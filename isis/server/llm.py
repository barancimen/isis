"""LLM help for query synthesis: a person describes in English what they want, and DeepSeek writes
the query for Search, or the word rules for Related.

The model gets the page's own guide for that box (How to search, or How Related and word rules
work), read from index.html so it stays as the page has it, with instructions written for it in
front. Every answer is checked with the query parser, and a wrong one is sent back with the reason,
up to MAX_ATTEMPTS calls in all.

Each call costs money, so the calls are counted before they are made: an hourly and a daily budget
for everybody together, which public mode adds a limit per address to (see public.py). The key is
never logged or sent to the page, nor is what DeepSeek says when it refuses a request."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from ..engine.search import TOKEN, QueryError, pattern_time_limit, split_query, steering

log = logging.getLogger(__name__)

API = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
KEY_ENV = "DEEPSEEK_API_KEY"
KEY_FILE = Path(__file__).resolve().parents[2] / ".deepseek_api_key"  # in the isis folder, ignored by git
WEB = Path(__file__).resolve().parents[2] / "web"

MAX_ATTEMPTS = 6  # one try and five retries
MAX_REQUEST = 600  # characters of English
MAX_QUERY = 300  # characters of the query written
CALL_SECONDS = 40  # one call to DeepSeek
TOTAL_SECONDS = 80  # all the attempts of one request, well inside what a proxy waits
MAX_TOKENS = 1500  # the model reasons before it answers, and that counts too (about 200 tokens in all, as seen)
GUIDES = {"search": "search-guide", "related": "related-guide"}  # the guide of each box, by its id in index.html


class Refused(Exception):
    """The request can't be served: the feature is off, the budget is spent, or DeepSeek refuses."""

    def __init__(self, status: int, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.status, self.retry_after = status, retry_after


class Unusable(ValueError):
    """An answer that isn't a query the box can take, with the reason, for the model to fix."""


class Upstream(Exception):
    """A call to DeepSeek that failed. `retry` says whether another call may do better."""

    def __init__(self, message: str, retry: bool):
        super().__init__(message)
        self.retry = retry


def load_key(path: str | os.PathLike | None = None) -> str | None:
    """The DeepSeek key: from DEEPSEEK_API_KEY, or else the file given, or else .deepseek_api_key in
    the isis folder. None if there is none, and then LLM help is off."""
    if key := os.environ.get(KEY_ENV, "").strip():
        return key
    file = Path(path or KEY_FILE)
    try:
        return file.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        if path:
            log.warning("there is no DeepSeek key file %s: LLM help is off", path)
    except OSError as e:  # such as a key the server's user may not read
        log.warning("cannot read the DeepSeek key file %s (%s): LLM help is off", file, e.strerror or e)
    return None


# ---- what the model is told ----

class _GuideText(HTMLParser):
    """The text of the element with one id, as plain text with markdown's marks for headings, list
    items and code."""

    def __init__(self, element_id: str):
        super().__init__(convert_charrefs=True)
        self.element_id, self.depth, self.out = element_id, 0, []

    def handle_starttag(self, tag, attrs):
        if self.depth:
            self.depth += tag not in ("br", "img", "input", "hr")
            self.out.append({"h3": "\n\n# ", "h4": "\n\n## ", "p": "\n\n", "li": "\n- ", "code": "`", "ul": "\n"}.get(tag, ""))
        elif dict(attrs).get("id") == self.element_id:
            self.depth = 1

    def handle_endtag(self, tag):
        if self.depth:
            self.depth -= 1
            if tag == "code":
                self.out.append("`")

    def handle_data(self, data):
        if self.depth:
            self.out.append(re.sub(r"\s+", " ", data))

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]*\n[ \t]*", "\n", "".join(self.out))).strip()


_guides: dict[str, tuple[float, str]] = {}


def guide(target: str, page: Path = WEB / "index.html") -> str:
    """The page's guide for this box, as text, read again when index.html changes."""
    try:
        mtime = page.stat().st_mtime
    except OSError:
        return ""
    cached = _guides.get(target)
    if cached and cached[0] == mtime:
        return cached[1]
    parser = _GuideText(GUIDES[target])
    parser.feed(page.read_text(encoding="utf-8"))
    text = parser.text()
    _guides[target] = (mtime, text)
    return text


RULES = """Rules for your answer:
- Reply with one JSON object and nothing else: {"query": "<the query>"}.
- The query is one line of at most 300 characters.
- Use only the syntax in the guide: plain words, "phrases", wildcards with *, /regular expressions/, + and - in front of any of them, site:host, and doc: in front of a word, phrase, wildcard or expression.
- Words aren't stemmed. Add a * to catch other endings, as in whal* for whale, whales and whaling.
- Plain words, phrases, wildcards and expressions rank results higher. They aren't required. Put + in front only of what every result must contain, and - in front of what no result may contain.
- Use a few words for each idea, including synonyms the documents may use, since any of them can match.
- Close every phrase with " and every regular expression with /. Follow the closing / with a space or the end of the query, and write a / inside an expression as \\/.
- Prefer words, phrases and wildcards to regular expressions. Use an expression only for what they can't say, such as digits or alternatives inside a word.
- A query made only of - rules finds nothing, so include at least one thing to look for.
- If the current query is given and the request reads as a change to it, change it. Otherwise write a new one.
- Don't explain anything."""

EXAMPLES = {
    "search": [
        ("passages about sperm whales hunting squid, but nothing about ships", '"sperm whale" squid hunt* -ship*'),
        ("pages from github that explain how to install something", "site:github.com install*"),
        ("years in the 1800s near the word voyage", "voyage /\\b18\\d\\d\\b/"),
        ("composting tips, only in documents that never mention tomatoes", "compost* garden* soil -doc:tomato*"),
        ("anything that quotes the phrase 'call me Ishmael', which must be there", '+"call me ishmael"'),
    ],
    "related": [
        ("focus on the voyage itself and leave out anything about finches", "voyage* sail* ship* -finch*"),
        ("prefer passages that mention a year", "/\\b1[89]\\d\\d\\b/"),
        ("only suggest passages that talk about islands, ideally volcanic ones", "+island* volcan*"),
        ("nothing from wikipedia, and lean towards experiments", "-site:wikipedia.org experiment*"),
    ],
}

INTRO = {
    "search": ("You write queries for Search in isis, a program for searching a person's own collection of Markdown "
               "documents. The person describes in English what they want to find, and you turn that into a query "
               "for the search box. The guide below is what the program shows the person about writing queries."),
    "related": ("You write word rules for Related in isis, a program for searching a person's own collection of "
                "Markdown documents. Related suggests passages that are related to the document the person is "
                "writing. Word rules steer those suggestions and use the same syntax as a search query. The person "
                "describes in English how they want to steer the suggestions, and you turn that into word rules. The "
                "guide below is what the program shows the person about Related and word rules."),
}


def instructions(target: str) -> str:
    examples = "\n".join(f'Request: {request}\nAnswer: {json.dumps({"query": query})}' for request, query in EXAMPLES[target])
    return f"{INTRO[target]}\n\n{RULES}\n\nExamples:\n{examples}\n\nThe guide:\n\n{guide(target)}"


def asking(target: str, request: str, current: str) -> str:
    box = "search box" if target == "search" else "word rules box"
    now = f"The {box} now holds: {current}" if current.strip() else f"The {box} is empty."
    return f"{now}\n\nRequest: {request}"


# ---- checking an answer ----

def _regex_available() -> bool:
    try:
        import regex  # noqa: F401
    except ImportError:
        return False
    return True


def check_query(text: str, target: str) -> str:
    """The query in an answer, if the box can take it as written. Raises Unusable with the reason
    otherwise: the parser itself lets a lot through, such as an unclosed phrase or expression,
    which it reads as being typed."""
    query = text.strip()
    if not query:
        raise Unusable("the query is empty")
    if "\n" in query or "\r" in query:
        raise Unusable("the query must be one line")
    if len(query) > MAX_QUERY:
        raise Unusable(f"the query is {len(query)} characters long, and the limit is {MAX_QUERY}")
    if query.count('"') % 2:
        raise Unusable('a phrase isn\'t closed: the query has an odd number of " marks')
    for m in TOKEN.finditer(query):
        op, doc, phrase, regex, partial, word = m.groups()
        if partial is not None:
            raise Unusable(f"the regular expression {partial} isn't closed with a / followed by a space or the end")
        if phrase is not None and not phrase.strip():
            raise Unusable("a phrase is empty")
        if word in ("+", "-"):
            raise Unusable(f"a {word} stands alone: put it right in front of what it applies to, with no space")
        if word and word.lstrip("+-").removeprefix("doc:").startswith("/"):
            raise Unusable(f"the regular expression {word} isn't closed with a / followed by a space or the end")
    clauses = split_query(query)
    if partial := next((c for c in clauses if c.kind == "partial"), None):
        raise Unusable(f"{partial.value} is incomplete: doc: and site: need something right after them")
    if not clauses:
        raise Unusable("the query has nothing to look for")
    if target == "search" and all(c.op == "-" for c in clauses):
        raise Unusable("a query made only of - rules finds nothing")
    try:
        if _regex_available():  # with the limits the public site sets on expressions
            with pattern_time_limit(3.0):
                steering(clauses)
        else:
            steering(clauses)
    except QueryError as e:
        raise Unusable(str(e)) from None
    return query


def query_in(answer: str, target: str) -> str:
    """The query in the model's answer, checked."""
    text = answer.strip()
    if text.startswith("```"):  # a JSON object in a code block
        text = re.sub(r"^```\w*\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except ValueError:
        raise Unusable('the answer isn\'t a JSON object. Reply with {"query": "..."} only') from None
    if not isinstance(data, dict) or not isinstance(data.get("query"), str):
        raise Unusable('the answer must be a JSON object with a "query" string, as in {"query": "..."}')
    return check_query(data["query"], target)


# ---- the budget ----

@dataclass
class Budget:
    """How many calls to DeepSeek may be made, by everyone together: a number per hour and per day.
    A call counts when it is made, whatever it answers."""

    per_hour: int = 60
    per_day: int = 300
    clock: object = time.time
    calls: deque = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def take(self) -> float:
        """Count one call and return 0, or return the seconds until one is allowed."""
        with self.lock:
            now = self.clock()
            while self.calls and self.calls[0] <= now - 86400:
                self.calls.popleft()
            hour = [t for t in self.calls if t > now - 3600]
            if len(self.calls) >= self.per_day:
                return self.calls[0] + 86400 - now
            if len(hour) >= self.per_hour:
                return hour[0] + 3600 - now
            self.calls.append(now)
            return 0.0


# ---- calling DeepSeek ----

def call_deepseek(key: str, messages: list[dict], timeout: float) -> str:
    """The text of DeepSeek's answer. Raises Upstream on failure, without what DeepSeek said, which
    can echo the request."""
    payload = {"model": MODEL, "messages": messages, "max_tokens": MAX_TOKENS, "reasoning_effort": "low",
               "response_format": {"type": "json_object"}}
    request = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            answer = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (401, 402, 403):
            raise Upstream(f"DeepSeek refused the request (HTTP {e.code}). The key or the account needs attention", retry=False) from None
        raise Upstream(f"DeepSeek answered HTTP {e.code}", retry=e.code in (429, 500, 502, 503, 504)) from None
    except (OSError, ValueError) as e:  # no connection, a timeout, or an answer that isn't JSON
        raise Upstream(f"DeepSeek couldn't be reached ({type(e).__name__})", retry=True) from None
    try:
        choice = answer["choices"][0]
        content = choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        raise Upstream("DeepSeek's answer had no text", retry=True) from None
    if choice.get("finish_reason") == "length":
        return ""  # it ran out of tokens while reasoning: an empty answer, to be tried again
    return content


class QuerySynth:
    """Writes queries for the page. `complete(messages, timeout)` returns the model's text; tests
    give one that needs no network."""

    def __init__(self, key: str | None, *, budget: Budget | None = None, complete=None, slots: int = 2):
        self.key = key
        self.budget = budget or Budget()
        self.complete = complete or (lambda messages, timeout: call_deepseek(key, messages, timeout))
        self.slots = threading.BoundedSemaphore(slots)

    @property
    def available(self) -> bool:
        return bool(self.key)

    def synthesize(self, target: str, request: str, current: str = "") -> dict:
        if not self.available:
            raise Refused(503, "LLM help is off on this server: it has no DeepSeek key")
        if target not in GUIDES:
            raise Refused(400, "the target must be search or related")
        request = request.strip()
        if not request:
            raise Refused(400, "write what you want to find first")
        if len(request) > MAX_REQUEST or len(current) > MAX_REQUEST:
            raise Refused(400, f"the request is too long: keep it under {MAX_REQUEST} characters")
        if not self.slots.acquire(timeout=10):
            raise Refused(503, "LLM help is busy. Try again in a moment", 5)
        try:
            return self._attempts(target, request, current.strip())
        finally:
            self.slots.release()

    def _attempts(self, target: str, request: str, current: str) -> dict:
        messages = [{"role": "system", "content": instructions(target)},
                    {"role": "user", "content": asking(target, request, current)}]
        started, problem, made = time.monotonic(), "", 0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            left = TOTAL_SECONDS - (time.monotonic() - started)
            if left < 5:
                break
            if wait := self.budget.take():
                raise Refused(429, "LLM help has been used as much as it may be for now. Try again later", wait)
            made = attempt
            try:
                answer = self.complete(messages, min(CALL_SECONDS, left))
            except Upstream as e:
                problem = str(e)
                log.warning("LLM help: %s", problem)
                if not e.retry:
                    raise Refused(502, problem) from None
                continue
            try:
                return {"query": query_in(answer, target), "attempts": attempt}
            except Unusable as e:
                problem = f"the model's answer couldn't be used: {e}"
                messages += [{"role": "assistant", "content": answer or "(no answer)"},
                             {"role": "user", "content": f"That can't be used: {e}. Reply again with only "
                                                         '{"query": "..."}, following the rules and the guide.'}]
        raise Refused(502, f"LLM help failed after {made} {'attempt' if made == 1 else 'attempts'}. "
                           f"The last problem: {problem or 'it took too long'}")
