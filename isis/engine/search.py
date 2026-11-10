"""Typed search: BM25 over passages, or BM25 fused with embedding similarity ("hybrid").

Query syntax: a query is a list of clauses, each a word, a "phrase" (verbatim, case and
spacing ignored) or a /regex/ (Python syntax; case-insensitive unless it has a capital letter).
    x      prefer: ranks passages up. Words are ranked with BM25 (more and rarer matches first), and
           a phrase or regex is scored the same way, as one more word. Without any + clause, a
           passage must match at least one of these to be found.
    +x     require: only passages matching x are found (and x counts for the ranking too).
    -x     never: passages matching x are dropped.
A word matches that word only, in any case: "whale" doesn't find "whales" (words are not stemmed).
A partly typed last word is expanded to the corpus terms it starts. correct_query fixes misspelled
plain words before a query from the page gets here. Words to prefer or require also match corpus
words a typo or two away, for typos in the documents (see typo_terms). The + and - clauses filter
every passage before ranking, in both modes, so similarity of meaning never brings back a
passage they exclude. The same clauses steer the Related panel (see related.py).
"""

from __future__ import annotations

import bisect
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .chunker import HEADING, document_title, web_host
from .index import Snapshot
from .lexical import B, K1
from .params import Group, tunable
from .text import term_spans, tokenize, tokenize_with_surface

# One clause: an optional operator, then a "phrase" (its closing quote may still be missing), a
# /regex/ that ends at a space or the end, a /regex still being typed, or a plain word.
TOKEN = re.compile(r"""([-+]?)(?:"([^"]*)(?:"|$)|/((?:\\.|[^/\\])+)/(?=\s|$)|(/(?:\\.|[^/\\])*)$|(\S+))""")
PREFER, REQUIRE, NEVER = "", "+", "-"
SITE = re.compile(r"site:(.*)", re.IGNORECASE)
MIN_PREFIX = 3
MAX_EXPANSIONS = 20
MIN_TYPO_LENGTH, MIN_TWO_TYPOS_LENGTH = 4, 8  # letters a word needs for one typo, and for two
# A typo is far rarer than the word it misspells: a word near yours counts as a typo of it, or yours
# as a typo of it, only if one is in this many times as many passages as the other. Two words of
# about the same frequency (form and farm) are more likely two real words.
TYPO_RARITY = 10
MAX_TYPO_TERMS = 50  # the most terms one word matches by typos, closest and most common first

KEYWORDS = Group("Keyword score", "How well the words in a passage match yours.")
FUSION = Group(
    "Mixing keywords and meaning",
    "Search makes two ranked lists of passages, one by keywords and one by meaning, "
    "then mixes them into one list of results.",
)


@dataclass
class SearchConfig:
    k1: float = tunable(
        K1, label="Words used many times", group=KEYWORDS, min=0.0, max=3.0, step=0.1,
        help="How much a word adds to the score when it appears many times in a passage.\n"
        "Raise it to rank passages that repeat your words higher.\n"
        "Lower it and repeats count for less. At 0, one use counts the same as many.",
    )
    b: float = tunable(
        B, label="Passage length", group=KEYWORDS, min=0.0, max=1.0, step=0.05,
        help="Long passages contain more words, so they can score well just because of their length. "
        "This sets how much a long passage's score is reduced.\n"
        "Raise it to rank short passages higher. At 1, length is fully corrected for.\n"
        "Lower it to rank long passages higher. At 0, length has no effect.",
    )
    rrf_k: int = tunable(
        20, label="Good rank in both lists", group=FUSION, min=1, max=100, step=1,
        help="Which passages come first when the keyword and meaning lists are mixed.\n"
        "Raise it to put passages that rank well in both lists first.\n"
        "Lower it to put the top passages of each list first, even if they rank badly in the other.",
    )
    keyword_weight: float = tunable(
        1.0, label="How important keywords are", group=FUSION, min=0.0, max=3.0, step=0.1,
        help="How much the keyword list counts in the mix.\n"
        "Raise it to rank results containing your words higher.\n"
        "Lower it and the meaning list counts for more. At 0, passages are ranked by meaning alone. "
        "Passages with no meaning data then count only through \"Passages with no meaning data\".",
    )
    meaning_weight: float = tunable(
        1.0, label="How important meaning is", group=FUSION, min=0.0, max=3.0, step=0.1,
        help="How much the meaning list counts in the mix.\n"
        "Raise it to rank results close to your words in meaning higher, even if they use different words.\n"
        "Lower it and the keyword list counts for more. At 0, passages are ranked by keywords alone. "
        "That's the same as turning the Meaning button off.",
    )
    unembedded_weight: float = tunable(
        1.0, label="Passages with no meaning data", group=FUSION, min=0.0, max=3.0, step=0.1,
        help="Some passages may have no meaning data, so they aren't in the meaning list. "
        "Their keyword rank stands in for a meaning rank, multiplied by this number.\n"
        "Raise it to rank these passages higher.\n"
        "Lower it to rank them lower. At 0, they count only through \"How important keywords are\". "
        "If that's also at 0, they won't appear.",
    )
    typo_weight: float = tunable(
        0.5, label="Words with typos", group=KEYWORDS, min=0.0, max=1.0, step=0.05,
        help="Search also finds words a typo away from yours, in case a document misspells them. "
        "This is how much such a word counts next to the word you typed. Two typos away counts this times itself.\n"
        "Raise it to rank passages with misspellings higher.\n"
        "Lower it to rank them lower. At 0, only your words are matched as you typed them. "
        "Words after + or - are always matched exactly.",
    )
    candidates: int = tunable(
        300, label="Length of each list", group=FUSION, min=20, max=2000, step=10,
        help="How many passages go into each list before mixing. "
        "A passage that's in neither list can't be shown.\n"
        "Raise it to find more passages, at the cost of speed.\n"
        "Lower it for faster results but fewer passages found.",
    )


@dataclass
class SnippetLine:
    line: int
    text: str
    in_passage: bool
    marks: list[tuple[int, int]]  # highlighted character ranges


@dataclass
class Hit:
    path: str
    heading: str
    start_line: int
    end_line: int
    score: float
    focus_line: int
    match: tuple[int, int, int] | None  # (line, ch, length) of the first highlighted word
    text: str
    snippet: list[SnippetLine]
    why: dict = field(default_factory=dict)
    marks: list[tuple[int, int, int]] = field(default_factory=list)  # (line, start, end) of every highlight in the passage
    title: str = ""  # of the document, if it has one (see document_title)


class QueryError(ValueError):
    """The query can't be run, e.g. an invalid regular expression."""


# A public server runs regexes under a time limit, and refuses ones that are costly to compile.
# The `regex` library takes a timeout, but expands counted repeats when it compiles, so
# /(x{65535}){65535}/ would use all the memory before any matching starts.
MAX_PATTERN_LENGTH = 300
MAX_PATTERN_REPEATS = 2000
COUNTED_REPEAT = re.compile(r"\{(\d*)(?:,(\d*))?\}")
FUZZY = re.compile(r"\{[^{}]*\b\d*[deis](?:<=?\d+|\b)[^{}]*\}")  # `regex` reads /a{e<=1}/ as fuzzy; `re` as plain text
_pattern_deadline: ContextVar[float | None] = ContextVar("isis_pattern_deadline", default=None)


@contextmanager
def pattern_time_limit(seconds: float):
    """Regexes compiled inside this block share `seconds` of matching time; running out raises
    QueryError. Outside any such block they are plain `re` patterns with no limit."""
    token = _pattern_deadline.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _pattern_deadline.reset(token)


@dataclass(frozen=True)
class TimedPattern:
    """A compiled `regex` pattern with the parts of re.Pattern that searching uses."""

    compiled: Any

    @property
    def pattern(self) -> str:
        return self.compiled.pattern

    @property
    def flags(self) -> int:
        return self.compiled.flags

    def finditer(self, text: str) -> Iterator:
        left = (_pattern_deadline.get() or 0) - time.monotonic()
        try:
            if left <= 0:
                raise TimeoutError
            # concurrent: other requests keep running while this one matches.
            yield from self.compiled.finditer(text, timeout=left, concurrent=True)
        except TimeoutError:
            raise QueryError(f"this pattern took too long /{self.pattern}/") from None


def timed_pattern(pattern: str, flags: int) -> TimedPattern:
    import regex

    if len(pattern) > MAX_PATTERN_LENGTH:
        raise QueryError(f"this pattern is too long (the limit is {MAX_PATTERN_LENGTH} characters)")
    if FUZZY.search(pattern):
        raise QueryError(f"invalid pattern /{pattern}/ because fuzzy matching isn't available")
    repeats = 1
    for low, high in COUNTED_REPEAT.findall(pattern):
        repeats *= max(int(low or 1), int(high or 1), 1)  # as if nested, the worst case
    if repeats > MAX_PATTERN_REPEATS:
        raise QueryError(f"this pattern has too many counted repeats (the limit is {MAX_PATTERN_REPEATS})")
    try:
        return TimedPattern(regex.compile(pattern, regex.IGNORECASE if flags & re.IGNORECASE else 0))
    except regex.error as e:
        raise QueryError(f"invalid pattern /{pattern}/ ({e})") from None


def compile_pattern(pattern: str) -> re.Pattern | TimedPattern:
    """A user's regex. Smart case: case-insensitive unless it contains a capital letter."""
    smart = 0 if any(c.isupper() for c in re.sub(r"\\.", "", pattern)) else re.IGNORECASE
    try:
        compiled = re.compile(pattern, smart)  # also when timed: the same syntax is valid everywhere
    except (re.error, OverflowError, RecursionError) as e:
        raise QueryError(f"invalid pattern /{pattern}/ ({e})") from None
    return compiled if _pattern_deadline.get() is None else timed_pattern(pattern, smart)


def phrase_pattern(phrase: str) -> re.Pattern:
    return re.compile(r"\s+".join(map(re.escape, phrase.split())), re.IGNORECASE)


@dataclass(frozen=True)
class Clause:
    op: str  # PREFER, REQUIRE or NEVER
    kind: str  # "word", "phrase", "regex", "site", or "partial" (a regex still being typed: ignored)
    value: str

    def __str__(self) -> str:
        return self.op + {"phrase": f'"{self.value}"', "regex": f"/{self.value}/", "site": f"site:{self.value}"}.get(self.kind, self.value)

    def terms(self) -> list[str]:
        return tokenize(self.value) if self.kind == "word" else []

    def pattern(self) -> re.Pattern | None:
        """The clause as text to match: a phrase, a regex, or a word that is not one search term
        (a stopword, or a hyphenated compound), matched as written."""
        if self.kind == "regex":
            return compile_pattern(self.value)
        if self.kind == "phrase":
            return phrase_pattern(self.value) if self.value.split() else None
        if self.kind == "word" and len(self.terms()) != 1:
            return re.compile(rf"(?<!\w){re.escape(self.value)}(?!\w)", re.IGNORECASE)
        return None


def split_query(query: str) -> list[Clause]:
    out = []
    for m in TOKEN.finditer(query):
        op, phrase, regex, partial, word = m.groups()
        if phrase is not None:
            out.append(Clause(op, "phrase", phrase))
        elif regex is not None:
            out.append(Clause(op, "regex", regex))
        elif partial is not None:
            out.append(Clause(op, "partial", partial))
        elif word and word not in (REQUIRE, NEVER):
            site = SITE.fullmatch(word)
            if not site:
                out.append(Clause(op, "word", word))
            elif site.group(1):
                out.append(Clause(op, "site", site.group(1)))
            else:
                out.append(Clause(op, "partial", word))  # still being typed
    return out


@dataclass
class Filters:
    """Hard constraints: terms and patterns a passage must contain, and ones it must not, and the
    websites its document must come from (any of them) or must not."""

    required_terms: set[str] = field(default_factory=set)
    required: list[re.Pattern] = field(default_factory=list)
    excluded_terms: set[str] = field(default_factory=set)
    excluded: list[re.Pattern] = field(default_factory=list)
    sites: set[str] = field(default_factory=set)
    excluded_sites: set[str] = field(default_factory=set)

    def __bool__(self) -> bool:
        return bool(self.requires or self.excluded_terms or self.excluded or self.sites or self.excluded_sites)

    @property
    def requires(self) -> bool:
        return bool(self.required_terms or self.required)

    def allowed(self, snapshot: Snapshot, rows=None) -> np.ndarray:
        """Which of `rows` (default: every passage) pass, as a boolean array aligned with them."""
        rows = np.arange(len(snapshot)) if rows is None else np.asarray(rows, np.int64)
        ok = np.ones(len(rows), bool)

        def containing(term: str) -> np.ndarray:
            has = np.zeros(len(snapshot), bool)
            tid = snapshot.term_id(term)
            if tid is not None:
                has[snapshot.bm25.matrix[:, tid].indices] = True
            return has[rows]

        for term in self.required_terms:
            ok &= containing(term)
        for term in self.excluded_terms:
            ok &= ~containing(term)
        for p in self.required:  # match counts are cached per snapshot, so typing doesn't rescan
            ok &= snapshot.pattern_counts(p, rows)[rows] > 0
        for p in self.excluded:
            ok &= snapshot.pattern_counts(p, rows)[rows] == 0
        if self.sites:
            ok &= snapshot.at_sites(self.sites)[rows]
        if self.excluded_sites:
            ok &= ~snapshot.at_sites(self.excluded_sites)[rows]
        return ok


@dataclass
class Steering:
    """What a list of clauses asks for, sorted by how each part is used."""

    words: Counter = field(default_factory=Counter)  # terms to rank by (prefer and require), with counts
    surfaces: dict = field(default_factory=dict)  # term -> the word as written
    patterns: list[re.Pattern] = field(default_factory=list)  # phrases and regexes to rank by
    filters: Filters = field(default_factory=Filters)
    text: str = ""  # what the meaning side embeds: the words and phrases to prefer or require

    @property
    def highlight(self) -> list[re.Pattern]:
        """The phrases and regexes to mark in what is shown."""
        return self.patterns + [p for p in self.filters.required if p not in self.patterns]


def steering(clauses: list[Clause]) -> Steering:
    """Compile clauses (raises QueryError for an invalid regex)."""
    out, text = Steering(), []
    for c in clauses:
        if c.kind == "partial":
            continue
        if c.kind == "site":
            (out.filters.excluded_sites if c.op == NEVER else out.filters.sites).add(web_host(c.value))
            continue
        own = c.terms()
        if c.op == NEVER:
            if c.kind == "word" and len(own) == 1:
                out.filters.excluded_terms.add(own[0])
            elif (p := c.pattern()) is not None:
                out.filters.excluded.append(p)
            continue
        if c.kind == "word":
            for term, surface in tokenize_with_surface(c.value):
                out.words[term] += 1
                out.surfaces.setdefault(term, surface)
            text.append(c.value)
            if c.op == REQUIRE:
                if len(own) == 1:
                    out.filters.required_terms.add(own[0])
                elif (p := c.pattern()) is not None:
                    out.filters.required.append(p)
            continue
        p = c.pattern()
        if p is None:
            continue
        out.patterns.append(p)
        if c.op == REQUIRE:
            out.filters.required.append(p)
        if c.kind == "phrase":
            text.append(c.value)
    out.text = " ".join(text)
    return out


@dataclass
class Query:
    term_ids: list[int]
    weights: list[float]
    terms: set[str]  # for highlighting
    filters: Filters
    text: str  # what the meaning side embeds
    patterns: list[re.Pattern] = field(default_factory=list)  # phrases and regexes to rank by
    highlight: list[re.Pattern] = field(default_factory=list)
    unknown: bool = False  # it has a word that no passage contains, which only meaning can look for


def expand_prefix(snapshot: Snapshot, prefix: str, limit: int = MAX_EXPANSIONS) -> list[str]:
    """Corpus terms starting with `prefix`, most common first."""
    terms = snapshot.sorted_terms
    lo = bisect.bisect_left(terms, prefix)
    hi = bisect.bisect_left(terms, prefix + "￿")
    found = terms[lo:hi]
    df = snapshot.bm25.doc_freq
    return sorted(found, key=lambda t: -df[snapshot.term_id(t)])[:limit]


def parse_query(snapshot: Snapshot, query: str, typo_weight: float = 0.0) -> Query:
    clauses = split_query(query)
    steer = steering(clauses)
    weights: dict[int, float] = {}
    terms: set[str] = set()
    for term, count in steer.words.items():
        tid = snapshot.term_id(term)
        if tid is not None:
            weights[tid] = float(count)
            terms.add(term)
    typing = None  # a partly typed last word
    last = clauses[-1] if clauses else None
    if query and not query[-1].isspace() and last is not None and last.kind == "word" and last.op != NEVER:
        pairs = tokenize_with_surface(last.value)
        if pairs and snapshot.term_id(pairs[-1][0]) is None and len(pairs[-1][1]) >= MIN_PREFIX:
            typing = pairs[-1][0]
            for t in expand_prefix(snapshot, pairs[-1][1]):
                weights.setdefault(snapshot.term_id(t), 1.0)
                terms.add(t)
    if typo_weight > 0:
        for term, count in steer.words.items():
            if term == typing:
                continue
            for tid, weight in typo_terms(snapshot, term, steer.surfaces[term], typo_weight).items():
                weights[tid] = max(weights.get(tid, 0.0), count * weight)
                terms.add(snapshot.vocab.terms[tid])
    unknown = any(snapshot.term_id(term) is None and len(word) >= MIN_PREFIX for term, word in steer.surfaces.items())
    return Query(list(weights), list(weights.values()), terms, steer.filters, steer.text, steer.patterns, steer.highlight, unknown)


def typo_limit(word: str) -> int:
    """How many typos a word may have: none if it's short or not all letters (a typo in a number
    is another number)."""
    if len(word) < MIN_TYPO_LENGTH or not word.isalpha():
        return 0
    return 2 if len(word) >= MIN_TWO_TYPOS_LENGTH else 1


def near_words(snapshot: Snapshot, word: str) -> list[tuple[str, int, int]]:
    """(word, term id, typos) for the corpus words a typo or two from `word`, by typo_limit, whose
    term is still in the corpus. The last few lookups are kept, since typing repeats them."""

    def find() -> list[tuple[str, int, int]]:
        view, df = snapshot.words, snapshot.bm25.doc_freq
        rows, typos = view.near(word, typo_limit(word))
        found = [(view.words[i], int(view.terms[i]), int(t)) for i, t in zip(rows.tolist(), typos.tolist())]
        return [(w, tid, t) for w, tid, t in found if tid < len(df) and df[tid] > 0]

    return snapshot.cached(snapshot.near_words, word, 256, find)


def typo_terms(snapshot: Snapshot, term: str, word: str, typo_weight: float) -> dict[int, float]:
    """Term id -> weight for the terms of the corpus words a typo or two from `word` (whose term is
    `term`), and for `term` itself if it is in the corpus.

    In BM25 a rare word weighs more, and a typo is about as rare as words get, so a passage with a
    typo would rank above those with the word spelt right. As in Lucene's fuzzy queries, every
    term here is weighed as if it were as common as the most common of them. A term then counts
    `typo_weight` times as much per typo."""
    df, idf = snapshot.bm25.doc_freq, snapshot.bm25.idf
    tid = snapshot.term_id(term)
    own = int(df[tid]) if tid is not None else 0
    best: dict[int, int] = {}
    for _, term, typos in near_words(snapshot, word):
        if term != tid and (not own or df[term] * TYPO_RARITY <= own or df[term] >= own * TYPO_RARITY):
            best[term] = min(typos, best.get(term, typos))
    if not best:
        return {}
    chosen = sorted(best, key=lambda t: (best[t], -df[t], t))[:MAX_TYPO_TERMS]
    common = min(float(idf[t]) for t in chosen + ([tid] if own else []))
    weights = {t: typo_weight ** best[t] * common / float(idf[t]) for t in chosen}
    if own:
        weights[tid] = common / float(idf[tid])
    return weights


def correct_query(snapshot: Snapshot, query: str) -> str | None:
    """The query with its misspelled words corrected, or None if it has none. Only a plain word that
    no passage contains is changed: + and - clauses, phrases and regexes mean what they say."""
    if not len(snapshot):
        return None
    df = snapshot.bm25.doc_freq
    parts, at = [], 0
    for m in TOKEN.finditer(query):
        written = m.group(5)
        pairs = tokenize_with_surface(written) if written and m.group(1) == PREFER else []
        if len(pairs) != 1 or snapshot.term_id(pairs[0][0]) is not None:
            continue
        word = pairs[0][1]
        if not typo_limit(word):
            continue
        if m.end() == len(query) and expand_prefix(snapshot, word, 1):
            continue  # still being typed
        near = near_words(snapshot, word)
        if near:  # the closest, then the one whose term is in the most passages
            parts += [query[at:m.start(5)], min(near, key=lambda n: (n[2], -df[n[1]], n[0]))[0]]
            at = m.end(5)
    return "".join(parts) + query[at:] if parts else None


def lexical_scores(snapshot: Snapshot, bm25, term_ids, weights, patterns, rows=None) -> np.ndarray:
    """BM25 for weighted terms, plus each phrase or regex scored as one more term. Patterns are
    only matched in `rows` (default: every passage), and score 0 elsewhere."""
    scores = bm25.scores(term_ids, weights)
    for p in patterns:
        scores = scores + bm25.pattern_scores(snapshot.pattern_counts(p, rows), rows)
    return scores


MAX_RUNS = 256  # beyond this many separate runs of rows, one product over every passage is quicker


def similarities(vectors: np.ndarray, query: np.ndarray, scope: np.ndarray | None, wanted: np.ndarray) -> np.ndarray:
    """vectors @ query where `wanted`, and -inf elsewhere. Within `scope` (None: everywhere) only
    its runs of rows are multiplied, as views without copying, so a search in one folder costs
    about as much as that folder."""
    if scope is None:
        out = vectors @ query
    else:
        edges = np.flatnonzero(np.diff(np.concatenate(([0], scope.view(np.int8), [0]))))
        if len(edges) > 2 * MAX_RUNS:
            out = vectors @ query
        else:
            out = np.full(len(vectors), -np.inf, np.float32)
            for start, end in zip(edges[::2], edges[1::2]):
                out[start:end] = vectors[start:end] @ query
    out[~wanted] = -np.inf
    return out


def top_rows(scores: np.ndarray, k: int, positive: bool = False) -> np.ndarray:
    """Row indices of the k highest scores, best first."""
    rows = np.flatnonzero(scores > 0) if positive else np.flatnonzero(np.isfinite(scores))
    if len(rows) > k:
        rows = rows[np.argpartition(-scores[rows], k)[:k]]
    return rows[np.argsort(-scores[rows], kind="stable")]


def remembering_lines(read_lines: Callable[[str], list[str]]) -> Callable[[str], list[str] | None]:
    """`read_lines` that reads each file once for the results being built, and gives None for a file
    deleted since indexing (the next refresh drops it)."""
    known: dict[str, list[str] | None] = {}

    def lines_of(path: str) -> list[str] | None:
        if path not in known:
            try:
                known[path] = read_lines(path)
            except (OSError, ValueError):
                known[path] = None
        return known[path]

    return lines_of


def search(
    snapshot: Snapshot,
    query: str,
    *,
    limit: int = 50,
    mode: str = "lexical",
    embedder=None,
    context_lines: int = 2,
    read_lines: Callable[[str], list[str]],
    config: SearchConfig | None = None,
    skip_folders=(),
    skip_files=(),
) -> list[Hit]:
    config = config or SearchConfig()
    if not len(snapshot) or not query.strip():
        return []
    q = parse_query(snapshot, query, config.typo_weight)
    if not q.term_ids and not q.patterns and not q.filters.requires and not q.filters.sites and not q.unknown:
        return []  # only exclusions, or a regex still being typed: nothing to look for
    # The passages this search may show: all of them, or those outside the folders and documents
    # left out. Only these are scanned for patterns and compared by meaning.
    scope, rows = None, None
    if skip_folders or skip_files:
        scope = snapshot.outside_folders(skip_folders) if skip_folders else np.ones(len(snapshot), bool)
        for path in skip_files:
            if (span := snapshot.file_rows.get(path)) is not None:
                scope[span[0]:span[1]] = False
        rows = np.flatnonzero(scope)
    if rows is None:
        allowed = q.filters.allowed(snapshot)
    else:
        allowed = np.zeros(len(snapshot), bool)
        allowed[rows] = q.filters.allowed(snapshot, rows)
    if mode == "hybrid" and config.keyword_weight + config.meaning_weight <= 0:
        return []  # nothing counts
    embeddings = mode == "hybrid" and config.meaning_weight > 0 and embedder is not None and snapshot.has_vector.any()
    stand_in = config.meaning_weight * config.unembedded_weight  # for the meaning rank a passage without embeddings lacks
    if config.keyword_weight + stand_in <= 0:  # nothing counts for a passage without embeddings: it is out
        allowed &= snapshot.has_vector
    bm25 = snapshot.bm25_for(config.k1, config.b)
    lexical = np.where(allowed, lexical_scores(snapshot, bm25, q.term_ids, q.weights, q.patterns, rows), 0.0)

    if embeddings and q.text.strip():
        semantic = similarities(snapshot.vectors, embedder.embed([q.text], cache=True, query=True)[0], scope,
                                snapshot.has_vector & allowed)
        lex_rank = {int(row): rank for rank, row in enumerate(top_rows(lexical, config.candidates, positive=True))}
        sem_rank = {int(row): rank for rank, row in enumerate(top_rows(semantic, config.candidates))}
        fused: dict[int, float] = {}
        for ranks, weight in ((lex_rank, config.keyword_weight), (sem_rank, config.meaning_weight)):
            for row, rank in ranks.items():
                fused[row] = fused.get(row, 0.0) + weight / (config.rrf_k + rank + 1)
        for row, rank in lex_rank.items():
            if not snapshot.has_vector[row]:
                fused[row] += stand_in / (config.rrf_k + rank + 1)
        ranked = sorted(((row, score) for row, score in fused.items() if score > 0), key=lambda kv: -kv[1])

        def why(row: int) -> dict:
            return {
                "bm25": float(lexical[row]),
                "cosine": float(semantic[row]),
                "keyword_rank": lex_rank[row] + 1 if row in lex_rank else None,  # None: not among the top keyword hits
                "meaning_rank": sem_rank[row] + 1 if row in sem_rank else None,
                "embedded": bool(snapshot.has_vector[row]),
            }
    else:
        rows = top_rows(lexical, len(lexical), positive=True)
        ranked = [(int(r), float(lexical[r])) for r in rows]

        def why(row: int) -> dict:
            return {"bm25": float(lexical[row])}

    # Passages that match every + clause but none of the others. A site only narrows what other
    # clauses find, and on its own lists what the site has.
    if q.filters.requires or (q.filters.sites and not q.term_ids and not q.patterns and not q.unknown):
        seen = {r for r, _ in ranked}
        ranked += [(int(r), 0.0) for r in np.flatnonzero(allowed) if r not in seen]

    lines_of = remembering_lines(read_lines)
    hits = []
    for row, score in ranked:
        lines = lines_of(snapshot.passages[row].path)
        if lines:
            hits.append(make_hit(snapshot, row, score, q.terms, lines, context_lines, why(row), q.highlight))
            if len(hits) >= limit:
                break
    return hits


def _spans(text: str, terms: set[str], patterns: list[re.Pattern]) -> list[tuple[int, int]]:
    spans = term_spans(text, terms)
    for p in patterns:
        spans += [m.span() for m in p.finditer(text) if m.end() > m.start()]
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):  # a word can match a term and a phrase: one mark
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def heading_line(lines: list[str], start: int, heading: str) -> int | None:
    """The line of the heading a passage sits under: the nearest heading above it."""
    for n in range(min(start, len(lines)) - 1, -1, -1):
        m = HEADING.match(lines[n])
        if m:
            return n if m.group(1) == heading else None
    return None


def make_hit(
    snapshot: Snapshot,
    row: int,
    score: float,
    terms: set[str],
    lines: list[str],
    context_lines: int,
    why: dict,
    patterns: list[re.Pattern],
) -> Hit:
    p = snapshot.passages[row]
    last_line = len(lines) - 1
    start, end = min(p.start_line, last_line), min(p.end_line, last_line)

    def spans_on(n: int) -> list[tuple[int, int]]:
        return _spans(lines[n], terms, patterns)

    def first_match_on(n: int) -> tuple[int, int, int] | None:
        spans = spans_on(n)
        return (n, spans[0][0], spans[0][1] - spans[0][0]) if spans else None

    match = next((m for n in range(start, end + 1) if (m := first_match_on(n))), None)
    above = None  # a heading line that holds the only match (phrases and regexes see headings)
    if match is None and p.heading:
        n = heading_line(lines, start, p.heading)
        if n is not None and (match := first_match_on(n)):
            above = n

    # The whole passage (what + adds to the task, and what the reader highlights), with a few
    # dimmed lines of context around it.
    first = max(0, start - context_lines)
    snippet = []
    if above is not None and above < first:  # show the heading, then the passage
        snippet += [SnippetLine(above, lines[above], False, spans_on(above)),
                    SnippetLine(above, "", False, [])]
    for n in range(first, min(last_line, end + context_lines) + 1):
        inside = start <= n <= end
        snippet.append(SnippetLine(n, lines[n], inside, spans_on(n) if inside or n == above else []))
    focus = match[0] if match else start
    marks = [(n, s, e) for n in ([above] if above is not None else []) + list(range(start, end + 1)) for s, e in spans_on(n)]

    return Hit(p.path, p.heading, p.start_line, p.end_line, score, focus, match, p.text, snippet, why, marks, title=document_title(lines))
