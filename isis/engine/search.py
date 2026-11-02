"""Typed search: BM25 over passages, or BM25 fused with embedding similarity ("hybrid").

Query syntax: a query is a list of clauses, each a word, a "phrase" (verbatim, case and
spacing ignored) or a /regex/ (Python syntax; case-insensitive unless it has a capital letter).
    x      prefer: ranks passages up. Words are ranked with BM25 (more and rarer matches first), and
           a phrase or regex is scored the same way, as one more word. Without any + clause, a
           passage must match at least one of these to be found.
    +x     require: only passages matching x are found (and x counts for the ranking too).
    -x     never: passages matching x are dropped. A word is matched by its stem, so -whale also
           drops "whales" and "whaling".
A partly typed last word is expanded to the corpus terms it starts. The + and - clauses filter
every passage before ranking, in both modes, so similarity of meaning never brings back a
passage they exclude. The same clauses steer the Related panel (see related.py).
"""

from __future__ import annotations

import bisect
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .chunker import HEADING
from .index import Snapshot
from .params import Group, tunable
from .text import term_spans, tokenize, tokenize_with_surface

# One clause: an optional operator, then a "phrase" (its closing quote may still be missing), a
# /regex/ that ends at a space or the end, a /regex still being typed, or a plain word.
TOKEN = re.compile(r"""([-+]?)(?:"([^"]*)(?:"|$)|/((?:\\.|[^/\\])+)/(?=\s|$)|(/(?:\\.|[^/\\])*)$|(\S+))""")
PREFER, REQUIRE, NEVER = "", "+", "-"
MIN_PREFIX = 3
MAX_EXPANSIONS = 20

KEYWORDS = Group("Keyword score", "The keyword score shows how much the words in a passage agree with your words.")
FUSION = Group(
    "Mixture of keywords and meaning",
    "The app makes two ranked lists of passages. One list uses keywords and the other list uses meaning. "
    "Then the app mixes the two lists into one list of results.",
)


@dataclass
class SearchConfig:
    k1: float = tunable(
        1.2, label="Words used many times", group=KEYWORDS, min=0.0, max=3.0, step=0.1,
        help="This slider sets how much a word adds to the score when it occurs many times in a passage.\n"
        "With a higher value, passages that use your words many times rank higher.\n"
        "With a lower value, more uses of a word add less to the score. At 0, a word that occurs one time counts the same as a word that occurs many times.",
    )
    b: float = tunable(
        0.35, label="Passage length", group=KEYWORDS, min=0.0, max=1.0, step=0.05,
        help="Long passages contain more words. Thus they can get a high score only because of their length. "
        "This slider sets how much the score of a long passage decreases.\n"
        "With a higher value, short passages rank higher. At 1, the app makes the full correction for length.\n"
        "With a lower value, long passages rank higher. At 0, length has no effect.",
    )
    rrf_k: int = tunable(
        20, label="Good rank in the two lists", group=FUSION, min=1, max=100, step=1,
        help="This slider sets which passages come first when the app mixes the keyword list and the meaning list.\n"
        "With a higher value, passages with a good rank in the two lists come first.\n"
        "With a lower value, the top passages of each list come first, also if they have a bad rank in the other list.",
    )
    keyword_weight: float = tunable(
        1.0, label="How important keywords are", group=FUSION, min=0.0, max=3.0, step=0.1,
        help="This slider sets how much the keyword list counts in the mix.\n"
        "With a higher value, results that contain your words rank higher.\n"
        "With a lower value, the meaning list counts more. At 0, the app ranks passages only by meaning. "
        "Passages with no meaning data then count only through the slider \"Passages with no meaning data\".",
    )
    meaning_weight: float = tunable(
        1.0, label="How important meaning is", group=FUSION, min=0.0, max=3.0, step=0.1,
        help="This slider sets how much the meaning list counts in the mix.\n"
        "With a higher value, results that are near your words in meaning rank higher, also if they use different words.\n"
        "With a lower value, the keyword list counts more. At 0, the app ranks passages only by keywords. "
        "This is the same as when the Meaning button is off.",
    )
    unembedded_weight: float = tunable(
        1.0, label="Passages with no meaning data", group=FUSION, min=0.0, max=3.0, step=0.1,
        help="The app can have no meaning data for some passages. These passages are not in the meaning list. "
        "The app uses their keyword rank value as their meaning rank, multiplied by this number.\n"
        "With a higher value, these passages rank higher.\n"
        "With a lower value, these passages rank lower. At 0, they count only through \"How important keywords are\". "
        "If that slider is also at 0, the app does not show these passages.",
    )
    candidates: int = tunable(
        300, label="Length of each list", group=FUSION, min=20, max=2000, step=10,
        help="This slider sets how many passages go into each list before the mix. "
        "The app cannot show a passage that is not in one of the two lists.\n"
        "With a higher value, the app can find more passages, but it can operate more slowly.\n"
        "With a lower value, the app operates more quickly, but it can find a smaller number of passages.",
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


class QueryError(ValueError):
    """The query can't be run, e.g. an invalid regular expression."""


def compile_pattern(pattern: str) -> re.Pattern:
    """A user's regex. Smart case: case-insensitive unless it contains a capital letter."""
    smart = 0 if any(c.isupper() for c in re.sub(r"\\.", "", pattern)) else re.IGNORECASE
    try:
        return re.compile(pattern, smart)
    except re.error as e:
        raise QueryError(f"invalid pattern /{pattern}/: {e}") from None


def phrase_pattern(phrase: str) -> re.Pattern:
    return re.compile(r"\s+".join(map(re.escape, phrase.split())), re.IGNORECASE)


@dataclass(frozen=True)
class Clause:
    op: str  # PREFER, REQUIRE or NEVER
    kind: str  # "word", "phrase", "regex", or "partial" (a regex still being typed: ignored)
    value: str

    def __str__(self) -> str:
        return self.op + {"phrase": f'"{self.value}"', "regex": f"/{self.value}/"}.get(self.kind, self.value)

    def stems(self) -> list[str]:
        return tokenize(self.value) if self.kind == "word" else []

    def pattern(self) -> re.Pattern | None:
        """The clause as text to match: a phrase, a regex, or a word that is not one search term
        (a stopword, or a hyphenated compound), matched as written."""
        if self.kind == "regex":
            return compile_pattern(self.value)
        if self.kind == "phrase":
            return phrase_pattern(self.value) if self.value.split() else None
        if self.kind == "word" and len(self.stems()) != 1:
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
            out.append(Clause(op, "word", word))
    return out


@dataclass
class Filters:
    """Hard constraints: terms and patterns a passage must contain, and ones it must not."""

    required_terms: set[str] = field(default_factory=set)  # stems
    required: list[re.Pattern] = field(default_factory=list)
    excluded_terms: set[str] = field(default_factory=set)
    excluded: list[re.Pattern] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.requires or self.excluded_terms or self.excluded)

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
            ok &= snapshot.pattern_counts(p)[rows] > 0
        for p in self.excluded:
            ok &= snapshot.pattern_counts(p)[rows] == 0
        return ok


@dataclass
class Steering:
    """What a list of clauses asks for, sorted by how each part is used."""

    words: Counter = field(default_factory=Counter)  # stems to rank by (prefer and require), with counts
    surfaces: dict = field(default_factory=dict)  # stem -> the word as written
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
        stems = c.stems()
        if c.op == NEVER:
            if c.kind == "word" and len(stems) == 1:
                out.filters.excluded_terms.add(stems[0])
            elif (p := c.pattern()) is not None:
                out.filters.excluded.append(p)
            continue
        if c.kind == "word":
            for stem, surface in tokenize_with_surface(c.value):
                out.words[stem] += 1
                out.surfaces.setdefault(stem, surface)
            text.append(c.value)
            if c.op == REQUIRE:
                if len(stems) == 1:
                    out.filters.required_terms.add(stems[0])
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


def expand_prefix(snapshot: Snapshot, prefix: str, limit: int = MAX_EXPANSIONS) -> list[str]:
    """Corpus terms starting with `prefix`, most common first."""
    terms = snapshot.sorted_terms
    lo = bisect.bisect_left(terms, prefix)
    hi = bisect.bisect_left(terms, prefix + "￿")
    found = terms[lo:hi]
    df = snapshot.bm25.doc_freq
    return sorted(found, key=lambda t: -df[snapshot.term_id(t)])[:limit]


def parse_query(snapshot: Snapshot, query: str) -> Query:
    clauses = split_query(query)
    steer = steering(clauses)
    weights: dict[int, float] = {}
    terms: set[str] = set()
    for stem, count in steer.words.items():
        tid = snapshot.term_id(stem)
        if tid is not None:
            weights[tid] = float(count)
            terms.add(stem)
    last = clauses[-1] if clauses else None
    if query and not query[-1].isspace() and last is not None and last.kind == "word" and last.op != NEVER:
        pairs = tokenize_with_surface(last.value)
        if pairs and snapshot.term_id(pairs[-1][0]) is None and len(pairs[-1][1]) >= MIN_PREFIX:
            for t in expand_prefix(snapshot, pairs[-1][1]):  # a partly typed last word
                weights.setdefault(snapshot.term_id(t), 1.0)
                terms.add(t)
    return Query(list(weights), list(weights.values()), terms, steer.filters, steer.text, steer.patterns, steer.highlight)


def lexical_scores(snapshot: Snapshot, bm25, term_ids, weights, patterns) -> np.ndarray:
    """BM25 for weighted terms, plus each phrase or regex scored as one more term."""
    scores = bm25.scores(term_ids, weights)
    for p in patterns:
        scores = scores + bm25.pattern_scores(snapshot.pattern_counts(p))
    return scores


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
) -> list[Hit]:
    config = config or SearchConfig()
    if not len(snapshot) or not query.strip():
        return []
    q = parse_query(snapshot, query)
    if not q.term_ids and not q.patterns and not q.filters.requires:
        return []  # only exclusions, or a regex still being typed: nothing to look for
    allowed = q.filters.allowed(snapshot) if q.filters else np.ones(len(snapshot), bool)
    if skip_folders:
        allowed &= snapshot.outside_folders(skip_folders)
    if mode == "hybrid" and config.keyword_weight + config.meaning_weight <= 0:
        return []  # nothing counts
    embeddings = mode == "hybrid" and config.meaning_weight > 0 and embedder is not None and snapshot.has_vector.any()
    stand_in = config.meaning_weight * config.unembedded_weight  # for the meaning rank a passage without embeddings lacks
    if config.keyword_weight + stand_in <= 0:  # nothing counts for a passage without embeddings: it is out
        allowed &= snapshot.has_vector
    bm25 = snapshot.bm25_for(config.k1, config.b)
    lexical = np.where(allowed, lexical_scores(snapshot, bm25, q.term_ids, q.weights, q.patterns), 0.0)

    if embeddings and q.text.strip():
        semantic = snapshot.vectors @ embedder.embed([q.text], cache=True)[0]
        semantic[~(snapshot.has_vector & allowed)] = -np.inf
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

    if q.filters.requires:  # passages that match every + clause, but none of the others
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

    return Hit(p.path, p.heading, p.start_line, p.end_line, score, focus, match, p.text, snippet, why, marks)
