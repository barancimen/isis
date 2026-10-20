"""Typed search: BM25 over passages, or BM25 fused with embedding similarity ("hybrid").

Query syntax:
    words          ranked with BM25: any word may match; more and rarer matches rank higher.
                   A partly typed last word is expanded to the corpus terms it starts.
    "a phrase"     must appear verbatim (case and spacing ignored).
    /regex/        must match (Python syntax; case-insensitive unless it has a capital letter).
    -word          must not appear, in any form that shares its stem (-whale also drops "whales").
    -"a phrase"    must not appear.
    -/regex/       must not match.
The constraints filter every passage before ranking, in both modes, so similarity of meaning
never brings back a passage they exclude.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .index import Snapshot
from .params import tunable
from .text import term_spans, tokenize, tokenize_with_surface

# One query token: optional "-", then a "phrase" (its closing quote may still be missing), a
# /regex/ that ends at a space or the end, a /regex still being typed, or a plain word.
TOKEN = re.compile(r'''(-?)(?:"([^"]*)(?:"|$)|/((?:\\.|[^/\\])+)/(?=\s|$)|(/(?:\\.|[^/\\])*)$|(\S+))''')
MIN_PREFIX = 3
MAX_EXPANSIONS = 20
SNIPPET_LINES = 8  # passage lines shown per hit, centred on the first match

KEYWORDS = "Keywords (BM25)"
FUSION = "Fusion (when 'meaning' is ticked)"


@dataclass
class SearchConfig:
    k1: float = tunable(
        1.2, label="k1", group=KEYWORDS, min=0.0, max=3.0, step=0.1,
        help="How fast repeated words stop adding to a passage's score. 0: one occurrence counts as much as many.",
    )
    b: float = tunable(
        0.35, label="b", group=KEYWORDS, min=0.0, max=1.0, step=0.05,
        help="How much long passages are penalised. 0: length ignored; 1: fully normalised.",
    )
    rrf_k: int = tunable(
        20, label="Fusion k", group=FUSION, min=1, max=100, step=1,
        help="Reciprocal-rank fusion: each list adds weight / (k + rank). Small k keeps the best of each "
        "list on top; large k favours passages ranked fairly well in both.",
    )
    keyword_weight: float = tunable(
        1.0, label="Keyword weight", group=FUSION, min=0.0, max=3.0, step=0.1,
        help="Weight of the keyword ranking in the fusion. 0: rank by meaning only.",
    )
    meaning_weight: float = tunable(
        1.0, label="Meaning weight", group=FUSION, min=0.0, max=3.0, step=0.1,
        help="Weight of the embedding ranking in the fusion. 0: rank by keywords only.",
    )
    candidates: int = tunable(
        300, label="Candidates per list", group=FUSION, min=20, max=2000, step=10,
        help="How deep each ranking goes before fusing. A passage outside both lists can't be found.",
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


@dataclass
class Filters:
    """Hard constraints on passages: patterns that must match, and terms and patterns that must not."""

    required: list[re.Pattern] = field(default_factory=list)
    excluded_terms: set[str] = field(default_factory=set)  # stems: a passage with any of them is out
    excluded: list[re.Pattern] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.required or self.excluded_terms or self.excluded)

    def exclude(self, kind: str, value: str) -> None:
        """Exclude a word (by its stem), a phrase or a regex, as written in a query."""
        if kind == "regex":
            self.excluded.append(compile_pattern(value))
        elif kind == "phrase":
            if value.split():
                self.excluded.append(phrase_pattern(value))
        else:
            stems = tokenize(value)
            if len(stems) == 1:
                self.excluded_terms.add(stems[0])
            elif value.strip("-"):  # a stopword or a hyphenated compound: match it as written
                self.excluded.append(re.compile(rf"(?<!\w){re.escape(value.strip('-'))}(?!\w)", re.IGNORECASE))

    def allowed(self, snapshot: Snapshot, rows=None) -> np.ndarray:
        """Which of `rows` (default: every passage) pass, as a boolean array aligned with them."""
        rows = np.arange(len(snapshot)) if rows is None else np.asarray(rows, np.int64)
        ok = np.ones(len(rows), bool)
        if self.excluded_terms:
            out = np.zeros(len(snapshot), bool)
            for term in self.excluded_terms:
                tid = snapshot.term_id(term)
                if tid is not None:
                    out[snapshot.bm25.matrix[:, tid].indices] = True
            ok &= ~out[rows]
        if self.required or self.excluded:
            for i in np.flatnonzero(ok):
                text = snapshot.passages[rows[i]].text
                ok[i] = all(p.search(text) for p in self.required) and not any(p.search(text) for p in self.excluded)
        return ok


def exclusion_filters(entries: list[str]) -> Filters:
    """Filters for a list of things never to show, each written as in a query: word, "phrase", /regex/."""
    filters = Filters()
    for entry in entries:
        for negated, kind, value in split_query(entry.strip().lstrip("-")):
            filters.exclude(kind, value)
    return filters


def split_query(query: str) -> list[tuple[bool, str, str]]:
    """(negated, kind, value) per token; kind is "word", "phrase", "regex" or "partial" (a regex
    still being typed, which is ignored)."""
    out = []
    for m in TOKEN.finditer(query):
        negated, phrase, regex, partial, word = m.groups()
        if phrase is not None:
            out.append((bool(negated), "phrase", phrase))
        elif regex is not None:
            out.append((bool(negated), "regex", regex))
        elif partial is not None:
            out.append((bool(negated), "partial", partial))
        elif word and word != "-":
            out.append((bool(negated), "word", word))
    return out


@dataclass
class Query:
    term_ids: list[int]
    weights: list[float]
    terms: set[str]  # for highlighting
    filters: Filters
    text: str  # what the meaning side embeds: the positive words and phrases
    highlight: list[re.Pattern] = field(default_factory=list)  # required phrases and patterns


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def expand_prefix(snapshot: Snapshot, prefix: str, limit: int = MAX_EXPANSIONS) -> list[str]:
    """Corpus terms starting with `prefix`, most common first."""
    terms = snapshot.sorted_terms
    lo = bisect.bisect_left(terms, prefix)
    hi = bisect.bisect_left(terms, prefix + "\uffff")
    found = terms[lo:hi]
    df = snapshot.bm25.doc_freq
    return sorted(found, key=lambda t: -df[snapshot.term_id(t)])[:limit]


def parse_query(snapshot: Snapshot, query: str) -> Query:
    tokens = split_query(query)
    filters = Filters()
    positive: list[str] = []  # words and phrases, in order: the BM25 and meaning query
    for negated, kind, value in tokens:
        if kind == "partial":
            continue
        if negated:
            filters.exclude(kind, value)
        elif kind == "phrase":
            if value.split():
                filters.required.append(phrase_pattern(value))
            positive.append(value)
        elif kind == "regex":
            filters.required.append(compile_pattern(value))
        else:
            positive.append(value)

    last = tokens[-1] if tokens else None
    still_typing = bool(query) and not query[-1].isspace() and last is not None and last[:2] == (False, "word")
    pairs = tokenize_with_surface(" ".join(positive))
    weights: dict[int, float] = {}
    terms: set[str] = set()
    for i, (term, surface) in enumerate(pairs):
        tid = snapshot.term_id(term)
        if tid is not None:
            weights[tid] = weights.get(tid, 0.0) + 1.0
            terms.add(term)
        elif still_typing and i == len(pairs) - 1 and len(surface) >= MIN_PREFIX:
            for t in expand_prefix(snapshot, surface):
                weights.setdefault(snapshot.term_id(t), 1.0)
                terms.add(t)
    return Query(list(weights), list(weights.values()), terms, filters, " ".join(positive), list(filters.required))


def top_rows(scores: np.ndarray, k: int, positive: bool = False) -> np.ndarray:
    """Row indices of the k highest scores, best first."""
    rows = np.flatnonzero(scores > 0) if positive else np.flatnonzero(np.isfinite(scores))
    if len(rows) > k:
        rows = rows[np.argpartition(-scores[rows], k)[:k]]
    return rows[np.argsort(-scores[rows], kind="stable")]


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
) -> list[Hit]:
    config = config or SearchConfig()
    if not len(snapshot) or not query.strip():
        return []
    q = parse_query(snapshot, query)
    if not q.term_ids and not q.filters.required:
        return []  # only exclusions, or a regex still being typed: nothing to look for
    allowed = q.filters.allowed(snapshot) if q.filters else np.ones(len(snapshot), bool)
    lexical = snapshot.bm25_for(config.k1, config.b).scores(q.term_ids, q.weights)
    lexical = np.where(allowed, lexical, 0.0)

    if mode == "hybrid" and q.text.strip() and embedder is not None and snapshot.has_vector.any():
        semantic = snapshot.vectors @ embedder.embed([q.text], cache=True)[0]
        semantic[~(snapshot.has_vector & allowed)] = -np.inf
        lex_rank = {int(row): rank for rank, row in enumerate(top_rows(lexical, config.candidates, positive=True))}
        sem_rank = {int(row): rank for rank, row in enumerate(top_rows(semantic, config.candidates))}
        fused: dict[int, float] = {}
        for ranks, weight in ((lex_rank, config.keyword_weight), (sem_rank, config.meaning_weight)):
            for row, rank in ranks.items():
                fused[row] = fused.get(row, 0.0) + weight / (config.rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        why = lambda row: {
            "bm25": float(lexical[row]),
            "cosine": float(semantic[row]),
            "keyword_rank": lex_rank[row] + 1 if row in lex_rank else None,  # None: not among the top keyword hits
            "meaning_rank": sem_rank[row] + 1 if row in sem_rank else None,
        }
    else:
        rows = top_rows(lexical, len(lexical), positive=True)
        ranked = [(int(r), float(lexical[r])) for r in rows]
        why = lambda row: {"bm25": float(lexical[row])}
    if q.filters.required:  # passages that match the phrases or patterns without any of the words
        seen = {r for r, _ in ranked}
        ranked += [(int(r), 0.0) for r in np.flatnonzero(allowed) if r not in seen]

    lines_cache: dict[str, list[str] | None] = {}
    hits = []
    for row, score in ranked:
        path = snapshot.passages[row].path
        if path not in lines_cache:
            try:
                lines_cache[path] = read_lines(path)
            except (OSError, ValueError):
                lines_cache[path] = None  # deleted since indexing; the next refresh drops it
        if lines_cache[path]:
            hits.append(make_hit(snapshot, row, score, q.terms, lines_cache[path], context_lines, why(row), q.highlight))
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


def make_hit(
    snapshot: Snapshot,
    row: int,
    score: float,
    terms: set[str],
    lines: list[str],
    context_lines: int,
    why: dict | None = None,
    patterns: list[re.Pattern] = (),
) -> Hit:
    p = snapshot.passages[row]
    last_line = len(lines) - 1
    start, end = min(p.start_line, last_line), min(p.end_line, last_line)
    patterns = list(patterns)

    match = None
    for n in range(start, end + 1):
        spans = _spans(lines[n], terms, patterns)
        if spans:
            match = (n, spans[0][0], spans[0][1] - spans[0][0])
            break
    focus = match[0] if match else start

    core_start = max(start, min(focus - 2, end - SNIPPET_LINES + 1))
    core_end = min(end, core_start + SNIPPET_LINES - 1)
    snippet = []
    for n in range(max(0, core_start - context_lines), min(last_line, core_end + context_lines) + 1):
        inside = start <= n <= end
        snippet.append(SnippetLine(n, lines[n], inside, _spans(lines[n], terms, patterns) if inside else []))

    return Hit(p.path, p.heading, p.start_line, p.end_line, score, focus, match, p.text, snippet, why or {})
