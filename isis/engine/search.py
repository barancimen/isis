"""Typed search: BM25 over passages, or BM25 fused with embedding similarity ("hybrid").

Query syntax: plain words are ranked with BM25 (any word may match; passages with more and rarer
matches rank higher), "quoted phrases" must appear verbatim, and a partly typed last word is
expanded to the corpus terms it starts.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .index import Snapshot
from .params import tunable
from .text import term_spans, tokenize_with_surface

QUOTED = re.compile(r'"([^"]+)"')
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
        0.75, label="b", group=KEYWORDS, min=0.0, max=1.0, step=0.05,
        help="How much long passages are penalised. 0: length ignored; 1: fully normalised.",
    )
    rrf_k: int = tunable(
        10, label="Fusion k", group=FUSION, min=1, max=100, step=1,
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


@dataclass
class Query:
    term_ids: list[int]
    weights: list[float]
    terms: set[str]  # for highlighting
    phrases: list[str]


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def expand_prefix(snapshot: Snapshot, prefix: str, limit: int = MAX_EXPANSIONS) -> list[str]:
    """Corpus terms starting with `prefix`, most common first."""
    terms = snapshot.sorted_terms
    lo = bisect.bisect_left(terms, prefix)
    hi = bisect.bisect_left(terms, prefix + "￿")
    found = terms[lo:hi]
    df = snapshot.bm25.doc_freq
    return sorted(found, key=lambda t: -df[snapshot.term_id(t)])[:limit]


def parse_query(snapshot: Snapshot, query: str) -> Query:
    phrases = [p.strip() for p in QUOTED.findall(query) if p.strip()]
    pairs = tokenize_with_surface(query.replace('"', " "))
    still_typing = bool(query) and not query[-1].isspace() and query[-1] != '"'

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
    return Query(list(weights), list(weights.values()), terms, phrases)


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
    lexical = snapshot.bm25_for(config.k1, config.b).scores(q.term_ids, q.weights)

    if mode == "hybrid" and embedder is not None and snapshot.has_vector.any():
        semantic = snapshot.vectors @ embedder.embed([query], cache=True)[0]
        semantic[~snapshot.has_vector] = -np.inf
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

    if q.phrases:
        phrases = [_norm(p) for p in q.phrases]
        ranked = [(r, s) for r, s in ranked if all(p in _norm(snapshot.passages[r].text) for p in phrases)]

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
            hits.append(make_hit(snapshot, row, score, q.terms, lines_cache[path], context_lines, why(row)))
            if len(hits) >= limit:
                break
    return hits


def make_hit(
    snapshot: Snapshot,
    row: int,
    score: float,
    terms: set[str],
    lines: list[str],
    context_lines: int,
    why: dict | None = None,
) -> Hit:
    p = snapshot.passages[row]
    last_line = len(lines) - 1
    start, end = min(p.start_line, last_line), min(p.end_line, last_line)

    match = None
    for n in range(start, end + 1):
        spans = term_spans(lines[n], terms)
        if spans:
            match = (n, spans[0][0], spans[0][1] - spans[0][0])
            break
    focus = match[0] if match else start

    core_start = max(start, min(focus - 2, end - SNIPPET_LINES + 1))
    core_end = min(end, core_start + SNIPPET_LINES - 1)
    snippet = []
    for n in range(max(0, core_start - context_lines), min(last_line, core_end + context_lines) + 1):
        inside = start <= n <= end
        snippet.append(SnippetLine(n, lines[n], inside, term_spans(lines[n], terms) if inside else []))

    return Hit(p.path, p.heading, p.start_line, p.end_line, score, focus, match, p.text, snippet, why or {})
