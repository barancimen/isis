"""Related passages for a task document: the task itself is the query.

1. units    Split the task into paragraphs ("units") and recognise the ones that are excerpts
            already collected from the corpus. Headings (or clusters of paragraphs) are themes.
2. signals  Score every corpus passage against the whole document, two ways:
              semantic  robust z-score of its cosine similarity to the document vector (the
                        length-weighted mean of the paragraph embeddings): how far it stands out
                        from the typical passage (median, and median absolute deviation), so
                        thresholds mean the same for every task and every model;
              lexical   BM25 for the document's most distinctive terms (TF-IDF), with the
                        user's pinned and banned terms applied.
            With query="paragraphs" each paragraph is a query of its own instead, and a passage
            scores by its best match (optionally weighted toward the cursor).
3. gate     Keep only clearly relevant passages: a strong semantic z-score, or a top lexical hit
            with some semantic support. Drop what is already collected or was dismissed, what
            contains an excluded word, phrase or pattern, and what is close to an avoided passage.
4. lanes    Rank the survivors four ways:
              core      reciprocal-rank fusion of both signals
              hidden    high meaning, low keyword overlap: the same idea in other words
              bridges   strongly related to two themes of the task that rarely meet
              gaps      related to the task's thinly covered themes
5. select   Fill each lane with MMR (relevance minus redundancy with what is already chosen or
            collected), a few per file at most. Slots per lane come from Thompson sampling over
            the user's add/dismiss feedback.

Every step is a plain function over numpy arrays, and every knob lives in RelatedConfig, so the
pipeline can be taken apart and tuned from a notebook.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .chunker import MAX_WORDS, chunk_markdown
from .index import Snapshot
from .params import tunable
from .search import Hit, exclusion_filters, make_hit, top_rows
from .text import tokenize, tokenize_with_surface

LANES = ("core", "hidden", "bridges", "gaps")
LANE_INFO = {
    "core": ("Core", "Strongest matches on both words and meaning"),
    "hidden": ("Hidden connections", "Close in meaning, with little keyword overlap"),
    "bridges": ("Bridges", "Strongly related to two of your themes that rarely meet"),
    "gaps": ("Gaps", "For the thinly covered themes of your task"),
}


EMPTY_NOTE = {
    "core": "Nothing in the corpus is clearly related yet.",
    "hidden": "Nothing close in meaning without also sharing the task's keywords.",
    "bridges": "No passage relates clearly to two of your themes at once.",
    "gaps": "Nothing clearly related to your thinly covered themes.",
}


QUERY, KEYWORDS, FILTER, LANES_GROUP, SELECTION = "Query", "Keywords (BM25)", "Relevance filter", "Lanes", "Selection"


@dataclass
class RelatedConfig:
    query: str = tunable(
        "document", label="Query", group=QUERY, choices=["document", "paragraphs"],
        help="document: the whole task is one query. paragraphs: each paragraph is a query, and a passage "
        "scores by its best match (strong matches to any one topic, not to their overlap).",
    )
    n_query_terms: int = tunable(
        30, label="Keywords taken", group=QUERY, min=1, max=60, step=1,
        help="How many of the task's most distinctive words (TF-IDF) make up the keyword query.",
    )
    excerpt_weight: float = tunable(
        0.5, label="Excerpt weight", group=QUERY, min=0.0, max=3.0, step=0.1,
        help="Weight of passages you collected from the corpus, relative to your own writing.",
    )
    cursor_boost: float = 1.0  # > 1 weights the paragraph at cursor_line up (1 = the whole document counts equally)
    cursor_decay: float = 1.0  # paragraphs over which that boost fades (1/e per this many)
    bm25_k1: float = tunable(
        1.2, label="k1", group=KEYWORDS, min=0.0, max=3.0, step=0.1,
        help="How fast repeated words stop adding to a passage's keyword score.",
    )
    bm25_b: float = tunable(
        0.35, label="b", group=KEYWORDS, min=0.0, max=1.0, step=0.05,
        help="How much long passages are penalised. 0: length ignored; 1: fully normalised.",
    )
    candidates: int = tunable(
        300, label="Candidates per signal", group=FILTER, min=20, max=2000, step=10,
        help="How deep the keyword and meaning rankings go before the filter.",
    )
    z_strong: float = tunable(
        3.25, label="Meaning alone", group=FILTER, min=0.0, max=5.0, step=0.1,
        help="Meaning z-score that gets a passage through the filter on its own.",
    )
    z_weak: float = tunable(
        1.75, label="Meaning with keywords", group=FILTER, min=-1.0, max=4.0, step=0.1,
        help="Meaning z-score that is enough for one of the top keyword matches.",
    )
    lexical_top: int = tunable(
        50, label="Top keyword ranks", group=FILTER, min=0, max=300, step=1,
        help="How many of the best keyword matches count as top keyword matches.",
    )
    duplicate_cos: float = tunable(
        0.95, label="Duplicate similarity", group=FILTER, min=0.7, max=1.0, step=0.01,
        help="A passage this similar to a paragraph of the task counts as already collected.",
    )
    negative_weight: float = tunable(
        0.5, label="Dismissal push", group=FILTER, min=0.0, max=2.0, step=0.1,
        help="How strongly passages similar to ones you dismissed are pushed down.",
    )
    avoid_z: float = tunable(
        3.0, label="Avoid: similarity", group=FILTER, min=0.5, max=6.0, step=0.1,
        help="Passages you avoid (⊘) also hide every passage at least this similar to them (z-score). "
        "Lower hides more.",
    )
    rrf_k: int = tunable(
        30, label="Core fusion k", group=LANES_GROUP, min=1, max=100, step=1,
        help="Core ranks by reciprocal-rank fusion. Large k favours passages good on both signals; small k "
        "lets the best of either signal through.",
    )
    hidden_z: float = tunable(
        2.5, label="Hidden: meaning", group=LANES_GROUP, min=0.0, max=5.0, step=0.1,
        help="Meaning z-score a hidden connection needs.",
    )
    hidden_max_obviousness: float = tunable(
        0.3, label="Hidden: max overlap", group=LANES_GROUP, min=0.0, max=1.0, step=0.05,
        help="A hidden connection's keyword score may be at most this share of the best keyword match.",
    )
    bridge_z: float = tunable(
        1.5, label="Bridges: per theme", group=LANES_GROUP, min=0.0, max=4.0, step=0.1,
        help="Meaning z-score a bridge needs on each of two themes (also lets it through the filter).",
    )
    gap_z: float = tunable(
        2.5, label="Gaps: meaning", group=LANES_GROUP, min=0.0, max=5.0, step=0.1,
        help="Meaning z-score a passage needs on a thinly covered theme.",
    )
    theme_distance: float = tunable(
        0.2, label="Theme split", group=LANES_GROUP, min=0.05, max=1.0, step=0.05,
        help="Without headings, paragraphs further apart than this (cosine distance) become separate themes.",
    )
    total_slots: int = tunable(
        24, label="Results", group=SELECTION, min=4, max=80, step=1,
        help="Results across all lanes.",
    )
    min_slots: int = tunable(
        2, label="Minimum per lane", group=SELECTION, min=0, max=10, step=1,
        help="Slots every non-empty lane gets before the rest is shared out by feedback.",
    )
    max_per_file: int = tunable(
        2, label="Per file, per lane", group=SELECTION, min=1, max=20, step=1,
        help="At most this many results from one file in one lane.",
    )
    mmr_lambda: dict = tunable(
        label="Relevance vs variety", group=SELECTION, min=0.0, max=1.0, step=0.05,
        keys=["core", "hidden", "bridges", "gaps"],
        default_factory=lambda: {"core": 0.7, "hidden": 0.6, "bridges": 0.6, "gaps": 0.6},
        help="MMR: 1 ranks by relevance only; lower values prefer results unlike the ones already shown "
        "and the passages you collected.",
    )
    mmr_pool: int = 100
    lane_prior: dict = field(
        default_factory=lambda: {"core": (3.0, 1.0), "hidden": (1.0, 1.0), "bridges": (1.0, 1.0), "gaps": (1.0, 1.0)}
    )


@dataclass
class TaskPrefs:
    """What the user has told us about one task. Stored next to the task document."""

    pins: list[str] = field(default_factory=list)
    bans: list[str] = field(default_factory=list)
    dismissed: list[dict] = field(default_factory=list)  # {"path", "text", "lane", "start_line", "end_line", "avoid"?}
    feedback: dict = field(default_factory=dict)  # lane -> [adds, dismisses]
    excluded: list[str] = field(default_factory=list)  # never shown: words, "phrases" and /regexes/

    @classmethod
    def from_dict(cls, data: dict | None) -> "TaskPrefs":
        data = data or {}
        return cls(**{k: data[k] for k in ("pins", "bans", "dismissed", "feedback", "excluded") if k in data})

    def to_dict(self) -> dict:
        return asdict(self)

    def record(self, lane: str, added: bool) -> None:
        counts = self.feedback.setdefault(lane, [0, 0])
        counts[0 if added else 1] += 1

    def dismiss(self, lane: str, path: str, text: str, start_line: int | None = None, end_line: int | None = None,
                avoid: bool = False) -> None:
        """Hide a passage. `avoid` also hides every passage close to it in meaning (an anti-query)."""
        self.record(lane, added=False)
        if text:
            entry = {"path": path, "text": text, "lane": lane, "start_line": start_line, "end_line": end_line}
            self.dismissed.append({**entry, "avoid": True} if avoid else entry)

    def restore(self, path: str, text: str) -> None:
        """Undo the dismissals of a passage: it may come back, and its lanes lose those dismissals."""
        key = (path, _norm(text))
        kept = []
        for d in self.dismissed:
            if (d.get("path"), _norm(d.get("text", ""))) != key:
                kept.append(d)
            elif d.get("lane") in self.feedback:
                counts = self.feedback[d["lane"]]
                counts[1] = max(counts[1] - 1, 0)
        self.dismissed = kept


@dataclass
class Unit:
    heading: str
    start_line: int
    end_line: int
    text: str
    weight: float = 1.0
    excerpt: bool = False
    source_rows: list[int] = field(default_factory=list)  # corpus passages it was taken from
    theme: int = 0


@dataclass
class TermChip:
    term: str  # stemmed, as used by the index
    label: str  # the word as it appears in the task
    weight: float
    pinned: bool = False
    banned: bool = False


@dataclass
class Theme:
    label: str
    units: int
    excerpts: int


@dataclass
class Lane:
    name: str
    title: str
    description: str
    items: list[Hit]
    note: str = ""


@dataclass
class RelatedResult:
    lanes: list[Lane]
    terms: list[TermChip]
    themes: list[Theme]
    stats: dict


# ---- 1. units ----


def task_units(text: str) -> list[Unit]:
    return [
        Unit(p.heading, p.start_line, p.end_line, p.text)
        for p in chunk_markdown("<task>", text, target_words=0, max_words=MAX_WORDS)
    ]


def weight_units(units: list[Unit], cursor_line: int | None, config: RelatedConfig) -> None:
    if cursor_line is None or not units:
        weights = np.ones(len(units))
    else:
        distance_in_lines = [
            0 if u.start_line <= cursor_line <= u.end_line else min(abs(cursor_line - u.start_line), abs(cursor_line - u.end_line))
            for u in units
        ]
        at = int(np.argmin(distance_in_lines))
        steps = np.abs(np.arange(len(units)) - at)
        weights = 1 + (config.cursor_boost - 1) * np.exp(-steps / config.cursor_decay)
    weights = weights * np.array([config.excerpt_weight if u.excerpt else 1.0 for u in units])
    for u, w in zip(units, weights / max(weights.max(), 1e-9) if len(units) else weights):
        u.weight = float(w)


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def find_excerpts(snapshot: Snapshot, units: list[Unit], cos: np.ndarray | None) -> None:
    """Mark units copied from the corpus, and remember which passages they came from.

    A unit is an excerpt when its text appears inside a corpus passage (or contains one). The
    places to look are the unit's best lexical and semantic matches."""
    for j, unit in enumerate(units):
        needle = _norm(unit.text)
        if len(needle) < 20:
            continue
        ids = [i for t in set(tokenize(unit.text)) if (i := snapshot.term_id(t)) is not None]
        rows = list(top_rows(snapshot.bm25.scores(ids, np.ones(len(ids))), 5, positive=True))
        if cos is not None:
            rows += list(top_rows(cos[:, j], 5))
        for row in dict.fromkeys(int(r) for r in rows):
            hay = _norm(snapshot.passages[row].text)
            if needle in hay or (len(hay) >= 20 and hay in needle):
                unit.excerpt = True
                unit.source_rows.append(row)


def assign_themes(units: list[Unit], unit_vecs: np.ndarray | None, config: RelatedConfig) -> list[Theme]:
    """Themes are the task's sections when it has at least two; otherwise clusters of similar
    paragraphs (average linkage on cosine distance)."""
    headings = list(dict.fromkeys(u.heading for u in units))
    if len(headings) >= 2:
        for u in units:
            u.theme = headings.index(u.heading)
        labels = [h or "(top)" for h in headings]
    elif unit_vecs is not None and len(units) >= 3:
        from scipy.cluster.hierarchy import fcluster, linkage

        clusters = fcluster(linkage(unit_vecs, "average", metric="cosine"), config.theme_distance, "distance")
        order = list(dict.fromkeys(clusters))  # number themes in reading order
        for u, c in zip(units, clusters):
            u.theme = order.index(c)
        labels = [_cluster_label([u for u in units if u.theme == t]) for t in range(len(order))]
    else:
        for u in units:
            u.theme = 0
        labels = [units[0].heading or "Everything"] if units else []

    return [
        Theme(label, sum(u.theme == t for u in units), sum(u.theme == t and u.excerpt for u in units))
        for t, label in enumerate(labels)
    ]


def _cluster_label(units: list[Unit]) -> str:
    counts = Counter(surface for u in units for _, surface in tokenize_with_surface(u.text))
    return " · ".join(w for w, _ in counts.most_common(3)) or "(untitled)"


# ---- 2. signals ----


def zscores(snapshot: Snapshot, vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity of every passage to each vector, and the same as robust z-scores:
    distance from the corpus median in units of (scaled) median absolute deviation. Unlike mean
    and standard deviation, these are not dragged around by a large on-topic cluster. Passages
    without embeddings get -inf."""
    cos = snapshot.vectors @ vecs.T
    has = snapshot.has_vector
    median = np.median(cos[has], axis=0)
    spread = 1.4826 * np.median(np.abs(cos[has] - median), axis=0) + 1e-6
    z = (cos - median) / spread
    z[~has] = -np.inf
    cos[~has] = -np.inf
    return cos, z


def task_terms(snapshot: Snapshot, units: list[Unit], prefs: TaskPrefs, config: RelatedConfig) -> list[TermChip]:
    """The task's most distinctive terms: weighted term frequency in the task x corpus IDF."""
    tf: dict[str, float] = defaultdict(float)
    surfaces: dict[str, Counter] = defaultdict(Counter)
    for u in units:
        for term, surface in tokenize_with_surface(u.text):
            tf[term] += u.weight
            surfaces[term][surface] += 1
    for heading in {u.heading for u in units if u.heading}:
        for term, surface in tokenize_with_surface(heading):
            tf[term] += 1.0
            surfaces[term][surface] += 1

    scored = []
    for term, f in tf.items():
        i = snapshot.term_id(term)
        if i is not None and snapshot.bm25.doc_freq[i] > 0:
            scored.append((f * float(snapshot.bm25.idf[i]), term))
    scored.sort(reverse=True)

    bans, pins = set(prefs.bans), set(prefs.pins)
    excluded = exclusion_filters(prefs.excluded).excluded_terms
    top = [(s, t) for s, t in scored if t not in bans and t not in excluded][: config.n_query_terms]
    best = top[0][0] if top else 1.0
    chosen = {t for _, t in top}
    for t in sorted(pins - chosen - excluded):  # pinned terms always take part, at full weight
        if snapshot.term_id(t) is not None:
            top.append((best, t))

    label = lambda t: surfaces[t].most_common(1)[0][0] if surfaces.get(t) else t
    chips = [TermChip(t, label(t), s / best, pinned=t in pins) for s, t in top]
    chips += [TermChip(t, label(t), 0.0, banned=True) for t in prefs.bans]
    return chips


def ranks(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """0 for the best score; len(scores) for rows that are not valid."""
    out = np.full(len(scores), len(scores), np.int64)
    rows = np.flatnonzero(valid)
    out[rows[np.argsort(-scores[rows], kind="stable")]] = np.arange(len(rows))
    return out


# ---- 5. select ----


def allocate_slots(lanes: list[str], prefs: TaskPrefs, config: RelatedConfig, seed: int) -> dict[str, int]:
    """Thompson sampling: draw each lane's add-rate from Beta(prior + adds, prior + dismisses)
    and share the slots in proportion. The seed only changes when feedback does, so the split
    stays put while you type."""
    if not lanes:
        return {}
    rng = np.random.default_rng(seed)
    draws = []
    for lane in lanes:
        a, b = config.lane_prior.get(lane, (1.0, 1.0))
        adds, dismisses = prefs.feedback.get(lane, [0, 0])
        draws.append(rng.beta(a + adds, b + dismisses))
    base = min(config.min_slots, config.total_slots // len(lanes))  # the total always wins
    spare = config.total_slots - base * len(lanes)
    shares = np.array(draws) / sum(draws) * spare
    slots = np.floor(shares).astype(int)
    for i in np.argsort(-(shares - slots))[: spare - slots.sum()]:  # largest remainders
        slots[i] += 1
    return {lane: base + int(n) for lane, n in zip(lanes, slots)}


def mmr_select(
    scores: dict[int, float],
    k: int,
    lam: float,
    snapshot: Snapshot,
    chosen_vecs: list[np.ndarray],
    taken: set[int],
    config: RelatedConfig,
    per_file: Counter | None = None,
) -> list[tuple[int, float]]:
    """Maximal marginal relevance: repeatedly take the passage with the best
    lam * relevance - (1 - lam) * (similarity to anything already chosen or collected)."""
    per_file = Counter() if per_file is None else per_file
    pool = [r for r in sorted(scores, key=lambda r: -scores[r]) if r not in taken][: config.mmr_pool]
    if not pool:
        return []
    best = max(max(scores[r] for r in pool), 1e-9)
    rel = np.array([scores[r] / best for r in pool])
    vecs = snapshot.vectors[pool]
    usable = snapshot.has_vector[pool]
    redundancy = np.zeros(len(pool))
    for v in chosen_vecs:
        redundancy = np.maximum(redundancy, np.where(usable, vecs @ v, 0.0))

    picked, alive = [], np.ones(len(pool), bool)
    while len(picked) < k and alive.any():
        value = np.where(alive, lam * rel - (1 - lam) * redundancy, -np.inf)
        i = int(np.argmax(value))
        alive[i] = False
        row = pool[i]
        path = snapshot.passages[row].path
        if per_file[path] >= config.max_per_file:
            continue
        per_file[path] += 1
        picked.append((row, float(value[i])))
        taken.add(row)
        if usable[i]:
            chosen_vecs.append(vecs[i])
            redundancy = np.maximum(redundancy, np.where(usable, vecs @ vecs[i], 0.0))
    return picked


# ---- the pipeline ----


def related(
    snapshot: Snapshot,
    task_text: str,
    *,
    embedder=None,
    cursor_line: int | None = None,
    prefs: TaskPrefs | None = None,
    config: RelatedConfig | None = None,
    seed: int = 0,
    read_lines: Callable[[str], list[str]],
    context_lines: int = 1,
) -> RelatedResult:
    prefs = prefs or TaskPrefs()
    config = config or RelatedConfig()
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(name):
        nonlocal clock
        now = time.perf_counter()
        timings[name] = round((now - clock) * 1000, 1)
        clock = now

    units = task_units(task_text)
    n = len(snapshot)
    if not units or not n:
        note = "Write or collect something in the task to get suggestions." if n else "The corpus is empty."
        return RelatedResult([Lane(l, *LANE_INFO[l], [], note) for l in LANES], [], [], {"passages": n})

    # 1-2: units and the semantic signal
    semantic = embedder is not None and snapshot.has_vector.any()
    unit_vecs = cos = z = None
    if semantic:
        unit_vecs = embedder.embed([u.heading + "\n" + u.text if u.heading else u.text for u in units], cache=True)
        cos, z = zscores(snapshot, unit_vecs)
    find_excerpts(snapshot, units, cos)
    weight_units(units, cursor_line, config)
    themes = assign_themes(units, unit_vecs, config)
    lap("units")

    weights = np.array([u.weight for u in units], np.float32)
    theme_z = second = avoided = None
    if semantic:
        if config.query == "document":
            sizes = np.array([len(u.text.split()) for u in units], np.float32) * weights
            document = (sizes[:, None] * unit_vecs).sum(0)
            _, z_doc = zscores(snapshot, (document / max(np.linalg.norm(document), 1e-9))[None, :])
            sem_any = sem_w = z_doc[:, 0]
            best_unit = np.where(np.isfinite(cos[:, 0]), cos.argmax(1), 0)  # closest paragraph, for display
        else:
            sem_any = z.max(1)
            sem_w = (z * weights).max(1)
            best_unit = (z * weights).argmax(1)
        dismissed = [d for d in prefs.dismissed if d.get("text")]
        if dismissed:
            _, z_neg = zscores(snapshot, embedder.embed([d["text"] for d in dismissed], cache=True))
            penalty = config.negative_weight * np.maximum(z_neg.max(1), 0)
            sem_any, sem_w = sem_any - penalty, sem_w - penalty
            avoid = [i for i, d in enumerate(dismissed) if d.get("avoid")]
            if avoid:  # anti-queries: everything clearly like them is out
                avoided = z_neg[:, avoid].max(1) >= config.avoid_z
        if len(themes) >= 2:  # each passage's z-score per theme, and its second-best theme's
            theme_of = np.array([u.theme for u in units])
            theme_z = np.stack([z[:, theme_of == t].max(1) for t in range(len(themes))], axis=1)
            second = np.sort(theme_z, axis=1)[:, -2]
    else:
        sem_any = sem_w = np.full(n, -np.inf)
        best_unit = np.zeros(n, np.int64)
    lap("semantic")

    chips = task_terms(snapshot, units, prefs, config)
    query = [(snapshot.term_id(c.term), c.weight) for c in chips if not c.banned]
    lex = snapshot.bm25_for(config.bm25_k1, config.bm25_b).scores([i for i, _ in query], [w for _, w in query])
    lex_rank = ranks(lex, lex > 0)
    sem_rank = ranks(sem_w, np.isfinite(sem_w))
    lap("lexical")

    # 3: gate. A passage gets in on strong meaning, on a top keyword match with some meaning
    # behind it, or (bridges) on being clearly related to two themes at once: two independent
    # themes at z >= bridge_z is about as unlikely by chance as one at z >= z_strong.
    has = snapshot.has_vector & semantic
    relevant = np.where(
        has,
        (sem_any >= config.z_strong) | ((lex_rank < config.lexical_top) & (sem_any >= config.z_weak)),
        lex_rank < config.lexical_top,  # no embedding for this passage (yet): keywords alone
    )
    if second is not None:
        relevant |= has & (second >= config.bridge_z)
    if semantic:
        relevant &= ~(has & (np.max(np.where(has[:, None], cos, -1.0), axis=1) >= config.duplicate_cos))

    candidates = set(top_rows(lex, config.candidates, positive=True)) | set(top_rows(sem_w, config.candidates))
    if second is not None:
        candidates |= set(top_rows(second, config.candidates))
    collected = {r for u in units for r in u.source_rows}
    dismissed_keys = {(d.get("path"), _norm(d.get("text", ""))) for d in prefs.dismissed}
    excerpts = [e for u in units if u.excerpt and len(e := _norm(u.text)) >= 20]

    rows = np.array(sorted(map(int, candidates)), np.int64)
    rows = rows[relevant[rows]]
    if avoided is not None:
        rows = rows[~avoided[rows]]
    exclusions = exclusion_filters(prefs.excluded)
    if exclusions:
        rows = rows[exclusions.allowed(snapshot, rows)]

    gated = []
    for row in map(int, rows):
        if row in collected:
            continue
        p = snapshot.passages[row]
        if dismissed_keys or excerpts:
            hay = _norm(p.text)
            if (p.path, hay) in dismissed_keys or any(e in hay or hay in e for e in excerpts):
                continue
        gated.append(row)
    lap("gate")

    # 4: lanes
    lane_scores: dict[str, dict[int, float]] = {lane: {} for lane in LANES}
    lane_why: dict[str, dict[int, dict]] = {lane: {} for lane in LANES}
    notes: dict[str, str] = {}
    lex_best = float(lex.max()) if n else 0.0

    for row in gated:
        lane_scores["core"][row] = 1 / (config.rrf_k + lex_rank[row] + 1) + 1 / (config.rrf_k + sem_rank[row] + 1)

    if not semantic:
        for lane in ("hidden", "bridges", "gaps"):
            notes[lane] = "Needs embeddings (still indexing, or running without a model)."
    else:
        for row in gated:
            obviousness = float(lex[row] / lex_best) if lex_best > 0 else 0.0
            if sem_any[row] >= config.hidden_z and obviousness <= config.hidden_max_obviousness:
                lane_scores["hidden"][row] = float(sem_w[row]) * (1 - obviousness)
                lane_why["hidden"][row] = {"obviousness": round(obviousness, 3)}

        if theme_z is None:
            notes["bridges"] = notes["gaps"] = "Needs at least two themes: add headings, or write about more than one thing."
        else:
            strong = theme_z >= config.bridge_z
            size = strong.sum(0)
            both = strong.T.astype(np.int32) @ strong.astype(np.int32)
            overlap = both / np.maximum(np.minimum.outer(size, size), 1)  # how often two themes meet
            need = np.array([1 / (1 + t.excerpts) for t in themes])

            for row in gated:
                order = np.argsort(-theme_z[row])
                f1, f2 = int(order[0]), int(order[1])
                if theme_z[row, f2] >= config.bridge_z:
                    rarity = 1 - overlap[f1, f2]
                    lane_scores["bridges"][row] = float(theme_z[row, f2]) * (0.5 + 0.5 * rarity)
                    lane_why["bridges"][row] = {
                        "themes": [themes[f1].label, themes[f2].label],
                        "theme_z": [round(float(theme_z[row, f1]), 2), round(float(theme_z[row, f2]), 2)],
                        "rarity": round(float(rarity), 3),
                    }
                gap = np.where(theme_z[row] >= config.gap_z, need * theme_z[row], -np.inf)
                if np.isfinite(gap.max()):
                    f = int(gap.argmax())
                    lane_scores["gaps"][row] = float(gap[f])
                    lane_why["gaps"][row] = {"themes": [themes[f].label], "collected_in_theme": themes[f].excerpts}
    lap("lanes")

    # 5: select
    active = [lane for lane in LANES if lane_scores[lane]]
    feedback_events = sum(sum(v) for v in prefs.feedback.values())
    slots = allocate_slots(active, prefs, config, seed + feedback_events)
    taken: set[int] = set()
    chosen_vecs = [unit_vecs[j] for j, u in enumerate(units) if u.excerpt] if semantic else []
    picked: dict[str, list[tuple[int, float]]] = {}
    per_file = {lane: Counter() for lane in active}
    for lane in active:
        lam = config.mmr_lambda.get(lane, 0.7)
        picked[lane] = mmr_select(lane_scores[lane], slots[lane], lam, snapshot, chosen_vecs, taken, config, per_file[lane])
    if "core" in picked:  # slots other lanes could not fill go to core
        spare = sum(slots.values()) - sum(len(v) for v in picked.values())
        if spare > 0:
            lam = config.mmr_lambda.get("core", 0.7)
            picked["core"] += mmr_select(
                lane_scores["core"], spare, lam, snapshot, chosen_vecs, taken, config, per_file["core"]
            )
    lap("select")

    terms = {c.term for c in chips if not c.banned}
    lines_cache: dict[str, list[str] | None] = {}

    def hit(lane: str, row: int, value: float) -> Hit | None:
        path = snapshot.passages[row].path
        if path not in lines_cache:
            try:
                lines_cache[path] = read_lines(path)
            except (OSError, ValueError):
                lines_cache[path] = None
        if not lines_cache[path]:
            return None
        u = units[int(best_unit[row])]
        why = {
            "lane_score": round(lane_scores[lane][row], 4),
            "mmr": round(value, 4),
            "semantic_z": round(float(sem_any[row]), 2) if np.isfinite(sem_any[row]) else None,
            "bm25": round(float(lex[row]), 3),
            "lexical_rank": int(lex_rank[row]) + 1 if lex[row] > 0 else None,
            "matches_task_line": u.start_line if semantic else None,
            **lane_why[lane].get(row, {}),
        }
        return make_hit(snapshot, row, lane_scores[lane][row], terms, lines_cache[path], context_lines, why)

    lanes = []
    for lane in LANES:
        items = [h for row, value in picked.get(lane, []) if (h := hit(lane, row, value)) is not None]
        note = notes.get(lane, "" if items or lane in active else EMPTY_NOTE[lane])
        lanes.append(Lane(lane, *LANE_INFO[lane], items, note))
    lap("snippets")

    return RelatedResult(
        lanes=lanes,
        terms=chips,
        themes=themes,
        stats={
            "passages": n,
            "units": len(units),
            "excerpts": sum(u.excerpt for u in units),
            "candidates": len(candidates),
            "gated": len(gated),
            "slots": slots,
            "ms": timings,
        },
    )
