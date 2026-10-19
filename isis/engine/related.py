"""Related passages for a task document: the task itself is the query.

1. units    Split the task into paragraphs ("units"). Weight them by distance from the cursor
            and recognise the ones that are excerpts already collected from the corpus.
2. signals  Score every corpus passage two ways:
              semantic  robust z-score of its cosine similarity to each unit: how far it
                        stands out from the typical passage (median, and median absolute
                        deviation), so thresholds mean the same for every unit and every model;
              lexical   BM25 for the task's most distinctive terms (TF-IDF), with the user's
                        pinned and banned terms applied.
3. gate     Keep only clearly relevant passages: a strong semantic z-score, or a top lexical hit
            with some semantic support. Drop what is already collected or was dismissed.
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
from .search import Hit, make_hit, top_rows
from .text import tokenize, tokenize_with_surface

LANES = ("core", "hidden", "bridges", "gaps")
LANE_INFO = {
    "core": ("Core", "Strongest matches on both words and meaning"),
    "hidden": ("Hidden connections", "Close in meaning, with little keyword overlap"),
    "bridges": ("Bridges", "Strongly related to two of your themes that rarely meet"),
    "gaps": ("Gaps", "For the thinly covered themes of your task"),
}


@dataclass
class RelatedConfig:
    n_query_terms: int = 20  # lexical query = the task's top TF-IDF terms
    cursor_boost: float = 3.0  # weight of the paragraph at the cursor relative to distant ones
    cursor_decay: float = 1.0  # paragraphs over which that boost fades (1/e per this many)
    excerpt_weight: float = 1.0  # weight of collected excerpts relative to your own writing
    candidates: int = 300  # per signal
    z_strong: float = 2.5  # gate: semantic z-score that is enough on its own
    z_weak: float = 1.0  # gate: semantic z-score that is enough for a top lexical hit
    lexical_top: int = 30  # ... where "top lexical hit" means this rank or better
    duplicate_cos: float = 0.95  # this similar to a task paragraph counts as already collected
    negative_weight: float = 0.5  # push away from dismissed passages (Rocchio-style)
    rrf_k: int = 60
    hidden_z: float = 3.0  # stricter than the gate: a small model's similarities are noisy
    hidden_max_obviousness: float = 0.3  # BM25 relative to the best lexical hit
    bridge_z: float = 2.0  # both themes must reach this
    gap_z: float = 2.5
    theme_distance: float = 0.5  # cosine distance cut when clustering paragraphs into themes
    total_slots: int = 24
    min_slots: int = 2  # per non-empty lane
    max_per_file: int = 2  # per lane
    mmr_pool: int = 100
    mmr_lambda: dict = field(default_factory=lambda: {"core": 0.7, "hidden": 0.6, "bridges": 0.6, "gaps": 0.6})
    lane_prior: dict = field(
        default_factory=lambda: {"core": (3.0, 1.0), "hidden": (1.0, 1.0), "bridges": (1.0, 1.0), "gaps": (1.0, 1.0)}
    )


@dataclass
class TaskPrefs:
    """What the user has told us about one task. Stored next to the task document."""

    pins: list[str] = field(default_factory=list)
    bans: list[str] = field(default_factory=list)
    dismissed: list[dict] = field(default_factory=list)  # {"path", "text"}
    feedback: dict = field(default_factory=dict)  # lane -> [adds, dismisses]

    @classmethod
    def from_dict(cls, data: dict | None) -> "TaskPrefs":
        data = data or {}
        return cls(**{k: data[k] for k in ("pins", "bans", "dismissed", "feedback") if k in data})

    def to_dict(self) -> dict:
        return asdict(self)

    def record(self, lane: str, added: bool) -> None:
        counts = self.feedback.setdefault(lane, [0, 0])
        counts[0 if added else 1] += 1


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
    top = [(s, t) for s, t in scored if t not in bans][: config.n_query_terms]
    best = top[0][0] if top else 1.0
    chosen = {t for _, t in top}
    for t in sorted(pins - chosen):  # pinned terms always take part, at full weight
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
    spare = max(0, config.total_slots - config.min_slots * len(lanes))
    shares = np.array(draws) / sum(draws) * spare
    slots = np.floor(shares).astype(int)
    for i in np.argsort(-(shares - slots))[: spare - slots.sum()]:  # largest remainders
        slots[i] += 1
    return {lane: config.min_slots + int(n) for lane, n in zip(lanes, slots)}


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
    if semantic:
        sem_any = z.max(1)
        sem_w = (z * weights).max(1)
        best_unit = (z * weights).argmax(1)
        dismissed = [d["text"] for d in prefs.dismissed if d.get("text")]
        if dismissed:
            _, z_neg = zscores(snapshot, embedder.embed(dismissed, cache=True))
            penalty = config.negative_weight * np.maximum(z_neg.max(1), 0)
            sem_any, sem_w = sem_any - penalty, sem_w - penalty
    else:
        sem_any = sem_w = np.full(n, -np.inf)
        best_unit = np.zeros(n, np.int64)
    lap("semantic")

    chips = task_terms(snapshot, units, prefs, config)
    query = [(snapshot.term_id(c.term), c.weight) for c in chips if not c.banned]
    lex = snapshot.bm25.scores([i for i, _ in query], [w for _, w in query])
    lex_rank = ranks(lex, lex > 0)
    sem_rank = ranks(sem_w, np.isfinite(sem_w))
    lap("lexical")

    # 3: gate
    candidates = set(top_rows(lex, config.candidates, positive=True)) | set(top_rows(sem_w, config.candidates))
    collected = {r for u in units for r in u.source_rows}
    dismissed_keys = {(d.get("path"), _norm(d.get("text", ""))) for d in prefs.dismissed}
    excerpts = [_norm(u.text) for u in units if u.excerpt]
    has = snapshot.has_vector

    gated = []
    for row in map(int, candidates):
        p = snapshot.passages[row]
        if row in collected or (p.path, _norm(p.text)) in dismissed_keys:
            continue
        if semantic and has[row] and cos[row].max() >= config.duplicate_cos:
            continue
        hay = _norm(p.text)
        if any(len(e) >= 20 and (e in hay or hay in e) for e in excerpts):
            continue
        if has[row] and semantic:
            ok = sem_any[row] >= config.z_strong or (lex_rank[row] < config.lexical_top and sem_any[row] >= config.z_weak)
        else:  # no embedding for this passage (yet): lexical evidence alone
            ok = lex_rank[row] < config.lexical_top
        if ok:
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

        if len(themes) < 2:
            notes["bridges"] = notes["gaps"] = "Needs at least two themes: add headings, or write about more than one thing."
        else:
            theme_of = np.array([u.theme for u in units])
            theme_z = np.stack([z[:, theme_of == t].max(1) for t in range(len(themes))], axis=1)
            strong = theme_z >= config.bridge_z
            size = strong.sum(0)
            both = strong.T.astype(np.int32) @ strong.astype(np.int32)
            overlap = both / np.maximum(np.minimum.outer(size, size), 1)
            need = np.array([1 / (1 + t.excerpts) for t in themes])

            for row in gated:
                order = np.argsort(-theme_z[row])
                f1, f2 = int(order[0]), int(order[1])
                if theme_z[row, f2] >= config.bridge_z:
                    rarity = 1 - overlap[f1, f2]
                    lane_scores["bridges"][row] = float(theme_z[row, f2]) * (0.5 + 0.5 * rarity)
                    lane_why["bridges"][row] = {"themes": [themes[f1].label, themes[f2].label], "rarity": round(float(rarity), 3)}
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
        note = notes.get(lane, "" if items or lane in active else "Nothing passed the relevance gate for this lane.")
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
