"""Related passages for a task document: the task itself is the query.

1. units    Split the task into paragraphs ("units") and recognise the ones that are excerpts
            already collected from the corpus. Clusters of similar paragraphs are themes.
2. signals  Score every corpus passage against the whole document, two ways:
              semantic  robust z-score of its cosine similarity to the document vector (the
                        length-weighted mean of the paragraph embeddings): how far it stands out
                        from the typical passage (median, and median absolute deviation), so
                        thresholds mean the same for every task and every model;
              lexical   BM25 for the document's most distinctive terms (TF-IDF), minus the ones
                        the user ignores, plus the user's steer line: its words join the query at
                        full weight, and its phrases and regexes are scored like words.
            With query="paragraphs" each paragraph is a query of its own instead, and a passage
            scores by its best match (optionally weighted toward the cursor).
3. gate     Keep only clearly relevant passages: a strong semantic z-score, or a top lexical hit
            with some semantic support. Drop what is already collected or was dismissed, what
            the steer line rules out (+x requires x, -x excludes it), and what is close to an
            avoided passage.
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

import hashlib
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import lru_cache, partial

import numpy as np

from .chunker import MAX_WORDS, chunk_markdown, heading_texts
from .index import Snapshot
from .params import tunable
from .search import (NEVER, PREFER, REQUIRE, Clause, Hit, QueryError, Steering, lexical_scores, make_hit,
                     remembering_lines, split_query, steering, top_rows)
from .text import tokenize, tokenize_with_surface

MAX_EXCERPT_MATCHES = 50_000  # paragraphs whose excerpt check is remembered
MAX_THEME_SCORES = 48  # themes whose scores are remembered: two arrays of the corpus's size each
THEME_BATCH = 32  # paragraphs scored at a time, which bounds the memory a large theme needs

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
    like_weight: float = tunable(
        1.0, label="More like this", group=QUERY, min=0.0, max=3.0, step=0.1,
        help="How strongly a passage you marked ↑ pulls the query toward it, in meaning and keywords. 1: each one "
        "counts as much as your whole document. 0: they are ignored.",
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
        0.0, label="Dismissal push", group=FILTER, min=0.0, max=2.0, step=0.1,
        help="How strongly passages similar to ones you dismissed are pushed down. Off by default: what you "
        "dismiss is usually still on topic, so pushing away from it pushed relevant passages down too.",
    )
    avoid_z: float = tunable(
        3.5, label="Avoid: similarity", group=FILTER, min=0.5, max=6.0, step=0.1,
        help="Passages you avoid (⊘) also hide every passage at least this similar to them (z-score). "
        "Lower hides more.",
    )
    rrf_k: int = tunable(
        30, label="Core fusion k", group=LANES_GROUP, min=1, max=100, step=1,
        help="Core ranks by reciprocal-rank fusion. Large k favours passages good on both signals; small k "
        "lets the best of either signal through.",
    )
    unembedded_weight: float = tunable(
        1.0, label="Core: not embedded", group=LANES_GROUP, min=0.0, max=3.0, step=0.1,
        help="A passage without embeddings has no meaning rank, so its keyword rank counts in its place, times "
        "this. 0: keywords only.",
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
        0.15, label="Theme split", group=LANES_GROUP, min=0.05, max=1.0, step=0.05,
        help="Paragraphs further apart than this (cosine distance) become separate themes.",
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
        keys=list(LANES),
        default_factory=lambda: {"core": 0.7, "hidden": 0.6, "bridges": 0.6, "gaps": 0.6},
        help="MMR: 1 ranks by relevance only; lower values prefer results unlike the ones already shown "
        "and the passages you collected.",
    )
    mmr_pool: int = 100
    lane_prior: dict = field(
        default_factory=lambda: {"core": (3.0, 1.0), "hidden": (1.0, 1.0), "bridges": (1.0, 1.0), "gaps": (1.0, 1.0)}
    )


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _strings(v) -> list[str]:
    return [x for x in v if isinstance(x, str)] if isinstance(v, list) else []


def _remembered_passages(v) -> list[dict]:
    """Saved passages that have their text, with a path and line numbers of the right type."""
    return [
        {**d, "path": d.get("path") if isinstance(d.get("path"), str) else "",
         "start_line": d.get("start_line") if _is_int(d.get("start_line")) else None,
         "end_line": d.get("end_line") if _is_int(d.get("end_line")) else None}
        for d in (v if isinstance(v, list) else [])
        if isinstance(d, dict) and isinstance(d.get("text"), str)
    ]


def _complete_clauses(text: str) -> list[Clause]:
    return [c for c in split_query(text) if c.kind != "partial"]


@dataclass
class TaskPrefs:
    """What the user has told us about one task. Stored next to the task document."""

    steer: str = ""  # clauses as in a search query: x prefer, +x require, -x never show
    ignored: list[str] = field(default_factory=list)  # stems of task keywords not to search for
    dismissed: list[dict] = field(default_factory=list)  # {"path", "text", "lane", "start_line", "end_line", "avoid"?}
    feedback: dict = field(default_factory=dict)  # lane -> [adds, dismisses]
    liked: list[dict] = field(default_factory=list)  # "more like this": {"path", "text", "lane", "start_line", "end_line"}

    @classmethod
    def from_dict(cls, data: dict | None) -> TaskPrefs:
        """Preferences as saved. Values of the wrong type (a hand-edited file) are dropped one by
        one rather than failing later; anything but an object raises ValueError."""
        data = {} if data is None else data
        if not isinstance(data, dict):
            raise ValueError("task preferences must be a JSON object")
        prefs = cls(
            steer=data["steer"] if isinstance(data.get("steer"), str) else "",
            ignored=_strings(data.get("ignored")),
            dismissed=_remembered_passages(data.get("dismissed")),
            liked=_remembered_passages(data.get("liked")),
            feedback={
                lane: [max(0, v[0]), max(0, v[1])]  # the lane draw refuses negative counts
                for lane, v in (data.get("feedback") if isinstance(data.get("feedback"), dict) else {}).items()
                if isinstance(v, list) and len(v) == 2 and all(map(_is_int, v))
            },
        )
        # Saved before the steer line: pinned terms become words to prefer, exclusions become
        # -clauses, and banned terms are what is now called ignored.
        if "steer" not in data and (data.get("pins") or data.get("excluded")):
            prefs.steer = " ".join([*_strings(data.get("pins")), *(f"-{e}" for e in _strings(data.get("excluded")))])
        if "ignored" not in data and data.get("bans"):
            prefs.ignored = _strings(data["bans"])
        return prefs

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self, preview: int = 200) -> dict:
        """What the page shows: as to_dict, but each remembered passage carries an id and only the
        start of its text. Hundreds of them would otherwise travel with every refresh."""
        out = self.to_dict()
        for name in ("dismissed", "liked"):
            out[name] = [{**d, "id": passage_id(d), "text": d["text"][:preview]} for d in getattr(self, name)]
        return out

    # ---- the steer line ----

    def clauses(self) -> list[Clause]:
        return _complete_clauses(self.steer)

    def set_steer(self, text: str) -> None:
        """Replace the steer line (raises QueryError for an invalid regex). A word named on it is
        no longer ignored, as when a chip is boosted or excluded."""
        clauses = _complete_clauses(text)
        steering(clauses)
        self._set_clauses(clauses)
        named = {stem for c in clauses for stem in c.stems()}
        self.ignored = [t for t in self.ignored if t not in named]

    def _set_clauses(self, clauses: list[Clause]) -> None:
        self.steer = " ".join(map(str, clauses))

    def _without(self, stem: str, ops) -> list[Clause]:
        return [c for c in self.clauses() if not (c.kind == "word" and c.op in ops and c.stems() == [stem])]

    def _put_word(self, word: str, op: str, instead_of: tuple[str, ...]) -> None:
        """Put a word on the steer line with this operator, in place of the operators `instead_of`.
        A word it can't search for (a stopword, or several words) is left out."""
        stems = tokenize(word)
        if len(stems) != 1:
            return
        clauses = self._without(stems[0], instead_of)
        if not any(c.kind == "word" and c.stems() == stems for c in clauses):
            clauses.append(Clause(op, "word", word))
        self._set_clauses(clauses)
        self.unignore(stems[0])

    def boost(self, word: str) -> None:
        """Prefer passages with this word: it joins the keyword query at full weight."""
        self._put_word(word, PREFER, (NEVER,))

    def unboost(self, stem: str) -> None:
        self._set_clauses(self._without(stem, (PREFER, REQUIRE)))

    def exclude(self, word: str) -> None:
        """Never show passages containing this word (in any form sharing its stem)."""
        self._put_word(word, NEVER, (PREFER, REQUIRE))

    def ignore(self, stem: str) -> None:
        """Stop searching for one of the task's own keywords. Passages containing it can still be
        suggested (to rule them out, exclude the word instead)."""
        self.unboost(stem)
        if stem not in self.ignored:
            self.ignored.append(stem)

    def unignore(self, stem: str) -> None:
        self.ignored = [t for t in self.ignored if t != stem]

    # ---- feedback on suggestions ----

    def record(self, lane: str, added: bool) -> None:
        counts = self.feedback.setdefault(lane, [0, 0])
        counts[0 if added else 1] += 1

    def dismiss(self, lane: str, path: str, text: str, start_line: int | None = None, end_line: int | None = None,
                avoid: bool = False) -> None:
        """Hide a passage. `avoid` also hides every passage close to it in meaning (an anti-query)."""
        self.record(lane, added=False)
        self.unlike(passage_id({"path": path, "text": text}))
        if text:
            entry = {"path": path, "text": text, "lane": lane, "start_line": start_line, "end_line": end_line}
            self.dismissed.append({**entry, "avoid": True} if avoid else entry)

    def like(self, lane: str, path: str, text: str, start_line: int | None = None, end_line: int | None = None) -> None:
        """More like this: the passage pulls the query toward it, without joining the document."""
        if not text or any(passage_id(d) == passage_id({"path": path, "text": text}) for d in self.liked):
            return
        self.record(lane, added=True)
        self.liked.append({"path": path, "text": text, "lane": lane, "start_line": start_line, "end_line": end_line})

    def unlike(self, ident: str) -> None:
        """Take a like back, along with the add it counted for its lane."""
        kept = []
        for d in self.liked:
            if passage_id(d) != ident:
                kept.append(d)
            elif counts := self.feedback.get(d.get("lane")):
                counts[0] = max(counts[0] - 1, 0)
        self.liked = kept

    def restore_id(self, ident: str) -> None:
        """restore() for the dismissal with this id (see summary())."""
        for d in self.dismissed:
            if passage_id(d) == ident:
                self.restore(d.get("path", ""), d["text"])
                return

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


def passage_id(d: dict) -> str:
    """A short stable name for a remembered passage: its path and text, hashed."""
    return hashlib.sha1(f"{d.get('path', '')}\0{_norm(d.get('text', ''))}".encode()).hexdigest()[:12]


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

    @property
    def embed_text(self) -> str:
        return f"{self.heading}\n{self.text}" if self.heading else self.text


@dataclass
class TermChip:
    term: str  # stemmed, as used by the index
    label: str  # the word as it appears in the task
    weight: float
    boosted: bool = False  # from the steer line: searched for at full weight
    ignored: bool = False  # not searched for


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


def _closest_in_words(snapshot: Snapshot, text: str) -> np.ndarray:
    ids = [i for t in set(tokenize(text)) if (i := snapshot.term_id(t)) is not None]
    return top_rows(snapshot.bm25.scores(ids, np.ones(len(ids))), 5, positive=True)


def _closest_in_meaning(snapshot: Snapshot, vec: np.ndarray) -> np.ndarray:
    scores = snapshot.vectors @ vec
    scores[~snapshot.has_vector] = -np.inf
    return top_rows(scores, 5)


def _sharing_text(snapshot: Snapshot, closest: Callable[[], np.ndarray], needle: str) -> list[int]:
    """The closest passages that hold this normalised text, or are held by it."""
    found = []
    for row in dict.fromkeys(int(r) for r in closest()):
        hay = _norm(snapshot.passages[row].text)
        if needle in hay or (len(hay) >= 20 and hay in needle):
            found.append(row)
    return found


def find_excerpts(snapshot: Snapshot, units: list[Unit], unit_vecs: np.ndarray | None = None) -> None:
    """Mark units copied from the corpus, and remember which passages they came from.

    A unit is an excerpt when its text appears inside a corpus passage (or contains one). The
    places to look are the unit's best lexical matches; with `unit_vecs`, the closest passages in
    meaning of the units not matched yet (the second pass, once the units are embedded). Answers
    are kept per text for as long as the index stays the same, so a long task only pays for the
    paragraphs that changed."""
    for j, unit in enumerate(units):
        needle = _norm(unit.text)
        if len(needle) < 20 or (unit_vecs is not None and unit.excerpt):
            continue
        if unit_vecs is None:
            key, closest = (needle,), partial(_closest_in_words, snapshot, unit.text)
        else:
            key, closest = (needle, unit.heading, unit.text), partial(_closest_in_meaning, snapshot, unit_vecs[j])
        matches = partial(_sharing_text, snapshot, closest, needle)
        for row in snapshot.cached(snapshot.excerpt_matches, key, MAX_EXCERPT_MATCHES, matches):
            if row not in unit.source_rows:
                unit.excerpt = True
                unit.source_rows.append(row)


class ExcerptMatcher:
    """Whether a passage is inside a collected excerpt or holds one, without comparing it to each
    of them: the same answer as any(e in hay or hay in e for e in excerpts), for normalised text.
    An excerpt can start and end mid-word, but the words inside it are whole: it can only be in a
    passage that has its longest inner word, so only those passages compare it."""

    def __init__(self, excerpts: list[str]):
        self.joined = "\0".join(excerpts)
        self.by_word: dict[str, list[str]] = defaultdict(list)
        self.no_inner_word: list[str] = []
        for e in excerpts:
            inner = e.split(" ")[1:-1]
            if inner:
                self.by_word[max(inner, key=len)].append(e)
            else:
                self.no_inner_word.append(e)

    def __call__(self, hay: str) -> bool:
        if not self.joined:
            return False
        if hay in self.joined:
            return True
        if any(e in hay for e in self.no_inner_word):
            return True
        for word in set(hay.split(" ")):
            for e in self.by_word.get(word, ()):
                if e in hay:
                    return True
        return False


def same_passage(snapshot: Snapshot, rows, text: str) -> int | None:
    """The row among `rows` whose passage is exactly `text` (up to case and spacing) and has a
    vector, so that vector can be reused instead of embedding the text again."""
    needle = _norm(text)
    for row in rows:
        if snapshot.has_vector[row] and _norm(snapshot.passages[row].text) == needle:
            return int(row)
    return None


def embed_reusing(snapshot: Snapshot, embedder, texts: list[str], rows: list[int | None]) -> np.ndarray:
    """Embeddings of `texts`, taking the corpus vector where `rows` names the passage a text is.
    A long task is mostly collected passages, so most of it is already embedded in the index."""
    out = np.zeros((len(texts), snapshot.vectors.shape[1]), np.float32)
    todo = [j for j, row in enumerate(rows) if row is None]
    for j, row in enumerate(rows):
        if row is not None:
            out[j] = snapshot.vectors[row]
    if todo:
        out[todo] = embedder.embed([texts[j] for j in todo], cache=True)
    return out


def remembered_row(snapshot: Snapshot, d: dict) -> int | None:
    """The corpus row of a passage remembered by path, line and text (as dismissals are)."""
    lo, hi = snapshot.file_rows.get(d.get("path", ""), (0, 0))
    start_line = d.get("start_line")
    rows = [r for r in range(lo, hi) if start_line is None or snapshot.passages[r].start_line == start_line]
    return same_passage(snapshot, rows, d["text"])


def remembered_vectors(snapshot: Snapshot, embedder, remembered: list[dict]) -> np.ndarray:
    """Embeddings of remembered passages, taken from the index for those still in the corpus."""
    return embed_reusing(snapshot, embedder, [d["text"] for d in remembered], [remembered_row(snapshot, d) for d in remembered])


def assign_themes(units: list[Unit], unit_vecs: np.ndarray | None, config: RelatedConfig,
                  snapshot: Snapshot | None = None) -> list[Theme]:
    """Themes are clusters of similar paragraphs (average linkage on cosine distance), at most
    2 * sqrt(paragraphs) of them. They come from the content alone: a heading can be pasted in with
    an excerpt, or say little about what follows it."""
    if unit_vecs is not None and len(units) >= 2:
        clusters = _clusters(unit_vecs, config.theme_distance)
        order = list(dict.fromkeys(clusters))  # number themes in reading order
        for u, c in zip(units, clusters):
            u.theme = order.index(c)
        labels = [_cluster_label([u for u in units if u.theme == t], snapshot) for t in range(len(order))]
    else:
        for u in units:
            u.theme = 0
        labels = ["Everything"] if units else []

    return [
        Theme(label, sum(u.theme == t for u in units), sum(u.theme == t and u.excerpt for u in units))
        for t, label in enumerate(labels)
    ]


def _clusters(vecs: np.ndarray, distance: float) -> np.ndarray:
    """Cluster ids for the paragraphs. A long task about one subject would otherwise split into
    a theme per paragraph. Small themes are kept: on long tasks, merging single paragraphs into
    bigger themes cost relevance in the bridges and gaps lanes (CALIBRATE.md)."""
    from scipy.cluster.hierarchy import fcluster, linkage

    tree = linkage(vecs, "average", metric="cosine")
    clusters = fcluster(tree, distance, "distance")
    cap = 2 * math.ceil(math.sqrt(len(vecs)))
    if len(set(clusters)) > cap:
        clusters = fcluster(tree, cap, "maxclust")
    return clusters


def _cluster_label(units: list[Unit], snapshot: Snapshot | None = None) -> str:
    """The theme's three most distinctive words: frequency in the theme times rarity in the corpus."""
    counts, surfaces = Counter(), defaultdict(Counter)
    for u in units:
        for term, surface in tokenize_with_surface(u.text):
            counts[term] += 1
            surfaces[term][surface] += 1

    def weight(term):
        i = snapshot.term_id(term) if snapshot is not None else None
        return counts[term] * (float(snapshot.bm25.idf[i]) if i is not None else 1.0)

    best = sorted(counts, key=lambda t: (-weight(t), t))[:3]
    return " · ".join(surfaces[t].most_common(1)[0][0] for t in best) or "(untitled)"


# ---- 2. signals ----


@dataclass
class SemanticSignals:
    """How every corpus passage relates to the task in meaning, as robust z-scores."""

    strongest: np.ndarray  # against the query, unweighted: what the thresholds compare
    weighted: np.ndarray  # the same with paragraph weights: what the meaning ranking sorts by
    best_unit: np.ndarray | None = None  # query="paragraphs": the paragraph each passage matches best
    closest: np.ndarray | None = None  # best cosine to any paragraph of the task
    theme_z: np.ndarray | None = None  # [passages, themes], with two themes or more
    second: np.ndarray | None = None  # each passage's z-score on its second-best theme
    avoided: np.ndarray | None = None  # clearly like a passage the user avoids

    def only_in(self, keep: np.ndarray) -> None:
        """Rule out the other passages. Left-out folders still count for the corpus statistics, so
        thresholds mean the same with or without them."""
        self.strongest = np.where(keep, self.strongest, -np.inf)
        self.weighted = np.where(keep, self.weighted, -np.inf)
        if self.second is not None:
            self.second = np.where(keep, self.second, -np.inf)


def _normalized(v: np.ndarray) -> np.ndarray:
    return v / max(np.linalg.norm(v), 1e-9)


def semantic_signals(snapshot: Snapshot, embedder, units: list[Unit], unit_vecs: np.ndarray, themes: list[Theme],
                     liked: list[dict], prefs: TaskPrefs, config: RelatedConfig) -> SemanticSignals:
    weights = np.array([u.weight for u in units], np.float32)
    like_vecs = remembered_vectors(snapshot, embedder, liked) if liked else None
    if config.query == "document":
        sizes = np.array([len(u.text.split()) for u in units], np.float32) * weights
        document = _normalized((sizes[:, None] * unit_vecs).sum(0))
        if liked:  # a fixed pull per passage, however long the document has grown
            document = _normalized(document + config.like_weight * like_vecs.sum(0))
        _, z_doc = zscores(snapshot, document[None, :])
        out = SemanticSignals(z_doc[:, 0], z_doc[:, 0])  # the closest paragraph is worked out for the passages shown
    else:
        _, z = zscores(snapshot, unit_vecs)
        out = SemanticSignals(z.max(1), (z * weights).max(1), best_unit=(z * weights).argmax(1))
        if liked:  # each liked passage is one more paragraph to match
            _, z_like = zscores(snapshot, like_vecs)
            out.strongest = np.maximum(out.strongest, z_like.max(1))
            out.weighted = np.maximum(out.weighted, min(config.like_weight, 1.0) * z_like.max(1))

    dismissed = [d for d in prefs.dismissed if d.get("text")]
    if config.negative_weight <= 0:  # no penalty: only the anti-queries need scoring
        dismissed = [d for d in dismissed if d.get("avoid")]
    if dismissed:
        _, z_neg = zscores(snapshot, remembered_vectors(snapshot, embedder, dismissed))
        if config.negative_weight > 0:
            penalty = config.negative_weight * np.maximum(z_neg.max(1), 0)
            out.strongest, out.weighted = out.strongest - penalty, out.weighted - penalty
        avoid = [i for i, d in enumerate(dismissed) if d.get("avoid")]
        if avoid:  # anti-queries: everything clearly like them is out
            out.avoided = z_neg[:, avoid].max(1) >= config.avoid_z

    theme_of = np.array([u.theme for u in units])
    per_theme = [
        theme_scores(snapshot, [u for u in units if u.theme == t], unit_vecs[theme_of == t])
        for t in range(len(themes))
    ]
    out.closest = np.max([best_cos for _, best_cos in per_theme], axis=0)
    if len(themes) >= 2:
        out.theme_z = np.stack([best_z for best_z, _ in per_theme], axis=1)
        out.second = np.partition(out.theme_z, -2, axis=1)[:, -2]
    return out


def zscores(snapshot: Snapshot, vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity of every passage to each vector, and the same as robust z-scores:
    distance from the corpus median in units of (scaled) median absolute deviation. Unlike mean
    and standard deviation, these are not dragged around by a large on-topic cluster. Passages
    without embeddings get -inf."""
    cos = snapshot.vectors @ vecs.T
    has = snapshot.has_vector
    typical = cos[snapshot.typical_rows()]
    median = np.median(typical, axis=0)
    spread = 1.4826 * np.median(np.abs(typical - median), axis=0) + 1e-6
    z = (cos - median) / spread
    z[~has] = -np.inf
    cos[~has] = -np.inf
    return cos, z


@lru_cache(maxsize=50_000)
def _terms_of(text: str) -> tuple:
    return tuple(tokenize_with_surface(text))


def task_terms(snapshot: Snapshot, units: list[Unit], prefs: TaskPrefs, config: RelatedConfig,
               steer: Steering | None = None, headings: list[str] = ()) -> list[TermChip]:
    """The task's most distinctive terms (weighted term frequency in the task x corpus IDF), minus
    the ignored and excluded ones, plus the steer line's words at full weight. `headings` are the
    task's heading lines: their words count like any other line of text."""
    steer = steer or Steering()
    tf: dict[str, float] = defaultdict(float)
    surfaces: dict[str, Counter] = defaultdict(Counter)
    for u in units:
        for term, surface in _terms_of(u.text):
            tf[term] += u.weight
            surfaces[term][surface] += 1
    for heading in headings:
        for term, surface in tokenize_with_surface(heading):
            tf[term] += 1.0
            surfaces[term][surface] += 1
    if config.like_weight > 0:  # each liked passage adds a share of the task's own word count
        total = max(sum(tf.values()), 1.0)
        for d in prefs.liked:
            pairs = tokenize_with_surface(d.get("text", ""))
            for term, surface in pairs:
                tf[term] += config.like_weight * total / len(pairs)
                surfaces[term][surface] += 1

    scored = []
    for term, f in tf.items():
        i = snapshot.term_id(term)
        if i is not None and snapshot.bm25.doc_freq[i] > 0:
            scored.append((f * float(snapshot.bm25.idf[i]), term))
    scored.sort(reverse=True)

    skip = set(prefs.ignored) | steer.filters.excluded_terms
    boosted = set(steer.words) - steer.filters.excluded_terms
    top = [(s, t) for s, t in scored if t not in skip][: config.n_query_terms]
    best = top[0][0] if top and top[0][0] > 0 else 1.0  # all 0 when every paragraph is an excerpt at weight 0
    top = [(best if t in boosted else s, t) for s, t in top]
    chosen = {t for _, t in top}
    for t in sorted(boosted - chosen):  # steered words always take part, at full weight
        if snapshot.term_id(t) is not None:
            top.append((best, t))

    def label(t):
        if surfaces.get(t):
            return surfaces[t].most_common(1)[0][0]
        return steer.surfaces.get(t, t)

    chips = [TermChip(t, label(t), s / best, boosted=t in boosted) for s, t in top]
    chips += [TermChip(t, label(t), 0.0, ignored=True) for t in prefs.ignored]
    return chips


def theme_scores(snapshot: Snapshot, theme_units: list[Unit], vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For one theme, every passage's best z-score and best cosine against its paragraphs. Kept
    for as long as the theme's text and the index stay the same: typing changes one theme, and
    the table of all passages against all paragraphs would not fit in memory for a large corpus."""

    def score():
        best_z, best_cos = None, None
        for start in range(0, len(vecs), THEME_BATCH):
            cos, z = zscores(snapshot, vecs[start : start + THEME_BATCH])
            z, cos = z.max(1), cos.max(1)
            best_z = z if best_z is None else np.maximum(best_z, z)
            best_cos = cos if best_cos is None else np.maximum(best_cos, cos)
        return best_z, best_cos

    key = tuple(sorted(f"{u.heading}\0{u.text}" for u in theme_units))
    return snapshot.cached(snapshot.theme_scores, key, MAX_THEME_SCORES, score)


def ranks(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """0 for the best score; len(scores) for rows that are not valid."""
    out = np.full(len(scores), len(scores), np.int64)
    rows = np.flatnonzero(valid)
    out[rows[np.argsort(-scores[rows], kind="stable")]] = np.arange(len(rows))
    return out


# ---- 3. gate ----


def gate(snapshot: Snapshot, units: list[Unit], signals: SemanticSignals, lex: np.ndarray, lex_rank: np.ndarray,
         steer: Steering, prefs: TaskPrefs, config: RelatedConfig, semantic: bool) -> tuple[list[int], int]:
    """The passages that may be suggested, and how many candidates the rankings put forward. A
    passage gets in on strong meaning, on a top keyword match with some meaning behind it, or
    (bridges) on being clearly related to two themes at once: two independent themes at
    z >= bridge_z is about as unlikely by chance as one at z >= z_strong."""
    has = snapshot.has_vector & semantic
    relevant = np.where(
        has,
        (signals.strongest >= config.z_strong) | ((lex_rank < config.lexical_top) & (signals.strongest >= config.z_weak)),
        lex_rank < config.lexical_top,  # no embedding for this passage (yet): keywords alone
    )
    if signals.second is not None:
        relevant |= has & (signals.second >= config.bridge_z)
    if semantic:
        relevant &= ~(has & (signals.closest >= config.duplicate_cos))

    candidates = set(top_rows(lex, config.candidates, positive=True)) | set(top_rows(signals.weighted, config.candidates))
    if signals.second is not None:
        candidates |= set(top_rows(signals.second, config.candidates))

    rows = np.array(sorted(map(int, candidates)), np.int64)
    rows = rows[relevant[rows]]
    if signals.avoided is not None:
        rows = rows[~signals.avoided[rows]]
    if steer.filters:
        rows = rows[steer.filters.allowed(snapshot, rows)]

    collected = {r for u in units for r in u.source_rows}
    dismissed_keys = {(d.get("path"), _norm(d.get("text", ""))) for d in prefs.dismissed}
    excerpts = [e for u in units if u.excerpt and len(e := _norm(u.text)) >= 20]
    near_excerpt = ExcerptMatcher(excerpts)

    gated = []
    for row in map(int, rows):
        if row in collected:
            continue
        p = snapshot.passages[row]
        if dismissed_keys or excerpts:
            hay = _norm(p.text)
            if (p.path, hay) in dismissed_keys or near_excerpt(hay):
                continue
        gated.append(row)
    return gated, len(candidates)


# ---- 4. lanes ----


@dataclass
class LaneScores:
    """What the passages that got through the gate score in each lane they qualify for."""

    scores: dict[str, dict[int, float]] = field(default_factory=lambda: {lane: {} for lane in LANES})
    why: dict[str, dict[int, dict]] = field(default_factory=lambda: {lane: {} for lane in LANES})
    notes: dict[str, str] = field(default_factory=dict)


def score_core(snapshot: Snapshot, gated: list[int], lanes: LaneScores, lex_rank: np.ndarray, sem_rank: np.ndarray,
               config: RelatedConfig, semantic: bool) -> None:
    for row in gated:
        keyword = 1 / (config.rrf_k + lex_rank[row] + 1)
        no_vector = semantic and not snapshot.has_vector[row]  # the keyword rank stands in for the meaning rank
        meaning = config.unembedded_weight * keyword if no_vector else 1 / (config.rrf_k + sem_rank[row] + 1)
        lanes.scores["core"][row] = keyword + meaning


def score_hidden(gated: list[int], lanes: LaneScores, signals: SemanticSignals, lex: np.ndarray, config: RelatedConfig) -> None:
    lex_best = float(lex.max())
    for row in gated:
        obviousness = float(lex[row] / lex_best) if lex_best > 0 else 0.0
        if signals.strongest[row] >= config.hidden_z and obviousness <= config.hidden_max_obviousness:
            lanes.scores["hidden"][row] = float(signals.weighted[row]) * (1 - obviousness)
            lanes.why["hidden"][row] = {"obviousness": round(obviousness, 3)}


def score_bridges_and_gaps(gated: list[int], lanes: LaneScores, theme_z: np.ndarray, themes: list[Theme], config: RelatedConfig) -> None:
    strong = theme_z >= config.bridge_z
    size = strong.sum(0)
    as_float = strong.astype(np.float32)  # exact counts below 16 million passages, and much faster
    both = as_float.T @ as_float
    overlap = both / np.maximum(np.minimum.outer(size, size), 1)  # how often two themes meet
    need = np.array([1 / (1 + t.excerpts) for t in themes])

    for row in gated:
        order = np.argsort(-theme_z[row])
        f1, f2 = int(order[0]), int(order[1])
        if theme_z[row, f2] >= config.bridge_z:
            rarity = 1 - overlap[f1, f2]
            lanes.scores["bridges"][row] = float(theme_z[row, f2]) * (0.5 + 0.5 * rarity)
            lanes.why["bridges"][row] = {
                "themes": [themes[f1].label, themes[f2].label],
                "theme_z": [round(float(theme_z[row, f1]), 2), round(float(theme_z[row, f2]), 2)],
                "rarity": round(float(rarity), 3),
            }
        gap = np.where(theme_z[row] >= config.gap_z, need * theme_z[row], -np.inf)
        if np.isfinite(gap.max()):
            f = int(gap.argmax())
            lanes.scores["gaps"][row] = float(gap[f])
            lanes.why["gaps"][row] = {"themes": [themes[f].label], "collected_in_theme": themes[f].excerpts}


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
    wanted: set[int] = frozenset(),
) -> list[tuple[int, float]]:
    """Maximal marginal relevance: repeatedly take the passage with the best
    lam * relevance - (1 - lam) * (similarity to anything already chosen or collected). Passages
    marked "more like this" (`wanted`) don't count as chosen: similar ones are what was asked for."""
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
        if usable[i] and row not in wanted:
            chosen_vecs.append(vecs[i])
            redundancy = np.maximum(redundancy, np.where(usable, vecs @ vecs[i], 0.0))
    return picked


def select(snapshot: Snapshot, scores: dict[str, dict[int, float]], collected_vecs: list[np.ndarray], liked: list[dict],
           prefs: TaskPrefs, config: RelatedConfig, seed: int) -> tuple[dict[str, list[tuple[int, float]]], dict[str, int]]:
    """Each lane's picks as (row, MMR value), and the slots each lane was given."""
    active = [lane for lane in LANES if scores[lane]]
    feedback_events = sum(sum(v) for v in prefs.feedback.values())
    slots = allocate_slots(active, prefs, config, seed + feedback_events)
    taken: set[int] = set()
    chosen_vecs = list(collected_vecs)
    wanted = {r for d in liked if (r := remembered_row(snapshot, d)) is not None}
    per_file = {lane: Counter() for lane in active}

    def fill(lane: str, k: int) -> list[tuple[int, float]]:
        lam = config.mmr_lambda.get(lane, 0.7)
        return mmr_select(scores[lane], k, lam, snapshot, chosen_vecs, taken, config, per_file[lane], wanted)

    picked = {lane: fill(lane, slots[lane]) for lane in active}
    if "core" in picked:  # slots other lanes could not fill go to core
        spare = sum(slots.values()) - sum(len(v) for v in picked.values())
        if spare > 0:
            picked["core"] += fill("core", spare)
    return picked, slots


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
    skip_folders=(),
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
        return RelatedResult([Lane(lane, *LANE_INFO[lane], [], note) for lane in LANES], [], [], {"passages": n})

    # 1: units and themes
    semantic = embedder is not None and snapshot.has_vector.any()
    unit_vecs = None
    find_excerpts(snapshot, units)
    if semantic:
        reuse = [same_passage(snapshot, u.source_rows, u.text) for u in units]
        unit_vecs = embed_reusing(snapshot, embedder, [u.embed_text for u in units], reuse)
        find_excerpts(snapshot, units, unit_vecs)
    weight_units(units, cursor_line, config)
    themes = assign_themes(units, unit_vecs, config, snapshot)
    lap("units")

    # 2: signals. Left-out folders are gone before anything is ranked, so they take no places.
    liked = [d for d in prefs.liked if d.get("text")] if config.like_weight > 0 else []
    if semantic:
        signals = semantic_signals(snapshot, embedder, units, unit_vecs, themes, liked, prefs, config)
    else:
        signals = SemanticSignals(np.full(n, -np.inf), np.full(n, -np.inf))
    keep = snapshot.outside_folders(skip_folders) if skip_folders else None
    if keep is not None:
        signals.only_in(keep)
    lap("semantic")

    try:
        steer = steering(prefs.clauses())
    except QueryError:  # saved lines are checked when saved; an old one may not be
        steer = Steering()
    chips = task_terms(snapshot, units, prefs, config, steer, heading_texts(task_text))
    query = [(snapshot.term_id(c.term), c.weight) for c in chips if not c.ignored]
    bm25 = snapshot.bm25_for(config.bm25_k1, config.bm25_b)
    lex = lexical_scores(snapshot, bm25, [i for i, _ in query], [w for _, w in query], steer.patterns)
    if keep is not None:
        lex = np.where(keep, lex, 0.0)
    lex_rank = ranks(lex, lex > 0)
    sem_rank = ranks(signals.weighted, np.isfinite(signals.weighted))
    lap("lexical")

    # 3: gate
    gated, n_candidates = gate(snapshot, units, signals, lex, lex_rank, steer, prefs, config, semantic)
    lap("gate")

    # 4: lanes
    scored = LaneScores()
    score_core(snapshot, gated, scored, lex_rank, sem_rank, config, semantic)
    if not semantic:
        for lane in ("hidden", "bridges", "gaps"):
            scored.notes[lane] = "Needs embeddings (still indexing, or running without a model)."
    else:
        score_hidden(gated, scored, signals, lex, config)
        if signals.theme_z is None:
            scored.notes["bridges"] = scored.notes["gaps"] = "Needs at least two themes: write about more than one thing."
        else:
            score_bridges_and_gaps(gated, scored, signals.theme_z, themes, config)
    lap("lanes")

    # 5: select
    collected_vecs = [unit_vecs[j] for j, u in enumerate(units) if u.excerpt] if semantic else []
    picked, slots = select(snapshot, scored.scores, collected_vecs, liked, prefs, config, seed)
    lap("select")

    terms = {c.term for c in chips if not c.ignored}
    lines_of = remembering_lines(read_lines)
    liked_ids = {passage_id(d) for d in prefs.liked}

    def hit(lane: str, row: int, value: float) -> Hit | None:
        p = snapshot.passages[row]
        lines = lines_of(p.path)
        if not lines:
            return None
        matches_task_line = None
        if semantic:  # the closest paragraph, for display
            best = signals.best_unit[row] if signals.best_unit is not None else np.argmax(snapshot.vectors[row] @ unit_vecs.T)
            matches_task_line = units[int(best)].start_line
        ident = passage_id({"path": p.path, "text": p.text}) if liked_ids else None
        why = {
            "lane_score": round(scored.scores[lane][row], 4),
            "mmr": round(value, 4),
            "semantic_z": round(float(signals.strongest[row]), 2) if np.isfinite(signals.strongest[row]) else None,
            "bm25": round(float(lex[row]), 3),
            "lexical_rank": int(lex_rank[row]) + 1 if lex[row] > 0 else None,
            "matches_task_line": matches_task_line,
            "liked": ident if ident in liked_ids else None,
            **scored.why[lane].get(row, {}),
        }
        return make_hit(snapshot, row, scored.scores[lane][row], terms, lines, context_lines, why, steer.highlight)

    lanes = []
    for lane in LANES:
        items = [h for row, value in picked.get(lane, []) if (h := hit(lane, row, value)) is not None]
        note = scored.notes.get(lane, "" if items or lane in picked else EMPTY_NOTE[lane])
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
            "candidates": n_candidates,
            "gated": len(gated),
            "slots": slots,
            "ms": timings,
        },
    )
