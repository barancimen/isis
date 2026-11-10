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
            scores by its best match.
3. gate     Keep only clearly relevant passages: a strong semantic z-score, or a top lexical hit
            with some semantic support. Drop what is already collected or was dismissed, what
            the steer line rules out (+x requires x, -x excludes it), what is close to an
            avoided passage, and every passage of an excluded document.
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
from functools import partial

import numpy as np

from .chunker import MAX_WORDS, chunk_markdown, heading_texts
from .index import Snapshot
from .lexical import B, K1
from .params import Group, tunable
from .search import (NEVER, PREFER, REQUIRE, Clause, Hit, QueryError, Steering, lexical_scores, make_hit,
                     remembering_lines, scoped_product, split_query, steering, top_rows)
from .text import tokenize, tokenize_with_surface

MAX_EXCERPT_MATCHES = 2_000  # paragraphs whose excerpt check is remembered
MAX_THEME_SCORES = 48  # themes whose scores are remembered: two arrays of the corpus's size each
THEME_BATCH = 32  # paragraphs scored at a time, which bounds the memory a large theme needs

LANES = ("core", "hidden", "bridges", "gaps")
LANE_INFO = {
    "core": ("Best results", "These passages match your task in both words and meaning."),
    "hidden": ("Same meaning, different words", "These passages are close to your task in meaning but use few of its keywords."),
    "bridges": ("Connections between topics", "These passages touch on two topics in your task that don't usually appear together."),
    "gaps": ("Small topics", "These passages relate to topics in your task where you've collected little so far."),
}
EMPTY_NOTE = {
    "core": "No passages are clearly related to your task.",
    "hidden": "No passages are close to your task in meaning while using different words.",
    "bridges": "No passages are clearly related to two topics in your task.",
    "gaps": "No passages found for the small topics in your task.",
}

QUERY = Group("How your task is read", "Related reads your task, then finds passages close to it in words and in meaning.")
KEYWORDS = Group("Keyword score", "How well the words in a passage match your task's keywords.")
FILTER = Group(
    "Which passages can appear",
    "A passage appears only if it passes this filter. " + "The meaning score shows how close a passage is to your task in meaning. A typical passage in your documents scores 0. A score of 3 is far above typical.",
)
LANES_GROUP = Group(
    "Result groups",
    "Results are sorted into four groups, each related to your task in a different way. "
    "A topic is a set of paragraphs in your task that are close in meaning.",
)
SELECTION = Group("Result list", "How many results you see and how much they differ from each other.")


@dataclass
class RelatedConfig:
    query: str = tunable(
        "document", label="Compare passages with", group=QUERY, choices={"document": "All of the task", "paragraphs": "Each paragraph"},
        help="How passages are compared with your task.\n"
        "\"All of the task\" treats the task as one text, so a passage has to fit all of it.\n"
        "\"Each paragraph\" compares a passage with every paragraph and uses the closest. "
        "A passage only has to be close to one paragraph.",
    )
    n_query_terms: int = tunable(
        30, label="Keywords to use", group=QUERY, min=1, max=60, step=1,
        help="How many keywords are taken from your task. "
        "These are words common in your task but not in your other documents.\n"
        "Raise it to use more words, so results can cover more topics.\n"
        "Lower it to use only the most distinctive words.",
    )
    excerpt_weight: float = tunable(
        0.5, label="How important collected passages are", group=QUERY, min=0.0, max=3.0, step=0.1,
        help="How much the passages you've added to your task count, compared with your notes.\n"
        "At 1, they count the same as your notes.\n"
        "Raise it to move results closer to the passages you've collected.\n"
        "Lower it to move results closer to your notes. At 0, collected passages aren't used.",
    )
    like_weight: float = tunable(
        1.0, label="How strong ↑ is", group=QUERY, min=0.0, max=3.0, step=0.1,
        help="When you click ↑ on a result, Related looks for more passages like it. "
        "This sets how strong that pull is, in meaning and in keywords.\n"
        "At 1, each passage with ↑ counts the same as all of your task.\n"
        "Raise it to move results closer to the passages with ↑.\n"
        "Lower it to keep results closer to your task. At 0, ↑ has no effect.",
    )
    bm25_k1: float = tunable(
        K1, label="Words used many times", group=KEYWORDS, min=0.0, max=3.0, step=0.1,
        help="How much a word adds to the score when it appears many times in a passage.\n"
        "Raise it to rank passages that repeat a keyword higher.\n"
        "Lower it and repeats count for less. At 0, one use counts the same as many.",
    )
    bm25_b: float = tunable(
        B, label="Passage length", group=KEYWORDS, min=0.0, max=1.0, step=0.05,
        help="Long passages contain more words, so they can score well just because of their length. "
        "This sets how much a long passage's score is reduced.\n"
        "Raise it to rank short passages higher. At 1, length is fully corrected for.\n"
        "Lower it to rank long passages higher. At 0, length has no effect.",
    )
    candidates: int = tunable(
        300, label="Passages to examine", group=FILTER, min=20, max=2000, step=10,
        help="Related makes one list by keywords and one by meaning. "
        "This sets how many top passages from each list reach the filter.\n"
        "Raise it to find more passages, at the cost of speed.\n"
        "Lower it for faster results but fewer passages found.",
    )
    z_strong: float = tunable(
        3.25, label="Meaning score alone", group=FILTER, min=0.0, max=5.0, step=0.1,
        help="A passage with this meaning score or higher passes the filter, even if it isn't a top keyword result.\n"
        "Raise it for fewer results that are more clearly related.\n"
        "Lower it for more results, though some are less clearly related.",
    )
    z_weak: float = tunable(
        1.75, label="Meaning score with keywords", group=FILTER, min=-1.0, max=4.0, step=0.1,
        help="A top keyword result with this meaning score or higher passes the filter. "
        "\"Top keyword ranks\" sets which passages count as top keyword results.\n"
        "Raise it so keyword results must also be closer in meaning, which gives fewer results.\n"
        "Lower it to let more keyword results through, even if they're less close in meaning.",
    )
    lexical_top: int = tunable(
        50, label="Top keyword ranks", group=FILTER, min=0, max=300, step=1,
        help="How many of the best keyword results count as top keyword results. "
        "These pass the filter with the lower meaning score. A passage with no meaning data passes only if it's one of them.\n"
        "Raise it to give more passages the lower limit, which can mean more results.\n"
        "Lower it to give fewer passages the lower limit. At 0, none get it.",
    )
    duplicate_cos: float = tunable(
        0.95, label="Duplicate limit", group=FILTER, min=0.7, max=1.0, step=0.01,
        help="A passage this close to a paragraph in your task is hidden, "
        "because your task probably already contains that text.\n"
        "Raise it to hide only near-identical passages. At 1, they must match exactly in meaning.\n"
        "Lower it to also hide passages that are less close to your text.",
    )
    negative_weight: float = tunable(
        0.0, label="Effect of removed passages", group=FILTER, min=0.0, max=2.0, step=0.1,
        help="How much lower a passage ranks when it's close to one you removed.\n"
        "Raise it to push those passages much lower.\n"
        "At 0, removed passages don't affect others. That's the default, "
        "because a passage you remove is usually still related to your task. "
        "In tests, this also pushed good results down.",
    )
    avoid_z: float = tunable(
        3.5, label="How near for ⊘", group=FILTER, min=0.5, max=6.0, step=0.1,
        help="When you click ⊘ on a result, it's removed along with passages close to it in meaning. "
        "This sets how close they must be, as a meaning score measured from the removed passage.\n"
        "Raise it to remove only the closest passages.\n"
        "Lower it to remove more passages.",
    )
    rrf_k: int = tunable(
        30, label="Good rank in both lists (Best results)", group=LANES_GROUP, min=1, max=100, step=1,
        help="\"Best results\" mixes each passage's keyword rank and meaning rank.\n"
        "Raise it to put passages that rank well in both lists first.\n"
        "Lower it to put the top passages of each list first, even if they rank badly in the other.",
    )
    unembedded_weight: float = tunable(
        1.0, label="No meaning data (Best results)", group=LANES_GROUP, min=0.0, max=3.0, step=0.1,
        help="Some passages may have no meaning data, so they have no meaning rank. "
        "Their keyword rank stands in for it, multiplied by this number.\n"
        "Raise it to rank these passages higher.\n"
        "Lower it to rank them lower. At 0, only their keyword rank counts.",
    )
    hidden_z: float = tunable(
        2.5, label="Meaning score (Same meaning, different words)", group=LANES_GROUP, min=0.0, max=5.0, step=0.1,
        help="The meaning score a passage needs to join \"Same meaning, different words\".\n"
        "Raise it for fewer passages that are closer in meaning.\n"
        "Lower it to let more passages in.",
    )
    hidden_max_obviousness: float = tunable(
        0.3, label="Keyword limit (Same meaning, different words)", group=LANES_GROUP, min=0.0, max=1.0, step=0.05,
        help="The highest keyword score a passage can have and still join \"Same meaning, different words\". "
        "It's a share of the best keyword result's score, so 0.3 means 30%.\n"
        "Raise it to allow more words from your task in this group.\n"
        "Lower it to require more different wording.",
    )
    bridge_z: float = tunable(
        1.5, label="Meaning score (Connections between topics)", group=LANES_GROUP, min=0.0, max=4.0, step=0.1,
        help="The meaning score a passage needs for two topics to join \"Connections between topics\". "
        "A passage with this score also passes the filter.\n"
        "Raise it for fewer passages that are more clearly related to both topics.\n"
        "Lower it to let more passages in, which can also add results to other groups.",
    )
    gap_z: float = tunable(
        2.5, label="Meaning score (Small topics)", group=LANES_GROUP, min=0.0, max=5.0, step=0.1,
        help="The meaning score a passage needs to join a topic in \"Small topics\".\n"
        "Raise it for fewer passages that are closer to the topic.\n"
        "Lower it to let more passages in.",
    )
    theme_distance: float = tunable(
        0.15, label="Number of topics", group=LANES_GROUP, min=0.05, max=1.0, step=0.05,
        help="How different two paragraphs must be to count as separate topics. "
        "Topics are used by \"Connections between topics\" and \"Small topics\".\n"
        "Raise it for fewer, larger topics.\n"
        "Lower it for more, smaller topics, up to a maximum for each task.",
    )
    total_slots: int = tunable(
        24, label="Number of results", group=SELECTION, min=4, max=80, step=1,
        help="The most results to show across all groups.\n"
        "Raise it to see more results, though the last ones may be less related.\n"
        "Lower it to see only the top results.",
    )
    min_slots: int = tunable(
        2, label="Minimum for each group", group=SELECTION, min=0, max=10, step=1,
        help="Each group with results is guaranteed this many places. "
        "The remaining places are shared out and groups you add more passages from usually get more.\n"
        "Raise it to even out the groups.\n"
        "Lower it to give the groups you use most more of the results.",
    )
    max_per_file: int = tunable(
        2, label="Maximum from one document", group=SELECTION, min=1, max=20, step=1,
        help="The most results one document can have in each group.\n"
        "Raise it to let one document fill more of a group.\n"
        "Lower it to draw results from more documents.",
    )
    mmr_lambda: dict = tunable(
        label="Closer or more varied", group=SELECTION, min=0.0, max=1.0, step=0.05,
        keys={lane: LANE_INFO[lane][0] for lane in LANES},
        default_factory=lambda: {"core": 0.7, "hidden": 0.6, "bridges": 0.6, "gaps": 0.6},
        help="How the results for this group are chosen.\n"
        "Raise it to choose the passages most related to your task. At 1, nothing else counts.\n"
        "Lower it to choose passages that differ more from the results already chosen, "
        "and from the passages you've collected.",
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
    ignored: list[str] = field(default_factory=list)  # task keywords (terms) not to search for
    dismissed: list[dict] = field(default_factory=list)  # {"path", "text", "lane", "start_line", "end_line", "avoid"?}
    feedback: dict = field(default_factory=dict)  # lane -> [adds, dismisses]
    liked: list[dict] = field(default_factory=list)  # "more like this": {"path", "text", "lane", "start_line", "end_line"}
    excluded_docs: list[str] = field(default_factory=list)  # paths of documents never to suggest from

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
            excluded_docs=_strings(data.get("excluded_docs")),
            feedback={
                lane: [max(0, v[0]), max(0, v[1])]  # the lane draw refuses negative counts
                for lane, v in (data.get("feedback") if isinstance(data.get("feedback"), dict) else {}).items()
                if isinstance(v, list) and len(v) == 2 and all(map(_is_int, v))
            },
        )
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
        named = {term for c in clauses for term in c.terms()}
        self.ignored = [t for t in self.ignored if t not in named]

    def _set_clauses(self, clauses: list[Clause]) -> None:
        self.steer = " ".join(map(str, clauses))

    def _without(self, term: str, ops) -> list[Clause]:
        return [c for c in self.clauses() if not (c.kind == "word" and c.op in ops and c.terms() == [term])]

    def _put_word(self, word: str, op: str, instead_of: tuple[str, ...]) -> None:
        """Put a word on the steer line with this operator, in place of the operators `instead_of`.
        A word it can't search for (a stopword, or several words) is left out."""
        terms = tokenize(word)
        if len(terms) != 1:
            return
        clauses = self._without(terms[0], instead_of)
        if not any(c.kind == "word" and c.terms() == terms for c in clauses):
            clauses.append(Clause(op, "word", word))
        self._set_clauses(clauses)
        self.unignore(terms[0])

    def boost(self, word: str) -> None:
        """Prefer passages with this word: it joins the keyword query at full weight."""
        self._put_word(word, PREFER, (NEVER,))

    def unboost(self, term: str) -> None:
        self._set_clauses(self._without(term, (PREFER, REQUIRE)))

    def exclude(self, word: str) -> None:
        """Never show passages containing this word."""
        self._put_word(word, NEVER, (PREFER, REQUIRE))

    def ignore(self, term: str) -> None:
        """Stop searching for one of the task's own keywords. Passages containing it can still be
        suggested (to rule them out, exclude the word instead)."""
        self.unboost(term)
        if term not in self.ignored:
            self.ignored.append(term)

    def unignore(self, term: str) -> None:
        self.ignored = [t for t in self.ignored if t != term]

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

    def exclude_doc(self, path: str) -> None:
        """Never suggest anything from this document. Passages liked in it keep pulling: wanting
        more like a paragraph, but nothing from where it came, is a fair thing to ask. This judges
        the document rather than the lane that showed it, so the lane counts stay out of it."""
        if path and path not in self.excluded_docs:
            self.excluded_docs.append(path)

    def include_doc(self, path: str) -> None:
        self.excluded_docs = [p for p in self.excluded_docs if p != path]

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
    term: str  # as used by the index: the word in lowercase
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


def weight_units(units: list[Unit], config: RelatedConfig) -> None:
    """Your own paragraphs weigh 1 and collected excerpts `excerpt_weight`, scaled so the most is 1."""
    weights = [config.excerpt_weight if u.excerpt else 1.0 for u in units]
    top = max(max(weights, default=1.0), 1e-9)
    for u, w in zip(units, weights):
        u.weight = w / top


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
        labels = ["All of the task"] if units else []

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
        """Rule out the other passages. The z-scores were measured against the whole corpus, so
        thresholds mean the same with or without the left-out folders."""
        self.strongest = np.where(keep, self.strongest, -np.inf)
        self.weighted = np.where(keep, self.weighted, -np.inf)
        if self.second is not None:
            self.second = np.where(keep, self.second, -np.inf)


def _normalized(v: np.ndarray) -> np.ndarray:
    return v / max(np.linalg.norm(v), 1e-9)


def semantic_signals(snapshot: Snapshot, embedder, units: list[Unit], unit_vecs: np.ndarray, themes: list[Theme],
                     liked: list[dict], prefs: TaskPrefs, config: RelatedConfig, scope: np.ndarray | None = None,
                     scope_key=()) -> SemanticSignals:
    """Every passage's meaning scores, worked out only in `scope` (None: everywhere; -inf
    elsewhere). `scope_key` names the scope, for the scores kept between refreshes."""
    weights = np.array([u.weight for u in units], np.float32)
    like_vecs = remembered_vectors(snapshot, embedder, liked) if liked else None
    if config.query == "document":
        sizes = np.array([len(u.text.split()) for u in units], np.float32) * weights
        document = _normalized((sizes[:, None] * unit_vecs).sum(0))
        if liked:  # a fixed pull per passage, however long the document has grown
            document = _normalized(document + config.like_weight * like_vecs.sum(0))
        _, z_doc = zscores(snapshot, document[None, :], scope)
        out = SemanticSignals(z_doc[:, 0], z_doc[:, 0])  # the closest paragraph is worked out for the passages shown
    else:
        _, z = zscores(snapshot, unit_vecs, scope)
        out = SemanticSignals(z.max(1), (z * weights).max(1), best_unit=(z * weights).argmax(1))
        if liked:  # each liked passage is one more paragraph to match
            _, z_like = zscores(snapshot, like_vecs, scope)
            out.strongest = np.maximum(out.strongest, z_like.max(1))
            out.weighted = np.maximum(out.weighted, min(config.like_weight, 1.0) * z_like.max(1))

    dismissed = [d for d in prefs.dismissed if d.get("text")]
    if config.negative_weight <= 0:  # no penalty: only the anti-queries need scoring
        dismissed = [d for d in dismissed if d.get("avoid")]
    if dismissed:
        _, z_neg = zscores(snapshot, remembered_vectors(snapshot, embedder, dismissed), scope)
        if config.negative_weight > 0:
            penalty = config.negative_weight * np.maximum(z_neg.max(1), 0)
            out.strongest, out.weighted = out.strongest - penalty, out.weighted - penalty
        avoid = [i for i, d in enumerate(dismissed) if d.get("avoid")]
        if avoid:  # anti-queries: everything clearly like them is out
            out.avoided = z_neg[:, avoid].max(1) >= config.avoid_z

    theme_of = np.array([u.theme for u in units])
    per_theme = [
        theme_scores(snapshot, [u for u in units if u.theme == t], unit_vecs[theme_of == t], scope, scope_key)
        for t in range(len(themes))
    ]
    out.closest = np.max([best_cos for _, best_cos in per_theme], axis=0)
    if len(themes) >= 2:
        out.theme_z = np.stack([best_z for best_z, _ in per_theme], axis=1)
        out.second = np.partition(out.theme_z, -2, axis=1)[:, -2]
    return out


def zscores(snapshot: Snapshot, vecs: np.ndarray, scope: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity of every passage to each vector, and the same as robust z-scores:
    distance from the corpus median in units of (scaled) median absolute deviation. Unlike mean
    and standard deviation, these are not dragged around by a large on-topic cluster. Passages
    without embeddings, or outside `scope` (None: none are), get -inf. The median is the whole
    corpus's either way, so a z-score means the same with folders left out."""
    rows = snapshot.typical_rows()
    if scope is None or len(rows) == int(snapshot.has_vector.sum()):  # a corpus small enough to use all of it
        cos = snapshot.vectors @ vecs.T
        typical = cos[rows]
        if scope is not None:
            cos[~scope] = -np.inf
    else:
        cos = scoped_product(snapshot.vectors, vecs.T, scope)
        typical = snapshot.typical_vectors() @ vecs.T
    has = snapshot.has_vector
    median = np.median(typical, axis=0)
    spread = 1.4826 * np.median(np.abs(typical - median), axis=0) + 1e-6
    z = (cos - median) / spread
    z[~has] = -np.inf
    cos[~has] = -np.inf
    return cos, z


def task_terms(snapshot: Snapshot, units: list[Unit], prefs: TaskPrefs, config: RelatedConfig,
               steer: Steering | None = None, headings: list[str] = ()) -> list[TermChip]:
    """The task's most distinctive terms (weighted term frequency in the task x corpus IDF), minus
    the ignored and excluded ones, plus the steer line's words at full weight. `headings` are the
    task's heading lines: their words count like any other line of text."""
    steer = steer or Steering()
    tf: dict[str, float] = defaultdict(float)
    surfaces: dict[str, Counter] = defaultdict(Counter)
    for u in units:
        for term, surface in tokenize_with_surface(u.text):
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


def theme_scores(snapshot: Snapshot, theme_units: list[Unit], vecs: np.ndarray, scope: np.ndarray | None = None,
                 scope_key=()) -> tuple[np.ndarray, np.ndarray]:
    """For one theme, every passage's best z-score and best cosine against its paragraphs. Kept
    for as long as the theme's text and the index stay the same: typing changes one theme, and
    the table of all passages against all paragraphs would not fit in memory for a large corpus."""

    def score():
        best_z, best_cos = None, None
        for start in range(0, len(vecs), THEME_BATCH):
            cos, z = zscores(snapshot, vecs[start : start + THEME_BATCH], scope)
            z, cos = z.max(1), cos.max(1)
            best_z = z if best_z is None else np.maximum(best_z, z)
            best_cos = cos if best_cos is None else np.maximum(best_cos, cos)
        return best_z, best_cos

    key = (scope_key, *sorted(f"{u.heading}\0{u.text}" for u in theme_units))
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
    excluded_docs = set(prefs.excluded_docs)
    excerpts = [e for u in units if u.excerpt and len(e := _norm(u.text)) >= 20]
    joined = "\0".join(excerpts)

    def near_excerpt(hay: str) -> bool:  # inside a collected excerpt, or holding one
        return hay in joined or any(e in hay for e in excerpts)

    gated = []
    for row in map(int, rows):
        if row in collected:
            continue
        p = snapshot.passages[row]
        if p.path in excluded_docs:
            continue
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
        note = "Write or collect text in your task to see results." if n else "No documents found."
        return RelatedResult([Lane(lane, *LANE_INFO[lane], [], note) for lane in LANES], [], [], {"passages": n})

    # 1: units and themes
    semantic = embedder is not None and snapshot.has_vector.any()
    unit_vecs = None
    find_excerpts(snapshot, units)
    if semantic:
        reuse = [same_passage(snapshot, u.source_rows, u.text) for u in units]
        unit_vecs = embed_reusing(snapshot, embedder, [u.embed_text for u in units], reuse)
        find_excerpts(snapshot, units, unit_vecs)
    weight_units(units, config)
    themes = assign_themes(units, unit_vecs, config, snapshot)
    lap("units")

    # 2: signals. Left-out folders are gone before anything is ranked, so they take no places, and
    # are neither compared by meaning nor scanned for patterns.
    liked = [d for d in prefs.liked if d.get("text")] if config.like_weight > 0 else []
    keep = snapshot.outside_folders(skip_folders) if skip_folders else None
    rows = None if keep is None else np.flatnonzero(keep)
    if semantic:
        signals = semantic_signals(snapshot, embedder, units, unit_vecs, themes, liked, prefs, config, keep,
                                   tuple(sorted(skip_folders)))
    else:
        signals = SemanticSignals(np.full(n, -np.inf), np.full(n, -np.inf))
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
    lex = lexical_scores(snapshot, bm25, [i for i, _ in query], [w for _, w in query], steer.patterns, rows)
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
            scored.notes[lane] = "This group needs meaning data, which isn't available yet. Wait until it's ready or start the app with a meaning model."
    else:
        score_hidden(gated, scored, signals, lex, config)
        if signals.theme_z is None:
            scored.notes["bridges"] = scored.notes["gaps"] = "This group needs at least two topics. Write about more than one topic in your task."
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
