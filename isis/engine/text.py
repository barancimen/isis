"""Tokenising for lexical search: lowercase words, English stopwords dropped. Words are not
stemmed: a search for "conditional" finds "conditional", not "condition" or "conditions"."""

from __future__ import annotations

import re

WORD = re.compile(r"[^\W_]+")

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing don down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its itself
    just me more most my myself no nor not now of off on once only or other our ours ourselves out
    over own same she should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where which while who
    whom why will with would you your yours yourself yourselves""".split()
)


def _words(text: str) -> list[tuple[int, int, str]]:
    """(start, end, lowercase word) for every word that counts as a search term."""
    out = []
    for m in WORD.finditer(text):
        word = m.group().lower()
        if len(word) > 1 and word not in STOPWORDS:
            out.append((m.start(), m.end(), word))
    return out


def tokenize(text: str) -> list[str]:
    """Search terms in `text`, in order."""
    return [w for _, _, w in _words(text)]


def tokenize_with_surface(text: str) -> list[tuple[str, str]]:
    """(term, the word as written) pairs. A term is the word in lowercase, so the two are the same
    today; callers keep both so that the index can map written words to terms."""
    return [(w, w) for _, _, w in _words(text)]


def term_spans(text: str, terms: set[str]) -> list[tuple[int, int]]:
    """Character ranges of the words in `text` whose term is in `terms`, for highlighting."""
    if not terms:
        return []
    return [(s, e) for s, e, w in _words(text) if w in terms]
