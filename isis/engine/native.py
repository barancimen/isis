"""The optional Rust extension, isis_native (native/ in the repository; `pip install ./native`).

Two jobs of a search are loops over text in Python's regex engine: finding the words to highlight
in the lines of a result, and counting a phrase in every passage. The extension does both several
times faster, with the same answers (tests/test_native.py). Without it every function here answers
None, and the caller runs its Python code.

It never takes part in building the index: the words of the corpus are always split by text.py."""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable

from .text import STOPWORDS

try:
    import isis_native
except ImportError:  # not built
    isis_native = None

MAX_PHRASES = 256  # phrase patterns whose words are remembered
_phrases: dict[tuple[str, int], tuple[str, ...]] = {}
_lock = threading.Lock()


def line_spans(lines: list[str], terms: set[str]) -> list[list[tuple[int, int]]] | None:
    """term_spans (text.py) of each line, in one call."""
    if isis_native is None:
        return None
    wanted = terms - STOPWORDS  # the tokeniser drops them, so they are never highlighted
    return isis_native.term_spans(lines, wanted) if wanted else [[] for _ in lines]


def remember_phrase(pattern: re.Pattern, words: list[str]) -> None:
    """Note that `pattern` is these words with whitespace between them, in any case, so that
    phrase_counts can count it without the regex. A regex a user writes to the same pattern means
    the same, and is counted the same way."""
    if isis_native is None or not words:
        return
    with _lock:
        while len(_phrases) >= MAX_PHRASES:
            _phrases.pop(next(iter(_phrases)))
        _phrases[(pattern.pattern, pattern.flags)] = tuple(words)


def phrase_counts(pattern, texts: Iterable[str]) -> list[int] | None:
    """How often a pattern noted by remember_phrase matches in each text. None for any other
    pattern, and `texts` is then left untouched."""
    if isis_native is None:
        return None
    with _lock:
        words = _phrases.get((pattern.pattern, pattern.flags))
    if not words:
        return None
    try:
        return isis_native.count_phrase(list(texts), list(words))
    except (ValueError, UnicodeError):  # text it can't take, such as half of a surrogate pair
        return None
