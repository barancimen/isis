"""The corpus words within a typo or two of a word, for typo-tolerant search.

Every distinct word of the corpus is kept as written (lowercased), with the search term it stems
to. Typos are measured on these words, not on stems: a typo near the end of a word changes how it
stems, so "informaton" is four letters from the stem of "information" ("inform") but one typo
from the word.

Looking up a word is a scan of the whole list in two steps. A filter in numpy keeps only the
words that could be close, and rapidfuzz (C++) counts the typos in those. A typo (a wrong,
missing or extra letter, or two letters swapped) changes a word's length by at most one and its
set of letters by at most two, so the filter never drops a word that is close enough.

Like Lucene's spelling suggester (prefix length 1), a near word must start with the same letter:
typists and OCR rarely get the first letter wrong, while many real words are a first letter
apart from each other (squid and quid, sea and tea).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import OSA

MAX_WORD_LENGTH = 40  # longer "words" are mostly junk (hashes, run-together text) and nobody types them


def _bit(code: int) -> int:
    if 97 <= code <= 122:  # a to z
        return code - 97
    if 48 <= code <= 57:  # 0 to 9
        return code - 22
    return 36 + code % 28


def letter_mask(word: str) -> int:
    """The set of characters in `word` as bits: a to z, the digits, and the rest folded into 28
    bits. Folding can only make two words look closer, never further apart."""
    mask = 0
    for ch in word:
        mask |= 1 << _bit(ord(ch))
    return mask


_BITS = np.array([0] + [1 << _bit(c) for c in range(1, 0x10000)], np.uint64)  # 0 pads shorter words


def letter_masks(words: list[str]) -> np.ndarray:
    """letter_mask of each word, in numpy."""
    codes = np.array(words, dtype=str)
    codes = codes.view(np.uint32).reshape(len(words), -1) if len(words) else np.zeros((0, 1), np.uint32)
    if codes.size and codes.max() >= 0x10000:  # outside the Basic Multilingual Plane: rare
        return np.array([letter_mask(w) for w in words], np.uint64)
    return np.bitwise_or.reduce(_BITS[codes], axis=1)


@dataclass(frozen=True)
class WordView:
    """The first words of a Words list, as they were when a snapshot was made."""

    words: list[str]  # shared with the Words list, which only appends: read the first len(terms)
    terms: np.ndarray  # int32 term id of each word
    masks: np.ndarray  # uint64 letter_mask of each word
    lengths: np.ndarray  # uint8
    firsts: np.ndarray  # uint32 code of each word's first character

    def __len__(self) -> int:
        return len(self.terms)

    def near(self, word: str, limit: int) -> tuple[np.ndarray, np.ndarray]:
        """(indices, typos) of the words 1 to `limit` typos from `word` that start like it."""
        if not len(self) or limit < 1 or not word:
            return np.zeros(0, np.int64), np.zeros(0, np.uint8)
        mask = np.uint64(letter_mask(word))
        close = self.firsts == ord(word[0])
        close &= np.abs(self.lengths.astype(np.int16) - len(word)) <= limit
        close &= np.bitwise_count(self.masks ^ mask) <= 2 * limit
        rows = np.flatnonzero(close)
        if not len(rows):
            return rows, np.zeros(0, np.uint8)
        typos = process.cdist([word], [self.words[i] for i in rows], scorer=OSA.distance,
                              score_cutoff=limit, dtype=np.uint8)[0]
        keep = (typos >= 1) & (typos <= limit)
        return rows[keep], typos[keep]


class Words:
    """Every word met so far, with its term id. Like the Vocabulary it only grows, so a snapshot's
    view of the first n words stays valid while a refresh adds more."""

    def __init__(self):
        self.ids: dict[str, int] = {}
        self.words: list[str] = []
        self._terms = np.zeros(1024, np.int32)
        self._masks = np.zeros(1024, np.uint64)
        self._lengths = np.zeros(1024, np.uint8)
        self._firsts = np.zeros(1024, np.uint32)

    def __len__(self) -> int:
        return len(self.words)

    def add(self, word: str, term: int) -> None:
        if not word or word in self.ids or len(word) > MAX_WORD_LENGTH:
            return
        n = len(self.words)
        if n == len(self._terms):  # new arrays: views of the old ones keep them
            self._terms, self._masks, self._lengths, self._firsts = (
                np.concatenate([a, np.zeros_like(a)]) for a in (self._terms, self._masks, self._lengths, self._firsts))
        self._terms[n], self._masks[n], self._lengths[n], self._firsts[n] = term, letter_mask(word), len(word), ord(word[0])
        self.ids[word] = n
        self.words.append(word)

    def extend(self, words: list[str], terms, masks: np.ndarray | None = None) -> None:
        """add() for many words at once, as when a cache is loaded. `masks` are their letter_masks
        if known, as a cache keeps them; otherwise they are worked out in numpy."""
        n = len(self.words)
        keep, fresh, fresh_terms = [], [], []
        for i, (w, t) in enumerate(zip(words, terms)):
            if w and len(w) <= MAX_WORD_LENGTH and w not in self.ids:
                self.ids[w] = n + len(fresh)
                fresh.append(w)
                fresh_terms.append(t)
                keep.append(i)
        m = len(fresh)
        if not m:
            return
        if n + m > len(self._terms):
            size = max(n + m, 2 * len(self._terms))
            self._terms, self._masks, self._lengths, self._firsts = (
                np.concatenate([a[:n], np.zeros(size - n, a.dtype)])
                for a in (self._terms, self._masks, self._lengths, self._firsts))
        self._terms[n:n + m] = fresh_terms
        self._lengths[n:n + m] = [len(w) for w in fresh]
        self._firsts[n:n + m] = [ord(w[0]) for w in fresh]
        if masks is not None:
            self._masks[n:n + m] = np.asarray(masks, np.uint64)[keep]
        else:
            for start in range(0, m, 65536):
                block = fresh[start:start + 65536]
                self._masks[n + start:n + start + len(block)] = letter_masks(block)
        self.words.extend(fresh)

    def view(self) -> WordView:
        n = len(self.words)
        return WordView(self.words, self._terms[:n], self._masks[:n], self._lengths[:n], self._firsts[:n])

    def of_terms(self, terms: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
        """The words whose term is one of the sorted `terms`, with each one's index in `terms` and
        its letter_mask."""
        n = len(self.words)
        where = np.searchsorted(terms, self._terms[:n]).clip(0, max(len(terms) - 1, 0))
        rows = np.flatnonzero(terms[where] == self._terms[:n]) if len(terms) else np.zeros(0, np.int64)
        return [self.words[i] for i in rows], where[rows].astype(np.int32), self._masks[rows]
