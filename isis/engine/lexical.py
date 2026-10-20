"""BM25 as a sparse matrix.

Each row is a passage, each column a term, and each value that term's BM25 contribution to that
passage. Scoring a query with per-term weights is then one sparse matrix-vector product,
`scores = W[:, terms] @ weights`, so typed queries and the weighted queries built from a task
document go through the same code.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


class Vocabulary:
    """Term <-> column id. Only grows, so ids stay valid across index updates."""

    def __init__(self, terms: list[str] | None = None):
        self.terms: list[str] = list(terms or [])
        self.ids: dict[str, int] = {t: i for i, t in enumerate(self.terms)}

    def __len__(self) -> int:
        return len(self.terms)

    def get(self, term: str) -> int | None:
        return self.ids.get(term)

    def add(self, term: str) -> int:
        i = self.ids.get(term)
        if i is None:
            i = self.ids[term] = len(self.terms)
            self.terms.append(term)
        return i


def term_counts(token_lists: list[list[str]], vocab: Vocabulary):
    """CSR pieces (indptr, term ids, counts) for a list of tokenised passages, growing `vocab`."""
    indptr = np.zeros(len(token_lists) + 1, dtype=np.int64)
    id_parts, count_parts = [], []
    for row, tokens in enumerate(token_lists):
        ids, counts = np.unique(np.fromiter((vocab.add(t) for t in tokens), np.int32, len(tokens)), return_counts=True)
        id_parts.append(ids)
        count_parts.append(counts.astype(np.float32))
        indptr[row + 1] = indptr[row] + len(ids)
    term_ids = np.concatenate(id_parts) if id_parts else np.zeros(0, np.int32)
    counts = np.concatenate(count_parts) if count_parts else np.zeros(0, np.float32)
    return indptr, term_ids, counts


class BM25:
    def __init__(self, indptr, term_ids, counts, n_terms: int, k1: float = 1.2, b: float = 0.75):
        n_docs = len(indptr) - 1
        self._raw = (indptr, term_ids, counts)
        self.k1, self.b = k1, b
        rows = np.repeat(np.arange(n_docs), np.diff(indptr))
        doc_len = np.bincount(rows, weights=counts, minlength=n_docs)
        avg_len = max(doc_len.mean(), 1e-9) if n_docs else 1.0

        self.n_docs = n_docs
        self.n_terms = n_terms
        self.doc_len, self.avg_len = doc_len.astype(np.float32), float(avg_len)
        self.doc_freq = np.bincount(term_ids, minlength=n_terms)
        self.idf = np.log1p((n_docs - self.doc_freq + 0.5) / (self.doc_freq + 0.5)).astype(np.float32)

        tf = counts
        data = self.idf[term_ids] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len[rows] / avg_len))
        self.matrix = sparse.csr_matrix(
            (data.astype(np.float32), term_ids, indptr), shape=(n_docs, n_terms)
        ).tocsc()

    def with_params(self, k1: float, b: float) -> "BM25":
        """The same index weighted with another k1 (how fast repeated words saturate) and b (how
        much long passages are normalised)."""
        if (k1, b) == (self.k1, self.b):
            return self
        return BM25(*self._raw, n_terms=self.n_terms, k1=k1, b=b)

    def pattern_scores(self, counts: np.ndarray) -> np.ndarray:
        """BM25 of a pseudo-term (a phrase or a regex) from how often it matches in each passage, so
        that it ranks like one more word of the query."""
        df = int((counts > 0).sum())
        if not df:
            return np.zeros(self.n_docs, np.float32)
        idf = np.log1p((self.n_docs - df + 0.5) / (df + 0.5))
        tf = counts.astype(np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avg_len)
        return (idf * tf * (self.k1 + 1) / (tf + norm)).astype(np.float32)

    def scores(self, term_ids, weights) -> np.ndarray:
        """BM25 score of every passage for a query given as term ids with weights."""
        if len(term_ids) == 0:
            return np.zeros(self.n_docs, np.float32)
        return self.matrix[:, np.asarray(term_ids)] @ np.asarray(weights, np.float32)
