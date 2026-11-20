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

    def extend(self, terms: list[str]) -> np.ndarray:
        """add() for many terms at once, as when a cache is loaded: their ids, as int32."""
        if not self.terms:  # the first folder's cache: no loop in Python
            ids = dict(zip(terms, range(len(terms))))
            if len(ids) == len(terms):
                self.terms, self.ids = list(terms), ids
                return np.arange(len(terms), dtype=np.int32)
        start = len(self.ids)
        found = np.fromiter((self.ids.setdefault(t, len(self.ids)) for t in terms), np.int32, len(terms))
        new = [""] * (len(self.ids) - start)  # the terms met for the first time, in the order of their ids
        for t, i in zip(terms, found.tolist()):
            if i >= start:
                new[i - start] = t
        self.terms.extend(new)
        return found


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


# The defaults of Search and Related. A snapshot's own matrix uses them, so that a search or a
# Related refresh with default settings never has to build another one (seconds, in a large corpus).
K1, B = 1.2, 0.35


class BM25:
    """Built from its postings: for each term, the passages it is in (rows) and how often (tf), as
    the columns of a CSC matrix. A cache keeps them, so loading one needn't turn every passage's
    terms around, and other k1 and b only weigh them again."""

    def __init__(self, indptr, term_ids, counts, n_terms: int, k1: float = 1.2, b: float = 0.75):
        """From each passage's terms, as CSR pieces."""
        self._weigh(*postings_of(indptr, term_ids, counts, n_terms), len(indptr) - 1, k1, b)

    @classmethod
    def from_postings(cls, indptr, rows, tf, n_docs: int, k1: float = 1.2, b: float = 0.75, doc_len=None) -> BM25:
        bm25 = cls.__new__(cls)
        bm25._weigh(indptr, rows, tf, n_docs, k1, b, doc_len)
        return bm25

    def _weigh(self, indptr, rows, tf, n_docs: int, k1: float, b: float, doc_len=None) -> None:
        self.postings = (indptr, rows, tf)
        self.k1, self.b = k1, b
        self.n_docs, self.n_terms = n_docs, len(indptr) - 1
        if doc_len is None:
            doc_len = np.bincount(rows, weights=tf, minlength=n_docs).astype(np.float32)
        self.doc_len = doc_len
        self.avg_len = float(max(doc_len.mean(), 1e-9)) if n_docs else 1.0
        self.doc_freq = np.diff(indptr)
        self.idf = np.log1p((n_docs - self.doc_freq + 0.5) / (self.doc_freq + 0.5)).astype(np.float32)
        norm = (k1 * (1 - b + b * doc_len / self.avg_len)).astype(np.float32)
        denominator = norm[rows]
        denominator += tf
        data = np.multiply(tf, np.float32(k1 + 1), dtype=np.float32)
        data /= denominator
        data *= np.repeat(self.idf, self.doc_freq)
        self.matrix = sparse.csc_matrix((data, rows, indptr), shape=(n_docs, self.n_terms))

    def with_params(self, k1: float, b: float) -> BM25:
        """The same index weighted with another k1 (how fast repeated words saturate) and b (how
        much long passages are normalised)."""
        if (k1, b) == (self.k1, self.b):
            return self
        return BM25.from_postings(*self.postings, self.n_docs, k1, b, self.doc_len)


    def pattern_scores(self, counts: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        """BM25 of a pseudo-term (a phrase or a regex) from how often it matches in each passage, so
        that it ranks like one more word of the query. If it was only matched in `rows`, how many
        passages of the corpus it is in is estimated from those, as words count the whole corpus."""
        if rows is None:
            df = float((counts > 0).sum())
        else:
            df = float((counts[rows] > 0).sum()) * self.n_docs / max(len(rows), 1)
        if not df:
            return np.zeros(self.n_docs, np.float32)
        idf = np.log1p((self.n_docs - df + 0.5) / (df + 0.5))
        tf = counts.astype(np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avg_len)
        # Only where it matches, like the terms in the matrix. Elsewhere the score is 0, and the
        # sum below would be 0/0 if norm is 0: with k1 at 0, or b at 1 for a passage of stopwords.
        scores = np.zeros(self.n_docs, np.float32)
        hit = tf > 0
        scores[hit] = idf * tf[hit] * (self.k1 + 1) / (tf[hit] + norm[hit])
        return scores

    def scores(self, term_ids, weights) -> np.ndarray:
        """BM25 score of every passage for a query given as term ids with weights."""
        if len(term_ids) == 0:
            return np.zeros(self.n_docs, np.float32)
        return self.matrix[:, np.asarray(term_ids)] @ np.asarray(weights, np.float32)


def merge_postings(blocks, n_terms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One set of postings (CSC pieces) from several, each (indptr, rows, tf, terms, row offset): the
    passages of each are the next rows, and `terms` gives the term of each of its columns (None: the
    same column). Each term's passages stay in the order of the blocks. O(n), with no sort."""
    if len(blocks) == 1 and blocks[0][3] is None and blocks[0][4] == 0 and len(blocks[0][0]) == n_terms + 1:
        return blocks[0][:3]
    columns = []
    sizes = np.zeros(n_terms, np.int64)
    for indptr, _, _, terms, _ in blocks:
        counts = np.diff(indptr)
        used = np.flatnonzero(counts)  # a block's columns are its terms once each, so plain indexing adds them
        cols = used if terms is None else np.asarray(terms, np.int64)[used]
        sizes[cols] += counts[used]
        columns.append((cols, counts[used], np.asarray(indptr)[used]))
    out_indptr = np.zeros(n_terms + 1, np.int64)
    np.cumsum(sizes, out=out_indptr[1:])
    out_rows, out_tf = np.empty(out_indptr[-1], np.int32), np.empty(out_indptr[-1], np.float32)
    filled = out_indptr[:-1].copy()
    for (_, rows, tf, _, offset), (cols, counts, starts) in zip(blocks, columns):
        where = np.repeat(filled[cols] - starts, counts) + np.arange(len(rows))
        out_rows[where] = rows + offset if offset else rows
        out_tf[where] = tf
        filled[cols] += counts
    return out_indptr, out_rows, out_tf


def postings_of(indptr, term_ids, counts, n_terms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The postings (CSC pieces) of passages given by their terms (CSR pieces)."""
    by_term = sparse.csr_matrix((np.asarray(counts, np.float32), np.asarray(term_ids), np.asarray(indptr)),
                                shape=(len(indptr) - 1, n_terms)).tocsc()
    return by_term.indptr.astype(np.int64, copy=False), by_term.indices.astype(np.int32, copy=False), by_term.data
