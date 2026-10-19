import math

import numpy as np
import pytest

from isis.engine.lexical import BM25, Vocabulary, term_counts

DOCS = [
    ["whale", "sea", "whale"],
    ["ship", "sea"],
    ["ship", "captain", "ahab", "ship", "sea", "storm"],
]


def build(docs=DOCS):
    vocab = Vocabulary()
    return vocab, BM25(*term_counts(docs, vocab), n_terms=len(vocab))


def reference_score(docs, query, k1=1.2, b=0.75):
    """Textbook BM25 (Lucene idf), term by term."""
    avg = sum(map(len, docs)) / len(docs)
    out = []
    for doc in docs:
        total = 0.0
        for term in query:
            df = sum(term in d for d in docs)
            idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
            tf = doc.count(term)
            total += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avg))
        out.append(total)
    return out


def test_matches_textbook_bm25():
    vocab, bm25 = build()
    query = ["whale", "ship", "storm"]
    got = bm25.scores([vocab.get(t) for t in query], [1, 1, 1])
    assert got == pytest.approx(reference_score(DOCS, query), rel=1e-5)


def test_query_weights_scale_each_term():
    vocab, bm25 = build()
    ids = [vocab.get("whale"), vocab.get("ship")]
    one = bm25.scores([ids[0]], [1]) * 2 + bm25.scores([ids[1]], [1]) * 0.5
    assert bm25.scores(ids, [2, 0.5]) == pytest.approx(one)


def test_empty_query_and_document_frequencies():
    vocab, bm25 = build()
    assert not bm25.scores([], []).any()
    assert bm25.doc_freq[vocab.get("sea")] == 3
    assert bm25.doc_freq[vocab.get("ahab")] == 1


def test_vocabulary_only_grows():
    vocab = Vocabulary(["a", "b"])
    assert vocab.add("b") == 1 and vocab.add("c") == 2 and vocab.get("zzz") is None


def test_empty_index():
    vocab, bm25 = build([])
    assert bm25.n_docs == 0 and bm25.scores([], []).shape == (0,)
    assert np.isfinite(bm25.idf).all()
