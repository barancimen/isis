import math
import warnings

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
    _, bm25 = build([])
    assert bm25.n_docs == 0 and bm25.scores([], []).shape == (0,)
    assert np.isfinite(bm25.idf).all()


@pytest.mark.parametrize("k1,b", [(0.0, 0.75), (1.2, 1.0), (0.0, 1.0), (0.0, 0.0), (3.0, 1.0)])
def test_pattern_scores_are_finite_at_the_ends_of_the_sliders(k1, b):
    """A passage with no words (only stopwords, say) has length 0. Passages without a match score 0."""
    vocab = Vocabulary()
    bm25 = BM25(*term_counts([["whale", "sea"], [], ["ship"]], vocab), n_terms=len(vocab), k1=k1, b=b)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scores = bm25.pattern_scores(np.array([2.0, 0.0, 0.0]))
    assert np.isfinite(scores).all()
    assert scores[0] > 0 and scores[1] == scores[2] == 0


def test_pattern_scores_are_unchanged_where_they_match():
    vocab = Vocabulary()
    bm25 = BM25(*term_counts([["whale", "sea"], ["ship", "sea", "oil"], ["ship"]], vocab), n_terms=len(vocab), k1=1.2, b=0.75)
    counts = np.array([2.0, 1.0, 0.0])
    idf = np.log1p((3 - 2 + 0.5) / (2 + 0.5))
    expected = idf * counts * 2.2 / (counts + 1.2 * (0.25 + 0.75 * bm25.doc_len / bm25.avg_len))
    assert np.allclose(bm25.pattern_scores(counts)[:2], expected[:2]) and bm25.pattern_scores(counts)[2] == 0


def test_bm25_variants_match_a_fresh_index():
    docs = [["whale", "sea", "whale"], ["ship", "sea"], ["ship", "captain", "ship", "sea", "storm", "storm"]]
    vocab = Vocabulary()
    pieces = term_counts(docs, vocab)
    base = BM25(*pieces, n_terms=len(vocab))
    ids, weights = [vocab.get("ship"), vocab.get("storm")], [1.0, 1.0]
    fresh = BM25(*pieces, n_terms=len(vocab), k1=2.0, b=0.2).scores(ids, weights)
    assert base.with_params(2.0, 0.2).scores(ids, weights) == pytest.approx(fresh)
    assert base.with_params(1.2, 0.75) is base


def test_snapshot_keeps_a_few_variants(engine):
    snap = engine.snapshot
    assert snap.bm25_for(1.2, 0.75) is snap.bm25
    first = snap.bm25_for(0.5, 0.5)
    assert snap.bm25_for(0.5, 0.5) is first
    for k1 in (0.1, 0.2, 0.3, 0.4, 0.6):
        snap.bm25_for(k1, 0.5)
    assert len(snap._bm25_variants) <= 4
