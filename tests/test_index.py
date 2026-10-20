import os
import pickle
import time

import numpy as np
import pytest

from isis.engine.index import Engine
from isis.engine.semantic import HashEmbedder


def paths(hits):
    return [h.path for h in hits]


def test_indexes_markdown_only_and_skips_hidden_folders(engine):
    assert sorted(engine.files) == ["finches.md", "ships/pequod.md", "whales.md"]
    assert engine.status.state == "ready" and engine.status.passages == 5
    assert engine.snapshot.has_vector.all()


def test_search_returns_location_snippet_and_highlights(engine):
    (hit,) = engine.search("ambergris")
    assert (hit.path, hit.heading, hit.start_line, hit.end_line) == ("whales.md", "Whales", 4, 4)
    assert hit.match == (4, 0, len("Ambergris"))
    line = next(s for s in hit.snippet if s.line == 4)
    assert line.in_passage and line.marks == [(0, 9)]
    assert [s.line for s in hit.snippet] == [2, 3, 4, 5]  # two lines of context either side


def test_ranking_prefers_more_and_rarer_matches(engine):
    hits = engine.search("whale captain")
    assert hits[0].path == "ships/pequod.md" and "Ahab" in hits[0].text


def test_partly_typed_last_word_is_expanded(engine):
    assert "finches.md" in paths(engine.search("galap"))
    assert engine.search("galap ") == []  # a finished word is not expanded


def test_quoted_phrases_must_appear_verbatim(engine):
    assert paths(engine.search('"white whale"')) == ["ships/pequod.md"]
    assert engine.search('"whale white"') == []


def test_hybrid_mode_fuses_both_signals(engine):
    hits = engine.search("whale", mode="hybrid")
    assert {"whales.md", "ships/pequod.md"} <= set(paths(hits))
    assert {"bm25", "cosine", "keyword_rank", "meaning_rank"} <= set(hits[0].why)


def test_hybrid_mode_keeps_the_best_of_each_signal(engine):
    hits = engine.search("finches beaks", mode="hybrid", limit=3)
    assert hits[0].path == "finches.md" and hits[0].why["keyword_rank"] == 1 and hits[0].why["meaning_rank"] == 1
    meaning_only = [h for h in engine.search("finches beaks", mode="hybrid") if h.why["keyword_rank"] is None]
    assert meaning_only and all(h.why["meaning_rank"] for h in meaning_only)  # found by meaning alone, and labelled so


def test_refresh_picks_up_edits_new_files_and_deletions(engine, corpus):
    (corpus / "whales.md").write_text("# Whales\n\nNothing about that substance now.\n")
    os.utime(corpus / "whales.md", ns=(time.time_ns(), time.time_ns() + 10**9))
    (corpus / "new.md").write_text("Ambergris washed up on the beach.\n")
    (corpus / "finches.md").unlink()
    stats = engine.refresh()
    assert (stats["changed"], stats["removed"]) == (2, 1)
    assert paths(engine.search("ambergris")) == ["new.md"]
    assert "finches.md" not in engine.snapshot.file_rows


def test_cache_is_reused_and_model_change_re_embeds(corpus, tmp_path):
    index = tmp_path / "index"
    Engine(corpus, embedder=HashEmbedder(), index_dir=index).refresh()

    again = Engine(corpus, embedder=HashEmbedder(), index_dir=index)
    assert again.snapshot.has_vector.all()  # usable before any refresh
    stats = again.refresh()
    assert (stats["changed"], stats["removed"], stats["embedded"]) == (0, 0, 0)

    other = Engine(corpus, embedder=HashEmbedder(dim=32), index_dir=index)
    assert not other.snapshot.has_vector.any()
    stats = other.refresh()
    assert stats["changed"] == 0 and stats["embedded"] == other.status.passages
    assert other.snapshot.vectors.shape[1] == 32



def test_a_run_without_a_model_keeps_the_embeddings(corpus, tmp_path):
    index = tmp_path / "index"
    Engine(corpus, embedder=HashEmbedder(), index_dir=index).refresh()
    keywords_only = Engine(corpus, index_dir=index)
    (corpus / "new.md").write_text("Ambergris washed up on the beach.\n")
    assert keywords_only.refresh()["changed"] == 1  # saves the index

    again = Engine(corpus, embedder=HashEmbedder(), index_dir=index)
    assert again.refresh()["embedded"] == 1  # only the new file


class Planted:
    def __reduce__(self):
        return (os.system, ("touch planted-ran",))


def test_a_planted_pickle_is_never_loaded(corpus, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (corpus / ".isis").mkdir()
    (corpus / ".isis" / "index.pkl").write_bytes(pickle.dumps(Planted()))
    e = Engine(corpus, embedder=HashEmbedder())
    e.refresh()
    assert not (tmp_path / "planted-ran").exists()
    assert not (corpus / ".isis" / "index.pkl").exists()  # replaced by the new cache
    assert Engine(corpus, embedder=HashEmbedder()).snapshot.has_vector.all()

def test_lexical_only_engine_works_without_a_model(corpus, tmp_path):
    e = Engine(corpus, index_dir=tmp_path / "index")
    e.refresh()
    assert paths(e.search("ambergris")) == ["whales.md"]
    assert e.snapshot.vectors.shape == (e.status.passages, 0)


def test_excluded_folders_are_skipped(corpus, tmp_path):
    e = Engine(corpus, index_dir=tmp_path / "index", exclude=(corpus / "ships",))
    e.refresh()
    assert "ships/pequod.md" not in e.files


def test_paths_outside_the_corpus_are_refused(engine):
    for bad in ("../outside.md", "/etc/passwd", "notes.txt"):
        with pytest.raises(ValueError):
            engine.resolve(bad)


def test_snapshot_vectors_are_normalised(engine):
    norms = np.linalg.norm(engine.snapshot.vectors, axis=1)
    assert np.allclose(norms, 1, atol=1e-2)
