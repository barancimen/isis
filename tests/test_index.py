import json
import os
import pickle
import re
import time
from pathlib import Path

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


def test_switching_back_to_a_model_reuses_its_kept_embeddings(corpus, tmp_path):
    index = tmp_path / "index"
    first = Engine(corpus, embedder=HashEmbedder(), index_dir=index)
    first.refresh()
    vectors = first.snapshot.vectors.copy()
    Engine(corpus, embedder=HashEmbedder(dim=32), index_dir=index).refresh()
    (corpus / "finches.md").write_text("# Finches\n\nTheir beaks differ from island to island.\n")
    Engine(corpus, embedder=HashEmbedder(dim=32), index_dir=index).refresh()  # an edit under the other model

    back = Engine(corpus, embedder=HashEmbedder(), index_dir=index)
    stats = back.refresh()
    assert stats["embedded"] == 1 and back.snapshot.has_vector.all()  # only the edited passage
    rows = back.snapshot.file_rows["whales.md"]
    assert np.allclose(back.snapshot.vectors[rows[0]:rows[1]], vectors[rows[0]:rows[1]], atol=1e-3)
    assert Engine(corpus, embedder=HashEmbedder(dim=32), index_dir=index).refresh()["embedded"] == 0



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


def test_an_edit_re_embeds_only_the_changed_passages(corpus, tmp_path):
    e = Engine(corpus, embedder=HashEmbedder(), index_dir=tmp_path / "index", target_words=0)
    e.refresh()
    before = {p.text: e.snapshot.vectors[i].copy() for i, p in enumerate(e.snapshot.passages)}
    with open(corpus / "whales.md", "a") as f:
        f.write("\nSperm whales sleep upright.\n")
    stats = e.refresh()
    assert (stats["changed"], stats["embedded"]) == (1, 1)
    for i, p in enumerate(e.snapshot.passages):
        if p.text in before:
            assert np.allclose(e.snapshot.vectors[i], before[p.text], atol=1e-3)


def test_a_failed_refresh_is_reported(corpus, tmp_path, monkeypatch):
    e = Engine(corpus, embedder=HashEmbedder(), index_dir=tmp_path / "index")
    monkeypatch.setattr(e.embedder, "embed", lambda texts, cache=False: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        e.refresh()
    assert e.status.state == "error" and "ZeroDivisionError" in e.status.error
    monkeypatch.undo()
    e.refresh()
    assert (e.status.state, e.status.error) == ("ready", "")


def test_a_cache_with_missing_parts_is_rebuilt(corpus, tmp_path):
    index = tmp_path / "index"
    Engine(corpus, embedder=HashEmbedder(), index_dir=index).refresh()
    meta = json.loads((index / "index.json").read_text())
    del meta["vocab"]
    (index / "index.json").write_text(json.dumps(meta))
    e = Engine(corpus, embedder=HashEmbedder(), index_dir=index)  # no crash
    assert e.refresh()["changed"] == 3


def linked_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "real").mkdir(parents=True)
    (corpus / "real" / "here.md").write_text("Heronword is in the corpus.\n")
    outside = tmp_path / "outside"
    (outside / "sub" / "deeper").mkdir(parents=True)
    (outside / "sub" / "deeper" / "there.md").write_text("Otterword is outside the corpus.\n")
    (outside / "secret.txt").write_text("not markdown")
    return corpus, outside


def test_linked_folders_are_scanned_and_open(tmp_path):
    corpus, outside = linked_corpus(tmp_path)
    (corpus / "link").symlink_to(outside)
    e = Engine(corpus, index_dir=tmp_path / "index")
    e.refresh()
    assert paths(e.search("otterword")) == ["link/sub/deeper/there.md"]
    assert e.read_lines("link/sub/deeper/there.md")[0].startswith("Otterword")
    assert e.links == {"link"}


def test_linked_paths_cannot_climb_out_or_reach_other_files(tmp_path):
    corpus, outside = linked_corpus(tmp_path)
    (corpus / "link").symlink_to(outside)
    (corpus / "disguised.md").symlink_to(outside / "secret.txt")
    e = Engine(corpus, index_dir=tmp_path / "index")
    e.refresh()
    assert "disguised.md" not in e.snapshot.file_rows
    for bad in ("../outside/sub/deeper/there.md", "link/../../outside/sub/deeper/there.md", "disguised.md", "link/secret.txt"):
        with pytest.raises(ValueError):
            e.resolve(bad)


def test_link_loops_end_and_nothing_is_indexed_twice(tmp_path):
    corpus, outside = linked_corpus(tmp_path)
    (corpus / "a-to-real").symlink_to(corpus / "real")  # sorts before the real folder
    (corpus / "real" / "up").symlink_to(corpus)
    (outside / "back").symlink_to(outside)
    (corpus / "one").symlink_to(outside)
    (corpus / "two").symlink_to(outside)
    (corpus / "a-file.md").symlink_to(corpus / "real" / "here.md")
    (corpus / "broken").symlink_to(tmp_path / "nothing")
    (corpus / "broken.md").symlink_to(tmp_path / "nothing.md")
    e = Engine(corpus, index_dir=tmp_path / "index")
    e.refresh()
    assert sorted(e.snapshot.file_rows) == ["one/sub/deeper/there.md", "real/here.md"]


def test_a_link_to_a_file_outside_is_indexed_and_follows_its_target(tmp_path):
    corpus, outside = linked_corpus(tmp_path)
    target = outside / "sub" / "deeper" / "there.md"
    (corpus / "single.md").symlink_to(target)
    e = Engine(corpus, index_dir=tmp_path / "index")
    e.refresh()
    assert paths(e.search("otterword")) == ["single.md"]
    target.write_text("Stoatword replaced it, at some length.\n")
    assert e.refresh()["changed"] == 1
    assert paths(e.search("stoatword")) == ["single.md"]


def test_links_into_the_tasks_folder_stay_out(tmp_path):
    corpus, _ = linked_corpus(tmp_path)
    tasks = corpus / "_tasks"
    tasks.mkdir()
    (tasks / "note.md").write_text("Newtword is a task.\n")
    (corpus / "tasks-link").symlink_to(tasks)
    (corpus / "note-link.md").symlink_to(tasks / "note.md")
    e = Engine(corpus, index_dir=tmp_path / "index", exclude=(tasks,))
    e.refresh()
    assert sorted(e.snapshot.file_rows) == ["real/here.md"]


def test_the_index_says_what_it_did(tmp_path, caplog):
    corpus, outside = linked_corpus(tmp_path)
    (corpus / "link").symlink_to(outside)
    (corpus / "broken").symlink_to(tmp_path / "nothing")
    index = tmp_path / "index"

    def said(action):
        caplog.clear()
        with caplog.at_level("INFO", logger="isis"):
            action()
        return [r.getMessage() for r in caplog.records]

    e = Engine(corpus, embedder=HashEmbedder(), index_dir=index)
    first = said(e.refresh)
    assert any(m.startswith("following link link -> ") for m in first)
    assert "2 markdown files: 2 new files (link/sub/deeper/there.md, real/here.md)" in first
    assert "embedding 2 passages" in first
    assert any("link broken skipped" in m for m in first)
    assert any(m.startswith("index ready: 2 files, 2 passages") for m in first)
    assert said(e.refresh) == []  # nothing changed: nothing to say

    (corpus / "real" / "more.md").write_text("Pikeword is new.\n")
    (corpus / "real" / "here.md").write_text("Heronword changed, in other words.\n")
    third = said(e.refresh)
    assert "1 new file (real/more.md), 1 changed (real/here.md)" in third
    assert "embedding 2 passages" in third

    again = Engine(corpus, embedder=HashEmbedder(), index_dir=index)
    fourth = said(again.refresh)
    assert "3 markdown files, no changes since the last run" in fourth


def test_a_no_embed_file_keeps_a_folder_to_keywords(corpus, tmp_path):
    index = tmp_path / "index"
    (corpus / "ships" / ".no-embed").touch()
    e = Engine(corpus, embedder=HashEmbedder(), index_dir=index)
    e.refresh()
    has = {p.path: bool(v) for p, v in zip(e.snapshot.passages, e.snapshot.has_vector)}
    assert has["whales.md"] and not has["ships/pequod.md"]
    assert e.status.unembedded == sum(p.path.startswith("ships/") for p in e.snapshot.passages)
    assert paths(e.search("pequod")) and paths(e.search("pequod"))[0] == "ships/pequod.md"  # still found by keywords
    assert ".no-embed" not in " ".join(e.snapshot.file_rows)

    (corpus / "ships" / "deep").mkdir()  # subfolders too
    (corpus / "ships" / "deep" / "more.md").write_text("Walrusword below the marker.\n")
    assert e.refresh()["embedded"] == 0

    (corpus / "ships" / ".no-embed").unlink()  # removing it embeds the folder
    stats = e.refresh()
    assert stats["embedded"] == sum(p.path.startswith("ships/") for p in e.snapshot.passages) and e.status.unembedded == 0

    (corpus / ".no-embed").touch()  # at the top: the whole corpus; embeddings already made are dropped
    assert e.refresh()["embedded"] == 0
    assert not e.snapshot.has_vector.any() and e.status.unembedded == len(e.snapshot)
    again = Engine(corpus, embedder=HashEmbedder(), index_dir=index)
    assert not again.snapshot.has_vector.any() and again.refresh()["embedded"] == 0


def test_hybrid_search_does_not_bury_passages_without_embeddings(tmp_path):
    from isis.engine.search import SearchConfig

    root = tmp_path / "c"
    (root / "plain").mkdir(parents=True)
    (root / "plain" / ".no-embed").touch()
    (root / "plain" / "exact.md").write_text("Coral reefs and coral atolls.\n")
    for i in range(6):  # embedded passages that share a word and so are close in meaning
        (root / f"e{i}.md").write_text(f"Coral and the sea, number {i}, with other words here.\n")
    e = Engine(root, embedder=HashEmbedder(), index_dir=tmp_path / "index")
    e.refresh()
    rank = lambda w: paths(e.search("coral reefs", mode="hybrid", config=SearchConfig(unembedded_weight=w))).index("plain/exact.md")
    assert rank(1.0) == 0 and rank(0.0) > 0
    hit = next(h for h in e.search("coral reefs", mode="hybrid") if h.path == "plain/exact.md")
    assert hit.why["embedded"] is False and hit.why["meaning_rank"] is None


def two_folders(tmp_path):
    a, b = tmp_path / "x" / "notes", tmp_path / "books"
    (a / "sub").mkdir(parents=True)
    b.mkdir()
    (a / "sub" / "n.md").write_text("Aardvarkword in the notes.\n")
    (b / "b.md").write_text("Bisonword in the books.\n")
    return a, b


def test_several_folders_are_one_corpus(tmp_path):
    a, b = two_folders(tmp_path)
    e = Engine([a, b], embedder=HashEmbedder())
    e.refresh()
    assert sorted(e.snapshot.file_rows) == ["books/b.md", "notes/sub/n.md"]
    assert paths(e.search("bisonword")) == ["books/b.md"]
    assert e.read_lines("notes/sub/n.md")[0].startswith("Aardvarkword")
    for bad in ("sub/n.md", "other/b.md", "notes/../../books/b.md", "notes"):
        with pytest.raises(ValueError):
            e.resolve(bad)

    (b / "c.md").write_text("Camelword arrives later.\n")  # edits and removals work per folder
    (a / "sub" / "n.md").unlink()
    stats = e.refresh()
    assert (stats["new"], stats["removed"]) == (1, 1)
    assert sorted(e.snapshot.file_rows) == ["books/b.md", "books/c.md"]


def test_each_folder_keeps_its_own_cache(tmp_path):
    a, b = two_folders(tmp_path)
    Engine([a, b], embedder=HashEmbedder()).refresh()
    assert (a / ".isis" / "index.json").exists() and (b / ".isis" / "index.json").exists()

    alone = Engine(b, embedder=HashEmbedder())  # served alone later: nothing to embed again
    assert alone.snapshot.has_vector.all()
    assert alone.refresh()["embedded"] == 0 and paths(alone.search("bisonword")) == ["b.md"]

    alone2 = Engine([b, a], embedder=HashEmbedder())  # either order, and terms keep their meaning
    assert alone2.refresh()["embedded"] == 0
    assert paths(alone2.search("aardvarkword")) == ["notes/sub/n.md"] and paths(alone2.search("bisonword")) == ["books/b.md"]
    assert alone2.index_bytes() > 0


def test_folders_with_the_same_name_or_inside_each_other(tmp_path):
    a, b = tmp_path / "one" / "notes", tmp_path / "two" / "notes"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "a.md").write_text("Aardvarkword.\n")
    (b / "b.md").write_text("Bisonword.\n")
    e = Engine([a, b], index_dir=tmp_path / "i")
    e.refresh()
    assert sorted(e.snapshot.file_rows) == ["notes-2/b.md", "notes/a.md"]

    nested = Engine([tmp_path / "one", a, tmp_path / "one"], index_dir=tmp_path / "j")  # a repeat, and a folder inside another
    nested.refresh()
    assert sorted(nested.snapshot.file_rows) == ["one/notes/a.md"]


def plain_and_embedded(tmp_path, embedded=10):
    root = tmp_path / "c"
    (root / "plain").mkdir(parents=True)
    (root / "plain" / ".no-embed").touch()
    for i in range(5):
        (root / "plain" / f"p{i}.md").write_text(f"The coral reef number {i} is plain.\n")
    for i in range(embedded):
        (root / f"e{i}.md").write_text(f"Coral and fish number {i} in the sea.\n")
    e = Engine(root, embedder=HashEmbedder(), index_dir=tmp_path / "i")
    e.refresh()
    return e


@pytest.mark.parametrize("query", ["coral reef", '"coral reef"', "/coral/", "+coral", "coral +/reef|fish/"])
def test_with_nothing_to_count_for_them_passages_without_embeddings_are_left_out(tmp_path, query):
    from isis.engine.search import SearchConfig

    e = plain_and_embedded(tmp_path)
    plain = lambda hits: [h for h in hits if h.path.startswith("plain/")]
    hits = e.search(query, mode="hybrid", config=SearchConfig(keyword_weight=0, unembedded_weight=0))
    assert hits and not plain(hits)
    assert plain(e.search(query, mode="hybrid", config=SearchConfig(keyword_weight=0.5, unembedded_weight=0)))
    assert plain(e.search(query, mode="hybrid"))  # the defaults keep them
    assert plain(e.search(query, mode="lexical"))
    lexical = e.search(query, mode="lexical", config=SearchConfig(keyword_weight=0, unembedded_weight=0))
    assert not plain(lexical)  # with or without 'meaning' ticked


def test_either_weight_keeps_passages_without_embeddings(tmp_path):
    from isis.engine.search import SearchConfig

    e = plain_and_embedded(tmp_path)
    hits = e.search("coral reef", mode="hybrid", config=SearchConfig(keyword_weight=0, unembedded_weight=1))
    assert any(h.path.startswith("plain/") for h in hits)
    assert all(h.score > 0 for h in hits)


def test_folders_can_be_left_out_of_a_search_or_suggestions(tmp_path):
    a, b = two_folders(tmp_path)
    for i in range(30):  # the left-out folder's many matches must not crowd out the other's
        (b / f"more{i}.md").write_text(f"The sperm whale harpoon ship number {i} sails on.\n")
    (a / "w.md").write_text("The sperm whale and the harpoon, on a ship.\n")
    e = Engine([a, b], embedder=HashEmbedder())
    e.refresh()
    for mode in ("lexical", "hybrid"):
        assert {h.path.split("/")[0] for h in e.search("whale harpoon", mode=mode)} == {"notes", "books"}
        assert {h.path.split("/")[0] for h in e.search("whale harpoon", mode=mode, skip_folders=["books"])} == {"notes"}
        assert e.search("whale harpoon", mode=mode, skip_folders=["books", "notes"]) == []
        assert e.search("+harpoon", mode=mode, skip_folders=["books"])[0].path == "notes/w.md"
    assert len(e.search("whale harpoon", skip_folders=["nonsense"])) == len(e.search("whale harpoon"))


def refused_walk(monkeypatch, folder):
    """os.walk, as if `folder` could not be read."""
    real = os.walk

    def walk(top, onerror=None, **kw):
        for dirpath, dirnames, filenames in real(top, **kw):
            if folder in dirnames:
                dirnames.remove(folder)
                onerror(OSError(13, "Permission denied", str(Path(dirpath) / folder)))
            yield dirpath, dirnames, filenames

    monkeypatch.setattr("isis.engine.index.os.walk", walk)


def test_a_folder_it_cannot_read_keeps_its_files_and_embeddings(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    (corpus / "sub").mkdir(parents=True)
    (corpus / "sub" / "a.md").write_text("Stoatword is here.\n")
    (corpus / "b.md").write_text("Pikeword is here.\n")
    e = Engine(corpus, embedder=HashEmbedder(), index_dir=tmp_path / "index")
    e.refresh()
    before = e.files["sub/a.md"].vectors
    with monkeypatch.context() as m:
        refused_walk(m, "sub")
        (corpus / "b.md").unlink()
        e.refresh()
    assert sorted(e.files) == ["sub/a.md"]  # what is really gone still goes
    assert e.files["sub/a.md"].vectors is before
    e.refresh()
    assert sorted(e.files) == ["sub/a.md"]
    (corpus / "sub" / "a.md").unlink()
    e.refresh()
    assert sorted(e.files) == []  # readable again: its deletions count


def test_a_folder_that_went_away_keeps_its_files_and_is_not_embedded_meanwhile(tmp_path):
    corpus, other = tmp_path / "corpus", tmp_path / "other"
    corpus.mkdir()
    (corpus / "a.md").write_text("Stoatword is here.\n")
    index = tmp_path / "index"
    Engine(corpus, index_dir=index).refresh()  # no embeddings yet
    corpus.rename(other)
    e = Engine(corpus, embedder=HashEmbedder(), index_dir=index)
    assert e.refresh()["embedded"] == 0
    assert sorted(e.files) == ["a.md"]
    assert e.search("stoatword") == []  # nothing to show from it for now
    other.rename(corpus)
    assert e.refresh()["embedded"] == 1
    assert e.search("stoatword")


def test_a_failed_save_leaves_no_temporary_files(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("Stoatword is here.\n")
    index = tmp_path / "index"
    e = Engine(corpus, index_dir=index)
    e.refresh()

    def full(*_, **__):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("isis.engine.index.json.dumps", full)
    with pytest.raises(OSError):
        e._save()
    assert not [p.name for p in index.rglob("*") if ".tmp" in p.name]


def forget_chunker(index):
    """Make a cache look like one written before files were split the current way."""
    for path in index.rglob("index.json"):
        meta = json.loads(path.read_text())
        del meta["chunker"]
        path.write_text(json.dumps(meta))


class RecordingEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.seen = []

    def embed(self, texts):
        self.seen += texts
        return super().embed(texts)


def test_a_cache_split_the_old_way_is_read_again_keeping_embeddings(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "rule.md").write_text("---\nStoatword opens this note.\n---\n\nPikeword closes it.\n")
    (corpus / "plain.md").write_text("Herringword is here.\n")
    index = tmp_path / "index"
    old = "isis.engine.chunker.YAML_LINE"
    with monkeypatch.context() as m:
        m.setattr(old, re.compile(""))  # what the chunker did: anything between two rules was skipped
        Engine(corpus, embedder=HashEmbedder(), index_dir=index, target_words=0).refresh()
    forget_chunker(index)

    no_model = Engine(corpus, index_dir=index, target_words=0)  # can't embed: it waits for a run that can
    no_model.refresh()
    assert not no_model.search("stoatword")
    assert no_model.files["plain.md"].vectors is not None
    (corpus / "new.md").write_text("Troutword is new.\n")
    no_model.refresh()  # and saving meanwhile doesn't mark the cache as done

    embedder = RecordingEmbedder()
    e = Engine(corpus, embedder=embedder, index_dir=index, target_words=0)
    e.refresh()
    assert e.search("stoatword")
    assert sorted(embedder.seen) == ["Stoatword opens this note.\n---", "Troutword is new."]  # the others kept theirs
