import math
import random

import numpy as np
import pytest

from conftest import write_corpus
import isis.engine.index as I
import isis.engine.related as R
from isis.engine.index import Engine
from isis.engine.related import (EMPTY_NOTE, LANES, RelatedConfig, TaskPrefs, Unit, _cluster_label, allocate_slots,
                                 assign_themes, mmr_select, passage_id, task_units, zscores)
from isis.engine.semantic import HashEmbedder

TOPICS = {
    "sea": "whale harpoon ocean ship sail captain wave deck anchor mast".split(),
    "birds": "finch beak island seed species feather nest song wing flock".split(),
    "rocks": "granite fossil stratum erosion volcano basalt sediment quartz lava cliff".split(),
}
FILLER = "time day long great little old new first last good".split()
SEA_TASK = "whale harpoon ship captain sail".split()
BIRD_TASK = "finch beak species nest song".split()


def paragraph(rng, *topics, n=14):
    words = [w for t in topics for w in rng.sample(TOPICS[t], 6)] + rng.sample(FILLER, 3)
    rng.shuffle(words)
    return " ".join(words[:n]).capitalize() + "."


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    rng = random.Random(1)
    root = tmp_path_factory.mktemp("corpus")
    for topic in TOPICS:
        for f in range(8):
            paras = [paragraph(rng, topic) for _ in range(5)]
            (root / f"{topic}-{f}.md").write_text(f"# {topic} {f}\n\n" + "\n\n".join(paras) + "\n")
    for f in range(3):  # passages where the sea meets the birds, in the words of the bridge test
        paras = [" ".join(rng.sample(SEA_TASK, 4) + rng.sample(BIRD_TASK, 4)) + "." for _ in range(4)]
        (root / f"voyage-{f}.md").write_text("# Voyage\n\n" + "\n\n".join(paras) + "\n")
    e = Engine(root, embedder=HashEmbedder(dim=1024), index_dir=tmp_path_factory.mktemp("index"), target_words=0)
    e.refresh()
    return e


def items(result, lane=None):
    return [h for l in result.lanes if lane in (None, l.name) for h in l.items]


def test_empty_task_explains_itself(engine):
    result = engine.related("")
    assert not items(result) and all("Write or collect" in l.note for l in result.lanes)


def test_results_pass_the_gate_and_lanes_do_not_overlap(engine):
    task = "# Sea\nThe whale and the harpoon, the ship and its captain.\n"
    result = engine.related(task)
    hits = items(result)
    assert hits, result.stats
    config = RelatedConfig()
    assert all(h.why["semantic_z"] >= config.z_weak for h in hits)
    keys = [(h.path, h.start_line) for h in hits]
    assert len(keys) == len(set(keys))
    assert all(h.path.startswith(("sea", "voyage")) for h in items(result, "core"))
    for lane in result.lanes:
        per_file = {}
        for h in lane.items:
            per_file[h.path] = per_file.get(h.path, 0) + 1
        assert max(per_file.values(), default=0) <= config.max_per_file


def test_collected_excerpts_are_recognised_and_not_suggested_again(engine):
    source = engine.snapshot.passages[engine.snapshot.file_rows["sea-0.md"][0]]
    task = f"# Sea\nWhales, ships and harpoons.\n\n{source.text}\n"
    result = engine.related(task)
    assert result.stats["excerpts"] == 1
    assert (source.path, source.start_line) not in {(h.path, h.start_line) for h in items(result)}


def test_a_copy_with_a_word_changed_still_counts_as_collected(engine):
    # Two neighbouring passages as one paragraph, about as long as a real one, with a word changed in each.
    rows = range(*engine.snapshot.file_rows["sea-1.md"])
    source, after = engine.snapshot.passages[rows[1]], engine.snapshot.passages[rows[2]]

    def edited(text):
        words = text.split()
        words[len(words) // 2] = "zebra"
        return " ".join(words)

    task = f"# Sea\nWhales, ships and harpoons.\n\n{edited(source.text)} {edited(after.text)}\n"
    result = engine.related(task)
    assert result.stats["excerpts"] == 1
    assert (source.path, source.start_line) not in {(h.path, h.start_line) for h in items(result)}
    exact_only = engine.related(task, config=RelatedConfig(copy_match=1.0))
    assert exact_only.stats["excerpts"] == 0  # at 1, only exact copies count


def test_copy_score_compares_wording_only():
    from isis.engine.related import copy_score
    text = ("the white whale swam past the harpoon ship while ahab watched the sea from the deck "
            "and the crew waited for the order to lower the boats into the grey water")
    assert copy_score(text, f"before it, {text}, and after it") == 1.0
    assert copy_score(text.replace("swam", "drifted"), text) >= 0.9
    assert copy_score("a whale is a big fish that swims in the deep sea", text) < 0.9
    assert copy_score("the white whale", text) == 0.0  # too short to say it was copied


def test_collected_passages_count_as_much_as_notes_by_default():
    assert RelatedConfig().excerpt_weight == 1.0


def test_terms_follow_the_task_the_steer_line_and_ignored_words(engine):
    task = "Whale whale whale harpoon ocean."
    chips = {c.term: c for c in engine.related(task).terms}
    assert chips["whale"].weight == 1.0 and chips["harpoon"].weight < 1.0

    prefs = TaskPrefs(steer="granite harpoon -ocean", ignored=["whale"])
    chips = {c.term: c for c in engine.related(task, prefs=prefs).terms}
    assert chips["whale"].ignored and chips["whale"].weight == 0
    assert chips["granite"].boosted and chips["granite"].label == "granite" and chips["granite"].weight == 1.0
    assert chips["harpoon"].boosted and chips["harpoon"].weight == 1.0  # boosted to full weight
    assert "ocean" not in chips  # excluded words are not searched for


def test_chip_shortcuts_edit_the_steer_line():
    prefs = TaskPrefs(steer="+whales /beak/", ignored=["squid"])
    prefs.boost("Whales")  # already required: unchanged
    prefs.boost("squid")
    assert prefs.steer == "+whales /beak/ squid" and prefs.ignored == []
    prefs.exclude("whales")
    assert prefs.steer == "/beak/ squid -whales"
    prefs.exclude("whaling")  # another word, not another form of it
    assert prefs.steer == "/beak/ squid -whales -whaling"
    prefs.boost("whaling")
    assert prefs.steer == "/beak/ squid -whales whaling"
    prefs.ignore("squid")
    assert prefs.steer == "/beak/ -whales whaling" and prefs.ignored == ["squid"]


def test_dismissed_passages_stay_away(engine):
    task = "# Sea\nThe whale and the harpoon, the ship and its captain.\n"
    first = items(engine.related(task), "core")[0]
    prefs = TaskPrefs()
    prefs.dismiss("core", first.path, first.text)
    again = engine.related(task, prefs=prefs)
    assert (first.path, first.start_line) not in {(h.path, h.start_line) for h in items(again)}

    prefs.restore(first.path, first.text)
    assert prefs.dismissed == [] and prefs.feedback == {"core": [0, 0]}
    assert items(engine.related(task, prefs=prefs), "core")[0].text == first.text


def test_passages_from_the_corpus_are_not_embedded_again(engine, monkeypatch):
    rows = engine.snapshot.file_rows
    collected = [engine.snapshot.passages[rows[f"sea-{i}.md"][0]] for i in range(3)]
    task = "# Sea\nThe whale and the harpoon, the ship and its captain.\n\n" + "\n\n".join(p.text for p in collected) + "\n"
    shown = items(engine.related(task), "core")
    embedded = []
    real = engine.embedder.embed
    monkeypatch.setattr(engine.embedder, "embed", lambda texts, cache=False: embedded.extend(texts) or real(texts, cache))

    reused = lambda: not any(p.text in t for p in collected for t in embedded)
    result = engine.related(task + "\nA new note about lances.\n")
    assert result.stats["excerpts"] == 3
    assert "A new note about lances." in embedded and reused()  # collected passages keep their vectors

    prefs = TaskPrefs()
    prefs.dismiss("core", shown[0].path, shown[0].text, shown[0].start_line)
    prefs.dismiss("core", shown[1].path, shown[1].text, shown[1].start_line, avoid=True)
    embedded.clear()
    keys = {(h.path, h.start_line) for h in items(engine.related(task, prefs=prefs))}
    assert not {shown[0].text, shown[1].text} & set(embedded) and (shown[1].path, shown[1].start_line) not in keys

    prefs.dismiss("core", shown[2].path, shown[2].text + " (edited since)", shown[2].start_line, avoid=True)
    engine.related(task, prefs=prefs)
    assert shown[2].text + " (edited since)" in embedded  # no longer the corpus passage: embedded


def test_long_tasks_get_a_few_themes(engine):
    rng = random.Random(7)
    task = "\n\n".join(paragraph(rng, rng.choice(["sea", "birds"])) for _ in range(40)) + "\n"
    result = engine.related(task)
    assert 2 <= len(result.themes) <= 2 * math.ceil(math.sqrt(40))


def test_theme_labels_prefer_distinctive_words(engine):
    units = [Unit("", 0, 0, "A long day by the old basalt.")]
    assert _cluster_label(units, engine.snapshot).startswith("basalt")  # rarer in the corpus than "long", "day" or "old"


def test_bridges_link_two_themes(engine):
    task = f"# Sea\n{' '.join(SEA_TASK)}.\n\n# Birds\n{' '.join(BIRD_TASK)}.\n"
    result = engine.related(task)
    labels = [t.label for t in result.themes]
    assert len(labels) == 2
    bridges = items(result, "bridges")
    assert bridges and all(set(h.why["themes"]) == set(labels) for h in bridges)
    assert any(h.path.startswith("voyage") for h in bridges)


def test_single_theme_tasks_explain_missing_lanes(engine):
    lanes = {l.name: l for l in engine.related("Whale harpoon ship captain sail.").lanes}
    assert lanes["bridges"].note and lanes["bridges"].note != EMPTY_NOTE["bridges"] and not lanes["bridges"].items


def test_the_whole_document_is_the_query(engine):
    task = f"{' '.join(SEA_TASK)}.\n\n{' '.join(BIRD_TASK)}.\n"
    core = items(engine.related(task), "core")
    assert core[0].path.startswith("voyage")  # passages about both topics fit the whole document best


def test_slot_allocation_adds_up_and_learns():
    config = RelatedConfig()
    slots = allocate_slots(list(LANES), TaskPrefs(), config, seed=3)
    assert sum(slots.values()) == config.total_slots and min(slots.values()) >= config.min_slots
    for total in (6, 3):  # fewer slots than the per-lane minimums add up to: the total still holds
        assert sum(allocate_slots(list(LANES), TaskPrefs(), RelatedConfig(total_slots=total), seed=3).values()) == total
    assert allocate_slots(list(LANES), TaskPrefs(), config, seed=3) == slots

    keen = TaskPrefs(feedback={"hidden": [20, 0], "core": [0, 20]})
    mean_hidden = np.mean([allocate_slots(list(LANES), keen, config, seed=s)["hidden"] for s in range(50)])
    mean_plain = np.mean([allocate_slots(list(LANES), TaskPrefs(), config, seed=s)["hidden"] for s in range(50)])
    assert mean_hidden > mean_plain + 3


def test_lexical_only_engine_still_suggests(engine, tmp_path):
    lexical = Engine(engine.root, index_dir=tmp_path / "lex", target_words=0)
    lexical.refresh()
    result = lexical.related("Whale harpoon ship captain sail.")
    lanes = {l.name: l for l in result.lanes}
    assert lanes["core"].items and "meaning data" in lanes["hidden"].note


def test_task_units_are_paragraphs_with_lines():
    units = task_units("# A\none\n\ntwo\n# B\nthree")
    assert [(u.heading, u.start_line, u.text) for u in units] == [("A", 1, "one"), ("A", 3, "two"), ("B", 5, "three")]


def test_folders_can_be_left_out_of_the_suggestions(tmp_path):
    rng = random.Random(3)
    a, b = tmp_path / "notes", tmp_path / "books"
    for root, files in ((a, 4), (b, 12)):  # the left-out folder's many matches must not crowd out the other's
        root.mkdir()
        for topic in TOPICS:
            for f in range(files):
                (root / f"{topic}-{f}.md").write_text("\n\n".join(paragraph(rng, topic) for _ in range(4)) + "\n")
    e = Engine([a, b], embedder=HashEmbedder(dim=1024), target_words=0)
    e.refresh()
    task = "The whale and the harpoon, the ship and its captain.\n"
    folders = lambda **k: {h.path.split("/")[0] for h in items(e.related(task, **k))}
    assert folders() == {"notes", "books"}
    assert folders(skip_folders=["books"]) == {"notes"}
    assert not folders(skip_folders=["books", "notes"])


def test_more_like_this_pulls_the_suggestions(engine):
    task = f"{' '.join(SEA_TASK)}.\n\n{' '.join(BIRD_TASK)}.\n"
    birds = lambda result: sum(h.path.startswith("birds") for h in items(result))
    base = engine.related(task)
    pick = engine.snapshot.passages[engine.snapshot.file_rows["birds-0.md"][0]]

    prefs = TaskPrefs()
    prefs.like("core", pick.path, pick.text, pick.start_line, pick.end_line)
    prefs.like("core", pick.path, pick.text, pick.start_line, pick.end_line)  # once is enough
    assert len(prefs.liked) == 1 and prefs.feedback == {"core": [1, 0]}
    result = engine.related(task, prefs=prefs)
    assert birds(result) > birds(base)
    shown = [h for h in items(result) if h.why["liked"]]
    assert [(h.path, h.start_line) for h in shown] == [(pick.path, pick.start_line)]  # on show, and marked
    assert not engine.related(task, prefs=prefs).stats.get("excerpts")  # and is not part of the task

    off = engine.related(task, prefs=prefs, config=RelatedConfig(like_weight=0))
    assert [c.term for c in off.terms] == [c.term for c in engine.related(task).terms]  # 0: no pull

    # With paragraph queries too: the liked passage raises the meaning scores of passages like it.
    from isis.engine import related as R
    snap = engine.snapshot
    units = R.task_units(task)
    vecs = engine.embedder.embed([u.embed_text for u in units])
    themes = R.assign_themes(units, vecs, RelatedConfig(), snap)
    birds_rows = [r for path, (start, end) in snap.file_rows.items() if path.startswith("birds")
                  for r in range(start, end) if r != snap.file_rows[pick.path][0]]
    pull = lambda weight: R.semantic_signals(snap, engine.embedder, units, vecs, themes, prefs.liked, prefs,
                                             RelatedConfig(query="paragraphs", like_weight=weight)).weighted[birds_rows].mean()
    assert pull(1.0) > pull(0.0)

    prefs.unlike(prefs.summary()["liked"][0]["id"])
    assert prefs.liked == [] and prefs.feedback == {"core": [0, 0]}


def test_an_excluded_document_is_never_suggested(engine):
    task = "# Sea\nThe whale and the harpoon, the ship and its captain.\n"
    shown = items(engine.related(task))
    path = shown[0].path
    assert {h.path for h in shown} - {path}, "the corpus should offer another document too"

    prefs = TaskPrefs()
    prefs.exclude_doc(path)
    prefs.exclude_doc(path)
    assert prefs.excluded_docs == [path]
    again = items(engine.related(task, prefs=prefs))
    assert again and path not in {h.path for h in again}

    prefs = TaskPrefs.from_dict(prefs.to_dict())  # it is remembered
    assert path not in {h.path for h in items(engine.related(task, prefs=prefs))}
    prefs.include_doc(path)
    assert path in {h.path for h in items(engine.related(task, prefs=prefs))}


def test_a_passage_liked_in_an_excluded_document_keeps_pulling(engine):
    task = "# Sea\nThe whale and the harpoon, the ship and its captain.\n"
    first = items(engine.related(task))[0]
    prefs = TaskPrefs()
    prefs.like("core", first.path, first.text, first.start_line, first.end_line)
    prefs.exclude_doc(first.path)
    assert [d["path"] for d in prefs.liked] == [first.path]
    assert prefs.feedback == {"core": [1, 0]}

    pulled = items(engine.related(task, prefs=prefs))
    assert pulled and first.path not in {h.path for h in pulled}
    prefs.unlike(passage_id({"path": first.path, "text": first.text}))
    plain = items(engine.related(task, prefs=prefs))
    assert [h.text for h in pulled] != [h.text for h in plain]  # the like still changed the results


def test_a_marked_passage_does_not_count_as_redundant(engine):
    rows = {i: 1.0 - i / 100 for i in range(*engine.snapshot.file_rows["sea-0.md"])}
    first = min(rows)
    config = RelatedConfig(max_per_file=99)
    without = mmr_select(dict(rows), 3, 0.5, engine.snapshot, [], set(), config)
    wanted = mmr_select(dict(rows), 3, 0.5, engine.snapshot, [], set(), config, None, {first})
    assert without[0][0] == wanted[0][0] == first
    assert wanted[1][1] >= without[1][1]  # the next ones are not pushed away from it


def test_headings_do_not_decide_the_themes(engine):
    config = RelatedConfig()
    plain = "Whale harpoon ship captain sail.\n\nWhale ship harpoon sail deck.\n\nFinch beak species nest song.\n"
    headed = "# One\nWhale harpoon ship captain sail.\n\n# Two\nWhale ship harpoon sail deck.\n\n# Three\nFinch beak species nest song.\n"

    def themes(text):
        units = task_units(text)
        vecs = engine.embedder.embed([u.text for u in units])
        found = assign_themes(units, vecs, config, engine.snapshot)
        return [u.theme for u in units], [t.label for t in found]

    assert themes(headed) == themes(plain)  # three headings, the same two groups of paragraphs
    assert "One" not in themes(headed)[1]
    assert engine.related("# Only a title\nJust one short note about a whale.\n").themes[0].label == "All of the task"


def test_heading_words_count_as_plain_text(engine):
    weights = lambda text: {c.term: c.weight for c in engine.related(text).terms}
    as_heading = weights("# Harpoon\nThe whale and the ship.\n\n# Harpoon\nThe captain and the sail.\n\n# Harpoon\n")
    as_text = weights("Harpoon\n\nThe whale and the ship.\n\nHarpoon\n\nThe captain and the sail.\n\nHarpoon\n")
    assert as_heading == as_text  # every one counts, even with nothing under it


def test_a_refresh_only_rescores_the_theme_that_changed(engine, monkeypatch):
    sea = " ".join(SEA_TASK) + "."
    birds = " ".join(BIRD_TASK) + "."
    task = f"{sea}\n\n{birds}\n\n{birds} Nest.\n"
    engine.related(task)  # the themes are now remembered
    scored, searched = [], []
    real_scores, real_closest = R.zscores, R._closest_in_words
    monkeypatch.setattr(R, "zscores", lambda snap, vecs, *a: scored.append(len(vecs)) or real_scores(snap, vecs, *a))
    monkeypatch.setattr(R, "_closest_in_words", lambda *a: searched.append(1) or real_closest(*a))
    engine.related(task)
    assert scored == [1]  # the document vector, and no paragraph
    assert not searched  # no paragraph's excerpt check either
    scored.clear()
    engine.related(task + "\nGranite fossil volcano basalt lava.\n")
    assert sum(scored) <= 3  # the document, and the new paragraph's own theme
    theme_counts = [len(k) for k in engine.snapshot.theme_scores]
    assert len(theme_counts) >= 2


def test_large_corpora_estimate_typical_similarity_from_a_fixed_sample(engine, monkeypatch):
    snap = engine.snapshot
    rows = snap.typical_rows()
    assert len(rows) == int(snap.has_vector.sum())  # small corpus: all of it
    vec = snap.vectors[:1]
    full = zscores(snap, vec)[1]

    monkeypatch.setattr(I, "SAMPLE_ABOVE", 20)
    monkeypatch.setattr(I, "TYPICAL_SAMPLE", 50)
    snap._typical.clear()
    rows = snap.typical_rows()
    assert len(rows) == 50 and list(rows) == sorted(set(rows))
    snap._typical.clear()
    assert list(snap.typical_rows()) == list(rows)  # the same sample each time
    sampled = zscores(snap, vec)[1]
    assert sampled.shape == full.shape and not np.array_equal(sampled, full)
    assert np.corrcoef(sampled[:, 0], full[:, 0])[0, 1] > 0.99
    snap._typical.clear()


@pytest.mark.parametrize("line", ["whales", "+whales", "-whales"])
def test_naming_a_word_on_the_steer_line_stops_ignoring_it(line):
    prefs = TaskPrefs(ignored=["whales", "whale", "harpoon"])
    prefs.set_steer(line)
    assert prefs.ignored == ["whale", "harpoon"]


def test_excerpt_weight_zero_on_a_task_of_excerpts_still_answers(engine):
    text = engine.snapshot.passages[0].text
    result = engine.related(text, config=RelatedConfig(excerpt_weight=0.0))
    assert all(t.weight == t.weight for t in result.terms)  # no NaN


def test_negative_saved_feedback_counts_are_clamped(engine):
    prefs = TaskPrefs.from_dict({"feedback": {"core": [-1, -2]}})
    assert prefs.feedback == {"core": [0, 0]}
    engine.related("# Sea\nwhales and ships", prefs=prefs)


def test_dismissing_a_liked_passage_takes_back_its_add():
    prefs = TaskPrefs()
    prefs.like("core", "a.md", "Some text", 1, 2)
    prefs.dismiss("core", "a.md", " some  TEXT ")  # the same passage, however it is spaced
    assert prefs.liked == [] and prefs.feedback == {"core": [0, 1]}


@pytest.mark.parametrize("sampled", [False, True], ids=["small corpus", "typical rows sampled"])
def test_suggestions_in_some_folders_only_look_at_them(tmp_path, monkeypatch, sampled):
    rng = random.Random(4)
    files = {f"{topic}-{f}.md": "\n\n".join(paragraph(rng, topic) for _ in range(4)) + "\n" for topic in TOPICS for f in range(5)}
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):  # alike, so a pattern's share of passages is the same in one as in both
        root.mkdir()
        for name, text in files.items():
            (root / name).write_text(text)
    e = Engine([a, b], embedder=HashEmbedder(dim=1024), target_words=0)
    e.refresh()
    if sampled:
        monkeypatch.setattr(I, "SAMPLE_ABOVE", 20)
        monkeypatch.setattr(I, "TYPICAL_SAMPLE", 50)
    task = f"{' '.join(SEA_TASK)}.\n\n{' '.join(BIRD_TASK)}.\n"
    prefs = TaskPrefs(steer="/harpoon|beak/")
    config = RelatedConfig(z_strong=1.0, z_weak=-1.0)  # a small corpus, all on topic: let more through
    shown = lambda result: [(l.name, h.path, h.start_line, round(h.score, 5)) for l in result.lanes for h in l.items]

    scoped = shown(e.related(task, prefs=prefs, config=config, skip_folders=["b"]))
    start, end = e.snapshot.folder_rows()["b"]
    for counts, done in e.snapshot._pattern_counts.values():
        assert not done[start:end].any()  # the steer line's pattern was not looked for in b

    real_zscores, real_lexical = R.zscores, R.lexical_scores  # as when everything was scored, then b dropped
    monkeypatch.setattr(R, "zscores", lambda snap, vecs, scope=None: real_zscores(snap, vecs))
    monkeypatch.setattr(R, "lexical_scores", lambda *a: real_lexical(*a[:5]))
    e.snapshot.theme_scores.clear()
    assert scoped and all(path.startswith("a/") for _, path, _, _ in scoped)
    assert shown(e.related(task, prefs=prefs, config=config, skip_folders=["b"])) == scoped


def test_word_rules_match_passages_without_their_headings_and_documents_with_them(tmp_path):
    engine = Engine(write_corpus(tmp_path / "corpus"), embedder=HashEmbedder(), index_dir=tmp_path / "index", target_words=0)
    engine.refresh()
    task = "The whale and the squid in the deep sea, and ambergris.\n\nAhab hunted the whale from his ship.\n"
    found = lambda steer: sorted({h.path for l in engine.related(task, prefs=TaskPrefs(steer=steer),
                                                                 config=RelatedConfig(z_strong=1.0, z_weak=-1.0)).lanes
                                  for h in l.items})
    assert found("") == ["finches.md", "ships/pequod.md", "whales.md"]
    assert found("+/^Whales/") == []  # only the heading starts with it
    assert found("doc:/^Whales/") == ["whales.md"]
    assert found("-doc:/^Whales/") == ["finches.md", "ships/pequod.md"]


def topic_engine(tmp_path, files: dict[str, str], **options) -> Engine:
    engine = Engine(write_corpus(tmp_path / "corpus", files), embedder=HashEmbedder(dim=64), index_dir=tmp_path / "index",
                    target_words=0, **options)
    engine.refresh()
    return engine


def test_word_rules_and_hidden_documents_take_no_places_in_the_rankings(tmp_path):
    """Eighty strong matches from other sites once held every top keyword rank, so with
    site:github.com nothing was left, though ten passages there matched."""
    rng = random.Random(3)
    sea = "whale harpoon ocean ship sail captain wave deck anchor mast".split()
    other = "granite fossil stratum erosion volcano basalt sediment quartz lava cliff finch beak island seed".split()
    files = {f"a-{f:02}.md": f"---\nurl: https://example.org/a{f}\n---\n# A {f}\n\nwhale harpoon ship captain sail {' '.join(rng.sample(sea, 4))} a{f}.\n"
             for f in range(80)}
    files |= {f"g-{f:02}.md": f"---\nurl: https://github.com/x/g{f}\n---\n# G {f}\n\nwhale harpoon ship captain {' '.join(rng.sample(other, 5))} g{f}.\n"
              for f in range(10)}
    files |= {f"rock-{f:03}.md": f"# Rock {f}\n\n{' '.join(rng.sample(other, 8))} filler{f}.\n" for f in range(100)}
    engine = Engine(write_corpus(tmp_path / "corpus", files), index_dir=tmp_path / "index", target_words=0)  # keywords only
    engine.refresh()
    task = "Notes on the whale, the harpoon, the ship, its captain and the sail.\n"
    shown = lambda prefs: sorted({h.path[0] for lane in engine.related(task, prefs=prefs).lanes for h in lane.items})
    assert shown(TaskPrefs()) == ["a"]
    assert shown(TaskPrefs(steer="site:github.com")) == ["g"]
    assert shown(TaskPrefs(steer="-site:example.org")) == ["g"]
    assert shown(TaskPrefs(steer="doc:g3")) == ["g"]
    assert shown(TaskPrefs(steer="-harpoon")) == []
    assert shown(TaskPrefs(excluded_docs=[f"a-{f:02}.md" for f in range(80)])) == ["g"]


def test_one_long_document_does_not_empty_a_group(tmp_path):
    rng = random.Random(3)
    sea = "whale harpoon ocean ship sail captain wave deck anchor mast".split()
    other = "granite fossil stratum erosion volcano basalt sediment quartz lava cliff finch beak island seed".split()
    files = {"book.md": "# Book\n\n" + "\n\n".join(f"whale harpoon ship captain sail {' '.join(rng.sample(sea, 3))} note{i}." for i in range(150)) + "\n"}
    files |= {f"note-{f:02}.md": f"# Note {f}\n\nwhale harpoon ship {' '.join(rng.sample(other, 4))} extra{f}.\n" for f in range(30)}
    files |= {f"rock-{f:03}.md": f"# Rock {f}\n\n{' '.join(rng.sample(other, 8))} filler{f}.\n" for f in range(100)}
    engine = topic_engine(tmp_path, files)
    result = engine.related("Notes on the whale, the harpoon, the ship, its captain and the sail.\n",
                            config=RelatedConfig(z_strong=1.0, z_weak=-1.0, lexical_top=300))
    core = result.lanes[0].items
    assert sum(h.path == "book.md" for h in core) == 2
    assert any(h.path.startswith("note-") for h in core)  # the book's passages alone were the pool, and these never got in


def test_a_task_of_thousands_of_paragraphs_stays_in_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "MAX_CLUSTERED", 40)
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((300, 16)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    clusters = R._clusters(vecs, 0.15)
    assert len(clusters) == 300 and len(set(clusters)) <= 2 * math.ceil(math.sqrt(40))
    some = np.linspace(0, 299, 40).astype(int)
    assert list(clusters[some]) == list(R._clusters(vecs[some], 0.15))  # the ones that were compared keep their clusters

    engine = topic_engine(tmp_path, {f"d{f}.md": f"# D {f}\n\nThe whale and the ship number {f}.\n\nA finch on island {f}.\n" for f in range(20)})
    task = "\n\n".join(f"The whale number {i} and the ship." if i % 2 else f"A finch number {i} on the island." for i in range(90))
    config = RelatedConfig(query="paragraphs", z_strong=0.5, z_weak=-1.0)
    monkeypatch.setattr(R, "THEME_BATCH", 1000)
    whole = engine.related(task, config=config)
    monkeypatch.setattr(R, "THEME_BATCH", 7)  # the paragraphs a batch at a time give the same answer
    engine.snapshot.theme_scores.clear()
    batched = engine.related(task, config=config)
    assert [[(h.path, h.start_line, h.why["matches_task_line"]) for h in lane.items] for lane in batched.lanes] == \
        [[(h.path, h.start_line, h.why["matches_task_line"]) for h in lane.items] for lane in whole.lanes]
    assert any(lane.items for lane in whole.lanes)


def test_mostly_identical_passages_do_not_send_meaning_scores_through_the_roof(tmp_path):
    files = {f"d{i}.md": "# D\n\nSubscribe to the newsletter for more stories.\n" for i in range(8)}
    files |= {"w.md": "# W\n\nThe whale and the harpoon on the ship.\n", "x.md": "# X\n\nA finch with a beak on the island.\n",
              "y.md": "# Y\n\nGranite and basalt in the cliff.\n"}
    engine = topic_engine(tmp_path, files)
    result = engine.related("Notes about the whale and its harpoon.\n\nAlso the captain of the ship and the sail.\n",
                            config=RelatedConfig(query="paragraphs", z_strong=0.0, z_weak=-1.0))
    scores = [h.why["semantic_z"] for lane in result.lanes for h in lane.items if h.why["semantic_z"] is not None]
    assert scores and max(scores) < 100  # they were in the hundreds of thousands


def test_a_passage_without_meaning_data_names_no_closest_paragraph(tmp_path):
    a = write_corpus(tmp_path / "notes", {"sea.md": "# Sea\n\nThe whale and the harpoon and the ship.\n"})
    b = write_corpus(tmp_path / "big", {"sea2.md": "# Sea\n\nA whale, a harpoon and a ship again.\n", ".no-embed": ""})
    engine = Engine([a, b], embedder=HashEmbedder(dim=64), index_dir=tmp_path / "index", target_words=0)
    engine.refresh()
    result = engine.related("Rocks first.\n\nThe whale, the harpoon and the ship.\n", config=RelatedConfig(z_strong=0.0, z_weak=-1.0))
    lines = {h.path: h.why["matches_task_line"] for lane in result.lanes for h in lane.items}
    assert lines["big/sea2.md"] is None and lines["notes/sea.md"] == 2


def test_settings_of_the_wrong_kind_are_ignored_not_fatal():
    from isis.engine.params import configure

    for overrides in ({"candidates": 10**400}, {"query": ["document"]}, {"query": {"a": 1}}, {"mmr_lambda.core": 10**400},
                      {"total_slots": float("inf")}, {"total_slots": "nan"}, {"z_strong": None}):
        config = configure(RelatedConfig, overrides)
        assert config.query == "document" and 4 <= config.total_slots <= 80
