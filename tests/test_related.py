import math
import random

import numpy as np
import pytest

from isis.engine.index import Engine
from isis.engine.related import LANES, RelatedConfig, TaskPrefs, allocate_slots, task_units
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


def test_terms_follow_the_task_the_steer_line_and_ignored_words(engine):
    task = "Whale whale whale harpoon ocean."
    chips = {c.term: c for c in engine.related(task).terms}
    assert chips["whale"].weight == 1.0 and chips["harpoon"].weight < 1.0

    prefs = TaskPrefs(steer="granite harpoon -ocean", ignored=["whale"])
    chips = {c.term: c for c in engine.related(task, prefs=prefs).terms}
    assert chips["whale"].ignored and chips["whale"].weight == 0
    assert chips["granit"].boosted and chips["granit"].label == "granite" and chips["granit"].weight == 1.0
    assert chips["harpoon"].boosted and chips["harpoon"].weight == 1.0  # boosted to full weight
    assert "ocean" not in chips  # excluded words are not searched for


def test_old_preferences_migrate_to_the_steer_line():
    prefs = TaskPrefs.from_dict({"pins": ["harpoon"], "bans": ["whale"], "excluded": ["squid", "/Ahab/"]})
    assert prefs.steer == "harpoon -squid -/Ahab/" and prefs.ignored == ["whale"]


def test_chip_shortcuts_edit_the_steer_line():
    prefs = TaskPrefs(steer="+whales /beak/", ignored=["squid"])
    prefs.boost("whale")  # already required: unchanged
    prefs.boost("squid")
    assert prefs.steer == "+whales /beak/ squid" and prefs.ignored == []
    prefs.exclude("whaling")
    assert prefs.steer == "/beak/ squid -whaling"
    prefs.ignore("squid")
    assert prefs.steer == "/beak/ -whaling" and prefs.ignored == ["squid"]


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
    assert "Sea\nA new note about lances." in embedded and reused()  # collected passages keep their vectors

    prefs = TaskPrefs()
    prefs.dismiss("core", shown[0].path, shown[0].text, shown[0].start_line)
    prefs.dismiss("core", shown[1].path, shown[1].text, shown[1].start_line, avoid=True)
    embedded.clear()
    keys = {(h.path, h.start_line) for h in items(engine.related(task, prefs=prefs))}
    assert not {shown[0].text, shown[1].text} & set(embedded) and (shown[1].path, shown[1].start_line) not in keys

    prefs.dismiss("core", shown[2].path, shown[2].text + " (edited since)", shown[2].start_line, avoid=True)
    engine.related(task, prefs=prefs)
    assert shown[2].text + " (edited since)" in embedded  # no longer the corpus passage: embedded


def test_long_tasks_get_a_few_themes_with_readable_labels(engine):
    rng = random.Random(7)
    task = "\n\n".join(paragraph(rng, rng.choice(["sea", "birds"])) for _ in range(40)) + "\n"
    result = engine.related(task)
    assert 2 <= len(result.themes) <= 2 * math.ceil(math.sqrt(40))


def test_theme_labels_prefer_distinctive_words(engine):
    from isis.engine.related import Unit, _cluster_label

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
    assert "two themes" in lanes["bridges"].note and not lanes["bridges"].items


def test_the_whole_document_is_the_query(engine):
    task = f"{' '.join(SEA_TASK)}.\n\n{' '.join(BIRD_TASK)}.\n"
    core = items(engine.related(task), "core")
    assert core[0].path.startswith("voyage")  # passages about both topics fit the whole document best
    keys = lambda r: [(h.path, h.start_line) for h in items(r)]
    assert keys(engine.related(task, cursor_line=0)) == keys(engine.related(task, cursor_line=2))


def test_paragraph_mode_can_follow_the_cursor(engine):
    task = "Whale harpoon ship captain sail.\n\nGranite fossil volcano basalt lava.\n"
    config = RelatedConfig(query="paragraphs", cursor_boost=3.0)
    share = lambda line: np.mean(
        [h.path.startswith("rocks") for h in items(engine.related(task, cursor_line=line, config=config), "core")[:8]]
    )
    assert share(2) > share(0)


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
    assert lanes["core"].items and "Needs embeddings" in lanes["hidden"].note


def test_task_units_are_paragraphs_with_lines():
    units = task_units("# A\none\n\ntwo\n# B\nthree")
    assert [(u.heading, u.start_line, u.text) for u in units] == [("A", 1, "one"), ("A", 3, "two"), ("B", 5, "three")]


def test_holdout_recovers_hidden_excerpts(engine):
    from isis.engine.evaluate import compare, holdout

    rows = engine.snapshot.file_rows
    excerpts = [engine.snapshot.passages[rows[f"sea-{i}.md"][0]].text for i in range(6)]
    task = "# Sea\n\n" + "\n\n".join(excerpts) + "\n"
    result = holdout(engine, task, fraction=0.5, seed=1)
    assert result.hidden == 3 and 0 <= result.found <= 3 and result.recall == result.found / 3
    table = compare(engine, [task], {"default": RelatedConfig(), "narrow": RelatedConfig(total_slots=8)}, seeds=[0])
    assert [r["config"] for r in table] == ["default", "narrow"] and all(0 <= r["recall"] <= 1 for r in table)


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

    core_birds = lambda weight: sum(h.path.startswith("birds") for h in items(
        engine.related(task, prefs=prefs, config=RelatedConfig(query="paragraphs", like_weight=weight)), "core"))
    assert core_birds(1.0) > core_birds(0.0)  # with paragraph queries too (same feedback, so the same slots)

    prefs.unlike(prefs.summary()["liked"][0]["id"])
    assert prefs.liked == [] and prefs.feedback == {"core": [0, 0]}


def test_dismissing_a_passage_takes_back_more_like_this():
    prefs = TaskPrefs()
    prefs.like("core", "a.md", "Some text", 1, 2)
    prefs.dismiss("core", "a.md", " some  TEXT ")
    assert prefs.liked == []


def test_a_marked_passage_does_not_count_as_redundant(engine):
    from isis.engine.related import mmr_select

    rows = {i: 1.0 - i / 100 for i in range(*engine.snapshot.file_rows["sea-0.md"])}
    first = min(rows)
    config = RelatedConfig(max_per_file=99)
    without = mmr_select(dict(rows), 3, 0.5, engine.snapshot, [], set(), config)
    wanted = mmr_select(dict(rows), 3, 0.5, engine.snapshot, [], set(), config, None, {first})
    assert without[0][0] == wanted[0][0] == first
    assert wanted[1][1] >= without[1][1]  # the next ones are not pushed away from it


def test_headings_do_not_decide_the_themes(engine):
    from isis.engine.related import assign_themes, task_units

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
    assert engine.related("# Only a title\nJust one short note about a whale.\n").themes[0].label == "Everything"


def test_heading_words_count_as_plain_text(engine):
    weights = lambda text: {c.term: c.weight for c in engine.related(text).terms}
    as_heading = weights("# Harpoon\nThe whale and the ship.\n\n# Harpoon\nThe captain and the sail.\n\n# Harpoon\n")
    as_text = weights("Harpoon\n\nThe whale and the ship.\n\nHarpoon\n\nThe captain and the sail.\n\nHarpoon\n")
    assert as_heading == as_text  # every one counts, even with nothing under it


def test_excerpt_matcher_agrees_with_comparing_every_excerpt():
    from isis.engine.related import ExcerptMatcher

    rng = random.Random(9)
    words = "the whale sea ship sail white captain ocean deep harpoon".split()
    text = lambda n: " ".join(rng.choice(words) for _ in range(n))
    excerpts = [e for e in (text(rng.randint(4, 12)) for _ in range(40)) if len(e) >= 20]
    excerpts += [e[2:-2] for e in excerpts[:10]]  # selections that start and end mid-word
    excerpts += ["harpoonharpoon whalewhalewhale", "onewordwithoutanyspacesatall"]  # no word inside them
    match = ExcerptMatcher(excerpts)
    hays = [text(rng.randint(1, 30)) for _ in range(800)] + excerpts + [e[3:15] for e in excerpts] + ["x " + e[1:] + " y" for e in excerpts] + ["ab" + e + "cd" for e in excerpts]
    answers = [any(e in hay or hay in e for e in excerpts) for hay in hays]
    assert [match(hay) for hay in hays] == answers
    assert any(answers) and not all(answers)
    assert not ExcerptMatcher([])("anything")


def test_a_refresh_only_rescores_the_theme_that_changed(engine, monkeypatch):
    import isis.engine.related as R

    sea = " ".join(SEA_TASK) + "."
    birds = " ".join(BIRD_TASK) + "."
    task = f"{sea}\n\n{birds}\n\n{birds} Nest.\n"
    engine.related(task)  # the themes are now remembered
    scored, searched = [], []
    real_scores, real_bm25 = R.zscores, engine.snapshot.bm25.scores
    monkeypatch.setattr(R, "zscores", lambda snap, vecs: scored.append(len(vecs)) or real_scores(snap, vecs))
    monkeypatch.setattr(engine.snapshot.bm25, "scores", lambda *a: searched.append(1) or real_bm25(*a))
    engine.related(task)
    assert scored == [1]  # the document vector, and no paragraph
    assert not searched  # no paragraph's excerpt check either
    scored.clear()
    engine.related(task + "\nGranite fossil volcano basalt lava.\n")
    assert sum(scored) <= 3  # the document, and the new paragraph's own theme
    theme_counts = [len(k) for k in engine.snapshot.theme_scores]
    assert len(theme_counts) >= 2


def test_large_corpora_estimate_typical_similarity_from_a_fixed_sample(engine, monkeypatch):
    import isis.engine.index as I
    from isis.engine.related import zscores

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
    prefs = TaskPrefs(ignored=["whale", "harpoon"])
    prefs.set_steer(line)
    assert prefs.ignored == ["harpoon"]
