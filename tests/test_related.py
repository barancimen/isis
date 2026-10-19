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


def test_terms_follow_the_task_and_respect_pins_and_bans(engine):
    task = "Whale whale whale harpoon ocean."
    chips = {c.term: c for c in engine.related(task).terms}
    assert chips["whale"].weight == 1.0

    prefs = TaskPrefs(pins=["granit"], bans=["whale"])
    chips = {c.term: c for c in engine.related(task, prefs=prefs).terms}
    assert chips["whale"].banned and chips["whale"].weight == 0
    assert chips["granit"].pinned and chips["granit"].weight == 1.0


def test_dismissed_passages_stay_away(engine):
    task = "# Sea\nThe whale and the harpoon, the ship and its captain.\n"
    first = items(engine.related(task), "core")[0]
    prefs = TaskPrefs(dismissed=[{"path": first.path, "text": first.text}])
    again = engine.related(task, prefs=prefs)
    assert (first.path, first.start_line) not in {(h.path, h.start_line) for h in items(again)}


def test_bridges_link_two_themes(engine):
    task = f"# Sea\n{' '.join(SEA_TASK)}.\n\n# Birds\n{' '.join(BIRD_TASK)}.\n"
    result = engine.related(task)
    assert [t.label for t in result.themes] == ["Sea", "Birds"]
    bridges = items(result, "bridges")
    assert bridges and all(set(h.why["themes"]) == {"Sea", "Birds"} for h in bridges)
    assert any(h.path.startswith("voyage") for h in bridges)


def test_single_theme_tasks_explain_missing_lanes(engine):
    lanes = {l.name: l for l in engine.related("Whale harpoon ship captain sail.").lanes}
    assert "two themes" in lanes["bridges"].note and not lanes["bridges"].items


def test_cursor_shifts_results_toward_the_nearby_paragraph(engine):
    task = "Whale harpoon ship captain sail.\n\nGranite fossil volcano basalt lava.\n"
    share = lambda line: np.mean([h.path.startswith("rocks") for h in items(engine.related(task, cursor_line=line), "core")[:8]])
    assert share(2) > share(0)


def test_slot_allocation_adds_up_and_learns():
    config = RelatedConfig()
    slots = allocate_slots(list(LANES), TaskPrefs(), config, seed=3)
    assert sum(slots.values()) == config.total_slots and min(slots.values()) >= config.min_slots
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
