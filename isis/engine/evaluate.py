"""Measure the related pipeline by hiding part of what a task collected and checking what comes
back.

`holdout` removes a random share of a task's excerpts, runs `related` on the rest, and reports
how many of the hidden excerpts' source passages were suggested (recall), in which lane, and how
findable they were by keywords alone. `compare` runs that over several tasks, seeds and configs.
`make_task` builds a test task from the corpus when you don't have real ones yet.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .related import RelatedConfig, TaskPrefs, find_excerpts, task_units


@dataclass
class HoldoutResult:
    hidden: int
    found: int
    recall: float
    by_lane: dict[str, int] = field(default_factory=dict)
    lexical_ranks: list[int | None] = field(default_factory=list)  # of the found ones; None: no keyword match


def make_task(engine, query: str, n: int = 8, pool: int = 40, seed: int = 0) -> str:
    """A task made of `n` passages drawn at random from the top `pool` hybrid search hits."""
    hits = engine.search(query, limit=pool, mode="hybrid")
    picked = random.Random(seed).sample(hits, min(n, len(hits)))
    return f"# {query}\n\n" + "\n\n".join(h.text for h in picked) + "\n"


def holdout(
    engine,
    task_text: str,
    *,
    fraction: float = 0.3,
    seed: int = 0,
    config: RelatedConfig | None = None,
    prefs: TaskPrefs | None = None,
) -> HoldoutResult:
    snapshot = engine.snapshot
    units = task_units(task_text)
    find_excerpts(snapshot, units, None)
    excerpts = [u for u in units if u.excerpt]
    if not excerpts:
        raise ValueError("the task has no excerpts from the corpus to hold out")

    k = max(1, round(fraction * len(excerpts)))
    hidden = random.Random(seed).sample(excerpts, k)
    lines = task_text.split("\n")
    drop = {n for u in hidden for n in range(u.start_line, u.end_line + 1)}
    reduced = "\n".join(line for n, line in enumerate(lines) if n not in drop)

    targets = {(snapshot.passages[r].path, snapshot.passages[r].start_line) for u in hidden for r in u.source_rows}
    result = engine.related(reduced, prefs=prefs, config=config, seed=seed)

    by_lane: dict[str, int] = {}
    ranks: list[int | None] = []
    for lane in result.lanes:
        for hit in lane.items:
            if (hit.path, hit.start_line) in targets:
                by_lane[lane.name] = by_lane.get(lane.name, 0) + 1
                ranks.append(hit.why.get("lexical_rank"))
    found = sum(by_lane.values())
    return HoldoutResult(len(targets), found, found / max(len(targets), 1), by_lane, ranks)


def compare(engine, tasks: list[str], configs: dict[str, RelatedConfig], seeds=range(3), fraction: float = 0.3):
    """Mean hold-out recall per config, plus how much of it came from each lane."""
    rows = []
    for name, config in configs.items():
        runs = [holdout(engine, t, fraction=fraction, seed=s, config=config) for t in tasks for s in seeds]
        lanes: dict[str, int] = {}
        for r in runs:
            for lane, count in r.by_lane.items():
                lanes[lane] = lanes.get(lane, 0) + count
        rows.append(
            {
                "config": name,
                "recall": round(float(np.mean([r.recall for r in runs])), 3),
                "found": sum(r.found for r in runs),
                "hidden": sum(r.hidden for r in runs),
                **{f"via_{lane}": n for lane, n in sorted(lanes.items())},
            }
        )
    return rows
