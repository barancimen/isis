"""Related calibration: pool suggestions for judging, score settings, study the signals, and
simulate dismissing and avoiding.

    python calibration/related.py build                      # data/tasks.json -> data/task_docs.json
    python calibration/related.py pool  [--grid z_strong=2.5,3,3.5 ...]
    python calibration/related.py eval  [--set ...] [--grid ...] [--top 20]
    python calibration/related.py signals                    # relevance by signal value, for the filter
    python calibration/related.py feedback [--pool]          # dismiss / avoid simulation

pool      runs every task document in data/task_docs.json with each setting (and the defaults),
          prints the suggestions nobody has judged, numbered, and saves their order in
          data/pool_related.json for judge.py (qrels_related.json).
eval      scores each setting over all tasks: utility per task (relevant +1, partly relevant +0.5,
          irrelevant -1, summed over the suggestions shown), precision, suggestions shown,
          relevant suggestions that keywords alone would not have found ("nonobv": keyword rank
          over 30 or none), judged coverage, precision per lane, and per-task wins and losses
          against the defaults.
signals   tabulates the judged passages by the signals the filter uses (document z-score,
          best-paragraph z-score, keyword rank): where relevance actually starts.
feedback  dismisses (then avoids) the first three irrelevant suggestions of every task and
          re-runs, for several values of negative_weight (then avoid_z). --pool prints and saves
          the unjudged suggestions this produces (data/pool_feedback.json) instead.
"""

from __future__ import annotations

import argparse
import statistics

import numpy as np

from lib import (UTILITY, RelatedConfig, jload, jsave, key, label, load_engine, override_args, overrides,
                 related_config)


def suggestions(engine, doc, over, prefs=None):
    result = engine.related(doc, prefs=prefs, config=related_config(over))
    return [(lane.name, h) for lane in result.lanes for h in lane.items]


def build(engine, args):
    by_key = {key(p.path, p.start_line): p for p in engine.snapshot.passages}
    docs = {}
    for task in jload("tasks.json", []):
        blocks = []
        for part in task["parts"]:
            if "heading" in part:
                blocks.append(f"# {part['heading']}")
            elif "note" in part:
                blocks.append(part["note"])
            else:
                blocks.append(by_key[part["excerpt"]].text)
        docs[task["id"]] = "\n\n".join(blocks) + "\n"
    jsave("task_docs.json", docs)
    print(f"{len(docs)} task documents written to data/task_docs.json")


def print_pool(found, qrels, name):
    intents = {t["id"]: t["intent"] for t in jload("tasks.json", [])}
    pools = {}
    for tid, hits in found.items():
        todo = [k for k in hits if k not in qrels.get(tid, {})]
        pools[tid] = todo
        if todo:
            print(f"\n=== {tid}: {intents.get(tid, '')}  ({len(todo)} to judge)")
            for i, k in enumerate(todo):
                print(f"[{i}] {k} | {hits[k].heading[:50]}\n{' '.join(hits[k].text.split())}")
    jsave(name, pools)
    print(f"\n{sum(map(len, pools.values()))} passages to judge; record grades with judge.py qrels_related.json {name}")


def pool(engine, args, docs, qrels):
    settings = [{}] + [o for o in overrides(args, RelatedConfig) if o]
    found = {}
    for tid, doc in docs.items():
        for over in settings:
            for _, h in suggestions(engine, doc, over):
                found.setdefault(tid, {}).setdefault(key(h.path, h.start_line), h)
    print_pool(found, qrels, "pool_related.json")


def score(engine, docs, qrels, over):
    total = dict(n=0, util=0.0, rel=0, junk=0, unjudged=0, nonobv=0)
    lanes, per_task = {}, {}
    for tid, doc in docs.items():
        q = qrels.get(tid, {})
        util = 0.0
        for lane, h in suggestions(engine, doc, over):
            stats = lanes.setdefault(lane, [0, 0, 0])  # shown, relevant, judged
            g = q.get(key(h.path, h.start_line))
            total["n"] += 1
            stats[0] += 1
            if g is None:
                total["unjudged"] += 1
                continue
            stats[2] += 1
            util += UTILITY[g]
            if g >= 1:
                total["rel"] += 1
                stats[1] += 1
                rank = h.why.get("lexical_rank")
                total["nonobv"] += rank is None or rank > 30
            else:
                total["junk"] += 1
        per_task[tid] = util
        total["util"] += util
    judged = total["n"] - total["unjudged"]
    return dict(util=total["util"] / len(docs), prec=total["rel"] / max(judged, 1), n=total["n"] / len(docs),
                junk=total["junk"], nonobv=total["nonobv"], cover=judged / max(total["n"], 1),
                lanes={k: (v[0], v[1] / max(v[2], 1)) for k, v in lanes.items()}, per_task=per_task)


def show(name, m, base=None):
    lanes = " ".join(f"{k}:{n}/{p:.2f}" for k, (n, p) in m["lanes"].items())
    versus = ""
    if base:
        wins = sum(m["per_task"][t] > base["per_task"][t] + 1e-9 for t in m["per_task"])
        losses = sum(m["per_task"][t] < base["per_task"][t] - 1e-9 for t in m["per_task"])
        versus = f" | vs defaults +{wins} -{losses}"
    print(f"  {name:60s} util={m['util']:5.2f} prec={m['prec']:.3f} shown={m['n']:4.1f} junk={m['junk']:3d} "
          f"nonobv={m['nonobv']:3d} judged={m['cover']:.2f} | {lanes}{versus}")


def evaluate(engine, args, docs, qrels):
    base = score(engine, docs, qrels, {})
    show("(defaults)", base)
    rows = [(over, score(engine, docs, qrels, over)) for over in overrides(args, RelatedConfig) if over]
    for over, m in sorted(rows, key=lambda r: -r[1]["util"])[: args.top]:
        show(label(over), m, base)


def signals(engine, args, docs, qrels):
    from isis.engine.related import (TaskPrefs, assign_themes, find_excerpts, ranks, task_terms, task_units,
                                     weight_units, zscores)

    cfg = RelatedConfig()
    snap = engine.snapshot
    index = {key(p.path, p.start_line): i for i, p in enumerate(snap.passages)}
    rows = []
    for tid, doc in docs.items():
        units = task_units(doc)
        vecs = engine.embedder.embed([u.heading + "\n" + u.text if u.heading else u.text for u in units], cache=True)
        cos, z = zscores(snap, vecs)
        find_excerpts(snap, units, cos)
        weight_units(units, None, cfg)
        assign_themes(units, vecs, cfg)
        sizes = np.array([len(u.text.split()) * u.weight for u in units], np.float32)
        document = (sizes[:, None] * vecs).sum(0)
        _, z_doc = zscores(snap, (document / np.linalg.norm(document))[None, :])
        chips = task_terms(snap, units, TaskPrefs(), cfg)
        query = [(snap.term_id(c.term), c.weight) for c in chips if not c.ignored]
        lex = snap.bm25_for(cfg.bm25_k1, cfg.bm25_b).scores([i for i, _ in query], [w for _, w in query])
        lex_rank = ranks(lex, lex > 0)
        collected = {r for u in units for r in u.source_rows}
        z_best = z.max(1)
        for k, g in qrels.get(tid, {}).items():
            i = index.get(k)
            if i is not None and i not in collected:
                rows.append((g, float(z_doc[i, 0]), float(z_best[i]), int(lex_rank[i])))

    def table(title, value, edges):
        print(f"\n{title}")
        for lo, hi in zip(edges, edges[1:]):
            grades = [r[0] for r in rows if lo <= value(r) < hi]
            if grades:
                print(f"  [{lo:>6}, {hi:>6})  n={len(grades):3d}  relevant={np.mean([g >= 1 for g in grades]):.2f}"
                      f"  fully={np.mean([g == 2 for g in grades]):.2f}  irrelevant={sum(g == 0 for g in grades)}")

    inf = float("inf")
    table("document z-score (the 'meaning' signal of the filter)", lambda r: r[1], [-inf, 1, 1.5, 2, 2.5, 3, 3.5, 4, inf])
    table("best single-paragraph z-score", lambda r: r[2], [-inf, 1.5, 2, 2.5, 3, 3.5, 4, 5, inf])
    table("keyword rank (0-based)", lambda r: r[3], [0, 10, 30, 60, 150, inf])
    print("\ndocument z-score, with and without a top-30 keyword rank")
    for lo, hi in [(-inf, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, inf)]:
        for name, test in (("top-30 keyword", lambda r: r[3] < 30), ("no keyword support", lambda r: r[3] >= 30)):
            grades = [r[0] for r in rows if lo <= r[1] < hi and test(r)]
            if grades:
                print(f"  z [{lo}, {hi})  {name:18s} n={len(grades):3d}  relevant={np.mean([g >= 1 for g in grades]):.2f}")


def feedback(engine, args, docs, qrels):
    from isis.engine.related import TaskPrefs

    found = {}
    for mode, param, values in (("dismiss", "negative_weight", [0.0, 0.25, 0.5, 1.0]),
                                ("avoid", "avoid_z", [2.5, 3.0, 3.5, 4.0])):
        if not args.pool:
            print(f"\n{mode} the first 3 irrelevant suggestions of each task, then re-run ({param})")
        for value in values:
            over = {param: value}
            util = relevant = junk = unjudged = lost = before_total = 0
            for tid, doc in docs.items():
                q = qrels.get(tid, {})
                first = suggestions(engine, doc, over)
                prefs = TaskPrefs()
                for lane, h in [(lane, h) for lane, h in first if q.get(key(h.path, h.start_line)) == 0][:3]:
                    prefs.dismiss(lane, h.path, h.text, avoid=mode == "avoid")
                after = suggestions(engine, doc, over, prefs)
                shown = {key(h.path, h.start_line) for _, h in after}
                before = [key(h.path, h.start_line) for _, h in first if q.get(key(h.path, h.start_line), 0) >= 1]
                before_total += len(before)
                lost += sum(k not in shown for k in before)
                for _, h in after:
                    g = q.get(key(h.path, h.start_line))
                    if g is None:
                        unjudged += 1
                        found.setdefault(tid, {}).setdefault(key(h.path, h.start_line), h)
                    else:
                        util += UTILITY[g]
                        relevant += g >= 1
                        junk += g == 0
            if not args.pool:
                print(f"  {param}={value}: util={util / len(docs):5.2f} relevant={relevant} irrelevant={junk} "
                      f"unjudged={unjudged} | relevant suggestions lost: {lost}/{before_total}")
    if args.pool:
        print_pool(found, qrels, "pool_feedback.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["build", "pool", "eval", "signals", "feedback"])
    parser.add_argument("--top", type=int, default=20, help="eval: settings to print")
    parser.add_argument("--pool", action="store_true", help="feedback: print the unjudged suggestions instead")
    override_args(parser)
    args = parser.parse_args()
    engine = load_engine()
    if args.action == "build":
        return build(engine, args)
    docs, qrels = jload("task_docs.json", {}), jload("qrels_related.json", {})
    {"pool": pool, "eval": evaluate, "signals": signals, "feedback": feedback}[args.action](engine, args, docs, qrels)


if __name__ == "__main__":
    main()
