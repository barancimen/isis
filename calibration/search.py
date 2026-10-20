"""Search calibration: pool results for judging, and score settings by nDCG@10.

    python calibration/search.py pool  [--mode hybrid] [--grid rrf_k=10,30,60] [--augment]
    python calibration/search.py eval  [--mode lexical|hybrid] [--set b=0.35] [--grid k1=0.9,1.2,1.5] [--top 15]

pool   runs every query in data/queries.json with each setting (plus the current defaults in both
       modes), prints the top results nobody has judged yet, numbered, and saves their order in
       data/pool_search.json for judge.py. --augment also runs the hand-written helper queries in
       data/augment.json, to find relevant passages that no setting ranks well (paraphrases).
eval   scores each setting: mean nDCG@10 overall and per query kind, how much of the top 10 is
       judged (unjudged results count as irrelevant, so keep this near 1.00), and per-query wins
       and losses against the current defaults.
"""

from __future__ import annotations

import argparse
import math
import statistics

from lib import jload, jsave, key, label, load_engine, overrides, override_args, search_config, SearchConfig


def ndcg(keys, rel, k=10):
    gains = [2 ** rel.get(x, 0) - 1 for x in keys[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((2 ** g - 1 for g in rel.values()), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def ranked(engine, query, mode, over, limit=10):
    return engine.search(query, limit=limit, mode=mode, config=search_config(over))


def pool(engine, args, queries, qrels):
    settings = [("lexical", {}), ("hybrid", {})] + [(args.mode, o) for o in overrides(args, SearchConfig) if o]
    helpers = jload("augment.json", {}) if args.augment else {}
    pools = {}
    for q in queries:
        found = {}
        for mode, over in settings:
            for h in ranked(engine, q["q"], mode, over, args.depth):
                found.setdefault(key(h.path, h.start_line), h)
        for helper in helpers.get(q["id"], []):
            for mode in ("lexical", "hybrid"):
                for h in ranked(engine, helper, mode, {}, 8):
                    found.setdefault(key(h.path, h.start_line), h)
        todo = [k for k in found if k not in qrels.get(q["id"], {})]
        pools[q["id"]] = todo
        if todo:
            print(f"\n=== {q['id']} [{q['kind']}] {q['q']!r}  ({len(todo)} to judge)")
            for i, k in enumerate(todo):
                print(f"[{i}] {k} | {found[k].heading[:50]}\n{' '.join(found[k].text.split())}")
    jsave("pool_search.json", pools)
    print(f"\n{sum(map(len, pools.values()))} passages to judge; record grades with judge.py qrels_search.json pool_search.json")


def evaluate(engine, args, queries, qrels):
    def run(over):
        per = {}
        for q in queries:
            keys = [key(h.path, h.start_line) for h in ranked(engine, q["q"], args.mode, over)]
            per[q["id"]] = (ndcg(keys, qrels.get(q["id"], {})), sum(k in qrels.get(q["id"], {}) for k in keys) / max(len(keys), 1))
        return per

    base = run({})
    rows = []
    for over in overrides(args, SearchConfig):
        per = run(over)
        kinds = {}
        for q in queries:
            kinds.setdefault(q["kind"], []).append(per[q["id"]][0])
        wins = sum(per[i][0] > base[i][0] + 1e-9 for i in per)
        losses = sum(per[i][0] < base[i][0] - 1e-9 for i in per)
        rows.append((statistics.mean(v[0] for v in per.values()), over, kinds, statistics.mean(v[1] for v in per.values()), wins, losses))
    print(f"{args.mode}: defaults {statistics.mean(v[0] for v in base.values()):.3f}")
    for score, over, kinds, judged, wins, losses in sorted(rows, key=lambda r: -r[0])[: args.top]:
        by_kind = " ".join(f"{k}={statistics.mean(v):.3f}" for k, v in kinds.items())
        print(f"  {label(over):50s} ndcg@10={score:.3f}  {by_kind}  judged={judged:.2f}  vs defaults +{wins} -{losses}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["pool", "eval"])
    parser.add_argument("--mode", choices=["lexical", "hybrid"], default="hybrid")
    parser.add_argument("--depth", type=int, default=10, help="pool: results per query and setting")
    parser.add_argument("--augment", action="store_true", help="pool: also run the helper queries")
    parser.add_argument("--top", type=int, default=15, help="eval: settings to print")
    override_args(parser)
    args = parser.parse_args()
    engine = load_engine()
    queries, qrels = jload("queries.json", []), jload("qrels_search.json", {})
    (pool if args.action == "pool" else evaluate)(engine, args, queries, qrels)


if __name__ == "__main__":
    main()
