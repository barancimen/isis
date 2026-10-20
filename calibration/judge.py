"""Record judgments by position in a pool file printed by one of the pool scripts.

    python calibration/judge.py qrels_search.json pool_search.json "q05: 2 2 1 0 ..." "q06: ..."

Each argument gives one grade per listed passage, in the printed order:
2 = what you were looking for, 1 = partly relevant or useful context, 0 = not relevant.
"""

from __future__ import annotations

import sys

from lib import jload, jsave

if len(sys.argv) < 4:
    sys.exit(__doc__)
qrels_name, pool_name = sys.argv[1], sys.argv[2]
qrels, pools = jload(qrels_name, {}), jload(pool_name, {})
for arg in sys.argv[3:]:
    qid, _, grades = arg.partition(":")
    qid, grades = qid.strip(), [int(g) for g in grades.split()]
    keys = pools[qid]
    if len(grades) != len(keys):
        sys.exit(f"{qid}: {len(grades)} grades for {len(keys)} passages; nothing saved")
    if any(g not in (0, 1, 2) for g in grades):
        sys.exit(f"{qid}: grades must be 0, 1 or 2; nothing saved")
    qrels.setdefault(qid, {}).update(dict(zip(keys, grades)))
jsave(qrels_name, qrels)
print(f"{sum(len(v) for v in qrels.values())} judgments in {qrels_name}")
