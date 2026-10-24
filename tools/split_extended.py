#!/usr/bin/env python3
"""Carve the tuning, validation and test sets out of PMR's training split.

  python tools/split_extended.py           # writes data/ext_{dev_add,val,test}.csv + data/ext_split.json
  python tools/split_extended.py --dry     # print the counts, write nothing

The judgment-only baseline tuned prompt and thresholds on dev's 16 urgent messages (4 at L3) and its
1-3 day cutoff did not survive test. PMR's training split (data/pmr_extended.csv)
is untouched -- nothing here is trained -- so it supplies three disjoint sets,
sampled per rounded PMR level with a fixed seed. Each is named ext_ for the file
it came from, so it is never confused with PMR's own dev and test splits
(data/pmr_dev.csv, data/pmr_test.csv):

  dev_add  urgent only, pooled with dev for tuning. Scored against PMR's level,
           not hand-labelled: the clinician kept 80/85 PMR-urgent messages
           urgent and moved no PMR-non-urgent message up.
  val      checked after each tuning round and for the threshold pick.
  test     scored once, at the frozen setup.

val and test oversample urgent (L3 most, since that boundary failed) so the
miss rate has enough cases behind it. Per-class rates are read directly;
anything mixing classes is reweighted to the pool's natural mix (urgent ~18%).
Both are clinician-labelled before use. PMR's own test split was used by
the earlier designs and is not reused.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pmr import round_level  # noqa: E402

DATA = Path("data")
SOURCE = DATA / "pmr_extended.csv"
SPLIT_FILE = DATA / "ext_split.json"
SEED = 20261020

# rounded PMR level -> how many to draw, per set. L3 uses all 34 the pool has.
PLAN = {
    "ext_dev_add": {1: 20, 2: 8, 3: 12},
    "ext_val":     {1: 10, 2: 6, 3: 8, 4: 18, 5: 9, 6: 9},
    "ext_test":    {1: 16, 2: 10, 3: 14, 4: 30, 5: 15, 6: 15},
}


def split(rows: list[dict]) -> dict[str, list[dict]]:
    by_level: dict[int, list[dict]] = {}
    for r in sorted(rows, key=lambda r: r["id"]):
        by_level.setdefault(round_level(float(r["level"])), []).append(r)
    rng = random.Random(SEED)
    for pool in by_level.values():
        rng.shuffle(pool)
    out: dict[str, list[dict]] = {name: [] for name in PLAN}
    for name, plan in PLAN.items():
        for level, n in plan.items():
            pool = by_level.get(level, [])
            if len(pool) < n:
                raise SystemExit(f"{name}: wants {n} at L{level}, pool has {len(pool)} left")
            out[name] += [dict(r, split=name) for r in pool[:n]]
            del pool[:n]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    with open(SOURCE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        columns = list(rows[0])
    out = split(rows)
    for name, sel in out.items():
        refs = Counter(r["reference"] for r in sel)
        levels = Counter(round_level(float(r["level"])) for r in sel)
        print(f"{name:12s} n={len(sel):3d}  {dict(sorted(refs.items()))}  "
              f"levels {dict(sorted(levels.items()))}")
    print(f"to label: {len(out['ext_val']) + len(out['ext_test'])} (val + test)")
    if args.dry:
        return 0
    for name, sel in out.items():
        with open(DATA / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
            w.writeheader()
            w.writerows(sel)
    SPLIT_FILE.write_text(json.dumps(
        {"source": str(SOURCE), "seed": SEED, "plan": PLAN,
         "ids": {name: [r["id"] for r in sel] for name, sel in out.items()}},
        indent=1) + "\n")
    print(f"wrote {', '.join(f'data/{n}.csv' for n in out)} and {SPLIT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
