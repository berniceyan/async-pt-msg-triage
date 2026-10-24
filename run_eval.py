#!/usr/bin/env python3
"""Evaluation runner. Routes a dataset and writes results/<dataset>.<config>.json.

  python run_eval.py --data data/ext_test.csv      # test
  python run_eval.py --data data/pmr_dev.csv        # the split tuning is allowed on
  python run_eval.py --firm-only                    # drop judgment-sensitive cases
  python run_eval.py --data data/pmr_dev.csv data/ext_dev_add.csv --pmr-fallback ext_dev_add
                                                    # tuning set; dev_add scored on PMR

Every run is scored twice: against the clinician's labels
(data/clinician_labels.csv) on the rows they cover -- the primary reference --
and against PMR's folded levels on every row.

Results are keyed by DATASET AND CONFIG, so two runs that answer different
questions never share a filename.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from src.env import api_key
from src.evaluate import (error_table, load_labels, load_reference, print_report,
                          relabel, save_run, score, score_incumbent)
from src.evaluate import Row
from src.judgment import cached_judge, cached_screen, judge, screen
from src.pipeline import route_judgment_only, route_message_scored

# model:     baseline, judgment only, can't-miss thresholds escalate.
# screen:    urgency screen first; unsure escalates.
# screen_uq: the same calls; unsure goes to the queue instead.
# Degraded mode (model down) is not computed: it is stated by construction.
CONFIGS = {"model": None, "screen": "urgent", "screen_uq": "queue"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/ext_test.csv"])
    ap.add_argument("--pmr-fallback", nargs="*", default=[], metavar="SPLIT",
                    help="rows of these splits with no clinician label are scored on PMR's level")
    ap.add_argument("--config", default=None, choices=list(CONFIGS))
    ap.add_argument("--firm-only", action="store_true")
    ap.add_argument("--no-cache", action="store_true",
                    help="re-sample every judgment, to measure run-to-run variation")
    ap.add_argument("--no-retry", action="store_true",
                    help="single attempt per call; MODEL_FAILURE then measures one call")
    args = ap.parse_args()

    rows = [r for path in args.data for r in load_reference(path)]
    labels = load_labels()
    dataset = "+".join(Path(p).stem for p in args.data)   # ext_test | pmr_dev+ext_dev_add | ...
    written: list[tuple[Path, str | None]] = []

    live = judge if not args.no_retry else (lambda m, **kw: judge(m, retries=0, **kw))
    model_judge = live if args.no_cache else cached_judge(live)
    clin_rows, unsure = relabel(rows, labels)
    fallback = [Row(**{**r.__dict__, "level": None}) for r in rows
                if r.split in args.pmr_fallback and r.id not in labels]
    if fallback:
        print(f"{len(fallback)} rows from {args.pmr_fallback} scored on PMR's level (no clinician label)")
        clin_rows += fallback
    print(f"{len(rows)} rows · {len(clin_rows)} with clinician labels ({unsure} unsure)")

    incumbent = score_incumbent(rows, args.firm_only)
    if incumbent is None:
        print("no incumbent labels in this dataset")
    else:
        print_report("INCUMBENT (provided dispositions)", incumbent)

    names = [args.config] if args.config else list(CONFIGS)
    # the screen is judge() with its own prompt, budget and schema, cached alike
    model_screen = ((lambda m, **kw: screen(m, judge_fn=live, **kw)) if args.no_cache
                    else cached_screen(live))
    if not api_key():
        print("\n!! no API key (OPENROUTER_API_KEY): every judgment must come from the cache;")
        print("   the first miss aborts the run. Copy .env.example to .env.")

    for name in names:
        judge_fn = model_judge
        predicted, telemetry = {}, []
        t0 = time.time()
        for r in rows:
            if CONFIGS[name] is None:
                d, meta = route_judgment_only(r.message, message_id=r.id, judge_fn=judge_fn)
                telemetry.append(meta)
            else:
                d, smeta, meta = route_message_scored(
                    r.message, message_id=r.id, screen_fn=model_screen,
                    judge_fn=judge_fn, unsure_to=CONFIGS[name])
                telemetry += [smeta, meta]
            predicted[r.id] = d

        s = score(rows, predicted, args.firm_only)
        clin = score(clin_rows, predicted) if clin_rows else None
        if clin:
            clin["unsure"] = unsure
            s["clinician"] = clin
        errs = error_table(clin_rows or rows, predicted)
        called = [m for m in telemetry if m.get("model") not in ("not_called", "STUB")]
        tel = {
            "wall_s": round(time.time() - t0, 1),
            "model_calls": len(called),
            "in_tokens": sum(m.get("in_tokens") or 0 for m in called),
            "out_tokens": sum(m.get("out_tokens") or 0 for m in called),
            "reasoning_tokens": sum(m.get("reasoning_tokens") or 0 for m in called),
            # what this run's judgments cost when they were fetched, cached or not
            "cost_usd": round(sum(m.get("cost_usd") or 0 for m in called), 4),
            "latencies": sorted(m["latency_s"] for m in called if m.get("latency_s")),
        }
        if tel["latencies"]:
            L = tel["latencies"]
            tel["p50_s"] = L[len(L) // 2]
            tel["p95_s"] = L[int(len(L) * 0.95) - 1] if len(L) > 1 else L[0]
        tel.pop("latencies", None)
        # why each failed call failed: truncation, unparseable JSON or a schema
        # violation are different fixes, and the decision record does not say
        per_row = len(telemetry) // len(rows)
        tel["failures"] = [{"id": rows[i // per_row].id, "finish_reason": m.get("finish_reason"),
                            "out_tokens": m.get("out_tokens"),
                            "error": (m.get("error") or "unparseable")[:200]}
                           for i, m in enumerate(telemetry) if not m.get("ok")]
        tel["attempts_over_1"] = sum(1 for m in called if (m.get("attempts") or 1) > 1)
        tel["retry_causes"] = sorted({c for m in called for c in m.get("retry_causes", [])})
        tel["dataset"] = args.data
        if CONFIGS[name] is not None:
            from collections import Counter
            tel["screen_verdicts"] = dict(Counter(
                predicted[r.id].screen.verdict if predicted[r.id].screen else "failed" for r in rows))
        tel["run_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        cstats = getattr(model_judge, "stats", None)
        tel["cache"] = ({"hits": cstats["hits"], "misses": cstats["misses"],
                         "served_drift": cstats["served_drift"]} if cstats else "disabled")

        if clin:
            print_report(f"{name} vs CLINICIAN labels", clin)
        print_report(f"{name} vs PMR", s)
        print(f"  {'model calls':22} {tel['model_calls']}   "
              f"tokens in/out {tel['in_tokens']}/{tel['out_tokens']}   ${tel['cost_usd']}")
        if errs:
            from collections import Counter
            print(f"  {'error codes':22} {dict(Counter(e['code'] for e in errs))}")
        out = Path(f"results/{dataset}.{name}.json")
        prior = None
        if out.exists():
            try:
                prior = json.loads(out.read_text()).get("telemetry", {}).get("run_at")
            except (OSError, ValueError):
                prior = None
        save_run(out, name, s, errs, tel)
        written.append((out, prior))

    cstats = getattr(model_judge, "stats", None)
    if cstats and cstats["hits"]:
        print(f"\n:: {cstats['hits']} judgments served from cache, {cstats['misses']} live.")
    if cstats and cstats["served_drift"]:
        print(f"!! the served model changed: {', '.join(cstats['served_drift'])}")
        print("   cached and live judgments in this run came from different weights.")
    print(f"\nwrote {len(written)} file(s):")
    for out, prior in written:
        print(f"  {out}" + (f"   (replaced a run from {prior})" if prior else ""))

    return 0


if __name__ == "__main__":
    sys.exit(main())
