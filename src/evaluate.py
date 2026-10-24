"""Metrics, ablation and the error taxonomy.

Denominators are stated on every metric because that is what makes them
auditable. Counts before percentages, and every percentage carries its Wilson
95% interval: 0/61 critical misses is not "0%", it is "no worse than about 6%",
and the interval is what says so.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .pmr import round_level
from .schema import DEFERRAL_CODES, Disposition, ReasonCode, RoutingDecision

RANK = {Disposition.SELF_CARE: 0, Disposition.ASYNC_PHYSICIAN: 1,
        Disposition.URGENT_ESCALATION: 2}


@dataclass
class Row:
    id: str
    message: str
    given: str                       # an incumbent workflow's label; "" when there is none
    reference: str                   # the reference standard, folded to three destinations
    confidence: str                  # firm | soft
    rationale: str = ""
    level: float | None = None       # the source's own urgency level (PMR: 1-6)
    source: str = ""                 # reddit | synth | adversarial ...
    split: str = ""


def load_reference(path: str | Path) -> list[Row]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(Row(id=r["id"], message=r["message"], given=r.get("given", ""),
                            reference=r["reference"], confidence=r.get("confidence", "firm"),
                            rationale=r.get("rationale", ""),
                            level=float(r["level"]) if r.get("level") else None,
                            source=r.get("source", ""), split=r.get("split", "")))
    return rows


# --------------------------------------------------------------------------
# primary metrics
# --------------------------------------------------------------------------

def wilson(n: int, d: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% score interval for n/d. Sound at 0/d and d/d, where the normal one is not."""
    if not d:
        return None
    p = n / d
    centre = (p + z * z / (2 * d)) / (1 + z * z / d)
    half = z * math.sqrt(p * (1 - p) / d + z * z / (4 * d * d)) / (1 + z * z / d)
    return max(0.0, centre - half), min(1.0, centre + half)


def _frac(n, d):
    ci = wilson(n, d)
    return {"n": n, "d": d, "pct": round(100 * n / d, 1) if d else None,
            "ci": [round(100 * ci[0], 1), round(100 * ci[1], 1)] if ci else None,
            "display": f"{n}/{d}" + (f" ({round(100*n/d)}%, CI {100*ci[0]:.0f}-{100*ci[1]:.0f}%)"
                                     if d else "")}


def score(rows: list[Row], predicted: dict[str, RoutingDecision],
          firm_only: bool = False) -> dict:
    pool = [r for r in rows if not (firm_only and r.confidence == "soft")]

    ref_urgent = [r for r in pool if r.reference == "URGENT_ESCALATION"]
    ref_nonurgent = [r for r in pool if r.reference != "URGENT_ESCALATION"]
    ref_selfcare = [r for r in pool if r.reference == "SELF_CARE"]
    # The messages the reference says need no physician. Not the same cohort as
    # ref_nonurgent: on PMR test 331 non-urgent against 254 non-queue.
    ref_nonasync = [r for r in pool if r.reference != "ASYNC_PHYSICIAN"]

    def pred(r):
        return predicted[r.id].disposition

    crit = [r for r in ref_urgent if RANK[pred(r)] < RANK[Disposition.URGENT_ESCALATION]]
    over = [r for r in ref_nonurgent if pred(r) is Disposition.URGENT_ESCALATION]
    sc = [r for r in ref_selfcare if pred(r) is Disposition.SELF_CARE]
    agree = [r for r in pool if pred(r).value == r.reference]

    # Of the messages that should not have needed a physician, the ones routed
    # correctly without one. Denominated on the reference like critical miss and
    # over-triage, not on the pool: counting every non-queue prediction would
    # score a wrongly cleared ASYNC_PHYSICIAN case as an automation win.
    automated = [r for r in ref_nonasync if pred(r).value == r.reference]
    deferred = [r for r in pool if predicted[r.id].reason in DEFERRAL_CODES]

    return {
        "n": len(pool),
        "agreement": _frac(len(agree), len(pool)),
        "critical_miss": _frac(len(crit), len(ref_urgent)),
        "critical_miss_ids": [r.id for r in crit],
        "over_triage": _frac(len(over), len(ref_nonurgent)),
        "over_triage_ids": [r.id for r in over],
        "self_care_capture": _frac(len(sc), len(ref_selfcare)),
        "automated": _frac(len(automated), len(ref_nonasync)),
        "automated_self_care": _frac(
            len([r for r in automated if r.reference == "SELF_CARE"]), len(ref_nonasync)),
        "automated_urgent": _frac(
            len([r for r in automated if r.reference == "URGENT_ESCALATION"]),
            len(ref_nonasync)),
        # What the column used to count: any prediction that is not the queue,
        # over the whole inbox. Kept as the throughput figure it always was.
        "queue_removed": _frac(
            len([r for r in pool if pred(r) is not Disposition.ASYNC_PHYSICIAN]), len(pool)),
        "wrongly_automated_ids": [
            r.id for r in pool
            if r.reference == "ASYNC_PHYSICIAN" and pred(r) is not Disposition.ASYNC_PHYSICIAN],
        "deferred": _frac(len(deferred), len(pool)),
        "deferral_reasons": dict(Counter(
            predicted[r.id].reason.value for r in deferred)),
        "secondary": _conventional(pool, predicted),
        **_breakdowns(pool, predicted),
    }


def _breakdowns(pool: list[Row], predicted: dict[str, RoutingDecision]) -> dict:
    """Where the misses sit. Empty for a dataset with no source or level columns.

    by_source: Reddit labels are an LLM's reading of an expert reply and Synth
    labels are clinicians' own, so the two are not one reference standard and
    are never only reported pooled. By level: an L1 (emergency department)
    sent to the queue is a worse miss than an L2 (same day), and over-triage
    of an L3 (seen within days) is a milder error than of an L6.
    """
    out: dict = {}
    sources = sorted({r.source for r in pool if r.source})
    if len(sources) > 1:
        out["by_source"] = {}
        for src in sources:
            sub = [r for r in pool if r.source == src]
            s = score(sub, predicted)
            out["by_source"][src] = {k: s[k] for k in (
                "n", "agreement", "critical_miss", "over_triage", "self_care_capture",
                "automated")}
    if any(r.level is not None for r in pool):
        def pred(r):
            return predicted[r.id].disposition
        by_cm, by_ot = {}, {}
        for lvl in range(1, 7):
            rows = [r for r in pool if r.level is not None and round_level(r.level) == lvl]
            if not rows:
                continue
            if rows[0].reference == "URGENT_ESCALATION":
                miss = [r for r in rows if pred(r) is not Disposition.URGENT_ESCALATION]
                by_cm[f"L{lvl}"] = _frac(len(miss), len(rows))
            else:
                over = [r for r in rows if pred(r) is Disposition.URGENT_ESCALATION]
                by_ot[f"L{lvl}"] = _frac(len(over), len(rows))
        out["critical_miss_by_level"] = by_cm
        out["over_triage_by_level"] = by_ot
    return out


def _conventional(pool: list[Row], predicted: dict[str, RoutingDecision]) -> dict:
    """Accuracy / macro P,R,F1. Secondary -- and per-class URGENT recall is the
    critical-miss rate inverted, not independent confirmation."""
    labels = [d.value for d in Disposition]
    per = {}
    for lab in labels:
        tp = sum(1 for r in pool if r.reference == lab and predicted[r.id].disposition.value == lab)
        fp = sum(1 for r in pool if r.reference != lab and predicted[r.id].disposition.value == lab)
        fn = sum(1 for r in pool if r.reference == lab and predicted[r.id].disposition.value != lab)
        p = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
        per[lab] = {"precision": round(p, 3), "recall": round(rec, 3),
                    "f1": round(f1, 3), "support": tp + fn}
    acc = sum(1 for r in pool if predicted[r.id].disposition.value == r.reference) / len(pool)
    return {"accuracy": round(acc, 3),
            "macro_f1": round(sum(v["f1"] for v in per.values()) / len(per), 3),
            "per_class": per}


def score_incumbent(rows: list[Row], firm_only: bool = False) -> dict | None:
    """Row one of the ablation: the provided dispositions through the same scorer.

    None when the dataset carries no incumbent label. PMR-Bench has none, so on
    it the ladder starts at model_only and nothing is compared to a workflow."""
    if not rows or any(not r.given for r in rows):
        return None
    fake = {r.id: RoutingDecision(message_id=r.id,
                                  disposition=Disposition(r.given),
                                  reason=ReasonCode.CLINICAL_JUDGMENT)
            for r in rows}
    out = score(rows, fake, firm_only)
    out["deferred"] = _frac(0, out["n"])
    out["deferral_reasons"] = {}
    return out


# --------------------------------------------------------------------------
# the clinician reference
# --------------------------------------------------------------------------

LABELS_PATH = Path("data/clinician_labels.csv")
LABEL_DESTINATION = {"urgent": "URGENT_ESCALATION", "queue": "ASYNC_PHYSICIAN",
                     "automated": "SELF_CARE"}


def load_labels(path: str | Path = LABELS_PATH) -> dict[str, str]:
    """id -> urgent | queue | automated | unsure. Empty when the file is absent.

    One clinician's three-way call, made against this system's own definitions
    (a queue that can take three days; self-care meaning no clinician reads it).
    PMR's levels were not built for either, so they are reported alongside, not
    instead."""
    if not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r["label"] for r in csv.DictReader(f)}


def relabel(rows: list[Row], labels: dict[str, str]) -> tuple[list[Row], int]:
    """The labelled rows with the clinician's call as the reference, and how many
    "unsure" rows were set aside. The PMR level is dropped: the by-level
    breakdown assumes a level maps to one destination, which no longer holds."""
    out, unsure = [], 0
    for r in rows:
        lab = labels.get(r.id)
        if lab is None:
            continue
        if lab not in LABEL_DESTINATION:
            unsure += 1
            continue
        out.append(Row(**{**r.__dict__, "reference": LABEL_DESTINATION[lab],
                          "confidence": "firm", "level": None}))
    return out, unsure


# --------------------------------------------------------------------------
# error taxonomy
# --------------------------------------------------------------------------

TAXONOMY = ["COMPREHENSION_MISS", "JUDGMENT_MISS", "MODEL_FAILURE", "POLICY_ERROR",
            "REFERENCE_AMBIGUITY", "REFERENCE_ERROR"]


def classify_error(row: Row, d: RoutingDecision) -> str | None:
    """First-pass automatic label. COMPREHENSION vs JUDGMENT needs you to read
    the evidence on the failed case -- the code guesses, you confirm."""
    if d.disposition.value == row.reference:
        return None
    if row.confidence == "soft":
        return "REFERENCE_AMBIGUITY"
    if d.reason is ReasonCode.MODEL_FAILURE:
        return "MODEL_FAILURE"
    return "JUDGMENT_MISS"          # confirm by reading d.judgment.cant_miss


def error_table(rows: list[Row], predicted: dict[str, RoutingDecision]) -> list[dict]:
    out = []
    for r in rows:
        d = predicted[r.id]
        code = classify_error(r, d)
        if code:
            out.append({"id": r.id, "code": code, "reference": r.reference,
                        "predicted": d.disposition.value, "reason": d.reason.value,
                        "basis": d.basis[:160],
                        "cant_miss": ([f"{c.condition} [{c.probability:g}%, {c.how_soon}]"
                                       for c in d.judgment.cant_miss] if d.judgment else []),
                        "needs_physician_action": (d.judgment.needs_physician_action.reason
                                                   if d.judgment else None)})
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def print_report(name: str, s: dict) -> None:
    print(f"\n=== {name}  (n={s['n']}) ===")
    for k in ["agreement", "critical_miss", "over_triage", "self_care_capture",
              "automated", "automated_self_care", "automated_urgent",
              "queue_removed", "deferred"]:
        print(f"  {k:22} {s[k]['display']}")
    if s["critical_miss_ids"]:
        print(f"  {'critical misses':22} {', '.join(s['critical_miss_ids'])}")
    if s["over_triage_ids"]:
        print(f"  {'over-triaged':22} {', '.join(s['over_triage_ids'])}")
    if s["wrongly_automated_ids"]:
        print(f"  {'wrongly automated':22} {', '.join(s['wrongly_automated_ids'])}")
    if s["deferral_reasons"]:
        print(f"  {'deferral reasons':22} {s['deferral_reasons']}")
    sec = s["secondary"]
    print(f"  {'[secondary] accuracy':22} {sec['accuracy']}   macro-F1 {sec['macro_f1']}")
    for src, b in s.get("by_source", {}).items():
        print(f"  {'[' + src + ']':22} n={b['n']}  agree {b['agreement']['display']}  "
              f"crit {b['critical_miss']['display']}  over {b['over_triage']['display']}")
    if s.get("critical_miss_by_level"):
        print(f"  {'critical miss by level':22} " + "  ".join(
            f"{k} {v['n']}/{v['d']}" for k, v in s["critical_miss_by_level"].items()))
        print(f"  {'over-triage by level':22} " + "  ".join(
            f"{k} {v['n']}/{v['d']}" for k, v in s["over_triage_by_level"].items()))


def save_run(path: str | Path, name: str, s: dict, errors: list[dict],
             telemetry: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(
        {"config": name, "metrics": s, "errors": errors, "telemetry": telemetry or {}},
        indent=2))
