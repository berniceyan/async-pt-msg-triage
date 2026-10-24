#!/usr/bin/env python3
"""Build the evaluation data from PMR-Bench, at a pinned revision.

  python tools/fetch_pmr.py           # writes data/pmr_{test,dev,extended}.csv + manifest
  python tools/fetch_pmr.py --check   # rebuild in memory, compare to the manifest, write nothing

The CSVs are NOT committed. The Reddit-derived text is CC BY-NC 4.0 and also
under Reddit's data policy, so the repository carries the recipe instead: the
pinned dataset revisions below, the folding rule in src/pmr.py, and
data/pmr_manifest.json with the sha256 of every file this writes. A rebuild
that does not reproduce those hashes means the source moved or the recipe did,
and either one invalidates every result computed from the old files.

Only PMR-Reddit is used. Its labels are an LLM's reading of a verified
clinician's reply to the post alone -- message-only, like this system. PMR-Synth
is not: its levels are a relative ranking within each 30-message inbox (exactly
five per level), made by clinicians with the patient's chart in view, so they
are neither absolute urgency nor reachable from the message.

Splits follow the source's own, so nothing here chooses which messages are hard:
  test      PMR-Reddit test posts (every unique text across the 1,502 pairs)
  dev       PMR-Reddit dev pool -- the only split anything may be tuned on
  extended  PMR-Reddit train pool -- nothing here is trained, so it is a second
            held-out set, scored once at the end

Source: Gatto et al., "Medical Triage as Pairwise Ranking: A Benchmark for
Urgency in Patient Portal Messages", arXiv 2601.13178.
https://huggingface.co/collections/PortalPal-AI/medical-triage-as-pairwise-ranking
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pmr import (MAPPING_VERSION, confidence_for, disposition_for,  # noqa: E402
                     rationale_for)

ORG = "PortalPal-AI"
REVISIONS = {
    "PMR-Reddit-Test-Pairs": "f72f5fe3d322397505ebc3bcb216c5f86af14c04",
    "PMR-Reddit-Train-Dev-Pool": "dea9728b18f356ed7ab1ee08e27b56391861bc7b",
}
DATA = Path("data")
MANIFEST = DATA / "pmr_manifest.json"
COLUMNS = ["id", "message", "given", "reference", "confidence", "rationale",
           "level", "source", "split"]
SPLITS = ("test", "dev", "extended")
PREFIX = {"test": "RT", "dev": "RD", "extended": "RX"}


def _rows(dataset: str, filename: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    path = hf_hub_download(f"{ORG}/{dataset}", filename, repo_type="dataset",
                           revision=REVISIONS[dataset])
    return pq.read_table(path).to_pylist()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _row(text: str, level: float, split: str) -> dict:
    rid = f"{PREFIX[split]}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]}"
    return {"id": rid, "message": text, "given": "",
            "reference": disposition_for(level),
            "confidence": confidence_for(level),
            "rationale": rationale_for(level, "reddit", split),
            "level": f"{level:g}", "source": "reddit", "split": split}


def build() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {s: [] for s in SPLITS}

    # Reddit test: the pairs repeat posts, so collect unique texts in first-seen order.
    levels: dict[str, float] = {}
    for r in _rows("PMR-Reddit-Test-Pairs", "data/test-00000-of-00001.parquet"):
        for side in ("chosen", "rejected"):
            text, level = r[side], float(r[f"{side}_level"])
            if levels.setdefault(text, level) != level:
                raise SystemExit(f"one post, two levels: {levels[text]} vs {level}")
    out["test"] += [_row(t, l, "test") for t, l in levels.items()]

    pool = "PMR-Reddit-Train-Dev-Pool"
    out["dev"] += [_row(r["text"], float(r["level"]), "dev")
                   for r in _rows(pool, "data/dev-00000-of-00001.parquet")]
    out["extended"] += [_row(r["text"], float(r["level"]), "extended")
                        for r in _rows(pool, "data/train-00000-of-00001.parquet")]

    # No message may sit in two splits, or in one split twice, under any spelling.
    seen: dict[str, str] = {}
    for split in SPLITS:
        for r in out[split]:
            k = _norm(r["message"])
            if k in seen:
                raise SystemExit(f"{r['id']} duplicates a message already in {seen[k]}")
            seen[k] = split
    ids = [r["id"] for s in SPLITS for r in out[s]]
    if len(ids) != len(set(ids)):
        raise SystemExit("id collision -- widen the hash prefix")
    return out


def _csv_bytes(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def manifest_for(out: dict[str, list[dict]]) -> dict:
    splits = {}
    for split, rows in out.items():
        words = [len(r["message"].split()) for r in rows]
        splits[split] = {
            "file": f"data/pmr_{split}.csv",
            "sha256": hashlib.sha256(_csv_bytes(rows)).hexdigest(),
            "n": len(rows),
            "by_source": dict(sorted(Counter(r["source"] for r in rows).items())),
            "reference": dict(sorted(Counter(r["reference"] for r in rows).items())),
            "soft": sum(r["confidence"] == "soft" for r in rows),
            "median_words": statistics.median(words),
        }
    return {"source": "PMR-Bench, arXiv 2601.13178", "license": "CC BY-NC 4.0",
            "revisions": REVISIONS, "mapping_version": MAPPING_VERSION, "splits": splits}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory and compare against the committed manifest")
    args = ap.parse_args()

    out = build()
    manifest = manifest_for(out)
    if args.check:
        if not MANIFEST.exists():
            print(f"no {MANIFEST} to check against"); return 1
        if json.loads(MANIFEST.read_text()) != manifest:
            print("REBUILD DOES NOT MATCH THE MANIFEST -- the source or the recipe moved")
            return 1
        print("rebuild matches the manifest"); return 0

    for split, rows in out.items():
        Path(manifest["splits"][split]["file"]).write_bytes(_csv_bytes(rows))
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    for split, s in manifest["splits"].items():
        print(f"{s['file']:24} n={s['n']:4}  {s['reference']}  soft={s['soft']}  "
              f"median {s['median_words']} words")
    print(f"{MANIFEST}  mapping {MAPPING_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
