"""The PMR-Bench fold and the files built from it. No network."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from src import pmr
from src.schema import Disposition

MANIFEST = Path("data/pmr_manifest.json")
COLUMNS = ["id", "message", "given", "reference", "confidence", "rationale",
           "level", "source", "split"]


@pytest.mark.parametrize("level,dest", [
    (1, "URGENT_ESCALATION"), (2, "URGENT_ESCALATION"),
    (3, "URGENT_ESCALATION"), (4, "ASYNC_PHYSICIAN"),
    (5, "SELF_CARE"), (6, "SELF_CARE"),
])
def test_integer_levels_fold_to_three_destinations(level, dest):
    assert pmr.disposition_for(level) == dest
    assert pmr.confidence_for(level) == "firm"


def test_every_destination_is_a_real_disposition():
    assert set(pmr.LEVEL_TO_DISPOSITION.values()) == {d.value for d in Disposition}


@pytest.mark.parametrize("level,dest,conf", [
    (1.5, "URGENT_ESCALATION", "firm"),    # both neighbours urgent
    (3.5, "URGENT_ESCALATION", "soft"),    # straddles urgent | queue; a tie escalates
    (4.333, "ASYNC_PHYSICIAN", "soft"),    # straddles async | self-care
    (4.5, "ASYNC_PHYSICIAN", "soft"),      # a tie rounds toward the more urgent level
    (4.667, "SELF_CARE", "soft"),
    (5.5, "SELF_CARE", "firm"),
])
def test_fractional_levels(level, dest, conf):
    assert pmr.disposition_for(level) == dest
    assert pmr.confidence_for(level) == conf


def test_the_urgent_boundary_sits_after_l3():
    """A queue that can take three days cannot hold a 'visit within 1-3 days'."""
    assert pmr.disposition_for(2.5) == "URGENT_ESCALATION"
    assert pmr.confidence_for(2.5) == "firm"      # both neighbours urgent now
    assert pmr.disposition_for(3.67) == "ASYNC_PHYSICIAN"


def test_out_of_range_levels_are_refused():
    with pytest.raises(ValueError):
        pmr.disposition_for(0.5)
    with pytest.raises(ValueError):
        pmr.disposition_for(6.5)


def test_rationale_names_the_level_never_a_diagnosis():
    assert pmr.rationale_for(2, "reddit", "test") == "PMR L2 (same-day care) · reddit-test"


# --------------------------------------------------------------------------
# the built files -- skipped until `make data` has run
# --------------------------------------------------------------------------

def _manifest():
    if not MANIFEST.exists():
        pytest.skip("no manifest")
    m = json.loads(MANIFEST.read_text())
    if not all(Path(s["file"]).exists() for s in m["splits"].values()):
        pytest.skip("PMR data not fetched -- run `make data`")
    return m


def test_files_match_the_manifest():
    m = _manifest()
    assert m["mapping_version"] == pmr.MAPPING_VERSION
    for s in m["splits"].values():
        assert hashlib.sha256(Path(s["file"]).read_bytes()).hexdigest() == s["sha256"], s["file"]


def test_schema_ids_and_labels():
    m = _manifest()
    seen_ids, seen_text = set(), set()
    for split, s in m["splits"].items():
        with open(s["file"], newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == COLUMNS
            rows = list(reader)
        assert len(rows) == s["n"]
        for r in rows:
            assert r["id"] not in seen_ids
            seen_ids.add(r["id"])
            key = " ".join(r["message"].split()).lower()
            assert key not in seen_text, f"{r['id']} repeats a message across splits"
            seen_text.add(key)
            assert r["given"] == ""
            assert r["split"] == split
            assert r["reference"] == pmr.disposition_for(float(r["level"]))
            assert r["confidence"] in ("firm", "soft")
            assert r["source"] == "reddit"            # Synth is not used: relative, chart-based labels
