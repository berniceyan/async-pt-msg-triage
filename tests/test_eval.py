"""Invariants over the runner and the clinician-mark scoring."""
from pathlib import Path

import run_eval
from src.evaluate import Row, relabel


def test_every_config_name_is_a_valid_results_filename():
    for name in run_eval.CONFIGS:
        assert name.replace("_", "").isalnum(), f"{name} breaks results/<dataset>.{name}.json"


def test_results_path_is_keyed_by_dataset_and_config():
    """Two datasets share the config names, so the results path must carry the
    dataset too."""
    src = Path("run_eval.py").read_text()
    assert 'f"results/{dataset}.{name}.json"' in src
    assert 'f"results/{name}.json"' not in src


def _row(i, ref="SELF_CARE"):
    return Row(id=i, message="m", given="", reference=ref, confidence="soft", level=5.5)


def test_relabel_swaps_the_reference_and_sets_aside_unsure():
    rows = [_row("a"), _row("b"), _row("c"), _row("unlabelled")]
    out, unsure = relabel(rows, {"a": "urgent", "b": "queue", "c": "unsure"})
    assert unsure == 1
    assert [(r.id, r.reference) for r in out] == [("a", "URGENT_ESCALATION"),
                                                  ("b", "ASYNC_PHYSICIAN")]
    assert all(r.confidence == "firm" and r.level is None for r in out)
