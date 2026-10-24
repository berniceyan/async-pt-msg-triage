"""The scorer's arithmetic: intervals, the absent incumbent, the breakdowns."""
import pytest

from src.evaluate import Row, score, score_incumbent, wilson
from src.schema import Disposition, ReasonCode, RoutingDecision


def _d(rid, disp):
    return RoutingDecision(message_id=rid, disposition=Disposition(disp),
                           reason=ReasonCode.CLINICAL_JUDGMENT)


def test_wilson_is_informative_at_zero():
    lo, hi = wilson(0, 61)
    assert lo == 0.0 and 0.05 < hi < 0.07       # 0/61 means "no worse than ~6%"


def test_wilson_brackets_the_point_estimate():
    lo, hi = wilson(30, 100)
    assert lo < 0.30 < hi


def test_wilson_undefined_on_an_empty_pool():
    assert wilson(0, 0) is None


def test_incumbent_is_absent_when_the_dataset_has_none():
    rows = [Row("a", "m", "", "SELF_CARE", "firm")]
    assert score_incumbent(rows) is None


def test_incumbent_still_scores_when_present():
    rows = [Row("a", "m", "SELF_CARE", "SELF_CARE", "firm")]
    assert score_incumbent(rows)["agreement"]["n"] == 1


def test_breakdowns_split_by_source_and_level():
    rows = [
        Row("u1", "m", "", "URGENT_ESCALATION", "firm", level=1, source="reddit"),
        Row("u2", "m", "", "URGENT_ESCALATION", "firm", level=2, source="synth"),
        Row("a3", "m", "", "ASYNC_PHYSICIAN", "firm", level=3, source="reddit"),
        Row("s6", "m", "", "SELF_CARE", "firm", level=6, source="synth"),
    ]
    pred = {"u1": _d("u1", "ASYNC_PHYSICIAN"), "u2": _d("u2", "URGENT_ESCALATION"),
            "a3": _d("a3", "URGENT_ESCALATION"), "s6": _d("s6", "SELF_CARE")}
    s = score(rows, pred)
    assert s["critical_miss_by_level"]["L1"]["n"] == 1
    assert s["critical_miss_by_level"]["L2"]["n"] == 0
    assert s["over_triage_by_level"]["L3"]["n"] == 1
    assert s["over_triage_by_level"]["L6"]["n"] == 0
    assert s["by_source"]["reddit"]["critical_miss"]["display"].startswith("1/1")
    assert s["by_source"]["synth"]["agreement"]["n"] == 2


def test_no_breakdowns_without_source_or_level_columns():
    rows = [Row("a", "m", "SELF_CARE", "SELF_CARE", "firm")]
    s = score(rows, {"a": _d("a", "SELF_CARE")})
    assert "by_source" not in s and "critical_miss_by_level" not in s


@pytest.mark.parametrize("n,d,text", [(0, 61, "0/61 (0%, CI 0-6%)")])
def test_display_leads_with_the_count(n, d, text):
    from src.evaluate import _frac
    assert _frac(n, d)["display"] == text
