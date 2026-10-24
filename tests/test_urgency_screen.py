"""Urgency screen: policy invariants, run with the model stubbed out."""
import pytest

from src.pipeline import route_message_scored
from src.policy import decide_screened
from src.schema import (CantMiss, ClinicalJudgment, Disposition, EscalationLevel,
                        PhysicianAction, ReasonCode, ScreenVerdict)


def sv(verdict, how_soon=None):
    return ScreenVerdict(verdict=verdict, how_soon=how_soon, reason="r")


def j(physician=False, cant_miss=()):
    return ClinicalJudgment(cant_miss=list(cant_miss), basis="b",
                            needs_physician_action=PhysicianAction(value=physician))


def test_a_failed_screen_escalates():
    d = decide_screened("X", None, j())
    assert d.disposition is Disposition.URGENT_ESCALATION
    assert d.reason is ReasonCode.MODEL_FAILURE


def test_not_safe_escalates_whatever_the_judgment_says():
    for unsure_to in ("urgent", "queue"):
        d = decide_screened("X", sv("not_safe", "today"), j(), unsure_to)
        assert d.disposition is Disposition.URGENT_ESCALATION
        assert d.level is EscalationLevel.SAME_DAY


def test_unsure_follows_the_setting():
    assert decide_screened("X", sv("unsure"), j(), "urgent").disposition is Disposition.URGENT_ESCALATION
    d = decide_screened("X", sv("unsure"), j(), "queue")
    assert d.disposition is Disposition.ASYNC_PHYSICIAN
    assert d.reason is ReasonCode.URGENCY_SCREEN


def test_unsure_to_queue_never_reaches_self_care():
    assert decide_screened("X", sv("unsure"), None, "queue").disposition is Disposition.ASYNC_PHYSICIAN


def test_an_escalation_with_no_timeline_is_within_days():
    assert decide_screened("X", sv("not_safe"), j()).level is EscalationLevel.WITHIN_DAYS


def test_cant_miss_probabilities_no_longer_escalate():
    high = j(cant_miss=[CantMiss(condition="x", probability=99, how_soon="emergency_now")])
    assert decide_screened("X", sv("safe_to_wait"), high).disposition is Disposition.SELF_CARE


def test_safe_to_wait_then_judgment_decides_queue_or_self_care():
    assert decide_screened("X", sv("safe_to_wait"), j(True)).disposition is Disposition.ASYNC_PHYSICIAN
    assert decide_screened("X", sv("safe_to_wait"), j(False)).disposition is Disposition.SELF_CARE
    d = decide_screened("X", sv("safe_to_wait"), None)
    assert d.disposition is Disposition.ASYNC_PHYSICIAN and d.reason is ReasonCode.MODEL_FAILURE


def test_a_bad_setting_is_refused():
    with pytest.raises(ValueError):
        decide_screened("X", sv("unsure"), j(), "self_care")


def test_the_scored_route_calls_both_even_when_escalating():
    calls = []
    def scr(m, **kw): calls.append("screen"); return sv("not_safe", "emergency_now"), {"ok": True}
    def jud(m, **kw): calls.append("judge"); return j(), {"ok": True}
    d, _, _ = route_message_scored("m", "t", screen_fn=scr, judge_fn=jud)
    assert calls == ["screen", "judge"]
    assert d.disposition is Disposition.URGENT_ESCALATION


def test_screen_verdict_rejects_an_unknown_answer():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ScreenVerdict.model_validate({"verdict": "maybe"})
