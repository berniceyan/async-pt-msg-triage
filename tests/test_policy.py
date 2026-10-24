"""Policy invariants. These test whether the SYSTEM has safety properties --
not whether the model is smart. Every one runs with the model stubbed out."""
from src.policy import decide
from src.schema import (CantMiss, ClinicalJudgment, Disposition, EscalationLevel,
                        PhysicianAction, ReasonCode)


T = {"emergency_now": 5.0, "today": 10.0, "within_1_3_days": 20.0}


def cm(probability=50, how_soon="today", condition="x"):
    return CantMiss(condition=condition, probability=probability, how_soon=how_soon)


def j(cant_miss=(), physician=False, basis="b"):
    return ClinicalJudgment(cant_miss=list(cant_miss), basis=basis,
                            needs_physician_action=PhysicianAction(value=physician))


def test_no_judgment_lands_in_the_queue():
    d = decide("X", None, T)
    assert d.disposition is Disposition.ASYNC_PHYSICIAN
    assert d.reason is ReasonCode.MODEL_FAILURE


def test_a_probability_at_threshold_escalates():
    d = decide("X", j([cm(20, "within_1_3_days")]), T)
    assert d.disposition is Disposition.URGENT_ESCALATION
    assert d.reason is ReasonCode.CANT_MISS
    assert d.level is EscalationLevel.WITHIN_DAYS


def test_below_threshold_does_not_escalate():
    assert decide("X", j([cm(19, "within_1_3_days")]), T).disposition is Disposition.SELF_CARE


def test_the_bar_is_lower_the_sooner_care_is_needed():
    assert decide("X", j([cm(6, "emergency_now")]), T).disposition is Disposition.URGENT_ESCALATION
    assert decide("X", j([cm(6, "within_1_3_days")]), T).disposition is Disposition.SELF_CARE


def test_physician_action_cannot_lower_an_escalation():
    assert decide("X", j([cm(50)], physician=False), T).disposition is Disposition.URGENT_ESCALATION


def test_the_most_urgent_passing_timeline_sets_the_level():
    d = decide("X", j([cm(30, "within_1_3_days"), cm(6, "emergency_now"), cm(9, "today")]), T)
    assert d.level is EscalationLevel.EMERGENCY_NOW


def test_an_entry_below_threshold_never_sets_the_level():
    d = decide("X", j([cm(30, "within_1_3_days"), cm(4, "emergency_now")]), T)
    assert d.level is EscalationLevel.WITHIN_DAYS


def test_physician_action_goes_to_the_queue():
    d = decide("X", j([cm(1)], physician=True), T)
    assert d.disposition is Disposition.ASYNC_PHYSICIAN
    assert d.reason is ReasonCode.PHYSICIAN_ACTION


def test_only_no_escalation_and_no_action_clears():
    d = decide("X", j([], physician=False), T)
    assert d.disposition is Disposition.SELF_CARE
    assert d.reason is ReasonCode.CLINICAL_JUDGMENT


def test_frozen_thresholds_are_monotone_in_urgency():
    from src.policy import THRESHOLDS as F
    assert F["emergency_now"] <= F["today"] <= F["within_1_3_days"]
