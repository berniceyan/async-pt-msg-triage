"""The pipeline is screen -> judgment -> policy, and the judgment only runs
when the screen lets a message wait."""
import pytest

from src.judgment import judge_stub
from src.pipeline import (route_judgment_only, route_message, route_message_scored,
                          route_message_traced)
from src.schema import (CantMiss, ClinicalJudgment, Disposition, PhysicianAction,
                        ReasonCode, ScreenVerdict)


def _fixed(x):
    return lambda message, **kw: (x, {"ok": True})


def sv(verdict, how_soon=None):
    return ScreenVerdict(verdict=verdict, how_soon=how_soon, reason="r")


def j(physician=False):
    return ClinicalJudgment(needs_physician_action=PhysicianAction(value=physician))


def _recording(calls, name, value):
    def fn(message, **kw):
        calls.append(name)
        return value, {"ok": True}
    return fn


def test_screen_down_escalates_without_asking_the_judgment():
    calls = []
    d, _, meta = route_message("I have crushing chest pain", "t", screen_fn=judge_stub,
                               judge_fn=_recording(calls, "judge", j()))
    assert d.disposition is Disposition.URGENT_ESCALATION
    assert d.reason is ReasonCode.MODEL_FAILURE
    assert calls == [] and meta["model"] == "not_called"


@pytest.mark.parametrize("verdict", ["not_safe", "unsure"])
def test_an_escalation_never_calls_the_judgment(verdict):
    calls = []
    d, _, meta = route_message("m", "t", screen_fn=_fixed(sv(verdict, "today")),
                               judge_fn=_recording(calls, "judge", j()))
    assert d.disposition is Disposition.URGENT_ESCALATION
    assert calls == [] and meta["ok"] is None


def test_safe_to_wait_asks_the_judgment():
    calls = []
    d, _, _ = route_message("m", "t", screen_fn=_fixed(sv("safe_to_wait")),
                            judge_fn=_recording(calls, "judge", j(physician=True)))
    assert calls == ["judge"]
    assert d.disposition is Disposition.ASYNC_PHYSICIAN


def test_judgment_down_after_safe_to_wait_lands_in_the_queue():
    d, _, _ = route_message("m", "t", screen_fn=_fixed(sv("safe_to_wait")),
                            judge_fn=judge_stub)
    assert d.disposition is Disposition.ASYNC_PHYSICIAN
    assert d.reason is ReasonCode.MODEL_FAILURE


@pytest.mark.parametrize("verdict", ["safe_to_wait", "unsure", "not_safe", None])
@pytest.mark.parametrize("unsure_to", ["urgent", "queue"])
@pytest.mark.parametrize("physician", [True, False])
def test_the_scored_route_decides_what_the_shipped_route_decides(verdict, unsure_to, physician):
    """run_eval fetches both calls on every message; the shipped route skips the
    judgment it does not need. The numbers are only about the shipped system if
    the two agree everywhere."""
    scr = _fixed(sv(verdict, "today") if verdict else None)
    jud = _fixed(j(physician))
    shipped, _, _ = route_message("m", "t", screen_fn=scr, judge_fn=jud, unsure_to=unsure_to)
    scored, _, _ = route_message_scored("m", "t", screen_fn=scr, judge_fn=jud, unsure_to=unsure_to)
    assert (shipped.disposition, shipped.reason, shipped.level) == \
           (scored.disposition, scored.reason, scored.level)


def test_on_stage_fires_screen_model_policy_even_when_the_model_is_skipped():
    for verdict in ("safe_to_wait", "not_safe"):
        seen = []
        route_message_traced("m", "t", screen_fn=_fixed(sv(verdict)), judge_fn=_fixed(j()),
                             on_stage=seen.append)
        assert seen == ["screen", "model", "policy"]


def test_on_stage_observes_and_cannot_change_the_decision():
    args = dict(screen_fn=_fixed(sv("safe_to_wait")), judge_fn=_fixed(j()))
    quiet = route_message_traced("m", "t", **args)
    noisy = route_message_traced("m", "t", on_stage=lambda name: None, **args)
    assert quiet[0].disposition == noisy[0].disposition
    assert quiet[0].reason == noisy[0].reason
    assert quiet[3].model_called == noisy[3].model_called


def test_the_judgment_only_baseline_still_routes_on_cant_miss():
    jj = ClinicalJudgment(cant_miss=[CantMiss(condition="x", probability=90,
                                              how_soon="emergency_now")],
                          needs_physician_action=PhysicianAction(value=False))
    d, _ = route_judgment_only("m", "t", judge_fn=_fixed(jj))
    assert d.disposition is Disposition.URGENT_ESCALATION
    assert d.reason is ReasonCode.CANT_MISS


def test_an_out_of_range_probability_is_a_model_failure():
    """A judgment that does not validate never routes: judge() returns None."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ClinicalJudgment.model_validate(
            {"cant_miss": [{"condition": "x", "probability": 140, "how_soon": "today"}],
             "needs_physician_action": {"value": False}})
