"""Wires urgency screen -> judgment -> policy. The only place the three meet.

The screen asks one question -- can this safely wait up to three days for
a physician? -- and a message it will not clear is escalated before the
judgment runs. The judgment only decides queue vs self-care for a message the
screen let wait.

The lexical and semantic screen that once ran first is gone: on PMR's Reddit
test split it caught 25 of 51 urgent messages and over-escalated 87 of 311
non-urgent ones (lexical alone: 1 of 51 caught, 3 of 311 over-escalated).
"""
from __future__ import annotations

from dataclasses import dataclass

from .judgment import judge, screen
from .policy import decide, decide_screened
from .schema import RoutingDecision

# What the telemetry of a stage that did not run says, so a skipped call can
# never be read as a failed one.
NOT_CALLED = {"model": "not_called", "ok": None, "latency_s": None}


@dataclass
class RouteTrace:
    """Which stages ran. `reason` lives on the decision, not here."""
    screen_meta: dict
    model_called: bool


def route_message(message: str, message_id: str | None = None,
                  screen_fn=screen, judge_fn=judge, unsure_to: str = "urgent"
                  ) -> tuple[RoutingDecision, dict, dict]:
    """The shipped route. Returns (decision, screen telemetry, judgment telemetry)."""
    d, smeta, meta, _ = route_message_traced(message, message_id, screen_fn,
                                             judge_fn, unsure_to)
    return d, smeta, meta


def route_message_traced(message: str, message_id: str | None = None,
                         screen_fn=screen, judge_fn=judge,
                         unsure_to: str = "urgent", on_stage=None
                         ) -> tuple[RoutingDecision, dict, dict, RouteTrace]:
    """route_message, plus a record of what ran.

    `on_stage(name)` is called as each stage finishes -- "screen", "model",
    then "policy" -- including the model stage when it was skipped, because a
    skipped stage is still a stage. It observes and cannot change a decision.
    """
    verdict, smeta = screen_fn(message)
    if on_stage is not None:
        on_stage("screen")

    # Policy reads the judgment only on safe_to_wait, so nothing else needs one.
    if verdict is not None and verdict.verdict == "safe_to_wait":
        judgment, meta = judge_fn(message)
        called = True
    else:
        judgment, meta, called = None, dict(NOT_CALLED), False
    if on_stage is not None:
        on_stage("model")

    decision = decide_screened(message_id, verdict, judgment, unsure_to)
    if on_stage is not None:
        on_stage("policy")
    return decision, smeta, meta, RouteTrace(screen_meta=smeta, model_called=called)


def route_message_scored(message: str, message_id: str | None = None,
                         screen_fn=screen, judge_fn=judge,
                         unsure_to: str = "urgent"
                         ) -> tuple[RoutingDecision, dict, dict]:
    """The evaluation's route: both calls on every message.

    Same decision as route_message -- policy ignores the judgment once the
    screen escalates -- but with the judgment always fetched, so both unsure_to
    settings and the judgment-only baseline score from one set of calls.
    """
    verdict, smeta = screen_fn(message)
    judgment, meta = judge_fn(message)
    return decide_screened(message_id, verdict, judgment, unsure_to), smeta, meta


def route_judgment_only(message: str, message_id: str | None = None,
                        judge_fn=judge) -> tuple[RoutingDecision, dict]:
    """Judgment only, kept as the baseline: no screen, can't-miss thresholds escalate."""
    judgment, meta = judge_fn(message)
    return decide(message_id, judgment), meta
