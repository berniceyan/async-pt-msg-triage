"""Deterministic policy. Contains NO clinical knowledge -- only a fixed reading
of the model's fields.

  1. Any failure to decide lands in the physician queue with a reason code.
  2. A can't-miss whose probability reaches the threshold for its timeline
     escalates, at the most urgent such timeline. Nothing else is consulted, so
     nothing can lower it.
  3. Otherwise a clinician action sends it to the queue.
  4. Only a judgment with no such can't-miss and no physician action clears a
     message.

The thresholds are lower the sooner care is needed: a one-in-twenty chance of
an emergency is worth escalating, the same chance of something that only needs
a visit this week is not. They were selected on dev by a rule fixed in advance
and are frozen with the prompt.
"""
from __future__ import annotations

from .schema import (HOW_SOON_LEVEL, LEVEL_RANK, CantMiss, ClinicalJudgment,
                     Disposition, EscalationLevel, ReasonCode, RoutingDecision,
                     ScreenVerdict)

POLICY_VERSION = "judgment_only"

# percent; see the module docstring
THRESHOLDS = {"emergency_now": 5.0, "today": 20.0, "within_1_3_days": 50.0}  # dev sweep pick, frozen 2026-10-20


def escalating(judgment: ClinicalJudgment,
               thresholds: dict[str, float] = THRESHOLDS) -> list[CantMiss]:
    """The can't-miss entries that send the message urgent."""
    return [c for c in judgment.cant_miss if c.probability >= thresholds[c.how_soon]]


# The urgency screen replaces the can't-miss thresholds for escalation.
SCREEN_POLICY_VERSION = "screened"
UNSURE_TO = ("urgent", "queue")


def decide_screened(message_id: str | None, screen: ScreenVerdict | None,
                    judgment: ClinicalJudgment | None,
                    unsure_to: str = "urgent") -> RoutingDecision:
    """Screened routing.

      1. A screen that failed escalates: the screen exists to catch what cannot
         wait, so its failure fails toward escalation.
      2. not_safe escalates; unsure escalates or goes to the queue per unsure_to.
         Nothing downstream can lower it.
      3. Otherwise the judgment decides queue vs self-care exactly as in the baseline,
         minus the can't-miss thresholds, which no longer route anything.
    """
    if unsure_to not in UNSURE_TO:
        raise ValueError(f"unsure_to must be one of {UNSURE_TO}")

    def out(disposition, reason, basis, level=None):
        return RoutingDecision(message_id=message_id, disposition=disposition,
                               reason=reason, level=level, judgment=judgment,
                               screen=screen, basis=basis,
                               policy_version=SCREEN_POLICY_VERSION)

    if screen is None:
        return out(Disposition.URGENT_ESCALATION, ReasonCode.MODEL_FAILURE,
                   "urgency screen produced no verdict")
    level = HOW_SOON_LEVEL[screen.how_soon] if screen.how_soon else EscalationLevel.WITHIN_DAYS
    if screen.verdict == "not_safe" or (screen.verdict == "unsure" and unsure_to == "urgent"):
        return out(Disposition.URGENT_ESCALATION, ReasonCode.URGENCY_SCREEN,
                   screen.reason, level)
    if screen.verdict == "unsure":
        return out(Disposition.ASYNC_PHYSICIAN, ReasonCode.URGENCY_SCREEN, screen.reason)

    if judgment is None:
        return out(Disposition.ASYNC_PHYSICIAN, ReasonCode.MODEL_FAILURE,
                   "no judgment produced")
    if judgment.needs_physician_action.value:
        return out(Disposition.ASYNC_PHYSICIAN, ReasonCode.PHYSICIAN_ACTION,
                   judgment.basis)
    return out(Disposition.SELF_CARE, ReasonCode.CLINICAL_JUDGMENT, judgment.basis)


def decide(message_id: str | None, judgment: ClinicalJudgment | None,
           thresholds: dict[str, float] = THRESHOLDS) -> RoutingDecision:
    def out(disposition, reason, basis, level=None):
        return RoutingDecision(message_id=message_id, disposition=disposition,
                               reason=reason, level=level, judgment=judgment,
                               basis=basis, policy_version=POLICY_VERSION)

    if judgment is None:
        return out(Disposition.ASYNC_PHYSICIAN, ReasonCode.MODEL_FAILURE,
                   "no judgment produced")

    concerns = escalating(judgment, thresholds)
    if concerns:
        level = max((HOW_SOON_LEVEL[c.how_soon] for c in concerns),
                    key=lambda l: LEVEL_RANK[l])
        return out(Disposition.URGENT_ESCALATION, ReasonCode.CANT_MISS,
                   judgment.basis, level)

    if judgment.needs_physician_action.value:
        return out(Disposition.ASYNC_PHYSICIAN, ReasonCode.PHYSICIAN_ACTION,
                   judgment.basis)

    return out(Disposition.SELF_CARE, ReasonCode.CLINICAL_JUDGMENT, judgment.basis)
