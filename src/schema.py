"""Types shared across the system. No logic lives here."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Disposition(str, Enum):
    SELF_CARE = "SELF_CARE"
    ASYNC_PHYSICIAN = "ASYNC_PHYSICIAN"
    URGENT_ESCALATION = "URGENT_ESCALATION"


class ReasonCode(str, Enum):
    # positive decisions
    CLINICAL_JUDGMENT = "CLINICAL_JUDGMENT"
    ADMIN_PROTOCOL = "ADMIN_PROTOCOL"   # reserved for a future admin lane; policy never emits it
    CANT_MISS = "CANT_MISS"            # a plausible or likely can't-miss: urgent
    PHYSICIAN_ACTION = "PHYSICIAN_ACTION"  # no can't-miss, but a clinician must act: the queue
    URGENCY_SCREEN = "URGENCY_SCREEN"  # the screen said not safe (or unsure) to wait three days
    # deferrals
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
    CLINICAL_AMBIGUITY = "CLINICAL_AMBIGUITY"
    MODEL_FAILURE = "MODEL_FAILURE"


DEFERRAL_CODES = {
    ReasonCode.INSUFFICIENT_INFORMATION,
    ReasonCode.CLINICAL_AMBIGUITY,
    ReasonCode.MODEL_FAILURE,
}

# Escalation levels carried as reason detail, not as separate destinations.
class EscalationLevel(str, Enum):
    EMERGENCY_NOW = "EMERGENCY_NOW"
    SAME_DAY = "SAME_DAY"
    WITHIN_DAYS = "WITHIN_DAYS"        # an in-person visit within 1-3 days
    BEHAVIORAL_CRISIS = "BEHAVIORAL_CRISIS"


LEVEL_RANK = {
    EscalationLevel.WITHIN_DAYS: 0,
    EscalationLevel.SAME_DAY: 1,
    EscalationLevel.BEHAVIORAL_CRISIS: 2,
    EscalationLevel.EMERGENCY_NOW: 3,
}


HowSoon = Literal["emergency_now", "today", "within_1_3_days"]

HOW_SOON_LEVEL = {
    "emergency_now": EscalationLevel.EMERGENCY_NOW,
    "today": EscalationLevel.SAME_DAY,
    "within_1_3_days": EscalationLevel.WITHIN_DAYS,
}


class Explanation(BaseModel):
    condition: str
    supporting_features: list[str] = Field(default_factory=list)


class CantMiss(BaseModel):
    condition: str
    # percent chance the condition is present AND this patient needs care on
    # the how_soon timeline because of it. Policy, not the model, decides what
    # probability is enough, separately for each timeline.
    probability: float = Field(ge=0, le=100)
    how_soon: HowSoon
    supporting_features: list[str] = Field(default_factory=list)
    would_rule_in_or_out: str = ""


class PhysicianAction(BaseModel):
    value: bool
    reason: str = ""


class ClinicalJudgment(BaseModel):
    """What the model is permitted to emit. It proposes no destination: policy
    reads cant_miss and needs_physician_action and routes. The differential is
    internal reasoning for audit, never patient-facing; basis is."""
    # --- routing surface ---
    cant_miss: list[CantMiss] = Field(default_factory=list)
    needs_physician_action: PhysicianAction
    # --- reasoning and evidence ---
    most_likely: list[Explanation] = Field(default_factory=list)
    complaints: list[str] = Field(default_factory=list)
    negated: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    basis: str = ""


class ScreenVerdict(BaseModel):
    """The urgency screen: one question -- can this wait three days for a
    physician? -- answered on three levels. Policy decides what unsure means."""
    verdict: Literal["safe_to_wait", "unsure", "not_safe"]
    how_soon: HowSoon | None = None
    reason: str = ""


class RoutingDecision(BaseModel):
    message_id: str | None = None
    disposition: Disposition
    reason: ReasonCode
    level: EscalationLevel | None = None
    judgment: ClinicalJudgment | None = None
    screen: ScreenVerdict | None = None
    basis: str = ""
    policy_version: str = "unset"
