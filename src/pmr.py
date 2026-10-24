"""PMR-Bench urgency levels -> the three destinations.

PMR-Bench (Gatto et al., arXiv 2601.13178) labels each message on a six-level
ordinal scale written with clinicians. This project routes to three
destinations, so the scale is folded, once, here. Nothing else in the code
knows the six levels exist.

  L1  life-threatening, emergency department     URGENT_ESCALATION
  L2  same-day or urgent care                    URGENT_ESCALATION
  L3  in-person visit within 1-3 days            URGENT_ESCALATION
  L4  in-person visit, not time-sensitive        ASYNC_PHYSICIAN
  L5  self-treatable with remote guidance        SELF_CARE
  L6  no action needed                           SELF_CARE

The line between URGENT and the queue is set by how long the queue can take,
and the service's answer is: up to three days. Anything that cannot safely wait
that long is URGENT. L2 obviously cannot. L3 -- a visit within 1-3 days --
cannot either: a physician reading it on day three is already past the window
before anyone has booked the visit. L4 can wait for the queue, and the queue is
where it gets scheduled.

Reddit levels can be fractional -- the paper averages over expert replies. A
fractional level is rounded to the nearer integer, ties toward the more urgent
level, and marked soft when its two neighbours fold to different destinations
(3.5 sits between URGENT and ASYNC; 4.33, 4.5, 4.67 between ASYNC and
SELF_CARE). Every other level is firm.
Soft rows feed the REFERENCE_AMBIGUITY taxonomy and the --firm-only pool, which
is what the physician's firm/soft flag fed before.
"""
from __future__ import annotations

import math

MAPPING_VERSION = "pmr3-v1"

LEVEL_TO_DISPOSITION = {
    1: "URGENT_ESCALATION",
    2: "URGENT_ESCALATION",
    3: "URGENT_ESCALATION",
    4: "ASYNC_PHYSICIAN",
    5: "SELF_CARE",
    6: "SELF_CARE",
}

LEVEL_MEANING = {
    1: "emergency department",
    2: "same-day care",
    3: "visit within 1-3 days",
    4: "visit, not time-sensitive",
    5: "self-care with guidance",
    6: "no action",
}


def round_level(level: float) -> int:
    """Nearest integer, ties toward the more urgent (lower) level."""
    if not 1 <= level <= 6:
        raise ValueError(f"PMR level out of range: {level}")
    lo = math.floor(level)
    return lo if level - lo <= 0.5 else lo + 1


def disposition_for(level: float) -> str:
    return LEVEL_TO_DISPOSITION[round_level(level)]


def confidence_for(level: float) -> str:
    """soft only when the fractional level straddles a destination boundary."""
    lo, hi = math.floor(level), math.ceil(level)
    return "soft" if LEVEL_TO_DISPOSITION[lo] != LEVEL_TO_DISPOSITION[hi] else "firm"


def rationale_for(level: float, source: str, split: str) -> str:
    return f"PMR L{level:g} ({LEVEL_MEANING[round_level(level)]}) · {source}-{split}"
