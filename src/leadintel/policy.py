"""The decision layer: turns a calibrated probability into a recommended action.

Kept deliberately separate from the model. The model answers "how likely is this lead to
subscribe?"; this policy answers "given our costs and how much we have already contacted
them, what should the team do next?". Every threshold is an explicit, editable
assumption (PolicyConfig), never hidden inside the model.

Decision logic (first matching rule wins):

  1. The model's uncertainty interval straddles the break-even line  -> HUMAN REVIEW
  2. p >= priority threshold                                          -> PRIORITISE
  3. p >= break-even and heavily contacted this campaign              -> CHANGE APPROACH
  4. p >= break-even                                                  -> FOLLOW UP
  5. p <  break-even and heavily contacted                            -> STOP FOLLOW-UP
  6. p <  break-even                                                  -> DEPRIORITISE

Break-even comes from expected value (Elkan 2001): a follow-up is worth making when
p x value_of_conversion > cost_of_contact, i.e. p > cost / value.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field, computed_field

from leadintel.patterns import HEAVY_CONTACT


class Action(StrEnum):
    PRIORITISE = "prioritise"
    FOLLOW_UP = "follow_up"
    CHANGE_APPROACH = "change_approach"
    REVIEW = "review"
    DEPRIORITISE = "deprioritise"
    STOP = "stop"


ACTION_ORDER: tuple[Action, ...] = (
    Action.PRIORITISE,
    Action.FOLLOW_UP,
    Action.CHANGE_APPROACH,
    Action.REVIEW,
    Action.DEPRIORITISE,
    Action.STOP,
)

ACTION_TITLE: dict[Action, str] = {
    Action.PRIORITISE: "Prioritise follow-up",
    Action.FOLLOW_UP: "Follow up",
    Action.CHANGE_APPROACH: "Follow up — change approach",
    Action.REVIEW: "Human review recommended",
    Action.DEPRIORITISE: "Deprioritise",
    Action.STOP: "Stop repeated follow-up",
}

ACTION_NEXT_STEP: dict[Action, str] = {
    Action.PRIORITISE: "Contact in the next calling block, ahead of standard leads.",
    Action.FOLLOW_UP: "Keep in the standard follow-up queue.",
    Action.CHANGE_APPROACH: (
        "Still worth pursuing, but repeated contact this campaign has not converted. "
        "Before calling again, review channel, timing or offer with the relationship manager."
    ),
    Action.REVIEW: (
        "The estimate is too uncertain to call either way. A relationship manager should "
        "decide using context the model cannot see."
    ),
    Action.DEPRIORITISE: "No active follow-up this campaign; keep for future campaigns.",
    Action.STOP: (
        "Expected value of another call is below cost and the lead has already been "
        "contacted repeatedly. Stop follow-up for this campaign."
    ),
}


class PolicyConfig(BaseModel):
    """Business assumptions. Defaults are illustrative, NOT real bank economics."""

    contact_cost: float = Field(
        default=5.0, gt=0, le=1000, description="Cost of one follow-up contact (EUR, assumed)"
    )
    conversion_value: float = Field(
        default=50.0, gt=0, le=100_000, description="Net value of one subscription (EUR, assumed)"
    )
    priority_threshold: float = Field(
        default=0.35, gt=0, lt=1, description="Probability at which a lead jumps the queue"
    )
    heavy_contact: int = Field(
        default=HEAVY_CONTACT,
        ge=2,
        le=20,
        description="Contacts this campaign = 'heavily contacted'",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def break_even(self) -> float:
        return min(0.99, self.contact_cost / self.conversion_value)


class Decision(BaseModel):
    action: Action
    title: str
    next_step: str
    reasons: list[str]
    expected_value: float  # p * value - cost, EUR per follow-up (under the assumptions)


def decide(
    probability: float,
    interval: tuple[float, float],
    campaign: int,
    config: PolicyConfig,
) -> Decision:
    p, (lo, hi) = probability, interval
    be = config.break_even
    heavy = campaign >= config.heavy_contact
    ev = p * config.conversion_value - config.contact_cost
    reasons: list[str] = []

    if lo < be <= hi:
        action = Action.REVIEW
        reasons.append(
            f"Model estimate {p:.0%}, but re-trained variants of the model range from "
            f"{lo:.0%} to {hi:.0%}, which spans the {be:.0%} break-even line."
        )
    elif p >= config.priority_threshold:
        action = Action.PRIORITISE
        reasons.append(
            f"Estimate {p:.0%} is at or above the "
            f"{config.priority_threshold:.0%} priority threshold."
        )
    elif p >= be:
        action = Action.CHANGE_APPROACH if heavy else Action.FOLLOW_UP
        reasons.append(f"Estimate {p:.0%} is above the {be:.0%} break-even line.")
    else:
        action = Action.STOP if heavy else Action.DEPRIORITISE
        reasons.append(f"Estimate {p:.0%} is below the {be:.0%} break-even line.")

    if heavy and action in {Action.CHANGE_APPROACH, Action.STOP}:
        reasons.append(
            f"Already contacted {campaign} times this campaign (≥ {config.heavy_contact}); "
            "conversion falls steadily with repeated contact."
        )
    reasons.append(
        f"Expected value of one more contact ≈ €{ev:,.2f} "
        f"(assumes €{config.conversion_value:,.0f} per subscription, "
        f"€{config.contact_cost:,.0f} per contact)."
    )
    return Decision(
        action=action,
        title=ACTION_TITLE[action],
        next_step=ACTION_NEXT_STEP[action],
        reasons=reasons,
        expected_value=round(ev, 2),
    )


def decide_many(
    probability: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    campaign: np.ndarray,
    config: PolicyConfig,
) -> np.ndarray:
    """Vectorised `decide` for the whole portfolio (same rules, same order)."""
    be = config.break_even
    heavy = campaign >= config.heavy_contact
    return np.select(
        [
            (low < be) & (be <= high),
            probability >= config.priority_threshold,
            (probability >= be) & heavy,
            probability >= be,
            heavy,
        ],
        [
            Action.REVIEW.value,
            Action.PRIORITISE.value,
            Action.CHANGE_APPROACH.value,
            Action.FOLLOW_UP.value,
            Action.STOP.value,
        ],
        default=Action.DEPRIORITISE.value,
    )
