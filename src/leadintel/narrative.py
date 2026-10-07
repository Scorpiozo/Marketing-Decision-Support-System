"""Plain-English evidence built only from an Assessment's numbers.

No free text is invented: every sentence is a template filled from the model output,
the SHAP contributions, the observed pattern rate or the policy decision.
"""

from __future__ import annotations

from leadintel.explain import Contribution
from leadintel.service import Assessment


def driver_sentence(c: Contribution) -> str:
    verb = "raised" if c.direction == "up" else "lowered"
    factor = c.odds_multiplier if c.direction == "up" else 1 / c.odds_multiplier
    return f"{c.label}: {c.value} — {verb} the odds ×{factor:.1f}"


def summary(a: Assessment) -> str:
    """One-paragraph 'what happened and what the model thinks'."""
    lead, pat = a.lead, a.pattern
    contacts = f"{lead.campaign} contact{'s' if lead.campaign != 1 else ''} this campaign"
    parts = [f"{pat.history} lead, {contacts}."]
    estimate = f"Estimated chance of subscribing: {a.probability:.0%}"
    if pat.observed_rate is not None and pat.observed_n >= 30:
        estimate += (
            f"; historically, {pat.observed_rate:.0%} of the {pat.observed_n:,} past leads "
            "with the same interaction pattern subscribed"
        )
    parts.append(estimate + ".")
    ups = [c for c in a.explanation.drivers_up if not c.is_context][:2]
    downs = [c for c in a.explanation.drivers_down if not c.is_context][:2]
    if ups:
        parts.append(
            "Supporting signals: " + "; ".join(f"{c.label.lower()} ({c.value})" for c in ups) + "."
        )
    if downs:
        parts.append(
            "Holding it back: " + "; ".join(f"{c.label.lower()} ({c.value})" for c in downs) + "."
        )
    return " ".join(parts)


def robustness(a: Assessment) -> tuple[str, str]:
    """(label, explanation) describing how stable the recommendation is."""
    lo, hi = a.interval
    if a.decision.action.value == "review":
        return (
            "Borderline",
            f"Re-trained versions of the model put this lead between {lo:.0%} and {hi:.0%}, "
            "on both sides of the follow-up line.",
        )
    return (
        "Stable",
        f"Re-trained versions of the model agree: {lo:.0%}–{hi:.0%}, all on the same side "
        "of the follow-up line.",
    )
