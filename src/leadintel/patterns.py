"""Interaction patterns: a readable description of a lead's contact history.

Two transparent axes, built only from UCI columns:
  * history   — what happened in *previous* campaigns (`previous`, `poutcome`)
  * intensity — how hard *this* campaign has already pushed (`campaign`)

These are descriptive labels for humans, not model inputs (the model sees the raw
columns). Observed conversion rates per pattern come from the training split and are
shown next to the label so a user can sanity-check the model against plain counts.
"""

from __future__ import annotations

HISTORY_ORDER = (
    "Previously converted",
    "Previously declined",
    "Previously contacted, outcome unclear",
    "New to campaigns",
)
INTENSITY_ORDER = ("1 contact", "2–3 contacts", "4+ contacts")

# Contacts this campaign at which we consider the lead "heavily contacted". Observed
# conversion falls from 14.6 % (1 contact) to 9.0 % (4) and keeps falling (4.2 % at 10+);
# this is a proxy for contact fatigue, not a measured fatigue label.
HEAVY_CONTACT = 4


def history_label(previous: int, poutcome: str) -> str:
    if previous == 0:
        return "New to campaigns"
    if poutcome == "success":
        return "Previously converted"
    if poutcome == "failure":
        return "Previously declined"
    return "Previously contacted, outcome unclear"


def intensity_label(campaign: int) -> str:
    if campaign <= 1:
        return "1 contact"
    if campaign < HEAVY_CONTACT:
        return "2–3 contacts"
    return "4+ contacts"


def pattern_label(previous: int, poutcome: str, campaign: int) -> str:
    return f"{history_label(previous, poutcome)} · {intensity_label(campaign)}"
