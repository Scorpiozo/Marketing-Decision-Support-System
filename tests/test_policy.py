from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from leadintel.policy import Action, PolicyConfig, decide, decide_many

CFG = PolicyConfig(contact_cost=5, conversion_value=50, priority_threshold=0.35, heavy_contact=4)


@pytest.mark.parametrize(
    "p,interval,campaign,expected",
    [
        (0.60, (0.55, 0.65), 1, Action.PRIORITISE),
        (0.60, (0.55, 0.65), 9, Action.PRIORITISE),  # high value outranks fatigue
        (0.20, (0.17, 0.22), 1, Action.FOLLOW_UP),
        (0.20, (0.17, 0.22), 4, Action.CHANGE_APPROACH),
        (0.11, (0.08, 0.13), 1, Action.REVIEW),  # interval straddles break-even (0.10)
        (0.05, (0.04, 0.06), 1, Action.DEPRIORITISE),
        (0.05, (0.04, 0.06), 6, Action.STOP),
    ],
)
def test_rules(p: float, interval: tuple[float, float], campaign: int, expected: Action) -> None:
    assert decide(p, interval, campaign, CFG).action == expected


def test_break_even_is_cost_over_value() -> None:
    assert PolicyConfig(contact_cost=4, conversion_value=80).break_even == pytest.approx(0.05)


def test_reasons_mention_assumptions() -> None:
    d = decide(0.05, (0.04, 0.06), 6, CFG)
    assert any("assumes €50" in r for r in d.reasons)
    assert d.expected_value == pytest.approx(0.05 * 50 - 5)


def test_vectorised_policy_matches_scalar() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.8, 3000)
    width = rng.uniform(0, 0.1, 3000)
    lo, hi = np.clip(p - width, 0, 1), np.clip(p + width, 0, 1)
    camp = rng.integers(1, 10, 3000)
    many = decide_many(p, lo, hi, camp, CFG)
    for i in range(3000):
        assert many[i] == decide(p[i], (lo[i], hi[i]), int(camp[i]), CFG).action.value


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValidationError):
        PolicyConfig(contact_cost=-1)
