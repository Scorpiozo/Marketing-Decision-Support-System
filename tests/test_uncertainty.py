from __future__ import annotations

import numpy as np

from leadintel.uncertainty import centered_interval


def test_interval_always_contains_estimate() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 500)
    members = np.clip(p * 0.9 + rng.normal(0, 0.05, (15, 500)), 0, 1)  # biased, noisy members
    lo, hi = centered_interval(p, members, (0.25, 0.75))
    assert ((lo <= p) & (p <= hi)).all()
    assert ((lo >= 0) & (hi <= 1)).all()
