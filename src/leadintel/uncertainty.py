"""Model (epistemic) uncertainty from a bootstrap ensemble.

Each ensemble member is the same model re-trained on a bootstrap resample of the training
data. How much they disagree about a lead says how sensitive its estimate is to *which*
historical leads we happened to train on.

Members trained on resamples are systematically a little less extreme than the full
model, so raw member quantiles can exclude the very estimate we display (e.g. "88 %,
range 83-88 %"). We therefore centre the members' spread on the main estimate:
    [p - (median - q_low), p + (q_high - median)]
"""

from __future__ import annotations

import numpy as np


def centered_interval(
    p: np.ndarray, member_preds: np.ndarray, quantiles: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """`member_preds` has shape [members, leads]; returns (low, high) arrays, clipped to [0, 1]."""
    q_lo, med, q_hi = np.quantile(member_preds, [quantiles[0], 0.5, quantiles[1]], axis=0)
    low = np.clip(p - (med - q_lo), 0.0, 1.0)
    high = np.clip(p + (q_hi - med), 0.0, 1.0)
    return low, high
