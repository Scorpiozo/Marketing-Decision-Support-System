"""Descriptive statistics over the scored portfolio, used by the dashboard charts.

Everything here is a plain group-by on observed outcomes or a ranking curve on the
model's scores; no extra modelling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from leadintel.patterns import HISTORY_ORDER

AGE_BANDS = ([17, 29, 39, 49, 59, 120], ["18–29", "30–39", "40–49", "50–59", "60+"])
CONTACT_BINS = ([0, 1, 2, 3, 4, 5, 7, 1000], ["1", "2", "3", "4", "5", "6–7", "8+"])


def _rates(key: pd.Series, y: pd.Series, order: list[str]) -> list[dict[str, Any]]:
    g = pd.DataFrame({"k": key.astype(str).to_numpy(), "y": y.to_numpy()}).groupby("k")["y"]
    mean, size = g.mean(), g.size()
    return [
        {"label": k, "rate": float(mean[k]), "n": int(size[k])} for k in order if k in mean.index
    ]


def gains_curve(probability: pd.Series, y: pd.Series, points: int = 40) -> list[list[float]]:
    """Share of all subscribers reached when contacting the top-x % of leads by score."""
    order = np.argsort(-probability.to_numpy(), kind="stable")
    hits = np.cumsum(y.to_numpy()[order]) / max(int(y.sum()), 1)
    n = len(order)
    out = [[0.0, 0.0]]
    for i in range(1, points + 1):
        k = round(i / points * n)
        out.append([round(k / n, 4), round(float(hits[k - 1]), 4)])
    return out


def compute(portfolio: pd.DataFrame) -> dict[str, Any]:
    y = portfolio["y"]
    return {
        "base_rate": float(y.mean()),
        "by_contacts": _rates(
            pd.cut(portfolio["campaign"], CONTACT_BINS[0], labels=CONTACT_BINS[1]),
            y,
            CONTACT_BINS[1],
        ),
        "by_history": _rates(portfolio["history"], y, list(HISTORY_ORDER)),
        "by_age": _rates(
            pd.cut(portfolio["age"], AGE_BANDS[0], labels=AGE_BANDS[1]), y, AGE_BANDS[1]
        ),
        "gains": gains_curve(portfolio["probability"], y),
    }
