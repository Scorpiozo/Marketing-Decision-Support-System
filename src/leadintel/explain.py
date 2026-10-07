"""Explanations for the gradient-boosting model via exact TreeSHAP.

SHAP values are in log-odds and add up exactly to the model's output:
    logit(p) = base_value + sum(contributions)
so each contribution can be read as an *odds multiplier* exp(phi): "this factor
multiplied the odds of subscribing by 2.4". That reading is exact, unlike percentage-point
splits of a probability, which are not additive.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
import shap
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from leadintel.features import FEATURE_GROUP_OF, FEATURE_LABELS

# Features that describe *when/how* the campaign reached the lead. In this dataset they
# partly encode the collection period (drift), so they are explained but never suggested
# as levers in the what-if simulator or the narrative.
CONTEXT_CAVEAT = {"contact", "month", "day"}

Strength = Literal["strong", "moderate", "slight"]


class Contribution(BaseModel):
    feature: str
    label: str
    group: str
    value: str  # human-readable raw value
    log_odds: float
    odds_multiplier: float
    direction: Literal["up", "down"]
    strength: Strength
    is_context: bool


class Explanation(BaseModel):
    base_probability: float  # model output for an "average" lead (expected value)
    probability: float
    contributions: list[Contribution]  # sorted by |log_odds|, descending

    @property
    def drivers_up(self) -> list[Contribution]:
        return [c for c in self.contributions if c.direction == "up" and c.strength != "slight"]

    @property
    def drivers_down(self) -> list[Contribution]:
        return [c for c in self.contributions if c.direction == "down" and c.strength != "slight"]


def _strength(phi: float) -> Strength:
    # |phi| >= 0.5 => odds x1.65 or more; >= 0.2 => x1.22 or more.
    a = abs(phi)
    return "strong" if a >= 0.5 else "moderate" if a >= 0.2 else "slight"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def display_value(feature: str, raw: object) -> str:
    if feature == "pdays":
        return "never contacted" if int(raw) == -1 else f"{int(raw)} days ago"  # type: ignore[arg-type]
    if feature == "balance":
        return f"€{int(raw):,}"  # type: ignore[arg-type]
    if feature in {"campaign", "previous"}:
        n = int(raw)  # type: ignore[arg-type]
        return f"{n} contact" + ("" if n == 1 else "s")
    return str(raw)


class Explainer:
    """Wraps a fitted (TreeEncoder -> HistGradientBoosting) pipeline. Build once, reuse."""

    def __init__(self, pipeline: Pipeline, features: list[str]) -> None:
        self.prep = pipeline.named_steps["prep"]
        self.tree = shap.TreeExplainer(pipeline.named_steps["clf"])
        self.features = features
        self.base_value = float(np.atleast_1d(self.tree.expected_value)[-1])

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        encoded = self.prep.transform(X[self.features])
        values = np.asarray(self.tree.shap_values(encoded))
        return values[..., 1] if values.ndim == 3 else values

    def explain_row(self, row: pd.Series, phi: np.ndarray) -> Explanation:
        logit = self.base_value + float(phi.sum())
        contributions = [
            Contribution(
                feature=f,
                label=FEATURE_LABELS[f],
                group=FEATURE_GROUP_OF[f],
                value=display_value(f, row[f]),
                log_odds=round(float(v), 4),
                odds_multiplier=round(math.exp(float(v)), 3),
                direction="up" if v >= 0 else "down",
                strength=_strength(float(v)),
                is_context=f in CONTEXT_CAVEAT,
            )
            for f, v in zip(self.features, phi, strict=True)
        ]
        contributions.sort(key=lambda c: -abs(c.log_odds))
        return Explanation(
            base_probability=round(_sigmoid(self.base_value), 4),
            probability=round(_sigmoid(logit), 4),
            contributions=contributions,
        )

    def explain(self, X: pd.DataFrame) -> list[Explanation]:
        phis = self.shap_values(X)
        return [self.explain_row(X.iloc[i], phis[i]) for i in range(len(X))]


def global_importance(
    X: pd.DataFrame, phis: np.ndarray, features: list[str]
) -> list[dict[str, object]]:
    """Mean |SHAP| per feature plus how its effect varies across its values."""
    out: list[dict[str, object]] = []
    for j, f in enumerate(features):
        col, phi = X[f], phis[:, j]
        if col.dtype.kind in "iuf" and f != "pdays" and col.nunique() > 12:
            bins = pd.qcut(col, q=6, duplicates="drop")
            grouped = pd.DataFrame({"bin": bins, "phi": phi}).groupby("bin", observed=True)["phi"]
            levels = [
                {
                    "value": f"{int(iv.left)}–{int(iv.right)}",
                    "mean_log_odds": round(float(m), 3),
                    "n": int(n),
                }
                for iv, m, n in zip(
                    grouped.mean().index, grouped.mean(), grouped.size(), strict=True
                )
            ]
        else:
            key = col.map(
                lambda v, f=f: (
                    display_value(f, v)
                    if f != "pdays"
                    else ("never contacted" if v == -1 else "contacted before")
                )
            )
            grouped = pd.DataFrame({"k": key, "phi": phi}).groupby("k")["phi"]
            levels = [
                {"value": str(k), "mean_log_odds": round(float(m), 3), "n": int(n)}
                for k, m, n in zip(
                    grouped.mean().index, grouped.mean(), grouped.size(), strict=True
                )
            ]
            levels.sort(key=lambda d: -float(d["mean_log_odds"]))  # type: ignore[arg-type]
        out.append(
            {
                "feature": f,
                "label": FEATURE_LABELS[f],
                "group": FEATURE_GROUP_OF[f],
                "mean_abs_log_odds": round(float(np.abs(phi).mean()), 4),
                "is_context": f in CONTEXT_CAVEAT,
                "levels": levels,
            }
        )
    out.sort(key=lambda d: -float(d["mean_abs_log_odds"]))  # type: ignore[arg-type]
    return out
