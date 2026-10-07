"""Metrics for an imbalanced, decision-oriented problem.

Accuracy alone is misleading at an 11.7 % base rate (predicting "no" for everyone scores
88 %), so we report ranking quality (ROC-AUC, PR-AUC, lift), probability quality
(Brier, log-loss, calibration error) and threshold metrics at an explicit threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

Floats = NDArray[np.float64]


def expected_calibration_error(y: ArrayLike, p: ArrayLike, bins: int = 10) -> float:
    """Weighted mean |observed rate - mean prediction| over equal-frequency bins."""
    y_arr, p_arr = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p_arr)
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) * abs(y_arr[chunk].mean() - p_arr[chunk].mean())
    return float(err / len(p_arr))


def lift_at(y: ArrayLike, p: ArrayLike, fraction: float) -> dict[str, float]:
    """Conversion rate and share of all conversions captured in the top `fraction` by score."""
    y_arr, p_arr = np.asarray(y, float), np.asarray(p, float)
    k = max(1, round(fraction * len(p_arr)))
    top = np.argsort(-p_arr, kind="stable")[:k]
    rate = y_arr[top].mean()
    return {
        "fraction": fraction,
        "conversion_rate": float(rate),
        "lift": float(rate / y_arr.mean()),
        "recall": float(y_arr[top].sum() / y_arr.sum()),
    }


def threshold_metrics(y: ArrayLike, p: ArrayLike, threshold: float) -> dict[str, Any]:
    y_arr = np.asarray(y)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, pred, labels=[0, 1]).ravel()
    # sklearn ships no type stubs; pyright infers `zero_division: str` from its default.
    zd: Any = 0
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_arr, pred)),
        "precision": float(precision_score(y_arr, pred, zero_division=zd)),
        "recall": float(recall_score(y_arr, pred, zero_division=zd)),
        "f1": float(f1_score(y_arr, pred, zero_division=zd)),
        "flagged_share": float(pred.mean()),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def score_report(y: ArrayLike, p: ArrayLike, threshold: float) -> dict[str, Any]:
    y_arr, p_arr = np.asarray(y), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {
        "n": len(y_arr),
        "base_rate": float(y_arr.mean()),
        "roc_auc": float(roc_auc_score(y_arr, p_arr)),
        "pr_auc": float(average_precision_score(y_arr, p_arr)),
        "brier": float(brier_score_loss(y_arr, p_arr)),
        "log_loss": float(log_loss(y_arr, p_arr)),
        "ece": expected_calibration_error(y_arr, p_arr),
        "mean_predicted": float(p_arr.mean()),
        "lift_top10": lift_at(y_arr, p_arr, 0.10),
        "lift_top20": lift_at(y_arr, p_arr, 0.20),
        "at_threshold": threshold_metrics(y_arr, p_arr, threshold),
    }


def best_f1_threshold(y: ArrayLike, p: ArrayLike) -> float:
    prec, rec, thr = precision_recall_curve(np.asarray(y), np.asarray(p))
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def _downsample(xs: Floats, ys: Floats, n: int = 120) -> list[list[float]]:
    idx = np.unique(np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int))
    return [[round(float(xs[i]), 4), round(float(ys[i]), 4)] for i in idx]


def curves(y: ArrayLike, p: ArrayLike, bins: int = 10) -> dict[str, Any]:
    """Compact curve data for charts (ROC, PR, reliability)."""
    y_arr, p_arr = np.asarray(y), np.asarray(p, float)
    fpr, tpr, _ = roc_curve(y_arr, p_arr)
    prec, rec, _ = precision_recall_curve(y_arr, p_arr)
    order = np.argsort(p_arr)
    reliability = []
    for chunk in np.array_split(order, bins):
        reliability.append(
            {
                "mean_predicted": round(float(p_arr[chunk].mean()), 4),
                "observed": round(float(y_arr[chunk].mean()), 4),
                "n": len(chunk),
            }
        )
    return {
        "roc": _downsample(fpr, tpr),
        "pr": _downsample(rec[::-1], prec[::-1]),
        "reliability": reliability,
    }


def bootstrap_ci(
    y: ArrayLike, p: ArrayLike, metric: str = "roc_auc", n: int = 500, seed: int = 0
) -> tuple[float, float]:
    """95 % percentile bootstrap interval for ROC-AUC or PR-AUC on a fixed test set."""
    fn = roc_auc_score if metric == "roc_auc" else average_precision_score
    y_arr, p_arr = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        if y_arr[idx].min() != y_arr[idx].max():
            stats.append(fn(y_arr[idx], p_arr[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)
