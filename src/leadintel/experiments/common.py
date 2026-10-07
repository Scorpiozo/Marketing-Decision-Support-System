"""Shared splits, cross-validation and report writing for all experiments."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from leadintel.config import CV_FOLDS, REPORTS_DIR, SEED, TEST_SIZE
from leadintel.data import load_dataset

EXPERIMENTS_DIR = REPORTS_DIR / "experiments"


@dataclass(frozen=True)
class Split:
    dev: pd.DataFrame  # 80 %: cross-validation, tuning, policy thresholds
    test: pd.DataFrame  # 20 %: touched once for final numbers; becomes the app's portfolio


def random_split(df: pd.DataFrame | None = None) -> Split:
    df = load_dataset() if df is None else df
    dev, test = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(df, test_size=TEST_SIZE, stratify=df["y"], random_state=SEED),
    )
    return Split(dev=dev.sort_values("row_order"), test=test.sort_values("row_order"))


def temporal_split(df: pd.DataFrame | None = None) -> Split:
    """Train on the earliest 80 % of contacts, test on the latest 20 % (out-of-time)."""
    df = load_dataset() if df is None else df
    cut = int(len(df) * (1 - TEST_SIZE))
    return Split(dev=df.iloc[:cut], test=df.iloc[cut:])


def oof_predict(
    make: Callable[[], Pipeline], X: pd.DataFrame, y: pd.Series, folds: int = CV_FOLDS
) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold probabilities for every dev row + per-fold fit seconds."""
    oof = np.zeros(len(X))
    fit_secs: list[float] = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for tr, va in skf.split(X, y):
        model = cast(Pipeline, clone(make()))
        t0 = time.perf_counter()
        model.fit(X.iloc[tr], y.iloc[tr])
        fit_secs.append(time.perf_counter() - t0)
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    return oof, fit_secs


def fold_scores(
    oof: np.ndarray, y: pd.Series, metric: Callable[[Any, Any], float], folds: int = CV_FOLDS
) -> tuple[float, float]:
    """Mean and std of a metric across the same CV folds used for `oof`."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    vals = [metric(y.iloc[va], oof[va]) for _, va in skf.split(oof, y)]
    return float(np.mean(vals)), float(np.std(vals))


def write_report(name: str, payload: dict[str, Any]) -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_jsonable))
    return path


def _jsonable(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)
