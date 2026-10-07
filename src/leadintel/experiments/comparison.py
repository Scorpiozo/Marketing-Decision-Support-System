"""Experiment 1 — model comparison on the deployable feature set (no `duration`).

For every candidate: 5-fold CV on the dev split (mean ± sd), one evaluation on the
held-out test split, fit time, single-row latency and serialized size. Also re-runs the
legacy class-balanced setting to quantify its effect on probability quality.
"""

from __future__ import annotations

import io
import time
from typing import Any

import joblib
import numpy as np
from loguru import logger
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from leadintel.evaluation import best_f1_threshold, bootstrap_ci, curves, score_report
from leadintel.experiments.common import fold_scores, oof_predict, r, random_split, write_report
from leadintel.features import MODEL_FEATURES
from leadintel.models import CANDIDATES, DISPLAY_NAMES, build


def _latency_ms(model: Any, row: Any, repeats: int = 200) -> float:
    model.predict_proba(row)  # warm-up
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.predict_proba(row)
    return (time.perf_counter() - t0) / repeats * 1000


def _size_kb(model: Any) -> float:
    buf = io.BytesIO()
    joblib.dump(model, buf, compress=3)
    return buf.tell() / 1024


def evaluate_candidate(name: str, balanced: bool = False) -> dict[str, Any]:
    split = random_split()
    Xd, yd = split.dev[MODEL_FEATURES], split.dev["y"]
    Xt, yt = split.test[MODEL_FEATURES], split.test["y"]

    oof, fit_secs = oof_predict(lambda: build(name, MODEL_FEATURES, balanced), Xd, yd)
    # A threshold is a decision, so it is chosen on dev (OOF) predictions, never on test.
    thr = best_f1_threshold(yd, oof)

    model = build(name, MODEL_FEATURES, balanced).fit(Xd, yd)
    pt = model.predict_proba(Xt)[:, 1]
    test = score_report(yt, pt, thr)
    auc_lo, auc_hi = bootstrap_ci(yt, pt, "roc_auc")
    pr_lo, pr_hi = bootstrap_ci(yt, pt, "pr_auc")
    return {
        "model": name,
        "display_name": DISPLAY_NAMES[name] + (" — balanced weights" if balanced else ""),
        "balanced": balanced,
        "cv": {
            "roc_auc": [r(v) for v in fold_scores(oof, yd, roc_auc_score)],
            "pr_auc": [r(v) for v in fold_scores(oof, yd, average_precision_score)],
            "brier": [r(v) for v in fold_scores(oof, yd, brier_score_loss)],
        },
        "f1_threshold_from_cv": r(thr),
        "test": test,
        "test_ci95": {"roc_auc": [r(auc_lo), r(auc_hi)], "pr_auc": [r(pr_lo), r(pr_hi)]},
        "test_curves": curves(yt, pt),
        "fit_seconds": r(float(np.mean(fit_secs)), 2),
        "latency_ms_single_row": r(_latency_ms(model, Xt.iloc[[0]]), 2),
        "size_kb": r(_size_kb(model), 1),
    }


def main() -> dict[str, Any]:
    results = []
    for name in CANDIDATES:
        logger.info(f"comparison: {name}")
        results.append(evaluate_candidate(name))
    for name in ("logistic_regression", "gradient_boosting"):
        logger.info(f"comparison: {name} (balanced)")
        results.append(evaluate_candidate(name, balanced=True))
    payload = {"feature_set": "full (no duration)", "results": results}
    write_report("comparison", payload)
    return payload


if __name__ == "__main__":
    main()
