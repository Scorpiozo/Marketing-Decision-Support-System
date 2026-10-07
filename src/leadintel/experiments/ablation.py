"""Experiment 2 — feature-group ablation: do interaction-history signals add value?

Feature sets (see features.py):
  A  profile                 customer demographics + finances
  B  profile+context         + campaign context (channel, contact month/day)
  B' profile+interaction     + interaction history, *without* campaign context
  C  full                    profile + context + interaction history   (deployed)
  L  full+duration           C + last-call duration — leakage benchmark, NOT deployable

Measured on ranking (AUC/PR-AUC), probability quality (Brier, ECE) and decision value
(conversions captured when calling only the top 20 %), with 5-fold CV variability.
Repeated under an out-of-time split, where the dataset's drift makes things harder.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from leadintel.evaluation import bootstrap_ci, lift_at, score_report
from leadintel.experiments.common import (
    Split,
    fold_scores,
    oof_predict,
    r,
    random_split,
    temporal_split,
    write_report,
)
from leadintel.features import FEATURE_SETS
from leadintel.models import build

MODELS = ("logistic_regression", "gradient_boosting")


def _run(split: Split, with_cv: bool) -> list[dict[str, Any]]:
    rows = []
    for fs, cols in FEATURE_SETS.items():
        for name in MODELS:
            logger.info(f"ablation: {fs} / {name}")
            Xd, yd, Xt, yt = split.dev[cols], split.dev["y"], split.test[cols], split.test["y"]
            row: dict[str, Any] = {"feature_set": fs, "model": name, "n_features": len(cols)}
            if with_cv:
                oof, _ = oof_predict(lambda n=name, c=cols: build(n, c), Xd, yd)
                row["cv"] = {
                    "roc_auc": [r(v) for v in fold_scores(oof, yd, roc_auc_score)],
                    "pr_auc": [r(v) for v in fold_scores(oof, yd, average_precision_score)],
                    "brier": [r(v) for v in fold_scores(oof, yd, brier_score_loss)],
                    "top20_recall": r(lift_at(yd, oof, 0.2)["recall"]),
                }
            pt = build(name, cols).fit(Xd, yd).predict_proba(Xt)[:, 1]
            rep = score_report(yt, pt, threshold=0.5)
            test: dict[str, Any] = {
                k: r(rep[k]) for k in ("roc_auc", "pr_auc", "brier", "ece", "mean_predicted")
            }
            test["base_rate"] = r(rep["base_rate"])
            test["top20_recall"] = r(rep["lift_top20"]["recall"])
            test["top10_conversion"] = r(rep["lift_top10"]["conversion_rate"])
            test["roc_auc_ci95"] = [r(v) for v in bootstrap_ci(yt, pt, "roc_auc", n=300)]
            row["test"] = test
            rows.append(row)
    return rows


def main() -> dict[str, Any]:
    payload = {
        "random_split": _run(random_split(), with_cv=True),
        "temporal_split": _run(temporal_split(), with_cv=False),
    }
    write_report("ablation", payload)
    return payload


if __name__ == "__main__":
    main()
