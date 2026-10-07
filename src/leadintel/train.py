"""Train the deployed model and write every artifact the web app needs.

    uv run leadintel-train

Outputs (artifacts/):
  model.joblib     final pipeline + bootstrap ensemble (uncertainty) + feature list
  portfolio.csv    held-out test leads: raw fields, outcome, probability, interval, SHAP
  model_card.json  data fingerprint, versions, metrics, curves, global explanations,
                   observed pattern rates, error analysis

The web app only *loads* these. Training never happens at app start-up.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from loguru import logger
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from leadintel.config import ARTIFACTS_DIR, DATA_PATH, N_BOOTSTRAP, SEED
from leadintel.data import file_sha256, load_dataset
from leadintel.evaluation import curves, score_report
from leadintel.experiments.common import random_split
from leadintel.explain import Explainer, global_importance
from leadintel.features import MODEL_FEATURES
from leadintel.models import DISPLAY_NAMES, FINAL_MODEL, build
from leadintel.patterns import HISTORY_ORDER, INTENSITY_ORDER, history_label, intensity_label
from leadintel.policy import PolicyConfig
from leadintel.uncertainty import centered_interval

# Middle half of the bootstrap models. Chosen on a validation split carved from the dev
# data (experiments/review_band.py), never on the test set: see that report.
INTERVAL_QUANTILES = (0.25, 0.75)


def fit_bootstrap(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> list[Any]:
    """Re-fit the classifier on bootstrap resamples (same encoder) -> epistemic spread."""
    encoded = pipeline.named_steps["prep"].transform(X)
    rng = np.random.default_rng(SEED)
    members = []
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(X), len(X))
        clf = clone(pipeline.named_steps["clf"]).set_params(random_state=SEED + i + 1)
        members.append(clf.fit(encoded.iloc[idx], y.iloc[idx]))
    return members


def bootstrap_interval(
    pipeline: Pipeline,
    members: list[Any],
    X: pd.DataFrame,
    p: np.ndarray,
    quantiles: tuple[float, float] = INTERVAL_QUANTILES,
) -> tuple[np.ndarray, np.ndarray]:
    encoded = pipeline.named_steps["prep"].transform(X)
    preds = np.stack([m.predict_proba(encoded)[:, 1] for m in members])
    return centered_interval(p, preds, quantiles)


def pattern_stats(dev: pd.DataFrame) -> list[dict[str, Any]]:
    d = dev.assign(
        history=[
            history_label(p, o) for p, o in zip(dev["previous"], dev["poutcome"], strict=True)
        ],
        intensity=dev["campaign"].map(intensity_label),
    )
    grouped = d.groupby(["history", "intensity"])["y"]
    rates, sizes = grouped.mean().to_dict(), grouped.size().to_dict()
    return [
        {
            "history": h,
            "intensity": i,
            "conversion_rate": round(float(rates[(h, i)]), 4),
            "n": int(sizes[(h, i)]),
        }
        for h in HISTORY_ORDER
        for i in INTENSITY_ORDER
        if (h, i) in rates
    ]


def error_analysis(portfolio: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Characterise false positives / negatives at the follow-up (break-even) line."""
    pred = portfolio["probability"] >= threshold
    groups = {
        "false_positives": portfolio[pred & (portfolio["y"] == 0)],
        "false_negatives": portfolio[~pred & (portfolio["y"] == 1)],
        "true_positives": portfolio[pred & (portfolio["y"] == 1)],
    }
    out: dict[str, Any] = {"threshold": threshold}
    for name, g in groups.items():
        out[name] = {
            "n": len(g),
            "mean_probability": round(float(g["probability"].mean()), 4) if len(g) else None,
            "share_new_to_campaigns": round(float((g["previous"] == 0).mean()), 3)
            if len(g)
            else None,
            "share_contact_unknown": round(float((g["contact"] == "unknown").mean()), 3)
            if len(g)
            else None,
            "median_contacts": float(g["campaign"].median()) if len(g) else None,
            "top_history": g["history"].value_counts().head(3).to_dict(),
            "examples": g.sort_values("probability", ascending=name == "false_negatives")
            .head(5)["lead_id"]
            .tolist(),
        }
    return out


def main() -> None:
    t0 = time.perf_counter()
    df = load_dataset()
    split = random_split(df)
    Xd, yd = split.dev[MODEL_FEATURES], split.dev["y"]
    test = split.test

    logger.info(f"Fitting {FINAL_MODEL} on {len(Xd):,} dev rows")
    pipeline = build(FINAL_MODEL, MODEL_FEATURES).fit(Xd, yd)
    members = fit_bootstrap(pipeline, Xd, yd)

    p = pipeline.predict_proba(test[MODEL_FEATURES])[:, 1]
    lo, hi = bootstrap_interval(pipeline, members, test[MODEL_FEATURES], p)
    explainer = Explainer(pipeline, MODEL_FEATURES)
    phis = explainer.shap_values(test[MODEL_FEATURES])

    # Guard against silently-wrong explanations (see journal: HistGB categorical bug).
    logits = np.log(p / (1 - p))
    additivity = float(np.abs(explainer.base_value + phis.sum(axis=1) - logits).max())
    if additivity > 1e-6:
        raise RuntimeError(f"SHAP values do not add up to the prediction (max err {additivity})")

    portfolio = test[["row_order", *MODEL_FEATURES, "y"]].copy()
    portfolio.insert(0, "lead_id", [f"L{n:05d}" for n in portfolio["row_order"]])
    portfolio["probability"] = p.round(5)
    portfolio["p_low"] = lo.round(5)
    portfolio["p_high"] = hi.round(5)
    portfolio["history"] = [
        history_label(a, b)
        for a, b in zip(portfolio["previous"], portfolio["poutcome"], strict=True)
    ]
    portfolio["intensity"] = portfolio["campaign"].map(intensity_label)
    for j, f in enumerate(MODEL_FEATURES):
        portfolio[f"shap_{f}"] = phis[:, j].round(5)

    report = score_report(test["y"], p, threshold=PolicyConfig().break_even)
    card = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": FINAL_MODEL,
        "model_display_name": DISPLAY_NAMES[FINAL_MODEL],
        "features": MODEL_FEATURES,
        "excluded_features": {"duration": "post-outcome leakage (known only after the call)"},
        "data": {
            "file": DATA_PATH.name,
            "sha256": file_sha256(DATA_PATH),
            "rows": len(df),
            "dev_rows": len(split.dev),
            "test_rows": len(test),
            "base_rate": round(float(df["y"].mean()), 4),
        },
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "seed": SEED,
        "bootstrap_members": N_BOOTSTRAP,
        "interval_quantiles": INTERVAL_QUANTILES,
        "shap_base_probability": round(1 / (1 + np.exp(-explainer.base_value)), 4),
        "shap_additivity_max_error": additivity,
        "test_metrics": report,
        "test_curves": curves(test["y"], p),
        "global_importance": global_importance(test[MODEL_FEATURES], phis, MODEL_FEATURES),
        "pattern_stats_dev": pattern_stats(split.dev),
        "error_analysis": error_analysis(portfolio, PolicyConfig().break_even),
    }

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "members": members, "features": MODEL_FEATURES},
        ARTIFACTS_DIR / "model.joblib",
        compress=3,
    )
    portfolio.to_csv(ARTIFACTS_DIR / "portfolio.csv", index=False)
    (ARTIFACTS_DIR / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    logger.info(
        f"Done in {time.perf_counter() - t0:.1f}s — test AUC {report['roc_auc']:.3f}, "
        f"PR-AUC {report['pr_auc']:.3f}, Brier {report['brier']:.4f}"
    )


if __name__ == "__main__":
    main()
