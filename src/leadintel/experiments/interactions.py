"""Experiment 4 — do interaction-pattern variables *interact*?

Three complementary tests, so the narrative is earned rather than asserted:
  1. Additive vs. full boosting: the same GB model with `interaction_cst="no_interactions"`
     (each feature acts alone) vs. unconstrained. The gap is the predictive value of all
     feature interactions combined.
  2. SHAP interaction values (TreeSHAP) on held-out leads: which pairs carry the most
     interaction effect, relative to the main effects.
  3. Observed conversion tables for the hypothesised pairs (dev split only), with counts,
     so a human can see the pattern directly.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from leadintel.config import SEED
from leadintel.experiments.common import oof_predict, r, random_split, write_report
from leadintel.features import FEATURE_LABELS, MODEL_FEATURES, tree_encoder
from leadintel.models import HGB_PARAMS, NO_EARLY_STOP, build
from leadintel.patterns import history_label, intensity_label

PAIRS = [
    ("campaign", "poutcome"),
    ("campaign", "contact"),
    ("pdays", "poutcome"),
    ("previous", "poutcome"),
    ("campaign", "previous"),
]


def _additive() -> Pipeline:
    clf = HistGradientBoostingClassifier(
        **HGB_PARAMS, interaction_cst="no_interactions", **NO_EARLY_STOP, random_state=SEED
    )
    return Pipeline([("prep", tree_encoder(MODEL_FEATURES)), ("clf", clf)])


def main() -> dict[str, Any]:
    split = random_split()
    Xd, yd = split.dev[MODEL_FEATURES], split.dev["y"]

    # 1. additive vs full
    test1 = {}
    for label, make in [
        ("additive (no interactions)", _additive),
        ("full (interactions allowed)", lambda: build("gradient_boosting", MODEL_FEATURES)),
    ]:
        oof, _ = oof_predict(make, Xd, yd)
        test1[label] = {
            "cv_roc_auc": r(roc_auc_score(yd, oof)),
            "cv_pr_auc": r(average_precision_score(yd, oof)),
            "cv_log_loss": r(log_loss(yd, oof)),
        }
        logger.info(f"{label}: {test1[label]}")

    # 2. SHAP interaction values on a sample of held-out leads
    model = build("gradient_boosting", MODEL_FEATURES).fit(Xd, yd)
    sample = split.test[MODEL_FEATURES].sample(1500, random_state=SEED)
    enc = model.named_steps["prep"].transform(sample)
    inter = np.asarray(shap.TreeExplainer(model.named_steps["clf"]).shap_interaction_values(enc))
    mean_abs = np.abs(inter).mean(axis=0)  # [f, f]
    main_effects = {MODEL_FEATURES[i]: r(mean_abs[i, i]) for i in range(len(MODEL_FEATURES))}
    pairs = []
    for i, j in combinations(range(len(MODEL_FEATURES)), 2):
        pairs.append(
            {
                "pair": [MODEL_FEATURES[i], MODEL_FEATURES[j]],
                "label": f"{FEATURE_LABELS[MODEL_FEATURES[i]]} × "
                f"{FEATURE_LABELS[MODEL_FEATURES[j]]}",
                # off-diagonal is split symmetrically, so the pair's total is 2x one cell
                "mean_abs_interaction": r(2 * mean_abs[i, j]),
            }
        )
    pairs.sort(key=lambda p: -p["mean_abs_interaction"])
    total_main = float(np.trace(mean_abs))
    total_inter = float(mean_abs.sum() - np.trace(mean_abs))

    # 3. observed conversion tables (dev only)
    dev = split.dev.copy()
    dev["history"] = [
        history_label(p, o) for p, o in zip(dev["previous"], dev["poutcome"], strict=True)
    ]
    dev["intensity"] = dev["campaign"].map(intensity_label)
    tables = {}
    for a, b in [("history", "intensity"), ("history", "contact")]:
        g = dev.groupby([a, b], observed=True)["y"].agg(["mean", "size"]).reset_index()
        tables[f"{a}_x_{b}"] = [
            {a: row[a], b: row[b], "conversion_rate": r(row["mean"]), "n": int(row["size"])}
            for _, row in g.iterrows()
        ]

    payload = {
        "additive_vs_full": test1,
        "shap_interaction": {
            "sample_size": len(sample),
            "share_of_attribution_from_interactions": r(total_inter / (total_main + total_inter)),
            "main_effects_mean_abs": main_effects,
            "top_pairs": pairs[:12],
            "hypothesised_pairs": [
                p for p in pairs if tuple(p["pair"]) in PAIRS or tuple(p["pair"][::-1]) in PAIRS
            ],
        },
        "observed_tables": tables,
    }
    write_report("interactions", payload)
    return payload


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
