"""Small, honest hyper-parameter search (5-fold CV log-loss on the dev split only).

Deliberately modest: the research question is about *features and decisions*, not about
squeezing the last 0.001 AUC. The chosen values are copied into models.py.
"""

from __future__ import annotations

import itertools
from typing import Any

from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from leadintel.config import SEED
from leadintel.experiments.common import oof_predict, r, random_split, write_report
from leadintel.features import MODEL_FEATURES, linear_encoder, tree_encoder
from leadintel.models import NO_EARLY_STOP

GRIDS: dict[str, dict[str, list[Any]]] = {
    "gradient_boosting": {
        "learning_rate": [0.03, 0.05, 0.1],
        "max_leaf_nodes": [15, 31],
        "min_samples_leaf": [40],
        "l2_regularization": [1.0],
        "max_iter": [300],
    },
    "random_forest": {"min_samples_leaf": [3, 10, 30], "max_features": ["sqrt", 0.5]},
    "logistic_regression": {"C": [0.03, 0.3, 3.0]},
}


def _make(name: str, params: dict[str, Any]) -> Pipeline:
    cols = MODEL_FEATURES
    if name == "gradient_boosting":
        clf = HistGradientBoostingClassifier(**params, **NO_EARLY_STOP, random_state=SEED)
        return Pipeline([("prep", tree_encoder(cols)), ("clf", clf)])
    if name == "random_forest":
        clf = RandomForestClassifier(300, **params, n_jobs=-1, random_state=SEED)
        return Pipeline([("prep", tree_encoder(cols)), ("clf", clf)])
    clf = LogisticRegression(max_iter=5000, **params)
    return Pipeline([("prep", linear_encoder(cols)), ("clf", clf)])


def main() -> dict[str, Any]:
    split = random_split()
    X, y = split.dev[MODEL_FEATURES], split.dev["y"]
    results: dict[str, Any] = {}
    for name, grid in GRIDS.items():
        rows = []
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid.keys(), values, strict=True))
            oof, _ = oof_predict(lambda n=name, p=params: _make(n, p), X, y)
            rows.append(
                {
                    "params": params,
                    "log_loss": r(log_loss(y, oof)),
                    "roc_auc": r(roc_auc_score(y, oof)),
                }
            )
            logger.info(f"{name} {params} -> {rows[-1]['log_loss']} / {rows[-1]['roc_auc']}")
        rows.sort(key=lambda row: row["log_loss"])
        results[name] = {"best": rows[0], "all": rows}
    write_report("tuning", results)
    return results


if __name__ == "__main__":
    main()
