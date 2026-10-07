"""Candidate model pipelines. Each takes the raw lead DataFrame and outputs P(subscribe).

All candidates train *unweighted* on log-loss so their probabilities stay honest
(the legacy class-balanced models over-stated probabilities 2-3x). Class imbalance is
handled where it belongs — in the decision thresholds (policy.py), not in the loss.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from leadintel.config import SEED
from leadintel.features import linear_encoder, tree_encoder

# Hyper-parameters chosen by the 5-fold CV grid in experiments/tuning.py (log-loss);
# results are in reports/experiments/tuning.json.
HGB_PARAMS: dict[str, Any] = {
    "learning_rate": 0.03,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
}
RF_PARAMS: dict[str, Any] = {"n_estimators": 300, "min_samples_leaf": 10, "max_features": 0.5}
LOGREG_PARAMS: dict[str, Any] = {"C": 0.3}
# A fixed iteration count keeps boosting fully deterministic. (Typed as a dict because
# sklearn has no stubs and pyright would infer `early_stopping: str` from its default.)
NO_EARLY_STOP: dict[str, Any] = {"early_stopping": False}


def _logreg(cols: list[str], balanced: bool = False) -> Pipeline:
    clf = LogisticRegression(
        max_iter=5000, class_weight="balanced" if balanced else None, **LOGREG_PARAMS
    )
    return Pipeline([("prep", linear_encoder(cols)), ("clf", clf)])


def _tree(cols: list[str], balanced: bool = False) -> Pipeline:
    # Shallow on purpose: the point of this candidate is a tree a human can read.
    clf = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=100,
        class_weight="balanced" if balanced else None,
        random_state=SEED,
    )
    return Pipeline([("prep", tree_encoder(cols)), ("clf", clf)])


def _forest(cols: list[str], balanced: bool = False) -> Pipeline:
    clf = RandomForestClassifier(
        **RF_PARAMS,
        class_weight="balanced" if balanced else None,
        n_jobs=-1,
        random_state=SEED,
    )
    return Pipeline([("prep", tree_encoder(cols)), ("clf", clf)])


def _boosting(cols: list[str], balanced: bool = False) -> Pipeline:
    clf = HistGradientBoostingClassifier(
        **HGB_PARAMS,
        class_weight="balanced" if balanced else None,
        **NO_EARLY_STOP,  # fixed iteration count -> fully deterministic
        random_state=SEED,
    )
    return Pipeline([("prep", tree_encoder(cols)), ("clf", clf)])


def _mlp(cols: list[str], balanced: bool = False) -> Pipeline:
    # The appropriate neural baseline for tabular data: an MLP on one-hot/scaled inputs
    # (no fake "spatial" structure, unlike the legacy 1D CNN). sklearn's MLP has no
    # class weights, so `balanced` is ignored.
    del balanced
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=1e-3,
        learning_rate_init=1e-3,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        max_iter=300,
        random_state=SEED,
    )
    return Pipeline([("prep", linear_encoder(cols)), ("clf", clf)])


CANDIDATES: dict[str, Callable[..., Pipeline]] = {
    "logistic_regression": _logreg,
    "decision_tree": _tree,
    "random_forest": _forest,
    "gradient_boosting": _boosting,
    "mlp": _mlp,
}

DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree (depth 5)",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting (HistGB)",
    "mlp": "Neural network (MLP)",
    "cnn_1d": "1D CNN (legacy architecture)",
}

FINAL_MODEL = "gradient_boosting"


def build(name: str, cols: list[str], balanced: bool = False) -> Pipeline:
    return CANDIDATES[name](cols, balanced=balanced)


def classifier(pipe: Pipeline) -> ClassifierMixin:
    return pipe.named_steps["clf"]
