"""
Model training, evaluation and persistence.

Exactly two predictive models are supported:

* ``gradient_boosting`` - scikit-learn GradientBoostingClassifier
* ``random_forest``     - scikit-learn RandomForestClassifier

Both are trained on the same preprocessed feature matrix (see
src/preprocessing.py and models/feature_order.json) and both use balanced
class weighting, so their probabilities sit on a comparable scale for the
fixed HOT / WARM / COLD thresholds in src/prediction.py.
"""

import json
import os

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from src.config import MODELS_DIR, PRODUCTION_MODEL_PATH

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.5


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    # GradientBoostingClassifier has no class_weight argument; balanced
    # sample weights give the same ~88/12 imbalance handling used by the
    # Random Forest.
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.9,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    return model


# Single source of truth for the supported models.
MODEL_REGISTRY = {
    "gradient_boosting": {
        "display_name": "Gradient Boosting",
        "filename": "gradient_boosting.joblib",
        "trainer": train_gradient_boosting,
    },
    "random_forest": {
        "display_name": "Random Forest",
        "filename": "random_forest.joblib",
        "trainer": train_random_forest,
    },
}


def model_path(model_key, models_dir=MODELS_DIR):
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_key}'. Supported models: {sorted(MODEL_REGISTRY)}"
        )
    return os.path.join(models_dir, MODEL_REGISTRY[model_key]["filename"])


def display_name(model_key):
    return MODEL_REGISTRY[model_key]["display_name"]


def predict_proba(model, X):
    """Return P(y=1) for each row of the preprocessed matrix X."""
    return model.predict_proba(np.asarray(X))[:, 1]


# Evaluation

def evaluate_model(y_true, y_pred, y_prob=None, model_name="model"):
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        metrics["PR-AUC"] = average_precision_score(y_true, y_prob)
    return metrics


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


# Persistence

def save_model(model, model_key, models_dir=MODELS_DIR):
    path = model_path(model_key, models_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(model_key, models_dir=MODELS_DIR):
    path = model_path(model_key, models_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'. Run `python train_and_evaluate.py` first."
        )
    return joblib.load(path)


def save_production_model_info(model_key, selected_by, metrics, path=PRODUCTION_MODEL_PATH):
    payload = {
        "model_key": model_key,
        "display_name": display_name(model_key),
        "filename": MODEL_REGISTRY[model_key]["filename"],
        "selected_by": selected_by,
        "test_metrics": {k: (round(float(v), 4) if k != "Model" else v) for k, v in metrics.items()},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def load_production_model_info(path=PRODUCTION_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Production model metadata not found at '{path}'. "
            "Run `python train_and_evaluate.py` first."
        )
    with open(path) as f:
        info = json.load(f)
    if info.get("model_key") not in MODEL_REGISTRY:
        raise ValueError(f"production_model.json names an unsupported model: {info.get('model_key')!r}")
    return info


def load_production_model(models_dir=MODELS_DIR, info_path=PRODUCTION_MODEL_PATH):
    """Return (model, info) for the model used to generate decision outputs."""
    info = load_production_model_info(info_path)
    return load_model(info["model_key"], models_dir), info


# Evaluation details for the API / reports

def roc_points(y_true, y_prob, max_points=120):
    """ROC curve downsampled to at most ``max_points`` (fpr, tpr, threshold) rows."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y_true, y_prob)
    if len(fpr) > max_points:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).round().astype(int))
        fpr, tpr, thr = fpr[idx], tpr[idx], thr[idx]
    thr = np.clip(thr, 0.0, 1.0)  # sklearn's first threshold is +inf
    return [[round(float(a), 4), round(float(b), 4), round(float(c), 4)] for a, b, c in zip(fpr, tpr, thr)]


def grouped_feature_importances(model, fitted_preprocessor):
    """Sum one-hot importances back onto their original input column.

    Returns [{"feature": name, "importance": share}] sorted descending; the
    shares sum to 1.
    """
    from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    importances = np.asarray(model.feature_importances_, dtype=float)
    encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
    owners = list(NUMERIC_FEATURES)
    for col, cats in zip(CATEGORICAL_FEATURES, encoder.categories_):
        owners.extend([col] * len(cats))
    if len(owners) != len(importances):
        raise ValueError("Feature importances do not line up with the preprocessor's features.")
    totals = {}
    for owner, value in zip(owners, importances):
        totals[owner] = totals.get(owner, 0.0) + value
    total = sum(totals.values()) or 1.0
    rows = [{"feature": k, "importance": round(v / total, 4)} for k, v in totals.items()]
    return sorted(rows, key=lambda r: r["importance"], reverse=True)
