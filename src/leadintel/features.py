"""Feature groups and the two encoders used by the model families.

Feature groups are the backbone of the ablation study (does interaction history add value
over profile + campaign context?). Every feature here is a raw UCI column or a transparent
re-expression of one — nothing is invented.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from leadintel.schema import CATEGORY_LEVELS

PROFILE = ["age", "job", "marital", "education", "default", "balance", "housing", "loan"]
CAMPAIGN_CONTEXT = ["contact", "day", "month"]
INTERACTION = ["campaign", "pdays", "previous", "poutcome"]
LEAKY = ["duration"]  # post-outcome; only ever used in the labelled leakage benchmark

FeatureSet = Literal["profile", "profile+context", "profile+interaction", "full", "full+duration"]

FEATURE_SETS: dict[str, list[str]] = {
    "profile": PROFILE,
    "profile+context": PROFILE + CAMPAIGN_CONTEXT,
    "profile+interaction": PROFILE + INTERACTION,
    "full": PROFILE + CAMPAIGN_CONTEXT + INTERACTION,
    "full+duration": PROFILE + CAMPAIGN_CONTEXT + INTERACTION + LEAKY,
}
MODEL_FEATURES: list[str] = FEATURE_SETS["full"]

FEATURE_GROUP_OF: dict[str, str] = {
    **{f: "Customer profile" for f in PROFILE},
    **{f: "Campaign context" for f in CAMPAIGN_CONTEXT},
    **{f: "Interaction history" for f in INTERACTION},
    **{f: "Leakage (post-call)" for f in LEAKY},
}

FEATURE_LABELS: dict[str, str] = {
    "age": "Age",
    "job": "Occupation",
    "marital": "Marital status",
    "education": "Education",
    "default": "Credit in default",
    "balance": "Account balance",
    "housing": "Housing loan",
    "loan": "Personal loan",
    "contact": "Contact channel",
    "day": "Day of month",
    "month": "Contact month",
    "campaign": "Contacts this campaign",
    "pdays": "Days since previous contact",
    "previous": "Contacts before this campaign",
    "poutcome": "Previous campaign outcome",
    "duration": "Last call duration",
}


def categorical(cols: list[str]) -> list[str]:
    return [c for c in cols if c in CATEGORY_LEVELS]


def numeric(cols: list[str]) -> list[str]:
    return [c for c in cols if c not in CATEGORY_LEVELS]


def _pdays_as_missing(x: pd.DataFrame) -> pd.DataFrame:
    # pdays = -1 is a sentinel for "never contacted", not a number of days. Turning it into
    # NaN lets each model handle it properly instead of treating -1 as "very recent".
    out = x.copy()
    if "pdays" in out:
        out["pdays"] = out["pdays"].where(out["pdays"] >= 0, np.nan)
    return out


class TreeEncoder(TransformerMixin, BaseEstimator):
    """Encode raw lead columns for tree models: one numeric column per raw feature.

    Categoricals become fixed ordinal codes (vocabulary from the schema, so unseen levels
    become NaN rather than crashing); ``pdays = -1`` becomes NaN. Tree models split on
    ordinal codes fine, and — unlike scikit-learn's native categorical splits — this keeps
    SHAP's TreeExplainer exact (verified by an additivity test). Keeping one model column
    per raw column means every explanation is about a business-level feature.
    """

    def __init__(self, cols: list[str]) -> None:
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: object = None) -> TreeEncoder:
        del X, y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = _pdays_as_missing(X[self.cols])
        for c in categorical(self.cols):
            codes = {level: float(i) for i, level in enumerate(CATEGORY_LEVELS[c])}
            out[c] = out[c].map(codes).astype(float)
        return out.astype(float)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        del input_features
        return np.asarray(self.cols, dtype=object)


def tree_encoder(cols: list[str]) -> TreeEncoder:
    return TreeEncoder(list(cols))


def linear_encoder(cols: list[str]) -> Pipeline:
    """One-hot categoricals; impute + flag missing pdays; scale numerics; log heavy tails."""
    nums = numeric(cols)
    heavy = [c for c in nums if c in {"campaign", "previous", "duration"}]
    plain = [c for c in nums if c not in heavy and c != "pdays"]
    # Fixed vocabulary from the schema (typed Any: sklearn has no stubs for `categories`).
    vocab: Any = [list(CATEGORY_LEVELS[c]) for c in categorical(cols)]
    parts: list[tuple[str, object, list[str]]] = [
        (
            "cat",
            OneHotEncoder(
                categories=vocab,
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical(cols),
        ),
        ("num", StandardScaler(), plain),
        (
            "heavy",
            Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]),
            heavy,
        ),
    ]
    if "pdays" in cols:
        parts.append(
            (
                "pdays",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["pdays"],
            )
        )
    return Pipeline(
        [
            ("sentinel", FunctionTransformer(_pdays_as_missing, feature_names_out="one-to-one")),
            ("encode", ColumnTransformer(parts)),
        ]
    )
