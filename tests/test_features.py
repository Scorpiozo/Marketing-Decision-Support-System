from __future__ import annotations

import numpy as np
import pandas as pd

from leadintel.features import FEATURE_SETS, MODEL_FEATURES, linear_encoder, tree_encoder


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "age": 30,
                "job": "student",
                "marital": "single",
                "education": "tertiary",
                "default": "no",
                "balance": 10,
                "housing": "no",
                "loan": "no",
                "contact": "cellular",
                "day": 1,
                "month": "mar",
                "campaign": 1,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown",
            },
            {
                "age": 60,
                "job": "retired",
                "marital": "married",
                "education": "primary",
                "default": "no",
                "balance": 9000,
                "housing": "yes",
                "loan": "no",
                "contact": "telephone",
                "day": 30,
                "month": "oct",
                "campaign": 5,
                "pdays": 90,
                "previous": 3,
                "poutcome": "success",
            },
        ]
    )


def test_deployed_features_exclude_duration() -> None:
    # Regression: `duration` leaked the outcome in the legacy model.
    assert "duration" not in MODEL_FEATURES
    assert "duration" in FEATURE_SETS["full+duration"]


def test_tree_encoder_one_column_per_raw_feature() -> None:
    out = tree_encoder(MODEL_FEATURES).fit(_rows()).transform(_rows())
    assert list(out.columns) == MODEL_FEATURES
    assert pd.isna(out.loc[0, "pdays"])  # -1 sentinel -> missing, not "-1 days"
    assert out.loc[1, "pdays"] == 90


def test_tree_encoder_unknown_level_becomes_nan_not_crash() -> None:
    rows = _rows().assign(job=["student", "astronaut"])
    out = tree_encoder(MODEL_FEATURES).transform(rows)
    assert pd.isna(out.loc[1, "job"])


def test_linear_encoder_is_finite() -> None:
    out = np.asarray(linear_encoder(MODEL_FEATURES).fit_transform(_rows()))
    assert np.isfinite(out).all()
