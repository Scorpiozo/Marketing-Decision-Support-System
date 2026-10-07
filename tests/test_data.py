from __future__ import annotations

import pandas as pd
import pytest

from leadintel.data import EXPECTED_ROWS, DataValidationError, load_dataset, validate


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_dataset()


def test_dataset_is_bank_full(df: pd.DataFrame) -> None:
    assert len(df) == EXPECTED_ROWS
    assert df["y"].mean() == pytest.approx(0.117, abs=0.001)
    assert df["row_order"].is_monotonic_increasing


def test_every_row_is_a_valid_lead(df: pd.DataFrame) -> None:
    from leadintel.schema import Lead

    for rec in df.sample(2000, random_state=0).to_dict("records"):
        Lead.model_validate(rec)


def test_validation_catches_broken_invariant(df: pd.DataFrame) -> None:
    raw = df.drop(columns=["row_order"]).assign(y=df["y"].map({1: "yes", 0: "no"}))
    broken = raw.copy()
    broken.loc[0, "pdays"] = 10  # previous == 0 but pdays != -1
    with pytest.raises(DataValidationError, match="previous == 0"):
        validate(broken)
