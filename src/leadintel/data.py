"""Load and validate the UCI Bank Marketing (bank-full) dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from leadintel.config import DATA_PATH
from leadintel.schema import CATEGORY_LEVELS

EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y",
]  # fmt: skip
EXPECTED_ROWS = 45_211


class DataValidationError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Return the dataset with `y` as 0/1 and a `row_order` column (chronological index).

    Accepts both the original UCI format (``;``-separated, quoted) and plain CSV.
    Fails loudly if the schema or the documented invariants don't hold.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download bank-full.csv from "
            "https://archive.ics.uci.edu/dataset/222/bank+marketing into data/."
        )
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().strip('"') for c in df.columns]
    validate(df)
    df["y"] = (df["y"] == "yes").astype(int)
    # bank-full is ordered by contact date (May 2008 -> Nov 2010); keep that order explicit
    # because it powers the out-of-time robustness evaluation.
    df["row_order"] = range(len(df))
    return df


def validate(df: pd.DataFrame) -> None:
    problems: list[str] = []
    if list(df.columns) != EXPECTED_COLUMNS:
        problems.append(f"columns {list(df.columns)} != expected {EXPECTED_COLUMNS}")
    else:
        if df.isna().any().any():
            problems.append("unexpected missing values (UCI encodes missing as 'unknown')")
        for col, levels in CATEGORY_LEVELS.items():
            extra = set(df[col].unique()) - set(levels)
            if extra:
                problems.append(f"{col}: unexpected levels {sorted(extra)}")
        if not set(df["y"].unique()) <= {"yes", "no"}:
            problems.append("target y must be yes/no")
        if ((df["previous"] == 0) != (df["pdays"] == -1)).any():
            problems.append("invariant broken: previous == 0 <=> pdays == -1")
        if ((df["previous"] == 0) & (df["poutcome"] != "unknown")).any():
            problems.append("invariant broken: previous == 0 => poutcome == 'unknown'")
    if problems:
        raise DataValidationError("Dataset failed validation:\n- " + "\n- ".join(problems))
