"""
preprocessing.py
-----------------
Data loading and preprocessing utilities for the UCI Bank Marketing dataset.

This module is responsible for:
    * Loading the raw dataset from disk (or downloading it if missing)
    * Cleaning / handling missing values
    * Splitting the data into train and test sets
    * Building a scikit-learn ColumnTransformer that:
          - scales numeric columns
          - one-hot encodes categorical columns
    * Fitting that transformer ONLY on the training data to avoid leakage,
      then re-using it (via .transform) on the test set and on any new
      customer records that need a prediction.

Keeping all of this in one place means the notebook, the batch-prediction
script, and the Streamlit dashboard all preprocess data in EXACTLY the same
way, which is essential for consistent predictions.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ---------------------------------------------------------------------------
# Column definitions (UCI Bank Marketing - bank-full.csv, 17 columns)
# ---------------------------------------------------------------------------
# NOTE ON FEATURE ORDER (used by the 1D CNN - see src/model.py):
# The three groups below are combined, IN THIS FIXED ORDER, into the final
# preprocessed feature vector fed to every model, including the CNN:
#
#   1. Numerical customer features   (scaled, kept in this exact order)
#   2. Customer categorical features (one-hot encoded, in this exact order)
#   3. Campaign/contact features     (one-hot encoded, in this exact order)
#
# This order is fixed by the literal Python lists below - it does NOT depend
# on incidental DataFrame column order, so it is reproducible every time the
# preprocessor is (re)fit. It is also saved to disk (see
# save_feature_order()/FEATURE_ORDER_PATH) so the exact order used to train
# the CNN can always be inspected later.
NUMERIC_FEATURES = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

CUSTOMER_CATEGORICAL_FEATURES = ["job", "marital", "education", "default", "housing", "loan"]

CAMPAIGN_CATEGORICAL_FEATURES = ["contact", "month", "poutcome"]

# Kept for backward compatibility with existing code that expects a single
# combined categorical list, in the same fixed order as above.
CATEGORICAL_FEATURES = CUSTOMER_CATEGORICAL_FEATURES + CAMPAIGN_CATEGORICAL_FEATURES

FEATURE_GROUPS = {
    "numerical_customer_features": NUMERIC_FEATURES,
    "customer_categorical_features": CUSTOMER_CATEGORICAL_FEATURES,
    "campaign_contact_features": CAMPAIGN_CATEGORICAL_FEATURES,
}

TARGET_COLUMN = "y"

# Columns kept "as is" (not fed to the model) but useful for the final
# report / dashboard so a human can recognise the customer.
DISPLAY_COLUMNS = ["age", "job", "balance", "contact", "campaign", "previous", "poutcome"]


def load_raw_data(data_path="data/bank-full.csv"):
    """
    Load the raw Bank Marketing dataset from a local CSV file.

    The official UCI file uses ';' as a separator. We first try to read the
    file as a normal comma separated file (already-cleaned copy used by this
    project); if that fails to produce the expected columns we fall back to
    the original UCI semicolon-separated / quoted format.

    Parameters
    ----------
    data_path : str
        Path to the bank-full.csv file.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find the dataset at '{data_path}'.\n"
            "Please download 'bank-full.csv' from the UCI Bank Marketing "
            "dataset (https://archive.ics.uci.edu/dataset/222/bank+marketing) "
            "and place it in the data/ folder. See data/README.md for details."
        )

    # Try plain comma-separated first
    df = pd.read_csv(data_path)
    if df.shape[1] == 1:
        # Probably still semicolon separated -> re-read properly
        df = pd.read_csv(data_path, sep=";")

    # Strip any stray quote characters / whitespace from column names & string cells
    df.columns = [c.strip().strip('"') for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.strip('"')

    return df


def clean_data(df):
    """
    Handle missing / unknown values.

    The UCI dataset does not contain NaNs, but several categorical columns
    use the literal string 'unknown' to represent a missing value. We treat
    'unknown' as its own valid category (this is common practice for this
    dataset) rather than dropping rows, since dropping would throw away a
    large fraction of the data (e.g. ~29% of 'poutcome' is 'unknown').

    We additionally guard against genuine NaNs (e.g. if the CSV was edited)
    by filling numeric NaNs with the median and categorical NaNs with the
    column mode.
    """
    df = df.copy()

    for col in NUMERIC_FEATURES:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_FEATURES:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].replace("nan", "unknown")

    return df


def encode_target(df):
    """Convert the 'yes'/'no' target column into binary 1/0."""
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"yes": 1, "no": 0}).astype(int)
    return df


def train_test_split_data(df, test_size=0.2, random_state=42):
    """
    Split into train/test sets. Stratified on the target so that the class
    balance (roughly 88% no / 12% yes) is preserved in both splits.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor():
    """
    Build (but do not fit) a scikit-learn ColumnTransformer that:
        * scales numeric columns with StandardScaler
        * one-hot encodes categorical columns

    Returns
    -------
    sklearn.compose.ColumnTransformer
    """
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def fit_transform_train_test(X_train, X_test, preprocessor=None):
    """
    Fit the preprocessor on TRAIN data only, then transform both train and
    test sets. This is the key step that avoids data leakage: the scaler's
    mean/std and the one-hot encoder's category list are learned exclusively
    from the training split.

    Returns
    -------
    X_train_proc, X_test_proc : np.ndarray
    preprocessor : fitted ColumnTransformer
    feature_names : list[str]
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    return X_train_proc, X_test_proc, preprocessor, feature_names


def get_feature_names(fitted_preprocessor):
    """Return human-readable feature names after the ColumnTransformer, in
    the exact fixed order used everywhere downstream (including the CNN's
    input vector). This order follows FEATURE_GROUPS above:
    numerical customer features -> customer categorical features (one-hot)
    -> campaign/contact categorical features (one-hot)."""
    num_names = NUMERIC_FEATURES
    cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return list(num_names) + cat_names


FEATURE_ORDER_PATH = "models/feature_order.json"


def save_feature_order(fitted_preprocessor, path=FEATURE_ORDER_PATH):
    """
    Save the exact, fixed feature order used to build the input vector for
    every model (including the 1D CNN's (n_features, 1) input) to a small
    JSON file. This makes the CNN's feature ordering explicit, reproducible,
    and auditable, instead of relying on implicit DataFrame/ColumnTransformer
    ordering that a reader would otherwise have to infer from code.
    """
    import json

    feature_names = get_feature_names(fitted_preprocessor)
    payload = {
        "description": (
            "Fixed, reproducible order of engineered features fed to every "
            "model (Logistic Regression, Decision Tree, Random Forest, and "
            "the 1D CNN). The CNN reshapes this vector from (n_features,) "
            "to (n_features, 1) in exactly this order."
        ),
        "feature_groups": {
            "1_numerical_customer_features": NUMERIC_FEATURES,
            "2_customer_categorical_features": CUSTOMER_CATEGORICAL_FEATURES,
            "3_campaign_contact_features": CAMPAIGN_CATEGORICAL_FEATURES,
        },
        "n_features_total": len(feature_names),
        "final_feature_order": feature_names,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def load_feature_order(path=FEATURE_ORDER_PATH):
    """Load the previously saved, fixed feature order (see save_feature_order)."""
    import json

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature order file not found at '{path}'. Run the training "
            "pipeline first so it can be created and saved."
        )
    with open(path) as f:
        return json.load(f)


def transform_new_records(df_new, fitted_preprocessor):
    """
    Transform brand-new customer records (e.g. for the dashboard / batch
    prediction) using an ALREADY-FITTED preprocessor, so that scaling and
    encoding are perfectly consistent with what the models were trained on.
    """
    missing_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in df_new.columns]
    if missing_cols:
        raise ValueError(f"New records are missing required columns: {missing_cols}")
    return fitted_preprocessor.transform(df_new)


def get_categorical_options(fitted_preprocessor):
    """
    Return the exact category values the OneHotEncoder learned during
    training, keyed by column name. Used by the dashboard's "New Customer
    Analysis" form so the selectbox choices come directly from the same
    preprocessing definitions used to train the models - no categories are
    invented or hand-typed separately.
    """
    cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
    return {
        col: list(cats)
        for col, cats in zip(CATEGORICAL_FEATURES, cat_encoder.categories_)
    }


def save_preprocessor(preprocessor, path="models/preprocessor.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path="models/preprocessor.joblib"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Preprocessor not found at '{path}'. Run the notebook / training "
            "script first so it can be created and saved."
        )
    return joblib.load(path)


def load_and_prepare(data_path="data/bank-full.csv", test_size=0.2, random_state=42):
    """
    Convenience function that runs the full pipeline:
    load -> clean -> encode target -> split -> fit/transform.

    Returns a dict with everything downstream code needs.
    """
    df = load_raw_data(data_path)
    df = clean_data(df)
    df = encode_target(df)

    X_train, X_test, y_train, y_test = train_test_split_data(
        df, test_size=test_size, random_state=random_state
    )

    X_train_proc, X_test_proc, preprocessor, feature_names = fit_transform_train_test(
        X_train, X_test
    )

    return {
        "raw_df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    # Quick manual test when running `python src/preprocessing.py`
    result = load_and_prepare(data_path="data/bank-full.csv")
    print("Raw shape:", result["raw_df"].shape)
    print("Train shape (processed):", result["X_train_proc"].shape)
    print("Test shape (processed):", result["X_test_proc"].shape)
    print("Number of engineered features:", len(result["feature_names"]))
