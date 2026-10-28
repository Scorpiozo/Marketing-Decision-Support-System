import json
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import DATA_PATH, FEATURE_ORDER_PATH, PREPROCESSOR_PATH


NUMERIC_FEATURES = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

CUSTOMER_CATEGORICAL_FEATURES = ["job", "marital", "education", "default", "housing", "loan"]

CAMPAIGN_CATEGORICAL_FEATURES = ["contact", "month", "poutcome"]

CATEGORICAL_FEATURES = CUSTOMER_CATEGORICAL_FEATURES + CAMPAIGN_CATEGORICAL_FEATURES

# Group order defines the engineered feature order (numeric first, then
# customer categoricals, then campaign/contact categoricals). Do not reorder.
FEATURE_GROUPS = {
    "1_numerical_customer_features": NUMERIC_FEATURES,
    "2_customer_categorical_features": CUSTOMER_CATEGORICAL_FEATURES,
    "3_campaign_contact_features": CAMPAIGN_CATEGORICAL_FEATURES,
}

REQUIRED_INPUT_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = "y"


def load_raw_data(data_path=DATA_PATH):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find the dataset at '{data_path}'.\n"
            "Please download 'bank-full.csv' from the UCI Bank Marketing "
            "dataset (https://archive.ics.uci.edu/dataset/222/bank+marketing) "
            "and place it in the data/ folder. See data/README.md for details."
        )

    df = pd.read_csv(data_path)
    if df.shape[1] == 1:
        df = pd.read_csv(data_path, sep=";")

    df.columns = [c.strip().strip('"') for c in df.columns]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip().str.strip('"')

    return df


def clean_data(df):
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
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"yes": 1, "no": 0}).astype(int)
    return df


def train_test_split_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor():
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
    if preprocessor is None:
        preprocessor = build_preprocessor()

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    return X_train_proc, X_test_proc, preprocessor, feature_names


def get_feature_names(fitted_preprocessor):
    num_names = NUMERIC_FEATURES
    cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return list(num_names) + cat_names


def save_feature_order(fitted_preprocessor, path=FEATURE_ORDER_PATH):
    feature_names = get_feature_names(fitted_preprocessor)
    payload = {
        "description": (
            "Fixed, reproducible order of engineered features fed to every "
            "model (Gradient Boosting and Random Forest). Produced by "
            "models/preprocessor.joblib; new records must be transformed by "
            "that same preprocessor so columns arrive in exactly this order."
        ),
        "feature_groups": FEATURE_GROUPS,
        "n_features_total": len(feature_names),
        "final_feature_order": feature_names,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def load_feature_order(path=FEATURE_ORDER_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature order file not found at '{path}'. Run the training "
            "pipeline first so it can be created and saved."
        )
    with open(path) as f:
        return json.load(f)


def validate_feature_order(fitted_preprocessor, path=FEATURE_ORDER_PATH):
    """Raise if the preprocessor's output columns differ from feature_order.json.

    Guards against pairing a model with a preprocessor from a different run.
    """
    expected = load_feature_order(path)["final_feature_order"]
    actual = get_feature_names(fitted_preprocessor)
    if actual != expected:
        raise ValueError(
            "Preprocessor feature order does not match models/feature_order.json. "
            "Re-run `python train_and_evaluate.py` so all artifacts come from the same run."
        )
    return actual


def transform_new_records(df_new, fitted_preprocessor):
    missing_cols = [c for c in REQUIRED_INPUT_COLUMNS if c not in df_new.columns]
    if missing_cols:
        raise ValueError(f"New records are missing required columns: {missing_cols}")
    return fitted_preprocessor.transform(df_new)


def get_categorical_options(fitted_preprocessor):
    cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
    return {
        col: list(cats)
        for col, cats in zip(CATEGORICAL_FEATURES, cat_encoder.categories_)
    }


def save_preprocessor(preprocessor, path=PREPROCESSOR_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path=PREPROCESSOR_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Preprocessor not found at '{path}'. Run `python train_and_evaluate.py` "
            "first so it can be created and saved."
        )
    return joblib.load(path)


def load_and_prepare(data_path=DATA_PATH, test_size=0.2, random_state=42):
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
    result = load_and_prepare()
    print("Raw shape:", result["raw_df"].shape)
    print("Train shape (processed):", result["X_train_proc"].shape)
    print("Test shape (processed):", result["X_test_proc"].shape)
    print("Number of engineered features:", len(result["feature_names"]))
