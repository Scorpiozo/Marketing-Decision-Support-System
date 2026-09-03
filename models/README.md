# Models folder

This folder holds trained artifacts produced by `train_and_evaluate.py`
(or by running the notebook):

| File | Description |
|---|---|
| `preprocessor.joblib` | Fitted scikit-learn `ColumnTransformer` (scaling + one-hot encoding), fit on the training split only |
| `feature_order.json` | The exact, fixed, documented order of engineered features fed to every model (including the 1D CNN's `(n_features,) → (n_features, 1)` input) — see README.md §7 |
| `logistic_regression.joblib` | Trained Logistic Regression baseline |
| `decision_tree.joblib` | Trained (balanced) Decision Tree baseline |
| `random_forest.joblib` | Trained Random Forest baseline |
| `cnn_model.keras` | Trained 1D CNN (used by the Streamlit dashboard's Marketing Overview/Customer Analysis pages via `outputs/final_predictions.csv`, and loaded directly by the **New Customer Analysis** page for on-demand single-record scoring) |

These files are regenerated every time `train_and_evaluate.py` is run, so
it's safe to delete everything in this folder and re-run the script.
`preprocessor.joblib` and `cnn_model.keras` are also loaded directly (never
retrained) by the dashboard's New Customer Analysis page.
