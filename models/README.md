# Models folder

Trained artifacts produced by `train_and_evaluate.py`:

| File | Description |
|---|---|
| `preprocessor.joblib` | Fitted scikit-learn `ColumnTransformer` (scaling + one-hot encoding), fit on the training split only |
| `feature_order.json` | The exact, fixed order of the 51 engineered features fed to both models (see README.md §5) |
| `gradient_boosting.joblib` | Trained Gradient Boosting classifier |
| `random_forest.joblib` | Trained Random Forest classifier |
| `production_model.json` | Which of the two models generates the decision outputs, how it was selected, and its test metrics |

The API (`api/service.py`) loads the preprocessor, checks it against
`feature_order.json`, and loads both model files once at startup.
`production_model.json` decides which model the dashboard uses by default.
Nothing is retrained by the API or the dashboard.

All files are regenerated every time `train_and_evaluate.py` runs, so it is
safe to delete everything in this folder (except this README) and re-run
the script.
