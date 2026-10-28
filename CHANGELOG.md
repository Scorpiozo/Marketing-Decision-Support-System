# Changelog

## Frontend migration and integration (2026-10-12)

The Streamlit dashboard is replaced by a Flask API plus a plain
HTML/CSS/JavaScript frontend. The backend logic in `src/` is unchanged
apart from two small helpers.

### New
- **`api/service.py`** loads everything once at startup (about 1-2 s): the
  historical dataset, the preprocessor (checked against
  `feature_order.json`), both models, the held-out test split
  (transformed once) and each model's scores and decision table. Requests
  only filter in-memory data. Aggregates are memoised per
  (model, filters) in a bounded LRU cache, and the model report is
  computed once.
- **`api/app.py`** has 11 JSON/CSV endpoints and also serves the frontend
  (see README §18). Inputs are validated, with HTTP 400 and a plain
  message on error; unknown API paths return JSON 404/405. Responses are
  small: about 0.3-7.5 KB, and the customer table is paginated.
- **`run.py`** starts the dashboard (`--host`, `--port`).
- **`frontend/`**: `index.html`, `css/styles.css`, `js/charts.js`
  (dependency-free SVG/HTML charts with tooltips, resize handling and
  table views) and `js/app.js`. No framework, no build step, no JS
  packages.
- **Four sections** replace the eight Streamlit pages:
  - Executive overview: Overview, plus the About notes in the footer.
  - Campaign insights: Campaign Drivers + Customer Segments.
  - Actionable targeting: Actionable Targeting + Customer Lookup + New
    Customer Analysis.
  - Model performance: Model Performance.
- Model choice (**Gradient Boosting / Random Forest**) on the targeting
  section and in the scoring results. Only these two models exist
  anywhere in the app.
- Model performance now shows real ROC curves, confusion matrices and
  feature importances for both models, computed from the saved models on
  the test set. Before, it showed a static PNG.
- `src/model.py`: `roc_points()` and `grouped_feature_importances()`.
- Tests: `tests/test_api.py` (35 API tests) and `tests/test_frontend.py`
  (9 browser tests against a real server; skipped if Playwright is not
  installed).

### Fixed or improved compared with the Streamlit app
- Customer Lookup rendered all 9,043 customers as expanders on one page.
  The new table is paginated (25 rows per page) and sorted on the server.
- The filter reset button only reran the script. Filters are now
  removable chips with "Clear all".
- One dominant histogram bin no longer flattens the HOT/WARM tail: it is
  clipped and labelled with its true value.
- The reference "average" line label never overlaps bars.
- The layout is responsive down to 390 px without horizontal scrolling.
  Dark mode follows the OS setting.

### Removed (each one was checked for references first)
| File | Reason |
|---|---|
| `app/streamlit_app.py` (and `app/`) | Replaced by `api/` + `frontend/`. Nothing in `src/`, the training script or the tests imported it. |
| `tests/test_dashboard.py` | Tested the Streamlit app; replaced by `tests/test_api.py` and `tests/test_frontend.py`. |
| `streamlit`, `plotly` from requirements | No longer used. `flask` added. `playwright` is an optional dev dependency. |
| Streamlit entries in `.gitignore`, launchers and docs | Obsolete |

### Unchanged (verified by tests)
- Preprocessing, feature order, both trained models, business rules and
  analytics.
- For the production model, the API's scores are identical to
  `outputs/final_predictions.csv`.
- The API's metrics match `outputs/model_comparison.csv`.
- Launchers now skip training when trained models already exist.

## Backend cleanup (2026-10-12)

Scope: backend only. The Streamlit dashboard keeps all 8 pages and its
look; no new frontend or API yet.

### Models: exactly two
- The active pipeline now trains, saves and compares **Gradient Boosting**
  and **Random Forest** only (`src/model.py::MODEL_REGISTRY`).
- **Removed:** 1D CNN (TensorFlow/Keras), Logistic Regression and Decision
  Tree code, their saved artifacts and their plots.
- **New Gradient Boosting model** (there was none before):
  `GradientBoostingClassifier(300 trees, lr 0.1, depth 4, subsample 0.9)`
  with balanced sample weights, matching the balanced weighting every
  original model used. The HOT/WARM/COLD thresholds assume that
  probability scale.
- **Random Forest** settings unchanged. After retraining, its test
  probabilities match the original saved model to within 2e-16, and its
  metrics match the original run exactly.
- **Production model selection:** the dashboard previously used the CNN.
  It now uses whichever model has the higher test ROC-AUC (Gradient
  Boosting in this run: 0.930 vs 0.924), recorded in
  `models/production_model.json`. Override with
  `--production-model random_forest|gradient_boosting`.

### Preserved (verified)
- Preprocessing and the 51-column feature order: the saved preprocessor
  output is identical to the original; `feature_order.json` has the same
  `final_feature_order`.
- Business rules unchanged: lead-temperature thresholds, contact-fatigue
  bands, interaction-pattern rules, recommendation table. Given the same
  probabilities, `build_prediction_dataframe` output is identical to the
  original code's.
- Analytics and segmentation (`src/analytics.py`) unchanged apart from the
  small cleanups below.
- `data/bank-full.csv` is untouched (same MD5 before and after).

### Bugs fixed
- **Customer Lookup page crashed.** `page_customer_analysis` had its whole
  body pasted twice. That rendered the page twice and created duplicate
  widgets, which raises `StreamlitDuplicateElementId`. The duplicate copy
  was removed (the later, corrected copy was kept).
- Removed `st.metric("")` with an empty label (Streamlit warns about it);
  it now uses a real label with `label_visibility="collapsed"`.
- Replaced deprecated `use_container_width=True` (11 places) with
  `width="stretch"`. Streamlit has announced that `use_container_width`
  will be removed.
- `load_raw_data` uses `select_dtypes(include=["object", "string"])` to
  avoid the pandas 3 deprecation. It still works on pandas 2.

### Redundancy and loading
- The dashboard no longer reads `top_priority_customers.csv`: it was loaded
  on every run but never used. The file is still generated.
- The dashboard loads the preprocessor and model once per process
  (`st.cache_resource`), and the old TensorFlow import at startup is gone.
  `src/model.py` used to import TensorFlow at import time, even for pages
  that only show analytics.
- Age-group bins are defined once (`analytics.add_age_group`) instead of
  being hard-coded again in the dashboard.
- Removed a redundant `df.copy()` in `load_campaign_dataset`.
- The training loop is now driven by the model registry, so the train,
  evaluate and save code is written once instead of per model.

### Safety and robustness
- New `src/config.py` holds absolute paths, so the training script, the
  dashboard and the tests work from any working directory. The training
  script previously needed to be run from the project root.
- New `validate_feature_order()`: the dashboard refuses to score if the
  saved preprocessor doesn't match `feature_order.json`.
- `predict_single_customer` now checks that all required columns are
  present (it uses the previously unused `transform_new_records`).

### Files removed (each one was checked for references first)
| File | Reason |
|---|---|
| `temp_streamlit.py` | Older UTF-16 copy of `app/streamlit_app.py`. Nothing referenced it. |
| `notebooks/bank_marketing_cnn.ipynb` | Duplicated the training pipeline and trained the removed CNN/LR/DT models. It was not used by the script or the dashboard. |
| `models/cnn_model.keras`, `models/logistic_regression.joblib`, `models/decision_tree.joblib` | Artifacts of the removed models |
| `outputs/cnn_confusion_matrix.*`, `outputs/cnn_training_history.png`, `outputs/roc_curves_all_models.png` | Outputs of the removed models; replaced by `confusion_matrix.*` and `roc_curves.png` |
| Unused `DISPLAY_COLUMNS`, unused `numpy`/`plotly.express` imports | Dead code |

### Files added
`src/config.py`, `models/gradient_boosting.joblib`,
`models/production_model.json`, `tests/test_pipeline.py`,
`tests/test_dashboard.py`, `requirements-dev.txt`, `pytest.ini`,
`run_project.sh`, `CHANGELOG.md`.

### Dependencies
Removed `tensorflow-cpu`, `ucimlrepo`, `jupyter` and `nbformat`. Raised
the minimum to `streamlit>=1.50`. Added `requirements-dev.txt` (pytest).
