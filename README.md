# Bank Marketing — Marketing Decision Support System

A college project built on the **UCI Bank Marketing dataset**: a
prototype decision-support tool that helps a bank's marketing team decide
who to follow up with and what marketing action to consider next.

> **This is a prototype.** It is not connected to a live bank CRM. The
> dashboard shows historical campaign analysis, predictions for the
> held-out test portion of the UCI dataset, and a form that scores one
> manually entered customer (§14).

**Quick start**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py                         # then open http://localhost:5000
```

Trained models and outputs already ship in `models/` and `outputs/`;
`python train_and_evaluate.py` rebuilds them (about a minute).

---

## 1. Problem statement

A bank runs phone-call marketing campaigns to sell term deposits. Calling
every single customer is expensive and time-consuming, and calling the
same customer too often annoys them. The marketing team needs a tool that
answers, in order:

1. **How is the campaign performing overall?**
2. **Which customer/campaign factors are associated with conversion?**
3. **Who is most likely to subscribe, and which segments should be
   prioritized?**
4. **What action should the bank consider next for a given customer or
   segment?**

The first two are answered with historical, observed campaign analysis
(Executive overview, Campaign insights); the last two
with model-predicted, per-customer targeting and recommendations
(Actionable targeting).

---

## 2. Dataset

**UCI Bank Marketing dataset**
(https://archive.ics.uci.edu/dataset/222/bank+marketing) — 45,211 customer
records from a Portuguese bank's phone marketing campaigns, with 16 input
features and one target column `y` (`yes`/`no` — did the customer
subscribe to a term deposit?). The target is imbalanced: about 88% "no"
and 12% "yes".

The file lives at `data/bank-full.csv`. It is **read-only** for this
project: no script writes to it. See `data/README.md` for its source.

---

## 3. Objective

Go from *"what's the conversion rate?"* to *"who converts?"* to *"what
factors matter?"* to *"who should the bank target, and what should they
do next?"* — using honest historical analysis (§14A) and per-customer
model predictions (§14B: **Conversion Probability**, **Lead
Temperature**, **Interaction Pattern**, **Contact Fatigue**, combined into
a plain-English **Recommendation**).

---

## 4. System architecture

```
Raw customer data (data/bank-full.csv, 45,211 rows, read-only)
        │
        ├─────────────────────────────────────────────┐
        ▼                                              ▼
 Preprocessing (src/preprocessing.py)           Historical analytics
   - keep "unknown" as a category                (src/analytics.py)
   - scale numeric columns                        - actual conversion rates
   - one-hot encode categorical columns            by month/contacts/outcome/
   - stratified 80/20 split (fit on TRAIN only)    duration/segment, with sample
   - fixed feature order -> feature_order.json     sizes and reliability flags
        │                                              │
        ▼                                              │
 Model training (src/model.py)                          │
   - Gradient Boosting   ┐ both trained & compared;     │
   - Random Forest       ┘ best test ROC-AUC = production model
        │                                              │
        ▼                                              │
 Prediction / decision layer (src/prediction.py)        │
   MODEL OUTPUT:    conversion_probability              │
   DECISION LAYER:  lead_temperature   (HOT / WARM / COLD)
                    interaction_pattern (src/interaction_analysis.py)
                    contact_fatigue     (LOW / MEDIUM / HIGH)
   RECOMMENDATION:  plain-English next action, combining all three
        │                                              │
        ▼                                              ▼
 Flask API (api/)  - loads data + both models ONCE at startup
        │            - JSON endpoints; filtered aggregates are memoised
        ▼
 Frontend (frontend/: HTML + CSS + vanilla JavaScript, SVG charts)
   1. Executive overview    (historical KPIs + model lead snapshot)
   2. Campaign insights     (historical drivers + customer segments)
   3. Actionable targeting  (model-predicted leads, customer table,
                             customer detail, score a new customer)
   4. Model performance     (metrics, ROC, confusion, feature importance)
```

There is exactly one implementation of preprocessing, the models, the
rules and the recommendation logic, all in `src/`. `train_and_evaluate.py`
is the single training entry point. The API only loads its saved
artifacts and never retrains; the frontend only talks to the API.

**Two data sources, kept clearly separate:** Executive overview and
Campaign insights use `src/analytics.py`, which computes plain
counts/rates from the **full historical dataset with real outcomes** — no
model involved. Actionable targeting (and the lead snapshot on the
overview) uses **model-predicted** probabilities for the held-out test
split. Each page header states which kind of number it shows.

**Model output vs. decision layer:** the model only ever produces one
number: *"probability that the customer subscribes."* Lead Temperature,
Interaction Pattern, Contact Fatigue and the Recommendation are
**project-created rules** applied on top of that number and on existing
UCI fields. None of them are official UCI labels.

---

## 5. Data preprocessing

Implemented in `src/preprocessing.py`:

- **Missing values**: the UCI file has no real NaNs, but several columns
  use the literal string `"unknown"` (e.g. `poutcome` is `"unknown"` for
  ~82% of rows). `"unknown"` is kept as its own category. Any genuine NaNs
  are filled defensively (median for numeric, mode for categorical).
- **Categorical encoding**: `OneHotEncoder(handle_unknown="ignore")`, so an
  unseen category in a new record becomes all-zeros instead of an error.
- **Numerical scaling**: `StandardScaler` on `age`, `balance`, `day`,
  `duration`, `campaign`, `pdays`, `previous`. (Tree models don't need
  scaling; it is kept so the saved feature representation is unchanged.)
- **Train/test split**: 80/20, stratified on the target, `random_state=42`.
- **No leakage**: the `ColumnTransformer` is `fit()` on the **training
  split only**, then only `.transform()`-ed on the test split and on new
  records.

### Feature order — fixed, documented, saved

The engineered feature vector (51 columns) is built from three groups in
this exact order (`src/preprocessing.py::FEATURE_GROUPS`):

| Order | Group | Fields |
|---|---|---|
| 1 | Numerical customer features (scaled) | `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous` |
| 2 | Customer categorical features (one-hot) | `job`, `marital`, `education`, `default`, `housing`, `loan` |
| 3 | Campaign/contact features (one-hot) | `contact`, `month`, `poutcome` |

The full order, including every one-hot column name, is written to
`models/feature_order.json` on every training run. When the dashboard
loads the scoring artifacts it checks that the saved preprocessor still
produces exactly this order (`validate_feature_order`) and refuses to
score if a mismatched preprocessor/model pair is detected.

---

## 6. Models

Exactly two predictive models are supported (`src/model.py::MODEL_REGISTRY`):

| Model | Configuration | Purpose |
|---|---|---|
| **Gradient Boosting** | `GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, subsample=0.9)`, balanced sample weights | Sequential ensemble of shallow trees; each tree corrects the errors of the previous ones |
| **Random Forest** | `RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced")` | Parallel ensemble of decorrelated trees; robust classical baseline |

Both use `random_state=42`, are trained on the **same** preprocessed
train/test split, and both handle the ~88/12 imbalance with balanced
weighting. This keeps their probabilities on a comparable scale for the
fixed HOT/WARM/COLD thresholds (§9).

---

## 7. Production model selection

`train_and_evaluate.py` trains both models, then picks the one with the
highest **test ROC-AUC** as the *production model*. It writes the choice
to `models/production_model.json`, which the dashboard reads. To force a
specific model:

```bash
python train_and_evaluate.py --production-model random_forest
python train_and_evaluate.py --production-model gradient_boosting
```

Both models are always saved, so you can switch without changing code.

---

## 8. Model comparison

Results from the shipped run (`outputs/model_comparison.csv`; same
held-out test set of 9,043 customers, 0.5 decision threshold):

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | **0.858** | **0.445** | **0.862** | **0.587** | **0.930** | **0.621** |
| Random Forest | 0.846 | 0.423 | 0.861 | 0.567 | 0.924 | 0.603 |

Gradient Boosting leads on every metric in this run, but the margins are
small (ROC-AUC 0.930 vs 0.924; Recall is nearly identical). It is therefore
the production model for the shipped outputs. Both models have moderate
precision on the minority "yes" class, as expected with balanced weighting.

Also generated: `outputs/roc_curves.png` (both models) and
`outputs/confusion_matrix.png/.csv` (production model).

---

## 9. Lead classification (HOT / WARM / COLD)

**Model output → decision layer.** From `src/prediction.py`, based on the
production model's `conversion_probability`:

| Probability | Lead temperature |
|---|---|
| ≥ 0.70 | HOT |
| 0.40 – 0.70 | WARM |
| < 0.40 | COLD |

These are the project's default thresholds. `lead_temperature` is a
**project-created decision label**, not something provided by the UCI
dataset.

---

## 10. Interaction patterns

Implemented in `src/interaction_analysis.py`. **These categories are
project-defined behavioural labels — not official UCI classifications.**
They are built from three existing fields: `campaign` (current-campaign
contact count), `previous` (previous-campaign contact count), and
`poutcome` (previous campaign's outcome).

| Category | Rule |
|---|---|
| **Interested** | Previous campaign was a success AND current contact count is low (≤ 3) |
| **Repeatedly Contacted** | Current contact count is high (> 3) AND previous outcome was not a success |
| **No Previous Interaction** | `previous == 0` |
| **Previously Unsuccessful** | `poutcome == "failure"` |
| **Unclear** | None of the above apply |

Rules are checked in this order, and the first match wins — keeping the
logic simple, deterministic, and fully explainable to a non-technical
reader. The `campaign <= 3` threshold was chosen from the distribution of
the `campaign` column (median ≈ 2; most customers are contacted 1-3 times
in this data) — a simple, documented choice, not a scientifically
validated cutoff.

---

## 11. Contact fatigue

**New, project-created derived indicator.** Uses only the `campaign` field
(number of contacts made during the *current* campaign) to flag customers
at risk of being over-contacted:

| `campaign` | `contact_fatigue` |
|---|---|
| ≤ 2 | LOW |
| 3 – 4 | MEDIUM |
| ≥ 5 | HIGH |

Implemented in `src/prediction.py::classify_contact_fatigue`. Like
`interaction_pattern`, **`contact_fatigue` is not an official UCI label**
— it is a simple, explainable rule created for this project so the
recommendation engine can factor in "have we already contacted this
person a lot this campaign?" separately from their longer-term history.

---

## 12. Recommendation engine

**Decision layer → recommendation.** A small, transparent lookup table
combines `lead_temperature`, `interaction_pattern`, and `contact_fatigue`
into a plain-English recommendation (see
`src/prediction.py::recommend_action`). Contact fatigue is checked first
within each lead-temperature bucket, since "don't over-contact this
customer" is usually the most important caveat regardless of interaction
history. Examples:

- **HOT + Interested + LOW fatigue** → "Prioritize follow-up soon - strong
  positive signal."
- **HOT + Repeatedly Contacted + HIGH fatigue** → "High conversion
  potential, but avoid excessive contact."
- **WARM + No Previous Interaction + LOW fatigue** → "Good potential -
  attempt a first meaningful contact."
- **COLD + Repeatedly Contacted + HIGH fatigue** → "Low priority and
  already over-contacted - stop further outreach for now."
- **COLD + Previously Unsuccessful + LOW fatigue** → "Low priority -
  previous attempt failed; contact later if at all."

The rule table is intentionally small (a handful of cases) so it can be
explained line-by-line in a viva.

---

## 13. Final marketing output

`outputs/final_predictions.csv` contains, per test-set customer:

| Column | Meaning |
|---|---|
| `customer_record_id` | **Project-generated** row identifier — see note below |
| `age`, `job`, `balance`, `contact`, `campaign`, `previous`, `poutcome`, ... | Original customer/campaign attributes |
| `conversion_probability` | **Model output** — production-model probability of subscription |
| `lead_temperature` | **Decision layer** — HOT / WARM / COLD |
| `interaction_pattern` | **Decision layer** — behavioural category |
| `contact_fatigue` | **Decision layer** — LOW / MEDIUM / HIGH |
| `recommendation` | **Recommendation** — combines the three decision-layer outputs |

Sorted by `conversion_probability` descending.
`outputs/top_priority_customers.csv` holds the top 50 by that ranking.

**Note on `customer_record_id`:** the UCI dataset has no real customer
identifier. `customer_record_id` is generated from each customer's row
position in the test split so the script and dashboard can refer to
"this customer" consistently. It is **not** a genuine customer ID.

---

## 14. Historical analysis vs. model-predicted scoring

### A. Historical / Observed Campaign Analysis

**Executive overview** and **Campaign insights** (`src/analytics.py`)
compute plain counts and rates from the **full 45,211-row dataset's actual
`y` outcomes** — no model is involved. This is correlation, not a
guaranteed outcome for any individual customer.

### B. Model-Predicted Scoring

At startup the API rebuilds the held-out **test split** (the same
deterministic split used in training: real UCI records the models never
saw) and scores it once with **each** model. Actionable targeting can show
either model's scores. For the production model this is identical to
`outputs/final_predictions.csv` (checked by the test suite).

### C. New Customer Demonstration

**Score a new customer** (on Actionable targeting) scores one manually
entered record on demand with the same saved preprocessor and both saved
models. Scoring a test-set record through this path reproduces its row in
`final_predictions.csv` (also covered by the tests).

All three remain academic-prototype outputs; none is connected to a live
bank system.

---

## 15. Dashboard

Start it with `python run.py` and open http://localhost:5000. It has four
sections in a left rail (top tabs on phones), and one filter row shared by
the first three.

**Filters** (Age group, Job, Education, Marital status, Housing loan,
Personal loan, Campaign month, Previous outcome) are multi-select
dropdowns. Active filters appear as removable chips. They apply to every
KPI, chart and table on Executive overview, Campaign insights and
Actionable targeting; Model performance always uses the full test set.

- **Executive overview** — a headline sentence with subscriptions,
  customers and conversion rate for the current selection, and the change
  versus all customers. Below it: contacts per customer, best job segment
  and strongest month (both only from groups of 30+ customers), hot-lead
  count, conversion rate by month, contact volume by month, and the
  HOT / WARM / COLD split of the production model's scores.
- **Campaign insights** — conversion rate by month, contacts in the
  current campaign, previous outcome and call duration, plus a segment
  explorer (Age group, Job, Education, Marital status, Housing loan,
  Personal loan, Credit default). A dashed line marks the selection's
  average. Groups under 30 customers are drawn faded with a dashed outline
  and flagged in the tooltip, not hidden.
- **Actionable targeting** — switch between **Gradient Boosting** and
  **Random Forest** (the production model is marked "In use"). The KPI
  strip (scored, hot, warm, cold, high fatigue) doubles as quick filters
  for the table. Charts show the predicted-probability distribution with
  the 40% / 70% thresholds and the recommended-action mix. The priority
  customer table is paginated (25 rows), sortable (probability, age,
  balance, contacts), searchable by record ID and filterable by lead
  temperature and contact fatigue, with CSV export of the full filtered
  list. Clicking a row opens the customer's profile, decision outputs and
  both models' probabilities. **Score a new customer** opens a form with
  all 16 inputs. Dropdown options come from the fitted preprocessor, and
  the server validates every value.
- **Model performance** — metric comparison (chart + table, best value
  in bold), ROC curves with AUC, feature importance per model (one-hot
  columns summed back to their input field), and confusion matrices at
  the 0.5 threshold.

Every chart has hover tooltips, and most have a **Table** toggle that
shows the exact numbers. Charts redraw on resize, and light and dark
themes follow the operating system. The fixed palette: teal for
conversion, blue / orange for Gradient Boosting / Random Forest, and
brick / amber / slate for HOT / WARM / COLD.

---

## 16. `duration` and the post-interaction assumption

The dataset contains `duration`, representing the duration of the last
phone call. The UCI documentation notes that `duration` is only known
**after** the call ends, and `duration=0` almost always means `y='no'`.
Using it makes prediction unrealistically easy if the goal is deciding
**who to call first**.

**This project is explicitly a POST-INTERACTION prototype**: the system
supports **follow-up** decisions after a contact attempt has already
happened, so having the most recent call's duration is legitimate
information at that point. `duration` is therefore **kept**, not silently
dropped or silently used — this assumption is documented here, in the
dashboard footer and in the scoring form.

If this system were changed into a **PRE-CALL** prediction system (deciding
who to call for the very first time), `duration` would need to be dropped
and the models retrained, since it would not be available at prediction
time in that scenario.

In the **Score a new customer** form (§15), this field is labelled
**"Last call duration (s)"**, with the note:

> "Call duration is assumed known: this supports follow-up after a call."

---

## 17. Installation

Requires **Python 3.10+** (verified with Python 3.13). From the project root:

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
python -m pip install -r requirements.txt
```

Runtime dependencies are pandas, numpy, scikit-learn, joblib, matplotlib
(training report images) and Flask. The frontend is plain
HTML/CSS/JavaScript with no build step and no JS packages. It loads one web
font from Google Fonts and falls back to system fonts when offline.

For the tests:

```bash
python -m pip install -r requirements-dev.txt
python -m playwright install chromium     # only for the browser tests
```

All paths are resolved from the project folder (`src/config.py`), so
commands work from any working directory.

---

## 18. How to run

**Start the dashboard:**

```bash
python run.py                    # http://localhost:5000
python run.py --port 8080        # another port
python run.py --host 0.0.0.0     # reachable from other devices on your network
```

Startup loads the dataset, the preprocessor and both models once (about 1-2
seconds), then prints the URL. Stop it with CTRL+C.

**One-step launchers:** `run_project.bat` (Windows, double-click) and
`run_project.sh` (macOS/Linux). They install dependencies, train only if
`models/production_model.json` is missing, then start the server; the
Windows one also opens the browser.

**(Optional) retrain and regenerate artifacts** — only needed after
changing code or data:

```bash
python train_and_evaluate.py                                   # picks the best ROC-AUC model
python train_and_evaluate.py --production-model random_forest   # or force one
```

| Folder | Files |
|---|---|
| `models/` | `preprocessor.joblib`, `feature_order.json`, `gradient_boosting.joblib`, `random_forest.joblib`, `production_model.json` |
| `outputs/` | `model_comparison.csv`, `final_predictions.csv`, `top_priority_customers.csv`, `roc_curves.png`, `confusion_matrix.png/.csv`, `run_summary.json` |

Restart `run.py` after retraining so the API loads the new artifacts.

**Run the tests:**

```bash
python -m pytest -q
```

| File | Covers | Time |
|---|---|---|
| `tests/test_pipeline.py` | preprocessing, feature order, model loading, predictions, business rules, analytics | ~5 s |
| `tests/test_api.py` | every endpoint, filters, pagination/sort/search, CSV export, predictions for both models, input validation, consistency with `final_predictions.csv` and `model_comparison.csv` | ~5 s |
| `tests/test_frontend.py` | real server + headless Chromium: all sections, filters, chips, tooltips, table toggles, model switch, KPI quick filters, sort, paging, search, customer detail, CSV download, scoring form + validation, mobile layout, no JS errors | ~15 s |

`test_frontend.py` is skipped automatically if Playwright/Chromium are
not installed.

### API reference

All endpoints return JSON except the CSV export. Filters are query
parameters and may repeat (`?job=student&job=retired&month=mar`).
`model` is `gradient_boosting` or `random_forest` (default: the
production model). Invalid input returns HTTP 400 with `{"error": "..."}`.

| Method & path | Returns |
|---|---|
| `GET /api/health` | status, startup time, production model |
| `GET /api/meta` | filter options, models, segments, thresholds, form options/ranges |
| `GET /api/overview` | historical KPIs, conversion by month, best job segment |
| `GET /api/insights/drivers` | conversion by month / contacts / previous outcome / duration |
| `GET /api/insights/segments?segment=job` | conversion by one customer segment |
| `GET /api/targeting/summary?model=` | lead KPIs, recommendation mix, probability histogram |
| `GET /api/targeting/customers?model=&temperature=&fatigue=&q=&sort=&order=&page=&page_size=` | one page of ranked customers (max 100 per page) |
| `GET /api/targeting/export?...` | the same filtered list as CSV |
| `GET /api/customers/<id>?model=` | one test-set customer, with both models' probabilities |
| `POST /api/predict?model=` | score one customer (JSON body with all 16 input fields) |
| `GET /api/models` | metrics, ROC points, confusion matrices, feature importances for both models |

---

## 19. Limitations

- **Correlation, not causation**: Campaign insights shows *associations*
  between attributes and observed conversion, not causes. Confounders are
  not controlled for.
- **Small samples**: groups with fewer than 30 customers (after filters)
  are flagged as low-confidence (faded, dashed outline, tooltip note).
- **Prototype, not a live system**: predictions cover the held-out test
  split; the scoring form handles one manually entered record. There is
  no CRM connection, authentication or database.
- **Development server**: `run.py` uses Flask's built-in threaded server,
  which is fine for local or classroom use. A shared deployment should run
  `api.app:create_app()` behind a production WSGI server (e.g. gunicorn or
  waitress) and add authentication.
- **In-memory state**: each server process holds the dataset and both
  models in memory (about 300 MB measured). Restart after retraining.
- **`duration` assumption**: see §16. A pre-call version must drop
  `duration` and retrain.
- **Rules are manually designed**: interaction-pattern and contact-fatigue
  rules are transparent, not statistically optimal.
- **Customer Record ID is not a real identifier**; search is by that ID only.
- **Historical data only**: one Portuguese bank, 2008-2010.
- **Class imbalance**: positive-class precision is moderate (~0.44) even
  though Recall and ROC-AUC are strong.
- **Production model chosen on the test set**: the ROC-AUC comparison that
  picks the production model uses the same test split the metrics are
  reported on. With two closely matched models this has little effect,
  but a separate validation split or cross-validation would be cleaner.
- **Web font needs internet**: offline, the UI falls back to the system
  font (layout unaffected).

---

## 20. Future improvements

- Select the production model with cross-validation rather than the test
  split.
- Hyperparameter-tune both models with a proper search.
- Per-customer explanations (e.g. SHAP) in the customer detail view.
- Probability calibration (balanced weighting inflates raw probabilities;
  the HOT/WARM/COLD thresholds currently assume that scale).
- Authentication, a database and a production WSGI deployment.
- Track model drift by retraining on new campaign data over time.

---

## 21. Project structure

```
IV-Proj/
│
├── data/
│   ├── bank-full.csv            (original dataset, read-only)
│   └── README.md
│
├── src/                         (backend logic - single implementation)
│   ├── config.py                (absolute project paths)
│   ├── preprocessing.py         (cleaning, split, ColumnTransformer, feature order)
│   ├── model.py                 (Gradient Boosting + Random Forest, evaluation, persistence)
│   ├── interaction_analysis.py  (interaction-pattern rules)
│   ├── prediction.py            (lead temperature, contact fatigue, recommendations, scoring)
│   └── analytics.py             (historical KPIs, segmentation, filters)
│
├── api/
│   ├── app.py                   (Flask routes, error handling, static file serving)
│   └── service.py               (load-once data/model cache, filtering, memoised aggregates)
│
├── frontend/                    (no build step)
│   ├── index.html
│   ├── css/styles.css
│   └── js/
│       ├── charts.js            (dependency-free SVG/HTML charts)
│       └── app.js               (routing, filters, API calls, sections)
│
├── tests/
│   ├── test_pipeline.py
│   ├── test_api.py
│   └── test_frontend.py
│
├── models/                      (generated by train_and_evaluate.py)
├── outputs/                     (generated by train_and_evaluate.py)
│
├── run.py                       (start the dashboard)
├── train_and_evaluate.py        (train + evaluate both models)
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── README.md
├── PROJECT_OVERVIEW.md
├── CHANGELOG.md
├── run_project.bat
└── run_project.sh
```
