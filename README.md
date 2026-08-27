# Bank Marketing — Marketing Decision Support System

A college project built on the **UCI Bank Marketing dataset**: a
prototype decision-support tool that helps a bank's marketing team decide
who to follow up with and what marketing action to consider next.

This README is written for a **college student**, not an ML researcher —
it explains *what* each part does and *why*, without assuming prior deep
learning experience.

> **This is a prototype.** It is not connected to a live bank CRM. The
> dashboard displays predictions already generated for the existing/test
> portion of the UCI dataset — it does not accept brand-new customers in
> real time. This is an intentional scope decision for an academic
> project, explained further in §14.

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

This project answers all four — first with historical, observed campaign
analysis (Executive Overview, Campaign Drivers, Customer Segments), then
with model-predicted, per-customer targeting and recommendations
(Actionable Targeting) — and ranks customers so the team knows who to
prioritize first.

---

## 2. Dataset

**UCI Bank Marketing dataset**
(https://archive.ics.uci.edu/dataset/222/bank+marketing) — 45,211 customer
records from a Portuguese bank's phone marketing campaigns, with 16 input
features (age, job, marital status, education, account balance, loan
status, contact type, campaign history, etc.) and one target column `y`
(`yes`/`no` — did the customer subscribe to a term deposit?).

The target is imbalanced: about 88% "no" and 12% "yes".

See `data/README.md` for exactly where to get the CSV file.

---

## 3. Objective

Build a dashboard and modeling pipeline that lets someone unfamiliar with
the project go from *"what's the conversion rate?"* to *"who converts?"*
to *"what factors matter?"* to *"who should the bank target, and what
should they do next?"* — using a mix of honest historical/observed
analysis (§14A) and per-customer model predictions (§14B: **Conversion
Probability**, **Lead Temperature**, **Interaction Pattern**, **Contact
Fatigue**, combined into a plain-English **Recommendation**) — comparing
classical ML models against a required advanced deep-learning model (1D
CNN) honestly, without exaggerating either side.

---

## 4. System architecture (in simple language)

```
Raw customer data (bank-full.csv, 45,211 rows)
        │
        ├─────────────────────────────────────────────┐
        ▼                                              ▼
 Preprocessing (src/preprocessing.py)           Historical analytics
   - fill/label "unknown" categories             (src/analytics.py)
   - scale numeric columns                        - actual conversion rates
   - one-hot encode categorical columns            - by month/contacts/outcome/
   - train/test split (fit on TRAIN only)            duration/segment, w/ sample
        │                                            sizes and reliability flags
        ▼                                              │
 Model training (src/model.py)                          │
   - baselines: Logistic Regression,                     │
     Decision Tree, Random Forest                         │
   - 1D CNN (TensorFlow/Keras)  <-- used for final outputs │
        │                                                  │
        ▼                                                  │
 Prediction / decision layer (src/prediction.py)            │
   MODEL OUTPUT:        conversion_probability (test split)  │
   DECISION LAYER:       lead_temperature   (HOT / WARM / COLD)
                         interaction_pattern (src/interaction_analysis.py)
                         contact_fatigue     (LOW / MEDIUM / HIGH)
   RECOMMENDATION:       plain-English next action, combining all three
        │                                                  │
        ▼                                                  │
 Final ranked dataframe + Top Priority Customers table       │
        │                                                    │
        ▼                                                    ▼
 Streamlit dashboard (app/streamlit_app.py) - business hierarchy:
   1. Executive Overview     (historical KPIs)
   2. Campaign Drivers       (historical conversion-rate charts)
   3. Customer Segments      (historical conversion-rate charts)
   4. Actionable Targeting   (model-predicted leads + recommendations)
   -- supporting pages --
   5. Customer Lookup        (look up one test-set customer)
   6. New Customer Analysis  (live single-customer scoring)
   7. Model Performance      (honest technical comparison)
   8. About This Project
```

The notebook (`notebooks/bank_marketing_cnn.ipynb`) walks through the model
training half of this diagram interactively and produces the same
artifacts as the scripted pipeline (`train_and_evaluate.py`), which is
meant for a single one-command run. Both reuse the exact same code in
`src/` — there is only one implementation of preprocessing, the models,
the rules, and the recommendation logic.

**Two data sources, kept clearly separate:** the dashboard's first three
pages (Overview, Drivers, Segments) use `src/analytics.py`, which computes
plain counts/rates directly from the **full historical dataset with real
outcomes** — no model involved. Actionable Targeting, Customer Lookup, and
New Customer Analysis use `src/prediction.py` /
`outputs/final_predictions.csv`, which are **model-predicted** probabilities
on the held-out test split. Every relevant page states which kind of number
it is showing, per §14.

**Model output vs. decision layer — an important distinction for the
viva:** the CNN/ML model only ever produces one number:
*"probability that the customer subscribes."* Everything else — Lead
Temperature, Interaction Pattern, Contact Fatigue, and the Recommendation
— is a **project-created rule** applied on top of that number and on
existing UCI fields. None of these four are official UCI labels; this is
called out again in each relevant section below.

---

## 5. Data preprocessing

Implemented in `src/preprocessing.py`:

- **Missing values**: the UCI file has no real NaNs, but several columns
  use the literal string `"unknown"` (e.g. `poutcome` is `"unknown"` for
  ~82% of rows, since most customers were never contacted before). We keep
  `"unknown"` as its own category rather than dropping those rows.
- **Categorical encoding**: One-Hot Encoding via `sklearn.OneHotEncoder`.
- **Numerical scaling**: `StandardScaler` on `age`, `balance`, `day`,
  `duration`, `campaign`, `pdays`, `previous`.
- **Train/test split**: 80/20, stratified on the target so both splits keep
  the same ~88/12 class balance.
- **No leakage**: the `ColumnTransformer` (scaler + encoder) is `fit()` on
  the **training split only**, then `.transform()`-ed on the test split and
  on any new customer records — never re-fit on test/new data.

---

## 6. Models

| Model | Purpose |
|---|---|
| Logistic Regression | Simple, fast, highly interpretable baseline |
| Decision Tree | Interpretable tree-based baseline |
| Random Forest | Strong classical ML model (ensemble of decision trees) |
| **1D CNN** | Advanced deep-learning model — required by the project brief, included to compare deep learning against conventional ML and to learn nonlinear patterns among neighbouring engineered features |

All four are trained and evaluated on the **same** preprocessed train/test
split for a fair comparison (see §8).

---

## 7. CNN architecture

CNNs (Convolutional Neural Networks) were designed for **images**, where
nearby pixels are related. Our data is **tabular** — each column is an
independent customer attribute, and **the dataset is not, and is never
treated as, an image**.

To meet the project requirement of using a CNN, `Conv1D` is applied the
way it's commonly used for tabular data in applied ML:

> *The 1D CNN is being evaluated as an advanced deep-learning architecture
> on a structured feature representation. The feature ordering is fixed
> and documented so that the convolution operation is reproducible.
> Because tabular features do not possess the same natural spatial
> structure as images, the CNN is treated as an experimental/academic
> deep-learning approach rather than being assumed to be inherently
> superior to tree-based models.*

Concretely:

1. Every customer is turned into a fixed-length numeric vector (after
   scaling numeric columns and one-hot encoding categorical ones — 51
   numbers in this dataset).
2. That vector is reshaped from `(51,)` to `(51, 1)` and treated as an
   ordered 1-D "signal".
3. `Conv1D` layers slide a small filter across this vector, learning to
   detect useful **local combinations of adjacent engineered features**.
4. The result is flattened and passed through Dense layers to a single
   sigmoid output (probability of subscription).

```
Input (51 features, 1 channel)
  → Conv1D(32 filters, kernel=3) → BatchNorm → MaxPool1D
  → Conv1D(64 filters, kernel=3) → BatchNorm → MaxPool1D
  → Flatten
  → Dense(64, relu) → Dropout(0.3)
  → Dense(32, relu) → Dropout(0.2)
  → Dense(1, sigmoid)
```

- Optimizer: Adam (lr=1e-3), Loss: binary cross-entropy
- `EarlyStopping` on validation loss (patience=6, restores best weights)
- Class weights applied to handle the ~88/12 imbalance
- Random seeds set for reproducibility (`src/model.py::set_seeds`)
- Kept intentionally small — this is a decision-support tool, not a
  large-scale production model.

### Feature ordering — fixed, documented, and saved

Because Conv1D's convolution only means something reproducible if the
position of each value in the vector is fixed, the feature order used to
build the CNN's `(51,) → (51, 1)` input is **explicitly controlled**, not
left to incidental DataFrame or `ColumnTransformer` ordering. It is defined
as three fixed groups, combined in this exact order
(`src/preprocessing.py::FEATURE_GROUPS`):

| Order | Group | Fields |
|---|---|---|
| 1 | Numerical customer features (scaled) | `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous` |
| 2 | Customer categorical features (one-hot) | `job`, `marital`, `education`, `default`, `housing`, `loan` |
| 3 | Campaign/contact features (one-hot) | `contact`, `month`, `poutcome` |

This exact order — including every individual one-hot column name — is
written to `models/feature_order.json` every time `train_and_evaluate.py`
runs (via `src/preprocessing.py::save_feature_order()`), so the precise
input vector the CNN was trained on can always be inspected and
reproduced, rather than inferred from code.

**Why CNN is included:** (1) it is an academic requirement of the project
brief, (2) it can learn nonlinear feature interactions that a linear model
cannot, and (3) it gives a meaningful point of comparison between deep
learning and conventional ML on this dataset — not because it is assumed
to automatically outperform everything else (see the honest comparison
below).

---

## 8. Model comparison

All models are trained and evaluated on the **same** preprocessed
train/test split. Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC,
PR-AUC, and confusion matrix (see `outputs/model_comparison.csv`,
`outputs/cnn_confusion_matrix.png`, and `outputs/roc_curves_all_models.png`
after running the pipeline).

**Actual results from the latest run** (also saved in
`outputs/model_comparison.csv` — re-run `train_and_evaluate.py` to
regenerate, numbers vary slightly run to run):

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.846 | 0.418 | 0.815 | 0.553 | 0.908 | 0.538 |
| **Decision Tree (balanced)** | 0.795 | 0.350 | **0.877** | 0.500 | 0.885 | 0.519 |
| **Random Forest** | **0.863** | **0.452** | 0.819 | **0.582** | 0.921 | **0.598** |
| 1D CNN | 0.856 | 0.440 | 0.846 | 0.579 | **0.924** | 0.587 |

**Read honestly, not selectively:**
- **Random Forest** has the best **F1-score** and Precision — the strongest
  all-round classical model here.
- **Decision Tree (balanced)** has the best **Recall** — it misses the
  fewest actual subscribers, at the cost of many more false alarms.
- **1D CNN** has the best **ROC-AUC** in this run, and Recall competitive
  with Random Forest — but it does **not** clearly "win" on every metric,
  and in other runs Random Forest's ROC-AUC is sometimes marginally ahead
  (they are consistently close, ~0.92 for both).

**Which model is used for the final decision-support output?** The **1D
CNN** is used to generate `conversion_probability` for every customer in
`outputs/final_predictions.csv` and the Streamlit dashboard. This is
because it is the project's required advanced deep-learning model and it
remains competitive with (and in this run, ahead of) Random Forest on
ROC-AUC — **not** because its numbers were selectively presented to make
it look best. A faculty reviewer can substitute Random Forest's
probabilities instead (both are saved in `models/`) without changing any
other part of the system, since the decision layer only needs a
probability column.

---

## 9. Lead classification (HOT / WARM / COLD)

**Model output → decision layer.** From `src/prediction.py`, based on the
CNN's `conversion_probability`:

| Probability | Lead temperature |
|---|---|
| ≥ 0.70 | HOT |
| 0.40 – 0.70 | WARM |
| < 0.40 | COLD |

These are the project's default thresholds. They can be adjusted, but any
change should be re-documented here to stay consistent with the code.
`lead_temperature` is a **project-created decision label**, not something
provided by the UCI dataset.

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

The final prediction table (`outputs/final_predictions.csv`) contains, per
customer:

| Column | Meaning |
|---|---|
| `customer_record_id` | **Project-generated** row identifier — see note below |
| `age`, `job`, `balance`, `contact`, `campaign`, `previous`, `poutcome`, ... | Original customer/campaign attributes |
| `conversion_probability` | **Model output** — CNN-predicted probability of subscription |
| `lead_temperature` | **Decision layer** — HOT / WARM / COLD |
| `interaction_pattern` | **Decision layer** — behavioural category |
| `contact_fatigue` | **Decision layer** — LOW / MEDIUM / HIGH |
| `recommendation` | **Recommendation** — combines the three decision-layer outputs |

Sorted by `conversion_probability` descending. `outputs/top_priority_customers.csv`
holds the top 50 by that same ranking.

**Note on `customer_record_id`:** the UCI Bank Marketing dataset does not
provide a real customer identifier. `customer_record_id` is generated from
each customer's row position in the working dataframe, purely so the
notebook, script, and dashboard can refer to "this customer" consistently.
It is **not** an official UCI field and should not be treated as a genuine
customer ID in any real deployment.

---

## 14. Historical analysis vs. model-predicted scoring

This project shows **three distinct kinds of output**, and they should not
be confused with each other:

### A. Historical / Observed Campaign Analysis

The **Executive Overview**, **Campaign Drivers**, and **Customer
Segments** dashboard pages (`src/analytics.py`) compute plain counts and
rates directly from the **full 45,211-row dataset's actual, observed `y`
outcomes** — no model is involved. A "43% conversion rate in March" here
means 43% of real historical customers contacted in March actually
subscribed. This is used to answer *"how did the campaign perform?"* and
*"which factors are associated with conversion?"* — correlation, not a
guaranteed outcome for any individual customer.

### B. Model-Predicted Scoring

`outputs/final_predictions.csv` contains **CNN-predicted probabilities**
generated once (by `train_and_evaluate.py`) for the dataset's **test
split** — real UCI records the model never saw during training. This
powers:

- **model evaluation** (§8 — Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC on
  a consistent, held-out set),
- **customer ranking** (the Top Priority Customers table on Actionable
  Targeting), and
- **the full decision-support output** (probability → lead temperature →
  interaction pattern → contact fatigue → recommendation) on real UCI
  data, viewable per-customer on Customer Lookup.

A "78% conversion probability" here is a **model estimate for one
customer**, not an observed historical rate — see §15 for exactly which
pages use which source.

### C. New Customer Demonstration

The dashboard's **New Customer Analysis** page demonstrates how the
trained system *could* score a single customer record that is not already
in the pre-computed table — for example, to show a faculty reviewer the
live inference path rather than only a static CSV. It uses the same saved
preprocessor and saved CNN as (B), but scores a manually entered record
on demand.

**All three remain academic-prototype outputs.** None is connected to a
live bank CRM or a real, ongoing customer database — (A) and (B) work off
a static, offline dataset, and (C) only proves the scoring path can run on
any single row shaped like a UCI record.

---

## 15. Dashboard

Run with `streamlit run app/streamlit_app.py`. The dashboard is a
**prototype** — it is not wired up to a live bank CRM. It follows a
business-analysis hierarchy — **Overview → Campaign Drivers → Customer
Segments → Actionable Targeting** — so someone unfamiliar with the project
can go from "what's the conversion rate?" to "who should we target?" in
four pages, followed by supporting pages for individual customer lookups
and technical review.

**Two distinct data sources power the dashboard, and every page states
which one it uses:**

- **Historical/observed** (`src/analytics.py::load_campaign_dataset`) —
  the full 45,211-row dataset with real, actual campaign outcomes. Powers
  the first three pages.
- **Model-predicted** (`outputs/final_predictions.csv`) — CNN conversion
  probabilities on the 9,043-row held-out test split. Powers Actionable
  Targeting, Customer Lookup, and (via live inference) New Customer
  Analysis.

Pages:

- **Executive Overview** — top-line, honestly-denominated KPIs computed on
  the (filtered) historical dataset: Customers Contacted, Subscriptions,
  Conversion Rate, and Avg. Contacts per Customer, plus a
  Contacted-vs-Subscribed chart and the best-performing job segment —
  shown **only** if it has at least 30 customers (see §19's discussion of
  small-sample handling via `MIN_RELIABLE_SAMPLE_SIZE`), otherwise the
  page says so explicitly instead of highlighting an unreliable small
  group.
- **Campaign Drivers** — four rate-based charts computed on the historical
  data: Subscription Rate by Month, by Number of Contacts, by Previous
  Campaign Outcome, and by Last Call Duration. Titles describe the
  business question ("Subscription Rate by X"), not just "distribution of
  X". Every bar shows its sample size in the tooltip; bars below the
  minimum reliable sample size are rendered at reduced opacity rather than
  hidden or presented as equally trustworthy.
- **Customer Segments** — a single selectable segment (Age Group, Job,
  Education, Marital Status, Housing Loan, Personal Loan, or Credit
  Default) at a time, shown as one conversion-rate bar chart plus an
  expandable numbers table. Segments are not all auto-plotted at once, to
  avoid an overwhelming wall of charts.
- **Actionable Targeting** — explicitly labelled as **model-predicted**,
  not historical. KPI cards (Customers Scored, Hot/Warm/Cold Leads, High
  Contact Fatigue), a Recommended Action Mix chart, and the ranked Top
  Priority Customers table (`customer_record_id`, attributes, conversion
  probability, lead temperature, interaction pattern, contact fatigue,
  recommendation), downloadable as CSV.
- **Customer Lookup** — filter by lead temperature / contact fatigue and
  select one test-set customer to see their full model-predicted profile.
- **New Customer Analysis** — a form (age, job, marital status, education,
  default, balance, housing, loan, contact type, day, month, duration,
  campaign, pdays, previous, poutcome) that scores ONE manually entered
  customer on demand using the SAME saved preprocessor
  (`models/preprocessor.joblib`) and SAME saved 1D CNN
  (`models/cnn_model.keras`) used everywhere else in this project — see
  §14C and §16. Categorical dropdown options are read directly from the
  fitted preprocessor's learned categories
  (`src/preprocessing.py::get_categorical_options`), so no category is
  invented separately from the training data. **No model is retrained
  when the "Analyze Customer" button is clicked** — the saved artifacts
  are only loaded (once, cached) and used for inference.
- **Model Performance** — the same honest model-comparison table as §8,
  deliberately placed near the end since this dashboard is built for
  marketing decisions first and technical ML review second.
- **About This Project** — plain-language description of the system and
  its key assumptions (post-interaction use of `duration`,
  project-generated Customer Record ID, prototype/no-live-CRM scope, the
  historical-vs-new-customer distinction from §14).

**Filters:** a sidebar filter panel (Age Group, Job, Education, Marital
Status, Housing Loan, Personal Loan, Campaign Month, Previous Campaign
Outcome) is shown on the first four (business-hierarchy) pages, and is
applied consistently to whichever dataframe that page uses — every KPI,
chart, and table on a page reflects the same filter selections, since the
historical dataset and `final_predictions.csv` share the same underlying
customer attribute columns.

**Visual design:** the same fixed colour palette is used throughout every
page, including the new ones — Deep Navy (`#19355F`) for headers/sidebar/
structure, Teal (`#087A75`) for positive/primary chart and action
elements, Warm Cream (`#D7C9AA`) for cards and neutral panels, and Deep
Brick (`#7B2D26`) for warnings and HIGH contact fatigue. Charts avoid pie/
donut/3D styles, rainbow palettes, and decorative visuals — every chart is
a plain bar chart answering one stated business question, with the
sample size in the tooltip.

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
notebook, and in the dashboard's "About This Project" page.

If this system were changed into a **PRE-CALL** prediction system (deciding
who to call for the very first time), `duration` would need to be dropped
and the models retrained, since it would not be available at prediction
time in that scenario.

In the **New Customer Analysis** form (§15), this field is labelled
**"Last call duration (seconds)"**, with the note:

> "Duration is assumed to be known because this prototype supports
> post-interaction follow-up decisions."

---

## 17. Installation

```bash
# from the project root
python -m pip install -r requirements.txt
```

Requires Python 3.9+ (tested with Python 3.12) and roughly 1-2 GB of free
disk space for TensorFlow. Only relative paths are used throughout the
project, so it can be extracted anywhere and run from its own root folder.

---

## 18. How to run

**Notebook:**

```bash
cd notebooks
jupyter notebook bank_marketing_cnn.ipynb
```

Run all cells top to bottom. It loads/cleans the data, trains all models,
plots results, runs the interaction-pattern and contact-fatigue analysis,
and saves the final prediction tables + trained models into `../outputs/`
and `../models/`.

**Scripted pipeline (recommended before running the dashboard):**

```bash
python train_and_evaluate.py
```

Does the same work as the notebook without needing Jupyter, and regenerates
all `outputs/` and `models/` files.

**Dashboard:**

```bash
streamlit run app/streamlit_app.py
```

Then open the printed local URL (usually `http://localhost:8501`).

**Windows one-step launcher:** `run_project.bat` installs dependencies,
runs `train_and_evaluate.py`, and launches the dashboard, in that order.

---

## 19. Limitations

- **Correlation, not causation**: the Campaign Drivers and Customer
  Segments pages show *associations* between customer/campaign attributes
  and observed conversion — e.g. retired customers or March campaigns
  showing higher rates does not mean age or month *causes* subscription.
  Confounding factors (e.g. call duration correlating with genuine
  interest) are not controlled for in these simple rate breakdowns.
- **Small samples**: any segment or driver combination with fewer than 30
  customers (after filters are applied) is flagged as low-confidence in
  the dashboard (reduced chart opacity, explicit caption) rather than
  presented as a reliable finding — but with enough simultaneous filters
  applied, even "reliable" groups can shrink, so filtered results should
  still be read with the sample size in mind (shown in every chart
  tooltip and in Customer Segments' "View underlying numbers" table).
- **Prototype dashboard, not a live system**: predictions are pre-computed
  for the dataset's test split; there is no real-time scoring of brand-new
  customers or connection to an actual bank CRM.
- **`duration` assumption**: as discussed in §16, this system assumes a
  post-interaction use case. A version meant to score customers who have
  *never* been contacted should drop `duration` and retrain.
- **Interaction-pattern and contact-fatigue rules are simple and manually
  designed**, not learned from data — they are meant to be transparent and
  explainable, not statistically optimal.
- **Customer Record ID is not a real identifier** — it is generated from
  row position and would need to be replaced with a genuine customer key
  in any real deployment.
- **Historical data only**: the dataset covers a single Portuguese bank's
  campaigns from 2008-2010; predictions may not generalise to other banks,
  products, countries, or time periods without retraining.
- **Class imbalance**: even with class weighting, the "yes" class (~12% of
  data) is harder to predict precisely — Precision for the positive class
  is moderate (~0.44) even though Recall and ROC-AUC are strong.
- **No single model dominates**: as shown in §8, Random Forest, the 1D
  CNN, and even the balanced Decision Tree each lead on a different metric.
  The CNN is used for the final output mainly because it is the required
  advanced model, not because it is unambiguously the best performer.

---

## 20. Future improvements

- Try feature-embedding based tabular deep learning (e.g. TabNet, entity
  embeddings for categoricals) as an additional comparison model.
- Hyperparameter-tune the CNN (filters, kernel size, learning rate) with a
  proper search instead of a single hand-picked architecture.
- Add SHAP or permutation feature importance to explain individual CNN
  predictions, not just the overall interaction-pattern rules.
- Extend the dashboard to accept a brand-new customer record and score it
  live, once a real backend/database is available.
- Track model drift by re-training on new campaign data over time.
- Add authentication and a database backend for real deployment instead of
  static CSV files.

---

## 21. Project structure

```
bank_marketing_project/
│
├── data/
│   ├── bank-full.csv
│   └── README.md
│
├── notebooks/
│   └── bank_marketing_cnn.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── interaction_analysis.py
│   ├── prediction.py
│   └── analytics.py
│
├── app/
│   └── streamlit_app.py
│
├── models/            (generated)
├── outputs/            (generated)
│
├── train_and_evaluate.py
├── requirements.txt
├── README.md
├── PROJECT_OVERVIEW.md
└── run_project.bat
```
