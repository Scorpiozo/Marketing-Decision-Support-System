# Baseline Audit — the original (legacy) implementation

*Audited 2026-10-07 at commit `01df404`. Every number below was produced by actually
running the original code in a clean Python 3.12 environment on Apple Silicon (macOS),
not copied from the original README.*

This document records what the repository looked like **before** the redesign, so the
improvements can be judged against evidence. The original code is preserved in git
history (commits `850341e` → `01df404`).

---

## 1. Repository map (as found)

```
README.md (711 lines) · PROJECT_OVERVIEW.md · requirements.txt · run_project.bat (Windows only)
train_and_evaluate.py      end-to-end training script (LR, DT, RF, 1D CNN)
temp_streamlit.py          100 KB UTF-16 file — an accidental editor/temp copy of the app
app/streamlit_app.py       2,089-line single-file Streamlit dashboard (≈570 lines of inline CSS)
src/preprocessing.py       load / clean / split / ColumnTransformer / persistence
src/model.py               model builders, CNN, metrics, persistence
src/prediction.py          HOT/WARM/COLD thresholds, fatigue, recommendation text
src/interaction_analysis.py  5 rule-based "interaction pattern" labels
src/analytics.py           historical conversion-rate breakdowns for the dashboard
notebooks/bank_marketing_cnn.ipynb   44-cell notebook duplicating the pipeline
data/bank-full.csv         the dataset (3.5 MB, committed)
models/                    23 MB of artifacts — random_forest.joblib alone is 23 MB
outputs/                   CSVs + PNGs incl. 9,043-row final_predictions.csv
```

No tests, no CI, no lockfile, no Docker/deploy config, no type checking or linting.
No secrets or credentials found (`git grep` for keys/passwords/tokens/local paths: clean).

## 2. Data

| Item | Finding |
|---|---|
| File | `data/bank-full.csv` — UCI Bank Marketing (id 222), **`bank-full` variant** (17 columns, *not* the 21-column `bank-additional` variant with socio-economic indicators). Converted from the original `;`-separated, quoted format to plain CSV. |
| Rows / columns | 45,211 rows × 16 features + target `y` |
| Target | `y` ∈ {yes, no}; **positive rate 11.70 %** (5,289 yes) — imbalanced ~7.5 : 1 |
| Missing values | 0 explicit nulls. Missingness is encoded as the string `"unknown"`: `poutcome` 81.7 %, `contact` 28.8 %, `education` 4.1 %, `job` 0.6 % |
| Sentinels | `pdays = -1` means "never previously contacted" (81.7 % of rows) — a sentinel mixed into a numeric column |
| Duplicates | 0 exact duplicate rows |
| Numeric | age, balance, day, duration, campaign, pdays, previous |
| Categorical | job, marital, education, default, housing, loan, contact, month, poutcome |
| **Chronological order** | UCI documents `bank-full` as ordered by date, May 2008 → Nov 2010. Verified: 30 contiguous month-runs. **Strong drift**: conversion rate by row-quintile = 3.4 % → 5.6 % → 6.1 % → 11.8 % → 31.6 %. |
| **Era artifacts** | `contact = "unknown"` is 100 % of the first quintile and ~0 % later; `poutcome` is never known in the first 40 % of rows. These columns partly encode *when* a record was collected, not *how the customer behaved*. |

## 3. ML pipeline (as found)

| Aspect | Finding |
|---|---|
| Split | Single stratified random 80/20 split, `random_state=42`. No validation set for sklearn models; CNN uses Keras `validation_split=0.15`. No cross-validation. |
| Preprocessing | `ColumnTransformer` (StandardScaler + OneHotEncoder) **fit on train only** ✅. `pdays=-1` sentinel scaled as if it were a real number of days. |
| Features | **All 16 columns, including `duration`** (see §5). |
| Imbalance | `class_weight="balanced"` (sklearn) / inverse-frequency weights (CNN). |
| Models | Logistic Regression, Decision Tree (depth 7), Random Forest (200 trees, depth 12), 1D CNN (2×Conv1D+BN+Pool → Dense 64/32 → sigmoid) |
| CNN input | The 51-dim one-hot/scaled vector reshaped to `(51, 1)` and convolved with kernel size 3. "Neighbouring" positions are an artefact of column order (e.g. `duration` next to `campaign`; `job_student` next to `job_technician`). There is no spatial or sequential structure for a convolution to exploit. The original docs acknowledge this. |
| Threshold | Fixed 0.5 for all models (on balanced-weighted, i.e. inflated, probabilities) |
| Calibration | Not measured, not corrected (see §6) |
| Model used downstream | **1D CNN**, explicitly chosen because "the project brief requires an advanced deep-learning model", not because it performed best |
| Reproducibility | sklearn models reproducible. **CNN is not**: committed ROC-AUC 0.9215, re-run 0.9180 (TF nondeterminism; early-stops at ~14 epochs) |

### Reproduced metrics (test set n = 9,043, threshold 0.5, *with `duration`*)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.846 | 0.418 | 0.815 | 0.553 | 0.908 | 0.538 |
| Decision Tree (balanced) | 0.795 | 0.350 | 0.877 | 0.500 | 0.885 | 0.519 |
| Random Forest | 0.847 | 0.425 | 0.862 | 0.569 | **0.924** | **0.603** |
| 1D CNN (re-run) | 0.847 | 0.423 | 0.851 | 0.565 | 0.918 | 0.570 |

Training run: **26.8 s wall-clock, 508 MB peak RSS**.

## 4. Decision layer (as found)

* `lead_temperature`: HOT ≥ 0.70, WARM ≥ 0.40, else COLD — applied to the CNN's
  (uncalibrated) probability. Thresholds are not justified by cost or data.
* `interaction_pattern`: 5 hand-written rules over `campaign`, `previous`, `poutcome`
  (Interested / Repeatedly Contacted / No Previous Interaction / Previously Unsuccessful / Unclear).
* `contact_fatigue`: `campaign` ≤ 2 LOW, ≤ 4 MEDIUM, else HIGH.
* `recommendation`: a nested if/else producing free-text strings.
* There is no uncertainty handling or "human review" state, and no evaluation of whether
  the segments are useful.

## 5. Leakage probe — `duration`

UCI's own documentation for this dataset (attribute 11 in `bank-additional-names.txt`, inside the official dataset-222 archive; the note is absent from `bank-names.txt`): *"duration highly affects the output target (e.g., if
duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also,
after the end of the call y is obviously known. Thus, this input should only be included
for benchmark purposes and should be discarded if the intention is to have a realistic
predictive model."*

The original README keeps `duration` under a "post-interaction follow-up" assumption.
That assumption does not hold up: the *last* contact is the call in which the deposit is
sold or refused, so its length is effectively a *consequence* of the outcome. Long calls
are the sales. Subscription rate by `duration` quintile: 0.7 % → 3.7 % → 7.9 % → 13.1 % → 33.2 %.

Measured impact (same split, same seed):

| Feature set | LR AUC | RF AUC | HistGB AUC | HistGB PR-AUC |
|---|---|---|---|---|
| All 16 (original) | 0.906 | 0.932 | 0.934 | 0.632 |
| Without `duration` | 0.772 | 0.806 | 0.803 | 0.465 |
| Without `duration`, `day`, `month` | 0.753 | 0.762 | 0.767 | 0.409 |

**≈ 0.13 of the headline AUC comes from a variable that is not available when the
decision ("should we follow up?") is made.**

## 6. Calibration probe

Actual conversion rate inside each original lead-temperature bucket (test set):

| Bucket | n (CNN) | CNN mean predicted | **Actually converted** | RF mean predicted | RF actual |
|---|---|---|---|---|---|
| COLD < 0.40 | 6,650 | 5.4 % | 1.9 % | 15.0 % | 1.4 % |
| WARM 0.40–0.70 | 1,070 | 57.0 % | **20.9 %** | 55.8 % | 23.4 % |
| HOT ≥ 0.70 | 1,323 | 82.1 % | **53.4 %** | 79.5 % | 58.0 % |

Brier score: CNN 0.098, RF 0.107. Mean predicted probability 0.23–0.30 vs. true base rate
0.117. **The "Conversion Probability" shown to users is roughly 2–3× too high**, a direct
consequence of balanced class weights with no recalibration.

## 7. Application (as found)

| Measure | Result |
|---|---|
| Install | `requirements.txt` **fails on macOS/Apple Silicon** (`tensorflow-cpu` has no arm64/macOS wheels). Unpinned `>=` ranges, no lockfile. Pulls in Jupyter + TensorFlow (~1.5 GB env) for a dashboard. |
| Launch | Windows-only `run_project.bat` **retrains every model before launching the app**. |
| Server ready | 3.3 s (`/_stcore/health`) |
| First live CNN prediction | ~1.05 s (TensorFlow import + model load, then cached via `st.cache_resource`) ✅ caching present |
| Subsequent prediction | ~0.10 s (full Streamlit script rerun) |
| TF import + CNN load (cold) | 2.96 s |
| Pages | Executive Overview · Campaign Drivers · Customer Segments · Actionable Targeting · Customer Lookup · New Customer Analysis · Model Performance · About |

### UX observations (exercised in a browser)

* Visually tidy (custom navy/teal theme), but **framed as descriptive campaign analytics**,
  not as a decision workflow. 4 of 8 pages are historical bar charts.
* No lead-level decision card that combines prediction, action, evidence and confidence.
* **"Why this result?" is static template text** describing the rules. It contains no
  per-lead model explanation. There is no XAI anywhere in the app.
* Shows a raw "Conversion Probability 52.4 %" (miscalibrated, see §6) with no uncertainty.
* New-customer form accepts **contradictory inputs** by default (`previous = 0`,
  `pdays = -1` but `poutcome = failure`) without validation.
* `duration` is a required input for "new customer" scoring, which is impossible to know before a call.
* "Campaign Drivers" charts conversion by call duration, which reinforces the leakage.
* Streamlit "Deploy" button and default chrome visible.

## 8. What the original got right (worth keeping)

* Preprocessor fit on the training split only, no test-set contamination of scaling/encoding.
* Honest framing: it documents that the CNN isn't the best model and that the dataset is
  not a real CRM, and it flags small-sample segments (n < 30) instead of hiding them.
* Clear separation in wording between *historical* rates and *model-predicted* scores.
* `st.cache_data` / `st.cache_resource` used for the data and models.
* Interaction-pattern idea (`campaign` × `previous` × `poutcome`) is the right raw material.
  It just needs to be grounded, evaluated, and integrated with the model.
