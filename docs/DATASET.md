# Dataset

## Identity

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository, dataset 222, *Bank Marketing* — https://archive.ics.uci.edu/dataset/222/bank+marketing |
| Variant used | **`bank-full`** (17 columns: 16 inputs + `y`). Not `bank`, not `bank-additional(-full)` (21 columns with socio-economic indicators). |
| File in repo | `data/bank-full.csv`, converted from UCI's `;`-separated, quoted format to plain CSV. The loader accepts either format. |
| Rows × columns | 45,211 × 17 |
| SHA-256 (file used for the deployed model) | `b44507c26634736d4a0619ec2d7c4b6ff56a2c63cbeb5176d3134e7b25e416f0` (from `artifacts/model_card.json`) |
| Period | May 2008 – November 2010, **ordered by date** (no year column) |
| Licence | CC BY 4.0 |

### Citations

* Dataset: Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
* Paper: Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31. https://doi.org/10.1016/j.dss.2014.03.001
* The `bank-full` readme (`bank-names.txt`) asks users to cite the earlier paper: Moro, S., Laureano, R., & Cortez, P. (2011). Using data mining for bank direct marketing: an application of the CRISP-DM methodology. ESM'2011, pp. 117–121.

## Variable dictionary

Descriptions are quoted from `bank-names.txt` (inside the official UCI archive). The UCI web table labels `day` as `day_of_week`, which is wrong for `bank-full`: it is the **day of the month**.

| Column | Type | UCI description | Group in this project |
|---|---|---|---|
| age | numeric | age | Customer profile |
| job | categorical (12) | type of job | Customer profile |
| marital | categorical (3) | marital status | Customer profile |
| education | categorical (4) | education level | Customer profile |
| default | yes/no | has credit in default? | Customer profile |
| balance | numeric | average yearly balance, in euros | Customer profile |
| housing | yes/no | has housing loan? | Customer profile |
| loan | yes/no | has personal loan? | Customer profile |
| contact | categorical | "contact communication type (categorical: "unknown","telephone","cellular")" | Campaign context |
| day | numeric | "last contact day of the month (numeric)" | Campaign context |
| month | categorical | "last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")" | Campaign context |
| duration | numeric | "last contact duration, in seconds (numeric)" | **Excluded — leakage** ([details](DATA_LEAKAGE_AND_FEATURE_AVAILABILITY.md)) |
| campaign | numeric | "number of contacts performed during this campaign and for this client (numeric, includes last contact)" | Interaction history |
| pdays | numeric | "number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)" | Interaction history |
| previous | numeric | "number of contacts performed before this campaign and for this client (numeric)" | Interaction history |
| poutcome | categorical | "outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")" | Interaction history |
| y | yes/no | has the client subscribed a term deposit? | **Target** |

## Target

| y | Count | Share |
|---|---|---|
| no | 39,922 | 88.3 % |
| yes | 5,289 | **11.7 %** |

Imbalanced about 7.5 : 1. A model that always predicts "no" is 88 % accurate, so accuracy is not used as a headline metric (see [ML_METHODOLOGY.md](ML_METHODOLOGY.md)).

## Missingness and sentinels

There are no explicit nulls. Missing information is encoded as values:

| Column | Encoding | Share of rows |
|---|---|---|
| poutcome | `"unknown"` | 81.7 % (36,959) |
| contact | `"unknown"` | 28.8 % (13,020) |
| education | `"unknown"` | 4.1 % |
| job | `"unknown"` | 0.6 % |
| pdays | `-1` = never previously contacted | 81.7 % (36,954) |

`pdays = -1` is a sentinel, not a number of days. Both encoders convert it to missing (see [ML_METHODOLOGY.md](ML_METHODOLOGY.md#encoders)).

Exact duplicate rows: 0.

## Invariants enforced on every load (`leadintel.data.validate`)

These hold for all 45,211 rows. Loading fails loudly if any is broken.

1. Columns equal the 17 expected names, in order.
2. No explicit missing values.
3. Every categorical value is in the documented vocabulary (`schema.CATEGORY_LEVELS`).
4. `y ∈ {yes, no}`.
5. `previous == 0` **if and only if** `pdays == -1`.
6. `previous == 0` ⇒ `poutcome == "unknown"`. The reverse does not hold: 5 rows have `previous > 0` with `poutcome = "unknown"`.

Rules 5 and 6 also validate user input in the app and the API: the original app accepted the impossible combination "never contacted, previous outcome = failure".

## Chronological order and drift

The rows are in contact-date order (30 contiguous month runs; the first rows are `may`, the last are `nov`). Splitting the file into row quintiles shows strong drift:

| Row quintile (time →) | Subscription rate | `poutcome` known | `contact = unknown` |
|---|---|---|---|
| 1 (earliest) | 3.4 % | 0.0 % | 100 % |
| 2 | 5.6 % | 0.0 % | 40.2 % |
| 3 | 6.1 % | 8.5 % | 0.9 % |
| 4 | 11.8 % | 33.7 % | 0.0 % |
| 5 (latest) | 31.6 % | 49.0 % | 2.8 % |

Consequences:

* **Era artifacts.** `contact = "unknown"` is effectively a marker for the earliest collection period, and `poutcome` is never known in the first 40 % of rows. Part of what these columns "predict" is *when* a record was collected.
* **Calendar reconstruction.** `day` × `month` together approximate the contact date, so a model can learn period-specific base rates. This shows up as a top SHAP interaction ([EXPERIMENTS.md](EXPERIMENTS.md#4-interaction-effects)).
* **Random splits overstate generalisation.** An out-of-time evaluation is therefore reported alongside the random split ([EXPERIMENTS.md](EXPERIMENTS.md#3-feature-group-ablation)).

## What the dataset does NOT contain

* Call transcripts, messages, CRM notes or customer objections
* Sentiment or conversation-quality signals
* A per-lead timeline of individual touches (only three summary counts: `campaign`, `previous`, `pdays`)
* Agent identity or experience, call direction, interest-rate context. Moro et al. (2014) used these; they are not in `bank-full`.
* Customer identifiers. App lead IDs (`L` + 5-digit original row number) are derived from row position.
* Any data from India or any bank other than the one Portuguese institution.

"Interaction pattern" in this project therefore means **contact history**, not conversation. See [LIMITATIONS.md](LIMITATIONS.md).
