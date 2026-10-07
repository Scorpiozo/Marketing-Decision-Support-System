# Model evaluation (model card)

Source: `artifacts/model_card.json` (trained 2026-10-07T17:01:48Z), `artifacts/portfolio.csv` and `reports/experiments/*.json`.

## Model details

| Item | Value |
|---|---|
| Model | scikit-learn `HistGradientBoostingClassifier` (lr 0.03, 300 iterations, 31 leaves, min leaf 40, l2 1.0), behind an ordinal `TreeEncoder` |
| Inputs (15) | age, job, marital, education, default, balance, housing, loan, contact, day, month, campaign, pdays, previous, poutcome |
| Excluded | `duration` (post-outcome leakage) |
| Output | P(subscribe), plus a 25–75 % bootstrap interval (15 members, centred) and exact TreeSHAP contributions |
| Training data | 36,168 dev rows of UCI `bank-full` (SHA-256 `b44507c2…6f0`), seed 42 |
| Evaluation data | 9,043 held-out rows (stratified), base rate 11.70 % |
| Software | Python 3.12.13, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, shap 0.52.0 |

## Held-out performance

| Metric | Value |
|---|---|
| ROC-AUC | **0.808** (95 % bootstrap CI 0.792–0.823) |
| PR-AUC | **0.466** (CI 0.434–0.499; random = 0.117) |
| Brier score | 0.0804 (base-rate predictor: 0.1033) |
| Log-loss | 0.280 |
| Expected calibration error (10 bins) | 0.86 % |
| Mean predicted vs observed | 11.54 % vs 11.70 % |
| CV ROC-AUC (5-fold, dev) | 0.799 ± 0.005 |

### Calibration

Reliability by decile of predicted probability (each about 904 leads):

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean predicted | 2.4 % | 3.3 % | 4.0 % | 4.8 % | 5.8 % | 6.9 % | 8.3 % | 10.6 % | 18.6 % | 50.7 % |
| Observed | 1.6 % | 3.7 % | 4.1 % | 4.1 % | 5.6 % | 6.3 % | 7.2 % | 10.5 % | 21.4 % | 52.7 % |

Close agreement throughout. The ninth decile is slightly under-predicted (18.6 % vs 21.4 %).

### Ranking / lift

| Call only the top… | Their subscription rate | Lift vs base | Share of all subscriptions captured |
|---|---|---|---|
| 10 % of leads | 52.7 % | 4.5× | 45.0 % |
| 20 % of leads | 37.0 % | 3.2× | 63.2 % |

### Confusion matrix at the default follow-up line (10 %)

| | Model: follow up | Model: don't |
|---|---|---|
| Subscribed | 738 (TP) | 320 (FN) |
| Did not | 1,699 (FP) | 6,286 (TN) |

Precision 30.3 %, recall 69.8 %, accuracy 77.7 %, 26.9 % of leads flagged. The low line is a deliberate consequence of assumed costs (a missed subscriber is worth more than a wasted call); see [DECISION_POLICY.md](DECISION_POLICY.md).

### Segment calibration under the default policy

| Action segment | Mean estimate | Observed |
|---|---|---|
| Prioritise | 54.5 % | 55.3 % |
| Follow up | 19.2 % | 22.5 % |
| Change approach | 18.9 % | 16.7 % |
| Human review | 9.9 % | 9.7 % |
| Deprioritise | 5.0 % | 4.6 % |
| Stop | 4.7 % | 4.5 % |

## Out-of-time performance

Trained on the earliest 80 % of contacts, tested on the latest 20 % (base rate 31.6 %; [EXPERIMENTS.md](EXPERIMENTS.md#3-feature-group-ablation)):

| Model / features | ROC-AUC | PR-AUC | ECE |
|---|---|---|---|
| GB, deployed features | 0.665 | 0.482 | 19.6 % |
| LR, profile + interaction | 0.723 | 0.552 | 19.3 % |

The deployed model ranks much worse, and is badly miscalibrated, on a later period. That is the most important limitation for any real use.

## Subgroup check

Computed on the held-out portfolio (`artifacts/portfolio.csv`); top 5 jobs by count. This is a basic performance-and-calibration check, **not** a fairness audit.

| Group | n | ROC-AUC | Mean predicted | Observed |
|---|---|---|---|---|
| Age < 30 | 1,003 | 0.804 | 17.1 % | 17.8 % |
| Age 30–59 | 7,703 | 0.791 | 9.9 % | 10.0 % |
| Age 60+ | 337 | 0.772 | 31.9 % | 33.2 % |
| Divorced | 1,037 | 0.752 | 11.0 % | 10.7 % |
| Married | 5,443 | 0.812 | 10.2 % | 10.5 % |
| Single | 2,563 | 0.803 | 14.7 % | 14.7 % |
| admin. | 1,030 | 0.809 | 11.9 % | 11.8 % |
| blue-collar | 1,902 | 0.775 | 7.2 % | 7.2 % |
| management | 1,947 | 0.791 | 12.9 % | 13.4 % |
| services | 806 | 0.749 | 8.9 % | 8.8 % |
| technician | 1,529 | 0.848 | 12.0 % | 11.6 % |

**Reading it honestly.**
* Calibration holds in every subgroup (mean predicted within ~1.5 points of observed), so the probabilities mean the same thing across groups.
* Ranking quality varies (0.749 for services to 0.848 for technicians). Smaller groups (60+: 337 leads) have wide uncertainty; no confidence intervals were computed for subgroups.
* Age, job and marital status are model inputs, so the system treats groups differently by design. Some groups have much higher observed subscription rates (60+: 33.2 % here; students: 28.7 % across the full dataset). Whether using such attributes for contact prioritisation is acceptable is a policy and legal question for a real bank, not something this prototype settles. See [LIMITATIONS.md](LIMITATIONS.md).

## Known failure patterns

* **Leads with no history.** 82.8 % of missed subscribers are new to campaigns (vs 53.1 % of correctly flagged ones), often from the early `contact = unknown` period.
* **Era-driven confidence.** The highest-scored leads, both correct and incorrect, come from the latest part of the file, where base rates were highest.
* **Context fields as drivers.** Contact channel has the largest mean |SHAP| (0.41) but partly encodes the collection period.

## Intended use

* Teaching and research demonstration of separating prediction from decision, with explanations, on a public dataset.
* Ranking and triaging leads *within the same data distribution*, with a human making the call.

## Out-of-scope use

* Real customer contact decisions at any bank, including Indian banks, without retraining on that bank's data, recalibration, governance and legal review.
* Judging a customer's intent (no conversation data).
* Inferring causal effects ("calling less *causes* subscriptions").
* Use without the human-review step, or with the illustrative cost assumptions treated as real.
