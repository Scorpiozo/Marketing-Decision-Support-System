# Explainability (XAI)

The goal is that a user can follow the whole chain on one screen:

```
lead data → features → model → probability → explanation → interaction pattern → recommended action
```

Explanations are not a separate "XAI page". They sit inside the lead card next to the decision, and global views are on the **Model** and **Insights** pages.

## Method choice: exact TreeSHAP

| Requirement | Why TreeSHAP on the gradient-boosting model meets it |
|---|---|
| Faithful | SHAP values are additive: `logit(p) = base + Σ contributions`, exactly (Lundberg & Lee, 2017) |
| Exact, not sampled | TreeSHAP computes exact Shapley values for trees in polynomial time (Lundberg et al., 2020) |
| Fast enough for live what-if | ~0.14 ms per lead for HistGB (100 leads in 0.014 s, measured) |
| Per business field | `TreeEncoder` keeps one model column per raw field, so there is one contribution per field, not per one-hot dummy |
| Verified | Additivity is asserted on every training run (max error 1.6e-14) and in `tests/test_model_and_service.py` |

### A pitfall caught and guarded

scikit-learn's HistGB can split natively on categorical features. In this setup, SHAP's `TreeExplainer` (shap 0.52.0) produced attributions that **did not add up** to the prediction, off by up to **2.68 log-odds**, with no warning. Ordinal-encoding categoricals instead gives exact values (error ~2e-15 in the probe). The project therefore uses ordinal codes, and two guards would catch a regression:

1. `train.py` raises if max |base + Σφ − logit(p)| > 1e-6.
2. `test_shap_adds_up_to_prediction` checks the same on 500 leads.

## Local explanation (one lead)

| Element | Built from | Example (lead L43013) |
|---|---|---|
| Waterfall (diverging bars) | SHAP log-odds per field; top 8 + "N other fields" combined | "Days since previous contact: 183 days ago ×4.39" |
| Odds multipliers | `exp(φ)`. Exact reading: "this field multiplied the odds by…" | ×2.33 for previous outcome = success |
| Baseline | Model output for an average lead (`expected_value`) | 7.9 % |
| Raising / lowering lists | Contributions with \|φ\| ≥ 0.2 (odds ×1.22 or more) | "▲ Previous campaign outcome: success — raised the odds ×2.3" |
| Summary sentence | Template filled from history label, estimate, observed pattern rate and top non-context drivers | "Previously converted lead, 2 contacts this campaign. Estimated chance of subscribing: 88%; historically, 64% of the 433 training leads with the same interaction pattern subscribed…" |
| Robustness | Bootstrap interval vs break-even line | "Recommendation is stable. Re-trained versions of the model agree: 87%–90%…" |
| Caveats | Rules: always "no conversation data"; context driver in top 3; no history; pattern n < 100 | "Day of month is among the main drivers…" |

**No LLM is used.** Every sentence is a fixed template filled from model outputs (`narrative.py`, `policy.py`, `service._caveats`), so explanations cannot invent evidence. Odds multipliers were chosen over "percentage points" because SHAP contributions are additive in log-odds, not in probability.

**Context flags.** `contact`, `month` and `day` are marked ⚑ in the waterfall, "⚑ context" in driver lists, and kept out of the summary's "supporting signals", because in this dataset they partly encode the collection period ([DATA_LEAKAGE_AND_FEATURE_AVAILABILITY.md](DATA_LEAKAGE_AND_FEATURE_AVAILABILITY.md)).

## Global explanation

* **Importance:** mean |SHAP| over the 9,043 held-out leads. Top five: contact channel 0.41, days since previous contact 0.27, contact month 0.20, previous outcome 0.18, housing loan 0.14 (log-odds). Context fields are drawn in amber with an explanatory note.
* **Per-value effects** (Interaction patterns page): average odds multiplier for each value or bin, e.g. previous outcome success ×1.83 vs failure ×0.53. Count features use explicit bins (1 / 2 / 3 / 4–5 / 6+), and quantile bins show their real min–max.
* **Interaction effects:** SHAP interaction values, plus an additive-vs-full model comparison (0.775 vs 0.799 CV AUC). The strongest genuine pair is recency × previous outcome. The hypothesised contacts × outcome interaction is weak ([EXPERIMENTS.md](EXPERIMENTS.md#4-interaction-effects)).

## Counterfactual / what-if

The simulator re-scores an edited lead live with the same model, the same bootstrap ensemble and the same SHAP explainer (~27 ms). The result splits changes into:

* **Fields you changed:** how much each edited field's contribution moved.
* **Unchanged fields whose effect shifted:** the model's interaction effects made visible. Example: setting previous outcome to failure also shrinks the lift from "183 days ago" (×0.38), because recency matters much more after a past success.

Invalid combinations (e.g. "never contacted" with a previous outcome) are refused with a reason, not scored.

## Explanations across model families

| Model | Natural explanation | Used? | Notes |
|---|---|---|---|
| Logistic regression | Coefficients (log-odds per unit / per category) | Comparison only | Fully transparent, but 0.036 AUC behind |
| Decision tree (depth 5) | Readable rules | Comparison only | Transparent, weakest ranking (0.728) |
| Random forest | TreeSHAP | Comparison only | Exact but ~26 ms per lead (2.66 s / 100 leads, 200-tree probe) |
| **Gradient boosting** | **TreeSHAP** | **Deployed** | Exact, ~0.14 ms per lead |
| MLP / 1D CNN | KernelSHAP or gradient methods | No | Approximate and slower. For the CNN, the permutation test showed its "local patterns" are meaningless, so filter-based explanations would not be trustworthy |

## Limitations of these explanations

* **Associational, not causal.** SHAP explains what the model learned from 2008–2010 data. "Fewer contacts → higher estimate" does not mean calling less causes subscriptions.
* **Correlated features share credit.** TreeSHAP's path-dependent attribution can split or move credit between correlated fields (e.g. `pdays`, `previous`, `poutcome`). Read groups of related fields together.
* **Era effects look like drivers.** The top global driver (contact channel) is partly a data-collection artifact; this is flagged rather than hidden.
* **Uncertainty of explanations** is not quantified per lead. Only the probability has an interval.

References: Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*, NeurIPS, arXiv:1705.07874. Lundberg et al. (2020), *From local explanations to global understanding with explainable AI for trees*, Nature Machine Intelligence 2:56–67, doi:10.1038/s42256-019-0138-9. Molnar, *Interpretable Machine Learning*, 3rd ed. (2025), https://christophm.github.io/interpretable-ml-book/.
