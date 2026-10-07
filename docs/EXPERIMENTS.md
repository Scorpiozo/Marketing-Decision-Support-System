# Experiments

All experiments use the leak-free feature set unless stated otherwise. Unless noted, numbers are read from `reports/experiments/*.json`, which these commands regenerate:

```bash
uv run leadintel-experiments          # experiments 1–4 + final training
uv sync --group dl && uv run python -m leadintel.experiments.cnn   # experiment 5
```

Setup common to all: stratified 80/20 dev/test split (seed 42), 5-fold CV on dev, base rate 11.7 %. See [ML_METHODOLOGY.md](ML_METHODOLOGY.md).

---

## 1. Hyper-parameter tuning

**Question.** Are the candidates reasonably configured before comparing them?
**Setup.** Small grid, 5-fold CV log-loss on dev (`python -m leadintel.experiments.tuning`).

| Model | Best params | CV log-loss | CV ROC-AUC | Worst in grid (log-loss) |
|---|---|---|---|---|
| Gradient boosting | lr 0.03, 31 leaves, min leaf 40, l2 1.0, 300 iters | 0.2843 | 0.799 | 0.2919 |
| Random forest | min leaf 10, max_features 0.5 | 0.2870 | 0.797 | 0.2933 |
| Logistic regression | C 0.3 | 0.3017 | 0.764 | 0.3025 |

**Interpretation.** Configuration changes within a family move CV AUC by only ~0.01, much less than the differences between feature sets (experiment 3).

## 2. Model comparison

**Question.** Which model family gives the best combination of ranking, probability quality, speed and explainability?
**Setup.** `python -m leadintel.experiments.comparison`. CV = mean ± sd over 5 folds. Test CI = 95 % bootstrap (500 resamples). F1 uses the F1-optimal threshold from out-of-fold predictions. Latency = one `predict_proba` call on one row. Size = joblib, compress=3.

| Model | CV ROC-AUC | Test ROC-AUC [95 % CI] | Test PR-AUC | Brier | ECE | Mean pred. | F1 | Top-10 % conv. | Latency | Size |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | 0.764 ± 0.010 | 0.772 [0.754–0.789] | 0.411 | 0.0851 | 1.3 % | 11.7 % | 0.445 | 49.0 % | 3.3 ms | 3 KB |
| Decision tree (depth 5) | 0.716 ± 0.004 | 0.728 [0.711–0.744] | 0.323 | 0.0894 | 3.8 % | 11.7 % | 0.369 | 39.9 % | 1.4 ms | 3 KB |
| Random forest | 0.797 ± 0.005 | 0.802 [0.787–0.817] | 0.461 | 0.0810 | 1.4 % | 11.6 % | 0.487 | 52.1 % | 16.8 ms | 14,306 KB |
| **Gradient boosting (HistGB)** | **0.799 ± 0.005** | **0.808 [0.792–0.823]** | **0.466** | **0.0804** | **0.9 %** | 11.5 % | **0.501** | **52.7 %** | 23.3 ms | 417 KB |
| MLP | 0.777 ± 0.011 | 0.794 [0.778–0.808] | 0.440 | 0.0822 | 0.8 % | 11.0 % | 0.484 | 51.2 % | 2.9 ms | 130 KB |
| LR, balanced weights | 0.765 ± 0.010 | 0.772 [0.756–0.789] | 0.410 | 0.1831 | 29.8 % | 41.5 % | 0.446 | 48.3 % | 2.9 ms | 3 KB |
| GB, balanced weights | 0.797 ± 0.005 | 0.806 [0.791–0.822] | 0.468 | 0.1532 | 25.3 % | 37.0 % | 0.495 | 51.5 % | 24.7 ms | 419 KB |
| 1D CNN (legacy arch., 3 seeds) | — | 0.792 ± 0.004 | 0.429 | 0.0832 | — | — | — | — | — | — |

**Interpretation.**
* Gradient boosting and random forest are statistically tied on ranking (overlapping CIs). Gradient boosting is chosen for its best Brier and ECE, a 34× smaller artifact, and exact, fast TreeSHAP (0.14 ms/row vs ~26 ms/row for the forest; [EXPLAINABILITY.md](EXPLAINABILITY.md)).
* Logistic regression, the transparent baseline, is ~0.04 AUC behind.
* Single-row latency is dominated by sklearn per-call overhead. In the app, single-lead scoring runs single-threaded: 24–27 ms for the estimate + 15 bootstrap members + SHAP ([PERFORMANCE.md](PERFORMANCE.md)).

## 3. Feature-group ablation

**Question (central hypothesis).** Do interaction-history signals add value beyond the customer profile and campaign context?
**Setup.** `python -m leadintel.experiments.ablation`. Groups: profile (8 fields), campaign context (contact, day, month), interaction history (campaign, pdays, previous, poutcome). Random split with CV, plus an **out-of-time** split (train earliest 80 % of rows, test latest 20 %, base rate 31.6 %).

| Feature set | Model | Test ROC-AUC | Test PR-AUC | Brier | Top-20 % recall | **Out-of-time ROC-AUC** | Out-of-time PR-AUC |
|---|---|---|---|---|---|---|---|
| profile | LR | 0.666 | 0.214 | 0.0991 | 39 % | 0.634 | 0.411 |
| profile | GB | 0.699 | 0.265 | 0.0958 | 45 % | 0.639 | 0.414 |
| profile + context | LR | 0.746 | 0.326 | 0.0915 | 51 % | 0.616 | 0.418 |
| profile + context | GB | 0.793 | 0.395 | 0.0850 | 60 % | 0.596 | 0.407 |
| profile + interaction | LR | 0.731 | 0.357 | 0.0894 | 49 % | **0.723** | **0.552** |
| profile + interaction | GB | 0.748 | 0.395 | 0.0874 | 53 % | 0.687 | 0.493 |
| **full (deployed)** | LR | 0.772 | 0.411 | 0.0851 | 56 % | 0.686 | 0.492 |
| **full (deployed)** | **GB** | **0.808** | **0.466** | **0.0804** | **63 %** | 0.665 | 0.482 |
| full + duration (leakage benchmark) | LR | 0.906 | 0.568 | 0.0694 | 76 % | 0.798 | 0.591 |
| full + duration (leakage benchmark) | GB | 0.936 | 0.646 | 0.0606 | 84 % | 0.794 | 0.590 |

**Interpretation.**
* **Supported:** interaction history adds value. Over profile alone it lifts GB PR-AUC 0.265 → 0.395. Over profile + context it adds a further 0.395 → 0.466 PR-AUC and 60 % → 63 % top-20 % recall.
* **Supported, and stronger out-of-time:** interaction history is the *most robust* group. Profile + interaction (LR) is the best non-leaky out-of-time model (0.723). Campaign context **hurts** out-of-time (0.639 → 0.596 for GB), consistent with it encoding the collection period.
* **Caveat:** every model degrades out-of-time, and calibration fails there (ECE 0.17–0.23) because the base rate jumps from ~8 % to 31.6 %. A deployed system would need regular retraining and recalibration.

### 3b. Dropping calendar proxies (exploratory probe)

Run interactively in the development session (not yet a scripted experiment; same splits and models):

| Features | GB test ROC / PR | GB out-of-time ROC | LR test ROC / PR | LR out-of-time ROC |
|---|---|---|---|---|
| full | 0.808 / 0.466 | 0.665 | 0.772 / 0.411 | 0.686 |
| − day | 0.798 / 0.450 | 0.642 | 0.772 / 0.412 | 0.693 |
| − day, month | 0.768 / 0.409 | 0.682 | 0.753 / 0.368 | 0.717 |
| − day, month, contact | 0.748 / 0.395 | 0.687 | 0.731 / 0.357 | 0.723 |

**Interpretation.** No free lunch. Removing calendar proxies costs 0.01–0.06 AUC in-period and helps out-of-time mainly for logistic regression. The deployed model keeps them, flagged as context, and this trade-off is documented in [LIMITATIONS.md](LIMITATIONS.md).

## 4. Interaction effects

**Question.** Do interaction-pattern variables *interact*, or act independently?
**Setup.** `python -m leadintel.experiments.interactions`. (a) Same GB with `interaction_cst="no_interactions"` vs unconstrained (5-fold CV). (b) TreeSHAP interaction values on 1,500 held-out leads. (c) Observed conversion tables on dev.

**(a) Additive vs full**

| GB variant | CV ROC-AUC | CV PR-AUC | CV log-loss |
|---|---|---|---|
| Additive (no interactions) | 0.7745 | 0.411 | 0.3005 |
| Full | 0.7990 | 0.451 | 0.2843 |

Interactions are worth ~0.025 AUC. 52 % of total SHAP attribution mass sits in off-diagonal (interaction) terms.

**(b) Strongest pairs (mean |SHAP interaction|, log-odds)**

| Pair | Strength | Reading |
|---|---|---|
| Days since previous contact × previous outcome | 0.231 | Genuine interaction-history effect |
| Day of month × contact month | 0.212 | Calendar reconstruction → era effect |
| Contact channel × contact month | 0.192 | Era effect |
| Housing loan × contact month | 0.120 | Mostly era |
| Contacts this campaign × previous outcome | **0.010** | Hypothesised; **not supported** |
| Contacts before × previous outcome | 0.012 | Weak |

**Direction of recency × outcome (verified in-session).** Mean SHAP of `pdays` on portfolio leads by previous outcome and recency: for contacts within ~200 days, recency lifts the estimate **1.35–1.42** log-odds after a past success vs **0.40–0.68** after a past failure. Leads last contacted more than 400 days ago get a large lift whatever the outcome (1.77–1.86). Such leads exist only late in the file, so this is attributed to drift.

**(c) Observed conversion (dev), history × contacts this campaign**

| Previous campaigns | 1 contact | 2–3 | 4+ |
|---|---|---|---|
| Previously converted | 67.6 % (678) | 63.5 % (433) | 48.9 % (94) |
| Previously declined | 13.4 % (1,939) | 12.0 % (1,503) | 8.3 % (447) |
| Previously contacted, outcome unclear | 19.1 % (581) | 17.1 % (613) | 10.5 % (296) |
| New to campaigns | 11.2 % (10,828) | 8.9 % (11,886) | 6.6 % (6,870) |

**Interpretation.** Repeated contact lowers conversion roughly in parallel across all histories. That matches the weak campaign × outcome SHAP interaction: contact fatigue (proxy) acts mostly on its own.

## 5. Legacy 1D CNN

**Question.** Is a convolutional network justified on this tabular data?
**Setup.** `python -m leadintel.experiments.cnn` (needs `uv sync --group dl`). Legacy architecture, unweighted loss, input = 51 one-hot/scaled features reshaped to (51, 1). Three seeds in the original column order, plus three random column permutations (falsification test: a CNN exploiting local structure should suffer when neighbours are shuffled).

| Input order | ROC-AUC (mean ± sd) | PR-AUC | Brier |
|---|---|---|---|
| Original | 0.792 ± 0.004 | 0.429 | 0.0832 |
| Randomly permuted | 0.789 ± 0.002 | 0.437 | 0.0830 |

Mean training time: 12.5 s per CNN fit (vs mean per-fold fit times of 4.2 s for GB and 0.7 s for the MLP in experiment 2).

**Interpretation.** Permutation makes no meaningful difference, so the convolution exploits no spatial or sequential structure; the column order is arbitrary. The CNN behaves like a generic neural network, roughly matching the MLP (0.794) and trailing gradient boosting (0.808). This agrees with Grinsztajn et al. (2022) and Shwartz-Ziv & Armon (2022). Kept as a documented experiment, not deployed. The legacy run was also nondeterministic (committed AUC 0.9215 vs re-run 0.9180, both with `duration`).

## 6. Decision-policy back-test

**Question.** Does the policy produce useful segmentation on leads the model never saw?
**Setup.** Default policy (contact €5, subscription €50 → follow-up line 10 %, priority 35 %, heavy contact = 4). Evaluated on the 9,043 held-out leads against their real outcomes.
`uv run python -c "from leadintel.service import ScoringService; from leadintel.policy import PolicyConfig; print(ScoringService().policy_backtest(PolicyConfig()))"`

| Action | Leads | Share | Mean estimate | Actually subscribed | Share of all subscriptions |
|---|---|---|---|---|---|
| Prioritise | 747 | 8.3 % | 54.5 % | 55.3 % | 39.0 % |
| Follow up | 1,091 | 12.1 % | 19.2 % | 22.5 % | 23.2 % |
| Follow up — change approach | 162 | 1.8 % | 18.9 % | 16.7 % | 2.6 % |
| Human review | 999 | 11.0 % | 9.9 % | 9.7 % | 9.2 % |
| Deprioritise | 4,512 | 49.9 % | 5.0 % | 4.6 % | 19.6 % |
| Stop repeated follow-up | 1,532 | 16.9 % | 4.7 % | 4.5 % | 6.5 % |

Contacting 33.2 % of leads (including review) captures **73.9 %** of subscriptions; a random selection of the same size would capture ~33 %. 6,044 calls avoided, at the cost of 276 of 1,058 subscriptions. Mean estimate ≈ observed rate in every segment, so calibration holds at segment level. Sensitivity analysis: [DECISION_POLICY.md](DECISION_POLICY.md#sensitivity).

## 7. Human-review band

**Question.** How wide should the bootstrap interval be before a decision counts as "not robust"?
**Criteria (set in advance).** A manageable review load (roughly a tenth of leads), and flagged leads must be measurably harder to predict.

**7a. Selection on dev-only data (decisive)** — `uv run python -m leadintel.experiments.review_band` → `reports/experiments/review_band.json`. The dev split is re-split 75/25; model + 15 bootstrap members are fit on the 75 % and evaluated on the 25 %. The test set is not touched.

| Band (centred) | Leads flagged | Their observed rate | Brier inside flag | Brier outside |
|---|---|---|---|---|
| 10–90 % | 23.3 % | 9.7 % | 0.0866 | 0.0803 |
| 20–80 % | 15.7 % | 9.1 % | 0.0828 | 0.0816 |
| **25–75 % (chosen)** | **12.8 %** | 9.5 % | **0.0855** | **0.0812** |
| 30–70 % | 10.2 % | 9.5 % | 0.0857 | 0.0813 |

**Decision.** 25–75 % (the conventional interquartile range, not a tuned value) satisfies both criteria. The information carried by the flag is **modest**: flagged leads are only somewhat harder (Brier 0.086 vs 0.081). The flag's main job is to mark leads sitting on the break-even line, where either action is defensible.

**7b. Earlier exploratory look on held-out data (disclosed).** Before 7a existed, the band was first examined on the test set (uncentred quantiles):

| Band | Leads flagged | Their observed rate | Brier inside flag | Brier outside |
|---|---|---|---|---|
| 10–90 % | 20.9 % | 10.2 % | 0.0907 | 0.0776 |
| 20–80 % | 13.4 % | 10.1 % | 0.0894 | 0.0790 |
| **25–75 % (chosen)** | **10.6 %** | 10.4 % | **0.0931** | **0.0789** |

That look is why 7a was added: a choice made by looking at test data is not an independent validation. The dev-only result supports the same band, with a smaller (but still positive) difference in difficulty.

## 8. Error analysis

At the follow-up line (10 %), on held-out leads (`artifacts/model_card.json → error_analysis`):

| Group | n | Mean estimate | New to campaigns | Channel unknown | Median contacts |
|---|---|---|---|---|---|
| Correctly flagged (TP) | 738 | 41.3 % | 53.1 % | 1.6 % | 1 |
| Flagged, did not subscribe (FP) | 1,699 | 23.0 % | 72.3 % | 1.4 % | 2 |
| Missed subscribers (FN) | 320 | 6.1 % | **82.8 %** | **28.4 %** | 2 |

* Missed subscribers are mostly leads with **no previous-campaign history**, often from the early `contact = unknown` period. The model has little interaction signal for them, which is where conversational CRM data would help most.
* The most confident false positives (e.g. L45098, L43135) and the top true positives (L41241, L42986) both come from the latest rows, where base rates were highest: the model has learned the era.
