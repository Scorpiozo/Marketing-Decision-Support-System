# ML methodology

## Problem framing

* **Prediction target:** `y` = the client subscribed to a term deposit (binary, 11.7 % positive).
* **Decision supported:** what to do with a lead before the next contact. The model outputs a **calibrated probability**; a separate, explicit policy turns it into an action ([DECISION_POLICY.md](DECISION_POLICY.md)).
* **Inputs:** 15 fields (profile, campaign context, interaction history). `duration` is excluded ([DATA_LEAKAGE_AND_FEATURE_AVAILABILITY.md](DATA_LEAKAGE_AND_FEATURE_AVAILABILITY.md)).

## Splits

| Split | How | Used for |
|---|---|---|
| Dev / test | Stratified random 80/20, `random_state=42` (`experiments.common.random_split`) | Dev: 36,168 rows. Test: 9,043 held-out leads that become the app's portfolio |
| Cross-validation | 5-fold `StratifiedKFold(shuffle=True, random_state=42)` on dev | Tuning, model comparison (mean ± sd), out-of-fold predictions for thresholds |
| Out-of-time | First 80 % of rows (by date order) train, last 20 % test (`temporal_split`) | Robustness check under drift (test base rate 31.6 %) |

The test split is used once per experiment to report final numbers. Thresholds and hyper-parameters come from dev/CV.

## Encoders

Two encoders, chosen by model family (`features.py`). Both treat `pdays = -1` as **missing**, not as "−1 days".

| Encoder | Models | Categoricals | Numerics | `pdays` |
|---|---|---|---|---|
| `TreeEncoder` (stateless, vocabulary from `schema.CATEGORY_LEVELS`) | Decision tree, random forest, gradient boosting | Fixed ordinal codes; unseen level → NaN | Passed through | NaN when −1 |
| `linear_encoder` (sklearn `ColumnTransformer`) | Logistic regression, MLP, CNN study | One-hot over the fixed vocabulary | `StandardScaler`; `campaign`, `previous` (and `duration` in the benchmark) are `log1p` then scaled | Median-imputed **plus** a missing-indicator column, then scaled |

**Why ordinal codes for trees, not native categorical splits?** scikit-learn's `HistGradientBoostingClassifier` supports native categoricals, but SHAP's `TreeExplainer` returned **wrong** attributions for them in this setup: they did not add up to the prediction, off by up to 2.68 log-odds, with no warning. With ordinal codes, attributions are exact (error ~1e-14). Every training run asserts additivity, and a test guards it ([EXPLAINABILITY.md](EXPLAINABILITY.md)).

## Candidate models

Hyper-parameters were chosen by a small 5-fold CV grid on dev, minimising log-loss (`experiments/tuning.py` → `reports/experiments/tuning.json`).

| Candidate | Settings (final) | Grid searched | Best CV log-loss / ROC-AUC |
|---|---|---|---|
| Logistic regression | `C=0.3`, `max_iter=5000` | C ∈ {0.03, 0.3, 3} | 0.3017 / 0.764 |
| Decision tree | `max_depth=5`, `min_samples_leaf=100` (kept shallow so a human can read it) | not tuned | — |
| Random forest | 300 trees, `min_samples_leaf=10`, `max_features=0.5` | leaf ∈ {3, 10, 30} × max_features ∈ {sqrt, 0.5} | 0.2870 / 0.797 |
| **Gradient boosting (HistGB)** | `learning_rate=0.03`, `max_iter=300`, `max_leaf_nodes=31`, `min_samples_leaf=40`, `l2_regularization=1.0`, no early stopping | lr ∈ {0.03, 0.05, 0.1} × leaves ∈ {15, 31} | **0.2843 / 0.799** |
| MLP (neural baseline) | (64, 32), `alpha=1e-3`, early stopping on 15 % | not tuned | — |
| 1D CNN (legacy architecture) | 2×(Conv1D-BN-Pool) → Dense 64/32, Adam 1e-3, early stopping | not tuned; re-evaluated only | — |

The grid is deliberately modest. Differences inside a family were small (HistGB log-loss 0.2843–0.2919), so the research effort went into features and decisions, not tuning.

## Class imbalance: unweighted training

All deployed candidates train **unweighted** on log-loss. Class-balanced weights or resampling change the effective base rate, which distorts predicted probabilities without improving ranking (van den Goorbergh et al., 2022, *JAMIA* 29(9):1525–1534). Measured here on held-out data:

| Gradient boosting | ROC-AUC | Brier | Calibration error (ECE) | Mean predicted (true 11.7 %) |
|---|---|---|---|---|
| Unweighted (deployed) | 0.808 | 0.0804 | 0.9 % | 11.5 % |
| `class_weight="balanced"` | 0.806 | 0.1532 | 25.3 % | 37.0 % |

The imbalance is handled where it belongs: in the **decision threshold**, derived from costs (Elkan, 2001).

## Calibration

The final model needs **no post-hoc calibration**. Held-out ECE is 0.86 % (10 equal-frequency bins), and Brier is 0.0804 against a 0.1033 baseline that always predicts the base rate (0.117 × 0.883). A calibration layer such as isotonic regression would also break the exact link between SHAP values and the displayed probability, so it was not added. Reliability deciles are in [MODEL_EVALUATION.md](MODEL_EVALUATION.md#calibration).

## Uncertainty (bootstrap ensemble)

* 15 copies of the final classifier, each refit on a bootstrap resample of dev (same encoder, seeds 43–57).
* For a lead, the members' spread is the **25th–75th percentile**, **centred on the main model's estimate**: `[p − (median − q25), p + (q75 − median)]`, clipped to [0, 1]. Centring is needed because resample-trained members are systematically less extreme. Without it, an "88 %" lead showed a range of 83–88 %, which excludes its own estimate.
* This measures **epistemic** uncertainty only: sensitivity to which historical leads were sampled. Outcome noise is already expressed by the probability itself.
* Why 25–75: selected on a dev-only validation split (test untouched). An 80 % band flagged 23 % of validation leads; the interquartile band flags 12.8 %, and flagged leads are modestly harder (Brier 0.086 vs 0.081). See [EXPERIMENTS.md](EXPERIMENTS.md#7-human-review-band).

## Metrics (and why)

| Metric | Why it is reported |
|---|---|
| ROC-AUC (+ 95 % bootstrap CI, 500 resamples) | Ranking quality, comparable with the literature |
| PR-AUC | More informative than ROC when positives are rare (Saito & Rehmsmeier, 2015, *PLoS ONE* 10(3):e0118432); random = base rate |
| Brier, log-loss | Probability quality; the policy multiplies probabilities by money |
| ECE (10 quantile bins) and reliability curve | Are "30 %" leads really 30 %? |
| Lift / recall at top 10 % and 20 % | What a capacity-limited call team would get |
| Precision, recall, F1 at an explicit threshold | Operational trade-off. The threshold is chosen on CV predictions or by the policy, never on test |
| Accuracy | Reported in the JSON for completeness, not used for selection |

## Determinism and reproducibility

* Fixed seeds everywhere (`SEED = 42`). HistGB uses a fixed iteration count (`early_stopping=False`). Re-running `leadintel-train` reproduced identical metrics (AUC 0.808, PR-AUC 0.466, Brier 0.0804) across four runs in development.
* The CNN study (TensorFlow) is not bit-for-bit deterministic, so it reports mean ± sd over seeds.
* Dependencies are locked in `uv.lock`; Python 3.12. The data file is fingerprinted by SHA-256 in `artifacts/model_card.json`.

```bash
uv sync                                   # exact locked environment
uv run leadintel-train                    # final model + artifacts (~30 s)
uv run leadintel-experiments              # tuning, comparison, ablation, interactions, train (~4 min)
uv sync --group dl && uv run python -m leadintel.experiments.cnn   # legacy CNN study (~2 min)
uv run pytest                             # includes calibration and SHAP-additivity regression tests
```
