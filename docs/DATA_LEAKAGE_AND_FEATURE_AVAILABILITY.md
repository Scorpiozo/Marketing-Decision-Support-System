# Data leakage and feature availability

**Leakage** is information about the target that would not legitimately be available when the prediction is used (Kaufman, Rosset, Perlich & Stitelman, 2012, *Leakage in data mining: Formulation, detection, and avoidance*, ACM TKDD 6(4), doi:10.1145/2382577.2382579). Their remedy is a strict *learn–predict separation*: only use what is known at prediction time.

**Decision moment in this project:** *before the next follow-up contact for a lead*. The question is "is another call worth making, and how?"

## Per-feature availability

| Feature | Group | Known before the decision? | Status | Decision | Reason |
|---|---|---|---|---|---|
| age, job, marital, education, default, balance, housing, loan | Profile | Yes (customer record) | Safe | Used | Static attributes held by the bank |
| contact | Campaign context | Yes (channel being used) | Era proxy | Used, **flagged as context** | `unknown` occurs almost only in the earliest records ([DATASET.md](DATASET.md#chronological-order-and-drift)) |
| month | Campaign context | Yes (planned contact month) | Era proxy | Used, **flagged as context** | Together with `day` it reconstructs the calendar date |
| day | Campaign context | Yes (planned contact day) | Era proxy | Used, **flagged as context** | Little behavioural meaning; its value comes mainly from the day×month interaction |
| campaign | Interaction history | Yes, as "contacts made so far" | Safe, see note | Used | Counts the last contact too; the count of calls already made is known at decision time |
| pdays | Interaction history | Yes (previous campaign is in the past) | Safe | Used (`-1` → missing) | Pre-campaign information |
| previous | Interaction history | Yes | Safe | Used | Pre-campaign information |
| poutcome | Interaction history | Yes | Safe | Used | Outcome of an *earlier* campaign |
| **duration** | Post-contact | **No.** Known only after the call ends | **Leakage** | **Excluded** | The final call is where the outcome is decided; its length is effectively a consequence of `y` |

"Flagged as context" means the feature stays in the model, but explanations mark it with ⚑. The what-if simulator labels it "context, not a lever", and every lead whose top drivers include it gets a caveat.

## `duration`: evidence for exclusion

1. **UCI's own note.** Attribute 11 in `bank-additional-names.txt`, part of the same dataset-222 archive (the note is absent from `bank-names.txt`): "Important note: this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model."
2. **The dataset authors excluded it.** Moro et al. (2014) say their earlier models only worked "when using attributes that are only known on call execution, such as call duration. Thus, … such models cannot be used for prediction."
3. **Measured in this repo.** Subscription rate by `duration` quintile is 0.7 % → 3.7 % → 7.9 % → 13.1 % → 33.2 %. Held-out results from `reports/experiments/ablation.json`:

| Feature set | Model | ROC-AUC | PR-AUC | Top-20 % recall |
|---|---|---|---|---|
| full (deployed) | Gradient boosting | 0.808 | 0.466 | 63 % |
| full + duration | Gradient boosting | 0.936 | 0.646 | 84 % |
| full (deployed) | Logistic regression | 0.772 | 0.411 | 56 % |
| full + duration | Logistic regression | 0.906 | 0.568 | 76 % |

About 0.13 ROC-AUC comes from a variable that cannot exist at decision time. The legacy project kept `duration` and argued for a "post-interaction" use case. The argument fails because the last call *is* the one that decides the outcome. `duration` survives only as an explicitly labelled benchmark row in the ablation.

## Era proxies are not leakage, but they need caution

`contact`, `month` and `day` are available before a call, so they are not leakage in Kaufman et al.'s sense. In this file, though, they partly encode the **collection period**:

* SHAP interaction strength: day × month 0.212 and channel × month 0.192, second and third behind recency × outcome (0.231). See [EXPERIMENTS.md](EXPERIMENTS.md#4-interaction-effects).
* Out-of-time evaluation: adding campaign context to profile **lowers** ROC-AUC (gradient boosting 0.639 → 0.596; logistic regression 0.634 → 0.616).

Decision: they are kept in the deployed model, which is the best in-period, matching how the demo portfolio is drawn. They are flagged everywhere they appear, and the drift result is a headline limitation ([LIMITATIONS.md](LIMITATIONS.md)).

## Leakage audit checklist

| Check | Status | Where |
|---|---|---|
| `duration` not used in the deployed model or the app/API input schema | ✔ | `features.MODEL_FEATURES`; `schema.Lead` has no `duration`; regression tests in `tests/test_features.py` and `tests/test_schema.py` |
| Preprocessing fitted only on training data | ✔ | Encoders live inside sklearn `Pipeline`s and are refit within every CV fold. `TreeEncoder` is stateless, with its vocabulary taken from the schema rather than the data |
| Hyper-parameters chosen on dev CV only | ✔ | `experiments/tuning.py` uses the dev split |
| Classification threshold for F1 chosen on CV (out-of-fold) predictions, not test | ✔ | `experiments/comparison.py` (`f1_threshold_from_cv`) |
| Policy defaults not tuned on the test set | ✔ | Defaults come from cost assumptions (break-even = cost/value) and observed contact-count rates; the review band was selected on a dev-only validation split, `experiments/review_band.py` ([EXPERIMENTS.md](EXPERIMENTS.md#7-human-review-band)) |
| Test set used for final numbers only; no model selection on it | ✔ / caveat | The model family was selected by CV. The test set is used to *report* the policy back-test (descriptive, not tuned). The review band was first looked at on test data, then re-selected on dev-only data; both are disclosed |
| Interaction-pattern rates shown in the app come from training data | ✔ | `train.pattern_stats(split.dev)` |
| Held-out outcomes not exposed through the API | ✔ | `_api_payload` excludes `actual_outcome`; tested in `tests/test_web.py` |
| Out-of-time check performed | ✔ | `experiments.common.temporal_split` |
