# Gap Analysis

*2026-10-07. Derived from [BASELINE_AUDIT.md](BASELINE_AUDIT.md). Severity: 🔴 critical · 🟠 major · 🟡 minor.*

## Critical bugs
| # | Issue | Evidence | Fix direction |
|---|---|---|---|
| C1 🔴 | `duration` (post-outcome) used as a model input | AUC 0.93 → 0.80 without it | Exclude from the decision model; keep only as a labelled "leakage upper bound" in the ablation |
| C2 🔴 | Displayed probabilities 2–3× too high | WARM leads shown ~57 %, convert 21 % | Train unweighted or recalibrate (isotonic/Platt on a held-out fold); report Brier + reliability curve |
| C3 🔴 | Doesn't install on macOS arm64 | `tensorflow-cpu` has no wheels | `uv` + lockfile; keep TensorFlow out of the serving path |
| C4 🟠 | Form accepts contradictory history (`previous=0` + `poutcome=failure`) | Default form values | Schema validation (Pydantic) with clear error messages |

## ML / data bugs
* 🟠 Accuracy at a 0.5 threshold on balanced-weighted models is the headline. With a 11.7 % base rate, PR-AUC, calibration and top-k lift are the relevant metrics.
* 🟠 Only a random split. The data is chronological with severe drift (3 % → 32 % conversion). No out-of-time evaluation, so generalisation is overstated.
* 🟠 `contact="unknown"` and `month` partly encode *data-collection era*, not behaviour. Their effect must be interpreted carefully.
* 🟡 `pdays=-1` sentinel standard-scaled as a number. Should be split into `was_previously_contacted` + days-since (where known).
* 🟡 CNN is nondeterministic (AUC varies between runs) and trained on an arbitrary column order.
* 🟡 No cross-validation, no confidence intervals, no hyper-parameter search, no validation set for threshold choice.

## Architecture problems
* 🟠 2,089-line single-file UI with ~570 lines of inline CSS; UI, data access and business rules are interleaved.
* 🟠 Duplicate pipelines: `src/`, `train_and_evaluate.py` and the notebook each re-implement steps.
* 🟠 TensorFlow (~600 MB) in the serving path to score one row.
* 🟡 23 MB Random Forest artifact committed; `temp_streamlit.py` (UTF-16 junk copy) committed.
* 🟡 Relative paths that assume CWD = repo root (`"data/bank-full.csv"`, `"models/..."`).
* 🟡 Windows-only launcher that retrains models on every launch.

## Performance problems
* 🟡 Every widget interaction reruns the whole 2,000-line script (Streamlit model). Caching mitigates the data loads, but page transitions still re-execute everything.
* 🟡 Cold start pays a ~3 s TensorFlow import.
* 🟡 Not measured anywhere. No latency evidence existed.

## UX problems
* 🟠 Descriptive analytics dashboard, not a decision workflow. There's no "here's who to call and why".
* 🔴 No real explanation of individual predictions. "Why this result?" is static text.
* 🟠 No confidence/uncertainty, no "human review" state.
* 🟠 Lead table is rank-by-probability only; no action column grounded in policy.
* 🟡 No what-if exploration (the form scores once; you can't see *what changes the decision*).
* 🟡 Streamlit default chrome (Deploy button, hamburger) visible in a "product".

## Research problems
* 🟠 Research question is "can a CNN classify subscription?". The interesting question (do *interaction* signals add decision value?) is never tested.
* 🔴 No ablation (context vs. +campaign vs. +interaction features).
* 🟠 No interpretability analysis, no error analysis, no interaction-effect analysis.
* 🟠 Claims (AUC 0.92) stronger than the deployable signal supports.
* 🟡 CNN choice justified by "faculty requirement", not by evidence. It should be kept as a *controlled experiment* that answers "does convolution help on tabular data?".

## Product gaps
* 🟠 No explicit decision policy: thresholds aren't derived from costs, capacity, or data.
* 🟠 No separation of *prediction* (model) from *decision* (policy) in code or UI.
* 🟡 No evaluation of the decision layer (does "Prioritise" actually concentrate conversions? does "Stop" avoid wasted calls?).
* 🟡 No portfolio/operational view (how many calls does the policy recommend at a given capacity?).

## Kept from the original
Train-only preprocessing fit · honest wording about CNN and dataset limits · small-sample
flags · historical-vs-predicted distinction · the `campaign × previous × poutcome`
interaction idea.
