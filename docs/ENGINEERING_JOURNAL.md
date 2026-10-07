# Engineering Journal

Timestamped trace of decisions, findings and dead ends — newest at the bottom.
(This is the project's `journal.md`; it lives in `docs/` alongside the other design docs.)

## 2026-10-07 16:00 — Phase 0: inherited a legacy repo; audited before touching anything
Ran the original training pipeline (27 s, 508 MB) and Streamlit app in an isolated venv.
The original `requirements.txt` doesn't install on macOS arm64 (`tensorflow-cpu` has no
wheels); swapped to `tensorflow` just to reproduce. Reproduced the metric table to within
CNN noise (CNN AUC 0.9215 committed vs 0.9180 re-run, so the CNN is nondeterministic).

## 2026-10-07 16:20 — The headline AUC of 0.92 is mostly `duration` leakage
Dropping `duration` takes AUC from 0.93 to 0.80 across LR/RF/boosting. The original
README argued for a "post-interaction" use case, but the last call *is* the sale, so its
length is a consequence of the label. Decision: the decision model will **not** use
`duration`. It survives only as an explicitly-labelled "leakage upper bound" row in the
ablation, which is more persuasive to an examiner than quietly dropping it.

## 2026-10-07 16:30 — Probabilities shown to users are 2–3× too high
`class_weight="balanced"` inflates probabilities: "WARM" leads displayed at ~57 % convert
at 21 %. Balanced weights help a 0.5-threshold classifier but damage probability
estimates. A decision-support tool needs honest probabilities, so the new models train
unweighted (log-loss) and calibration is measured (Brier + reliability) and corrected only
if needed. Thresholds move into an explicit decision policy instead of being hidden in
class weights.

## 2026-10-07 16:40 — The data is chronological with heavy drift
Row-quintile conversion: 3.4 → 5.6 → 6.1 → 11.8 → 31.6 %. `contact="unknown"` is 100 %
of the first quintile (collection artifact). So a random split overstates
generalisation, and some "effects" (contact=unknown, month) are partly era effects.
Plan: random stratified split for model selection (comparable to literature) **plus** an
out-of-time evaluation as a robustness study, and say plainly in the limitations what that shows.

## 2026-10-07 21:30 — Rejected LightGBM (macOS libomp), chose sklearn HistGB + SHAP — and caught a silent SHAP bug
LightGBM needs `brew install libomp` on macOS: an extra system step for every examiner's
laptop. sklearn's HistGradientBoosting has no system deps. But SHAP's TreeExplainer gives
**wrong** values for HistGB *native categorical* splits (additivity error 2.68 log-odds,
no warning). With ordinal-coded categoricals it is exact (1e-15) and ~0.14 ms/row.
Decision: ordinal `TreeEncoder` + HistGB, and a unit test that asserts SHAP additivity so
this can never silently regress.

## 2026-10-07 22:05 — Experiments 1-2: GB wins narrowly; interaction history is the *robust* signal
Random split, no duration: GB AUC 0.808 / PR-AUC 0.466 / Brier 0.0804 / ECE 0.009 —
already well calibrated, so **no post-hoc calibration layer** (keeps SHAP explaining the
exact probability shown). RF is within noise but 14 MB and slower; LR 0.772.
Balanced weights: same AUC, Brier 0.080 → 0.153, ECE 0.009 → 0.253 — the legacy
miscalibration reproduced and explained.
Ablation: +interaction over profile+context raises PR-AUC 0.395 → 0.466. Out-of-time
(train early 80 %, test latest 20 %): campaign context *hurts* (0.639 → 0.596) because
month/channel encode collection era; profile+interaction is the most robust set (LR 0.723).
Decision: deploy the full GB model (best in-distribution, which is how the demo
portfolio is drawn), but label month/channel as "context, not levers" in the UI and make
the drift result a headline limitation, not a footnote.
