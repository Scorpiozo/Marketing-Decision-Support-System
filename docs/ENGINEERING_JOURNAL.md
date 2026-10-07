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

## 2026-10-07 22:20 — Decision layer: 10-90 bootstrap band flagged 21 % for review; switched to 25-75
The follow-up line (10 %) sits where most leads cluster, so an 80 % band flagged one lead
in five, which is unusable. The middle-half band flags ~11 %, and those leads really are
harder (Brier 0.093 inside vs 0.079 outside). Also found that bootstrap members are
less extreme than the full model (an "88 %" lead had range 83-88 %), so the spread is
now centred on the main estimate (`uncertainty.centered_interval`).

## 2026-10-07 22:25 — Single-lead scoring 161 ms -> 24 ms by disabling OpenMP threads per call
Thread start-up dominated one-row predictions across 16 HistGB models + SHAP.
`threadpool_limits(1)` around live scoring; batch work (training) keeps all cores.

## 2026-10-07 22:45 — Web stack: FastAPI + Jinja + HTMX, hand-written CSS instead of Tailwind CDN
Followed the house stack except Tailwind: its docs say the Play CDN is "for development
purposes only" (in-browser runtime compile). A ~300-line token-based stylesheet gives
light/dark themes, no FOUC, no build step. HTMX is vendored (pinned 2.0.10, sha256 in
docs) so the demo works offline. Charts are server-rendered SVG — no JS chart library.
Action colours validated for colour-vision deficiency; light-mode low-contrast hues are
always paired with text labels. axe WCAG 2.2 AA: 0 violations across 8 pages (fixed:
muted text contrast, colour-only links, coloured numbers, focusable scroll regions).

## 2026-10-07 22:50 — Caught two of my own over-claims while reviewing the UI
(1) "A past success matters more when the contact was recent" was asserted from the
interaction *magnitude* only. Checked direction: true for <=200 days (SHAP lift 1.35-1.42
vs 0.40-0.68), but >400-day leads get a big lift regardless — a drift artifact. Copy now
says exactly that. (2) Quantile bins labelled "0-2 contacts" and "1-37 previous" were
wrong; count features now use explicit bins and others show real min-max per bin.

## 2026-10-07 23:40 — Removed a test-set influence on the review band; fixed a policy edge case
A doc reviewer pointed out that the 25-75 % review band had been chosen by looking at
the held-out test set. Added `experiments/review_band.py`: re-split dev 75/25, fit model
+ bootstrap on 75 %, compare bands on 25 %. Result: 25-75 % flags 12.8 % with Brier
0.086 inside vs 0.081 outside, so the same band passes both pre-stated criteria, but the
flag is less informative than the test-set look suggested (0.093 vs 0.079). Kept the
band (it is the conventional IQR), and the docs now state the modest effect and disclose
the earlier look.
Also: a priority threshold set below break-even let below-break-even leads be
"prioritised". `PolicyConfig.priority_line = max(priority, break_even)` + regression test.
