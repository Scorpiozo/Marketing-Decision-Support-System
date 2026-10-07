# Testing

## Current result

```
uv run pytest -q -p no:warnings      → 56 passed (≈1.9 s)
uv run ruff check . && ruff format --check .  → clean
uv run pyright                        → 0 errors, 7 warnings
```

The 7 pyright warnings are unresolved `tensorflow` imports in `experiments/cnn.py`; TensorFlow
is an optional dependency group (`uv sync --group dl`). Tests use the committed artifacts,
so no training happens during the test run.

## Test inventory

| File | Layer | What it covers |
|---|---|---|
| `test_schema.py` | Unit / validation | Valid lead; `duration` not part of a lead; inconsistent histories rejected (3 cases); out-of-range and unknown values rejected (4 cases) |
| `test_data.py` | Data | Dataset is `bank-full` (45,211 rows, 11.7 % positive, chronological `row_order`); 2,000 sampled rows validate as `Lead`; broken invariant is detected |
| `test_features.py` | Unit | Deployed features exclude `duration`; tree encoder keeps one column per raw field and maps `pdays = -1` to missing; unknown category → NaN not crash; linear encoder output finite |
| `test_policy.py` | Unit | Each of the six rules (7 parametrised cases); break-even = cost ÷ value; reasons state the assumptions; vectorised policy equals scalar policy on 3,000 random leads; invalid config rejected; regression: priority line can never sit below break-even |
| `test_uncertainty.py` | Unit | Interval always contains the estimate and stays in [0, 1] |
| `test_model_and_service.py` | ML + integration | Artifacts load and feature schema matches; probabilities finite, in range, inside their interval and calibrated (mean within 1 pp of base rate, ECE < 2 %); SHAP additivity; cached vs live scoring agree; end-to-end lead → model → explanation → policy; more contacts drive a low lead to "stop"; back-test partitions the portfolio and "prioritise" converts > 4× "stop" |
| `test_web.py` | Application | All 7 pages render (parametrised); lead page shows decision, explanation, caveat, what-if; unknown lead 404; HTMX table returns a fragment; empty filter state; what-if changes the decision; invalid what-if explained not crashed; policy update sets cookie and changes segmentation; invalid policy shows error; API round-trip; API rejects invalid lead (422); password gate incl. 401 for API and open-redirect guard; euro formatting; per-IP rate limit on live scoring (429) |

### Regression tests (each tied to a bug found in this project)

| Bug | Test |
|---|---|
| Legacy model used post-outcome `duration` | `test_deployed_features_exclude_duration`, `test_duration_is_not_part_of_a_lead` |
| Legacy form scored impossible histories | `test_inconsistent_history_rejected` |
| Legacy probabilities 2–3× too high | `test_portfolio_probabilities_valid_and_calibrated` |
| SHAP silently wrong for native categorical splits | `test_shap_adds_up_to_prediction` |
| Cached portfolio values could drift from live scoring | `test_cached_and_live_scoring_agree` |
| Negative euro values rendered as "€-82,745" | `test_euro_formatting_handles_negatives` |
| Login `next` could redirect off-site | `test_password_gate` (open-redirect assertion) |

## Accessibility

axe-core, WCAG 2.0/2.1/2.2 A and AA tags, on `/`, `/leads`, `/leads/L43013`, `/patterns`,
`/model`, `/policy`, `/method`, `/login`: **0 violations** after these fixes:

* muted text token darkened (contrast was 3.8–4.1:1, needed 4.5:1);
* links inside running text underlined (were colour-only);
* odds numbers moved to text colour with a coloured ▲/▼ glyph (coloured numbers failed contrast);
* small counts inside heat-map cells set to primary text colour;
* horizontally scrollable table wrappers made focusable regions;
* button colour darkened, with a separate dark-mode token pair.

Action colours were checked with a colour-vision-deficiency palette validator; every
coloured mark is paired with a text label.

## Browser end-to-end checks (performed during development)

| Check | Result |
|---|---|
| What-if on L43013: previous outcome → failure, contacts → 8 | 88 % Prioritise → 12 % Follow up — change approach; changed and interaction-shifted fields listed |
| Policy slider: contact cost €5 → €15 | Follow-up line 10 % → 30 %; contacted share 33 % → 12 %; cookie persisted |
| Invalid what-if (previous 0, outcome failure) | Message shown, nothing scored |
| Mobile width 390 px | No horizontal page scroll; nav moves to its own scrollable row |
| Dark mode | Tokens switch; contrast preserved |
| Production Docker image with `APP_PASSWORD` | Unauthenticated → redirect to login; after login all pages 200; process runs as non-root `app` |

## Usability review by persona

| Persona | Question | Where it is answered | Gap / note |
|---|---|---|---|
| First-time user | Where do I start? | Overview headline, 5-step pipeline strip, "Start here: top priority leads" | — |
| First-time user | What does this number mean? | "estimated chance of subscribing · likely range", "How to read this" under SHAP | Odds multipliers still need one sentence of explanation |
| First-time user | What should I do? | Decision card title + next step | — |
| Bank analyst | What should I do with this lead, and why? | Decision card, policy reasons, SHAP drivers, plain-English summary | — |
| Bank analyst | How confident are you? | Likely range bar with policy lines; "stable / borderline" | Interval reflects model uncertainty only, not data drift |
| Bank analyst | Can I compare leads? | Ranked, filterable table with range and action | No side-by-side comparison view |
| Bank analyst | Can I change assumptions? | Decision-policy sliders with live back-test | — |
| Faculty guide | What is novel / what is the interaction component? | Interaction-patterns page; overview findings | — |
| Faculty guide | Why these models? Why not the CNN? | Model & XAI comparison and CNN permutation test | — |
| Faculty guide | What are the limitations? | Method & limits page; caveats on each lead | — |
| Skeptical ML reviewer | Is there leakage? | Ablation row "full+duration — leakage, not deployable"; overview finding | — |
| Skeptical ML reviewer | Is the evaluation honest / calibrated? | CV ± sd, bootstrap CIs, reliability diagram, out-of-time column | Single random seed for the split |
| Skeptical ML reviewer | Do interaction features really help? | Ablation in-period and out-of-time | Gains out-of-time are modest in absolute terms |
| Skeptical ML reviewer | Are explanations trustworthy? | Additivity check, context-feature flags, interaction-direction caveat | SHAP explains the model, not causation (stated) |

## Known testing gaps

* No automated browser (Playwright) suite in the repository; browser checks were manual.
* No load test beyond sequential latency measurement.
* The CNN experiment is not run in CI (optional TensorFlow).
