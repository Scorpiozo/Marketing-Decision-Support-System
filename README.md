# Lead Decision Intelligence

**An interpretable, interaction-pattern-based decision-support prototype that tells a bank's
sales team which leads deserve another call, which need a different approach, and which
should be left alone — and shows the evidence behind every recommendation.**

**Live demo:** [https://web-production-4cf5b6.up.railway.app](https://web-production-4cf5b6.up.railway.app)

| | |
|---|---|
| Data | UCI Bank Marketing, `bank-full` (45,211 phone-campaign contacts, Portuguese bank, 2008–2010) |
| Model | Gradient boosting (scikit-learn HistGB), unweighted, call `duration` excluded as leakage |
| Held-out quality | ROC-AUC **0.808** · PR-AUC **0.466** (base rate 0.117) · Brier **0.0804** · calibration error **0.9 %** |
| Decision back-test | Contacting **33 %** of held-out leads captures **74 %** of actual subscriptions; "Prioritise" leads subscribe at **55 %** |
| Explanations | Exact TreeSHAP per lead (verified additivity error 1.6e-14), global SHAP, interaction values, live what-if |
| Stack | FastAPI + Jinja2 + HTMX, server-rendered SVG charts, `uv`, `ruff`, `pyright`, 59 tests |

This is a **research prototype for human decision support**. It is not an autonomous
decision-maker, it is not connected to any bank, and it makes no regulatory claim.

---

## The problem

Banks contact the same leads repeatedly. Some are worth another call, some need a
different approach, and some have been called too often already. The original version of
this project answered *"will this customer subscribe?"* with a CNN and a 0.92 AUC. An audit
([BASELINE_AUDIT](docs/BASELINE_AUDIT.md)) found that this number came mostly from `duration`
(the length of the final call, known only after the outcome), that displayed probabilities
were 2–3× too high, and that no recommendation was explained.

This redesign asks the research question:

> Can customer and campaign **interaction patterns** be turned into an **interpretable lead
> decision system** that decides who deserves continued engagement, who needs a different
> approach, and whose follow-up should stop?

## Why interaction patterns matter

In this dataset an interaction pattern is the observable contact history of a lead:
previous-campaign outcome (`poutcome`), contacts before this campaign (`previous`), days
since that contact (`pdays`) and contacts made in this campaign (`campaign`).

* A previous-campaign success converts at roughly 6–7× the rate of a lead new to campaigns
  (training split: 68 % vs 11 % at one contact).
* Conversion falls with repeated contact in every history group (e.g. previously converted
  68 % → 49 %, new leads 11 % → 7 % from 1 to 4+ contacts).
* Adding interaction history to profile + campaign context raises held-out PR-AUC from
  0.395 to 0.466, and it is the most robust feature group when testing on a later period.

The dataset contains **no** transcripts, messages, sentiment or CRM notes. "Interaction"
here means contact history, not conversation — see [LIMITATIONS](docs/LIMITATIONS.md).

## What the system does

| Page | Answers |
|---|---|
| Dashboard | KPIs, recommended-action mix, targeting-power (gains) curve, conversion by history, top leads |
| Leads | All scored leads ranked by likelihood, with action filters and search |
| Lead page | Likelihood gauge, decision card (range, confidence, value), SHAP explanation, interaction profile, **what-if simulator** |
| Insights | History × contact-intensity heat map, signal-group lift, conversion by contacts / history / age, per-value SHAP effects |
| Model | Six-model comparison (incl. 1D CNN), ROC, predicted-vs-actual, confusion matrix, global SHAP |
| Strategy | Editable business assumptions with live action mix and outcomes |
| About | Workflow, data, model, technology and integration API |

Recommended actions: **Prioritise · Follow up · Follow up — change approach · Human review ·
Deprioritise · Stop repeated follow-up.**

## Architecture

```
OFFLINE  (uv run leadintel-train, ~30 s)              ONLINE  (one FastAPI process)
data/bank-full.csv                                     Browser ──HTMX──▶ routes ──▶ Jinja2 HTML + SVG charts
  └─ validate schema + invariants                                          │
  └─ stratified split 80 % dev / 20 % held-out                             ▼
  └─ fit HistGB + 15 bootstrap re-fits                  ScoringService ─┬─ predict_proba       (estimate)
  └─ score + TreeSHAP the held-out portfolio                            ├─ bootstrap ensemble  (uncertainty)
  └─ artifacts/: model.joblib, portfolio.csv,                          ├─ TreeSHAP            (explanation)
     model_card.json                                                   ├─ patterns            (history label)
                                                                       └─ policy.decide        (action + reasons)
uv run leadintel-experiments → reports/experiments/*.json      CRM ──▶ POST /api/v1/assess (JSON)
```

Prediction (model) and decision (policy) are separate modules. Details:
[ARCHITECTURE](docs/ARCHITECTURE.md) · [ARCHITECTURE_DECISION](docs/ARCHITECTURE_DECISION.md).

## Dataset

UCI Bank Marketing (id 222), `bank-full` variant: 45,211 rows, 16 inputs + target `y`
(term-deposit subscription, 11.7 % yes), ordered by date May 2008 – Nov 2010 with strong
drift (row-quintile conversion 3.4 % → 31.6 %). `duration` is excluded from every deployed
model. See [DATASET](docs/DATASET.md) and
[DATA_LEAKAGE_AND_FEATURE_AVAILABILITY](docs/DATA_LEAKAGE_AND_FEATURE_AVAILABILITY.md).

## Models (held-out test set, n = 9,043, no `duration`)

| Model | ROC-AUC | PR-AUC | Brier | Notes |
|---|---|---|---|---|
| **Gradient Boosting (HistGB)** — deployed | **0.808** | **0.466** | **0.0804** | 417 KB, calibrated without post-processing |
| Random Forest | 0.802 | 0.461 | 0.0810 | statistically tied, 14 MB |
| Neural network (MLP) | 0.794 | 0.440 | 0.0822 | appropriate tabular neural baseline |
| 1D CNN (legacy architecture, 3 seeds) | 0.792 | 0.429 | 0.0832 | shuffling input columns: 0.789 — no spatial structure used |
| Logistic Regression | 0.772 | 0.411 | 0.0851 | transparent baseline |
| Decision Tree (depth 5) | 0.728 | 0.323 | 0.0894 | readable tree |
| GB with balanced class weights | 0.806 | 0.468 | 0.1532 | same ranking, probabilities inflated (mean 37 % vs 11.7 %) |

Method: 5-fold CV on the dev split for selection, one held-out evaluation with bootstrap CIs.
See [ML_METHODOLOGY](docs/ML_METHODOLOGY.md), [EXPERIMENTS](docs/EXPERIMENTS.md),
[MODEL_EVALUATION](docs/MODEL_EVALUATION.md).

## Decision intelligence

The model estimates P(subscribe). An explicit, editable policy turns it into an action:
break-even line = contact cost ÷ conversion value (defaults €5 / €50 = 10 %, **illustrative
assumptions**), a priority threshold (35 %), a "heavily contacted" proxy (≥ 4 contacts this
campaign), and **Human review** when bootstrap re-fits of the model disagree about which
side of the break-even line a lead falls on. Back-test on held-out leads:

| Action | Leads | Actually subscribed | Share of all subscriptions |
|---|---|---|---|
| Prioritise follow-up | 747 | 55 % | 39 % |
| Follow up | 1,091 | 22 % | 23 % |
| Follow up — change approach | 162 | 17 % | 3 % |
| Human review recommended | 999 | 10 % | 9 % |
| Deprioritise | 4,512 | 5 % | 20 % |
| Stop repeated follow-up | 1,532 | 5 % | 7 % |

See [DECISION_POLICY](docs/DECISION_POLICY.md).

## Explainability

* **Local:** exact TreeSHAP for each lead, shown as odds multipliers that add up to the estimate.
* **Global:** mean |SHAP| per field and average effect per value.
* **Interactions:** SHAP interaction values; blocking all interactions lowers CV ROC-AUC 0.799 → 0.774.
* **Counterfactual:** the what-if simulator re-scores an edited lead and separates fields you
  changed from unchanged fields whose effect shifted (interaction effects made visible).
* **Guarded:** SHAP silently mis-explains scikit-learn boosting with *native* categorical
  splits (additivity error 2.68 log-odds); categoricals are ordinal-encoded and a test
  asserts additivity on every run.

See [EXPLAINABILITY](docs/EXPLAINABILITY.md).

## Results in one paragraph

Without leakage, gradient boosting reaches ROC-AUC 0.808 and PR-AUC 0.466 with
near-perfect calibration (mean estimate 11.5 % vs 11.7 % actual). Interaction history adds
measurable value and generalises to a later period better than campaign context. The
legacy CNN performs like a generic MLP and is not used in production. The cost-based policy
concentrates 74 % of subscriptions into a third of the leads and flags about 11 % of leads
for human review where the model is genuinely uncertain.

## Demo

A 5-minute walkthrough script (overview → lead L43013 → what-if → model → limitations) is in
[FACULTY_VIVA_GUIDE](docs/FACULTY_VIVA_GUIDE.md).

## Screenshots

Stored in `docs/screenshots/`: `overview.png`, `lead.png`, `model.png`.

## Local setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is installed by uv automatically.

```bash
uv sync                                         # install locked dependencies
uv run leadintel-train                          # (optional) retrain; artifacts are committed
uv run uvicorn leadintel.web.app:app --reload   # http://127.0.0.1:8000
uv run leadintel-experiments                    # reproduce all experiments (~4 min)
uv sync --group dl && uv run leadintel-experiments --cnn   # + legacy CNN study (TensorFlow)
```

## Deployment

Production image: `Dockerfile` (no training, no TensorFlow, non-root). Railway via CLI;
set `APP_PASSWORD` and `SESSION_SECRET` for any public URL. See [DEPLOYMENT](docs/DEPLOYMENT.md).

## Testing

```bash
uv run pytest            # 59 tests: unit, ML, integration, web, regression
uv run ruff check . && uv run ruff format --check .
uv run pyright           # 0 errors
```

Accessibility: axe WCAG 2.2 AA, 0 violations across 8 pages. See [TESTING](docs/TESTING.md)
and [PERFORMANCE](docs/PERFORMANCE.md).

## Limitations

No conversation data; one Portuguese bank, 2008–2010; strong time drift (out-of-time ROC-AUC
falls to about 0.67–0.72); channel/month/day partly encode the collection period; SHAP
explains the model, not causation; costs are assumed. Full list: [LIMITATIONS](docs/LIMITATIONS.md).

## Future work

Lead-level CRM event histories and conversation signals (each audited for leakage), a
time-based validation and recalibration schedule, real cost and contact-frequency rules,
fairness review across customer groups, and feedback capture of what relationship managers
actually did.

## Academic and research context

Redesign of an earlier student project (original preserved at git tag `legacy-v1`). The
audit, experiments and decisions are traced in [ENGINEERING_JOURNAL](docs/ENGINEERING_JOURNAL.md).

* Dataset: Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C5K306 — licence CC BY 4.0.
* Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of
  bank telemarketing. *Decision Support Systems*, 62, 22–31.
  https://doi.org/10.1016/j.dss.2014.03.001
* Further sources: [RESEARCH_NOTES](docs/RESEARCH_NOTES.md).

## Documentation index

| Document | Contents |
|---|---|
| [PRODUCT_VISION](docs/PRODUCT_VISION.md) | Problem, personas, user stories, failure modes, scenarios |
| [RESEARCH_NOTES](docs/RESEARCH_NOTES.md) | Verified sources and what they imply |
| [BASELINE_AUDIT](docs/BASELINE_AUDIT.md) | Measured state of the original implementation |
| [GAP_ANALYSIS](docs/GAP_ANALYSIS.md) | What was wrong and how it was addressed |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | Components, data flow, modules, API, security |
| [ARCHITECTURE_DECISION](docs/ARCHITECTURE_DECISION.md) | Why FastAPI + HTMX, why HistGB + SHAP, why no LLM |
| [DATASET](docs/DATASET.md) | Variables, invariants, drift |
| [DATA_LEAKAGE_AND_FEATURE_AVAILABILITY](docs/DATA_LEAKAGE_AND_FEATURE_AVAILABILITY.md) | Per-feature availability audit |
| [ML_METHODOLOGY](docs/ML_METHODOLOGY.md) | Splits, pipelines, tuning, reproducibility |
| [EXPERIMENTS](docs/EXPERIMENTS.md) | Comparison, ablation, interactions, CNN study |
| [MODEL_EVALUATION](docs/MODEL_EVALUATION.md) | Metrics, calibration, error analysis |
| [DECISION_POLICY](docs/DECISION_POLICY.md) | Rules, thresholds, assumptions, back-test |
| [EXPLAINABILITY](docs/EXPLAINABILITY.md) | Local/global SHAP, interactions, what-if |
| [PERFORMANCE](docs/PERFORMANCE.md) | Before/after latency and memory |
| [TESTING](docs/TESTING.md) | Test inventory, accessibility, usability review |
| [DEPLOYMENT](docs/DEPLOYMENT.md) | Docker, Railway, environment, security checklist |
| [LIMITATIONS](docs/LIMITATIONS.md) | What this prototype cannot claim |
| [LEAD_INTELLIGENCE_FINAL_HANDOFF_GUIDE](docs/LEAD_INTELLIGENCE_FINAL_HANDOFF_GUIDE.md) ([.docx](docs/LEAD_INTELLIGENCE_FINAL_HANDOFF_GUIDE.docx)) | **Start here:** complete handoff, user guide, architecture, deployment and viva kit |
| [FACULTY_VIVA_GUIDE](docs/FACULTY_VIVA_GUIDE.md) | Plain-language explanation and examiner Q&A |
| [ENGINEERING_JOURNAL](docs/ENGINEERING_JOURNAL.md) | Timestamped decisions and findings |
