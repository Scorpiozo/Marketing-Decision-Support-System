# Architecture

One Python package (`leadintel`), two modes: **offline** training/experiments that write
versioned artifacts, and **online** serving that only loads them. No database, queue or
external service is needed: the held-out portfolio is 9,043 rows and fits in memory.

## Data flow

```
                 OFFLINE                                              ONLINE
┌──────────────────────────────────────────┐        ┌──────────────────────────────────────────────┐
│ data/bank-full.csv                       │        │ uvicorn leadintel.web.app:app                 │
│   │ data.load_dataset (schema+invariants)│        │   lifespan: ScoringService(artifacts/) once  │
│   ▼                                      │        │                                              │
│ experiments/common.random_split          │        │ Browser ──GET/POST (HTMX)──▶ FastAPI routes  │
│   80 % dev ──▶ CV, tuning, model choice  │        │                    │                         │
│   20 % held-out ──▶ final eval/portfolio │        │                    ▼                         │
│   │                                      │        │ ScoringService.assess(lead, PolicyConfig)    │
│   ▼ train.main                           │        │   ├ cached row (portfolio) or live score     │
│ HistGB pipeline + 15 bootstrap re-fits   │ ─────▶ │   ├ uncertainty.centered_interval            │
│ TreeSHAP for every held-out lead         │artifacts│  ├ explain.Explainer (TreeSHAP)             │
│ model card (metrics, curves, SHAP, ...)  │        │   ├ patterns.history/intensity               │
└──────────────────────────────────────────┘        │   └ policy.decide ──▶ Decision + reasons     │
 experiments/run_all ─▶ reports/experiments/*.json  │ narrative.* ──▶ plain-English evidence       │
                                                    │ Jinja2 templates + SVG charts ──▶ HTML       │
                                                    │ CRM ──▶ POST /api/v1/assess ──▶ JSON         │
                                                    └──────────────────────────────────────────────┘
```

## Module map (`src/leadintel/`)

| Module | Responsibility |
|---|---|
| `schema.py` | `Lead` Pydantic model: field ranges, category vocabularies, history-consistency rules; no `duration` |
| `config.py` | Repo-relative paths, seeds, split sizes, runtime `Settings` (`APP_PASSWORD`, `SESSION_SECRET`) |
| `data.py` | Load `bank-full.csv` (either UCI format), validate schema and invariants, add `row_order` |
| `features.py` | Feature groups (profile / context / interaction / leakage), `TreeEncoder`, linear encoder |
| `models.py` | Candidate pipelines (LR, tree, RF, HistGB, MLP), tuned hyper-parameters, `FINAL_MODEL` |
| `evaluation.py` | ROC/PR-AUC, Brier, log-loss, ECE, lift, threshold metrics, curves, bootstrap CIs |
| `experiments/common.py` | Random and out-of-time splits, out-of-fold CV, report writing |
| `experiments/tuning.py` | Small CV grid (log-loss) for HistGB, RF, LR |
| `experiments/comparison.py` | All candidates: CV, held-out metrics, latency, size; balanced-weight variants |
| `experiments/ablation.py` | Feature-group ablation, random and out-of-time |
| `experiments/interactions.py` | Additive vs full model, SHAP interaction values, observed tables |
| `experiments/cnn.py` | Legacy 1D CNN re-evaluation and column-permutation test (optional TensorFlow) |
| `experiments/run_all.py` | Reproduces every experiment and the artifacts |
| `explain.py` | Exact TreeSHAP wrapper, per-lead `Explanation`, global importance |
| `uncertainty.py` | Bootstrap interval centred on the main estimate |
| `patterns.py` | Interaction-pattern labels (history × intensity) and the heavy-contact proxy |
| `policy.py` | `PolicyConfig`, `decide` (one lead) and `decide_many` (vectorised, same rules) |
| `service.py` | `ScoringService`: loads artifacts once; assess, portfolio table, policy back-test |
| `narrative.py` | Template sentences built only from assessment numbers |
| `train.py` | Fits the deployed model and writes `artifacts/` with an additivity guard |
| `web/app.py` | FastAPI routes, password gate, policy cookie, JSON API, view helpers |
| `web/charts.py` | Chart geometry (bars, diverging SHAP bars, ROC/PR/reliability) for SVG |
| `web/templates/` | Jinja2 pages and HTMX fragments (`_assessment`, `_whatif_result`, `_lead_table`, `_backtest`) |
| `web/static/` | Hand-written `app.css` (light/dark tokens) and vendored `htmx.min.js` 2.0.10 |

## Request lifecycles

**Lead page `GET /leads/{id}`**
1. Policy read from the `policy` cookie (falls back to defaults if missing/invalid).
2. `ScoringService.assess` finds the lead in the portfolio and uses its **precomputed**
   probability, interval and SHAP values (no model call).
3. `policy.decide` produces action, reasons and expected value; `narrative` writes the summary.
4. Jinja renders the decision card, SHAP chart, pattern and caveats.

**What-if `POST /leads/{id}/whatif` (HTMX, on field change)**
1. Form validated as a `Lead`; validation errors return a message fragment.
2. Fields differ from the cached row, so the lead is scored **live**: pipeline + 15 bootstrap
   members + TreeSHAP inside `threadpool_limits(1)` (single-row thread overhead removed).
3. SHAP deltas split into *fields you changed* and *unchanged fields whose effect shifted*.
4. Fragment swapped into `#whatif-result`.

## Artifacts

| File | Produced by | Contents |
|---|---|---|
| `artifacts/model.joblib` | `train.py` | Pipeline, 15 bootstrap classifiers, feature list |
| `artifacts/portfolio.csv` | `train.py` | Held-out leads: fields, outcome, probability, interval, SHAP per field, pattern |
| `artifacts/model_card.json` | `train.py` | Data SHA-256, versions, metrics, curves, global SHAP, pattern rates, error analysis |
| `reports/experiments/*.json` | experiments | tuning, comparison, ablation, interactions, cnn |
| `reports/performance.json` | `scripts/benchmark.py` | Latency and memory measurements |

## HTTP endpoints

| Method / path | Purpose |
|---|---|
| `GET /`, `/leads`, `/leads/{id}`, `/patterns`, `/model`, `/policy`, `/about` | Pages: Dashboard, Leads, lead page, Insights, Model, Strategy, About (`/method` → 301 to `/about`) |
| `POST /leads/{id}/whatif` | Live re-score fragment |
| `POST /policy` | Update assumptions (cookie) and return back-test fragment |
| `POST /api/v1/assess` | Score any valid lead (JSON in/out) |
| `GET /api/v1/leads/{id}` | Assessment for a portfolio lead |
| `GET /healthz` | Liveness (unauthenticated) |
| `GET/POST /login` | Password gate when `APP_PASSWORD` is set |
| `GET /api/docs` | OpenAPI docs |

## Security

* Optional password gate (`APP_PASSWORD`): every route except `/login`, `/static`, `/healthz`
  requires an **httponly, SameSite=Lax** cookie holding an HMAC-SHA256 token keyed by
  `SESSION_SECRET` + password; comparisons use `hmac.compare_digest`.
* API returns 401 (not a redirect) when unauthenticated.
* Login `next` parameter is restricted to local paths (open-redirect guard, tested).
* No secrets in the repository; `.env` is git- and docker-ignored; `.env.example` documents keys.
* Held-out outcomes (`actual_outcome`) are excluded from API responses (tested).
* Jinja auto-escaping is on; no user input reaches SQL (there is no database).
* Container runs as an unprivileged user; image excludes tests, data, docs and dev tools.

## CRM integration path

`POST /api/v1/assess` already accepts a lead record and returns probability, interval,
action, reasons, SHAP contributions and a summary. A production integration would add:
authenticated service tokens, a mapping from CRM fields to the schema, per-touch event
history as new features (each leakage-audited), outcome feedback for monitoring, and
scheduled retraining. The model/policy separation means business rules can change without
retraining.
