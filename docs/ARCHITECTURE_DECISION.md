# Architecture Decision Records

## ADR-1 — Web architecture: FastAPI + server-rendered Jinja2 + HTMX (replacing Streamlit)

*Status: accepted, implemented 2026-10-07.*

### Context: the legacy architecture

A single 2,089-line `app/streamlit_app.py` (about 570 lines of inline CSS) mixed UI, data
access and business rules. It loaded a TensorFlow CNN to score one row, required post-call
`duration` as input, and launched through a Windows-only script that retrained every model
first. Measured: server ready 3.3 s; first live prediction ~1.05 s (TensorFlow load);
subsequent interactions ~0.10 s per full-script rerun ([BASELINE_AUDIT](BASELINE_AUDIT.md)).

Streamlit's execution model (official docs):
* "any time something must be updated on the screen, Streamlit reruns your entire Python
  script from top to bottom" — on every widget interaction
  (https://docs.streamlit.io/get-started/fundamentals/main-concepts);
* `st.cache_data` / `st.cache_resource` mitigate repeated loading; `st.cache_resource` objects
  are shared singletons and the docs warn about non-thread-safe objects
  (https://docs.streamlit.io/develop/concepts/architecture/caching);
* `st.fragment` (since 1.37) reruns only part of a page, with limitations, e.g. it cannot be
  combined with caching on the same function
  (https://docs.streamlit.io/develop/concepts/architecture/fragments).

### Options

| | A. Optimised Streamlit | B. FastAPI + Jinja2 + HTMX | C. FastAPI + React/Next.js | D. Static site + serverless scoring |
|---|---|---|---|---|
| Development complexity | Low | Low–medium | High (two codebases, build step) | Medium |
| Interaction latency | Full-script reruns (mitigated) | One small fragment per request | Fast after JS bundle | Network per call |
| UX control (layout, a11y, URLs) | Limited by widget model | Full HTML/CSS control, real URLs | Full | Full |
| Deployment | Simple | One container | Two artifacts or SSR server | Two platforms |
| ML integration | Direct Python | Direct Python | Through API | Through API |
| Reproducibility | Good | Good, no JS build | npm toolchain to pin | Mixed |
| Future CRM integration | Separate API needed | Same app exposes JSON API | API exists | API exists |
| Academic explainability of the code | Good | Good | Harder for a non-JS student | Medium |

### Decision

**Option B.** The app is a read-mostly decision workflow over a 9k-row in-memory
portfolio; every interaction (filter, what-if, policy slider) is "send a few fields, get a
small HTML fragment back". HTMX does exactly that without a JavaScript build step, and the
same FastAPI process exposes the JSON API a CRM would call. Server-side SVG charts need no
chart library.

### Consequences and trade-offs

* + Pages render in 0.5–10 ms server time; what-if re-score 27 ms p50 ([PERFORMANCE](PERFORMANCE.md)).
* + Real URLs per lead (`/leads/L43013`), browser back/forward, works without JavaScript for reading.
* + WCAG 2.2 AA clean under axe (8 pages) — achievable because the markup is ours.
* − Charts are simpler than an interactive charting library (hover tooltips via SVG `<title>`).
* − More template code than Streamlit for the same page.

**Styling: hand-written CSS instead of the Tailwind CDN.** The house stack lists Tailwind
via CDN, but Tailwind's documentation states "The Play CDN is designed for development
purposes only, and is not intended for production"
(https://tailwindcss.com/docs/installation/play-cdn) because it compiles CSS in the browser.
A ~300-line token-based stylesheet gives light/dark themes, no flash of unstyled content and
no build step. HTMX 2.0.10 is vendored into `static/` (works offline; one fewer runtime
dependency) and assets are cache-busted with a content hash.

**Deliberately not added:** React/Next.js (no need for client state; would add a toolchain);
a database (portfolio fits in memory, no writes needed); Redis/caching layer (nothing is slow);
microservices/Kubernetes (one process serves everything); an LLM (ADR-3).

### Migration plan (as executed)

1. Audit and measure the legacy app; tag it `legacy-v1`.
2. Rebuild the ML core as a tested package; artifacts written by a training command.
3. Build the web app on the service layer; verify each page in a browser and with axe.
4. Remove legacy files from the tree (preserved at the tag).
5. Containerise without TensorFlow; deploy behind a password.

## ADR-2 — Model library: scikit-learn HistGradientBoosting + SHAP (LightGBM rejected)

*Status: accepted.*

* **LightGBM** gives native TreeSHAP via `pred_contrib`, but its macOS wheel needs a system
  OpenMP library (`brew install libomp`); importing it failed on the development Mac. An
  extra system step on every examiner's laptop conflicts with "reproducible from a clean
  environment".
* **scikit-learn HistGradientBoosting** has no system dependencies, matched the best model in
  experiments (held-out ROC-AUC 0.808 vs Random Forest 0.802) and was well calibrated
  unweighted (calibration error 0.9 %), so no post-hoc calibration layer is needed — the SHAP
  values therefore explain the exact probability shown.
* **The SHAP pitfall:** `shap.TreeExplainer` returned values for HistGB with *native
  categorical* splits that did not add up to the model output (max additivity error 2.68
  log-odds, no warning). With ordinal-coded categoricals the error is ~1e-15. Decision:
  `TreeEncoder` ordinal-codes categoricals with a fixed vocabulary; `train.py` aborts if
  additivity error exceeds 1e-6, and a regression test checks it.
* TensorFlow is an optional `dl` dependency group used only for the legacy CNN experiment.

## ADR-3 — No LLM

The core value is grounded evidence: probabilities, SHAP contributions, observed pattern
rates and explicit policy rules. An LLM would add cost, latency, an API key and a risk of
fabricated explanations, without adding information. Explanations are templates filled
only from computed values (`narrative.py`). Revisit only if real free-text CRM notes become
available — and then only to summarise evidence that is computed elsewhere.

## ADR-4 — Prediction separated from decision

The model outputs P(subscribe); `policy.py` owns all thresholds as editable assumptions
(break-even = cost ÷ value, priority threshold, heavy-contact proxy) plus an uncertainty
rule for human review. Class imbalance is handled in thresholds, not in training weights,
because balanced weights inflated probabilities (Brier 0.0804 → 0.1532 for HistGB) with no
ranking gain ([EXPERIMENTS](EXPERIMENTS.md)).
