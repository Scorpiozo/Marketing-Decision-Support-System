# Performance

Measured, not estimated. "Before" numbers come from running the legacy app
([BASELINE_AUDIT](BASELINE_AUDIT.md)); "after" numbers from `scripts/benchmark.py`
(`reports/performance.json`, Darwin arm64, Python 3.12.13, 8 cores, local loopback).

## Before vs after

| Measure | Legacy (Streamlit + TensorFlow CNN) | Redesign (FastAPI + HistGB) |
|---|---|---|
| Install on macOS arm64 | **Fails** (`tensorflow-cpu` has no wheel) | `uv sync` from lockfile |
| Launch | Windows `.bat` retrains all models first (26.8 s, 508 MB) | Loads committed artifacts |
| Server ready | 3.3 s (health endpoint) | **2.26 s** cold start to healthy, model loaded |
| Model load | TensorFlow import + CNN load 2.96 s | Artifacts loaded in ~0.26 s at start-up |
| First live prediction | ~1.05 s | 27.1 ms p50 (same path as every what-if) |
| Repeated interaction | ~0.10 s full-script rerun | What-if 27.1 ms p50 / 31.0 ms p95 |
| Explanation per lead | None (static text) | Included in the 27 ms (exact TreeSHAP) |
| Server memory | not measured | 145 MB RSS |

Docker image size: _measured in DEPLOYMENT.md_

## Page latency (server time, 30 requests each)

| Route | p50 | p95 |
|---|---|---|
| `/` overview (policy back-test over 9,043 leads) | 9.6 ms | 13.1 ms |
| `/leads` | 5.9 ms | 9.2 ms |
| `/leads?action=prioritise&page=3` | 4.6 ms | 6.8 ms |
| `/leads/L43013` | 1.0 ms | 1.2 ms |
| `/patterns` | 0.8 ms | 0.9 ms |
| `/model` | 1.3 ms | 1.5 ms |
| `/policy` | 6.7 ms | 7.4 ms |
| `/method` | 0.5 ms | 0.6 ms |
| `POST /leads/{id}/whatif` (50 requests) | 27.1 ms | 31.0 ms |
| `POST /api/v1/assess` (50 requests) | 27.1 ms | 30.1 ms |
| `POST /policy` (re-run back-test) | 6.8 ms | 7.8 ms |

First page after start-up: 29.5 ms (template compilation).

## What was optimised and why

| Change | Effect |
|---|---|
| Training separated from serving; artifacts committed | No training at start-up; deterministic deploys |
| Held-out portfolio scored and SHAP-explained once at training time | Lead pages need no model call (~1 ms) |
| `ScoringService` loaded once in the FastAPI lifespan | No repeated model or CSV loading |
| `threadpool_limits(1)` around single-lead scoring | Pipeline + 15 bootstrap members + SHAP: **161 ms → 24 ms** (thread start-up dominated one-row predictions) |
| Vectorised policy (`decide_many`) | Re-segmenting all 9,043 leads takes a few ms |
| No TensorFlow in serving | Removes a ~3 s import and a large dependency |
| Server-rendered SVG charts, vendored htmx, hand-written CSS | No chart library or CSS compiler in the browser |
| Content-hashed static URLs (`?v=`) | Browsers never use a stale stylesheet after deploy |
| HTMX fragments | Interactions return 2–6 KB instead of a full page |

## Reproduce

```bash
uv run python scripts/benchmark.py   # starts its own server on a free port, writes reports/performance.json
```

Numbers vary with hardware; relative comparisons are the point.
