# Deployment

## Status

Status: **deployed** on Railway, project `lead-decision-intelligence`, service `web`, environment
`production`: https://web-production-4cf5b6.up.railway.app (public, no password, per-IP rate limit on live scoring). Verified 2026-10-08
with a live smoke test (all pages, lead pages, what-if, strategy, API, errors, assets, latency).

## Local

```bash
uv sync
uv run uvicorn leadintel.web.app:app --reload        # http://127.0.0.1:8000
```

Artifacts (`artifacts/`) are committed, so the app runs without retraining. To rebuild them:
`uv run leadintel-train` (~30 s).

## Docker

The image serves the pre-trained model only: no training, no TensorFlow, no dev tools,
no dataset, runs as an unprivileged user, includes a `/healthz` health check.

```bash
docker build -t leadintel .
docker run --rm -p 8000:8000 \
  -e APP_PASSWORD='choose-a-password' \
  -e SESSION_SECRET="$(openssl rand -hex 32)" \
  leadintel
# open http://localhost:8000 → login page
```

Docker image size (measured 2026-10-07, `docker images`): **1.11 GB**, down from 2.75 GB for the
first draft. The first draft lost space to a `chown -R` layer that duplicated the 650 MB virtualenv,
the uv download cache baked into the image, and `fastapi[standard]` cloud/telemetry extras. The
remaining weight is the scientific stack (scipy, pandas, scikit-learn) plus numba/llvmlite, which
SHAP needs at runtime (~200 MB). Container check: `/healthz` OK, all pages 200, JSON API OK,
runs as non-root user `app`, honours an injected `PORT`, model + portfolio load in 0.54 s.

## Railway (CLI): the procedure actually used

```bash
unset RAILWAY_TOKEN                          # a stale token overrides `railway login`
railway logout && railway login              # interactive, in the browser
git switch redesign && git status            # deploy a clean working tree
railway init --name lead-decision-intelligence
railway add --service web                    # empty service for the app
railway variables --set "PORT=8000" --service web
railway up --service web                     # uploads the repo, builds the Dockerfile
railway domain --service web                 # public URL (target port 8000)
curl https://<domain>/healthz                # {"status":"ok","model_loaded":true}
```

Redeploying a change: commit, then `railway up --service web --ci` (streams the build and
exits when the deploy finishes). `railway logs --service web` is the first stop when
anything breaks. To require a login, also set `APP_PASSWORD` and a random `SESSION_SECRET`.

* Railway detects the root `Dockerfile` automatically.
* Railway injects `PORT`. The Dockerfile `CMD` uses **shell form**
  (`uvicorn ... --port ${PORT}`) so the variable expands; exec-form JSON would pass the
  literal string `${PORT}`.
* `--proxy-headers` is set because Railway terminates TLS in front of the container.
* Secrets live only in Railway variables, never in the repository or the image
  (`.env` is git- and docker-ignored).

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `APP_PASSWORD` | Yes for any public URL | Enables the login gate for every page and the API |
| `SESSION_SECRET` | Yes when `APP_PASSWORD` is set | Key for the HMAC session cookie; long random value |
| `PORT` | Set by the platform | Listening port (defaults to 8000 in the image) |
| `ARTIFACTS_DIR` | No | Override artifact location (defaults to `artifacts/`) |

## Security checklist

- [x] Password gate on every page and API route; `/healthz` and static files only are open
- [x] httponly, SameSite=Lax session cookie; constant-time comparisons
- [x] Login redirect restricted to local paths
- [x] No secrets committed; `.env.example` documents the keys
- [x] Held-out outcomes never returned by the API
- [x] Container runs as non-root; no dataset or tests in the image
- [ ] Rate limiting on `/login` (not implemented; acceptable for a gated demo)

## Verify a deployment

```bash
curl -s https://<domain>/healthz                     # {"status":"ok","model_loaded":true}
curl -s -o /dev/null -w "%{http_code}\n" https://<domain>/   # 303 to /login when gated
```

Then walk the demo path: Dashboard → a lead → what-if → Strategy → Model.
