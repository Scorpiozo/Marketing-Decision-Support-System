# Deployment

## Status

Status: not yet deployed (awaiting Railway login) — updated after deploy.

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

Docker image size: _to be recorded after the slimmed build_.

## Railway (CLI)

```bash
railway login                       # interactive, once
railway init                        # create a project (or `railway link` to an existing one)
railway variables --set "APP_PASSWORD=..." --set "SESSION_SECRET=..."
railway up                          # builds the Dockerfile and deploys
railway domain                      # generate a public URL
railway logs                        # first stop when anything breaks
railway status
```

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

Then log in and walk the demo path: Overview → a lead → what-if → Model & XAI → Method & limits.
