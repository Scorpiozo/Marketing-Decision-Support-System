# Production image: serves the pre-trained model. No training, no TensorFlow, no dev tools.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# Unprivileged user owns /app from the start, so no costly `chown -R` layer is needed.
RUN useradd --create-home app && mkdir /app && chown app /app
USER app
WORKDIR /app

# Dependencies first (cached layer): locked, main group only.
COPY --chown=app pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Application code + trained artifacts + the experiment reports the UI displays.
COPY --chown=app src ./src
COPY --chown=app artifacts ./artifacts
COPY --chown=app reports ./reports
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/healthz')"
# Shell form so $PORT (injected by Railway and most PaaS) expands.
CMD uvicorn leadintel.web.app:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips="*"
