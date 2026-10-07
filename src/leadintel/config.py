"""Paths and runtime settings. Everything resolves from the repo root, never the CWD."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "bank-full.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

SEED = 42
TEST_SIZE = 0.20  # held-out "portfolio" the app displays; never used for fitting or tuning
CV_FOLDS = 5
N_BOOTSTRAP = 15  # bootstrap re-fits used to estimate model (epistemic) uncertainty


class Settings(BaseSettings):
    """Runtime settings from the environment / .env (only what the web app needs)."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    artifacts_dir: Path = ARTIFACTS_DIR
    # When set, the web app requires this password (httponly cookie session).
    app_password: str | None = None
    session_secret: str = "dev-only-change-me"
