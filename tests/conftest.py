from __future__ import annotations

from typing import Any

import pytest

from leadintel.service import ScoringService


@pytest.fixture(scope="session")
def service() -> ScoringService:
    return ScoringService()


@pytest.fixture
def lead_fields() -> dict[str, Any]:
    """A valid, previously-converted lead (all fields in UCI vocabulary)."""
    return {
        "age": 41, "job": "management", "marital": "married", "education": "tertiary",
        "default": "no", "balance": 1500, "housing": "no", "loan": "no",
        "contact": "cellular", "day": 12, "month": "may",
        "campaign": 1, "pdays": 120, "previous": 2, "poutcome": "success",
    }  # fmt: skip
