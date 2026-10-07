from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from leadintel.schema import Lead


def test_valid_lead(lead_fields: dict[str, Any]) -> None:
    assert Lead(**lead_fields).poutcome == "success"


def test_duration_is_not_part_of_a_lead(lead_fields: dict[str, Any]) -> None:
    # Regression: the legacy app required post-call `duration` to score a new lead.
    lead = Lead.model_validate({**lead_fields, "duration": 900})
    assert "duration" not in lead.model_dump()


@pytest.mark.parametrize(
    "changes",
    [
        # Legacy form default: never contacted, yet a previous outcome of "failure".
        {"previous": 0, "pdays": -1, "poutcome": "failure"},
        {"previous": 0, "pdays": 30, "poutcome": "unknown"},
        {"previous": 3, "pdays": -1, "poutcome": "failure"},
    ],
)
def test_inconsistent_history_rejected(
    lead_fields: dict[str, Any], changes: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError, match="Inconsistent history"):
        Lead(**{**lead_fields, **changes})


@pytest.mark.parametrize(
    "field,value", [("age", 12), ("campaign", 0), ("job", "astronaut"), ("month", "may2010")]
)
def test_out_of_range_or_unknown_values_rejected(
    lead_fields: dict[str, Any], field: str, value: Any
) -> None:
    with pytest.raises(ValidationError):
        Lead(**{**lead_fields, field: value})
