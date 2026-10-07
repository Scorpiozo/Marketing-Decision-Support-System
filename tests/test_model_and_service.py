"""Integration tests: artifacts -> model -> explanation -> policy (no web layer)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from leadintel.policy import PolicyConfig
from leadintel.schema import Lead
from leadintel.service import ScoringService


def test_artifacts_load_and_schema_matches(service: ScoringService) -> None:
    assert service.features == service.card["features"]
    assert "duration" not in service.features
    assert len(service.portfolio) == service.card["data"]["test_rows"]


def test_portfolio_probabilities_valid_and_calibrated(service: ScoringService) -> None:
    pf = service.portfolio
    assert np.isfinite(pf["probability"]).all()
    assert pf["probability"].between(0, 1).all()
    assert ((pf["p_low"] <= pf["probability"]) & (pf["probability"] <= pf["p_high"])).all()
    # Regression: legacy probabilities averaged 0.23-0.30 against an 11.7 % base rate.
    assert pf["probability"].mean() == pytest.approx(pf["y"].mean(), abs=0.01)
    assert service.card["test_metrics"]["ece"] < 0.02


def test_shap_adds_up_to_prediction(service: ScoringService) -> None:
    # Regression: SHAP silently mis-explains HistGB *native categorical* splits.
    pf = service.portfolio.sample(500, random_state=0)
    phis = service.explainer.shap_values(pf[service.features])
    p = service.pipeline.predict_proba(pf[service.features])[:, 1]
    logits = np.log(p / (1 - p))
    assert np.abs(service.explainer.base_value + phis.sum(axis=1) - logits).max() < 1e-6


def test_cached_and_live_scoring_agree(service: ScoringService) -> None:
    lead_id = service.portfolio.index[123]
    lead = service.lead(lead_id)
    cached = service.assess(lead, PolicyConfig(), lead_id)
    live = service.assess(lead, PolicyConfig(), lead_id=None)
    assert live.probability == pytest.approx(cached.probability, abs=1e-4)
    assert live.interval == pytest.approx(cached.interval, abs=1e-4)
    assert live.decision.action == cached.decision.action
    assert [c.feature for c in live.explanation.contributions[:5]] == [
        c.feature for c in cached.explanation.contributions[:5]
    ]


def test_end_to_end_assessment(service: ScoringService, lead_fields: dict[str, Any]) -> None:
    a = service.assess(Lead(**lead_fields), PolicyConfig())
    assert 0 < a.probability < 1
    assert a.pattern.history == "Previously converted"
    assert a.explanation.probability == pytest.approx(a.probability, abs=1e-3)
    assert a.decision.reasons and a.caveats
    # previous-campaign success should push the estimate up
    poutcome = next(c for c in a.explanation.contributions if c.feature == "poutcome")
    assert poutcome.direction == "up"


def test_more_contacts_never_raise_the_action_for_low_leads(service: ScoringService) -> None:
    lead = service.lead(service.portfolio.sort_values("probability").index[0])
    tired = lead.model_copy(update={"campaign": 8})
    assert service.assess(tired, PolicyConfig()).decision.action.value == "stop"


def test_backtest_partitions_portfolio(service: ScoringService) -> None:
    b = service.policy_backtest(PolicyConfig())
    assert sum(r["n"] for r in b["rows"]) == b["n"]
    assert sum(r["share_of_conversions"] for r in b["rows"]) == pytest.approx(1.0)
    prioritise = next(r for r in b["rows"] if r["action"] == "prioritise")
    stop = next(r for r in b["rows"] if r["action"] == "stop")
    assert prioritise["observed_rate"] > 4 * stop["observed_rate"]
