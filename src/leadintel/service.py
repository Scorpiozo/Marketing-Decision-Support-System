"""ScoringService: the one place where model, explanation, patterns and policy meet.

Loaded once per process. Portfolio leads use pre-computed scores and SHAP values (instant);
what-if leads are scored live in ~25 ms (single-threaded, see `_live`).
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from threadpoolctl import threadpool_limits

from leadintel.config import REPORTS_DIR, Settings
from leadintel.explain import Explainer, Explanation
from leadintel.patterns import history_label, intensity_label
from leadintel.policy import (
    ACTION_ORDER,
    ACTION_TITLE,
    Action,
    Decision,
    PolicyConfig,
    decide,
    decide_many,
)
from leadintel.schema import Lead
from leadintel.uncertainty import centered_interval

NO_CONVERSATION_CAVEAT = (
    "This dataset has no call transcripts, messages or sentiment. The estimate uses profile "
    "and contact-history fields only, so it cannot judge the customer's stated intent."
)


class Pattern(BaseModel):
    history: str
    intensity: str
    observed_rate: float | None  # conversion rate of similar leads in the training data
    observed_n: int


class Assessment(BaseModel):
    lead_id: str | None
    lead: Lead
    probability: float
    interval: tuple[float, float]
    explanation: Explanation
    pattern: Pattern
    decision: Decision
    caveats: list[str]
    actual_outcome: int | None = None  # only for held-out portfolio leads (back-testing)


class ArtifactsMissing(RuntimeError):
    pass


class ScoringService:
    def __init__(self, artifacts_dir: Path | None = None) -> None:
        d = artifacts_dir or Settings().artifacts_dir
        try:
            bundle = joblib.load(d / "model.joblib")
            self.portfolio = pd.read_csv(d / "portfolio.csv").set_index("lead_id", drop=False)
            self.card: dict[str, Any] = json.loads((d / "model_card.json").read_text())
        except FileNotFoundError as e:
            raise ArtifactsMissing(
                f"Model artifacts not found in {d}. Run `uv run leadintel-train` first."
            ) from e
        self.pipeline = bundle["pipeline"]
        self.members = bundle["members"]
        self.features: list[str] = bundle["features"]
        self.explainer = Explainer(self.pipeline, self.features)
        self._shap_cols = [f"shap_{f}" for f in self.features]
        self._pattern_rates = {
            (s["history"], s["intensity"]): (s["conversion_rate"], s["n"])
            for s in self.card["pattern_stats_dev"]
        }
        lo, hi = self.card["interval_quantiles"]
        self._quantiles = (float(lo), float(hi))

    # ---- single lead -------------------------------------------------------------

    def _live(self, lead: Lead) -> tuple[float, tuple[float, float], np.ndarray]:
        row = pd.DataFrame([lead.model_dump()])[self.features]
        # One row is far too small to benefit from OpenMP threads; thread start-up made
        # single-lead scoring ~7x slower (161 ms -> 24 ms with one thread).
        with threadpool_limits(1):
            p = float(self.pipeline.predict_proba(row)[0, 1])
            enc = self.pipeline.named_steps["prep"].transform(row)
            spread = np.array([[m.predict_proba(enc)[0, 1]] for m in self.members])
            phi = self.explainer.shap_values(row)[0]
        lo, hi = centered_interval(np.array([p]), spread, self._quantiles)
        return p, (float(lo[0]), float(hi[0])), phi

    def assess(self, lead: Lead, config: PolicyConfig, lead_id: str | None = None) -> Assessment:
        """Score a lead. Portfolio leads use cached values unless their fields were edited."""
        cached = self._row(lead_id) if lead_id in self.portfolio.index else None
        if cached is not None and self.lead_from_row(cached) == lead:
            p = float(cached["probability"])
            interval = (float(cached["p_low"]), float(cached["p_high"]))
            phi = cached[self._shap_cols].to_numpy(dtype=float)
            actual = int(cached["y"])
        else:
            p, interval, phi = self._live(lead)
            actual = None
        row = pd.Series(lead.model_dump())
        explanation = self.explainer.explain_row(row, phi)
        pattern = self.pattern(lead)
        decision = decide(p, interval, lead.campaign, config)
        return Assessment(
            lead_id=lead_id,
            lead=lead,
            probability=p,
            interval=interval,
            explanation=explanation,
            pattern=pattern,
            decision=decision,
            caveats=self._caveats(explanation, pattern),
            actual_outcome=actual,
        )

    def pattern(self, lead: Lead) -> Pattern:
        h, i = history_label(lead.previous, lead.poutcome), intensity_label(lead.campaign)
        rate, n = self._pattern_rates.get((h, i), (None, 0))
        return Pattern(history=h, intensity=i, observed_rate=rate, observed_n=n)

    def _caveats(self, explanation: Explanation, pattern: Pattern) -> list[str]:
        caveats = [NO_CONVERSATION_CAVEAT]
        top_context = [
            c.label
            for c in explanation.contributions[:3]
            if c.is_context and c.strength != "slight"
        ]
        if top_context:
            caveats.append(
                f"{' and '.join(top_context)} {'is' if len(top_context) == 1 else 'are'} among the "
                "main drivers. In this dataset, contact channel, month and day partly reflect "
                "*when* the data was collected (strong drift over 2008–2010), not how the "
                "customer behaved."
            )
        if pattern.history == "New to campaigns":
            caveats.append(
                "No previous-campaign history: the estimate relies on profile and campaign "
                "context, which carry weaker signal."
            )
        if pattern.observed_n < 100:
            caveats.append(
                f"Only {pattern.observed_n} similar leads in the training data for this "
                "interaction pattern; the observed rate is noisy."
            )
        return caveats

    @staticmethod
    def lead_from_row(row: pd.Series) -> Lead:
        return Lead.model_validate(row.to_dict())

    def _row(self, lead_id: str | None) -> pd.Series:
        row = self.portfolio.loc[lead_id]
        assert isinstance(row, pd.Series), "lead ids are unique"
        return row

    def lead(self, lead_id: str) -> Lead:
        return self.lead_from_row(self._row(lead_id))

    # ---- portfolio ---------------------------------------------------------------

    def actions(self, config: PolicyConfig) -> pd.Series:
        pf = self.portfolio
        acts = decide_many(
            pf["probability"].to_numpy(),
            pf["p_low"].to_numpy(),
            pf["p_high"].to_numpy(),
            pf["campaign"].to_numpy(),
            config,
        )
        return pd.Series(acts, index=pf.index, name="action")

    def policy_backtest(self, config: PolicyConfig) -> dict[str, Any]:
        """How the policy segments the held-out leads, judged against what actually happened."""
        pf = self.portfolio.assign(action=self.actions(config))
        total_conv = int(pf["y"].sum())
        rows = []
        for a in ACTION_ORDER:
            g = pf[pf["action"] == a.value]
            rows.append(
                {
                    "action": a.value,
                    "title": ACTION_TITLE[a],
                    "n": len(g),
                    "share": len(g) / len(pf),
                    "observed_rate": float(g["y"].mean()) if len(g) else None,
                    "mean_probability": float(g["probability"].mean()) if len(g) else None,
                    "conversions": int(g["y"].sum()),
                    "share_of_conversions": float(g["y"].sum() / total_conv),
                }
            )
        contact = pf["action"].isin(
            [Action.PRIORITISE, Action.FOLLOW_UP, Action.CHANGE_APPROACH, Action.REVIEW]
        )
        v, c = config.conversion_value, config.contact_cost

        def value(mask: pd.Series) -> float:
            return float(pf.loc[mask, "y"].sum() * v - mask.sum() * c)

        everyone = pd.Series(True, index=pf.index)
        stop_or_dep = pf["action"].isin([Action.STOP, Action.DEPRIORITISE])
        return {
            "rows": rows,
            "n": len(pf),
            "total_conversions": total_conv,
            "contacted_share": float(contact.mean()),
            "conversions_captured": float(pf.loc[contact, "y"].sum() / total_conv),
            "value_policy": value(contact),
            "value_contact_everyone": value(everyone),
            # A random selection of the same size would capture this many conversions:
            "random_same_size_conversions": float(contact.mean() * total_conv),
            "calls_saved": int(stop_or_dep.sum()),
            "conversions_forgone": int(pf.loc[stop_or_dep, "y"].sum()),
        }

    def table(
        self,
        config: PolicyConfig,
        action: str | None = None,
        history: str | None = None,
        query: str | None = None,
        sort: str = "probability",
        page: int = 1,
        per_page: int = 25,
    ) -> tuple[pd.DataFrame, int]:
        pf = self.portfolio.assign(action=self.actions(config))
        if action:
            pf = pf[pf["action"] == action]
        if history:
            pf = pf[pf["history"] == history]
        if query:
            q = query.strip().upper()
            pf = pf[pf["lead_id"].str.contains(q, regex=False)]
        if sort == "contacts":
            pf = pf.sort_values(["campaign", "probability"], ascending=[False, False])
        else:
            pf = pf.sort_values("probability", ascending=False)
        total = len(pf)
        start = (max(page, 1) - 1) * per_page
        return pf.iloc[start : start + per_page], total

    # ---- research results (read-only) ----------------------------------------------

    @cached_property
    def experiments(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in ("comparison", "ablation", "interactions", "cnn", "tuning"):
            path = REPORTS_DIR / "experiments" / f"{name}.json"
            if path.exists():
                out[name] = json.loads(path.read_text())
        return out
