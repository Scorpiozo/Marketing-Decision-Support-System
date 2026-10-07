"""View models: turn service outputs into chart geometry and display-ready values.

Kept separate from app.py so routes stay thin and each page's data is easy to read.
"""

from __future__ import annotations

import math
from typing import Any

from leadintel import narrative
from leadintel.models import DISPLAY_NAMES, FINAL_MODEL
from leadintel.policy import ACTION_NEXT_STEP, Action, PolicyConfig
from leadintel.service import Assessment, ScoringService
from leadintel.web import charts

FEATURE_SET_LABELS = {
    "profile": "Customer profile only",
    "profile+context": "Profile + campaign context",
    "profile+interaction": "Profile + interaction history",
    "full": "All signals (deployed)",
}


def action_donut(backtest: dict[str, Any]) -> charts.Chart:
    return charts.donut(
        [
            (r["title"], r["share"], f"act-{r['action']}",
             f"{r['title']}: {r['n']:,} leads ({r['share']:.0%}), "
             f"{(r['observed_rate'] or 0):.0%} subscribed")
            for r in backtest["rows"]
            if r["n"]
        ]
    )  # fmt: skip


def gains_chart(s: ScoringService, contacted_share: float, captured: float) -> charts.Chart:
    chart = charts.xy({"model": s.insights["gains"]}, width=560, height=300, pad=56)
    pad, pw, ph = chart.extras["pad"], chart.extras["pw"], chart.extras["ph"]
    chart.extras["marker"] = {
        "x": pad + pw * contacted_share,
        "y": 12 + ph * (1 - captured),
        "label": f"{contacted_share:.0%} contacted → {captured:.0%} reached",
    }
    return chart


def rate_columns(rows: list[dict[str, Any]], css: str = "") -> charts.Chart:
    return charts.columns(
        [
            (r["label"], r["rate"], f"{r['rate']:.0%}",
             f"{r['label']}: {r['rate']:.1%} subscribed ({r['n']:,} leads)")
            for r in rows
        ],
        css=css,
    )  # fmt: skip


def dashboard(s: ScoringService, cfg: PolicyConfig) -> dict[str, Any]:
    backtest = s.policy_backtest(cfg)
    top, _ = s.table(cfg, action=Action.PRIORITISE.value, per_page=6)
    history = s.insights["by_history"]
    return {
        "backtest": backtest,
        "donut": action_donut(backtest),
        "gains": gains_chart(s, backtest["contacted_share"], backtest["conversions_captured"]),
        "history_chart": charts.hbars(
            [
                (
                    r["label"],
                    r["rate"],
                    f"{r['rate']:.0%}",
                    f"{r['label']}: {r['rate']:.1%} subscribed ({r['n']:,} leads)",
                )
                for r in history
            ],
            vmax=1.0,
            label_w=250,
        ),
        "top": top.to_dict("records"),
        "card": s.card,
        "lift": history[0]["rate"] / s.insights["base_rate"] if history else None,
    }


def insights_page(s: ScoringService) -> dict[str, Any]:
    abl = {
        r["feature_set"]: r
        for r in s.experiments.get("ablation", {}).get("random_split", [])
        if r["model"] == FINAL_MODEL and r["feature_set"] in FEATURE_SET_LABELS
    }
    signal_chart = charts.hbars(
        [
            (label, abl[k]["test"]["pr_auc"], f"{abl[k]['test']['pr_auc']:.3f}",
             f"{label}: precision-recall score {abl[k]['test']['pr_auc']:.3f}")
            for k, label in FEATURE_SET_LABELS.items()
            if k in abl
        ],
        vmax=0.5,
        label_w=230,
        css=["muted-bar", "muted-bar", "muted-bar", ""],
    ) if abl else None  # fmt: skip
    gain = None
    if "profile+context" in abl and "full" in abl:
        before, after = abl["profile+context"]["test"]["pr_auc"], abl["full"]["test"]["pr_auc"]
        gain = after / before - 1
    grid = {(r["history"], r["intensity"]): r for r in s.card["pattern_stats_dev"]}
    return {
        "grid": grid,
        "contacts_chart": rate_columns(s.insights["by_contacts"]),
        "history_chart": rate_columns(s.insights["by_history"]),
        "age_chart": rate_columns(s.insights["by_age"]),
        "signal_chart": signal_chart,
        "signal_gain": gain,
        "importance": s.card["global_importance"],
    }


def model_page(s: ScoringService) -> dict[str, Any]:
    card = s.card
    cmp = [r for r in s.experiments.get("comparison", {}).get("results", []) if not r["balanced"]]
    cnn = s.experiments.get("cnn", {}).get("original_order_mean_std")
    rows = [
        {
            "name": DISPLAY_NAMES[r["model"]],
            "chosen": r["model"] == FINAL_MODEL,
            "roc_auc": r["test"]["roc_auc"],
            "pr_auc": r["test"]["pr_auc"],
            "f1": r["test"]["at_threshold"]["f1"],
            "latency": r["latency_ms_single_row"],
        }
        for r in cmp
    ]
    if cnn:
        rows.append(
            {"name": "1D CNN", "chosen": False, "roc_auc": cnn["roc_auc"][0],
             "pr_auc": cnn["pr_auc"][0], "f1": None, "latency": None}
        )  # fmt: skip
    rows.sort(key=lambda r: -r["roc_auc"])
    model_chart = charts.hbars(
        [(r["name"], r["roc_auc"], f"{r['roc_auc']:.3f}",
          f"{r['name']}: ROC-AUC {r['roc_auc']:.3f}") for r in rows],
        vmax=1.0,
        label_w=210,
        css=["" if r["chosen"] else "muted-bar" for r in rows],
    )  # fmt: skip
    curves = card["test_curves"]
    rel = [[r["mean_predicted"], r["observed"]] for r in curves["reliability"]]
    rel_max = min(1.0, round(max(max(x, y) for x, y in rel) + 0.05, 1))
    importance = card["global_importance"]
    return {
        "card": card,
        "rows": rows,
        "model_chart": model_chart,
        "roc": charts.xy({"model": curves["roc"]}, width=520, height=320, pad=52),
        "reliability": charts.xy(
            {"model": rel}, xmax=rel_max, ymax=rel_max, width=520, height=300, pad=56
        ),
        "imp_chart": charts.hbars(
            [
                (
                    i["label"],
                    i["mean_abs_log_odds"],
                    f"{i['mean_abs_log_odds']:.2f}",
                    f"{i['label']}: average impact {i['mean_abs_log_odds']:.3f} log-odds",
                )
                for i in importance
            ],
            label_w=230,
        ),
    }


def assessment(a: Assessment, cfg: PolicyConfig) -> dict[str, Any]:
    shown = a.explanation.contributions[:8]
    rest = a.explanation.contributions[8:]
    rows = [
        (f"{c.label}: {c.value}", c.log_odds, f"×{c.odds_multiplier:.2f}",
         f"{c.label} = {c.value}: odds ×{c.odds_multiplier:.2f}")
        for c in shown
    ]  # fmt: skip
    if rest:
        total = sum(c.log_odds for c in rest)
        rows.append(
            (f"{len(rest)} other fields", total, f"×{math.exp(total):.2f}",
             "Combined effect of the remaining fields")
        )  # fmt: skip
    confident = a.decision.action != Action.REVIEW
    return {
        "waterfall": charts.diverging(rows),
        "gauge": charts.gauge(
            a.probability, [(cfg.break_even, "follow-up"), (cfg.priority_line, "priority")]
        ),
        "summary": narrative.summary(a),
        "drivers_up": [narrative.driver_sentence(c) for c in a.explanation.drivers_up[:4]],
        "drivers_down": [narrative.driver_sentence(c) for c in a.explanation.drivers_down[:4]],
        "confidence": "High" if confident else "Needs review",
        "confidence_why": narrative.robustness(a)[1],
        "next_step": ACTION_NEXT_STEP[a.decision.action],
    }


def what_changed(before: Assessment, after: Assessment) -> dict[str, list[dict[str, Any]]]:
    """Split SHAP shifts into fields the user edited and unchanged fields whose effect moved.

    The second group is the model's interaction effects made visible: e.g. a recent
    previous contact helps a lot after a past success, much less after a past failure.
    """
    b = {c.feature: c for c in before.explanation.contributions}
    edited: list[dict[str, Any]] = []
    shifted: list[dict[str, Any]] = []
    for c in after.explanation.contributions:
        old = b[c.feature]
        d = c.log_odds - old.log_odds
        row = {"label": c.label, "before": old.value, "after": c.value, "delta": d}
        if getattr(before.lead, c.feature) != getattr(after.lead, c.feature):
            edited.append(row)
        elif abs(d) >= 0.1:
            shifted.append(row)
    return {
        "edited": sorted(edited, key=lambda x: -abs(x["delta"])),
        "shifted": sorted(shifted, key=lambda x: -abs(x["delta"]))[:4],
    }
