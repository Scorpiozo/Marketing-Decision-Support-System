"""FastAPI + Jinja2 + HTMX web app.  Run:  uv run fastapi dev src/leadintel/web/app.py

Server-rendered pages; HTMX swaps fragments for the interactive parts (what-if, table
filters, policy sliders). The model and portfolio are loaded once at start-up.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import parse_qsl, urlencode

from fastapi import FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from pydantic import ValidationError

from leadintel import narrative
from leadintel.config import Settings
from leadintel.features import FEATURE_GROUP_OF, FEATURE_LABELS
from leadintel.models import DISPLAY_NAMES
from leadintel.patterns import HISTORY_ORDER, INTENSITY_ORDER
from leadintel.policy import ACTION_NEXT_STEP, ACTION_ORDER, ACTION_TITLE, Action, PolicyConfig
from leadintel.schema import CATEGORY_LEVELS, Lead
from leadintel.service import Assessment, ScoringService
from leadintel.web import charts

HERE = Path(__file__).parent
POLICY_COOKIE = "policy"
SESSION_COOKIE = "session"
settings = Settings()
state: dict[str, ScoringService] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    t0 = time.perf_counter()
    state["svc"] = ScoringService(settings.artifacts_dir)
    logger.info(f"Model and portfolio loaded in {time.perf_counter() - t0:.2f}s")
    yield


app = FastAPI(title="Lead Decision Intelligence", lifespan=lifespan, docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")
# Content hash of static assets, appended as ?v= so browsers never serve a stale stylesheet
# or script after a deploy.
ASSET_VERSION = hashlib.sha256(
    b"".join(f.read_bytes() for f in sorted((HERE / "static").iterdir()) if f.is_file())
).hexdigest()[:10]
templates.env.globals.update(
    ASSET_VERSION=ASSET_VERSION,
    ACTION_TITLE={a.value: t for a, t in ACTION_TITLE.items()},
    ACTION_ORDER=[a.value for a in ACTION_ORDER],
    FEATURE_LABELS=FEATURE_LABELS,
    FEATURE_GROUP_OF=FEATURE_GROUP_OF,
    DISPLAY_NAMES=DISPLAY_NAMES,
)


def _eur(x: float) -> str:
    return f"{'−' if x < 0 else ''}€{abs(x):,.0f}"


templates.env.filters["eur"] = _eur


def svc() -> ScoringService:
    return state["svc"]


# ---- auth (only when APP_PASSWORD is set) -------------------------------------------


def _token() -> str:
    key = (settings.session_secret + (settings.app_password or "")).encode()
    return hmac.new(key, b"authenticated", hashlib.sha256).hexdigest()


@app.middleware("http")
async def password_gate(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    open_paths = ("/login", "/static", "/healthz")
    if (
        settings.app_password
        and not request.url.path.startswith(open_paths)
        and not hmac.compare_digest(request.cookies.get(SESSION_COOKIE, ""), _token())
    ):
        if request.url.path.startswith("/api"):
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        return RedirectResponse(f"/login?next={request.url.path}", status_code=303)
    return await call_next(request)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/", error: bool = False) -> HTMLResponse:
    return templates.TemplateResponse(request, "login.html", {"next": next, "error": error})


@app.post("/login")
def login(password: Annotated[str, Form()], next: Annotated[str, Form()] = "/") -> Response:
    target = next if next.startswith("/") and not next.startswith("//") else "/"
    if not settings.app_password or not hmac.compare_digest(password, settings.app_password):
        return RedirectResponse(f"/login?error=1&next={target}", status_code=303)
    resp = RedirectResponse(target, status_code=303)
    resp.set_cookie(SESSION_COOKIE, _token(), httponly=True, samesite="lax", max_age=7 * 86400)
    return resp


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": "svc" in state}


# ---- policy (per-browser, stored in a cookie) ---------------------------------------


def policy_from(request: Request) -> PolicyConfig:
    raw = request.cookies.get(POLICY_COOKIE)
    if raw:
        try:
            return PolicyConfig.model_validate(dict(parse_qsl(raw)))
        except ValidationError:
            pass
    return PolicyConfig()


def render(request: Request, name: str, **ctx: Any) -> HTMLResponse:
    ctx.setdefault("policy", policy_from(request))
    ctx.setdefault("active", name.split(".")[0])
    return templates.TemplateResponse(request, name, ctx)


# ---- pages ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def overview(request: Request) -> HTMLResponse:
    s, cfg = svc(), policy_from(request)
    backtest = s.policy_backtest(cfg)
    top, _ = s.table(cfg, action=Action.PRIORITISE.value, per_page=6)
    ablation = {
        (r["feature_set"], r["model"]): r
        for r in s.experiments.get("ablation", {}).get("random_split", [])
    }
    return render(
        request,
        "overview.html",
        backtest=backtest,
        mix=_action_mix_chart(backtest),
        top=top.to_dict("records"),
        card=s.card,
        ablation=ablation,
    )


@app.get("/leads", response_class=HTMLResponse)
def leads(
    request: Request,
    action: str = "",
    history: str = "",
    q: str = "",
    sort: str = "probability",
    page: int = 1,
) -> HTMLResponse:
    cfg = policy_from(request)
    rows, total = svc().table(cfg, action or None, history or None, q or None, sort, page)
    ctx = {
        "rows": rows.to_dict("records"),
        "total": total,
        "page": page,
        "pages": max(1, -(-total // 25)),
        "filters": {"action": action, "history": history, "q": q, "sort": sort},
        "histories": HISTORY_ORDER,
        "counts": svc().actions(cfg).value_counts().to_dict(),
        "qs": lambda **kw: urlencode(
            {**{"action": action, "history": history, "q": q, "sort": sort}, **kw}
        ),
    }
    template = "_lead_table.html" if request.headers.get("HX-Request") else "leads.html"
    return render(request, template, active="leads", **ctx)


@app.get("/leads/{lead_id}", response_class=HTMLResponse)
def lead_page(request: Request, lead_id: str) -> HTMLResponse:
    s = svc()
    if lead_id not in s.portfolio.index:
        raise HTTPException(404, f"No lead with id {lead_id}")
    a = s.assess(s.lead(lead_id), policy_from(request), lead_id)
    return render(
        request, "lead.html", active="leads", a=a, view=_assessment_view(a), levels=CATEGORY_LEVELS
    )


@app.post("/leads/{lead_id}/whatif", response_class=HTMLResponse)
async def whatif(request: Request, lead_id: str) -> HTMLResponse:
    s, cfg = svc(), policy_from(request)
    original = s.assess(s.lead(lead_id), cfg, lead_id)
    form = dict(await request.form())
    try:
        edited = Lead.model_validate(form)
    except ValidationError as e:
        errors = [err["msg"].removeprefix("Value error, ") for err in e.errors()]
        return render(request, "_whatif_result.html", errors=errors)
    a = s.assess(edited, cfg, lead_id)
    return render(
        request,
        "_whatif_result.html",
        a=a,
        view=_assessment_view(a),
        original=original,
        changes=_what_changed(original, a),
    )


@app.get("/patterns", response_class=HTMLResponse)
def patterns(request: Request) -> HTMLResponse:
    s = svc()
    grid = {(r["history"], r["intensity"]): r for r in s.card["pattern_stats_dev"]}
    abl = {
        r["feature_set"]: r
        for r in s.experiments.get("ablation", {}).get("random_split", [])
        if r["model"] == "gradient_boosting" and r["feature_set"] != "full+duration"
    }
    abl_labels = {
        "profile": "Profile only",
        "profile+context": "Profile + campaign context",
        "profile+interaction": "Profile + interaction history",
        "full": "All three (deployed)",
    }
    abl_chart = charts.hbars(
        [
            (label, abl[k]["test"]["pr_auc"], f"{abl[k]['test']['pr_auc']:.3f}",
             f"{label}: PR-AUC {abl[k]['test']['pr_auc']:.3f}, "
             f"ROC-AUC {abl[k]['test']['roc_auc']:.3f}")
            for k, label in abl_labels.items() if k in abl
        ],
        vmax=0.5,
        label_w=210,
    ) if abl else None  # fmt: skip
    inter = s.experiments.get("interactions", {}).get("shap_interaction", {})
    pairs = inter.get("top_pairs", [])[:7]
    pair_chart = charts.hbars(
        [(p["label"], p["mean_abs_interaction"], f"{p['mean_abs_interaction']:.3f}",
          f"{p['label']}: {p['mean_abs_interaction']:.3f}") for p in pairs],
        label_w=330,
    ) if pairs else None  # fmt: skip
    hyp = {
        tuple(sorted(p["pair"])): p["mean_abs_interaction"]
        for p in inter.get("hypothesised_pairs", [])
    }
    return render(
        request,
        "patterns.html",
        grid=grid,
        histories=HISTORY_ORDER,
        intensities=INTENSITY_ORDER,
        exp=s.experiments,
        importance=s.card["global_importance"],
        abl_chart=abl_chart,
        pair_chart=pair_chart,
        hyp_campaign=hyp.get(("campaign", "poutcome"), 0.0),
        base_rate=s.card["data"]["base_rate"],
    )


@app.get("/model", response_class=HTMLResponse)
def model_page(request: Request) -> HTMLResponse:
    s = svc()
    card = s.card
    curves = card["test_curves"]
    rel = [[r["mean_predicted"], r["observed"]] for r in curves["reliability"]]
    top = max(max(x, y) for x, y in rel)
    rel_max = min(1.0, round(top + 0.05, 1))
    importance = card["global_importance"]
    imp_chart = charts.hbars(
        [
            (i["label"], i["mean_abs_log_odds"], f"{i['mean_abs_log_odds']:.2f}",
             f"{i['label']}: mean |SHAP| {i['mean_abs_log_odds']:.3f} log-odds")
            for i in importance
        ],
        css=["context" if i["is_context"] else "" for i in importance],
        label_w=230,
    )  # fmt: skip
    return render(
        request,
        "model.html",
        card=card,
        exp=s.experiments,
        roc=charts.xy({"model": curves["roc"]}),
        pr=charts.xy({"model": curves["pr"]}, diagonal=False),
        reliability=charts.xy({"model": rel}, xmax=rel_max, ymax=rel_max),
        importance=importance,
        imp_chart=imp_chart,
    )


@app.get("/policy", response_class=HTMLResponse)
def policy_page(request: Request) -> HTMLResponse:
    cfg = policy_from(request)
    b = svc().policy_backtest(cfg)
    return render(request, "policy.html", backtest=b, mix=_action_mix_chart(b))


@app.post("/policy", response_class=HTMLResponse)
async def update_policy(request: Request) -> HTMLResponse:
    form = dict(await request.form())
    try:
        cfg = PolicyConfig.model_validate(form) if "reset" not in form else PolicyConfig()
        error = None
    except ValidationError as e:
        cfg, error = policy_from(request), "; ".join(err["msg"] for err in e.errors())
    b = svc().policy_backtest(cfg)
    resp = render(
        request, "_backtest.html", policy=cfg, backtest=b, mix=_action_mix_chart(b), error=error
    )
    fields = cfg.model_dump(exclude={"break_even"})
    resp.set_cookie(POLICY_COOKIE, urlencode(fields), samesite="lax", max_age=30 * 86400)
    if "reset" in form:
        resp.headers["HX-Refresh"] = "true"
    return resp


@app.get("/method", response_class=HTMLResponse)
def method(request: Request) -> HTMLResponse:
    return render(request, "method.html", card=svc().card, exp=svc().experiments)


# ---- JSON API: the integration surface a CRM would call -----------------------------


@app.post("/api/v1/assess")
def api_assess(lead: Lead, request: Request) -> dict[str, Any]:
    """Score one lead; returns probability, interval, action, reasons and SHAP drivers."""
    return _api_payload(svc().assess(lead, policy_from(request)))


@app.get("/api/v1/leads/{lead_id}")
def api_lead(lead_id: str, request: Request) -> dict[str, Any]:
    s = svc()
    if lead_id not in s.portfolio.index:
        raise HTTPException(404, f"No lead with id {lead_id}")
    return _api_payload(s.assess(s.lead(lead_id), policy_from(request), lead_id))


def _api_payload(a: Assessment) -> dict[str, Any]:
    out = a.model_dump(exclude={"actual_outcome"})
    out["summary"] = narrative.summary(a)
    return out


# ---- view helpers --------------------------------------------------------------------


def _assessment_view(a: Assessment) -> dict[str, Any]:
    shown = a.explanation.contributions[:8]
    rest = a.explanation.contributions[8:]
    rows = [
        (f"{c.label}: {c.value}", c.log_odds, f"×{c.odds_multiplier:.2f}",
         f"{c.label} = {c.value}: odds ×{c.odds_multiplier:.2f} ({c.log_odds:+.2f} log-odds)")
        for c in shown
    ]  # fmt: skip
    if rest:
        total = sum(c.log_odds for c in rest)
        rows.append(
            (f"{len(rest)} other fields", total, f"×{2.718281828 ** total:.2f}",
             "Combined effect of the remaining fields")
        )  # fmt: skip
    label, why = narrative.robustness(a)
    return {
        "waterfall": charts.diverging(rows),
        "context_rows": {i for i, c in enumerate(shown) if c.is_context},
        "summary": narrative.summary(a),
        "drivers_up": [narrative.driver_sentence(c) for c in a.explanation.drivers_up[:4]],
        "drivers_down": [narrative.driver_sentence(c) for c in a.explanation.drivers_down[:4]],
        "robustness": label,
        "robustness_why": why,
        "next_step": ACTION_NEXT_STEP[a.decision.action],
    }


def _what_changed(before: Assessment, after: Assessment) -> dict[str, list[dict[str, Any]]]:
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


def _action_mix_chart(backtest: dict[str, Any]) -> charts.Chart:
    rows = backtest["rows"]
    return charts.hbars(
        [
            (r["title"], r["share"], f"{r['share']:.0%}",
             f"{r['title']}: {r['n']:,} leads · {(r['observed_rate'] or 0):.0%} actually converted")
            for r in rows
        ],
        vmax=1.0,
        css=[f"act-{r['action']}" for r in rows],
    )  # fmt: skip
