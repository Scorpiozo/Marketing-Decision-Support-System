"""FastAPI + Jinja2 + HTMX web app.  Run:  uv run fastapi dev src/leadintel/web/app.py

Server-rendered pages; HTMX swaps fragments for the interactive parts (what-if, table
filters, policy sliders). The model and portfolio are loaded once at start-up.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import defaultdict, deque
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
from leadintel.policy import ACTION_ORDER, ACTION_TITLE, PolicyConfig
from leadintel.schema import CATEGORY_LEVELS, Lead
from leadintel.service import Assessment, ScoringService
from leadintel.web import views

HERE = Path(__file__).parent
POLICY_COOKIE = "policy"
SESSION_COOKIE = "session"
settings = Settings()
state: dict[str, ScoringService] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    t0 = time.perf_counter()
    if (
        settings.app_password
        and settings.session_secret == Settings.model_fields["session_secret"].default
    ):
        logger.warning("APP_PASSWORD is set but SESSION_SECRET is the default; set a random value.")
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


# ---- rate limit for endpoints that run the model live ---------------------------------
# The public demo has no login, so cap live scoring per client IP to stop a bot from
# burning hosting credit. In-memory sliding window: fine for a single-process demo.
RATE_LIMIT = 120  # live scoring requests per IP per minute (a human uses a handful)
_hits: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def rate_limit(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if request.method == "POST" and request.url.path != "/login":
        ip = request.client.host if request.client else "unknown"
        now, window = time.monotonic(), _hits[ip]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= RATE_LIMIT:
            return JSONResponse({"detail": "Too many requests; try again in a minute."}, 429)
        window.append(now)
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
    return render(request, "overview.html", **views.dashboard(svc(), policy_from(request)))


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
    s, cfg = svc(), policy_from(request)
    if lead_id not in s.portfolio.index:
        raise HTTPException(404, f"No lead with id {lead_id}")
    a = s.assess(s.lead(lead_id), cfg, lead_id)
    return render(
        request,
        "lead.html",
        active="leads",
        a=a,
        view=views.assessment(a, cfg),
        levels=CATEGORY_LEVELS,
    )


@app.post("/leads/{lead_id}/whatif", response_class=HTMLResponse)
async def whatif(request: Request, lead_id: str) -> HTMLResponse:
    s, cfg = svc(), policy_from(request)
    if lead_id not in s.portfolio.index:
        raise HTTPException(404, f"No lead with id {lead_id}")
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
        view=views.assessment(a, cfg),
        original=original,
        changes=views.what_changed(original, a),
    )


@app.get("/patterns", response_class=HTMLResponse)
def patterns(request: Request) -> HTMLResponse:
    return render(
        request,
        "patterns.html",
        histories=HISTORY_ORDER,
        intensities=INTENSITY_ORDER,
        **views.insights_page(svc()),
    )


@app.get("/model", response_class=HTMLResponse)
def model_page(request: Request) -> HTMLResponse:
    return render(request, "model.html", **views.model_page(svc()))


@app.get("/policy", response_class=HTMLResponse)
def policy_page(request: Request) -> HTMLResponse:
    b = svc().policy_backtest(policy_from(request))
    return render(request, "policy.html", backtest=b, donut=views.action_donut(b))


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
        request,
        "_backtest.html",
        policy=cfg,
        backtest=b,
        donut=views.action_donut(b),
        error=error,
    )
    fields = cfg.model_dump(exclude={"break_even", "priority_line"})
    resp.set_cookie(POLICY_COOKIE, urlencode(fields), samesite="lax", max_age=30 * 86400)
    if "reset" in form:
        resp.headers["HX-Refresh"] = "true"
    return resp


@app.get("/about", response_class=HTMLResponse)
def about(request: Request) -> HTMLResponse:
    return render(request, "about.html", card=svc().card)


@app.get("/method")
def method_redirect() -> RedirectResponse:
    return RedirectResponse("/about", status_code=301)


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
