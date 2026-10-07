"""Application tests: every page renders, interactions work, errors are graceful."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from leadintel.web import app as web


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(web.app) as c:
        yield c


@pytest.fixture(scope="module")
def lead_id(client: TestClient) -> str:
    del client  # ensures the app (and service) started
    return str(web.svc().portfolio.index[0])


@pytest.mark.parametrize(
    "path,needle",
    [
        ("/", "Know which leads to call next"),
        ("/leads", "Lead prioritisation"),
        ("/leads?action=stop&page=2", "Stop repeated follow-up"),
        ("/patterns", "Subscription rate by interaction pattern"),
        ("/model", "Model comparison"),
        ("/policy", "Contact strategy simulator"),
        ("/about", "How it works"),
    ],
)
def test_pages_render(client: TestClient, path: str, needle: str) -> None:
    r = client.get(path)
    assert r.status_code == 200
    assert needle in r.text


def test_lead_page_has_decision_explanation_and_caveat(client: TestClient, lead_id: str) -> None:
    r = client.get(f"/leads/{lead_id}")
    assert r.status_code == 200
    for needle in (
        "Recommended action",
        "Why this recommendation",
        "likelihood to subscribe",
        "What-if",
    ):
        assert needle in r.text


def test_unknown_lead_is_404(client: TestClient) -> None:
    assert client.get("/leads/L99999").status_code == 404


def test_htmx_table_returns_fragment(client: TestClient) -> None:
    r = client.get("/leads?action=prioritise", headers={"HX-Request": "true"})
    assert r.status_code == 200
    assert "<html" not in r.text and "Prioritise follow-up" in r.text


def test_empty_filter_shows_empty_state(client: TestClient) -> None:
    r = client.get("/leads?q=NOPE", headers={"HX-Request": "true"})
    assert "No leads match" in r.text


def _form(lead_id: str, **changes: Any) -> dict[str, Any]:
    fields = web.svc().lead(lead_id).model_dump()
    return {**fields, **changes}


def test_whatif_rescoring_changes_decision(client: TestClient) -> None:
    s = web.svc()
    top = str(s.portfolio.sort_values("probability").index[-1])  # a high-probability lead
    r = client.post(f"/leads/{top}/whatif", data=_form(top, poutcome="failure", campaign=9))
    assert r.status_code == 200
    assert "What-if result" in r.text and "Fields you changed" in r.text
    assert "The recommendation changed" in r.text


def test_whatif_invalid_history_is_explained_not_crashed(client: TestClient, lead_id: str) -> None:
    r = client.post(
        f"/leads/{lead_id}/whatif", data=_form(lead_id, previous=0, pdays=-1, poutcome="failure")
    )
    assert r.status_code == 200
    assert "can't be scored" in r.text and "Inconsistent history" in r.text


def test_policy_update_sets_cookie_and_changes_segmentation(client: TestClient) -> None:
    r = client.post(
        "/policy",
        data={
            "contact_cost": 20,
            "conversion_value": 50,
            "priority_threshold": 0.5,
            "heavy_contact": 3,
        },
    )
    assert r.status_code == 200 and "follow-up line 40.0%" in r.text
    assert "policy" in r.cookies
    client.cookies.clear()


def test_invalid_policy_shows_error(client: TestClient) -> None:
    r = client.post(
        "/policy",
        data={
            "contact_cost": -5,
            "conversion_value": 50,
            "priority_threshold": 0.35,
            "heavy_contact": 4,
        },
    )
    assert r.status_code == 200 and "Invalid assumption" in r.text
    client.cookies.clear()


def test_api_assess_roundtrip(client: TestClient, lead_id: str) -> None:
    r = client.post("/api/v1/assess", json=web.svc().lead(lead_id).model_dump())
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["probability"] < 1
    assert body["decision"]["action"] in {
        "prioritise",
        "follow_up",
        "change_approach",
        "review",
        "deprioritise",
        "stop",
    }
    assert "actual_outcome" not in body  # never leak the held-out label through the API
    assert len(body["explanation"]["contributions"]) == 15


def test_api_rejects_invalid_lead(client: TestClient, lead_id: str) -> None:
    bad = {**web.svc().lead(lead_id).model_dump(), "age": 5}
    assert client.post("/api/v1/assess", json=bad).status_code == 422


def test_password_gate(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    monkeypatch.setattr(web.settings, "app_password", "s3cret")
    assert client.get("/", follow_redirects=False).status_code == 303
    assert client.get("/api/v1/leads/L00000").status_code == 401
    assert client.get("/healthz").status_code == 200
    bad = client.post("/login", data={"password": "nope"}, follow_redirects=False)
    assert "error=1" in bad.headers["location"]
    ok = client.post(
        "/login", data={"password": "s3cret", "next": "/model"}, follow_redirects=False
    )
    assert ok.headers["location"] == "/model"
    assert client.get("/model").status_code == 200
    # open-redirect guard
    evil = client.post(
        "/login", data={"password": "s3cret", "next": "//evil.example"}, follow_redirects=False
    )
    assert evil.headers["location"] == "/"
    client.cookies.clear()


def test_euro_formatting_handles_negatives() -> None:
    assert web._eur(-82745.0) == "−€82,745"
    assert web._eur(24105.4) == "€24,105"


def test_live_scoring_is_rate_limited(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, lead_id: str
) -> None:
    monkeypatch.setattr(web, "RATE_LIMIT", 3)
    web._hits.clear()
    body = web.svc().lead(lead_id).model_dump()
    codes = [client.post("/api/v1/assess", json=body).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    web._hits.clear()


def test_old_method_route_redirects(client: TestClient) -> None:
    r = client.get("/method", follow_redirects=False)
    assert r.status_code == 301 and r.headers["location"] == "/about"


def test_insights_are_consistent(client: TestClient) -> None:
    ins = web.svc().insights
    assert ins["gains"][0] == [0.0, 0.0] and ins["gains"][-1] == [1.0, 1.0]
    assert sum(r["n"] for r in ins["by_contacts"]) == len(web.svc().portfolio)


def test_ui_has_no_prototype_wording(client: TestClient) -> None:
    for path in ("/", "/leads", "/patterns", "/model", "/policy", "/about"):
        text = client.get(path).text.lower()
        assert "prototype" not in text and "faculty" not in text and "academic" not in text
