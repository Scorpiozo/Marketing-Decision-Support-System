"""Measure the running app: cold start, page latency, what-if latency, memory.

    uv run python scripts/benchmark.py            # starts its own server on a free port

Writes reports/performance.json. Numbers are wall-clock on the machine it runs on.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    "/",
    "/leads",
    "/leads?action=prioritise&page=3",
    "/patterns",
    "/model",
    "/policy",
    "/method",
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def timed(client: httpx.Client, method: str, url: str, n: int, **kw: object) -> dict[str, float]:
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.request(method, url, **kw)  # type: ignore[arg-type]
        r.raise_for_status()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return {
        "p50_ms": round(statistics.median(samples), 1),
        "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 1),
        "bytes": len(r.content),
    }


def main() -> None:
    port = free_port()
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "leadintel.web.app:app", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base, timeout=30) as c:
            while True:
                try:
                    if c.get("/healthz").status_code == 200:
                        break
                except httpx.TransportError:
                    time.sleep(0.05)
            cold_start = time.perf_counter() - t0
            first_page = timed(c, "GET", "/", 1)
            pages = {p: timed(c, "GET", p, 30) for p in PAGES}
            lead_id = c.get("/api/v1/leads/L43013").json()["lead_id"]
            lead = c.get(f"/api/v1/leads/{lead_id}").json()["lead"]
            pages[f"/leads/{lead_id}"] = timed(c, "GET", f"/leads/{lead_id}", 30)
            whatif = timed(c, "POST", f"/leads/{lead_id}/whatif", 50, data={**lead, "campaign": 7})
            api = timed(c, "POST", "/api/v1/assess", 50, json={**lead, "campaign": 7})
            policy = timed(
                c,
                "POST",
                "/policy",
                30,
                data={
                    "contact_cost": 8,
                    "conversion_value": 60,
                    "priority_threshold": 0.4,
                    "heavy_contact": 5,
                },
            )
        rss_mb = None
        if platform.system() != "Windows":
            out = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(proc.pid)], capture_output=True, text=True
            )
            rss_mb = round(int(out.stdout.strip()) / 1024, 1) if out.stdout.strip() else None
    finally:
        proc.terminate()
        proc.wait()

    report = {
        "machine": (
            f"{platform.system()} {platform.machine()}, "
            f"Python {platform.python_version()}, {os.cpu_count()} cores"
        ),
        "cold_start_to_healthy_s": round(cold_start, 2),
        "first_page_ms": first_page["p50_ms"],
        "pages": pages,
        "whatif_rescore": whatif,
        "api_assess": api,
        "policy_backtest_update": policy,
        "server_rss_mb": rss_mb,
    }
    out_path = ROOT / "reports" / "performance.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
