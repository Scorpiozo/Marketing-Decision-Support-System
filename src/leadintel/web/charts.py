"""Chart geometry for server-rendered SVG. Python computes coordinates; Jinja draws them.

Keeping charts as plain SVG (no JS charting library) means zero client-side work,
instant rendering, crisp print output and charts that work with JavaScript disabled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Bar:
    label: str
    value: float
    x: float
    y: float
    w: float
    h: float
    text: str
    tooltip: str
    css: str = ""


@dataclass
class Chart:
    width: int
    height: int
    bars: list[Bar] = field(default_factory=list)
    lines: list[dict[str, Any]] = field(default_factory=list)
    ticks: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


def hbars(
    rows: list[tuple[str, float, str, str]],
    vmax: float | None = None,
    width: int = 560,
    label_w: int = 190,
    row_h: int = 30,
    bar_h: int = 18,
    css: list[str] | None = None,
) -> Chart:
    """Horizontal bars from a zero baseline. rows = (label, value, value_text, tooltip)."""
    vmax = vmax or max((v for _, v, _, _ in rows), default=1.0) or 1.0
    plot_w = width - label_w - 70
    chart = Chart(width=width, height=row_h * len(rows) + 6, extras={"label_w": label_w})
    for i, (label, v, text, tip) in enumerate(rows):
        w = max(2.0, plot_w * v / vmax)
        chart.bars.append(
            Bar(label, v, label_w, i * row_h + (row_h - bar_h) / 2, w, bar_h, text, tip,
                css[i] if css else "")
        )  # fmt: skip
    return chart


def diverging(
    rows: list[tuple[str, float, str, str]], width: int = 560, label_w: int = 230, row_h: int = 28
) -> Chart:
    """Bars left/right of a centre line (SHAP log-odds). Positive = raises the odds."""
    vmax = max((abs(v) for _, v, _, _ in rows), default=1.0) or 1.0
    half = (width - label_w - 20) / 2
    centre = label_w + half
    chart = Chart(width=width, height=row_h * len(rows) + 6, extras={"centre": centre})
    for i, (label, v, text, tip) in enumerate(rows):
        w = max(2.0, half * abs(v) / vmax * 0.82)
        x = centre if v >= 0 else centre - w
        chart.bars.append(
            Bar(label, v, x, i * row_h + 5, w, row_h - 10, text, tip, "up" if v >= 0 else "down")
        )
    return chart


def xy(
    series: dict[str, list[list[float]]],
    width: int = 320,
    height: int = 260,
    pad: int = 36,
    diagonal: bool = True,
    xmax: float = 1.0,
    ymax: float = 1.0,
) -> Chart:
    """Unit-square line chart (ROC, PR, reliability). Points are [x, y] in data units."""
    pw, ph = width - pad - 12, height - pad - 12
    chart = Chart(width=width, height=height, extras={"pad": pad, "pw": pw, "ph": ph})

    def sx(x: float) -> float:
        return pad + pw * x / xmax

    def sy(y: float) -> float:
        return 12 + ph * (1 - y / ymax)

    for name, pts in series.items():
        chart.lines.append(
            {
                "name": name,
                "d": "M" + " L".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts),
                "points": [{"x": sx(x), "y": sy(y), "tip": f"{x:.2f}, {y:.2f}"} for x, y in pts],
            }
        )
    if diagonal:
        chart.extras["diagonal"] = f"M{sx(0):.1f},{sy(0):.1f} L{sx(xmax):.1f},{sy(ymax):.1f}"
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        chart.ticks.append(
            {
                "x": sx(t * xmax),
                "y": sy(t * ymax),
                "label": f"{t * xmax:g}",
                "ylabel": f"{t * ymax:g}",
            }
        )
    return chart


def donut(
    rows: list[tuple[str, float, str, str]], size: int = 220, ring: int = 30, gap: float = 2.0
) -> Chart:
    """Share-of-whole ring. rows = (label, share 0-1, css class, tooltip). Drawn with
    stroke-dasharray on concentric circles, so no path maths in the template."""
    r = (size - ring) / 2
    circ = 2 * math.pi * r
    chart = Chart(width=size, height=size, extras={"r": r, "c": size / 2, "ring": ring})
    start = 0.0
    for label, share, css, tip in rows:
        length = max(share * circ - gap, 0.0)
        chart.bars.append(Bar(label, share, 0, 0, length, circ - length, f"{share:.0%}", tip, css))
        chart.bars[-1].x = -start * circ  # dash offset (reuse field)
        start += share
    return chart


def columns(
    rows: list[tuple[str, float, str, str]],
    width: int = 560,
    height: int = 240,
    vmax: float | None = None,
    css: str = "",
) -> Chart:
    """Vertical columns from a zero baseline. rows = (label, value, value_text, tooltip)."""
    vmax = vmax or max((v for _, v, _, _ in rows), default=1.0) or 1.0
    top, bottom = 22, 34
    plot_h = height - top - bottom
    slot = width / max(len(rows), 1)
    bar_w = min(44.0, slot * 0.62)
    chart = Chart(width=width, height=height, extras={"base": top + plot_h})
    for i, (label, v, text, tip) in enumerate(rows):
        h = max(2.0, plot_h * v / vmax)
        x = i * slot + (slot - bar_w) / 2
        chart.bars.append(Bar(label, v, x, top + plot_h - h, bar_w, h, text, tip, css))
    return chart


def gauge(p: float, marks: list[tuple[float, str]], width: int = 260) -> dict[str, Any]:
    """Half-ring probability gauge with tick marks at policy thresholds."""
    r, cx = width / 2 - 18, width / 2
    cy = r + 12
    arc_len = math.pi * r

    def point(v: float, rr: float) -> tuple[float, float]:
        a = math.pi * (1 - v)
        return cx + rr * math.cos(a), cy - rr * math.sin(a)

    ticks = []
    for v, label in marks:
        x1, y1 = point(v, r - 16)
        x2, y2 = point(v, r + 16)
        ticks.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": label})
    return {
        "width": width,
        "height": cy + 10,
        "d": f"M{cx - r:.1f},{cy:.1f} A{r:.1f},{r:.1f} 0 0 1 {cx + r:.1f},{cy:.1f}",
        "len": arc_len,
        "fill": arc_len * min(max(p, 0.0), 1.0),
        "ticks": ticks,
        "cx": cx,
        "cy": cy,
    }
