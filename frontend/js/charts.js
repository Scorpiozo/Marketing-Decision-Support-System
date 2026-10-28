/*
 * Minimal, dependency-free SVG/HTML charts for the dashboard.
 *
 *   Charts.bar(el, opts)      vertical bars (rates, counts, histograms)
 *   Charts.hbar(el, opts)     horizontal bars in HTML (long labels wrap)
 *   Charts.pairs(el, opts)    two-series horizontal bars (model vs model)
 *   Charts.line(el, opts)     multi-series line chart (ROC curves)
 *   Charts.grouped(el, opts)  grouped vertical bars (metric comparison)
 *   Charts.table(el, cols, rows)
 *
 * Colours are CSS custom properties, so light/dark themes switch without
 * re-rendering. Every chart re-renders on container resize.
 */
(function () {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const registry = new WeakMap();
  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const job = registry.get(entry.target);
      if (job && Math.abs(entry.contentRect.width - job.width) > 4) job.draw();
    }
  });

  function mount(el, drawFn) {
    const job = {
      width: 0,
      draw() {
        job.width = el.clientWidth;
        el.replaceChildren();
        drawFn(Math.max(job.width, 240));
      },
    };
    registry.set(el, job);
    resizeObserver.observe(el);
    job.draw();
  }

  function svgEl(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  function niceTicks(max, count) {
    if (!(max > 0)) return [0, 1];
    const raw = max / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const ticks = [];
    for (let v = 0; v <= max + step * 0.001; v += step) ticks.push(+v.toFixed(10));
    if (ticks[ticks.length - 1] < max) ticks.push(+(ticks[ticks.length - 1] + step).toFixed(10));
    return ticks;
  }

  // Bar with 4px rounded data-end, square at the baseline.
  function barPath(x, y, w, h, r) {
    r = Math.min(r, w / 2, h);
    if (h <= 0) return "";
    return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
  }

  // Tooltip (single shared element)
  let tip;
  function tooltip() {
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "chart-tip";
      tip.setAttribute("role", "status");
      document.body.appendChild(tip);
    }
    return tip;
  }
  function showTip(html, evt) {
    const t = tooltip();
    t.innerHTML = html;
    t.style.display = "block";
    const pad = 14;
    const r = t.getBoundingClientRect();
    let x = evt.clientX + pad;
    let y = evt.clientY + pad;
    if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
    t.style.left = `${Math.max(8, x)}px`;
    t.style.top = `${Math.max(8, y)}px`;
  }
  function hideTip() {
    if (tip) tip.style.display = "none";
  }
  window.addEventListener("scroll", hideTip, { passive: true });

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function empty(el, message) {
    el.replaceChildren();
    const p = document.createElement("p");
    p.className = "chart-empty";
    p.textContent = message || "No customers match the current filters.";
    el.appendChild(p);
  }

  /* Vertical bars.
   * opts.rows: [{label, value, color?, faded?, tip?}]
   * opts.format(v), opts.height, opts.refLine {value, label}, opts.yMax,
   * opts.markers [{at: index boundary (float), label}]  (vertical guides)
   */
  function bar(el, opts) {
    const rows = opts.rows || [];
    if (!rows.length) return empty(el, opts.emptyText);
    mount(el, (width) => {
      const height = opts.height || 260;
      const fmt = opts.format || ((v) => v);
      const labels = rows.map((r) => String(r.label));
      const longest = Math.max(...labels.map((l) => l.length));
      const m = { top: opts.refLine ? 30 : 16, right: 12, bottom: 28, left: 48 };
      const plotW0 = width - m.left - m.right;
      const band0 = plotW0 / rows.length;
      const rotate = !opts.hideLabels && longest * 6.8 > band0 - 4;
      if (rotate) m.bottom = Math.min(90, 14 + longest * 5.2);
      if (opts.hideLabels) m.bottom = opts.axisBottom ? 26 : 10;
      const plotW = width - m.left - m.right;
      const plotH = height - m.top - m.bottom;
      let maxVal = Math.max(opts.yMax || 0, ...rows.map((r) => r.value), opts.refLine ? opts.refLine.value : 0);
      // Clip one dominant bar so the rest stay readable (its true value is labelled).
      let clipAt = null;
      if (opts.clip) {
        const sorted = rows.map((r) => r.value).sort((a, b) => b - a);
        if (sorted.length > 2 && sorted[0] > sorted[1] * 3) { clipAt = sorted[1] * 1.6; maxVal = clipAt; }
      }
      const ticks = niceTicks(maxVal, 4);
      const top = ticks[ticks.length - 1];
      const y = (v) => m.top + plotH - (v / top) * plotH;
      const band = plotW / rows.length;
      const gap = Math.max(2, Math.min(band * 0.28, 18));

      const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": opts.ariaLabel || "Bar chart" }, el);
      const grid = svgEl("g", { class: "grid" }, svg);
      ticks.forEach((t) => {
        svgEl("line", { x1: m.left, x2: width - m.right, y1: y(t), y2: y(t), class: t === 0 ? "axis-line" : "grid-line" }, grid);
        const label = svgEl("text", { x: m.left - 8, y: y(t) + 4, class: "tick", "text-anchor": "end" }, grid);
        label.textContent = fmt(t, true);
      });

      const bars = svgEl("g", {}, svg);
      rows.forEach((r, i) => {
        const x = m.left + i * band + gap / 2;
        const w = Math.max(1, band - gap);
        const shown = clipAt != null ? Math.min(r.value, top) : r.value;
        const h = Math.max(0, y(0) - y(shown));
        const g = svgEl("g", { class: "bar" + (r.faded ? " bar--faded" : "") }, bars);
        svgEl("path", { d: barPath(x, y(shown), w, h, 4), style: `fill:${r.color || opts.color || "var(--series-1)"}` }, g);
        if (clipAt != null && r.value > top) {
          const cy = y(top) + 10;
          svgEl("line", { x1: x - 1, x2: x + w + 1, y1: cy + 3, y2: cy - 3, class: "clip-mark" }, g);
          const vt = svgEl("text", { x: x + w + 4, y: y(top) + 10, class: "bar-value" }, g);
          vt.textContent = fmt(r.value);
        }
        // Hit target taller than the mark
        const hit = svgEl("rect", { x: m.left + i * band, y: m.top, width: band, height: plotH, class: "hit" }, g);
        const html = r.tip || `<strong>${esc(r.label)}</strong><br>${esc(fmt(r.value))}`;
        hit.addEventListener("mousemove", (e) => { g.classList.add("is-hover"); showTip(html, e); });
        hit.addEventListener("mouseleave", () => { g.classList.remove("is-hover"); hideTip(); });
        if (opts.onClick) {
          hit.style.cursor = "pointer";
          hit.addEventListener("click", () => opts.onClick(r, i));
        }
        if (!opts.hideLabels) {
          const lx = m.left + i * band + band / 2;
          const ly = y(0) + 16;
          const t = svgEl("text", { x: lx, y: ly, class: "tick tick--x", "text-anchor": rotate ? "end" : "middle" }, svg);
          if (rotate) t.setAttribute("transform", `rotate(-40 ${lx} ${ly - 6})`);
          t.textContent = r.label;
        }
      });

      (opts.markers || []).forEach((mk) => {
        const x = m.left + mk.at * band;
        svgEl("line", { x1: x, x2: x, y1: m.top - 6, y2: y(0), class: "marker-line" }, svg);
        const t = svgEl("text", { x: x + 4, y: m.top + 4, class: "marker-label" }, svg);
        t.textContent = mk.label;
      });
      if (opts.axisBottom) {
        opts.axisBottom.forEach((a) => {
          const t = svgEl("text", { x: m.left + a.at * band, y: y(0) + 16, class: "tick tick--x", "text-anchor": "middle" }, svg);
          t.textContent = a.label;
        });
      }

      if (opts.refLine) {
        const ry = y(opts.refLine.value);
        svgEl("line", { x1: m.left, x2: width - m.right, y1: ry, y2: ry, class: "ref-line" }, svg);
        // Key above the plot, so it never collides with bars
        const t = svgEl("text", { x: width - m.right, y: 12, class: "ref-label", "text-anchor": "end" }, svg);
        t.textContent = opts.refLine.label;
        const tw = opts.refLine.label.length * 6.3;
        svgEl("line", { x1: width - m.right - tw - 26, x2: width - m.right - tw - 6, y1: 8, y2: 8, class: "ref-key" }, svg);
      }
    });
  }

  /* Horizontal bars in HTML. rows: [{label, value, display?, color?, onClick?}] */
  function hbar(el, opts) {
    const rows = opts.rows || [];
    if (!rows.length) return empty(el, opts.emptyText);
    registry.delete(el);
    el.replaceChildren();
    const max = Math.max(...rows.map((r) => r.value)) || 1;
    const list = document.createElement("div");
    list.className = "hbars";
    rows.forEach((r) => {
      const row = document.createElement("div");
      row.className = "hbar";
      row.innerHTML = `
        <span class="hbar__label">${esc(r.label)}</span>
        <span class="hbar__track"><span class="hbar__fill" style="width:${(r.value / max) * 100}%;background:${r.color || opts.color || "var(--series-1)"}"></span></span>
        <span class="hbar__value">${esc(r.display != null ? r.display : r.value)}</span>`;
      list.appendChild(row);
    });
    el.appendChild(list);
  }

  /* Two series per row, horizontal. rows: [{label, values:[a,b]}], series:[{name,color}] */
  function pairs(el, opts) {
    const rows = opts.rows || [];
    if (!rows.length) return empty(el);
    registry.delete(el);
    el.replaceChildren();
    const fmt = opts.format || ((v) => v);
    const max = Math.max(...rows.flatMap((r) => r.values)) || 1;
    const list = document.createElement("div");
    list.className = "pairs";
    rows.forEach((r) => {
      const row = document.createElement("div");
      row.className = "pairs__row";
      const bars = r.values.map((v, i) => `
        <span class="pairs__bar"><span class="pairs__fill" style="width:${(v / max) * 100}%;background:${opts.series[i].color}"></span>
        <span class="pairs__value">${esc(fmt(v))}</span></span>`).join("");
      row.innerHTML = `<span class="pairs__label">${esc(r.label)}</span><span class="pairs__bars">${bars}</span>`;
      row.title = `${r.label}: ` + r.values.map((v, i) => `${opts.series[i].name} ${fmt(v)}`).join(", ");
      list.appendChild(row);
    });
    el.appendChild(list);
  }

  /* Lines. series: [{name, color, points:[[x,y,extra], ...]}]; axes 0..1 */
  function line(el, opts) {
    mount(el, (width) => {
      const height = Math.min(opts.height || 320, width * 0.9);
      const m = { top: 12, right: 14, bottom: 40, left: 48 };
      const plotW = width - m.left - m.right;
      const plotH = height - m.top - m.bottom;
      const x = (v) => m.left + v * plotW;
      const y = (v) => m.top + plotH - v * plotH;
      const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": opts.ariaLabel || "Line chart" }, el);
      const ticks = [0, 0.25, 0.5, 0.75, 1];
      ticks.forEach((t) => {
        svgEl("line", { x1: m.left, x2: width - m.right, y1: y(t), y2: y(t), class: t === 0 ? "axis-line" : "grid-line" }, svg);
        const ty = svgEl("text", { x: m.left - 8, y: y(t) + 4, class: "tick", "text-anchor": "end" }, svg);
        ty.textContent = t;
        const tx = svgEl("text", { x: x(t), y: y(0) + 16, class: "tick", "text-anchor": "middle" }, svg);
        tx.textContent = t;
      });
      const xl = svgEl("text", { x: m.left + plotW / 2, y: height - 4, class: "axis-title", "text-anchor": "middle" }, svg);
      xl.textContent = opts.xLabel || "";
      const yl = svgEl("text", { x: 12, y: m.top + plotH / 2, class: "axis-title", "text-anchor": "middle", transform: `rotate(-90 12 ${m.top + plotH / 2})` }, svg);
      yl.textContent = opts.yLabel || "";
      if (opts.diagonal) svgEl("line", { x1: x(0), y1: y(0), x2: x(1), y2: y(1), class: "ref-line" }, svg);

      opts.series.forEach((s) => {
        const d = s.points.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
        svgEl("path", { d, class: "series-line", style: `stroke:${s.color}` }, svg);
      });

      const cross = svgEl("line", { y1: m.top, y2: y(0), class: "crosshair", visibility: "hidden" }, svg);
      const dots = opts.series.map((s) => svgEl("circle", { r: 4.5, class: "series-dot", style: `fill:${s.color}`, visibility: "hidden" }, svg));
      const hit = svgEl("rect", { x: m.left, y: m.top, width: plotW, height: plotH, class: "hit" }, svg);
      hit.addEventListener("mousemove", (e) => {
        const rect = svg.getBoundingClientRect();
        const fx = Math.min(1, Math.max(0, ((e.clientX - rect.left) * (width / rect.width) - m.left) / plotW));
        cross.setAttribute("x1", x(fx)); cross.setAttribute("x2", x(fx)); cross.setAttribute("visibility", "visible");
        const lines = opts.series.map((s, i) => {
          let best = s.points[0];
          for (const p of s.points) if (Math.abs(p[0] - fx) < Math.abs(best[0] - fx)) best = p;
          dots[i].setAttribute("cx", x(best[0])); dots[i].setAttribute("cy", y(best[1])); dots[i].setAttribute("visibility", "visible");
          return opts.tipLine ? opts.tipLine(s, best) : `${esc(s.name)}: ${best[1]}`;
        });
        showTip(lines.join("<br>"), e);
      });
      hit.addEventListener("mouseleave", () => {
        cross.setAttribute("visibility", "hidden");
        dots.forEach((d) => d.setAttribute("visibility", "hidden"));
        hideTip();
      });
    });
  }

  /* Grouped bars. categories: [..], series: [{name,color,values:[..]}] */
  function grouped(el, opts) {
    mount(el, (width) => {
      const height = opts.height || 260;
      const fmt = opts.format || ((v) => v);
      const m = { top: 16, right: 12, bottom: 30, left: 44 };
      const plotW = width - m.left - m.right;
      const plotH = height - m.top - m.bottom;
      const y = (v) => m.top + plotH - v * plotH; // 0..1 scale
      const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": opts.ariaLabel || "Grouped bar chart" }, el);
      [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
        svgEl("line", { x1: m.left, x2: width - m.right, y1: y(t), y2: y(t), class: t === 0 ? "axis-line" : "grid-line" }, svg);
        const lab = svgEl("text", { x: m.left - 8, y: y(t) + 4, class: "tick", "text-anchor": "end" }, svg);
        lab.textContent = t;
      });
      const band = plotW / opts.categories.length;
      const inner = band * 0.72;
      const bw = (inner - 2 * (opts.series.length - 1)) / opts.series.length;
      opts.categories.forEach((cat, ci) => {
        const x0 = m.left + ci * band + (band - inner) / 2;
        opts.series.forEach((s, si) => {
          const v = s.values[ci];
          const x = x0 + si * (bw + 2);
          const g = svgEl("g", { class: "bar" }, svg);
          svgEl("path", { d: barPath(x, y(v), bw, y(0) - y(v), 4), style: `fill:${s.color}` }, g);
        });
        const hit = svgEl("rect", { x: m.left + ci * band, y: m.top, width: band, height: plotH, class: "hit" }, svg);
        const html = `<strong>${esc(cat)}</strong><br>` + opts.series.map((s) => `<span class="swatch" style="background:${s.color}"></span>${esc(s.name)}: ${esc(fmt(s.values[ci]))}`).join("<br>");
        hit.addEventListener("mousemove", (e) => showTip(html, e));
        hit.addEventListener("mouseleave", hideTip);
        const t = svgEl("text", { x: m.left + ci * band + band / 2, y: y(0) + 18, class: "tick tick--x", "text-anchor": "middle" }, svg);
        t.textContent = cat;
      });
    });
  }

  function table(el, columns, rows) {
    registry.delete(el);
    el.replaceChildren();
    if (!rows.length) return empty(el);
    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    const head = columns.map((c) => `<th scope="col" class="${c.num ? "num" : ""}">${esc(c.label)}</th>`).join("");
    const body = rows.map((r) => "<tr>" + columns.map((c) => `<td class="${c.num ? "num" : ""}">${esc(c.get(r))}</td>`).join("") + "</tr>").join("");
    wrap.innerHTML = `<table class="data-table compact"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    el.appendChild(wrap);
  }

  window.Charts = { bar, hbar, pairs, line, grouped, table, empty, hideTip, esc };
})();
