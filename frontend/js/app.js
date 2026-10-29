(function () {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const esc = Charts.esc;
  const nf = new Intl.NumberFormat("en-US");
  const int = (v) => nf.format(Math.round(v));
  const pct = (v, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
  const MODEL_COLORS = { gradient_boosting: "var(--series-1)", random_forest: "var(--series-2)" };
  const TEMP_COLORS = { HOT: "var(--hot)", WARM: "var(--warm)", COLD: "var(--cold)" };

  const SECTIONS = {
    overview: { title: "Executive overview", filters: true },
    insights: { title: "Campaign insights", filters: true },
    targeting: { title: "Actionable targeting", filters: true },
    models: { title: "Model performance", filters: false },
  };

  const state = {
    meta: null,
    section: "overview",
    filters: {},
    model: null,
    segment: "job",
    table: { q: "", temperature: "", fatigue: "", sort: "probability", order: "desc", page: 1, pageSize: 25 },
    dirty: { overview: true, insights: true, targeting: true, models: true },
    charts: {},
    tableView: new Set(),
  };

  const controllers = {};
  function query(params, withFilters = true) {
    const qs = new URLSearchParams();
    if (withFilters) {
      for (const [key, set] of Object.entries(state.filters)) for (const v of set) qs.append(key, v);
    }
    for (const [k, v] of Object.entries(params || {})) if (v !== "" && v != null) qs.set(k, v);
    const s = qs.toString();
    return s ? `?${s}` : "";
  }

  async function api(path, { params, slot, filters = true, method = "GET", body } = {}) {
    if (slot) {
      if (controllers[slot]) controllers[slot].abort();
      controllers[slot] = new AbortController();
    }
    const res = await fetch(`/api/${path}${query(params, filters)}`, {
      method,
      signal: slot ? controllers[slot].signal : undefined,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    let data = null;
    try { data = await res.json(); } catch (_) {}
    if (!res.ok) throw new Error((data && data.error) || `Request failed (${res.status})`);
    return data;
  }

  function showError(err) {
    if (err && err.name === "AbortError") return;
    const box = $("#global-error");
    box.textContent = err instanceof TypeError
      ? "Can't reach the dashboard server. Check that `python run.py` is still running, then reload."
      : `Couldn't load data: ${err.message}`;
    box.hidden = false;
  }
  function clearError() { $("#global-error").hidden = true; }

  function chart(id, draw, table) {
    state.charts[id] = { draw, table };
    renderChart(id);
  }
  function renderChart(id) {
    const spec = state.charts[id];
    const el = document.getElementById(id);
    if (!spec || !el) return;
    const btn = document.querySelector(`[data-table-toggle="${id}"]`);
    const asTable = state.tableView.has(id) && spec.table;
    if (btn) btn.textContent = asTable ? "Chart" : "Table";
    if (asTable) Charts.table(el, spec.table.columns, spec.table.rows);
    else spec.draw(el);
  }
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-table-toggle]");
    if (!btn) return;
    const id = btn.dataset.tableToggle;
    state.tableView.has(id) ? state.tableView.delete(id) : state.tableView.add(id);
    renderChart(id);
  });

  const RATE_COLUMNS = (label) => [
    { label, get: (r) => r.category },
    { label: "Customers", num: true, get: (r) => int(r.count) },
    { label: "Subscribed", num: true, get: (r) => int(r.conversions) },
    { label: "Conversion", num: true, get: (r) => pct(r.rate) },
    { label: "Sample", get: (r) => (r.reliable ? "Reliable" : "Small") },
  ];

  function rateTip(r) {
    return `<strong>${esc(r.category)}</strong><br>${pct(r.rate)} converted<br>` +
      `<span class="muted">${int(r.conversions)} of ${int(r.count)} customers</span>` +
      (r.reliable ? "" : `<br><span class="muted">Small sample - read with caution</span>`);
  }

  function rateChart(id, rows, label, color) {
    const total = rows.reduce((a, r) => a + r.count, 0);
    const conv = rows.reduce((a, r) => a + r.conversions, 0);
    chart(id, (el) => Charts.bar(el, {
      rows: rows.map((r) => ({ label: r.category, value: r.rate, faded: !r.reliable, tip: rateTip(r) })),
      format: (v) => pct(v, v < 0.1 && v > 0 ? 1 : 0),
      color: color || "var(--teal)",
      refLine: total ? { value: conv / total, label: `Average ${pct(conv / total)}` } : null,
      ariaLabel: `Conversion rate by ${label}`,
    }), { columns: RATE_COLUMNS(label), rows });
  }

  function activeFilterCount() {
    return Object.values(state.filters).reduce((a, s) => a + s.size, 0);
  }

  function buildFilters() {
    const wrap = $("#filter-controls");
    wrap.replaceChildren();
    for (const f of state.meta.filters) {
      const box = document.createElement("div");
      box.className = "filter";
      box.innerHTML = `
        <button type="button" class="filter__btn" aria-expanded="false" aria-haspopup="true" data-filter="${f.key}">
          <span class="filter__name">${esc(f.label)}</span><span class="caret" aria-hidden="true"></span>
        </button>
        <div class="filter__panel" role="group" aria-label="${esc(f.label)}" hidden>
          ${f.options.map((o) => `<label><input type="checkbox" value="${esc(o)}" data-key="${f.key}">${esc(o)}</label>`).join("")}
        </div>`;
      wrap.appendChild(box);
    }
    wrap.addEventListener("click", (e) => {
      const btn = e.target.closest(".filter__btn");
      if (!btn) return;
      const panel = btn.nextElementSibling;
      const open = panel.hidden;
      closePanels();
      panel.hidden = !open;
      btn.setAttribute("aria-expanded", String(open));
      if (open) { const first = panel.querySelector("input"); if (first) first.focus(); }
    });
    wrap.addEventListener("change", (e) => {
      const input = e.target;
      if (!input.dataset.key) return;
      const set = state.filters[input.dataset.key] || (state.filters[input.dataset.key] = new Set());
      input.checked ? set.add(input.value) : set.delete(input.value);
      if (!set.size) delete state.filters[input.dataset.key];
      filtersChanged();
    });
    document.addEventListener("click", (e) => { if (!e.target.closest(".filter")) closePanels(); });
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      const open = document.querySelector(".filter__panel:not([hidden])");
      if (open) { const btn = open.previousElementSibling; closePanels(); btn.focus(); }
    });
  }

  function closePanels() {
    document.querySelectorAll(".filter__panel").forEach((p) => {
      p.hidden = true;
      p.previousElementSibling.setAttribute("aria-expanded", "false");
    });
  }

  function syncFilterUI() {
    const labels = Object.fromEntries(state.meta.filters.map((f) => [f.key, f.label]));
    document.querySelectorAll(".filter__btn").forEach((btn) => {
      const key = btn.dataset.filter;
      const n = state.filters[key] ? state.filters[key].size : 0;
      btn.classList.toggle("is-active", n > 0);
      btn.querySelector(".filter__name").textContent = n ? `${labels[key]} (${n})` : labels[key];
    });
    document.querySelectorAll(".filter__panel input").forEach((i) => {
      i.checked = !!(state.filters[i.dataset.key] && state.filters[i.dataset.key].has(i.value));
    });
    const chips = $("#filter-chips");
    chips.replaceChildren();
    for (const [key, set] of Object.entries(state.filters)) {
      for (const v of set) {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.innerHTML = `${esc(labels[key])}: ${esc(v)} <button type="button" aria-label="Remove ${esc(labels[key])} ${esc(v)}">×</button>`;
        chip.querySelector("button").addEventListener("click", () => {
          set.delete(v);
          if (!set.size) delete state.filters[key];
          filtersChanged();
        });
        chips.appendChild(chip);
      }
    }
    if (activeFilterCount() > 1) {
      const clear = document.createElement("button");
      clear.type = "button";
      clear.className = "chip chip--clear";
      clear.textContent = "Clear all filters";
      clear.addEventListener("click", () => { state.filters = {}; filtersChanged(); });
      chips.appendChild(clear);
    }
  }

  let filterTimer;
  function filtersChanged() {
    syncFilterUI();
    state.dirty.overview = state.dirty.insights = state.dirty.targeting = true;
    state.table.page = 1;
    clearTimeout(filterTimer);
    filterTimer = setTimeout(() => load(state.section), 120);
  }

  function route() {
    const name = (location.hash || "#overview").slice(1);
    const section = SECTIONS[name] ? name : "overview";
    state.section = section;
    document.querySelectorAll("[data-section]").forEach((s) => { s.hidden = s.dataset.section !== section; });
    document.querySelectorAll("[data-nav]").forEach((a) => {
      if (a.dataset.nav === section) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
    $("#page-title").textContent = SECTIONS[section].title;
    $("#filterbar").hidden = !SECTIONS[section].filters;
    document.title = `${SECTIONS[section].title} - Marketing Decision Support`;
    setSource();
    Charts.hideTip();
    load(section);
  }

  function setSource() {
    const m = state.meta;
    if (!m) return;
    const modelName = m.models.find((x) => x.key === state.model).name;
    const src = {
      overview: `Historical outcomes for ${int(m.dataset.customers)} contacted customers. Lead counts are model-predicted.`,
      insights: "Historical conversion rates. These are associations, not causes.",
      targeting: `${modelName} predictions for ${int(m.dataset.test_customers)} held-out customers the models never trained on.`,
      models: `Both models evaluated on the same ${int(m.dataset.test_customers)} held-out customers.`,
    };
    $("#page-source").textContent = src[state.section];
  }

  async function load(section) {
    if (!state.meta || !state.dirty[section]) return;
    const el = document.getElementById(`section-${section}`);
    el.classList.add("is-loading");
    try {
      await ({ overview: loadOverview, insights: loadInsights, targeting: loadTargeting, models: loadModels })[section]();
      state.dirty[section] = false;
      clearError();
    } catch (err) {
      showError(err);
    } finally {
      el.classList.remove("is-loading");
    }
  }

  async function loadOverview() {
    const [ov, tg] = await Promise.all([
      api("overview", { slot: "overview" }),
      api("targeting/summary", { slot: "overview-leads", params: { model: state.meta.production_model.key } }),
    ]);
    const k = ov.kpis;
    const hero = $("#hero");
    if (!k.customers_contacted) {
      hero.innerHTML = "No customers match these filters. Remove a filter to see results.";
      $("#overview-kpis").replaceChildren();
      ["ov-month", "ov-volume"].forEach((id) => { delete state.charts[id]; Charts.empty(document.getElementById(id)); });
      renderLeadBar(tg);
      return;
    }
    const filtered = activeFilterCount() > 0;
    const diff = (k.conversion_rate - ov.baseline_rate) * 100;
    hero.innerHTML =
      `<b>${int(k.subscriptions)}</b> of <b>${int(k.customers_contacted)}</b> ${filtered ? "customers in this selection" : "contacted customers"} subscribed to a term deposit, a <b>${pct(k.conversion_rate)}</b> conversion rate.` +
      `<span class="delta">${filtered
        ? `${diff >= 0 ? "+" : "−"}${Math.abs(diff).toFixed(1)} percentage points compared with all customers (${pct(ov.baseline_rate)}).`
        : "Use the filters to compare any group against this baseline."}</span>`;

    const reliableMonths = ov.by_month.filter((r) => r.reliable);
    const topMonth = reliableMonths.sort((a, b) => b.rate - a.rate)[0];
    const best = ov.best_segment;
    $("#overview-kpis").innerHTML = [
      kpi("Contacts per customer", k.avg_contacts.toFixed(1), "average in the current campaign"),
      kpi("Best job segment", best ? best.category : "–", best ? `${pct(best.conversion_rate)} of ${int(best.count)}` : "no group of 30+"),
      kpi("Strongest month", topMonth ? capital(topMonth.category) : "–", topMonth ? `${pct(topMonth.rate)} of ${int(topMonth.count)}` : "no month with 30+"),
      kpi("Hot leads", int(tg.kpis.hot), `of ${int(tg.kpis.scored)} scored test customers`),
    ].join("");

    rateChart("ov-month", ov.by_month.map((r) => ({ ...r, category: capital(r.category) })), "Month");
    chart("ov-volume", (el) => Charts.bar(el, {
      rows: ov.by_month.map((r) => ({
        label: capital(r.category), value: r.count,
        tip: `<strong>${capital(r.category)}</strong><br>${int(r.count)} customers contacted<br><span class="muted">${int(r.conversions)} subscribed</span>`,
      })),
      format: (v, axis) => (axis && v >= 1000 ? `${v / 1000}k` : int(v)),
      color: "var(--volume)",
      ariaLabel: "Customers contacted by month",
    }), { columns: RATE_COLUMNS("Month"), rows: ov.by_month });
    renderLeadBar(tg);
  }

  function renderLeadBar(tg) {
    const k = tg.kpis;
    const name = state.meta.production_model.name;
    $("#ov-leads-sub").textContent = k.scored
      ? `${name} scores for ${int(k.scored)} held-out customers matching the filters.`
      : "No held-out customers match these filters.";
    const box = $("#ov-leads");
    if (!k.scored) { box.replaceChildren(); return; }
    const parts = [["HOT", k.hot], ["WARM", k.warm], ["COLD", k.cold]];
    box.innerHTML =
      `<div class="leadbar__track" role="img" aria-label="${parts.map(([t, n]) => `${t} ${n}`).join(", ")}">` +
      parts.map(([t, n]) => n ? `<span class="leadbar__seg" style="flex:${n};background:${TEMP_COLORS[t]}" title="${t}: ${int(n)}"></span>` : "").join("") +
      `</div><div class="leadbar__legend">` +
      parts.map(([t, n]) => `<span><span class="swatch" style="background:${TEMP_COLORS[t]}"></span>${capital(t.toLowerCase())} <b>${int(n)}</b> ${pct(n / k.scored, 0)}</span>`).join("") +
      `<span>High contact fatigue <b>${int(k.high_fatigue)}</b></span></div>`;
  }

  function kpi(label, value, note, extra = "") {
    return `<div class="kpi ${extra}"><span class="kpi__label">${esc(label)}</span><span class="kpi__value" title="${esc(value)}">${esc(value)}</span><span class="kpi__note">${esc(note)}</span></div>`;
  }
  const capital = (s) => s.charAt(0).toUpperCase() + s.slice(1);

  function buildInsights() {
    const wrap = $("#drivers");
    wrap.innerHTML = state.meta.drivers.map((d) => `
      <figure class="card">
        <figcaption class="card__head"><h2>Conversion by ${esc(d.label.toLowerCase())}</h2><button class="link-btn" data-table-toggle="dr-${d.key}">Table</button></figcaption>
        <div class="chart" id="dr-${d.key}"></div>
      </figure>`).join("");
    const picker = $("#segment-picker");
    picker.innerHTML = state.meta.segments.map((s) =>
      `<button type="button" role="radio" aria-checked="${s.key === state.segment}" data-segment="${s.key}">${esc(s.label)}</button>`).join("");
    picker.addEventListener("click", (e) => {
      const b = e.target.closest("[data-segment]");
      if (!b || b.dataset.segment === state.segment) return;
      state.segment = b.dataset.segment;
      picker.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
      loadSegment().catch(showError);
    });
    $("#insights-note").textContent =
      `Faded bars with a dashed outline cover fewer than ${state.meta.thresholds.min_reliable_sample} customers in the current selection, so treat them as low-confidence. ` +
      "Call duration is only known after a call ends: it describes what happened, it is not a lever.";
  }

  async function loadInsights() {
    const [dr] = await Promise.all([api("insights/drivers", { slot: "drivers" }), loadSegment()]);
    for (const d of dr.drivers) {
      const rows = d.key === "month" ? d.rows.map((r) => ({ ...r, category: capital(r.category) })) : d.rows;
      rateChart(`dr-${d.key}`, rows, d.label);
    }
  }

  async function loadSegment() {
    const seg = await api("insights/segments", { slot: "segment", params: { segment: state.segment } });
    rateChart("seg-chart", seg.rows, seg.label);
  }

  function buildTargeting() {
    const picker = $("#model-picker");
    picker.innerHTML = state.meta.models.map((m) =>
      `<button type="button" role="radio" aria-checked="${m.key === state.model}" data-model="${m.key}">
        <span class="swatch" style="background:${MODEL_COLORS[m.key]}"></span>${esc(m.name)}${m.production ? ' <span class="tag">In use</span>' : ""}</button>`).join("");
    picker.addEventListener("click", (e) => {
      const b = e.target.closest("[data-model]");
      if (!b || b.dataset.model === state.model) return;
      state.model = b.dataset.model;
      picker.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
      state.table.page = 1;
      state.dirty.targeting = true;
      setSource();
      load("targeting");
    });

    $("#tg-kpis").addEventListener("click", (e) => {
      const b = e.target.closest("button.kpi");
      if (!b) return;
      const t = state.table;
      if (b.dataset.temp !== undefined) t.temperature = t.temperature === b.dataset.temp ? "" : b.dataset.temp;
      if (b.dataset.fatigue !== undefined) t.fatigue = t.fatigue === b.dataset.fatigue ? "" : b.dataset.fatigue;
      if (b.dataset.reset !== undefined) { t.temperature = ""; t.fatigue = ""; }
      $("#tc-temp").value = t.temperature;
      $("#tc-fatigue").value = t.fatigue;
      t.page = 1;
      markKpis();
      loadTable().catch(showError);
    });

    let qTimer;
    $("#tc-q").addEventListener("input", (e) => {
      const digits = e.target.value.replace(/\D/g, "");
      if (digits !== e.target.value) e.target.value = digits;
      clearTimeout(qTimer);
      qTimer = setTimeout(() => { state.table.q = digits; state.table.page = 1; loadTable().catch(showError); }, 250);
    });
    $("#tc-temp").addEventListener("change", (e) => { state.table.temperature = e.target.value; state.table.page = 1; markKpis(); loadTable().catch(showError); });
    $("#tc-fatigue").addEventListener("change", (e) => { state.table.fatigue = e.target.value; state.table.page = 1; markKpis(); loadTable().catch(showError); });

    document.querySelectorAll("#customer-table .sort").forEach((b) => b.addEventListener("click", () => {
      const t = state.table;
      if (t.sort === b.dataset.sort) t.order = t.order === "desc" ? "asc" : "desc";
      else { t.sort = b.dataset.sort; t.order = "desc"; }
      t.page = 1;
      loadTable().catch(showError);
    }));

    const tbody = $("#customer-table tbody");
    tbody.addEventListener("click", (e) => { const tr = e.target.closest("tr[data-id]"); if (tr) openCustomer(+tr.dataset.id); });
    tbody.addEventListener("keydown", (e) => {
      const tr = e.target.closest("tr[data-id]");
      if (tr && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openCustomer(+tr.dataset.id); }
    });

    $("#pager").addEventListener("click", (e) => {
      const b = e.target.closest("[data-page]");
      if (!b) return;
      state.table.page = +b.dataset.page;
      loadTable().then(() => $("#customer-table").scrollIntoView({ block: "nearest" })).catch(showError);
    });

    $("#export-csv").addEventListener("click", () => {
      const a = document.createElement("a");
      a.href = `/api/targeting/export${query(tableParams())}`;
      a.download = "";
      document.body.appendChild(a);
      a.click();
      a.remove();
    });

    $("#open-score").addEventListener("click", openScoreDialog);
  }

  function tableParams() {
    const t = state.table;
    return { model: state.model, temperature: t.temperature, fatigue: t.fatigue, q: t.q, sort: t.sort, order: t.order };
  }

  async function loadTargeting() {
    const [sum] = await Promise.all([
      api("targeting/summary", { slot: "tg-summary", params: { model: state.model } }),
      loadTable(),
    ]);
    const k = sum.kpis;
    const share = (n) => (k.scored ? pct(n / k.scored, 0) : "–");
    $("#tg-kpis").innerHTML = [
      `<button type="button" class="kpi" data-reset><span class="kpi__label">Customers scored</span><span class="kpi__value">${int(k.scored)}</span><span class="kpi__note">avg probability ${pct(k.mean_probability)}</span></button>`,
      `<button type="button" class="kpi kpi--hot" data-temp="HOT"><span class="kpi__label">Hot leads</span><span class="kpi__value">${int(k.hot)}</span><span class="kpi__note">${share(k.hot)}, probability 70%+</span></button>`,
      `<button type="button" class="kpi kpi--warm" data-temp="WARM"><span class="kpi__label">Warm leads</span><span class="kpi__value">${int(k.warm)}</span><span class="kpi__note">${share(k.warm)}, 40–70%</span></button>`,
      `<button type="button" class="kpi kpi--cold" data-temp="COLD"><span class="kpi__label">Cold leads</span><span class="kpi__value">${int(k.cold)}</span><span class="kpi__note">${share(k.cold)}, under 40%</span></button>`,
      `<button type="button" class="kpi kpi--hot" data-fatigue="HIGH"><span class="kpi__label">High contact fatigue</span><span class="kpi__value">${int(k.high_fatigue)}</span><span class="kpi__note">5+ contacts this campaign</span></button>`,
    ].join("");
    markKpis();

    const hot = state.meta.thresholds.hot, warm = state.meta.thresholds.warm;
    const band = (from) => (from >= hot - 1e-9 ? "HOT" : from >= warm - 1e-9 ? "WARM" : "COLD");
    const width = sum.histogram.length ? sum.histogram[0].to - sum.histogram[0].from : 0.05;
    chart("tg-hist", (el) => Charts.bar(el, {
      rows: sum.histogram.map((b) => ({
        label: `${Math.round(b.from * 100)}–${Math.round(b.to * 100)}%`,
        value: b.count, color: TEMP_COLORS[band(b.from)],
        tip: `<strong>${Math.round(b.from * 100)}–${Math.round(b.to * 100)}% probability</strong><br>${int(b.count)} customers<br><span class="muted">${band(b.from)} lead</span>`,
      })),
      hideLabels: true,
      axisBottom: [0, 0.2, 0.4, 0.6, 0.8, 1].map((v) => ({ at: v / width, label: `${v * 100}%` })),
      markers: [{ at: warm / width, label: "Warm" }, { at: hot / width, label: "Hot" }],
      format: (v, axis) => (axis && v >= 1000 ? `${v / 1000}k` : int(v)),
      clip: true,
      height: 260,
      emptyText: "No customers match these filters.",
      ariaLabel: "Distribution of predicted conversion probability",
    }), {
      columns: [
        { label: "Probability", get: (b) => `${Math.round(b.from * 100)}–${Math.round(b.to * 100)}%` },
        { label: "Lead", get: (b) => band(b.from) },
        { label: "Customers", num: true, get: (b) => int(b.count) },
      ],
      rows: sum.histogram,
    });

    chart("tg-recs", (el) => Charts.hbar(el, {
      rows: sum.recommendations.map((r) => ({ label: r.label, value: r.count, display: int(r.count) })),
      color: "var(--teal)",
      emptyText: "No customers match these filters.",
    }));
  }

  function markKpis() {
    const t = state.table;
    document.querySelectorAll("#tg-kpis button.kpi").forEach((b) => {
      const on = (b.dataset.temp !== undefined && b.dataset.temp === t.temperature) ||
        (b.dataset.fatigue !== undefined && b.dataset.fatigue === t.fatigue);
      b.setAttribute("aria-pressed", String(on));
    });
  }

  async function loadTable() {
    const t = state.table;
    const data = await api("targeting/customers", { slot: "tg-table", params: { ...tableParams(), page: t.page, page_size: t.pageSize } });
    const ix = Object.fromEntries(data.columns.map((c, i) => [c, i]));
    $("#tg-count").textContent = `(${int(data.total)})`;
    const tbody = $("#customer-table tbody");
    if (!data.rows.length) {
      tbody.innerHTML = `<tr><td colspan="10" class="chart-empty">${t.q ? `No customer with an ID containing "${esc(t.q)}" in this selection.` : "No customers match these filters."}</td></tr>`;
    } else {
      tbody.innerHTML = data.rows.map((r) => {
        const p = r[ix.conversion_probability];
        return `<tr tabindex="0" data-id="${r[ix.customer_record_id]}" aria-label="Open customer ${r[ix.customer_record_id]}">
          <td>${r[ix.customer_record_id]}</td>
          <td class="num">${r[ix.age]}</td>
          <td>${esc(r[ix.job])}</td>
          <td class="num">${int(r[ix.balance])}</td>
          <td class="num">${r[ix.campaign]}</td>
          <td>${esc(r[ix.poutcome])}</td>
          <td class="num"><span class="prob"><span class="prob__bar"><span style="width:${p * 100}%"></span></span>${pct(p)}</span></td>
          <td><span class="pill pill--${r[ix.lead_temperature]}">${r[ix.lead_temperature]}</span></td>
          <td><span class="pill pill--${r[ix.contact_fatigue]}">${r[ix.contact_fatigue]}</span></td>
          <td class="rec">${esc(r[ix.recommendation])}</td></tr>`;
      }).join("");
    }
    document.querySelectorAll("#customer-table .sort").forEach((b) => {
      if (b.dataset.sort === t.sort) b.setAttribute("aria-sort", t.order === "desc" ? "descending" : "ascending");
      else b.removeAttribute("aria-sort");
    });
    const from = data.total ? (data.page - 1) * data.page_size + 1 : 0;
    const to = Math.min(data.total, data.page * data.page_size);
    $("#pager").innerHTML = `<span>Showing ${int(from)}–${int(to)} of ${int(data.total)}</span>
      <span class="pager__btns">
        <button class="btn" data-page="${data.page - 1}" ${data.page <= 1 ? "disabled" : ""}>Previous</button>
        <span style="align-self:center">Page ${data.page} of ${data.pages}</span>
        <button class="btn" data-page="${data.page + 1}" ${data.page >= data.pages ? "disabled" : ""}>Next</button>
      </span>`;
  }

  const FIELD_LABELS = {
    age: "Age", job: "Job", marital: "Marital status", education: "Education", default: "Credit in default",
    balance: "Account balance", housing: "Housing loan", loan: "Personal loan", contact: "Contact type",
    day: "Last contact day", month: "Last contact month", duration: "Last call duration (s)",
    campaign: "Contacts this campaign", pdays: "Days since previous campaign (-1 = never)",
    previous: "Contacts before this campaign", poutcome: "Previous campaign outcome",
  };

  function decisionBlock(r) {
    return `<div class="decision">
      <div class="decision__row"><span class="pill pill--${r.lead_temperature}">${r.lead_temperature} lead</span>
        <span class="pill pill--${r.contact_fatigue}">${r.contact_fatigue} contact fatigue</span>
        <span>${esc(r.interaction_pattern)}</span></div>
      ${r.interaction_description ? `<p class="muted">${esc(r.interaction_description)}</p>` : ""}
      <p class="recommendation">${esc(r.recommendation)}</p></div>`;
  }

  function modelProbs(probs) {
    return `<div class="detail__models">` + state.meta.models.map((m) =>
      `<span><span class="swatch" style="background:${MODEL_COLORS[m.key]}"></span>${esc(m.name)} <b>${pct(probs[m.key])}</b></span>`).join("") + `</div>`;
  }

  async function openCustomer(id) {
    const dlg = $("#customer-dialog");
    const box = $("#customer-detail");
    box.innerHTML = `<p class="muted">Loading customer ${id}…</p>`;
    if (!dlg.open) dlg.showModal();
    try {
      const c = await api(`customers/${id}`, { params: { model: state.model }, filters: false });
      const modelName = state.meta.models.find((m) => m.key === c.model).name;
      box.innerHTML = `<div class="detail">
        <div><h2 id="cd-title">Customer ${c.customer_record_id}</h2>
          <p class="muted">Held-out test customer. ${esc(modelName)} conversion probability:</p></div>
        <div class="detail__prob">${pct(c.conversion_probability)}</div>
        ${modelProbs(c.probability_by_model)}
        ${decisionBlock(c)}
        <dl class="dl">${Object.entries(c.record).map(([k, v]) => `<div><dt>${esc(FIELD_LABELS[k] || k)}</dt><dd>${esc(typeof v === "number" ? int(v) : v)}</dd></div>`).join("")}</dl>
      </div>`;
    } catch (err) {
      box.innerHTML = `<p class="form-error">${esc(err.message)}</p>`;
    }
  }

  const FORM_DEFAULTS = { age: 35, balance: 1000, day: 15, duration: 180, campaign: 2, pdays: -1, previous: 0,
    job: "management", marital: "married", education: "secondary", default: "no", housing: "yes", loan: "no",
    contact: "cellular", month: "may", poutcome: "unknown" };
  const FORM_ORDER = ["age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"];

  function buildScoreForm() {
    const f = state.meta.form;
    const form = $("#score-form");
    form.innerHTML = FORM_ORDER.map((key) => {
      const label = esc(FIELD_LABELS[key]);
      if (f.numeric[key]) {
        const r = f.numeric[key];
        return `<label>${label}<input type="number" name="${key}" required step="1" min="${r.min}" max="${r.max}" value="${FORM_DEFAULTS[key]}"></label>`;
      }
      const opts = f.categorical[key];
      const def = opts.includes(FORM_DEFAULTS[key]) ? FORM_DEFAULTS[key] : opts[0];
      return `<label>${label}<select name="${key}">${opts.map((o) => `<option ${o === def ? "selected" : ""}>${esc(o)}</option>`).join("")}</select></label>`;
    }).join("") +
      `<div class="actions"><button class="btn btn--primary" type="submit">Score customer</button>
       <span class="hint">Call duration is assumed known: this supports follow-up after a call.</span></div>`;
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const out = $("#score-result");
      let firstBad = null;
      form.querySelectorAll("input").forEach((i) => {
        const ok = i.checkValidity();
        i.setAttribute("aria-invalid", String(!ok));
        if (!ok && !firstBad) firstBad = i;
      });
      if (firstBad) {
        out.innerHTML = `<p class="form-error">${esc(FIELD_LABELS[firstBad.name])} must be a whole number between ${firstBad.min} and ${firstBad.max}.</p>`;
        firstBad.focus();
        return;
      }
      const body = {};
      new FormData(form).forEach((v, k) => { body[k] = f.numeric[k] ? Number(v) : v; });
      const btn = form.querySelector("[type=submit]");
      btn.disabled = true;
      try {
        const res = await api("predict", { method: "POST", body, params: { model: state.model }, filters: false });
        const name = state.meta.models.find((m) => m.key === res.model).name;
        out.innerHTML = `<div class="score-result">
          <div><p class="muted">${esc(name)} conversion probability</p><div class="detail__prob">${pct(res.result.conversion_probability)}</div>${modelProbs(res.probability_by_model)}</div>
          ${decisionBlock(res.result)}</div>`;
      } catch (err) {
        out.innerHTML = `<p class="form-error">${esc(err.message)}</p>`;
      } finally {
        btn.disabled = false;
      }
    });
  }

  function openScoreDialog() {
    $("#score-result").replaceChildren();
    $("#score-dialog").showModal();
  }

  async function loadModels() {
    const r = await api("models", { slot: "models", filters: false });
    const models = state.meta.models;
    const byKey = Object.fromEntries(r.comparison.map((c) => [c.key, c]));
    const prod = models.find((m) => m.key === r.production_model);
    $("#models-lede").textContent =
      `${prod.name} generates the targeting scores (chosen by ${r.selected_by}). Both models use the same preprocessed features ` +
      `and are scored on the same ${int(r.test_customers)} held-out customers, ${int(r.test_positives)} of whom subscribed. ` +
      `Classification metrics use a ${r.threshold} probability threshold.`;
    const legend = models.map((m) => `<span><span class="swatch" style="background:${MODEL_COLORS[m.key]}"></span>${esc(m.name)}${m.production ? " (in use)" : ""}</span>`).join("");
    $("#models-legend").innerHTML = legend;
    document.querySelectorAll("#section-models figure .card__head").forEach((h) => {
      if (!h.nextElementSibling.classList.contains("legend")) h.insertAdjacentHTML("afterend", `<div class="legend legend--row">${legend}</div>`);
    });

    const series = models.map((m) => ({ name: m.name, color: MODEL_COLORS[m.key], values: r.metrics.map((k) => byKey[m.key][k]) }));
    chart("md-metrics", (el) => Charts.grouped(el, { categories: r.metrics, series, format: (v) => v.toFixed(3), height: 240, ariaLabel: "Metric comparison" }));

    const best = Object.fromEntries(r.metrics.map((k) => [k, Math.max(...r.comparison.map((c) => c[k]))]));
    $("#md-table").innerHTML = `<div class="table-wrap"><table class="data-table compact"><thead><tr><th scope="col">Model</th>${r.metrics.map((k) => `<th scope="col" class="num">${esc(k)}</th>`).join("")}</tr></thead><tbody>` +
      models.map((m) => `<tr><td><span class="swatch" style="background:${MODEL_COLORS[m.key]}"></span>${esc(m.name)}</td>` +
        r.metrics.map((k) => { const v = byKey[m.key][k]; return `<td class="num">${v === best[k] ? `<b>${v.toFixed(4)}</b>` : v.toFixed(4)}</td>`; }).join("") + "</tr>").join("") +
      `</tbody></table></div>`;

    chart("md-roc", (el) => Charts.line(el, {
      series: models.map((m) => ({ name: `${m.name} (AUC ${byKey[m.key]["ROC-AUC"].toFixed(3)})`, color: MODEL_COLORS[m.key], points: r.roc[m.key] })),
      diagonal: true, xLabel: "False positive rate", yLabel: "True positive rate", height: 440,
      tipLine: (s, p) => `<span class="swatch" style="background:${s.color}"></span>${esc(s.name.split(" (")[0])}: catches ${pct(p[1])} of subscribers, ${pct(p[0])} false alarms <span class="muted">at threshold ${p[2].toFixed(2)}</span>`,
      ariaLabel: "ROC curves for both models",
    }));

    const order = r.importances[r.production_model].slice(0, 10).map((x) => x.feature);
    const lookup = Object.fromEntries(models.map((m) => [m.key, Object.fromEntries(r.importances[m.key].map((x) => [x.feature, x.importance]))]));
    chart("md-importance", (el) => Charts.pairs(el, {
      rows: order.map((f) => ({ label: f, values: models.map((m) => lookup[m.key][f] || 0) })),
      series: models.map((m) => ({ name: m.name, color: MODEL_COLORS[m.key] })),
      format: (v) => pct(v),
    }));

    $("#md-confusion").innerHTML = models.map((m) => {
      const [[tn, fp], [fn, tp]] = r.confusion[m.key];
      const cell = (n, rowTotal, label, good) => `<div class="cm__cell ${good ? "cm__cell--good" : ""}"><b>${int(n)}</b><span>${label}, ${pct(n / rowTotal)} of row</span></div>`;
      return `<div class="cm"><h3><span class="swatch" style="background:${MODEL_COLORS[m.key]}"></span>${esc(m.name)}</h3>
        <div class="cm__grid" role="table" aria-label="${esc(m.name)} confusion matrix">
          <div></div><div class="cm__axis">Predicted no</div><div class="cm__axis">Predicted yes</div>
          <div class="cm__axis">Did not subscribe</div>${cell(tn, tn + fp, "correct", true)}${cell(fp, tn + fp, "false alarm", false)}
          <div class="cm__axis">Subscribed</div>${cell(fn, fn + tp, "missed", false)}${cell(tp, fn + tp, "caught", true)}
        </div></div>`;
    }).join("");
  }

  async function init() {
    try {
      state.meta = await api("meta", { filters: false });
    } catch (err) {
      showError(err);
      $("#rail-note").textContent = "Server not reachable.";
      return;
    }
    state.model = state.meta.production_model.key;
    $("#rail-note").innerHTML = `Targeting model: <strong>${esc(state.meta.production_model.name)}</strong>, chosen by ${esc(state.meta.production_model.selected_by)}.`;
    buildFilters();
    syncFilterUI();
    buildInsights();
    buildTargeting();
    buildScoreForm();
    window.addEventListener("hashchange", route);
    route();
  }

  init();
})();
