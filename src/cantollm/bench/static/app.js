/* CantoLLM bench control panel. Hash-routed, 1 s polling, no build step.
   Charts: hand-rolled SVG bars per the dataviz mark specs — thin marks,
   4px rounded data ends, hairline grid, hover tooltips, one axis each. */

"use strict";

const $ = (sel, el) => (el || document).querySelector(sel);
const view = $("#view");
const tooltip = $("#tooltip");
let pollTimer = null;

/* Model output + prompt text is untrusted — always escape before innerHTML. */
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[c]));

const fmt = {
  num(v, digits = 1) {
    return v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(digits);
  },
  secs(v) { return v == null ? "–" : `${Number(v).toFixed(2)}s`; },
  pct(v) {
    if (v == null) return "";
    const s = v >= 0 ? "+" : "";
    return `${s}${v.toFixed(1)}%`;
  },
};

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (e) { /* keep */ }
    throw new Error(detail);
  }
  return r.json();
}

/* ── router ─────────────────────────────────────────────────────── */

const routes = {
  "#/launch": renderLaunch,
  "#/live": renderLive,
  "#/history": renderHistory,
  "#/compare": renderCompare,
};

function navigate() {
  const hash = location.hash.split("?")[0] || "#/launch";
  document.querySelectorAll("nav a").forEach((a) =>
    a.classList.toggle("here", a.getAttribute("href") === hash));
  clearInterval(pollTimer);
  const route = routes[hash] || (hash.startsWith("#/run/") ? renderRunDetail : renderLaunch);
  route().catch((e) => { view.innerHTML = `<p class="error">${e.message}</p>`; });
}
window.addEventListener("hashchange", navigate);

/* active-run pill in the header, refreshed on every poll */
async function refreshPill() {
  try {
    const { active } = await api("/api/runs");
    const pill = $("#active-pill");
    if (active) {
      pill.textContent = `● running: ${active.config_name}`;
      pill.classList.remove("hidden");
      pill.onclick = () => { location.hash = "#/live"; };
    } else {
      pill.classList.add("hidden");
    }
  } catch (e) { /* server going away is fine */ }
}

/* ── Launch ─────────────────────────────────────────────────────── */

async function renderLaunch() {
  const configs = await api("/api/configs");
  const workloads = await api("/api/workloads");
  view.innerHTML = `
    <h2>Launch a run</h2>
    <div class="card">
      <div class="row">
        <select id="cfg">${configs.map((c) =>
          `<option value="${c.name}" ${c.error ? "disabled" : ""}>
             ${c.name}${c.error ? " (broken)" : ` — ${c.n_cells} cells`}</option>`).join("")}
        </select>
        <input type="text" id="attach-url" placeholder="attach URL (attach-mode configs only)" size="34">
        <label class="check"><input type="checkbox" id="capture-text" checked>
          capture output text</label>
        <button class="primary" id="start">Start run</button>
        <span id="launch-msg" class="sub"></span>
      </div>
      <div id="preview"></div>
    </div>
    <h3>Workloads</h3>
    <div class="card"><table>
      <tr><th>set</th><th>prompts</th><th>input tokens (min–max)</th><th>verified</th></tr>
      ${workloads.map((w) => `<tr>
        <td class="name">${w.name}</td><td>${w.prompts ?? "–"}</td>
        <td>${w.input_tokens_min ?? "?"}–${w.input_tokens_max ?? "?"}</td>
        <td>${w.error ? `<span class="error">${w.error}</span>` : w.verified ? "yes" : "no"}</td>
      </tr>`).join("")}
    </table></div>`;

  const preview = async () => {
    const name = $("#cfg").value;
    if (!name) return;
    try {
      const exp = await api(`/api/configs/${name}/expand`);
      $("#preview").innerHTML = `
        <h3>Matrix preview — ${exp.cells.length} cells${exp.attach ? " (attach mode)" : ""}</h3>
        <table>
          <tr><th>cell</th><th>server</th><th>workload</th><th>mode</th><th>level</th><th>requests</th></tr>
          ${exp.cells.map((c) => `<tr>
            <td class="mono">${c.cell_id}</td>
            <td class="sub">${c.server ? Object.entries(c.server)
              .filter(([k]) => !["port", "health_timeout_s", "respawn", "in_process"].includes(k))
              .map(([k, v]) => `${k}=${v}`).join(" ") : "attach"}</td>
            <td>${c.workload}</td><td>${c.mode}</td><td>${c.level}</td><td>${c.requests}</td>
          </tr>`).join("")}
        </table>`;
    } catch (e) { $("#preview").innerHTML = `<p class="error">${e.message}</p>`; }
  };
  $("#cfg").onchange = preview;
  if (configs.length) await preview();

  $("#start").onclick = async () => {
    $("#start").disabled = true;
    $("#launch-msg").textContent = "starting…";
    try {
      const body = { config: $("#cfg").value };
      const url = $("#attach-url").value.trim();
      if (url) body.attach_url = url;
      body.capture_text = $("#capture-text").checked;
      await api("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      location.hash = "#/live";
    } catch (e) {
      $("#launch-msg").innerHTML = `<span class="error">${e.message}</span>`;
      $("#start").disabled = false;
    }
  };
}

/* ── Live ───────────────────────────────────────────────────────── */

async function renderLive() {
  const draw = async () => {
    const { active } = await api("/api/runs");
    await refreshPill();
    if (!active) {
      clearInterval(pollTimer);
      view.innerHTML = `<h2>Live</h2><div class="card">
        <p>No active run. <a href="#/launch">Launch one</a> or check
        <a href="#/history">History</a> for the latest results.</p></div>`;
      return;
    }
    const run = await api(`/api/runs/${active.run_id}`).catch(() => null);
    const done = active.cell_index;
    const total = active.n_cells || 1;
    view.innerHTML = `
      <h2>Live — <span class="mono">${active.run_id}</span></h2>
      <div class="card">
        <div class="row spread">
          <div>cell <b>${Math.min(done + 1, total)}/${total}</b>
               · repeat <b>${active.repeat_index + 1}</b>
               · requests done <b>${active.requests_done}</b></div>
          <button class="danger" id="abort">Abort run</button>
        </div>
        <div class="progress" style="margin-top:10px"><div style="width:${(done / total) * 100}%"></div></div>
      </div>
      ${run ? cellsTable(run.cells) : ""}`;
    $("#abort").onclick = async () => {
      if (!confirm("Abort the active run? Partials are kept.")) return;
      await api(`/api/runs/${active.run_id}/abort`, { method: "POST" });
    };
  };
  await draw();
  pollTimer = setInterval(() => draw().catch(() => {}), 1000);
}

/* ── History + run detail ───────────────────────────────────────── */

async function renderHistory() {
  const { history } = await api("/api/runs");
  view.innerHTML = `
    <h2>History</h2>
    <div class="card"><table>
      <tr><th>run</th><th>status</th><th>started</th><th>cells</th><th>device</th><th>git</th><th>warnings</th></tr>
      ${history.map((r) => `<tr>
        <td class="name"><a href="#/run/${r.run_id}" class="mono">${r.run_id}</a></td>
        <td><span class="status ${r.status}">${r.status}</span></td>
        <td class="sub">${r.started || "–"}</td>
        <td>${r.n_cells}</td>
        <td class="sub">${(r.env || {}).device_name || "–"}</td>
        <td class="mono">${((r.env || {}).git_sha || "").slice(0, 7)}${(r.env || {}).git_dirty ? "*" : ""}</td>
        <td>${r.warnings ? `<span class="warnbadge">${r.warnings}</span>` : ""}</td>
      </tr>`).join("") || "<tr><td>no runs yet</td></tr>"}
    </table></div>`;
}

async function renderRunDetail() {
  const runId = location.hash.slice("#/run/".length);
  const run = await api(`/api/runs/${runId}`);
  const cells = (run.cells || []).filter((c) => (c.median || {}).aggregate_tok_s != null);
  view.innerHTML = `
    <h2><span class="mono">${run.run_id}</span>
        <span class="status ${run.status}">${run.status}</span></h2>
    <div class="card sub">
      ${(run.env || {}).device_name || "?"} · torch ${(run.env || {}).torch || "?"}
      · git <span class="mono">${((run.env || {}).git_sha || "").slice(0, 7)}</span>
      · ${run.started || ""} → ${run.finished || "…"}
    </div>
    ${cells.length ? `<div class="card chart" id="chart"></div>` : ""}
    ${cellsTable(run.cells || [], true)}
    <div id="cell-detail"></div>`;
  document.querySelectorAll("tr.cellrow").forEach((tr) => {
    tr.onclick = () => showCellRequests(run.run_id, tr.dataset.cell);
  });
  if (cells.length) {
    barChart($("#chart"), {
      title: "Aggregate throughput by cell (tok/s, median of repeats)",
      unit: "tok/s",
      groups: cells.map((c) => ({
        label: `${c.workload} ${c.mode[0]}@${c.level}`,
        values: [{ name: "agg tok/s", value: c.median.aggregate_tok_s, color: "var(--series-1)" }],
      })),
    });
  }
}

function cellsTable(cells, clickable) {
  if (!cells.length) return "";
  return `<h3>Cells${clickable
    ? ' <span class="sub">— click a row for per-request transcripts</span>' : ""}</h3>
  <div class="card"><table>
    <tr><th>cell</th><th>status</th><th>agg tok/s</th><th>ttft p50</th><th>ttft p99</th>
        <th>compl p50</th><th>engine ITL p50</th><th>occup.</th><th>KV fill</th><th></th></tr>
    ${cells.map((c) => {
      const m = c.median || {};
      const warnings = (m.warnings || []).length;
      return `<tr${clickable ? ` class="cellrow" data-cell="${c.cell_id}"` : ""}>
        <td class="name">${c.workload} ${c.mode}@${c.level}
            <div class="sub">${c.cell_id}${c.reason ? ` — ${c.reason.split("\n")[0]}` : ""}</div></td>
        <td><span class="status ${c.status}">${c.status}</span></td>
        <td>${fmt.num(m.aggregate_tok_s)}</td>
        <td>${fmt.secs(m.ttft_p50)}</td><td>${fmt.secs(m.ttft_p99)}</td>
        <td>${fmt.secs(m.completion_p50)}</td>
        <td>${m.engine_itl_p50 == null ? "–" : (m.engine_itl_p50 * 1000).toFixed(1) + "ms"}</td>
        <td>${m.occupancy_mean == null ? "–" : (m.occupancy_mean * 100).toFixed(0) + "%"}</td>
        <td>${m.kv_fill_mean == null ? "–" : (m.kv_fill_mean * 100).toFixed(1) + "%"}</td>
        <td>${warnings ? `<span class="warnbadge" title="${(m.warnings || []).join("\n")}">${warnings}</span>` : ""}</td>
      </tr>`;
    }).join("")}
  </table></div>`;
}

/* ── Per-cell requests + transcript modal ───────────────────────── */

async function showCellRequests(runId, cellId) {
  const panel = $("#cell-detail");
  panel.innerHTML = `<h3>Requests — <span class="mono">${esc(cellId)}</span></h3>
    <div class="card sub">loading…</div>`;
  let data;
  try {
    data = await api(`/api/runs/${runId}/chats?cell=${encodeURIComponent(cellId)}`);
  } catch (e) {
    panel.innerHTML = `<p class="error">${esc(e.message)}</p>`;
    return;
  }
  const note = data.captured ? "" : `<p class="sub">Output text was not captured
    for this run, so transcripts show the prompt side only. Launch with
    “capture output text” checked to keep responses.</p>`;
  panel.innerHTML = `
    <h3>Requests — <span class="mono">${esc(cellId)}</span>
        <span class="sub">${esc(data.workload || "")} · ${data.rows.length} records
        incl. warmup · click a row for the transcript</span></h3>
    <div class="card">${note}<table>
      <tr><th>repeat</th><th>#</th><th>prompt</th><th>in tok</th><th>out tok</th>
          <th>ttft</th><th>compl</th><th>finish</th><th></th></tr>
      ${data.rows.map((r, i) => `<tr class="reqrow" data-i="${i}">
        <td>${r.excluded ? '<span class="warm">warm</span>' : r.repeat}</td>
        <td>${r.request_index}</td>
        <td class="name mono">${esc(r.prompt_id)}</td>
        <td>${r.input_tokens || "–"}</td><td>${r.output_tokens || "–"}</td>
        <td>${fmt.secs(r.ttft_s)}</td><td>${fmt.secs(r.completion_s)}</td>
        <td>${r.error ? `<span class="error">${esc(r.error)}</span>`
                      : (r.finish_reason || "–")}</td>
        <td class="sub">view ›</td>
      </tr>`).join("")}
    </table></div>`;
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  panel.querySelectorAll("tr.reqrow").forEach((tr) => {
    tr.onclick = () => openTranscript(data, Number(tr.dataset.i), cellId);
  });
}

function openTranscript(data, i, cellId) {
  const r = data.rows[i];
  const p = r.prompt;
  const bubble = (cls, who, body) =>
    `<div class="bubble ${cls}"><div class="who">${who}</div>${body}</div>`;
  const bubbles = [];
  if (p && p.system) bubbles.push(bubble("system", "system", esc(p.system)));
  (p ? p.messages : []).forEach((m) => {
    bubbles.push(bubble(m.role === "user" ? "user" : "assistant",
                        esc(m.role), esc(m.content)));
  });
  if (!p) {
    bubbles.push(`<p class="sub">Prompt <span class="mono">${esc(r.prompt_id)}</span>
      is not in the current workload file — it may have changed since this run.</p>`);
  }
  if (r.output_text != null) {
    bubbles.push(bubble("response", "response", esc(r.output_text)));
  } else {
    bubbles.push(bubble("response missing", "response",
      `<span class="sub">${data.captured
        ? "no text captured for this request"
        : "not captured — run was launched without “capture output text”"}</span>`));
  }

  const overlay = document.createElement("div");
  overlay.className = "overlay";
  overlay.innerHTML = `
    <div class="modal card">
      <div class="row spread">
        <div><b class="mono">${esc(r.prompt_id)}</b>
          <span class="sub">${esc(cellId)}
            · ${r.excluded ? "warmup" : `repeat ${r.repeat}`}
            · #${r.request_index} · ${esc(r.dialect || "")}</span></div>
        <button id="modal-close">Close</button>
      </div>
      <div class="sub" style="margin:6px 0 12px">
        ttft ${fmt.secs(r.ttft_s)} · completion ${fmt.secs(r.completion_s)}
        · ${r.input_tokens || "?"} in / ${r.output_tokens || "?"} out
        ${r.reasoning_tokens ? ` (${r.reasoning_tokens} thinking)` : ""}
        · finish: ${r.finish_reason || "–"}
        ${r.error ? ` · <span class="error">${esc(r.error)}</span>` : ""}
      </div>
      <div class="transcript">${bubbles.join("")}</div>
    </div>`;
  const onKey = (e) => { if (e.key === "Escape") close(); };
  const close = () => {
    overlay.remove();
    document.removeEventListener("keydown", onKey);
  };
  overlay.onclick = (e) => { if (e.target === overlay) close(); };
  overlay.querySelector("#modal-close").onclick = close;
  document.addEventListener("keydown", onKey);
  document.body.appendChild(overlay);
}

/* ── Compare ────────────────────────────────────────────────────── */

async function renderCompare() {
  const { history } = await api("/api/runs");
  const finished = history.filter((r) => r.status !== "running");
  const params = new URLSearchParams((location.hash.split("?")[1] || ""));
  const selA = params.get("a") || (finished[1] || {}).run_id || "";
  const selB = params.get("b") || (finished[0] || {}).run_id || "";
  const options = (sel) => finished.map((r) =>
    `<option value="${r.run_id}" ${r.run_id === sel ? "selected" : ""}>${r.run_id}</option>`).join("");

  view.innerHTML = `
    <h2>Compare</h2>
    <div class="card"><div class="row">
      A (baseline): <select id="run-a">${options(selA)}</select>
      B (candidate): <select id="run-b">${options(selB)}</select>
      <button class="primary" id="go">Compare</button>
    </div></div>
    <div id="cmp"></div>`;

  const go = async () => {
    const a = $("#run-a").value, b = $("#run-b").value;
    if (!a || !b) return;
    try {
      const cmp = await api(`/api/compare?runs=${a},${b}`);
      const rows = cmp.rows.filter((r) => r.b && r.a.aggregate_tok_s != null);
      $("#cmp").innerHTML = `
        <div class="legend">
          <span><span class="chip" style="background:var(--series-1)"></span>A ${cmp.run_a.run_id}</span>
          <span><span class="chip" style="background:var(--series-2)"></span>B ${cmp.run_b.run_id}</span>
        </div>
        <div class="card chart" id="cmp-tps"></div>
        <div class="card chart" id="cmp-ttft"></div>
        <h3>Deltas (B vs A)</h3>
        <div class="card"><table>
          <tr><th>cell</th><th>agg tok/s A</th><th>B</th><th>Δ</th>
              <th>ttft p50 A</th><th>B</th><th>Δ</th><th>engine ITL Δ</th></tr>
          ${cmp.rows.map((r) => {
            const d = r.deltas || {};
            const dTps = d.aggregate_tok_s, dTtft = d.ttft_p50, dItl = d.engine_itl_p50;
            return `<tr>
              <td class="name">${r.workload} ${r.mode}@${r.level}</td>
              <td>${fmt.num((r.a || {}).aggregate_tok_s)}</td>
              <td>${fmt.num((r.b || {}).aggregate_tok_s)}</td>
              <td class="${dTps && dTps.abs >= 0 ? "delta-up" : "delta-down"}">${dTps ? fmt.pct(dTps.pct) : "–"}</td>
              <td>${fmt.secs((r.a || {}).ttft_p50)}</td>
              <td>${r.b ? fmt.secs(r.b.ttft_p50) : "–"}</td>
              <td class="${dTtft && dTtft.abs <= 0 ? "delta-up" : "delta-down"}">${dTtft ? fmt.pct(dTtft.pct) : "–"}</td>
              <td class="${dItl && dItl.abs <= 0 ? "delta-up" : "delta-down"}">${dItl ? fmt.pct(dItl.pct) : "–"}</td>
            </tr>`;
          }).join("")}
        </table></div>`;
      if (rows.length) {
        barChart($("#cmp-tps"), {
          title: "Aggregate throughput (tok/s) — higher is better",
          unit: "tok/s",
          groups: rows.map((r) => ({
            label: `${r.workload} ${r.mode[0]}@${r.level}`,
            values: [
              { name: "A", value: r.a.aggregate_tok_s, color: "var(--series-1)" },
              { name: "B", value: r.b.aggregate_tok_s, color: "var(--series-2)" },
            ],
          })),
        });
        barChart($("#cmp-ttft"), {
          title: "TTFT p50 (s) — lower is better",
          unit: "s",
          groups: rows.map((r) => ({
            label: `${r.workload} ${r.mode[0]}@${r.level}`,
            values: [
              { name: "A", value: r.a.ttft_p50, color: "var(--series-1)" },
              { name: "B", value: r.b.ttft_p50, color: "var(--series-2)" },
            ],
          })),
        });
      }
    } catch (e) { $("#cmp").innerHTML = `<p class="error">${e.message}</p>`; }
  };
  $("#go").onclick = go;
  if (selA && selB) await go();
}

/* ── SVG grouped bar chart ──────────────────────────────────────── */

function barChart(container, { title, unit, groups }) {
  const W = 960, H = 240, padL = 48, padR = 12, padT = 28, padB = 40;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const values = groups.flatMap((g) => g.values.map((v) => v.value)).filter((v) => v != null);
  const max = Math.max(...values, 0) || 1;
  const ticks = niceTicks(max, 4);
  const top = ticks[ticks.length - 1];
  const y = (v) => padT + plotH - (v / top) * plotH;

  const groupW = plotW / groups.length;
  const nSeries = groups[0].values.length;
  const barW = Math.min(28, (groupW * 0.6) / nSeries);   // thin marks
  const gap = 2;                                          // 2px surface gap

  let bars = "";
  groups.forEach((g, gi) => {
    const cx = padL + gi * groupW + groupW / 2;
    const total = nSeries * barW + (nSeries - 1) * gap;
    g.values.forEach((v, si) => {
      if (v.value == null) return;
      const x = cx - total / 2 + si * (barW + gap);
      const yTop = y(v.value);
      const h = Math.max(padT + plotH - yTop, 1);
      const r = Math.min(4, barW / 2, h);                 // 4px rounded data end
      bars += `<path d="M${x},${padT + plotH}
        L${x},${yTop + r} Q${x},${yTop} ${x + r},${yTop}
        L${x + barW - r},${yTop} Q${x + barW},${yTop} ${x + barW},${yTop + r}
        L${x + barW},${padT + plotH} Z" fill="${v.color}"
        data-tip="<b>${g.label}</b><br>${v.name}: <b>${Number(v.value).toFixed(2)}</b> ${unit}"/>`;
    });
    const label = g.label.length > 14 ? g.label.slice(0, 13) + "…" : g.label;
    bars += `<text x="${cx}" y="${H - padB + 16}" text-anchor="middle">${label}</text>`;
  });

  const grid = ticks.map((t) =>
    `<line x1="${padL}" x2="${W - padR}" y1="${y(t)}" y2="${y(t)}"/>
     <text x="${padL - 8}" y="${y(t) + 4}" text-anchor="end">${t}</text>`).join("");

  container.innerHTML = `
    <div class="sub" style="margin-bottom:6px">${title}</div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}">
      <g class="grid">${grid}</g>
      <line class="baseline" x1="${padL}" x2="${W - padR}"
            y1="${padT + plotH}" y2="${padT + plotH}"/>
      ${bars}
    </svg>`;

  container.querySelectorAll("path[data-tip]").forEach((el) => {
    el.addEventListener("mousemove", (e) => {
      tooltip.innerHTML = el.dataset.tip;
      tooltip.style.left = `${e.clientX + 14}px`;
      tooltip.style.top = `${e.clientY + 14}px`;
      tooltip.classList.remove("hidden");
    });
    el.addEventListener("mouseleave", () => tooltip.classList.add("hidden"));
  });
}

function niceTicks(max, n) {
  const raw = max / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s * n >= max);
  return Array.from({ length: n + 1 }, (_, i) => Number((i * step).toPrecision(10)));
}

/* ── boot ───────────────────────────────────────────────────────── */

refreshPill();
setInterval(refreshPill, 3000);
navigate();
