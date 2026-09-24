"use strict";
// Civitas viewer: replays and live-follows runs written by `civitas run`.

const $ = (id) => document.getElementById(id);
const S = { runId: null, meta: null, frames: [], idx: 0, playing: false, timer: null, poll: null,
            finished: true, overlay: "none", tab: "agents", follow: true };

const ROLE_COLORS = () => {
  const cs = getComputedStyle(document.documentElement);
  return { city: cs.getPropertyValue("--s-city").trim(), mayor: cs.getPropertyValue("--s-mayor").trim(),
           planning: cs.getPropertyValue("--s-planning").trim(), utilities: cs.getPropertyValue("--s-utilities").trim() };
};
const dark = () => matchMedia("(prefers-color-scheme: dark)").matches;

// ---------------------------------------------------------------- palette
const TILE = {
  ".": ["#c9dcae", "#2f3a2a"], "~": ["#6fa3d6", "#27496b"], "#": ["#5b5f66", "#8a8f97"], "=": ["#9b8a6e", "#8c7a5c"],
  "H": ["#2b2f35", "#d9d9d9"], "T": ["#7a4fb3", "#b391e6"], "P": ["#3d8b40", "#3f8f47"],
  "E": ["#3a3a3a", "#bdbdbd"], "G": ["#d9822b", "#e39a4c"], "S": ["#e8c547", "#e8c547"], "W": ["#9fd8e0", "#7cc6d1"],
  "U": ["#1f8a8a", "#39b3b3"], "V": ["#146c7c", "#3aa0b3"],
};
// services, landmarks and other buildings added later: one colour each (fill, outline)
for (const [ch, c] of Object.entries({ Y: "#8d8a82", "%": "#6d5f3f", "*": "#4f8f6f", K: "#1f4fa8", J: "#7d8388", F: "#c62828",
  "^": "#b8b39c", A: "#e8e8e8", h: "#cfd8dc", ":": "#d32f2f", "?": "#8e24aa", ">": "#b0bec5", L: "#7a6a4f", X: "#5b5e62", Z: "#6f9f5a", N: "#f3d99b", B: "#b5654a", Q: "#efe9dc",
  D: "#2e5b3c", "|": "#6f8f4f", "@": "#b23b3b", "&": "#f1c40f", "+": "#d8d2c4", $: "#dfe3e8", "!": "#e0b43a", u: "#9fd3c7", z: "#6a6a6a" })) TILE[ch] = [c, c];
const ZONE = { R: [[120, 190, 110], [44, 150, 70]], C: [[120, 160, 230], [48, 108, 214]], I: [[232, 190, 110], [196, 132, 38]],
  O: [[190, 180, 235], [110, 100, 200]], M: [[220, 200, 176], [161, 136, 127]] };
const NAMES = { ".": "empty land", "~": "river", "#": "road", "=": "bridge", "H": "highway", "T": "town hall", "P": "park",
  "E": "coal plant", "G": "gas plant", "S": "solar farm", "W": "wind turbine", "U": "water pump", "V": "water tower",
  "Y": "hydro dam", "%": "sewage outlet", "*": "treatment plant", "K": "police", "J": "jail", "F": "fire station", "^": "shelter",
  "A": "clinic", "h": "hospital", ":": "metro station", "?": "casino", ">": "airport", "m": "mixed use (zoned)", "M": "mixed use", "L": "landfill", "X": "incinerator", "Z": "cemetery", "N": "elementary school", "B": "high school",
  "Q": "university", "D": "library", "|": "levee", "@": "train station", "&": "harbour", "+": "plaza", "$": "stadium", "!": "monument",
  "r": "residential (zoned)", "c": "commercial (zoned)", "i": "industrial (zoned)", "o": "offices (zoned)",
  "R": "residential", "C": "commercial", "I": "industrial", "O": "offices", "u": "nuclear plant", "z": "crematorium" };

function zoneColor(ch, level) {
  const up = ch.toUpperCase(); const [lo, hi] = ZONE[up];
  if (ch !== up || level === 0) return `rgba(${lo.join(",")},${dark() ? 0.35 : 0.45})`;
  const t = Math.min(1, (level - 1) / 4); const c = lo.map((v, i) => Math.round(v + (hi[i] - v) * (0.35 + 0.65 * t)));
  return `rgb(${c.join(",")})`;
}
function heat(v) { // 0..1 -> transparent-to-red ramp
  const a = 0.08 + 0.72 * v; return `rgba(${Math.round(200 + 55 * v)},${Math.round(160 - 140 * v)},40,${a})`;
}
function lvColor(v) { const a = 0.15 + 0.6 * v; return `rgba(40,${Math.round(90 + 110 * v)},${Math.round(200 - 80 * v)},${a})`; }

// ------------------------------------------------------------------- api
async function api(path) { const r = await fetch(path, { cache: "no-store" }); if (!r.ok) throw new Error(path); return r.json(); }

async function loadRuns() {
  const runs = await api("/api/runs");
  const sel = $("runSelect"); sel.innerHTML = "";
  if (!runs.length) { sel.innerHTML = "<option>No runs yet: civitas run -c configs/baseline.toml</option>"; return; }
  for (const r of runs) {
    const o = document.createElement("option"); o.value = r.run_id;
    const score = r.scores ? ` · city ${r.scores.city.toFixed(1)}` : " · running…";
    o.textContent = `${r.run_id}${score}`; sel.appendChild(o);
  }
  const want = new URLSearchParams(location.search).get("run");
  sel.value = want && runs.some(r => r.run_id === want) ? want : runs[0].run_id;
  await openRun(sel.value);
}

async function openRun(id) {
  stop(); clearInterval(S.poll);
  S.runId = id; S.frames = []; S.idx = 0; S.follow = true;
  S.meta = await api(`/api/runs/${id}/meta`);
  history.replaceState(null, "", `?run=${encodeURIComponent(id)}`);
  await fetchFrames();
  S.idx = S.finished ? 0 : S.frames.length - 1;
  if (!S.finished) S.poll = setInterval(fetchFrames, 1500);
  render();
}

async function fetchFrames() {
  const d = await api(`/api/runs/${S.runId}/frames?since=${S.frames.length}`);
  const atEnd = S.idx >= S.frames.length - 1;
  for (const f of d.frames) {
    if (f.cities) Object.assign(f, f.cities.north);            // region runs: the 2D view shows the north city
    if (!(S.meta.frame_format >= 3)) f.tiles = f.tiles.replaceAll("M", "h");   // old runs: hospital was 'M'
    if (!(S.meta.frame_format >= 2)) f.tiles = f.tiles.replaceAll("O", "V");   // old runs: water tower was 'O'
  }
  S.frames.push(...d.frames); S.finished = d.finished;
  $("liveBadge").hidden = d.finished;
  if (d.finished) clearInterval(S.poll);
  $("monthSlider").max = Math.max(0, S.frames.length - 1);
  if (d.frames.length && !S.finished && atEnd && S.follow) { S.idx = S.frames.length - 1; render(); }
  else if (d.frames.length) renderCharts();
}

// -------------------------------------------------------------- playback
function play() { if (S.idx >= S.frames.length - 1) S.idx = 0; S.playing = true; $("playBtn").textContent = "❚❚"; tick(); }
function stop() { S.playing = false; clearTimeout(S.timer); $("playBtn").textContent = "▶"; }
function tick() {
  if (!S.playing) return;
  if (S.idx < S.frames.length - 1) { S.idx++; render(); }
  else if (S.finished) { stop(); return; }
  S.timer = setTimeout(tick, +$("speedSelect").value);
}

// ---------------------------------------------------------------- render
function render() {
  if (!S.frames.length) return;
  S.idx = Math.min(S.idx, S.frames.length - 1);
  $("monthSlider").value = S.idx;
  const f = S.frames[S.idx];
  $("monthLabel").textContent = `Y${Math.floor(f.month / 12) + 1} M${(f.month % 12) + 1} · ${f.month}/${S.meta.months}`;
  drawMap(f); renderScores(f); renderKpis(f); renderCharts(); renderTab(f); renderLegend();
}

function drawMap(f) {
  const W = S.meta.width, H = S.meta.height, cv = $("map");
  const px = Math.max(6, Math.floor(Math.min(cv.parentElement.clientWidth, 900) / W));
  const dpr = window.devicePixelRatio || 1;
  cv.width = W * px * dpr; cv.height = H * px * dpr; cv.style.width = `${W * px}px`;
  const g = cv.getContext("2d"); g.scale(dpr, dpr);
  const di = dark() ? 1 : 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, ch = f.tiles[i], lvl = +f.levels[i];
    g.fillStyle = TILE["."][di]; g.fillRect(x * px, y * px, px, px);
    if ("rciomRCIOM".includes(ch)) {
      g.fillStyle = zoneColor(ch, lvl); g.fillRect(x * px + 0.5, y * px + 0.5, px - 1, px - 1);
      if (lvl > 0 && px >= 8) { g.fillStyle = "rgba(255,255,255,.55)"; for (let k = 0; k < lvl; k++) g.fillRect(x * px + 2 + k * 3, y * px + px - 4, 2, 2); }
    } else if (ch !== ".") {
      g.fillStyle = (TILE[ch] || ["#f0f", "#f0f"])[di];
      const inset = "EGSWUOTP".includes(ch) ? 1 : 0;
      g.fillRect(x * px + inset, y * px + inset, px - 2 * inset, px - 2 * inset);
    }
    if (S.overlay !== "none" && ch !== "~") {
      if (S.overlay === "pollution") { const v = +f.pollution[i] / 9; if (v > 0.05) { g.fillStyle = heat(v); g.fillRect(x * px, y * px, px, px); } }
      else if (S.overlay === "land_value") { g.fillStyle = lvColor(+f.land_value[i] / 9); g.fillRect(x * px, y * px, px, px); }
      else if ("rciomRCIOM".includes(ch)) {
        const on = f[S.overlay][i] === "1";
        g.fillStyle = on ? "rgba(46,160,90,.55)" : "rgba(220,60,40,.7)"; g.fillRect(x * px + 1, y * px + 1, px - 2, px - 2);
      }
    }
  }
  cv.dataset.px = px;
}

function renderLegend() {
  const L = $("legend"), di = dark() ? 1 : 0;
  if (S.overlay === "pollution") L.innerHTML = `Pollution <span class="ramp" style="background:linear-gradient(90deg,${heat(0)},${heat(1)})"></span> high`;
  else if (S.overlay === "land_value") L.innerHTML = `Land value low <span class="ramp" style="background:linear-gradient(90deg,${lvColor(0)},${lvColor(1)})"></span> high`;
  else if (S.overlay === "power" || S.overlay === "water") L.innerHTML = `<span><i style="background:rgba(46,160,90,.8)"></i>zone has ${S.overlay}</span><span><i style="background:rgba(220,60,40,.8)"></i>no ${S.overlay}</span>`;
  else {
    const items = [["R", zoneColor("R", 3)], ["C", zoneColor("C", 3)], ["I", zoneColor("I", 3)], ["zoned, empty", zoneColor("r", 0)],
      ["road", TILE["#"][di]], ["river", TILE["~"][di]], ["park", TILE.P[di]], ["town hall", TILE.T[di]], ["coal", TILE.E[di]],
      ["gas", TILE.G[di]], ["solar", TILE.S[di]], ["wind", TILE.W[di]], ["pump", TILE.U[di]], ["tower", TILE.V[di]]];
    L.innerHTML = items.map(([n, c]) => `<span><i style="background:${c}"></i>${n}</span>`).join("") + `<span class="muted">dots = building level</span>`;
  }
}

function fmtMoney(v) { const a = Math.abs(v); const s = a >= 1e6 ? (a / 1e6).toFixed(2) + "M" : a >= 1e4 ? (a / 1e3).toFixed(1) + "k" : Math.round(a).toLocaleString(); return (v < 0 ? "−$" : "$") + s; }
const pct = (v) => v == null ? "—" : `${(100 * v).toFixed(0)}%`;

function renderScores(f) {
  const C = ROLE_COLORS(), prev = S.frames[Math.max(0, S.idx - 12)];
  const keys = ["city", ...S.meta.roles];
  $("scoreCards").innerHTML = keys.map(k => {
    const v = f.scores[k] ?? 0, d = v - (prev.scores[k] ?? 0);
    const agent = k === "city" ? "overall" : (S.meta.agents[k] || "");
    const dcls = Math.abs(d) < 0.05 ? "" : d > 0 ? "up" : "down";
    return `<div class="score" style="--c:${C[k]}"><div class="name"><span>${k}</span>
      <span class="delta ${dcls}">${Math.abs(d) < 0.05 ? "" : (d > 0 ? "+" : "") + d.toFixed(1)}</span></div>
      <div class="val">${v.toFixed(1)}</div><div class="agent" title="${agent}">${agent}</div></div>`;
  }).join("");
}

function renderKpis(f) {
  const m = f.metrics;
  const k = [
    ["Population", m.population.toLocaleString()], ["Jobs C / I", `${m.jobs_com} <small>/ ${m.jobs_ind}</small>`],
    ["Unemployment", pct(m.unemployment)], ["Happiness", pct(m.happiness)],
    ["Treasury", fmtMoney(m.treasury)], ["Dept funds", `${fmtMoney(m.dept_funds.planning)} <small>P</small> ${fmtMoney(m.dept_funds.utilities)} <small>U</small>`],
    ["Net / month", fmtMoney(m.income - m.expenses)], ["Debt", fmtMoney(m.debt)],
    ["Taxes R/C/I", `${(m.taxes.R * 100).toFixed(0)} / ${(m.taxes.C * 100).toFixed(0)} / ${(m.taxes.I * 100).toFixed(0)}%`],
    ["Demand R/C/I", `${m.demand.R.toFixed(2)} / ${m.demand.C.toFixed(2)} / ${m.demand.I.toFixed(2)}`],
    ["Power", `${Math.round(m.power_supply)} <small>/ ${Math.round(m.power_demand)} · ${pct(m.power_coverage)}</small>`],
    ["Water", `${Math.round(m.water_supply)} <small>/ ${Math.round(m.water_demand)} · ${pct(m.water_coverage)}</small>`],
    ["Pollution at homes", pct(m.pollution_res)], ["Land value", pct(m.land_value_dev)],
    ["Developed / zoned", `${m.developed_total} <small>/ ${m.zoned_total}</small>`], ["Roads · parks", `${m.road_tiles} <small>· ${m.parks}</small>`],
  ];
  $("kpis").innerHTML = k.map(([a, b]) => `<div class="kpi"><div class="k">${a}</div><div class="v">${b}</div></div>`).join("");
}

// ---------------------------------------------------------------- charts
function lineChart(svg, series, { yFmt = (v) => v, yMin = null, yMax = null } = {}) {
  const w = svg.clientWidth || 300, h = svg.clientHeight || 120, pl = 40, pr = 6, pt = 6, pb = 16;
  const n = S.meta.months + 1;
  const all = series.flatMap(s => s.values.filter(v => v != null));
  let lo = yMin ?? Math.min(0, ...all), hi = yMax ?? Math.max(1, ...all);
  if (hi === lo) hi = lo + 1;
  const X = (i) => pl + (w - pl - pr) * i / Math.max(1, n - 1), Y = (v) => pt + (h - pt - pb) * (1 - (v - lo) / (hi - lo));
  let out = `<line class="axis" x1="${pl}" y1="${Y(lo)}" x2="${w - pr}" y2="${Y(lo)}"/>`;
  for (const t of [lo, (lo + hi) / 2, hi]) out += `<text x="${pl - 4}" y="${Y(t) + 3}" text-anchor="end">${yFmt(t)}</text><line class="axis" x1="${pl}" x2="${w - pr}" y1="${Y(t)}" y2="${Y(t)}" opacity=".5"/>`;
  for (let yr = 0; yr * 12 < n; yr += Math.ceil(n / 12 / 5)) out += `<text x="${X(yr * 12)}" y="${h - 3}" text-anchor="middle">y${yr}</text>`;
  for (const s of series) {
    const pts = s.values.map((v, i) => v == null ? null : `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).filter(Boolean).join(" ");
    out += `<polyline fill="none" stroke="${s.color}" stroke-width="${s.width || 1.8}" stroke-linejoin="round" points="${pts}"/>`;
  }
  const cx = X(S.frames[S.idx]?.month ?? 0);
  out += `<line class="cursor" x1="${cx}" x2="${cx}" y1="${pt}" y2="${h - pb}"/>`;
  if (series.length > 1) out += series.map((s, i) => `<text x="${pl + 6 + i * 70}" y="${pt + 8}" style="fill:${s.color}">${s.name}</text>`).join("");
  svg.innerHTML = out;
}

function renderCharts() {
  if (!S.frames.length) return;
  const C = ROLE_COLORS(), fr = S.frames, M = (k) => fr.map(f => f.metrics[k]);
  const acc = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  const kfmt = (v) => Math.abs(v) >= 1000 ? (v / 1000).toFixed(0) + "k" : Math.round(v);
  lineChart($("chPop"), [{ name: "population", values: M("population"), color: acc }], { yFmt: kfmt });
  lineChart($("chMoney"), [{ name: "treasury", values: M("treasury"), color: acc }], { yFmt: kfmt });
  lineChart($("chScores"), ["city", ...S.meta.roles].map(k => ({ name: k, values: fr.map(f => f.scores[k] ?? 0), color: C[k], width: k === "city" ? 2.6 : 1.5 })), { yMin: 0, yMax: 100 });
  lineChart($("chHappy"), [
    { name: "happy", values: M("happiness"), color: C.mayor },
    { name: "power", values: M("power_coverage"), color: C.utilities },
    { name: "water", values: fr.map(f => f.metrics.water_coverage), color: "#3b8fc4" }], { yMin: 0, yMax: 1, yFmt: (v) => `${Math.round(v * 100)}%` });
}

// ------------------------------------------------------------------ tabs
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function actText(a) { const { type, ...rest } = a; return `${type} ${Object.entries(rest).map(([k, v]) => `${k}=${v}`).join(" ")}`; }

function renderTab(f) {
  const C = ROLE_COLORS(), body = $("tabBody");
  if (S.tab === "agents") {
    $("tabHint").textContent = f.month ? `decisions made during month ${f.month - 1}` : "start of game";
    if (!f.turns.length) { body.innerHTML = `<div class="empty">No agent turns in this frame.</div>`; return; }
    body.innerHTML = f.turns.map(t => `<div class="turn" style="--c:${C[t.role]}">
      <div class="who"><span>${t.role} <span class="pill">${esc(S.meta.agents[t.role])}</span></span>
        <span class="small muted">${t.invalid ? `${t.invalid} failed` : ""}${t.parse_error ? ` · ${esc(t.parse_error).slice(0, 60)}` : ""}</span></div>
      ${t.thinking ? `<div class="think">${esc(t.thinking)}</div>` : ""}
      ${t.results.map(r => `<div class="act ${r.ok ? "ok" : "bad"}"><span class="st">${r.ok ? "✓" : "✗"}</span>
        <span>${esc(actText(r.action))}<br><span class="msg">${esc(r.message)}</span></span></div>`).join("") || `<div class="muted small">no actions</div>`}
      ${t.messages.map(m => `<div class="msgline small"><b style="--c:${C[t.role]}">→ ${esc(m.to)}</b> ${esc(m.text)}</div>`).join("")}
    </div>`).join("");
  } else if (S.tab === "council") {
    const msgs = S.frames.slice(0, S.idx + 1).flatMap(fr => fr.council || []).slice(-60).reverse();
    $("tabHint").textContent = `${msgs.length} messages up to now`;
    body.innerHTML = msgs.length ? msgs.map(m => `<div class="msgline" style="--c:${C[m.sender]}"><span class="pill">m${m.month}</span>
      <b>${esc(m.sender)}</b> → ${esc(m.to)}: ${esc(m.text)}</div>`).join("") : `<div class="empty">No council messages yet.</div>`;
  } else if (S.tab === "events") {
    const ev = S.frames.slice(0, S.idx + 1).flatMap(fr => (fr.events || []).map(e => [fr.month, e])).slice(-80).reverse();
    $("tabHint").textContent = "";
    body.innerHTML = ev.length ? ev.map(([m, e]) => `<div class="msgline"><span class="pill">m${m}</span> ${esc(e)}</div>`).join("") : `<div class="empty">No events.</div>`;
  } else if (S.tab === "board") {
    $("tabHint").textContent = "all finished runs, grouped by agent line-up";
    api("/api/leaderboard").then(rows => {
      if (!rows.length) { body.innerHTML = `<div class="empty">No finished runs.</div>`; return; }
      const roles = Object.keys(rows[0].roles);
      body.innerHTML = `<table class="lb"><tr><th>agents</th><th>n</th><th>city</th><th>pop</th>${roles.map(r => `<th>${r}</th>`).join("")}<th>bankrupt</th></tr>
        ${rows.map(r => `<tr><td>${esc(r.signature)}</td><td>${r.n}</td><td><b>${r.city.mean.toFixed(1)}</b> ±${r.city.std.toFixed(1)}</td>
        <td>${Math.round(r.population.mean).toLocaleString()}</td>${roles.map(k => `<td>${(r.roles[k]?.mean ?? 0).toFixed(1)}</td>`).join("")}<td>${r.bankruptcies}</td></tr>`).join("")}</table>`;
    });
  }
}

// ---------------------------------------------------------------- events
$("runSelect").onchange = (e) => openRun(e.target.value);
$("playBtn").onclick = () => (S.playing ? stop() : play());
$("monthSlider").oninput = (e) => { stop(); S.idx = +e.target.value; S.follow = S.idx === S.frames.length - 1; render(); };
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT" && e.target.type !== "range") return;
  if (e.code === "Space") { e.preventDefault(); S.playing ? stop() : play(); }
  if (e.code === "ArrowRight") { stop(); S.idx = Math.min(S.frames.length - 1, S.idx + 1); render(); }
  if (e.code === "ArrowLeft") { stop(); S.idx = Math.max(0, S.idx - 1); render(); }
});
for (const b of $("overlaySeg").querySelectorAll("button")) b.onclick = () => {
  $("overlaySeg").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); S.overlay = b.dataset.ov; render();
};
for (const b of $("tabSeg").querySelectorAll("button")) b.onclick = () => {
  $("tabSeg").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); S.tab = b.dataset.tab; render();
};
$("map").addEventListener("mousemove", (e) => {
  const f = S.frames[S.idx]; if (!f) return;
  const px = +$("map").dataset.px, r = $("map").getBoundingClientRect();
  const x = Math.floor((e.clientX - r.left) / px), y = Math.floor((e.clientY - r.top) / px);
  if (x < 0 || y < 0 || x >= S.meta.width || y >= S.meta.height) { $("tip").hidden = true; return; }
  const i = y * S.meta.width + x, ch = f.tiles[i], zone = "rciomRCIOM".includes(ch);
  $("tip").innerHTML = `<b>(${x}, ${y})</b> ${NAMES[ch] || ch}${zone ? ` · level ${f.levels[i]}` : +f.levels[i] > 0 ? ` · tier ${f.levels[i]}` : ""}<br>
    pollution ${f.pollution[i]}/9 · land value ${f.land_value[i]}/9${zone ? `<br>power ${f.power[i] === "1" ? "yes" : "no"} · water ${f.water[i] === "1" ? "yes" : "no"}` : ""}`;
  const tip = $("tip"); tip.hidden = false;
  const lx = e.clientX - r.left + 14, ly = e.clientY - r.top + 14;
  tip.style.left = `${Math.min(lx, r.width - tip.offsetWidth - 4)}px`; tip.style.top = `${ly}px`;
});
$("map").addEventListener("mouseleave", () => ($("tip").hidden = true));
window.addEventListener("resize", () => render());
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => render());

loadRuns().catch(err => { $("tabBody").innerHTML = `<div class="empty">Could not reach the API: ${esc(err.message)}</div>`; });
