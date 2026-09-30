// Civitas control center: replays & live-follows runs, launches new ones. The city is drawn by the pixel view
// (pixel.js, OpenTTD-style) or the 3D view (scene.js); both have the same interface and `scene` is the active one.
import { CityScene, OVERLAY_RAMPS, OVERLAYS, OVERLAY_BY_ID, DISTRICT_COLORS } from "./scene.js";
import { PixelScene } from "./pixel.js";
import { RegionMap, regionLegend, SPEC_COLOR } from "./region.js";
import { CollabGraph, BeamLayer } from "./collab.js";

const $ = (id) => document.getElementById(id);
const app = $("app");
const ROLE_C = { city: "#ffffff", mayor: "#b99bff", planning: "#5fd38d", utilities: "#ffb454", transport: "#e07bd8",
  police: "#5b8cff", fire: "#ff7a59", health: "#3fd0c9", education: "#f5d547" };
const roleColor = (r) => ROLE_C[r] || ROLE_C[String(r).split(/[./]/).pop()] || "#cccccc";
const NAMES = { ".": "empty land", "~": "river", "#": "road", "=": "bridge", H: "highway", T: "town hall", P: "park",
  E: "coal plant", G: "gas plant", S: "solar farm", W: "wind turbine", U: "water pump", V: "water tower", Y: "hydro dam",
  "%": "sewage outlet", "*": "sewage treatment plant", K: "police station", J: "jail", F: "fire station", "^": "emergency shelter",
  A: "clinic", h: "hospital", L: "landfill", ":": "metro station", "?": "casino", ">": "airport", m: "mixed-use zone (empty)", M: "mixed use", X: "incinerator", Z: "cemetery", N: "elementary school", B: "high school",
  Q: "university", D: "library", "|": "levee", "@": "train station", "&": "harbour", "+": "plaza", $: "stadium", "!": "monument",
  r: "residential zone (empty)", c: "commercial zone (empty)", i: "industrial zone (empty)", o: "office zone (empty)",
  R: "residential", C: "commercial", I: "industrial", O: "offices", u: "nuclear plant", z: "crematorium" };
// facilities keep their tile character as they move up a tier; the frame level is the tier (0 in old replays = base)
// keep in step with civitas/sim/catalog.py TIER_LINES (the source of truth for tiers and their names)
const TIER_NAMES = { E: { 2: "advanced coal plant" }, G: { 2: "combined-cycle gas plant" }, S: { 2: "solar power plant" },
  W: { 2: "advanced wind turbine" }, U: { 2: "pumping station", 3: "water works" }, V: { 2: "inland water plant" },
  "*": { 3: "advanced treatment plant" }, L: { 2: "recycling centre", 3: "waste processing complex" },
  h: { 3: "medical centre" }, K: { 2: "police HQ" }, J: { 2: "prison" }, F: { 2: "fire HQ" },
  N: { 2: "large elementary school" }, B: { 2: "large high school" } };
const nameOf = (ch, lvl) => (TIER_NAMES[ch] && TIER_NAMES[ch][lvl]) || NAMES[ch] || ch;
const LEVEL_NAME = { R: ["", "houses", "apartments", "high-rise", "tower", "luxury tower"], C: ["", "shops", "offices", "skyscraper", "mall tower", "megatower"],
  I: ["", "workshop", "factory", "heavy industry", "plant", "tech campus"], O: ["", "small offices", "office block", "office tower", "HQ tower", "landmark HQ"],
  M: ["", "corner shop & flat", "shop-house", "mid-rise", "high-rise", "tower"] };
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const store = { get(k, d) { try { const v = localStorage.getItem("civitas." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("civitas." + k, JSON.stringify(v)); } catch { /* ignore */ } } };

const S = {
  runs: [], runId: null, meta: null, frames: [], idx: 0, status: "—", playing: false, timer: null, poll: null,
  tab: store.get("tab", "agents"), overlay: store.get("overlay", "none"), roleFilter: new Set(), selected: -1,
  follow: true, configs: [], issuesView: store.get("issues", "agents"),
};
if (S.tab === "problems" || S.tab === "events") { S.issuesView = S.tab === "events" ? "city" : "agents"; S.tab = "issues"; }

// ================================================================ api
async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : { cache: "no-store" });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `${r.status} ${path}`);
  return j;
}
function toast(msg, err = false) {
  const t = document.createElement("div"); t.className = "toast" + (err ? " err" : ""); t.innerHTML = msg;
  $("toasts").appendChild(t); setTimeout(() => t.remove(), err ? 7000 : 3500);
}

// ================================================================ scene
const sceneOpts = { onHover: (tile, ev) => showTooltip(tile, ev), onClick: (tile) => select(tile) };
const scenes = {};
const makeScene = (kind) => scenes[kind] || (scenes[kind] = kind === "3d" ? new CityScene($("viewport"), sceneOpts) : new PixelScene($("viewport"), sceneOpts));
let viewMode = store.get("view", "pixel") === "3d" ? "3d" : "pixel";
let scene = makeScene(viewMode);
function setView(kind) {
  if (kind === viewMode) return;
  scene.setActive(false);
  viewMode = kind; store.set("view", kind);
  scene = makeScene(kind); scene.setActive(true);
  if (S.meta) { scene.setMeta(S.meta); if (S.frames.length) render(null, false); }
  scene.setOverlay(S.overlay); scene.setNight(store.get("night", false)); scene.setQuality(store.get("quality", "high"));
  scene.setSelected(S.selected);
  $("vMode").textContent = kind === "3d" ? "3D" : "PX";
  $("vMode").title = kind === "3d" ? "3D view: switch to pixel view (X)" : "Pixel view: switch to 3D view (X)";
  $("vTop").title = kind === "3d" ? "Top-down view (T)" : "Zoom out to the whole region (T)";
  $("vRotate").title = kind === "3d" ? "Auto-rotate (O)" : "Auto-rotate in quarter turns (O)";
}

// ================================================================ runs
async function refreshRuns() {
  try { S.runs = await api("/api/runs"); } catch (e) { toast("Cannot reach the Civitas server: " + esc(e.message), true); return; }
  renderRunMenu();
  $("empty").hidden = S.runs.length > 0;
}
function statusPill(st) { return `<span class="pill ${st}">${st}</span>`; }
function renderRunMenu() {
  $("runMenu").innerHTML = S.runs.length ? S.runs.map(r => `
    <div class="item ${r.run_id === S.runId ? "cur" : ""}" data-run="${esc(r.run_id)}">
      <span class="ellipsis"><b>${esc(r.name)}</b> · seed ${r.seed} <span class="muted small">${esc(r.run_id.slice(0, 15))}</span></span>
      <span>${r.scores ? `<b>${r.scores.city.toFixed(1)}</b>` : ""} ${statusPill(r.status)}</span>
      <span class="sig">${esc(r.signature || "")}</span></div>`).join("") : `<div class="emptytab">No runs yet</div>`;
  $("runMenu").querySelectorAll(".item").forEach(el => el.onclick = () => { $("runMenu").hidden = true; openRun(el.dataset.run); });
}

// ---------------------------------------------------------------- region mode (two cities)
function cityMeta(meta, c) {
  if (!meta.region) return meta;
  const base = [...new Set(meta.roles.map(r => r.split(".").pop()))];
  const agents = Object.fromEntries(base.map(r => [r, meta.agents[`${c}.${r}`] || meta.agents[r] || "?"]));
  return { ...meta, roles: base, agents, resources: (meta.city_meta[c] || {}).resources, rawRoles: meta.roles };
}
function cityFrame(f, c) {
  if (!f.cities) return f;
  const cf = f.cities[c], pre = c + ".";
  const local = (m) => ({ ...m, sender: m.sender.startsWith(pre) ? m.sender.slice(pre.length) : m.sender,
    to: m.to.startsWith(pre) ? m.to.slice(pre.length) : m.to });
  return { ...cf, month: f.month, council: (f.council || []).map(local), deals: f.deals, regionScores: f.scores };
}
function renderCitySwitch() {
  const el = $("citySwitch");
  $("regionTab").hidden = $("regionBtn").hidden = !(S.metaRaw && S.metaRaw.region);
  if ($("regionBtn").hidden) $("regionModal").hidden = true;
  if ($("regionTab").hidden && S.tab === "region") S.tab = "agents";
  if (!S.metaRaw || !S.metaRaw.region) { el.hidden = true; return; }
  el.hidden = false;
  el.innerHTML = S.metaRaw.cities.map(c => `<button data-city="${c}" class="${c === S.city ? "on" : ""}">${c}</button>`).join("");
  el.querySelectorAll("button").forEach(b => b.onclick = () => switchCity(b.dataset.city));
}
function switchCity(c) {
  if (c === S.city) return;
  S.city = c; S.meta = cityMeta(S.metaRaw, c); scene.setMeta(S.meta);
  S.frames = S.raw.map(f => cityFrame(f, c)); S.traces = {};
  S.collab = null; S.teamPick = null; loadCollab(true);
  renderCitySwitch();
  $("agentsLine").innerHTML = `<b>${c}</b> · ` + S.meta.roles.map(r => `<span style="color:${roleColor(r)}">${r}</span> ${esc(S.meta.agents[r])}`).join(" · ");
  render(null, false);
}

async function openRun(id, { month = null } = {}) {
  stop(); clearInterval(S.poll);
  S.runId = id; S.frames = []; S.raw = []; S.idx = 0; S.selected = -1; $("inspector").hidden = true;
  S.collab = null; S.collabErr = null; S.teamPick = null; collabAt = 0;
  try { S.metaRaw = await api(`/api/runs/${encodeURIComponent(id)}/meta`); } catch (e) { toast("Could not open run: " + esc(e.message), true); return; }
  S.city = S.metaRaw.region ? S.metaRaw.cities[0] : null;
  S.meta = cityMeta(S.metaRaw, S.city);
  renderCitySwitch();
  scene.setMeta(S.meta);
  $("runLabel").textContent = `${S.meta.name} · seed ${S.meta.seed}`;
  $("agentsLine").innerHTML = (S.city ? `<b>${S.city}</b> · ` : "") + S.meta.roles.map(r => `<span style="color:${roleColor(r)}">${r}</span> ${esc(S.meta.agents[r])}`).join(" · ");
  await fetchFrames(true);
  S.idx = month != null ? Math.min(month, S.frames.length - 1) : (S.status === "running" ? S.frames.length - 1 : S.frames.length - 1);
  S.poll = setInterval(() => fetchFrames(false), 2000);
  store.set("lastRun", id);
  history.replaceState(null, "", `?run=${encodeURIComponent(id)}`);
  renderRunMenu();
  render(null, false);
  if (S.status !== "running" && S.frames.length > 1 && month == null) {
    toast(`Loaded <b>${esc(S.meta.name)}</b> (${S.frames.length - 1} months). Press <kbd>Space</kbd> to replay from the start or scrub the timeline.`);
  }
}

async function fetchFrames(first) {
  if (!S.runId) return;
  let d;
  try { d = await api(`/api/runs/${encodeURIComponent(S.runId)}/frames?since=${S.frames.length}`); } catch { return; }
  const wasEnd = S.idx >= S.frames.length - 1;
  const legacy = (t) => { if (!(S.meta.frame_format >= 3)) t = t.replaceAll("M", "h"); if (!(S.meta.frame_format >= 2)) t = t.replaceAll("O", "V"); return t; };
  if (!(S.meta.frame_format >= 3)) for (const f of d.frames) {
    if (f.tiles) f.tiles = legacy(f.tiles);
    for (const cf of Object.values(f.cities || {})) cf.tiles = legacy(cf.tiles);
  }
  S.raw.push(...d.frames);
  S.frames.push(...d.frames.map(f => cityFrame(f, S.city)));
  const prevStatus = S.status; S.status = d.status;
  $("statusPill").className = "pill " + d.status; $("statusPill").textContent = d.status;
  $("slider").max = Math.max(0, S.frames.length - 1);
  if (d.frames.length) loadProblems();          // the timeline's problem marks; refetched as a live run grows
  if (d.frames.length) loadCollab();
  if (d.status !== "running") clearInterval(S.poll);
  if (prevStatus === "running" && d.status !== "running" && !first) { toast(`Run ${esc(d.status)}.`); refreshRuns(); }
  if (!first && d.frames.length) {
    drawTrackMarks();
    if (S.follow && wasEnd && !S.playing) { const p = S.idx; S.idx = S.frames.length - 1; render(S.frames[p], S.idx === p + 1); }
    else renderCharts();
  }
}

// ================================================================ playback
function play() { if (S.idx >= S.frames.length - 1) S.idx = 0; S.playing = true; $("tPlay").textContent = "❚❚"; render(null, false); tick(); }
function stop() { S.playing = false; clearTimeout(S.timer); $("tPlay").textContent = "▶"; }
function tick() {
  if (!S.playing) return;
  if (S.idx < S.frames.length - 1) { const p = S.frames[S.idx]; S.idx++; render(p, true); }
  else if (S.status !== "running") { stop(); return; }
  S.timer = setTimeout(tick, +$("speed").value);
}
function go(i, animate = false) {
  if (!S.frames.length) return;
  const p = S.frames[S.idx]; stop();
  S.idx = Math.max(0, Math.min(S.frames.length - 1, i));
  S.follow = S.idx === S.frames.length - 1 && $("follow").checked;
  render(p, animate && S.idx === S.frames.indexOf(p) + 1);
}

// ================================================================ render
function render(prev, animate) {
  if (!S.frames.length) return;
  const f = S.frames[S.idx];
  S.anim = !!animate;
  if (scene.setRegion) scene.setRegion(S.metaRaw && S.metaRaw.region ? { meta: S.metaRaw, frame: S.raw[S.idx], city: S.city } : null);
  scene.setFrame(f, prev, { animate });
  $("slider").value = S.idx;
  const arrow = f.wind ? "→↘↓↙←↖↑↗"[((Math.round(Math.atan2(f.wind[1], f.wind[0]) / (Math.PI / 4)) % 8) + 8) % 8] : "";
  $("monthLabel").textContent = `Year ${Math.floor(f.month / 12) + 1} · M${(f.month % 12) + 1}` + (f.season ? ` · ${f.season}` : "") + (arrow ? ` ${arrow}` : "");
  $("monthSub").textContent = `month ${f.month} / ${S.meta.months}`;
  renderScores(f); renderKpis(f); renderDemand(f); renderUtil(f); renderServices(f); renderMoney(f);
  renderTab(); drawMinimap(); drawTrackMarks(); updateRegionMap(); updateTeamView(); updateBeams();
  if (S.selected >= 0) renderInspector(S.selected);
}

const fmt$ = (v) => { const a = Math.abs(v); const s = a >= 1e6 ? (a / 1e6).toFixed(2) + "M" : a >= 1e4 ? (a / 1e3).toFixed(1) + "k" : Math.round(a).toLocaleString(); return (v < 0 ? "−$" : "$") + s; };
const pct = (v) => v == null ? "—" : `${Math.round(100 * v)}%`;

function spark(values, color, w = 70, h = 22) {
  const v = values.filter(x => x != null); if (v.length < 2) return "";
  const lo = Math.min(...v), hi = Math.max(...v, lo + 1e-6);
  const pts = values.map((x, i) => `${(i / (values.length - 1) * w).toFixed(1)},${(h - 2 - (x - lo) / (hi - lo) * (h - 4)).toFixed(1)}`).join(" ");
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline fill="none" stroke="${color}" stroke-width="1.5" points="${pts}"/></svg>`;
}

// ---------------------------------------------------------------- left dock
// read top to bottom: the city score, what needs attention now, four vitals, the departments, then folded detail
const YEAR = 12;
const ago = () => S.frames[Math.max(0, S.idx - YEAR)];
function trend(v, p, { digits = 1, pct: asPct = false, zero = "" } = {}) {
  const d = v - p, show = asPct ? Math.round(d * 100) : +d.toFixed(digits);
  if (!show) return zero;
  return `<span class="${d > 0 ? "up" : "down"}">${d > 0 ? "▲" : "▼"}${Math.abs(show).toLocaleString()}${asPct ? " pts" : ""}</span>`;
}
function renderScores(f) { renderHero(f); renderAlerts(f); renderVitals(f); renderDepts(f); }
function heroSpark(vals, months) {         // the score over the whole game so far, on a 0-100 scale
  const w = 272, h = 38, n = Math.max(months || vals.length, 2), y = (v) => (h - 3 - v / 100 * (h - 6)).toFixed(1);
  if (vals.length < 2) return `<svg viewBox="0 0 ${w} ${h}" aria-hidden="true"></svg>`;
  const line = vals.map((v, i) => `${(i / (n - 1) * w).toFixed(1)},${y(v)}`).join(" "), xe = ((vals.length - 1) / (n - 1) * w).toFixed(1);
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">
    <defs><linearGradient id="heroFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#3ecf9e" stop-opacity=".32"/><stop offset="1" stop-color="#3ecf9e" stop-opacity="0"/></linearGradient></defs>
    <line x1="0" x2="${w}" y1="${y(50)}" y2="${y(50)}" stroke="rgba(255,255,255,.09)" stroke-dasharray="2 4" vector-effect="non-scaling-stroke"/>
    <polygon fill="url(#heroFill)" points="0,${h} ${line} ${xe},${h}"/>
    <polyline fill="none" stroke="#3ecf9e" stroke-width="1.8" stroke-linejoin="round" vector-effect="non-scaling-stroke" points="${line}"/></svg>`;
}
function renderHero(f) {
  const v = f.scores.city ?? 0, hist = S.frames.slice(0, S.idx + 1).map(fr => fr.scores.city ?? 0);
  const side = f.regionScores ? `${esc(S.city)} · region ${(f.regionScores.city || 0).toFixed(1)}` : "out of 100";
  $("hero").innerHTML = `<div class="hlab"><span>City score</span><span>${side}</span></div>
    <div class="hval"><b>${v.toFixed(1)}</b><span class="d">${trend(v, ago().scores.city ?? 0, { zero: '<span class="flat">steady</span>' })}</span>
    <span class="muted small">${S.idx >= YEAR ? "over the last year" : "since the start"}</span></div>${heroSpark(hist, S.meta.months)}`;
}
function alertsOf(m) {
  const out = [], bad = "var(--bad)", warn = "var(--warn)";
  const short = (what, sup, dem) => { if (dem > 0 && sup < dem * 0.98) out.push([`${what} short by ${Math.round(dem - sup).toLocaleString()}: ${Math.round(100 * sup / dem)}% of demand met`, bad]); };
  short("Power", m.power_supply, m.power_demand); short("Water", m.water_supply, m.water_demand);
  if (m.sewage_demand != null) short("Sewage", m.sewage_supply, m.sewage_demand);
  if (m.treasury < 0) out.push([`Treasury in the red at ${fmt$(m.treasury)}${m.months_negative ? ` for ${m.months_negative} month${m.months_negative > 1 ? "s" : ""}` : ""}`, bad]);
  if (m.fires > 0) out.push([`${m.fires} fire${m.fires > 1 ? "s" : ""} burning`, bad]);
  if (m.approval != null && m.approval < 0.35) out.push([`Mayor approval at ${pct(m.approval)}: recall risk`, bad]);
  if (m.debt >= 40000) out.push([`Debt ${fmt$(m.debt)}, near the $50k limit`, warn]);
  if (m.unemployment > 0.1) out.push([`Unemployment at ${pct(m.unemployment)}`, warn]);
  if (m.happiness < 0.45) out.push([`Happiness down to ${pct(m.happiness)}`, warn]);
  const unserved = (m.zoned_no_power || 0) + (m.zoned_no_water || 0);
  if (unserved >= 10) out.push([`${unserved} zoned lots without power or water`, warn]);
  if ((m.zoned_no_access || 0) >= 10) out.push([`${m.zoned_no_access} zoned lots with no road access`, warn]);
  if ((m.garbage_backlog || 0) > 50) out.push([`Garbage piling up: ${Math.round(m.garbage_backlog).toLocaleString()} uncollected`, warn]);
  return out;
}
function renderAlerts(f) {
  const a = alertsOf(f.metrics), shown = a.slice(0, 4);
  $("alerts").innerHTML = a.length ? shown.map(([t, c]) => `<div class="alert" style="--c:${c}">${esc(t)}</div>`).join("")
    + (a.length > shown.length ? `<div class="alert calm" title="${esc(a.slice(4).map(x => x[0]).join("\n"))}">and ${a.length - 4} more</div>` : "")
    : `<div class="alert calm">Nothing urgent this month</div>`;
}
function renderVitals(f) {
  const m = f.metrics, p = ago().metrics, net = m.income - m.expenses;
  const V = [["Population", m.population.toLocaleString(), trend(m.population, p.population, { digits: 0 }) || "no change"],
    ["Happiness", pct(m.happiness), trend(m.happiness, p.happiness, { pct: true }) || "no change"],
    ["Treasury", fmt$(m.treasury), `<span class="${net >= 0 ? "up" : "down"}">${net >= 0 ? "+" : ""}${fmt$(net)}</span> a month`],
    ["Unemployment", pct(m.unemployment), `${(m.workers || 0).toLocaleString()} workers`]];
  $("vitals").innerHTML = V.map(([k, v, sub]) => `<div class="vital"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${sub}</div></div>`).join("");
}
function renderDepts(f) {
  const roles = S.meta.roles, p = ago();
  const kinds = [...new Set(roles.map(r => S.meta.agents[r]))];
  $("lineup").textContent = kinds.length === 1 ? kinds[0] : `${kinds.length} kinds of agent`;
  $("lineup").title = roles.map(r => `${r}: ${S.meta.agents[r]}`).join("\n");
  $("scoreCards").innerHTML = roles.map(r => {
    const v = f.scores[r] ?? 0;
    return `<div class="drow ${roleOn(r) ? "" : "off"}" data-role="${r}" style="--c:${roleColor(r)}" title="${esc(r)} (${esc(S.meta.agents[r])}): click to show only this department in the right panel">
      <span class="n">${r}</span><span class="t"><i style="width:${Math.max(0, Math.min(100, v)).toFixed(1)}%"></i></span>
      <span class="sc">${v.toFixed(1)}</span><span class="dd">${trend(v, p.scores[r] ?? 0)}</span></div>`;
  }).join("");
  $("scoreCards").querySelectorAll(".drow").forEach(el => el.onclick = () => {
    const r = el.dataset.role;
    S.roleFilter = S.roleFilter.size === 1 && S.roleFilter.has(r) ? new Set() : new Set([r]);
    renderDepts(S.frames[S.idx]); renderTab(); updateBeams();
  });
}
function renderKpis(f) {
  const m = f.metrics;
  const k = [["Population", m.population.toLocaleString()], ["Happiness", pct(m.happiness)],
    ["Jobs in shops / industry / offices", `${m.jobs_com.toLocaleString()} <small>/ ${m.jobs_ind.toLocaleString()}${m.jobs_off != null ? " / " + m.jobs_off.toLocaleString() : ""}</small>`], ["Unemployment", pct(m.unemployment)],
    ["Pollution at homes", pct(m.pollution_res)], ["Land value", pct(m.land_value_dev)],
    ["Developed / zoned lots", `${m.developed_total} <small>/ ${m.zoned_total}</small>`], ["Road tiles · parks", `${m.road_tiles} <small>· ${m.parks}</small>`]];
  if (m.approval != null) k.splice(2, 0, ["Mayor approval", `<span style="color:${m.approval < 0.3 ? "var(--bad)" : m.approval < 0.45 ? "var(--warn)" : "inherit"}">${pct(m.approval)}</span>`],
    ["Noise at homes", pct(m.noise_res)]);
  if (m.soil_res != null) k.push(["Soil pollution at homes", pct(m.soil_res)], ["Historic · mixed use", `${m.historic_tiles} <small>· ${m.mixed_use}</small>`]);
  if (m.metro_lines) k.push(["Metro", `${m.metro_lines} lines <small>· ${Math.round(m.metro_riders)} riders</small>`]);
  if (m.exports != null) k.push(["Exports · imports", `${Math.round(m.exports)} <small>· ${Math.round(m.imports)} / ${Math.round(m.freight_capacity)}</small>`],
    ["Businesses losing money", `${m.businesses_losing} <small>· ${m.business_closures} closed</small>`]);
  if (m.milestones) k.push(["Latest milestone", m.milestones.length ? esc(m.milestones[m.milestones.length - 1]) + ` <small>(${m.milestones.length})</small>` : "none yet"],
    ["Tourists a month", Math.round(m.tourists || 0).toLocaleString()]);
  if (m.wealth) k.push(["Homes by wealth, low / mid / high", `${m.wealth.low} <small>/ ${m.wealth.middle} / ${m.wealth.high}</small>`],
    ["Level 4-5 buildings", `${m.levels_4_5}`]);
  if (m.commute_minutes != null) k.push(["Commute", `${Math.round(m.commute_minutes)} <small>min</small>`],
    ["Jammed roads · by bus", `${pct(m.congestion_index)} <small>· ${pct(m.transit_share)}</small>`]);
  if (m.crime_res != null) k.push(["Crime at homes", pct(m.crime_res)], ["Health", pct(m.health)],
    ["Adults with high school+", pct(m.education_index)], ["Fires · tiles lost", `${m.fires} <small>· ${m.fire_tiles_lost}</small>`]);
  $("kpis").innerHTML = k.map(([a, b]) => `<div class="srow"><span>${a}</span><b>${b}</b></div>`).join("");
  $("statSum").textContent = `${k.length} figures`;
}
const DEMAND = { R: ["Homes", "#5fd38d"], C: ["Shops", "#4aa3ff"], I: ["Industry", "#ffc857"], O: ["Offices", "#a99cf0"] };
function renderDemand(f) {
  const d = f.metrics.demand, keys = "RCIO".split("").filter(k => d[k] != null);
  $("demand").innerHTML = keys.map(k => {
    const v = d[k], wdt = Math.min(50, Math.abs(v) * 100), [name, c] = DEMAND[k];
    const bar = v >= 0 ? `left:50%;width:${wdt}%;background:${c}` : `right:50%;width:${wdt}%;background:${c};opacity:.45`;
    return `<div class="drg" title="${name} demand ${v.toFixed(2)}"><span>${name}</span><span class="axis"><i style="${bar}"></i></span><span class="vv">${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}</span></div>`;
  }).join("");
  const want = keys.filter(k => d[k] > 0.05).sort((a, b) => d[b] - d[a]).map(k => DEMAND[k][0].toLowerCase());
  $("demandSum").textContent = want.length ? `wants ${want.join(", ")}` : "nothing wanted";
}
function meter(label, supply, demand, color) {
  const max = Math.max(supply, demand, 1) * 1.1;
  const warn = supply < demand ? `<span style="color:var(--bad)">short ${Math.round(demand - supply)}</span>` : `<span class="muted">${Math.round((supply / Math.max(demand, 1e-9) - 1) * 100)}% spare</span>`;
  return `<div class="meter"><div class="top"><span>${label} <b>${Math.round(supply)}</b> <span class="muted">/ ${Math.round(demand)} used</span></span>${demand ? warn : ""}</div>
    <div class="bar"><i style="width:${supply / max * 100}%;background:${color};opacity:.35"></i><i style="width:${Math.min(supply, demand) / max * 100}%;background:${color}"></i>
    ${demand > supply ? `<b style="left:${demand / max * 100}%"></b>` : ""}</div></div>`;
}
const covered = (label, v) => `<span style="color:${v < 0.9 ? "var(--bad)" : "inherit"}">${label} ${pct(v)}</span>`;
function renderUtil(f) {
  const m = f.metrics;
  $("utilBars").innerHTML = meter("Power", m.power_supply, m.power_demand, "#ffc857") + meter("Water", m.water_supply, m.water_demand, "#4aa3ff") +
    (m.sewage_supply != null ? meter("Sewage", m.sewage_supply, m.sewage_demand, "#a1887f") : "") +
    `<div class="small muted">Homes covered: power ${pct(m.power_coverage)}, water ${pct(m.water_coverage)}${m.sewage_coverage !== undefined ? `, sewage ${pct(m.sewage_coverage)}` : ""}. Clean power ${m.power_supply ? Math.round(100 * (m.power_clean_capacity || 0) / m.power_supply) : 0}%` +
    (m.river_pollution != null ? `. River ${pct(m.river_pollution)} polluted (worst ${pct(m.river_pollution_max)})` : "") + `.</div>`;
  $("utilSum").innerHTML = [covered("power", m.power_coverage), covered("water", m.water_coverage), m.sewage_coverage != null ? covered("sewage", m.sewage_coverage) : ""].filter(Boolean).join(" · ");
}
function renderServices(f) {
  const m = f.metrics, el = $("services");
  if (m.police_coverage == null) { el.closest("details").hidden = true; return; }
  el.closest("details").hidden = false;
  const bar = (label, v, color, note = "") => `<div class="meter"><div class="top"><span>${label}</span><span>${pct(v)}${note ? ` <span class="muted">${note}</span>` : ""}</span></div>
    <div class="bar"><i style="width:${Math.round(100 * (v || 0))}%;background:${color}"></i></div></div>`;
  const e = m.enrollment || {}, sb = m.service_buildings || {};
  el.innerHTML = bar("Police cover", m.police_coverage, ROLE_C.police, `${sb.police_station || 0} stations`) +
    bar("Fire cover", m.fire_coverage, ROLE_C.fire, `${sb.fire_station || 0} stations`) +
    bar("Clinic cover", m.health_coverage, ROLE_C.health, `${(sb.clinic || 0) + (sb.hospital || 0)} buildings`) +
    bar("Garbage collected", m.garbage_coverage, "#b0916b", m.landfill_fill_pct != null ? `landfills ${Math.round(m.landfill_fill_pct)}% full` : "") +
    bar("School enrolment", e.elementary, ROLE_C.education, `high school ${pct(e.high)}`);
  $("servSum").innerHTML = [covered("police", m.police_coverage), covered("fire", m.fire_coverage), covered("clinics", m.health_coverage)].join(" · ");
}
function renderMoney(f) {
  const m = f.metrics, net = m.income - m.expenses, sign = (v) => `<span class="${v >= 0 ? "up" : "down"}">${v >= 0 ? "+" : ""}${fmt$(v)}</span>`;
  const rows = [["Treasury", fmt$(m.treasury)], ["Net a month", sign(net)], ["Income a month", fmt$(m.income)], ["Expenses a month", fmt$(m.expenses)],
    ["Debt", m.debt ? fmt$(m.debt) : "none"], ["Taxes: homes / shops / industry", `${Math.round(m.taxes.R * 100)}% / ${Math.round(m.taxes.C * 100)}% / ${Math.round(m.taxes.I * 100)}%`]];
  const extra = [];
  if (m.funding) {
    const off = Object.entries(m.funding).filter(([, v]) => Math.abs(v - 1) > 1e-6);
    extra.push(["Service funding", off.length ? off.map(([d, v]) => `${d} ${Math.round(100 * v)}%`).join(", ") : "all at 100%"]);
  }
  if (m.policies) extra.push(["Policies", m.policies.length ? `${m.policies.map(esc).join(", ")} <span class="muted">(${fmt$(m.policy_cost || 0)} a month)</span>` : "none"]);
  $("money").innerHTML = rows.map(([a, b]) => `<div class="moneyrow"><span class="muted">${a}</span><b>${b}</b></div>`).join("") +
    `<div class="budgets" title="What each department holds to spend">${Object.entries(m.dept_funds).map(([d, v]) => `<span style="--c:${roleColor(d)}"><em>${d}</em><b>${fmt$(v)}</b></span>`).join("")}</div>` +
    extra.map(([a, b], i) => `<div class="moneyrow wrap ${i ? "" : "mfirst"}"><span class="muted">${a}</span><b>${b}</b></div>`).join("") +
    (m.months_negative ? `<div class="small" style="color:var(--bad);margin-top:6px">Treasury negative for ${m.months_negative} month(s)</div>` : "");
  $("moneySum").innerHTML = `${fmt$(m.treasury)} · ${sign(net)} a month`;
}
// folded sections remember whether they are open
document.querySelectorAll("#left details.fold").forEach(d => {
  d.open = store.get("fold." + d.dataset.fold, false);
  d.addEventListener("toggle", () => store.set("fold." + d.dataset.fold, d.open));
});

// ================================================================ tabs
function actRect(a) {
  if (a.x1 != null && a.y1 != null) return { x1: +a.x1, y1: +a.y1, x2: +(a.x2 ?? a.x1), y2: +(a.y2 ?? a.y1) };
  if (a.x != null && a.y != null) return { x1: +a.x, y1: +a.y, x2: +a.x, y2: +a.y };
  return null;
}
function actText(a) { const { type, ...rest } = a; return `<b>${esc(type)}</b> ${esc(Object.entries(rest).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" "))}`; }
function roleChips() {
  const roles = S.meta ? S.meta.roles : [];
  const SHORT = { planning: "plan", utilities: "util", transport: "trans", education: "edu" };
  $("tabTools").innerHTML = roles.map(r => `<button class="chip ${S.roleFilter.size === 0 || S.roleFilter.has(r) ? "on" : ""}" data-role="${r}" data-name="${r}" style="--c:${roleColor(r)}"><span class="dot"></span>${SHORT[r] || r}</button>`).join("");
  // a click shows only that role (again: all roles); shift / cmd / ctrl-click adds or removes one
  $("tabTools").querySelectorAll(".chip").forEach(b => b.onclick = (ev) => {
    const r = b.dataset.role;
    if (ev.shiftKey || ev.metaKey || ev.ctrlKey) {
      if (S.roleFilter.size === 0) S.roleFilter = new Set([r]);
      else if (S.roleFilter.has(r)) S.roleFilter.delete(r); else S.roleFilter.add(r);
    } else S.roleFilter = S.roleFilter.size === 1 && S.roleFilter.has(r) ? new Set() : new Set([r]);
    if (S.roleFilter.size === roles.length) S.roleFilter.clear();
    renderTab();
  });
  $("tabTools").querySelectorAll(".chip").forEach(b => b.title = `${b.dataset.name}: show only this department (click again for all; shift-click to add or remove)`);
}
const roleOn = (r) => S.roleFilter.size === 0 || S.roleFilter.has(r);

// the quarter on show: turns are logged on the frame after the month they were taken in, and agents act every few months
function turnFrame(idx) { for (let j = Math.min(idx, S.frames.length - 1); j >= 0; j--) if ((S.frames[j].turns || []).length) return j; return -1; }
function nextTurnFrame(idx) { for (let j = idx + 1; j < S.frames.length; j++) if ((S.frames[j].turns || []).length) return j; return -1; }

// one line per turn from its successful actions: what a reader wants before the detail
const nOf = (rs, re) => rs.reduce((s, r) => s + (+((r.message || "").match(re) || [])[1] || 0), 0);
function built(r) {
  const m = (r.message || "").match(/^built (?:an? )?(.+?)(?: \(tier \d+\))? at /) || (r.message || "").match(/^upgraded (.+?) to/);
  const name = m ? m[1] : r.action.type.replace(/^build_/, "").replace(/_/g, " ");
  return BUILT_NAME[name] || name;
}
// the build messages use short kinds; the table says what they are
const BUILT_NAME = { coal: "coal plant", gas: "gas plant", wind: "wind turbine", solar: "solar farm", nuclear: "nuclear plant", pump: "water pump",
  tower: "water tower", outlet: "sewage outlet", treatment: "treatment plant", elementary: "elementary school", high: "high school" };
function headlinePart(type, rs) {
  if (type === "allocate_budget") {
    const out = rs.filter(r => +r.action.amount > 0), back = rs.filter(r => +r.action.amount < 0);
    const byDept = (xs, sign) => { const m = new Map(); for (const r of xs) m.set(r.action.department, (m.get(r.action.department) || 0) + sign * +r.action.amount); return [...m].map(([d, v]) => `${d} ${fmt$(v)}`).join(", "); };
    return [out.length ? "funded " + byDept(out, 1) : "", back.length ? "took back " + byDept(back, -1) : ""].filter(Boolean).join("; ");
  }
  const total = (k) => rs.reduce((s, r) => s + (+r.action[k] || 0), 0);
  if (type === "take_loan") return `borrowed ${fmt$(total("amount"))}`;
  if (type === "repay_loan") return `repaid ${fmt$(total("amount"))}`;
  if (type === "zone") return `zoned ${nOf(rs, /zoned (\d+) tiles/)} tiles`;
  if (type === "dezone") return `dezoned ${nOf(rs, /de-zoned (\d+) tiles/)} tiles`;
  if (type === "build_road") return `${nOf(rs, /built (\d+) road tiles/)} road tiles`;
  if (type === "set_tax") return "taxes " + rs.map(r => `${r.action.sector} ${r.action.rate}%`).join(", ");
  if (type.startsWith("build_") || type.startsWith("upgrade")) return rs.map(built).join(", ");
  return type.replace(/_/g, " ") + (rs.length > 1 ? ` ×${rs.length}` : "");
}
function turnHeadline(t) {
  const ok = t.results.filter(r => r.ok), groups = new Map();
  for (const r of ok) { const k = r.action.type; groups.set(k, [...(groups.get(k) || []), r]); }
  return ok.length ? esc([...groups].map(([k, rs]) => headlinePart(k, rs)).filter(Boolean).join(" · "))
    : `<span class="muted">${t.results.length ? "nothing succeeded" : "no actions"}</span>`;
}
// the trace summary (/turns) fills refusals and how the turn ended once it has loaded
function fillQuarterRow(role, s) {
  const ref = document.querySelector(`.qref[data-role="${role}"]`), end = document.querySelector(`.qend[data-role="${role}"]`);
  if (ref) ref.innerHTML = s.refused ? `<span class="warn" title="calls refused by a guard">⊘ ${s.refused}</span>` : "";
  if (end) end.innerHTML = !s.finish_reason || s.finish_reason === "end_turn" ? "" : `<span class="warn" title="how the turn ended: ${esc(s.finish_reason)}">${esc(s.finish_reason.replace("max_steps", "step limit").replace("no_tool_calls", "no call").replace("turn_tokens", "token budget"))}</span>`;
}

function renderTab() {
  document.querySelectorAll("#tabs button[data-tab]").forEach(b => b.classList.toggle("on", b.dataset.tab === S.tab));
  const more = document.querySelector(`#moreMenu button[data-tab="${S.tab}"]`);
  $("moreBtn").classList.toggle("on", !!more); $("moreBtn").textContent = more ? `${more.textContent} ▾` : "⋯";
  const body = $("tabBody");
  if (!S.frames.length && S.tab !== "runs") { $("tabTools").innerHTML = ""; body.innerHTML = `<div class="emptytab">Open or launch a run.</div>`; return; }
  const f = S.frames[S.idx];
  if (S.tab !== "team" && teamGraph) { teamGraph.destroy(); teamGraph = null; }
  if (S.tab === "team") {
    roleChips();
    renderTeam();
  } else if (S.tab === "agents") {
    roleChips();
    renderTurns(f);
  } else if (S.tab === "issues") {
    const v = S.issuesView;
    if (v === "agents") roleChips(); else $("tabTools").innerHTML = "";
    $("tabTools").insertAdjacentHTML("afterbegin", `<div class="seg issueseg" role="tablist"><button data-v="agents" class="${v === "agents" ? "on" : ""}">Agent mistakes</button><button data-v="city" class="${v === "city" ? "on" : ""}">City events</button></div>`);
    $("tabTools").querySelectorAll(".issueseg button").forEach(b => b.onclick = () => { S.issuesView = b.dataset.v; store.set("issues", S.issuesView); renderTab(); });
    if (v === "agents") renderProblems(); else renderEvents(f);
  } else if (S.tab === "money") {
    roleChips();
    renderMoneyTab();
  } else if (S.tab === "council") {
    roleChips();
    const msgs = S.frames.slice(0, S.idx + 1).flatMap(fr => fr.council || []).filter(m => roleOn(m.sender)).reverse();
    body.innerHTML = msgs.length ? msgs.map(m => `<div class="msgline" style="--c:${roleColor(m.sender)}"><span class="tag">m${m.month}</span>
      <b>${esc(m.sender)}</b> <span class="muted">→ ${esc(m.to)}</span><div>${esc(m.text)}</div></div>`).join("") : `<div class="emptytab">No messages yet.</div>`;
  } else if (S.tab === "charts") {
    $("tabTools").innerHTML = "";
    renderCharts();
  } else if (S.tab === "region") {
    $("tabTools").innerHTML = "";
    renderRegion();
  } else if (S.tab === "runs") {
    $("tabTools").innerHTML = `<button class="btn" id="rRefresh">↻ Refresh</button><button class="btn primary" id="rNew">＋ New run</button>`;
    $("rRefresh").onclick = () => { refreshRuns().then(renderTab); };
    $("rNew").onclick = openLaunch;
    renderRunsTab();
  }
}

// ---------------------------------------------------------------- turns
// one compact row per department that opens into its card: reasoning, actions (📍 flies there), the messages it
// sent with whether they were answered (the Team data), and its tool calls. Open rows stay open across quarters.
S.openTurns = new Set(store.get("openTurns", []));
function msgThread(t, m) {
  if (!S.collab) return null;
  const head = String(m.text || "").slice(0, 60);
  return S.collab.threads.find(th => th.from === t.role && th.month === t.month && th.to === m.to && th.text.startsWith(head)) || null;
}
function turnCard(t, ti, em) {
  const ok = t.results.filter(r => r.ok).length, bad = t.results.length - ok, open = S.openTurns.has(t.role);
  const spent = t.results.filter(r => r.ok).reduce((a, r) => a + (r.cost || 0), 0);
  const acts = t.results.map((r, ri) => { const rc = actRect(r.action); return `<div class="act ${r.ok ? "ok" : "bad"} ${rc ? "loc" : ""}" data-t="${ti}" data-r="${ri}">
    <span class="st">${r.ok ? "✓" : "✗"}</span><span><code>${actText(r.action)}</code><span class="msg">${esc(r.message)}${r.cost ? ` · ${fmt$(r.cost)}` : ""}</span></span>${rc ? `<span class="pin">📍</span>` : ""}</div>`; }).join("");
  const msgs = (t.messages || []).map(m => {
    const th = msgThread(t, m), st = th ? threadState(th, em) : null;
    return `<div class="tmsg"><span class="tto" style="color:${roleColor(m.to)}">to ${esc(m.to)}</span>${st ? `<span class="st st-${st}">${ST_TEXT[st] || st}</span>` : ""}<div class="ttxt">${esc(m.text)}</div></div>`;
  }).join("");
  return `<div class="tcard ${open ? "open" : ""}" id="turn-${esc(t.role)}" style="--c:${roleColor(t.role)}">
    <button class="thead" data-role="${esc(t.role)}" aria-expanded="${open}"><span class="trole">${esc(t.role)}</span><span class="twhat">${turnHeadline(t)}</span>
      <span class="tmeta">${spent ? `<span>${fmt$(spent)}</span>` : ""}<span>${ok} ✓</span>${bad ? `<span class="badn">${bad} ✗</span>` : ""}${(t.messages || []).length ? `<span title="messages sent">✉ ${t.messages.length}</span>` : ""}<span class="qref" data-role="${esc(t.role)}"></span><span class="qend" data-role="${esc(t.role)}"></span></span><span class="chev">▸</span></button>
    <div class="tbody" ${open ? "" : "hidden"}>
      <div class="tagline"><span>${esc(S.meta.agents[t.role] || "")}</span><span>month ${t.month}</span></div>
      ${t.parse_error ? `<div class="small" style="color:var(--warn)">⚠ ${esc(t.parse_error).slice(0, 140)}</div>` : ""}
      ${t.thinking ? `<div class="think" title="click to read it all">${esc(t.thinking)}</div>` : ""}
      <div class="tsec">Actions</div>${acts || `<div class="small muted">no actions</div>`}
      ${msgs ? `<div class="tsec">Messages sent</div>${msgs}` : ""}
      <div class="trace" data-role="${esc(t.role)}"></div>
    </div></div>`;
}
function renderTurns(f) {
  const body = $("tabBody");
  // agents act every few months: between their turns, keep showing the latest quarter instead of an empty tab
  const j = turnFrame(S.idx);
  S.agentsFrame = j;
  if (j < 0) { body.innerHTML = `<div class="emptytab">${f.month === 0 ? "Start of the game: press ▶ or step forward to see the agents' first decisions." : "No agent turns yet."}</div>`; return; }
  const tf = S.frames[j], turns = tf.turns.filter(t => roleOn(t.role));
  const prevJ = turnFrame(j - 1), nextJ = nextTurnFrame(S.idx), em = tf.month - 1;
  const since = j === S.idx ? "" : ` <span class="muted">· now month ${f.month}</span>`;
  const allOpen = turns.length && turns.every(t => S.openTurns.has(t.role));
  body.innerHTML = `<div class="qhead"><button class="btn qnav" id="qPrev" ${prevJ < 0 ? "disabled" : ""} title="previous quarter">◀</button>
      <div><b>Decisions from month ${em}</b>${since}</div><button class="qall" id="qAll">${allOpen ? "Close all" : "Open all"}</button>
      <button class="btn qnav" id="qNext" ${nextJ < 0 ? "disabled" : ""} title="next quarter">▶</button></div>` +
    (turns.length ? turns.map((t, ti) => turnCard(t, ti, teamMonth())).join("") : `<div class="emptytab">No turns for the selected roles.</div>`);
  loadTraces(em);
  if (prevJ >= 0) $("qPrev").onclick = () => go(prevJ);
  if (nextJ >= 0) $("qNext").onclick = () => go(nextJ);
  const save = () => store.set("openTurns", [...S.openTurns]);
  $("qAll").onclick = () => { for (const t of turns) allOpen ? S.openTurns.delete(t.role) : S.openTurns.add(t.role); save(); renderTurns(f); };
  body.querySelectorAll(".thead").forEach(el => el.onclick = () => {
    const r = el.dataset.role, card = el.parentElement, open = !S.openTurns.has(r);
    open ? S.openTurns.add(r) : S.openTurns.delete(r); save();
    card.classList.toggle("open", open); card.querySelector(".tbody").hidden = !open; el.setAttribute("aria-expanded", open);
  });
  body.querySelectorAll(".think, .ttxt").forEach(el => el.onclick = () => el.classList.toggle("full"));
  body.querySelectorAll(".act.loc").forEach(el => el.onclick = () => {
    const r = turns[+el.dataset.t].results[+el.dataset.r], rc = actRect(r.action);
    scene.highlightRect(rc); scene.flyTo((rc.x1 + rc.x2) / 2, (rc.y1 + rc.y2) / 2, 12 + Math.max(Math.abs(rc.x2 - rc.x1), Math.abs(rc.y2 - rc.y1)) * 0.6);
  });
}
// the city's own events (disasters, milestones, offers, recalls) and what residents complain about; the agents'
// failed calls are under Agent mistakes
function renderEvents(f) {
  const body = $("tabBody");
  const ev = S.frames.slice(0, S.idx + 1).flatMap(fr => (fr.events || []).map(e => [fr.month, e, /BANKRUPT|negative|recall|FULL|destroyed|approval|FLOOD|HEATWAVE|RECESSION|OFFER|MILESTONE/i.test(e)])).reverse();
  const comp = (f.metrics.complaints || []);
  const head = comp.length ? `<div class="complaints"><div class="tsec">What residents complain about this month</div>${comp.map(c =>
    `<div class="complaint"><span class="bar" style="width:${Math.min(100, Math.round(c.drag * 600))}%"></span><span>${esc(c.text)}</span></div>`).join("")}</div>` : "";
  body.innerHTML = head + (ev.length ? ev.slice(0, 300).map(([m, e, w]) => `<div class="msgline"><span class="tag ${w ? "warn" : ""}">m${m}</span> ${esc(e)}</div>`).join("") : `<div class="emptytab">Nothing has happened to the city yet.</div>`);
}

// ---------------------------------------------------------------- team (collab.js)
// the departments as a live network: requests, money and builds of each turn play out as the replay moves
let teamGraph = null, teamBig = null, collabAt = 0;
const beams = new BeamLayer($("beams"), { colorOf: roleColor, project: (x, y) => scene.tileToScreen ? scene.tileToScreen(x, y) : null });
const beamsOn = () => store.get("beams", true);
function updateBeams() {
  if (!S.collab || !S.frames.length) { beams.setMonth(-1, { on: false }); return; }
  beams.setData(S.collab);
  beams.setMonth(teamMonth(), { animate: S.anim, filter: roleOn, on: beamsOn(),
    budgetMs: S.playing ? Math.min(2600, +$("speed").value * (S.collab.act_every || 1) * 0.9) : 1600 });
}
function setBeams(on) { store.set("beams", on); toggleBtn("vBeams", on); updateBeams(); }
async function loadCollab(force = false) {
  const id = S.runId;
  if (!id || (!force && S.status === "running" && Date.now() - collabAt < 6000 && S.collab)) return;
  collabAt = Date.now();
  let d;
  try { d = await api(`/api/runs/${encodeURIComponent(id)}/collab${S.city ? `?city=${encodeURIComponent(S.city)}` : ""}`); }
  catch (e) {
    if (S.runId !== id) return;
    S.collabErr = /404/.test(e.message) ? "This server has no Team data yet: restart <code>civitas serve</code> (it doesn't reload Python changes)."
      : `Could not load the team: ${esc(e.message)}`;
    if (S.tab === "team") renderTeam();
    return;
  }
  if (S.runId !== id) return;
  S.collab = d; S.collabErr = null;
  for (const gph of [teamGraph, teamBig]) if (gph) gph.setData(d);
  S.anim = false; updateBeams();
  if (S.tab === "team") renderTeam(); updateTeamView();
}
const teamMonth = () => S.frames.length ? S.frames[S.idx].month - 1 : -1;     // turns are logged on the next frame
function teamStep(gph) {
  const f = S.frames[S.idx], m = f.metrics || {};
  const act = (S.collab && S.collab.act_every) || 1;
  gph.setMonth(teamMonth(), { funds: { ...(m.dept_funds || {}), __treasury: m.treasury }, animate: S.anim, filter: roleOn,
    budgetMs: S.playing ? Math.min(2600, +$("speed").value * act * 0.9) : 1600 });
}
// a thread's state as of the month on show
function threadState(t, em) {
  if (t.response && t.response.month <= em) return t.status;
  return em - t.month > (S.collab.window || 6) ? "unanswered" : "waiting";
}
const ST_TEXT = { met: "funded", partial: "partly funded", acted: "acted on", replied: "replied", waiting: "waiting", unanswered: "no answer", open: "waiting" };
function teamThreads(em) {
  const d = S.collab, pick = S.teamPick, win = 2 * (d.act_every || 1);
  return d.threads.filter(t => t.month <= em && t.month > em - win && (roleOn(t.from) || roleOn(t.to))
    && (!pick || (pick.role ? t.from === pick.role || t.to === pick.role : t.from === pick.from && t.to === pick.to))).reverse();
}
function threadHTML(t, em) {
  const st = threadState(t, em), r = t.response, o = t.outcome, see = (x) => x && x.month <= em;
  const steps = [];
  if (see(r)) steps.push(`<div class="tstep">↳ <span class="jump" data-m="${r.month}">m${r.month}</span> ${r.kind === "money" ? `${t.to} sent <b>${fmt$(r.amount)}</b>` : `${t.to}: ${esc(r.text).slice(0, 90)}`}</div>`);
  if (see(o)) steps.push(`<div class="tstep ${o.tile ? "loc" : ""}" data-x="${o.tile ? o.tile[0] : ""}" data-y="${o.tile ? o.tile[1] : ""}">↳ <span class="jump" data-m="${o.month}">m${o.month}</span> ${t.from} <b>${esc(o.what)}</b>${o.tile ? ` at (${o.tile[0]},${o.tile[1]}) 📍` : ""}</div>`);
  return `<div class="thread" style="--c:${roleColor(t.from)}"><span class="tag jump" data-m="${t.month}">m${t.month}</span>
    <b style="color:${roleColor(t.from)}">${esc(t.from)}</b> → <b style="color:${roleColor(t.to)}">${esc(t.to)}</b>${t.amount ? ` · ${fmt$(t.amount)}` : ""}
    <span class="st st-${st}">${ST_TEXT[st] || st}</span><div class="ttext">${esc(t.text)}</div>${steps.join("")}</div>`;
}
function teamSummary(em) {
  const d = S.collab, act = d.act_every || 1, inQ = (m) => m <= em && m > em - act;
  const asks = d.threads.filter(t => inQ(t.month));
  const answered = asks.filter(t => t.response && t.response.month <= em).length;
  const money = d.events.filter(e => e.kind === "money" && inQ(e.month) && !e.take_back).reduce((a, e) => a + e.amount, 0);
  const builds = d.events.filter(e => e.kind === "build" && inQ(e.month)).length;
  return `<div class="teamsum"><div><b>${asks.length}</b><span>requests</span></div><div><b>${asks.length ? Math.round(100 * answered / asks.length) + "%" : "—"}</b><span>answered so far</span></div>
    <div><b>${fmt$(money)}</b><span>money moved</span></div><div><b>${builds}</b><span>builds</span></div></div>`;
}
const TEAM_LEGEND = `<div class="teamlegend"><span><i style="background:#fff;box-shadow:0 0 6px #fff"></i>request</span><span><i style="background:#8a93a0"></i>report</span>
  <span><i style="background:#ffd34d"></i>money</span><span><i style="border:2px solid #5fd38d;background:none"></i>build</span>
  <span><i style="background:#45d483"></i>answered</span><span><i style="background:#ffc857"></i>open / partly</span><span><i style="background:#ff6b5e"></i>no answer</span></div>`;
function bindThreads(root) {
  root.querySelectorAll(".jump").forEach(el => el.onclick = (e) => {
    e.stopPropagation(); const m = +el.dataset.m, i = S.frames.findIndex(fr => fr.month === m + 1); if (i >= 0) go(i);
  });
  root.querySelectorAll(".tstep.loc").forEach(el => el.onclick = () => {
    const x = +el.dataset.x, y = +el.dataset.y; scene.highlightRect({ x1: x, y1: y, x2: x, y2: y }); scene.flyTo(x, y, 12);
  });
}
function teamList(em) {
  const th = teamThreads(em), pick = S.teamPick;
  const head = pick ? `<div class="teampick muted">Showing ${pick.role ? `<b style="color:${roleColor(pick.role)}">${esc(pick.role)}</b>'s threads` : `<b>${esc(pick.from)} → ${esc(pick.to)}</b>`} · <a href="#" class="unpick">show all</a></div>` : "";
  return head + (th.length ? th.map(t => threadHTML(t, em)).join("") : `<div class="emptytab">No requests in the last ${2 * (S.collab.act_every || 1)} months${pick ? " here" : ""}.</div>`);
}
function handoffList(em) {
  const win = S.collab.window || 6, pick = S.teamPick;
  const hs = (S.collab.handoffs || []).filter(h => h.month_to <= em && em - h.month_to < win && roleOn(h.to)
    && (!pick || (pick.role ? h.to === pick.role || h.from === pick.role : pick.from === h.from && pick.to === h.to))).reverse();
  if (!hs.length) return "";
  return `<div class="teamh">Land handed over · ⇢ on the map</div>` + hs.slice(0, 8).map(h =>
    `<div class="handoff" data-x="${h.tile_to[0]}" data-y="${h.tile_to[1]}" data-r="${h.rect_from.join(",")}"><span class="tag">m${h.month_to}</span>
      <b style="color:${roleColor(h.from)}">${esc(h.from)}</b>'s ${esc(h.what_from)} (m${h.month}) → <b style="color:${roleColor(h.to)}">${esc(h.to)}</b> ${esc(h.what)} 📍</div>`).join("");
}
function bindHandoffs(root) {
  root.querySelectorAll(".handoff").forEach(el => el.onclick = () => {
    const [x1, y1, x2, y2] = el.dataset.r.split(",").map(Number), x = +el.dataset.x, y = +el.dataset.y;
    scene.highlightRect({ x1: Math.min(x1, x), y1: Math.min(y1, y), x2: Math.max(x2, x), y2: Math.max(y2, y) });
    scene.flyTo((Math.min(x1, x) + Math.max(x2, x)) / 2, (Math.min(y1, y) + Math.max(y2, y)) / 2, 16);
  });
}
function onTeamPick(h) { S.teamPick = h; if (S.tab === "team") renderTeam(); updateTeamView(); }
function renderTeam() {
  const body = $("tabBody");
  $("tabTools").insertAdjacentHTML("beforeend", `<button class="btn" id="teamExpand" title="Open the full-screen Team view">⤢ Expand</button>`);
  $("teamExpand").onclick = openTeamView;
  if (!S.collab) {
    body.innerHTML = `<div class="emptytab">${S.collabErr || "Loading the team…"}</div>`;
    if (!S.collabErr && Date.now() - collabAt > 3000) loadCollab(true);
    return;
  }
  if (!teamGraph || !document.getElementById("teamCanvas")) {
    if (teamGraph) teamGraph.destroy();
    body.innerHTML = `<canvas id="teamCanvas" class="teamcanvas"></canvas>${TEAM_LEGEND}<div id="teamSum"></div><div id="teamList"></div>`;
    teamGraph = new CollabGraph($("teamCanvas"), { colorOf: roleColor, onPick: onTeamPick });
    teamGraph.setData(S.collab);
    S.anim = false;
  }
  teamStep(teamGraph);
  const em = teamMonth();
  $("teamSum").innerHTML = teamSummary(em);
  $("teamList").innerHTML = handoffList(em) + `<div class="teamh">Requests</div>` + teamList(em);
  bindThreads($("teamList")); bindHandoffs($("teamList"));
  const un = $("teamList").querySelector(".unpick"); if (un) un.onclick = (e) => { e.preventDefault(); teamGraph.selected = null; onTeamPick(null); };
}
function openTeamView() {
  $("teamModal").hidden = false;
  if (!teamBig) teamBig = new CollabGraph($("teamBig"), { colorOf: roleColor, onPick: onTeamPick, big: true });
  teamBig.setActive(true); teamBig.resize();
  if (S.collab) teamBig.setData(S.collab);
  S.anim = false; updateTeamView();
}
function updateTeamView() {
  if (!teamBig || $("teamModal").hidden || !S.collab || !S.frames.length) { if (teamBig) teamBig.setActive(!$("teamModal").hidden); return; }
  teamStep(teamBig);
  const em = teamMonth();
  $("teamMonth").textContent = `month ${S.frames[S.idx].month}`;
  $("teamSide").innerHTML = teamSummary(em) + TEAM_LEGEND + handoffList(em) + `<div class="teamh">Requests</div>` + teamList(em);
  bindThreads($("teamSide")); bindHandoffs($("teamSide"));
  const un = $("teamSide").querySelector(".unpick"); if (un) un.onclick = (e) => { e.preventDefault(); teamBig.selected = null; onTeamPick(null); };
}

// ---------------------------------------------------------------- region map
// the Region tab and the full-screen region map share one renderer (region.js)
let regionTabMap = null, regionBigMap = null;
function pickCity(c) { switchCity(c); if (S.tab === "region") renderTab(); }
function regionSummary(f, meta) {
  const flows = f.flows || [], tot = (k) => Math.round(flows.reduce((s, x) => s + (k === "goods" ? x.goods + x.raw : x[k]), 0));
  const rows = meta.cities.map(c => { const m = f.cities[c].metrics, sc = f.scores[`${c}.city`] ?? 0, spec = m.specializations || {};
    return `<tr data-city="${c}" class="${c === S.city ? "cur" : ""}"><td><b>${c}</b></td><td>${sc.toFixed(1)}</td><td>${m.population.toLocaleString()}</td><td>${Object.entries(spec).filter(([, v]) => v >= 0.33).sort((a, b) => b[1] - a[1]).map(([k, v]) => `<span class="spec" style="--c:${SPEC_COLOR[k]}">${k} ${Math.round(v * 100)}%</span>`).join(" ") || "—"}</td></tr>`; }).join("");
  const links = (f.links || []).map(l => `<span class="chip">${l.kind} · ${l.a}–${l.b}</span>`).join(" ");
  return `<div class="regionstats"><div><b>${tot("goods").toLocaleString()}</b><span>goods & raw / mo</span></div><div><b>${tot("commuters").toLocaleString()}</b><span>commuters</span></div><div><b>${tot("tourists").toLocaleString()}</b><span>tourists</span></div><div><b>${(f.scores.city || 0).toFixed(1)}</b><span>region score</span></div></div>
    <table class="lb"><tr><th>city</th><th>score</th><th>pop</th><th>specialisations</th></tr>${rows}</table>
    <div class="small muted" style="margin-top:8px">${links || "No links yet: cities must each build their side of a highway or rail connector."}</div>`;
}
function renderRegion() {
  const f = S.raw[S.idx], meta = S.metaRaw, body = $("tabBody");
  if (!f || !f.cities) { body.innerHTML = `<div class="emptytab">Not a region run.</div>`; return; }
  $("tabTools").innerHTML = `<button class="btn" id="regionExpand" title="Open the full-screen region map">⤢ Expand</button>`;
  $("regionExpand").onclick = openRegionMap;
  body.innerHTML = `<div class="small muted">Each city at its place in the region. Highways and rail start from the connectors the cities built; dots are this month's goods, commuters and tourists. Click a city to view it.</div>
    <canvas id="regionCanvas" class="regioncanvas"></canvas><div class="legendline">${regionLegend()}</div>${regionSummary(f, meta)}`;
  regionTabMap = new RegionMap($("regionCanvas"), { mini: MINI, onPick: pickCity });
  regionTabMap.set(f, meta, S.city);
  body.querySelectorAll("tr[data-city]").forEach(el => el.onclick = () => pickCity(el.dataset.city));
}
function openRegionMap() {
  $("regionModal").hidden = false;
  const cv = $("regionBig");
  cv.dataset.maxh = String(Math.max(300, window.innerHeight - 190));
  regionBigMap = regionBigMap || new RegionMap(cv, { mini: MINI, onPick: (c) => { pickCity(c); $("regionModal").hidden = true; } });
  updateRegionMap();
}
function updateRegionMap() {
  const f = S.raw && S.raw[S.idx];
  if (!f || !f.cities) return;
  if (regionBigMap && !$("regionModal").hidden) {
    regionBigMap.set(f, S.metaRaw, S.city);
    $("regionSide").innerHTML = `<div class="legendline">${regionLegend()}</div>` + regionSummary(f, S.metaRaw);
    $("regionSide").querySelectorAll("tr[data-city]").forEach(el => el.onclick = () => { pickCity(el.dataset.city); $("regionModal").hidden = true; });
    $("regionMonth").textContent = `month ${f.month}`;
  }
}

// leaderboard line-up without role names; fleet runs lead with their mode and fleet_version (grouped by both)
function boardLabel(r) {
  const sig = r.signature.replace(/[\w/]+=/g, "");
  if (!r.fleet_version) return sig;
  const mode = (sig.match(/\[(\w+)\]\s*$/) || [])[1] || "fleet";
  return `[${mode} ${r.fleet_version.slice(0, 6)}] ` + sig.replace(/\s*\[\w+\]\s*$/, "");
}

// full per-turn traces (what the agent saw, model reasoning, raw reply), loaded lazily
S.traces = {};
async function loadTraces(month) {
  if (month < 0 || !S.runId) return;
  const key = `${S.runId}:${month}`;
  if (!S.traces[key]) { try { S.traces[key] = await api(`/api/runs/${encodeURIComponent(S.runId)}/turns?month=${month}`); } catch { S.traces[key] = []; } }
  if (S.tab !== "agents" || S.frames[S.agentsFrame]?.month - 1 !== month) return;
  for (const tr of S.traces[key]) {
    if (S.city && !tr.role.startsWith(S.city + ".")) continue;
    if (tr.trace) fillQuarterRow(tr.role.split(".").pop(), tr.trace);
    const el = document.querySelector(`.trace[data-role="${tr.role.split(".").pop()}"]`); if (!el) continue;
    const u = tr.usage || {}, parts = [], fleet = !!tr.trace;
    if (tr.reasoning) parts.push(["Model reasoning", tr.reasoning]);
    if (tr.response) parts.push([fleet ? "End-of-turn report (or last reply)" : "Raw reply", tr.response]);
    if (tr.observation) parts.push([fleet ? "Briefing (first prompt)" : "What the agent saw (prompt)", tr.observation]);
    const stats = [tr.latency_s ? `${tr.latency_s.toFixed(1)} s` : "", u.prompt_tokens ? `${u.prompt_tokens}→${u.completion_tokens} tokens` : "",
      tr.finish_reason && tr.finish_reason !== "stop" && !fleet ? `finish: ${tr.finish_reason}` : "", tr.reward != null ? `reward ${tr.reward >= 0 ? "+" : ""}${tr.reward.toFixed(4)}` : ""].filter(Boolean).join(" · ");
    el.innerHTML = (stats ? `<div class="small muted" style="margin-top:6px">${esc(stats)}</div>` : "") + (fleet ? callsPanel(month, tr) : "") +
      parts.map(([t, v]) => `<details><summary>${t} <span class="muted">(${v.length.toLocaleString()} chars)</span></summary><pre class="log big">${esc(v)}</pre></details>`).join("");
    if (fleet) bindCalls(el.querySelector("details.calls"));
  }
}

// a fleet turn's tool calls (/trace), fetched when its panel is first opened; open panels stay open across re-renders
S.calls = {}; S.callsOpen = new Set();
const OUTCOME = { ok: ["✓", "ok"], failed: ["✗", "failed"], refused: ["⊘", "refused"], repeat: ["↺", "repeat"] };
const plural = (n, w, ws = w + "s") => `${n} ${n === 1 ? w : ws}`;
function callsHeader(s) {
  const bad = (t) => `<span class="bad">${t}</span>`;
  return [plural(s.calls, "call") + (s.model_calls ? ` <span class="muted">in ${plural(s.model_calls, "reply", "replies")}</span>` : ""),
    plural(s.actions, "action") + (s.failed_actions ? ` (${bad(`${s.failed_actions} failed`)})` : ""),
    s.errors ? bad(plural(s.errors, "error")) : "", s.refused ? `<span class="warn">${s.refused} refused</span>` : "",
    s.repeats ? plural(s.repeats, "repeat") : "", s.spent ? `${fmt$(s.spent)} spent` : "",
    s.finish_reason === "end_turn" ? `<span class="muted">ended with end_turn</span>` : `<span class="warn">stopped: ${esc(s.finish_reason || "?")}</span>`,
  ].filter(Boolean).join(" · ");
}
function callsPanel(month, tr) {
  const key = `${S.runId}:${month}:${tr.role}`;
  return `<details class="calls" data-key="${esc(key)}" data-month="${month}" data-role="${esc(tr.role)}" ${S.callsOpen.has(key) ? "open" : ""}>
    <summary>Tool calls &amp; reasoning <span class="callsum">${callsHeader(tr.trace)}</span></summary><div class="calllist"></div></details>`;
}
function argsText(a) {
  if (a == null || typeof a !== "object") return String(a ?? "");
  return Object.entries(a).map(([k, v]) => `${k}=${typeof v === "object" && v !== null ? JSON.stringify(v) : v}`).join(" ");
}
// the model's reasoning for one reply, shown above the calls that reply made (clamped; click to expand)
function stepBlock(st) {
  return `<div class="rstep" title="click to expand"><span class="rs-h">💭 reply ${st.step + 1} reasoning <span class="muted">(${st.chars.toLocaleString()} chars${st.chars > st.text.length ? `, first ${st.text.length.toLocaleString()} shown` : ""})</span></span><div class="rs-t">${esc(st.text)}</div></div>`;
}
function callRows(d) {
  const steps = new Map((d.steps || []).map(st => [st.step, st])), shown = new Set();
  const before = (step) => { const st = steps.get(step ?? 0); if (!st || shown.has(st.step)) return ""; shown.add(st.step); return stepBlock(st); };
  if (!d.calls.length) return [...steps.values()].map(stepBlock).join("") + `<div class="small muted">No tool calls: the turn ended before the model called a tool (${esc(d.summary.finish_reason)}).</div>`;
  const rest = () => [...steps.values()].filter(st => !shown.has(st.step)).map(stepBlock).join("");   // e.g. the last reply's thoughts
  return d.calls.map((c, i) => before(c.step) + (() => {
    const [glyph, word] = OUTCOME[c.outcome] || ["·", c.outcome], raw = typeof c.args === "string";
    const nested = !raw && c.args && Object.values(c.args).some(v => v && typeof v === "object");   // flat args read fine on the line
    const step = `model reply ${(c.step ?? 0) + 1}${c.finish_reason && c.finish_reason !== "tool_calls" ? ` (finish: ${c.finish_reason})` : ""}`;
    const extra = [step, c.budget_left != null ? `budget left ${fmt$(c.budget_left)}` : "", c.refused ? `refused: ${c.refused}` : "",
      c.action && !c.ran && c.outcome === "failed" ? "the action did not run" : ""].filter(Boolean).join(" · ");
    return `<div class="call o-${esc(c.outcome)}${nested ? " nested" : ""}${i && c.step !== d.calls[i - 1].step ? " newstep" : ""}" title="call ${c.n} · ${esc(step)} · click to expand">
      <span class="cn">${c.n}</span><span class="st">${glyph}</span>
      <span class="cb"><span class="cl"><b class="${c.action ? "gact" : ""}" title="${c.action ? "game action" : "look-up, memory or turn tool"}">${esc(c.tool)}</b> <code>${esc(raw ? c.args : argsText(c.args))}</code></span>
        ${c.text ? `<span class="cm">${esc(c.text)}</span>` : ""}
        <span class="cx small muted">${esc(extra)}${raw ? " · arguments were not a JSON object" : ""}</span>
        ${nested ? `<pre class="log cx">${esc(JSON.stringify(c.args, null, 1))}</pre>` : ""}</span>
      <span class="co">${c.outcome === "ok" ? "" : `<span class="tag ${c.outcome === "repeat" ? "" : "warn"}">${word}</span>`}${c.cost ? `<span class="cost">${fmt$(c.cost)}</span>` : ""}</span></div>`;
  })()).join("") + rest();
}
function fillCalls(det, d) {
  const list = det.querySelector(".calllist");
  if (d.error) { list.innerHTML = `<div class="small bad">Could not load the tool calls: ${esc(d.error)}</div>`; return; }
  det.querySelector(".callsum").innerHTML = callsHeader(d.summary);
  list.innerHTML = callRows(d);
  list.querySelectorAll(".call").forEach(el => el.onclick = (ev) => { if (!window.getSelection().toString()) el.classList.toggle("open"); ev.stopPropagation(); });
  list.querySelectorAll(".rstep").forEach(el => el.onclick = (ev) => { if (!window.getSelection().toString()) el.classList.toggle("open"); ev.stopPropagation(); });
  const fc = S.focusCall;                     // opened from the Problems tab: show that call
  if (fc && fc.key === det.dataset.key) {
    const el = list.querySelectorAll(".call")[fc.n - 1];
    if (el) { el.classList.add("focus", "open"); el.scrollIntoView({ behavior: "smooth", block: "center" }); }
    S.focusCall = null;
  }
}
S.callsLoading = {};
async function showCalls(det) {
  const key = det.dataset.key;
  if (S.calls[key]) return fillCalls(det, S.calls[key]);
  det.querySelector(".calllist").innerHTML = `<div class="small muted">loading…</div>`;
  S.callsLoading[key] = S.callsLoading[key] || api(`/api/runs/${encodeURIComponent(S.runId)}/trace?month=${det.dataset.month}&role=${encodeURIComponent(det.dataset.role)}`)
    .then(d => (S.calls[key] = d), e => ({ error: e.message }))      // an error isn't cached: opening again retries
    .finally(() => delete S.callsLoading[key]);
  const d = await S.callsLoading[key];
  const cur = [...document.querySelectorAll("details.calls")].find(x => x.dataset.key === key);   // re-rendered meanwhile?
  if (cur && cur.open) fillCalls(cur, d);
}
function bindCalls(det) {
  if (!det) return;
  det.addEventListener("toggle", () => {
    if (det.open) { S.callsOpen.add(det.dataset.key); showCalls(det); } else S.callsOpen.delete(det.dataset.key);
  });
  if (det.open) showCalls(det);
}

// ---------------------------------------------------------------- money
// the quarter's money: what the mayor saw (its briefing's money lines), what it moved, and each department's money
// from its turn to the next one; frames carry treasury, debt and dept_funds / dept_spent / dept_upkeep per month
const MONEY_LINE = /^- (treasury|last month|debt|to spend|to fund|first buildings|blocked for lack of land|\$[\d,.k]+ idle)/;
const MONEY_ACTS = new Set(["allocate_budget", "take_loan", "repay_loan", "set_tax", "set_funding", "set_district_tax", "set_service_imports"]);
function renderMoneyTab() {
  const body = $("tabBody"), j = turnFrame(S.idx);
  if (j < 0) { body.innerHTML = `<div class="emptytab">No mayor turn yet.</div>`; return; }
  const tf = S.frames[j], month = tf.month - 1, seen = S.frames[Math.max(0, j - 1)].metrics;
  const nj = nextTurnFrame(j), end = nj < 0 ? S.frames.length : nj, quarter = S.frames.slice(j, end), last = quarter[quarter.length - 1].metrics;
  const prevJ = turnFrame(j - 1), nextJ = nextTurnFrame(S.idx);
  const mayor = tf.turns.find(t => t.role === "mayor");
  const card = (v, label, cls = "") => `<div class="pcard ${cls}"><b>${v}</b><span>${label}</span></div>`;
  const head = `<div class="qhead"><button class="btn qnav" id="mPrev" ${prevJ < 0 ? "disabled" : ""} title="previous quarter">◀</button>
      <div><b>Money in the quarter from month ${month}</b> <span class="muted">· months ${month}-${quarter[quarter.length - 1].month - 1}</span></div>
      <button class="btn qnav" id="mNext" ${nextJ < 0 ? "disabled" : ""} title="next quarter">▶</button></div>
    <div class="small muted" style="margin-bottom:4px">At the mayor's turn (month ${month}):</div>
    <div class="pcards">${card(fmt$(seen.treasury), "treasury", seen.treasury < 0 ? "bad" : "")}${card(fmt$(seen.debt || 0), "debt", seen.debt ? "bad" : "")}
      ${card(fmt$(seen.net_worth ?? seen.treasury), "net worth")}${card(fmt$(seen.income || 0), "income / month", "good")}${card(fmt$(seen.expenses || 0), "expenses / month")}</div>`;
  // the mayor's own money lines, from its briefing (loaded with the turn traces)
  const tr = (S.traces[`${S.runId}:${month}`] || []).find(x => x.role.split(".").pop() === "mayor");
  const lines = tr ? (tr.observation || "").split("\n").filter(l => MONEY_LINE.test(l)) : null;
  const seenBlock = `<details class="mlines" open><summary>What the mayor read</summary>${lines == null ? `<div class="small muted">loading…</div>` :
    lines.length ? `<pre class="log">${esc(lines.join("\n"))}</pre>` : `<div class="small muted">(no money lines in this briefing: an older run)</div>`}</details>`;
  if (!tr) loadTraces(month).then(() => { if (S.tab === "money") renderMoneyTab(); });
  const moves = mayor ? mayor.results.filter(r => MONEY_ACTS.has(r.action.type)) : [];
  const movesBlock = `<div class="small muted" style="margin:8px 0 4px">What the mayor did:</div>` + (moves.length ? moves.map(r =>
    `<div class="act ${r.ok ? "ok" : "bad"}"><span class="st">${r.ok ? "✓" : "✗"}</span><span><code>${actText(r.action)}</code><span class="msg">${esc(r.message)}</span></span></div>`).join("")
    : `<div class="small muted">no money moves</div>`);
  // per department over the quarter: given / taken from the mayor's calls, spent from the monthly ledgers
  const given = {}, taken = {};
  for (const r of moves) if (r.ok && r.action.type === "allocate_budget") {
    const d = r.action.department, a = +r.action.amount; if (a > 0) given[d] = (given[d] || 0) + a; else taken[d] = (taken[d] || 0) - a;
  }
  const depts = Object.keys(last.dept_funds || {}).filter(roleOn);
  const rows = depts.map(d => {
    const held = (seen.dept_funds || {})[d] || 0, now = (last.dept_funds || {})[d] || 0;
    const spent = quarter.reduce((s, f) => s + ((f.metrics.dept_spent || {})[d] || 0), 0), up = (last.dept_upkeep || {})[d] || 0;
    const idle = now >= 1000 && spent === 0;
    return `<tr style="--c:${roleColor(d)}" class="${idle ? "midle" : ""}"><td class="qrole"><span class="dot"></span>${esc(d)}</td><td>${fmt$(held)}</td>
      <td>${given[d] ? `<span class="good">+${fmt$(given[d])}</span>` : "—"}</td><td>${taken[d] ? `<span class="warn">−${fmt$(taken[d])}</span>` : "—"}</td>
      <td>${spent ? fmt$(spent) : "—"}</td><td>${fmt$(now)}${idle ? ` <span class="tag warn" title="held $1,000+ and spent nothing all quarter">idle</span>` : ""}</td><td>${up ? fmt$(up) : "—"}</td></tr>`;
  }).join("");
  const table = `<div class="small muted" style="margin:10px 0 4px">Each department, month ${month} to ${quarter[quarter.length - 1].month - 1}:</div>
    <table class="lb ptab mtab"><tr><th>role</th><th title="funds at the mayor's turn">held</th><th title="topped up this turn">given</th><th title="taken back this turn">taken</th><th title="spent over the quarter">spent</th><th title="funds at the end of the quarter">holds</th><th title="upkeep per month">upkeep</th></tr>${rows}</table>`;
  const chart = `<div class="chart"><div class="ct"><span>Money over the run</span><b>${fmt$(S.frames[S.idx].metrics.treasury)}</b></div><svg id="mChart"></svg>
    <div class="legendline"><span style="--c:#ffc857"><i></i>treasury</span><span style="--c:#ff6b5e"><i></i>debt</span><span style="--c:#4aa3ff"><i></i>department funds</span><span style="--c:#e9ecef"><i></i>net worth</span></div></div>`;
  body.innerHTML = head + seenBlock + movesBlock + table + chart;
  if (prevJ >= 0) $("mPrev").onclick = () => go(prevJ);
  if (nextJ >= 0) $("mNext").onclick = () => go(nextJ);
  const fr = S.frames, sum = (o) => Object.values(o || {}).reduce((a, b) => a + b, 0), kf = (v) => Math.abs(v) >= 1000 ? (v / 1000).toFixed(0) + "k" : Math.round(v);
  lineChart($("mChart"), [{ values: fr.map(f => f.metrics.treasury), color: "#ffc857" }, { values: fr.map(f => f.metrics.debt || 0), color: "#ff6b5e" },
    { values: fr.map(f => sum(f.metrics.dept_funds)), color: "#4aa3ff" }, { values: fr.map(f => f.metrics.net_worth ?? null), color: "#e9ecef", width: 1.2 }], { yFmt: kf });
}

// ---------------------------------------------------------------- problems
// every failed / refused call and bad turn end in the run (/problems), and what the agent did about it
S.problems = null; S.probFilter = new Set();
const UNSOLVED = new Set(["dropped", "repeated", "ended"]);
const HANDLED = { fixed: ["fixed", "a later call of the same tool worked"], escalated: ["escalated", "gave up but messaged another role"],
  dropped: ["dropped", "gave up on it this turn"], repeated: ["repeated", "sent the same failing call again"],
  heeded: ["heeded", "a guard refused and the agent changed course"], overridden: ["overridden", "repeated the refused call with confirm"],
  ended: ["turn ended badly", "step limit, no tool call, token budget or an unreadable reply"] };
function probsAt(month) { const d = S.problems && S.problems.runId === S.runId ? S.problems.data : null; return d ? (d.byMonth[month] || []) : []; }
async function loadProblems() {
  const runId = S.runId, n = S.frames.length;
  if (!runId || (S.problems && S.problems.runId === runId && S.problems.at === n)) return;
  try {
    const d = await api(`/api/runs/${encodeURIComponent(runId)}/problems`);
    d.byMonth = {}; for (const p of d.problems) (d.byMonth[p.month] = d.byMonth[p.month] || []).push(p);
    S.problems = { runId, at: n, data: d };
  } catch (e) { S.problems = { runId, at: n, data: null, error: e.message }; }
  if (S.runId === runId) { drawTrackMarks(); if (S.tab === "issues" && S.issuesView === "agents") renderProblems(); }
}
const pct0 = (a, b) => b ? `${Math.round(100 * a / b)}%` : "—";
function renderProblems() {
  const body = $("tabBody");
  if (!S.problems || S.problems.runId !== S.runId || S.problems.at !== S.frames.length) {
    if (!S.problems || S.problems.runId !== S.runId) body.innerHTML = `<div class="emptytab">Loading problems…</div>`;
    loadProblems(); if (!S.problems || S.problems.runId !== S.runId) return;
  }
  const d = S.problems.data;
  if (!d) { body.innerHTML = `<div class="emptytab">Could not load problems: ${esc(S.problems.error || "")}</div>`; return; }
  const t = d.total || {};
  if (!d.problems.length) { body.innerHTML = `<div class="emptytab">No failed or refused calls in this run.</div>`; return; }
  const card = (v, label, title, cls = "") => `<div class="pcard ${cls}" title="${esc(title)}"><b>${v}</b><span>${label}</span></div>`;
  const head = `<div class="pcards">
      ${card(t.problems, "problems", "failed or refused calls and bad turn ends")}
      ${card(pct0(t.fixed, t.fixable), "fixed in turn", `${t.fixed} of ${t.fixable} failed calls: a later call of the same tool worked`, "good")}
      ${card(pct0(t.escalated, t.fixable), "escalated", `${t.escalated}: gave up but messaged another role (e.g. asked the mayor for money)`)}
      ${card(pct0(t.dropped + t.repeated, t.fixable), "unsolved", `${t.dropped} dropped, ${t.repeated} repeated the same failing call`, "bad")}
      ${card(t.recurred, "recurred", "the same role hit the same cause on its next turn", t.recurred ? "bad" : "")}
      ${card(t.guards ? `${t.heeded}/${t.guards}` : "—", "guards heeded", `${t.overridden} overridden with confirm`)}
      ${card(t.bad_ends, "bad turn ends", "step limit, no tool call, token budget, unreadable reply", t.bad_ends ? "bad" : "")}</div>`;
  const roles = Object.entries(d.roles).filter(([r, v]) => roleOn(r) && v.problems);
  const table = `<table class="lb ptab"><tr><th>role</th><th title="problems">⚠</th><th title="fixed in the same turn / fixable">fixed</th><th>escal.</th><th title="dropped + repeated">unsolved</th><th title="recurred next turn">↻</th><th title="guards heeded / refused">guards</th><th title="bad turn ends">ends</th></tr>` +
    roles.map(([r, v]) => `<tr style="--c:${roleColor(r)}"><td class="qrole"><span class="dot"></span>${esc(r)}</td><td>${v.problems}</td><td>${pct0(v.fixed, v.fixable)}</td><td>${v.escalated || 0}</td>
      <td>${v.dropped + v.repeated ? `<span class="bad">${v.dropped + v.repeated}</span>` : "0"}</td><td>${v.recurred ? `<span class="bad">${v.recurred}</span>` : "0"}</td>
      <td>${v.guards ? `${v.heeded}/${v.guards}` : "—"}</td><td>${v.bad_ends ? `<span class="warn">${v.bad_ends}</span>` : "0"}</td></tr>`).join("") + `</table>`;
  const hints = (d.hints || []).length ? `<details class="phints"><summary>Coaching hints from the fleet report (${d.hints.length})</summary>${d.hints.map(h => `<div class="msgline small">${esc(h.text || "")}</div>`).join("")}</details>` : "";
  const chips = `<div class="pchips">${Object.entries(HANDLED).map(([k, [w, title]]) => `<button class="chip ${S.probFilter.size === 0 || S.probFilter.has(k) ? "on" : ""} h-${k}" data-h="${k}" title="${esc(title)}">${w}</button>`).join("")}
    <button class="chip ${S.probFilter.has("recurred") ? "on" : ""} h-recurred" data-h="recurred" title="only problems whose cause came back on the role's next turn">↻ recurred only</button></div>`;
  const onlyRecurred = S.probFilter.has("recurred"), hs = new Set([...S.probFilter].filter(k => k !== "recurred"));
  const list = d.problems.filter(p => roleOn(p.role) && (hs.size === 0 || hs.has(p.handled)) && (!onlyRecurred || p.recurred));
  const cur = S.frames[turnFrame(S.idx)]?.month - 1;
  let last = null, rows = "";
  for (const p of list.slice(0, 500)) {
    if (p.month !== last) { rows += `<div class="pmonth ${p.month === cur ? "cur" : ""}">Month ${p.month}</div>`; last = p.month; }
    const [hw] = HANDLED[p.handled] || [p.handled];
    rows += `<div class="prow h-${esc(p.handled)}" data-m="${p.month}" data-role="${esc(p.role)}" data-n="${p.n ?? ""}" style="--c:${roleColor(p.role)}">
      <span class="prole">${esc(p.role)}</span>
      <span class="pbody"><span class="pline">${p.tool ? `<b>${esc(p.tool)}</b> <code>${esc(argsText(p.args))}</code>` : `<b>${esc(p.cause)}</b>`}</span>
        <span class="pmsg">${esc(p.text)}</span></span>
      <span class="pbadges"><span class="tag">${esc(p.kind)}</span><span class="tag hb h-${esc(p.handled)}" title="${esc((HANDLED[p.handled] || ["", ""])[1])}">${esc(hw)}${p.at ? ` · call ${p.at}` : ""}</span>${p.recurred ? `<span class="tag bad" title="the same cause on this role's next turn">↻</span>` : ""}</span></div>`;
  }
  body.innerHTML = head + compareBlock(d) + table + hints + chips + (rows || `<div class="emptytab">No problems match.</div>`) + (list.length > 500 ? `<div class="small muted">+${list.length - 500} more: filter by role or outcome</div>` : "");
  body.querySelectorAll(".pchips .chip").forEach(b => b.onclick = () => {
    const k = b.dataset.h;
    if (k === "recurred") { S.probFilter.has(k) ? S.probFilter.delete(k) : S.probFilter.add(k); }
    else if (S.probFilter.size === 0 || [...S.probFilter].every(x => x === "recurred")) { S.probFilter.add(k); }
    else if (S.probFilter.has(k)) S.probFilter.delete(k); else S.probFilter.add(k);
    renderProblems();
  });
  body.querySelectorAll(".prow").forEach(el => el.onclick = () => openProblem(+el.dataset.m, el.dataset.role, el.dataset.n ? +el.dataset.n : null));
  bindCompare();
}

// compare problem-solving rates across runs (per turn, so runs of different lengths compare); picks are remembered
S.compare = store.get("compare", []); S.cmpData = {};
async function loadCompare(id) {
  if (S.cmpData[id]) return;
  S.cmpData[id] = { loading: true };
  try {
    const [p, sm] = await Promise.all([api(`/api/runs/${encodeURIComponent(id)}/problems`), api(`/api/runs/${encodeURIComponent(id)}/summary`).catch(() => null)]);
    S.cmpData[id] = { p, sm };
  } catch (e) { S.cmpData[id] = { error: e.message }; }
  if (S.tab === "issues" && S.issuesView === "agents") renderProblems();
}
function cmpRates(p, sm) {
  const t = p.total || {}, turns = t.turns || 1;
  return { turns: t.turns || 0, perTurn: (t.problems || 0) / turns, fixed: t.fixable ? t.fixed / t.fixable : null, esc: t.fixable ? t.escalated / t.fixable : null,
    unsolved: t.fixable ? (t.dropped + t.repeated) / t.fixable : null, recurred: (t.recurred || 0) / turns, guards: t.guards ? t.heeded / t.guards : null,
    ends: (t.bad_ends || 0) / turns, city: sm && sm.final_scores ? sm.final_scores.city : null, pop: sm && sm.final_metrics ? sm.final_metrics.population : null,
    roles: Object.fromEntries(Object.entries(p.roles || {}).map(([r, v]) => [r, v.fixable ? v.fixed / v.fixable : null])) };
}
function compareBlock(d) {
  const ids = [S.runId, ...S.compare.filter(x => x !== S.runId)].slice(0, 5);
  const fleetRuns = (S.runs || []).filter(r => /fleet:/.test(r.signature || "") && !ids.includes(r.run_id));
  const picker = `<div class="pcmp-h"><b>Compare runs</b> <select id="pcmpAdd"><option value="">+ add a fleet run…</option>${fleetRuns.map(r => `<option value="${esc(r.run_id)}">${esc(r.run_id)}</option>`).join("")}</select></div>`;
  if (ids.length < 2) return `<div class="pcmp">${picker}<div class="small muted">Add runs to compare their problem-solving rates with this one.</div></div>`;
  const cur = { p: d, sm: null };
  const cols = ids.map(id => id === S.runId ? { id, ...cur, sm: S.cmpData[id]?.sm || null } : { id, ...(S.cmpData[id] || {}) });
  for (const c of cols) if (c.id === S.runId ? !S.cmpData[c.id] : !S.cmpData[c.id]) loadCompare(c.id);   // the open run's summary too
  const R = cols.map(c => c.p ? cmpRates(c.p, c.sm) : null);
  const f = { pct: (v) => v == null ? "—" : `${Math.round(v * 100)}%`, num: (v) => v == null ? "—" : v.toFixed(2), int: (v) => v == null ? "—" : Math.round(v).toLocaleString(), sc: (v) => v == null ? "—" : v.toFixed(1) };
  // higher is better unless flagged; the best value per row is highlighted
  const rows = [["final city score", "city", f.sc, 1], ["population", "pop", f.int, 1], ["turns", "turns", f.int, 0], ["problems per turn", "perTurn", f.num, -1],
    ["fixed in turn", "fixed", f.pct, 1], ["escalated", "esc", f.pct, 0], ["unsolved", "unsolved", f.pct, -1], ["recurred per turn", "recurred", f.num, -1],
    ["guards heeded", "guards", f.pct, 1], ["bad turn ends per turn", "ends", f.num, -1]];
  const cell = (r, k, fm, dir) => {
    const vals = R.map(x => x ? x[k] : null).filter(v => v != null), best = dir > 0 ? Math.max(...vals) : dir < 0 ? Math.min(...vals) : null;
    return R.map((x, i) => `<td class="${dir && x && x[k] != null && vals.length > 1 && x[k] === best ? "best" : ""}">${x ? fm(x[k]) : (cols[i].error ? "error" : "…")}</td>`).join("");
  };
  const head = `<tr><th></th>${cols.map(c => `<th title="${esc(c.id)}">${c.id === S.runId ? "<b>this run</b>" : `${esc(c.id.replace(/^fleet_/, "").slice(0, 16))} <span class="pcmp-x" data-x="${esc(c.id)}" title="remove">×</span>`}</th>`).join("")}</tr>`;
  const main = rows.map(([label, k, fm, dir]) => `<tr><td>${label}</td>${cell(null, k, fm, dir)}</tr>`).join("");
  const roles = [...new Set(R.flatMap(x => x ? Object.keys(x.roles) : []))];
  const per = roles.map(r => `<tr style="--c:${roleColor(r)}"><td class="qrole"><span class="dot"></span>${esc(r)}</td>${(() => {
    const vals = R.map(x => x ? x.roles[r] : null).filter(v => v != null), best = Math.max(...vals);
    return R.map(x => `<td class="${x && x.roles[r] != null && vals.length > 1 && x.roles[r] === best ? "best" : ""}">${x ? f.pct(x.roles[r]) : "…"}</td>`).join(""); })()}</tr>`).join("");
  return `<div class="pcmp">${picker}<table class="lb pcmp-t">${head}${main}</table>
    <details class="phints"><summary>Fixed in turn, per role</summary><table class="lb pcmp-t">${head.replace(/<span class="pcmp-x"[^<]*<\/span>/g, "")}${per}</table></details></div>`;
}
function bindCompare() {
  const sel = $("pcmpAdd"); if (sel) sel.onchange = () => { if (!sel.value) return; S.compare = [...S.compare.filter(x => x !== sel.value), sel.value].slice(-4); store.set("compare", S.compare); renderProblems(); };
  document.querySelectorAll(".pcmp-x").forEach(x => x.onclick = (ev) => { ev.stopPropagation(); S.compare = S.compare.filter(id => id !== x.dataset.x); store.set("compare", S.compare); renderProblems(); });
}
// jump to the quarter, the role's card and (for a call) its tool calls with that call shown
function openProblem(month, role, n) {
  const j = S.frames.findIndex(f => f.month === month + 1); if (j < 0) return;
  S.openTurns.add(role); store.set("openTurns", [...S.openTurns]);
  go(j); S.tab = "agents"; renderTab();
  const cardEl = document.getElementById(`turn-${role}`); if (!cardEl) return;
  cardEl.scrollIntoView({ behavior: "smooth", block: "start" }); cardEl.classList.remove("flash"); void cardEl.offsetWidth; cardEl.classList.add("flash");
  if (n == null) return;
  const open = () => {
    const det = cardEl.querySelector("details.calls"); if (!det) return false;
    S.focusCall = { key: det.dataset.key, n };
    if (det.open) showCalls(det); else det.open = true;             // the toggle handler loads and fills the calls
    return true;
  };
  if (!open()) { let tries = 0; const t = setInterval(() => { if (open() || ++tries > 40) clearInterval(t); }, 100); }   // traces load after the card
}

async function renderRunsTab() {
  const body = $("tabBody");
  let procs = [], board = [];
  try { [procs, board] = await Promise.all([api("/api/processes"), api("/api/leaderboard")]); } catch { /* server without launcher */ }
  if (S.tab !== "runs") return;
  const live = procs.filter(p => p.running);
  body.innerHTML = `
    ${live.length ? `<h4>Running now</h4>${live.map(p => `<div class="runrow" data-run="${esc(p.run_id)}"><span class="ellipsis"><b>${esc(p.run_id)}</b></span>
      <button class="btn danger" data-stop="${esc(p.run_id)}">Stop</button><pre class="log">${esc(p.log_tail || "starting…")}</pre></div>`).join("")}` : ""}
    <h4>Leaderboard <span class="muted small">(finished runs, grouped by agent line-up)</span></h4>
    ${board.length ? `<table class="lb"><tr><th>agents</th><th>n</th><th>city</th><th>pop</th>${Object.keys(board[0].roles).map(r => `<th style="color:${roleColor(r)}">${r.slice(0, 4)}</th>`).join("")}</tr>
      ${board.map(r => `<tr title="${esc(r.signature + (r.fleet_version ? `\nfleet_version ${r.fleet_version}` : ""))}"><td class="ellipsis" style="max-width:130px">${esc(boardLabel(r))}</td><td>${r.n}</td>
      <td><b>${r.city.mean.toFixed(1)}</b><span class="muted">±${r.city.std.toFixed(1)}</span></td><td>${Math.round(r.population.mean).toLocaleString()}</td>
      ${Object.keys(r.roles).map(k => `<td>${r.roles[k].mean.toFixed(0)}</td>`).join("")}</tr>`).join("")}</table>` : `<div class="emptytab">No finished runs yet.</div>`}
    <h4 style="margin-top:14px">All runs</h4>
    ${S.runs.map(r => `<div class="runrow" data-run="${esc(r.run_id)}"><span class="ellipsis"><b>${esc(r.name)}</b> · seed ${r.seed} <span class="muted small">${esc(r.run_id.slice(0, 15))}</span></span>
      <span>${r.scores ? `<b>${r.scores.city.toFixed(1)}</b> ` : ""}<span class="pill ${r.status}">${r.status}</span></span><span class="sig">${esc(r.signature || "")}</span></div>`).join("")}`;
  body.querySelectorAll(".runrow").forEach(el => el.onclick = (e) => { if (!e.target.dataset.stop) openRun(el.dataset.run); });
  body.querySelectorAll("[data-stop]").forEach(b => b.onclick = async (e) => {
    e.stopPropagation();
    try { await api("/api/stop", { run_id: b.dataset.stop }); toast("Run stopped."); } catch (err) { toast(esc(err.message), true); }
    await refreshRuns(); renderRunsTab();
  });
}

// ================================================================ charts
function lineChart(el, series, { yMin = null, yMax = null, yFmt = (v) => v } = {}) {
  const w = el.clientWidth || 340, h = 110, pl = 38, pr = 6, pt = 8, pb = 16, n = Math.max(2, S.meta.months + 1);
  const all = series.flatMap(s => s.values.filter(v => v != null));
  let lo = yMin ?? Math.min(0, ...all), hi = yMax ?? Math.max(1, ...all); if (hi === lo) hi = lo + 1;
  const X = (i) => pl + (w - pl - pr) * i / (n - 1), Y = (v) => pt + (h - pt - pb) * (1 - (v - lo) / (hi - lo));
  let o = "";
  for (const t of [lo, (lo + hi) / 2, hi]) o += `<line class="ax" x1="${pl}" x2="${w - pr}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${pl - 4}" y="${Y(t) + 3}" text-anchor="end">${yFmt(t)}</text>`;
  const step = Math.max(1, Math.ceil(n / 12 / 6));
  for (let yr = 0; yr * 12 < n; yr += step) o += `<text x="${X(yr * 12)}" y="${h - 3}" text-anchor="middle">y${yr}</text>`;
  for (const s of series) {
    const pts = s.values.map((v, i) => v == null ? null : `${X(S.frames[i]?.month ?? i).toFixed(1)},${Y(v).toFixed(1)}`).filter(Boolean).join(" ");
    o += `<polyline fill="none" stroke="${s.color}" stroke-width="${s.width || 1.7}" stroke-linejoin="round" points="${pts}"/>`;
  }
  const cx = X(S.frames[S.idx].month);
  o += `<line class="cur" x1="${cx}" x2="${cx}" y1="${pt}" y2="${h - pb}"/>`;
  el.setAttribute("viewBox", `0 0 ${w} ${h}`); el.innerHTML = o;
  el.onclick = (e) => { const r = el.getBoundingClientRect(), m = Math.round((e.clientX - r.left - pl) / (r.width - pl - pr) * (n - 1)); go(S.frames.findIndex(f => f.month >= m)); };
}
function renderCharts() {
  if (S.tab !== "charts" || !S.frames.length) return;
  const fr = S.frames, M = (k) => fr.map(f => f.metrics[k]), cur = fr[S.idx].metrics, kf = (v) => Math.abs(v) >= 1000 ? (v / 1000).toFixed(0) + "k" : Math.round(v);
  const defs = [
    ["Population", cur.population.toLocaleString(), [{ values: M("population"), color: "#3ecf9e" }], { yFmt: kf }],
    ["Scores", `city ${fr[S.idx].scores.city.toFixed(1)}`, ["city", ...S.meta.roles].map(k => ({ name: k, values: fr.map(f => f.scores[k] ?? 0), color: roleColor(k), width: k === "city" ? 2.6 : 1.4 })), { yMin: 0, yMax: 100 }],
    ["Treasury", fmt$(cur.treasury), [{ values: M("treasury"), color: "#ffc857" }], { yFmt: kf }],
    ["Income vs expenses", `${fmt$(cur.income)} / ${fmt$(cur.expenses)}`, [{ name: "income", values: M("income"), color: "#45d483" }, { name: "expenses", values: M("expenses"), color: "#ff6b5e" }], { yFmt: kf }],
    ["Happiness · coverage", pct(cur.happiness), [{ name: "happiness", values: M("happiness"), color: "#b99bff" }, { name: "approval", values: M("approval"), color: "#ffffff" }, { name: "power", values: M("power_coverage"), color: "#ffc857" }, { name: "water", values: M("water_coverage"), color: "#4aa3ff" }], { yMin: 0, yMax: 1, yFmt: (v) => Math.round(v * 100) + "%" }],
    ["Demand R / C / I", "", [{ name: "R", values: fr.map(f => f.metrics.demand.R), color: "#5fd38d" }, { name: "C", values: fr.map(f => f.metrics.demand.C), color: "#4aa3ff" }, { name: "I", values: fr.map(f => f.metrics.demand.I), color: "#ffc857" }], { yMin: -1, yMax: 1, yFmt: (v) => v.toFixed(1) }],
    ["Jobs & workers", "", [{ name: "workers", values: M("workers"), color: "#e9ecef" }, { name: "commercial", values: M("jobs_com"), color: "#4aa3ff" }, { name: "industrial", values: M("jobs_ind"), color: "#ffc857" }], { yFmt: kf }],
    ["Pollution at homes", pct(cur.pollution_res), [{ values: M("pollution_res"), color: "#ff6b5e" }], { yMin: 0, yMax: 1, yFmt: (v) => Math.round(v * 100) + "%" }],
  ];
  if (cur.commute_minutes != null) defs.push(
    ["Commute · traffic", `${Math.round(cur.commute_minutes)} min`, [{ name: "commute (min/60)", values: fr.map(f => f.metrics.commute_minutes != null ? f.metrics.commute_minutes / 60 : null), color: ROLE_C.transport },
      { name: "jammed roads", values: M("congestion_index"), color: "#ff6b5e" }, { name: "bus share", values: M("transit_share"), color: "#4aa3ff" }], { yMin: 0, yMax: 1, yFmt: (v) => Math.round(v * 100) + "%" }]);
  if (cur.exports != null) defs.push(
    ["Trade & freight", `${Math.round(cur.exports)} out · ${Math.round(cur.imports)} in`, [{ name: "exports", values: M("exports"), color: "#45d483" },
      { name: "imports", values: M("imports"), color: "#ff6b5e" }, { name: "freight capacity", values: M("freight_capacity"), color: "#9aa3ad" }], { yFmt: (v) => Math.abs(v) >= 1000 ? (v / 1000).toFixed(1) + "k" : Math.round(v) }]);
  if (cur.police_coverage != null) defs.push(
    ["Service coverage", "", [{ name: "police", values: M("police_coverage"), color: ROLE_C.police }, { name: "fire", values: M("fire_coverage"), color: ROLE_C.fire },
      { name: "clinics", values: M("health_coverage"), color: ROLE_C.health }, { name: "garbage", values: M("garbage_coverage"), color: "#b0916b" }], { yMin: 0, yMax: 1, yFmt: (v) => Math.round(v * 100) + "%" }],
    ["Crime · health", `${pct(cur.crime_res)} · ${pct(cur.health)}`, [{ name: "crime at homes", values: M("crime_res"), color: "#ff6b5e" }, { name: "health", values: M("health"), color: ROLE_C.health }], { yMin: 0, yMax: 1, yFmt: (v) => Math.round(v * 100) + "%" }],
    ["Education", pct(cur.education_index), [{ name: "adults high school+", values: M("education_index"), color: ROLE_C.education },
      { name: "elementary enrolment", values: fr.map(f => f.metrics.enrollment ? f.metrics.enrollment.elementary : null), color: "#9ccc65" },
      { name: "high school enrolment", values: fr.map(f => f.metrics.enrollment ? f.metrics.enrollment.high : null), color: "#4aa3ff" }], { yMin: 0, yMax: 1, yFmt: (v) => Math.round(v * 100) + "%" }]);
  $("tabBody").innerHTML = defs.map(([t, v, ser], i) => `<div class="chart"><div class="ct"><span>${t}</span><b>${v}</b></div><svg id="ch${i}"></svg>
    ${ser.length > 1 ? `<div class="legendline">${ser.map(s => `<span style="--c:${s.color}"><i></i>${s.name}</span>`).join("")}</div>` : ""}</div>`).join("") +
    `<div class="small muted">Click a chart to jump to that month.</div>`;
  defs.forEach(([, , ser, opt], i) => lineChart($(`ch${i}`), ser, opt));
}

// ================================================================ inspector & tooltip
function tileInfo(i) {
  const f = S.frames[S.idx], W = S.meta.width, ch = f.tiles[i];
  const L = f.layers || {}, g = (k) => L[k] ? +L[k][i] : null;
  return { x: i % W, y: Math.floor(i / W), ch, lvl: +f.levels[i], pol: +f.pollution[i], lv: +f.land_value[i], pw: f.power[i] === "1", wt: f.water[i] === "1", zone: "rciomRCIOM".includes(ch),
    crime: g("crime"), health: g("health"), garbage: g("garbage"), police: g("police"), fire: g("fire"), school: g("school"), happy: g("happiness"),
    variant: f.variant ? +f.variant[i] : 0, density: f.density ? +f.density[i] : null, resource: S.meta.resources ? +S.meta.resources[i] : 0 };
}
const IND_TYPE = ["", "farm", "sawmill", "mine", "oil field", "high-tech"], WEALTH = ["", "low income", "middle income", "high income"];
const RESOURCE = ["", "fertile soil", "forest", "ore", "oil"];
function tileTitle(t) {
  const up = t.ch.toUpperCase();
  if (!(t.zone && t.lvl > 0)) return nameOf(t.ch, t.lvl) + (t.ch === "." && t.resource ? ` (${RESOURCE[t.resource]})` : "");
  const kind = up === "I" && t.variant ? IND_TYPE[t.variant] : LEVEL_NAME[up][t.lvl];
  const extra = ["RM".includes(up) && t.variant ? WEALTH[t.variant] : "", t.density != null && "RCOM".includes(up) ? (t.density ? "high density" : "low density") : ""].filter(Boolean).join(", ");
  return `${NAMES[t.ch]} · ${kind} (L${t.lvl})${extra ? " · " + extra : ""}`;
}
function showTooltip(tile, ev) {
  const tip = $("tooltip");
  if (tile < 0 || !S.frames.length || !ev) { tip.hidden = true; return; }
  const t = tileInfo(tile);
  tip.innerHTML = `<b>${esc(tileTitle(t))}</b> <span class="muted">(${t.x}, ${t.y})</span><br>pollution ${t.pol}/9 · land value ${t.lv}/9${t.zone ? ` · ${t.pw ? "⚡" : "<span style='color:var(--bad)'>no power</span>"} ${t.wt ? "💧" : "<span style='color:var(--bad)'>no water</span>"}` : ""}`;
  tip.hidden = false;
  const r = app.getBoundingClientRect();
  tip.style.left = Math.min(ev.clientX + 14, r.width - tip.offsetWidth - 8) + "px";
  tip.style.top = Math.min(ev.clientY + 14, r.height - tip.offsetHeight - 8) + "px";
}
function select(tile) {
  S.selected = tile;
  scene.setSelected(tile);
  if (tile < 0) { $("inspector").hidden = true; return; }
  renderInspector(tile);
}
function covers(a, x, y) { const r = actRect(a); return r && x >= Math.min(r.x1, r.x2) && x <= Math.max(r.x1, r.x2) && y >= Math.min(r.y1, r.y2) && y <= Math.max(r.y1, r.y2); }
function roadCovers(a, x, y) {   // L-shaped roads: horizontal from (x1,y1) to (x2,y1), then vertical to y2
  if (a.type !== "build_road") return covers(a, x, y);
  const inH = y === +a.y1 && x >= Math.min(a.x1, a.x2) && x <= Math.max(a.x1, a.x2);
  const inV = x === +a.x2 && y >= Math.min(a.y1, a.y2) && y <= Math.max(a.y1, a.y2);
  return inH || inV;
}
function renderInspector(tile) {
  const t = tileInfo(tile), el = $("inspector");
  const hist = S.frames.slice(0, S.idx + 1).map(f => f.tiles[tile] === t.ch ? +f.levels[tile] : 0);
  const acts = [];
  for (const f of S.frames.slice(0, S.idx + 1)) for (const tr of f.turns || []) for (const r of tr.results)
    if (r.ok && roadCovers(r.action, t.x, t.y)) acts.push({ month: f.month - 1, role: tr.role, a: r.action, idx: S.frames.indexOf(f) });
  el.innerHTML = `<div class="head"><div><b>${esc(tileTitle(t))}</b><div class="muted small">tile (${t.x}, ${t.y})</div></div><button class="btn icon" id="insClose">✕</button></div>
    <div class="grid"><span>Pollution</span><b>${t.pol}/9</b><span>Land value</span><b>${t.lv}/9</b>
    ${t.zone ? `<span>Power</span><b style="color:${t.pw ? "var(--ok)" : "var(--bad)"}">${t.pw ? "connected" : "none"}</b><span>Water</span><b style="color:${t.wt ? "var(--ok)" : "var(--bad)"}">${t.wt ? "connected" : "none"}</b>` : ""}
    ${t.zone && t.crime != null ? [["Crime", t.crime], ["Health", t.health], ["Garbage", t.garbage], ["Police cover", t.police], ["Fire cover", t.fire], ["School", t.school]].map(([a, v]) => `<span>${a}</span><b>${v}/9</b>`).join("") : ""}</div>
    ${t.zone ? `<div class="small muted" style="margin-top:6px">Building level over time</div>${spark(hist, "#3ecf9e", 260, 36)}` : ""}
    <div class="hist">${acts.length ? acts.slice(-12).reverse().map(a => `<div data-i="${a.idx}"><span class="tag">m${a.month}</span> <b style="color:${roleColor(a.role)}">${a.role}</b> ${esc(a.a.type)}${a.a.zone ? " " + esc(a.a.zone) : ""}${a.a.kind ? " " + esc(a.a.kind) : ""}</div>`).join("") : `<span class="muted">No agent action touched this tile yet.</span>`}</div>`;
  el.hidden = false;
  $("insClose").onclick = () => select(-1);
  el.querySelectorAll(".hist [data-i]").forEach(d => d.onclick = () => { go(+d.dataset.i); S.tab = "agents"; renderTab(); });
}

// ================================================================ minimap & timeline marks
const MINI = { ".": "#7fb069", "~": "#3d8fd1", "=": "#b8b3a8", "#": "#3d4148", H: "#222", T: "#b99bff", P: "#3e8f3a", E: "#444", G: "#d9822b", S: "#1f3a73", W: "#f4f5f6", U: "#1f8a8a", V: "#2fa3b3",
  Y: "#8d8a82", "%": "#6d5f3f", "*": "#4f8f6f", K: "#1f4fa8", J: "#7d8388", F: "#c62828", "^": "#b8b39c", A: "#fafafa", h: "#eceff1", L: "#7a6a4f", X: "#5b5e62",
  ":": "#d32f2f", "?": "#8e24aa", ">": "#b0bec5", m: "#dcc9b0", M: "#a1887f",
  Z: "#6f9f5a", N: "#f3d99b", B: "#b5654a", Q: "#efe9dc", D: "#2e5b3c", "|": "#6f8f4f", "@": "#b23b3b", "&": "#f1c40f", "+": "#d8d2c4", $: "#dfe3e8", "!": "#e0b43a",
  r: "#b4c98a", c: "#a8bfd8", i: "#d8c9a0", o: "#c9c2e6", R: "#43a85a", C: "#3f7fd6", I: "#d99a2b", O: "#7d86bd", u: "#9fd3c7", z: "#6a6a6a" };
function drawMinimap() {
  if (!S.frames.length) return;
  const cv = $("minimap"), g = cv.getContext("2d"), f = S.frames[S.idx], W = S.meta.width, H = S.meta.height, s = cv.width / Math.max(W, H);
  g.fillStyle = "#111"; g.fillRect(0, 0, cv.width, cv.height);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, ch = f.tiles[i];
    g.fillStyle = MINI[ch] || "#f0f";
    if ("RCIOM".includes(ch)) g.globalAlpha = 0.5 + 0.1 * +f.levels[i];
    g.fillRect(x * s, y * s, Math.ceil(s), Math.ceil(s)); g.globalAlpha = 1;
  }
  const c = scene.cameraFootprint(), r = Math.min(cv.width / 2, c.dist * s * 0.45);
  g.strokeStyle = "#fff"; g.lineWidth = 1.5; g.beginPath(); g.arc((c.x + 0.5) * s, (c.y + 0.5) * s, Math.max(6, r * 0.35), 0, Math.PI * 2); g.stroke();
  g.beginPath(); g.moveTo((c.x + 0.5) * s, (c.y + 0.5) * s); g.lineTo((c.x + 0.5) * s - Math.sin(c.angle) * -r * 0.6, (c.y + 0.5) * s - Math.cos(c.angle) * -r * 0.6); g.stroke();
  if (S.selected >= 0) { g.strokeStyle = "#ffd23f"; g.strokeRect((S.selected % W) * s - 1, Math.floor(S.selected / W) * s - 1, s + 2, s + 2); }
}
function drawTrackMarks() {
  const cv = $("trackMarks"); if (!S.meta) return;
  const w = cv.clientWidth, h = 12; cv.width = w; cv.height = h;
  const g = cv.getContext("2d"), n = Math.max(1, S.frames.length - 1);
  g.clearRect(0, 0, w, h);
  S.frames.forEach((f, i) => {
    const x = 7 + (w - 14) * i / n;
    const bad = (f.turns || []).reduce((a, t) => a + (t.invalid || 0), 0), msgs = (f.council || []).length;
    if ((f.events || []).some(e => /negative|BANKRUPT|recall|LOST/.test(e))) { g.fillStyle = "#ff6b5e"; g.fillRect(x - 1, 0, 2, h); }
    else if ((f.events || []).some(e => /^(FLOOD|HEATWAVE|RECESSION|OFFER)/.test(e))) { g.fillStyle = "#4aa3ff"; g.fillRect(x - 1, 0, 3, h); }
    else if ((f.events || []).some(e => /^MILESTONE/.test(e))) { g.fillStyle = "#ffd23f"; g.fillRect(x - 1, 0, 3, h); }
    else if (bad) { g.fillStyle = "rgba(255,200,87,.8)"; g.fillRect(x - 1, 6, 2, 6); }
    if (msgs) { g.fillStyle = "rgba(185,155,255,.9)"; g.fillRect(x - 1, 0, 2, 4); }
    if (f.month && f.month % 12 === 0) { g.fillStyle = "rgba(255,255,255,.35)"; g.fillRect(x, 4, 1, 4); }
    const pr = probsAt(f.month - 1);          // this frame shows the turns of month - 1
    if (pr.length) { g.fillStyle = pr.some(p => UNSOLVED.has(p.handled)) ? "#ff6b5e" : "#ffc857"; g.fillRect(x - 2, h - 3, 4, 3); }
  });
}
function minimapJump(e) {
  if (!S.meta) return;
  const cv = $("minimap"), r = cv.getBoundingClientRect(), s = r.width / Math.max(S.meta.width, S.meta.height);
  scene.flyTo((e.clientX - r.left) / s - 0.5, (e.clientY - r.top) / s - 0.5, scene.cameraFootprint().dist);
}

// ================================================================ legend & view prefs
function renderLegend() {
  const L = $("legend"), o = OVERLAY_BY_ID[S.overlay];
  if (o && o.ramp) L.innerHTML = `<span>${o.lo}</span><span class="ramp" style="background:linear-gradient(90deg,${OVERLAY_RAMPS[o.ramp].join(",")})"></span><span>${o.hi}</span>`;
  else if (o && o.bool) L.innerHTML = `<span><i style="background:#34c759"></i>zone has ${o.id}</span><span><i style="background:#ff3b30"></i>zone without ${o.id}</span>`;
  else if (o && o.districts) { const names = (S.frames[S.idx] || {}).district_names || {};
    L.innerHTML = Object.keys(names).length ? Object.entries(names).map(([id, n]) => `<span><i style="background:${DISTRICT_COLORS[+id]}"></i>${esc(n)}</span>`).join("") : "<span>No districts in this run: Planning paints them with <code>paint_district</code> (the scripted planner never does)</span>"; }
  else if (o && o.cat) L.innerHTML = Object.entries(o.cat).map(([k, c]) => `<span><i style="background:${c}"></i>${o.names[k]}</span>`).join("");
  else L.innerHTML = "";
}
function buildLayerMenu() {
  const groups = [...new Set(OVERLAYS.map(o => o.group))];
  $("layerMenu").innerHTML = groups.map(g => `<div class="lgroup"><h4>${g}</h4>${OVERLAYS.filter(o => o.group === g).map(o =>
    `<button class="btn" data-ov="${o.id}">${o.label}${OVERLAYS.indexOf(o) < 9 ? ` <kbd>${OVERLAYS.indexOf(o) + 1}</kbd>` : ""}</button>`).join("")}</div>`).join("");
  $("layerMenu").querySelectorAll("button").forEach(b => b.onclick = () => { setOverlay(b.dataset.ov); $("layerMenu").hidden = true; });
}
function setOverlay(o) {
  if (!OVERLAY_BY_ID[o]) o = "none";
  S.overlay = o; store.set("overlay", o); scene.setOverlay(o);
  document.querySelectorAll("#layerMenu button").forEach(b => b.classList.toggle("on", b.dataset.ov === o));
  $("layerBtn").innerHTML = `◧ ${OVERLAY_BY_ID[o].label} <span class="caret">▾</span>`;
  $("layerBtn").classList.toggle("on", o !== "none");
  renderLegend();
}
function toggleBtn(id, on) { $(id).classList.toggle("on", on); }
function setNight(on) { store.set("night", on); scene.setNight(on); toggleBtn("vNight", on); }
function setQuality(q) { store.set("quality", q); scene.setQuality(q); $("vQuality").textContent = q === "low" ? "LQ" : "HQ"; }
function setDocks(l, r) { app.classList.toggle("hide-left", !l); app.classList.toggle("hide-right", !r); store.set("docks", [l, r]); setTimeout(() => { drawTrackMarks(); if (S.tab === "charts") renderCharts(); }, 300); }

// ================================================================ launch modal
async function openLaunch() {
  try { S.configs = (await api("/api/configs")).filter(c => !c.remote && !c.error); } catch (e) { toast("Launching needs `civitas serve` from the project folder: " + esc(e.message), true); return; }
  if (!S.configs.length) { toast("No configs found. Run `civitas serve` from the project folder (it reads configs/*.toml).", true); return; }
  const sel = $("lConfig");
  sel.innerHTML = S.configs.map(c => `<option value="${esc(c.file)}">${esc(c.name)} (${esc(c.file)})</option>`).join("");
  const last = store.get("lastConfig", "baseline.toml");
  if (S.configs.some(c => c.file === last)) sel.value = last;
  fillLaunch();
  $("launchModal").hidden = false;
}
// launcher: the config's own agent per role; only roles the user changes are sent as overrides
const DEFAULT_MODEL = "arcee/trinity-large-thinking";
const MODEL_KINDS = ["llm", "fleet"];
function roleDefault(a, kind) {          // provider/model shown for `kind` in a role whose config entry is `a`
  if (kind === a.kind && a.model) return `${a.provider || "arcee"}/${a.model}`;
  if (kind === "fleet" && a.fleet) return `${a.fleet.provider || "arcee"}/${a.fleet.model}`;
  return DEFAULT_MODEL;
}
function fillLaunch() {
  const c = S.configs.find(x => x.file === $("lConfig").value); if (!c) return;
  $("lDoc").textContent = c.doc || "";
  $("lMonths").placeholder = c.months; $("lAct").placeholder = c.act_every;
  const kinds = ["scripted", "llm", ...(c.region ? [] : ["fleet"]), "noop", "random"];   // fleets are single-city
  $("lAgents").innerHTML = Object.keys(c.agents).map(r => {
    const a = c.agents[r], opts = kinds.includes(a.kind) ? kinds : [a.kind, ...kinds];
    return `<div class="ar" style="--c:${roleColor(r)}"><b>${r}</b>
      <select data-role="${r}">${opts.map(k => `<option ${k === a.kind ? "selected" : ""}>${k}</option>`).join("")}</select>
      <input data-model="${r}" placeholder="provider/model" value="${esc(roleDefault(a, a.kind))}" ${MODEL_KINDS.includes(a.kind) ? "" : "disabled"}></div>`;
  }).join("");
  $("lAgents").querySelectorAll("select").forEach(s => s.onchange = () => {
    const i = $("lAgents").querySelector(`[data-model=${s.dataset.role}]`);
    i.disabled = !MODEL_KINDS.includes(s.value); i.value = roleDefault(c.agents[s.dataset.role], s.value); estimate(); });
  $("lAgents").querySelectorAll("input").forEach(i => i.oninput = estimate);
  estimate();
}
function estimate() {
  const c = S.configs.find(x => x.file === $("lConfig").value); if (!c) return;
  const months = +$("lMonths").value || c.months, every = +$("lAct").value || c.act_every, turns = Math.ceil(months / every);
  const sel = [...$("lAgents").querySelectorAll("select")];
  const llm = sel.filter(s => s.value === "llm").length, fleet = sel.filter(s => s.value === "fleet");
  const steps = Math.max(0, ...fleet.map(s => (c.agents[s.dataset.role].fleet || {}).max_steps || 20));
  const parts = [llm ? `≈ ${turns * llm} LLM calls` : "",
    fleet.length ? `${turns * fleet.length} fleet turns (up to ${steps} model calls each)` : ""].filter(Boolean);
  $("lCost").textContent = parts.length ? parts.join(" + ") : "no LLM calls · finishes in seconds";
}
async function launch() {
  const c = S.configs.find(x => x.file === $("lConfig").value), agents = {};
  $("lAgents").querySelectorAll("select").forEach(s => {
    const r = s.dataset.role, a = c.agents[r], m = $("lAgents").querySelector(`[data-model=${r}]`).value.trim();
    const modelChanged = MODEL_KINDS.includes(s.value) && m && m !== roleDefault(a, s.value);
    if (s.value === a.kind && !modelChanged) return;           // untouched: the config decides
    agents[r] = s.value === "llm" ? `llm:${m || DEFAULT_MODEL}` : s.value === "fleet" && modelChanged ? `fleet:${m}` : s.value;
  });
  const body = { config: $("lConfig").value, seed: +$("lSeed").value || 0, months: $("lMonths").value, act_every: $("lAct").value, agents };
  store.set("lastConfig", body.config);
  $("lGo").disabled = true;
  try {
    const r = await api("/api/launch", body);
    $("launchModal").hidden = true;
    toast(`Launched <b>${esc(r.run_id)}</b>`);
    for (let k = 0; k < 40; k++) {                // wait for the run folder
      await new Promise(res => setTimeout(res, 500));
      await refreshRuns();
      if (S.runs.some(x => x.run_id === r.run_id)) { S.follow = true; $("follow").checked = true; await openRun(r.run_id); break; }
      const procs = await api("/api/processes").catch(() => []);
      const p = procs.find(x => x.run_id === r.run_id);
      if (p && !p.running) { toast(`Run failed to start:<pre class="log">${esc(p.log_tail)}</pre>`, true); break; }
    }
  } catch (e) { toast("Launch failed: " + esc(e.message), true); }
  $("lGo").disabled = false;
}

// ================================================================ wiring
$("runBtn").onclick = (e) => { e.stopPropagation(); $("runMenu").hidden = !$("runMenu").hidden; if (!$("runMenu").hidden) refreshRuns(); };
document.addEventListener("click", (e) => { if (!$("runMenu").contains(e.target)) $("runMenu").hidden = true;
  if (!$("layerMenu").contains(e.target)) $("layerMenu").hidden = true; });
$("newRunBtn").onclick = openLaunch; $("emptyLaunch").onclick = openLaunch;
$("helpBtn").onclick = () => ($("helpModal").hidden = false);
$("regionBtn").onclick = openRegionMap;
document.querySelectorAll("[data-close]").forEach(b => b.onclick = () => { b.closest(".modal").hidden = true; if (teamBig) teamBig.setActive(!$("teamModal").hidden); });
document.querySelectorAll(".modal").forEach(m => m.onclick = (e) => { if (e.target === m) m.hidden = true; });
$("lConfig").onchange = fillLaunch; $("lMonths").oninput = estimate; $("lAct").oninput = estimate; $("lGo").onclick = launch;
document.querySelectorAll("#tabs button[data-tab]").forEach(b => b.onclick = () => { S.tab = b.dataset.tab; store.set("tab", S.tab); $("moreMenu").hidden = true; renderTab(); });
$("moreBtn").onclick = (e) => { e.stopPropagation(); $("moreMenu").hidden = !$("moreMenu").hidden; };
document.addEventListener("click", (e) => { if (!e.target.closest(".tabmore")) $("moreMenu").hidden = true; });
buildLayerMenu();
$("layerBtn").onclick = (e) => { e.stopPropagation(); $("layerMenu").hidden = !$("layerMenu").hidden; };
$("vReset").onclick = () => scene.resetView(); $("vTop").onclick = () => scene.topView();
$("vMode").onclick = () => setView(viewMode === "3d" ? "pixel" : "3d");
$("vRotate").onclick = () => { const on = !$("vRotate").classList.contains("on"); scene.setAutoRotate(on); toggleBtn("vRotate", on); };
$("vNight").onclick = () => setNight(!$("vNight").classList.contains("on"));
$("vBeams").onclick = () => setBeams(!beamsOn());
toggleBtn("vBeams", beamsOn());
$("vQuality").onclick = () => setQuality(scene.quality === "low" ? "high" : "low");
$("vShot").onclick = () => { const a = document.createElement("a"); a.href = scene.screenshot(); a.download = `civitas_${S.runId || "city"}_m${S.frames[S.idx]?.month ?? 0}.png`; a.click(); };
$("vFull").onclick = () => document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen?.();
$("vLeft").onclick = () => setDocks(app.classList.contains("hide-left"), !app.classList.contains("hide-right"));
$("vRight").onclick = () => setDocks(!app.classList.contains("hide-left"), app.classList.contains("hide-right"));
$("tPlay").onclick = () => (S.playing ? stop() : play());
$("tPrev").onclick = () => go(S.idx - 1); $("tNext").onclick = () => go(S.idx + 1, true);
$("tFirst").onclick = () => go(0); $("tLast").onclick = () => go(S.frames.length - 1);
$("slider").oninput = (e) => go(+e.target.value);
$("follow").onchange = (e) => { S.follow = e.target.checked; if (S.follow) go(S.frames.length - 1); };
$("minimap").addEventListener("click", minimapJump);
setInterval(drawMinimap, 250);

document.addEventListener("keydown", (e) => {
  const tag = e.target.tagName;
  if ((tag === "INPUT" && e.target.type !== "range") || tag === "SELECT" || tag === "TEXTAREA") return;
  if (e.key === "Escape") { document.querySelectorAll(".modal").forEach(m => m.hidden = true); $("runMenu").hidden = true; $("layerMenu").hidden = true; select(-1); scene.highlightRect(null); return; }
  if (!$("launchModal").hidden || !$("helpModal").hidden) return;
  const k = e.key;
  const map = {
    " ": () => (S.playing ? stop() : play()),
    ArrowRight: () => e.shiftKey ? go(S.idx + 12) : go(S.idx + 1, true), ArrowLeft: () => go(S.idx - (e.shiftKey ? 12 : 1)),
    Home: () => go(0), End: () => go(S.frames.length - 1),
    w: () => scene.panBy(0, 1), s: () => scene.panBy(0, -1), a: () => scene.panBy(-1, 0), d: () => scene.panBy(1, 0),
    ArrowUp: () => scene.panBy(0, 1), ArrowDown: () => scene.panBy(0, -1),
    q: () => scene.rotateBy(0.15), e: () => scene.rotateBy(-0.15), "+": () => scene.zoomBy(0.85), "=": () => scene.zoomBy(0.85), "-": () => scene.zoomBy(1.18),
    "0": () => scene.resetView(), t: () => scene.topView(), o: () => $("vRotate").click(), n: () => $("vNight").click(), g: () => $("vQuality").click(),
    p: () => $("vShot").click(), f: () => $("vFull").click(), l: () => { $("follow").checked = !$("follow").checked; $("follow").onchange({ target: $("follow") }); },
    "[": () => $("vLeft").click(), "]": () => $("vRight").click(), r: () => $("runBtn").click(), "?": () => ($("helpModal").hidden = false),
    v: () => $("layerBtn").click(), x: () => $("vMode").click(), b: () => $("vBeams").click(),
  };
  OVERLAYS.slice(0, 9).forEach((o, k) => { map[String(k + 1)] = () => setOverlay(o.id); });
  const fn = map[k] || map[k.toLowerCase()];
  if (fn) { e.preventDefault(); fn(); }
});
window.addEventListener("resize", () => { drawTrackMarks(); if (S.tab === "charts") renderCharts(); if (S.tab === "money") renderMoneyTab(); });

// ================================================================ boot
(async function boot() {
  const small = window.innerWidth < 820;
  const [l, r] = small ? [false, false] : store.get("docks", [true, true]);
  setDocks(l, r); setOverlay(S.overlay); setNight(store.get("night", false)); setQuality(store.get("quality", "high"));
  $("vMode").textContent = viewMode === "3d" ? "3D" : "PX";
  await refreshRuns();
  const want = new URLSearchParams(location.search).get("run") || store.get("lastRun", null);
  const pickRun = S.runs.find(x => x.run_id === want) || S.runs.find(x => x.status === "running") || S.runs[0];
  if (pickRun) await openRun(pickRun.run_id);
  else renderTab();
  window.__civitas = { S, get scene() { return scene; }, setView, go, openRun };      // handy for debugging from the console
})();
