// Compare view: several runs side by side (/api/compare, civitas/bench/compare.py). A table with the runs as columns
// (best value per row highlighted) and overlaid line charts of the city score, population and Teamwork over months,
// each run's scripted warm-up shaded and, from its hand-over, its scripted yardstick dashed (yardstick.json; the
// "since hand-over" rows show it beside the run's own change). A region run is one column (region values: the region's city score and
// population, the departments' and Teamwork's mean over its cities) or, split, one more column per city.

// categorical run colours, fixed order (a run keeps its slot while it stays picked): the dataviz palette's dark steps
export const RUN_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
export const COMPARE_MAX = 8;
const DASH = ["", "6 3", "2 3", "8 3 2 3", "1 2", "4 4"];   // a split region's cities
const YARD_DASH = "4 3";                                      // a warm-up run's scripted yardstick, thinner
const COMPS = [["responsiveness", "answers"], ["follow_through", "follow-through"], ["handoffs", "hand-offs"], ["money_aim", "money aim"]];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (d) => (v) => v == null ? "—" : v.toFixed(d);
const int = (v) => v == null ? "—" : Math.round(v).toLocaleString();
const signed = (d) => (v) => v == null ? "—" : (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(d);
const signedInt = (v) => v == null ? "—" : (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(Math.round(v)).toLocaleString();
const kfmt = (v) => Math.abs(v) >= 1000 ? (v / 1000).toFixed(Math.abs(v) >= 10000 ? 0 : 1) + "k" : String(Math.round(v));
const dur = (s) => s == null ? "—" : s >= 3600 ? `${(s / 3600).toFixed(1)} h` : s >= 60 ? `${Math.round(s / 60)} min` : `${s.toFixed(0)} s`;
const per = (c, k) => c && c.turns ? c[k] / c.turns : null;

export const shortName = (r) => `${r.name || r.run_id}${r.seed != null ? ` s${r.seed}` : ""}`;
// a short distinguishing tail of the run id: a version suffix (v19) or the launcher's timestamp
const idTail = (r) => { const m = /_(v\d+)$/.exec(r.run_id); if (m) return m[1]; const t = /^(\d{8})-(\d{4})/.exec(r.run_id); return t ? `${t[1].slice(4, 6)}/${t[1].slice(6)} ${t[2].slice(0, 2)}:${t[2].slice(2)}` : ""; };

// a column's name in the legend and tooltips: run, seed, the id's tail (v19 / launch time), the city of a split region
const colName = (c) => `${shortName(c.run)}${idTail(c.run) ? ` ${idTail(c.run)}` : ""}${c.city ? ` · ${c.city}` : ""}`;

// the table and chart columns: a run, or with `split` a region run and then each of its cities
export function columns(runs, { split = false, colorOf }) {
  const cols = [];
  for (const r of runs) {
    const color = colorOf(r.run_id);
    cols.push({ key: r.run_id, run: r, e: r, city: null, color, dash: "" });
    if (split && r.region) (r.cities || []).forEach((c, i) => cols.push({ key: `${r.run_id}/${c}`, run: r, e: (r.by_city || {})[c] || {}, city: c, color, dash: DASH[(i + 1) % DASH.length] }));
  }
  return cols;
}

// rows: [label, value(col), format, direction (+1 higher is better, -1 lower, 0 none), tip]; null = a group header
function rowsFor(cols) {
  const any = (f) => cols.some(c => f(c) != null);
  const warm = cols.some(c => c.run.warmup), region = cols.some(c => c.run.region), calls = any(c => c.e.calls);
  const yard = cols.some(c => c.e.yardstick);
  const roles = [...new Set(cols.flatMap(c => Object.keys(c.e.roles || {})))];
  const R = [
    ["Run"],
    ["kind", c => c.run.kind + (c.run.tags.length ? ` · ${c.run.tags.join(" · ")}` : ""), (v) => esc(v), 0, null, null, "text"],
    ["model", c => c.run.model || "—", (v) => esc(v), 0, null, null, "text"],
    ["months played", c => c.run.months, (v, c) => int(v) + (c.run.months_configured && c.run.months_configured !== v ? `<span class="muted"> / ${c.run.months_configured}</span>` : ""), 0],
    ...(warm ? [["scripted warm-up", c => c.run.warmup || null, (v) => v ? `${v} months` : "—", 0, "months 0 to N−1 played by the scripted fleet before the configured agents took over"]] : []),
    ["status", c => c.run.status, (v) => `<span class="pill ${esc(v)}">${esc(v)}</span>`, 0],
    ["City"],
    ["final city score", c => c.e.city, num(1), 1],
    ["average city score", c => c.e.avg_city, num(1), 1, "the mean of the monthly city score over every month played"],
    ...(warm ? [["average after warm-up", c => c.e.avg_after_warmup, vsYard(num(1), y => y.avg_city_score), 1, "the mean city score over the configured agents' months" + (yard ? YARD_TIP : "")],
      ["city score since hand-over", c => c.e.at_handover && c.e.city != null ? c.e.city - (c.e.at_handover.city || 0) : null, vsYard(signed(1), y => y.change.city), 1, "final city score minus the score when the warm-up ended" + (yard ? YARD_TIP : "")]] : []),
    ["population", c => c.e.population, int, 1],
    ...(warm ? [["population since hand-over", c => c.e.at_handover && c.e.population != null ? c.e.population - c.e.at_handover.population : null, vsYard(signedInt, y => y.change.population), 1, yard ? YARD_TIP.trim() : null]] : []),
    ["bankrupt", c => c.city ? null : c.run.bankrupt, (v) => v == null ? "—" : v ? `<span class="bad">yes</span>` : "no", 0],
    ["Departments (final score)"],
    ...roles.map(r => [r, c => (c.e.roles || {})[r], num(1), 1, null, r]),
    ["Teamwork"],
    ["Teamwork score", c => c.e.teamwork ? c.e.teamwork.score : null, (v) => v == null ? `<span class="muted">n/a</span>` : `<b>${v.toFixed(1)}</b>`, 1,
      "the mean of the four components with enough cases each (the Team tab's score); a region run's own column: the mean over its cities"],
    ...COMPS.map(([k, label]) => [label, c => c.e.teamwork ? c.e.teamwork.components[k] : null, bar, 1, null, null, "comp"]),
    ["requests", c => c.e.teamwork ? c.e.teamwork.requests : null, int, 0, "requests between departments (the Team tab's threads)"],
    ...(region ? [["asks between cities", c => c.city ? null : c.e.teamwork && c.e.teamwork.asks, (v) => v == null ? "—" : v.toFixed(1), 1, "share of asks to another city answered (n/a under the minimum cases)"],
      ["deals accepted", c => c.city ? null : c.e.teamwork && c.e.teamwork.deals, (v) => v == null ? "—" : v.toFixed(1), 1, "share of deal offers accepted"]] : []),
  ];
  if (calls) R.push(
    ["Fleet calls (fleet_report.json)"],
    ["turns", c => c.e.calls ? c.e.calls.turns : null, int, 0],
    ["tool calls per turn", c => per(c.e.calls, "tool_calls"), num(1), 0],
    ["failed actions per turn", c => per(c.e.calls, "failed"), num(2), -1],
    ["refused per turn", c => per(c.e.calls, "refused"), num(2), -1, "calls a guard refused"],
    ["repeats per turn", c => per(c.e.calls, "repeats"), num(2), -1, "a failed call sent again (refused by the tool box)"],
    ["tool errors per turn", c => per(c.e.calls, "tool_errors"), num(2), -1, "bad arguments, unknown tools"],
    ["overrides", c => c.e.calls ? c.e.calls.overrides : null, int, 0, "refusals overridden with confirm"]);
  R.push(["playing time", c => c.city ? null : c.run.wall_time_s, dur, 0]);
  return R;
}
// a warm-up run's "since hand-over" cells: the run's agents, then the scripted agents from the same start
// (yardstick.json, bench/yardstick.py) beside it; best-value highlighting stays on the run's own number
const YARD_TIP = "\nscripted: the same game from the same hand-over played by scripted agents (the run's yardstick)";
const vsYard = (fmt, pick) => (v, c) => {
  const y = c.e.yardstick, yv = y ? pick(y) : null;
  return fmt(v) + (yv == null ? "" : ` <span class="muted cmp-ys" title="scripted agents from the same start">· scripted ${fmt(yv)}</span>`);
};
function bar(v) {
  if (v == null) return `<span class="muted">n/a</span>`;
  return `<span class="cmp-bar"><i style="width:${Math.max(0, Math.min(100, v)).toFixed(0)}%"></i></span><span>${v.toFixed(0)}</span>`;
}

export function tableHTML(cols, { current, roleColor = () => "#ccc" } = {}) {
  const head = `<tr><th></th>${cols.map(c => {
    const r = c.run, cur = r.run_id === current && !c.city;
    return `<th class="${c.city ? "cmp-city" : ""}" style="--c:${c.color}" title="${esc(r.run_id)}\n${esc(r.signature || "")}">
      <div class="cmp-h">${c.city ? `<span class="cmp-sw${c.dash ? " dash" : ""}"></span><span class="cmp-cityname">${esc(c.city)}</span>`
        : `<span class="cmp-sw"></span><button class="cmp-open" data-open="${esc(r.run_id)}" title="Open this run">${esc(shortName(r))}</button>${cur ? `<span class="cmp-cur">open</span>` : ""}
           <button class="cmp-x" data-x="${esc(r.run_id)}" title="Remove from the comparison" aria-label="Remove">×</button>`}</div>
      <div class="cmp-sub">${c.city ? `of ${esc(shortName(r))}` : esc(idTail(r))}</div></th>`;
  }).join("")}</tr>`;
  const body = rowsFor(cols).map(([label, get, fmt, dir, tip, role, cls]) => {
    if (!get) return `<tr class="cmp-g"><td colspan="${cols.length + 1}">${esc(label)}</td></tr>`;
    const vals = cols.map(c => { try { return get(c); } catch { return null; } });
    const nums = vals.filter(v => typeof v === "number" && isFinite(v));
    const best = dir && nums.length > 1 && new Set(nums).size > 1 ? (dir > 0 ? Math.max(...nums) : Math.min(...nums)) : null;
    const lab = role ? `<span class="cmp-role" style="--c:${roleColor(role)}">${esc(label)}</span>` : esc(label);
    return `<tr class="${cls ? "cmp-" + cls : ""}"><td${tip ? ` title="${esc(tip)}"` : ""}>${lab}${tip ? ` <span class="muted">ⓘ</span>` : ""}</td>${vals.map((v, i) =>
      `<td class="${best != null && v === best ? "best" : ""}${cols[i].city ? " cmp-city" : ""}">${fmt(v, cols[i])}</td>`).join("")}</tr>`;
  }).join("");
  return `<table class="cmp-t">${head}${body}</table>`;
}

// ---------------------------------------------------------------- charts
export const CHARTS = [
  { id: "city", title: "City score", series: (c) => c.e.series ? c.e.series.city : [], months: (c) => c.run.month, yMin: 0, yMax: 100, fmt: (v) => v.toFixed(1),
    yard: (y) => y.series.city },
  { id: "population", title: "Population", series: (c) => c.e.series ? c.e.series.population : [], months: (c) => c.run.month, yMin: 0, fmt: (v) => Math.round(v).toLocaleString(), axis: kfmt,
    yard: (y) => y.series.population },
  { id: "teamwork", title: "Teamwork", series: (c) => c.e.series ? c.e.series.teamwork : [], months: (c, s) => s.map((_, i) => i), yMin: 0, yMax: 100, fmt: (v) => v.toFixed(1),
    note: "as of each month" },
];

export function chartsHTML() {
  return CHARTS.map(ch => `<div class="cmp-chart" data-chart="${ch.id}"><div class="ct"><span>${ch.title}</span><span class="muted small">${ch.note || ""}</span></div>
    <div class="cmp-plot"><svg></svg><div class="cmp-tip" hidden></div></div></div>`).join("");
}

export function legendHTML(cols, { yard = true } = {}) {
  const warm = cols.some(c => c.run.warmup), ys = cols.some(c => c.e.yardstick);
  return `<div class="cmp-legend">${cols.map(c => `<span style="--c:${c.color}"><svg width="22" height="8" aria-hidden="true"><line x1="1" x2="21" y1="4" y2="4" stroke="${c.color}" stroke-width="2.4" stroke-dasharray="${c.dash}"/></svg>${esc(colName(c))}</span>`).join("")}
    ${warm ? `<span class="muted"><i class="cmp-warmkey"></i>scripted warm-up (faded line)</span>` : ""}
    ${ys ? `<label class="muted cmp-yskey" title="A warm-up run's yardstick: the same game from the same hand-over played by scripted agents (yardstick.json)"><input type="checkbox" id="cmpYard" ${yard ? "checked" : ""}><svg width="22" height="8" aria-hidden="true"><line x1="1" x2="21" y1="4" y2="4" stroke="currentColor" stroke-width="1.8" stroke-dasharray="${YARD_DASH}"/></svg>scripted from the same start</label>` : ""}</div>`;
}

export function drawCharts(root, cols, { yard = true } = {}) {
  const lastMonth = Math.max(1, ...cols.map(c => (c.run.month || []).slice(-1)[0] || 0));
  for (const ch of CHARTS) {
    const box = root.querySelector(`[data-chart="${ch.id}"]`); if (!box) continue;
    const svg = box.querySelector("svg"), tip = box.querySelector(".cmp-tip");
    const lines = cols.map(c => { const s = ch.series(c) || []; return { c, s, m: ch.months(c, s) || [] }; });
    // the scripted yardstick from the hand-over, dashed in the run's colour (a run with yardstick.json)
    const yards = yard && ch.yard ? cols.filter(c => c.e.yardstick).map(c => ({ c, s: ch.yard(c.e.yardstick) || [], m: c.e.yardstick.series.month || [] })) : [];
    const all = [...lines, ...yards].flatMap(l => l.s.filter(v => v != null));
    const w = Math.max(240, svg.clientWidth || box.clientWidth || 320), h = 150, pl = 40, pr = 8, pt = 10, pb = 18;
    let lo = ch.yMin ?? Math.min(...all), hi = ch.yMax ?? Math.max(1, ...all) * 1.05; if (hi <= lo) hi = lo + 1;
    const X = (m) => pl + (w - pl - pr) * m / lastMonth, Y = (v) => pt + (h - pt - pb) * (1 - (v - lo) / (hi - lo));
    let o = "";
    // the warm-up spans: one band per run with a warm-up, stacked thin at the top, and a light wash under the lines
    const warm = [...new Map(cols.filter(c => c.run.warmup && !c.city).map(c => [c.run.run_id, c])).values()];
    warm.forEach((c, i) => {
      o += `<rect x="${X(0)}" y="${pt}" width="${X(c.run.warmup) - X(0)}" height="${h - pt - pb}" fill="${c.color}" opacity="0.07"/>`;
      o += `<rect x="${X(0)}" y="${pt - 8 + i * 3}" width="${X(c.run.warmup) - X(0)}" height="2" rx="1" fill="${c.color}" opacity="0.9"><title>${esc(shortName(c.run))}: scripted warm-up, months 0–${c.run.warmup - 1}</title></rect>`;
    });
    for (const t of [lo, (lo + hi) / 2, hi]) o += `<line class="ax" x1="${pl}" x2="${w - pr}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${pl - 5}" y="${Y(t) + 3}" text-anchor="end">${(ch.axis || ch.fmt)(t)}</text>`;
    const step = Math.max(1, Math.ceil(lastMonth / 12 / Math.max(2, Math.floor((w - pl) / 46))));
    for (let yr = 0; yr * 12 <= lastMonth; yr += step) o += `<text x="${X(yr * 12)}" y="${h - 4}" text-anchor="middle">y${yr}</text>`;
    for (const { c, s, m } of lines) {
      const warmN = c.run.warmup || 0;
      const pts = (keep) => s.map((v, i) => v == null || m[i] == null || !keep(m[i]) ? null : `${X(m[i]).toFixed(1)},${Y(v).toFixed(1)}`).filter(Boolean).join(" ");
      const dash = c.dash ? ` stroke-dasharray="${c.dash}"` : "";
      if (warmN) o += `<polyline fill="none" stroke="${c.color}" stroke-width="2" opacity="0.4" stroke-linejoin="round"${dash} points="${pts(x => x <= warmN)}"/>`;
      o += `<polyline fill="none" stroke="${c.color}" stroke-width="2" stroke-linejoin="round"${dash} points="${pts(x => x >= warmN)}"/>`;
    }
    for (const { c, s, m } of yards) {
      const pts = s.map((v, i) => v == null || m[i] == null ? null : `${X(m[i]).toFixed(1)},${Y(v).toFixed(1)}`).filter(Boolean).join(" ");
      o += `<polyline class="cmp-yard" fill="none" stroke="${c.color}" stroke-width="1.6" opacity="0.85" stroke-linejoin="round" stroke-dasharray="${YARD_DASH}" points="${pts}"><title>${esc(colName(c))}: scripted from the same start</title></polyline>`;
    }
    o += `<line class="cmp-cross" x1="0" x2="0" y1="${pt}" y2="${h - pb}" visibility="hidden"/>`;
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`); svg.setAttribute("height", h); svg.innerHTML = o;
    const cross = svg.querySelector(".cmp-cross");
    svg.onmousemove = (e) => {
      const r = svg.getBoundingClientRect(), x = (e.clientX - r.left) * w / r.width;
      const mo = Math.max(0, Math.min(lastMonth, Math.round((x - pl) / (w - pl - pr) * lastMonth)));
      cross.setAttribute("x1", X(mo)); cross.setAttribute("x2", X(mo)); cross.setAttribute("visibility", "visible");
      const rows = lines.map(({ c, s, m }) => { const i = m.indexOf(mo); const v = i >= 0 ? s[i] : null;
        return `<div><i style="background:${c.color}"></i>${esc(colName(c))}${c.run.warmup && mo < c.run.warmup ? ` <span class="muted">(warm-up)</span>` : ""}<b>${v == null ? "—" : ch.fmt(v)}</b></div>`; }).join("")
        + yards.map(({ c, s, m }) => { const i = m.indexOf(mo); if (i < 0 || s[i] == null) return "";
          return `<div class="muted"><i class="ys" style="border-color:${c.color}"></i>${esc(colName(c))} · scripted<b>${ch.fmt(s[i])}</b></div>`; }).join("");
      tip.innerHTML = `<div class="muted">month ${mo} · year ${Math.floor(mo / 12)}</div>${rows}`; tip.hidden = false;
      const px = (X(mo) / w) * r.width;
      const tw = tip.offsetWidth, side = px > r.width / 2 ? px - tw - 12 : px + 12;   // beside the cursor, kept in the plot
      tip.style.left = `${Math.max(0, Math.min(side, r.width - tw))}px`;
    };
    svg.onmouseleave = () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; };
  }
}
