// The Evals page (top bar "Evals"): add a model, press Evaluate, read a ranked result. Three tabs over the server's
// Evals endpoints (civitas/server.py): Models (the registry configs/models.toml, API keys, preflight tests), Evals
// (running evals with progress and Stop, finished ones with their scorecard) and Leaderboard (per environment version).
// A strip on top shows the environment version and whether the code still matches it (bench/envs.py). Everything
// reads without the launcher; edits need `civitas serve` with launching on, from this computer (`can_edit`).
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const BUILTIN = ["arcee", "openrouter", "openai", "anthropic"];
const POLL_MS = 3000;
const tok = (n) => n == null ? "—" : n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e7 ? 1 : 2)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(Math.round(n));
const usd = (v) => v == null ? "—" : v >= 100 ? `$${Math.round(v).toLocaleString()}` : `$${v.toFixed(2)}`;
const dur = (s) => s == null ? "—" : s >= 5400 ? `${(s / 3600).toFixed(1)} h` : s >= 60 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`;
const sgn = (v, d = 1) => v == null ? "—" : (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(d);
const when = (t) => t ? new Date(t * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "";

// fetch → {ok, status, data}: the page reads error bodies too ({"errors": [...]} on a refused save or start)
async function req(path, body) {
  try {
    const r = await fetch(path, body === undefined ? { cache: "no-store" } :
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await r.json().catch(() => ({}));
    return { ok: r.ok, status: r.status, data };
  } catch (e) { return { ok: false, status: 0, data: { error: `cannot reach the server (${e.message})` } }; }
}
const errText = (r) => (r.data && (r.data.errors || [r.data.error]).filter(Boolean).join("; ")) || `HTTP ${r.status}`;

// ---------------------------------------------------------------- markdown (scorecard.md): a small, safe subset
// headers, tables, lists, paragraphs, **bold**, *italic*, `code`; every character of the source is HTML-escaped first
function inline(s) {
  return String(s).split(/(`[^`]+`)/).map((part, i) => i % 2 ? `<code>${esc(part.slice(1, -1))}</code>` :
    esc(part).replace(/\*\*([^*]+?)\*\*/g, "<b>$1</b>").replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s).,;:!?]|$)/g, "$1<i>$2</i>")).join("");
}
const cells = (line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split(/(?<!\\)\|/).map(c => c.trim().replace(/\\\|/g, "|"));
const isSep = (line) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line);
export function md(src) {
  const lines = String(src ?? "").replace(/\r/g, "").split("\n");
  const out = [];
  let para = [];
  const flush = () => { if (para.length) out.push(`<p>${inline(para.join(" "))}</p>`); para = []; };
  for (let i = 0; i < lines.length;) {
    const line = lines[i];
    if (!line.trim()) { flush(); i++; continue; }
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) { flush(); const n = Math.min(h[1].length, 4); out.push(`<div class="md-h md-h${n}">${inline(h[2])}</div>`); i++; continue; }
    if (/^\s*\|/.test(line)) {
      flush();
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
      const head = rows.length > 1 && isSep(rows[1]) ? cells(rows[0]) : null;
      const align = head ? cells(rows[1]).map(c => c.endsWith(":") ? (c.startsWith(":") ? "center" : "right") : "") : [];
      const body = (head ? rows.slice(2) : rows).filter(r => !isSep(r)).map(cells);
      const td = (tag, c, j) => `<${tag}${align[j] ? ` style="text-align:${align[j]}"` : ""}>${inline(c)}</${tag}>`;
      out.push(`<div class="ev-scroll"><table class="md-t">${head ? `<thead><tr>${head.map((c, j) => td("th", c, j)).join("")}</tr></thead>` : ""}<tbody>${body.map(r => `<tr>${r.map((c, j) => td("td", c, j)).join("")}</tr>`).join("")}</tbody></table></div>`);
      continue;
    }
    const li = /^\s*([-*+]|\d+[.)])\s+(.*)$/.exec(line);
    if (li) {
      flush();
      const ordered = /\d/.test(li[1]), items = [];
      while (i < lines.length) {
        const m = /^\s*([-*+]|\d+[.)])\s+(.*)$/.exec(lines[i]);
        if (m && /\d/.test(m[1]) === ordered) { items.push(m[2]); i++; }
        else if (items.length && /^\s{2,}\S/.test(lines[i])) { items[items.length - 1] += " " + lines[i].trim(); i++; }
        else break;
      }
      out.push(`<${ordered ? "ol" : "ul"}>${items.map(x => `<li>${inline(x)}</li>`).join("")}</${ordered ? "ol" : "ul"}>`);
      continue;
    }
    para.push(line.trim()); i++;
  }
  flush();
  return `<div class="md">${out.join("")}</div>`;
}

// ---------------------------------------------------------------- the page
export function initEvals({ toast = () => {}, onCompareEval = null } = {}) {
  const P = {
    tab: (() => { try { return JSON.parse(localStorage.getItem("civitas.evalsTab")) || "models"; } catch { return "models"; } })(),
    models: null, providers: [], canEdit: false, modelsErr: null, env: null, envs: [], evals: null, jobs: [],
    progress: {}, lb: null, lbEnv: null, lbErr: null, suites: null, pf: {}, open: new Set(), cards: {}, keyFor: null,
    timer: null, envTimer: null, lastList: 0,
  };
  const modal = $("evalsModal"), body = $("evBody");
  const shown = () => !modal.hidden;
  const saveTab = () => { try { localStorage.setItem("civitas.evalsTab", JSON.stringify(P.tab)); } catch { /* ignore */ } };

  // ------------------------------------------------ loading
  async function loadModels() {
    const r = await req("/api/models");
    if (r.ok) { P.models = r.data.models || []; P.providers = r.data.providers || []; P.canEdit = !!r.data.can_edit; P.modelsErr = null; }
    else { P.models = []; P.modelsErr = r.status === 501 ? "The model registry isn't in this checkout yet (civitas/bench/models.py)." : errText(r); P.canEdit = false; }
    if (P.tab === "models") render();
  }
  async function loadEnv(refresh = false) {
    clearTimeout(P.envTimer);
    const r = await req(`/api/env${refresh ? "?refresh=1" : ""}`);
    P.env = r.ok ? r.data : { error: r.status === 501 ? "environment versions aren't in this checkout yet (civitas/bench/envs.py)" : errText(r), unavailable: r.status === 501 };
    renderEnv();
    if (P.env.computing && shown()) P.envTimer = setTimeout(() => loadEnv(), 2000);
    else if (P.env && !P.env.computing && P.tab === "models") render();   // the latest scores follow the version
  }
  async function loadEnvs() { const r = await req("/api/envs"); P.envs = r.ok && Array.isArray(r.data) ? r.data : []; }
  async function loadEvals() {
    const [a, b] = await Promise.all([req("/api/evals"), req("/api/evals/jobs")]);
    P.evals = a.ok && Array.isArray(a.data) ? a.data : [];
    P.jobs = b.ok && Array.isArray(b.data) ? b.data : [];
    P.lastList = Date.now();
    await loadProgress();
    if (P.tab === "evals") render();
  }
  async function loadProgress() {
    const refs = running().map(x => x.ref);
    const res = await Promise.all(refs.map(ref => req(`/api/evals/progress?dir=${encodeURIComponent(ref)}`)));
    refs.forEach((ref, i) => { if (res[i].ok) P.progress[ref] = res[i].data; });
  }
  async function loadBoard() {
    const env = P.lbEnv || (P.env && P.env.env) || "";
    const r = await req(`/api/evals/leaderboard${env ? `?env=${encodeURIComponent(env)}` : ""}`);
    P.lb = r.ok ? r.data : null;
    P.lbErr = r.ok ? null : r.status === 501 ? "The leaderboard isn't in this checkout yet (evals.leaderboard)." : errText(r);
    if (P.lb && !P.lbEnv) P.lbEnv = P.lb.env || null;
    if (P.tab === "board") render();
  }
  async function loadSuites() { if (P.suites) return P.suites; const r = await req("/api/evals/suites"); P.suites = r.ok ? r.data : []; return P.suites; }

  // a job that ended before writing eval.json (e.g. the eval's own check refused): it never started, whether or not
  // its folder was named in advance (a stamped eval knows it from the start)
  const failedStart = (j) => !!j && !j.running && !j.eval_json;
  // the evals still going: jobs this server started (until their folder is listed) and evals whose process lives
  function running() {
    const out = [], seen = new Set();
    for (const j of P.jobs) {
      if (!j.running && !failedStart(j)) continue;                       // ended after planning: listed as an eval
      if (failedStart(j) && Date.now() / 1000 - j.started > 600) continue;   // a failed start, shown for ten minutes
      out.push({ ref: j.eval_dir || j.job, job: j, name: j.job, model: j.model });
      if (j.eval_dir) seen.add(j.eval_dir);
    }
    for (const e of P.evals || []) if (e.status === "running" && !seen.has(e.dir)) out.push({ ref: e.dir, ev: e, name: e.name, model: e.model });
    return out;
  }

  // ------------------------------------------------ polling: progress every few seconds while the page is open
  function poll() {
    clearTimeout(P.timer);
    if (!shown()) return;
    P.timer = setTimeout(async () => {
      if (!shown()) return;
      const live = running();
      if (P.tab === "evals" || live.length) {
        const before = new Set(live.filter(x => !failedStart(x.job)).map(x => x.ref));   // a failed start stays put
        if (Date.now() - P.lastList > 15000 || !live.length) await loadEvals();
        else { await loadProgress(); const jb = await req("/api/evals/jobs"); if (jb.ok) P.jobs = jb.data; }
        const after = new Set(running().map(x => x.ref));
        if ([...before].some(r => !after.has(r) || (P.progress[r] && ["finished", "partial", "interrupted", "stopped", "failed"].includes(P.progress[r].status)))) {
          await loadEvals(); loadModels(); if (P.tab === "board") loadBoard();
        }
        if (P.tab === "evals") render();
      }
      poll();
    }, POLL_MS);
  }

  // ------------------------------------------------ the environment strip
  function renderEnv() {
    const e = P.env, el = $("evEnv");
    let st;
    if (!e) st = `<span class="ev-st muted">…</span>`;
    else if (e.computing) st = `<span class="ev-st busy"><span class="ev-spin"></span>checking… <span class="muted small">the scripted fleet replays every suite seed (~30 s, longer while evals start)</span></span>`;
    else if (e.unavailable_readonly) st = `<span class="ev-st muted" title="The probe runs only in a civitas serve with launching on">not checked (read-only)</span>`;
    else if (e.error) st = `<span class="ev-st bad" title="${esc(e.error)}">${e.unavailable ? "" : "check failed: "}${esc(e.error)}</span>`;
    else if (!e.frozen || !e.env) st = `<span class="ev-st warn">no version frozen</span>`;
    else if (e.match) st = `<span class="ev-st ok">✓ matches</span>`;
    else st = `<span class="ev-st warn" title="This code plays a different game from the frozen version: evals now are recorded as drifted and not ranked">⚠ changed: ${esc((e.diff || []).join(", ") || "?")}</span>`;
    if (e && e.stale && !e.computing) st += ` <span class="muted small" title="The code changed since this check; a read-only server doesn't probe again">(checked on older code)</span>`;
    const fp = e && e.fingerprint ? `<span class="muted small mono" title="this code's fingerprint">${esc(String(e.fingerprint).slice(0, 10))}</span>` : "";
    el.innerHTML = `<span class="ev-env-k">Environment</span><b class="ev-env-v">${esc((e && e.env) || "—")}</b>${st}
      ${e && e.suite ? `<span class="muted small">suite ${esc(e.suite)}</span>` : ""}${fp}<div class="grow"></div>
      ${e && !e.unavailable ? `<button class="btn" id="evRecheck" ${e.computing || !P.canEdit ? "disabled" : ""} title="Probe this code again">↻ Recheck</button>` : ""}
      ${e && !e.unavailable ? `<button class="btn" id="evFreeze" ${!P.canEdit || e.computing ? "disabled" : ""} title="${P.canEdit ? "Freeze the environment as it is now under a new version name" : "Editing needs civitas serve with launching on, on this computer"}">Freeze new version…</button>` : ""}`;
    if ($("evRecheck")) $("evRecheck").onclick = () => loadEnv(true);
    if ($("evFreeze")) $("evFreeze").onclick = freezeSheet;
  }

  // ------------------------------------------------ sheets: dialogs over the page (model form, evaluate, freeze)
  function sheet(html, bind) {
    const s = $("evSheet");
    s.innerHTML = `<div class="ev-dialog glass" role="dialog" aria-modal="true">${html}</div>`;
    s.hidden = false;
    s.onclick = (e) => { if (e.target === s) closeSheet(); };
    s.onkeydown = (e) => { if (e.key === "Escape") { e.stopPropagation(); closeSheet(); } };
    s.querySelectorAll("[data-sheet-close]").forEach(b => b.onclick = closeSheet);
    bind(s);
    const f = s.querySelector("input:not([type=checkbox]):not([readonly]), select, textarea"); if (f) f.focus();
  }
  function closeSheet() { const s = $("evSheet"); s.hidden = true; s.innerHTML = ""; }

  function freezeSheet() {
    const cur = (P.env && P.env.env) || "";
    const m = /^(.*?)(\d+)$/.exec(cur);
    const suggest = m ? `${m[1]}${+m[2] + 1}` : "civitas-v1";
    sheet(`<div class="modal-head"><h3>Freeze a new environment version</h3><button class="btn icon" data-sheet-close aria-label="Close">✕</button></div>
      <p class="small">Records the game as this code plays it (config, prompts, tools, and the scripted fleet's briefings, results and metrics on every suite seed) under a new name: <code>configs/envs/&lt;name&gt;.json</code> and a git tag <code>env/&lt;name&gt;</code>.</p>
      <p class="ev-warnbox small">Evals of different versions are never ranked together. After freezing, the reference models (and every model you want on the leaderboard) need a fresh eval on the new version, about 12M tokens each.</p>
      <label>Version name<input id="fzName" value="${esc(suggest)}" maxlength="64" autocomplete="off" spellcheck="false"></label>
      <div id="fzErr" class="small bad"></div>
      <div class="modal-foot"><div class="grow"></div><button class="btn" data-sheet-close>Cancel</button><button class="btn primary" id="fzGo">Freeze</button></div>`, (s) => {
      const go = async () => {
        const name = $("fzName").value.trim();
        if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(name)) { $("fzErr").textContent = "Letters, digits, . _ - only (e.g. civitas-v2)."; return; }
        $("fzGo").disabled = true; $("fzGo").textContent = "Freezing…";
        const r = await req("/api/env/freeze", { name });
        if (!r.ok) { $("fzErr").textContent = errText(r); $("fzGo").disabled = false; $("fzGo").textContent = "Freeze"; return; }
        closeSheet(); toast(`Froze environment <b>${esc(name)}</b>. Re-run the reference models on it.`);
        P.lbEnv = null; loadEnv(); loadEnvs(); if (P.tab === "board") loadBoard();
      };
      $("fzGo").onclick = go; $("fzName").onkeydown = (e) => { if (e.key === "Enter") go(); };
    });
  }

  function modelSheet(m = null) {
    const provs = [...new Set([...BUILTIN, ...P.providers.map(p => p.name), ...(m ? [m.provider] : [])])].filter(Boolean);
    const v = (k) => m && m[k] != null ? esc(m[k]) : "";
    const extra = m && m.extra_body && Object.keys(m.extra_body).length ? esc(JSON.stringify(m.extra_body, null, 2)) : "";
    sheet(`<div class="modal-head"><h3>${m ? `Edit ${esc(m.key)}` : "Add a model"}</h3><button class="btn icon" data-sheet-close aria-label="Close">✕</button></div>
      <div class="ev-form">
        <label>Key <span class="ev-hint">the name in evals and on the leaderboard</span><input id="mfKey" value="${v("key")}" ${m ? "readonly" : ""} placeholder="e.g. gpt-5" autocomplete="off" spellcheck="false"></label>
        <label>Provider<select id="mfProv">${provs.map(p => `<option ${(m ? m.provider : "arcee") === p ? "selected" : ""}>${esc(p)}</option>`).join("")}</select></label>
        <label class="ev-wide">Served id <span class="ev-hint">the id the API takes, when it differs from the key</span><input id="mfId" value="${v("id")}" placeholder="e.g. openai/gpt-5 on OpenRouter" autocomplete="off" spellcheck="false"></label>
        <label>Temperature <span class="ev-hint">blank: the suite's</span><input id="mfTemp" type="number" step="0.05" min="0" max="2" value="${m && m.temperature !== "default" ? v("temperature") : ""}" ${m && m.temperature === "default" ? "disabled" : ""}></label>
        <label class="ev-check small" title="Send no temperature at all: the provider's own (OpenAI's reasoning models refuse any other)"><input type="checkbox" id="mfTempDef" ${m && m.temperature === "default" ? "checked" : ""}> Provider default temperature <span class="ev-hint">send none</span></label>
        <label>Max tokens <span class="ev-hint">per reply</span><input id="mfMax" type="number" step="1" min="1" value="${v("max_tokens")}"></label>
        <label class="ev-wide">extra_body <span class="ev-hint">JSON object sent with every request, e.g. {"reasoning": {"effort": "high"}}</span><textarea id="mfExtra" rows="3" spellcheck="false" placeholder="{}">${extra}</textarea></label>
        <label class="ev-wide">Notes<input id="mfNotes" value="${v("notes")}" autocomplete="off"></label>
      </div>
      <p class="muted small">The compute budget (steps, tokens per turn, turns, playbooks) is the environment's, the same for every model: it can't be set here.</p>
      <div id="mfErr" class="small bad"></div>
      <div class="modal-foot"><div class="grow"></div><button class="btn" data-sheet-close>Cancel</button><button class="btn primary" id="mfGo">${m ? "Save" : "Add model"}</button></div>`, (s) => {
      const check = () => {
        const t = $("mfExtra").value.trim();
        if (!t) { $("mfExtra").classList.remove("bad-input"); return {}; }
        try { const o = JSON.parse(t); if (!o || typeof o !== "object" || Array.isArray(o)) throw new Error("not an object"); $("mfExtra").classList.remove("bad-input"); return o; }
        catch (e) { $("mfExtra").classList.add("bad-input"); return null; }
      };
      $("mfTempDef").onchange = () => { $("mfTemp").disabled = $("mfTempDef").checked; };
      $("mfExtra").oninput = () => { $("mfErr").textContent = check() === null ? "extra_body must be a JSON object, e.g. {\"top_p\": 0.9}" : ""; };
      $("mfGo").onclick = async () => {
        const key = $("mfKey").value.trim(), extra = check();
        if (!key) { $("mfErr").textContent = "Give the model a key."; return; }
        if (extra === null) { $("mfErr").textContent = "extra_body must be a JSON object."; return; }
        const fields = { provider: $("mfProv").value };
        const id = $("mfId").value.trim(); if (id && id !== key) fields.id = id;
        const t = $("mfTemp").value.trim();
        if ($("mfTempDef").checked) fields.temperature = "default"; else if (t !== "") fields.temperature = Number(t);
        const mx = $("mfMax").value.trim(); if (mx !== "") fields.max_tokens = Math.round(Number(mx));
        if (Object.keys(extra).length) fields.extra_body = extra;
        const notes = $("mfNotes").value.trim(); if (notes) fields.notes = notes;
        if (Number.isNaN(fields.temperature) || Number.isNaN(fields.max_tokens)) { $("mfErr").textContent = "Temperature and max tokens must be numbers."; return; }
        $("mfGo").disabled = true;
        const r = await req("/api/models", { key, fields });
        $("mfGo").disabled = false;
        if (!r.ok) { $("mfErr").innerHTML = (r.data.errors || [r.data.error || `HTTP ${r.status}`]).map(esc).join("<br>"); return; }
        closeSheet(); toast(`${m ? "Saved" : "Added"} <b>${esc(key)}</b>.`); loadModels();
      };
    });
  }

  async function evaluateSheet(m) {
    const suites = await loadSuites();
    let suite = (suites.find(s => s.default) || suites[0] || {}).file || "";
    let allow = false, plan = null, busy = false;
    const draw = (s) => {
      const box = s.querySelector("#evPlan");
      if (busy) { box.innerHTML = `<div class="ev-st busy"><span class="ev-spin"></span>Checking: the preflight (one request to the model, unless one passed in the last 10 minutes) and the environment probe…</div>`; }
      else if (!plan) box.innerHTML = "";
      else if (plan.error) box.innerHTML = `<div class="small bad">${esc(plan.error)}</div>`;
      else {
        const e = plan.estimate || {}, env = plan.env || {}, pf = plan.preflight;
        const drift = env.match === false || env.drift;
        box.innerHTML = `<div class="ev-kv">
            <span>Suite</span><b>${esc((plan.suite && plan.suite.name) || suite)}</b>
            <span>Runs</span><b>${plan.runs} <span class="muted">(${plan.units} units, ${plan.workers || "?"} at a time)</span></b>
            <span>Tokens</span><b>${tok(e.prompt_tokens)} prompt + ${tok(e.completion_tokens)} completion</b>
            <span>Time</span><b>${e.minutes == null ? "—" : dur(e.minutes * 60)}</b>
            <span>Cost</span><b>${e.cost == null ? `<span class="muted">no price for this model in the suite</span>` : usd(e.cost)}</b>
            <span>Environment</span><b>${esc(env.name || "none frozen")} ${env.match === true ? `<span class="good">✓ matches</span>` : drift ? `<span class="warnc">⚠ changed: ${esc((env.diff || []).join(", ") || "?")}</span>` : ""}</b>
          </div>
          ${e.reference_model ? `<div class="muted small">Estimates scale the suite's reference numbers (${esc(e.reference_model)}); another model's tokens and time can differ a lot.</div>` : ""}
          ${pf ? `<div class="ev-sub">Preflight</div>${pfHTML(pf)}` : ""}
          ${drift ? `<label class="ev-check"><input type="checkbox" id="evAllow" ${allow ? "checked" : ""}> Allow drift: run on the changed code anyway <span class="muted">(recorded as drifted, never ranked)</span></label>` : ""}
          ${plan.errors.length ? `<div class="ev-errs"><b>It can't start:</b><ul>${plan.errors.map(x => `<li>${esc(x)}</li>`).join("")}</ul></div>` : `<div class="good small">Ready to start.</div>`}`;
        const a = s.querySelector("#evAllow"); if (a) a.onchange = () => { allow = a.checked; check(s); };
      }
      s.querySelector("#evGo").disabled = busy || !plan || !!plan.error || plan.errors.length > 0;
    };
    // a preflight that passed in the last 10 minutes is reused by the server (no paid request per suite change or
    // drift toggle); "Check again" asks the model again
    const check = async (s, fresh = false) => {
      busy = true; plan = null; draw(s);
      const r = await req("/api/evals/plan", { model: m.key, suite, allow_drift: allow, ...(fresh ? { fresh_preflight: true } : {}) });
      busy = false; plan = r.ok ? r.data : { error: errText(r) }; draw(s);
    };
    sheet(`<div class="modal-head"><h3>Evaluate ${esc(m.key)}</h3><button class="btn icon" data-sheet-close aria-label="Close">✕</button></div>
      <div class="ev-form"><label class="ev-wide">Suite<select id="evSuite">${suites.map(x => `<option value="${esc(x.file)}" ${x.file === suite ? "selected" : ""} ${x.error ? "disabled" : ""}>${esc(x.name || x.file)}${x.units ? ` · ${x.units} units` : ""}${x.default ? " · default" : ""}${x.error ? " (broken)" : ""}</option>`).join("")}</select></label></div>
      <div id="evPlan" class="ev-plan"></div>
      <div class="modal-foot"><button class="btn" id="evRe">↻ Check again</button><div class="grow"></div><button class="btn" data-sheet-close>Cancel</button><button class="btn primary" id="evGo" disabled>Start eval ▸</button></div>`, (s) => {
      s.querySelector("#evSuite").onchange = (e) => { suite = e.target.value; allow = false; check(s); };
      s.querySelector("#evRe").onclick = () => check(s, true);
      s.querySelector("#evGo").onclick = async () => {
        const b = s.querySelector("#evGo"); b.disabled = true; b.textContent = "Starting…";
        const r = await req("/api/evals/start", { model: m.key, suite, allow_drift: allow });
        if (!r.ok) { plan = { ...(plan || {}), errors: r.data.errors || [errText(r)] }; b.textContent = "Start eval ▸"; draw(s); return; }
        closeSheet();
        toast(`Started the eval of <b>${esc(m.key)}</b>${r.data.eval_dir ? ` (${esc(r.data.eval_dir)})` : ""}.`);
        P.tab = "evals"; saveTab(); await loadEvals(); render(); poll();
      };
      check(s);
    });
  }

  const pfHTML = (pf) => `<div class="ev-pf">${(pf.steps || []).map(st => `<div class="ev-step ${st.ok ? "ok" : "bad"}"><span class="st">${st.ok ? "✓" : "✗"}</span><b>${esc(st.name)}</b><span class="muted">${esc(st.detail || "")}</span></div>`).join("")}
    ${pf.error ? `<div class="small bad">${esc(pf.error)}</div>` : ""}${pf.seconds != null ? `<div class="muted small">${pf.ok ? "passed" : "failed"} in ${Number(pf.seconds).toFixed(1)} s${pf.served_id ? ` · served id ${esc(pf.served_id)}` : ""}${pf.cached ? ` · reused from ${pf.checked ? esc(when(pf.checked)) : "earlier"} (no new request; Check again asks again)` : ""}</div>` : ""}</div>`;

  // ------------------------------------------------ tab: Models
  function modelsHTML() {
    if (P.models == null) return `<div class="small muted">Loading models…</div>`;
    const ro = !P.canEdit ? `<div class="ev-note small">Read-only: adding models, keys and evals needs <code>civitas serve</code> with launching on, opened on this computer.</div>` : "";
    if (P.modelsErr) return `${ro}<div class="emptytab">${esc(P.modelsErr)}</div>`;
    const dis = P.canEdit ? "" : "disabled";
    const envName = (P.env && P.env.env) || "";
    const rows = P.models.map(m => {
      const ks = m.key_status || {}, lt = m.latest;
      const iv = lt && lt.interval && lt.interval.lo != null ? ` <span class="muted">[${sgn(lt.interval.lo)}, ${sgn(lt.interval.hi)}]</span>` : "";
      const pf = P.pf[m.key];
      return `<tr>
        <td><b>${esc(m.key)}</b>${m.notes ? `<div class="ev-sub2">${esc(m.notes)}</div>` : ""}</td>
        <td>${esc(m.provider)}</td><td class="mono small">${esc(m.served_id || m.id || m.key)}</td>
        <td title="${esc(ks.env || "")}">${ks.set ? `<span class="good">✓</span>` : `<span class="badc">✗</span>`} <span class="muted small">${esc(ks.env || "")}</span></td>
        <td>${lt && lt.score != null ? `<b class="${lt.score > 0 ? "good" : lt.score < 0 ? "badc" : ""}">${sgn(lt.score)}</b>${iv}<div class="ev-sub2">vs scripted · ${esc(lt.env || envName)}</div>` : `<span class="muted">—</span>`}</td>
        <td class="ev-acts"><button class="btn" data-test="${esc(m.key)}" ${dis} title="One real request with the fleet's tools: it must come back as a tool call">${pf === "busy" ? "Testing…" : "Test"}</button><button class="btn primary" data-eval="${esc(m.key)}" ${dis}>Evaluate</button><button class="btn" data-edit="${esc(m.key)}" ${dis}>Edit</button><button class="btn danger" data-del="${esc(m.key)}" ${dis} aria-label="Delete ${esc(m.key)}">Delete</button></td></tr>
        ${pf && pf !== "busy" ? `<tr class="ev-pfrow"><td colspan="6">${pf.error && !pf.steps ? `<span class="bad small">${esc(pf.error)}</span>` : pfHTML(pf)}</td></tr>` : ""}`;
    }).join("");
    const keyRows = P.providers.filter(p => p.env).map(p => `<tr>
        <td><b>${esc(p.name)}</b>${p.builtin ? "" : ` <span class="muted small">(registry)</span>`}</td><td class="mono small">${esc(p.env)}</td>
        <td>${p.set ? `<span class="good">✓ set</span>` : `<span class="badc">✗ not set</span>`}</td>
        <td class="ev-acts">${P.keyFor === p.env ? `<input type="password" id="kfVal" placeholder="paste the key" autocomplete="off" spellcheck="false"><button class="btn primary" id="kfGo">Save</button><button class="btn" id="kfX">Cancel</button>`
          : `<button class="btn" data-key="${esc(p.env)}" ${dis}>${p.set ? "Replace key" : "Set API key"}</button>`}</td></tr>`).join("");
    return `${ro}<div class="ev-card">
        <div class="ev-head"><h4>Models <span class="muted">${P.models.length}</span></h4><div class="grow"></div><button class="btn primary" id="evAdd" ${dis}>＋ Add model</button></div>
        ${P.models.length ? `<div class="ev-scroll"><table class="ev-t"><thead><tr><th>Model</th><th>Provider</th><th>Served id</th><th>API key</th><th>Latest eval</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`
          : `<div class="emptytab">No models in the registry yet (configs/models.toml). Add one to evaluate it.</div>`}
      </div>
      <div class="ev-card">
        <div class="ev-head"><h4>API keys</h4><span class="muted small">kept in ./.env on this computer; the page only learns whether each is set</span></div>
        <div class="ev-scroll"><table class="ev-t"><thead><tr><th>Provider</th><th>Variable</th><th>Status</th><th></th></tr></thead><tbody>${keyRows}</tbody></table></div>
      </div>`;
  }
  function bindModels() {
    if ($("evAdd")) $("evAdd").onclick = () => modelSheet();
    const byKey = (k) => P.models.find(m => m.key === k);
    body.querySelectorAll("[data-edit]").forEach(b => b.onclick = () => modelSheet(byKey(b.dataset.edit)));
    body.querySelectorAll("[data-eval]").forEach(b => b.onclick = () => evaluateSheet(byKey(b.dataset.eval)));
    body.querySelectorAll("[data-del]").forEach(b => b.onclick = async () => {
      if (!confirm(`Delete ${b.dataset.del} from the registry? Its evals stay in runs/.`)) return;
      const r = await req("/api/models/delete", { key: b.dataset.del });
      if (!r.ok) toast(`Could not delete: ${esc(errText(r))}`, true); else toast(`Deleted <b>${esc(b.dataset.del)}</b>.`);
      loadModels();
    });
    body.querySelectorAll("[data-test]").forEach(b => b.onclick = async () => {
      const k = b.dataset.test; P.pf[k] = "busy"; render();
      const r = await req("/api/models/preflight", { key: k });
      P.pf[k] = r.ok ? r.data : { ok: false, error: errText(r) }; render();
    });
    body.querySelectorAll("[data-key]").forEach(b => b.onclick = () => { P.keyFor = b.dataset.key; render(); $("kfVal") && $("kfVal").focus(); });
    if ($("kfX")) $("kfX").onclick = () => { P.keyFor = null; render(); };
    if ($("kfGo")) {
      const save = async () => {
        const value = $("kfVal").value.trim(); if (!value) return;
        const env = P.keyFor;
        $("kfVal").value = "";                       // the key leaves the page with the request and nowhere else
        const r = await req("/api/keys", { env, value });
        if (!r.ok) { toast(`Could not set ${esc(env)}: ${esc(errText(r))}`, true); return; }
        P.keyFor = null; toast(`${esc(env)} is set.`); loadModels();
      };
      $("kfGo").onclick = save; $("kfVal").onkeydown = (e) => { if (e.key === "Enter") save(); if (e.key === "Escape") { P.keyFor = null; render(); } };
    }
  }

  // ------------------------------------------------ tab: Evals
  function verdictOf(dir) { const row = ((P.lb && P.lb.rows) || []).find(r => r.eval_dir === dir); return row ? row.verdict : null; }
  const ciText = (a) => !a || a.mean == null ? "—" : `${sgn(a.mean)}${a.lo != null ? ` [${sgn(a.lo)}, ${sgn(a.hi)}]` : a.std != null ? ` ± ${a.std.toFixed(1)}` : ""}`;
  function runningHTML(x) {
    const p = P.progress[x.ref] || {}, j = x.job;
    if (failedStart(j)) return `<div class="ev-run"><div class="ev-run-h"><b>${esc(x.name)}</b><span class="muted small">${esc(x.model || "")}</span><div class="grow"></div><span class="pill failed">failed to start</span></div><div class="small bad" style="white-space:pre-wrap">${esc(j.error || p.error || "the eval process ended before planning its runs")}</div></div>`;
    const starting = j && !j.eval_json;
    const total = p.total || 0, done = p.finished || 0, failed = p.failed || 0;
    const pct = total ? 100 * done / total : 0, fpct = total ? 100 * failed / total : 0;
    const chips = (p.runs || []).map(r => `<span class="ev-chip s-${esc(r.status)}" title="${esc(r.run_id)}: ${esc(r.status)}">s${esc(r.seed)}${r.months ? ` <span class="muted">${r.month ?? 0}/${r.months}</span>` : ""}</span>`).join("");
    return `<div class="ev-run">
      <div class="ev-run-h"><b>${esc(x.name)}</b><span class="muted small">${esc(x.model || "")}${x.ev && x.ev.suite ? ` · ${esc(x.ev.suite.name || "")}` : ""}${x.ref.startsWith("eval_") ? ` · ${esc(x.ref)}` : ""}</span><div class="grow"></div>
        ${j && j.stopping ? `<span class="pill pausing">stopping</span>` : `<span class="pill running">${starting ? "starting" : "running"}</span>`}
        ${P.canEdit ? `<button class="btn danger" data-stop="${esc(x.ref)}" ${j && j.stopping ? "disabled" : ""} title="Stop the eval and its runs (civitas eval --resume continues it later)">■ Stop</button>` : ""}</div>
      ${starting ? `<div class="ev-st busy small"><span class="ev-spin"></span>Starting: preflight, environment check, planning the runs…</div>` : `
      <div class="ev-bar" role="progressbar" aria-valuemin="0" aria-valuemax="${total}" aria-valuenow="${done}"><span style="width:${pct}%"></span><span class="f" style="left:${pct}%;width:${fpct}%"></span></div>
      <div class="ev-run-n small"><span><b>${done}</b> of ${total || "?"} runs finished${failed ? `, <span class="badc">${failed} failed</span>` : ""}${p.running ? `, ${p.running} running` : ""}</span><span>${tok(p.tokens)} tokens so far</span><span>${p.eta_s != null ? `about ${dur(p.eta_s)} left` : "time left: —"}</span></div>
      ${chips ? `<div class="ev-chips">${chips}</div>` : ""}`}
    </div>`;
  }
  function evalsHTML() {
    if (P.evals == null) return `<div class="small muted">Loading evals…</div>`;
    const live = running(), liveDirs = new Set(live.map(x => x.ref));
    const done = P.evals.filter(e => !liveDirs.has(e.dir) && e.status !== "running");
    const rows = done.map(e => {
      const a = e.aggregate || {}, v = verdictOf(e.dir), open = P.open.has(e.dir);
      const st = e.status || "?";
      return `<div class="ev-item ${open ? "open" : ""}">
        <div class="ev-item-h">
          <button class="ev-tog" data-open="${esc(e.dir)}" aria-expanded="${open}"><span class="caret">${open ? "▾" : "▸"}</span><b>${esc(e.name || e.dir)}</b></button>
          <span class="muted small">${esc(e.model || "?")} · ${esc((e.suite && e.suite.name) || "?")} · ${e.runs.length} runs · ${esc(when(e.created))}</span>
          <span class="pill ${esc(st === "finished" ? "finished" : st === "partial" ? "incomplete" : "stopped")}">${esc(st)}</span>
          <div class="grow"></div>
          <span class="ev-score" title="the city's change since the hand-over minus the scripted yardstick's, mean and 95% interval over units">vs scripted <b class="${a.city_delta && a.city_delta.mean > 0 ? "good" : a.city_delta && a.city_delta.mean < 0 ? "badc" : ""}">${ciText(a.city_delta)}</b></span>
          ${onCompareEval ? `<button class="btn" data-cmp="${esc(e.dir)}" title="Open its runs in the Compare view">Compare runs</button>` : ""}
        </div>
        ${v ? `<div class="ev-verdict small">${esc(v)}</div>` : ""}
        ${open ? `<div class="ev-card-md">${P.cards[e.dir] == null ? `<div class="small muted">Loading the scorecard…</div>` : P.cards[e.dir].error ? `<div class="small muted">${esc(P.cards[e.dir].error)}</div>` : md(P.cards[e.dir].markdown)}</div>` : ""}
      </div>`;
    }).join("");
    return `<div class="ev-card"><div class="ev-head"><h4>Running <span class="muted">${live.length}</span></h4></div>
        ${live.length ? live.map(runningHTML).join("") : `<div class="small muted">No eval is running. Start one from the Models tab (Evaluate).</div>`}</div>
      <div class="ev-card"><div class="ev-head"><h4>Finished <span class="muted">${done.length}</span></h4><div class="grow"></div><span class="muted small">newest first · the verdict is the paired comparison with the suite's baseline model</span></div>
        ${done.length ? rows : `<div class="small muted">No finished evals yet.</div>`}</div>`;
  }
  function bindEvals() {
    body.querySelectorAll("[data-stop]").forEach(b => b.onclick = async () => {
      if (!confirm("Stop this eval and its runs? What finished is kept; civitas eval --resume <folder> reruns the rest.")) return;
      b.disabled = true;
      const r = await req("/api/evals/stop", { dir: b.dataset.stop });
      if (!r.ok) toast(`Could not stop it: ${esc(errText(r))}`, true); else toast("Stopping the eval and its runs…");
      const jb = await req("/api/evals/jobs"); if (jb.ok) P.jobs = jb.data; render();
    });
    body.querySelectorAll("[data-open]").forEach(b => b.onclick = async () => {
      const d = b.dataset.open;
      if (P.open.has(d)) { P.open.delete(d); render(); return; }
      P.open.add(d); render();
      if (P.cards[d] == null || P.cards[d].error) {
        const r = await req(`/api/evals/scorecard?dir=${encodeURIComponent(d)}`);
        P.cards[d] = r.ok ? r.data : { error: r.status === 404 ? "No scorecard yet: civitas eval --report <folder> writes it." : errText(r) };
        render();
      }
    });
    body.querySelectorAll("[data-cmp]").forEach(b => b.onclick = () => { modal.hidden = true; onCompareEval(b.dataset.cmp); });
  }

  // ------------------------------------------------ tab: Leaderboard
  function ciBar(v, lo, hi) {
    if (!v || v.mean == null) return "";
    const x = (t) => `${(100 * (t - lo) / (hi - lo)).toFixed(2)}%`;
    const a = v.lo ?? v.mean, b = v.hi ?? v.mean;
    const cls = a > 0 ? "up" : b < 0 ? "down" : "";
    return `<div class="ev-ci ${cls}" aria-hidden="true"><span class="z" style="left:${x(0)}"></span><span class="r" style="left:${x(a)};width:calc(${x(b)} - ${x(a)})"></span><span class="d" style="left:${x(v.mean)}"></span></div>`;
  }
  function boardHTML() {
    const names = [...new Set([...(P.envs || []).map(e => e.name), ...((P.lb && P.lb.envs) || []), ...(P.lbEnv ? [P.lbEnv] : [])])];
    const pick = `<label class="ev-inline">Version <select id="lbEnv">${names.length ? names.map(n => `<option ${n === P.lbEnv ? "selected" : ""}>${esc(n)}</option>`).join("") : `<option value="">—</option>`}</select></label>`;
    if (P.lbErr) return `<div class="ev-tools">${pick}</div><div class="emptytab">${esc(P.lbErr)}</div>`;
    if (!P.lb) return `<div class="ev-tools">${pick}</div><div class="small muted">Loading the leaderboard…</div>`;
    const rows = (P.lb.rows || []).slice().sort((a, b) => ((b.vs_yardstick || {}).mean ?? -1e9) - ((a.vs_yardstick || {}).mean ?? -1e9));
    const vals = rows.flatMap(r => { const v = r.vs_yardstick || {}; return [v.lo, v.hi, v.mean].filter(x => x != null); });
    let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
    const pad = Math.max(0.5, (hi - lo) * 0.08); lo -= pad; hi += pad;
    const base = P.lb.baseline;
    const tr = rows.map((r, i) => {
      const v = r.vs_yardstick || {}, s = r.score || {};
      return `<tr>
        <td class="ev-rank">${i + 1}</td>
        <td><b>${esc(r.model)}</b>${r.model === base ? ` <span class="ev-tag">baseline</span>` : ""}<div class="ev-sub2 mono">${esc(r.eval_dir || "")}</div></td>
        <td class="ev-cicell"><div class="ev-cinum"><b class="${v.lo > 0 ? "good" : v.hi < 0 ? "badc" : ""}">${sgn(v.mean)}</b>${v.lo != null ? ` <span class="muted">[${sgn(v.lo)}, ${sgn(v.hi)}]</span>` : ""}</div>${ciBar(v, lo, hi)}</td>
        <td class="small">${sgn((r.pop_term || {}).mean)} <span class="muted">/</span> ${sgn((r.quality || {}).mean)}</td>
        <td>${s.mean != null ? s.mean.toFixed(1) : "—"}${s.lo != null ? ` <span class="muted small">[${s.lo.toFixed(1)}, ${s.hi.toFixed(1)}]</span>` : ""}</td>
        <td>${r.n_units ?? "—"}</td>
        <td>${r.teamwork != null ? r.teamwork.toFixed(1) : "—"}</td>
        <td class="small">${r.failed_per_turn != null ? r.failed_per_turn.toFixed(2) : "—"} <span class="muted">/</span> ${r.refused_per_turn != null ? r.refused_per_turn.toFixed(2) : "—"}</td>
        <td class="ev-vd">${r.model === base ? `<span class="muted">—</span>` : r.verdict ? esc(r.verdict) : `<span class="muted">no paired baseline</span>`}</td>
        <td>${tok(r.tokens)}</td><td>${usd(r.cost)}</td></tr>`;
    }).join("");
    const skipped = (P.lb.skipped || []).map(s => `<li><span class="mono">${esc(s.eval_dir)}</span>: ${esc(s.reason)}</li>`).join("");
    return `<div class="ev-tools">${pick}<button class="btn" id="lbRe">↻ Refresh</button><div class="grow"></div><span class="muted small">the latest finished, non-drifted eval of each model on this version</span></div>
      <div class="ev-card">${rows.length ? `<div class="ev-scroll"><table class="ev-t ev-board"><thead><tr><th>#</th><th>Model</th><th>vs scripted <span class="muted">(95% interval)</span></th><th title="vs scripted split: the population term (50 x sqrt(population / target)) / the quality terms (the rest)">pop / quality</th><th>City score</th><th>Units</th><th title="mean Teamwork score over the runs">Teamwork</th><th title="failed actions / guard refusals per model turn">failed / refused per turn</th><th>vs baseline${base ? ` (${esc(base)})` : ""}</th><th>Tokens</th><th>Cost</th></tr></thead><tbody>${tr}</tbody></table></div>
        <div class="muted small ev-foot">vs scripted: the city's change since the hand-over minus the scripted agents' change from the same hand-over, mean over units with its 95% interval (the bar; the line is 0). An interval clear of 0 beats or trails scripted beyond seed luck. pop / quality: how much of it came from the population term and how much from the quality terms.</div>`
        : `<div class="emptytab">No ranked evals on ${esc(P.lb.env || "this version")} yet.</div>`}</div>
      ${skipped ? `<div class="ev-card ev-skipped"><div class="ev-head"><h4>Not ranked</h4></div><ul class="small muted">${skipped}</ul></div>` : ""}`;
  }
  function bindBoard() {
    if ($("lbEnv")) $("lbEnv").onchange = (e) => { P.lbEnv = e.target.value || null; P.lb = null; render(); loadBoard(); };
    if ($("lbRe")) $("lbRe").onclick = () => loadBoard();
  }

  // ------------------------------------------------ render
  function render() {
    if (!shown()) return;
    document.querySelectorAll("#evTabs [data-t]").forEach(b => { b.classList.toggle("on", b.dataset.t === P.tab); b.setAttribute("aria-selected", b.dataset.t === P.tab); });
    const live = running().length;
    const evTab = document.querySelector('#evTabs [data-t="evals"]'); if (evTab) evTab.innerHTML = `Evals${live ? ` <span class="ev-dot">${live}</span>` : ""}`;
    const focus = document.activeElement && document.activeElement.id;   // keep a key being typed
    const val = $("kfVal") ? $("kfVal").value : null;
    const y = body.scrollTop;
    if (P.tab === "models") { body.innerHTML = modelsHTML(); bindModels(); }
    else if (P.tab === "evals") { body.innerHTML = evalsHTML(); bindEvals(); }
    else { body.innerHTML = boardHTML(); bindBoard(); }
    body.scrollTop = y;
    if (val != null && $("kfVal")) $("kfVal").value = val;
    if (focus && $(focus) && body.contains($(focus))) $(focus).focus();
  }

  document.querySelectorAll("#evTabs [data-t]").forEach(b => b.onclick = () => {
    P.tab = b.dataset.t; saveTab(); render();
    if (P.tab === "evals") loadEvals(); else if (P.tab === "board") { loadEnvs().then(render); loadBoard(); } else loadModels();
  });

  async function open() {
    modal.hidden = false;
    closeSheet();
    render(); renderEnv();
    loadEnv(); loadModels(); loadEvals(); loadEnvs(); loadBoard();
    poll();
  }
  return { open, md };
}
