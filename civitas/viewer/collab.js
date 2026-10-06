// Team view: the departments as a living network. The mayor sits in the centre and the departments on a ring around
// it. Edges glow with recent traffic, and each month's events play out: requests fly as comets, reports as pulses, the
// mayor's money as a stream of coins, builds as bursts, and a thread closing flashes its edge (green answered, amber
// partly, red when nobody answered). Data: /api/runs/<id>/collab (bench/collab.py). One class, used by the Team tab
// and the full-screen Team view.
const GOLD = "#ffd34d", OK = "#45d483", PART = "#ffc857", BAD = "#ff6b5e";
const DECAY = 0.82;                 // edge activity kept per month
const HISTORY = 12;                 // months of activity an edge shows
const fmtK = (v) => { const a = Math.abs(v); return "$" + (a >= 1e4 ? (a / 1e3).toFixed(1).replace(/\.0$/, "") + "k" : Math.round(a).toLocaleString()); };
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

function hexA(hex, a) {
  const h = hex.replace("#", ""), n = parseInt(h.length === 3 ? h.split("").map(c => c + c).join("") : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

export class CollabGraph {
  constructor(canvas, { colorOf, onPick, big = false } = {}) {
    this.cv = canvas; this.g = canvas.getContext("2d"); this.colorOf = colorOf || (() => "#ccc");
    this.onPick = onPick; this.big = big;
    this.data = null; this.month = -1; this.funds = {}; this.answers = {}; this.filter = () => true;
    this.parts = []; this.floats = []; this.flash = new Map(); this.pulse = new Map(); this.act = new Map();
    this.pending = new Map(); this.hover = null; this.mouse = null; this.selected = null; this.active = true;
    this._loop = this._loop.bind(this);
    canvas.addEventListener("mousemove", (e) => { this.mouse = this._xy(e); this.hover = this._hit(this.mouse); canvas.style.cursor = this.hover ? "pointer" : "default"; });
    canvas.addEventListener("mouseleave", () => { this.mouse = null; this.hover = null; });
    canvas.addEventListener("click", (e) => {
      const h = this._hit(this._xy(e));
      this.selected = h && this.selected && this._key(h) === this._key(this.selected) ? null : h;
      if (this.onPick) this.onPick(this.selected);
    });
    this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(canvas);
    this.resize();
    requestAnimationFrame(this._loop);
  }

  destroy() { this.dead = true; this.ro.disconnect(); }
  setActive(on) { this.active = on; }

  setData(data) {
    this.data = data;
    this.roles = data.roles || [];
    this._layout();
    const m = this.month; this.month = -1;
    if (m >= 0) this.setMonth(m, { funds: this.funds });
  }

  // month: the turn month on show (a frame's month - 1). animate: launch that month's events (and those since prev).
  // answers: role -> {s, a, n}, its responsiveness to that month (score or null under 3 cases, answered, settled).
  setMonth(month, { funds = {}, answers = {}, animate = false, budgetMs = 1600, filter } = {}) {
    if (filter) this.filter = filter;
    this.funds = funds; this.answers = answers;
    const prev = this.month;
    this.month = month;
    if (!this.data) return;
    this._activity(month);
    if (animate && prev >= 0 && month > prev && month - prev <= 12) this._launch(prev, month, budgetMs);
    else if (!animate) { this.parts = []; this.floats = []; }
  }

  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = this.cv.clientWidth || 340, h = this.big ? (this.cv.clientHeight || 600) : Math.round(Math.min(420, Math.max(280, w * 0.92)));
    if (!this.big) this.cv.style.height = `${h}px`;
    this.W = w; this.H = h; this.dpr = dpr;
    this.cv.width = Math.round(w * dpr); this.cv.height = Math.round(h * dpr);
    this._layout();
  }

  // ------------------------------------------------------------ model
  _layout() {
    if (!this.roles) return;
    const cx = this.W / 2, cy = this.H / 2 + (this.big ? 0 : 4), R = Math.min(this.W, this.H) * (this.big ? 0.36 : 0.35);
    const ring = this.roles.filter(r => r !== "mayor");
    this.pos = {};
    if (this.roles.includes("mayor")) this.pos.mayor = [cx, cy];
    ring.forEach((r, i) => { const a = -Math.PI / 2 + i * 2 * Math.PI / ring.length; this.pos[r] = [cx + R * Math.cos(a), cy + R * Math.sin(a)]; });
    this.scale = Math.min(1.45, Math.min(this.W, this.H) / 380);    // big view: bigger ring, not giant labels
  }

  _activity(m) {
    this.act = new Map(); this.pending = new Map();
    for (const e of this.data.events) {
      if (e.month > m) break;
      if (e.month <= m - HISTORY || (e.kind !== "message" && e.kind !== "money") || e.to === "all" || e.from === e.to) continue;
      if (!this.pos[e.from] || !this.pos[e.to]) continue;
      const w = e.kind === "money" ? 0.6 + Math.log10(1 + e.amount) / 3 : e.ask ? 1.2 : 0.6;
      const k = `${e.from}>${e.to}`;
      this.act.set(k, (this.act.get(k) || 0) + w * Math.pow(DECAY, m - e.month));
    }
    for (const t of this.data.threads) {
      if (t.month > m || m - t.month > this.data.window) continue;
      if (!(t.response && t.response.month <= m)) this.pending.set(t.from, (this.pending.get(t.from) || 0) + 1);
    }
  }

  _launch(prev, month, budget) {
    const now = performance.now();
    const evs = this.data.events.filter(e => e.month > prev && e.month <= month && this.filter(e.from) && (e.to === "all" || this.filter(e.to) || e.from === e.to));
    const dur = Math.max(420, Math.min(1400, budget * 0.55));
    const gap = evs.length > 1 ? Math.max(0, budget - dur) / (evs.length - 1) : 0;
    evs.forEach((e, i) => {
      const t0 = now + i * Math.min(gap, 260);
      const cA = this.colorOf(e.from);
      if (e.kind === "message") {
        if (e.to === "all" || !this.pos[e.to]) this.parts.push({ type: "ring", a: e.from, t0, dur: dur * 1.2, color: cA });
        else this.parts.push({ type: e.ask ? "comet" : "pulse", a: e.from, b: e.to, t0, dur, color: cA, text: e.ask && e.amount ? fmtK(e.amount) + "?" : null });
      } else if (e.kind === "money" && this.pos[e.to]) {
        const n = Math.max(2, Math.min(8, Math.round(Math.log10(1 + e.amount) * 1.6)));
        for (let j = 0; j < n; j++) this.parts.push({ type: "coin", a: e.from, b: e.to, t0: t0 + j * 70, dur: dur * 1.05, color: e.take_back ? "#b8b2a0" : GOLD, last: j === n - 1, text: (e.take_back ? "−" : "+") + fmtK(e.amount) });
      } else if (e.kind === "build") {
        this.parts.push({ type: "burst", a: e.from, t0: t0 + dur * 0.3, dur: dur * 1.3, color: cA, text: e.text });
      }
    });
    // threads closing this step: answers flash their edge, unanswered ones fade out in red
    for (const t of this.data.threads) {
      if (!this.filter(t.from)) continue;
      const r = t.response;
      if (r && r.month > prev && r.month <= month) this.flash.set(`${t.from}>${t.to}`, { color: t.status === "partial" ? PART : OK, t0: now + dur, dur: 900 });
      const end = t.month + this.data.window;
      if (t.status === "unanswered" && end > prev && end <= month) this.flash.set(`${t.from}>${t.to}`, { color: BAD, t0: now, dur: 1100, cross: true });
    }
  }

  // ------------------------------------------------------------ geometry
  _ctrl(a, b) {
    const [ax, ay] = this.pos[a], [bx, by] = this.pos[b];
    return [(ax + bx) / 2 - (by - ay) * 0.16, (ay + by) / 2 + (bx - ax) * 0.16];
  }
  _at(a, b, t) {
    const [ax, ay] = this.pos[a], [bx, by] = this.pos[b], [cx, cy] = this._ctrl(a, b), u = 1 - t;
    return [u * u * ax + 2 * u * t * cx + t * t * bx, u * u * ay + 2 * u * t * cy + t * t * by];
  }
  _radius(r) {
    const s = this.scale, f = r === "mayor" ? (this.funds.__treasury || 0) / 4 : this.funds[r] || 0;
    return (r === "mayor" ? 21 : 14) * s + Math.min(12 * s, Math.sqrt(Math.max(0, f)) / 9 * s);
  }
  _xy(e) { const b = this.cv.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top]; }
  _key(h) { return h ? (h.role || `${h.from}>${h.to}`) : ""; }
  _hit([x, y]) {
    if (!this.pos) return null;
    for (const r of this.roles) { const p = this.pos[r]; if (p && Math.hypot(p[0] - x, p[1] - y) <= this._radius(r) + 4) return { role: r }; }
    let best = null, bd = 7;
    for (const k of this.act.keys()) {
      const [a, b] = k.split(">");
      for (let i = 1; i < 20; i++) { const [px, py] = this._at(a, b, i / 20), d = Math.hypot(px - x, py - y); if (d < bd) { bd = d; best = { from: a, to: b }; } }
    }
    return best;
  }

  // ------------------------------------------------------------ drawing
  _loop(now) {
    if (this.dead) return;
    if (this.active && this.pos && this.W) this._draw(now);
    requestAnimationFrame(this._loop);
  }

  _draw(now) {
    const g = this.g, W = this.W, H = this.H, s = this.scale;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const bg = g.createRadialGradient(W / 2, H / 2, 10, W / 2, H / 2, Math.max(W, H) * 0.7);
    bg.addColorStop(0, "#131a2b"); bg.addColorStop(1, "#070a12");
    g.fillStyle = bg; g.fillRect(0, 0, W, H);
    // the ring the departments sit on, and faint spokes to the mayor
    const c = this.pos.mayor || [W / 2, H / 2], ring = this.roles.filter(r => r !== "mayor");
    if (ring.length) {
      const R = Math.hypot(this.pos[ring[0]][0] - c[0], this.pos[ring[0]][1] - c[1]);
      g.strokeStyle = "rgba(140,160,210,0.08)"; g.lineWidth = 1; g.setLineDash([2, 6]);
      g.beginPath(); g.arc(c[0], c[1], R, 0, 2 * Math.PI); g.stroke(); g.setLineDash([]);
    }
    if (!this.data) { this._label("loading the team's messages…", W / 2, H / 2, "#9aa3ad", 12); return; }
    const sel = this.selected, focus = (a, b) => !sel || (sel.role ? a === sel.role || b === sel.role : sel.from === a && sel.to === b);
    // edges: width and brightness by recent traffic, dashes flowing from sender to receiver
    for (const [k, v] of this.act) {
      const [a, b] = k.split(">");
      const on = this.filter(a) && this.filter(b) && focus(a, b), w = Math.min(7, 0.8 + v * 1.1) * s, al = on ? Math.min(0.85, 0.18 + v * 0.22) : 0.05;
      const [ax, ay] = this.pos[a], [bx, by] = this.pos[b], [cx, cy] = this._ctrl(a, b);
      const grad = g.createLinearGradient(ax, ay, bx, by);
      grad.addColorStop(0, hexA(this.colorOf(a), al)); grad.addColorStop(1, hexA(this.colorOf(b), al));
      g.strokeStyle = grad; g.lineWidth = w; g.lineCap = "round";
      g.beginPath(); g.moveTo(ax, ay); g.quadraticCurveTo(cx, cy, bx, by); g.stroke();
      if (on) {
        g.setLineDash([1.5 * s, 11 * s]); g.lineDashOffset = -now / 40; g.strokeStyle = `rgba(255,255,255,${Math.min(0.5, 0.12 + v * 0.1)})`;
        g.lineWidth = Math.max(1, w * 0.5); g.beginPath(); g.moveTo(ax, ay); g.quadraticCurveTo(cx, cy, bx, by); g.stroke(); g.setLineDash([]);
      }
    }
    // thread flashes on their edge
    for (const [k, f] of this.flash) {
      const p = (now - f.t0) / f.dur;
      if (p > 1) { this.flash.delete(k); continue; }
      if (p < 0) continue;
      const [a, b] = k.split(">");
      if (!this.pos[a] || !this.pos[b]) continue;
      const [ax, ay] = this.pos[a], [bx, by] = this.pos[b], [cx, cy] = this._ctrl(a, b), al = Math.sin(p * Math.PI);
      g.save(); g.shadowColor = f.color; g.shadowBlur = 16 * s; g.strokeStyle = hexA(f.color, 0.9 * al); g.lineWidth = 3.5 * s;
      g.beginPath(); g.moveTo(ax, ay); g.quadraticCurveTo(cx, cy, bx, by); g.stroke(); g.restore();
      if (f.cross) { const [mx, my] = this._at(a, b, 0.5); this._label("✕", mx, my + 5 * s, hexA(BAD, al), 16 * s, true); }
    }
    // travelling particles
    this.parts = this.parts.filter(q => {
      const p = (now - q.t0) / q.dur;
      if (p < 0) return true;
      if (p >= 1) { this._arrive(q, now); return false; }
      this._part(q, p, now);
      return true;
    });
    // nodes
    for (const r of this.roles) this._node(r, now, this.filter(r) && focus(r, r) || (sel && !sel.role && (sel.from === r || sel.to === r)) || (sel && sel.role === r));
    // floating labels
    this.floats = this.floats.filter(f => {
      const p = (now - f.t0) / f.dur;
      if (p >= 1) return false;
      if (p >= 0) this._label(f.text, f.x, f.y - p * 22 * s, hexA(f.color, 1 - p * p), (f.size || 11) * s, true);
      return true;
    });
    if (this.hover && this.mouse) this._tip(this.hover, this.mouse);
  }

  _part(q, p, now) {
    const g = this.g, s = this.scale;
    if (q.type === "ring") {
      const [x, y] = this.pos[q.a];
      g.strokeStyle = hexA(q.color, 0.6 * (1 - p)); g.lineWidth = 2 * s;
      g.beginPath(); g.arc(x, y, this._radius(q.a) + p * 70 * s, 0, 2 * Math.PI); g.stroke();
      return;
    }
    if (q.type === "burst") {
      const [x, y] = this.pos[q.a], r0 = this._radius(q.a);
      for (let i = 0; i < 2; i++) {
        const pp = Math.max(0, p - i * 0.18);
        g.strokeStyle = hexA(q.color, 0.8 * (1 - pp)); g.lineWidth = (2.5 - i) * s;
        g.beginPath(); g.arc(x, y, r0 + ease(pp) * 30 * s, 0, 2 * Math.PI); g.stroke();
      }
      for (let i = 0; i < 8; i++) {                     // sparks
        const a = i * Math.PI / 4 + q.t0 % 1, d = r0 + ease(p) * 36 * s;
        g.fillStyle = hexA(q.color, 1 - p); g.beginPath(); g.arc(x + Math.cos(a) * d, y + Math.sin(a) * d, 1.6 * s, 0, 2 * Math.PI); g.fill();
      }
      return;
    }
    const t = ease(p), [x, y] = this._at(q.a, q.b, t);
    if (q.type === "comet") {
      for (let i = 8; i >= 1; i--) {                     // the trail
        const [tx, ty] = this._at(q.a, q.b, Math.max(0, t - i * 0.025));
        g.fillStyle = hexA(q.color, 0.5 * (1 - i / 9)); g.beginPath(); g.arc(tx, ty, (3.4 - i * 0.3) * s, 0, 2 * Math.PI); g.fill();
      }
      g.save(); g.shadowColor = q.color; g.shadowBlur = 14 * s; g.fillStyle = "#fff";
      g.beginPath(); g.arc(x, y, 3.6 * s, 0, 2 * Math.PI); g.fill(); g.restore();
      g.fillStyle = q.color; g.beginPath(); g.arc(x, y, 2.4 * s, 0, 2 * Math.PI); g.fill();
      if (q.text && t > 0.15 && t < 0.7) this._label(q.text, x, y - 8 * s, hexA("#ffffff", 0.9 * Math.sin((t - 0.15) / 0.55 * Math.PI)), 9.5 * s, true);
    } else if (q.type === "pulse") {
      g.fillStyle = hexA(q.color, 0.75); g.beginPath(); g.arc(x, y, 2.2 * s, 0, 2 * Math.PI); g.fill();
    } else if (q.type === "coin") {
      g.save(); g.shadowColor = q.color; g.shadowBlur = 10 * s;
      g.fillStyle = q.color; g.beginPath(); g.ellipse(x, y, 4 * s, 4 * s * Math.abs(Math.cos(now / 90 + q.t0)), 0, 0, 2 * Math.PI); g.fill();
      g.restore();
    }
  }

  _arrive(q, now) {
    if (q.type === "coin" && q.last) {
      const [x, y] = this.pos[q.b];
      this.floats.push({ text: q.text, x, y: y - this._radius(q.b) - 8 * this.scale, color: q.color, t0: now, dur: 1400, size: 12 });
      this.pulse.set(q.b, { t0: now, color: q.color });
    } else if (q.type === "comet") this.pulse.set(q.b, { t0: now, color: q.color });
    else if (q.type === "burst" && q.text) {
      const [x, y] = this.pos[q.a];
      this.floats.push({ text: q.text, x, y: y + this._radius(q.a) + 40 * this.scale, color: q.color, t0: now - 200, dur: 1500, size: 10 });
    }
  }

  _node(r, now, on) {
    const g = this.g, s = this.scale, [x, y] = this.pos[r], rad = this._radius(r), col = this.colorOf(r);
    const breathe = 1 + 0.03 * Math.sin(now / 700 + x);
    g.save(); g.globalAlpha = on ? 1 : 0.25;
    g.shadowColor = col; g.shadowBlur = (r === "mayor" ? 26 : 18) * s;
    const grad = g.createRadialGradient(x - rad * 0.3, y - rad * 0.35, rad * 0.1, x, y, rad * breathe);
    grad.addColorStop(0, "#ffffff"); grad.addColorStop(0.25, col); grad.addColorStop(1, hexA(col, 0.35));
    g.fillStyle = grad; g.beginPath(); g.arc(x, y, rad * breathe, 0, 2 * Math.PI); g.fill();
    g.shadowBlur = 0;
    const pu = this.pulse.get(r);                       // something just arrived
    if (pu) {
      const p = (now - pu.t0) / 700;
      if (p > 1) this.pulse.delete(r);
      else { g.strokeStyle = hexA(pu.color, 1 - p); g.lineWidth = 2 * s; g.beginPath(); g.arc(x, y, rad + p * 14 * s, 0, 2 * Math.PI); g.stroke(); }
    }
    const open = this.pending.get(r) || 0;               // open requests: an arc round the node, one fifth each
    if (open) {
      g.strokeStyle = hexA(PART, 0.9); g.lineWidth = 3 * s; g.lineCap = "round";
      g.beginPath(); g.arc(x, y, rad + 5 * s, -Math.PI / 2, -Math.PI / 2 + Math.min(1, open / 5) * 2 * Math.PI); g.stroke();
      this._label(String(open), x + rad * 0.85, y - rad * 0.85, "#111", 9 * s, true, PART);
    }
    g.restore();
    g.globalAlpha = on ? 1 : 0.35;
    this._label(r, x, y + rad + 13 * s, "#e9ecef", 11 * s, true);
    const f = r === "mayor" ? this.funds.__treasury : this.funds[r];
    if (f != null) this._label(fmtK(f), x, y + rad + 25 * s, "#9aa3ad", 9.5 * s, true);
    const an = this.answers[r];                          // its responsiveness as the one asked
    if (an) this._label(an.s != null ? `answers ${Math.round(an.s)}%` : `answers ${an.a} of ${an.n}`, x, y + rad + (f != null ? 37 : 25) * s,
      an.s == null ? "#9aa3ad" : an.s >= 70 ? "#7fdcaa" : an.s >= 40 ? "#ffd98a" : "#ff9a90", 9.5 * s, true);
    g.globalAlpha = 1;
  }

  _label(text, x, y, color, size, center, badge) {
    const g = this.g;
    g.font = `600 ${Math.round(size)}px Inter, system-ui, sans-serif`;
    g.textAlign = center ? "center" : "left"; g.textBaseline = "alphabetic";
    if (badge) {
      const w = g.measureText(text).width + 6;
      g.fillStyle = badge; g.beginPath(); g.roundRect(x - w / 2, y - size, w, size + 3, 4); g.fill();
      g.fillStyle = color; g.fillText(text, x, y); return;
    }
    g.lineWidth = 3; g.strokeStyle = "rgba(7,10,18,0.8)"; g.strokeText(text, x, y);
    g.fillStyle = color; g.fillText(text, x, y);
  }

  _tip(h, [mx, my]) {
    const st = (this.data.stats || {})[h.role] || {}, an = h.role && this.answers[h.role];
    const lines = h.role
      ? [h.role, `asked ${st.asked || 0} · answered ${st.answered || 0}`, `asked of it ${st.asked_of || 0} · it answered ${st.answered_by || 0}`,
         ...(an ? [`answered ${an.a} of ${an.n} settled requests to it${an.s != null ? ` (${Math.round(an.s)}%)` : ""}`] : []),
         `money in ${fmtK(st.money_in || 0)}${st.money_out ? ` · out ${fmtK(st.money_out)}` : ""}`, `${st.builds || 0} builds · ${st.messages || 0} messages`]
      : this._edgeLines(h);
    const g = this.g, pad = 7;
    g.font = "12px Inter, system-ui, sans-serif";
    const w = Math.max(...lines.map(l => g.measureText(l).width)) + pad * 2, hh = lines.length * 16 + pad * 2 - 4;
    const x = Math.min(this.W - w - 4, mx + 12), y = Math.min(this.H - hh - 4, my + 12);
    g.fillStyle = "rgba(12,16,26,0.94)"; g.strokeStyle = "rgba(255,255,255,0.12)"; g.lineWidth = 1;
    g.beginPath(); g.roundRect(x, y, w, hh, 7); g.fill(); g.stroke();
    g.textAlign = "left";
    lines.forEach((l, i) => { g.fillStyle = i === 0 ? this.colorOf(h.role || h.from) : "#d5dae0"; g.font = `${i === 0 ? "600 " : ""}12px Inter, system-ui, sans-serif`; g.fillText(l, x + pad, y + pad + 11 + i * 16); });
  }

  _edgeLines({ from, to }) {
    const th = this.data.threads.filter(t => t.from === from && t.to === to && t.month <= this.month);
    const money = this.data.events.filter(e => e.kind === "money" && e.from === from && e.to === to && e.month <= this.month).reduce((a, e) => a + e.amount, 0);
    const msgs = this.data.events.filter(e => e.kind === "message" && e.from === from && e.to === to && e.month <= this.month).length;
    const done = th.filter(t => t.response && t.response.month <= this.month).length;
    return [`${from} → ${to}`, `${msgs} messages · ${th.length} requests (${done} answered)`, money ? `${fmtK(money)} moved` : "no money moved", "click to see the threads"];
  }
}

// Hand-off beams over the city: planning's road or freed lot glows, an arc carries a light from it to the building
// another department put there, and the building bursts with "road → school". Drawn on an overlay canvas above
// either city renderer through its tileToScreen(); recent hand-offs stay as faint arcs while paused.
export class BeamLayer {
  constructor(canvas, { colorOf, project }) {
    this.cv = canvas; this.g = canvas.getContext("2d"); this.colorOf = colorOf; this.project = project;
    this.data = null; this.month = -1; this.on = true; this.filter = () => true; this.live = [];
    this._loop = this._loop.bind(this);
    this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(canvas.parentElement);
    this.resize(); requestAnimationFrame(this._loop);
  }
  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1), el = this.cv.parentElement;
    this.W = el.clientWidth; this.H = el.clientHeight; this.dpr = dpr;
    this.cv.width = Math.round(this.W * dpr); this.cv.height = Math.round(this.H * dpr);
  }
  setData(data) { this.data = data; }
  setMonth(month, { animate = false, budgetMs = 1600, filter, on = true } = {}) {
    this.on = on; if (filter) this.filter = filter;
    const prev = this.month; this.month = month;
    if (!this.data || !on) { this.live = []; return; }
    if (animate && prev >= 0 && month > prev && month - prev <= 12) {
      const now = performance.now(), hs = this.data.handoffs.filter(h => h.month_to > prev && h.month_to <= month && this.filter(h.to));
      const dur = Math.max(900, Math.min(2200, budgetMs * 0.8));
      hs.forEach((h, i) => this.live.push({ h, t0: now + i * Math.min(350, budgetMs * 0.25), dur }));
    } else if (!animate) this.live = [];
  }
  _arc(a, b) {                                     // screen-space arc lifted above the city
    const d = Math.hypot(b[0] - a[0], b[1] - a[1]);
    return [(a[0] + b[0]) / 2, Math.min(a[1], b[1]) - Math.min(170, 30 + d * 0.45)];
  }
  _pt(a, c, b, t) { const u = 1 - t; return [u * u * a[0] + 2 * u * t * c[0] + t * t * b[0], u * u * a[1] + 2 * u * t * c[1] + t * t * b[1]]; }
  _from(h) { const r = h.rect_from; return this.project((r[0] + r[2]) / 2, (r[1] + r[3]) / 2); }   // the opener's middle
  _rect(r) {                                       // a tile rectangle's corners on screen
    const P = [[r[0] - 0.5, r[1] - 0.5], [r[2] + 0.5, r[1] - 0.5], [r[2] + 0.5, r[3] + 0.5], [r[0] - 0.5, r[3] + 0.5]].map(([x, y]) => this.project(x, y));
    return P.every(Boolean) ? P : null;
  }
  _loop(now) {
    if (this.ro === null) return;
    const g = this.g;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0); g.clearRect(0, 0, this.W, this.H);
    if (this.on && this.data) {
      const m = this.month, win = this.data.window || 6;
      // resting arcs: the hand-offs of the last window, fading with age
      for (const h of this.data.handoffs) {
        if (h.month_to > m || m - h.month_to >= win || !this.filter(h.to)) continue;
        if (this.live.some(l => l.h === h)) continue;
        const a = this._from(h), b = this.project(...h.tile_to);
        if (!a || !b) continue;
        const c = this._arc(a, b), al = 0.75 * (1 - (m - h.month_to) / win) + 0.2;
        g.strokeStyle = `rgba(7,10,18,${0.45 * al})`; g.lineWidth = 4.5;           // a dark underlay reads on snow too
        g.beginPath(); g.moveTo(a[0], a[1]); g.quadraticCurveTo(c[0], c[1], b[0], b[1]); g.stroke();
        const gr = g.createLinearGradient(a[0], a[1], b[0], b[1]);
        gr.addColorStop(0, hexA(this.colorOf(h.from), al)); gr.addColorStop(1, hexA(this.colorOf(h.to), al));
        g.strokeStyle = gr; g.lineWidth = 2.2;
        g.beginPath(); g.moveTo(a[0], a[1]); g.quadraticCurveTo(c[0], c[1], b[0], b[1]); g.stroke();
        g.setLineDash([2, 14]); g.lineDashOffset = -now / 45; g.strokeStyle = `rgba(255,255,255,${0.8 * al})`; g.lineWidth = 2.2;
        g.beginPath(); g.moveTo(a[0], a[1]); g.quadraticCurveTo(c[0], c[1], b[0], b[1]); g.stroke(); g.setLineDash([]);
        g.save(); g.shadowColor = this.colorOf(h.to); g.shadowBlur = 10; g.fillStyle = hexA(this.colorOf(h.to), Math.min(1, al + 0.2));
        g.beginPath(); g.arc(b[0], b[1], 3.5, 0, 2 * Math.PI); g.fill(); g.restore();
      }
      this.live = this.live.filter(l => { const p = (now - l.t0) / l.dur; if (p >= 1) return false; if (p >= 0) this._beam(l.h, p); return true; });
    }
    requestAnimationFrame(this._loop);
  }
  _beam(h, p) {
    const g = this.g, a = this._from(h), b = this.project(...h.tile_to);
    if (!a || !b) return;
    const cf = this.colorOf(h.from), ct = this.colorOf(h.to), c = this._arc(a, b);
    const rc = this._rect(h.rect_from);
    if (rc && p < 0.6) {                           // 1. the road or lot planning opened glows
      const al = Math.sin(Math.min(1, p / 0.6) * Math.PI);
      g.save(); g.shadowColor = cf; g.shadowBlur = 18; g.strokeStyle = hexA(cf, 0.9 * al); g.fillStyle = hexA(cf, 0.18 * al); g.lineWidth = 2.5;
      g.beginPath(); rc.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.closePath(); g.fill(); g.stroke(); g.restore();
    }
    const q = Math.min(1, Math.max(0, (p - 0.15) / 0.55)), t = ease(q);
    if (q > 0) {                                   // 2. the light travels, leaving the arc behind it
      const gr = g.createLinearGradient(a[0], a[1], b[0], b[1]);
      gr.addColorStop(0, hexA(cf, 0.85)); gr.addColorStop(1, hexA(ct, 0.85));
      g.strokeStyle = gr; g.lineWidth = 2.5; g.beginPath(); g.moveTo(a[0], a[1]);
      for (let i = 1; i <= 24; i++) { const [x, y] = this._pt(a, c, b, t * i / 24); g.lineTo(x, y); }
      g.stroke();
      if (q < 1) {
        const [x, y] = this._pt(a, c, b, t);
        g.save(); g.shadowColor = "#fff"; g.shadowBlur = 20; g.fillStyle = "#fff"; g.beginPath(); g.arc(x, y, 4.5, 0, 2 * Math.PI); g.fill(); g.restore();
      }
    }
    if (p > 0.68) {                                // 3. the new building bursts, and the hand-off is named
      const r = (p - 0.68) / 0.32;
      g.strokeStyle = hexA(ct, 1 - r); g.lineWidth = 3;
      g.beginPath(); g.arc(b[0], b[1], 6 + ease(r) * 34, 0, 2 * Math.PI); g.stroke();
      g.font = "600 13px Inter, system-ui, sans-serif"; g.textAlign = "center";
      const text = `${h.what_from} → ${h.what.replace(/^built /, "")}`, y = b[1] - 16 - r * 14;
      g.lineWidth = 4; g.strokeStyle = `rgba(7,10,18,${0.85 * (1 - r * r)})`; g.strokeText(text, b[0], y);
      g.fillStyle = hexA(ct, 1 - r * r); g.fillText(text, b[0], y);
    }
  }
}

// ================================================================ region view
// The Team tab's Region view: the cities at their place in the region (node size by population), the mayors' deals as
// edges between them (solid with light flowing from seller to buyer while active, dashed while offered, faint for a
// year once ended, labelled with amount, resource and price), and the messages between cities as comets like the
// department graph's. Each month a deal's payment travels as coins from buyer to seller; an offer flies as a contract,
// an acceptance flashes the edge green, a cancellation red. Data: /api/runs/<id>/collab?region=1 (bench/collab.py).
const CITY_C = ["#4fc3f7", "#ff8a65", "#c5a3ff", "#7bd88f", "#f06292", "#ffd54f"];
export const RES_C = { power: "#ffd34d", water: "#5ab8ff" };
const ENDED_SHOWN = 12;                 // months an ended deal stays on the graph
const DONE = ["met", "partial", "acted", "replied"];
export const cityColor = (cities, c) => { const i = (cities || []).indexOf(c); return CITY_C[(i < 0 ? 0 : i) % CITY_C.length]; };
// a deal's state in month m: null before its offer, "offered", "active" or "ended"
export function dealState(d, m) {
  if (d.offered == null || m < d.offered) return null;
  if (d.ended != null && d.ended <= m) return "ended";
  return d.accepted != null && d.accepted <= m ? "active" : "offered";
}
// months a deal was paid up to month m: from the month it was accepted to the month it ran out (a cancelled deal
// was last paid the month before)
export function dealPaid(d, m) {
  if (d.accepted == null || m < d.accepted) return 0;
  const last = d.ended == null ? m : Math.min(m, d.end === "expired" ? d.ended : d.ended - 1);
  return Math.max(0, last - d.accepted + 1);
}
export function dealEnd(d) {
  if (d.end === "expired") return d.accepted == null ? "offer expired" : "ran out";
  if (d.accepted == null) return d.ended_by === d.proposer ? "withdrawn" : "declined";
  return `cancelled${d.ended_by ? ` by ${d.ended_by}` : ""}`;
}
// one city's dealings with the others up to month m
export function cityTally(d, c, m) {
  const msgs = d.events.filter(e => e.kind === "message" && e.month <= m);
  const asks = d.threads.filter(t => t.kind !== "deal" && t.month <= m);
  const ok = (t) => t.response && t.response.month <= m && DONE.includes(t.status);
  const deals = d.deals.filter(x => dealState(x, m));
  const sum = (xs) => xs.reduce((a, x) => a + x.monthly * dealPaid(x, m), 0);
  return { sent: msgs.filter(e => e.from_city === c).length, received: msgs.filter(e => e.to_city === c).length,
    asks_out: asks.filter(t => t.from_city === c).length, asks_out_ok: asks.filter(t => t.from_city === c && ok(t)).length,
    asks_in: asks.filter(t => t.to_city === c).length, asks_in_ok: asks.filter(t => t.to_city === c && ok(t)).length,
    proposed: deals.filter(x => x.proposer === c).length,
    accepted: deals.filter(x => x.partner === c && x.accepted != null && x.accepted <= m).length,
    active: deals.filter(x => (x.seller === c || x.buyer === c) && dealState(x, m) === "active").length,
    sold: sum(deals.filter(x => x.seller === c)), bought: sum(deals.filter(x => x.buyer === c)) };
}

export class RegionGraph {
  constructor(canvas, { colorOf, onPick, onOpen, big = false } = {}) {
    this.cv = canvas; this.g = canvas.getContext("2d"); this.colorOf = colorOf || (() => "#ccc");
    this.onPick = onPick; this.onOpen = onOpen; this.big = big;
    this.data = null; this.layout = {}; this.month = -1; this.info = {}; this.links = [];
    this.parts = []; this.floats = []; this.flash = new Map(); this.pulse = new Map(); this.act = new Map(); this.pending = new Map();
    this.hover = null; this.mouse = null; this.selected = null; this.active = true;
    this._loop = this._loop.bind(this);
    canvas.addEventListener("mousemove", (e) => { this.mouse = this._xy(e); this.hover = this._hit(this.mouse); canvas.style.cursor = this.hover ? "pointer" : "default"; });
    canvas.addEventListener("mouseleave", () => { this.mouse = null; this.hover = null; });
    canvas.addEventListener("click", (e) => {
      const h = this._hit(this._xy(e));
      this.selected = h && this.selected && this._key(h) === this._key(this.selected) ? null : h;
      if (this.onPick) this.onPick(this.selected);
    });
    canvas.addEventListener("dblclick", (e) => { const h = this._hit(this._xy(e)); if (h && h.city && this.onOpen) this.onOpen(h.city); });
    this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(canvas);
    this.resize();
    requestAnimationFrame(this._loop);
  }

  destroy() { this.dead = true; this.ro.disconnect(); }
  setActive(on) { this.active = on; }

  setData(data, layout) {
    this.data = data; this.cities = data.cities || []; this.layout = layout || {};
    this.resize();
    const m = this.month; this.month = -1;
    if (m >= 0) this.setMonth(m, { info: this.info, links: this.links });
  }

  // month: the turn month on show. info: {city: {pop, treasury}}. links: the frame's road / rail / boat / air links.
  setMonth(month, { info = {}, links = [], animate = false, budgetMs = 1600 } = {}) {
    this.info = info; this.links = links;
    const prev = this.month;
    this.month = month;
    if (!this.data) return;
    this._activity(month);
    if (animate && prev >= 0 && month > prev && month - prev <= 12) this._launch(prev, month, budgetMs);
    else if (!animate) { this.parts = []; this.floats = []; }
  }

  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1), rows = this._grid().rows;
    const w = this.cv.clientWidth || 340, h = this.big ? (this.cv.clientHeight || 600) : Math.round(Math.min(480, Math.max(260, rows * 210)));
    if (!this.big) this.cv.style.height = `${h}px`;
    this.W = w; this.H = h; this.dpr = dpr;
    this.cv.width = Math.round(w * dpr); this.cv.height = Math.round(h * dpr);
    this._layout();
  }

  // ------------------------------------------------------------ model
  _grid() {
    const P = (this.cities || []).map(c => this.layout[c] || [0, 0]);
    if (!P.length) return { x0: 0, y0: 0, cols: 1, rows: 1 };
    const xs = P.map(p => p[0]), ys = P.map(p => p[1]), x0 = Math.min(...xs), y0 = Math.min(...ys);
    return { x0, y0, cols: Math.max(...xs) - x0 + 1, rows: Math.max(...ys) - y0 + 1 };
  }
  _layout() {
    if (!this.cities) return;
    const { x0, y0, cols, rows } = this._grid(), px = 70, py = 40;
    this.scale = Math.min(1.45, Math.min(this.W, this.H) / 330);
    this.pos = {};
    this.cities.forEach((c, i) => {
      const [lx, ly] = this.layout[c] || [i, 0];
      this.pos[c] = [px + (lx - x0 + 0.5) * (this.W - 2 * px) / cols, py + (ly - y0 + 0.5) * (this.H - 2 * py) / rows];
    });
  }
  _activity(m) {
    this.act = new Map(); this.pending = new Map();
    for (const e of this.data.events) {
      if (e.month > m) break;
      if (e.kind !== "message" || e.deal || e.month <= m - HISTORY || !this.pos[e.from_city] || !this.pos[e.to_city]) continue;
      const k = `${e.from_city}>${e.to_city}`;
      this.act.set(k, (this.act.get(k) || 0) + (e.ask ? 1.2 : 0.6) * Math.pow(DECAY, m - e.month));
    }
    for (const t of this.data.threads) {
      if (t.month > m || (t.kind !== "deal" && m - t.month > this.data.window)) continue;
      if (t.kind === "deal") { const d = this._deal(t.deal); if (!d || dealState(d, m) !== "offered") continue; }
      if (!(t.response && t.response.month <= m)) this.pending.set(t.from_city, (this.pending.get(t.from_city) || 0) + 1);
    }
  }
  _deal(id) { return this.data.deals.find(d => d.id === id); }
  // the deals on show in month m, each with its pair (sorted city names) and its bend k (several deals fan out)
  _shown(m) {
    const out = [], byPair = new Map();
    for (const d of this.data.deals) {
      const st = dealState(d, m);
      if (!st || (st === "ended" && m - d.ended >= ENDED_SHOWN) || !this.pos[d.seller] || !this.pos[d.buyer]) continue;
      const pair = [d.seller, d.buyer].sort(), key = pair.join("|");
      if (!byPair.has(key)) byPair.set(key, []);
      byPair.get(key).push({ d, st, p: pair[0], q: pair[1] });
    }
    for (const list of byPair.values()) list.forEach((x, i) => { x.k = (i - (list.length - 1) / 2) * 0.2; x.lt = Math.max(0.25, Math.min(0.75, 0.5 + (i - (list.length - 1) / 2) * 0.24)); out.push(x); });
    return out;
  }
  _launch(prev, month, budget) {
    const now = performance.now(), shown = new Map(this._shown(month).map(x => [x.d.id, x]));
    const evs = this.data.events.filter(e => e.month > prev && e.month <= month && !(e.kind === "message" && e.deal));
    const dur = Math.max(480, Math.min(1500, budget * 0.6));
    const gap = evs.length > 1 ? Math.max(0, budget - dur) / (evs.length - 1) : 0;
    evs.forEach((e, i) => {
      const t0 = now + i * Math.min(gap, 300), a = e.from_city, b = e.to_city;
      if (!this.pos[a] || !this.pos[b]) return;
      if (e.kind === "message") { this.parts.push({ type: e.ask ? "comet" : "pulse", a, b, k: 0.2, t0, dur, color: this.colorOf(e.from), text: e.ask ? "?" : null }); return; }
      const col = RES_C[e.resource] || GOLD;
      if (e.what === "offered") this.parts.push({ type: "doc", a, b, k: 0.2, t0, dur: dur * 1.2, color: col, text: `${Math.round(e.amount)} ${e.resource}?` });
      else this.flash.set(e.deal, { color: e.what === "accepted" ? OK : e.what === "cancelled" ? BAD : "#8a93a0", t0: t0 + dur * 0.4, dur: 1000,
        cross: e.what === "cancelled", text: e.what === "accepted" ? `deal #${e.deal} ✓` : e.what === "cancelled" ? `#${e.deal} cancelled`
          : (this._deal(e.deal) || {}).accepted == null ? `#${e.deal} offer expired` : `#${e.deal} ran out` });
    });
    // this step's payments: coins from buyer to seller along the deal's edge
    for (const d of this.data.deals) {
      const months = dealPaid(d, month) - dealPaid(d, prev), x = shown.get(d.id) || this._shown(Math.min(month, d.ended ?? month)).find(y => y.d.id === d.id);
      if (months <= 0 || !x) continue;
      const amt = d.monthly * months, n = Math.max(2, Math.min(7, Math.round(Math.log10(1 + amt) * 1.5)));
      for (let j = 0; j < n; j++) this.parts.push({ type: "coin", a: d.buyer, b: d.seller, deal: x, t0: now + dur * 0.5 + j * 80, dur: dur * 1.1, color: GOLD, last: j === n - 1, text: "+" + fmtK(amt) });
    }
  }

  // ------------------------------------------------------------ geometry
  _ctrl(a, b, k) {
    const [ax, ay] = this.pos[a], [bx, by] = this.pos[b];
    return [(ax + bx) / 2 - (by - ay) * k, (ay + by) / 2 + (bx - ax) * k];
  }
  _at(a, b, k, t) {
    const [ax, ay] = this.pos[a], [bx, by] = this.pos[b], [cx, cy] = this._ctrl(a, b, k), u = 1 - t;
    return [u * u * ax + 2 * u * t * cx + t * t * bx, u * u * ay + 2 * u * t * cy + t * t * by];
  }
  _path(a, b, k) { const g = this.g, [ax, ay] = this.pos[a], [bx, by] = this.pos[b], [cx, cy] = this._ctrl(a, b, k); g.beginPath(); g.moveTo(ax, ay); g.quadraticCurveTo(cx, cy, bx, by); }
  // a point on a deal's edge, t running from a to b whichever way the edge is drawn
  _onDeal(x, a, t) { return a === x.p ? this._at(x.p, x.q, x.k, t) : this._at(x.p, x.q, x.k, 1 - t); }
  _radius(c) { const s = this.scale, pop = (this.info[c] || {}).pop || 0; return 20 * s + Math.min(16 * s, Math.sqrt(pop) / 11 * s); }
  _xy(e) { const b = this.cv.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top]; }
  _key(h) { return h ? (h.city || (h.deal != null ? `#${h.deal}` : `${h.from}>${h.to}`)) : ""; }
  _hit([x, y]) {
    if (!this.pos || !this.data) return null;
    for (const c of this.cities) { const p = this.pos[c]; if (p && Math.hypot(p[0] - x, p[1] - y) <= this._radius(c) + 4) return { city: c }; }
    let best = null, bd = 8;
    for (const s of this._shown(this.month)) for (let i = 1; i < 24; i++) {
      const [px, py] = this._at(s.p, s.q, s.k, i / 24), d = Math.hypot(px - x, py - y);
      if (d < bd) { bd = d; best = { deal: s.d.id }; }
    }
    if (best) return best;
    for (const k of this.act.keys()) {
      const [a, b] = k.split(">");
      for (let i = 1; i < 20; i++) { const [px, py] = this._at(a, b, 0.2, i / 20), d = Math.hypot(px - x, py - y); if (d < bd) { bd = d; best = { from: a, to: b }; } }
    }
    return best;
  }

  // ------------------------------------------------------------ drawing
  _loop(now) {
    if (this.dead) return;
    if (this.active && this.pos && this.W) this._draw(now);
    requestAnimationFrame(this._loop);
  }
  _draw(now) {
    const g = this.g, W = this.W, H = this.H, s = this.scale;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const bg = g.createRadialGradient(W / 2, H / 2, 10, W / 2, H / 2, Math.max(W, H) * 0.7);
    bg.addColorStop(0, "#131a2b"); bg.addColorStop(1, "#070a12");
    g.fillStyle = bg; g.fillRect(0, 0, W, H);
    if (!this.data) { this._label("loading the region…", W / 2, H / 2, "#9aa3ad", 12, true); return; }
    const m = this.month, sel = this.selected;
    const focusCity = (c) => !sel || (sel.city ? sel.city === c : sel.deal != null ? (() => { const d = this._deal(sel.deal); return d && (d.seller === c || d.buyer === c); })() : sel.from === c || sel.to === c);
    // the links the cities built between them (road, rail, boat, air): faint ground lines, named while no deal is drawn
    const seen = new Set(), shown = this._shown(m), dealPairs = new Set(shown.map(x => `${x.p}|${x.q}`));
    for (const l of this.links) {
      const key = [l.a, l.b].sort().join("|");
      if (seen.has(key) || !this.pos[l.a] || !this.pos[l.b]) continue;
      seen.add(key);
      const kinds = [...new Set(this.links.filter(x => [x.a, x.b].sort().join("|") === key).map(x => x.kind))];
      const [ax, ay] = this.pos[l.a], [bx, by] = this.pos[l.b];
      g.strokeStyle = "rgba(140,160,210,0.16)"; g.lineWidth = 6 * s; g.lineCap = "round";
      g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.stroke();
      const text = kinds.join(" · ");
      g.font = `600 ${Math.round(9 * s)}px Inter, system-ui, sans-serif`;       // named only where it fits between the halos
      const room = Math.hypot(bx - ax, by - ay) - this._radius(l.a) - this._radius(l.b) - 24 * s;
      if (!dealPairs.has(key) && g.measureText(text).width <= room) this._label(text, (ax + bx) / 2, (ay + by) / 2 + 3 * s, "rgba(160,175,205,0.6)", 9 * s, true);
    }
    // message traffic: arcs between the cities, brighter with recent messages
    for (const [k, v] of this.act) {
      const [a, b] = k.split(">"), on = focusCity(a) || focusCity(b);
      const w = Math.min(6, 0.8 + v) * s, al = on ? Math.min(0.75, 0.16 + v * 0.2) : 0.05;
      const grad = g.createLinearGradient(...this.pos[a], ...this.pos[b]);
      grad.addColorStop(0, hexA(cityColor(this.cities, a), al)); grad.addColorStop(1, hexA(cityColor(this.cities, b), al));
      g.strokeStyle = grad; g.lineWidth = w; g.lineCap = "round"; this._path(a, b, 0.2); g.stroke();
    }
    // deals: active ones glow and flow from seller to buyer, offers are dashed, ended ones fade
    for (const x of shown) this._dealEdge(x, now, !sel || (sel.deal != null ? sel.deal === x.d.id : focusCity(x.p) && focusCity(x.q)));
    for (const [id, f] of this.flash) {
      const p = (now - f.t0) / f.dur, x = shown.find(y => y.d.id === id);
      if (p > 1 || !x) { if (p > 1) this.flash.delete(id); continue; }
      if (p < 0) continue;
      const al = Math.sin(p * Math.PI);
      g.save(); g.shadowColor = f.color; g.shadowBlur = 18 * s; g.strokeStyle = hexA(f.color, 0.95 * al); g.lineWidth = 4.5 * s;
      this._path(x.p, x.q, x.k); g.stroke(); g.restore();
      const [mx, my] = this._at(x.p, x.q, x.k, 0.5);
      if (f.cross) this._label("✕", mx, my + 6 * s, hexA(BAD, al), 18 * s, true);
      if (f.text && !f.floated && (f.floated = true)) this.floats.push({ text: f.text, x: mx, y: my - 16 * s, color: f.color, t0: now, dur: 1600, size: 11 });
    }
    this.parts = this.parts.filter(q => {
      const p = (now - q.t0) / q.dur;
      if (p < 0) return true;
      if (p >= 1) { this._arrive(q, now); return false; }
      this._part(q, p, now);
      return true;
    });
    for (const c of this.cities) this._node(c, now, focusCity(c));
    this.floats = this.floats.filter(f => {
      const p = (now - f.t0) / f.dur;
      if (p >= 1) return false;
      if (p >= 0) this._label(f.text, f.x, f.y - p * 22 * s, hexA(f.color, 1 - p * p), (f.size || 11) * s, true);
      return true;
    });
    if (!this.data.deals.length && !this.data.events.length) this._label("no deals or messages between the cities yet", W / 2, H - 12 * s, "#6f7884", 10.5 * s, true);
    if (this.hover && this.mouse) this._tip(this.hover, this.mouse);
  }
  _dealEdge(x, now, on) {
    const g = this.g, s = this.scale, d = x.d, col = RES_C[d.resource] || "#d5dae0";
    const age = x.st === "ended" ? (this.month - d.ended) / ENDED_SHOWN : 0, dim = on ? 1 : 0.18;
    g.save(); g.lineCap = "round";
    if (x.st === "active") {
      g.shadowColor = col; g.shadowBlur = 12 * s; g.strokeStyle = hexA(col, 0.85 * dim); g.lineWidth = 3.4 * s;
      this._path(x.p, x.q, x.k); g.stroke(); g.shadowBlur = 0;
      g.setLineDash([2 * s, 10 * s]); g.lineDashOffset = (d.seller === x.p ? -1 : 1) * now / 28;
      g.strokeStyle = `rgba(255,255,255,${0.75 * dim})`; g.lineWidth = 2.2 * s; this._path(x.p, x.q, x.k); g.stroke();
    } else if (x.st === "offered") {
      g.setLineDash([7 * s, 5 * s]); g.lineDashOffset = (d.proposer === x.p ? -1 : 1) * now / 60;
      g.strokeStyle = hexA(col, 0.7 * dim); g.lineWidth = 1.8 * s; this._path(x.p, x.q, x.k); g.stroke();
    } else {
      g.setLineDash([1.5 * s, 5 * s]); g.strokeStyle = hexA("#8a93a0", 0.55 * (1 - age) * dim); g.lineWidth = 1.6 * s;
      this._path(x.p, x.q, x.k); g.stroke();
    }
    g.restore();
    // the label on the edge (several deals between two cities stack theirs along it): what, how much, at what price,
    // and where it stands
    const [mx, my] = this._at(x.p, x.q, x.k, x.lt), left = d.months - dealPaid(d, this.month);
    const main = `${Math.round(d.amount)} ${d.resource} · $${d.price}`;
    const sub = x.st === "active" ? `#${d.id} ${d.seller}→${d.buyer} · ${left} mo left` : x.st === "offered" ? `#${d.id} offer from ${d.proposer}` : `#${d.id} ${dealEnd(d)}`;
    g.globalAlpha = (x.st === "ended" ? 0.6 * (1 - age) + 0.2 : 1) * (on ? 1 : 0.35);
    this._label(main, mx, my - 2 * s, x.st === "ended" ? "#9aa3ad" : col, 10.5 * s, true);
    this._label(sub, mx, my + 10 * s, "#9aa3ad", 9 * s, true);
    g.globalAlpha = 1;
  }
  _part(q, p, now) {
    const g = this.g, s = this.scale, t = ease(p);
    if (q.type === "coin") {
      const [x, y] = this._onDeal(q.deal, q.a, t);
      g.save(); g.shadowColor = q.color; g.shadowBlur = 10 * s; g.fillStyle = q.color;
      g.beginPath(); g.ellipse(x, y, 4 * s, 4 * s * Math.abs(Math.cos(now / 90 + q.t0)), 0, 0, 2 * Math.PI); g.fill(); g.restore();
      return;
    }
    const [x, y] = this._at(q.a, q.b, q.k, t);
    if (q.type === "comet") {
      for (let i = 8; i >= 1; i--) {
        const [tx, ty] = this._at(q.a, q.b, q.k, Math.max(0, t - i * 0.025));
        g.fillStyle = hexA(q.color, 0.5 * (1 - i / 9)); g.beginPath(); g.arc(tx, ty, (3.6 - i * 0.3) * s, 0, 2 * Math.PI); g.fill();
      }
      g.save(); g.shadowColor = q.color; g.shadowBlur = 14 * s; g.fillStyle = "#fff"; g.beginPath(); g.arc(x, y, 3.8 * s, 0, 2 * Math.PI); g.fill(); g.restore();
      g.fillStyle = q.color; g.beginPath(); g.arc(x, y, 2.5 * s, 0, 2 * Math.PI); g.fill();
      if (q.text && t > 0.15 && t < 0.75) this._label(q.text, x, y - 8 * s, "#ffffff", 10 * s, true);
    } else if (q.type === "pulse") {
      g.fillStyle = hexA(q.color, 0.8); g.beginPath(); g.arc(x, y, 2.4 * s, 0, 2 * Math.PI); g.fill();
    } else if (q.type === "doc") {                       // a contract on its way: a little sheet with the offer
      g.save(); g.translate(x, y); g.rotate(Math.sin(now / 160) * 0.15);
      g.shadowColor = q.color; g.shadowBlur = 12 * s; g.fillStyle = "#f4f1e8";
      g.beginPath(); g.roundRect(-5 * s, -6.5 * s, 10 * s, 13 * s, 1.5 * s); g.fill(); g.shadowBlur = 0;
      g.fillStyle = q.color; for (let i = 0; i < 3; i++) g.fillRect(-3 * s, (-3.5 + i * 3) * s, 6 * s, 1.1 * s);
      g.restore();
      if (t > 0.1 && t < 0.85) this._label(q.text, x, y - 11 * s, q.color, 9.5 * s, true);
    }
  }
  _arrive(q, now) {
    if (q.type === "coin" && q.last) {
      const [x, y] = this.pos[q.b];
      this.floats.push({ text: q.text, x, y: y - this._radius(q.b) - 8 * this.scale, color: q.color, t0: now, dur: 1400, size: 12 });
      this.pulse.set(q.b, { t0: now, color: q.color });
    } else if (q.type === "comet" || q.type === "doc") this.pulse.set(q.b, { t0: now, color: q.color });
  }
  _node(c, now, on) {
    const g = this.g, s = this.scale, [x, y] = this.pos[c], rad = this._radius(c), col = cityColor(this.cities, c);
    const breathe = 1 + 0.025 * Math.sin(now / 800 + x);
    g.save(); g.globalAlpha = on ? 1 : 0.3;
    g.shadowColor = col; g.shadowBlur = 24 * s;
    const grad = g.createRadialGradient(x - rad * 0.3, y - rad * 0.35, rad * 0.1, x, y, rad * breathe);
    grad.addColorStop(0, "#ffffff"); grad.addColorStop(0.25, col); grad.addColorStop(1, hexA(col, 0.3));
    g.fillStyle = grad; g.beginPath(); g.arc(x, y, rad * breathe, 0, 2 * Math.PI); g.fill();
    g.shadowBlur = 0;
    g.strokeStyle = hexA(col, 0.5); g.lineWidth = 1.2 * s; g.setLineDash([2 * s, 4 * s]);      // the city limits
    g.beginPath(); g.arc(x, y, rad + 9 * s, 0, 2 * Math.PI); g.stroke(); g.setLineDash([]);
    const pu = this.pulse.get(c);
    if (pu) {
      const p = (now - pu.t0) / 700;
      if (p > 1) this.pulse.delete(c);
      else { g.strokeStyle = hexA(pu.color, 1 - p); g.lineWidth = 2 * s; g.beginPath(); g.arc(x, y, rad + p * 16 * s, 0, 2 * Math.PI); g.stroke(); }
    }
    const open = this.pending.get(c) || 0;
    if (open) {
      g.strokeStyle = hexA(PART, 0.9); g.lineWidth = 3 * s; g.lineCap = "round";
      g.beginPath(); g.arc(x, y, rad + 5 * s, -Math.PI / 2, -Math.PI / 2 + Math.min(1, open / 4) * 2 * Math.PI); g.stroke();
      this._label(String(open), x + rad * 0.85, y - rad * 0.85, "#111", 9 * s, true, PART);
    }
    g.restore();
    g.globalAlpha = on ? 1 : 0.4;
    // names go below the node, or above it for a city with a neighbour right below (labels clear of the edges)
    const lay = this.layout[c] || [0, 0], above = this.cities.some(o => o !== c && (this.layout[o] || [])[0] === lay[0] && (this.layout[o] || [])[1] > lay[1]);
    const ny = above ? y - rad - 26 * s : y + rad + 22 * s, py = above ? y - rad - 14 * s : y + rad + 34 * s;
    this._label(c.toUpperCase(), x, ny, "#e9ecef", 11.5 * s, true);
    const inf = this.info[c] || {};
    if (inf.pop != null) this._label(`${inf.pop.toLocaleString()} people${inf.treasury != null ? ` · ${fmtK(inf.treasury)}` : ""}`, x, py, "#9aa3ad", 9.5 * s, true);
    g.globalAlpha = 1;
  }
  _label(text, x, y, color, size, center, badge) { CollabGraph.prototype._label.call(this, text, x, y, color, size, center, badge); }
  _tip(h, [mx, my]) {
    const m = this.month, d = h.deal != null ? this._deal(h.deal) : null;
    let lines, color;
    if (h.city) {
      const t = cityTally(this.data, h.city, m), st = (this.data.stats || {})[h.city] || {};
      color = cityColor(this.cities, h.city);
      lines = [h.city.toUpperCase(), `messages to other cities ${t.sent} · from them ${t.received}`,
        `its asks answered ${t.asks_out_ok}/${t.asks_out} · it answered ${t.asks_in_ok}/${t.asks_in}`,
        `deals proposed ${t.proposed} · accepted ${t.accepted} · active ${t.active}`,
        `sold ${fmtK(t.sold)} · bought ${fmtK(t.bought)}`,
        `its team: ${st.answered || 0}/${st.requests || 0} requests answered (whole run)`, "double-click to open its team"];
    } else if (d) {
      const st = dealState(d, m), paid = dealPaid(d, m);
      color = RES_C[d.resource] || "#fff";
      lines = [`deal #${d.id} · ${d.resource}`, `${d.seller} sells ${Math.round(d.amount)} to ${d.buyer} at $${d.price}/unit/month`,
        `${fmtK(d.monthly)}/month for ${d.months} months · proposed by ${d.proposer} (m${d.offered})`,
        st === "active" ? `active since m${d.accepted} · ${d.months - paid} months left` : st === "offered" ? `waiting for ${d.partner} to accept` : `${dealEnd(d)} (m${d.ended})`,
        `paid so far ${fmtK(d.monthly * paid)}`];
    } else {
      const msgs = this.data.events.filter(e => e.kind === "message" && e.from_city === h.from && e.to_city === h.to && e.month <= m);
      const asks = this.data.threads.filter(t => t.kind !== "deal" && t.from_city === h.from && t.to_city === h.to && t.month <= m);
      color = cityColor(this.cities, h.from);
      lines = [`${h.from} → ${h.to}`, `${msgs.length} messages · ${asks.length} asks (${asks.filter(t => t.response && t.response.month <= m).length} answered)`, "click to see the threads"];
    }
    const g = this.g, pad = 7;
    g.font = "12px Inter, system-ui, sans-serif";
    const w = Math.max(...lines.map(l => g.measureText(l).width)) + pad * 2, hh = lines.length * 16 + pad * 2 - 4;
    const x = Math.max(4, Math.min(this.W - w - 4, mx + 12)), y = Math.max(4, Math.min(this.H - hh - 4, my + 12));
    g.fillStyle = "rgba(12,16,26,0.94)"; g.strokeStyle = "rgba(255,255,255,0.12)"; g.lineWidth = 1;
    g.beginPath(); g.roundRect(x, y, w, hh, 7); g.fill(); g.stroke();
    g.textAlign = "left";
    lines.forEach((l, i) => { g.fillStyle = i === 0 ? color : "#d5dae0"; g.font = `${i === 0 ? "600 " : ""}12px Inter, system-ui, sans-serif`; g.fillText(l, x + pad, y + pad + 11 + i * 16); });
  }
}
