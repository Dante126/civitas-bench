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
    this.data = null; this.month = -1; this.funds = {}; this.filter = () => true;
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
  setMonth(month, { funds = {}, animate = false, budgetMs = 1600, filter } = {}) {
    if (filter) this.filter = filter;
    this.funds = funds;
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
      this.floats.push({ text: q.text, x, y: y + this._radius(q.a) + 24 * this.scale, color: q.color, t0: now - 200, dur: 1500, size: 10 });
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
    const st = (this.data.stats || {})[h.role] || {};
    const lines = h.role
      ? [h.role, `asked ${st.asked || 0} · answered ${st.answered || 0}`, `asked of it ${st.asked_of || 0} · it answered ${st.answered_by || 0}`,
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
