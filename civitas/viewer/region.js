// Region map: every city drawn from its real tiles at its place in the region layout, the countryside and
// river between them, and the links (highways, rail, boats, planes) running from the actual connector tiles,
// with animated traffic sized by the goods, commuters and tourists that moved this month.
const GAP = 12;                                   // tiles of countryside between neighbouring cities
const LINK = {
  highway: { color: "#3b3f46", edge: "#f2c94c", width: 1.1, label: "highway" },
  rail: { color: "#7b4a2f", edge: "#d7ccc8", width: 0.7, label: "rail" },
  boat: { color: "#4aa3ff", width: 0.45, dash: [1.2, 1.2], label: "boat" },
  air: { color: "#b99bff", width: 0.35, dash: [2.4, 1.6], label: "air" },
};
export const SPEC_COLOR = { "mining & farming": "#c8a45a", "high tech": "#4fc3f7", gambling: "#e040fb", tourism: "#ffb454", trade: "#45d483", education: "#f5d547" };
const FLOW_COLOR = { goods: "#ffb454", commuters: "#ffffff", tourists: "#ff6fb5" };

const hash = (a, b) => { let h = (a * 374761393 + b * 668265263) | 0; h = (h ^ (h >>> 13)) * 1274126177; return ((h ^ (h >>> 16)) >>> 0) / 4294967296; };

export class RegionMap {
  constructor(canvas, { mini, onPick }) {
    this.cv = canvas; this.mini = mini; this.onPick = onPick; this.hover = null;
    canvas.addEventListener("mousemove", (e) => { const c = this._cityAt(e); if (c !== this.hover) { this.hover = c; canvas.style.cursor = c ? "pointer" : "default"; } });
    canvas.addEventListener("mouseleave", () => { this.hover = null; });
    canvas.addEventListener("click", (e) => { const c = this._cityAt(e); if (c && this.onPick) this.onPick(c); });
    this._loop = this._loop.bind(this);
  }

  set(frame, meta, current) {
    this.frame = frame; this.meta = meta; this.current = current;
    const L = meta.layout, xs = Object.values(L).map(p => p[0]), ys = Object.values(L).map(p => p[1]);
    this.x0 = Math.min(...xs); this.y0 = Math.min(...ys);
    this.cols = Math.max(...xs) - this.x0 + 1; this.rows = Math.max(...ys) - this.y0 + 1;
    this.W = meta.width; this.H = meta.height;
    this.TW = this.cols * this.W + (this.cols + 1) * GAP; this.TH = this.rows * this.H + (this.rows + 1) * GAP;
    this._resize();
    this.base = this._paintBase();
    this.paths = this._paths();
    if (!this.running) { this.running = true; requestAnimationFrame(this._loop); }
  }

  origin(c) { const p = this.meta.layout[c]; return [GAP + (p[0] - this.x0) * (this.W + GAP), GAP + (p[1] - this.y0) * (this.H + GAP)]; }

  _resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1), cssW = this.cv.clientWidth || 340;
    const maxH = this.cv.dataset.maxh ? +this.cv.dataset.maxh : Infinity;
    let s = cssW / this.TW;
    if (this.TH * s > maxH) s = maxH / this.TH;
    this.s = s; this.cssW = this.TW * s; this.cssH = this.TH * s;
    this.cv.style.height = `${this.cssH}px`;
    this.cv.width = Math.round(cssW * dpr); this.cv.height = Math.round(this.cssH * dpr);
    this.ox = (cssW - this.cssW) / 2; this.dpr = dpr;
  }

  // ------------------------------------------------------------ static layer
  _paintBase() {
    const s = this.s, off = document.createElement("canvas");
    off.width = Math.ceil(this.TW * s * this.dpr); off.height = Math.ceil(this.TH * s * this.dpr);
    const g = off.getContext("2d"); g.scale(this.dpr * s, this.dpr * s);
    // countryside: mostly meadow, with some crop fields and woods in 6x6-tile parcels
    g.fillStyle = "#6f9a55"; g.fillRect(0, 0, this.TW, this.TH);
    for (let y = 0; y < this.TH; y += 6) for (let x = 0; x < this.TW; x += 6) {
      const b = hash(x, y);
      if (b < 0.1) g.fillStyle = "#a9a765";                         // wheat
      else if (b < 0.18) g.fillStyle = "#8fa95a";                   // young crops
      else if (b < 0.36) g.fillStyle = "#5d8a48";                   // woods
      else g.fillStyle = b < 0.6 ? "#6f9a55" : b < 0.8 ? "#76a05a" : "#6a9450";
      g.fillRect(x, y, 6.05, 6.05);
      if (b >= 0.18 && b < 0.36) { g.fillStyle = "#4b7a3b"; for (let k = 0; k < 7; k++) { g.beginPath(); g.arc(x + 0.8 + hash(x + k, y) * 4.4, y + 0.8 + hash(y + k, x) * 4.4, 0.7, 0, Math.PI * 2); g.fill(); } }
      else if (b < 0.18) { g.fillStyle = "rgba(0,0,0,.07)"; for (let k = 1; k < 6; k += 2) g.fillRect(x, y + k, 6, 0.5); }
    }
    this._paintRivers(g);
    for (const c of this.meta.cities) this._paintCity(g, c);
    this._paintOuterRoads(g);
    return off;
  }

  _water(c, y) { const t = this.frame.cities[c].tiles, xs = []; for (let x = 0; x < this.W; x++) if ("~=Y".includes(t[y * this.W + x])) xs.push(x); return xs.length ? [Math.min(...xs), Math.max(...xs) + 1] : null; }

  _paintRivers(g) {
    g.fillStyle = "#3d8fd1";
    const chain = this.meta.river_chain || [];
    for (const c of this.meta.cities) {
      const [ox, oy] = this.origin(c), top = this._water(c, 0), bot = this._water(c, this.H - 1);
      const k = chain.indexOf(c), up = k > 0 ? chain[k - 1] : null, down = k >= 0 && k < chain.length - 1 ? chain[k + 1] : null;
      if (top && !up) this._band(g, ox + top[0], ox + top[1], oy, -1, this._blocked(c, -1) ? GAP / 2 : oy);
      if (bot && !down) this._band(g, ox + bot[0], ox + bot[1], oy + this.H, 1, this._blocked(c, 1) ? GAP / 2 : this.TH - oy - this.H);
      if (bot && down) {                              // the river flows on into the next city downstream
        const [dx, dy] = this.origin(down), dtop = this._water(down, 0);
        if (dtop) {
          g.beginPath(); g.moveTo(ox + bot[0], oy + this.H); g.lineTo(ox + bot[1], oy + this.H);
          const my = (oy + this.H + dy) / 2;
          g.bezierCurveTo(ox + bot[1], my, dx + dtop[1], my, dx + dtop[1], dy); g.lineTo(dx + dtop[0], dy);
          g.bezierCurveTo(dx + dtop[0], my, ox + bot[0], my, ox + bot[0], oy + this.H); g.fill();
        }
      }
    }
  }
  _blocked(c, dir) { const p = this.meta.layout[c]; return this.meta.cities.some(o => o !== c && this.meta.layout[o][0] === p[0] && Math.sign(this.meta.layout[o][1] - p[1]) === dir); }
  _band(g, x1, x2, y0, dir, len) {                 // river running on beyond the city, meandering a little
    const ye = y0 + dir * len, sway = Math.min(3, len / 4), a = y0 + dir * len / 3, b = y0 + dir * 2 * len / 3;
    g.beginPath(); g.moveTo(x1, y0); g.bezierCurveTo(x1 + sway, a, x1 - sway, b, x1, ye);
    g.lineTo(x2, ye); g.bezierCurveTo(x2 - sway, b, x2 + sway, a, x2, y0); g.fill();
  }

  _paintCity(g, c) {
    const [ox, oy] = this.origin(c), f = this.frame.cities[c], W = this.W;
    g.save(); g.shadowColor = "rgba(0,0,0,.35)"; g.shadowBlur = 2.5; g.fillStyle = "#7fb069"; g.fillRect(ox, oy, W, this.H); g.restore();
    for (let y = 0; y < this.H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, ch = f.tiles[i];
      if (ch === ".") { if (hash(i, 7) < 0.12) { g.fillStyle = "#5f9150"; g.fillRect(ox + x, oy + y, 1, 1); } continue; }
      g.fillStyle = this.mini[ch] || "#f0f";
      if ("RCIOM".includes(ch)) g.globalAlpha = 0.55 + 0.09 * +f.levels[i];
      g.fillRect(ox + x, oy + y, 1.02, 1.02); g.globalAlpha = 1;
    }
    g.strokeStyle = "rgba(255,255,255,.55)"; g.lineWidth = 0.25; g.strokeRect(ox, oy, W, this.H);
  }

  _paintOuterRoads(g) {                             // highways leaving a city towards the outside world
    const linked = new Set((this.frame.links || []).filter(l => l.kind === "highway").flatMap(l => [`${l.a}>${l.b}`, `${l.b}>${l.a}`]));
    for (const c of this.meta.cities) {
      const [ox, oy] = this.origin(c), t = this.frame.cities[c].tiles, W = this.W, H = this.H;
      for (const [side, dx, dy] of [["west", -1, 0], ["east", 1, 0], ["north", 0, -1], ["south", 0, 1]]) {
        const nb = this.meta.cities.find(o => { const a = this.meta.layout[c], b = this.meta.layout[o]; return b[0] - a[0] === dx && b[1] - a[1] === dy; });
        if (nb && linked.has(`${c}>${nb}`)) continue;
        for (const [x, y] of this._edgeTiles(t, side, "H")) {
          const len = nb ? GAP / 2 - 1 : GAP + 2;
          g.fillStyle = "#2a2d33"; g.fillRect(ox + x + (dx < 0 ? -len : dx > 0 ? 1 : 0), oy + y + (dy < 0 ? -len : dy > 0 ? 1 : 0), dx ? len : 1, dy ? len : 1);
        }
      }
    }
  }

  _edgeTiles(t, side, chars) {
    const W = this.W, H = this.H, out = [];
    const at = (x, y) => chars.includes(t[y * W + x]);
    if (side === "west") { for (let y = 0; y < H; y++) if (at(0, y)) out.push([0, y]); }
    if (side === "east") { for (let y = 0; y < H; y++) if (at(W - 1, y)) out.push([W - 1, y]); }
    if (side === "north") { for (let x = 0; x < W; x++) if (at(x, 0)) out.push([x, 0]); }
    if (side === "south") { for (let x = 0; x < W; x++) if (at(x, H - 1)) out.push([x, H - 1]); }
    return out;
  }
  _nearestTile(c, chars, px, py) {                  // the tile of that kind closest to a local point
    const t = this.frame.cities[c].tiles; let best = null, bd = Infinity;
    for (let i = 0; i < t.length; i++) if (chars.includes(t[i])) { const x = i % this.W, y = Math.floor(i / this.W), d = (x - px) ** 2 + (y - py) ** 2; if (d < bd) { bd = d; best = [x, y]; } }
    return best;
  }

  // ------------------------------------------------------------ link paths
  _paths() {
    const out = [], flows = this.frame.flows || [], links = this.frame.links || [];
    const pairFlow = (a, b) => flows.filter(f => (f.a === a && f.b === b) || (f.a === b && f.b === a))
      .reduce((s, f) => ({ goods: s.goods + f.goods + f.raw, commuters: s.commuters + f.commuters, tourists: s.tourists + f.tourists }), { goods: 0, commuters: 0, tourists: 0 });
    const perPair = {};
    for (const l of links) {
      const key = [l.a, l.b].sort().join("|"); perPair[key] = (perPair[key] || 0) + 1;
      const pts = this._linkPath(l, perPair[key] - 1);
      if (!pts) continue;
      const fl = pairFlow(l.a, l.b), share = l.people + l.goods > 0 ? 1 / links.filter(o => [o.a, o.b].sort().join("|") === key).length : 0;
      const segs = []; let len = 0;
      for (let k = 1; k < pts.length; k++) { const d = Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]); segs.push(d); len += d; }
      out.push({ link: l, pts, segs, len, flow: { goods: fl.goods * share, commuters: fl.commuters * share, tourists: fl.tourists * share } });
    }
    return out;
  }

  _linkPath(l, k) {
    const A = this.meta.layout[l.a], B = this.meta.layout[l.b], [ax, ay] = this.origin(l.a), [bx, by] = this.origin(l.b);
    const W = this.W, H = this.H, dx = Math.sign(B[0] - A[0]), dy = Math.sign(B[1] - A[1]);
    if (l.kind === "highway" || l.kind === "rail") {
      const side = dx > 0 ? "east" : dx < 0 ? "west" : dy > 0 ? "south" : "north";
      const opp = { east: "west", west: "east", north: "south", south: "north" }[side];
      const chars = l.kind === "highway" ? "H" : "@";
      const pick = (c, sd, o) => {
        const e = this._edgeTiles(this.frame.cities[c].tiles, sd, chars);
        if (e.length) return e[Math.min(e.length - 1, k)];
        const mid = { east: [W - 1, H / 2], west: [0, H / 2], north: [W / 2, 0], south: [W / 2, H - 1] }[sd];
        return this._nearestTile(c, chars, mid[0], mid[1]) || mid.map(Math.floor);
      };
      const pa = pick(l.a, side), pb = pick(l.b, opp);
      const edgePt = ([x, y], sd, [ox, oy]) => sd === "east" ? [ox + W, oy + y + 0.5] : sd === "west" ? [ox, oy + y + 0.5] : sd === "south" ? [ox + x + 0.5, oy + H] : [ox + x + 0.5, oy];
      const inner = ([x, y], [ox, oy]) => [ox + x + 0.5, oy + y + 0.5];
      const p1 = edgePt(pa, side, [ax, ay]), p2 = edgePt(pb, opp, [bx, by]);
      const pts = [inner(pa, [ax, ay]), p1];
      if (dx && !dy) { const mx = (p1[0] + p2[0]) / 2; pts.push([mx, p1[1]], [mx, p2[1]]); }
      else if (dy && !dx) { const my = (p1[1] + p2[1]) / 2; pts.push([p1[0], my], [p2[0], my]); }
      pts.push(p2, inner(pb, [bx, by]));
      return pts;
    }
    const anchor = (c, chars, [ox, oy]) => { const t = this._nearestTile(c, chars, W / 2, H / 2); return t ? [ox + t[0] + 0.5, oy + t[1] + 0.5] : [ox + W / 2, oy + H / 2]; };
    const p1 = anchor(l.a, l.kind === "boat" ? "&" : ">", [ax, ay]), p2 = anchor(l.b, l.kind === "boat" ? "&" : ">", [bx, by]);
    const n = 24, pts = [];
    const bend = l.kind === "air" ? 0.25 : 0.12, nx = -(p2[1] - p1[1]) * bend, ny = (p2[0] - p1[0]) * bend;
    for (let i = 0; i <= n; i++) { const t = i / n, b = 4 * t * (1 - t); pts.push([p1[0] + (p2[0] - p1[0]) * t + nx * b, p1[1] + (p2[1] - p1[1]) * t + ny * b]); }
    return pts;
  }

  _at(p, d) {                                       // point + heading at distance d along a path
    let k = 0; while (k < p.segs.length - 1 && d > p.segs[k]) { d -= p.segs[k]; k++; }
    const a = p.pts[k], b = p.pts[k + 1], t = p.segs[k] ? Math.min(1, d / p.segs[k]) : 0;
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, Math.atan2(b[1] - a[1], b[0] - a[0])];
  }

  // ------------------------------------------------------------ per frame
  _loop(now) {
    if (!this.cv.isConnected || this.cv.offsetParent === null) { this.running = false; return; }   // stops while hidden; set() restarts it
    this._draw(now / 1000);
    requestAnimationFrame(this._loop);
  }

  _draw(t) {
    const g = this.cv.getContext("2d"), s = this.s, dpr = this.dpr;
    g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, this.cv.width, this.cv.height);
    g.drawImage(this.base, this.ox * dpr, 0);
    g.setTransform(dpr * s, 0, 0, dpr * s, this.ox * dpr, 0);
    g.lineCap = "round"; g.lineJoin = "round";
    for (const p of this.paths) {                     // the links themselves
      const st = LINK[p.link.kind] || LINK.highway;
      g.setLineDash(st.dash || []);
      g.beginPath(); p.pts.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y));
      if (st.edge) { g.strokeStyle = st.edge; g.lineWidth = st.width + 0.35; g.stroke(); }
      g.strokeStyle = st.color; g.lineWidth = st.width; g.stroke();
      if (p.link.kind === "rail") { g.setLineDash([0.25, 0.55]); g.strokeStyle = "#e0d6cc"; g.lineWidth = st.width * 0.9; g.stroke(); }
    }
    g.setLineDash([]);
    for (const p of this.paths) {                     // traffic: goods trucks, commuters, tourists
      const kinds = Object.entries(p.flow).filter(([, v]) => v > 0);
      const speed = { highway: 5, rail: 9, boat: 2.5, air: 14 }[p.link.kind] || 5;
      for (const [kind, v] of kinds) {
        const n = Math.min(10, 1 + Math.round(Math.sqrt(v) / 6));
        for (let i = 0; i < n; i++) {
          const fwd = (i + (kind === "goods" ? 0 : 1)) % 2 === 0;
          let d = ((t * speed + (i / n) * p.len + hash(i, kind.length) * 3) % p.len);
          if (!fwd) d = p.len - d;
          const [x, y, a] = this._at(p, d);
          const side = (fwd ? 1 : -1) * (p.link.kind === "highway" ? 0.28 : 0);
          const px = x - Math.sin(a) * side, py = y + Math.cos(a) * side;
          g.fillStyle = FLOW_COLOR[kind];
          if (p.link.kind === "air") { this._plane(g, px, py, a + (fwd ? 0 : Math.PI)); continue; }
          if (p.link.kind === "rail") { g.save(); g.translate(px, py); g.rotate(a); g.fillRect(-1.2, -0.35, 2.4, 0.7); g.restore(); continue; }
          g.beginPath(); g.arc(px, py, p.link.kind === "boat" ? 0.55 : 0.4, 0, Math.PI * 2); g.fill();
        }
      }
    }
    for (const c of this.meta.cities) this._label(g, c);
  }

  _plane(g, x, y, a) {
    g.save(); g.translate(x, y); g.rotate(a); g.fillStyle = "#fff";
    g.beginPath(); g.moveTo(1.2, 0); g.lineTo(-0.9, -0.25); g.lineTo(-0.9, 0.25); g.closePath(); g.fill();
    g.fillRect(-0.3, -0.9, 0.35, 1.8); g.restore();
  }

  _label(g, c) {
    const [ox, oy] = this.origin(c), m = this.frame.cities[c].metrics || {}, sc = (this.frame.scores || {})[`${c}.city`];
    const cur = c === this.current, hov = c === this.hover;
    if (cur || hov) { g.strokeStyle = cur ? "#fff" : "rgba(255,255,255,.7)"; g.lineWidth = cur ? 0.7 : 0.45; g.strokeRect(ox - 0.6, oy - 0.6, this.W + 1.2, this.H + 1.2); }
    const px = 1 / this.s, fs = Math.max(2.4, 12 * px);           // at least ~12 css px
    const spec = Object.entries(m.specializations || {}).sort((a, b) => b[1] - a[1])[0];
    const title = c[0].toUpperCase() + c.slice(1), sub = `${(m.population || 0).toLocaleString()} · ${sc != null ? sc.toFixed(1) : "–"}`;
    g.font = `700 ${fs}px system-ui, sans-serif`; const w1 = g.measureText(title).width;
    g.font = `500 ${fs * 0.8}px system-ui, sans-serif`; const w2 = g.measureText(sub).width;
    const specTxt = spec && spec[1] >= 0.33 ? spec[0] : ""; const w3 = specTxt ? g.measureText(specTxt).width + fs * 0.9 : 0;
    const bw = Math.max(w1, w2, w3) + fs * 1.1, bh = fs * (specTxt ? 3.2 : 2.3), bx = ox + this.W / 2 - bw / 2, by = oy + fs * 0.6;
    g.fillStyle = cur ? "rgba(20,24,31,.9)" : "rgba(20,24,31,.72)";
    g.beginPath(); g.roundRect ? g.roundRect(bx, by, bw, bh, fs * 0.4) : g.rect(bx, by, bw, bh); g.fill();
    g.textAlign = "center"; g.textBaseline = "top";
    g.fillStyle = "#fff"; g.font = `700 ${fs}px system-ui, sans-serif`; g.fillText(title, ox + this.W / 2, by + fs * 0.3);
    g.fillStyle = "#c9ced6"; g.font = `500 ${fs * 0.8}px system-ui, sans-serif`; g.fillText(sub, ox + this.W / 2, by + fs * 1.35);
    if (specTxt) {
      g.fillStyle = SPEC_COLOR[specTxt] || "#fff";
      g.beginPath(); g.arc(ox + this.W / 2 - w3 / 2 + fs * 0.3, by + fs * 2.62, fs * 0.25, 0, Math.PI * 2); g.fill();
      g.fillText(specTxt, ox + this.W / 2 + fs * 0.3, by + fs * 2.25);
    }
  }

  _cityAt(e) {
    const r = this.cv.getBoundingClientRect(), x = (e.clientX - r.left - this.ox) / this.s, y = (e.clientY - r.top) / this.s;
    return this.meta.cities.find(c => { const [ox, oy] = this.origin(c); return x >= ox && y >= oy && x < ox + this.W && y < oy + this.H; }) || null;
  }
}

export function regionLegend() {
  return Object.entries(LINK).map(([k, st]) => `<span><i style="background:${st.color};height:4px;width:14px;border-radius:2px${st.edge ? `;box-shadow:0 0 0 1px ${st.edge}` : ""}"></i>${st.label}</span>`).join("")
    + Object.entries(FLOW_COLOR).map(([k, c]) => `<span><i style="background:${c}"></i>${k}</span>`).join("");
}
