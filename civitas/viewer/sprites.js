// Pixel-art sprite painter for the pixel view (pixel.js), in classic pixel-art style: 2:1 dimetric tiles,
// light from the top-left, hard pixel edges and dark silhouettes. A Builder collects 3D primitives in tile units
// (x, y along the map, z up, one height unit = ZS pixels); paint() rotates them to the view, depth-sorts them,
// shades every face from its normal and rasterizes the result without anti-aliasing. What to draw lives in
// recipes.js; pixel.js caches the painted sprites.

export const TW = 64, TH = 32, ZS = 40;          // tile width / height in pixels at 1x; pixels per height unit
const DZ = (TW / 2) / ZS;                         // (1, 1, DZ) points from the scene at the viewer
const LIGHT = [0.28, 0.64, 1.0];                  // right walls dark, left walls mid, tops full

export const hash = (i, k = 0) => {               // integer hash -> [0, 1), same as scene.js
  let h = Math.imul((i | 0) ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul((k | 0) + 0x632be5ab, 0xc2b2ae35);
  h ^= h >>> 15; h = Math.imul(h, 0x2c1b3c6d); h ^= h >>> 12; h = Math.imul(h, 0x297a2d39); h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
};
const _rgb = new Map();
export function rgb(c) {
  if (Array.isArray(c)) return c;
  let v = _rgb.get(c);
  if (!v) { const n = parseInt(c.slice(1), 16); v = [(n >> 16) & 255, (n >> 8) & 255, n & 255]; _rgb.set(c, v); }
  return v;
}
export const mix = (a, b, t) => { a = rgb(a); b = rgb(b); return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; };
const mul = (c, k) => [c[0] * k, c[1] * k, c[2] * k];
const SNOW = [236, 241, 246], NIGHT = [0.3, 0.34, 0.54];

// quarter-turn rotation shared by the map and every sprite
export function rotQ(x, y, r) { r &= 3; return r === 0 ? [x, y] : r === 1 ? [-y, x] : r === 2 ? [-x, -y] : [y, -x]; }
export const project = (u, v, z = 0) => [(u - v) * (TW / 2), (u + v) * (TH / 2) - z * ZS];

// ------------------------------------------------------------------ builder
export class Builder {
  constructor() {
    this.groups = []; this.c = 1; this.s = 0; this.tx = 0; this.ty = 0; this.k = 1; this.kz = 1;
    this.meta = { smoke: [], rotor: null };
  }
  turn(q) { q &= 3; this.c = [1, 0, -1, 0][q]; this.s = [0, 1, 0, -1][q]; return this; }
  angle(a) { this.c = Math.cos(a); this.s = Math.sin(a); return this; }
  scale(k, kz = k) { this.k = k; this.kz = kz; return this; }
  move(x, y) { this.tx = x; this.ty = y; return this; }
  P(x, y, z) { x *= this.k; y *= this.k; return [x * this.c - y * this.s + this.tx, x * this.s + y * this.c + this.ty, z * this.kz]; }
  N(nx, ny) { return [nx * this.c - ny * this.s, nx * this.s + ny * this.c, 0]; }
  _group(ctr, o) { const g = { faces: [], wins: [], blob: null, ctr, o }; this.groups.push(g); return g; }

  // axis-aligned box centred on (x, y), sx by sy, from z up h; o.top: roof colour, o.win: window grid
  box(x, y, sx, sy, z, h, col, o = {}) {
    const x0 = x - sx / 2, x1 = x + sx / 2, y0 = y - sy / 2, y1 = y + sy / 2, z1 = z + h, P = (a, b, c) => this.P(a, b, c);
    const g = this._group(P(x, y, z + h / 2), o);
    g.faces.push({ p: [P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)], c: o.top || col, o: { ...o, win: null } });
    const sides = [
      [[P(x0, y1, z), P(x1, y1, z), P(x1, y1, z1), P(x0, y1, z1)], [0, 1], sx, (t, zz) => P(x0 + t * sx, y1 + 0.002, zz)],
      [[P(x1, y0, z), P(x1, y1, z), P(x1, y1, z1), P(x1, y0, z1)], [1, 0], sy, (t, zz) => P(x1 + 0.002, y1 - t * sy, zz)],
      [[P(x0, y0, z), P(x1, y0, z), P(x1, y0, z1), P(x0, y0, z1)], [0, -1], sx, (t, zz) => P(x1 - t * sx, y0 - 0.002, zz)],
      [[P(x0, y0, z), P(x0, y1, z), P(x0, y1, z1), P(x0, y0, z1)], [-1, 0], sy, (t, zz) => P(x0 - 0.002, y0 + t * sy, zz)],
    ];
    for (const [p, n, len, at] of sides) {
      g.faces.push({ p, c: o.side || col, o });
      if (o.win) this._windows(g, o.win, len, z, h, at, this.N(n[0], n[1]));
    }
    return this;
  }
  _windows(g, w, len, z, h, at, n) {
    const sp = w.sp || 0.14, fl = w.fl || 0.2, z0 = w.z0 ?? 0.06, top = w.top ?? 0.05;
    const cols = Math.max(1, Math.floor((len * this.k - 0.04) / sp)), rows = Math.max(h - z0 - top > 0.08 ? 1 : 0, Math.floor((h - z0 - top) / fl + 1e-6));
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const idx = g.wins.length;
      g.wins.push({ p: at((c + 0.5) / cols, z + z0 + r * fl + fl * 0.75), n, w: w.w || 2, h: w.h || 3, c: w.c || "#3a4a5e",
        lit: hash((w.seed || 0) * 131 + idx, 17) < (w.lit ?? 0.55), glow: w.glow });
    }
  }
  // gable roof over a sx by sy rectangle, ridge along x (axis "x") or y; o.wall colours the gable ends
  gable(x, y, sx, sy, z, h, col, o = {}) {
    const P = (a, b, c) => this.P(a, b, c), zh = z + h;
    const g = this._group(P(x, y, z), o);
    const along = o.axis === "y";
    const x0 = x - sx / 2, x1 = x + sx / 2, y0 = y - sy / 2, y1 = y + sy / 2, wall = { ...o, nosnow: true };
    if (!along) {
      g.faces.push({ p: [P(x0, y1, z), P(x1, y1, z), P(x1, y, zh), P(x0, y, zh)], c: col, o });
      g.faces.push({ p: [P(x0, y0, z), P(x1, y0, z), P(x1, y, zh), P(x0, y, zh)], c: col, o });
      g.faces.push({ p: [P(x1, y0, z), P(x1, y1, z), P(x1, y, zh)], c: o.wall || col, o: wall });
      g.faces.push({ p: [P(x0, y0, z), P(x0, y1, z), P(x0, y, zh)], c: o.wall || col, o: wall });
    } else {
      g.faces.push({ p: [P(x1, y0, z), P(x1, y1, z), P(x, y1, zh), P(x, y0, zh)], c: col, o });
      g.faces.push({ p: [P(x0, y0, z), P(x0, y1, z), P(x, y1, zh), P(x, y0, zh)], c: col, o });
      g.faces.push({ p: [P(x0, y1, z), P(x1, y1, z), P(x, y1, zh)], c: o.wall || col, o: wall });
      g.faces.push({ p: [P(x0, y0, z), P(x1, y0, z), P(x, y0, zh)], c: o.wall || col, o: wall });
    }
    return this;
  }
  // hipped roof (a pyramid when square)
  hip(x, y, sx, sy, z, h, col, o = {}) {
    const P = (a, b, c) => this.P(a, b, c), zh = z + h;
    const g = this._group(P(x, y, z), o);
    const x0 = x - sx / 2, x1 = x + sx / 2, y0 = y - sy / 2, y1 = y + sy / 2;
    const d = Math.min(sx, sy) / 2;
    const [ra, rb] = sx >= sy ? [P(x0 + d, y, zh), P(x1 - d, y, zh)] : [P(x, y0 + d, zh), P(x, y1 - d, zh)];
    if (sx >= sy) {
      g.faces.push({ p: [P(x0, y1, z), P(x1, y1, z), rb, ra], c: col, o }, { p: [P(x0, y0, z), P(x1, y0, z), rb, ra], c: col, o },
        { p: [P(x1, y0, z), P(x1, y1, z), rb], c: col, o }, { p: [P(x0, y0, z), P(x0, y1, z), ra], c: col, o });
    } else {
      g.faces.push({ p: [P(x1, y0, z), P(x1, y1, z), rb, ra], c: col, o }, { p: [P(x0, y0, z), P(x0, y1, z), rb, ra], c: col, o },
        { p: [P(x0, y1, z), P(x1, y1, z), rb], c: col, o }, { p: [P(x0, y0, z), P(x1, y0, z), ra], c: col, o });
    }
    return this;
  }
  // surface of revolution: profile [[r, z], ...] bottom to top; cylinders, cones, domes, tanks
  lathe(x, y, prof, col, o = {}) {
    const seg = o.seg || 10, sy = o.sy || 1, a0 = o.a0 ?? Math.PI / seg;
    const ring = (r, z) => Array.from({ length: seg }, (_, k) => {
      const a = a0 + k * 2 * Math.PI / seg; return this.P(x + Math.cos(a) * r, y + Math.sin(a) * r * sy, z);
    });
    const g = this._group(this.P(x, y, (prof[0][1] + prof[prof.length - 1][1]) / 2), o);
    const rings = prof.map(([r, z]) => ring(r, z));
    for (let j = 0; j + 1 < rings.length; j++) for (let k = 0; k < seg; k++) {
      const a = rings[j][k], b = rings[j][(k + 1) % seg], c = rings[j + 1][(k + 1) % seg], d = rings[j + 1][k];
      g.faces.push({ p: prof[j + 1][0] === 0 ? [a, b, c] : [a, b, c, d], c: col, o });
    }
    const last = prof[prof.length - 1];
    if (last[0] > 0) g.faces.push({ p: rings[rings.length - 1], c: o.top || col, o: { ...o, win: null } });
    if (o.win) {                                   // window columns on every other side
      const [r0, zb] = prof[0], zt = prof[prof.length - 1][1];
      for (let k = 0; k < seg; k += 1) {
        const a = a0 + (k + 0.5) * 2 * Math.PI / seg, n = this.N(Math.cos(a), Math.sin(a) * sy);
        const at = (t, zz) => this.P(x + Math.cos(a) * (r0 + 0.003), y + Math.sin(a) * (r0 + 0.003) * sy, zz);
        this._windows(g, { ...o.win, sp: 1 }, 0.5, zb, zt - zb, at, n);
      }
    }
    return this;
  }
  cyl(x, y, r, z, h, col, o = {}) { return this.lathe(x, y, [[r, z], [o.rt ?? r, z + h]], col, o); }
  cone(x, y, r, z, h, col, o = {}) { return this.lathe(x, y, [[r, z], [0, z + h]], col, { seg: 8, ...o }); }
  dome(x, y, r, z, col, o = {}) {
    return this.lathe(x, y, [[r, z], [r * 0.92, z + r * 0.38], [r * 0.7, z + r * 0.72], [r * 0.36, z + r * 0.94]], col, { seg: 12, ...o });
  }
  // flat horizontal rectangle (lots, pools, pitches, road paint)
  flat(x, y, sx, sy, z, col, o = {}) {
    const x0 = x - sx / 2, x1 = x + sx / 2, y0 = y - sy / 2, y1 = y + sy / 2, P = (a, b, c) => this.P(a, b, c);
    const g = this._group(P(x, y, z - 1), o);
    g.faces.push({ p: [P(x0, y0, z), P(x1, y0, z), P(x1, y1, z), P(x0, y1, z)], c: col, o });
    return this;
  }
  // any polygon [[x, y, z], ...]; o.two: visible from both sides, o.ctr: a point behind it
  poly(pts, col, o = {}) {
    const g = this._group(o.ctr ? this.P(...o.ctr) : this.P(pts[0][0], pts[0][1], pts[0][2] - 1), o);
    g.faces.push({ p: pts.map(p => this.P(...p)), c: col, o });
    return this;
  }
  // wall panel facing out along (nx, ny); explicit normal, used for river banks
  wall(pts, n, col, o = {}) {
    const g = this._group(null, o);
    g.faces.push({ p: pts.map(p => this.P(...p)), n: this.N(n[0], n[1]), c: col, o });
    return this;
  }
  // soft round clump drawn in screen space (tree crowns, bushes, smoke-free domes)
  blob(x, y, z, r, col, o = {}) {
    const g = this._group(this.P(x, y, z), o);
    g.blob = { p: this.P(x, y, z), r: r * this.k, c: col, o };
    return this;
  }
  smoke(x, y, z, rate = 1, dark = 0.5, fire = false) { this.meta.smoke.push({ p: this.P(x, y, z), rate, dark, fire }); return this; }
  rotor(x, y, z, len) { this.meta.rotor = { p: this.P(x, y, z), len: len * this.kz }; return this; }
}

// ------------------------------------------------------------------ paint
const newell = (p) => {
  let nx = 0, ny = 0, nz = 0;
  for (let i = 0; i < p.length; i++) {
    const a = p[i], b = p[(i + 1) % p.length];
    nx += (a[1] - b[1]) * (a[2] + b[2]); ny += (a[2] - b[2]) * (a[0] + b[0]); nz += (a[0] - b[0]) * (a[1] + b[1]);
  }
  const l = Math.hypot(nx, ny, nz) || 1;
  return [nx / l, ny / l, nz / l];
};

export function fillPoly(buf, W, H, pts, col, gid, gbuf, tex) {
  let y0 = Infinity, y1 = -Infinity;
  for (const p of pts) { if (p[1] < y0) y0 = p[1]; if (p[1] > y1) y1 = p[1]; }
  const ya = Math.max(0, Math.ceil(y0 - 0.5)), yb = Math.min(H - 1, Math.floor(y1 - 0.5));
  const xs = [];
  for (let y = ya; y <= yb; y++) {
    const sy = y + 0.5; xs.length = 0;
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      if ((a[1] <= sy && b[1] > sy) || (b[1] <= sy && a[1] > sy)) xs.push(a[0] + (sy - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
    }
    xs.sort((p, q) => p - q);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const xa = Math.max(0, Math.ceil(xs[k] - 0.5)), xb = Math.min(W - 1, Math.floor(xs[k + 1] - 0.5));
      for (let x = xa; x <= xb; x++) { buf[y * W + x] = tex ? tex(x, y) : col; gbuf[y * W + x] = gid; }
    }
  }
}
const pack = (c) => (255 << 24) | (Math.max(0, Math.min(255, c[2] | 0)) << 16) | (Math.max(0, Math.min(255, c[1] | 0)) << 8) | Math.max(0, Math.min(255, c[0] | 0));
const unpack = (v) => [v & 255, (v >>> 8) & 255, (v >>> 16) & 255];

// Depth order between two groups' rotated boxes: -1 if a is drawn first, 1 if b, 0 if unknown.
function order(a, b) {
  const e = 1e-4;
  if (a.u1 <= b.u0 + e) return -1; if (b.u1 <= a.u0 + e) return 1;
  if (a.v1 <= b.v0 + e) return -1; if (b.v1 <= a.v0 + e) return 1;
  if (a.z1 <= b.z0 + e) return -1; if (b.z1 <= a.z0 + e) return 1;
  return a.d < b.d ? -1 : a.d > b.d ? 1 : 0;
}

/**
 * Rasterize a Builder. opts: rot (0-3 view rotation), night, season, outline (dark silhouette),
 * tint ([r, g, b] multiply, e.g. seasons on the ground). Returns { canvas, w, h, ox, oy, alpha, top, smoke, rotor }
 * where (ox, oy) is the sprite pixel of the local origin (0, 0, 0).
 */
export function paint(b, { rot = 0, night = false, season = null, outline = true, canvas = true } = {}) {
  const R = (p) => { const [u, v] = rotQ(p[0], p[1], rot); return [u, v, p[2]]; };
  const S = (p) => project(p[0], p[1], p[2]);
  const gs = [];
  let mnx = Infinity, mny = Infinity, mxx = -Infinity, mxy = -Infinity;
  const ext = (x, y) => { if (x < mnx) mnx = x; if (x > mxx) mxx = x; if (y < mny) mny = y; if (y > mxy) mxy = y; };
  for (const g of b.groups) {
    const G = { faces: [], wins: [], blob: null, u0: Infinity, u1: -Infinity, v0: Infinity, v1: -Infinity, z0: Infinity, z1: -Infinity,
      sx0: Infinity, sx1: -Infinity, sy0: Infinity, sy1: -Infinity };
    const box = (p) => {
      G.u0 = Math.min(G.u0, p[0]); G.u1 = Math.max(G.u1, p[0]); G.v0 = Math.min(G.v0, p[1]); G.v1 = Math.max(G.v1, p[1]);
      G.z0 = Math.min(G.z0, p[2]); G.z1 = Math.max(G.z1, p[2]);
    };
    const sbox = (x, y) => { G.sx0 = Math.min(G.sx0, x); G.sx1 = Math.max(G.sx1, x); G.sy0 = Math.min(G.sy0, y); G.sy1 = Math.max(G.sy1, y); ext(x, y); };
    const ctr = g.ctr && R(g.ctr);
    for (const f of g.faces) {
      const p = f.p.map(R);
      p.forEach(box);
      let n;
      if (f.n) { const [nu, nv] = rotQ(f.n[0], f.n[1], rot); n = [nu, nv, f.n[2] || 0]; }
      else {
        n = newell(p);
        if (ctr) {
          const cx = p.reduce((a, q) => a + q[0], 0) / p.length, cy = p.reduce((a, q) => a + q[1], 0) / p.length, cz = p.reduce((a, q) => a + q[2], 0) / p.length;
          if (n[0] * (cx - ctr[0]) + n[1] * (cy - ctr[1]) + n[2] * (cz - ctr[2]) < 0) n = [-n[0], -n[1], -n[2]];
        }
      }
      let vis = n[0] + n[1] + DZ * n[2];
      if (f.o.two && vis < 0) { n = [-n[0], -n[1], -n[2]]; vis = -vis; }
      if (vis <= 1e-6) continue;
      const s = p.map(S);
      s.forEach(q => sbox(q[0], q[1]));
      G.faces.push({ s, n, c: f.c, o: f.o });
    }
    for (const w of g.wins) {
      const [nu, nv] = rotQ(w.n[0], w.n[1], rot);
      if (nu + nv <= 1e-6) continue;
      const p = R(w.p), s = S(p);
      sbox(s[0], s[1]); sbox(s[0] + w.w, s[1] + w.h + 2);
      G.wins.push({ s, side: nu > nv ? "r" : "l", w });
    }
    if (g.blob) {
      const p = R(g.blob.p), s = S(p), rr = g.blob.r * 45;
      box([p[0] - g.blob.r, p[1] - g.blob.r, p[2] - g.blob.r]); box([p[0] + g.blob.r, p[1] + g.blob.r, p[2] + g.blob.r]);
      sbox(s[0] - rr * 1.2, s[1] - rr * 1.2); sbox(s[0] + rr * 1.2, s[1] + rr * 1.2);
      G.blob = { s, r: rr, c: g.blob.c, o: g.blob.o };
    }
    if (!G.faces.length && !G.wins.length && !G.blob) continue;
    G.d = (G.u0 + G.u1) / 2 + (G.v0 + G.v1) / 2 + DZ * (G.z0 + G.z1) / 2;
    gs.push(G);
  }
  // topological depth sort over groups whose screen boxes overlap
  const n = gs.length, before = gs.map(() => []), indeg = new Array(n).fill(0);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const a = gs[i], c = gs[j];
    if (a.sx1 < c.sx0 || c.sx1 < a.sx0 || a.sy1 < c.sy0 || c.sy1 < a.sy0) continue;
    const r = order(a, c);
    if (r < 0) { before[i].push(j); indeg[j]++; } else if (r > 0) { before[j].push(i); indeg[i]++; }
  }
  const done = new Array(n).fill(false), seq = [];
  for (let k = 0; k < n; k++) {
    let best = -1;
    for (let i = 0; i < n; i++) if (!done[i] && indeg[i] === 0 && (best < 0 || gs[i].d < gs[best].d)) best = i;
    if (best < 0) for (let i = 0; i < n; i++) if (!done[i] && (best < 0 || gs[i].d < gs[best].d)) best = i;   // cycle: nearest first
    done[best] = true; seq.push(gs[best]);
    for (const j of before[best]) indeg[j]--;
  }
  if (!isFinite(mnx)) { mnx = mny = 0; mxx = mxy = 1; }
  const ox = 2 - Math.floor(mnx), oy = 2 - Math.floor(mny);
  const W = Math.ceil(mxx) + ox + 2, H = Math.ceil(mxy) + oy + 2;
  const buf = new Uint32Array(W * H), gbuf = new Int16Array(W * H).fill(-1);
  const snowy = season === "winter";
  seq.forEach((G, gi) => {
    for (const f of G.faces) {
      let c = rgb(f.c);
      if (snowy && f.n[2] > 0.5 && !f.o.nosnow) c = mix(c, SNOW, f.o.snow ?? 0.85);
      const k = f.o.flat ? 1 : Math.max(0.3, Math.min(1.12, 0.5 + 0.5 * (LIGHT[0] * f.n[0] + LIGHT[1] * f.n[1] + LIGHT[2] * f.n[2])));
      c = mul(c, k);
      const glow = night && f.o.glow;
      const fin = (q) => glow ? rgb(f.o.glow) : night ? [q[0] * NIGHT[0], q[1] * NIGHT[1], q[2] * NIGHT[2]] : q;
      const pts = f.s.map(([x, y]) => [x + ox, y + oy]);
      let tex = null;
      if (f.o.tex && !glow) { const t = f.o.tex, base = c; tex = (x, y) => pack(fin(t(x - ox, y - oy, base))); }
      fillPoly(buf, W, H, pts, pack(fin(c)), gi, gbuf, tex);
    }
    for (const { s, side, w } of G.wins) {        // window stamps follow the 2:1 slope of their wall
      const lit = night && w.lit;
      let c = lit ? rgb(w.glow || (hash(s[0] * 7 + s[1], 3) < 0.5 ? "#ffd86a" : "#ffeaa6")) : mul(rgb(w.c), side === "r" ? 0.72 : 0.9);
      if (night && !lit) c = [c[0] * NIGHT[0], c[1] * NIGHT[1], c[2] * NIGHT[2]];
      const pc = pack(c), x0 = Math.round(s[0] + ox), y0 = Math.round(s[1] + oy);
      for (let dx = 0; dx < w.w; dx++) {
        const sl = side === "l" ? (dx >> 1) : -(dx >> 1);
        for (let dy = 0; dy < w.h; dy++) {
          const x = x0 + dx, y = y0 + dy + sl - w.h;
          if (x >= 0 && y >= 0 && x < W && y < H && buf[y * W + x] && gbuf[y * W + x] === gi) buf[y * W + x] = pc;
        }
      }
    }
    if (G.blob) {                                   // lumpy lit sphere with ordered dither
      const { s, r, o } = G.blob, cx = s[0] + ox, cy = s[1] + oy, seed = o.seed || 0;
      let base = rgb(G.blob.c);
      if (snowy && !o.nosnow) base = mix(base, SNOW, o.snow ?? 0.45);
      const bay = [0, 0.5, 0.75, 0.25];
      for (let y = Math.floor(cy - r * 1.2); y <= Math.ceil(cy + r * 1.2); y++) for (let x = Math.floor(cx - r * 1.2); x <= Math.ceil(cx + r * 1.2); x++) {
        if (x < 0 || y < 0 || x >= W || y >= H) continue;
        const dx = x + 0.5 - cx, dy = y + 0.5 - cy, a = Math.atan2(dy, dx);
        const rr = r * (1 + (o.lumpy ?? 0.14) * Math.sin(5 * a + seed) + 0.06 * Math.sin(11 * a + seed * 3));
        const d = Math.hypot(dx, dy) / rr;
        if (d > 1) continue;
        const nz = Math.sqrt(1 - d * d), nx = dx / rr, ny = dy / rr;
        let l = 0.62 + 0.5 * (-0.45 * nx - 0.55 * ny + 0.7 * nz) - 0.25;
        l = Math.floor(l * 4 + bay[(x & 1) + 2 * (y & 1)]) / 4;
        let c = mul(base, 0.55 + Math.max(0, Math.min(1, l)) * 0.6);
        if (o.speck && hash(x * 31 + y * 17, seed) < o.speck) c = mix(c, o.speckC || "#ffffff", 0.7);
        if (night && !o.glow) c = [c[0] * NIGHT[0], c[1] * NIGHT[1], c[2] * NIGHT[2]];
        buf[y * W + x] = pack(c); gbuf[y * W + x] = gi;
      }
    }
  });
  if (outline) {                                   // darken the silhouette into a sprite outline
    const edge = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!buf[i]) continue;
      if (x === 0 || y === 0 || x === W - 1 || y === H - 1 || !buf[i - 1] || !buf[i + 1] || !buf[i - W] || !buf[i + W]) edge.push(i);
    }
    for (const i of edge) { const c = unpack(buf[i]); buf[i] = pack(mul(c, 0.5)); }
  }
  const alpha = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) alpha[i] = buf[i] ? 1 : 0;
  const proj = (p) => { const s = S(R(p)); return [s[0], s[1]]; };
  const out = { w: W, h: H, ox, oy, alpha, buf, top: oy,
    smoke: b.meta.smoke.map(e => ({ ...e, s: proj(e.p) })), rotor: b.meta.rotor && { s: proj(b.meta.rotor.p), len: b.meta.rotor.len } };
  if (canvas) out.canvas = toCanvas(buf, W, H);
  return out;
}

export function toCanvas(buf, W, H) {
  const cv = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(W, H) : Object.assign(document.createElement("canvas"), { width: W, height: H });
  const g = cv.getContext("2d");
  const img = g.createImageData(W, H);
  new Uint32Array(img.data.buffer).set(buf);
  g.putImageData(img, 0, 0);
  return cv;
}
