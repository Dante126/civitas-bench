// Civitas pixel view: an OpenTTD-style 2D isometric renderer with the same interface as CityScene (scene.js).
// Ground tiles are baked into one canvas per (frame, rotation, night, overlay); buildings, trees, vehicles and smoke
// are depth-sorted and blitted from cached sprites every animation frame. Sprites come from recipes.js / sprites.js.
// Around the city a distant view (one quarter scale, fading into haze) shows the countryside, the river and highways
// running on, villages, and in region runs the neighbouring cities at their place in the layout.
import { TW, TH, ZS, hash, rotQ, project, paint, fillPoly, toCanvas, Builder, mix, rgb } from "./sprites.js";
import * as R from "./recipes.js";
import { LifeSim } from "./lifesim.js";
import { OVERLAY_BY_ID, OVERLAY_RAMPS, DISTRICT_COLORS, SPECIALS } from "./scene.js";

const MARGIN = 12;                                  // decorative land around the playable map (as in scene.js)
const ZOOMS = [0.125, 0.25, 0.5, 1, 2, 3, 4];
const Z_HALF = ZOOMS.indexOf(0.5), Z_ONE = ZOOMS.indexOf(1);
const BUS_COLORS = ["#e07bd8", "#4aa3ff", "#ffb454", "#5fd38d", "#ff6b5e", "#f5d547"];
const NATIVE_SIZE = ">";
const BUILD_MS = 1600, DUST_MS = 900;
const GAP = 36;                                     // tiles of countryside between neighbouring cities in the distant view
const FAR = 4;                                      // the distant view is painted at 1/FAR scale
const HAZE = { day: "#c9d6de", night: "#0d1422" };
const SIGNAL_C = { g: "#3cf06a", y: "#ffc400", r: "#ff3b30" };
const hex = (c) => "#" + rgb(c).map(v => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, "0")).join("");
const layerStr = (f, o) => o.src ? o.src(f) : (f.layers && f.layers[o.layer]);
function ramp(stops, t) {
  t = Math.max(0, Math.min(1, t));
  const n = stops.length - 1, i = Math.min(n - 1, Math.floor(t * n));
  return hex(mix(stops[i], stops[i + 1], t * n - i));
}

export class PixelScene {
  constructor(container, { onHover, onClick } = {}) {
    this.container = container; this.onHover = onHover; this.onClick = onClick;
    this.overlay = "none"; this.night = false; this.quality = "high";
    this.rot = 0; this.zi = Z_ONE; this.cam = { x: 0, y: 0 }; this.camAnim = null;
    this.frame = null; this.meta = null; this.active = true;
    this.cache = new Map(); this.anims = new Map(); this.dust = []; this.hover = -1; this.selected = -1; this.hl = null;
    this.cv = document.createElement("canvas");
    this.cv.className = "pixelcanvas";
    this.cv.style.imageRendering = "pixelated";
    container.appendChild(this.cv);
    this.g = this.cv.getContext("2d");
    this.life = new LifeSim(this);
    this.waterFrame = 0; this._lastWater = 0; this._autoT = 0;
    this._bind();
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this._t0 = performance.now(); this._last = this._t0;
    this._raf = requestAnimationFrame(() => this._loop());
  }

  // ------------------------------------------------------------ host interface for LifeSim
  get margin() { return MARGIN; }
  charAt(f, x, y) {                                 // terrain beyond the map mirrors the edge rows (scene.js:_outerChar)
    const W = this.W, H = this.H;
    const cx = Math.min(W - 1, Math.max(0, x)), cy = Math.min(H - 1, Math.max(0, y));
    const ch = f.tiles[cy * W + cx];
    const across = (x < 0 || x >= W) !== (y < 0 || y >= H);
    if (across && ch === "H") return "#";
    if (across && ch === "@") return "r";
    if (x >= 0 && x < W && (ch === "~" || ch === "=" || ch === "Y")) return "~";
    return ".";
  }
  busColor(k) { return BUS_COLORS[k % BUS_COLORS.length]; }
  airports() {                                      // other region airports (centres, tiles from this city): flight destinations
    const out = [];
    if (!this.region || !this.city) return out;
    for (const c of this._allCities().slice(1)) for (const [a, fw, fh] of c.C.f.multi || [])
      if (c.C.f.tiles[a] === ">") out.push([c.ox + a % this.W + (fw - 1) / 2, c.oy + Math.floor(a / this.W) + (fh - 1) / 2]);
    return out;
  }

  // ------------------------------------------------------------ geometry
  _rt(x, y) { return rotQ(x - this.cxm, y - this.cym, this.rot); }
  world(x, y, z = 0) { const [u, v] = this._rt(x, y); return project(u, v, z); }
  depth(x, y) { const [u, v] = this._rt(x, y); return u + v; }
  toMap(px, py) {                                   // world pixels (z = 0) -> map tile coordinates
    const a = px / (TW / 2), b = py / (TH / 2), u = (a + b) / 2, v = (b - a) / 2;
    const [x, y] = rotQ(u, v, (4 - this.rot) & 3);
    return [x + this.cxm, y + this.cym];
  }
  get scale() { return ZOOMS[this.zi] * this.dpr; }
  _screenToWorld(cx, cy) {
    const r = this.cv.getBoundingClientRect(), s = ZOOMS[this.zi];
    const [ox, oy] = this.world(this.cam.x, this.cam.y);
    return [ox + (cx - r.left - r.width / 2) / s, oy + (cy - r.top - r.height / 2) / s];
  }

  // ------------------------------------------------------------ meta / frames
  setMeta(meta) {
    this.meta = meta; this.W = meta.width; this.H = meta.height;
    this.cxm = (this.W - 1) / 2; this.cym = (this.H - 1) / 2;
    this.frame = null; this.ground = null; this.far = null; this._farKey = null; this.objects = []; this.anims.clear(); this.dust = [];
    this.life = new LifeSim(this);
    this.resetView(false);
  }
  setFrame(frame, prev, { animate = true } = {}) {
    if (!this.meta) return;
    const now = performance.now(), T = frame.tiles, L = frame.levels;
    if (animate && prev && prev.tiles.length === T.length) {
      for (let i = 0; i < T.length; i++) {
        const was = prev.tiles[i], is = T[i];
        const built = ("RCIOM".includes(is) && +L[i] > 0) || (SPECIALS.includes(is) && is !== "H");
        const had = ("RCIOM".includes(was) && +prev.levels[i] > 0) || (SPECIALS.includes(was) && was !== "H");
        if (built && (was !== is || +prev.levels[i] < +L[i])) this.anims.set(i, now);
        else if (had && !built) this.dust.push({ x: i % this.W, y: Math.floor(i / this.W), t0: now });
      }
    }
    this.frame = frame;
    this.season = frame.season || null;
    this.wind = frame.wind || [1, 0];
    this.city = this._makeCity(frame, this.meta.resources);
    this.part = this.city.part; this.size = this.city.size;
    this.life.setFrame(frame);
    this._dirty = true;
  }
  // everything the tile rules need about one city: its frame, resources and multi-tile buildings
  _makeCity(f, res) {
    const W = this.W, H = this.H, part = new Set(), size = new Map();
    for (const [a, fw, fh] of f.multi || []) {
      size.set(a, [fw, fh]);
      for (let j = 0; j < fh; j++) for (let k = 0; k < fw; k++) if (j || k) part.add(a + j * W + k);
    }
    return { f, W, H, res, part, size, ch: (x, y) => x >= 0 && y >= 0 && x < W && y < H ? f.tiles[y * W + x] : this.charAt(f, x, y) };
  }
  _ch(x, y) { return this.city.ch(x, y); }
  // region runs: app.js passes the raw region frame so neighbouring cities can be drawn in the distant view
  setRegion(r) { this.region = r && r.meta && r.frame && r.frame.cities ? r : null; }

  // ------------------------------------------------------------ sprite cache
  _sprite(key, build, o = {}) {
    const full = `${key}|${this.rot}|${this.night ? 1 : 0}|${this.season || ""}`;
    let s = this.cache.get(full);
    if (s === undefined) {
      const b = new Builder();
      if (build(b) === false || !b.groups.length) s = null;
      else s = paint(b, { rot: o.rot ?? this.rot, night: this.night, season: this.season, outline: o.outline ?? true });
      this.cache.set(full, s);
      if (this.cache.size > 6000) this.cache.clear();
    }
    return s;
  }

  // ------------------------------------------------------------ ground
  _tileSeed(x, y) { return x >= 0 && y >= 0 && x < this.W && y < this.H ? y * this.W + x : 50000 + (y + 50) * 997 + x; }
  // countryside: smooth value noise for forests, brick-offset parcels of fields and meadow elsewhere
  _noise(x, y) {
    const oct = (s, k) => {
      const gx = Math.floor(x / s), gy = Math.floor(y / s), u = x / s - gx, v = y / s - gy, sm = (t) => t * t * (3 - 2 * t);
      const h = (a, b) => hash(a * 7919 + b * 104729, k), a = sm(u), b = sm(v);
      return (h(gx, gy) * (1 - a) + h(gx + 1, gy) * a) * (1 - b) + (h(gx, gy + 1) * (1 - a) + h(gx + 1, gy + 1) * a) * b;
    };
    return 0.68 * oct(11, 81) + 0.32 * oct(4, 82);
  }
  _parcel(x, y) {                                    // [value, row, column] of the 6x6 field parcel holding (x, y)
    const row = Math.floor((y + 120) / 6), col = Math.floor((x + 120 + Math.floor(hash(row, 73) * 6)) / 6);
    return [hash(col * 131 + row, 70), row, col];
  }
  _groundParams(x, y, C = this.city) {
    const f = C.f, W = C.W, H = C.H, inside = x >= 0 && y >= 0 && x < W && y < H, i = inside ? y * W + x : -1;
    const ch = C.ch(x, y), seed = this._tileSeed(x, y), v = Math.floor(hash(seed, 1) * 4);
    const isWater = (c) => c === "~" || c === "=" || c === "Y";
    const M = MARGIN, inArea = (xx, yy) => xx >= -M && yy >= -M && xx < W + M && yy < H + M;
    if (isWater(ch)) {
      let mask = 0;
      [[1, 0], [0, 1], [-1, 0], [0, -1]].forEach(([dx, dy], k) => { if (inArea(x + dx, y + dy) && !isWater(C.ch(x + dx, y + dy))) mask |= 1 << k; });
      const dirty = inside && f.layers && f.layers.river ? Math.round(+f.layers.river[i] / 9 * 4) : 0;
      if (ch === "=") {
        const road = (c) => "#=H".includes(c);
        return { kind: "bridge", mask, dirty, ew: road(C.ch(x - 1, y)) || road(C.ch(x + 1, y)), anim: false };
      }
      return { kind: "water", mask, dirty, anim: ch === "~" };
    }
    if (ch === "#" || ch === "H") {
      let mask = 0;
      [[1, 0], [0, 1], [-1, 0], [0, -1]].forEach(([dx, dy], k) => { if (inArea(x + dx, y + dy) && "#=H".includes(C.ch(x + dx, y + dy))) mask |= 1 << k; });
      const cls = !inside || ch === "H" ? 3 : f.road_class ? Math.max(1, +f.road_class[i]) : 1;
      return { kind: "road", mask, cls, v, outer: !inside };
    }
    if (ch === "r") return { kind: "rail", ew: x < 0 || x >= W, v };
    if (ch === "P") return { kind: "park", v };
    if ("rciom".includes(ch)) return { kind: "lot", zone: ch, v };
    if ("RCIOM".includes(ch)) return +f.levels[i] > 0 ? { kind: "dev", zone: ch, v, q: this._facing(x, y, i, C) } : { kind: "lot", zone: ch.toLowerCase(), v };
    if (SPECIALS.includes(ch) && ch !== "|" && ch !== "Z") return { kind: "dev", zone: "S", v };
    if (inside) {
      const rs = C.res ? +C.res[i] : 0;
      if (rs === 4 && hash(i, 2) < 0.35) return { kind: "oil", v };
      const hh = hash(i, 2);
      return { kind: "grass", v, fertile: rs === 1, flowers: hh >= 0.23 && hh < 0.36 };
    }
    return this._countryGround(x, y);
  }
  _countryGround(x, y) {                            // fields, meadows and woods floors outside the cities
    const v = Math.floor(hash(this._tileSeed(x, y), 1) * 4), n = this._noise(x, y), [b, row, col] = this._parcel(x, y);
    if (n > 0.6) return { kind: "grass", v, dark: true };
    const dir = Math.floor(hash(col * 7 + row, 71) * 2);
    if (b < 0.2) return { kind: "field", crop: "crop", dir, v };
    if (b < 0.29) return { kind: "field", crop: "plough", dir, v };
    return { kind: "grass", v, flowers: b > 0.86 };
  }
  _groundSprite(p, frameNo = 0) {
    const q = { ...p, season: this.season, frame: p.anim ? frameNo : 0 };
    delete q.anim;
    return this._sprite("g" + JSON.stringify(q), (b) => { b.groups = R.ground(q).groups; }, { outline: false });
  }
  _clipDiamond(s) {                                 // keep only the tile's own z=0 diamond: for re-drawing animated water
    if (s._clip) return s._clip;
    const W = s.w, H = s.h, mask = new Uint32Array(W * H), g = new Int16Array(W * H);
    fillPoly(mask, W, H, [[s.ox, s.oy - TH / 2], [s.ox + TW / 2, s.oy], [s.ox, s.oy + TH / 2], [s.ox - TW / 2, s.oy]], 1, 0, g, null);
    const buf = new Uint32Array(W * H);
    for (let i = 0; i < W * H; i++) if (mask[i]) buf[i] = s.buf[i];
    s._clip = toCanvas(buf, W, H);
    return s._clip;
  }
  _tiles() {                                        // every tile of the area, back to front for this rotation
    const key = `${this.W}x${this.H}r${this.rot}`;
    if (this._tilesKey === key) return this._tileList;
    const M = MARGIN, list = [];
    for (let y = -M; y < this.H + M; y++) for (let x = -M; x < this.W + M; x++) list.push([x, y, this.depth(x, y)]);
    list.sort((a, b) => a[2] - b[2]);
    this._tilesKey = key; this._tileList = list;
    return list;
  }
  _bakeGround() {
    const M = MARGIN, W = this.W, H = this.H, f = this.frame;
    const corners = [[-M - 0.5, -M - 0.5], [W + M - 0.5, -M - 0.5], [-M - 0.5, H + M - 0.5], [W + M - 0.5, H + M - 0.5]].map(([x, y]) => {
      const [u, v] = rotQ(x - this.cxm, y - this.cym, this.rot); return project(u, v);
    });
    const x0 = Math.floor(Math.min(...corners.map(c => c[0]))) - 4, x1 = Math.ceil(Math.max(...corners.map(c => c[0]))) + 4;
    const y0 = Math.floor(Math.min(...corners.map(c => c[1]))) - 24, y1 = Math.ceil(Math.max(...corners.map(c => c[1]))) + 30;
    if (!this.ground || this.ground.width !== x1 - x0 || this.ground.height !== y1 - y0) {
      this.ground = document.createElement("canvas"); this.ground.width = x1 - x0; this.ground.height = y1 - y0;
    }
    this.gx0 = x0; this.gy0 = y0;
    const g = this.ground.getContext("2d");
    g.clearRect(0, 0, this.ground.width, this.ground.height);
    const on = this.overlay !== "none";
    const flooded = new Set(f.flooded || []);
    this.waterTiles = [];
    for (const [x, y] of this._tiles()) {
      const p = this._groundParams(x, y), s = this._groundSprite(p, this.waterFrame);
      const [px, py] = this.world(x, y);
      if (s) g.drawImage(s.canvas, px - s.ox - x0, py - s.oy - y0);
      if (p.anim && !on) this.waterTiles.push({ p, px, py });
      const inside = x >= 0 && y >= 0 && x < W && y < H, i = inside ? y * W + x : -1;
      if (on) {
        const c = inside ? this._overlayColor(i) : null;
        if (c) this._blitTint(g, c, px - x0, py - y0, 0.72);
      }
      if (inside && flooded.has(i)) this._blitTint(g, "#5a7ab0", px - x0, py - y0, 0.6);
    }
    this._groundKey = this._gkey();
  }
  _blitTint(g, color, px, py, alpha) {
    const s = this._sprite(`tint|${color}`, (b) => { b.groups = R.ground({ kind: "tint", color }).groups; }, { outline: false, rot: 0 });
    g.globalAlpha = alpha; g.drawImage(s.canvas, px - s.ox, py - s.oy); g.globalAlpha = 1;
  }
  _animateWater() {                                  // palette-cycle style shimmer: redraw river tiles every few ticks
    if (!this.ground || !this.waterTiles || !this.waterTiles.length) return;
    const g = this.ground.getContext("2d");
    for (const { p, px, py } of this.waterTiles) {
      const s = this._groundSprite(p, this.waterFrame);
      if (s) g.drawImage(this._clipDiamond(s), px - s.ox - this.gx0, py - s.oy - this.gy0);
    }
  }
  _gkey() { return [this.frame && this.frame.month, this.frame && this.frame.tiles.length, this.rot, this.night, this.overlay, this.season, this.W].join("|"); }

  // ------------------------------------------------------------ overlays (colours match scene.js:_overlayColor)
  _overlayColor(i) {
    const f = this.frame, ch = f.tiles[i], zone = "rciomRCIOM".includes(ch), o = OVERLAY_BY_ID[this.overlay];
    const road = "#H".includes(ch) ? "#3d4148" : null, grey = "#9aa19a";
    if (o && o.districts) { const d = f.districts ? +f.districts[i] : 0; return d ? DISTRICT_COLORS[d] : (road || grey); }
    if (o && o.cat) { const r = this.meta.resources ? +this.meta.resources[i] : 0; return r ? o.cat[r] : (road || grey); }
    const str = o && layerStr(f, o);
    if (!str) return road || grey;
    if (o.water && !"~=Y".includes(ch)) return road || grey;
    if (o.water) return ramp(OVERLAY_RAMPS[o.ramp], +str[i] / 9);
    if (o.roads && !"#=H".includes(ch)) return grey;
    if (o.mask && !o.mask.includes(ch)) return "#H=".includes(ch) ? "#3d4148" : grey;
    if (o.ramp) return ramp(OVERLAY_RAMPS[o.ramp], +str[i] / 9);
    if (!zone) return road || grey;
    return str[i] === "1" ? "#34c759" : "#ff3b30";
  }
  setOverlay(mode) { this.overlay = mode; this._dirty = true; }

  // ------------------------------------------------------------ objects on tiles
  _facing(x, y, i, C = this.city) {
    const road = (xx, yy) => "#=H".includes(C.ch(xx, yy));
    if (road(x, y + 1)) return 0; if (road(x - 1, y)) return 1; if (road(x, y - 1)) return 2; if (road(x + 1, y)) return 3;
    return Math.floor(hash(i, 2) * 4);
  }
  _facingWater(x, y, i, C = this.city) {           // harbours put their quay and crane on the river side
    const wet = (xx, yy) => "~=Y".includes(C.ch(xx, yy));
    if (wet(x, y + 1)) return 0; if (wet(x - 1, y)) return 1; if (wet(x, y - 1)) return 2; if (wet(x + 1, y)) return 3;
    return this._facing(x, y, i, C);
  }
  _objParams(x, y, C = this.city) {
    const f = C.f, W = C.W, H = C.H, inside = x >= 0 && y >= 0 && x < W && y < H, i = inside ? y * W + x : -1;
    const ch = C.ch(x, y), season = this.season, near = C === this.city;
    if (inside && C.part.has(i)) return null;
    if (inside && "RCIOM".includes(ch) && +f.levels[i] > 0) {
      const c = { ch, lv: +f.levels[i], vr: f.variant ? +f.variant[i] : 0, dn: f.density ? +f.density[i] : 1, v: Math.floor(hash(i, 5) * 8),
        hist: f.layers && f.layers.historic && +f.layers.historic[i] >= 5 };
      const q = this._facing(x, y, i, C);
      return { key: `b|${ch}|${c.lv}|${c.vr}|${c.dn}|${c.v}|${q}|${c.hist ? 1 : 0}`, build: (b) => { b.turn(q); R.building(b, { ...c, season }); }, build0: true };
    }
    if (inside && SPECIALS.includes(ch)) {
      if (ch === "H" && !(x === 0 || y === 0 || x === W - 1 || y === H - 1)) return null;
      const [fw, fh] = C.size.get(i) || [1, 1];
      const k = NATIVE_SIZE.includes(ch) ? 1 : Math.min(fw, fh) * 0.92, kz = k > 1 ? 1 + (k - 1) * 0.55 : 1;
      const q = fw > 1 || fh > 1 || "YH|".includes(ch) ? 0 : ch === "&" ? this._facingWater(x, y, i, C) : this._facing(x, y, i, C);
      return { key: `s|${ch}|${fw}|${fh}|${q}`, fw, fh, build: (b) => { b.scale(k, kz).turn(q); R.special(b, ch, i % 7, season); }, build0: ch !== "H" };
    }
    if (ch === "P") { const v = Math.floor(hash(i, 5) * 8); return { key: `p|${v}`, build: (b) => R.park(b, v, season) }; }
    if (ch === "#" && inside) {
      if (!near) return null;
      const stop = this._stops && this._stops.get(i);
      const sig = this.life.signals.has(i);
      if (stop) return { key: `stop|${stop}|${sig ? 1 : 0}`, signal: sig, build: (b) => { R.busStop(b, stop); if (sig) R.signals(b); } };
      if (sig) return { key: "signals", signal: true, build: (b) => R.signals(b) };
      const road = (xx, yy) => "#=H".includes(C.ch(xx, yy));
      const ew = road(x - 1, y) || road(x + 1, y), ns = road(x, y - 1) || road(x, y + 1);
      if (ew !== ns && hash(i, 2) < 0.28) {
        const s = hash(i, 3) < 0.5 ? 1 : -1, lx = ew ? 0 : 0.44 * s, ly = ew ? 0.44 * s : 0;
        return { key: `lamp|${lx}|${ly}`, lamp: true, build: (b) => R.lamp(b, lx, ly) };
      }
      return null;
    }
    if (ch !== ".") return null;
    if (inside) {
      const rs = C.res ? +C.res[i] : 0, v = Math.floor(hash(i, 6) * 64);
      const hh = hash(v, 2), h3 = hash(v, 3);
      if (rs !== 2 && rs !== 3 && hh >= 0.23) return null;
      return { key: `w|${v}|${rs}`, build: (b) => R.wild(b, v, hh, h3, season, rs) };
    }
    return this._countryObj(x, y);
  }
  _countryObj(x, y) {                               // woods, lone trees and farms outside the cities
    const season = this.season, n = this._noise(x, y), [b0] = this._parcel(x, y), seed = this._tileSeed(x, y), h1 = hash(seed, 1), v = Math.floor(hash(seed, 6) * 32);
    if (n > 0.6) {                                   // forest, denser towards its heart
      if (h1 < 0.55 + (n - 0.6) * 3) { const k = Math.min(3, 1 + Math.floor((n - 0.6) * 12 + h1 * 1.5)); return { key: `wd|${v}|${k}`, build: (b) => R.woods(b, v, k, season) }; }
    } else if (n > 0.52 && h1 < 0.25) return { key: `wd|${v}|1`, build: (b) => R.woods(b, v, 1, season) };   // forest edge
    else if (b0 >= 0.29 && h1 < 0.03) return { key: `wd|${v}|1`, build: (b) => R.woods(b, v, 1, season) };  // lone tree in a meadow
    else if (b0 < 0.2 && h1 > 0.985) return { key: "farm", build: (b) => R.farm(b) };
    return null;
  }
  _buildObjects() {
    const W = this.W, f = this.frame;
    this._stops = new Map();
    (f.bus_lines || []).forEach((line, k) => { for (const [x, y] of line.stops) this._stops.set(y * W + x, BUS_COLORS[k % BUS_COLORS.length]); });
    const objs = [];
    this.lamps = [];
    for (const [x, y] of this._tiles()) {
      const p = this._objParams(x, y);
      if (!p) continue;
      const s = this._sprite(p.key, p.build);
      if (!s) continue;
      const fw = p.fw || 1, fh = p.fh || 1, cx = x + (fw - 1) / 2, cy = y + (fh - 1) / 2;
      const inside = x >= 0 && y >= 0 && x < W && y < this.H, i = inside ? y * W + x : -1;
      const [px, py] = this.world(cx, cy);
      objs.push({ s, px, py, d: this.depth(cx, cy), tile: i, build0: p.build0, fw, fh, signal: p.signal ? [x, y] : null });
      if (p.lamp) this.lamps.push([px, py]);
    }
    this.objects = objs;                              // already back to front (tiles are sorted by depth)
    this.emitters = [];
    for (const o of objs) for (const e of o.s.smoke) this.emitters.push({ x: o.px + e.s[0], y: o.py + e.s[1], rate: e.rate, dark: e.dark, fire: e.fire, tile: o.tile });
    for (const t of f.burning || []) {
      const [px, py] = this.world(t % W, Math.floor(t / W));
      this.emitters.unshift({ x: px, y: py - 8, rate: 1.6, dark: 0.9, fire: true, tile: t });
    }
    this.emitters = this.emitters.slice(0, 120);
  }
  _rebuild() {
    this._bakeGround();
    this._buildObjects();
    if (this.night && this.lamps.length) {            // warm light pools under the street lamps
      const g = this.ground.getContext("2d");
      g.globalCompositeOperation = "lighter";
      for (const [px, py] of this.lamps) {
        const x = px - this.gx0, y = py - this.gy0, gr = g.createRadialGradient(x, y, 1, x, y, 26);
        gr.addColorStop(0, "rgba(255,190,90,0.35)"); gr.addColorStop(1, "rgba(255,190,90,0)");
        g.fillStyle = gr; g.beginPath(); g.ellipse(x, y, 26, 13, 0, 0, Math.PI * 2); g.fill();
      }
      g.globalCompositeOperation = "source-over";
    }
    this._dirty = false;
  }

  // ------------------------------------------------------------ distant view
  _allCities() {                                    // this city and its neighbours, with offsets (tiles) from this one
    const r = this.region, L = r ? r.meta.layout || {} : {}, me = r && L[r.city];
    const out = [{ name: r ? r.city : "", lx: me ? me[0] : 0, ly: me ? me[1] : 0, ox: 0, oy: 0, C: this.city }];
    if (!me) return out;
    for (const c of r.meta.cities) if (c !== r.city && L[c] && r.frame.cities[c])
      out.push({ name: c, lx: L[c][0], ly: L[c][1], ox: (L[c][0] - me[0]) * (this.W + GAP), oy: (L[c][1] - me[1]) * (this.H + GAP),
        C: this._makeCity(r.frame.cities[c], ((r.meta.city_meta || {})[c] || {}).resources) });
    return out;
  }
  // Lay out the land between and around the cities so it makes sense: rivers flow on into the city downstream
  // (or meander out to the horizon), highways and railways are routed between the connector tiles of each link,
  // other edge highways and railways join that network or leave the region, with a village on the way.
  // Every city edge feature first runs straight for MARGIN tiles, as the detailed margin draws it (charAt).
  // Returns Map "x,y" -> { ch: "~" water | "#" road | "=" road bridge | "r" rail | "b" rail bridge | "+" level crossing | "v" house, ... }
  _farPlan(all, B) {
    const W = this.W, H = this.H, M = MARGIN, [bx0, by0, bx1, by1] = B, map = new Map(), K = (x, y) => x + "," + y;
    const inCity = (x, y) => all.some(c => x >= c.ox && y >= c.oy && x < c.ox + W && y < c.oy + H);
    const near = (x, y) => all.some(c => x >= c.ox - M && y >= c.oy - M && x < c.ox + W + M && y < c.oy + H + M);
    const inB = (x, y) => x >= bx0 + 1 && y >= by0 + 1 && x < bx1 - 1 && y < by1 - 1;
    const put = (x, y, f) => { if (inB(x, y) && !inCity(x, y)) map.set(K(x, y), f); };
    const span = (c, row) => {
      let lo = Infinity, hi = -Infinity;
      for (let x = 0; x < W; x++) if ("~=Y".includes(c.C.f.tiles[row * W + x])) { lo = Math.min(lo, x); hi = Math.max(hi, x); }
      return isFinite(lo) ? [lo, hi] : null;
    };
    const nb = (c, dx, dy) => all.find(o => o.lx === c.lx + dx && o.ly === c.ly + dy);
    // ---- rivers (they run north-south through the cities)
    const riverOut = (c, sp, dir) => {
      const edge = dir > 0 ? c.oy + H - 1 : c.oy, ph = c.ox * 0.37 + dir;
      let prev = null;
      for (let k = 1; k < 400; k++) {
        const y = edge + dir * k, m = k <= M ? 0 : Math.round(3 * Math.sin((k - M) * 0.11 + ph) * Math.min(1, (k - M) / 12));
        const lo = c.ox + sp[0] + m, hi = c.ox + sp[1] + m;
        if (!inB(lo, y)) break;
        if (k > M && (near(lo - 1, y) || near(hi + 1, y))) {            // runs into another city: ends in a lake
          const cx = (lo + hi) / 2, cy = y - dir * 3;
          for (let yy = -3; yy <= 3; yy++) for (let xx = -4; xx <= 4; xx++) if ((xx / 4.5) ** 2 + (yy / 3.2) ** 2 <= 1 && !near(Math.round(cx + xx), cy + yy)) put(Math.round(cx + xx), cy + yy, { ch: "~" });
          break;
        }
        const a = prev ? Math.min(prev[0], lo) : lo, b = prev ? Math.max(prev[1], hi) : hi;   // keep the banks joined when it bends
        for (let x = a; x <= b; x++) put(x, y, { ch: "~" });
        prev = [lo, hi];
      }
    };
    for (const c of all) {
      const bot = span(c, H - 1), top = span(c, 0), dn = nb(c, 0, 1), up = nb(c, 0, -1);
      const dnTop = dn && span(dn, 0), upBot = up && span(up, H - 1);
      if (bot && dnTop) {                          // flows on into the next city, bending across the middle of the gap
        const y0 = c.oy + H, y1 = dn.oy - 1;
        let prev = null;
        for (let y = y0; y <= y1; y++) {
          const t = Math.max(0, Math.min(1, (y - y0 - M) / Math.max(1, y1 - y0 - 2 * M))), e = t * t * (3 - 2 * t);
          const lo = c.ox + Math.round(bot[0] + (dnTop[0] - bot[0]) * e), hi = c.ox + Math.round(bot[1] + (dnTop[1] - bot[1]) * e);
          for (let x = prev ? Math.min(prev[0], lo) : lo; x <= (prev ? Math.max(prev[1], hi) : hi); x++) put(x, y, { ch: "~" });
          prev = [lo, hi];
        }
      } else if (bot) riverOut(c, bot, 1);
      if (top && !upBot) riverOut(c, top, -1);
    }
    // ---- highway and railway ends on every city edge (connectors first)
    const ends = [];
    const dirOf = (x, y) => x === 0 ? [-1, 0] : x === W - 1 ? [1, 0] : y === 0 ? [0, -1] : [0, 1];
    for (const c of all) {
      const T = c.C.f.tiles, seen = new Set();
      for (const cn of (c.C.f.metrics && c.C.f.metrics.connectors) || []) {
        ends.push({ c, x: cn.x, y: cn.y, d: dirOf(cn.x, cn.y), kind: cn.kind, to: cn.to }); seen.add(K(cn.x, cn.y));
      }
      const edge = [];
      for (let x = 1; x < W - 1; x++) edge.push([x, 0], [x, H - 1]);
      for (let y = 1; y < H - 1; y++) edge.push([0, y], [W - 1, y]);
      for (const [x, y] of edge) {
        const ch = T[y * W + x];
        if ((ch === "H" || ch === "@") && !seen.has(K(x, y))) ends.push({ c, x, y, d: dirOf(x, y), kind: ch === "H" ? "highway" : "rail", to: null });
      }
    }
    const isKind = (f, kind) => f && (kind === "highway" ? "#=+".includes(f.ch) : "rb+".includes(f.ch));
    const lay = (x, y, kind, ew) => {                // one tile of road or track, bridging water and crossing the other kind
      const f = map.get(K(x, y));
      if (kind === "highway") put(x, y, f && f.ch === "~" ? { ch: "=", ew } : f && "rb".includes(f.ch) ? { ch: "+", railEw: !ew } : f && f.ch === "+" ? f : { ch: "#" });
      else put(x, y, f && f.ch === "~" ? { ch: "b", ew } : f && "#=".includes(f.ch) ? { ch: "+", railEw: ew } : f && f.ch === "+" ? f : { ch: "r" });
    };
    for (const e of ends) {                          // the straight run across the margin
      e.gx = e.c.ox + e.x; e.gy = e.c.oy + e.y;
      for (let k = 1; k <= M; k++) lay(e.gx + e.d[0] * k, e.gy + e.d[1] * k, e.kind, e.d[0] !== 0);
      e.sx = e.gx + e.d[0] * (M + 1); e.sy = e.gy + e.d[1] * (M + 1);
    }
    const walk = (path, kind) => { for (let k = 0; k < path.length; k++) { const [x, y] = path[k], q = path[k + 1] || path[k - 1]; lay(x, y, kind, q ? q[1] === y : true); } };
    // ---- links between cities
    for (const l of (this.region && this.region.frame.links) || []) {
      if (l.kind !== "highway" && l.kind !== "rail") continue;
      const ea = ends.find(e => !e.used && e.c.name === l.a && e.to === l.b && e.kind === l.kind);
      const eb = ends.find(e => !e.used && e.c.name === l.b && e.to === l.a && e.kind === l.kind);
      if (!ea || !eb) continue;
      const path = this._route(map, B, near, ea, (x, y) => x === eb.sx && y === eb.sy, l.kind, [eb.sx, eb.sy]);
      if (path) { walk(path, l.kind); ea.used = eb.used = true; }
    }
    // ---- the rest: out of the region if the way is clear, else join the network
    for (const e of ends) {
      if (e.used) continue;
      let clear = true, n = 0;
      for (let x = e.sx, y = e.sy; inB(x, y); x += e.d[0], y += e.d[1]) if (near(x, y)) { clear = false; break; }
      if (clear) {
        const path = [];
        for (let x = e.sx, y = e.sy; inB(x, y); x += e.d[0], y += e.d[1]) path.push([x, y]);
        walk(path, e.kind);
        if (e.kind === "highway") for (let k = 8; k < 22; k++, n++) {   // a village along the road out of the region
          const [x, y] = path[k] || [];
          if (x == null) break;
          for (const s of [-1, 1]) for (let o = 1; o <= 2; o++) {
            const vx = x + (e.d[1] ? s * o : 0), vy = y + (e.d[0] ? s * o : 0);
            if (!map.has(K(vx, vy)) && !near(vx, vy) && hash(vx * 31 + vy * 7, 44) < (o === 1 ? 0.62 : 0.3)) put(vx, vy, { ch: "v", road: e.d[1] ? [-s, 0] : [0, -s] });
          }
        }
      } else {
        const path = this._route(map, B, near, e, (x, y) => isKind(map.get(K(x, y)), e.kind), e.kind, null);
        if (path) walk(path.slice(0, -1), e.kind);
      }
    }
    return map;
  }
  // Cheapest path from an end's stub over open land: straight runs preferred, no turning on bridges or crossings,
  // never through a city or its margin. goal(x, y) says when to stop; target (optional) guides the search.
  _route(map, B, near, e, goal, kind, target) {
    const [bx0, by0, bx1, by1] = B, GW = bx1 - bx0, GH = by1 - by0, N = GW * GH * 4, K = (x, y) => x + "," + y;
    const D = [[1, 0], [0, 1], [-1, 0], [0, -1]];
    const dist = new Float32Array(N).fill(Infinity), prev = new Int32Array(N).fill(-1);
    const id = (x, y, d) => ((y - by0) * GW + (x - bx0)) * 4 + d;
    const h = (x, y) => target ? Math.abs(x - target[0]) + Math.abs(y - target[1]) : 0;
    const heap = [], push = (f, s) => {
      heap.push([f, s]); let i = heap.length - 1;
      while (i > 0) { const p = (i - 1) >> 1; if (heap[p][0] <= heap[i][0]) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; }
    };
    const pop = () => {
      const top = heap[0], last = heap.pop();
      if (heap.length) {
        heap[0] = last; let i = 0;
        for (;;) { const l = 2 * i + 1, r = l + 1; let m = i;
          if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r;
          if (m === i) break; [heap[m], heap[i]] = [heap[i], heap[m]]; i = m; }
      }
      return top;
    };
    const d0 = D.findIndex(([dx, dy]) => dx === e.d[0] && dy === e.d[1]);
    if (near(e.sx, e.sy)) return null;
    const s0 = id(e.sx, e.sy, d0);
    dist[s0] = 0; push(h(e.sx, e.sy), s0);
    for (let it = 0; heap.length && it < 600000; it++) {
      const [, s] = pop(), d = s & 3, cell = s >> 2, x = bx0 + cell % GW, y = by0 + Math.floor(cell / GW), g = dist[s];
      if (goal(x, y) && !(x === e.sx && y === e.sy && !target)) {
        const path = []; for (let k = s; k >= 0; k = prev[k]) { const c = k >> 2; path.push([bx0 + c % GW, by0 + Math.floor(c / GW)]); }
        return path.reverse();
      }
      const here = map.get(K(x, y)), locked = here && "~=b+".includes(here.ch) || (here && here.ch === (kind === "highway" ? "r" : "#"));
      for (let nd = 0; nd < 4; nd++) {
        if (nd === (d + 2) % 4 || (locked && nd !== d)) continue;
        const nx = x + D[nd][0], ny = y + D[nd][1];
        if (nx <= bx0 || ny <= by0 || nx >= bx1 - 1 || ny >= by1 - 1) continue;
        const f = map.get(K(nx, ny)), isGoal = goal(nx, ny);
        if (!isGoal && near(nx, ny)) continue;
        if (f && f.ch === "v") continue;
        let c = 1 + (nd !== d ? 0.8 : 0);
        if (f && !isGoal) c += f.ch === "~" ? 2 : "#=".includes(f.ch) ? (kind === "highway" ? 1.5 : 4) : "rb".includes(f.ch) ? (kind === "rail" ? 1.5 : 4) : 3;
        const ns = id(nx, ny, nd);
        if (g + c < dist[ns]) { dist[ns] = g + c; prev[ns] = s; push(g + c + h(nx, ny), ns); }
      }
    }
    return null;
  }
  _farKeys() {
    const f = this.frame, W = this.W, H = this.H, T = f.tiles;
    let edges = T.slice(0, W) + T.slice((H - 1) * W);
    for (let y = 0; y < H; y++) edges += T[y * W] + T[y * W + W - 1];
    const r = this.region;
    return { fixed: [this.rot, this.night, this.season, W, H, edges, JSON.stringify((f.metrics || {}).connectors || []), r ? r.city + r.meta.cities.join() : ""].join("|"),
      month: r ? r.frame.month : 0 };
  }
  _mini(s) {                                        // quarter-scale copy with its anchor on a whole pixel
    if (s._mini) return s._mini;
    const px = (FAR - s.ox % FAR) % FAR, py = (FAR - s.oy % FAR) % FAR;
    const cv = document.createElement("canvas");
    cv.width = Math.ceil((s.w + px) / FAR) + 1; cv.height = Math.ceil((s.h + py) / FAR) + 1;
    const g = cv.getContext("2d");
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
    g.drawImage(s.canvas, px / FAR, py / FAR, s.w / FAR, s.h / FAR);
    s._mini = { canvas: cv, ox: (s.ox + px) / FAR, oy: (s.oy + py) / FAR };
    return s._mini;
  }
  _bakeFar() {
    const W = this.W, H = this.H, M = MARGIN, all = this._allCities();
    const PAD = M + (this.region ? 30 : 44);
    const bx0 = Math.min(...all.map(c => c.ox)) - PAD, by0 = Math.min(...all.map(c => c.oy)) - PAD;
    const bx1 = Math.max(...all.map(c => c.ox + W)) + PAD, by1 = Math.max(...all.map(c => c.oy + H)) + PAD;
    const B = this.farBounds = [bx0, by0, bx1, by1];
    const planKey = all.map(c => { const T = c.C.f.tiles; let s = T.slice(0, W) + T.slice((H - 1) * W); for (let y = 0; y < H; y++) s += T[y * W] + T[y * W + W - 1];
      return `${c.name}@${c.ox},${c.oy}:${s}:${JSON.stringify((c.C.f.metrics || {}).connectors || [])}`; }).join("|") + JSON.stringify((this.region && this.region.frame.links) || []);
    if (planKey !== this._planKey) { this._plan = this._farPlan(all, B); this._planKey = planKey; }
    const plan = this._plan, K = (x, y) => x + "," + y;
    const cs = [[bx0 - 0.5, by0 - 0.5], [bx1 - 0.5, by0 - 0.5], [bx0 - 0.5, by1 - 0.5], [bx1 - 0.5, by1 - 0.5]].map(([x, y]) => this.world(x, y));
    const x0 = Math.floor((Math.min(...cs.map(c => c[0])) - 8) / FAR) * FAR, x1 = Math.max(...cs.map(c => c[0])) + 8;
    const y0 = Math.floor((Math.min(...cs.map(c => c[1])) - 220) / FAR) * FAR, y1 = Math.max(...cs.map(c => c[1])) + 16;
    const cw = Math.ceil((x1 - x0) / FAR), chh = Math.ceil((y1 - y0) / FAR);
    if (!this.far || this.far.width !== cw || this.far.height !== chh) { this.far = document.createElement("canvas"); this.far.width = cw; this.far.height = chh; }
    this.fx0 = x0; this.fy0 = y0;
    const g = this.far.getContext("2d");
    g.clearRect(0, 0, cw, chh);
    const poly = (e) => [[bx0 - 0.5 + e, by0 - 0.5 + e], [bx1 - 0.5 - e, by0 - 0.5 + e], [bx1 - 0.5 - e, by1 - 0.5 - e], [bx0 - 0.5 + e, by1 - 0.5 - e]]
      .map(([x, y]) => { const [px, py] = this.world(x, y); return [(px - x0) / FAR, (py - y0) / FAR]; });
    const path = (pts) => { g.moveTo(pts[0][0], pts[0][1]); for (const q of pts.slice(1)) g.lineTo(q[0], q[1]); g.closePath(); };
    g.fillStyle = R.GRASS[this.season] || R.GRASS.null; g.beginPath(); path(poly(0)); g.fill();   // no seams between tiles
    const inDetail = (x, y) => x >= -M && y >= -M && x < W + M && y < H + M;
    const cityAt = (x, y) => all.find(c => x >= c.ox && y >= c.oy && x < c.ox + W && y < c.oy + H);
    const at = (x, y) => { if (inDetail(x, y)) return this.city.ch(x, y); const c = cityAt(x, y); if (c) return c.C.ch(x - c.ox, y - c.oy); const f = plan.get(K(x, y)); return f ? f.ch : "."; };
    const N4 = [[1, 0], [0, 1], [-1, 0], [0, -1]];
    const mask = (x, y, ok) => N4.reduce((m, [dx, dy], k) => m | (ok(at(x + dx, y + dy)) ? 1 << k : 0), 0);
    const land = (c) => !"~=Yb".includes(c), roadish = (c) => "#=H+".includes(c), railish = (c) => "rb+@".includes(c);
    const tiles = [];
    for (let y = by0; y < by1; y++) for (let x = bx0; x < bx1; x++) if (!inDetail(x, y)) tiles.push([x, y, this.depth(x, y)]);
    tiles.sort((a, b) => a[2] - b[2]);
    const objs = [];
    for (const [x, y] of tiles) {
      const c = cityAt(x, y), f = !c && plan.get(K(x, y));
      let gp = null, op = null, fw = 1, fh = 1;
      if (c) {
        const lx = x - c.ox, ly = y - c.oy;
        gp = this._groundParams(lx, ly, c.C); op = this._objParams(lx, ly, c.C);
        if (op) { fw = op.fw || 1; fh = op.fh || 1; }
      } else if (f) {
        if (f.ch === "~") gp = { kind: "water", mask: mask(x, y, land), dirty: 0 };
        else if (f.ch === "#") gp = { kind: "road", mask: mask(x, y, roadish), cls: 3, v: 0, outer: true };
        else if (f.ch === "=") gp = { kind: "bridge", mask: mask(x, y, land), dirty: 0, ew: f.ew };
        else if (f.ch === "r") gp = { kind: "rail", mask: mask(x, y, railish), v: 0 };
        else if (f.ch === "b") gp = { kind: "railbridge", mask: mask(x, y, land), dirty: 0, ew: f.ew };
        else if (f.ch === "+") gp = { kind: "crossing", mask: mask(x, y, roadish), cls: 3, v: 0, outer: true, railEw: f.railEw };
        else if (f.ch === "v") {
          const v = Math.floor(hash(x * 13 + y * 101, 5) * 8), q = f.road[0] > 0 ? 3 : f.road[0] < 0 ? 1 : f.road[1] > 0 ? 0 : 2;
          const big = hash(x * 7 + y, 9) < 0.15;
          gp = { kind: "dev", zone: "R", v: v & 3, q };
          op = { key: `b|R|${big ? 2 : 1}|0|0|${v}|${q}|0`, build: (b) => { b.turn(q); R.building(b, { ch: "R", lv: big ? 2 : 1, vr: 0, dn: 0, v, season: this.season }); } };
        }
      } else { gp = this._countryGround(x, y); op = this._countryObj(x, y); }
      const [px, py] = this.world(x, y);
      const s = gp && this._groundSprite(gp);
      if (s) { const m = this._mini(s); g.drawImage(m.canvas, (px - x0) / FAR - m.ox, (py - y0) / FAR - m.oy); }
      if (op) objs.push([op, x + (fw - 1) / 2, y + (fh - 1) / 2]);
    }
    for (const [op, cx, cy] of objs) {
      const s = this._sprite(op.key, op.build);
      if (!s) continue;
      const m = this._mini(s), [px, py] = this.world(cx, cy);
      g.drawImage(m.canvas, (px - x0) / FAR - m.ox, (py - y0) / FAR - m.oy);
    }
    // haze: a light veil over all of it, thickening towards the horizon
    const hz = this.night ? HAZE.night : HAZE.day, N = 28, step = Math.min(24, PAD - M) / N;
    g.fillStyle = hz; g.globalAlpha = 0.05; g.beginPath(); path(poly(0)); g.fill();
    for (let k = 0; k < N; k++) {
      g.globalAlpha = Math.pow(1 - k / N, 1.8) * 0.9;
      g.beginPath(); path(poly(k * step)); path(poly((k + 1) * step).reverse()); g.fill("evenodd");
    }
    g.globalAlpha = 1;
  }

  // ------------------------------------------------------------ vehicles
  _vehicles(t) {
    const L = this.life, out = [];
    if (!L.roads || this.overlay !== "none") return out;
    const low = this.quality === "low";
    const add = (x, y, z, key, build, alpha = 1) => {
      const s = this._sprite(key, build);
      if (!s) return;
      const [px, py] = this.world(x, y, z);
      out.push({ s, px, py, d: this.depth(x, y) + 0.01, alpha });
    };
    const dirIdx = (dx, dy) => dx > 0 ? 0 : dy > 0 ? 1 : dx < 0 ? 2 : 3;
    L.cars.forEach((c, n) => {
      if (low && n % 2) return;
      const p = L.car(c), d = dirIdx(p.dx, p.dy);
      add(p.x, p.y, p.bridge ? 0.08 : 0, `car|${c.color}|${d}`, (b) => { b.turn(d); R.car(b, c.color); });
    });
    for (const bu of L.buses) {
      const p = L.bus(bu), d = dirIdx(p.dx, p.dy);
      add(p.x, p.y, p.bridge ? 0.08 : 0, `bus|${bu.color}|${d}`, (b) => { b.turn(d); R.bus(b, bu.color); });
    }
    for (const tr of L.trains) for (const k of L.train(tr, t) || []) {
      const col = k.loco ? tr.color : "#eceff1";
      add(k.x, k.y, 0, `wagon|${col}|${k.ew ? 0 : 1}|${k.loco ? 1 : 0}`, (b) => { b.turn(k.ew ? 0 : 1); R.wagon(b, col, k.loco); });
    }
    L.boats.forEach((bo, n) => {
      const p = L.boat(bo, t, n), d = dirIdx(p.dx, p.dy);
      add(p.x, p.y, -0.1 + p.bob * 0.012, `boat|${bo.color}|${bo.sail ? 1 : 0}|${d}`, (b) => { b.turn(d); R.boat(b, bo.color, bo.sail); });
    });
    for (const sh of L.ships) {                      // container ships calling at the harbours
      const p = L.ship(sh);
      if (!p) continue;
      const d = dirIdx(p.dx, p.dy);
      add(p.x, p.y, -0.1, `ship|${sh.color}|${sh.v}|${d}`, (b) => { b.turn(d); R.cargoShip(b, sh.color, sh.v); }, p.fade);
    }
    for (const pl of L.planes) {                     // planes on the ground (landing roll, taxiing, at a stand)
      if (pl.cross) continue;
      const q = L.plane(pl, t);
      if (!q.ground) continue;
      const hd = ((Math.round(q.heading / (Math.PI / 8)) % 16) + 16) % 16;
      add(q.x, q.y, 0.03, `plane|${hd}|0`, (b) => { b.angle(hd * Math.PI / 8); R.plane(b, false); });
    }
    return out;
  }
  _air(t) {                                          // planes and birds: drawn above everything, with ground shadows
    const L = this.life, out = [];
    if (!L.roads || this.overlay !== "none") return out;
    for (const p of L.planes) {
      const q = L.plane(p, t);
      if (q.ground) continue;
      const hd = ((Math.round(q.heading / (Math.PI / 8)) % 16) + 16) % 16;
      const s = this._sprite(`plane|${hd}|${q.big ? 1 : 0}`, (b) => { b.angle(hd * Math.PI / 8); R.plane(b, q.big); });
      const [gx, gy] = this.world(q.x, q.y), [px, py] = this.world(q.x, q.y, q.h);
      out.push({ s, px, py, gx, gy });
    }
    return out;
  }

  // ------------------------------------------------------------ camera
  _clampZoom(z) { return Math.max(0, Math.min(ZOOMS.length - 1, z)); }
  resetView(anim = true) {
    if (!this.W) return;
    const r = this.cv.getBoundingClientRect(), span = (this.W + this.H) * TW / 2;
    let zi = 0;
    for (let k = ZOOMS.length - 1; k >= 0; k--) if (span * ZOOMS[k] <= Math.max(320, r.width) * 1.05) { zi = k; break; }
    this.zi = Math.max(Z_HALF, zi);
    this._goto(this.cxm, this.cym, anim);
  }
  topView() {                                        // zoom right out over the whole region
    const b = this.farBounds, r = this.cv.getBoundingClientRect();
    const span = b ? (b[2] - b[0] + b[3] - b[1]) * TW / 2 * 0.8 : (this.W + this.H) * TW;
    this.zi = 0;
    for (let k = Z_ONE; k >= 0; k--) if (span * ZOOMS[k] <= Math.max(320, r.width)) { this.zi = k; break; }
    this._goto(b ? (b[0] + b[2] - 1) / 2 : this.cxm, b ? (b[1] + b[3] - 1) / 2 : this.cym, true);
  }
  flyTo(x, y) { this._goto(x, y, true); }
  _goto(x, y, anim) {
    if (anim) this.camAnim = { t0: performance.now(), ms: 600, fx: this.cam.x, fy: this.cam.y, tx: x, ty: y };
    else { this.cam.x = x; this.cam.y = y; this.camAnim = null; }
  }
  panBy(dx, dz) {                                    // dx right, dz up the screen, a tenth of the view per step
    const r = this.cv.getBoundingClientRect(), s = ZOOMS[this.zi];
    const [ox, oy] = this.world(this.cam.x, this.cam.y);
    const [x, y] = this.toMap(ox + dx * r.width * 0.1 / s, oy - dz * r.height * 0.1 / s);
    this.cam.x = x; this.cam.y = y; this.camAnim = null;
  }
  rotateBy(a) {
    this.rot = (this.rot + (a > 0 ? 1 : 3)) & 3;
    this._dirty = true;
  }
  zoomBy(f, cx, cy) {
    const zi = this._clampZoom(this.zi + (f < 1 ? 1 : -1));
    if (zi === this.zi) return;
    if (cx != null) {                                // keep the point under the cursor in place
      const [wx, wy] = this._screenToWorld(cx, cy);
      const [mx, my] = this.toMap(wx, wy);
      const k = ZOOMS[this.zi] / ZOOMS[zi];
      this.cam.x = mx + (this.cam.x - mx) * k; this.cam.y = my + (this.cam.y - my) * k;
    }
    this.zi = zi;
  }
  setAutoRotate(on) { this.autoRotate = on; this._autoT = performance.now(); }
  setNight(on) { if (this.night !== on) { this.night = on; this._dirty = true; } }
  setQuality(q) { this.quality = q; }
  cameraFootprint() {
    const r = this.cv.getBoundingClientRect(), [fx, fy] = rotQ(-1, -1, (4 - this.rot) & 3);
    return { x: this.cam.x, y: this.cam.y, angle: Math.atan2(fx, fy), dist: r.width / (TW * ZOOMS[this.zi]) * 1.2 };
  }
  screenshot() { this._draw(); return this.cv.toDataURL("image/png"); }
  setActive(on) {
    this.active = on; this.cv.style.display = on ? "" : "none";
    if (on) { cancelAnimationFrame(this._raf); this._last = performance.now(); this._raf = requestAnimationFrame(() => this._loop()); this.resize(); }
    else cancelAnimationFrame(this._raf);
  }

  // ------------------------------------------------------------ highlights
  setHover(tile) { this.hover = tile == null ? -1 : tile; }
  setSelected(tile) { this.selected = tile == null ? -1 : tile; }
  highlightRect(r) { this.hl = r || null; this.hlT0 = performance.now(); }

  // ------------------------------------------------------------ picking
  pick(cx, cy) {
    if (!this.frame) return -1;
    const [wx, wy] = this._screenToWorld(cx, cy);
    const list = this._drawn || [];
    for (let k = list.length - 1; k >= 0; k--) {
      const o = list[k];
      if (o.tile < 0) continue;
      const sx = Math.floor(wx - (o.px - o.s.ox)), sy = Math.floor(wy - (o.py - o.s.oy));
      if (sx < 0 || sy < 0 || sx >= o.s.w || sy >= o.s.h) continue;
      if (o.s.alpha[sy * o.s.w + sx]) return o.tile;
    }
    const [mx, my] = this.toMap(wx, wy), x = Math.round(mx), y = Math.round(my);
    if (x < 0 || y < 0 || x >= this.W || y >= this.H) return -1;
    let i = y * this.W + x;
    if (this.part.has(i)) for (const [a, [fw, fh]] of this.size) {
      const ax = a % this.W, ay = Math.floor(a / this.W);
      if (x >= ax && x < ax + fw && y >= ay && y < ay + fh) { i = a; break; }
    }
    return i;
  }
  _bind() {
    const el = this.cv;
    let down = null, pending = null;
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    el.addEventListener("pointerdown", (e) => {
      down = { x: e.clientX, y: e.clientY, cx: this.cam.x, cy: this.cam.y, moved: false, b: e.button };
      el.setPointerCapture(e.pointerId); this.camAnim = null;
    });
    el.addEventListener("pointermove", (e) => {
      if (down) {
        const dx = e.clientX - down.x, dy = e.clientY - down.y;
        if (Math.hypot(dx, dy) > 4) down.moved = true;
        if (down.moved) {                          // drag to scroll, like OpenTTD
          const s = ZOOMS[this.zi], [ox, oy] = this.world(down.cx, down.cy);
          const [x, y] = this.toMap(ox - dx / s, oy - dy / s);
          this.cam.x = x; this.cam.y = y;
          el.style.cursor = "grabbing";
          return;
        }
      }
      pending = e;
    });
    el.addEventListener("pointerup", (e) => {
      el.style.cursor = "";
      if (down && !down.moved && down.b === 0) this.onClick && this.onClick(this.pick(e.clientX, e.clientY), e);
      down = null;
    });
    el.addEventListener("pointerleave", () => { this.setHover(-1); this.onHover && this.onHover(-1); });
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      this._wheel = (this._wheel || 0) + e.deltaY;
      if (Math.abs(this._wheel) >= 60) { this.zoomBy(this._wheel < 0 ? 0.85 : 1.18, e.clientX, e.clientY); this._wheel = 0; }
    }, { passive: false });
    this._pollHover = () => {
      if (!pending) return;
      const e = pending; pending = null;
      const t = this.pick(e.clientX, e.clientY); this.setHover(t);
      this.onHover && this.onHover(t, e);
    };
  }
  resize() {
    const w = this.container.clientWidth || 1, h = this.container.clientHeight || 1;
    this.dpr = Math.min(3, window.devicePixelRatio || 1);
    this.cv.width = Math.round(w * this.dpr); this.cv.height = Math.round(h * this.dpr);
    this.cv.style.width = "100%"; this.cv.style.height = "100%";
  }

  // ------------------------------------------------------------ drawing
  _loop() {
    if (!this.active) return;
    const now = performance.now(), dt = Math.min(0.05, (now - this._last) / 1000);
    this._last = now;
    if (this.camAnim) {
      const a = this.camAnim, k = Math.min(1, (now - a.t0) / a.ms), e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
      this.cam.x = a.fx + (a.tx - a.fx) * e; this.cam.y = a.fy + (a.ty - a.fy) * e;
      if (k >= 1) this.camAnim = null;
    }
    if (this.autoRotate && now - this._autoT > 6000) { this._autoT = now; this.rotateBy(1); }
    if (this.frame && this.overlay === "none") this.life.step(dt);
    if (this.frame && now - this._lastWater > 280) { this._lastWater = now; this.waterFrame = (this.waterFrame + 1) & 3; if (!this._dirty) this._animateWater(); }
    this._draw(now);
    this._pollHover && this._pollHover();
    this._raf = requestAnimationFrame(() => this._loop());
  }
  _draw(now = performance.now()) {
    const g = this.g, cw = this.cv.width, ch = this.cv.height;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.fillStyle = this.night ? HAZE.night : HAZE.day; g.fillRect(0, 0, cw, ch);
    if (!this.frame) return;
    if (this._dirty || this._groundKey !== this._gkey()) this._rebuild();
    const fk = this._farKeys();                     // the distant view: rebuilt when its inputs change, neighbours at most every 1.5 s
    if (!this._farKey || fk.fixed !== this._farKey.fixed || (fk.month !== this._farKey.month && now - this._farT > 1500)) {
      this._bakeFar(); this._farKey = fk; this._farT = now;
    }
    const t = (now - this._t0) / 1000, s = this.scale;
    const [ox, oy] = this.world(this.cam.x, this.cam.y);
    const tx = Math.round(cw / 2 - ox * s), ty = Math.round(ch / 2 - oy * s);
    g.setTransform(s, 0, 0, s, tx, ty);
    g.imageSmoothingEnabled = s < 1;
    const vx0 = -tx / s, vy0 = -ty / s, vx1 = vx0 + cw / s, vy1 = vy0 + ch / s;
    if (this.far) {
      g.imageSmoothingEnabled = true;
      g.drawImage(this.far, this.fx0, this.fy0, this.far.width * FAR, this.far.height * FAR);
      g.imageSmoothingEnabled = s < 1;
    }
    // ground
    const gx = Math.max(vx0, this.gx0), gy = Math.max(vy0, this.gy0);
    const gw = Math.min(vx1, this.gx0 + this.ground.width) - gx, gh = Math.min(vy1, this.gy0 + this.ground.height) - gy;
    if (gw > 0 && gh > 0) {
      const sx = Math.floor(gx - this.gx0), sy = Math.floor(gy - this.gy0), w = Math.ceil(gw) + 1, h = Math.ceil(gh) + 1;
      g.drawImage(this.ground, sx, sy, w, h, sx + this.gx0, sy + this.gy0, w, h);
    }
    if (this.overlay === "transit") this._drawRoutes(g);
    if (this.hl) this._drawRect(g, now);
    if (this.hover >= 0) this._drawCursor(g, this.hover, "#ffffff");
    if (this.selected >= 0) this._drawCursor(g, this.selected, "#ffd23f");
    // shadows of planes
    const air = this._air(t);
    g.fillStyle = "rgba(0,0,0,0.22)";
    for (const a of air) { g.beginPath(); g.ellipse(a.gx, a.gy, 12, 5, 0, 0, Math.PI * 2); g.fill(); }
    // objects + vehicles, back to front
    const veh = this._vehicles(t).sort((a, b) => a.d - b.d);
    const objs = this.objects, drawn = [];
    const alpha = this.overlay === "none" ? 1 : 0.38;
    let vi = 0;
    const blit = (o) => {
      const x = o.px - o.s.ox, y = o.py - o.s.oy;
      if (x > vx1 || y > vy1 || x + o.s.w < vx0 || y + o.s.h < vy0) return;
      g.drawImage(o.s.canvas, x, y);
    };
    g.globalAlpha = alpha;
    for (let k = 0; k <= objs.length; k++) {
      const o = objs[k], d = o ? o.d : Infinity;
      while (vi < veh.length && veh[vi].d <= d) { g.globalAlpha = veh[vi].alpha ?? 1; blit(veh[vi++]); }
      if (!o) break;
      g.globalAlpha = alpha;
      const a0 = o.tile >= 0 && this.anims.get(o.tile);
      if (a0 && o.build0) { if (this._drawBuilding(g, o, now - a0)) drawn.push(o); continue; }
      blit(o);
      drawn.push(o);
      if (o.s.rotor) this._drawRotor(g, o, t);
      if (o.signal) this._drawSignal(g, o);
    }
    g.globalAlpha = 1;
    this._drawn = drawn;
    this._people(g, t);
    this._drawSmoke(g, t);
    this._drawDust(g, now);
    for (const a of air) blit(a);
    this._birds(g, t);
  }

  _drawBuilding(g, o, el) {                          // construction: foundations, then the building rises inside scaffolding
    const k = el / BUILD_MS, s = o.s;
    if (k >= 1) { this.anims.delete(o.tile); g.drawImage(s.canvas, o.px - s.ox, o.py - s.oy); return true; }
    const fnd = this._sprite(`fnd|${o.fw}`, (b) => { b.scale(o.fw); R.foundation(b); });
    if (k < 0.3) { g.drawImage(fnd.canvas, o.px - fnd.ox, o.py - fnd.oy); return false; }
    const f = (k - 0.3) / 0.7, rh = Math.max(1, Math.round(s.h * f));
    g.drawImage(s.canvas, 0, s.h - rh, s.w, rh, o.px - s.ox, o.py - s.oy + s.h - rh, s.w, rh);
    const hu = Math.max(0.2, Math.round(((s.oy - TH / 2 - 2) / ZS) * 10) / 10);
    if (f < 0.92) {
      const sc = this._sprite(`scaf|${hu}|${o.fw}`, (b) => { b.scale(o.fw, 1); R.scaffold(b, hu * 0.95); });
      g.drawImage(sc.canvas, o.px - sc.ox, o.py - sc.oy);
    }
    return true;
  }
  _drawRotor(g, o, t) {
    const r = o.s.rotor, cx = Math.round(o.px + r.s[0]), cy = Math.round(o.py + r.s[1]), L = r.len * ZS * 0.9;
    g.fillStyle = this.night ? "#6a7080" : "#f4f5f6";
    for (let k = 0; k < 3; k++) {
      const a = -t * 2.4 + k * Math.PI * 2 / 3 + (o.tile % 7);
      this._line(g, cx, cy, cx + Math.cos(a) * L, cy + Math.sin(a) * L * 0.9);
    }
    g.fillStyle = "#9aa0a6"; g.fillRect(cx - 1, cy - 1, 2, 2);
  }
  _drawSignal(g, o) {                              // lit lamps on the two traffic light poles (see recipes.js:signals)
    const [x, y] = o.signal, st = this.life.signal(o.tile);
    if (!st) return;
    for (const [d, s] of [[0.44, st.ew], [-0.44, st.ns]]) {       // red on top, amber in the middle, green below
      const [px, py] = this.world(x + d, y + d, 0.31), X = Math.round(px) - 1, Y = Math.round(py) - 1 + (s === "r" ? -2 : s === "y" ? 0 : 2);
      if (this.night) { g.fillStyle = SIGNAL_C[s] + "44"; g.fillRect(X - 2, Y - 2, 6, 6); }
      g.fillStyle = SIGNAL_C[s]; g.fillRect(X, Y, 2, 2);
    }
  }
  _line(g, x0, y0, x1, y1) {                         // crisp 1px line
    x0 = Math.round(x0); y0 = Math.round(y0); x1 = Math.round(x1); y1 = Math.round(y1);
    const dx = Math.abs(x1 - x0), dy = -Math.abs(y1 - y0), sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    let e = dx + dy;
    for (let n = 0; n < 400; n++) {
      g.fillRect(x0, y0, 1, 1);
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * e;
      if (e2 >= dy) { e += dy; x0 += sx; }
      if (e2 <= dx) { e += dx; y0 += sy; }
    }
  }
  _diamond(g, x, y, w, h, color) {                   // outline of a w x h tile block with top-left tile (x, y)
    const c = [[x - 0.5, y - 0.5], [x + w - 0.5, y - 0.5], [x + w - 0.5, y + h - 0.5], [x - 0.5, y + h - 0.5]].map(([a, b]) => this.world(a, b));
    g.fillStyle = color;
    for (let k = 0; k < 4; k++) this._line(g, c[k][0], c[k][1], c[(k + 1) % 4][0], c[(k + 1) % 4][1]);
  }
  _drawCursor(g, tile, color) {
    const W = this.W, [fw, fh] = this.size.get(tile) || [1, 1];
    this._diamond(g, tile % W, Math.floor(tile / W), fw, fh, color);
  }
  _drawRect(g, now) {
    const r = this.hl, xa = Math.min(r.x1, r.x2), xb = Math.max(r.x1, r.x2), ya = Math.min(r.y1, r.y2), yb = Math.max(r.y1, r.y2);
    const a = 0.22 + 0.14 * Math.sin((now - this.hlT0) / 180);
    for (let y = ya; y <= yb; y++) for (let x = xa; x <= xb; x++) { const [px, py] = this.world(x, y); this._blitTint(g, "#ffd23f", px, py, a); }
    this._diamond(g, xa, ya, xb - xa + 1, yb - ya + 1, "#ffd23f");
  }
  _drawRoutes(g) {
    const f = this.frame;
    g.lineWidth = 3; g.lineJoin = "round";
    (f.metro_lines || []).forEach((line) => {
      g.strokeStyle = "rgba(211,47,47,0.8)"; g.setLineDash([6, 4]); g.beginPath();
      line.stations.forEach(([x, y], k) => { const [px, py] = this.world(x, y); k ? g.lineTo(px, py) : g.moveTo(px, py); });
      g.stroke(); g.setLineDash([]);
    });
    (f.bus_lines || []).forEach((line, k) => {
      const route = this.life.routePath(line.stops);
      if (route.length < 2) return;
      const off = (k % 3 - 1) * 3;
      g.strokeStyle = BUS_COLORS[k % BUS_COLORS.length]; g.beginPath();
      route.forEach(([x, y], n) => { const [px, py] = this.world(x, y); n ? g.lineTo(px, py + off) : g.moveTo(px, py + off); });
      g.stroke();
    });
  }
  _people(g, t) {
    if (this.quality === "low" || this.overlay !== "none" || ZOOMS[this.zi] < 1) return;
    const L = this.life;
    L.people.forEach((p, n) => {
      const q = L.person(p, t, n), [px, py] = this.world(q.x, q.y);
      const x = Math.round(px), y = Math.round(py - q.bob);
      g.fillStyle = this.night ? "#40485a" : p.color; g.fillRect(x, y - 3, 1, 2);
      g.fillStyle = this.night ? "#4a4a50" : "#f0c8a0"; g.fillRect(x, y - 4, 1, 1);
    });
  }
  _birds(g, t) {
    if (this.quality === "low" || this.overlay !== "none") return;
    g.fillStyle = this.night ? "#8a90a0" : "#2b2b2b";
    for (const b of this.life.birdPoints(t)) {
      const [px, py] = this.world(b.x, b.y, b.h), x = Math.round(px), y = Math.round(py), up = b.flap > 0 ? 1 : 0;
      g.fillRect(x - 2, y - up, 2, 1); g.fillRect(x + 1, y - up, 2, 1); g.fillRect(x, y, 1, 1);
    }
  }
  _drawSmoke(g, t) {
    if (this.quality === "low") return;
    if (this.overlay !== "none" && this.overlay !== "pollution") return;
    const [wu, wv] = rotQ(this.wind[0], this.wind[1], this.rot), [wx, wy] = project(wu, wv);
    const per = 9;
    this.emitters.forEach((e, n) => {
      for (let k = 0; k < per; k++) {
        const life = 4.5 / (0.6 + e.rate * 0.4), ph = (k / per + hash(n * 31 + k, 7) * 0.1) % 1, q = ((t / life) + ph) % 1;
        const x = e.x + q * 1.1 * wx + Math.sin(q * 6 + k) * 3, y = e.y + q * 1.1 * wy - q * (1.2 + e.rate * 0.5) * ZS * 0.8;
        const sz = Math.round(2 + q * 5), a = (1 - q) * Math.min(1, q * 5) * 0.8;
        if (e.fire) g.fillStyle = `rgba(${255 - q * 120 | 0},${Math.max(40, 200 - q * 230) | 0},${30 + q * 50 | 0},${a})`;
        else { const c = Math.round(255 * Math.min(1, e.dark + q * 0.35) * (this.night ? 0.5 : 1)); g.fillStyle = `rgba(${c},${c},${c + 4},${a})`; }
        g.fillRect(Math.round(x - sz / 2), Math.round(y - sz / 2), sz, sz);
      }
    });
  }
  _drawDust(g, now) {
    this.dust = this.dust.filter(d => now - d.t0 < DUST_MS);
    for (const d of this.dust) {
      const k = (now - d.t0) / DUST_MS, [px, py] = this.world(d.x, d.y);
      for (let n = 0; n < 10; n++) {
        const a = n * 2.4, r = 4 + k * 18 * (0.6 + hash(n, 9) * 0.6), sz = Math.round(3 + k * 5);
        g.fillStyle = `rgba(150,130,100,${(1 - k) * 0.7})`;
        g.fillRect(Math.round(px + Math.cos(a) * r - sz / 2), Math.round(py + Math.sin(a) * r * 0.5 - k * 10 - sz / 2), sz, sz);
      }
    }
  }
}
