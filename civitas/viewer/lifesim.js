// Moving life, renderer-agnostic: cars, buses, trains, boats, planes, pedestrians and birds.
// Everything is cosmetic: it reads the replay frame (tiles, traffic layer, bus lines, stations) and keeps
// positions in tile coordinates. `life.js` (3D) and `pixel.js` (2D) each draw it their own way.
// `host` gives W, H, margin, charAt(frame, x, y) and busColor(k).

const rnd = (a, b) => a + Math.random() * (b - a);
const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
export const CAR_COLORS = ["#e53935", "#1e88e5", "#fdd835", "#fafafa", "#212121", "#43a047", "#8e24aa", "#fb8c00", "#90a4ae", "#6d4c41", "#00acc1"];
export const CLOTHES = ["#e53935", "#1e88e5", "#fdd835", "#fafafa", "#37474f", "#43a047", "#8e24aa", "#fb8c00", "#f06292", "#26a69a"];
export const TRAIN_COLORS = ["#c62828", "#1565c0", "#2e7d32", "#ef6c00"];
export const BOAT_COLORS = ["#fafafa", "#c62828", "#1565c0", "#f9a825", "#2e7d32"];
const DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1]];
// Traffic lights: each junction cycles east-west green, amber, north-south green, amber (seconds).
const CYCLE = [5, 1.2, 5, 1.2], PERIOD = CYCLE.reduce((a, b) => a + b, 0);
const STOP = 0.6, GAP = 0.27;                       // stop line (fraction of the way to the junction) and queue spacing

export class LifeSim {
  constructor(host) {
    this.host = host;
    this.cars = []; this.buses = []; this.trains = []; this.boats = []; this.planes = []; this.people = []; this.birds = []; this.ships = [];
    this.roads = null; this.time = 0; this.signals = new Map();
  }

  // ------------------------------------------------------------ setup
  setFrame(frame) {
    const sc = this.host, W = sc.W, H = sc.H, M = sc.margin, T = frame.tiles;
    const key = T + (frame.road_class || "") + JSON.stringify(frame.bus_lines || []);
    const traffic = frame.layers && frame.layers.traffic;
    this.traffic = traffic;
    if (key === this._key) return;
    this._key = key;
    const ch = (x, y) => (x >= 0 && y >= 0 && x < W && y < H) ? T[y * W + x] : sc.charAt(frame, x, y);
    this.ch = ch;
    const isRoad = (x, y) => x >= -M && y >= -M && x < W + M && y < H + M && "#=H".includes(ch(x, y));
    const isWater = (x, y) => x >= -M && y >= -M && x < W + M && y < H + M && "~=Y".includes(ch(x, y));
    this.isRoad = isRoad; this.isWater = isWater;
    const roads = [], water = [], walk = [];
    let pop = 0;
    for (let y = -M; y < H + M; y++) for (let x = -M; x < W + M; x++) {
      if (isRoad(x, y)) roads.push([x, y]);
      if (isWater(x, y) && ch(x, y) !== "=") water.push([x, y]);
    }
    for (const [x, y] of roads) {
      if (x < 0 || y < 0 || x >= W || y >= H) continue;
      let near = 0;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]) {
        const c = ch(x + dx, y + dy);
        if ("RCOMrcom".includes(c)) near++;
      }
      if (near) walk.push([x, y, near]);
    }
    for (let i = 0; i < T.length; i++) if ("RM".includes(T[i])) pop += +frame.levels[i];
    this.roads = roads; this.water = water; this.walk = walk;
    // traffic lights on street and avenue junctions with three or more arms
    this.signals = new Map();
    for (let i = 0; i < T.length; i++) {
      if (T[i] !== "#" || (frame.road_class && +frame.road_class[i] >= 3)) continue;
      const x = i % W, y = Math.floor(i / W);
      let arms = 0;
      for (const [dx, dy] of DIRS) if (isRoad(x + dx, y + dy)) arms++;
      if (arms >= 3) this.signals.set(i, ((Math.imul(i, 2654435761) >>> 0) % 1000) / 1000 * PERIOD);
    }
    // how busy: traffic layer + city size
    let busy = 0;
    if (traffic) for (let i = 0; i < traffic.length; i++) if ("#=H".includes(T[i])) busy += +traffic[i];
    const nCars = Math.min(560, Math.round(18 + roads.length * 0.35 + busy * 1.6 + pop * 0.25));
    const nPeople = Math.min(480, Math.round(walk.length * 0.8 + pop * 0.4));
    this._spawn(this.cars, nCars, () => this._newCar());
    this._spawn(this.people, walk.length ? nPeople : 0, () => this._newPerson());
    // buses follow their line along the roads
    this.buses = [];
    for (const [k, line] of (frame.bus_lines || []).entries()) {
      const path = this._routePath(line.stops);
      if (path.length > 1) for (let n = 0; n < (path.length > 25 ? 2 : 1); n++)
        this.buses.push({ path, i: n ? Math.floor(path.length / 2) : 0, t: 0, dir: 1, wait: 0, stops: new Set(line.stops.map(([x, y]) => `${x},${y}`)),
          color: sc.busColor(k) });
    }
    // trains shuttle between each station and the map edge
    this.trains = [];
    for (let i = 0; i < T.length; i++) if (T[i] === "@") {
      const x = i % W, y = Math.floor(i / W);
      const out = x === 0 ? [-1, 0] : x === W - 1 ? [1, 0] : y === 0 ? [0, -1] : y === H - 1 ? [0, 1] : null;
      if (out) this.trains.push({ x, y, out, phase: Math.random() * 30, color: pick(TRAIN_COLORS) });
    }
    // boats on the river, more with a harbour
    const harbours = (T.match(/&/g) || []).length;
    this._spawn(this.boats, water.length ? Math.min(30, 3 + Math.round(water.length / 120) + 4 * harbours) : 0, () => this._newBoat());
    // planes land at each airport, taxi to a stand, wait, and take off again (to another region airport if any)
    this.planes = [];
    for (const [a, fw, fh] of frame.multi || []) if (T[a] === ">") {
      const x = a % W + (fw - 1) / 2, y = Math.floor(a / W) + (fh - 1) / 2;
      for (let n = 0; n < 2; n++) {
        const p = { airport: [x, y], stand: n ? 0.6 : 0.15, n };
        this._flight(p);
        p.clock = n ? p.sched.end * 0.5 : Math.random() * 6;
        this.planes.push(p);
      }
    }
    this.planes.push({ cross: true, t: Math.random() * 40 });
    // container ships sail in from the river mouth, berth at each harbour, and leave again
    this.ships = [];
    for (let i = 0; i < T.length; i++) if (T[i] === "&") {
      const x = i % W, y = Math.floor(i / W);
      const dock = DIRS.map(([dx, dy]) => [x + dx, y + dy]).find(([a, b]) => isWater(a, b) && ch(a, b) !== "=");
      const path = dock && this._toEdge(dock, isWater);
      if (path && path.length > 2) {
        const leg = path.length / 0.8;
        this.ships.push({ path, leg, cycle: 2 * leg + 12 + 6, clock: Math.random() * 20, color: pick(["#1f3a5f", "#7a1f1f", "#2f4f2f", "#3a3a44"]), v: i % 5 });
      }
    }
    // birds circle over the map
    if (!this.birds.length) for (let n = 0; n < 3; n++)
      this.birds.push({ cx: W / 2 - 0.5 + rnd(-W / 3, W / 3), cy: H / 2 - 0.5 + rnd(-H / 3, H / 3), r: rnd(3, 7), ang: Math.random() * 6.28, h: rnd(2.2, 3.5), n: 6 + Math.floor(Math.random() * 6) });
  }

  // Flight plan around an airport (centre ax, ay; runway along x at y + 0.9): keyframes [time, x, y, h, ease].
  _flight(p) {
    const [ax, ay] = p.airport, R = ay + 0.9, dests = (this.host.airports ? this.host.airports() : []).filter(([x, y]) => Math.hypot(x - ax, y - ay) > 4);
    const from = dests.length ? dests[(p.n + (p.trips || 0)) % dests.length] : [ax - 20, R], to = dests.length ? dests[(p.n + (p.trips || 0) + 1) % dests.length] : [ax + 20, R];
    const hFrom = dests.length ? 0 : 5, hTo = dests.length ? 0 : 5;
    const pts = [], add = (x, y, h, dt, ease = "lin") => pts.push([(pts.length ? pts[pts.length - 1][0] : 0) + dt, x, y, h, ease]);
    const air = (x, y, h) => { const q = pts[pts.length - 1]; add(x, y, h, Math.hypot(x - q[1], y - q[2]) / 3.5 + 0.3); };
    const taxi = (x, y) => { const q = pts[pts.length - 1]; add(x, y, 0, Math.hypot(x - q[1], y - q[2]) / 0.35 + 0.2); };
    add(from[0], from[1], hFrom, 0);
    if (dests.length) air((from[0] + ax - 12) / 2, (from[1] + R) / 2, 6);          // cruise across the region
    air(ax - 12, R, 3); air(ax - 7, R, 1.6);
    add(ax - 1.3, R, 0, 2.6, "out");                                                   // touchdown
    add(ax + 1.0, R, 0, 2.8, "out");                                                   // roll out
    taxi(ax + 1.0, ay + 0.25); taxi(ax + p.stand, ay + 0.25); taxi(ax + p.stand, ay - 0.1);
    add(ax + p.stand, ay - 0.1, 0, 9);                                                 // at the stand
    taxi(ax + p.stand, ay + 0.25); taxi(ax - 1.3, ay + 0.25); taxi(ax - 1.3, R);
    add(ax + 0.9, R, 0, 3.2, "in");                                                    // take-off roll
    air(ax + 1.8, R, 0.5); air(ax + 8, R, 2.6);
    if (dests.length) air((to[0] + ax + 8) / 2, (to[1] + R) / 2, 6);
    air(to[0], to[1], hTo);
    p.sched = { pts, end: pts[pts.length - 1][0] };
  }
  // shortest water path to a tile from the edge of the drawn area (the river mouth): edge first, the tile last
  _toEdge([sx, sy], ok) {
    const M = this.host.margin, W = this.host.W, H = this.host.H, prev = new Map(), key = (x, y) => `${x},${y}`, q = [[sx, sy]];
    prev.set(key(sx, sy), null);
    for (let k = 0; k < q.length; k++) {
      const [x, y] = q[k];
      if (x === -M || y === -M || x === W + M - 1 || y === H + M - 1) {
        const out = []; for (let c = [x, y]; c; c = prev.get(key(c[0], c[1]))) out.push(c);
        return out;
      }
      for (const [dx, dy] of DIRS) { const nx = x + dx, ny = y + dy, kk = key(nx, ny); if (!prev.has(kk) && ok(nx, ny) && this.ch(nx, ny) !== "=") { prev.set(kk, [x, y]); q.push([nx, ny]); } }
    }
    return null;
  }

  _spawn(list, n, make) {
    while (list.length > n) list.pop();
    while (list.length < n) { const o = make(); if (!o) break; list.push(o); }
  }

  _newCar() {
    if (!this.roads.length) return null;
    const [x, y] = this._weightedRoad();
    const car = { x, y, px: x, py: y, t: Math.random(), speed: rnd(0.9, 1.6), color: pick(CAR_COLORS) };
    this._nextTile(car, this.isRoad);
    return car;
  }
  _weightedRoad() {
    const tr = this.traffic, W = this.host.W;
    for (let k = 0; k < 6; k++) {
      const [x, y] = pick(this.roads);
      const inside = x >= 0 && y >= 0 && x < W && y < this.host.H;
      const v = tr && inside ? +tr[y * W + x] : 1;
      if (Math.random() < 0.25 + v / 9) return [x, y];
    }
    return pick(this.roads);
  }
  _newPerson() {
    if (!this.walk.length) return null;
    const [x, y] = pick(this.walk);
    const p = { x, y, px: x, py: y, t: Math.random(), speed: rnd(0.18, 0.32), side: Math.random() < 0.5 ? 1 : -1, color: pick(CLOTHES) };
    this._nextTile(p, this.isRoad);
    return p;
  }
  _newBoat() {
    if (!this.water.length) return null;
    const [x, y] = pick(this.water);
    const b = { x, y, px: x, py: y, t: Math.random(), speed: rnd(0.25, 0.45), sail: Math.random() < 0.4, color: pick(BOAT_COLORS) };
    this._nextTile(b, this.isWater, true);
    return b;
  }

  // choose where to go after reaching (x, y), coming from (px, py)
  _nextTile(o, ok, river = false) {
    const opts = [];
    for (const [dx, dy] of DIRS) {
      const nx = o.x + dx, ny = o.y + dy;
      if (!ok(nx, ny)) continue;
      const back = nx === o.px && ny === o.py;
      const straight = dx === o.x - o.px && dy === o.y - o.py;
      let w = back ? 0.05 : straight ? 3 : 1;
      if (river && dy !== 0) w *= 2.5;              // boats mostly follow the river north-south
      if (this.traffic && !river && nx >= 0 && ny >= 0 && nx < this.host.W && ny < this.host.H) w *= 0.6 + +this.traffic[ny * this.host.W + nx] / 6;
      opts.push([nx, ny, w]);
    }
    if (!opts.length) { o.nx = o.x; o.ny = o.y; return; }
    let r = Math.random() * opts.reduce((a, b) => a + b[2], 0);
    for (const [nx, ny, w] of opts) { r -= w; if (r <= 0) { o.nx = nx; o.ny = ny; return; } }
    o.nx = opts[0][0]; o.ny = opts[0][1];
  }

  _advance(o, dt, ok, river) {
    o.t += o.speed * dt;
    while (o.t >= 1) {
      o.t -= 1; o.px = o.x; o.py = o.y; o.x = o.nx; o.y = o.ny;
      this._nextTile(o, ok, river);
    }
  }

  _routePath(stops) {
    const path = [];
    for (let k = 0; k + 1 < stops.length; k++) {
      const seg = this._bfs(stops[k], stops[k + 1]);
      if (!seg) continue;
      if (path.length) seg.shift();
      path.push(...seg);
    }
    return path;
  }
  _bfs([sx, sy], [ex, ey]) {
    const prev = new Map(), key = (x, y) => `${x},${y}`, q = [[sx, sy]];
    prev.set(key(sx, sy), null);
    while (q.length) {
      const [x, y] = q.shift();
      if (x === ex && y === ey) break;
      for (const [dx, dy] of DIRS) {
        const nx = x + dx, ny = y + dy, k = key(nx, ny);
        if (!prev.has(k) && this.isRoad(nx, ny)) { prev.set(k, [x, y]); q.push([nx, ny]); }
      }
    }
    if (!prev.has(key(ex, ey))) return null;
    const out = [];
    for (let c = [ex, ey]; c; c = prev.get(key(c[0], c[1]))) out.push(c);
    return out.reverse();
  }
  routePath(stops) { return this._routePath(stops); }

  // ------------------------------------------------------------- traffic lights
  // "g", "y" or "r" for traffic along x (ew) and along y (ns) at junction tile i, or null if it has no lights
  signal(i) {
    const off = this.signals.get(i);
    if (off === undefined) return null;
    let p = (this.time + off) % PERIOD;
    if (p < CYCLE[0]) return { ew: "g", ns: "r" };
    if ((p -= CYCLE[0]) < CYCLE[1]) return { ew: "y", ns: "r" };
    if ((p -= CYCLE[1]) < CYCLE[2]) return { ew: "r", ns: "g" };
    return { ew: "r", ns: "y" };
  }
  _redAhead(x, y, nx, ny) {                        // is the junction we are driving into showing red / amber to us?
    if (nx < 0 || ny < 0 || nx >= this.host.W || ny >= this.host.H) return false;
    const s = this.signal(ny * this.host.W + nx);
    return !!s && (nx !== x ? s.ew : s.ns) !== "g";
  }

  // ------------------------------------------------------------- step
  step(dt) {
    if (!this.roads) return;
    this.time += dt;
    const W = this.host.W, H = this.host.H, tr = this.traffic;
    const queues = new Map();                       // cars waiting at a red light, per approach
    for (const c of this.cars) {
      c.hold = null;
      if (c.t < STOP && this._redAhead(c.x, c.y, c.nx, c.ny)) {
        const k = `${c.nx},${c.ny},${c.nx - c.x},${c.ny - c.y}`;
        (queues.get(k) || queues.set(k, []).get(k)).push(c);
      }
    }
    for (const q of queues.values()) q.sort((a, b) => b.t - a.t).forEach((c, r) => { c.hold = Math.max(0.02, STOP - GAP * r); });
    for (const c of this.cars) {                    // slower where the traffic layer says the road is jammed
      const jam = tr && c.x >= 0 && c.y >= 0 && c.x < W && c.y < H ? +tr[c.y * W + c.x] / 9 : 0;
      if (c.hold != null) { if (c.t < c.hold) c.t = Math.min(c.hold, c.t + c.speed * dt * (1 - 0.65 * jam)); continue; }
      this._advance(c, dt * (1 - 0.65 * jam), this.isRoad, false);
    }
    for (const p of this.people) this._advance(p, dt, this.isRoad, false);
    for (const b of this.buses) {                   // along the route, pausing at stops
      if (b.wait > 0) { b.wait -= dt; continue; }
      const [ax, ay] = b.path[b.i], nb = b.path[Math.max(0, Math.min(b.path.length - 1, b.i + b.dir))];
      if (b.t < STOP && this._redAhead(ax, ay, nb[0], nb[1])) { b.t = Math.min(STOP, b.t + dt * 0.9); continue; }
      b.t += dt * 0.9;
      if (b.t >= 1) {
        b.t = 0; b.i += b.dir;
        if (b.i <= 0 || b.i >= b.path.length - 1) { b.i = Math.max(0, Math.min(b.path.length - 1, b.i)); b.dir *= -1; }
        const [x, y] = b.path[b.i];
        if (b.stops.has(`${x},${y}`)) b.wait = 1.4;
      }
    }
    for (const b of this.boats) this._advance(b, dt, this.isWater, true);
    for (const p of this.planes) if (!p.cross) {
      p.clock += dt;
      if (p.clock >= p.sched.end) { p.clock -= p.sched.end; p.trips = (p.trips || 0) + 1; this._flight(p); }
    }
    for (const s of this.ships) s.clock = (s.clock + dt) % s.cycle;
    for (const f of this.birds) f.ang += dt * 0.35;
  }

  // ------------------------------------------------------------ poses (tile coordinates)
  // cars drive on the right: offset 0.17 tiles from the centre line
  car(c) {
    const dx = c.nx - c.x, dy = c.ny - c.y;
    return { x: c.x + dx * c.t - dy * 0.17, y: c.y + dy * c.t + dx * 0.17, dx, dy, bridge: this.ch(c.x, c.y) === "=" };
  }
  person(p, t, n) {
    const dx = p.nx - p.x, dy = p.ny - p.y;
    return { x: p.x + dx * p.t - dy * 0.43 * p.side, y: p.y + dy * p.t + dx * 0.43 * p.side, bob: Math.abs(Math.sin(t * 9 + n)) };
  }
  bus(b) {
    const a = b.path[b.i], c = b.path[Math.max(0, Math.min(b.path.length - 1, b.i + b.dir))];
    const dx = c[0] - a[0], dy = c[1] - a[1], k = b.wait > 0 ? 0 : b.t;
    return { x: a[0] + dx * k - dy * 0.17, y: a[1] + dy * k + dx * 0.17, dx, dy, bridge: this.ch(a[0], a[1]) === "=" };
  }
  boat(b, t, n) {
    const dx = b.nx - b.x, dy = b.ny - b.y;
    return { x: b.x + dx * b.t + Math.sin(t * 0.6 + n) * 0.12, y: b.y + dy * b.t, dx, dy, bob: Math.sin(t * 2 + n) };
  }
  // trains arrive from beyond the map edge, wait, leave: 4 cars, or null while off-map
  train(tr, t) {
    const M = this.host.margin, cyc = (t + tr.phase) % 30;
    const d = cyc < 8 ? (1 - cyc / 8) * M : cyc < 14 ? 0 : cyc < 22 ? ((cyc - 14) / 8) * M : M + 5;
    if (d > M + 1) return null;
    const cars = [];
    for (let k = 0; k < 4; k++) {
      const dist = d + 0.5 + k * 0.52;
      cars.push({ x: tr.x + tr.out[0] * dist, y: tr.y + tr.out[1] * dist, ew: !!tr.out[0], loco: k === 0 });
    }
    return cars;
  }
  // circuits around airports, plus an airliner crossing high up now and then; heading in radians (map x/y)
  plane(p, t) {
    const W = this.host.W, H = this.host.H;
    if (p.cross) {
      const k = ((t + p.t) % 45) / 45, span = Math.max(W, H) * 1.6;
      return { x: W / 2 - 0.5 - span / 2 + k * span, y: H / 2 - 0.5 - span * 0.2 + k * span * 0.3, h: 14, heading: Math.atan2(0.3, 1), big: true };
    }
    const P = p.sched.pts, c = Math.min(p.clock, p.sched.end - 1e-6);
    let k = 1; while (k < P.length - 1 && P[k][0] <= c) k++;
    const a = P[k - 1], b = P[k], u = (c - a[0]) / Math.max(1e-6, b[0] - a[0]);
    const e = b[4] === "in" ? u * u : b[4] === "out" ? 1 - (1 - u) * (1 - u) : u;
    let hd = Math.atan2(b[2] - a[2], b[1] - a[1]);
    if (Math.hypot(b[1] - a[1], b[2] - a[2]) < 1e-6) { const q = P[k - 2] || a; hd = Math.atan2(a[2] - q[2], a[1] - q[1]); }  // parked: keep facing the way it came
    const h = a[3] + (b[3] - a[3]) * e;
    return { x: a[1] + (b[1] - a[1]) * e, y: a[2] + (b[2] - a[2]) * e, h, heading: hd, ground: h < 0.02 };
  }
  // container ship: sails the path in, berths, sails out; fade 0..1 near the edge of the drawn area
  ship(s) {
    const n = s.path.length - 1, c = s.clock, inT = s.leg, dockT = 12;
    let f, out = false;
    if (c < inT) f = c / inT; else if (c < inT + dockT) f = 1; else if (c < 2 * inT + dockT) { f = 1 - (c - inT - dockT) / inT; out = true; } else return null;
    const pos = f * n, k = Math.min(n - 1, Math.floor(pos)), t = pos - k, a = s.path[k], b = s.path[k + 1];
    let dx = b[0] - a[0], dy = b[1] - a[1];
    if (out) { dx = -dx; dy = -dy; }
    return { x: a[0] + (b[0] - a[0]) * t, y: a[1] + (b[1] - a[1]) * t, dx, dy, fade: Math.min(1, pos / 3), docked: f === 1 };
  }
  *birdPoints(t) {
    for (const f of this.birds) for (let k = 0; k < f.n; k++) {
      const a = f.ang - k * 0.09, flap = Math.sin(t * 8 + k) * 0.15;
      yield { x: f.cx + Math.cos(a) * (f.r + (k % 3) * 0.3), y: f.cy + Math.sin(a) * f.r, h: f.h + flap * 0.4 + (k % 2) * 0.2, a, flap };
    }
  }
}
