// Moving life for the 3D city: cars, buses, trains, boats, planes, pedestrians and birds.
// Positions come from `LifeSim` (lifesim.js, shared with the pixel view); this draws them as instanced meshes.
import * as THREE from "./vendor/three-bundle.min.js";
import { LifeSim } from "./lifesim.js";

const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _p = new THREE.Vector3(), _s = new THREE.Vector3(), _e = new THREE.Euler();
const _colors = new Map();
const C = (hex) => { let c = _colors.get(hex); if (!c) { c = new THREE.Color(hex); _colors.set(hex, c); } return c; };
const GLASS = new THREE.Color(0x9fb7c9), WAGON = new THREE.Color(0xeceff1);
const BOXES = ["#c0392b", "#2980b9", "#27ae60", "#f39c12", "#8e44ad"].map(c => new THREE.Color(c));
const LIGHT = { g: new THREE.Color(0x36e05a), y: new THREE.Color(0xffc400), r: new THREE.Color(0xff3b30) };

function box(w, h, d, y = 0) { const g = new THREE.BoxGeometry(w, h, d); g.translate(0, y + h / 2, 0); return g; }

class Fleet {
  constructor(parent, geom, mat, cap) {
    this.mesh = new THREE.InstancedMesh(geom, mat, cap);
    this.mesh.count = 0; this.mesh.frustumCulled = false; this.mesh.castShadow = true;
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.cap = cap; parent.add(this.mesh);
  }
  set(i, x, y, z, rot, sx = 1, sy = 1, sz = 1, pitch = 0) {
    _p.set(x, y, z); _s.set(sx, sy, sz); _q.setFromEuler(_e.set(0, rot, pitch, "YXZ"));
    this.mesh.setMatrixAt(i, _m.compose(_p, _q, _s));
  }
  color(i, c) { this.mesh.setColorAt(i, c); }
  done(n) {
    this.mesh.count = n; this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }
}

export class Life {
  constructor(scene) {
    this.scene = scene;                 // CityScene (for wx / wz / charAt)
    this.group = new THREE.Group();
    scene.root.add(this.group);
    const std = (o) => new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.15, ...o });
    this.carMat = std({ color: 0xffffff });
    this.fleets = {
      carBody: new Fleet(this.group, box(0.22, 0.06, 0.11, 0.012), this.carMat, 600),
      carTop: new Fleet(this.group, box(0.12, 0.05, 0.1, 0.07), std({ color: 0x9fb7c9, roughness: 0.2, metalness: 0.5 }), 600),
      bus: new Fleet(this.group, box(0.42, 0.13, 0.13, 0.015), std({ color: 0xffffff }), 40),
      busWin: new Fleet(this.group, box(0.4, 0.04, 0.135, 0.085), std({ color: 0x223344, roughness: 0.2 }), 40),
      train: new Fleet(this.group, box(0.48, 0.14, 0.15, 0.05), std({ color: 0xffffff }), 80),
      boatHull: new Fleet(this.group, box(0.36, 0.07, 0.14, -0.05), std({ color: 0xffffff }), 40),
      boatCabin: new Fleet(this.group, box(0.14, 0.07, 0.1, 0.02), std({ color: 0xeeeeee }), 40),
      sail: new Fleet(this.group, new THREE.ConeGeometry(0.08, 0.34, 3).translate(0, 0.19, 0), std({ color: 0xfafafa, roughness: 0.8 }), 40),
      plane: new Fleet(this.group, this._planeGeom(), std({ color: 0xf5f5f5 }), 8),
      person: new Fleet(this.group, box(0.028, 0.075, 0.028), std({ color: 0xffffff, roughness: 0.8 }), 500),
      bird: new Fleet(this.group, new THREE.ConeGeometry(0.05, 0.14, 3).rotateZ(Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0x2b2b2b }), 40),
      shipHull: new Fleet(this.group, box(0.92, 0.1, 0.22, -0.08), std({ color: 0xffffff }), 20),
      shipBoxes: new Fleet(this.group, box(0.5, 0.1, 0.18, 0.02).translate(0.1, 0, 0), std({ color: 0xffffff }), 20),
      shipBridge: new Fleet(this.group, box(0.14, 0.16, 0.18, 0.02).translate(-0.34, 0, 0), std({ color: 0xf2f2ee }), 20),
      sigPole: new Fleet(this.group, box(0.025, 0.34, 0.025), std({ color: 0x3a3d42 }), 1200),
      sigLamp: new Fleet(this.group, box(0.06, 0.06, 0.06, 0.3), new THREE.MeshBasicMaterial({ color: 0xffffff }), 1200),
    };
    this.sim = new LifeSim(scene);
    this.visible = true;
  }

  _planeGeom() {
    const g = [box(0.7, 0.08, 0.09), box(0.14, 0.02, 0.8, 0.03), box(0.08, 0.16, 0.02, 0.05)];
    g[2].translate(-0.3, 0, 0);
    const pos = [], idx = [];
    let off = 0;
    for (const x of g) {
      const p = x.attributes.position.array, id = x.index.array;
      for (const v of p) pos.push(v);
      for (const i of id) idx.push(i + off);
      off += p.length / 3;
    }
    const out = new THREE.BufferGeometry();
    out.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    out.setIndex(idx); out.computeVertexNormals();
    return out;
  }

  setFrame(frame) { this.sim.setFrame(frame); }
  routePath(stops) { return this.sim.routePath(stops); }

  // ------------------------------------------------------------- tick
  tick(dt, t, nightK) {
    const F = this.fleets, sc = this.scene, sim = this.sim;
    this.group.visible = this.visible;
    if (!this.visible || !sim.roads) return;
    sim.step(dt);
    let n = 0;
    for (const c of sim.cars) {
      const p = sim.car(c), X = sc.wx(p.x), Z = sc.wz(p.y), Y = p.bridge ? 0.082 : 0.004, rot = -Math.atan2(p.dy, p.dx);
      F.carBody.set(n, X, Y, Z, rot); F.carTop.set(n, X, Y, Z, rot);
      F.carBody.color(n, C(c.color)); F.carTop.color(n, C(c.color).clone().lerp(GLASS, 0.7));
      n++;
    }
    F.carBody.done(n); F.carTop.done(n);
    n = 0;                                          // pedestrians on the sidewalks
    for (const p of sim.people) {
      const q = sim.person(p, t, n);
      F.person.set(n, sc.wx(q.x), 0.004 + q.bob * 0.006, sc.wz(q.y), 0); F.person.color(n, C(p.color));
      n++;
    }
    F.person.done(n);
    n = 0;
    for (const b of sim.buses) {
      const p = sim.bus(b), X = sc.wx(p.x), Z = sc.wz(p.y), rot = -Math.atan2(p.dy, p.dx), Y = p.bridge ? 0.082 : 0.004;
      F.bus.set(n, X, Y, Z, rot); F.busWin.set(n, X, Y, Z, rot); F.bus.color(n, C(b.color));
      n++;
    }
    F.bus.done(n); F.busWin.done(n);
    n = 0;
    for (const tr of sim.trains) {
      for (const k of sim.train(tr, t) || []) {
        F.train.set(n, sc.wx(k.x), 0.02, sc.wz(k.y), k.ew ? 0 : Math.PI / 2); F.train.color(n, k.loco ? C(tr.color) : WAGON);
        n++;
      }
    }
    F.train.done(n);
    n = 0;
    for (const b of sim.boats) {
      const p = sim.boat(b, t, n), X = sc.wx(p.x), Z = sc.wz(p.y), rot = -Math.atan2(p.dy, p.dx), y = -0.1 + p.bob * 0.012, s = b.sail;
      F.boatHull.set(n, X, y, Z, rot); F.boatHull.color(n, C(b.color));
      F.boatCabin.set(n, X, y, Z, rot, s ? 0.001 : 1, s ? 0.001 : 1, s ? 0.001 : 1);
      F.sail.set(n, X, y, Z, rot, s ? 1 : 0.001, s ? 1 : 0.001, s ? 1 : 0.001);
      n++;
    }
    F.boatHull.done(n); F.boatCabin.done(n); F.sail.done(n);
    n = 0;                                          // container ships calling at the harbours
    for (const sh of sim.ships) {
      const p = sim.ship(sh);
      if (!p || p.fade < 0.3) continue;
      const rot = -Math.atan2(p.dy, p.dx), X = sc.wx(p.x), Z = sc.wz(p.y);
      F.shipHull.set(n, X, -0.08, Z, rot); F.shipHull.color(n, C(sh.color));
      F.shipBoxes.set(n, X, -0.08, Z, rot); F.shipBoxes.color(n, BOXES[sh.v % BOXES.length]);
      F.shipBridge.set(n, X, -0.08, Z, rot);
      n++;
    }
    F.shipHull.done(n); F.shipBoxes.done(n); F.shipBridge.done(n);
    n = 0;
    for (const p of sim.planes) {
      const q = sim.plane(p, t), s = q.big ? 1.4 : 1;
      F.plane.set(n++, sc.wx(q.x), q.h, sc.wz(q.y), -q.heading, s, s, s);
    }
    F.plane.done(n);
    n = 0;
    for (const b of sim.birdPoints(t)) F.bird.set(n++, sc.wx(b.x), b.h, sc.wz(b.y), -b.a - Math.PI / 2, 1, 1, 1, b.flap);
    F.bird.done(n);
    n = 0;                                          // traffic lights: one pole per direction on opposite corners
    for (const i of sim.signals.keys()) {
      const s = sim.signal(i), x = i % sc.W, y = Math.floor(i / sc.W);
      if (n + 2 > F.sigPole.cap) break;
      for (const [d, st] of [[0.44, s.ew], [-0.44, s.ns]]) {
        const X = sc.wx(x + d), Z = sc.wz(y + d);
        F.sigPole.set(n, X, 0, Z, 0); F.sigLamp.set(n, X, 0, Z, 0); F.sigLamp.color(n, LIGHT[st]);
        n++;
      }
    }
    F.sigPole.done(n); F.sigLamp.done(n);
  }
}
