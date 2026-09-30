// Civitas 3D city renderer (three.js). Renders one replay frame at a time.
import * as THREE from "./vendor/three-bundle.min.js";
import { OrbitControls } from "./vendor/three-bundle.min.js";
import { Life } from "./life.js";
import { tierAnnexes, tierRank, facilityTier } from "./recipes.js";

const MARGIN = 12;                       // decorative land around the playable map
const ZONES = "rciomRCIOM";
export const SPECIALS = "TEGSWUVHKJF^AhLXZNBQD%*Y|@&+$!:?>uz";
const NATIVE_SIZE = ">";            // models already drawn at their full footprint
const BUS_COLORS = ["#e07bd8", "#4aa3ff", "#ffb454", "#5fd38d", "#ff6b5e", "#f5d547"];
const hash = (i, k = 0) => {                      // integer hash -> [0, 1), no visible patterns
  let h = Math.imul((i | 0) ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul((k | 0) + 0x632be5ab, 0xc2b2ae35);
  h ^= h >>> 15; h = Math.imul(h, 0x2c1b3c6d); h ^= h >>> 12; h = Math.imul(h, 0x297a2d39); h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
};
const col = (h) => new THREE.Color(h);
const SNOW = new THREE.Color("#eef3f6");
const SIGNS = ["#e53935", "#fdd835", "#1e88e5", "#43a047", "#8e24aa", "#fb8c00"].map(c => new THREE.Color(c));
const FLOWERS = ["#f06292", "#fff176", "#ba68c8", "#ffffff", "#ff8a65", "#64b5f6"].map(c => new THREE.Color(c));
const lerpC = (a, b, t) => a.clone().lerp(b, t);

// ------------------------------------------------------------------ palette
const P = {
  grass: [col("#7aae5c"), col("#7db15e"), col("#78ab5a"), col("#7fb360")],
  grassOuter: col("#76a45a"),
  zoneLot: { r: col("#c8d8a0"), c: col("#b8cde0"), i: col("#ddd0b0"), o: col("#c9c2e6"), m: col("#dcc9b0") },
  lotDev: { R: col("#b9c79a"), C: col("#a9b8c6"), I: col("#b8ad98"), O: col("#aeb0c8"), M: col("#c2b09a") },
  asphalt: col("#3d4148"), asphaltDark: col("#2c3036"), sidewalk: col("#a8a59c"), park: col("#5fae4f"), bed: col("#b3a07a"), water: col("#4aa6e0"), dirty: col("#4f7a5c"),
  resWall: ["#f3e3c3", "#e9c9a7", "#f5efe0", "#dfe7ee", "#f0d5c8", "#e7dcc4"].map(col),
  resRoof: ["#a2553d", "#7a4b3a", "#5b5f66", "#8d3f33"].map(col),
  apt: ["#d9c2a3", "#c79b7c", "#e2d6c2", "#b9b2a8", "#d6b8a0"].map(col),
  glass: ["#8fb4d6", "#7aa5cc", "#a7c3db", "#6f97bf", "#9fb7c9"].map(col),
  shop: ["#f2f0ea", "#e8e2d6", "#dfe6ea"].map(col),
  awning: ["#d9534f", "#2f7fc1", "#3aa35b", "#e0a526", "#8e5bb5"].map(col),
  ind: ["#a9a79f", "#9c9a94", "#b5b0a2", "#8f949a"].map(col),
  indRoof: col("#6d7278"), chimney: col("#8a6f63"),
  trunk: col("#6b4a2f"), leaves: ["#3f7f3a", "#4e8f3f", "#2f6f3a", "#5a9a45"].map(col),
};

function ramp(stops, t) {
  t = Math.max(0, Math.min(1, t));
  const n = stops.length - 1, i = Math.min(n - 1, Math.floor(t * n)), f = t * n - i;
  return lerpC(stops[i], stops[i + 1], f);
}
const RAMPS = {
  pollution: ["#4caf50", "#cddc39", "#ffc107", "#f44336", "#7b1fa2"].map(col),     // low = good
  land_value: ["#283593", "#1e88e5", "#26c6da", "#c0ca33", "#ffeb3b"].map(col),    // high = good
  good: ["#b71c1c", "#ef6c00", "#fdd835", "#7cb342", "#1b9e5a"].map(col),          // coverage-like: high = good
  bad: ["#2e7d32", "#9ccc65", "#ffca28", "#f4511e", "#8e1b4a"].map(col),           // problem-like: high = bad
  water: ["#1e88e5", "#26a69a", "#9e9d24", "#8d6e63", "#4e342e"].map(col),         // river pollution
};
export const OVERLAY_RAMPS = Object.fromEntries(Object.entries(RAMPS).map(([k, v]) => [k, v.map(c => "#" + c.getHexString())]));

// Map layers the viewer can show. `src` reads a frame; `ramp` layers are 0-9 digit strings.
export const OVERLAYS = [
  { id: "none", label: "City", group: "Basics" },
  { id: "pollution", label: "Pollution", group: "Basics", ramp: "pollution", src: (f) => f.pollution, lo: "clean", hi: "heavy pollution" },
  { id: "land_value", label: "Land value", group: "Basics", ramp: "land_value", src: (f) => f.land_value, lo: "low", hi: "high land value" },
  { id: "happiness", label: "Happiness", group: "Basics", ramp: "good", layer: "happiness", lo: "unhappy", hi: "happy homes", mask: "RM" },
  { id: "noise", label: "Noise", group: "Basics", ramp: "bad", layer: "noise", lo: "quiet", hi: "noisy" },
  { id: "soil", label: "Soil pollution", group: "Basics", ramp: "bad", layer: "soil", lo: "clean soil", hi: "contaminated" },
  { id: "historic", label: "Historic", group: "Basics", ramp: "good", layer: "historic", lo: "", hi: "listed building" },
  { id: "resources", label: "Resources", group: "Economy", cat: { 1: "#c5e384", 2: "#2e7d32", 3: "#9e9e9e", 4: "#4e342e" },
    names: { 1: "fertile soil", 2: "forest", 3: "ore", 4: "oil" } },
  { id: "affordability", label: "Affordability", group: "Economy", ramp: "good", layer: "affordability", lo: "can't pay rent", hi: "affordable", mask: "RM" },
  { id: "power", label: "Power", group: "Utilities", bool: true, src: (f) => f.power },
  { id: "water", label: "Water", group: "Utilities", bool: true, src: (f) => f.water },
  { id: "sewage", label: "Sewage", group: "Utilities", bool: true, src: (f) => f.sewage },
  { id: "river", label: "River pollution", group: "Utilities", ramp: "water", layer: "river", lo: "clean river", hi: "polluted", water: true },
  { id: "traffic", label: "Traffic", group: "Transport", ramp: "bad", layer: "traffic", lo: "free flow", hi: "jammed", roads: true },
  { id: "transit", label: "Bus coverage", group: "Transport", ramp: "good", layer: "transit", lo: "no bus", hi: "near a stop", mask: "RCIOMrciom" },
  { id: "profit", label: "Business profit", group: "Economy", ramp: "good", layer: "profit", lo: "losing money", hi: "profitable", mask: "CIOM" },
  { id: "flood_risk", label: "Flood risk", group: "Events", ramp: "bad", layer: "flood_risk", lo: "safe", hi: "will flood (forecast)" },
  { id: "districts", label: "Districts", group: "Events", districts: true },
  { id: "police", label: "Police cover", group: "Services", ramp: "good", layer: "police", lo: "none", hi: "full cover", mask: "RCIOMrciom" },
  { id: "crime", label: "Crime", group: "Services", ramp: "bad", layer: "crime", lo: "safe", hi: "high crime", mask: "RCIOM?" },
  { id: "fire", label: "Fire cover", group: "Services", ramp: "good", layer: "fire", lo: "none", hi: "full cover", mask: "RCIOMrciom" },
  { id: "fire_risk", label: "Fire risk", group: "Services", ramp: "bad", layer: "fire_risk", lo: "low", hi: "high risk", mask: "RCIOM" },
  { id: "health", label: "Health", group: "Services", ramp: "good", layer: "health", lo: "sick", hi: "healthy", mask: "RCIOM" },
  { id: "garbage", label: "Garbage", group: "Services", ramp: "bad", layer: "garbage", lo: "clean", hi: "piling up" },
  { id: "school", label: "Schools", group: "Services", ramp: "good", layer: "school", lo: "no school", hi: "in range", mask: "RCIOMrciom" },
];
export const OVERLAY_BY_ID = Object.fromEntries(OVERLAYS.map(o => [o.id, o]));
export const DISTRICT_COLORS = ["#9aa19a", "#e07bd8", "#4aa3ff", "#ffb454", "#5fd38d", "#ff6b5e", "#f5d547", "#3fd0c9", "#b99bff"];
const layerStr = (f, o) => o.src ? o.src(f) : (f.layers && f.layers[o.layer]);

// ------------------------------------------------------------ textures
function windowTextures(seed, cols = 4, rows = 8) {
  const W = 64, H = 128, a = document.createElement("canvas"), b = document.createElement("canvas");
  a.width = b.width = W; a.height = b.height = H;
  const g = a.getContext("2d"), e = b.getContext("2d");
  g.fillStyle = "#e9e9e9"; g.fillRect(0, 0, W, H);
  e.fillStyle = "#000"; e.fillRect(0, 0, W, H);
  const cw = W / cols, rh = H / rows;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const x = c * cw + cw * 0.2, y = r * rh + rh * 0.22, w = cw * 0.6, h = rh * 0.56;
    g.fillStyle = "#5d6b7a"; g.fillRect(x, y, w, h);
    const lit = hash(seed * 97 + r * 13 + c, 3);
    if (lit > 0.35) { e.fillStyle = lit > 0.8 ? "#fff3c4" : "#ffcf7a"; e.fillRect(x, y, w, h); }
  }
  const map = new THREE.CanvasTexture(a), emi = new THREE.CanvasTexture(b);
  for (const t of [map, emi]) { t.colorSpace = THREE.SRGBColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 4; }
  return { map, emi };
}
function softDot() {
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const g = c.getContext("2d"), gr = g.createRadialGradient(32, 32, 2, 32, 32, 30);
  gr.addColorStop(0, "rgba(255,255,255,1)"); gr.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

// ----------------------------------------------------- instanced parts
const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _p = new THREE.Vector3(), _s = new THREE.Vector3(), _e = new THREE.Euler(), _c = new THREE.Color();

class Part {
  constructor(parent, geom, mat, cap, { shadow = true, receive = true } = {}) {
    this.mesh = new THREE.InstancedMesh(geom, mat, cap);
    this.mesh.count = 0; this.mesh.castShadow = shadow; this.mesh.receiveShadow = receive;
    this.mesh.frustumCulled = false;
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.items = []; this.tileOf = []; this.cap = cap;
    parent.add(this.mesh);
  }
  begin() { this.items.length = 0; }
  add(tile, x, y, z, sx, sy, sz, rot, color, grow = 0) {
    if (this.items.length < this.cap) this.items.push({ tile, x, y, z, sx, sy, sz, rot, color, grow });
  }
  write(i, it, f) {
    _p.set(it.x, it.y * f, it.z); _s.set(it.sx, Math.max(1e-3, it.sy * f), it.sz);
    _q.setFromEuler(_e.set(0, it.rot, 0));
    this.mesh.setMatrixAt(i, _m.compose(_p, _q, _s));
  }
  commit(now) {
    const n = this.items.length;
    this.anim = [];
    for (let i = 0; i < n; i++) {
      const it = this.items[i];
      it.t0 = it.grow ? now : 0;
      this.write(i, it, it.grow ? 0.02 : 1);
      if (it.grow) this.anim.push(i);
      this.mesh.setColorAt(i, it.color);
      this.tileOf[i] = it.tile;
    }
    this.mesh.count = n;
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
    this.mesh.boundingSphere = null; this.mesh.boundingBox = null;
  }
  recolor(fn) {
    for (let i = 0; i < this.items.length; i++) this.mesh.setColorAt(i, fn(this.items[i]));
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }
  tick(now) {
    if (!this.anim || !this.anim.length) return;
    const still = [];
    for (const i of this.anim) {
      const it = this.items[i], k = Math.min(1, (now - it.t0) / 650);
      const f = k >= 1 ? 1 : 1 - Math.pow(1 - k, 3) * (1 - 0.15 * Math.sin(k * 9));
      this.write(i, it, f);
      if (k < 1) still.push(i);
    }
    this.anim = still;
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}

function baseBox() { const g = new THREE.BoxGeometry(1, 1, 1); g.translate(0, 0.5, 0); return g; }
function baseCyl(seg = 12, top = 1, bot = 1) { const g = new THREE.CylinderGeometry(0.5 * top, 0.5 * bot, 1, seg); g.translate(0, 0.5, 0); return g; }

// ============================================================ CityScene
export class CityScene {
  constructor(container, { onHover, onClick } = {}) {
    this.container = container;
    this.onHover = onHover; this.onClick = onClick;
    this.overlay = "none"; this.night = false; this.nightK = 0; this.quality = "high";
    this.frame = null; this.specials = new Map(); this.emitters = []; this.turbines = [];

    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    r.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    r.shadowMap.enabled = true; r.shadowMap.type = THREE.PCFShadowMap;
    r.toneMapping = THREE.ACESFilmicToneMapping; r.toneMappingExposure = 1.05;
    container.appendChild(r.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 600);
    this.controls = new OrbitControls(this.camera, r.domElement);
    Object.assign(this.controls, { enableDamping: true, dampingFactor: 0.08, maxPolarAngle: 1.42, minDistance: 3,
      maxDistance: 160, screenSpacePanning: false, zoomToCursor: true });
    this.controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };

    this.hemi = new THREE.HemisphereLight(0xdfefff, 0x5a6b45, 0.8);
    this.sun = new THREE.DirectionalLight(0xfff2dd, 2.3);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0004; this.sun.shadow.normalBias = 0.02;
    this.scene.add(this.hemi, this.sun, this.sun.target);
    this.dayBg = col("#b9d7ef"); this.nightBg = col("#0c1426");
    this.scene.background = this.dayBg.clone();
    this.scene.fog = new THREE.Fog(this.dayBg.clone(), 60, 170);

    this._makeEnvironment();
    this.root = new THREE.Group(); this.scene.add(this.root);
    this.clock = new THREE.Timer();
    this.time = { value: 0 };
    this._makeMaterials();
    this._bind();
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => this._loop());
  }

  _makeEnvironment() {
    const sky = new THREE.Scene();
    const geo = new THREE.SphereGeometry(50, 32, 16), cols = [];
    const top = col("#6fa8dc"), hor = col("#f1f6fb"), bot = col("#7c9a63");
    const p = geo.attributes.position;
    for (let k = 0; k < p.count; k++) {
      const yv = p.getY(k) / 50, c = yv > 0 ? lerpC(hor, top, Math.pow(yv, 0.6)) : lerpC(hor, bot, Math.min(1, -yv * 3));
      cols.push(c.r, c.g, c.b);
    }
    geo.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
    sky.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide })));
    const sun = new THREE.Mesh(new THREE.SphereGeometry(3, 16, 8), new THREE.MeshBasicMaterial({ color: new THREE.Color(6, 5.6, 5) }));
    sun.position.set(-25, 35, 18); sky.add(sun);
    const pm = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pm.fromScene(sky, 0.02).texture;
    this.scene.environmentIntensity = 0.55;
    pm.dispose();
  }

  // ---------------------------------------------------------- materials
  _makeMaterials() {
    const std = (o) => new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, ...o });
    const wt = (seed, rep, o = {}) => {
      const { map, emi } = windowTextures(seed);
      map.repeat.set(1, rep); emi.repeat.set(1, rep);
      return std({ map, emissiveMap: emi, emissive: col("#ffd49a"), emissiveIntensity: 0, ...o });
    };
    const roof = std({ color: 0xffffff, roughness: 0.9 });
    this.windowMats = [];
    const faced = (side) => { this.windowMats.push(side); return [side, side, roof, roof, side, side]; };
    this.M = {
      land: std({ roughness: 1, envMapIntensity: 0.35 }),
      bed: std({ color: P.bed, roughness: 1 }),
      water: new THREE.MeshStandardMaterial({ color: P.water, roughness: 0.22, metalness: 0.0, transparent: true, opacity: 0.86,
        emissive: col("#2a78b0"), emissiveIntensity: 0.35, envMapIntensity: 1.0 }),
      plain: std({}),
      roof,
      apt: faced(wt(1, 1.2)),
      tower: faced(wt(2, 2.6)),
      glassMid: faced(wt(3, 1.8, { roughness: 0.35, metalness: 0.35 })),
      glassTall: faced(wt(4, 4.2, { roughness: 0.3, metalness: 0.4 })),
      shop: faced(wt(5, 0.5)),
      dash: std({ color: 0xf2e7b5, roughness: 0.6, emissive: col("#ffe9a8"), emissiveIntensity: 0 }),
      leaves: std({ roughness: 0.9 }),
      lamp: std({ color: 0xfff3c4, emissive: col("#ffd98a"), emissiveIntensity: 0 }),
      sign: std({ color: 0xffffff, emissive: col("#ffffff"), emissiveIntensity: 0.15, roughness: 0.4 }),
      cylTower: wt(6, 3.0),
      cylGlass: wt(7, 4.0, { roughness: 0.3, metalness: 0.4 }),
      overlayTint: null,
    };
    // animated waves on the river surface (vertex displacement + analytic normals)
    this.M.water.onBeforeCompile = (sh) => {
      sh.uniforms.uTime = this.time;
      const wave = `
        vec4 wp0 = modelMatrix * instanceMatrix * vec4(position, 1.0);
        float a1 = wp0.x * 1.7 + wp0.z * 0.6 + uTime * 1.3, a2 = wp0.z * 2.3 - wp0.x * 0.4 + uTime * 0.9, a3 = (wp0.x + wp0.z) * 4.1 + uTime * 2.1;
        float wh = 0.035 * sin(a1) + 0.03 * cos(a2) + 0.008 * sin(a3);
        float ddx = 0.035 * 1.7 * cos(a1) + 0.03 * 0.4 * sin(a2) + 0.008 * 4.1 * cos(a3);
        float ddz = 0.035 * 0.6 * cos(a1) - 0.03 * 2.3 * sin(a2) + 0.008 * 4.1 * cos(a3);`;
      sh.vertexShader = sh.vertexShader.replace("#include <common>", "#include <common>\nuniform float uTime;")
        .replace("#include <beginnormal_vertex>", wave + "\n        vec3 objectNormal = normalize(vec3(-ddx * 2.5, 1.0, -ddz * 2.5));")
        .replace("#include <begin_vertex>", "#include <begin_vertex>\n        transformed.y += wh;");
    };
    this.smokeTex = softDot();
  }

  // --------------------------------------------------------------- map
  setMeta(meta) {
    this.meta = meta;
    this.W = meta.width; this.H = meta.height;
    for (const g of this.specials.values()) this._disposeSpecial(g);
    this.root.clear(); this.specials.clear();
    const W = this.W, H = this.H, full = (W + 2 * MARGIN) * (H + 2 * MARGIN), cap = W * H;
    const R = this.root;
    const box = baseBox();
    this.parts = {
      land: new Part(R, new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), this.M.land, full, { shadow: false }),
      bank: new Part(R, new THREE.BoxGeometry(1, 0.4, 1).translate(0, -0.2, 0), this.M.bed, full, { shadow: false }),
      bed: new Part(R, new THREE.BoxGeometry(1, 0.3, 1).translate(0, -0.55, 0), this.M.bed, full, { shadow: false }),
      water: new Part(R, new THREE.PlaneGeometry(1.001, 1.001, 3, 3).rotateX(-Math.PI / 2), this.M.water, full, { shadow: false }),
      dash: new Part(R, box, this.M.dash, cap * 2, { shadow: false }),
      deck: new Part(R, box, this.M.plain, cap),
      rail: new Part(R, box, this.M.plain, cap * 2),
      pillar: new Part(R, baseCyl(8), this.M.plain, cap),
      house: new Part(R, box, this.M.plain, cap * 3),
      gable: new Part(R, new THREE.CylinderGeometry(0.5, 0.5, 1, 3).rotateZ(Math.PI / 2).rotateY(Math.PI / 2).scale(1, 1, 1).translate(0, 0.25, 0), this.M.plain, cap * 2),
      hroof: new Part(R, new THREE.ConeGeometry(0.72, 1, 4).rotateY(Math.PI / 4).translate(0, 0.5, 0), this.M.plain, cap * 2),
      apt: new Part(R, box, this.M.apt, cap),
      tower: new Part(R, box, this.M.tower, cap),
      shop: new Part(R, box, this.M.shop, cap),
      awning: new Part(R, box, this.M.plain, cap),
      glassMid: new Part(R, box, this.M.glassMid, cap),
      glassTall: new Part(R, box, this.M.glassTall, cap),
      podium: new Part(R, box, this.M.plain, cap),
      rooftop: new Part(R, box, this.M.plain, cap * 2),
      spire: new Part(R, baseCyl(6, 0.2, 1), this.M.plain, cap),
      ind: new Part(R, box, this.M.plain, cap),
      saw: new Part(R, new THREE.CylinderGeometry(0.5, 0.5, 1, 3).rotateZ(Math.PI / 2).rotateY(Math.PI / 2).translate(0, 0.25, 0), this.M.plain, cap * 3),
      chimney: new Part(R, baseCyl(10, 0.8, 1), this.M.plain, cap * 2),
      tank: new Part(R, baseCyl(16), this.M.plain, cap),
      trunk: new Part(R, baseCyl(6, 0.7, 1), this.M.plain, full),
      crown: new Part(R, new THREE.IcosahedronGeometry(0.5, 0).translate(0, 0.5, 0), this.M.leaves, full),
      pine: new Part(R, new THREE.ConeGeometry(0.5, 1, 7).translate(0, 0.5, 0), this.M.leaves, full),
      oak: new Part(R, new THREE.DodecahedronGeometry(0.5, 0).translate(0, 0.5, 0), this.M.leaves, full),
      bush: new Part(R, new THREE.IcosahedronGeometry(0.5, 0).translate(0, 0.35, 0), this.M.leaves, full * 2),
      flower: new Part(R, box, this.M.plain, full * 2, { shadow: false }),
      rock: new Part(R, new THREE.DodecahedronGeometry(0.5, 0).translate(0, 0.25, 0), this.M.plain, full),
      asphalt: new Part(R, new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), this.M.land, cap, { shadow: false }),
      pole: new Part(R, baseCyl(5), this.M.plain, cap),
      lamp: new Part(R, box, this.M.lamp, cap, { shadow: false }),
      balcony: new Part(R, box, this.M.plain, cap * 4),
      sign: new Part(R, box, this.M.sign, cap, { shadow: false }),
      cylTower: new Part(R, baseCyl(14), this.M.cylTower, cap),
      cylGlass: new Part(R, baseCyl(14), this.M.cylGlass, cap),
      parasol: new Part(R, new THREE.ConeGeometry(0.5, 0.35, 8).translate(0, 0.17, 0), this.M.plain, cap * 2),
      barn: new Part(R, box, this.M.plain, full),
      barnRoof: new Part(R, new THREE.CylinderGeometry(0.5, 0.5, 1, 3).rotateZ(Math.PI / 2).rotateY(Math.PI / 2).translate(0, 0.25, 0), this.M.plain, full),
    };
    this.life = new Life(this);
    this._busKey = null;
    this.pickables = [this.parts.land.mesh, this.parts.water.mesh, this.parts.house.mesh, this.parts.hroof.mesh,
      this.parts.apt.mesh, this.parts.tower.mesh, this.parts.shop.mesh, this.parts.glassMid.mesh, this.parts.glassTall.mesh,
      this.parts.ind.mesh, this.parts.deck.mesh, this.parts.podium.mesh, this.parts.cylTower.mesh, this.parts.cylGlass.mesh,
      this.parts.asphalt.mesh];
    this._mountains();

    // outer ground + playable-area frame
    const outer = new THREE.Mesh(new THREE.PlaneGeometry(900, 900).rotateX(-Math.PI / 2), new THREE.MeshStandardMaterial({ color: P.grassOuter, roughness: 1 }));
    outer.position.y = -0.42; outer.receiveShadow = true; R.add(outer); this.outerGround = outer;
    const fr = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(W, 0.001, H)),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.45 }));
    fr.position.y = 0.012; R.add(fr);

    // hover / selection / action highlight
    const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0));
    this.hoverBox = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0xffffff }));
    this.selBox = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0xffd23f }));
    this.hlMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0),
      new THREE.MeshBasicMaterial({ color: 0xffd23f, transparent: true, opacity: 0.22, depthWrite: false }));
    this.hlEdges = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0xffd23f }));
    for (const o of [this.hoverBox, this.selBox, this.hlMesh, this.hlEdges]) { o.visible = false; R.add(o); }

    // smoke
    const N = 1600;
    this.smokeGeo = new THREE.BufferGeometry();
    this.smokeGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    this.smokeGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(N * 4), 4));
    this.smokeGeo.setDrawRange(0, 0);
    this.smoke = new THREE.Points(this.smokeGeo, new THREE.PointsMaterial({ size: 0.9, map: this.smokeTex, vertexColors: true,
      transparent: true, depthWrite: false, sizeAttenuation: true }));
    this.smoke.frustumCulled = false; R.add(this.smoke);

    const s = Math.max(W, H) / 2 + MARGIN;
    Object.assign(this.sun.shadow.camera, { left: -s, right: s, top: s, bottom: -s, near: 1, far: 260 });
    this.sun.shadow.camera.updateProjectionMatrix();
    this.sun.position.set(-s * 0.9, s * 1.6, s * 0.7);
    this.scene.fog.near = s * 1.6; this.scene.fog.far = s * 5;
    this.resetView(false);
  }

  _mountains() {                        // a ring of low-poly hills and snowy peaks on the horizon
    const R = Math.max(this.W, this.H) / 2 + MARGIN + 18;
    this.hills = [];
    for (let k = 0; k < 22; k++) {
      const a = (k / 22) * Math.PI * 2 + hash(k, 90) * 0.2, r = R + hash(k, 91) * 22, h = 6 + hash(k, 92) * 14;
      const w = 10 + hash(k, 93) * 14, x = Math.cos(a) * r, z = Math.sin(a) * r;
      const hill = new THREE.Mesh(new THREE.ConeGeometry(w, h, 7, 1), new THREE.MeshStandardMaterial({ color: lerpC(col("#6f8f62"), col("#8a8f86"), hash(k, 94)), roughness: 1, flatShading: true }));
      hill.position.set(x, -0.42 + h / 2, z); hill.rotation.y = hash(k, 95) * 3; this.root.add(hill);
      this.hills.push({ mat: hill.material, base: hill.material.color.clone() });
      if (h > 13) {
        const cap = new THREE.Mesh(new THREE.ConeGeometry(w * 0.28, h * 0.28, 7, 1), new THREE.MeshStandardMaterial({ color: 0xf4f7f8, roughness: 0.9, flatShading: true }));
        cap.position.set(x, -0.42 + h - h * 0.14 + 0.01, z); cap.rotation.y = hill.rotation.y; this.root.add(cap);
      }
    }
  }

  wx(x) { return x - this.W / 2 + 0.5; }
  wz(y) { return y - this.H / 2 + 0.5; }

  // Terrain outside the playable area mirrors the edge rows so the river and highway run on.
  _outerChar(f, x, y) {
    const W = this.W, H = this.H;
    const cx = Math.min(W - 1, Math.max(0, x)), cy = Math.min(H - 1, Math.max(0, y));
    const ch = f.tiles[cy * W + cx];
    const across = (x < 0 || x >= W) !== (y < 0 || y >= H);     // straight out from one edge, not a corner
    if (across && ch === "H") return "#";                         // highway links run on to the region
    if (across && ch === "@") return "r";                         // rail lines run on from edge stations
    if (x >= 0 && x < W && (ch === "~" || ch === "=" || ch === "Y")) return "~";
    return ".";
  }
  charAt(f, x, y) { return this._outerChar(f, x, y); }
  get margin() { return MARGIN; }
  busColor(k) { return BUS_COLORS[k % BUS_COLORS.length]; }

  // ---------------------------------------------------------- frames
  setFrame(frame, prev, { animate = true } = {}) {
    if (!this.meta) return;
    this.frame = frame;
    const now = performance.now(), W = this.W, H = this.H, T = frame.tiles, L = frame.levels;
    this.windDir = frame.wind || [1, 0];
    const season = frame.season || null;
    this.leafPalette = season === "autumn" ? ["#d9822b", "#c0392b", "#e0a526", "#8d6e3f"].map(col)
      : season === "winter" ? ["#dfe6ea", "#c9d3d8", "#e8eef1", "#b8c4c9"].map(col) : P.leaves;
    const tint = (c) => season === "winter" ? lerpC(c, SNOW, 0.62) : season === "autumn" ? lerpC(c, col("#c2a257"), 0.28)
      : season === "summer" ? lerpC(c, col("#c9c46a"), 0.12) : c;
    if (this.outerGround) this.outerGround.material.color.copy(tint(P.grassOuter));
    for (const h of this.hills || []) h.mat.color.copy(season === "winter" ? lerpC(h.base, SNOW, 0.55) : tint(h.base));
    const hist = frame.layers && frame.layers.historic;
    const grew = (i) => animate && prev && (+prev.levels[i] < +L[i] || prev.tiles[i] !== T[i]);
    const Pt = this.parts;
    for (const k in Pt) Pt[k].begin();
    this.emitters = [];
    const isRoad = (x, y) => x >= 0 && y >= 0 && x < W && y < H && "#=H".includes(T[y * W + x]);

    for (let y = -MARGIN; y < H + MARGIN; y++) for (let x = -MARGIN; x < W + MARGIN; x++) {
      const inside = x >= 0 && y >= 0 && x < W && y < H, i = inside ? y * W + x : -1;
      const ch = inside ? T[i] : this._outerChar(frame, x, y);
      const X = this.wx(x), Z = this.wz(y), h1 = hash(inside ? i : 50000 + (y + 50) * 997 + x, 1);
      const water = ch === "~" || ch === "=" || ch === "Y";
      if (water) {
        Pt.bed.add(i, X, 0, Z, 1, 1, 1, 0, P.bed);
        const rp = inside && frame.layers && frame.layers.river ? +frame.layers.river[i] / 9 : 0;
        Pt.water.add(i, X, -0.14, Z, 1, 1, 1, 0, rp > 0.05 ? lerpC(P.water, P.dirty, 0.35 * Math.min(1, rp)) : P.water);   // a little murkier when polluted
      } else {
        let c;
        if ("#H".includes(ch)) c = inside ? P.sidewalk : P.asphalt;
        else if (ch === "r") c = tint(P.grassOuter);
        else if (ch === "P") c = P.park;
        else if ("rciom".includes(ch)) c = P.zoneLot[ch];
        else if ("RCIOM".includes(ch)) c = P.lotDev[ch];
        else if (inside && this.meta.resources && this.meta.resources[i] === "1") c = lerpC(P.grass[Math.floor(h1 * 4)], col("#a9c96a"), 0.5);
        else if (inside) c = lerpC(P.grass[Math.floor(h1 * 4)], col("#6f9e4f"), 0.5 * this._patch(x, y));
        else c = this._fieldColor(x, y, h1);
        if (ch === "." || ch === "P") c = tint(c);
        Pt.land.add(i, X, 0, Z, 1, 1, 1, 0, c);
        const wet = (dx, dy) => { const xx = x + dx, yy = y + dy;
          const cc = xx >= 0 && yy >= 0 && xx < W && yy < H ? T[yy * W + xx] : this._outerChar(frame, xx, yy);
          return cc === "~" || cc === "=" || cc === "Y"; };
        if (wet(1, 0) || wet(-1, 0) || wet(0, 1) || wet(0, -1)) Pt.bank.add(i, X, -0.003, Z, 1, 1, 1, 0, col("#7d6a4c"));
      }
      if (!inside) {
        if (ch === "r") {                               // rail track beyond the map edge
          const ew = x < 0 || x >= W;
          Pt.podium.add(-1, X, 0, Z, ew ? 1 : 0.46, 0.03, ew ? 0.46 : 1, 0, col("#8d8578"));
          for (const k of [-0.3, 0, 0.3]) Pt.podium.add(-1, X + (ew ? k : 0), 0.03, Z + (ew ? 0 : k), ew ? 0.06 : 0.4, 0.015, ew ? 0.4 : 0.06, 0, col("#5d4037"));
          for (const sgn of [-0.11, 0.11]) Pt.rail.add(-1, X + (ew ? 0 : sgn), 0.045, Z + (ew ? sgn : 0), ew ? 1 : 0.025, 0.025, ew ? 0.025 : 1, 0, col("#9e9e9e"));
          continue;
        }
        if (ch === "#") { Pt.dash.add(-1, X, 0.001, Z, 0.3, 0.012, 0.05, 0, P.asphalt.clone().offsetHSL(0, 0, 0.5)); }
        else if (ch === ".") this._outskirts(x, y, X, Z, h1, season);
        continue;
      }
      const g = grew(i), lv = +L[i], hh = hash(i, 2), h3 = hash(i, 3), rot = (Math.floor(hh * 4)) * Math.PI / 2;
      if (ch === "#" || ch === "H" || ch === "=") {
        const ew = isRoad(x - 1, y) || isRoad(x + 1, y), ns = isRoad(x, y - 1) || isRoad(x, y + 1);
        const r = ew && !ns ? 0 : (ns && !ew ? Math.PI / 2 : -1);
        const rc = frame.road_class ? +frame.road_class[i] : 1;
        if (ch !== "=") {                              // asphalt between two sidewalks
          const wdt = rc >= 3 || ch === "H" ? 1 : rc === 2 ? 0.86 : 0.7;
          Pt.asphalt.add(i, X, 0.003, Z, r === Math.PI / 2 ? wdt : 1, 1, r === 0 ? wdt : 1, 0, rc >= 2 ? P.asphaltDark : P.asphalt);
          if (r >= 0 && hh < 0.33) {                    // street lamp on the sidewalk
            const sx = X + (r ? 0.44 : 0) * (h3 < 0.5 ? 1 : -1), sz = Z + (r ? 0 : 0.44) * (h3 < 0.5 ? 1 : -1);
            Pt.pole.add(i, sx, 0, sz, 0.018, 0.34, 0.018, 0, col("#5f6368"));
            Pt.lamp.add(i, sx, 0.33, sz, 0.05, 0.025, 0.05, 0, col("#fff3c4"));
          }
        }
        if (rc >= 2 && r >= 0 && ch !== "H") {        // avenue: lane lines; highway: edge lines + barriers
          const yy = ch === "=" ? 0.082 : 0.007, off = 0.3;
          for (const sgn of [-1, 1]) Pt.dash.add(i, X + (r ? sgn * off : 0), yy, Z + (r ? 0 : sgn * off), r ? 0.03 : 0.9, 0.012, r ? 0.9 : 0.03, 0,
            rc === 3 ? col("#f5f5f5") : col("#cfd3d8"));
          if (rc === 3) for (const sgn of [-1, 1]) Pt.rail.add(i, X + (r ? sgn * 0.47 : 0), yy, Z + (r ? 0 : sgn * 0.47), r ? 0.05 : 1, 0.1, r ? 1 : 0.05, 0, col("#9ea3a8"));
        }
        if (ch === "=") {
          const rr = r < 0 ? 0 : r;
          Pt.deck.add(i, X, -0.02, Z, 1, 0.1, 1, rr, col("#b8b3a8"));
          Pt.pillar.add(i, X, -0.62, Z, 0.22, 0.62, 0.22, 0, col("#8d8a82"));
          for (const s of [-0.46, 0.46]) Pt.rail.add(i, X + (rr ? s : 0), 0.08, Z + (rr ? 0 : s), rr ? 0.06 : 1, 0.12, rr ? 1 : 0.06, 0, col("#d7d2c7"));
        }
        if (r >= 0) Pt.dash.add(i, X, ch === "=" ? 0.081 : 0.006, Z, r ? 0.05 : 0.32, 0.012, r ? 0.32 : 0.05, 0, col(rc >= 2 ? "#f2c94c" : "#f2e7b5"));
      } else if (ch === "P") {
        for (let k = 0; k < 3; k++) this._tree(i, X + (hash(i, 10 + k) - 0.5) * 0.7, Z + (hash(i, 20 + k) - 0.5) * 0.7, hash(i, 30 + k), g);
      } else if (ch === ".") {
        const rs = this.meta.resources ? +this.meta.resources[i] : 0;
        if (rs === 2) { for (let k = 0; k < 2; k++) this._tree(i, X + (hash(i, 50 + k) - 0.5) * 0.7, Z + (hash(i, 55 + k) - 0.5) * 0.7, 0.6 + 0.4 * hash(i, 58 + k), false); }
        else if (rs === 3 && hh < 0.5) Pt.crown.add(i, X + (h3 - 0.5) * 0.4, -0.05, Z + (hh - 0.25) * 0.8, 0.28, 0.2, 0.24, hh * 6, col("#8d8d8d"));
        else if (rs === 4 && hh < 0.35) Pt.podium.add(i, X, 0, Z, 0.5, 0.012, 0.4, hh * 6, col("#3e3530"));
        else this._wild(i, X, Z, hh, h3, season);
      } else if (lv > 0 && "RCIOM".includes(ch)) {
        this._building(ch, lv, i, X, Z, hh, h3, rot, g, frame.variant ? +frame.variant[i] : 0, frame.density ? +frame.density[i] : 1);
        if (hist && +hist[i] >= 5) {                 // listed historic building: a gold plaque by the door
          Pt.podium.add(i, X + 0.38, 0, Z + 0.38, 0.16, 0.22, 0.03, 0, col("#e0b43a"), g);
        }
      }
    }
    for (const t of frame.flooded || []) {            // floodwater over the flooded bank this month
      const x = t % W, y = Math.floor(t / W);
      Pt.water.add(t, this.wx(x), 0.06, this.wz(y), 1, 1, 1, 0, lerpC(P.water, P.dirty, 0.45));
    }
    for (const k in Pt) Pt[k].commit(now);
    this._specials(frame, prev, now);
    this.life.setFrame(frame);
    this._busLines(frame);
    this._applyOverlay();
  }

  _tree(i, X, Z, h, grow) {
    const s = 0.55 + h * 0.5, P2 = this.parts;
    const leaves = this.leafPalette || P.leaves, lc = leaves[Math.floor(h * 97) % 4];
    const species = Math.floor(hash(Math.floor(X * 7.3 + Z * 13.1), 77) * 4);
    if (h > 0.55 || species === 0) {                 // pine (stays green all year)
      P2.trunk.add(i, X, 0, Z, 0.07 * s, 0.25 * s, 0.07 * s, 0, P.trunk, grow);
      P2.pine.add(i, X, 0.18 * s, Z, 0.42 * s, 0.75 * s, 0.42 * s, h * 6, P.leaves[Math.floor(h * 97) % 4], grow);
    } else if (species === 1) {                      // birch: white trunk, slim crown
      P2.trunk.add(i, X, 0, Z, 0.05 * s, 0.4 * s, 0.05 * s, 0, col("#e8e6df"), grow);
      P2.crown.add(i, X, 0.3 * s, Z, 0.32 * s, 0.6 * s, 0.32 * s, h * 6, lc.clone().offsetHSL(0.03, 0, 0.08), grow);
    } else if (species === 2) {                      // oak: broad crown
      P2.trunk.add(i, X, 0, Z, 0.09 * s, 0.24 * s, 0.09 * s, 0, P.trunk, grow);
      P2.oak.add(i, X, 0.16 * s, Z, 0.68 * s, 0.52 * s, 0.68 * s, h * 6, lc.clone().offsetHSL(0, 0, -0.04), grow);
    } else {
      P2.trunk.add(i, X, 0, Z, 0.07 * s, 0.25 * s, 0.07 * s, 0, P.trunk, grow);
      P2.crown.add(i, X, 0.18 * s, Z, 0.5 * s, 0.5 * s, 0.5 * s, h * 6, lc, grow);
    }
  }

  // soft large-scale patches of darker / lighter grass
  _patch(x, y) { return 0.5 + 0.5 * Math.sin(x * 0.31 + Math.sin(y * 0.17) * 2.1) * Math.cos(y * 0.23 + x * 0.05); }

  _wild(i, X, Z, hh, h3, season) {                   // empty land: a few trees, bushes, flowers and stones
    const Pt = this.parts, leaves = this.leafPalette || P.leaves;
    if (hh < 0.05) this._tree(i, X + (h3 - 0.5) * 0.5, Z + (hh * 10 - 0.5) * 0.5, h3, false);
    else if (hh < 0.13) Pt.bush.add(i, X + (h3 - 0.5) * 0.6, 0, Z + (hash(i, 61) - 0.5) * 0.6, 0.22, 0.2, 0.22, h3 * 6, leaves[Math.floor(h3 * 4)].clone().offsetHSL(0, 0, -0.05));
    else if (hh < 0.16) Pt.rock.add(i, X + (h3 - 0.5) * 0.6, -0.02, Z + (hash(i, 62) - 0.5) * 0.6, 0.14 + h3 * 0.1, 0.12, 0.12, h3 * 6, col("#9a9890"));
    else if (hh < 0.3 && season !== "winter") {
      const fc = FLOWERS[Math.floor(h3 * FLOWERS.length)];
      for (let k = 0; k < 3; k++) Pt.flower.add(i, X + (hash(i, 63 + k) - 0.5) * 0.8, 0, Z + (hash(i, 66 + k) - 0.5) * 0.8, 0.035, 0.03, 0.035, 0, fc);
    }
  }

  _fieldColor(x, y, h1) {                            // the countryside around the map: fields, meadows, woods
    const b = hash(Math.floor((x + 100) / 5) * 131 + Math.floor((y + 100) / 5), 70);
    const base = P.grass[Math.floor(h1 * 4)];
    let c = base;
    if (b < 0.3) c = (Math.floor((y + 100) / 1) % 2) ? col("#c9b458") : col("#9fb34f");        // crop stripes
    else if (b < 0.45) c = col("#b89a5a");                                                     // ploughed field
    else if (b < 0.6) c = col("#5f8f45");                                                      // woods floor
    const season = this.frame && this.frame.season;
    if (season === "winter") c = lerpC(c, SNOW, 0.7);
    else if (season === "autumn") c = lerpC(c, col("#c2a257"), 0.25);
    return lerpC(c, P.grassOuter, 0.15);
  }

  _outskirts(x, y, X, Z, h1, season) {
    const b = hash(Math.floor((x + 100) / 5) * 131 + Math.floor((y + 100) / 5), 70);
    if (b >= 0.45 && b < 0.6) {                      // woods
      if (h1 < 0.6) this._tree(-1, X + (hash(x * 7, y) - 0.5) * 0.6, Z + (hash(y * 7, x) - 0.5) * 0.6, 0.3 + h1, false);
    } else if (b >= 0.6 && h1 < 0.1) this._tree(-1, X, Z, h1 * 5, false);
    else if (b < 0.3 && h1 > 0.985) {                // a farm on the fields
      this.parts.barn.add(-1, X, 0, Z, 0.5, 0.32, 0.36, 0, col("#a93226"));
      this.parts.barnRoof.add(-1, X, 0.32, Z, 0.54, 0.3, 0.4, 0, col("#5d4037"));
    }
  }

  _building(ch, lv, i, X, Z, hh, h3, rot, g, vr = 0, dn = 1) {
    const Pt = this.parts, pick = (arr, k) => arr[Math.floor(hash(i, k) * arr.length) % arr.length];
    if (ch === "I" && vr >= 1 && vr <= 5) return this._industry(vr, lv, i, X, Z, hh, h3, rot, g);
    if (ch === "R" && dn === 0 && lv === 2) {        // low density L2: large detached house (villa if wealthy)
      const rich = vr === 3, hgt = 0.42 + h3 * 0.1;
      Pt.house.add(i, X - 0.06, 0, Z - 0.05, 0.56, hgt, 0.48, rot, rich ? col("#f7f5ef") : pick(P.resWall, 4), g);
      Pt.hroof.add(i, X - 0.06, hgt, Z - 0.05, 0.62, 0.24, 0.5, rot, rich ? col("#3c3f45") : pick(P.resRoof, 5), g);
      if (rich) Pt.podium.add(i, X + 0.28, 0, Z + 0.28, 0.26, 0.02, 0.2, 0, col("#4fc3f7"), g);
      else Pt.house.add(i, X + 0.3, 0, Z + 0.3, 0.18, 0.18, 0.18, rot, pick(P.resWall, 14), g);
      return;
    }
    if (ch === "R" && dn === 1 && lv === 1) {        // high density L1: row houses
      for (let k = -1; k <= 1; k++) {
        const ox = X + (rot % Math.PI ? 0 : k * 0.27), oz = Z + (rot % Math.PI ? k * 0.27 : 0);
        Pt.house.add(i, ox, 0, oz, 0.25, 0.36, 0.5, rot, pick(P.resWall, 20 + k), g);
        Pt.hroof.add(i, ox, 0.36, oz, 0.3, 0.14, 0.5, rot, pick(P.resRoof, 24), g);
      }
      return;
    }
    if (ch === "R") {
      if (lv === 1) this._houses(i, X, Z, hh, h3, rot, g, pick);
      else if (lv === 2) {
        const hgt = 0.62 + h3 * 0.35, c = pick(P.apt, 6), sx = 0.74, sz = 0.7;
        if (hh < 0.35) {                              // L-shaped block around a courtyard
          Pt.apt.add(i, X - 0.12, 0, Z, 0.5, hgt, 0.74, rot, c, g);
          Pt.apt.add(i, X + 0.2, 0, Z - 0.2, 0.36, hgt * 0.85, 0.34, rot, c.clone().offsetHSL(0, 0, -0.05), g);
          this._tree(i, X + 0.26, Z + 0.24, 0.3 + h3 * 0.2, g);
        } else Pt.apt.add(i, X, 0, Z, sx, hgt, sz, rot, c, g);
        const floors = Math.max(1, Math.floor(hgt / 0.22));   // balconies on the street facade
        const fx = Math.round(Math.sin(rot)), fz = Math.round(Math.cos(rot));
        for (let f = 0; f < floors; f++) for (const s of [-0.2, 0.2]) {
          Pt.balcony.add(i, X + fx * 0.37 + (fz ? s : 0), 0.14 + f * 0.22, Z + fz * 0.37 + (fx ? s : 0),
            fz ? 0.2 : 0.07, 0.035, fz ? 0.07 : 0.2, 0, col("#e9e6df"), g);
        }
        Pt.rooftop.add(i, X + 0.12, hgt, Z - 0.1, 0.2, 0.08, 0.16, rot, col("#9aa0a6"), g);
      } else {
        const hgt = [0, 0, 0, 1.5, 2.3, 3.2][lv] + h3 * 1.2, w = lv >= 5 ? 0.66 : 0.6;
        const c = vr === 3 ? col("#e8eef2") : vr === 1 ? pick(["#b98a6e", "#c79b7c", "#a8876b"].map(col), 7) : pick(P.apt, 7);
        const style = Math.floor(hh * 4);
        if (style === 0) {                            // setback tower
          Pt.tower.add(i, X, 0, Z, w + 0.12, hgt * 0.55, w + 0.12, rot, c, g);
          Pt.tower.add(i, X, hgt * 0.55, Z, w - 0.08, hgt * 0.45, w - 0.08, rot, c.clone().offsetHSL(0, 0, 0.04), g);
        } else if (style === 1) {                     // round tower
          Pt.cylTower.add(i, X, 0, Z, w + 0.08, hgt, w + 0.08, 0, c, g);
        } else if (style === 2 && lv >= 4) {          // twin towers on a podium
          Pt.podium.add(i, X, 0, Z, 0.86, 0.2, 0.86, rot, col("#cfd2d6"), g);
          for (const s of [-0.2, 0.2]) Pt.tower.add(i, X + (rot % Math.PI ? 0 : s), 0.2, Z + (rot % Math.PI ? s : 0), 0.34, hgt * (s < 0 ? 1 : 0.8), 0.5, rot, c, g);
        } else {
          Pt.tower.add(i, X, 0, Z, w, hgt, w, rot, c, g);
          const fx = Math.round(Math.sin(rot)), fz = Math.round(Math.cos(rot));
          for (let f = 1; f < Math.floor(hgt / 0.35); f++) Pt.balcony.add(i, X + fx * (w / 2 + 0.03), f * 0.35, Z + fz * (w / 2 + 0.03), fz ? w * 0.8 : 0.06, 0.03, fz ? 0.06 : w * 0.8, 0, col("#dfe3e6"), g);
        }
        Pt.rooftop.add(i, X, hgt, Z, 0.3, 0.12, 0.3, rot, col("#8c9196"), g);
        if (lv >= 5) Pt.spire.add(i, X, hgt + 0.12, Z, 0.05, 0.4, 0.05, 0, col("#d9dde2"), g);
      }
    } else if (ch === "C") {
      if (lv === 1) {
        const c = pick(P.shop, 8), f = 0.36, ax = X + Math.sin(rot) * f, az = Z + Math.cos(rot) * f;
        if (hh < 0.3) {                               // café with parasols out front
          Pt.shop.add(i, X - Math.sin(rot) * 0.12, 0, Z - Math.cos(rot) * 0.12, 0.7, 0.3, 0.5, rot, c, g);
          for (const s of [-0.22, 0.22]) {
            const px = X + Math.sin(rot) * 0.3 + (Math.round(Math.cos(rot)) ? s : 0), pz = Z + Math.cos(rot) * 0.3 + (Math.round(Math.sin(rot)) ? s : 0);
            Pt.pole.add(i, px, 0, pz, 0.015, 0.2, 0.015, 0, col("#555"), g);
            Pt.parasol.add(i, px, 0.18, pz, 0.24, 0.35, 0.24, 0, pick(P.awning, 30 + s * 10), g);
          }
        } else {
          Pt.shop.add(i, X, 0, Z, 0.8, 0.34, 0.66, rot, c, g);
          Pt.awning.add(i, ax, 0.22, az, rot % Math.PI ? 0.14 : 0.82, 0.04, rot % Math.PI ? 0.82 : 0.14, 0, pick(P.awning, 9), g);
          Pt.sign.add(i, X, 0.34, Z, rot % Math.PI ? 0.05 : 0.4, 0.12, rot % Math.PI ? 0.4 : 0.05, 0, pick(SIGNS, 31), g);
        }
      } else if (lv === 2) {
        const hgt = 1.0 + h3 * 0.6;
        if (hh < 0.4) {                               // mall: wide low box + signage
          Pt.shop.add(i, X, 0, Z, 0.9, 0.5, 0.86, rot, pick(P.shop, 10), g);
          Pt.glassMid.add(i, X, 0.5, Z, 0.5, 0.2, 0.5, rot, pick(P.glass, 10), g);
          Pt.sign.add(i, X + Math.sin(rot) * 0.46, 0.34, Z + Math.cos(rot) * 0.46, Math.round(Math.cos(rot)) ? 0.5 : 0.03, 0.1, Math.round(Math.cos(rot)) ? 0.03 : 0.5, 0, pick(SIGNS, 32), g);
        } else {
          Pt.glassMid.add(i, X, 0, Z, 0.72, hgt, 0.72, rot, pick(P.glass, 10), g);
          Pt.rooftop.add(i, X - 0.1, hgt, Z + 0.1, 0.22, 0.1, 0.22, rot, col("#a0a6ad"), g);
          Pt.sign.add(i, X, hgt + 0.05, Z, 0.42, 0.14, 0.06, rot, pick(SIGNS, 33), g);
        }
      } else {
        const hgt = [0, 0, 0, 2.4, 3.2, 4.2][lv] + h3 * 1.8;
        Pt.podium.add(i, X, 0, Z, 0.86, 0.28, 0.86, rot, col("#c9cfd6"), g);
        if (hh < 0.3) Pt.cylGlass.add(i, X, 0.28, Z, 0.62, hgt, 0.62, 0, pick(P.glass, 11), g);
        else if (hh < 0.55) {                         // stepped crown
          Pt.glassTall.add(i, X, 0.28, Z, 0.62, hgt * 0.7, 0.62, rot, pick(P.glass, 11), g);
          Pt.glassTall.add(i, X, 0.28 + hgt * 0.7, Z, 0.44, hgt * 0.3, 0.44, rot, pick(P.glass, 12), g);
        } else Pt.glassTall.add(i, X, 0.28, Z, 0.56, hgt, 0.56, rot, pick(P.glass, 11), g);
        if (hh > 0.4 || lv >= 5) Pt.spire.add(i, X, hgt + 0.28, Z, 0.06, 0.5, 0.06, 0, col("#d9dde2"), g);
      }
    } else if (ch === "M") {                        // mixed use: shops at street level, flats above
      const hgt = [0, 0.45, 0.8, 1.4, 2.1, 2.9][lv] + h3 * 0.5;
      Pt.shop.add(i, X, 0, Z, 0.84, 0.28, 0.78, rot, pick(P.shop, 15), g);
      const f = 0.4, ax = X + Math.sin(rot) * f, az = Z + Math.cos(rot) * f;
      Pt.awning.add(i, ax, 0.2, az, rot % Math.PI ? 0.12 : 0.8, 0.04, rot % Math.PI ? 0.8 : 0.12, 0, pick(P.awning, 16), g);
      Pt.apt.add(i, X, 0.28, Z, 0.72, hgt, 0.66, rot, pick(P.apt, 17), g);
      const fx = Math.round(Math.sin(rot)), fz = Math.round(Math.cos(rot));
      for (let fl = 0; fl < Math.floor(hgt / 0.24); fl++) Pt.balcony.add(i, X + fx * 0.35, 0.4 + fl * 0.24, Z + fz * 0.35, fz ? 0.5 : 0.06, 0.03, fz ? 0.06 : 0.5, 0, col("#6d4c41"), g);
      Pt.rooftop.add(i, X + 0.1, 0.28 + hgt, Z, 0.2, 0.08, 0.18, rot, col("#9aa0a6"), g);
    } else if (ch === "O") {                        // offices: glass slabs, towers and campuses
      const w = [0, 0.6, 0.7, 0.7, 0.74, 0.78][lv], hgt = [0, 0.5, 1.0, 1.9, 2.8, 3.8][lv] + h3 * 0.8;
      const c = pick(["#8c93c7", "#7d86bd", "#9aa1cf", "#6f8fb0", "#a7b6c8"].map(col), 13);
      Pt.podium.add(i, X, 0, Z, 0.84, 0.12, 0.84, rot, col("#c3c6d6"), g);
      if (hh < 0.3 && lv >= 3) Pt.cylGlass.add(i, X, 0.12, Z, w, hgt, w, 0, c, g);
      else if (hh < 0.55 && lv >= 2) {               // two offset slabs
        Pt.glassMid.add(i, X - 0.12, 0.12, Z - 0.1, w * 0.6, hgt, w * 0.6, rot, c, g);
        Pt.glassMid.add(i, X + 0.16, 0.12, Z + 0.14, w * 0.5, hgt * 0.65, w * 0.5, rot, c.clone().offsetHSL(0, 0, 0.05), g);
      } else Pt.glassMid.add(i, X, 0.12, Z, w, hgt, w * 0.7, rot, c, g);
      Pt.rooftop.add(i, X, hgt + 0.12, Z, 0.24, 0.08, 0.2, rot, col("#9aa0a6"), g);
      if (lv >= 4) Pt.pole.add(i, X + 0.08, hgt + 0.2, Z, 0.02, 0.5, 0.02, 0, col("#d0d0d0"), g);
    } else if (ch === "I" && hh < 0.35) {            // warehouse with loading docks and trucks
      const w = [0, 0.8, 0.86, 0.9, 0.9, 0.92][lv], hgt = 0.26 + lv * 0.06, c = pick(P.ind, 12);
      Pt.ind.add(i, X, 0, Z - 0.08, w, hgt, w * 0.62, rot, c, g);
      Pt.gable.add(i, X, hgt, Z - 0.08, w, 0.18, w * 0.64, rot, P.indRoof, g);
      for (let k = 0; k < Math.min(3, lv + 1); k++) {
        const ox = X - 0.28 + k * 0.28;
        Pt.podium.add(i, ox, 0, Z + 0.3, 0.12, 0.08, 0.2, 0, col(k % 2 ? "#eceff1" : "#ef6c00"), g);  // trailers
      }
    } else if (ch === "I") {
      const c = pick(P.ind, 12);
      const w = [0, 0.8, 0.86, 0.9, 0.9, 0.92][lv], hgt = [0, 0.32, 0.48, 0.62, 0.72, 0.84][lv];
      Pt.ind.add(i, X, 0, Z, w, hgt, w * 0.85, rot, c, g);
      for (let k = 0; k < Math.min(3, lv + 1); k++) {
        const off = (k - (Math.min(3, lv + 1) - 1) / 2) * 0.26;
        Pt.saw.add(i, X + (rot % Math.PI ? 0 : off), hgt, Z + (rot % Math.PI ? off : 0), 0.24, 0.34, w * 0.8, rot, P.indRoof, g);
      }
      if (lv >= 2) {
        const n = Math.min(2, lv - 1);
        for (let k = 0; k < n; k++) {
          const cx = X + 0.3 - k * 0.18, cz = Z - 0.28, ch2 = lv >= 3 ? 1.3 : 0.9;
          Pt.chimney.add(i, cx, 0, cz, 0.12, ch2, 0.12, 0, P.chimney, g);
          this.emitters.push({ x: cx, y: ch2 + 0.05, z: cz, rate: 0.35 * lv, dark: 0.55, tile: i });
        }
      }
      if (lv >= 3) Pt.tank.add(i, X - 0.28, 0, Z + 0.28, 0.26, 0.35, 0.26, 0, col("#c8c4b8"), g);
    }
  }

  _houses(i, X, Z, hh, h3, rot, g, pick) {        // detached houses: gable / hip / flat roofs, chimneys, garages, gardens
    const Pt = this.parts, two = hh > 0.55;
    const spots = two ? [[-0.2, -0.18], [0.22, 0.2]] : [[-0.05, -0.05]];
    spots.forEach(([dx, dz], k) => {
      const w = two ? 0.36 : 0.46, hgt = 0.24 + hash(i, 40 + k) * 0.12, ox = X + (rot % Math.PI ? dz : dx), oz = Z + (rot % Math.PI ? dx : dz);
      const style = Math.floor(hash(i, 44 + k) * 3), wall = pick(P.resWall, 4 + k * 3), roof = pick(P.resRoof, 5 + k * 3);
      Pt.house.add(i, ox, 0, oz, w, hgt, w * 0.85, rot, wall, g);
      if (style === 0) Pt.gable.add(i, ox, hgt, oz, w * 1.08, 0.24, w * 0.9, rot, roof, g);
      else if (style === 1) Pt.hroof.add(i, ox, hgt, oz, w * 0.95, 0.2 + h3 * 0.06, w * 0.8, rot, roof, g);
      else {                                        // modern flat roof with a parapet
        Pt.rooftop.add(i, ox, hgt, oz, w * 1.02, 0.03, w * 0.87, rot, col("#6b6f75"), g);
        Pt.house.add(i, ox + 0.05, hgt, oz + 0.05, w * 0.5, 0.14, w * 0.5, rot, wall.clone().offsetHSL(0, 0, 0.06), g);
      }
      if (style !== 2 && hash(i, 48 + k) < 0.6) Pt.rooftop.add(i, ox + w * 0.25, hgt + 0.1, oz - w * 0.2, 0.05, 0.18, 0.05, 0, col("#8d6e63"), g);
    });
    if (!two) {
      if (h3 < 0.5) Pt.house.add(i, X + 0.3, 0, Z + 0.3, 0.2, 0.16, 0.24, rot, col("#d7d2c8"), g);   // garage
      else this._tree(i, X + 0.3, Z + 0.3, 0.25 + h3 * 0.2, g);
      if (hh < 0.3) for (const s of [-1, 1]) Pt.bush.add(i, X + s * 0.42, 0, Z + 0.42, 0.14, 0.12, 0.14, 0, col("#4f7d3a"), g);  // hedge
      else if (hh < 0.45) Pt.flower.add(i, X - 0.3, 0, Z + 0.35, 0.24, 0.03, 0.08, 0, FLOWERS[Math.floor(h3 * FLOWERS.length)], g);
    }
  }

  _industry(vr, lv, i, X, Z, hh, h3, rot, g) {
    const Pt = this.parts;
    if (vr === 1) {                                   // agriculture: crop rows, barn, silo
      for (let k = -2; k <= 2; k++) Pt.podium.add(i, X + (rot % Math.PI ? k * 0.17 : 0), 0, Z + (rot % Math.PI ? 0 : k * 0.17), rot % Math.PI ? 0.1 : 0.9, 0.03 + 0.02 * lv, rot % Math.PI ? 0.9 : 0.1, 0,
        (k + lv) % 2 ? col("#d9c45a") : col("#7fae4a"), g);
      if (lv >= 2) { Pt.house.add(i, X + 0.28, 0, Z - 0.28, 0.3, 0.26, 0.26, rot, col("#a93226"), g); Pt.hroof.add(i, X + 0.28, 0.26, Z - 0.28, 0.34, 0.14, 0.28, rot, col("#5d4037"), g); }
      if (lv >= 3) Pt.tank.add(i, X - 0.3, 0, Z - 0.3, 0.14, 0.62, 0.14, 0, col("#cfd8dc"), g);
    } else if (vr === 2) {                            // forestry: sawmill + log piles + a few trees
      Pt.ind.add(i, X - 0.1, 0, Z - 0.1, 0.5, 0.3 + 0.06 * lv, 0.4, rot, col("#8d6e63"), g);
      for (let k = 0; k < lv + 1; k++) Pt.podium.add(i, X + 0.28, 0.04 * k, Z + 0.25 - k * 0.02, 0.26, 0.05, 0.14, 0, col(k % 2 ? "#795548" : "#a1887f"), g);
      this._tree(i, X - 0.35, Z + 0.35, 0.8, g);
    } else if (vr === 3) {                            // ore mine: headframe + spoil heaps
      Pt.ind.add(i, X + 0.2, 0, Z + 0.2, 0.36, 0.26, 0.36, rot, col("#757575"), g);
      for (const d of [-0.1, 0.1]) Pt.ind.add(i, X - 0.2 + d, 0, Z - 0.2, 0.05, 0.6 + 0.12 * lv, 0.05, 0, col("#424242"), g);
      Pt.ind.add(i, X - 0.2, 0.6 + 0.12 * lv, Z - 0.2, 0.26, 0.05, 0.1, 0, col("#424242"), g);
      Pt.crown.add(i, X - 0.28, 0, Z + 0.28, 0.3 + 0.05 * lv, 0.2 + 0.05 * lv, 0.3, hh * 6, col("#9e9e9e"), g);
      this.emitters.push({ x: X + 0.2, y: 0.4, z: Z + 0.2, rate: 0.3, dark: 0.7, tile: i });
    } else if (vr === 4) {                            // oil: pumpjacks, tanks, a flare
      for (let k = 0; k < Math.min(3, lv); k++) {
        const ox = X - 0.25 + k * 0.25;
        Pt.ind.add(i, ox, 0, Z + 0.25, 0.06, 0.24, 0.06, 0, col("#37474f"), g);
        Pt.ind.add(i, ox, 0.24, Z + 0.25, 0.3, 0.05, 0.06, 0, col("#f9a825"), g);
      }
      Pt.tank.add(i, X - 0.2, 0, Z - 0.22, 0.3, 0.26, 0.3, 0, col("#eceff1"), g);
      if (lv >= 2) Pt.tank.add(i, X + 0.2, 0, Z - 0.22, 0.26, 0.22, 0.26, 0, col("#cfd8dc"), g);
      if (lv >= 3) { Pt.chimney.add(i, X + 0.38, 0, Z + 0.38, 0.06, 0.8, 0.06, 0, col("#616161"), g); this.emitters.push({ x: X + 0.38, y: 0.85, z: Z + 0.38, rate: 0.8, dark: 0.9, fire: true, tile: i }); }
    } else {                                          // high-tech campus: low glass buildings, green roof
      const hgt = [0, 0.35, 0.5, 0.7, 1.0, 1.4][lv];
      Pt.podium.add(i, X, 0, Z, 0.9, 0.03, 0.9, 0, col("#8bc34a"), g);
      Pt.glassMid.add(i, X - 0.12, 0.03, Z - 0.1, 0.56, hgt, 0.46, rot, col("#b3e5fc"), g);
      Pt.glassMid.add(i, X + 0.26, 0.03, Z + 0.24, 0.3, hgt * 0.6, 0.3, rot, col("#81d4fa"), g);
      Pt.rooftop.add(i, X - 0.12, hgt + 0.03, Z - 0.1, 0.4, 0.02, 0.3, rot, col("#1f3a73"), g);
    }
  }

  // ------------------------------------------------ unique structures
  _specials(frame, prev, now) {
    const W = this.W, want = new Map();
    const T = frame.tiles;
    const part = new Set(), size = new Map();
    for (const [a, fw, fh] of frame.multi || []) {
      size.set(a, [fw, fh]);
      for (let j = 0; j < fh; j++) for (let k = 0; k < fw; k++) if (j || k) part.add(a + j * W + k);
    }
    for (let i = 0; i < T.length; i++) {
      const ch = T[i];
      const lv = facilityTier(frame, i);                   // in the key, so an upgrade rebuilds the model
      if (SPECIALS.includes(ch) && !part.has(i)) want.set(`${i}:${ch}:${lv}`, { i, ch, lv, size: size.get(i) });
    }
    for (const [k, g] of this.specials) if (!want.has(k)) { this.root.remove(g); this._disposeSpecial(g); this.specials.delete(k); }
    this.turbines = [];
    const plantEmitters = [];
    for (const [k, { i, ch, lv, size: sz }] of want) {
      let g = this.specials.get(k);
      if (!g) {
        g = this._buildSpecial(ch, i, lv);
        const x = i % W, y = Math.floor(i / W), [fw, fh] = sz || [1, 1];
        g.position.set(this.wx(x + (fw - 1) / 2), 0, this.wz(y + (fh - 1) / 2));
        const s = NATIVE_SIZE.includes(ch) ? 1 : Math.min(fw, fh) * 0.92;
        g.userData.base = [s, NATIVE_SIZE.includes(ch) ? 1 : (s > 1 ? 1 + (s - 1) * 0.55 : 1), s];
        g.scale.set(...g.userData.base);
        g.userData.tile = i;
        if (ch === "&") {                            // quay and crane face the river
          const wet = (dx, dy) => "~=Y".includes(this._outerChar(frame, x + dx, y + dy)) || (x + dx >= 0 && y + dy >= 0 && x + dx < W && y + dy < this.H && "~=Y".includes(T[(y + dy) * W + x + dx]));
          g.rotation.y = wet(0, 1) ? 0 : wet(1, 0) ? Math.PI / 2 : wet(-1, 0) ? -Math.PI / 2 : wet(0, -1) ? Math.PI : 0;
        }
        g.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; o.userData.tile = i; } });
        if (prev) { g.userData.t0 = now; g.scale.set(0.01, 0.01, 0.01); }
        this.root.add(g); this.specials.set(k, g);
      }
      if (g.userData.rotor) this.turbines.push(g.userData.rotor);
      const b = g.userData.base || [1, 1, 1];
      for (const e of g.userData.stacks || []) plantEmitters.push({ x: g.position.x + e.x * b[0], y: e.y * b[1], z: g.position.z + e.z * b[2], rate: e.rate, dark: e.dark, tile: i, fire: e.fire });
    }
    const fires = (frame.burning || []).map(t => ({ x: this.wx(t % W), y: 0.25, z: this.wz(Math.floor(t / W)), rate: 1.6, dark: 0.9, fire: true, tile: t }));
    this.emitters = [...fires, ...plantEmitters, ...this.emitters];   // fires, then power plants first
    this._setupSmoke();
  }

  _disposeSpecial(g) {            // free a removed model's geometry and its own materials (not the shared _mat cache)
    const shared = new Set(this._mats ? this._mats.values() : []);
    g.traverse(o => {
      if (!o.isMesh) return;
      o.geometry.dispose();
      for (const m of [].concat(o.material)) if (!shared.has(m)) m.dispose();
    });
  }

  _mat(color, o = {}) {           // shared materials for unique structures
    const key = color + JSON.stringify(o);
    this._mats = this._mats || new Map();
    if (!this._mats.has(key)) this._mats.set(key, new THREE.MeshStandardMaterial({ color, roughness: 0.75, ...o }));
    return this._mats.get(key);
  }

  _buildSpecial(ch, i, lv = 1) {
    const g = new THREE.Group();
    const m = (geo, color, o = {}) => { const mesh = new THREE.Mesh(geo, this._mat(color, o)); g.add(mesh); return mesh; };
    const bx = (w, h, d, x, y, z, c, o) => { const me = m(new THREE.BoxGeometry(w, h, d), c, o); me.position.set(x, y + h / 2, z); return me; };
    const cy = (r1, r2, h, x, y, z, c, seg = 16, o) => { const me = m(new THREE.CylinderGeometry(r1, r2, h, seg), c, o); me.position.set(x, y + h / 2, z); return me; };
    if (ch === "T") {                               // town hall
      bx(0.92, 0.12, 0.92, 0, 0, 0, "#d8d2c4");
      bx(0.78, 0.5, 0.6, 0, 0.12, -0.05, "#efe9dc");
      for (let k = -2; k <= 2; k++) cy(0.035, 0.035, 0.42, k * 0.15, 0.12, 0.3, "#ffffff", 10);
      bx(0.82, 0.08, 0.2, 0, 0.54, 0.28, "#e6dfd0");
      cy(0.2, 0.22, 0.14, 0, 0.62, -0.05, "#efe9dc");
      const dome = m(new THREE.SphereGeometry(0.2, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2), "#7a4fb3", { metalness: 0.4, roughness: 0.4 });
      dome.position.set(0, 0.76, -0.05);
      cy(0.01, 0.01, 0.35, 0, 0.9, -0.05, "#ddd", 6);
      bx(0.16, 0.09, 0.01, 0.08, 1.14, -0.05, "#e0b43a");
    } else if (ch === "E") {                        // coal plant
      bx(0.88, 0.4, 0.62, 0, 0, 0.12, "#5d5f63");
      bx(0.4, 0.22, 0.3, -0.2, 0.4, 0.14, "#4a4c50");
      const st = [];
      for (const sx of [-0.25, 0.22]) {
        cy(0.08, 0.11, 1.5, sx, 0, -0.3, "#9c8b80", 16);
        cy(0.085, 0.085, 0.12, sx, 1.1, -0.3, "#c9453b", 16);
        st.push({ x: sx, y: 1.55, z: -0.3, rate: 1.4, dark: 0.3 });
      }
      bx(0.3, 0.18, 0.3, 0.28, 0.4, 0.2, "#2b2b2b");
      g.userData.stacks = st;
    } else if (ch === "G") {                        // gas plant
      bx(0.55, 0.36, 0.45, -0.15, 0, 0.18, "#b7b9bd");
      for (const [sx, sz] of [[0.22, -0.2], [0.22, 0.22]]) {
        const s = m(new THREE.SphereGeometry(0.17, 18, 12), "#f0f0ee", { metalness: 0.3, roughness: 0.4 }); s.position.set(sx, 0.3, sz);
        cy(0.03, 0.03, 0.14, sx, 0, sz, "#888", 6);
      }
      cy(0.05, 0.06, 0.95, -0.3, 0, -0.25, "#d9822b", 12);
      g.userData.stacks = [{ x: -0.3, y: 1.0, z: -0.25, rate: 0.6, dark: 0.75 }];
    } else if (ch === "S") {                        // solar farm
      bx(0.94, 0.02, 0.94, 0, 0, 0, "#9aa39a");
      for (const sx of [-0.23, 0.23]) for (const sz of [-0.23, 0.23]) {
        cy(0.015, 0.015, 0.12, sx, 0, sz, "#777", 6);
        const p = bx(0.4, 0.02, 0.36, sx, 0.12, sz, "#1f3a73", { metalness: 0.6, roughness: 0.25 });
        p.rotation.x = -0.5;
      }
    } else if (ch === "W") {                        // wind turbine
      const adv = tierRank(ch, lv) >= 2, th = adv ? 2.6 : 2.0, bl = adv ? 1.1 : 0.85;   // advanced turbine: taller, longer blades
      bx(0.5, 0.04, 0.5, 0, 0, 0, "#b8bdb5");
      cy(0.03, adv ? 0.08 : 0.06, th, 0, 0, 0, "#f4f5f6", 12);
      bx(adv ? 0.13 : 0.1, adv ? 0.12 : 0.1, adv ? 0.3 : 0.22, 0, th - 0.04, 0.02, "#e8eaec");
      const rotor = new THREE.Group(); rotor.position.set(0, th + 0.01, adv ? 0.18 : 0.14); g.add(rotor);
      for (let k = 0; k < 3; k++) {
        const b = new THREE.Mesh(new THREE.BoxGeometry(0.05, bl, 0.015).translate(0, bl / 2 + 0.005, 0), new THREE.MeshStandardMaterial({ color: "#fbfbfb", roughness: 0.5 }));
        b.rotation.z = k * Math.PI * 2 / 3; rotor.add(b);
      }
      rotor.rotation.z = hash(i, 40) * 6;
      g.userData.rotor = rotor;
    } else if (ch === "U") {                        // water pump
      bx(0.6, 0.34, 0.5, 0, 0, 0.1, "#dfe7ea");
      bx(0.64, 0.06, 0.54, 0, 0.34, 0.1, "#1f8a8a");
      const pipe = m(new THREE.CylinderGeometry(0.06, 0.06, 0.9, 10), "#1f8a8a", { metalness: 0.5, roughness: 0.4 });
      pipe.rotation.z = Math.PI / 2; pipe.position.set(0.1, 0.12, -0.25);
      cy(0.14, 0.14, 0.26, -0.28, 0, -0.22, "#8fb9c7", 14);
    } else if (ch === "V") {                        // water tower
      for (const [sx, sz] of [[-0.2, -0.2], [0.2, -0.2], [-0.2, 0.2], [0.2, 0.2]]) cy(0.025, 0.025, 1.1, sx, 0, sz, "#8a9aa0", 6);
      cy(0.34, 0.3, 0.45, 0, 1.05, 0, "#2fa3b3", 20, { metalness: 0.3, roughness: 0.4 });
      const top = m(new THREE.ConeGeometry(0.36, 0.2, 20), "#1d7480"); top.position.set(0, 1.6, 0);
    } else if (ch === "K") {                        // police station
      bx(0.86, 0.1, 0.86, 0, 0, 0, "#c9ced6");
      bx(0.72, 0.42, 0.56, 0, 0.1, 0.02, "#e8edf3");
      bx(0.74, 0.08, 0.58, 0, 0.52, 0.02, "#1f4fa8");
      bx(0.3, 0.06, 0.02, 0, 0.36, 0.31, "#1f4fa8");
      bx(0.12, 0.05, 0.06, -0.08, 0.6, 0.02, "#e53935", { emissive: "#e53935", emissiveIntensity: 0.6 });
      bx(0.12, 0.05, 0.06, 0.08, 0.6, 0.02, "#1e88e5", { emissive: "#1e88e5", emissiveIntensity: 0.6 });
      cy(0.012, 0.012, 0.75, 0.36, 0.1, 0.36, "#bbb", 6);
      bx(0.14, 0.09, 0.01, 0.43, 0.74, 0.36, "#1f4fa8");
    } else if (ch === "J") {                        // jail
      bx(0.94, 0.06, 0.94, 0, 0, 0, "#9ea3a8");
      for (const [w, d, x, z] of [[0.9, 0.06, 0, -0.42], [0.9, 0.06, 0, 0.42], [0.06, 0.9, -0.42, 0], [0.06, 0.9, 0.42, 0]]) bx(w, 0.32, d, x, 0.06, z, "#7d8388");
      bx(0.44, 0.46, 0.4, 0, 0.06, 0, "#b0b4b8");
      for (let k = -1; k <= 1; k++) bx(0.02, 0.2, 0.01, k * 0.12, 0.2, 0.205, "#44484c");
      cy(0.07, 0.07, 0.7, 0.36, 0.06, -0.36, "#8a9095", 8);
      bx(0.2, 0.12, 0.2, 0.36, 0.76, -0.36, "#5e6368");
    } else if (ch === "F") {                        // fire station
      bx(0.9, 0.08, 0.9, 0, 0, 0, "#c9c4bb");
      bx(0.8, 0.44, 0.6, 0, 0.08, 0.05, "#c62828");
      bx(0.8, 0.06, 0.62, 0, 0.52, 0.05, "#8e1c1c");
      for (const x of [-0.2, 0.2]) bx(0.3, 0.3, 0.02, x, 0.08, 0.36, "#eeeeee");
      bx(0.2, 0.75, 0.2, -0.3, 0.08, -0.25, "#b71c1c");
      bx(0.22, 0.05, 0.22, -0.3, 0.83, -0.25, "#6d1010");
    } else if (ch === "^") {                        // emergency shelter
      bx(0.9, 0.05, 0.9, 0, 0, 0, "#9aa39a");
      const d = m(new THREE.SphereGeometry(0.36, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2), "#b8b39c"); d.position.set(0, 0.05, 0); d.scale.y = 0.7;
      bx(0.2, 0.14, 0.04, 0, 0.05, 0.33, "#2e7d32", { emissive: "#2e7d32", emissiveIntensity: 0.3 });
    } else if (ch === "A") {                        // clinic
      bx(0.86, 0.06, 0.86, 0, 0, 0, "#cfd8dc");
      bx(0.7, 0.36, 0.55, 0, 0.06, 0, "#fafafa");
      bx(0.26, 0.06, 0.07, 0, 0.42, 0, "#e53935");
      bx(0.07, 0.06, 0.26, 0, 0.42, 0, "#e53935");
      bx(0.3, 0.14, 0.02, 0, 0.12, 0.285, "#90caf9");
    } else if (ch === "h") {                        // hospital (2x2)
      bx(0.94, 0.06, 0.94, 0, 0, 0, "#cfd8dc");
      bx(0.5, 1.1, 0.5, -0.16, 0.06, -0.12, "#f5f7f8");
      bx(0.86, 0.45, 0.34, 0, 0.06, 0.26, "#eceff1");
      const pad = m(new THREE.CylinderGeometry(0.2, 0.2, 0.02, 20), "#455a64"); pad.position.set(-0.16, 1.17, -0.12);
      bx(0.2, 0.025, 0.05, -0.16, 1.18, -0.12, "#ffffff"); bx(0.05, 0.025, 0.2, -0.16, 1.18, -0.12, "#ffffff");
      bx(0.26, 0.26, 0.02, 0.2, 0.2, 0.44, "#e53935", { emissive: "#e53935", emissiveIntensity: 0.35 });
    } else if (ch === "L") {                        // landfill
      bx(0.96, 0.02, 0.96, 0, 0, 0, "#8d7b5a");
      const mound = m(new THREE.SphereGeometry(0.42, 14, 8, 0, Math.PI * 2, 0, Math.PI / 2), "#7a6a4f", { flatShading: true }); mound.scale.set(1, 0.55, 0.85);
      for (let k = 0; k < 5; k++) bx(0.08, 0.06, 0.08, (hash(i, 60 + k) - 0.5) * 0.6, 0.16 + hash(i, 70 + k) * 0.06, (hash(i, 80 + k) - 0.5) * 0.5,
        ["#c0392b", "#2980b9", "#f1c40f", "#7f8c8d", "#27ae60"][k]);
      bx(0.16, 0.12, 0.26, 0.34, 0.02, 0.32, "#e0a526");
    } else if (ch === "X") {                        // incinerator
      bx(0.86, 0.06, 0.86, 0, 0, 0, "#8f8f8f");
      bx(0.62, 0.5, 0.5, -0.08, 0.06, 0.1, "#6b6e72");
      bx(0.62, 0.06, 0.52, -0.08, 0.56, 0.1, "#3d4044");
      cy(0.08, 0.1, 1.35, 0.28, 0.06, -0.26, "#5b5e62", 14);
      cy(0.085, 0.085, 0.1, 0.28, 1.2, -0.26, "#e0a526", 14);
      g.userData.stacks = [{ x: 0.28, y: 1.45, z: -0.26, rate: 0.9, dark: 0.35 }];
    } else if (ch === "Z") {                        // cemetery
      bx(0.96, 0.03, 0.96, 0, 0, 0, "#6f9f5a");
      for (let r = 0; r < 3; r++) for (let k = 0; k < 4; k++) bx(0.07, 0.09 + 0.03 * hash(i, r * 4 + k), 0.03, -0.3 + k * 0.2, 0.03, -0.3 + r * 0.25, "#b0b4b8");
      cy(0.03, 0.03, 0.2, 0.36, 0.03, 0.36, "#6b4a2f", 6);
      const cr = m(new THREE.IcosahedronGeometry(0.14, 0), "#2f5f3a"); cr.position.set(0.36, 0.33, 0.36);
    } else if (ch === "N") {                        // elementary school
      bx(0.92, 0.04, 0.92, 0, 0, 0, "#b9c79a");
      bx(0.7, 0.34, 0.4, 0, 0.04, -0.2, "#f3d99b");
      const roof = m(new THREE.CylinderGeometry(0.5, 0.5, 0.7, 3), "#d35400"); roof.rotation.z = Math.PI / 2; roof.rotation.y = Math.PI / 2;
      roof.scale.set(0.42, 1, 0.3); roof.position.set(0, 0.46, -0.2);
      bx(0.36, 0.02, 0.3, -0.18, 0.04, 0.26, "#c77d4e");
      bx(0.08, 0.18, 0.08, 0.22, 0.04, 0.26, "#e74c3c"); bx(0.16, 0.03, 0.05, 0.22, 0.22, 0.26, "#3498db");
    } else if (ch === "B") {                        // high school
      bx(0.96, 0.03, 0.96, 0, 0, 0, "#6fae5a");
      bx(0.86, 0.48, 0.36, 0, 0.03, -0.26, "#b5654a");
      bx(0.88, 0.05, 0.38, 0, 0.51, -0.26, "#6d3b2c");
      bx(0.7, 0.012, 0.36, 0, 0.03, 0.2, "#3f8f3a");
      bx(0.7, 0.014, 0.012, 0, 0.035, 0.2, "#ffffff");
      bx(0.012, 0.014, 0.36, 0, 0.035, 0.2, "#ffffff");
    } else if (ch === "Q") {                        // university
      bx(0.96, 0.03, 0.96, 0, 0, 0, "#7fb069");
      bx(0.86, 0.14, 0.66, 0, 0.03, -0.1, "#dcd6c8");
      bx(0.76, 0.5, 0.5, 0, 0.17, -0.12, "#efe9dc");
      for (let k = -3; k <= 3; k++) cy(0.03, 0.03, 0.46, k * 0.11, 0.17, 0.16, "#ffffff", 10);
      bx(0.8, 0.07, 0.12, 0, 0.63, 0.15, "#e6dfd0");
      const dome = m(new THREE.SphereGeometry(0.22, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2), "#2d6a9f", { metalness: 0.4, roughness: 0.4 });
      dome.position.set(0, 0.67, -0.12);
      for (const x of [-0.34, 0.34]) { const t = m(new THREE.IcosahedronGeometry(0.1, 0), "#3f7f3a"); t.position.set(x, 0.18, 0.36); }
    } else if (ch === "D") {                        // library
      bx(0.9, 0.08, 0.9, 0, 0, 0, "#d2ccbd");
      bx(0.72, 0.38, 0.5, 0, 0.08, -0.08, "#e4dccb");
      for (let k = -2; k <= 2; k++) cy(0.028, 0.028, 0.36, k * 0.14, 0.08, 0.22, "#f5f1e8", 10);
      const pd = m(new THREE.CylinderGeometry(0.5, 0.5, 0.78, 3), "#2e5b3c"); pd.rotation.z = Math.PI / 2; pd.rotation.y = Math.PI / 2;
      pd.scale.set(0.3, 1, 0.26); pd.position.set(0, 0.52, 0.02);
    } else if (ch === "%") {                        // sewage outlet
      bx(0.5, 0.2, 0.5, -0.1, 0, 0, "#8d8575");
      const pipe = m(new THREE.CylinderGeometry(0.07, 0.07, 0.6, 10), "#5d5446", { metalness: 0.4 });
      pipe.rotation.z = Math.PI / 2; pipe.position.set(0.28, 0.08, 0);
      bx(0.1, 0.02, 0.2, 0.52, -0.03, 0, "#6d5f3f");
    } else if (ch === "*") {                        // sewage treatment plant
      bx(0.94, 0.04, 0.94, 0, 0, 0, "#a8a89e");
      for (const [x, z] of [[-0.22, -0.2], [0.22, -0.2], [-0.22, 0.22]]) {
        cy(0.19, 0.19, 0.14, x, 0.04, z, "#9e9e96", 20);
        cy(0.16, 0.16, 0.01, x, 0.175, z, "#4f8f6f", 20, { roughness: 0.3 });
      }
      bx(0.26, 0.3, 0.26, 0.24, 0.04, 0.24, "#d7d4ca");
    } else if (ch === "Y") {                        // hydro dam (on the river)
      bx(1.0, 0.55, 0.3, 0, -0.35, 0, "#b8b3a8");
      bx(1.0, 0.05, 0.34, 0, 0.2, 0, "#8d8a82");
      bx(0.34, 0.24, 0.3, 0.2, 0.2, 0.2, "#d7d2c7");
      for (let k = -1; k <= 1; k++) bx(0.12, 0.2, 0.02, k * 0.28, -0.2, 0.16, "#6fb6e8", { emissive: "#2a78b0", emissiveIntensity: 0.4 });
    } else if (ch === "|") {                        // levee
      bx(1.0, 0.18, 0.7, 0, 0, 0, "#6f8f4f");
      bx(1.0, 0.04, 0.4, 0, 0.18, 0, "#8d7b5a");
    } else if (ch === "@") {                        // train station
      bx(0.96, 0.06, 0.96, 0, 0, 0, "#9e9e9e");
      for (const z of [-0.3, 0.3]) bx(0.96, 0.03, 0.05, 0, 0.06, z, "#5d4037");
      bx(0.96, 0.08, 0.3, 0, 0.06, 0, "#d7ccc8");
      for (const x of [-0.35, 0, 0.35]) cy(0.02, 0.02, 0.36, x, 0.14, 0, "#555", 6);
      bx(1.0, 0.03, 0.46, 0, 0.5, 0, "#b23b3b");
    } else if (ch === "&") {                        // commercial port: quay on the river side (+z, turned in _specials)
      bx(0.98, 0.08, 0.98, 0, 0, 0, "#9a968c");
      bx(0.98, 0.01, 0.06, 0, 0.08, 0.44, "#e8c040");
      const cols = ["#c0392b", "#2980b9", "#27ae60", "#f39c12", "#8e44ad", "#16a085"];
      for (let r = 0; r < 2; r++) for (let k = 0; k < 3; k++) for (let h = 0; h < 1 + ((r + k + i) % 3); h++)
        bx(0.26, 0.09, 0.12, -0.3 + k * 0.3, 0.08 + h * 0.09, -0.36 + r * 0.15, cols[(r * 3 + k + h * 2) % cols.length]);
      for (const x of [-0.26, 0.26]) for (const z of [0.02, 0.36]) bx(0.04, 0.9, 0.04, x, 0.08, z, "#f2b705");
      bx(0.6, 0.06, 0.06, 0, 0.98, 0.02, "#f2b705"); bx(0.6, 0.06, 0.06, 0, 0.98, 0.36, "#f2b705");
      bx(0.08, 0.06, 1.3, 0, 1.04, 0.5, "#f2b705");
      bx(0.14, 0.12, 0.14, 0, 0.92, 0.3, "#555a60");
      bx(0.01, 0.4, 0.01, 0, 0.6, 0.9, "#333333"); bx(0.16, 0.05, 0.1, 0, 0.56, 0.9, "#2980b9");
    } else if (ch === "+") {                        // plaza
      bx(0.98, 0.03, 0.98, 0, 0, 0, "#d8d2c4");
      cy(0.24, 0.26, 0.08, 0, 0.03, 0, "#bdb6a6", 20);
      cy(0.2, 0.2, 0.01, 0, 0.11, 0, "#4aa6e0", 20, { emissive: "#2a78b0", emissiveIntensity: 0.3 });
      cy(0.03, 0.05, 0.3, 0, 0.11, 0, "#e6dfd0", 10);
      for (const [x, z] of [[-0.38, -0.38], [0.38, -0.38], [-0.38, 0.38], [0.38, 0.38]]) { const t = m(new THREE.IcosahedronGeometry(0.1, 0), "#3f7f3a"); t.position.set(x, 0.14, z); }
    } else if (ch === "$") {                        // stadium
      const ring = m(new THREE.CylinderGeometry(0.48, 0.44, 0.36, 28, 1, true), "#dfe3e8", { side: THREE.DoubleSide }); ring.position.y = 0.18; ring.scale.z = 0.8;
      const pitch = m(new THREE.CylinderGeometry(0.34, 0.34, 0.02, 28), "#3f9f4a"); pitch.position.y = 0.02; pitch.scale.z = 0.8;
      const roof = m(new THREE.TorusGeometry(0.44, 0.05, 6, 28), "#b0bec5"); roof.rotation.x = Math.PI / 2; roof.position.y = 0.36; roof.scale.y = 0.8;
    } else if (ch === "!") {                        // monument
      bx(0.7, 0.1, 0.7, 0, 0, 0, "#d8d2c4");
      bx(0.4, 0.12, 0.4, 0, 0.1, 0, "#c9c1ae");
      const ob = m(new THREE.CylinderGeometry(0.04, 0.12, 1.8, 4), "#efe9dc"); ob.position.y = 1.12; ob.rotation.y = Math.PI / 4;
      const tip = m(new THREE.ConeGeometry(0.06, 0.14, 4), "#e0b43a", { metalness: 0.6, roughness: 0.3 }); tip.position.y = 2.09; tip.rotation.y = Math.PI / 4;
    } else if (ch === ":") {                        // metro station entrance
      bx(0.9, 0.04, 0.9, 0, 0, 0, "#cfd3d8");
      bx(0.5, 0.26, 0.36, 0, 0.04, 0, "#37474f");
      const arch = m(new THREE.CylinderGeometry(0.2, 0.2, 0.36, 16, 1, false, 0, Math.PI), "#90a4ae", { side: THREE.DoubleSide });
      arch.rotation.z = Math.PI / 2; arch.rotation.y = Math.PI / 2; arch.position.set(0, 0.3, 0);
      cy(0.02, 0.02, 0.7, 0.32, 0.04, 0.3, "#555", 6);
      const sign = m(new THREE.CylinderGeometry(0.1, 0.1, 0.03, 16), "#d32f2f", { emissive: "#d32f2f", emissiveIntensity: 0.6 });
      sign.rotation.x = Math.PI / 2; sign.position.set(0.32, 0.78, 0.3);
    } else if (ch === "?") {                        // casino (2x2 when placed)
      bx(0.94, 0.06, 0.94, 0, 0, 0, "#2b1d3a");
      bx(0.7, 0.42, 0.6, 0, 0.06, -0.08, "#6a1b9a");
      bx(0.36, 0.95, 0.36, -0.14, 0.06, -0.14, "#4a148c", { metalness: 0.4, roughness: 0.35 });
      bx(0.74, 0.05, 0.64, 0, 0.48, -0.08, "#ffd54f", { emissive: "#ffb300", emissiveIntensity: 0.9 });
      for (const [x, c] of [[-0.3, "#ff4081"], [0.0, "#40c4ff"], [0.3, "#ffea00"]])
        bx(0.08, 0.08, 0.02, x, 0.3, 0.23, c, { emissive: c, emissiveIntensity: 1.0 });
      const dome = m(new THREE.SphereGeometry(0.14, 16, 10, 0, Math.PI * 2, 0, Math.PI / 2), "#ffd54f", { metalness: 0.7, roughness: 0.25 });
      dome.position.set(-0.14, 1.01, -0.14);
      cy(0.12, 0.12, 0.02, 0.28, 0.06, 0.32, "#40c4ff", 16, { emissive: "#0288d1", emissiveIntensity: 0.4 });
    } else if (ch === ">") {                        // airport: drawn at its full 3x3 size
      bx(2.9, 0.03, 2.9, 0, 0, 0, "#9e9e9e");
      bx(2.8, 0.035, 0.5, 0, 0.005, 0.9, "#3a3d42");
      for (let k = -5; k <= 5; k++) bx(0.14, 0.04, 0.03, k * 0.24, 0.006, 0.9, "#f5f5f5");
      bx(1.2, 0.36, 0.55, -0.6, 0.03, -0.75, "#cfd8dc");
      bx(1.22, 0.05, 0.57, -0.6, 0.39, -0.75, "#90a4ae");
      bx(1.2, 0.18, 0.02, -0.6, 0.12, -0.47, "#81d4fa", { roughness: 0.2, metalness: 0.4 });
      cy(0.08, 0.1, 0.9, 0.8, 0.03, -0.8, "#eceff1", 12);
      cy(0.2, 0.16, 0.16, 0.8, 0.93, -0.8, "#4fc3f7", 12, { roughness: 0.2, metalness: 0.5 });
      const plane = new THREE.Group(); g.add(plane);
      const fus = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.8, 10), this._mat("#fafafa")); fus.rotation.z = Math.PI / 2; plane.add(fus);
      const wing = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.02, 0.8), this._mat("#e0e0e0")); plane.add(wing);
      const tail = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.2, 0.02), this._mat("#1976d2")); tail.position.set(-0.36, 0.1, 0); plane.add(tail);
      plane.position.set(-0.85, 0.12, -0.2); plane.rotation.y = Math.PI / 2;   // parked at a stand; life.js flies the others
    } else if (ch === "H") {                        // highway entrance sign
      cy(0.02, 0.02, 0.8, 0.3, 0, -0.42, "#777", 6); cy(0.02, 0.02, 0.8, -0.3, 0, -0.42, "#777", 6);
      bx(0.7, 0.22, 0.03, 0, 0.62, -0.42, "#1b7a3e");
    } else if (ch === "u") {                        // nuclear plant (2x2): two cooling towers, reactor dome, turbine hall
      bx(0.96, 0.03, 0.96, 0, 0, 0, "#a3a8a0");
      const tower = [[0.2, 0], [0.16, 0.27], [0.13, 0.53], [0.15, 0.75]].map(([r, y]) => new THREE.Vector2(r, y));
      const st = [];
      for (const [x, z] of [[-0.24, -0.2], [0.18, -0.26]]) {
        const t = m(new THREE.LatheGeometry(tower, 24), "#d9d6cf", { side: THREE.DoubleSide }); t.position.set(x, 0.03, z);
        const rim = m(new THREE.TorusGeometry(0.15, 0.01, 6, 24), "#8f8b84"); rim.rotation.x = Math.PI / 2; rim.position.set(x, 0.78, z);
        st.push({ x, y: 0.82, z, rate: 1.2, dark: 0.95 });
      }
      bx(0.44, 0.22, 0.24, -0.18, 0.03, 0.26, "#c9ced6");                        // turbine hall
      bx(0.46, 0.03, 0.26, -0.18, 0.25, 0.26, "#6f7a86");
      cy(0.14, 0.14, 0.22, 0.24, 0.03, 0.22, "#e6e3dc", 20);                     // containment
      const dome = m(new THREE.SphereGeometry(0.14, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2), "#eceae4"); dome.position.set(0.24, 0.25, 0.22);
      bx(0.1, 0.12, 0.1, 0.4, 0.03, -0.02, "#f2c230");                             // switchyard
      for (const x of [0.08, 0.42]) cy(0.01, 0.01, 0.28, x, 0.03, 0.4, "#8a8f96", 6);
      g.userData.stacks = st;
    } else if (ch === "z") {                        // crematorium: small chapel with a pitched roof and one chimney
      bx(0.92, 0.03, 0.92, 0, 0, 0, "#8fae78");
      bx(0.5, 0.3, 0.4, -0.06, 0.03, 0.02, "#d8d3c8");
      const roof = m(new THREE.CylinderGeometry(0.5, 0.5, 0.56, 3), "#4e4a52"); roof.rotation.z = Math.PI / 2; roof.rotation.y = Math.PI / 2;
      roof.scale.set(0.42, 1, 0.3); roof.position.set(-0.06, 0.41, 0.02);
      for (let k = -1; k <= 1; k++) bx(0.05, 0.14, 0.01, -0.06 + k * 0.14, 0.1, 0.225, "#6d7f99");
      bx(0.03, 0.14, 0.03, -0.06, 0.53, 0.02, "#e0b43a"); bx(0.1, 0.03, 0.03, -0.06, 0.61, 0.02, "#e0b43a");
      bx(0.1, 0.68, 0.1, 0.28, 0.03, -0.24, "#7a6f66"); bx(0.12, 0.04, 0.12, 0.28, 0.71, -0.24, "#4a4440");
      for (const [x, z] of [[-0.36, 0.36], [0.34, 0.34]]) { const c = m(new THREE.ConeGeometry(0.08, 0.3, 7), "#2f5f3a"); c.position.set(x, 0.2, z); }
      g.userData.stacks = [{ x: 0.28, y: 0.78, z: -0.24, rate: 0.35, dark: 0.7 }];
    }
    this._tierExtras(g, ch, lv, bx);
    return g;
  }

  // higher tiers (frame level = tier): an annex block with an accent roof, a second one at the top tier (recipes.js)
  _tierExtras(g, ch, lv, bx) {
    for (const [w, h, d, x, y, z, c] of tierAnnexes(ch, lv)) bx(w, h, d, x, y, z, c);
  }

  // -------------------------------------------------------- bus lines
  _busLines(frame) {
    const key = JSON.stringify([frame.bus_lines || [], frame.metro_lines || [], frame.tiles.length]) + (frame.road_class || "").length;
    if (key === this._busKey) return;
    this._busKey = key;
    if (this.busGroup) this.root.remove(this.busGroup);
    this.busGroup = new THREE.Group(); this.root.add(this.busGroup);
    this.metroGroup = new THREE.Group(); this.busGroup.add(this.metroGroup);
    this.routeGroup = new THREE.Group(); this.busGroup.add(this.routeGroup);
    (frame.metro_lines || []).forEach((line) => {    // metro tunnels: only drawn on the transit layer
      const pts = line.stations.map(([x, y]) => new THREE.Vector3(this.wx(x), 0.05, this.wz(y)));
      if (pts.length < 2) return;
      const path = new THREE.CurvePath();
      for (let k = 0; k + 1 < pts.length; k++) path.add(new THREE.LineCurve3(pts[k], pts[k + 1]));
      this.metroGroup.add(new THREE.Mesh(new THREE.TubeGeometry(path, pts.length * 20, 0.06, 6, false),
        new THREE.MeshBasicMaterial({ color: "#d32f2f", transparent: true, opacity: 0.75, depthTest: false })));
    });
    (frame.bus_lines || []).forEach((line, k) => {
      const c = BUS_COLORS[k % BUS_COLORS.length];
      // the route follows the roads between stops: a thin coloured band on the road (visible on the transit layer)
      const route = this.life.routePath(line.stops);
      if (route.length >= 2) {
        const path = new THREE.CurvePath();
        const pt = ([x, y]) => new THREE.Vector3(this.wx(x) + 0.12 * (k % 3 - 1), 0.06, this.wz(y) + 0.12 * (k % 3 - 1));
        for (let n = 0; n + 1 < route.length; n++) path.add(new THREE.LineCurve3(pt(route[n]), pt(route[n + 1])));
        this.routeGroup.add(new THREE.Mesh(new THREE.TubeGeometry(path, route.length * 2, 0.11, 4, false),
          new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.85 })));
      }
      const pts = line.stops.map(([x, y]) => new THREE.Vector3(this.wx(x), 0, this.wz(y)));
      for (const p of pts) {
        const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.55, 6), this._mat("#8a8f96"));
        pole.position.set(p.x + 0.42, 0.14, p.z + 0.42); pole.scale.set(1, 0.5, 1);
        const sign = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.08, 0.02), this._mat(c, { emissive: c, emissiveIntensity: 0.5 }));
        sign.position.set(p.x + 0.42, 0.3, p.z + 0.42);
        const roof = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.015, 0.1), this._mat("#90a4ae"));
        roof.position.set(p.x + 0.3, 0.16, p.z + 0.44);
        this.busGroup.add(pole, sign, roof);
      }
    });
  }

  // ------------------------------------------------------------ smoke
  _setupSmoke() {
    const cap = this.smokeGeo.attributes.position.count, per = 16;
    const list = this.emitters.slice(0, Math.floor(cap / per));
    this.smokeParts = [];
    for (let e = 0; e < list.length; e++) for (let k = 0; k < per; k++)
      this.smokeParts.push({ e: list[e], phase: (k / per + hash(e * 31 + k, 7) * 0.1) % 1 });
    this.smokeGeo.setDrawRange(0, this.smokeParts.length);
  }
  _tickSmoke(t) {
    if (!this.smokeParts) return;
    const pos = this.smokeGeo.attributes.position.array, colA = this.smokeGeo.attributes.color.array;
    for (let n = 0; n < this.smokeParts.length; n++) {
      const { e, phase } = this.smokeParts[n], life = 4.5 / (0.6 + e.rate * 0.4);
      const k = ((t / life) + phase) % 1;
      const [wx, wz] = this.windDir || [1, 0];
      pos[n * 3] = e.x + k * 1.1 * wx + Math.sin(k * 6 + n) * 0.08;
      pos[n * 3 + 1] = e.y + k * (1.2 + e.rate * 0.5);
      pos[n * 3 + 2] = e.z + k * 1.1 * wz;
      if (e.fire) {                     // flames: yellow -> orange -> dark smoke
        colA[n * 4] = 1.0 - k * 0.55; colA[n * 4 + 1] = Math.max(0.15, 0.75 - k * 0.9); colA[n * 4 + 2] = 0.1 + k * 0.2;
      } else {
        const c = e.dark + k * 0.35 + this.nightK * -0.2;
        colA[n * 4] = c; colA[n * 4 + 1] = c; colA[n * 4 + 2] = c + 0.02;
      }
      colA[n * 4 + 3] = (1 - k) * Math.min(1, k * 5) * 0.75;
    }
    this.smokeGeo.attributes.position.needsUpdate = true;
    this.smokeGeo.attributes.color.needsUpdate = true;
  }

  // ---------------------------------------------------------- overlays
  setOverlay(mode) { this.overlay = mode; this._applyOverlay(); }
  _overlayColor(i) {
    const f = this.frame, ch = f.tiles[i], zone = ZONES.includes(ch), o = OVERLAY_BY_ID[this.overlay];
    if (o && o.districts) { const d = f.districts ? +f.districts[i] : 0; return d ? col(DISTRICT_COLORS[d]) : ("#H".includes(ch) ? P.asphalt : col("#9aa19a")); }
    if (o && o.cat) { const r = this.meta.resources ? +this.meta.resources[i] : 0; return r ? col(o.cat[r]) : ("#H".includes(ch) ? P.asphalt : col("#9aa19a")); }
    const str = o && layerStr(f, o);
    if (!str) return "#H".includes(ch) ? P.asphalt : col("#9aa19a");
    if (o.water && !"~=Y".includes(ch)) return "#H".includes(ch) ? P.asphalt : col("#9aa19a");
    if (o.roads && !"#=H".includes(ch)) return col("#9aa19a");
    if (o.mask && !o.mask.includes(ch)) return "#H=".includes(ch) ? P.asphalt : col("#9aa19a");
    if (o.ramp) return ramp(RAMPS[o.ramp], +str[i] / 9);
    if (!zone) return "#H".includes(ch) ? P.asphalt : col("#9aa19a");
    return str[i] === "1" ? col("#34c759") : col("#ff3b30");
  }
  _applyOverlay() {
    if (!this.frame || !this.parts) return;
    const on = this.overlay !== "none";
    const f = this.frame;
    for (const [k, part] of Object.entries(this.parts)) {
      const o = OVERLAY_BY_ID[this.overlay];
      if (k === "water" && o && o.water) { part.recolor(it => it.tile < 0 ? it.color : this._overlayColor(it.tile)); continue; }
      if (["water", "bed", "bank", "dash", "trunk", "crown", "pine", "oak", "bush", "flower", "rock", "pole", "lamp", "barn", "barnRoof",
        "pillar", "rail", "deck"].includes(k)) continue;
      part.recolor(it => {
        if (!on || it.tile < 0) return it.color;
        const oc = this._overlayColor(it.tile);
        return k === "land" ? oc : lerpC(oc, col("#ffffff"), 0.18);
      });
    }
    for (const g of this.specials.values()) g.traverse(o => {
      if (!o.isMesh) return;
      o.material.transparent = on; o.material.opacity = on ? 0.55 : 1;
    });
    this.smoke.visible = !on || this.overlay === "pollution";
    if (this.life) this.life.visible = !on;
    if (this.metroGroup) this.metroGroup.visible = this.overlay === "transit";
    if (this.routeGroup) this.routeGroup.visible = this.overlay === "transit";
  }

  // ----------------------------------------------------- highlights
  _fitBox(obj, x1, y1, x2, y2, h) {
    const xa = Math.min(x1, x2), xb = Math.max(x1, x2), ya = Math.min(y1, y2), yb = Math.max(y1, y2);
    obj.position.set(this.wx((xa + xb) / 2), 0, this.wz((ya + yb) / 2));
    obj.scale.set(xb - xa + 1.04, h, yb - ya + 1.04);
    obj.visible = true;
  }
  tileHeight(i) {
    let h = 0.1;
    for (const part of Object.values(this.parts || {})) for (let k = 0; k < part.items.length; k++) {
      const it = part.items[k]; if (it.tile === i) h = Math.max(h, it.y + it.sy);
    }
    const g = [...this.specials.values()].find(s => s.userData.tile === i);
    if (g) h = Math.max(h, new THREE.Box3().setFromObject(g).max.y);
    return h + 0.05;
  }
  setHover(tile) {
    if (tile == null || tile < 0) { this.hoverBox.visible = false; return; }
    const x = tile % this.W, y = Math.floor(tile / this.W);
    this._fitBox(this.hoverBox, x, y, x, y, this.tileHeight(tile));
  }
  setSelected(tile) {
    if (tile == null || tile < 0) { this.selBox.visible = false; return; }
    const x = tile % this.W, y = Math.floor(tile / this.W);
    this._fitBox(this.selBox, x, y, x, y, this.tileHeight(tile));
  }
  highlightRect(r) {
    if (!r) { this.hlMesh.visible = this.hlEdges.visible = false; return; }
    this._fitBox(this.hlMesh, r.x1, r.y1, r.x2, r.y2, 0.6);
    this._fitBox(this.hlEdges, r.x1, r.y1, r.x2, r.y2, 0.6);
    this.hlT0 = performance.now();
  }

  // ---------------------------------------------------------- camera
  _animateCam(target, pos, ms = 800) {
    this.camAnim = { t0: performance.now(), ms, ft: this.controls.target.clone(), fp: this.camera.position.clone(), tt: target, tp: pos };
  }
  resetView(anim = true) {
    const s = Math.max(this.W || 48, this.H || 48);
    const t = new THREE.Vector3(0, 0, 2), p = new THREE.Vector3(s * 0.55, s * 0.62, s * 0.78);
    if (anim) this._animateCam(t, p); else { this.controls.target.copy(t); this.camera.position.copy(p); }
  }
  topView() {
    const s = Math.max(this.W, this.H);
    this._animateCam(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0.01, s * 1.25, 0.6));
  }
  flyTo(x, y, dist = 14) {
    const t = new THREE.Vector3(this.wx(x), 0, this.wz(y));
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    if (dir.y < 0.35) { dir.y = 0.6; dir.normalize(); }
    this._animateCam(t, t.clone().add(dir.multiplyScalar(dist)));
  }
  panBy(dx, dz) {
    const fwd = new THREE.Vector3(); this.camera.getWorldDirection(fwd); fwd.y = 0; fwd.normalize();
    const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 1, 0));
    const d = right.multiplyScalar(dx).add(fwd.multiplyScalar(dz)).multiplyScalar(this.camera.position.distanceTo(this.controls.target) * 0.04);
    this.controls.target.add(d); this.camera.position.add(d);
  }
  rotateBy(a) {
    const off = this.camera.position.clone().sub(this.controls.target);
    off.applyAxisAngle(new THREE.Vector3(0, 1, 0), a);
    this.camera.position.copy(this.controls.target).add(off);
  }
  zoomBy(f) {
    const off = this.camera.position.clone().sub(this.controls.target).multiplyScalar(f);
    if (off.length() > 3 && off.length() < 160) this.camera.position.copy(this.controls.target).add(off);
  }
  setAutoRotate(on) { this.controls.autoRotate = on; this.controls.autoRotateSpeed = 0.6; }
  setActive(on) {                 // hidden scenes stop rendering (app.js switches between this and the pixel view)
    this.renderer.domElement.style.display = on ? "" : "none";
    this.renderer.setAnimationLoop(on ? () => this._loop() : null);
    if (on) this.resize();
  }
  setNight(on) { this.night = on; }
  setQuality(q) {
    this.quality = q;
    this.renderer.shadowMap.enabled = q !== "low";
    this.renderer.setPixelRatio(q === "low" ? 1 : Math.min(2, window.devicePixelRatio || 1));
    this.scene.traverse(o => { if (o.material) o.material.needsUpdate = true; });
    this.resize();
  }
  tileToScreen(x, y, h = 0.2) {   // a tile's centre -> CSS pixels in the view (app.js beams layer); null behind the camera
    const v = new THREE.Vector3(this.wx(x), h, this.wz(y)).project(this.camera);
    if (v.z > 1) return null;
    const r = this.renderer.domElement.getBoundingClientRect();
    return [(v.x + 1) / 2 * r.width, (1 - v.y) / 2 * r.height];
  }
  cameraFootprint() {         // for the minimap: target tile + view direction
    const t = this.controls.target, d = new THREE.Vector3(); this.camera.getWorldDirection(d);
    return { x: t.x + this.W / 2 - 0.5, y: t.z + this.H / 2 - 0.5, angle: Math.atan2(d.x, d.z),
      dist: this.camera.position.distanceTo(t) };
  }
  screenshot() { this.renderer.render(this.scene, this.camera); return this.renderer.domElement.toDataURL("image/png"); }

  // ---------------------------------------------------------- picking
  _bind() {
    const el = this.renderer.domElement, ray = new THREE.Raycaster(), v = new THREE.Vector2();
    let down = null, pending = null;
    const pick = (ev) => {
      if (!this.parts) return -1;
      const r = el.getBoundingClientRect();
      v.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(v, this.camera);
      const objs = [...this.pickables, ...this.specials.values()];
      const hits = ray.intersectObjects(objs, true);
      for (const h of hits) {
        let tile = -1;
        if (h.object.isInstancedMesh && h.instanceId != null) {
          const part = Object.values(this.parts).find(p => p.mesh === h.object);
          tile = part ? part.tileOf[h.instanceId] : -1;
        } else if (h.object.userData.tile != null) tile = h.object.userData.tile;
        if (tile != null && tile >= 0) return tile;
      }
      return -1;
    };
    el.addEventListener("pointermove", (ev) => { pending = ev; });
    this._pollHover = () => {
      if (!pending) return;
      const ev = pending; pending = null;
      const t = pick(ev); this.setHover(t);
      this.onHover && this.onHover(t, ev);
    };
    el.addEventListener("pointerleave", () => { this.setHover(-1); this.onHover && this.onHover(-1); });
    el.addEventListener("pointerdown", (ev) => { down = [ev.clientX, ev.clientY]; this.camAnim = null; });
    el.addEventListener("pointerup", (ev) => {
      if (!down || Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) > 5 || ev.button !== 0) return;
      const t = pick(ev); this.onClick && this.onClick(t, ev);
    });
  }

  resize() {
    const w = this.container.clientWidth || 1, h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%"; this.renderer.domElement.style.height = "100%";
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  // ------------------------------------------------------------ loop
  _loop() {
    this.clock.update(); const dt = Math.min(0.05, this.clock.getDelta()), t = this.clock.getElapsed(), now = performance.now();
    this.time.value = t;
    if (this.camAnim) {
      const a = this.camAnim, k = Math.min(1, (now - a.t0) / a.ms), e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
      this.controls.target.lerpVectors(a.ft, a.tt, e); this.camera.position.lerpVectors(a.fp, a.tp, e);
      if (k >= 1) this.camAnim = null;
    }
    this.controls.update();
    if (this.parts) for (const p of Object.values(this.parts)) p.tick(now);
    for (const g of this.specials.values()) if (g.userData.t0) {
      const k = Math.min(1, (now - g.userData.t0) / 600), f = Math.max(0.01, 1 - Math.pow(1 - k, 3)), b = g.userData.base || [1, 1, 1];
      g.scale.set(b[0] * f, b[1] * f, b[2] * f);
      if (k >= 1) g.userData.t0 = 0;
    }
    for (const r of this.turbines) r.rotation.z -= dt * 2.4;
    if (this.life) this.life.tick(dt, t, this.nightK);
    this._tickSmoke(t);
    if (this.hlMesh && this.hlMesh.visible) this.hlMesh.material.opacity = 0.16 + 0.12 * Math.sin((now - this.hlT0) / 180);

    // day / night blend
    const goal = this.night ? 1 : 0;
    if (Math.abs(this.nightK - goal) > 1e-3) {
      this.nightK += (goal - this.nightK) * Math.min(1, dt * 3);
      const k = this.nightK;
      this.scene.background.copy(this.dayBg).lerp(this.nightBg, k);
      this.scene.fog.color.copy(this.scene.background);
      this.sun.intensity = 2.3 * (1 - k) + 0.25 * k;
      this.sun.color.set(k > 0.5 ? 0x9fb4ff : 0xfff2dd);
      this.hemi.intensity = 0.8 * (1 - k) + 0.3 * k;
      this.scene.environmentIntensity = 0.55 * (1 - k) + 0.08 * k;
      for (const m of this.windowMats) m.emissiveIntensity = 1.3 * k;
      this.M.dash.emissiveIntensity = 0.8 * k;
      this.M.lamp.emissiveIntensity = 2.2 * k;
      this.M.sign.emissiveIntensity = 0.15 + 1.2 * k;
      this.renderer.toneMappingExposure = 1.05 - 0.15 * k;
    }
    this._pollHover && this._pollHover();
    this.renderer.render(this.scene, this.camera);
  }
}
