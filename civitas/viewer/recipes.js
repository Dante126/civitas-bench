// What the pixel view draws: recipes that fill a sprites.js Builder for every building (tile character, level,
// density, variant), special structure, tree, ground tile and vehicle. Coordinates are tile units centred on the
// tile (or footprint); z is in height units. Buildings are written facing +y and pixel.js turns them to their road.
// Adding a building: give its tile character a branch in special() (and a MINI / NAMES entry in app.js).
import { Builder, hash, mix, rgb } from "./sprites.js";

const TALL = 0.7;                                  // towers are squatter than in 3D, as in OpenTTD
const pick = (arr, v, k) => arr[Math.floor(hash(v * 7919 + k, 11) * arr.length) % arr.length];
const lighten = (c, t) => mix(c, "#ffffff", t), darken = (c, t) => mix(c, "#000000", t);
const hx = (c) => { c = rgb(c); return "#" + c.map(v => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, "0")).join(""); };

export const PAL = {
  wall: ["#e8d8b0", "#d8b890", "#f0e8d8", "#c8d0d8", "#e0c0b0", "#d0c0a0"],
  brick: ["#b0583c", "#a04c34", "#c07048"],
  roof: ["#a83c2c", "#7c4430", "#5c6068", "#8c3428", "#b85c34", "#4c5c70"],
  apt: ["#d8c0a0", "#c89c7c", "#e0d4c0", "#b8b0a8", "#d4b49c", "#a8886c"],
  glass: ["#6c8cc0", "#5c7cb4", "#7c9ccc", "#4c6ca4", "#8cacd0"],
  shop: ["#f0ece0", "#e4dcd0", "#dce4e8", "#f0d8c0"],
  awning: ["#d83c34", "#2c7cc4", "#3ca454", "#e0a424", "#8c54b4"],
  sign: ["#e53935", "#fdd835", "#1e88e5", "#43a047", "#8e24aa", "#fb8c00"],
  ind: ["#a8a498", "#9c988c", "#b4ac9c", "#8c9094"],
  office: ["#8c93c7", "#7d86bd", "#9aa1cf", "#6f8fb0", "#a7b6c8"],
  flowers: ["#f06292", "#fff176", "#ba68c8", "#ffffff", "#ff8a65", "#64b5f6"],
};
const WIN = {
  house: (seed) => ({ c: "#3c4c60", fl: 0.2, sp: 0.16, w: 2, h: 3, z0: 0.04, lit: 0.5, seed }),
  flat: (seed) => ({ c: "#34465a", fl: 0.18, sp: 0.12, w: 2, h: 3, z0: 0.05, lit: 0.55, seed }),
  tower: (seed) => ({ c: "#2e4058", fl: 0.15, sp: 0.1, w: 2, h: 3, z0: 0.06, lit: 0.5, seed }),
  glass: (seed) => ({ c: "#c4dcf4", fl: 0.12, sp: 0.08, w: 2, h: 2, z0: 0.05, lit: 0.45, seed }),
  shop: (seed) => ({ c: "#a4cce8", fl: 0.22, sp: 0.18, w: 4, h: 5, z0: 0.03, lit: 0.9, glow: "#fff0b0", seed }),
  factory: (seed) => ({ c: "#4a5460", fl: 0.25, sp: 0.2, w: 3, h: 2, z0: 0.12, lit: 0.3, seed }),
};

// ------------------------------------------------------------------ trees
const LEAVES = { spring: ["#3c8c2c", "#4c9c34", "#2c7c2c", "#5cac3c"], summer: ["#3c7c2c", "#4c8430", "#2c6c2c", "#5c9434"],
  autumn: ["#d8822b", "#c0392b", "#e0a526", "#a86c2c"] };
export function tree(b, x, y, s, kind, season, seed = 0) {
  const leaves = LEAVES[season] || LEAVES.summer, lc = leaves[seed & 3];
  const bare = season === "winter";
  if (kind === 0) {                                 // pine: stays green, catches snow
    const pc = bare ? mix("#2c5c34", "#ffffff", 0.3) : "#2c6434";
    b.box(x, y, 0.04 * s, 0.04 * s, 0, 0.1 * s, "#5c3c24");
    b.cone(x, y, 0.15 * s, 0.07 * s, 0.26 * s, pc, { seg: 7 });
    b.cone(x, y, 0.11 * s, 0.2 * s, 0.24 * s, lighten(pc, 0.06), { seg: 7 });
    return;
  }
  const [th, tw, cz, cr, tc] = kind === 1 ? [0.22, 0.03, 0.26, 0.1, "#e8e4dc"] : kind === 2 ? [0.14, 0.05, 0.2, 0.16, "#5c3c24"] : [0.12, 0.04, 0.18, 0.13, "#6b4a2f"];
  b.box(x, y, tw * s, tw * s, 0, th * s, tc);
  const col = bare ? "#8a7c6c" : kind === 1 ? lighten(lc, 0.12) : kind === 2 ? darken(lc, 0.08) : lc;
  b.blob(x, y, cz * s, cr * s, col, bare ? { seed, lumpy: 0.3, speck: 0.3, speckC: "#ffffff", nosnow: true } : { seed });
}
const bush = (b, x, y, r, c, season, seed) => b.blob(x, y, r * 0.8, r, season === "winter" ? "#c8ccd0" : season === "autumn" ? LEAVES.autumn[seed & 3] : c, { seed, lumpy: 0.2 });

// ------------------------------------------------------------------ zone buildings
// c: { ch, lv, vr (variant: wealth / industry type), dn (density), v (look variant), season, hist }
export function building(b, c) {
  const { ch, lv, vr = 0, dn = 1, v, season } = c;
  const hh = hash(v, 2), h3 = hash(v, 3), P = (arr, k) => pick(arr, v, k);
  if (ch === "I" && vr >= 1 && vr <= 5) industry(b, vr, lv, v, hh, season);
  else if (ch === "R" && dn === 0 && lv === 2) {    // large detached house, a villa if wealthy
    const rich = vr === 3, hgt = 0.42 + h3 * 0.1, wall = rich ? "#f7f5ef" : P(PAL.wall, 4);
    b.box(-0.06, -0.05, 0.56, 0.48, 0, hgt, wall, { win: WIN.house(v) });
    b.hip(-0.06, -0.05, 0.64, 0.56, hgt, 0.22, rich ? "#3c3f45" : P(PAL.roof, 5));
    if (rich) b.flat(0.28, 0.3, 0.3, 0.22, 0.01, "#4fc3f7", { flat: true, nosnow: true });
    else { b.box(0.3, 0.3, 0.2, 0.2, 0, 0.16, P(PAL.wall, 14), { top: "#6b6f75" }); }
  } else if (ch === "R" && dn === 1 && lv === 1) {  // row houses under one roof
    for (let k = -1; k <= 1; k++) b.box(k * 0.27, 0, 0.27, 0.5, 0, 0.36, P(PAL.wall, 20 + k), { win: WIN.house(v + k) });
    b.gable(0, 0, 0.86, 0.56, 0.36, 0.18, P(PAL.roof, 24), { wall: P(PAL.wall, 20) });
    for (const k of [-0.27, 0.27]) b.box(k, -0.12, 0.05, 0.05, 0.44, 0.16, "#8d6e63");
  } else if (ch === "R" && lv === 1) houses(b, v, hh, h3, season);
  else if (ch === "R" && lv === 2) {                // apartments with balconies
    const hgt = 0.62 + h3 * 0.35, col = P(PAL.apt, 6);
    if (hh < 0.35) {                                // L-shaped block around a courtyard tree
      b.box(-0.12, 0, 0.5, 0.74, 0, hgt, col, { win: WIN.flat(v), top: "#8c8c8c" });
      b.box(0.2, -0.2, 0.36, 0.34, 0, hgt * 0.85, hx(darken(col, 0.05)), { win: WIN.flat(v + 1), top: "#8c8c8c" });
      tree(b, 0.28, 0.26, 0.6, 3, season, v);
    } else b.box(0, 0, 0.74, 0.7, 0, hgt, col, { win: WIN.flat(v), top: "#8c8c8c" });
    const floors = Math.max(1, Math.floor(hgt / 0.2));
    for (let f = 1; f < floors; f++) for (const s of [-0.2, 0.2]) b.box(s, 0.37, 0.2, 0.05, f * 0.2, 0.03, "#e9e6df");
    b.box(0.12, -0.1, 0.2, 0.16, hgt, 0.08, "#9aa0a6");
  } else if (ch === "R") {                          // high-rise towers
    const hgt = ([0, 0, 0, 1.5, 2.3, 3.2][lv] + h3 * 1.2) * TALL, w = lv >= 5 ? 0.66 : 0.6;
    const col = vr === 3 ? "#e8eef2" : vr === 1 ? P(["#b98a6e", "#c79b7c", "#a8876b"], 7) : P(PAL.apt, 7);
    const style = Math.floor(hh * 4), win = WIN.tower(v), top = "#8c9196";
    if (style === 0) {
      b.box(0, 0, w + 0.12, w + 0.12, 0, hgt * 0.55, col, { win, top });
      b.box(0, 0, w - 0.08, w - 0.08, hgt * 0.55, hgt * 0.45, hx(lighten(col, 0.06)), { win: WIN.tower(v + 1), top });
    } else if (style === 1) b.cyl(0, 0, (w + 0.08) / 2, 0, hgt, col, { seg: 12, win, top });
    else if (style === 2 && lv >= 4) {
      b.box(0, 0, 0.86, 0.86, 0, 0.2, "#cfd2d6", { win: WIN.shop(v) });
      for (const s of [-0.2, 0.2]) b.box(s, 0, 0.34, 0.5, 0.2, hgt * (s < 0 ? 1 : 0.8), col, { win: WIN.tower(v + s * 10), top });
    } else {
      b.box(0, 0, w, w, 0, hgt, col, { win, top });
      for (let f = 1; f < Math.floor(hgt / 0.3); f++) b.box(0, w / 2 + 0.02, w * 0.8, 0.04, f * 0.3, 0.025, "#dfe3e6");
    }
    if (style !== 2 || lv < 4) b.box(0, 0, 0.26, 0.26, hgt, 0.1, "#8c9196");
    if (lv >= 5) b.cyl(0, 0, 0.025, hgt + 0.1, 0.4, "#d9dde2", { seg: 6 });
  } else if (ch === "C") commercial(b, lv, v, hh, h3);
  else if (ch === "M") {                            // shops at street level, flats above
    const hgt = ([0, 0.45, 0.8, 1.4, 2.1, 2.9][lv] + h3 * 0.5) * (lv >= 3 ? TALL : 1);
    b.box(0, 0, 0.84, 0.78, 0, 0.28, P(PAL.shop, 15), { win: WIN.shop(v) });
    b.box(0, 0.42, 0.8, 0.12, 0.2, 0.04, P(PAL.awning, 16));
    b.box(0, -0.02, 0.72, 0.66, 0.28, hgt, P(PAL.apt, 17), { win: WIN.flat(v), top: "#8c8c8c" });
    for (let f = 0; f < Math.floor(hgt / 0.24); f++) b.box(0, 0.33, 0.5, 0.04, 0.4 + f * 0.24, 0.025, "#6d4c41");
    b.box(0.1, -0.02, 0.2, 0.18, 0.28 + hgt, 0.08, "#9aa0a6");
  } else if (ch === "O") {                          // offices: glass slabs, towers and campuses
    const w = [0, 0.6, 0.7, 0.7, 0.74, 0.78][lv], hgt = ([0, 0.5, 1.0, 1.9, 2.8, 3.8][lv] + h3 * 0.8) * (lv >= 3 ? TALL : 1);
    const col = P(PAL.office, 13), win = WIN.glass(v), top = "#7c8090";
    b.box(0, 0, 0.84, 0.84, 0, 0.12, "#c3c6d6");
    if (hh < 0.3 && lv >= 3) b.cyl(0, 0, w / 2, 0.12, hgt, col, { seg: 12, win, top });
    else if (hh < 0.55 && lv >= 2) {
      b.box(-0.12, -0.1, w * 0.6, w * 0.6, 0.12, hgt, col, { win, top });
      b.box(0.16, 0.14, w * 0.5, w * 0.5, 0.12, hgt * 0.65, hx(lighten(col, 0.08)), { win: WIN.glass(v + 1), top });
    } else b.box(0, 0, w, w * 0.7, 0.12, hgt, col, { win, top });
    b.box(0, 0, 0.22, 0.18, hgt + 0.12, 0.08, "#9aa0a6");
    if (lv >= 4) b.cyl(0.08, 0, 0.012, hgt + 0.2, 0.5, "#d0d0d0", { seg: 4 });
  } else if (ch === "I" && hh < 0.35) {             // warehouse with loading docks and trailers
    const w = [0, 0.8, 0.86, 0.9, 0.9, 0.92][lv], hgt = 0.26 + lv * 0.06, col = P(PAL.ind, 12);
    b.box(0, -0.08, w, w * 0.62, 0, hgt, col, { win: WIN.factory(v) });
    b.gable(0, -0.08, w, w * 0.64, hgt, 0.14, "#6d7278", { wall: col });
    for (let k = 0; k < Math.min(3, lv + 1); k++) b.box(-0.28 + k * 0.28, 0.3, 0.12, 0.2, 0, 0.08, k % 2 ? "#eceff1" : "#ef6c00");
  } else if (ch === "I") {                          // factory with sawtooth roofs, chimneys, a tank
    const col = P(PAL.ind, 12), w = [0, 0.8, 0.86, 0.9, 0.9, 0.92][lv], hgt = [0, 0.32, 0.48, 0.62, 0.72, 0.84][lv];
    b.box(0, 0, w, w * 0.85, 0, hgt, col, { win: WIN.factory(v) });
    const n = Math.min(3, lv + 1);
    for (let k = 0; k < n; k++) b.gable((k - (n - 1) / 2) * 0.26, 0, 0.24, w * 0.8, hgt, 0.16, "#6d7278", { axis: "y", wall: "#9cb4c8" });
    if (lv >= 2) for (let k = 0; k < Math.min(2, lv - 1); k++) {
      const cx = 0.3 - k * 0.18, cz = -0.28, h = lv >= 3 ? 1.3 : 0.9;
      b.cyl(cx, cz, 0.06, 0, h, "#8a6f63", { seg: 8, rt: 0.05 });
      b.cyl(cx, cz, 0.052, h - 0.12, 0.06, "#c9453b", { seg: 8 });
      b.smoke(cx, cz, h + 0.05, 0.35 * lv, 0.55);
    }
    if (lv >= 3) b.cyl(-0.28, 0.28, 0.13, 0, 0.35, "#c8c4b8", { seg: 10 });
  }
  if (c.hist) b.box(0.38, 0.4, 0.16, 0.03, 0, 0.22, "#e0b43a");
}

function houses(b, v, hh, h3, season) {           // detached houses: gable / hip / flat roofs, chimneys, garages, gardens
  const two = hh > 0.55, P = (arr, k) => pick(arr, v, k);
  const spots = two ? [[-0.2, -0.18], [0.22, 0.2]] : [[-0.05, -0.05]];
  spots.forEach(([dx, dy], k) => {
    const w = two ? 0.36 : 0.46, hgt = 0.24 + hash(v, 40 + k) * 0.12, style = Math.floor(hash(v, 44 + k) * 3);
    const wall = hash(v, 49 + k) < 0.25 ? P(PAL.brick, 4 + k) : P(PAL.wall, 4 + k * 3), roof = P(PAL.roof, 5 + k * 3);
    b.box(dx, dy, w, w * 0.85, 0, hgt, wall, { win: WIN.house(v * 3 + k) });
    if (style === 0) b.gable(dx, dy, w * 1.08, w * 0.95, hgt, 0.2, roof, { wall });
    else if (style === 1) b.hip(dx, dy, w * 1.06, w * 0.92, hgt, 0.18, roof);
    else {                                          // modern flat roof with a parapet
      b.box(dx, dy, w * 1.02, w * 0.87, hgt, 0.03, "#6b6f75");
      b.box(dx + 0.05, dy + 0.05, w * 0.5, w * 0.5, hgt + 0.03, 0.12, hx(lighten(wall, 0.06)), { win: WIN.house(v + 9) });
    }
    if (style !== 2 && hash(v, 48 + k) < 0.6) b.box(dx + w * 0.25, dy - w * 0.2, 0.05, 0.05, hgt + 0.06, 0.18, "#8d6e63");
  });
  if (!two) {
    if (h3 < 0.5) b.box(0.3, 0.3, 0.2, 0.24, 0, 0.16, "#d7d2c8", { top: "#8c8c8c" });   // garage
    else tree(b, 0.3, 0.3, 0.6 + h3 * 0.3, 3, season, v);
    if (hh < 0.3) for (const s of [-1, 1]) bush(b, s * 0.4, 0.42, 0.06, "#4f7d3a", season, v + s);
  }
}

function commercial(b, lv, v, hh, h3) {
  const P = (arr, k) => pick(arr, v, k);
  if (lv === 1) {
    if (hh < 0.3) {                                 // café with parasols out front
      b.box(0, -0.12, 0.7, 0.5, 0, 0.3, P(PAL.shop, 8), { win: WIN.shop(v), top: "#8c8c8c" });
      for (const s of [-0.22, 0.22]) {
        b.box(s, 0.3, 0.015, 0.015, 0, 0.2, "#555555");
        b.cone(s, 0.3, 0.12, 0.17, 0.08, P(PAL.awning, 30 + s * 10), { seg: 8 });
      }
    } else {
      b.box(0, 0, 0.8, 0.66, 0, 0.34, P(PAL.shop, 8), { win: WIN.shop(v), top: "#8c8c8c" });
      b.box(0, 0.37, 0.82, 0.12, 0.22, 0.04, P(PAL.awning, 9));
      b.box(0, 0, 0.4, 0.05, 0.34, 0.12, P(PAL.sign, 31), { glow: P(PAL.sign, 31) });
    }
  } else if (lv === 2) {
    const hgt = 1.0 + h3 * 0.6;
    if (hh < 0.4) {                                 // mall: wide low box + signage
      b.box(0, 0, 0.9, 0.86, 0, 0.5, P(PAL.shop, 10), { win: WIN.shop(v), top: "#8c8c8c" });
      b.box(0, 0, 0.5, 0.5, 0.5, 0.2, P(PAL.glass, 10), { win: WIN.glass(v) });
      b.box(0, 0.45, 0.5, 0.03, 0.34, 0.1, P(PAL.sign, 32), { glow: P(PAL.sign, 32) });
    } else {
      b.box(0, 0, 0.72, 0.72, 0, hgt, P(PAL.glass, 10), { win: WIN.glass(v), top: "#7c8090" });
      b.box(-0.1, 0.1, 0.22, 0.22, hgt, 0.1, "#a0a6ad");
      b.box(0, 0, 0.42, 0.06, hgt + 0.05, 0.14, P(PAL.sign, 33), { glow: P(PAL.sign, 33) });
    }
  } else {
    const hgt = ([0, 0, 0, 2.4, 3.2, 4.2][lv] + h3 * 1.8) * TALL, top = "#7c8090";
    b.box(0, 0, 0.86, 0.86, 0, 0.28, "#c9cfd6", { win: WIN.shop(v) });
    if (hh < 0.3) b.cyl(0, 0, 0.31, 0.28, hgt, P(PAL.glass, 11), { seg: 12, win: WIN.glass(v), top });
    else if (hh < 0.55) {                           // stepped crown
      b.box(0, 0, 0.62, 0.62, 0.28, hgt * 0.7, P(PAL.glass, 11), { win: WIN.glass(v), top });
      b.box(0, 0, 0.44, 0.44, 0.28 + hgt * 0.7, hgt * 0.3, P(PAL.glass, 12), { win: WIN.glass(v + 1), top });
    } else b.box(0, 0, 0.56, 0.56, 0.28, hgt, P(PAL.glass, 11), { win: WIN.glass(v), top });
    if (hh > 0.4 || lv >= 5) b.cyl(0, 0, 0.03, hgt + 0.28, 0.5, "#d9dde2", { seg: 6 });
  }
}

function industry(b, vr, lv, v, hh, season) {
  if (vr === 1) {                                   // agriculture: crop rows, barn, silo
    for (let k = -2; k <= 2; k++) b.box(0, k * 0.17, 0.9, 0.1, 0, 0.03 + 0.02 * lv, (k + lv) % 2 ? "#d9c45a" : "#7fae4a");
    if (lv >= 2) { b.box(0.28, -0.28, 0.3, 0.26, 0, 0.26, "#a93226"); b.gable(0.28, -0.28, 0.34, 0.3, 0.26, 0.14, "#5d4037", { wall: "#a93226" }); }
    if (lv >= 3) { b.cyl(-0.3, -0.3, 0.07, 0, 0.62, "#cfd8dc", { seg: 8 }); b.dome(-0.3, -0.3, 0.07, 0.62, "#b0bec5", { seg: 8 }); }
  } else if (vr === 2) {                            // forestry: sawmill, log piles, a tree
    b.box(-0.1, -0.1, 0.5, 0.4, 0, 0.3 + 0.06 * lv, "#8d6e63", { win: WIN.factory(v) });
    b.gable(-0.1, -0.1, 0.54, 0.44, 0.3 + 0.06 * lv, 0.12, "#5d4037", { wall: "#8d6e63" });
    for (let k = 0; k < lv + 1; k++) b.box(0.28, 0.25 - k * 0.02, 0.26, 0.14, 0.04 * k, 0.05, k % 2 ? "#795548" : "#a1887f");
    tree(b, -0.35, 0.35, 0.8, 0, season, v);
  } else if (vr === 3) {                            // ore mine: headframe, spoil heap
    b.box(0.2, 0.2, 0.36, 0.36, 0, 0.26, "#757575", { win: WIN.factory(v) });
    for (const d of [-0.1, 0.1]) b.box(-0.2 + d, -0.2, 0.04, 0.04, 0, 0.6 + 0.12 * lv, "#424242");
    b.box(-0.2, -0.2, 0.26, 0.05, 0.6 + 0.12 * lv, 0.05, "#424242");
    b.blob(-0.28, 0.28, 0.04, 0.12 + 0.02 * lv, "#9e9e9e", { seed: v, lumpy: 0.25 });
    b.smoke(0.2, 0.2, 0.3, 0.3, 0.7);
  } else if (vr === 4) {                            // oil: pumpjacks, tanks, a flare
    for (let k = 0; k < Math.min(3, lv); k++) {
      const ox = -0.25 + k * 0.25;
      b.box(ox, 0.25, 0.05, 0.05, 0, 0.24, "#37474f"); b.box(ox, 0.25, 0.3, 0.05, 0.24, 0.05, "#f9a825");
    }
    b.cyl(-0.2, -0.22, 0.15, 0, 0.26, "#eceff1", { seg: 10 });
    if (lv >= 2) b.cyl(0.2, -0.22, 0.13, 0, 0.22, "#cfd8dc", { seg: 10 });
    if (lv >= 3) { b.cyl(0.38, 0.38, 0.03, 0, 0.8, "#616161", { seg: 6 }); b.smoke(0.38, 0.38, 0.85, 0.8, 0.9, true); }
  } else {                                          // high-tech campus: low glass buildings, green roof
    const hgt = [0, 0.35, 0.5, 0.7, 1.0, 1.4][lv];
    b.flat(0, 0, 0.9, 0.9, 0.01, "#8bc34a", { flat: true });
    b.box(-0.12, -0.1, 0.56, 0.46, 0.01, hgt, "#b3e5fc", { win: WIN.glass(v), top: "#6c9c3c" });
    b.box(0.26, 0.24, 0.3, 0.3, 0.01, hgt * 0.6, "#81d4fa", { win: WIN.glass(v + 1), top: "#1f3a73" });
  }
}

// ------------------------------------------------------------------ special structures
// Written for a unit tile like scene.js:_buildSpecial; bigger footprints are scaled by pixel.js via Builder.scale.
export function special(b, ch, v = 0, season = null) {
  const bx = (w, h, d, x, y, z, c, o) => b.box(x, z, w, d, y, h, c, o);                 // scene.js argument order
  const cy = (r1, r2, h, x, y, z, c, seg = 12, o = {}) => b.cyl(x, z, r2, y, h, c, { seg, rt: r1, ...o });
  if (ch === "T") {                                 // town hall
    bx(0.92, 0.12, 0.92, 0, 0, 0, "#d8d2c4");
    bx(0.78, 0.5, 0.6, 0, 0.12, -0.05, "#efe9dc", { win: WIN.house(1) });
    for (let k = -2; k <= 2; k++) cy(0.035, 0.035, 0.42, k * 0.15, 0.12, 0.3, "#ffffff", 6);
    bx(0.82, 0.08, 0.2, 0, 0.54, 0.28, "#e6dfd0");
    cy(0.2, 0.22, 0.14, 0, 0.62, -0.05, "#efe9dc");
    b.dome(0, -0.05, 0.2, 0.76, "#7a4fb3");
    cy(0.01, 0.01, 0.3, 0, 0.94, -0.05, "#dddddd", 4);
    bx(0.14, 0.08, 0.01, 0.08, 1.14, -0.05, "#e0b43a");
  } else if (ch === "E") {                          // coal plant
    bx(0.88, 0.4, 0.62, 0, 0, 0.12, "#5d5f63", { win: WIN.factory(2) });
    bx(0.4, 0.22, 0.3, -0.2, 0.4, 0.14, "#4a4c50");
    for (const sx of [-0.25, 0.22]) {
      cy(0.08, 0.11, 1.5, sx, 0, -0.3, "#9c8b80");
      cy(0.085, 0.085, 0.12, sx, 1.1, -0.3, "#c9453b");
      b.smoke(sx, -0.3, 1.55, 1.4, 0.3);
    }
    bx(0.3, 0.18, 0.3, 0.28, 0.4, 0.2, "#2b2b2b");
    for (let k = 0; k < 4; k++) b.blob(0.3 + (k % 2) * 0.1 - 0.05, 0.35, 0.02, 0.05, "#1c1c1c", { seed: k, lumpy: 0.3 });
  } else if (ch === "G") {                          // gas plant
    bx(0.55, 0.36, 0.45, -0.15, 0, 0.18, "#b7b9bd", { win: WIN.factory(3) });
    for (const [sx, sz] of [[0.22, -0.2], [0.22, 0.22]]) { cy(0.03, 0.03, 0.14, sx, 0, sz, "#888888", 6); b.blob(sx, sz, 0.3, 0.17, "#f0f0ee", { lumpy: 0, nosnow: true }); }
    cy(0.05, 0.06, 0.95, -0.3, 0, -0.25, "#d9822b");
    b.smoke(-0.3, -0.25, 1.0, 0.6, 0.75);
  } else if (ch === "S") {                          // solar farm: tilted panel rows
    bx(0.94, 0.02, 0.94, 0, 0, 0, "#9aa39a");
    for (const sx of [-0.23, 0.23]) for (const sz of [-0.23, 0.23]) {
      cy(0.015, 0.015, 0.1, sx, 0, sz, "#777777", 4);
      b.poly([[sx - 0.2, sz + 0.16, 0.08], [sx + 0.2, sz + 0.16, 0.08], [sx + 0.2, sz - 0.16, 0.2], [sx - 0.2, sz - 0.16, 0.2]], "#1f3a73", { two: true, nosnow: true });
    }
  } else if (ch === "W") {                          // wind turbine (rotor drawn by pixel.js)
    bx(0.5, 0.04, 0.5, 0, 0, 0, "#b8bdb5");
    cy(0.03, 0.06, 2.0, 0, 0, 0, "#f4f5f6", 8);
    bx(0.08, 0.08, 0.2, 0, 1.96, 0.02, "#e8eaec");
    b.rotor(0, 0.13, 2.0, 0.85);
  } else if (ch === "U") {                          // water pump
    bx(0.6, 0.34, 0.5, 0, 0, 0.1, "#dfe7ea", { win: WIN.factory(4) });
    bx(0.64, 0.06, 0.54, 0, 0.34, 0.1, "#1f8a8a");
    bx(0.9, 0.1, 0.1, 0.1, 0.06, -0.25, "#1f8a8a");
    cy(0.14, 0.14, 0.26, -0.28, 0, -0.22, "#8fb9c7");
  } else if (ch === "V") {                          // water tower
    for (const [sx, sz] of [[-0.2, -0.2], [0.2, -0.2], [-0.2, 0.2], [0.2, 0.2]]) bx(0.03, 1.1, 0.03, sx, 0, sz, "#8a9aa0");
    cy(0.34, 0.3, 0.45, 0, 1.05, 0, "#2fa3b3", 12);
    b.cone(0, 0, 0.36, 1.5, 0.2, "#1d7480", { seg: 12 });
  } else if (ch === "K") {                          // police station
    bx(0.86, 0.1, 0.86, 0, 0, 0, "#c9ced6");
    bx(0.72, 0.42, 0.56, 0, 0.1, 0.02, "#e8edf3", { win: WIN.house(5) });
    bx(0.74, 0.08, 0.58, 0, 0.52, 0.02, "#1f4fa8");
    bx(0.3, 0.06, 0.02, 0, 0.36, 0.31, "#1f4fa8");
    bx(0.1, 0.05, 0.06, -0.08, 0.6, 0.02, "#e53935", { glow: "#ff5040" }); bx(0.1, 0.05, 0.06, 0.08, 0.6, 0.02, "#1e88e5", { glow: "#40a0ff" });
    bx(0.02, 0.75, 0.02, 0.36, 0.1, 0.36, "#bbbbbb"); bx(0.14, 0.09, 0.01, 0.43, 0.74, 0.36, "#1f4fa8");
  } else if (ch === "J") {                          // jail
    bx(0.94, 0.06, 0.94, 0, 0, 0, "#9ea3a8");
    for (const [w, d, x, z] of [[0.9, 0.05, 0, -0.42], [0.9, 0.05, 0, 0.42], [0.05, 0.9, -0.42, 0], [0.05, 0.9, 0.42, 0]]) bx(w, 0.32, d, x, 0.06, z, "#7d8388");
    bx(0.44, 0.46, 0.4, 0, 0.06, 0, "#b0b4b8", { win: { ...WIN.factory(6), w: 1, h: 3, sp: 0.08 } });
    bx(0.12, 0.7, 0.12, 0.36, 0.06, -0.36, "#8a9095"); bx(0.2, 0.12, 0.2, 0.36, 0.76, -0.36, "#5e6368");
  } else if (ch === "F") {                          // fire station
    bx(0.9, 0.08, 0.9, 0, 0, 0, "#c9c4bb");
    bx(0.8, 0.44, 0.6, 0, 0.08, 0.05, "#c62828");
    bx(0.8, 0.06, 0.62, 0, 0.52, 0.05, "#8e1c1c");
    for (const x of [-0.2, 0.2]) bx(0.3, 0.3, 0.02, x, 0.08, 0.36, "#eeeeee");
    bx(0.2, 0.75, 0.2, -0.3, 0.08, -0.25, "#b71c1c", { win: WIN.house(7) }); bx(0.22, 0.05, 0.22, -0.3, 0.83, -0.25, "#6d1010");
  } else if (ch === "^") {                          // emergency shelter
    bx(0.9, 0.05, 0.9, 0, 0, 0, "#9aa39a");
    b.lathe(0, 0, [[0.36, 0.05], [0.33, 0.16], [0.24, 0.26], [0.1, 0.3]], "#b8b39c", { seg: 12 });
    bx(0.2, 0.14, 0.04, 0, 0.05, 0.34, "#2e7d32", { glow: "#4caf50" });
  } else if (ch === "A") {                          // clinic
    bx(0.86, 0.06, 0.86, 0, 0, 0, "#cfd8dc");
    bx(0.7, 0.36, 0.55, 0, 0.06, 0, "#fafafa", { win: WIN.house(8) });
    bx(0.26, 0.06, 0.07, 0, 0.42, 0, "#e53935", { glow: "#ff5a50" }); bx(0.07, 0.06, 0.26, 0, 0.42, 0, "#e53935", { glow: "#ff5a50" });
    bx(0.3, 0.14, 0.02, 0, 0.12, 0.285, "#90caf9");
  } else if (ch === "h") {                          // hospital
    bx(0.94, 0.06, 0.94, 0, 0, 0, "#cfd8dc");
    bx(0.5, 1.1, 0.5, -0.16, 0.06, -0.12, "#f5f7f8", { win: WIN.flat(9) });
    bx(0.86, 0.45, 0.34, 0, 0.06, 0.26, "#eceff1", { win: WIN.flat(10) });
    b.cyl(-0.16, -0.12, 0.18, 1.16, 0.02, "#455a64", { seg: 12 });
    bx(0.18, 0.02, 0.05, -0.16, 1.18, -0.12, "#ffffff"); bx(0.05, 0.02, 0.18, -0.16, 1.18, -0.12, "#ffffff");
    bx(0.24, 0.24, 0.02, 0.2, 0.2, 0.44, "#e53935", { glow: "#ff5a50" });
  } else if (ch === "L") {                          // landfill
    bx(0.96, 0.02, 0.96, 0, 0, 0, "#8d7b5a");
    b.lathe(0, 0, [[0.42, 0.02], [0.36, 0.12], [0.2, 0.22], [0.06, 0.25]], "#7a6a4f", { seg: 9, sy: 0.85 });
    for (let k = 0; k < 5; k++) bx(0.07, 0.05, 0.07, (hash(v, 60 + k) - 0.5) * 0.5, 0.16 + hash(v, 70 + k) * 0.06, (hash(v, 80 + k) - 0.5) * 0.4,
      ["#c0392b", "#2980b9", "#f1c40f", "#7f8c8d", "#27ae60"][k]);
    bx(0.16, 0.12, 0.26, 0.34, 0.02, 0.32, "#e0a526");
  } else if (ch === "X") {                          // incinerator
    bx(0.86, 0.06, 0.86, 0, 0, 0, "#8f8f8f");
    bx(0.62, 0.5, 0.5, -0.08, 0.06, 0.1, "#6b6e72", { win: WIN.factory(11) });
    bx(0.62, 0.06, 0.52, -0.08, 0.56, 0.1, "#3d4044");
    cy(0.08, 0.1, 1.35, 0.28, 0.06, -0.26, "#5b5e62");
    cy(0.085, 0.085, 0.1, 0.28, 1.2, -0.26, "#e0a526");
    b.smoke(0.28, -0.26, 1.45, 0.9, 0.35);
  } else if (ch === "Z") {                          // cemetery
    bx(0.96, 0.03, 0.96, 0, 0, 0, "#6f9f5a");
    for (let r = 0; r < 3; r++) for (let k = 0; k < 4; k++) bx(0.06, 0.08 + 0.03 * hash(v, r * 4 + k), 0.03, -0.3 + k * 0.2, 0.03, -0.3 + r * 0.25, "#b0b4b8");
    tree(b, 0.36, 0.36, 0.9, 0, season, v);
  } else if (ch === "N") {                          // elementary school
    bx(0.92, 0.04, 0.92, 0, 0, 0, "#b9c79a");
    bx(0.7, 0.34, 0.4, 0, 0.04, -0.2, "#f3d99b", { win: WIN.house(12) });
    b.gable(0, -0.2, 0.76, 0.46, 0.38, 0.16, "#d35400", { wall: "#f3d99b" });
    bx(0.36, 0.02, 0.3, -0.18, 0.04, 0.26, "#c77d4e");
    bx(0.06, 0.18, 0.06, 0.22, 0.04, 0.26, "#e74c3c"); bx(0.16, 0.03, 0.05, 0.22, 0.22, 0.26, "#3498db");
  } else if (ch === "B") {                          // high school with sports field
    bx(0.96, 0.03, 0.96, 0, 0, 0, "#6fae5a");
    bx(0.86, 0.48, 0.36, 0, 0.03, -0.26, "#b5654a", { win: WIN.flat(13) });
    bx(0.88, 0.05, 0.38, 0, 0.51, -0.26, "#6d3b2c");
    b.flat(0, 0.2, 0.7, 0.36, 0.035, "#3f8f3a", { flat: true });
    b.flat(0, 0.2, 0.7, 0.03, 0.04, "#ffffff", { flat: true, nosnow: true }); b.flat(0, 0.2, 0.03, 0.36, 0.04, "#ffffff", { flat: true, nosnow: true });
  } else if (ch === "Q") {                          // university
    bx(0.96, 0.03, 0.96, 0, 0, 0, "#7fb069");
    bx(0.86, 0.14, 0.66, 0, 0.03, -0.1, "#dcd6c8");
    bx(0.76, 0.5, 0.5, 0, 0.17, -0.12, "#efe9dc", { win: WIN.house(14) });
    for (let k = -3; k <= 3; k++) cy(0.025, 0.025, 0.46, k * 0.11, 0.17, 0.16, "#ffffff", 4);
    bx(0.8, 0.07, 0.12, 0, 0.63, 0.15, "#e6dfd0");
    b.dome(0, -0.12, 0.22, 0.67, "#2d6a9f");
    for (const x of [-0.34, 0.34]) tree(b, x, 0.36, 0.7, 3, season, v + x * 10);
  } else if (ch === "D") {                          // library
    bx(0.9, 0.08, 0.9, 0, 0, 0, "#d2ccbd");
    bx(0.72, 0.38, 0.5, 0, 0.08, -0.08, "#e4dccb", { win: WIN.house(15) });
    for (let k = -2; k <= 2; k++) cy(0.025, 0.025, 0.36, k * 0.14, 0.08, 0.22, "#f5f1e8", 4);
    b.gable(0, 0.02, 0.78, 0.66, 0.46, 0.14, "#2e5b3c", { wall: "#e4dccb" });
  } else if (ch === "%") {                          // sewage outlet
    bx(0.5, 0.2, 0.5, -0.1, 0, 0, "#8d8575");
    bx(0.6, 0.12, 0.12, 0.28, 0.02, 0, "#5d5446");
    bx(0.1, 0.02, 0.2, 0.52, -0.03, 0, "#6d5f3f");
  } else if (ch === "*") {                          // sewage treatment plant
    bx(0.94, 0.04, 0.94, 0, 0, 0, "#a8a89e");
    for (const [x, z] of [[-0.22, -0.2], [0.22, -0.2], [-0.22, 0.22]]) {
      b.cyl(x, z, 0.19, 0.04, 0.12, "#9e9e96", { seg: 12, top: "#4f8f6f" });
    }
    bx(0.26, 0.3, 0.26, 0.24, 0.04, 0.24, "#d7d4ca", { win: WIN.factory(16) });
  } else if (ch === "Y") {                          // hydro dam across the river
    bx(1.0, 0.45, 0.3, 0, -0.25, 0, "#b8b3a8");
    bx(1.0, 0.05, 0.34, 0, 0.2, 0, "#8d8a82");
    bx(0.34, 0.24, 0.3, 0.2, 0.2, 0.2, "#d7d2c7", { win: WIN.factory(17) });
    for (let k = -1; k <= 1; k++) bx(0.12, 0.2, 0.02, k * 0.28, -0.2, 0.16, "#6fb6e8", { glow: "#8fd0ff" });
  } else if (ch === "|") {                          // levee
    bx(1.0, 0.18, 0.7, 0, 0, 0, "#6f8f4f");
    bx(1.0, 0.04, 0.4, 0, 0.18, 0, "#8d7b5a");
  } else if (ch === "@") {                          // train station
    bx(0.96, 0.06, 0.96, 0, 0, 0, "#9e9e9e");
    for (const z of [-0.3, 0.3]) bx(0.96, 0.03, 0.05, 0, 0.06, z, "#5d4037");
    bx(0.96, 0.08, 0.3, 0, 0.06, 0, "#d7ccc8");
    for (const x of [-0.35, 0, 0.35]) bx(0.03, 0.36, 0.03, x, 0.14, 0, "#555555");
    bx(1.0, 0.03, 0.46, 0, 0.5, 0, "#b23b3b");
  } else if (ch === "&") {                          // commercial port: quay on the river (+y), gantry crane, containers
    bx(0.98, 0.08, 0.98, 0, 0, 0, "#9a968c");
    b.flat(0, 0.44, 0.98, 0.06, 0.081, "#e8c040", { flat: true, nosnow: true });           // quay edge
    for (const x of [-0.36, 0, 0.36]) bx(0.05, 0.04, 0.05, x, 0.08, 0.42, "#3a3d42");       // bollards
    const box = ["#c0392b", "#2980b9", "#27ae60", "#f39c12", "#8e44ad", "#16a085", "#d35400"];
    for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++) {                               // container stacks
      const n = 1 + Math.floor(hash(v * 9 + r * 3 + c, 61) * 3);
      for (let k = 0; k < n; k++) bx(0.26, 0.09, 0.12, -0.3 + c * 0.3, 0.08 + k * 0.09, -0.36 + r * 0.15, box[(r * 3 + c + k * 2 + v) % box.length]);
    }
    bx(0.24, 0.16, 0.16, 0.34, 0.08, -0.34, "#e6e1d6", { win: WIN.house(22) });            // port office
    for (const x of [-0.26, 0.26]) for (const z of [0.02, 0.36]) bx(0.04, 0.9, 0.04, x, 0.08, z, "#f2b705");   // crane legs
    bx(0.6, 0.06, 0.06, 0, 0.98, 0.02, "#f2b705"); bx(0.6, 0.06, 0.06, 0, 0.98, 0.36, "#f2b705");
    bx(0.08, 0.06, 1.3, 0, 1.04, 0.5, "#f2b705");                                          // boom out over the water
    bx(0.14, 0.12, 0.14, 0, 0.92, 0.3, "#555a60", { win: { c: "#9cc4e4", sp: 0.1, fl: 0.2, w: 2, h: 2, z0: 0.02, top: 0.01, lit: 1 } });
    bx(0.01, 0.4, 0.01, 0, 0.6, 0.9, "#333333"); bx(0.16, 0.05, 0.1, 0, 0.56, 0.9, "#2980b9");   // cable + spreader
  } else if (ch === "+") {                          // plaza with a fountain
    bx(0.98, 0.03, 0.98, 0, 0, 0, "#d8d2c4");
    b.cyl(0, 0, 0.26, 0.03, 0.08, "#bdb6a6", { seg: 12, top: "#4aa6e0" });
    cy(0.03, 0.05, 0.3, 0, 0.11, 0, "#e6dfd0", 6);
    for (const [x, z] of [[-0.38, -0.38], [0.38, -0.38], [-0.38, 0.38], [0.38, 0.38]]) tree(b, x, z, 0.6, 3, season, v + x * 3 + z);
  } else if (ch === "$") {                          // stadium
    b.lathe(0, 0, [[0.48, 0], [0.46, 0.2], [0.42, 0.36]], "#dfe3e8", { seg: 16, sy: 0.8, top: "#3f9f4a" });
    b.flat(0, 0, 0.4, 0.3, 0.37, "#3f9f4a", { flat: true });
    b.flat(0, 0, 0.02, 0.3, 0.375, "#ffffff", { flat: true, nosnow: true });
  } else if (ch === "!") {                          // monument
    bx(0.7, 0.1, 0.7, 0, 0, 0, "#d8d2c4");
    bx(0.4, 0.12, 0.4, 0, 0.1, 0, "#c9c1ae");
    b.lathe(0, 0, [[0.12, 0.22], [0.05, 1.95], [0, 2.1]], "#efe9dc", { seg: 4, a0: 0 });
  } else if (ch === ":") {                          // metro station entrance
    bx(0.9, 0.04, 0.9, 0, 0, 0, "#cfd3d8");
    bx(0.5, 0.26, 0.36, 0, 0.04, 0, "#37474f");
    b.gable(0, 0, 0.5, 0.4, 0.3, 0.12, "#90a4ae", { axis: "y", wall: "#37474f" });
    bx(0.02, 0.7, 0.02, 0.32, 0.04, 0.3, "#555555"); bx(0.14, 0.14, 0.03, 0.32, 0.72, 0.3, "#d32f2f", { glow: "#ff4a40" });
  } else if (ch === "?") {                          // casino
    bx(0.94, 0.06, 0.94, 0, 0, 0, "#2b1d3a");
    bx(0.7, 0.42, 0.6, 0, 0.06, -0.08, "#6a1b9a", { win: WIN.shop(18) });
    bx(0.36, 0.95, 0.36, -0.14, 0.06, -0.14, "#4a148c", { win: { ...WIN.glass(19), c: "#e0b0ff", glow: "#ff80ff" } });
    bx(0.74, 0.05, 0.64, 0, 0.48, -0.08, "#ffd54f", { glow: "#ffc030" });
    for (const [x, c] of [[-0.3, "#ff4081"], [0, "#40c4ff"], [0.3, "#ffea00"]]) bx(0.08, 0.08, 0.02, x, 0.3, 0.23, c, { glow: c });
    b.dome(-0.14, -0.14, 0.14, 1.01, "#ffd54f");
  } else if (ch === ">") {                          // airport, native 3x3: runway along x at y = +0.9 (lifesim.js flies it)
    b.flat(0, 0, 2.96, 2.96, 0.005, "#6f9f5a", { flat: true });
    b.flat(0, 0.05, 2.9, 1.0, 0.012, "#a8a8a0", { flat: true, nosnow: true, tex: speckle(v, 0.1, 0.92) });   // apron
    b.flat(0, 0.9, 2.84, 0.46, 0.014, "#3a3d42", { flat: true, nosnow: true, tex: speckle(v + 1, 0.1, 0.9) });
    for (let k = -4; k <= 4; k++) b.flat(k * 0.28, 0.9, 0.14, 0.03, 0.016, "#f5f5f5", { flat: true, nosnow: true });
    for (const e of [-1.32, 1.32]) for (let k = -2; k <= 2; k++) b.flat(e, 0.9 + k * 0.08, 0.12, 0.035, 0.016, "#f5f5f5", { flat: true, nosnow: true });
    for (let k = -5; k <= 5; k++) for (const s of [0.66, 1.14]) b.box(k * 0.27, s, 0.02, 0.02, 0.014, 0.02, "#dddddd", { glow: s < 1 ? "#ffe9a0" : "#9fd0ff" });
    for (const x of [-1.3, 1.0]) { b.flat(x, 0.57, 0.16, 0.2, 0.014, "#55595f", { flat: true, nosnow: true }); b.flat(x, 0.57, 0.02, 0.2, 0.016, "#e8c040", { flat: true, nosnow: true }); }
    b.flat(-0.15, 0.25, 2.5, 0.12, 0.013, "#55595f", { flat: true, nosnow: true }); b.flat(-0.15, 0.25, 2.5, 0.015, 0.015, "#e8c040", { flat: true, nosnow: true });
    bx(1.8, 0.3, 0.46, -0.35, 0.012, -0.95, "#cfd8dc", { win: WIN.glass(20), top: "#90a4ae" });            // terminal
    bx(1.82, 0.04, 0.48, -0.35, 0.312, -0.95, "#78909c");
    for (const x of [-0.85, -0.35, 0.15, 0.6]) bx(0.05, 0.05, 0.36, x, 0.14, -0.55, "#b0bec5");               // jet bridges
    cy(0.07, 0.09, 0.9, 1.2, 0.012, -0.95, "#eceff1", 8);                                               // control tower
    cy(0.18, 0.14, 0.14, 1.2, 0.91, -0.95, "#4fc3f7", 8, { glow: "#8fe0ff" }); cy(0.19, 0.19, 0.03, 1.2, 1.05, -0.95, "#546e7a", 8);
    bx(0.42, 0.26, 0.36, 1.18, 0.012, -0.28, "#b0b4b8");                                                 // hangar
    b.gable(1.18, -0.28, 0.44, 0.38, 0.27, 0.1, "#78909c", { wall: "#b0b4b8" });
    b.flat(-1.2, -0.3, 0.42, 0.3, 0.013, "#6b6f75", { flat: true, nosnow: true });                       // car park
    for (let k = 0; k < 6; k++) bx(0.1, 0.04, 0.05, -1.34 + (k % 3) * 0.14, 0.013, -0.38 + Math.floor(k / 3) * 0.16, ["#e53935", "#1e88e5", "#fafafa", "#fdd835", "#43a047", "#212121"][k]);
    for (const x of [-0.85, -0.35]) { b.move(x, -0.2).angle(-Math.PI / 2); plane(b, false, true); b.move(0, 0).angle(0); }
  } else if (ch === "H") {                          // highway sign gantry at the map edge
    bx(0.02, 0.8, 0.02, 0.3, 0, -0.42, "#777777"); bx(0.02, 0.8, 0.02, -0.3, 0, -0.42, "#777777");
    bx(0.7, 0.22, 0.03, 0, 0.62, -0.42, "#1b7a3e");
  }
}

// ------------------------------------------------------------------ small things on tiles
export function wild(b, v, hh, h3, season, rs) {   // empty land: a few trees, bushes, rocks
  if (rs === 2) { for (let k = 0; k < 3; k++) tree(b, (hash(v, 50 + k) - 0.5) * 0.7, (hash(v, 55 + k) - 0.5) * 0.7, 0.6 + 0.4 * hash(v, 58 + k), hash(v, 59 + k) < 0.5 ? 0 : 2, season, v + k); return; }
  if (rs === 3) { b.blob((h3 - 0.5) * 0.4, (hh - 0.5) * 0.5, 0.02, 0.1, "#8d8d8d", { seed: v, lumpy: 0.3 }); b.blob(0.2, 0.15, 0.01, 0.06, "#9e9e9e", { seed: v + 1, lumpy: 0.3 }); return; }
  if (hh < 0.09) tree(b, (h3 - 0.5) * 0.5, (hash(v, 61) - 0.5) * 0.5, 0.6 + h3 * 0.5, Math.floor(hash(v, 62) * 4), season, v);
  else if (hh < 0.19) bush(b, (h3 - 0.5) * 0.6, (hash(v, 61) - 0.5) * 0.6, 0.07, pick(LEAVES.summer, v, 3), season, v);
  else if (hh < 0.23) b.blob((h3 - 0.5) * 0.6, (hash(v, 62) - 0.5) * 0.6, 0.02, 0.05, "#9a9890", { seed: v, lumpy: 0.25 });
}
export function woods(b, v, n, season) {           // countryside forest
  for (let k = 0; k < n; k++) tree(b, (hash(v, 70 + k) - 0.5) * 0.75, (hash(v, 80 + k) - 0.5) * 0.75, 0.7 + 0.5 * hash(v, 90 + k), hash(v, 95 + k) < 0.55 ? 0 : 1 + Math.floor(hash(v, 96 + k) * 3), season, v + k);
}
export function farm(b) { b.box(0, 0, 0.5, 0.36, 0, 0.32, "#a93226", { win: WIN.factory(21) }); b.gable(0, 0, 0.54, 0.4, 0.32, 0.18, "#5d4037", { wall: "#a93226" }); b.cyl(0.32, -0.1, 0.07, 0, 0.5, "#cfd8dc", { seg: 8 }); }
export function park(b, v, season) {
  for (let k = 0; k < 3; k++) tree(b, [-0.28, 0.26, -0.22][k] + (hash(v, 10 + k) - 0.5) * 0.12, [-0.26, -0.2, 0.28][k] + (hash(v, 20 + k) - 0.5) * 0.12, 0.7 + hash(v, 30 + k) * 0.4, Math.floor(hash(v, 35 + k) * 4), season, v + k);
  if (hash(v, 40) < 0.5) { b.box(0.28, 0.24, 0.16, 0.05, 0.03, 0.02, "#8d6e63"); b.box(0.28, 0.22, 0.16, 0.015, 0.05, 0.04, "#8d6e63"); }
  else b.cyl(0.26, 0.26, 0.1, 0, 0.05, "#bdb6a6", { seg: 8, top: "#4aa6e0" });
}
export function lamp(b, x, y) { b.box(x, y, 0.02, 0.02, 0, 0.3, "#5f6368"); b.box(x, y, 0.05, 0.05, 0.3, 0.025, "#fff3c4", { glow: "#fff2b0" }); }
export function busStop(b, color) {
  b.box(0.42, 0.42, 0.02, 0.02, 0, 0.3, "#8a8f96"); b.box(0.42, 0.42, 0.06, 0.02, 0.26, 0.06, color, { glow: color });
  b.box(0.3, 0.44, 0.2, 0.08, 0.14, 0.015, "#90a4ae");
}
export function signals(b) {                       // traffic light poles on opposite corners; pixel.js lights the lamps
  for (const d of [0.44, -0.44]) { b.box(d, d, 0.025, 0.025, 0, 0.24, "#3a3d42"); b.box(d, d, 0.07, 0.07, 0.22, 0.18, "#1c1e22", { top: "#2c2f34" }); }
}
export function plaque(b) { b.box(0.38, 0.4, 0.16, 0.03, 0, 0.22, "#e0b43a"); }

// ------------------------------------------------------------------ construction
export function foundation(b) {
  b.flat(0, 0, 0.84, 0.84, 0.005, "#8c6c40", { flat: true, tex: speckle(3, 0.2, 0.85) });
  b.box(0, 0, 0.6, 0.6, 0, 0.04, "#b0aca0");
  b.box(-0.3, 0.3, 0.12, 0.12, 0, 0.1, "#c89c5c"); b.box(-0.15, 0.32, 0.1, 0.1, 0, 0.08, "#b8894c");
}
export function scaffold(b, h, w = 0.7) {
  const d = w / 2, pole = "#9aa0a6", plank = "#c89c5c";
  for (const [x, y] of [[-d, -d], [d, -d], [-d, d], [d, d], [0, d], [d, 0]]) b.box(x, y, 0.025, 0.025, 0, h, pole);
  for (let z = 0.18; z < h; z += 0.2) {
    b.box(0, d, w, 0.03, z, 0.018, plank); b.box(d, 0, 0.03, w, z, 0.018, plank);
    b.box(0, -d, w, 0.02, z, 0.015, pole); b.box(-d, 0, 0.02, w, z, 0.015, pole);
  }
  if (h > 1) {                                       // tower crane
    b.box(-d - 0.06, -d - 0.06, 0.05, 0.05, 0, h + 0.5, "#f2b705");
    b.box(-d - 0.06 + 0.3, -d - 0.06, 0.9, 0.04, h + 0.5, 0.04, "#f2b705");
    b.box(-d - 0.3, -d - 0.06, 0.12, 0.08, h + 0.44, 0.08, "#555555");
  }
}

// ------------------------------------------------------------------ vehicles (canonical heading +x)
export function car(b, color) {
  b.box(0, 0, 0.22, 0.11, 0.012, 0.055, color);
  b.box(-0.015, 0, 0.12, 0.1, 0.067, 0.045, hx(mix(color, "#9fb7c9", 0.65)));
  b.box(0.112, 0, 0.004, 0.08, 0.03, 0.02, "#fff6c0", { glow: "#fff6c0" });
}
export function bus(b, color) {
  b.box(0, 0, 0.42, 0.13, 0.015, 0.13, color, { top: "#dddddd", win: { c: "#223344", fl: 0.2, sp: 0.07, w: 2, h: 2, z0: 0.06, top: 0.02, lit: 0.95, glow: "#fff0b0" } });
}
export function wagon(b, color, loco) {
  b.box(0, 0, 0.48, 0.15, 0.02, 0.14, color, { top: loco ? hx(darken(color, 0.3)) : "#b0b6ba", win: { c: "#2a3440", fl: 0.2, sp: 0.08, w: 2, h: 2, z0: 0.07, top: 0.02, lit: 0.9 } });
}
export function boat(b, color, sail) {
  b.box(0, 0, 0.36, 0.14, -0.04, 0.07, color, { nosnow: true });
  if (sail) { b.box(0, 0, 0.012, 0.012, 0.03, 0.34, "#8d6e63"); b.poly([[0.01, 0, 0.06], [0.01, 0, 0.36], [0.16, 0, 0.06]], "#fafafa", { two: true, nosnow: true }); }
  else b.box(-0.04, 0, 0.14, 0.1, 0.03, 0.07, "#eeeeee", { win: { c: "#34465a", sp: 0.05, fl: 0.2, w: 2, h: 2, z0: 0.02, top: 0.01, lit: 0.9 } });
}
export function cargoShip(b, hull, v = 0) {        // container ship, bow towards +x
  b.box(0, 0, 0.92, 0.22, -0.06, 0.1, hull, { nosnow: true, top: "#7a4a2f" });
  b.box(0.44, 0, 0.08, 0.14, -0.06, 0.1, hull, { nosnow: true });
  const box = ["#c0392b", "#2980b9", "#27ae60", "#f39c12", "#8e44ad", "#16a085"];
  for (let k = 0; k < 4; k++) for (let h = 0; h < 1 + ((k + v) % 2); h++) b.box(-0.1 + k * 0.14, 0, 0.12, 0.18, 0.04 + h * 0.06, 0.06, box[(k * 2 + h + v) % box.length], { nosnow: true });
  b.box(-0.35, 0, 0.14, 0.18, 0.04, 0.14, "#f2f2ee", { win: { c: "#34465a", sp: 0.06, fl: 0.2, w: 2, h: 2, z0: 0.08, top: 0.02, lit: 1 } });
  b.box(-0.4, 0, 0.05, 0.06, 0.18, 0.06, "#c0392b");
}
export function plane(b, big = false, parked = false) {
  const k = big ? 1.4 : 1, z = parked ? 0.06 : 0;
  b.box(0, 0, 0.7 * k, 0.09 * k, z, 0.08 * k, "#f5f5f5", { nosnow: !parked });
  b.box(0.02, 0, 0.14 * k, 0.8 * k, z + 0.03 * k, 0.02 * k, "#e0e0e0", { nosnow: !parked });
  b.box(-0.3 * k, 0, 0.08 * k, 0.02 * k, z + 0.05 * k, 0.16 * k, "#1976d2");
}

// ------------------------------------------------------------------ ground
// Textures run on sprite pixels; seeds give each variant its own pattern.
export function speckle(seed, p = 0.18, dk = 0.86, up = 0.07) {
  return (x, y, c) => { const h = hash(x * 73 + y * 1031, seed); return h < p ? [c[0] * dk, c[1] * dk, c[2] * dk] : h > 1 - up ? [c[0] * 1.1, c[1] * 1.1, c[2] * 1.1] : c; };
}
export const GRASS = { spring: "#5ca03c", summer: "#6a9c38", autumn: "#80983c", winter: "#e4eaf0", null: "#5c9c38" };
function grassTex(seed, season, flowers) {
  const winter = season === "winter";
  return (x, y, c) => {
    const h = hash(x * 73 + y * 1031, seed);
    if (winter) return h < 0.12 ? rgb("#c4ccd8") : h > 0.985 ? rgb("#7c9c6c") : c;
    if (flowers && h > 0.965) return rgb(flowers[Math.floor(hash(x + y * 7, seed + 1) * flowers.length)]);
    return h < 0.2 ? [c[0] * 0.84, c[1] * 0.86, c[2] * 0.84] : h > 0.93 ? [c[0] * 1.1, c[1] * 1.1, c[2] * 1.06] : c;
  };
}
const full = (b, col, tex) => b.flat(0, 0, 1, 1, 0, col, { flat: true, tex, nosnow: true });
const ZONE_C = { r: "#3cb44b", c: "#3c78e0", i: "#e0b030", o: "#9070d0", m: "#d08850" };

// p: { kind, v, season, ... } — see pixel.js:_groundKey for the inputs of every kind
export function ground(p) {
  const b = new Builder(), s = p.season, winter = s === "winter", v = p.v || 0;
  const grass = p.fertile && !winter ? mix(GRASS[s] || GRASS.null, "#a9c96a", 0.4) : (GRASS[s] || GRASS.null);
  switch (p.kind) {
    case "grass": full(b, p.dark && !winter ? mix(grass, "#3c6c2c", 0.22) : grass, grassTex(v, s, p.flowers && s !== "winter" && s !== "autumn" ? PAL.flowers : null)); break;
    case "oil": full(b, grass, grassTex(v, s)); b.flat(0, 0, 0.5, 0.4, 0.001, "#3e3530", { flat: true, nosnow: true }); break;
    case "field": {
      const [a, c] = winter ? ["#e8ecf0", "#d4dce4"] : p.crop === "plough" ? ["#8c6c40", "#7c5c34"] : s === "autumn" ? ["#c8a048", "#a88c3c"] : ["#c9b458", "#9fb34f"];
      b.turn(p.dir);
      for (let k = 0; k < 8; k++) b.flat(0, -0.5 + (k + 0.5) / 8, 1, 1 / 8, 0, k % 2 ? a : c, { flat: true, nosnow: true, tex: speckle(v + k, 0.15, 0.9) });
      break;
    }
    case "lot": {                                   // zoned but empty: grass with the zone's dashed border and stakes
      full(b, mix(grass, "#c8c090", winter ? 0 : 0.25), grassTex(v, s));
      const zc = ZONE_C[p.zone] || "#ffffff";
      for (let k = 0; k < 5; k++) {
        const t = -0.4 + k * 0.2;
        b.flat(t, 0.46, 0.1, 0.04, 0.001, zc, { flat: true }); b.flat(t, -0.46, 0.1, 0.04, 0.001, zc, { flat: true });
        b.flat(0.46, t, 0.04, 0.1, 0.001, zc, { flat: true }); b.flat(-0.46, t, 0.04, 0.1, 0.001, zc, { flat: true });
      }
      break;
    }
    case "dev": {                                   // under a building
      const z = p.zone;
      if (z === "R") { full(b, mix(grass, "#88c060", winter ? 0 : 0.2), grassTex(v, s)); b.turn(p.q || 0); b.flat(0, 0.3, 0.12, 0.4, 0.001, "#c8bca0", { flat: true, nosnow: true }); }
      else if (z === "I") full(b, "#a8a090", speckle(v, 0.22, 0.85));
      else if (z === "O") full(b, "#c0c0c8", speckle(v, 0.12, 0.9));
      else if (z === "S") full(b, "#b8b4a8", speckle(v, 0.14, 0.9));
      else full(b, "#b8b4a8", (x, y, c) => ((x >> 2) + (y >> 1)) % 2 ? [c[0] * 0.94, c[1] * 0.94, c[2] * 0.94] : c);
      break;
    }
    case "park": {
      full(b, mix(grass, "#5cbc44", winter ? 0 : 0.3), grassTex(v, s, winter ? null : PAL.flowers));
      b.flat(0, 0, 1, 0.1, 0.001, winter ? "#dde2e6" : "#d8c8a0", { flat: true, nosnow: true }); b.flat(0, 0, 0.1, 1, 0.001, winter ? "#dde2e6" : "#d8c8a0", { flat: true, nosnow: true });
      break;
    }
    case "road": road(b, p); break;
    case "rail": full(b, grass, grassTex(v, s)); rail(b, p.mask ?? (p.ew ? 5 : 10), v); break;
    case "crossing": {                              // level crossing: the road, with the rails laid across it
      road(b, p);
      b.turn(p.railEw ? 0 : 1);
      for (const t of [-0.11, 0.11]) b.flat(0, t, 1, 0.035, 0.003, "#c0c0c0", { flat: true, nosnow: true });
      for (const t of [-0.5, 0.5]) b.flat(t * 0.9, 0.3, 0.04, 0.04, 0.003, "#e53935", { flat: true, nosnow: true });
      break;
    }
    case "railbridge": water(b, p); bridge(b, p, true); break;
    case "water": case "bridge": water(b, p); if (p.kind === "bridge") bridge(b, p); break;
    case "tint": full(b, p.color); break;
  }
  return b;
}

// river blue when clean, only a little murkier at the worst pollution (dirty 0-4)
const WATER = "#3468c0", MURKY = "#4f7a5c", MURK_MAX = 0.35;
function water(b, p) {
  const dirty = p.dirty || 0, col = dirty ? mix(WATER, MURKY, MURK_MAX * Math.min(1, dirty / 4)) : rgb(WATER);
  const f = p.frame || 0, wv = mix(col, "#ffffff", 0.3);
  b.flat(0, 0, 1, 1, -0.1, col, { flat: true, nosnow: true, tex: (x, y, c) => {
    const w = (x + 2 * y + f * 4) & 15, h = hash(x * 7 + y * 13, 5);
    return w < 2 && h < 0.55 ? wv : h > 0.97 ? mix(c, "#ffffff", 0.18) : c;
  } });
  const bank = "#7c6444", m = p.mask || 0;           // banks where the neighbour is land (bits: +x, +y, -x, -y)
  if (m & 1) b.wall([[0.5, -0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, -0.1], [0.5, -0.5, -0.1]], [-1, 0], bank);
  if (m & 2) b.wall([[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, -0.1], [-0.5, 0.5, -0.1]], [0, -1], bank);
  if (m & 4) b.wall([[-0.5, -0.5, 0], [-0.5, 0.5, 0], [-0.5, 0.5, -0.1], [-0.5, -0.5, -0.1]], [1, 0], bank);
  if (m & 8) b.wall([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, -0.5, -0.1], [-0.5, -0.5, -0.1]], [0, 1], bank);
}
function bridge(b, p, rails = false) {
  b.turn(p.ew ? 0 : 1);
  b.box(0, 0, 0.2, 0.2, -0.1, 0.12, "#8d8a82");
  if (rails) {                                      // girder bridge carrying the track
    b.box(0, 0, 1, 0.56, 0.02, 0.06, "#8d8578", { top: "#8d8578", nosnow: true });
    for (const t of [-0.375, -0.125, 0.125, 0.375]) b.flat(t, 0, 0.07, 0.4, 0.081, "#5d4037", { flat: true, nosnow: true });
    for (const t of [-0.11, 0.11]) b.flat(0, t, 1, 0.035, 0.082, "#c0c0c0", { flat: true, nosnow: true });
    for (const s of [-0.3, 0.3]) b.box(0, s, 1, 0.03, 0.08, 0.1, "#7a4a2f");
    return;
  }
  b.box(0, 0, 1, 0.84, 0.02, 0.06, "#b8b3a8", { top: "#55595f", nosnow: true });
  for (const t of [-0.25, 0.25]) b.flat(t, 0, 0.18, 0.04, 0.081, "#f2e7b5", { flat: true, nosnow: true });
  for (const s of [-0.43, 0.43]) b.box(0, s, 1, 0.04, 0.08, 0.05, "#d7d2c7");
}
// track with arms towards the neighbours in mask (bits +x, +y, -x, -y); straight runs, corners and junctions
function rail(b, mask, v) {
  const ballast = { flat: true, tex: speckle(v, 0.3, 0.8) }, wood = { flat: true, nosnow: true }, steel = { flat: true, nosnow: true };
  if (mask === 5 || mask === 10 || !mask || mask === 1 || mask === 4 || mask === 2 || mask === 8) {
    b.turn(mask & 10 && !(mask & 5) ? 1 : 0);
    b.flat(0, 0, 1, 0.46, 0.001, "#8d8578", ballast);
    for (const t of [-0.375, -0.125, 0.125, 0.375]) b.flat(t, 0, 0.07, 0.4, 0.002, "#5d4037", wood);
    for (const t of [-0.11, 0.11]) b.flat(0, t, 1, 0.035, 0.003, "#c0c0c0", steel);
    b.turn(0);
    return;
  }
  b.flat(0, 0, 0.46, 0.46, 0.001, "#8d8578", ballast);
  [[1, 0], [0, 1], [-1, 0], [0, -1]].forEach(([dx, dy], k) => {
    if (!(mask & (1 << k))) return;
    b.flat(dx * 0.25, dy * 0.25, dx ? 0.5 : 0.46, dy ? 0.5 : 0.46, 0.001, "#8d8578", ballast);
    for (const t of [0.125, 0.375]) b.flat(dx * t, dy * t, dx ? 0.07 : 0.4, dy ? 0.4 : 0.07, 0.002, "#5d4037", wood);
    for (const o of [-0.11, 0.11]) b.flat(dx * 0.25 + (dy ? o : 0), dy * 0.25 + (dx ? o : 0), dx ? 0.5 + 0.11 : 0.035, dy ? 0.5 + 0.11 : 0.035, 0.003, "#c0c0c0", steel);
  });
}
function road(b, p) {
  const { mask = 0, cls = 1, v = 0 } = p, hw = cls >= 3, av = cls === 2;
  const a = hw ? 0.44 : av ? 0.34 : 0.26;
  full(b, p.outer && !hw ? "#8c8c84" : "#a8a8a0", speckle(v, 0.1, 0.9));
  const asph = av || hw ? "#48484c" : "#58585c", tex = speckle(v + 1, 0.12, 0.88), o = { flat: true, nosnow: true, tex };
  b.flat(0, 0, 2 * a, 2 * a, 0.001, asph, o);
  const arms = [[1, 0], [0, 1], [-1, 0], [0, -1]].filter((_, k) => mask & (1 << k));
  for (const [dx, dy] of arms) {
    const L = 0.5 - a, c = a + L / 2;
    b.flat(dx * c, dy * c, dx ? L : 2 * a, dy ? L : 2 * a, 0.001, asph, o);
  }
  const straight = mask === 5 || mask === 10, ew = mask === 5 || (!straight && (mask & 5) && !(mask & 10));
  const paint = { flat: true, nosnow: true };
  if (straight || arms.length === 1) {
    const along = straight ? (ew ? [1, 0] : [0, 1]) : arms[0];
    const ax = along[0] !== 0;
    const line = (off, col, dash) => {
      if (dash) for (const t of [-0.375, -0.125, 0.125, 0.375]) b.flat(ax ? t : off, ax ? off : t, ax ? 0.12 : 0.04, ax ? 0.04 : 0.12, 0.002, col, paint);
      else b.flat(ax ? 0 : off, ax ? off : 0, ax ? 1 : 0.035, ax ? 0.035 : 1, 0.002, col, paint);
    };
    if (hw) { line(0, "#e8c040", false); for (const s of [-1, 1]) { line(s * 0.2, "#e8e8e8", true); line(s * 0.4, "#e8e8e8", false); } }
    else if (av) { line(-0.025, "#e8c040", false); line(0.025, "#e8c040", false); for (const s of [-1, 1]) line(s * 0.17, "#d8d8d8", true); }
    else line(0, "#e8d890", true);
  } else if (arms.length >= 3) {                     // zebra crossings on each arm
    for (const [dx, dy] of arms) for (let k = -2; k <= 2; k++) {
      const t = k * a * 0.36, c = a + 0.04;
      b.flat(dx ? dx * c : t, dy ? dy * c : t, dx ? 0.06 : 0.05, dy ? 0.06 : 0.05, 0.002, "#e8e8e8", paint);
    }
  }
}
