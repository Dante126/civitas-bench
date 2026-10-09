// Seasons change gradually across the city instead of all at once. The game's season is by month (catalog.SEASONS:
// Dec-Feb winter, Mar-May spring, Jun-Aug summer, Sep-Nov autumn); here each tile turns at its own time within
// TRANSITION months around each boundary, in smooth patches (value noise plus a little per-tile jitter), so the grass
// and the leaves turn over a month or two. Snow is all or nothing: the whole city is under it in the winter months and
// none of it in the others (patchy snow looks like missing textures). Both renderers (pixel.js, scene.js) use it, so
// the patches match between views.
import { hash } from "./sprites.js";

export const SEASONS = ["winter", "spring", "summer", "autumn"];
const TRANSITION = 2.4;                             // months a boundary takes to sweep across the city

function noise(x, y) {                              // smooth value noise in [0, 1), patches of ~9 tiles
  const s = 9, gx = Math.floor(x / s), gy = Math.floor(y / s), u = x / s - gx, v = y / s - gy, sm = (t) => t * t * (3 - 2 * t);
  const h = (a, b) => hash(a * 7919 + b * 104729, 141), a = sm(u), b = sm(v);
  return (h(gx, gy) * (1 - a) + h(gx + 1, gy) * a) * (1 - b) + (h(gx, gy + 1) * (1 - a) + h(gx + 1, gy + 1) * a) * b;
}

const MID = (month) => { const r = (((month + 1) % 12) + 0.5) % 3; return r > TRANSITION / 2 && r < 3 - TRANSITION / 2; };
function tileTime(month, x, y) {                    // the tile's own months since the start of winter, in [0, 12)
  const t = ((month + 1) % 12) + 0.5;
  const k = 0.65 * noise(x + 300, y + 300) + 0.35 * hash((y + 300) * 1009 + x + 300, 142);
  return (t + (k - 0.5) * TRANSITION + 12) % 12;
}

// the season tile (x, y) shows in game month `month` (ground, trees, roofs); `season` is the frame's (null: seasons off)
export function seasonAt(month, x, y, season) {
  if (!season || season === "winter" || month == null || MID(month)) return season;
  const s = SEASONS[Math.floor(tileTime(month, x, y) / 3)];
  return s === "winter" ? season : s;              // no early snow, no late snow
}
