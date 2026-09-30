"""Role-scoped views of the central city state.

`build_observation` returns a JSON-serialisable dict (for RL, remote agents
and logging); `render_text` turns it into the prompt an LLM agent reads.
Every role sees the same city summary; each role gets a detailed section on
its own system (facilities, loads, hotspots with coordinates, candidate
sites) and only summaries of the others, so agents must talk on the council.
"""
from __future__ import annotations

import numpy as np

from ..sim import catalog as C
from ..sim.grid import box_sum, shift
from ..sim.world import World
from .actions import action_docs
from .roles import ROLES

MAP_LEGEND = (
    "Legend: . empty land  ~ river  # road  = bridge  H highway (outside connection)  T town hall  "
    "r/c/i/o/m zoned but empty  R/C/I/O/M developed zone (O = offices, M = mixed use)  P park  E coal  G gas  S solar  W wind  "
    "U water pump  V water tower  % sewage outlet  * treatment plant  K police  J jail  F fire station  A clinic  "
    "L landfill  X incinerator  Z cemetery  N elementary school  B high school  Q university (2x2)  "
    "D library  h hospital (2x2)  : metro station  ? casino (2x2)  > airport (3x3)  @ train station  & harbour  ^ shelter  | levee  Y hydro dam  + plaza  $ stadium  ! monument  "
    "u nuclear plant  z crematorium (higher tiers keep the letter).  "
    "x = column (left to right), y = row (top to bottom)."
)


def _pct(v):
    return None if v is None else round(100 * v, 1)


def _cov(v) -> str:
    return "n/a (no buildings yet)" if v is None else f"{v}%"


def _sites(lst, noun: str = "") -> str:
    if not lst:
        return "none"
    return ", ".join(f"({s['x']},{s['y']})" + (f" homes={s['homes_within_5']}" if "homes_within_5" in s else "")
                     + (f" need={s['need']}" if "need" in s else "")
                     + (f" pollution={s['pollution']}" if "pollution" in s else "")
                     + _crowd_note(s, noun) for s in lst)


def _crowd_note(site: dict, noun: str) -> str:
    n = site.get("crowd", 0)
    return f" near {n} {noun}{'s' if n > 1 else ''}" if n else ""


def _crowd_grid(world: World, line: str) -> np.ndarray:
    """Per tile, how many buildings (any tier) of `line`'s crowd group a new one there would crowd: the tiles whose
    footprint gap (Chebyshev, as in World._crowd_factors) to a building is within the group radius."""
    g = next(g for g in C.CROWD_GROUPS.values() if line in g["lines"])
    r, n = g["radius"], np.zeros(world.struct.shape, dtype=int)
    for st in {t["struct"] for ln in g["lines"] for t in C.TIER_LINES[ln]}:
        for x, y in world.facilities(st):
            fw, fh = world.footprint_at(x, y)
            n[max(0, y - r):y + fh + r, max(0, x - r):x + fw + r] += 1
    return n


def _uncrowded_first(world: World, sites: list[dict], line: str, k: int = 6) -> list[dict]:
    """The first k sites with uncrowded ones first (existing order kept), then by neighbour count; crowded ones get
    "crowd": n. Nothing is dropped unless there are more than k sites."""
    n = _crowd_grid(world, line)
    ranked = sorted(sites, key=lambda s: n[s["y"], s["x"]])[:k]    # stable: keeps the order among equals
    return [dict(s, crowd=int(n[s["y"], s["x"]])) if n[s["y"], s["x"]] else s for s in ranked]  # copies: lists share dicts


def _candidate_sites(world: World) -> dict:
    """Helpful pre-computed placements for the utilities role."""
    s = world.struct
    empty = (s == C.NONE) & (world.terrain == C.LAND)
    res = np.isin(s, C.HOMES).astype(float)
    homes_near = box_sum(res, 5)
    road = world.road_labels >= 0
    near_road = np.zeros_like(road)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                near_road |= shift(road, dy, dx, False)
    ok = empty & near_road
    ys, xs = np.nonzero(ok)
    sites = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        net = world.adjacent_network(x, y)
        if net >= 0 and world.networks[net].outside_connected:
            sites.append((int(homes_near[y, x]), x, y))
    far = sorted(sites)[:6]
    near = sorted(sites, key=lambda t: -t[0])[:6]
    banks = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        if world.is_riverbank(x, y) and world.adjacent_network(x, y) >= 0:
            banks.append({"x": x, "y": y, "pollution": round(float(max(world.pollution[y, x], world._river_near(x, y))), 2),
                          "network": world.adjacent_network(x, y)})
    outlets = [(x, y) for x, y in world.facilities(C.OUTLET) + world.facilities(C.TREATMENT)]
    pumps = world.facilities(C.PUMP)
    # pumps: clean water, upstream (north) of every outlet; sewage: downstream (south) of every pump
    # (crowded spots last in each list: catalog.CROWD_GROUPS)
    for_pumps = _uncrowded_first(world, sorted(banks, key=lambda b: (
        b["pollution"] + (0.5 if any(oy <= b["y"] for _, oy in outlets) else 0), b["y"])), "pump")
    for_sewage = _uncrowded_first(world, sorted(banks, key=lambda b: (any(py >= b["y"] for _, py in pumps), -b["y"])),
                                  "outlet")
    windy = _uncrowded_first(world, [{"x": x, "y": y, "wind": round(float(world.wind[y, x]), 2)}
                                     for h, x, y in sorted(sites, key=lambda t: -world.wind[t[2], t[1]])], "wind")
    out = {
        "far_from_homes": [{"x": x, "y": y, "homes_within_5": h} for h, x, y in far],
        "near_homes": [{"x": x, "y": y, "homes_within_5": h} for h, x, y in near],
        "windy": windy,
        "riverbank_next_to_road": for_pumps,
        "riverbank_for_sewage": for_sewage,
    }
    if world.unlocked("water_works") or world.unlocked("advanced_treatment") or world.unlocked("nuclear"):
        out["riverbank_2x2"] = world.site_candidates(np.ones(world.struct.shape), radius=1, k=3, riverbank=True, size=2)
    return out


def _river_summary(world: World) -> str:
    r = world.river_row
    n = len(r)
    parts = [(f"rows {a}-{b - 1}", float(r[a:b].mean())) for a, b in ((0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n))]
    return ", ".join(f"{k} {100 * v:.0f}%" for k, v in parts)


def _sample(mask: np.ndarray, n: int = 8) -> list[list[int]]:
    ys, xs = np.nonzero(mask)
    idx = np.linspace(0, len(ys) - 1, min(n, len(ys))).astype(int) if len(ys) else []
    return [[int(xs[i]), int(ys[i])] for i in idx]


def _residents(world: World) -> np.ndarray:
    return world.residents_grid()


def _far_from_homes(world: World) -> np.ndarray:
    """Tiles with many homes within 4 tiles (landfills / jails should avoid them)."""
    return box_sum(np.isin(world.struct, C.HOMES), 4) >= 3


# ------------------------------------------------------ role sections
def _road_runs(world: World, k: int = 14, min_len: int = 3) -> list[dict]:
    """The road network as straight segments (longest first), so agents need not read the map."""
    road = (world.road_labels >= 0)
    runs = []
    for y in range(world.h):                      # horizontal
        x = 0
        while x < world.w:
            if road[y, x]:
                x0 = x
                while x + 1 < world.w and road[y, x + 1]:
                    x += 1
                if x - x0 + 1 >= min_len:
                    runs.append((x - x0 + 1, x0, y, x, y))
            x += 1
    for x in range(world.w):                      # vertical
        y = 0
        while y < world.h:
            if road[y, x]:
                y0 = y
                while y + 1 < world.h and road[y + 1, x]:
                    y += 1
                if y - y0 + 1 >= min_len:
                    runs.append((y - y0 + 1, x, y0, x, y))
            y += 1
    runs.sort(reverse=True)
    out = []
    for n, x1, y1, x2, y2 in runs[:k]:
        net = int(world.road_labels[y1, x1])
        out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "tiles": n,
                    "connected": bool(net >= 0 and world.networks[net].outside_connected)})
    return out


def _zone_suggestions(world: World, per_type: int = 2) -> dict:
    """Empty strips along connected roads, ranked per zone type (ready-to-use zone rectangles)."""
    s = world.struct
    free = (s == C.NONE) & (world.terrain == C.LAND) & world.access
    homes = box_sum(np.isin(s, C.HOMES), 4)
    ind = box_sum(s == C.IND, 4)
    tx, ty = world.townhall
    cands = []
    for r in _road_runs(world, k=40, min_len=4):
        if not r["connected"]:
            continue
        horiz = r["y1"] == r["y2"]
        for side in (-1, 1):
            for off in ((1, 2),):
                if horiz:
                    ya, yb = sorted((r["y1"] + side * off[0], r["y1"] + side * off[1]))
                    xa, xb = r["x1"], r["x2"]
                else:
                    xa, xb = sorted((r["x1"] + side * off[0], r["x1"] + side * off[1]))
                    ya, yb = r["y1"], r["y2"]
                if not (0 <= xa and xb < world.w and 0 <= ya and yb < world.h):
                    continue
                cols = free[ya:yb + 1, xa:xb + 1]
                if cols.size == 0:
                    continue
                # trim to the longest mostly-free stretch
                ok = cols.all(0) if horiz else cols.all(1)
                best, cur, start, bstart = 0, 0, 0, 0
                for i, v in enumerate(ok):
                    cur = cur + 1 if v else 0
                    if cur == 1:
                        start = i
                    if cur > best:
                        best, bstart = cur, start
                if best < 3:
                    continue
                best = min(best, 150 // 2)
                if horiz:
                    rect = (xa + bstart, ya, xa + bstart + best - 1, yb)
                else:
                    rect = (xa, ya + bstart, xb, ya + bstart + best - 1)
                x1, y1, x2, y2 = rect
                sl = (slice(y1, y2 + 1), slice(x1, x2 + 1))
                cands.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "tiles": int(free[sl].sum()),
                              "pollution": float(world.pollution[sl].mean()), "land_value": float(world.land_value[sl].mean()),
                              "homes_near": float(homes[sl].mean()), "industry_near": float(ind[sl].mean()),
                              "centre_dist": abs((x1 + x2) / 2 - tx) + abs((y1 + y2) / 2 - ty)})
    score = {
        "R": lambda c: c["land_value"] - c["pollution"] - 0.05 * c["industry_near"] + 0.01 * c["tiles"],
        "C": lambda c: -0.03 * c["centre_dist"] + 0.02 * c["homes_near"] + 0.01 * c["tiles"],
        "I": lambda c: -0.1 * c["homes_near"] + 0.02 * c["industry_near"] + 0.01 * c["tiles"] - 0.01 * c["centre_dist"] * 0,
        "O": lambda c: c["land_value"] - 0.02 * c["centre_dist"] + 0.01 * c["tiles"],
    }
    out = {}
    for z, f in score.items():
        used, picks = [], []
        for c in sorted(cands, key=f, reverse=True):
            if any(not (c["x2"] < u["x1"] or c["x1"] > u["x2"] or c["y2"] < u["y1"] or c["y1"] > u["y2"]) for u in used):
                continue
            picks.append({k: c[k] for k in ("x1", "y1", "x2", "y2", "tiles")})
            used.append(c)
            if len(picks) >= per_type:
                break
        out[z] = picks
    return out


def _road_extensions(world: World, k: int = 4, max_len: int = 15) -> list[dict]:
    """Straight extensions from dead ends of the connected network into open land."""
    road = world.road_labels >= 0
    s = world.struct
    out = []
    for y, x in zip(*np.nonzero(road)):
        net = int(world.road_labels[y, x])
        if not world.networks[net].outside_connected or s[y, x] == C.HIGHWAY:
            continue
        nbrs = [(dx, dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if world.in_bounds(x + dx, y + dy) and road[y + dy, x + dx]]
        if len(nbrs) != 1:
            continue
        dx, dy = -nbrs[0][0], -nbrs[0][1]          # keep going away from the road behind
        n, water, opened = 0, 0, 0
        cx, cy = x, y
        while n < max_len:
            nx, ny = cx + dx, cy + dy
            if not world.in_bounds(nx, ny) or s[ny, nx] != C.NONE:
                break
            water += int(world.terrain[ny, nx] == C.WATER)
            cx, cy, n = nx, ny, n + 1
        if n >= 3:
            band = (s == C.NONE) & (world.terrain == C.LAND) & ~world.access
            y1, y2, x1, x2 = sorted((y, cy)) + sorted((x, cx))
            opened = int(band[max(0, y1 - 2):y2 + 3, max(0, x1 - 2):x2 + 3].sum())
            out.append({"x1": int(x), "y1": int(y), "x2": int(cx), "y2": int(cy), "tiles": n, "bridge_tiles": water,
                        "opens_land": opened})
    out.sort(key=lambda r: -r["opens_land"])
    return out[:k]


def _resource_spots(world: World) -> dict:
    """Where the natural resources are (industry zoned on them specialises)."""
    from ..sim.economy import RESOURCE_NAMES
    out = {}
    for k in range(1, 5):
        m = (world.resource == k) & (world.resource_left > 0.1) & (world.struct == C.NONE)
        if m.any():
            out[RESOURCE_NAMES[k]] = [[h["x"], h["y"]] for h in world.hotspots(m.astype(float), k=3, block=8)]
    return out


def _police_section(world: World, site_hints: bool) -> dict:
    people = world._people_grid()
    out = {
        "stations": world.service_info.get("police_station", []),
        "jails": world.service_info.get("jail", []),
        "inmates": round(world.stats.get("inmates", 0.0)), "jail_capacity": world.stats.get("jail_capacity", 0),
        "crime_hotspots": world.hotspots(world.crime * people),
    }
    gap = world.downtown_gap.get("police")
    out["downtown_uncovered"] = int(gap.sum()) if gap is not None else 0
    stations = world.upgrade_options([C.POLICE], need=people * gap if gap is not None else None, k=3, per_pair=False)
    over = out["inmates"] - out["jail_capacity"]
    jails = world.upgrade_options([C.JAIL], need={xy: over for xy in world.facilities(C.JAIL)}, k=1, min_need=0,
                                  per_pair=False)
    # one list by need (people x downtown gap for stations, inmate overflow for jails); an overcrowded jail stays in
    out["upgrades"] = sorted(stations[:3 - len(jails)] + jails, key=lambda o: -o.get("need", 0.0))
    if site_hints:
        out["candidate_sites"] = world.site_candidates(people * (1.0 - world.police_cov) * (0.3 + world.crime), radius=6)
        out["jail_sites"] = world.site_candidates(np.ones_like(people), radius=1, k=3, exclude=_far_from_homes(world))
    return out


def _fire_section(world: World, site_hints: bool) -> dict:
    dev = (np.isin(world.struct, C.ZONES) & (world.level > 0)).astype(float)
    risk_left = world.fire_risk * (1.0 - world.fire_cov)
    recent = world.history[-12:]
    out = {
        "stations": world.service_info.get("fire_station", []),
        "responses_per_station": C.SERVICES["fire_station"]["responses"],     # tier 1; kept for compatibility
        "callouts_per_month": int(sum(world.facility_spec(x, y)["responses"] for x, y in world.facilities(C.FIRE))
                                  * world.funding_eff("fire")),               # as _fire_events counts them
        "fires_last_12_months": sum(h.get("fires", 0) for h in recent),
        "buildings_lost_last_12_months": sum(h.get("fire_tiles_lost", 0) for h in recent),
        "uncovered_buildings": int((dev * (world.fire_cov < 0.3)).sum()),
        "risk_hotspots": world.hotspots(risk_left),
    }
    gap = world.downtown_gap.get("fire")
    out["downtown_uncovered"] = int(gap.sum()) if gap is not None else 0
    out["upgrades"] = world.upgrade_options([C.FIRE], need=dev * gap if gap is not None else None, k=3, per_pair=False)
    nxt = next((e for e in world.schedule if e["kind"] == "flood" and 0 <= e["month"] - world.month <= 3), None)
    out["flood_warning"] = None if nxt is None else {"rows": [nxt["y1"], nxt["y2"]], "in_months": nxt["month"] - world.month}
    out["levee_tiles"] = int((world.struct == C.LEVEE).sum())
    out["shelters"] = world.service_info.get("shelter", [])
    if site_hints:
        out["candidate_sites"] = world.site_candidates(dev * (0.2 + world.fire_risk) * (1.0 - world.fire_cov), radius=6)
        if nxt is not None:
            out["levee_lines"] = _levee_lines(world, nxt)
        res = _residents(world)
        out["shelter_sites"] = world.site_candidates(res * (1.0 - world.shelter_cov), radius=5, k=3)
    return out


def _levee_lines(world: World, e: dict, k: int = 6) -> list[dict]:
    """Vertical runs of empty riverbank tiles in the flood band that protect buildings."""
    zone = world.flood_zone(e)
    dev = zone & np.isin(world.struct, C.ZONES) & (world.level > 0)
    from ..sim.grid import dilate
    near = dilate(dev, C.LEVEE_PROTECT)
    ok = (world.terrain == C.LAND) & (world.struct == C.NONE) & (world.dist_water == 1) & near
    runs = []
    for x in range(world.w):
        col = ok[:, x]
        y = 0
        while y < world.h:
            if col[y]:
                y0 = y
                while y + 1 < world.h and col[y + 1]:
                    y += 1
                runs.append({"x1": x, "y1": y0, "x2": x, "y2": y, "tiles": y - y0 + 1})
            y += 1
    runs.sort(key=lambda r: -r["tiles"])
    return runs[:k]


GARBAGE_FILL_OFFER = 50      # landfills more than this % full get a recycling-centre offer
GARBAGE_BUSY_LOAD = 0.85     # ... and so do landfills whose trucks carry at least this share of their capacity
GARBAGE_BUSY_RANK = 100      # busy landfills rank above filling ones (need = this + load %)


def _health_section(world: World, site_hints: bool) -> dict:
    res = _residents(world)
    st = world.stats
    out = {
        "clinics": world.service_info.get("clinic", []), "hospitals": world.service_info.get("hospital", []),
        "landfills": world.service_info.get("landfill", []), "incinerators": world.service_info.get("incinerator", []),
        "cemeteries": world.service_info.get("cemetery", []),
        "garbage_produced": round(st.get("garbage_produced", 0.0)),
        "garbage_collected": round(st.get("garbage_collected", 0.0)),
        "garbage_uncollected_stock": round(st.get("garbage_backlog", 0.0)),
        "deaths_per_month": round(st.get("deaths", 0.0), 1),
        "deathcare_capacity": round(st.get("deathcare_capacity", 0.0), 1),
        "deceased_waiting": round(st.get("death_backlog", 0.0)),
        "garbage_hotspots": world.hotspots(world.garbage),
        "soil_hotspots": world.hotspots(world.soil_level() * (world.soil_level() > 0.15)),
        "low_health_hotspots": world.hotspots(res * (1.0 - world.health)),
    }
    gap = world.downtown_gap.get("health")
    out["downtown_uncovered"] = int(gap.sum()) if gap is not None else 0
    # landfill need: fill % (offered above GARBAGE_FILL_OFFER), or GARBAGE_BUSY_RANK + load % when the trucks are busy
    # (load >= GARBAGE_BUSY_LOAD x collection capacity), so busy landfills rank above filling ones
    fills = {}
    for l in world.service_info.get("landfill", []):
        need = l.get("fill_pct", 0.0)
        if l.get("capacity") and l.get("load", 0.0) >= GARBAGE_BUSY_LOAD * l["capacity"]:
            need = max(need, GARBAGE_BUSY_RANK + 100 * l["load"] / l["capacity"])
        fills[(l["x"], l["y"])] = need
    plots = {(c["x"], c["y"]): 100 - 100 * c["plots_left"] / C.SERVICES["cemetery"]["plots"]
             for c in world.service_info.get("cemetery", [])}
    out["crematoria"] = world.service_info.get("crematorium", [])
    out["upgrades"] = {
        "care": world.upgrade_options([C.CLINIC, C.HOSPITAL], need=res * gap if gap is not None else None, k=2,
                                      per_pair=False),
        "garbage": world.upgrade_options([C.LANDFILL], need=fills, k=3, min_need=GARBAGE_FILL_OFFER, per_pair=False),
        "deathcare": world.upgrade_options([C.CEMETERY], need=plots, k=2,
                                           min_need=None if out["deceased_waiting"] > 0 else 70, per_pair=False),
    }
    if site_hints:
        out["clinic_sites"] = world.site_candidates(res * (1.0 - world.health_cov), radius=5)
        out["hospital_sites"] = world.site_candidates(res * (1.0 - world.health_cov) + 0.01, radius=8, k=3, size=2)
        prod = world._garbage_production() + world.garbage
        out["landfill_sites"] = world.site_candidates(prod * (1.0 - world.garbage_cov) + 0.01, radius=10, k=4,
                                                      exclude=_far_from_homes(world))
    return out


def _education_section(world: World, site_hints: bool) -> dict:
    res = _residents(world)
    n = world.cohorts
    total = max(1.0, float(n.sum()))
    st = world.stats
    out = {
        "population": world.cohort_summary(),
        "elementary": {"students": int(n[0].sum()), "seats": int(st.get("seats_elementary", 0)),
                       "homes_in_range_pct": _pct(st.get("school_coverage", 0.0))},
        "high": {"students": int(n[1].sum()), "seats": int(st.get("seats_high", 0)),
                 "homes_in_range_pct": _pct(st.get("high_coverage", 0.0))},
        "university": {"students": int(n[2, 2] * C.STUDENT_AGE_SHARE), "seats": int(st.get("seats_university", 0))},
        "enrollment_pct": {k: _pct(v) for k, v in world.enrollment.items()},
        "skilled_jobs_filled_pct": _pct(st.get("skilled_fill", 1.0)),
        "schools": world.service_info.get("elementary_school", []),
        "high_schools": world.service_info.get("high_school", []),
        "libraries": world.service_info.get("library", []),
    }
    over = {(f["x"], f["y"]): f["load"] - f["capacity"]
            for f in world.service_info.get("elementary_school", []) + world.service_info.get("high_school", [])}
    out["upgrades"] = world.upgrade_options([C.SCHOOL, C.HIGHSCHOOL], need=over, k=3, min_need=0, per_pair=False)
    if site_hints:
        out["school_sites"] = world.site_candidates(res * n[0].sum() / total * (1.0 - world.school_cov), radius=6)
        out["high_school_sites"] = world.site_candidates(res * n[1].sum() / total * (1.0 - world.high_cov), radius=8, k=3)
        out["university_sites"] = world.site_candidates(res + 0.01, radius=8, k=3, size=2)
    return out


def _congested_runs(world: World, k: int = 5, thresh: float = 0.8) -> list[dict]:
    """Straight stretches of road that carry more than `thresh` x their capacity."""
    cong = world.congestion
    road = world.struct == C.ROAD
    hot = road & (cong > thresh)
    used = np.zeros_like(hot)
    ys, xs = np.nonzero(hot)
    order = np.argsort(-cong[ys, xs])
    runs = []
    for i in order.tolist():
        x, y = int(xs[i]), int(ys[i])
        if used[y, x]:
            continue
        best = None
        for dx, dy in ((1, 0), (0, 1)):
            a = b = 0
            while hot[y - (a + 1) * dy, x - (a + 1) * dx] if world.in_bounds(x - (a + 1) * dx, y - (a + 1) * dy) else False:
                a += 1
            while hot[y + (b + 1) * dy, x + (b + 1) * dx] if world.in_bounds(x + (b + 1) * dx, y + (b + 1) * dy) else False:
                b += 1
            if best is None or a + b > best[0]:
                best = (a + b, (x - a * dx, y - a * dy, x + b * dx, y + b * dy))
        x1, y1, x2, y2 = best[1]
        seg = cong[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1]
        used[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1] = True
        cls = {v: k for k, v in C.ROAD_CLASSES.items()}[int(world.road_class[y, x])]
        runs.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "tiles": int(seg.size),
                     "load_vs_capacity": round(float(seg.max()), 2), "class": cls})
        if len(runs) >= k:
            break
    return runs


def _transport_section(world: World, site_hints: bool) -> dict:
    st = world.stats
    out = {
        "commute_minutes": round(st.get("commute_minutes", 0.0), 1),
        "congested_road_share_pct": _pct(st.get("congestion_index", 0.0)),
        "commuters": round(st.get("trips", 0.0)), "bus_riders": round(st.get("transit_riders", 0.0)),
        "fare": world.fare, "free_transit": world.policy_on("free_transit"),
        "roads_by_class": {k: int((world.road_class == v).sum()) for k, v in C.ROAD_CLASSES.items()},
        "bus_lines": [{"id": l["id"], "stops": [list(p) for p in l["stops"]], "riders": l.get("riders", 0),
                       "capacity": l.get("capacity", C.LINE_CAPACITY)} for l in world.bus_lines],
        "metro_stations": [[x, y] for x, y in world.facilities(C.METRO)],
        "metro_lines": [{"id": l["id"], "stations": [list(p) for p in l["stations"]]} for l in world.metro_lines],
        "metro_riders": round(st.get("metro_riders", 0.0)),
        "job_centres": [{"x": h["x"], "y": h["y"], "jobs": round(h["jobs"]), "kind": h["kind"]} for h in world.hubs],
        "congested_roads": _congested_runs(world),
        "train_stations": [{"x": x, "y": y} for x, y in world.facilities(C.TRAIN)],
        "harbours": [{"x": x, "y": y} for x, y in world.facilities(C.HARBOUR)],
        "freight": {"exports": round(world.trade.get("exports", 0.0)), "imports": round(world.trade.get("imports", 0.0)),
                    "capacity": round(world.trade.get("freight_capacity", 0.0))},
    }
    if site_hints:
        out["suggested_stops"] = _suggest_stops(world)
        out["train_station_sites"] = _edge_sites(world)
        out["harbour_sites"] = _harbour_sites(world)
        if world.region_info:
            out["connector_sites"] = _connector_sites(world)
        if world.unlocked("metro"):
            dense = world.residents_grid() + world.shop_jobs_grid() + world._people_grid() * 0.5
            out["metro_station_sites"] = world.site_candidates(dense * (1.0 - world.metro_cov), radius=C.METRO_RADIUS,
                                                               k=4, spacing=8)
    return out


def _suggest_stops(world: World, n_homes: int = 4, n_jobs: int = 3, avoid: np.ndarray | None = None) -> list[list[int]]:
    """Road tiles in the biggest residential areas and at the biggest job centres (main network); with `avoid` (a
    tile mask, e.g. the bus lines' coverage) only homes outside it count (the fleet's next line: tools.new_line_stops)."""
    res = _residents(world)
    anchor = world._anchors()
    main = max(range(len(world.networks)), key=lambda i: (world.networks[i].outside_connected,
                                                           world.networks[i].road_tiles)) if world.networks else -1
    ok = anchor >= 0
    mass = np.bincount(anchor[ok].ravel(), weights=res[ok].ravel(), minlength=world.h * world.w).reshape(world.h, world.w)
    if avoid is not None:
        mass = mass * ~avoid
    mass = box_sum(mass, 3) * (world.road_labels == main)
    stops = []
    for h in sorted([h for h in world.hubs if h["kind"] == "jobs"], key=lambda h: -h["jobs"])[:n_jobs]:
        if world.road_labels[h["y"], h["x"]] == main:
            stops.append([h["x"], h["y"]])
    ys, xs = np.nonzero(mass > 0)
    for i in np.argsort(-mass[ys, xs]).tolist():
        x, y = int(xs[i]), int(ys[i])
        if all(abs(x - a) + abs(y - b) >= 7 for a, b in stops):
            stops.append([x, y])
        if len(stops) >= n_jobs + n_homes:
            break
    return stops


def _connector_sites(world: World) -> dict:
    """Cheapest edge tile towards each neighbouring city (shortest feeder road)."""
    from .actions import connector_feeder
    out = {}
    for city, side in world.region_info["neighbors"].items():
        edge = {"north": [(x, 0) for x in range(world.w)], "south": [(x, world.h - 1) for x in range(world.w)],
                "west": [(0, y) for y in range(world.h)], "east": [(world.w - 1, y) for y in range(world.h)]}[side]
        best = None
        for x, y in edge:
            if world.terrain[y, x] != C.LAND or world.struct[y, x] != C.NONE:
                continue
            f = connector_feeder(world, side, x, y)
            if f is not None and (best is None or len(f) < best[0]):
                best = (len(f), x, y)
        built = sorted({c["kind"] for c in world.connectors if c["to"] == city})
        out[city] = {"side": side, "built": built,
                     "site": None if best is None else {"x": best[1], "y": best[2], "feeder_tiles": best[0]}}
    return out


def _harbour_sites(world: World, k: int = 3) -> list[dict]:
    out = []
    for y in range(world.h - 1, world.h - 1 - C.HARBOUR_ROWS, -1):
        for x in range(world.w):
            if world.struct[y, x] == C.NONE and world.is_riverbank(x, y):
                net = world.adjacent_network(x, y)
                if net >= 0 and world.networks[net].outside_connected:
                    out.append({"x": x, "y": y})
                    if len(out) >= k:
                        return out
    return out


def _edge_sites(world: World, k: int = 3) -> list[dict]:
    out = []
    s = world.struct
    for x, y in [(x, y) for y in range(world.h) for x in (0, world.w - 1)] + \
                [(x, y) for x in range(1, world.w - 1) for y in (0, world.h - 1)]:
        if s[y, x] == C.NONE and world.terrain[y, x] == C.LAND:
            net = world.adjacent_network(x, y)
            if net >= 0 and world.networks[net].outside_connected:
                out.append({"x": x, "y": y})
                if len(out) >= k:
                    break
    return out


ROLE_SECTIONS = {"transport": _transport_section, "police": _police_section, "fire": _fire_section, "health": _health_section,
                 "education": _education_section}


def build_observation(world: World, role: str, *, council_msgs: list[dict], memory: dict,
                      scores: dict, months_total: int, include_map: bool = True,
                      site_hints: bool = True) -> dict:
    spec = ROLES[role]
    docs = action_docs()
    m = world.metrics()                                   # live (reflects actions this month)
    lm = world.history[-1] if world.history else m        # last completed month (money flows)
    trend = [{"month": h["month"], "population": h["population"], "treasury": round(h["treasury"]),
              "happiness": round(h["happiness"], 2), "income": round(h["income"]),
              "expenses": round(h["expenses"])} for h in world.history[-6:]]
    obs = {
        "role": role,
        "title": spec["title"],
        "mission": spec["mission"],
        "month": world.month,
        "date": f"Year {world.month // 12 + 1}, month {world.month % 12 + 1}"
                + (f", {world.season()}" if world.seasons_on else ""),
        "wind": f"blowing toward the {world.wind_name()} (air pollution drifts that way)",
        "months_left": months_total - world.month,
        "city": {
            "population": m["population"], "jobs_commercial": m["jobs_com"], "jobs_industrial": m["jobs_ind"],
            "workers": m["workers"], "unemployment_pct": _pct(m["unemployment"]),
            "happiness_pct": _pct(m["happiness"]),
            "demand": world.demand and {k: round(v, 2) for k, v in world.demand.items()},
            "tax_pct": {k: round(100 * v, 1) for k, v in world.taxes.items()},
            "power": {"supply": m["power_supply"], "demand": m["power_demand"],
                      "coverage_pct": _pct(m["power_coverage"])},
            "water": {"supply": m["water_supply"], "demand": m["water_demand"],
                      "coverage_pct": _pct(m["water_coverage"])},
            "sewage": {"supply": m["sewage_supply"], "demand": m["sewage_demand"],
                       "coverage_pct": _pct(m["sewage_coverage"])},
            "river_pollution_pct": _pct(m["river_pollution"]),
            "approval_pct": _pct(m["approval"]),
            "trade": {"exports": m["exports"], "imports": m["imports"], "freight_capacity": m["freight_capacity"],
                      "goods_shortage_pct": _pct(m["goods_shortage"]), "businesses_losing_money": m["businesses_losing"],
                      "business_closures_last_month": m["business_closures"]},
            "commute_minutes": m["commute_minutes"], "congested_road_pct": _pct(m["congestion_index"]),
            "complaints": [c["text"] for c in m["complaints"]],
            "pollution_at_homes_pct": _pct(m["pollution_res"]),
            "avg_land_value_pct": _pct(m["land_value_dev"]),
            "zoned_tiles": m["zoned"], "developed_tiles": m["developed"],
            "zoned_without_road_access": m["zoned_no_access"],
            "zoned_without_power": m["zoned_no_power"],
            "zoned_without_water": m["zoned_no_water"],
            "zoned_without_sewage": m["zoned_no_sewage"],
            "jobs_offices": m["jobs_off"],
            "demand_caps": m["demand_caps"],
            "road_tiles": m["road_tiles"], "parks": m["parks"],
            "services": {
                "crime_at_homes_pct": _pct(m["crime_res"]), "police_coverage_pct": _pct(m["police_coverage"]),
                "fire_coverage_pct": _pct(m["fire_coverage"]), "fires_last_month": m["fires"],
                "health_pct": _pct(m["health"]), "health_coverage_pct": _pct(m["health_coverage"]),
                "garbage_collection_pct": _pct(m["garbage_coverage"]),
                "homes_with_garbage_piling_pct": _pct(m["garbage_homes_pct"]),
                "school_coverage_pct": _pct(m["school_coverage"]),
                "adults_high_school_plus_pct": _pct(m["education_index"]),
            },
        },
        "finance": {
            "treasury": round(world.treasury, 2),
            "department_funds": {k: round(v, 2) for k, v in world.dept_funds.items()},
            "debt": round(world.debt, 2),
            "last_month_income": lm["income_breakdown"],
            "last_month_expenses": lm["expense_breakdown"],
            "last_month_department_spending": lm["dept_spent"],
            "months_with_negative_treasury": world.months_negative,
        },
        "your_budget": round(world.dept_funds[spec["budget"]], 2) if spec["budget"] else None,
        "scores": {"city": round(scores.get("city", 0), 1), "yours": round(scores.get(role, 0), 1)},
        "trend": trend,
        "events": world.last_events + world.events,
        "council": council_msgs,
        "last_turn": memory.get("last_results", []),
        "notes": memory.get("notes", ""),
        "actions": {a: {"example": docs[a][0], "help": docs[a][1]} for a in spec["actions"]},
        "progress": {"milestones_reached": list(world.milestones_reached), "next_milestone": world.next_milestone(),
                     "locked": world.locked_now()},
        "forecast": world.forecast(),
        "offers": [dict(o) for o in world.offers],
    }
    if role == "utilities":
        obs["networks"] = [n.to_dict() for n in world.networks if n.road_tiles > 1 or n.plants]
        obs["upgrades"] = world.upgrade_options(C.UTILITY_TYPES)
        obs["disconnected_facilities"] = world.disconnected_plants
        zone_mask = np.isin(world.struct, C.ZONES)
        obs["sample_tiles_needing_power"] = _sample(zone_mask & world.access & ~world.powered)
        obs["sample_tiles_needing_water"] = _sample(zone_mask & world.access & world.powered & ~world.watered)
        obs["sample_tiles_needing_sewage"] = _sample(zone_mask & world.access & world.watered & ~world.sewered)
        obs["river"] = _river_summary(world)
        if site_hints:
            obs["candidate_sites"] = _candidate_sites(world)
    if role == "planning":
        zone_mask = np.isin(world.struct, C.ZONES)
        empty_access = (world.struct == C.NONE) & (world.terrain == C.LAND) & world.access
        obs["empty_land_with_road_access"] = int(empty_access.sum())
        obs["sample_zoned_tiles_without_access"] = _sample(zone_mask & ~world.access)
        obs["levels"] = {L: [int(((world.struct == z) & (world.level == k)).sum()) for k in range(C.MAX_LEVEL + 1)]
                         for L, z in C.ZONE_BY_LETTER.items()}
        obs["high_density_zoned"] = m["density_high"]
        obs["historic"] = {"listed_tiles": m["historic_tiles"], "max": C.HISTORIC_MAX_TILES}
        obs["homes_by_wealth"] = m["wealth"]
        obs["industry_by_type"] = {k: v for k, v in m["industry_types"].items() if v}
        obs["business_margins"] = {"commercial": m["margins"]["com"], "industrial": m["margins"]["ind"],
                                   "offices": m["margins"]["off"]}
        obs["households"] = {"income_per_resident": round(m["income_per_resident"], 2),
                             "disposable_per_resident": round(m["disposable_income"], 2),
                             "homeless_pct": _pct(m["homeless_share"])}
        obs["natural_resources"] = _resource_spots(world)
        if site_hints and world.unlocked("casino"):
            obs["casino_sites"] = world.site_candidates(world._people_grid() + 0.01, radius=5, k=2, size=2,
                                                        exclude=box_sum(world.struct == C.SCHOOL, 4) > 0)
        obs["road_segments"] = _road_runs(world)
        if site_hints:
            obs["zoning_suggestions"] = _zone_suggestions(world)
            obs["road_suggestions"] = _road_extensions(world)
        obs["level_rules"] = ("L1 road+power, L2 +water, L3 +sewage +garbage pickup +land value 40%, "
                              "L4 +school in reach (homes) / skilled workers (jobs) +land value 58%, "
                              "L5 +land value 75% +low crime +park nearby. Low density stops at L2; "
                              "industry stops at L3 unless high-tech.")
    if role in ("mayor", "planning"):
        obs["districts"] = world.districts_summary()
    if role == "planning":
        obs["landmarks"] = {k: int((world.struct == v["struct"]).sum()) for k, v in C.LANDMARKS.items()}
        obs["tourists"] = round(world.stats.get("tourists", 0.0))
    if role == "mayor":
        obs["recent_allocations"] = [{"month": h["month"], **{k: v for k, v in h["dept_allocated"].items() if v}}
                                     for h in world.history[-6:]]
        obs["department_upkeep"] = m["dept_upkeep"]
        obs["funding_pct"] = {k: round(100 * v) for k, v in world.funding.items()}
        obs["policies"] = {"in_force": sorted(world.policies), "monthly_cost": m["policy_cost"],
                           "available": {k: v["desc"] for k, v in C.POLICIES.items() if k not in world.policies}}
        obs["service_imports"] = {"enabled": world.service_imports, "last_month": m["service_imports"],
                                  "price": {k: round(world.service_import_price(k), 2) for k in ("power", "water")},
                                  "power_sold": m["power_sold"]}
        obs["approval"] = {"pct": _pct(world.approval), "months_below_30": world.low_approval_months,
                           "needs_pct": {k: _pct(v) for k, v in world.need_avg.items()}}
    if role in ROLE_SECTIONS:
        obs[role] = ROLE_SECTIONS[role](world, site_hints)
    if include_map:
        obs["map"] = world.ascii_map()
        obs["map_legend"] = MAP_LEGEND
    return obs


def _kv(d: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())


def _facilities(lst: list[dict], extra=("load", "capacity")) -> str:
    if not lst:
        return "none"
    out = []
    for f in lst[:10]:
        bits = [f"{k} {f[k]}" for k in extra if k in f]
        for k in ("fill_pct", "plots_left"):
            if k in f:
                bits.append(f"{k.replace('_', ' ')} {f[k]}")
        out.append((f"{f['name'].replace('_', ' ')} " if _shares_struct(f) else "")
                   + f"({f['x']},{f['y']})" + (f" [{', '.join(bits)}]" if bits else ""))
    return "; ".join(out) + (f" (+{len(lst) - 10} more)" if len(lst) > 10 else "")


def _shares_struct(f: dict) -> bool:
    """A facility above the lowest tier drawn with its letter (police hq, recycling centre, medical centre)."""
    if f.get("tier", 1) <= 1 or f.get("name") not in C.TIER_BY_NAME:
        return False
    line, t = C.TIER_BY_NAME[f["name"]]
    return t > C.BASE_TIER[C.TIER_LINES[line][t - 1]["struct"]]


def _upgrades(lst: list[dict]) -> str:
    def gain(u):
        full = u.get("nameplate", {})
        def note(k, v):          # the effective figure can beat the nameplate on a windy site
            if k not in full or full[k] == v:
                return ""
            return f", {full[k]:g} at full output" if v < full[k] else f", nameplate {full[k]:g}"
        return ", ".join(f"+{v:g} {k}" + note(k, v) for k, v in u["gain"].items())
    return "; ".join(f"({u['x']},{u['y']}) {u['from'].replace('_', ' ')} -> {u['to'].replace('_', ' ')} "
                     f"${u['cost']:,} ({gain(u)})" + (f" (+{u['similar']} similar)" if u.get("similar") else "")
                     for u in lst)


def _crowding(plants: list[dict]) -> str:
    """'0.4 x20, 0.55 x8; worst clusters near (41,19), (38,24)' for a network's crowded plants ('' if none)."""
    crowded = sorted((p for p in plants if "crowd" in p), key=lambda p: (p["crowd"], -p.get("neighbours", 0)))
    if not crowded:
        return ""
    counts: dict[float, int] = {}
    for p in crowded:
        counts[round(p["crowd"], 2)] = counts.get(round(p["crowd"], 2), 0) + 1
    worst = []
    for p in crowded:
        if all(abs(p["x"] - x) + abs(p["y"] - y) > 4 for x, y in worst):
            worst.append((p["x"], p["y"]))
        if len(worst) >= 3:
            break
    return (", ".join(f"{c:g} x{n}" for c, n in counts.items()) + "; worst clusters near "
            + ", ".join(f"({x},{y})" for x, y in worst))


def _downtown(d: dict, L: list[str], offers: bool = False) -> None:
    if d.get("downtown_uncovered"):
        L.append(f"Downtown (level 4-5) tiles without tier-2 cover: {d['downtown_uncovered']} "
                 f"(tier-1 buildings give them only {int(C.T1_DOWNTOWN * 100)}%)"
                 + (" - offers below are ranked by downtown need" if offers else ""))


def _spots(lst: list[dict]) -> str:
    return ", ".join(f"around ({h['x']},{h['y']}) {h['amount']:g}" for h in lst) or "none"


def _render_role(obs: dict, L: list[str]) -> None:
    role = obs["role"]
    d = obs.get(role)
    if not d:
        return
    L.append("")
    if role == "transport":
        L.append("## Transport")
        L.append(f"Average commute {d['commute_minutes']} min for {d['commuters']} commuters; congested road tiles "
                 f"{d['congested_road_share_pct']}%; bus riders {d['bus_riders']} (fare ${d['fare']:.2f}"
                 f"{', free transit policy' if d['free_transit'] else ''})")
        L.append(f"Roads by class: {_kv(d['roads_by_class'])}")
        L.append("Job centres (commute destinations): " + (", ".join(f"({h['x']},{h['y']}) {h['kind']} {h['jobs']}"
                                                                        for h in d["job_centres"]) or "none"))
        L.append("Congested stretches (load / capacity): " + ("; ".join(
            f"({r['x1']},{r['y1']})-({r['x2']},{r['y2']}) {r['class']} x{r['load_vs_capacity']}" for r in d["congested_roads"]) or "none"))
        L.append("Bus lines: " + ("; ".join(f"line {l['id']} stops {l['stops']} riders {l['riders']:.0f}/{l['capacity']:.0f}"
                                            for l in d["bus_lines"]) or "none"))
        L.append("Metro stations: " + (", ".join(f"({x},{y})" for x, y in d["metro_stations"]) or "none") + "; lines: "
                 + ("; ".join(f"M{l['id']} {l['stations']}" for l in d["metro_lines"]) or "none")
                 + f"; metro riders {d['metro_riders']}")
        if d.get("metro_station_sites"):
            L.append(f"Suggested metro station sites (dense, not yet served): {_sites(d['metro_station_sites'])}")
        L.append("Train stations: " + (", ".join(f"({t['x']},{t['y']})" for t in d["train_stations"]) or "none")
                 + "; harbours: " + (", ".join(f"({t['x']},{t['y']})" for t in d["harbours"]) or "none"))
        fr = d["freight"]
        L.append(f"Freight: exports {fr['exports']} + imports {fr['imports']} of capacity {fr['capacity']} per month "
                 "(highway + train stations + harbour; trucks add to road traffic)")
        if "suggested_stops" in d:
            L.append(f"Suggested bus stops (big job centres + dense homes): {d['suggested_stops']}")
            L.append(f"Train station sites (map edge next to a connected road): {_sites(d['train_station_sites'])}")
            L.append(f"Harbour sites (riverbank near the south edge): {_sites(d['harbour_sites'])}")
            for city, cs in d.get("connector_sites", {}).items():
                site = cs["site"]
                L.append(f"Connector towards {city} ({cs['side']} edge): built {cs['built'] or 'nothing'}; "
                         + (f"best site ({site['x']},{site['y']}) needing {site['feeder_tiles']} feeder road tiles"
                            if site else "no edge tile within reach of a road yet"))
    elif role == "police":
        L.append("## Police")
        L.append(f"Stations [people in area / capacity]: {_facilities(d['stations'])}")
        _downtown(d, L, bool(d.get("upgrades")))
        L.append(f"Jails: {_facilities(d['jails'], ('capacity',))}; inmates {d['inmates']} / capacity {d['jail_capacity']}"
                 + (" OVERCROWDED" if d['inmates'] > d['jail_capacity'] else ""))
        if d.get("upgrades"):
            L.append('Tier upgrades (upgrade_facility {"module":"tier"}): ' + _upgrades(d["upgrades"]))
        L.append(f"Crime hotspots (crime x people): {_spots(d['crime_hotspots'])}")
        if "candidate_sites" in d:
            L.append(f"Suggested station sites (uncovered people nearby): {_sites(d['candidate_sites'])}")
            L.append(f"Suggested jail sites (away from homes): {_sites(d['jail_sites'])}")
    elif role == "fire":
        L.append("## Fire & emergencies")
        if d.get("flood_warning"):
            fw = d["flood_warning"]
            L.append(f"FLOOD WARNING: rows {fw['rows'][0]}-{fw['rows'][1]} in {fw['in_months']} month(s). "
                     f"Levee tiles built so far: {d['levee_tiles']}")
            if d.get("levee_lines"):
                L.append("  Riverbank stretches to protect (empty bank next to buildings): " + "; ".join(
                    f"({r['x1']},{r['y1']})-({r['x2']},{r['y2']}) {r['tiles']} tiles" for r in d["levee_lines"]))
        L.append(f"Shelters: {_facilities(d['shelters'])}" + (f"; suggested shelter sites: {_sites(d['shelter_sites'])}"
                                                               if d.get("shelter_sites") else ""))
        L.append(f"Stations [buildings in area / capacity]: {_facilities(d['stations'])}; "
                 f"call-out capacity {d['callouts_per_month']} per month (all stations, at current funding)")
        _downtown(d, L, bool(d.get("upgrades")))
        if d.get("upgrades"):
            L.append('Tier upgrades (upgrade_facility {"module":"tier"}): ' + _upgrades(d["upgrades"]))
        L.append(f"Last 12 months: {d['fires_last_12_months']} fires, {d['buildings_lost_last_12_months']} buildings lost; "
                 f"buildings without fire cover: {d['uncovered_buildings']}")
        L.append(f"Uncovered fire-risk hotspots: {_spots(d['risk_hotspots'])}")
        if "candidate_sites" in d:
            L.append(f"Suggested station sites: {_sites(d['candidate_sites'])}")
    elif role == "health":
        L.append("## Health & sanitation")
        L.append(f"Clinics [residents in area / capacity]: {_facilities(d['clinics'])}")
        L.append(f"Hospitals: {_facilities(d['hospitals'])}")
        _downtown(d, L, bool(d.get("upgrades", {}).get("care")))
        L.append(f"Garbage per month: produced {d['garbage_produced']}, collected {d['garbage_collected']}; "
                 f"uncollected stock {d['garbage_uncollected_stock']}")
        L.append(f"Landfills [garbage in area / collection capacity]: {_facilities(d['landfills'])}")
        L.append(f"Incinerators: {_facilities(d['incinerators'])}")
        L.append(f"Deathcare: {d['deaths_per_month']} deaths/month, capacity {d['deathcare_capacity']}/month, "
                 f"waiting {d['deceased_waiting']}; cemeteries: {_facilities(d['cemeteries'], ())}"
                 + (f"; crematoria: {_facilities(d['crematoria'], ('rate',))}" if d.get("crematoria") else ""))
        if d.get("upgrades") and any(d["upgrades"].values()):
            L.append('Tier upgrades (upgrade_facility {"module":"tier"}): '
                     + "; ".join(f"{k}: {_upgrades(v)}" for k, v in d["upgrades"].items() if v))
        L.append(f"Garbage hotspots: {_spots(d['garbage_hotspots'])}")
        L.append(f"Contaminated soil hotspots (clean_soil): {_spots(d['soil_hotspots'])}")
        L.append(f"Low-health hotspots: {_spots(d['low_health_hotspots'])}")
        if "clinic_sites" in d:
            L.append(f"Suggested clinic sites: {_sites(d['clinic_sites'])}")
            L.append(f"Suggested landfill sites (away from homes): {_sites(d['landfill_sites'])}")
            L.append(f"Suggested hospital sites (2x2, top-left tile): {_sites(d['hospital_sites'])}")
    elif role == "education":
        L.append("## Education")
        p = d["population"]
        L.append(f"Population: children {p['children']}, teens {p['teens']}, adults {p['adults']}, seniors {p['seniors']}; "
                 f"adult education %: {_kv(p['adult_education_pct'])}")
        for k in ("elementary", "high", "university"):
            v = d[k]
            L.append(f"{k}: students {v['students']}, seats {v['seats']}"
                     + (f", homes in range {v['homes_in_range_pct']}%" if "homes_in_range_pct" in v else "")
                     + f", enrolled {d['enrollment_pct'][k]}%")
        L.append(f"Skilled jobs (high school / university) filled: {d['skilled_jobs_filled_pct']}%")
        L.append(f"Elementary schools [students in area / seats]: {_facilities(d['schools'])}")
        L.append(f"High schools: {_facilities(d['high_schools'])}; libraries: {_facilities(d['libraries'], ())}")
        if d.get("upgrades"):
            L.append('Tier upgrades (upgrade_facility {"module":"tier"}): ' + _upgrades(d["upgrades"]))
        if "school_sites" in d:
            L.append(f"Suggested elementary school sites: {_sites(d['school_sites'])}")
            L.append(f"Suggested high school sites: {_sites(d['high_school_sites'])}")
            L.append(f"Suggested university sites (2x2, top-left tile): {_sites(d['university_sites'])}")


def render_text(obs: dict) -> str:
    c, f = obs["city"], obs["finance"]
    sv = c["services"]
    L = []
    L.append(f"=== {obs['title']} | {obs['date']} (month {obs['month']}, {obs['months_left']} months left) ===")
    L.append(f"Scores: city {obs['scores']['city']}/100, your role {obs['scores']['yours']}/100. Wind {obs['wind']}")
    if obs["your_budget"] is not None:
        L.append(f"YOUR BUDGET: ${obs['your_budget']:,.0f}" +
                 ("  (you cannot build anything until the mayor allocates funds; ask on the council board)"
                  if obs["your_budget"] < 1 else ""))
    L.append("")
    L.append("## City")
    L.append(f"Population {c['population']}, workers {c['workers']}, jobs C={c['jobs_commercial']} I={c['jobs_industrial']} O={c['jobs_offices']}, "
             f"unemployment {c['unemployment_pct']}%, happiness {c['happiness_pct']}%")
    L.append(f"Demand (-1..1, positive = people/businesses want more of this zone): {_kv(c['demand'])}"
             + (f" | CAPPED: {_kv(c['demand_caps'])} (a big city needs a train station and enough parks)" if c["demand_caps"] else ""))
    L.append(f"Taxes %: {_kv(c['tax_pct'])}")
    L.append(f"Power: supply {c['power']['supply']}, demand {c['power']['demand']}, coverage {_cov(c['power']['coverage_pct'])} | "
             f"Water: supply {c['water']['supply']}, demand {c['water']['demand']}, coverage {_cov(c['water']['coverage_pct'])}")
    L.append(f"Sewage: capacity {c['sewage']['supply']}, produced {c['sewage']['demand']}, coverage {_cov(c['sewage']['coverage_pct'])} | "
             f"River pollution {c['river_pollution_pct']}% (the river flows north to south)")
    L.append(f"Pollution at homes {c['pollution_at_homes_pct']}%, avg land value {c['avg_land_value_pct']}%")
    L.append(f"Services: crime at homes {sv['crime_at_homes_pct']}% (police cover {sv['police_coverage_pct']}%), "
             f"fire cover {sv['fire_coverage_pct']}% ({sv['fires_last_month']} fires last month), "
             f"health {sv['health_pct']}% (clinic cover {sv['health_coverage_pct']}%), "
             f"garbage collection {sv['garbage_collection_pct']}% (piling up at {sv['homes_with_garbage_piling_pct']}% of homes), "
             f"school cover {sv['school_coverage_pct']}%, adults with high school+ {sv['adults_high_school_plus_pct']}%")
    L.append(f"Traffic: average commute {c['commute_minutes']:.0f} min, congested road tiles {c['congested_road_pct']}%")
    tr = c["trade"]
    L.append(f"Trade: exports {tr['exports']:.0f}, imports {tr['imports']:.0f} of freight capacity {tr['freight_capacity']:.0f}"
             f" (goods shortage {tr['goods_shortage_pct']}%); businesses losing money {tr['businesses_losing_money']}, "
             f"closed last month {tr['business_closures_last_month']}")
    L.append(f"Mayor approval {c['approval_pct']}%. Top complaints: " + ("; ".join(c["complaints"]) or "none"))
    L.append(f"Zoned tiles {_kv(c['zoned_tiles'])}; developed {_kv(c['developed_tiles'])}; roads {c['road_tiles']}; parks {c['parks']}")
    L.append(f"Zoned tiles lacking: road access {c['zoned_without_road_access']}, power {c['zoned_without_power']}, "
             f"water {c['zoned_without_water']}")
    L.append("")
    L.append("## Finance")
    L.append(f"Treasury ${f['treasury']:,.0f}; department funds {_kv(f['department_funds'])}; debt ${f['debt']:,.0f}; "
             f"months with negative treasury: {f['months_with_negative_treasury']}")
    L.append(f"Last month income: {_kv(f['last_month_income'])}")
    L.append(f"Last month expenses: {_kv(f['last_month_expenses'])}")
    L.append(f"Last month department spending: {_kv(f['last_month_department_spending'])}")
    if obs["trend"]:
        L.append("Trend: " + " | ".join(f"m{t['month']}: pop {t['population']}, ${t['treasury']:,}, net {t['income'] - t['expenses']:+,}"
                                         for t in obs["trend"]))
    if "recent_allocations" in obs:
        L.append("Your recent allocations: " + " | ".join(
            f"m{a['month']}: " + (_kv({k: v for k, v in a.items() if k != 'month'}) or "none") for a in obs["recent_allocations"]))
        L.append(f"Department upkeep per month: {_kv(obs['department_upkeep'])}")
        L.append(f"Funding levels %: {_kv(obs['funding_pct'])}")
        pol = obs["policies"]
        L.append(f"Policies in force: {', '.join(pol['in_force']) or 'none'} (cost ${pol['monthly_cost']:,.0f}/month); "
                 f"available: " + "; ".join(f"{k} ({v})" for k, v in pol["available"].items()))
        si = obs["service_imports"]
        L.append(f"Service imports {'ON' if si['enabled'] else 'OFF'}: last month power {si['last_month']['power']}, water "
                 f"{si['last_month']['water']} (price ${si['price']['power']}/${si['price']['water']} per unit); surplus power "
                 f"sold {si['power_sold']}")
        ap = obs["approval"]
        L.append(f"Approval {ap['pct']}% (months below 30%: {ap['months_below_30']}); satisfied needs at homes %: {_kv(ap['needs_pct'])}")
    if "levels" in obs:
        L.append("")
        L.append(f"## Zones by level [empty, L1..L{C.MAX_LEVEL}]: " + _kv(obs["levels"]))
        L.append(f"Level rules: {obs['level_rules']}")
        L.append(f"Historic buildings listed: {obs['historic']['listed_tiles']}/{obs['historic']['max']} tiles")
        if obs.get("casino_sites"):
            L.append(f"Casino sites (2x2, top-left tile): {_sites(obs['casino_sites'])}")
        L.append(f"High-density zoned tiles {obs['high_density_zoned']}; homes by wealth {_kv(obs['homes_by_wealth'])}; "
                 f"industry by type {_kv(obs['industry_by_type']) or 'none'}")
        L.append(f"Average business profit margin: {_kv(obs['business_margins'])} (below 0 for "
                 f"{C.CLOSE_AFTER_MONTHS} months = closure)")
        hh = obs["households"]
        L.append(f"Households: income ${hh['income_per_resident']}/resident, left after rent ${hh['disposable_per_resident']}, "
                 f"homeless {hh['homeless_pct']}%")
        L.append("Natural resources (zone industry on them for farms, sawmills, mines, oil): "
                 + (_kv(obs["natural_resources"]) or "none left"))
        L.append("Road segments (x1,y1)-(x2,y2), longest first: " + "; ".join(
            f"({r['x1']},{r['y1']})-({r['x2']},{r['y2']})" + ("" if r["connected"] else " NOT connected")
            for r in obs["road_segments"]))
        if "zoning_suggestions" in obs:
            L.append("Empty strips along connected roads, ready to zone (x1,y1,x2,y2 tiles):")
            for z, lst in obs["zoning_suggestions"].items():
                L.append(f"  best for {z}: " + ("; ".join(f"({r['x1']},{r['y1']})-({r['x2']},{r['y2']}) {r['tiles']} tiles"
                                                         for r in lst) or "none - build roads first"))
            L.append("Road extensions from dead ends into open land: " + ("; ".join(
                f"({r['x1']},{r['y1']})-({r['x2']},{r['y2']}) {r['tiles']} tiles"
                + (f" incl. {r['bridge_tiles']} bridge" if r["bridge_tiles"] else "") + f", opens ~{r['opens_land']} tiles"
                for r in obs["road_suggestions"]) or "none"))
        L.append(f"Empty land tiles with road access: {obs['empty_land_with_road_access']}")
        if obs["sample_zoned_tiles_without_access"]:
            L.append(f"Some zoned tiles with no road access (x,y): {obs['sample_zoned_tiles_without_access']}")
    if "networks" in obs:
        L.append("")
        L.append("## Road networks (each is a separate power/water/sewage grid)")
        for n in obs["networks"]:
            plants = ", ".join(f"{p['kind']}@({p['x']},{p['y']})" + (f" q={p['quality']}" if 'quality' in p else "")
                               + (f" age {p['age_years']}y" if p.get('age_years', 0) >= 4 else "")
                               + (" OFFLINE" if p.get("offline") else "") for p in n["plants"]) or "none"
            L.append(f"net {n['id']}{' [connected to highway]' if n['outside_connected'] else ''}: roads {n['road_tiles']}, "
                     f"power {n['power_supply']}/{n['power_demand']} (supply/demand), water {n['water_supply']}/{n['water_demand']}, "
                     f"sewage {n['sewage_supply']}/{n['sewage_demand']}, water quality {n['water_quality']}; facilities: {plants}")
            if cr := _crowding(n["plants"]):
                L.append(f"  crowded plants (output kept): {cr}")
        if obs.get("upgrades"):
            L.append('Tier upgrades available (upgrade_facility {"module":"tier"}): ' + _upgrades(obs["upgrades"]))
        L.append(f"River pollution by section (north -> south): {obs['river']}")
        if obs["disconnected_facilities"]:
            L.append(f"Facilities not touching any road: {obs['disconnected_facilities']}")
        if obs["sample_tiles_needing_power"]:
            L.append(f"Zoned tiles needing power (x,y): {obs['sample_tiles_needing_power']}")
        if obs["sample_tiles_needing_water"]:
            L.append(f"Zoned tiles needing water (x,y): {obs['sample_tiles_needing_water']}")
        if obs["sample_tiles_needing_sewage"]:
            L.append(f"Buildings without sewage (x,y): {obs['sample_tiles_needing_sewage']}")
        if "candidate_sites" in obs:
            cs = obs["candidate_sites"]
            L.append("Candidate empty sites touching the highway-connected road network:")
            L.append(f"  far from homes (good for coal/gas): {_sites(cs['far_from_homes'])}")
            L.append(f"  near homes (fine for solar/wind/tower): {_sites(cs['near_homes'])}")
            L.append("  windiest, uncrowded first (wind turbines produce 40 x wind): "
                     + ", ".join(f"({s['x']},{s['y']}) wind={s['wind']}" + _crowd_note(s, "turbine")
                                 for s in cs.get('windy', [])))
            L.append("  riverbank next to a road, uncrowded then cleanest/upstream first (for pumps): "
                     + (_sites(cs['riverbank_next_to_road'], "intake") if cs['riverbank_next_to_road']
                        else "none - ask Planning to build a road to the river"))
            L.append("  riverbank, uncrowded then downstream of pumps first (for sewage): "
                     + _sites(cs['riverbank_for_sewage'], "outfall"))
            if cs.get("riverbank_2x2"):
                L.append("  riverbank 2x2 blocks (water works / advanced treatment / nuclear): " + _sites(cs["riverbank_2x2"]))
    _render_role(obs, L)
    pr = obs["progress"]
    L.append("")
    nm = pr["next_milestone"]
    L.append("## Progress: milestones reached " + (", ".join(pr["milestones_reached"]) or "none")
             + (f"; next '{nm['name']}' at {nm['pop']:,} residents (${nm['reward']:,} reward) unlocks "
                f"{', '.join(k.replace('_', ' ') for k in nm['unlocks']) or 'nothing new'}" if nm else ""))
    if pr["locked"]:
        L.append("Still locked: " + ", ".join(k.replace("_", " ") for k in pr["locked"]))
    if obs.get("districts"):
        L.append("Districts: " + "; ".join(f"{d['name']} ({d['tiles']} tiles, policies {d['policies'] or 'none'}, "
                                           f"tax {d['tax_mod_pct'] or 'city rate'}, industry {d['specialization'] or 'any'})"
                                           for d in obs["districts"]))
    if "landmarks" in obs:
        L.append(f"Landmarks: {_kv(obs['landmarks'])}; tourists per month {obs['tourists']}")
    if obs["forecast"] or obs["offers"]:
        L.append("")
        L.append("## Forecast & offers")
        L.extend(f"- {f}" for f in obs["forecast"])
        L.extend(f"- OPEN OFFER #{o['id']}: large factory, {o['jobs']} jobs, ${o['bonus']:,} bonus, heavy pollution "
                 f"(mayor: accept_offer / decline_offer before month {o['expires']})" for o in obs["offers"])
    L.append("")
    L.append("## Events")
    L.extend(f"- {e}" for e in obs["events"][-10:]) if obs["events"] else L.append("- none")
    L.append("")
    L.append("## Council board (recent)")
    if obs["council"]:
        L.extend(f"- [m{m['month']}] {m['sender']} -> {m['to']}: {m['text']}" for m in obs["council"][-12:])
    else:
        L.append("- no messages")
    L.append("")
    L.append("## Results of your previous turn")
    if obs["last_turn"]:
        L.extend(f"- {'OK ' if r['ok'] else 'FAILED'} {r['action'].get('type')}: {r['message']}" for r in obs["last_turn"])
    else:
        L.append("- (no actions)")
    if obs["notes"]:
        L.append("")
        L.append(f"## Your notes to self\n{obs['notes']}")
    if "map" in obs:
        L.append("")
        L.append("## Map")
        L.append(obs["map_legend"])
        L.append(obs["map"])
    return "\n".join(L)


def render_action_docs(role: str) -> str:
    lines = []
    docs = action_docs()
    for a in ROLES[role]["actions"]:
        ex, help_ = docs[a]
        lines.append(f"- {a}: {help_}\n  example: {ex}")
    return "\n".join(lines)
