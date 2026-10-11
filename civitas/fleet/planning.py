"""Planning's Next line and the sites it names (the briefing's Your budget and Ready sites blocks, some tool hints).

The 120-month LLM runs (trinity v19 / v23 seeds 0-1, the two-city region run, the late smokes): planning was the
lowest-scoring LLM department and repeated its mistakes across quarters (178 recurrences in the Problems tab, the next
role 33). Measured causes:
- no parks: 1 park tile in 244 turns. Every run's R demand sat at the +0.25 cap for lack of parks (World.demand_caps:
  under one park tile per 400 residents from 2,000) from about month 24 to the end, and no home was within 4 tiles of
  a park, so the land value of developed tiles was 14-19% (the scripted warm-up city: 44%). Land value and demand
  balance were planning's two biggest score losses. Nothing in its briefing named parks or the cap.
- utilities' riverbank lot: in 103 of 240 turns utilities' pump or outlet was blocked for a free riverbank lot by a
  road; planning spent 10-20 inspects a turn hunting for the lot (417 in 36 turns of v23 seed 0; 52 turns ended on the
  step limit) and dezoned roads, river and empty land.
- zoning over budget: 134 zones failed on money, 131 after other actions spent it, 72 recurring the next turn; the
  Ready sites zone lines had no price.
- requests to planning: 176 in those runs (98 from utilities for bank lots, 35 from the mayor, 26 from health), 85
  unanswered; in the late smokes it answered none (police's jail site, health's landfill and cemetery at the town edge).
So every turn planning reads a Next line (planning_next) like transport's: the builds that raise its score most now,
priced, with the exact calls; its zone lines carry their price; a parks line names the best park block; an Open
requests block (request_fix) says what to do for each department asking it, and its end_turn is held once while any
is open (ToolBox.end_turn, the model's toolbox only).
"""
from __future__ import annotations

import heapq
import json
import math
from typing import NamedTuple

import numpy as np

from ..core.actions import zone_density, zone_price
from ..sim import catalog as C
from ..sim.grid import box_sum
from .fmt import advice, money
from .tools import (AWAY_FROM_HOMES, BUILT_NAMES, DIRECTIONS, MAX_SITES, SiteList, Spot, _connected_roads, _empty_land,
                    _riverbank_by_roads, _touching, _wanted, _where, claim_spots, dept_lists, edge_site, funded_sites,
                    land_blocked, land_fix, road_cost, scarce_bank, short_of, site_price, spare_funded, zone_fit_rect,
                    zone_line, zoning_cost)

PARK_RADIUS = 4            # World._compute_environment: park tiles within box_sum(.., 4) raise land value,
PARK_STEP, PARK_MAX = 0.08, 0.3   # by 8 points each, at most 30 (4 park tiles in range)
PARK_TILES = 4             # a park block of up to 2x2 tiles (what reaches PARK_MAX for the homes around it)
HOMES_PER_PARK = 25        # the playbook's "about one per 25 homes"
ROADSIDE_PARK = 0.5        # a park on a roadside lot counts half: those lots are for zones and services
CAP_NEAR = 0.8             # the parks cap is named from this share of CAP_POP_PARKS on
WAITING_MAX = 10           # the playbook: skip a zone type with this many zoned tiles waiting to develop
ZONE_LETTERS = ("R", "C", "I", "O")     # what the Next line zones by demand (mixed use has no demand of its own)
BANK_ROAD_MAX = 15         # a road to free riverbank: at most this many new tiles
EDGE_ROAD_MAX = 20         # a road to the map edge for transport's first train station: at most this many new tiles
DETOUR_PAD = 3             # a detour around a building stays within this many tiles of the road it replaces
DETOUR_LEGS = 4            # and has at most this many straight legs
TURN_COST = 5              # a detour's turn costs this much more (fewer legs, fewer calls)
NEXT_ITEMS = 3             # builds on the Next line, most valuable first
BANK_PURPOSES = ("sewage", "pump")      # utilities' one-tile riverbank buildings, the outlet first (it pollutes the
                                        # river, so it goes downstream: south)


def call(action: str, args: dict) -> str:
    return f"{action} {json.dumps(args, separators=(',', ':'))}"


class Parks(NamedTuple):
    have: int               # World.leisure_tiles()
    by_pop: int             # what the R demand cap asks for: one per RESIDENTS_PER_PARK residents
    want: int               # that, or one per HOMES_PER_PARK developed homes if more
    capped: bool            # R demand capped for lack of parks now (World.parks_capped, as demand_caps decides)
    soon: bool              # within CAP_NEAR of the population where the cap starts, short of by_pop


def parks(w) -> Parks:
    """The park tiles against the population, by the World's own cap helpers (leisure_tiles, parks_capped) on the
    population the simulation reads (stats)."""
    pop = w.stats.get("population", 0.0)
    homes = int((np.isin(w.struct, C.HOMES) & (w.level > 0)).sum())
    have, by_pop = w.leisure_tiles(), math.ceil(pop / C.RESIDENTS_PER_PARK)
    capped = w.parks_capped()
    at_cap = max(by_pop, math.ceil(C.CAP_POP_PARKS / C.RESIDENTS_PER_PARK))
    soon = not capped and pop >= CAP_NEAR * C.CAP_POP_PARKS and have < at_cap
    return Parks(have, by_pop, max(by_pop, homes // HOMES_PER_PARK), capped, soon)


class ParkSite(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int
    tiles: int
    homes: int              # developed homes within PARK_RADIUS of it whose land value it raises

    @property
    def cost(self) -> float:
        return self.tiles * C.CATALOG["park"]["cost"]

    def args(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


def park_site(w, funds: float | None = None, avoid=frozenset()) -> ParkSite | None:
    """Where a park raises land value most: the empty land tile whose 9x9 neighbourhood holds the most developed
    homes, shops and offices still short of PARK_MAX from parks (each weighted by what one more park tile adds), a
    roadside lot at ROADSIDE_PARK (zones and services need those), not in `avoid` (lots funded departments or utilities
    are about to use); grown to a free block of up to PARK_TILES tiles (2x2, else 2x1) that `funds` pay for (one tile
    at least). None when no park would raise any developed tile."""
    s = w.struct
    target = np.isin(s, (C.RES, C.COM, C.OFF, C.MIX)) & (w.level > 0)
    near = box_sum(s == C.PARK, PARK_RADIUS)
    gain = np.minimum(PARK_MAX, PARK_STEP * (near + 1)) - np.minimum(PARK_MAX, PARK_STEP * near)
    free = _empty_land(w)
    for x, y in avoid:
        free[y, x] = False
    score = box_sum(target * gain, PARK_RADIUS) * np.where(_touching(_connected_roads(w)), ROADSIDE_PARK, 1.0)
    score = np.where(free, score, 0.0)
    if score.max() <= 1e-9:
        return None
    y, x = map(int, np.unravel_index(int(np.argmax(score)), score.shape))
    price = C.CATALOG["park"]["cost"]
    n = PARK_TILES if funds is None else max(1, min(PARK_TILES, int((funds + 1e-9) // price)))
    shapes = ([(2, 2)] if n >= 4 else []) + ([(2, 1), (1, 2)] if n >= 2 else []) + [(1, 1)]
    best = None
    for bw, bh in shapes:
        for x0 in range(x - bw + 1, x + 1):
            for y0 in range(y - bh + 1, y + 1):
                if x0 < 0 or y0 < 0 or x0 + bw > w.w or y0 + bh > w.h or not free[y0:y0 + bh, x0:x0 + bw].all():
                    continue
                total = float(score[y0:y0 + bh, x0:x0 + bw].sum())
                if best is None or total > best[0]:
                    best = (total, (x0, y0, x0 + bw - 1, y0 + bh - 1))
        if best is not None:
            break
    x1, y1, x2, y2 = best[1]
    homes = np.isin(s, C.HOMES) & (w.level > 0) & (near < PARK_MAX / PARK_STEP)
    r = PARK_RADIUS
    reach = int(homes[max(0, y1 - r):y2 + r + 1, max(0, x1 - r):x2 + r + 1].sum())
    return ParkSite(x1, y1, x2, y2, (x2 - x1 + 1) * (y2 - y1 + 1), reach)


def parks_line(p: Parks, site: ParkSite | None) -> list[str]:
    """Planning's Ready sites line on parks: the park tiles against what the population asks for (`p`), the R demand
    cap while it holds (or is near), and the best park block (`site`: the view's, the Next line's too) with its price
    and call; [] while there is no developed home, shop or office for a park to raise."""
    if site is None:
        return []
    cap = (f": R demand capped at +{C.CAP_LEVEL:g} until {p.by_pop}" if p.capped else
           f": R demand is capped at +{C.CAP_LEVEL:g} from {C.CAP_POP_PARKS:,} residents below "
           f"{math.ceil(C.CAP_POP_PARKS / C.RESIDENTS_PER_PARK)}" if p.soon else "")
    return [f"- parks: {p.have} of {p.want} park tiles{cap}; each raises land value {round(100 * PARK_STEP)} points "
            f"within {PARK_RADIUS} tiles (up to {round(100 * PARK_MAX)} from 4); best {_where(site._asdict())}, "
            f"{site.homes} homes near; {money(site.cost)}" + advice(f": {call('build_park', site.args())}")]


def cap_line(w) -> list[str]:
    """Planning's Key metrics line while its demand is capped (World.demand_caps): by parks (its own fix), by the
    missing regional link (transport's train station or harbour)."""
    caps = w.stats.get("demand_caps") or {}
    if not caps:
        return []
    p, pop = parks(w), w.stats.get("population", 0.0)
    linked = bool(w.facilities(C.TRAIN)) or bool(w.facilities(C.HARBOUR))
    why = ([f"parks {p.have} of {p.by_pop} tiles" + advice(" (yours: build_park)")] if p.capped else []) + (
        ["no train station or harbour (transport's)"] if pop >= C.CAP_POP_CONNECTION and not linked else [])
    return [f"- demand capped: {', '.join(f'{k} +{v:g}' for k, v in caps.items())}: "
            + (", ".join(why) or "see city_overview")]


# ------------------------------------------------------------------ the riverbank lot utilities waits for
class Item(NamedTuple):
    """One build on planning's Next line: what and why, its price, the exact call."""
    text: str
    cost: float
    call: str
    road: bool = False      # planning's road into open land (the build the mayor funds for it: mayor_ask)


def bank_purposes(w, depts: dict) -> list[str]:
    """Utilities' one-tile riverbank purposes (pump, sewage outlet) blocked for lack of land while their supply is
    wanted: planning frees a lot for them."""
    lists = depts.get("utilities") or {}
    return [p for p in BANK_PURPOSES if p in lists and lists[p].blocked and _wanted(w, p)]


def bank_lot(w, purpose: str) -> tuple[int, int] | None:
    """The zoned riverbank lot beside a highway-connected road to free for `purpose`: an empty lot first, then the
    lowest building level; an outlet's the southernmost (downstream), a pump's the northernmost. None without one."""
    lots, _ = _riverbank_by_roads(w)
    ys, xs = np.nonzero(lots & np.isin(w.struct, C.ZONES))
    if not len(xs):
        return None
    south = purpose == "sewage"
    x, y = min(zip(xs.tolist(), ys.tolist()), key=lambda t: (int(w.level[t[1], t[0]]), -t[1] if south else t[1], t[0]))
    return x, y


def bank_road(w) -> tuple[dict, tuple[int, int], float] | None:
    """The cheapest straight road from the highway-connected network over empty land (no bridges) that brings a free
    riverbank tile beside a road, at most BANK_ROAD_MAX new tiles: (build_road args, the lot, cost); None without one.
    Road tiles on the bank are fine; the lot is a bank tile beside the road's last tile, off the road."""
    s, land = w.struct, w.terrain == C.LAND
    bank = land & _touching(w.terrain == C.WATER, diagonal=False)
    free = (s == C.NONE) & land
    best = None
    ys, xs = np.nonzero(_connected_roads(w) & (s != C.HIGHWAY))
    for x0, y0 in zip(xs.tolist(), ys.tolist()):
        for dx, dy in DIRECTIONS:
            path = []
            x, y = x0 + dx, y0 + dy
            while len(path) < BANK_ROAD_MAX and 0 <= x < w.w and 0 <= y < w.h and free[y, x]:
                path.append((x, y))
                lot = next(((x + i, y + j) for j in (-1, 0, 1) for i in (-1, 0, 1)
                            if (i or j) and 0 <= x + i < w.w and 0 <= y + j < w.h and bank[y + j, x + i]
                            and free[y + j, x + i] and (x + i, y + j) not in path), None)
                if lot is not None:
                    if best is None or len(path) < best[0]:
                        best = (len(path), {"x1": x0, "y1": y0, "x2": x, "y2": y}, lot)
                    break
                x, y = x + dx, y + dy
    if best is None:
        return None
    return best[1], best[2], best[0] * C.CATALOG["road"]["cost"]


def _bank_beside(w, site: dict) -> tuple[int, int] | None:
    """A free riverbank tile beside a road_extension site's straight run (off the run), else None."""
    bank = (w.terrain == C.LAND) & _touching(w.terrain == C.WATER, diagonal=False) & _empty_land(w)
    sx = (site["x2"] > site["x1"]) - (site["x2"] < site["x1"])
    sy = (site["y2"] > site["y1"]) - (site["y2"] < site["y1"])
    n = max(abs(site["x2"] - site["x1"]), abs(site["y2"] - site["y1"]))
    run = [(site["x1"] + i * sx, site["y1"] + i * sy) for i in range(n + 1)]
    on = set(run)
    for x, y in run:
        for j in (-1, 0, 1):
            for i in (-1, 0, 1):
                if 0 <= x + i < w.w and 0 <= y + j < w.h and bank[y + j, x + i] and (x + i, y + j) not in on:
                    return x + i, y + j
    return None


def bank_item(w, depts: dict, lists: dict | None = None) -> Item | None:
    """Freeing the riverbank lot utilities' blocked pump or outlet waits for (bank_purposes): dezone a zoned bank lot
    beside a road (bank_lot, free), else planning's best road into open land when it passes free bank (one road for
    both), else a short road that reaches free bank (bank_road)."""
    wanted = bank_purposes(w, depts)
    if not wanted:
        return None
    p = wanted[0]
    pr = site_price(w, "utilities", p)
    name = f"utilities' {pr.building if pr else 'riverbank building'}"
    if (xy := bank_lot(w, p)) is not None:
        x, y = xy
        lv = int(w.level[y, x])
        held = "an empty lot" if not lv else f"razes a level-{lv} {BUILT_NAMES[C.LETTER_BY_ZONE[int(w.struct[y, x])]]}"
        return Item(f"free a riverbank lot for {name} ({held})", 0.0,
                    call("dezone", {"x1": x, "y1": y, "x2": x, "y2": y}))
    roads = (lists or {}).get("road_extension")
    if roads and roads.sites and (xy := _bank_beside(w, roads.sites[0])) is not None:
        return road_item(lists, f"it reaches free riverbank for {name}: leave ({xy[0]},{xy[1]}) empty")
    if (road := bank_road(w)) is not None:
        args, (lx, ly), cost = road
        return Item(f"a road to free riverbank for {name}, then leave ({lx},{ly}) empty", cost,
                    call("build_road", args))
    return None


def dezone_hint(w, depts: dict) -> str:
    """For a failed dezone: the zoned riverbank lot utilities waits for (bank_item's dezone), when it waits for one."""
    wanted = bank_purposes(w, depts)
    xy = bank_lot(w, wanted[0]) if wanted else None
    return "" if xy is None else f"; the riverbank lot utilities waits for: " + call(
        "dezone", {"x1": xy[0], "y1": xy[1], "x2": xy[0], "y2": xy[1]})


# ------------------------------------------------------------------ requests to planning
def request_fix(w, dept: str, v: "View", text: str = "") -> str:
    """What planning can do for a department asking it for land, for its Open requests block: utilities' riverbank lot
    (bank_item's call); a first building blocked for lack of land (land_blocked): a lot at the town edge for one that
    must stay away from homes (edge_site), else what it needs (land_fix); a funded building whose lot is free
    (claim_spots): say where; else to reply where it can build or why not. Of several blocked buildings, the one the
    request (`text`) names. The late smokes (seeds 0-1, the region): utilities' bank
    lots, police's jail and health's landfill and cemetery sites away from homes went unanswered every time. All from
    the view (planning_view), computed once a briefing."""
    if dept == "mayor":
        return "your Next line has the calls: tell it what you did, or why not"
    if dept == "utilities" and v.bank is not None:
        return f"{v.bank.call} ({money(v.bank.cost) if v.bank.cost else 'free'})"
    blocked = v.blocked
    if dept in blocked:
        named = lambda q: (site_price(w, dept, q).building if site_price(w, dept, q) else q.replace("_", " "))
        p = next((q for q in blocked[dept] if named(q).split()[-1] in text.lower()), blocked[dept][0])
        name = named(p)
        if p == "train_station" and v.link is not None:              # transport's first link: the edge road's call
            return f"its first {name} needs a road to the map edge: {v.link.call} ({money(v.link.cost)})"
        if p in AWAY_FROM_HOMES:
            site = edge_site(w, v.roads)
            return (f"its {name} needs a lot away from homes: leave ({site['x2']},{site['y2']}) free, the far end of "
                    f"road_extension {_where(site)}" if site else
                    f"its {name} needs a lot away from homes: extend a road into open land at the town edge and leave "
                    "its far end free")
        return f"its {name} needs {land_fix(w, dept, p) or 'a free roadside lot'}: leave or free (dezone) one"
    spot = next((sp for k, sp in v.claims.items() if k[0] == dept), None)
    if spot is not None:
        return f"its {spot.building} lot ({spot.xy[0]},{spot.xy[1]}) is free: tell it, and zone around it"
    return "send_message where it can build (its Ready sites) or why not"


# ------------------------------------------------------------------ zones by demand
def strips_shown(lists: dict[str, SiteList]) -> dict[str, dict]:
    """zone purpose -> the strip its Ready sites line shows (zone_line, one zone type per strip), in list order as
    ready_lines walks them."""
    strips: dict[str, dict] = {}
    for purpose, lst in lists.items():
        sites = lst.sites[:MAX_SITES]
        if purpose.startswith("zone_") and not lst.short and sites and "x2" in sites[0]:
            zone_line(purpose, sites, strips, lambda p, s: "")
    return strips


def _density(w, letter: str) -> str:
    if letter in ("I", "M") or not w.unlocked("high_density"):
        return "low"
    if letter == "R":
        return "high" if w.demand.get("R_high", 0.0) >= w.demand.get("R_low", 0.0) else "low"
    return "high"


def zone_item(w, city: dict, lists: dict[str, SiteList], funds: float,
              avoid=frozenset()) -> tuple[Item | None, list[str]]:
    """The zone type with the largest positive demand and fewer than WAITING_MAX zoned tiles waiting, on the strip its
    Ready sites line shows (strips_shown), sized to `funds` (zone_fit_rect, sparing `avoid`); a type whose strip has
    no tile left outside `avoid` gives way to the next. And the types with demand but no usable strip (a road into
    open land opens one); a locked type (offices, mixed use before their milestone) is neither. (None, ...) when no
    type qualifies."""
    zoned, dev = city.get("zoned_tiles") or {}, city.get("developed_tiles") or {}
    strips = strips_shown(lists)
    blocked, cands = [], []
    for letter in ZONE_LETTERS:
        d = w.demand.get(letter, 0.0)
        waiting = zoned.get(letter, 0) - dev.get(letter, 0)
        lst = lists.get(f"zone_{letter}")
        if d <= 0 or waiting >= WAITING_MAX or lst is None or (lst.short or "").startswith("locked"):
            continue
        if f"zone_{letter}" not in strips:
            blocked.append(letter)
            continue
        cands.append((d, -ZONE_LETTERS.index(letter), letter, waiting))
    for d, _, letter, waiting in sorted(cands, reverse=True):
        strip = strips[f"zone_{letter}"]
        density = _density(w, letter)
        args = {"zone": letter, "density": density, "x1": strip["x1"], "y1": strip["y1"], "x2": strip["x2"],
                "y2": strip["y2"]}
        price = zone_price(zone_density(letter, density))
        fit = zone_fit_rect(w, args, funds if not short_of(funds, price) else math.inf, avoid)  # broke: whole strip
        if fit is None:                                # every tile of it spared for others: no usable strip
            blocked.append(letter)
            continue
        (x1, y1, x2, y2), n, cost = fit
        args.update(x1=x1, y1=y1, x2=x2, y2=y2)
        if letter in ("I", "M"):
            args.pop("density")
        dens = "" if letter in ("I", "M") else f" {density} density"
        return Item(f"zone {n} tiles {letter}{dens} (demand {d:+.2f}, {waiting} zoned tile{'s' * (waiting != 1)} "
                    f"waiting)", cost, call("zone", args)), sorted(blocked, key=ZONE_LETTERS.index)
    return None, sorted(blocked, key=ZONE_LETTERS.index)


def road_item(lists: dict[str, SiteList], why: str) -> Item | None:
    """The best road_extension site (a dead end's extension, else a branch road with its dezone or bulldoze first)."""
    lst = lists.get("road_extension")
    if not lst or not lst.sites:
        return None
    s = lst.sites[0]
    args = {"x1": s["x1"], "y1": s["y1"], "x2": s["x2"], "y2": s["y2"]}
    first = s.get("first")
    zoning = zoning_cost(s)
    return Item(f"a road into open land ({why}; opens {s['opens_land']} tiles, zoning them about {money(zoning)})",
                road_cost(s), (f"{first}, then " if first else "") + call("build_road", args), road=True)


def edge_road(w) -> tuple[dict, tuple[int, int], float] | None:
    """The cheapest straight road from the highway-connected network over empty land to one tile short of the map edge,
    the edge tile beyond it empty land, at most EDGE_ROAD_MAX new tiles: (build_road args, the edge tile, cost); None
    without one. A train station goes on an empty map-edge tile beside a connected road (observation._edge_sites)."""
    s, free = w.struct, (w.struct == C.NONE) & (w.terrain == C.LAND)
    edge = lambda x, y: x in (0, w.w - 1) or y in (0, w.h - 1)
    best = None
    ys, xs = np.nonzero(_connected_roads(w) & (s != C.HIGHWAY))
    for x0, y0 in zip(xs.tolist(), ys.tolist()):
        for dx, dy in DIRECTIONS:
            n, x, y = 0, x0 + dx, y0 + dy
            while n < EDGE_ROAD_MAX and 0 <= x < w.w and 0 <= y < w.h and free[y, x] and not edge(x, y):
                n += 1
                nx, ny = x + dx, y + dy
                if 0 <= nx < w.w and 0 <= ny < w.h and edge(nx, ny) and free[ny, nx]:
                    if best is None or n < best[0]:
                        best = (n, {"x1": x0 + dx, "y1": y0 + dy, "x2": x, "y2": y}, (nx, ny))
                    break
                x, y = nx, ny
    if best is None:
        return None
    return best[1], best[2], best[0] * C.CATALOG["road"]["cost"]


def link_item(w, blocked: dict[str, list[str]]) -> Item | None:
    """A road to the map edge (edge_road) while transport's first train station is blocked for want of an edge lot by a
    road and R and C demand are capped without a link (land_blocked: tools.capped_unlinked). Seed 4 at month 96:
    6,052 residents, no train station or harbour, no edge lot by a road."""
    if "train_station" not in blocked.get("transport", ()) or (road := edge_road(w)) is None:
        return None
    args, (lx, ly), cost = road
    return Item(f"a road to the map edge for transport's first train station (R and C demand stay capped at "
                f"+{C.CAP_LEVEL:g} without one), then leave ({lx},{ly}) empty", cost, call("build_road", args))


# ------------------------------------------------------------------ the Next line
def away_item(w, blocked: dict[str, list[str]], lists: dict) -> Item | None:
    """A road into open land whose far end is away from homes (edge_site) while a first building that must stay away
    from homes is blocked for lack of land (land_blocked: police's jail, health's landfill or cemetery): build it and
    leave that end free. The late smokes: police asked for a jail site and health for landfill and cemetery sites at
    the town edge; planning answered none. `blocked`: land_blocked."""
    found = [(d, p) for d, ps in blocked.items() for p in ps if p in AWAY_FROM_HOMES]
    roads = (lists.get("road_extension") or SiteList([], None, None)).sites
    site = edge_site(w, roads) if found else None
    if site is None:
        return None
    names = []
    for d, p in found:
        pr = site_price(w, d, p)
        names.append(f"{d}'{'s' * (not d.endswith('s'))} {pr.building if pr else p}")
    one = {**lists, "road_extension": lists["road_extension"]._replace(sites=[site])}
    return road_item(one, f"a lot away from homes for {' and '.join(dict.fromkeys(names))}: leave ({site['x2']},"
                          f"{site['y2']}) free")


class View(NamedTuple):
    """What planning's briefing computes once (planning_view) and its Next line, Ready sites and Open requests blocks
    share: one observation per priced department and what follows from it."""
    depts: dict             # dept_lists
    funded: dict            # funded_sites
    claims: dict            # claim_spots: each funded building's own lot
    lists: dict             # planning's site lists with its zone strips sparing those lots (spare_funded)
    roads: list             # its road_extension sites
    blocked: dict           # land_blocked: first buildings blocked for lack of land
    avoid: set              # the claimed lots and scarce riverbank lots: zone and park calls spare them
    bank: Item | None       # bank_item
    away: Item | None       # away_item (None when bank_item is a road already)
    link: Item | None       # link_item
    parks: Parks
    park: ParkSite | None   # park_site, sized to the funds bank / away leave: the parks line and the Next line's

    @property
    def spots(self) -> list:
        return list(self.claims.values())


def planning_view(env, lists: dict[str, SiteList], funds: float) -> View:
    w = env.world
    depts = dept_lists(env)
    funded = funded_sites(w, depts)
    claims = claim_spots(funded)
    spared = spare_funded(lists, list(claims.values()))
    roads = lists["road_extension"].sites if "road_extension" in lists else []
    blocked = land_blocked(env, depts)
    avoid = {t for sp in claims.values() for t in sp.tiles} | scarce_bank(w)
    bank = bank_item(w, depts, spared)
    away = None if bank is not None and bank.road else away_item(w, blocked, spared)
    link = link_item(w, blocked)
    left = funds - sum(i.cost for i in (bank, link, away) if i is not None)
    return View(depts, funded, claims, spared, roads, blocked, avoid, bank, away, link, parks(w),
                park_site(w, left, avoid))


def planning_next(env, city: dict, v: View, funds: float, ask, most: int = NEXT_ITEMS) -> str:
    """Planning's Next line, every turn: up to `most` (NEXT_ITEMS; a refreshed Next line in a tool result, 2:
    briefing.next_line) builds that raise its score most now, in this order, each
    priced with its exact call: the riverbank lot utilities' blocked pump or outlet waits for (bank_item); a road to
    the map edge for transport's first train station while R and C demand are capped without one (link_item); a road
    leaving a lot away from homes for a blocked jail, landfill or cemetery (away_item); parks while R demand is capped
    for lack of them, or soon will be (the view's park, the parks line's); the zone type demand asks for most
    (zone_item, on its Ready sites strip, sized to the funds left); parks while under one per HOMES_PER_PARK homes; a
    road into open land (road_item) when a type with demand has no usable strip, nothing else is to build, or the
    funds left cover it. Each later item is sized and priced with what the earlier ones leave. When the items shown
    cost more than the funds together: the difference to ask the mayor for when mayor_ask (`ask`) says planning may
    ask now, else to say so in the end_turn report. The mayor's needs line prices only planning's road and zoning all
    it opens (road_build), so the two amounts can differ: making it follow the Next line would change funding_gaps,
    which the money guards read on the scripted fleet's turns too."""
    w = env.world
    items = [i for i in (v.bank, v.link, v.away) if i is not None]
    p, park = v.parks, v.park
    park_item = None if park is None else Item(
        f"{park.tiles} park tile{'s' * (park.tiles > 1)} by {park.homes} homes (parks {p.have} of {p.want}"
        + (f"; R demand capped at +{C.CAP_LEVEL:g} until {p.by_pop}" if p.capped else "") + ")",
        park.cost, call("build_park", park.args()))
    if park_item and (p.capped or p.soon):
        items.append(park_item)
    left = funds - sum(i.cost for i in items)
    zone, blocked = zone_item(w, city, v.lists, left, v.avoid)
    if zone is not None:
        items.append(zone)
    if park_item and not (p.capped or p.soon) and p.have < p.want:
        items.append(park_item)
    if any(w.demand.get(k, 0.0) > 0 for k in ZONE_LETTERS) and not any(i.road for i in items):
        road = road_item(v.lists, f"no free strip for {'/'.join(blocked)}" if blocked else "more land to zone")
        if road is not None and (blocked or not items or not short_of(funds - sum(i.cost for i in items), road.cost)):
            items.append(road)
    if not items:
        dem = ", ".join(f"{k} {w.demand.get(k, 0.0):+.2f}" for k in ZONE_LETTERS)
        return (f"- Next: nothing to build now (demand {dem}; no type with demand has a free strip and fewer than "
                f"{WAITING_MAX} zoned tiles waiting; parks {p.have} of {p.want}). End your turn")
    shown = items[:most]
    text = lambda i: f"{i.text} ({money(i.cost) if i.cost else 'free'}): {i.call}"
    line = "- Next: " + text(shown[0]) + "".join(f"; then {text(i)}" for i in shown[1:])
    if short_of(funds, total := sum(i.cost for i in shown)):
        lack = math.ceil(total - funds)
        line += f"; you have {money(math.floor(funds))}" + (
            f": ask the mayor for {money(lack)} for these (send_message)" if ask.ok else
            ": say what you need in your end_turn report")
    return line


# ------------------------------------------------------------------ a road around a building
def detour(w, line: list[tuple[int, int]]) -> list[str] | None:
    """For a straight build_road blocked by a building: the cheapest way from its first tile to its last over empty
    land (a road tile), river (a bridge) and road, within DETOUR_PAD tiles of the line, as build_road calls, one per
    straight leg with new tiles (a turn costs TURN_COST more); None when either end is taken, no way exists, or it
    takes more than DETOUR_LEGS legs. s1 v23: planning sent build_road through the water pump at (29,24) 16 times over
    10 turns, told "route around the building"."""
    (xa, ya), (xb, yb) = line[0], line[-1]
    ok = lambda x, y: w.struct[y, x] in (C.NONE, C.ROAD, C.HIGHWAY)
    if (xa, ya) == (xb, yb) or not ok(xa, ya) or not ok(xb, yb):
        return None
    x0, x1 = max(0, min(xa, xb) - DETOUR_PAD), min(w.w - 1, max(xa, xb) + DETOUR_PAD)
    y0, y1 = max(0, min(ya, yb) - DETOUR_PAD), min(w.h - 1, max(ya, yb) + DETOUR_PAD)

    def step(x, y) -> float:
        s = w.struct[y, x]
        if s in (C.ROAD, C.HIGHWAY):
            return 0.01
        return C.CATALOG["bridge" if w.terrain[y, x] == C.WATER else "road"]["cost"]

    start = (xa, ya, -1)
    dist, prev, heap = {start: 0.0}, {}, [(0.0, start)]
    end = None
    while heap:
        d, (x, y, k) = heapq.heappop(heap)
        if d > dist.get((x, y, k), math.inf):
            continue
        if (x, y) == (xb, yb):
            end = (x, y, k)
            break
        for i, (dx, dy) in enumerate(DIRECTIONS):
            nx, ny = x + dx, y + dy
            if not (x0 <= nx <= x1 and y0 <= ny <= y1) or not ok(nx, ny):
                continue
            nd = d + step(nx, ny) + (TURN_COST if k not in (-1, i) else 0.0)
            if nd < dist.get((nx, ny, i), math.inf):
                dist[(nx, ny, i)], prev[(nx, ny, i)] = nd, (x, y, k)
                heapq.heappush(heap, (nd, (nx, ny, i)))
    if end is None:
        return None
    path, node = [], end
    while node in prev or node == start:
        path.append(node)
        if node == start:
            break
        node = prev[node]
    path.reverse()
    legs, cur = [], [path[0][:2]]
    for a, b in zip(path, path[1:]):
        if len(cur) > 1 and b[2] != a[2]:
            legs.append(cur)
            cur = [a[:2]]
        cur.append(b[:2])
    legs.append(cur)
    new = [leg for leg in legs if any(w.struct[y, x] == C.NONE for x, y in leg)]
    if not new or len(new) > DETOUR_LEGS:
        return None
    return [call("build_road", {"x1": leg[0][0], "y1": leg[0][1], "x2": leg[-1][0], "y2": leg[-1][1]}) for leg in new]


def detour_hint(w, line: list[tuple[int, int]]) -> str:
    """detour() as a hint clause, with its cost: "; around it: build_road {...}, then build_road {...} ($80)"."""
    try:
        calls = detour(w, line)
    except Exception:
        return ""
    if not calls:
        return ""
    tiles = set()
    for c in calls:
        a = json.loads(c.split(" ", 1)[1])
        for x in range(min(a["x1"], a["x2"]), max(a["x1"], a["x2"]) + 1):
            for y in range(min(a["y1"], a["y2"]), max(a["y1"], a["y2"]) + 1):
                tiles.add((x, y))
    cost = sum(C.CATALOG["bridge" if w.terrain[y, x] == C.WATER else "road"]["cost"]
               for x, y in tiles if w.struct[y, x] == C.NONE)
    return f"; around it: {', then '.join(calls)} ({money(cost)})"


