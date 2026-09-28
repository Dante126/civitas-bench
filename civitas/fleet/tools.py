"""Fleet tools: what a tool-calling agent can look up and do on its turn.

`ToolBox(env, role)` exposes OpenAI-style function schemas (`schemas()`) and runs calls (`call(name, arguments)`,
never raises). Look-ups read the world through the observation builders; action tools are the role's existing
actions (schemas derived from `actions.action_docs()`) run through `apply_action`; turn-control tools write the
fleet memory (`FleetState`) and collect council messages and the end-of-turn report.
"""
from __future__ import annotations

import copy
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import Callable, NamedTuple

import numpy as np

from ..core.actions import (OWNED_STRUCTS, POWER_LINES, SCHOOL_KINDS, SEWAGE_LINES, WATER_LINES, _line, _point,
                            _rect, action_docs, apply_action)
from ..core.observation import MAP_LEGEND, _far_from_homes, _uncrowded_first, build_observation
from ..core.roles import ROLES
from ..sim import catalog as C
from ..sim.grid import label_components, shift
from .fmt import TOOL_MESSAGE_CHARS, a_or_an, money
from .memory import DEPARTMENTS, HISTORY_KINDS, LIMITS, FleetState

RESULT_CHARS = TOOL_MESSAGE_CHARS - 200     # the agent cuts tool messages at TOOL_MESSAGE_CHARS; lists are trimmed below
MAX_SITES = 6
MAX_RADIUS = 4
MAX_MONTHS = 12
MAX_ENTRIES = 20             # history entries / council messages per call
REPORT_MAX_CHARS = 1000      # end_turn reports are cut to this
REPEAT_ERROR_CHARS = 300     # the earlier error quoted when a repeated failed action or bad call is refused
INSPECTS_FREE = 3            # inspects run in a turn before their results point to map_overview (smoke v9-v11:
                             # planning made 5.4 a turn, other roles 0.4-1.2)
BLOCKERS_NAMED = 3           # tiles named per kind when a road is blocked
DEZONES_NAMED = 2            # dezone rects suggested for the lots in its way
ZONES_BESIDE = 2             # zone rects suggested beside a rectangle with no empty land
CONTENT_KINDS = 5            # kinds of tile named when a failed zone / dezone says what the rectangle holds
RIVER_BANDS = 3              # row bands the Map block describes the river in
ROAD_LINE_MIN = 3            # the Map block's road lines: straight runs of connected road at least this long,
ROAD_LINES = 8               # the longest this many named, then "+N more"
RESERVE_MIN = 3000           # the treasury reserve (mirrors the fleet configs' mayor playbook): this, or 3 months of
                             # expenses (6 in a deficit)
GAP_MIN, GAP_SHARE = 100, 0.1   # a department within max($100, 10% of its plan target) of it counts as funded
SPEND_MONTHS = 3             # the mayor's Departments block: what each department spent over this many months
IDLE_MULT, IDLE_MIN = 2, 1000   # a department keeps max(IDLE_MULT x that, its cheapest ready site, IDLE_MIN); funds
                                # above it by IDLE_MIN or more are idle
USED_SHARE = 0.5             # below its target only if it spent more than this share of the most it held (or never
                             # held what it keeps): DeptMoney.wants_more
HOLD = "hold; lower targets or wait for income"    # the advice when the treasury is at or under the reserve
ROAD_REMOVERS = {"bulldoze"}  # the only actions that remove road tiles (demolish takes buildings, never roads)
# buildings that link the road networks they touch to the region, as World.refresh() counts them (metro stations
# don't) -> their name in a refusal
LINKS = {C.TRAIN: "train station", C.HARBOUR: "harbour", C.AIRPORT: "airport"}
LINK_OWNERS = {r for r, owned in OWNED_STRUCTS.items() if owned & set(LINKS)}     # roles whose demolish takes one
# guarded action -> what its refusal and confirm parameter say the roads would be cut off from
CUT_FROM = {"bulldoze": "highway", "demolish": "region"}

# purpose -> path into build_observation(..., site_hints=True)
SITE_PURPOSES = {
    "mayor": {},
    "planning": {"zone_R": ("zoning_suggestions", "R"), "zone_C": ("zoning_suggestions", "C"),
                 "zone_I": ("zoning_suggestions", "I"), "zone_O": ("zoning_suggestions", "O"),
                 # no mixed-use ranking exists; shops below flats want what shops want (near homes and the centre)
                 "zone_M": ("zoning_suggestions", "C"),
                 "road_extension": ("road_suggestions",), "casino": ("casino_sites",)},
    "utilities": {"coal_or_gas": ("candidate_sites", "far_from_homes"), "wind": ("candidate_sites", "windy"),
                  "pump": ("candidate_sites", "riverbank_next_to_road"),
                  "sewage": ("candidate_sites", "riverbank_for_sewage"),
                  "near_homes": ("candidate_sites", "near_homes"),
                  "riverbank_2x2": ("candidate_sites", "riverbank_2x2")},
    "transport": {"bus_stops": ("transport", "suggested_stops"),
                  "train_station": ("transport", "train_station_sites"),
                  "harbour": ("transport", "harbour_sites"), "metro_station": ("transport", "metro_station_sites"),
                  "congested_roads": ("transport", "congested_roads")},
    "police": {"station": ("police", "candidate_sites"), "jail": ("police", "jail_sites")},
    "fire": {"station": ("fire", "candidate_sites"), "shelter": ("fire", "shelter_sites"),
             "levee": ("fire", "levee_lines")},
    "health": {"clinic": ("health", "clinic_sites"), "hospital": ("health", "hospital_sites"),
               "landfill": ("health", "landfill_sites"),
               # a cemetery serves the whole city (no radius), so it reuses the landfill list: roadside lots away
               # from homes, ranked by uncovered garbage nearby (which doesn't matter for a cemetery)
               "cemetery": ("health", "landfill_sites")},
    "education": {"school": ("education", "school_sites"), "high_school": ("education", "high_school_sites"),
                  "university": ("education", "university_sites"),
                  # a library speeds up schooling in the homes within its radius, so it reuses the school list:
                  # roadside lots near homes no school covers yet
                  "library": ("education", "school_sites")},
}
SITE_UNLOCKS = {"zone_O": ("office_zone",), "zone_M": ("mixed_use",), "casino": ("casino",),
                "riverbank_2x2": ("water_works", "advanced_treatment", "nuclear"), "train_station": ("train_station",),
                "harbour": ("harbour",), "metro_station": ("metro",), "jail": ("jail",), "hospital": ("hospital",),
                "high_school": ("high_school",), "university": ("university",), "library": ("library",)}
# purpose -> (short reason for the briefing, full note for find_sites) when its list is empty
SITE_NOTES = {"levee": ("no flood forecast", "no flood is forecast within 3 months; riverbank stretches to protect "
                        "appear with a flood warning"),
              "congested_roads": ("no road is congested", "no road is congested right now"),
              "bus_stops": ("too few homes and jobs on the roads yet (~1,500 residents)",
                            "too few homes and job centres on the highway-connected roads yet to suggest stops; "
                            "bus lines pay off once the city is bigger (~1,500 residents)")}
# purpose -> the population its list waits for (empty below it, with its SITE_NOTES reason): observation's suggested
# stops have no floor, so from month 1 transport read "can't pay (cheapest: bus_stops $1,500)" and the mayor "needs
# $1,500 for a bus line", against both playbooks (no bus line before ~1,500 residents)
SITE_MIN_POP = {"bus_stops": 1500}
RIVERBANK_PURPOSES = {"pump", "sewage", "riverbank_2x2"}      # empty when no road reaches free riverbank
BANK_TOP_UP = {"pump": "pump", "sewage": "outlet"}   # one-tile riverbank purposes site_lists tops up -> crowd line
RIVER_PURPOSES = RIVERBANK_PURPOSES | {"harbour", "levee"}    # purposes that belong on the river
SCARCE_BANK = 2              # at most this many free riverbank lots by the roads: other purposes leave them alone
# purposes whose lists skip lots with 3+ homes within 4 tiles (observation._far_from_homes): never empty for lack
# of need (the landfill ranking adds 0.01 everywhere, the jail's is flat), only for lack of such a lot
AWAY_FROM_HOMES = {"jail", "landfill", "cemetery"}
# the best site's most useful field for the briefing's Ready sites line, first match wins
SITE_FIELDS = [("wind", "wind {}"), ("homes_within_5", "{} homes within 5"), ("need", "need {}"),
               ("opens_land", "opens {} land tiles"), ("load_vs_capacity", "load {}x capacity"), ("tiles", "{} tiles"),
               ("pollution", "pollution {}")]
SITE_FIELDS_HIDDEN = {"sewage": {"pollution"}}    # the outfall's own water quality doesn't matter
SITE_FIELDS_NONZERO = {"near_homes"}              # "0 homes within 5" says nothing near homes (unlike coal_or_gas)
# purpose -> what its sites are for (tier names, or OFF_TIER_SPECS keys): the cheapest unlocked one prices the site
# (site_price). Planning's purposes aren't priced
SITE_BUILDS = {
    "utilities": {"coal_or_gas": ("coal", "gas"), "wind": ("wind",), "pump": ("pump",), "sewage": ("outlet",),
                  "near_homes": ("tower", "solar", "wind"),
                  "riverbank_2x2": ("water_works", "advanced_treatment", "nuclear")},
    "transport": {"bus_stops": ("bus_line",), "train_station": ("train_station",), "harbour": ("harbour",),
                  "metro_station": ("metro_station",), "congested_roads": ("avenue_tile",)},
    "police": {"station": ("police_station",), "jail": ("jail",)},
    "fire": {"station": ("fire_station",), "shelter": ("shelter",), "levee": ("levee_tile",)},
    "health": {"clinic": ("clinic",), "hospital": ("hospital",), "landfill": ("landfill",), "cemetery": ("cemetery",)},
    "education": {"school": ("elementary_school",), "high_school": ("high_school",), "university": ("university",),
                  "library": ("library",)},
}
# prices outside the tier lines -> (what the action charges, looked up when priced as the actions do, since a balance
# override may replace it; unlock key). A bus line: create_bus_line. Per tile (PER_TILE): build_levee charges each
# levee tile; upgrade_road each street tile it makes an avenue on land (the cheapest class upgrade; x3 on bridges)
OFF_TIER_SPECS = {"train_station": (lambda: C.TRAIN_STATION["cost"], "train_station"),
                  "harbour": (lambda: C.HARBOUR_SPEC["cost"], "harbour"),
                  "metro_station": (lambda: C.METRO_STATION["cost"], "metro"),
                  "bus_line": (lambda: C.BUS_LINE_COST, None),
                  "levee_tile": (lambda: C.CATALOG["levee"]["cost"], None),
                  "avenue_tile": (lambda: C.ROAD_UPGRADE_COST[C.ROAD_CLASSES["avenue"]], "avenue")}
PER_TILE = {"levee_tile", "avenue_tile"}
BRIDGE_UPGRADE = 3           # upgrade_road charges a bridge tile this many times a land tile's upgrade
# priced purposes that are never a department's first build: cheapest() skips them, so the mayor's needs line and the
# budget line name the primary price (their Ready lines keep theirs; affordable, they still keep the budget line
# away, which would be false). At $1,500 each they undercut the first build and were named instead. A library only
# multiplies school graduation (LIBRARY_BOOST): useless without the $2,500 school's seats. A shelter only matters in
# disasters (the playbook adds one past 3,000 residents): smoke v9's mayor read "needs $1,500 for a shelter", funded
# fire that much, and fire never reached the $2,500 station (0% coverage, fire score 31.4). A cemetery comes after
# the $2,000 landfill in the playbook. Train stations, harbours and metro stations are never transport's first build
# either: the playbook wants bus lines first (at 5,000+ residents, $3,000 read "can't pay (cheapest: harbour $10k)"
# with bus stops ready)
SUPPLEMENTARY = {"library", "shelter", "cemetery", "train_station", "harbour", "metro_station"}

ALL_ACTIONS = {a for spec in ROLES.values() for a in spec["actions"]}
# int in the doc example, but the handler reads a float (actions._num)
NUMBER_PARAMS = {("set_tax", "rate"), ("set_funding", "percent"), ("allocate_budget", "amount"),
                 ("take_loan", "amount"), ("repay_loan", "amount"), ("set_fare", "fare"), ("set_district_tax", "delta")}
_JSON_TYPE = {int: "integer", float: "number", str: "string", bool: "boolean", list: "array", dict: "object"}
# build actions -> the tier lines their "kind" can name
_BUILD_LINES = {"build_power_plant": POWER_LINES, "build_water": WATER_LINES, "build_sewage": SEWAGE_LINES,
                "build_school": tuple(SCHOOL_KINDS.values())}
# documented parameters the examples don't show
_EXTRA_PARAMS = {
    "build_park": {k: {"type": "integer"} for k in ("x1", "y1", "x2", "y2")},
    "enact_policy": {"district": {"type": "string", "description": "apply the policy in one district only"}},
    "repeal_policy": {"district": {"type": "string", "description": "repeal it in one district only"}},
    "upgrade_facility": {"tier": {"type": "integer", "description": 'target tier (with module "tier")'}},
}
_TIER_PARAM = {"type": "integer", "description": "tier within the building's line (1 = base); see the description"}


class ToolError(Exception):
    pass


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable
    properties: dict = field(default_factory=dict)
    required: tuple = ()
    strict: bool = True           # reject unknown arguments (action tools pass them to the handler)
    newest_last: bool = False     # trimming keeps the tail of long lists (logs, messages)
    trim_hint: str = ""           # how to ask for less when a result is trimmed

    def schema(self) -> dict:
        params = {"type": "object", "properties": self.properties}
        if self.required:
            params["required"] = list(self.required)
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": params}}


# ------------------------------------------------------------------ helpers
def _py(v):
    """JSON-safe copy: numpy scalars to Python, floats to 2 dp, tuples to lists, keys to str."""
    if isinstance(v, dict):
        return {str(k): _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 2) if math.isfinite(v) else None
    return v


def _size(v) -> int:
    return len(json.dumps(v))


def _fit(res: dict, limit: int = RESULT_CHARS, newest_last: bool = False, hint: str = "") -> dict:
    """Drop items from the biggest lists until the JSON, with its "truncated" note counting them, fits (from the
    end, or from the front for newest-last lists)."""
    note = lambda n: f"{n} {'older' if newest_last else 'more'} list items left out" + (f"; {hint}" if hint else "")
    dropped = 0
    while _size({**res, "truncated": note(dropped)} if dropped else res) > limit:
        lists = []

        def walk(v):
            if isinstance(v, list) and v:
                lists.append(v)
            for x in (v.values() if isinstance(v, dict) else v if isinstance(v, list) else ()):
                walk(x)
        walk(res)
        if not lists:
            break
        big = max(lists, key=_size)
        n = max(1, len(big) // 10)
        if newest_last:
            del big[:n]
        else:
            del big[-n:]
        dropped += n
    if dropped:
        res["truncated"] = note(dropped)
    return res


def _coerce(name: str, v, spec: dict):
    t = spec.get("type")
    bad = lambda what: ToolError(f"argument '{name}' must be {what}, got {v!r}")
    if t in ("integer", "number"):
        if isinstance(v, str):
            try:
                v = float(v.strip())
            except ValueError:
                raise bad("an integer" if t == "integer" else "a number")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise bad("an integer" if t == "integer" else "a number")
        v = int(v) if float(v).is_integer() else v       # fractions pass: handlers round where they need to
    elif t == "string":
        if isinstance(v, bool) or not isinstance(v, (str, int, float)):
            raise bad("a string")
        v = str(v)
    elif t == "boolean":
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            v = v.strip().lower() == "true"
        if not isinstance(v, bool):
            raise bad("true or false")
    elif t in ("array", "object"):
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                pass
        if not isinstance(v, list if t == "array" else dict):
            raise bad("a JSON array" if t == "array" else "a JSON object")
    if "enum" in spec and v not in spec["enum"]:
        raise ToolError(f"argument '{name}' must be one of {', '.join(map(str, spec['enum']))}; got {v!r}")
    return v


def _items_schema(example: list) -> dict:
    """Item schema from a doc example, e.g. stops [[5, 24], ...] -> arrays of integers (models sent {"x", "y"}
    objects to an untyped list)."""
    if not example:
        return {}
    first = example[0]
    if isinstance(first, list):
        return {"type": "array", "items": _items_schema(first) or {}}
    return {"type": _JSON_TYPE[type(first)]}


def _action_tool(name: str, doc: tuple[str, str], fn: Callable, confirm: bool = False) -> Tool:
    """The tool for action `name`, with a confirm parameter when the action is guarded for the role (guarded())."""
    example, text = doc
    props = {}
    for k, v in json.loads(example).items():
        if k != "type":
            t = "number" if (name, k) in NUMBER_PARAMS else _JSON_TYPE[type(v)]
            props[k] = {"type": t, **({"items": _items_schema(v)} if isinstance(v, list) else {})}
    lines = _BUILD_LINES.get(name) or ((C.TIER_BY_NAME[name[6:]][0],) if name[6:] in C.TIER_BY_NAME else ())
    if name.startswith("build_") and any(len(C.TIER_LINES[ln]) > 1 for ln in lines):
        props["tier"] = dict(_TIER_PARAM)
    props.update(_EXTRA_PARAMS.get(name, {}))
    if confirm:
        props["confirm"] = {"type": "boolean", "description": f"true only to repeat a {name} refused for cutting "
                            f"roads off the {CUT_FROM[name]}, once you are sure"}
    return Tool(name, text, fn, props, strict=False)


def _site(s) -> dict:
    s = {"x": s[0], "y": s[1]} if isinstance(s, (list, tuple)) else dict(s)
    if "x" not in s:                           # a rectangle or line: its first corner
        s = {"x": s["x1"], "y": s["y1"], **s}
    s.pop("network", None)
    return s


def _on_connected_road(w, s) -> bool:
    """False for a hint that names a road network off the highway (the riverbank lists take any network): a
    building there would get no power, water or customers."""
    net = s.get("network") if isinstance(s, dict) else None
    return net is None or bool(w.networks[net].outside_connected)


def _distance(s: dict, near: tuple[int, int]) -> int:
    nx, ny = near
    cx = min(max(nx, s.get("x1", s["x"])), s.get("x2", s["x"])) if "x2" in s else s["x"]
    cy = min(max(ny, s.get("y1", s["y"])), s.get("y2", s["y"])) if "y2" in s else s["y"]
    return abs(cx - nx) + abs(cy - ny)


def _connected_roads(w) -> np.ndarray:
    """Road tiles on networks connected to the highway."""
    labels = [i for i, n in enumerate(w.networks) if n.outside_connected]
    return np.isin(w.road_labels, labels) if labels else np.zeros(w.struct.shape, bool)


def _outside_labels(w, labels: np.ndarray, drop=frozenset()) -> set[int]:
    """The road components linked to the region, as World.refresh() finds them: those holding a highway tile, and
    those next to (8-neighbourhood) a train station, harbour or airport (LINKS) not anchored at a tile in `drop`."""
    out = {int(labels[y, x]) for y, x in zip(*np.nonzero(w.struct == C.HIGHWAY))} - {-1}
    for x, y in (a for st in LINKS for a in w.facilities(st) if a not in drop):
        for px, py in w.parts(x, y):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if w.in_bounds(px + dx, py + dy) and labels[py + dy, px + dx] >= 0:
                        out.add(int(labels[py + dy, px + dx]))
    return out


def cut_off(w, tiles, drop=frozenset()) -> np.ndarray:
    """The highway-connected road tiles (the removed ones excluded) that removing the road tiles among `tiles`, and
    the link buildings anchored at the tiles in `drop`, would cut off the region: the road network relabelled
    without them as World.refresh() labels it (4-neighbour components, linked by a highway tile or a station /
    harbour / airport)."""
    before = _connected_roads(w)
    gone = np.zeros(before.shape, bool)
    for x, y in tiles:
        gone[y, x] = w.struct[y, x] == C.ROAD
    if not (gone & before).any() and not drop:
        return np.zeros_like(before)
    labels, _ = label_components(((w.struct == C.ROAD) | (w.struct == C.HIGHWAY)) & ~gone)
    return before & ~gone & ~np.isin(labels, list(_outside_labels(w, labels, drop)))


def guarded(role: str, action: str) -> bool:
    """Whether `role`'s `action` is refused when it would cut roads off (confirm: true lets it run): a bulldoze, and
    a demolish by a role that can take a train station, harbour or airport (transport; LINK_OWNERS)."""
    return action in ROAD_REMOVERS or (action == "demolish" and role in LINK_OWNERS)


def _touching(mask: np.ndarray, diagonal: bool = True) -> np.ndarray:
    """Tiles with a neighbour in `mask`: 8-neighbourhood (as `adjacent_network` and the build actions check road
    access), or 4 (as `is_riverbank` checks water)."""
    out = np.zeros_like(mask)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dx or dy) and (diagonal or not (dx and dy)):
                out |= shift(mask, dy, dx, False)
    return out


def _empty_land(w) -> np.ndarray:
    return (w.struct == C.NONE) & (w.terrain == C.LAND)


def _roadside(w) -> np.ndarray:
    """Empty land tiles touching a highway-connected road: where a new building could go."""
    return _touching(_connected_roads(w)) & _empty_land(w)


def _free_roadside(w) -> int:
    return int(_roadside(w).sum())


def _riverbank_by_roads(w) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Riverbank land tiles (next to water, as the build actions check `is_riverbank`) that touch a
    highway-connected road, road tiles excluded: the lots a pump or outlet could use. And where those roads meet
    the river: bridgeheads, else their tiles on the bank (roads along it), else the bank lots beside them (a road
    passing the water diagonally); empty when no connected road reaches the river."""
    road, water = _connected_roads(w), w.terrain == C.WATER
    bank = (w.terrain == C.LAND) & _touching(water, diagonal=False)
    lots = bank & _touching(road) & (w.struct != C.ROAD) & (w.struct != C.HIGHWAY)
    heads = road & bank & _touching(road & water, diagonal=False)
    contact = next((m for m in (heads, road & bank) if m.any()), lots)
    ys, xs = np.nonzero(contact)
    return lots, [(int(x), int(y)) for y, x in zip(ys, xs)]


def free_bank(w) -> np.ndarray:
    """The free riverbank lots beside highway-connected roads: where a pump or outlet can go."""
    return _riverbank_by_roads(w)[0] & _empty_land(w)


def _bank_needed(w) -> list[str]:
    """What the city still lacks that takes a riverbank lot: a pump (neither a pump nor a water tower yet; a tower
    supplies water without the bank) and a sewage outlet (neither an outlet nor a treatment plant)."""
    return [name for name, structs in (("pump", (C.PUMP, C.TOWER)), ("sewage outlet", (C.OUTLET, C.TREATMENT)))
            if not any(w.facilities(s) for s in structs)]


def _ends(points: list) -> list:
    """The first and last point (reading order): a bridge's two ends, a riverside road's two ends."""
    return points[:1] + points[-1:] if len(points) > 1 else points


def _box(mask) -> tuple[list[int], list[int]] | None:
    ys, xs = np.nonzero(mask)
    return None if not len(xs) else ([int(xs.min()), int(xs.max())], [int(ys.min()), int(ys.max())])


def _span(a: int, b: int) -> str:
    return str(a) if a == b else f"{a}-{b}"


def _some(items: list[str], n: int, sep: str = ", ") -> str:
    """The first n items, and how many more there are."""
    return sep.join(items[:n]) + (f" +{len(items) - n} more" if len(items) > n else "")


def _roads(w) -> dict | None:
    """Extent and size of the highway-connected road network."""
    roads = _connected_roads(w)
    box = _box(roads)
    return box and {"x": box[0], "y": box[1], "tiles": int(roads.sum())}


def _roads_text(r: dict | None) -> str:
    return (f"highway-connected roads: x {_span(*r['x'])}, y {_span(*r['y'])}, {r['tiles']} tiles" if r else
            "no roads connected to the highway yet")


def _road_lines(w) -> list[dict]:
    """The maximal straight runs of highway-connected road, along rows ({"y", "x": [x0, x1]}) and down columns
    ({"x", "y": [y0, y1]}), of at least ROAD_LINE_MIN tiles: the highway tile, bridges and avenues count, and a
    crossing is in both its lines. Longest first, then by the first tile's y and x, rows before columns. Smoke
    v9-v11: given only the network's box (x 0-25, y 19-29; really y=19, y=24 and x=19), planning rebuilt existing
    road 13 times and made 5.4 inspect calls a turn."""
    roads = _connected_roads(w)
    runs = []
    for down, grid in ((False, roads), (True, roads.T)):
        for i, row in enumerate(grid):
            edges = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))))
            for a, b in zip(edges[::2].tolist(), (edges[1::2] - 1).tolist()):      # each run's first and last tile
                if b - a + 1 >= ROAD_LINE_MIN:
                    y, x = (a, i) if down else (i, a)
                    runs.append(((a - b, y, x, down), {"x": i, "y": [a, b]} if down else {"y": i, "x": [a, b]}))
    return [ln for _, ln in sorted(runs, key=lambda r: r[0])]


def _road_line_text(ln: dict) -> str:
    return f"y={ln['y']} x {_span(*ln['x'])}" if isinstance(ln["x"], list) else f"x={ln['x']} y {_span(*ln['y'])}"


def _on_road_line(ln: dict, x: int, y: int) -> bool:
    return (y == ln["y"] and ln["x"][0] <= x <= ln["x"][1]) if isinstance(ln["x"], list) else (
        x == ln["x"] and ln["y"][0] <= y <= ln["y"][1])


def _contents(w, tiles: list[tuple[int, int]]) -> str:
    """What the tiles hold by kind, largest first, the first CONTENT_KINDS kinds then "+N other tiles": road (bridges
    too), highway, zoned <letter> (with how many are developed), river, empty land, and any other structure by name
    (tiles, so a 2x2 hospital is "4 hospital"). "all 14 tiles are road", "10 zoned R (6 developed), 4 road"."""
    n, dev = Counter(), Counter()
    for x, y in tiles:
        s = int(w.struct[y, x])
        k = (f"zoned {C.LETTER_BY_ZONE[s]}" if s in C.ZONES else C.STRUCT_NAME[s].replace("_", " ") if s != C.NONE
             else "river" if w.terrain[y, x] == C.WATER else "empty land")
        n[k] += 1
        dev[k] += bool(s in C.ZONES and w.level[y, x] > 0)
    name = lambda k: k + (f" ({dev[k]} developed)" if dev[k] else "")
    if len(n) == 1:
        (k, c), = n.items()
        bare = k in ("road", "highway", "river", "empty land") or k.startswith("zoned ")
        return f"all {c} tiles are {name(k)}" if c > 1 else f"the tile is {name(k) if bare else a_or_an(k)}"
    kinds = n.most_common()
    rest = sum(c for _, c in kinds[CONTENT_KINDS:])
    return ", ".join(f"{c} {name(k)}" for k, c in kinds[:CONTENT_KINDS]) + (
        f" +{rest} other tile{'s' * (rest > 1)}" if rest else "")


def _zone_beside(w, tiles: list[tuple[int, int]]) -> list[str]:
    """Zone rects for the free roadside land (empty land touching a highway-connected road) on the lines just outside
    a rectangle's sides (`tiles` in reading order, as _rect gives them), over its span: each trimmed to its first and
    last such tile (one line, so always a valid zone), kept with 2 such tiles or half its side (a 1-tile side only
    beside a 1-tile rectangle, so a row's end caps aren't offered as single lots), the most tiles first (ties: north,
    west, east, south). While a pump or outlet is missing (_bank_needed) the free riverbank lots are left out, and a
    side whose rect would still cover one is dropped (v7: zoning took the last bank lots and utilities had nowhere to
    put them)."""
    (xa, ya), (xb, yb) = tiles[0], tiles[-1]
    free, sides = _roadside(w), []
    bank = free_bank(w) if _bank_needed(w) else np.zeros_like(free)
    free &= ~bank
    for side in ([(x, ya - 1) for x in range(xa, xb + 1)], [(xa - 1, y) for y in range(ya, yb + 1)],
                 [(xb + 1, y) for y in range(ya, yb + 1)], [(x, yb + 1) for x in range(xa, xb + 1)]):
        hits = [(x, y) for x, y in side if w.in_bounds(x, y) and free[y, x]]
        if hits and (len(hits) >= 2 or 2 * len(hits) >= len(side) > 1 or len(tiles) == 1):
            (x1, y1), (x2, y2) = hits[0], hits[-1]                 # along the side, so x1 <= x2 and y1 <= y2
            if not bank[y1:y2 + 1, x1:x2 + 1].any():
                sides.append((-len(hits), f"x1={x1},y1={y1},x2={x2},y2={y2}"))
    return [r for _, r in sorted(sides, key=lambda s: s[0])[:ZONES_BESIDE]]


def _river_cols(w) -> tuple[int, int, int, int] | None:
    """(x0, x1, y0, y1): the river's columns within the rows the connected roads span (the whole map before
    there are roads, or when the river misses those rows); None without a river."""
    road_rows = np.nonzero(_connected_roads(w).any(axis=1))[0]
    y0, y1 = (int(road_rows.min()), int(road_rows.max())) if len(road_rows) else (0, w.h - 1)
    cols = np.nonzero((w.terrain[y0:y1 + 1] == C.WATER).any(axis=0))[0]
    if not len(cols):
        cols = np.nonzero((w.terrain == C.WATER).any(axis=0))[0]
        y0, y1 = 0, w.h - 1
    return (int(cols.min()), int(cols.max()), y0, y1) if len(cols) else None


def _river_span(w) -> str | None:
    """Where the river runs, within the rows the connected roads span; None without a river."""
    r = _river_cols(w)
    return r and f"the river runs at x {r[0]}-{r[1]} in rows {r[2]}-{r[3]}"


def _edge(w, x: int, y: int) -> str:
    return "west" if x == 0 else "east" if x == w.w - 1 else "north" if y == 0 else "south" if y == w.h - 1 else "inner"


def map_facts(w) -> dict:
    """Orientation for the briefing's Map block and map_overview: size, highway entries, the river per row band
    (RIVER_BANDS; each [[x0, x1, first row], [x0, x1, last row]], the columns at the band's ends), the
    highway-connected roads' extent and straight lines (_road_lines; map_lines names ROAD_LINES), the road tiles off
    the highway network (None when there are none) and the built-up box."""
    hw = [{"x": int(x), "y": int(y), "edge": _edge(w, x, y)} for y, x in zip(*np.nonzero(w.struct == C.HIGHWAY))]
    water = w.terrain == C.WATER
    rows = np.nonzero(water.any(axis=1))[0]
    at = lambda y: [int(np.nonzero(water[y])[0].min()), int(np.nonzero(water[y])[0].max()), int(y)]
    river = [[at(band[0]), at(band[-1])] for band in np.array_split(rows, min(RIVER_BANDS, len(rows)))
             if len(band)] if len(rows) else []
    zoned = np.isin(w.struct, list(C.ZONES))
    built = (w.struct != C.NONE) & (w.struct != C.ROAD) & (w.struct != C.HIGHWAY)
    flat = np.arange(w.h * w.w).reshape(w.h, w.w)
    anchors = built & ~zoned & ((w.anchor < 0) | (w.anchor == flat))       # multi-tile buildings count once
    bb = _box(built)
    loose = (w.struct == C.ROAD) & ~_connected_roads(w)
    lb = _box(loose)
    return {"size": [w.w, w.h], "highway": hw, "river": river, "roads": _roads(w), "road_lines": _road_lines(w),
            "unconnected": lb and {"x": lb[0], "y": lb[1], "tiles": int(loose.sum())},
            "built": bb and {"x": bb[0], "y": bb[1], "zoned": int(zoned.sum()),
                             "developed": int((zoned & (w.level > 0)).sum()), "buildings": int(anchors.sum())}}


MAP_PARTS = ("size", "highway", "river", "roads", "road_lines", "unconnected", "built")
# the mayor's Map block: how far the roads reach, roads cut off, the built-up area. Size, highway entry, river and
# road lines made room for the departments' spending and idle money: with them the worst case (5 departments idle)
# took 7,737 of the 7,450 characters test_briefings_stay_bounded_at_the_limits allows; without, 7,369
MAYOR_MAP = ("roads", "unconnected", "built")


def map_lines(f: dict, parts: tuple = MAP_PARTS) -> list[str]:
    """The map_facts lines, only `parts` (MAP_PARTS keys, in MAP_PARTS order)."""
    W, H = f["size"]
    end = lambda e: f"x {_span(e[0], e[1])} at row {e[2]}"
    b = f["built"]
    lines = {
        "size": [f"- {W}x{H} tiles: x = column 0-{W - 1} (0 = west), y = row 0-{H - 1} (0 = north)"],
        "highway": ["- highway entry: " + (", ".join(f"({h['x']},{h['y']}) on the {h['edge']} edge"
                                                     for h in f["highway"]) or "none")],
        "river": ["- river: " + ", ".join(end(t) + (f" to {end(e)}" if e != t else "") for t, e in f["river"])
                  if f["river"] else "- no river"],
        "roads": ["- " + _roads_text(f["roads"])],
        "road_lines": ["- road lines: " + _some([_road_line_text(ln) for ln in lns], ROAD_LINES, "; ")]
                      if (lns := f.get("road_lines")) else [],
        "unconnected": [f"- roads not connected to the highway: {u['tiles']} tiles (x {_span(*u['x'])}, y "
                        f"{_span(*u['y'])}) - reconnect them"] if (u := f.get("unconnected")) else [],
        "built": [f"- built-up area: x {_span(*b['x'])}, y {_span(*b['y'])} ({b['zoned']} zoned tiles, "
                  f"{b['developed']} developed, {b['buildings']} building{'s' * (b['buildings'] != 1)})"
                  if b else "- built-up area: none yet"]}
    return [ln for p in MAP_PARTS if p in parts for ln in lines[p]]


TIER_NOTE = " (higher tiers keep the letter)."


def _legend(chars: set) -> str:
    """MAP_LEGEND cut to the characters on the map (the tier note moves to map_overview's "read")."""
    items = MAP_LEGEND.removeprefix("Legend: ").replace(TIER_NOTE, "").split("  ")
    return "Legend: " + "  ".join(i for i in items if set(i.split(" ", 1)[0].split("/")) & chars)


def _ascii(w, k: int) -> list[str]:
    """world.ascii_map() rows (column rulers, row numbers), or one character per k x k block: the highway entry,
    else its most common feature (ties: the first in reading order), river only where the block has nothing else."""
    if k == 1:
        return w.ascii_map().split("\n")
    xs, ys = range(0, w.w, k), range(0, w.h, k)
    rows = ["    " + "".join(str((x // 10) % 10) for x in xs), "    " + "".join(str(x % 10) for x in xs)]
    for y in ys:
        line = []
        for x in xs:
            block = [w.tile_char(tx, ty) for ty in range(y, min(y + k, w.h)) for tx in range(x, min(x + k, w.w))]
            main = ([c for c in block if c == "H"] or [c for c in block if c not in ".~"]
                    or [c for c in block if c == "~"] or ["."])
            line.append(Counter(main).most_common(1)[0][0])
        rows.append(f"{y:>3} " + "".join(line))
    return rows


def map_overview(w, limit: int = RESULT_CHARS, parts: tuple = MAP_PARTS) -> dict:
    """The whole map as ASCII rows with the map_facts summary, downsampled until the result fits `limit`. At full
    size the summary goes first: it shrinks to what the role's Map block (`parts`) leaves out, then to that without
    the road lines (the mayor's leaves out size, highway, river and road lines: MAYOR_MAP)."""
    facts = map_facts(w)
    summary = [ln[2:] for ln in map_lines(facts)]
    left = [p for p in MAP_PARTS if p not in parts]
    note = ["the rest: see the Map block in your briefing" if left else "see the Map block in your briefing"]
    short = [[ln[2:] for ln in map_lines(facts, ps)] + note
             for ps in dict.fromkeys([tuple(left), tuple(p for p in left if p != "road_lines")])]
    read = "top two rows: x (tens, units); rows start with y; higher tiers keep the letter"
    for k, lines in [(1, summary)] + [(1, s) for s in short] + [(k, summary) for k in range(2, max(w.w, w.h) + 1)]:
        grid = _ascii(w, k)
        body = {c for row in grid[2:] for c in row[4:]}
        res = {"ok": True, "summary": lines, "legend": _legend(body), "read": read, "map": grid}
        if k > 1:
            res["scale"] = f"each character is a {k}x{k} block, shown at its top-left x, y; inspect for detail"
        if _size(res) <= limit:
            break
    return res


def _plant(p: dict) -> dict:
    return {k: v for k, v in p.items() if k not in ("line", "neighbours")}


def overview(env) -> dict:
    """Headline metrics, finances and scores, with the change over the last 3 months (`trend`, None early on)."""
    w = env.world
    m = w.metrics()
    last = w.history[-1] if w.history else m
    pct = lambda v: None if v is None else round(100 * v, 1)
    out = {"ok": True, "month": w.month, "months_left": env.cfg.months - w.month,
           "population": m["population"], "happiness_pct": pct(m["happiness"]),
           "approval_pct": pct(m["approval"]), "unemployment_pct": pct(m["unemployment"]),
           "jobs": {"commercial": m["jobs_com"], "industrial": m["jobs_ind"], "offices": m["jobs_off"]},
           "treasury": round(w.treasury), "debt": round(w.debt),
           "last_month": {"income": round(last["income"]), "expenses": round(last["expenses"])},
           "department_funds": {k: round(v) for k, v in w.dept_funds.items()},
           "tax_pct": {k: round(100 * v, 1) for k, v in w.taxes.items()},
           "demand": {k: round(v, 2) for k, v in (w.demand or {}).items()},
           **{k: {"supply": m[f"{k}_supply"], "demand": m[f"{k}_demand"], "coverage_pct": pct(m[f"{k}_coverage"])}
              for k in ("power", "water", "sewage")},
           "coverage_pct": {k: pct(m[f"{k}_coverage"]) for k in ("police", "fire", "health", "garbage", "school")},
           "scores": {k: round(v, 1) for k, v in env.last_scores.items()}}
    past = next((h for h in w.history if h["month"] >= w.month - 3), None)
    out["trend"] = None if past is None else {
        "since_month": past["month"], "population": m["population"] - past["population"],
        "treasury": round(w.treasury - past["treasury"]),
        "happiness_pct": round(100 * (m["happiness"] - past["happiness"]), 1)}
    return out


def plan_slice(fleet: FleetState, role: str) -> dict:
    """The whole plan for the mayor; goals, notes and the role's budget target and priority for a department."""
    p = fleet.plan
    if not fleet.plan_versions:
        return {"ok": True, "plan": None, "note": "no plan yet" + (" - write one with update_plan"
                                                                  if role == "mayor" else "")}
    if role == "mayor":
        return {"ok": True, "version": len(fleet.plan_versions), **p}
    out = {"ok": True, "month": p["month"], "goals": p["goals"], "budget": p["budgets"].get(role),
           "priority": p["priorities"].get(role)}
    if p["notes"]:
        out["notes"] = p["notes"]
    return out


def reserve(w) -> float:
    """The treasury to keep back: RESERVE_MIN or last month's expenses x 3 (x 6 while running a deficit)."""
    if not w.history:
        return RESERVE_MIN
    last = w.history[-1]
    return max(RESERVE_MIN, (3 if last["income"] >= last["expenses"] else 6) * last["expenses"])


class DeptMoney:
    """What a department holds and spent, and how much of it is idle: it keeps max(IDLE_MULT x its spending over the
    last SPEND_MONTHS months, its dearest ready first building (first_build), IDLE_MIN), and the rest, when it is
    IDLE_MIN or more, is idle. Smoke v9-v12: planning held $7,000-10,700 unspent all game in 7 of 8 runs,
    spending about $1,000 a quarter, while the mayor took loans in 6 and education got no school in 5. Money moved to
    it this month doesn't count: the mayor acts first, so the rest sat through at least one of its turns (at month 0
    none has had one). Its observation (`build`) is built once, on first use, and only asked for when a site's price
    can matter: funds short of its dearest priced purpose, a per-tile purpose (a stretch has no such bound), or a
    price above what its spending keeps."""

    def __init__(self, env, dept: str):
        w = env.world
        self.env, self.dept = env, dept
        self.funds = w.dept_funds[dept]
        window = w.history[-SPEND_MONTHS:]
        self.months = len(window)               # 0 before the first month: nothing shown, nothing idle
        self.spent = sum(h["dept_spent"].get(dept, 0) for h in window)
        # the most it had in one of those months: month-end funds plus that month's spending (the mayor moves money
        # first), so a station bought in the month counts as held
        self.held_max = max((h["dept_funds"].get(dept, 0) + h["dept_spent"].get(dept, 0) for h in window), default=0)
        # without this month's top-ups; the ledger nets the month, so a take-back and a top-up in one turn can leave
        # part of the top-up counted as idle (rare)
        self.held = self.funds - max(0.0, w.ledger["allocated"].get(dept, 0.0))
        self.floor = max(IDLE_MULT * self.spent, IDLE_MIN)         # what it keeps whatever its sites cost
        priced = [pr for p in SITE_PURPOSES[dept] if (pr := site_price(w, dept, p))]
        self.bound = None if any(pr.per_tile for pr in priced) else max((pr.cost for pr in priced), default=0)

    @cached_property
    def build(self) -> FirstBuild | None:
        return first_build(self.env, self.dept)

    @cached_property
    def keep(self) -> float:
        if self.bound is not None and self.bound <= self.floor:
            return self.floor
        return max(self.floor, self.build.dearest if self.build else 0)

    @cached_property
    def idle(self) -> int:
        """Funds above what it keeps, rounded down, when IDLE_MIN or more; else 0."""
        if not self.months or self.held - self.floor < IDLE_MIN:       # not idle whatever its sites cost
            return 0
        idle = math.floor(self.held - self.keep)
        return idle if idle >= IDLE_MIN else 0

    @property
    def needs(self) -> FirstBuild | None:
        """Its cheapest ready site when its funds are short of it (the mayor's needs line), else None."""
        if self.bound is not None and not short_of(self.funds, self.bound):
            return None
        return self.build if self.build and short_of(self.funds, self.build.cost) else None

    @property
    def wants_more(self) -> bool:
        """Whether a top-up would be used: before the first month always; then only when it used more than USED_SHARE
        of the most it held over the last SPEND_MONTHS months, or never held what it keeps (its first build).
        After a take-back planning held its keep, 2 x its spending, below its $9,000 target, and read "below
        target" again: topped up, then idle, then taken back. More than, not at least: a take-back leaves exactly
        1 / IDLE_MULT = USED_SHARE used over the next quarter (test_a_take_back_starts_no_top_up_cycle)."""
        return (not self.months or self.spent > USED_SHARE * self.held_max
                or self.held_max < self.floor or self.held_max < self.keep)     # floor first: keep may need a build


class Funding(NamedTuple):
    gaps: dict[str, dict]   # {dept: {"funds", "target", "gap"}}, in turn order; idle departments left out
    total: int              # the gaps summed
    reserve: float          # reserve(w)
    spare: float            # treasury above the reserve (negative below it)
    idle: dict[str, int]    # {dept: idle funds} (DeptMoney), in turn order
    depts: dict[str, DeptMoney]     # every department in the game

    @property
    def short(self) -> bool:
        return self.total > self.spare

    @property
    def hold(self) -> bool:
        """The treasury is at or under the reserve: nothing to allocate (HOLD says what to do instead)."""
        return self.spare <= 0

    def idle_note(self) -> str:
        """The idle money and what to do with it, "" without any."""
        return (f"{money(sum(self.idle.values()))} idle in {', '.join(self.idle)}: take it back (negative "
                "allocate_budget) before loans or new top-ups") if self.idle else ""


def funding_gaps(env, fleet: FleetState) -> Funding:
    """Departments in the game whose funds are below their plan budget target by more than max(GAP_MIN, GAP_SHARE x
    target) and that would use a top-up (DeptMoney.wants_more), and whether the treasury above the reserve covers
    them: the update_plan result and the mayor's briefing show this, with the idle money. An idle department is no
    gap: smoke v12 seed 0 read "planning: funds $7,834 < target $9,000" and topped it up to $10,721."""
    w, budgets = env.world, fleet.plan["budgets"]
    depts = {d: DeptMoney(env, d) for d in DEPARTMENTS if d in env.roles}
    idle = {d: m.idle for d, m in depts.items() if m.idle}
    gaps = {}
    for d, m in depts.items():
        target = budgets.get(d)
        if (target is not None and d not in idle and target - m.funds >= max(GAP_MIN, GAP_SHARE * target)
                and m.wants_more):
            gaps[d] = {"funds": round(m.funds), "target": round(target), "gap": round(target - m.funds)}
    keep = reserve(w)
    return Funding(gaps, sum(g["gap"] for g in gaps.values()), keep, w.treasury - keep, idle, depts)


class SiteList(NamedTuple):
    sites: list[dict]       # ranked, as build_observation lists them (find_sites sorts by distance and cuts)
    short: str | None       # why the list is empty or unusable (locked), in a few words: the briefing's line
    note: str | None        # the same in full, with what to do: find_sites' "note"


def role_obs(env, role: str, site_hints: bool = True) -> dict:
    """`role`'s observation for the fleet's look-ups and briefings: no map, council messages or memory; with the
    site hints site_lists reads unless site_hints=False."""
    return build_observation(env.world, role, council_msgs=[], memory={}, scores={}, months_total=env.cfg.months,
                             include_map=False, site_hints=site_hints)


def site_lists(w, role: str, obs: dict) -> dict[str, SiteList]:
    """Every find_sites purpose of `role` from one build_observation(..., site_hints=True): the find_sites tool
    and the briefing's Ready sites block both read this, so they can't disagree."""
    out = {}
    for purpose, path in SITE_PURPOSES[role].items():
        found = obs
        for key in path:
            found = found.get(key) if isinstance(found, dict) else None
        sites = [_site(s) for s in (found or []) if _on_connected_road(w, s)]
        if purpose in BANK_TOP_UP and len(sites) < len(found or []):
            sites += _bank_top_up(w, sites, BANK_TOP_UP[purpose])
        if purpose == "congested_roads":         # avenues already last (kept): an avenue upgrade charges nothing there
            sites.sort(key=lambda s: not site_tiles(w, purpose, s)[0])
        keys = SITE_UNLOCKS.get(purpose, ())
        if keys and not any(w.unlocked(k) for k in keys):
            key = min(keys, key=C.UNLOCK_POP.get)
            short, note = f"locked (unlocks at {C.UNLOCK_POP[key]:,} residents)", "locked: " + w.unlock_note(key)
        elif w.metrics()["population"] < SITE_MIN_POP.get(purpose, 0):
            sites, (short, note) = [], SITE_NOTES[purpose]
        elif sites:
            short = note = None
        else:                    # models told to "inspect the map" scanned it blindly for whole turns
            short, note = SITE_NOTES.get(purpose) or _why_empty(w, role, purpose)
        out[purpose] = SiteList(sites, short, note)
    return out


def _bank_top_up(w, listed: list[dict], line: str) -> list[dict]:
    """Free riverbank lots beside highway-connected roads (where a pump or outlet can go) not in `listed`, as many as
    fit in MAX_SITES with it: the observation cuts its riverbank lists to 6 before site_lists drops the sites off
    the highway network, so off-highway roads by the bank could leave pump and sewage short, or empty ("all
    taken"), with free lots beside the connected roads. Ranked as observation._candidate_sites ranks them for
    `line` (pump: cleanest first, +0.5 at or south of an outlet, then north first; outlet: those at or north of a
    pump last, then south first; crowded spots last, marked "crowd"), with pollution as it computes it."""
    have = {(s["x"], s["y"]) for s in listed}
    ys, xs = np.nonzero(free_bank(w))
    more = [{"x": x, "y": y, "pollution": round(max(float(w.pollution[y, x]), w._river_near(x, y)), 2)}
            for x, y in zip(xs.tolist(), ys.tolist()) if (x, y) not in have]
    if line == "pump":
        outlets = w.facilities(C.OUTLET) + w.facilities(C.TREATMENT)
        more.sort(key=lambda b: (b["pollution"] + (0.5 if any(oy <= b["y"] for _, oy in outlets) else 0), b["y"]))
    else:
        pumps = w.facilities(C.PUMP)
        more.sort(key=lambda b: (any(py >= b["y"] for _, py in pumps), -b["y"]))
    k = MAX_SITES - len(listed)
    return _uncrowded_first(w, more, line, k) if k > 0 else []


def _why_empty(w, role: str, purpose: str) -> tuple[str, str]:
    """(short, full): why a site list is empty and what to do: no free roadside land (roads must grow), no road
    at the river or its bank lots taken, no free roadside lot away from homes (AWAY_FROM_HOMES purposes), or
    nothing needs the building yet (the lists only rank spots near uncovered homes / jobs)."""
    roads = lambda: _roads_text(_roads(w))
    if purpose == "road_extension":
        return ("no straight extension from a dead end: build_road from any road into free land",
                "no straight extension from a dead end into open land; build_road from any road tile into free "
                f"land (L-shaped roads work, bridges cross the river; {roads()})")
    if purpose in RIVERBANK_PURPOSES:
        river = _river_cols(w)
        if river is None:
            return "no river on this map", "this map has no river"
        lots, contact = _riverbank_by_roads(w)
        near = ", ".join(f"({x},{y})" for x, y in _ends(contact))
        if not contact:
            return (f"no connected road reaches the river yet: ask planning for a road to it (river at x "
                    f"{_span(river[0], river[1])})",
                    f"no free riverbank tile next to a highway-connected road: a road has to reach the river first "
                    f"({_river_span(w)}); ask planning (send_message) for a road to the riverbank, and build what "
                    "doesn't need the river meanwhile")
        free = [(int(x), int(y)) for y, x in zip(*np.nonzero(lots & _empty_land(w)))]
        if purpose == "riverbank_2x2" and free:
            return (f"no free 2x2 riverbank block by a road: ask planning to leave one near {near}",
                    f"no free 2x2 block on the riverbank beside a highway-connected road (free bank lots are single "
                    f"tiles, e.g. ({free[0][0]},{free[0][1]})); ask planning (send_message) to leave a 2x2 riverbank "
                    f"lot beside the road near {near}")
        return (f"riverbank beside roads all taken: ask planning to free a lot near {near}",
                "the river bank tiles next to highway-connected roads are all taken (zoned or built): ask planning "
                f"(send_message) to leave or free (dezone) a riverbank lot beside the road (e.g. near {near}, where "
                "the road meets the river), and build what doesn't need the river meanwhile")
    free = _free_roadside(w)
    if role == "planning":
        return ("no free roadside strip: extend the roads (find_sites road_extension)",
                f"no free strip along a highway-connected road ({roads()}; {free} free roadside tiles); "
                "extend the roads first (find_sites road_extension lists dead ends to build from)")
    if not free:
        return ("no free roadside land: ask planning for a lot",
                f"no free land next to a highway-connected road ({roads()}); ask planning (send_message) to extend a "
                "road or leave an empty lot by one")
    if purpose in AWAY_FROM_HOMES and not (_roadside(w) & ~_far_from_homes(w)).any():
        return ("no roadside lot away from homes (3+ within 4 tiles): ask planning for one at the town edge",
                f"all {free} free roadside tiles have 3 or more homes within 4 tiles, and a {purpose} must stay away "
                "from homes; ask planning (send_message) to leave an empty lot by a road at the edge of town, away "
                f"from the houses, or to extend a road into open land ({roads()})")
    return ("nothing needs it right now",
            f"nothing needs this right now: no uncovered homes or jobs near the {free} free roadside tiles "
            "(check coverage in city_overview / my_facilities before building)")


def _brief(v) -> str:
    return f"{round(v):,}" if abs(v) >= 10 else f"{round(float(v), 2):g}"


def _where(s: dict) -> str:
    return f"({s['x1']},{s['y1']})-({s['x2']},{s['y2']})" if "x2" in s else f"({s['x']},{s['y']})"


def _extra(purpose: str, s: dict) -> str:
    for k, fmt in SITE_FIELDS:
        if k in s and k not in SITE_FIELDS_HIDDEN.get(purpose, ()):
            return "" if purpose in SITE_FIELDS_NONZERO and not s[k] else ", " + fmt.format(_brief(s[k]))
    return ""


def scarce_bank(w) -> set[tuple[int, int]]:
    """The free riverbank lots beside highway-connected roads (where a pump or outlet can go) when there are at most
    SCARCE_BANK of them, else none: the Ready sites line of another purpose whose site is one of them says to keep
    it for pump / sewage."""
    ys, xs = np.nonzero(free_bank(w))
    return {(int(x), int(y)) for y, x in zip(ys, xs)} if len(xs) <= SCARCE_BANK else set()


def _price(w, name: str) -> float | None:
    """What `name` (a tier name or an OFF_TIER_SPECS key) costs: what its action charges. None while it is locked."""
    if name in OFF_TIER_SPECS:
        cost, key = OFF_TIER_SPECS[name]
        return cost() if key is None or w.unlocked(key) else None
    line, tier = C.TIER_BY_NAME[name]
    spec = C.TIER_LINES[line][tier - 1]
    return None if spec.get("unlock") and not w.unlocked(spec["unlock"]) else spec["cost"]


def building_name(name: str) -> str:
    """A SITE_BUILDS name for people: a line's base tier by its struct's name (tower -> water tower, outlet -> sewage
    outlet), a higher tier by its own (water works: the struct is the line's), the others as they are (bus line)."""
    if name in C.TIER_BY_NAME:
        line, tier = C.TIER_BY_NAME[name]
        name = C.STRUCT_NAME[C.TIER_LINES[line][0]["struct"]] if tier == 1 else name
    return name.replace("_", " ")


class Price(NamedTuple):
    cost: float             # as the action charges it, unrounded (one tile for PER_TILE names)
    building: str           # what it buys, for people (building_name: "water tower", "bus line", "levee tile")
    per_tile: bool = False
    first: bool = True      # a line's base tier or an off-tier spec: a first building (not a clinic's hospital)

    def text(self) -> str:
        return money(self.cost) + ("/tile" if self.per_tile else "")


def short_of(funds: float, cost: float) -> bool:
    """`funds` can't pay `cost`, as actions._spend decides (it lets float dust through)."""
    return funds + 1e-9 < cost


def site_price(w, role: str, purpose: str) -> Price | None:
    """What a site on `role`'s find_sites `purpose` list costs: the cheapest unlocked thing the list is for
    (SITE_BUILDS), as its action charges it (per tile for PER_TILE names). None for purposes without a price or while
    all of theirs are locked."""
    priced = [(c, n) for n in SITE_BUILDS.get(role, {}).get(purpose, ()) if (c := _price(w, n)) is not None]
    best = min(priced, key=lambda t: t[0], default=None)
    return None if best is None else Price(best[0], building_name(best[1]), per_tile=best[1] in PER_TILE,
                                           first=best[1] in OFF_TIER_SPECS or C.TIER_BY_NAME[best[1]][1] == 1)


def site_tiles(w, purpose: str, site: dict) -> tuple[int, int]:
    """(tiles, charged units) of one site of a per-tile purpose built whole: a congested stretch's roads below avenue
    class, as upgrade_road charges them (BRIDGE_UPGRADE units on a bridge; none on a stretch of avenues), else the
    site's "tiles" (a levee run: empty bank land, one unit each), one when it doesn't say."""
    if purpose == "congested_roads":
        xs = range(min(site["x1"], site["x2"]), max(site["x1"], site["x2"]) + 1)
        ys = range(min(site["y1"], site["y2"]), max(site["y1"], site["y2"]) + 1)
        road = [(x, y) for y in ys for x in xs
                if w.struct[y, x] == C.ROAD and w.road_class[y, x] < C.ROAD_CLASSES["avenue"]]
        return len(road), sum(BRIDGE_UPGRADE if w.terrain[y, x] == C.WATER else 1 for x, y in road)
    n = max(1, int(site.get("tiles", 1)))
    return n, n


def best_stretch(w, purpose: str, sites: list[dict]) -> tuple[int, int]:
    """site_tiles of the best (first) site, one tile when it has nothing to charge: site_lists ranks congested
    stretches that are avenues already last (seed 0's most congested roads were by month 60), so a first one with
    nothing to charge means none has."""
    t = site_tiles(w, purpose, sites[0])
    return t if t[0] else (1, 1)


def ready_prices(w, role: str, lists: dict[str, SiteList]) -> dict[str, Price]:
    """site_price of each of `role`'s priced purposes with ready sites (listed and unlocked): the Ready sites lines
    end with it, and funds short of every one buy no ready site."""
    return {p: pr for p, lst in lists.items() if lst.sites and not lst.short and (pr := site_price(w, role, p))}


def cheapest(prices: dict[str, Price], supplementary: bool = False) -> tuple[tuple[str, ...], Price] | None:
    """(purposes, price) of the lowest price, SUPPLEMENTARY purposes skipped unless `supplementary` (the mayor's
    needs line and the budget line name them); None when there is none. Every purpose at that price, in order:
    health's clinic and landfill both cost $2,000, and naming the first alone would pick the clinic over the
    playbook's landfill-first."""
    kept = {p: pr for p, pr in prices.items() if supplementary or p not in SUPPLEMENTARY}
    low = min(kept.values(), key=lambda pr: pr.cost, default=None)
    return None if low is None else (tuple(p for p, pr in kept.items() if pr.cost == low.cost), low)


class FirstBuild(NamedTuple):
    cost: float                 # the cheapest ready site (cheapest), a per-tile purpose's best stretch whole
    named: dict[str, int]       # what it buys ("a clinic", "3 avenue tiles") -> ready sites of the first purpose
                                # pricing it
    dearest: float              # the dearest ready first building (Price.first, SUPPLEMENTARY skipped): what it keeps


def first_build(env, dept: str) -> FirstBuild | None:
    """`dept`'s cheapest ready site from one observation, None when none is ready: the mayor's needs line names it.
    With it the dearest ready first building, which DeptMoney keeps: review replay (seed 0, month 9), utilities held
    $3,000 for a $2,500 turbine with power at capacity, read idle above its $1,200 outlet, and lost it. A tier 2+
    purpose (hospital, riverbank_2x2's water works) is an upgrade, not a first building. A per-tile purpose counts its
    best site whole ("$90 for 3 avenue tiles", best_stretch), also when picking the cheapest: one $30 tile undersold
    an upgrade and beat the $1,500 bus line. Every building at the lowest price is named, with the ready sites of the
    first purpose pricing it."""
    w = env.world
    lists = site_lists(w, dept, role_obs(env, dept))
    prices, tiles = ready_prices(w, dept, lists), {}
    for p, pr in prices.items():
        if pr.per_tile:
            tiles[p], units = best_stretch(w, p, lists[p].sites)
            prices[p] = pr._replace(cost=pr.cost * units)
    purposes, low = cheapest(prices) or ((), None)
    if low is None:
        return None
    named = {}
    for p in purposes:
        n, b = tiles.get(p, 1), prices[p].building
        named.setdefault(f"{n} {b}s" if n != 1 else a_or_an(b), len(lists[p].sites[:MAX_SITES]))
    return FirstBuild(low.cost, named, max((pr.cost for p, pr in prices.items() if p not in SUPPLEMENTARY and pr.first),
                                           default=0))


def ready_lines(role: str, lists: dict[str, SiteList], keep_bank: set = frozenset(),
                prices: dict[str, Price] | None = None, funds: float | None = None) -> list[str]:
    """The Ready sites lines: per purpose the count and best site, or the short reason. A best site already shown
    on an earlier line is named once: the line points back to it and shows the next distinct site instead (several
    zone types rank the same strip first); a purpose reading another's list says so. A site in `keep_bank`
    (scarce_bank) shown for a purpose that doesn't need the river is marked to keep for pump / sewage. A line with
    sites ends with its price (`prices`: ready_prices), and with `funds` (rounded down) when they are short of it."""
    lines, shown, paths = [], {}, {}                 # shown: where -> purpose; paths: hint path -> purpose

    def site(purpose: str, s: dict) -> str:
        bank = purpose not in RIVER_PURPOSES and "x2" not in s and (s["x"], s["y"]) in keep_bank
        return _where(s) + _extra(purpose, s) + (" (riverbank: keep for pump/sewage)" if bank else "")

    def price(purpose: str) -> str:
        pr = (prices or {}).get(purpose)
        return "" if pr is None else f"; {pr.text()}" + (f" - you have {money(math.floor(funds))}"
                                                         if funds is not None and short_of(funds, pr.cost) else "")

    for purpose, lst in lists.items():
        head = f"- {purpose}: "
        path = SITE_PURPOSES[role][purpose]
        if lst.short:
            lines.append(head + (lst.short if lst.short.startswith("locked") else f"none - {lst.short}"))
            continue
        sites = lst.sites[:MAX_SITES]
        head += f"{len(sites)} site{'s' * (len(sites) != 1)}, "
        if path in paths:
            lines.append(head + f"same list as {paths[path]}" + price(purpose))
            continue
        paths[path] = purpose
        best, first = sites[0], shown.get(_where(sites[0]))
        if first is None:
            shown[_where(best)] = purpose
            lines.append(head + f"best {site(purpose, best)}" + price(purpose))
            continue
        nxt = next((t for t in sites[1:] if _where(t) not in shown), None)
        if nxt is None:
            lines.append(head + f"same {'strip' if 'x2' in best else 'site'} as {first}" + price(purpose))
        else:
            shown[_where(nxt)] = purpose
            lines.append(head + f"best as {first}; next {site(purpose, nxt)}" + price(purpose))
    return lines


# ------------------------------------------------------------------ toolbox
class ToolBox:
    def __init__(self, env, role: str):
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        self.env, self.role = env, role
        self.world = env.world
        self.fleet = FleetState.of(env)
        self.ended = False
        self.report = ""
        self.executed: list[dict] = []
        self.messages: list[dict] = []
        self.n_calls = 0                                  # tool calls this turn; "call N" counts from 1
        self.looked: Counter = Counter()                  # look-ups run this turn by tool (repeats not counted)
        self._seen: dict[tuple, tuple[int, dict]] = {}    # look-up key -> (call, result) since the last change
        self._failed: dict[tuple, tuple[int, str]] = {}   # action key -> (call, error) since the last success
        self._invalid: dict[tuple, tuple[int, str]] = {}  # bad call key -> (call, error) since the last success
        self.tools: dict[str, Tool] = {}
        lookups = self._lookup_tools()
        self.lookups = {t.name for t in lookups}
        for t in lookups + self._turn_tools():
            self.tools[t.name] = t
        docs = action_docs()
        for a in ROLES[role]["actions"]:
            if a != "noop":
                self.tools[a] = _action_tool(a, docs[a], partial(self._act, a), confirm=guarded(role, a))

    # ------------------------------------------------------------ interface
    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, arguments: str | dict | None = None) -> dict:
        """Run one tool call. An identical repeat (same tool, same arguments after coercion) of a look-up returns
        the earlier result marked "repeat", and one of an action that failed is refused ("refused": "repeat")
        without running, as is one of any call with bad arguments or an unknown tool (same arguments as sent),
        until something changes: a successful action (or plan update) for look-ups, a successful action for failed
        actions and bad calls; a failed action only refreshes `history`, which lists it. A bulldoze that would cut
        roads off the highway, or transport's demolish of their only station / harbour / airport link, is refused
        before it runs ("refused": "disconnect"; not executed, logged or remembered as a failure) unless it passes
        confirm: true. Once INSPECTS_FREE inspects have run this turn (repeats don't count), each further inspect
        result notes that map_overview shows the whole map (only the count when its radius note says so), until
        map_overview runs; the note isn't cached, so a repeat comes without it."""
        self.n_calls += 1
        try:
            tool, args = self._resolve(name, arguments)
        except ToolError as e:
            bad = (name, self._sent_key(arguments))
            if bad in self._invalid:
                return self._refusal(*self._invalid[bad])
            self._invalid[bad] = (self.n_calls, str(e)[:REPEAT_ERROR_CHARS])
            return {"ok": False, "error": str(e)}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        key = (name, json.dumps(args, sort_keys=True, default=str))
        if key in self._seen:
            n, res = self._seen[key]
            return {**copy.deepcopy(res), "repeat": f"same as call {n} (nothing changed since)"}
        if key in self._failed:
            return self._refusal(*self._failed[key])
        executed = len(self.executed)
        res = self._run(tool, args)
        if name in self.lookups:
            self._seen[key] = (self.n_calls, copy.deepcopy(res))     # callers may change the result they get
            self.looked[name] += 1
            if name == "inspect" and self.looked[name] > INSPECTS_FREE and not self.looked["map_overview"]:
                count = f"inspect #{self.looked[name]} this turn"
                res["note"] = (f"{res['note']}; {count}" if res.get("note")     # a radius note names map_overview
                               else f"{count}: map_overview shows the whole map in one call")
        elif name in ALL_ACTIONS:
            if res.get("ok"):
                self._seen.clear()
                self._failed.clear()
                self._invalid.clear()
            elif "refused" not in res:               # a guard refusal ran nothing: an identical repeat is refused again
                self._failed[key] = (self.n_calls, str(res.get("error"))[:REPEAT_ERROR_CHARS])
                if len(self.executed) > executed:         # logged as a failure: history shows it
                    self._seen = {k: v for k, v in self._seen.items() if k[0] != "history"}
        elif name == "update_plan" and res.get("ok"):
            self._seen.clear()
        return res

    @staticmethod
    def _refusal(n: int, err: str) -> dict:
        """The refusal of an identical repeat of the call that failed at call `n` with `err`."""
        if "call end_turn" not in err:
            err += "; change the arguments or do something else"
        elif not err.endswith("call end_turn"):
            err += "; call end_turn"
        return {"ok": False, "refused": "repeat", "error": f"this exact call failed at call {n}: {err}"}

    @staticmethod
    def _sent_key(arguments) -> str:
        """Arguments as sent, for the repeat check of a call that didn't resolve: JSON with sorted keys (none or
        blank as {}, but "null" stays null), else the stripped text."""
        if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
            return "{}"
        try:
            return json.dumps(json.loads(arguments) if isinstance(arguments, str) else arguments, sort_keys=True,
                              default=str)
        except (TypeError, ValueError):
            return str(arguments).strip()

    def _resolve(self, name: str, arguments) -> tuple[Tool, dict]:
        """The tool and its checked, coerced arguments; ToolError for unknown tools and bad arguments."""
        tool = self.tools.get(name)
        if tool is None:
            if name == "update_plan":
                raise ToolError("update_plan is for the mayor only; send the mayor a message instead")
            if name == "noop":
                raise ToolError("there is no noop tool; to do nothing, call end_turn")
            if name in ALL_ACTIONS:
                raise ToolError(f"the {self.role} role cannot use '{name}'")
            raise ToolError(f"unknown tool {name!r}; available: {', '.join(self.tools)}")
        return tool, self._check(tool, self._parse(arguments))

    def _run(self, tool: Tool, args: dict) -> dict:
        try:
            res = tool.fn(**args)
            res = _fit(_py(res), newest_last=tool.newest_last, hint=tool.trim_hint)
            if res.get("ok") is False and "error" not in res:
                res["error"] = res.get("message", "failed")
            return res
        except ToolError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:                           # never let a tool crash the agent loop
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    @staticmethod
    def _parse(arguments) -> dict:
        if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
            return {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError as e:
                raise ToolError(f"arguments are not valid JSON ({e}); send an object such as {{\"x\": 3, \"y\": 4}}")
        if not isinstance(arguments, dict):
            raise ToolError(f"arguments must be a JSON object of named parameters, got {type(arguments).__name__}")
        return arguments

    @staticmethod
    def _check(tool: Tool, args: dict) -> dict:
        out = {}
        for k, v in args.items():
            if v is None:                                # optional arguments sent as null count as absent
                continue
            if k not in tool.properties:
                if tool.strict:
                    valid = ", ".join(tool.properties) or "none"
                    raise ToolError(f"{tool.name} has no argument {k!r}; valid arguments: {valid}")
                out[k] = v
                continue
            out[k] = _coerce(k, v, tool.properties[k])
        missing = [k for k in tool.required if k not in out]
        if missing:
            raise ToolError(f"{tool.name} needs argument(s) {', '.join(missing)}")
        return out

    # ------------------------------------------------------------- actions
    def _act(self, name: str, /, **args) -> dict:
        w = self.world
        cap = self.env.cfg.max_actions_per_turn
        if sum(r["ok"] for r in self.executed) >= cap:
            raise ToolError(f"action limit reached for this turn ({cap} successful actions); call end_turn")
        t = args.pop("type", None)
        if t is not None and t != name:
            raise ToolError(f"the tool name is the action ({name}); don't pass type={t!r} - did you mean kind?")
        if not args.pop("confirm", False) and guarded(self.role, name):
            refusal = self._disconnect_refusal(name, args)
            if refusal:
                return {"ok": False, "refused": "disconnect", "error": refusal}
        bank = free_bank(self.world) if name == "zone" else None
        res = apply_action(w, self.role, {**args, "type": name}).to_dict()
        self.executed.append(res)
        self.fleet.record(w.month, self.role, "action", action=res["action"], ok=res["ok"], message=res["message"],
                          cost=res["cost"])
        dept = ROLES[self.role]["budget"]
        if res["ok"]:                                     # advice goes to the model only; the log keeps the core text
            note = self._bank_warning(bank) if bank is not None else ""
        else:
            note = self._explain(name, args, res["message"])
        return {"ok": res["ok"], "message": res["message"] + note, "cost": res["cost"],
                "budget_left": round(w.dept_funds[dept] if dept else w.treasury, 2)}

    def _disconnect_refusal(self, name: str, args: dict) -> str | None:
        """Why a bulldoze or demolish (guarded()) with these arguments must not run: it would cut highway-connected
        road tiles off the region (seed 1, v7: planning cleared the 2-tile link to the highway and the city emptied;
        a demolished train station, harbour or airport unlinks the roads it was the only link of). None when it
        wouldn't, or when the arguments are bad (the action reports those)."""
        w = self.world
        try:
            tiles = _rect(w, args) if name in ROAD_REMOVERS else [_point(w, args)]
        except Exception:
            return None
        if name in ROAD_REMOVERS:
            lost, gone = cut_off(w, tiles), [(x, y) for x, y in tiles if w.struct[y, x] == C.ROAD]
            why = "; build a new link first or pick tiles that aren't on the only route"
        else:
            (x, y), = tiles
            link = LINKS.get(int(w.struct[y, x]))
            if link is None:                              # not a link: nothing to relabel
                return None
            lost, gone = cut_off(w, (), drop={w._anchor_of(x, y)}), w.parts(x, y)
            why = (f": the {link} is their only link; ask planning to connect them to the highway by road first, or "
                   "build another link")
        if not lost.any():
            return None
        ys, xs = np.nonzero(lost)
        i = min(range(len(xs)), key=lambda i: min(abs(xs[i] - x) + abs(ys[i] - y) for x, y in gone))
        return (f"this would cut {len(xs)} road tile{'s' * (len(xs) > 1)} (e.g. around ({xs[i]},{ys[i]})) off the "
                f"{CUT_FROM[name]}{why}; to do it anyway repeat with confirm: true")

    def _bank_warning(self, before: np.ndarray) -> str:
        """A warning for a zone action that took the last free riverbank lots beside roads (`before`: free_bank()
        before it) while the city still lacks a sewage outlet / treatment plant or has neither a pump nor a water
        tower (pumps and outlets need the bank; a tower supplies water without it)."""
        w = self.world
        missing = _bank_needed(w)
        if not missing or not before.any() or free_bank(self.world).any():
            return ""
        ys, xs = np.nonzero(before & np.isin(w.struct, list(C.ZONES)))
        if not len(xs):                                   # not taken by this zoning: someone else's doing
            return ""
        return (" WARNING: that zoned the last free riverbank lot(s) beside roads; utilities still needs one for its "
                f"{' and '.join(missing)} - dezone one near ({xs[0]},{ys[0]})")

    def _explain(self, name: str, args: dict, message: str) -> str:
        """How to fix a failed build_road, zone or dezone, appended to the core message in the tool result (v8, seed 0
        month 0: planning zoned row 23, then ran four roads north through those lots, still empty, told only
        "occupied by residential" and never that dezoning is free; it also re-built a road twice and zoned high
        density before 'Town'). A blocked road names the empty zoned lots and the buildings on its line (a multi-tile
        building once), and dezones that stay on the line: per straight leg, one per run of lots between buildings,
        from its first to its last lot (dezone razes developed tiles, and a box off the line would take other lots).
        Smoke v9-v11: 11 x zone "no empty land tiles in that area", often on the road just built (build_road
        (35,23)-(35,8), then zone R x=35 y 9-22), and 13 x "every tile on that line is already road" (e.g. the main
        road (0,24)-(47,24) again), told nothing more. So a zone or dezone that found nothing to do says what the
        rectangle holds, a zone adds the free roadside land beside it, and a road on existing road names the road
        lines it runs along (else, off every line, the connected roads' extent). "" otherwise."""
        w = self.world
        if name == "zone" and message == w.unlock_note("high_density"):
            return ' - zone with density "low" now'
        if (name, message) in (("zone", "no empty land tiles in that area"), ("dezone", "no zoned tiles in that area")):
            try:
                tiles = _rect(w, args)
            except Exception:
                return ""
            beside = _zone_beside(w, tiles) if name == "zone" else []
            return f" - {_contents(w, tiles)}" + (f"; zone beside it instead: {' or '.join(beside)}" if beside else "")
        if name != "build_road":
            return ""
        if message == "every tile on that line is already road":
            line = _line(w, args)
            on = [_road_line_text(ln) for ln in _road_lines(w)       # lines it runs along (a crossing tile isn't)
                  if sum(_on_road_line(ln, x, y) for x, y in line) >= min(2, len(line))]
            if not on:
                return (f" - {_roads_text(_roads(w))}; start on the road and run into free land (find_sites "
                        "road_extension)")
            many = len(on) > 1                                    # at most 2: one per leg, or a crossing tile's two
            return (f" - the road line{'s' * many} {' and '.join(on)} {'are' if many else 'is'} already built; branch "
                    f"off {'them' if many else 'it'} into free land (find_sites road_extension)")
        if " is occupied by " not in message:
            return ""
        line = _line(w, args)                             # build_road got this far, so the arguments parse
        kind = {}                                         # what build_road can't take, as it walks the line
        for x, y in line:
            if w.struct[y, x] not in (C.NONE, C.ROAD, C.HIGHWAY):
                kind[x, y] = "lot" if w.struct[y, x] in C.ZONES and w.level[y, x] == 0 else "built"
        lots = [p for p, k in kind.items() if k == "lot"]
        firsts = {}                                       # a building once, at its first tile on the line
        for p, k in kind.items():
            if k == "built":
                firsts.setdefault(w._anchor_of(*p), p)
        built = list(firsts.values())
        ls, bs = "s" * (len(lots) > 1), "s" * (len(built) > 1)
        tiles = lambda pts: _some([f"({x},{y})" for x, y in pts], BLOCKERS_NAMED)
        tail = f"; route around the building{bs} (the Map block has the road extent)"
        if not lots:
            return f" - in the way: building{bs} {tiles(built)}{tail}"
        n = next((i for i, (_, y) in enumerate(line) if y != line[0][1]), len(line))    # the horizontal leg comes first
        rects = []
        for leg in (line[:n], line[n:]) if n > 1 else (line,):
            run = []
            for p in leg + [None]:                        # a run of lots ends at a building or the leg's end
                if p is None or kind.get(p) == "built":
                    if run:
                        (xa, ya), (xb, yb) = run[0], run[-1]
                        rects.append(f"dezone x1={min(xa, xb)},y1={min(ya, yb)},x2={max(xa, xb)},y2={max(ya, yb)}")
                    run = []
                elif kind.get(p) == "lot":
                    run.append(p)
        head = f" - in the way: empty zoned lot{ls} {tiles(lots)}"
        fix = f"dezone the lot{ls} first (free): {_some(rects, DEZONES_NAMED, '; ')}"
        if not built:
            return f"{head}; {fix}, then retry"
        return f"{head} and building{bs} {tiles(built)}; {fix}{tail}"

    # ------------------------------------------------------------- look-ups
    def _lookup_tools(self) -> list[Tool]:
        if self.role == "mayor":                        # read-only: every department's lists, "<department>.<purpose>"
            purposes = [f"{d}.{p}" for d in DEPARTMENTS if d in self.env.roles for p in SITE_PURPOSES[d]]
            what = ("Check which sites a department has (read-only), e.g. before believing \"no sites\": the list "
                    "the department's own find_sites returns for <department>.<purpose>, up to "
                    f"{MAX_SITES} ranked sites (crowded spots last; sorted by distance to (near_x, near_y) when "
                    "given), or a note saying why there are none.")
        else:
            purposes = list(SITE_PURPOSES[self.role])
            what = (f"Up to {MAX_SITES} suggested sites for a purpose (ranked; crowded spots last), sorted by distance "
                    "to (near_x, near_y) when given.")
        tools = [
            Tool("city_overview", "Headline numbers: population, happiness, jobs, money, taxes, demand, utilities "
                 "supply vs demand, service coverage, scores and the change over the last 3 months.",
                 self.city_overview),
            Tool("map_overview", "The whole city as an ASCII map with rulers and a legend, plus where the "
                 "highway, river, roads and built-up area are. The Map block in your briefing usually suffices: call "
                 "this only when you need tile-level layout, at most once per turn; inspect for details.",
                 self.map_overview),
            Tool("inspect", f"Look at the map around (x, y) within `radius` tiles (0-{MAX_RADIUS}, default 2): every "
                 "non-empty tile with what is there (zone letter + level such as R3, a building's tier name, road, "
                 "river), plus power / water / pollution for zones; empty land is counted. x = column, y = row.",
                 self.inspect, {"x": {"type": "integer"}, "y": {"type": "integer"},
                                "radius": {"type": "integer", "description": f"0-{MAX_RADIUS}, default 2"}},
                 ("x", "y"), trim_hint="use a smaller radius"),
            Tool("my_facilities", "Your department's buildings (load, capacity, tier) and the tier upgrades on offer "
                 "for them (run one with upgrade_facility)." if self.role != "mayor" else
                 "Your levers: service funding levels, policies, service imports and department upkeep.",
                 self.my_facilities),
            Tool("unlocks", "Options still locked by population milestones, the population each needs, and the next "
                 "milestone.", self.unlocks),
            Tool("get_plan", "The mayor's city plan" + (" (goals, department budgets and priorities, notes)."
                 if self.role == "mayor" else ": the city goals and your department's budget target and priority."),
                 self.get_plan),
            Tool("get_messages", f"Council messages you can see from the last {self.env.cfg.council_window} months, "
                 "newest last.", self.get_messages, newest_last=True),
            Tool("history", "The fleet log, newest last: " + (
                 "actions of all roles ('action', or failed ones with 'failure'), reports" if self.role == "mayor"
                 else "your department's actions ('action', or failed ones with 'failure') and reports")
                 + ", plus city events and monthly metrics. Omit kind (or \"all\") for everything.", self.history,
                 {"kind": {"type": "string", "enum": [*HISTORY_KINDS, "all"]},
                  "months": {"type": "integer", "description": f"how many recent months (1-{MAX_MONTHS}, default 3)"}},
                 newest_last=True, trim_hint="ask for fewer months or one kind"),
        ]
        if purposes:
            tools.insert(3, Tool(
                "find_sites", what + " Rectangles and lines also carry x1, y1, x2, y2.",
                self.find_sites, {"purpose": {"type": "string", "enum": purposes},
                                  "near_x": {"type": "integer"}, "near_y": {"type": "integer"}}, ("purpose",)))
        return tools

    def city_overview(self) -> dict:
        return overview(self.env)

    def inspect(self, x: int, y: int, radius: int = 2) -> dict:
        w = self.world
        x, y = int(round(x)), int(round(y))
        if not w.in_bounds(x, y):
            raise ToolError(f"({x},{y}) is outside the {w.w}x{w.h} map (x 0-{w.w - 1}, y 0-{w.h - 1})")
        r = max(0, min(MAX_RADIUS, int(round(radius))))
        x0, x1, y0, y1 = max(0, x - r), min(w.w - 1, x + r), max(0, y - r), min(w.h - 1, y + r)
        buildings, zones, ground, seen, empty = [], [], [], set(), 0
        cls = {v: k for k, v in C.ROAD_CLASSES.items()}
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                st = int(w.struct[ty, tx])
                if st == C.NONE:
                    if w.terrain[ty, tx] == C.WATER:
                        ground.append({"x": tx, "y": ty, "what": "river"})
                    else:
                        empty += 1
                elif st in C.ZONES:
                    zones.append({"x": tx, "y": ty, "what": f"{C.LETTER_BY_ZONE[st]}{int(w.level[ty, tx])}",
                                  "powered": bool(w.powered[ty, tx]), "watered": bool(w.watered[ty, tx]),
                                  "pollution": round(float(w.pollution[ty, tx]), 2)})
                elif st == C.ROAD:
                    t = {"x": tx, "y": ty, "what": "bridge" if w.terrain[ty, tx] == C.WATER else "road"}
                    if int(w.road_class[ty, tx]) > 1:
                        t["class"] = cls.get(int(w.road_class[ty, tx]))
                    ground.append(t)
                else:
                    ax, ay = w._anchor_of(tx, ty)
                    if (ax, ay) in seen:
                        continue
                    seen.add((ax, ay))
                    spec = w.facility_spec(ax, ay)
                    b = {"x": ax, "y": ay, "what": spec["name"] if spec else C.STRUCT_NAME.get(st, str(st))}
                    if spec:
                        b["tier"] = spec["tier"]
                    fw, fh = w.footprint_at(ax, ay)
                    if fw * fh > 1:
                        b["size"] = f"{fw}x{fh}"
                    buildings.append(b)
        out = {"ok": True, "x": x, "y": y, "radius": r, "window": [x0, y0, x1, y1],
               "tiles": buildings + zones + ground, "empty_land": empty}
        if r != int(round(radius)):          # models ask for 10 or 20 to see more; say they got less
            out["note"] = f"radius {radius} clipped to {r} (0-{MAX_RADIUS}); map_overview shows the whole map"
        return out

    def map_overview(self) -> dict:
        return map_overview(self.world, parts=MAYOR_MAP if self.role == "mayor" else MAP_PARTS)

    def find_sites(self, purpose: str, near_x: int | None = None, near_y: int | None = None) -> dict:
        if (near_x is None) != (near_y is None):
            raise ToolError("give both near_x and near_y, or neither")
        w = self.world
        role, key = purpose.split(".", 1) if self.role == "mayor" else (self.role, purpose)
        lst = site_lists(w, role, role_obs(self.env, role))[key]
        sites = lst.sites
        if near_x is not None:
            near = (int(near_x), int(near_y))
            sites = sorted(({**s, "distance": _distance(s, near)} for s in sites), key=lambda s: s["distance"])
        out = {"ok": True, "purpose": purpose, "sites": sites[:MAX_SITES]}
        if lst.note:                  # written to the department: the mayor mustn't read "ask planning" as its own task
            out["note"] = lst.note if role == self.role else f"({role} department's note) {lst.note}"
        return out

    def my_facilities(self) -> dict:
        w, role = self.world, self.role
        obs = role_obs(self.env, role, site_hints=False)
        if role == "mayor":
            return {"ok": True, **{k: obs[k] for k in ("funding_pct", "policies", "service_imports",
                                                       "department_upkeep")}}
        if role == "utilities":
            nets = [{**{k: v for k, v in n.items() if k != "plants"}, "plants": [_plant(p) for p in n["plants"]]}
                    for n in obs["networks"]]
            return {"ok": True, "networks": nets, "disconnected": obs["disconnected_facilities"],
                    "upgrades": obs["upgrades"]}
        if role == "transport":
            d = obs["transport"]
            return {"ok": True, **{k: d[k] for k in ("train_stations", "harbours", "metro_stations", "bus_lines",
                                                     "metro_lines", "freight")},
                    "airports": [{"x": x, "y": y} for x, y in w.facilities(C.AIRPORT)]}
        if role == "planning":
            marks = {k: [[x, y] for x, y in w.facilities(v["struct"])] for k, v in C.LANDMARKS.items()}
            return {"ok": True, "landmarks": {k: v for k, v in marks.items() if v},
                    "casinos": [[x, y] for x, y in w.facilities(C.CASINO)], "historic": obs["historic"]}
        facilities = {}
        for st in C.STRUCTS_OF_DEPT.get(role, []):
            lst = [dict(f) for f in w.service_info.get(C.STRUCT_NAME[st], [])]
            for f in lst:
                sp = None if "name" in f else w.facility_spec(f["x"], f["y"])
                if sp:
                    f.update(name=sp["name"], tier=sp["tier"])
            if lst:
                facilities[C.STRUCT_NAME[st]] = lst
        out = {"ok": True, "facilities": facilities, "upgrades": obs[role].get("upgrades", [])}
        if role == "fire":
            out["levee_tiles"] = obs["fire"]["levee_tiles"]
        if not facilities:
            out["note"] = "you have no buildings yet"
        return out

    def unlocks(self) -> dict:
        w = self.world
        locked: dict[int, list[str]] = {}
        for k in sorted(w.locked_now(), key=lambda k: (C.UNLOCK_POP[k], k)):
            locked.setdefault(C.UNLOCK_POP[k], []).append(k)
        return {"ok": True, "population": w.metrics()["population"],
                "locked_by_population": locked, "next_milestone": w.next_milestone()}

    def get_plan(self) -> dict:
        return plan_slice(self.fleet, self.role)

    def get_messages(self) -> dict:
        env = self.env
        msgs = [env._local(m) for m in env.council.visible_to(env._name(self.role),
                                                             since_month=self.world.month - env.cfg.council_window)]
        return {"ok": True, "messages": msgs[-MAX_ENTRIES:]}

    def history(self, kind: str | None = None, months: int = 3) -> dict:
        kind = None if kind == "all" else kind
        months = max(1, min(MAX_MONTHS, int(months)))
        roles = None if self.role == "mayor" else (self.role, "city")    # departments: their own and the city's
        entries = self.fleet.history(kind, months, role=roles, now=self.world.month)
        return {"ok": True, "months": months, "entries": entries[-MAX_ENTRIES:]}

    # --------------------------------------------------------- turn control
    def _turn_tools(self) -> list[Tool]:
        others = [r for r in self.env.roles if r != self.role]
        strings = {"type": "array", "items": {"type": "string"}}
        cap = f"at most {LIMITS['list_items']} items of {LIMITS['item_chars']} chars"
        tools = [
            Tool("send_message", "Post on the council board (delivered at the end of your turn): to a role, or "
                 "'all'.", self.send_message, {"to": {"type": "string", "enum": [*others, "all"]},
                                               "text": {"type": "string"}}, ("to", "text")),
            Tool("update_memory", f"Save your notes for later turns. Each list you give replaces the old one ({cap}); "
                 "lists you leave out are kept.", self.update_memory,
                 {"commitments": strings, "open_tasks": strings, "lessons": strings}),
            Tool("end_turn", "Finish your turn with a short report for the mayor (what you did, what you need).",
                 self.end_turn, {"report": {"type": "string"}}, ("report",)),
        ]
        if self.role == "mayor":
            per_dept = lambda t: {"type": "object", "properties": {d: {"type": t} for d in DEPARTMENTS}}
            tools.insert(2, Tool(
                "update_plan", f"Write the city plan the departments follow. goals ({LIMITS['goals']} max) and notes "
                "replace the old ones; budgets (a $ target per department) and priorities (one line per "
                "department) merge per department, so send only the ones that change. To clear one, set its budget "
                "to 0 or its priority to \"\" (never null). Budgets are targets: move the money with allocate_budget.",
                self.update_plan, {"goals": strings, "budgets": per_dept("number"),
                                   "priorities": per_dept("string"), "notes": {"type": "string"}}))
        return tools

    def send_message(self, to: str, text: str) -> dict:
        if not text.strip():
            raise ToolError("text is empty")
        self.messages.append({"to": to, "text": text})
        return {"ok": True, "message": f"message to {to} will be posted at the end of your turn"}

    def update_memory(self, **fields) -> dict:
        try:
            return {"ok": True, "message": self.fleet.update_memory(self.role, **fields)}
        except ValueError as e:
            raise ToolError(str(e))

    def update_plan(self, **fields) -> dict:
        """Save the plan; the result lists the idle money to take back first and the departments now below their
        budget target (budgets are targets, the money moves with allocate_budget), and says when the gaps exceed the
        treasury above the reserve."""
        w = self.world
        try:
            out = {"ok": True, "message": self.fleet.update_plan(w.month, **fields)}
        except ValueError as e:
            raise ToolError(str(e))
        f = funding_gaps(self.env, self.fleet)
        nxt = [f.idle_note()] if f.idle else []
        if f.idle:
            out["idle"] = f.idle
        if f.gaps:
            nxt.append(f"the treasury ({money(w.treasury)}) is at or under the {money(f.reserve)} reserve: {HOLD}"
                       if f.hold else f"call allocate_budget to fund these (treasury {money(w.treasury)})")
            if f.short and not f.hold:
                nxt[-1] += (f"; the gaps total {money(f.total)}, more than the treasury above a {money(f.reserve)} "
                            f"reserve ({money(max(0.0, f.spare))}): fund the most urgent first or lower targets")
            out.update(below_target=f.gaps, treasury=round(w.treasury))
        if nxt:
            out["next"] = "; ".join(nxt)
        return out

    def end_turn(self, report: str) -> dict:
        report = report.strip()[:REPORT_MAX_CHARS]
        self.ended, self.report = True, report
        self.fleet.reports[self.role] = report
        self.fleet.record(self.world.month, self.role, "report", text=report)
        return {"ok": True, "message": "turn ended"}

