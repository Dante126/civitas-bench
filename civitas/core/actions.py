"""Validated actions: the only way agents can change the world.

Every action is a plain dict, e.g. {"type": "zone", "zone": "R", "x1": 3,
"y1": 20, "x2": 10, "y2": 22}. The same format is used by LLM agents,
scripted agents, remote HTTP agents and RL policies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..sim import catalog as C
from ..sim.world import DEPARTMENTS, World
from .roles import ROLES

MAX_LINE = 60
MAX_RECT = 150


@dataclass
class ActionResult:
    action: dict
    ok: bool
    message: str
    cost: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class ActionError(Exception):
    pass


# --------------------------------------------------------------- helpers
def _int(params: dict, key: str) -> int:
    if key not in params:
        raise ActionError(f"missing parameter '{key}'")
    try:
        return int(round(float(params[key])))
    except (TypeError, ValueError, OverflowError):
        raise ActionError(f"parameter '{key}' must be a number, got {params[key]!r}")


def _num(params: dict, key: str) -> float:
    if key not in params:
        raise ActionError(f"missing parameter '{key}'")
    try:
        return float(params[key])
    except (TypeError, ValueError):
        raise ActionError(f"parameter '{key}' must be a number, got {params[key]!r}")


def _point(world: World, params: dict, kx="x", ky="y") -> tuple[int, int]:
    x, y = _int(params, kx), _int(params, ky)
    if not world.in_bounds(x, y):
        raise ActionError(f"({x},{y}) is outside the {world.w}x{world.h} map")
    return x, y


def _rect(world: World, params: dict, max_tiles: int = MAX_RECT) -> list[tuple[int, int]]:
    if "x" in params and "x1" not in params:
        x, y = _point(world, params)
        return [(x, y)]
    x1, y1 = _point(world, params, "x1", "y1")
    x2, y2 = _point(world, params, "x2", "y2")
    xa, xb = sorted((x1, x2))
    ya, yb = sorted((y1, y2))
    n = (xb - xa + 1) * (yb - ya + 1)
    if n > max_tiles:
        raise ActionError(f"rectangle has {n} tiles; max {max_tiles} per action")
    return [(x, y) for y in range(ya, yb + 1) for x in range(xa, xb + 1)]


def _line(world: World, params: dict) -> list[tuple[int, int]]:
    """Straight line, or an L (horizontal first, then vertical)."""
    x1, y1 = _point(world, params, "x1", "y1")
    x2, y2 = _point(world, params, "x2", "y2")
    pts = []
    step = 1 if x2 >= x1 else -1
    for x in range(x1, x2 + step, step):
        pts.append((x, y1))
    step = 1 if y2 >= y1 else -1
    for y in range(y1 + step, y2 + step, step) if y2 != y1 else []:
        pts.append((x2, y))
    if len(pts) > MAX_LINE:
        raise ActionError(f"road has {len(pts)} tiles; max {MAX_LINE} per action")
    return pts


def _require(world: World, key: str) -> None:
    """Milestone gate: raise if a building / option is still locked."""
    if not world.unlocked(key):
        raise ActionError(world.unlock_note(key))


def _spend(world: World, dept: str, cost: float) -> None:
    have = world.dept_funds[dept]
    if cost > have + 1e-9:
        raise ActionError(f"costs ${cost:,.0f} but the {dept} budget only has ${have:,.0f}; "
                          f"ask the mayor for funds")
    world.dept_funds[dept] = have - cost
    world.ledger["spent"][dept] += cost


# ---------------------------------------------------------------- mayor
def set_tax(world: World, p: dict) -> tuple[str, float]:
    sector = str(p.get("sector", "")).upper()[:1]
    if sector not in ("R", "C", "I"):
        raise ActionError("sector must be 'R', 'C' or 'I'")
    rate = _num(p, "rate")
    if rate < 0 or rate > C.MAX_TAX * 100:
        raise ActionError(f"rate is a percentage between 0 and {C.MAX_TAX * 100:.0f}")
    old = world.taxes[sector]
    world.taxes[sector] = rate / 100.0
    world.tax_changed(old, world.taxes[sector])
    warn = " (tax hikes cost approval)" if world.taxes[sector] > old else ""
    return f"{sector} tax set to {rate:.1f}%{warn}", 0.0


def set_funding(world: World, p: dict) -> tuple[str, float]:
    dept = str(p.get("department", "")).lower()
    if dept not in world.funding:
        raise ActionError(f"department must be one of {list(world.funding)}")
    pct = _num(p, "percent")
    lo, hi = C.FUNDING_MIN * 100, C.FUNDING_MAX * 100
    if not lo <= pct <= hi:
        raise ActionError(f"percent must be between {lo:.0f} and {hi:.0f}")
    world.funding[dept] = pct / 100.0
    return (f"{dept} funded at {pct:.0f}% (effectiveness {100 * world.funding_eff(dept):.0f}%, "
            f"upkeep x{pct / 100:.2f})"), 0.0


def _policy_name(p: dict) -> str:
    name = str(p.get("policy", p.get("name", ""))).lower().replace(" ", "_").replace("-", "_")
    if name not in C.POLICIES:
        raise ActionError(f"policy must be one of {list(C.POLICIES)}")
    return name


def _check_district(world: World, p: dict) -> None:
    if "district" in p and str(p["district"]) not in getattr(world, "districts", {}):
        known = list(getattr(world, "districts", {}))
        raise ActionError(f"no district named {p['district']!r}" + (f"; districts: {known}" if known else
                                                                    " (Planning paints districts)"))


def enact_policy(world: World, p: dict) -> tuple[str, float]:
    name = _policy_name(p)
    _check_district(world, p)
    if "district" in p:
        return world.set_district_policy(str(p["district"]), name, True), 0.0
    if name in world.policies:
        raise ActionError(f"{name} is already in force")
    world.policies.add(name)
    return f"enacted {name}: {C.POLICIES[name]['desc']}", 0.0


def repeal_policy(world: World, p: dict) -> tuple[str, float]:
    name = _policy_name(p)
    _check_district(world, p)
    if "district" in p:
        return world.set_district_policy(str(p["district"]), name, False), 0.0
    if name not in world.policies:
        raise ActionError(f"{name} is not in force")
    world.policies.discard(name)
    return f"repealed {name}", 0.0


def allocate_budget(world: World, p: dict) -> tuple[str, float]:
    dept = str(p.get("department", "")).lower()
    if dept not in DEPARTMENTS:
        raise ActionError(f"department must be one of {list(DEPARTMENTS)}")
    amount = _num(p, "amount")
    if amount >= 0:
        if amount > world.treasury:
            raise ActionError(f"treasury only has ${world.treasury:,.0f}")
        world.treasury -= amount
        world.dept_funds[dept] += amount
        world.ledger["allocated"][dept] += amount
        return f"transferred ${amount:,.0f} to {dept} (now ${world.dept_funds[dept]:,.0f})", 0.0
    amount = -amount
    if amount > world.dept_funds[dept]:
        raise ActionError(f"{dept} only holds ${world.dept_funds[dept]:,.0f}")
    world.dept_funds[dept] -= amount
    world.treasury += amount
    world.ledger["allocated"][dept] -= amount
    return f"reclaimed ${amount:,.0f} from {dept}", 0.0


def take_loan(world: World, p: dict) -> tuple[str, float]:
    amount = _num(p, "amount")
    if amount <= 0:
        raise ActionError("amount must be positive")
    if world.debt + amount > C.MAX_DEBT:
        raise ActionError(f"debt limit is ${C.MAX_DEBT:,.0f}; current debt ${world.debt:,.0f}")
    world.debt += amount
    world.treasury += amount
    return f"borrowed ${amount:,.0f} at {C.LOAN_INTEREST * 100:.1f}%/month (debt ${world.debt:,.0f})", 0.0


def repay_loan(world: World, p: dict) -> tuple[str, float]:
    amount = min(_num(p, "amount"), world.debt)
    if amount <= 0:
        raise ActionError("nothing to repay")
    if amount > world.treasury:
        raise ActionError(f"treasury only has ${world.treasury:,.0f}")
    world.debt -= amount
    world.treasury -= amount
    return f"repaid ${amount:,.0f} (debt ${world.debt:,.0f})", 0.0


# ------------------------------------------------------------- planning
def build_road(world: World, p: dict) -> tuple[str, float]:
    pts = _line(world, p)
    new, cost = [], 0.0
    for x, y in pts:
        s = world.struct[y, x]
        if s in (C.ROAD, C.HIGHWAY):
            continue
        if s != C.NONE:
            raise ActionError(f"tile ({x},{y}) is occupied by {C.STRUCT_NAME[int(s)]}; "
                              f"bulldoze it or route around")
        water = world.terrain[y, x] == C.WATER
        cost += C.CATALOG["bridge" if water else "road"]["cost"]
        new.append((x, y))
    if not new:
        raise ActionError("every tile on that line is already road")
    _spend(world, "planning", cost)
    for x, y in new:
        world.struct[y, x] = C.ROAD
    bridges = sum(1 for x, y in new if world.terrain[y, x] == C.WATER)
    extra = f" incl. {bridges} bridge tiles" if bridges else ""
    return f"built {len(new)} road tiles{extra}", cost


NO_DENSITY = ("I", "M")                 # industry and mixed use have no density setting: charged as high density
REZONE_COST = 2                          # per zoned tile of the same type whose density a zone changes


def zone_density(letter: str, density: str) -> str:
    """The density a zone of `letter` is built and charged at."""
    return "high" if letter in NO_DENSITY else density


def zone_tiles(world: World, letter: str, density: str, rect) -> tuple[list, list]:
    """The tiles of `rect` a zone fills (empty land) and re-zones (same type, other density)."""
    z = C.ZONE_BY_LETTER[letter]
    tiles = [(x, y) for x, y in rect if world.struct[y, x] == C.NONE and world.terrain[y, x] == C.LAND]
    rezone = [(x, y) for x, y in rect if world.struct[y, x] == z and world.density[y, x] != (density == "high")]
    return tiles, rezone


def zone_price(density: str) -> float:
    """Per new tile at that density (zone_density)."""
    return C.CATALOG["zone_high" if density == "high" else "zone"]["cost"]


def zone_cost(new: int, rezone: int, density: str) -> float:
    return new * zone_price(density) + rezone * REZONE_COST


def zone(world: World, p: dict) -> tuple[str, float]:
    letter = str(p.get("zone", "")).upper()[:1]
    if letter not in C.ZONE_BY_LETTER:
        raise ActionError(f"zone must be one of {list(C.ZONE_BY_LETTER)}")
    density = str(p.get("density", "low")).lower()
    if density not in ("low", "high"):
        raise ActionError("density must be 'low' or 'high'")
    if letter == "O":
        _require(world, "office_zone")
    if letter == "M":
        _require(world, "mixed_use")
    density = zone_density(letter, density)
    if density == "high" and letter not in NO_DENSITY:
        _require(world, "high_density")
    rect = _rect(world, p)
    tiles, rezone = zone_tiles(world, letter, density, rect)
    z = C.ZONE_BY_LETTER[letter]
    if not tiles and not rezone:
        raise ActionError("no empty land tiles in that area")
    cost = zone_cost(len(tiles), len(rezone), density)
    _spend(world, "planning", cost)
    no_access = 0
    for x, y in tiles:
        world.struct[y, x] = z
        world.level[y, x] = 0
        no_access += 0 if world.access[y, x] else 1
    for x, y in tiles + rezone:
        world.density[y, x] = 1 if density == "high" else 0
    warn = f" WARNING: {no_access} of them have no road access yet" if no_access else ""
    extra = f", changed density of {len(rezone)} zoned tiles" if rezone else ""
    note = {"I": "industry has no density setting", "M": "mixed use is always dense"}.get(letter, f"{density} density")
    return f"zoned {len(tiles)} tiles {letter} ({note}){extra}{warn}", cost


def dezone(world: World, p: dict) -> tuple[str, float]:
    tiles = [(x, y) for x, y in _rect(world, p) if world.struct[y, x] in C.ZONES]
    if not tiles:
        raise ActionError("no zoned tiles in that area")
    for x, y in tiles:
        world.struct[y, x] = C.NONE
        world.level[y, x] = 0
    return f"de-zoned {len(tiles)} tiles", 0.0


def build_park(world: World, p: dict) -> tuple[str, float]:
    tiles = [(x, y) for x, y in _rect(world, p)
             if world.struct[y, x] == C.NONE and world.terrain[y, x] == C.LAND]
    if not tiles:
        raise ActionError("no empty land tiles there")
    if len(tiles) > 9:
        raise ActionError("at most 9 park tiles per action")
    cost = len(tiles) * C.CATALOG["park"]["cost"]
    _spend(world, "planning", cost)
    for x, y in tiles:
        world.struct[y, x] = C.PARK
    return f"built {len(tiles)} park tile(s)", cost


def bulldoze(world: World, p: dict) -> tuple[str, float]:
    allowed = {C.ROAD, C.PARK, *C.ZONES}
    tiles = [(x, y) for x, y in _rect(world, p) if world.struct[y, x] in allowed]
    if not tiles:
        raise ActionError("nothing Planning can bulldoze there (roads, parks, zones only)")
    cost = len(tiles) * C.CATALOG["bulldoze"]["cost"]
    _spend(world, "planning", cost)
    for x, y in tiles:
        world.struct[y, x] = C.NONE
        world.level[y, x] = 0
    return f"bulldozed {len(tiles)} tiles", cost


# ------------------------------------------------------------ utilities
POWER_LINES = tuple(C.POWER_PLANTS)
WATER_LINES = tuple(C.WATER_PLANTS)
SEWAGE_LINES = tuple(dict.fromkeys(C.LINE_OF_STRUCT[st] for st in C.SEWAGE_PLANTS.values()))


def _kind_name(kind) -> str:
    """'Pumping Station' / 'pumping-station' -> 'pumping_station'."""
    return str(kind or "").strip().lower().replace(" ", "_").replace("-", "_")


def _tier_spec(world: World, kind, p: dict, lines: tuple, explicit: bool = True) -> dict:
    """Spec for a build action: kind is a line ('pump') or a tier name ('pumping_station'); "tier" picks a tier
    (absolute within the line). explicit=False: kind is the handler's default, so "tier" may override it freely."""
    names = [t["name"] for line in lines for t in C.TIER_LINES[line]]
    kind = _kind_name(kind)
    if kind not in C.TIER_BY_NAME or C.TIER_BY_NAME[kind][0] not in lines:
        raise ActionError(f"kind must be one of {names}")
    line, tier = C.TIER_BY_NAME[kind]
    tiers = C.TIER_LINES[line]
    if p.get("tier") is not None:
        want = _int(p, "tier")
        if not 1 <= want <= len(tiers):
            if len(tiers) == 1:
                raise ActionError(f'{line} has a single tier ({tiers[0]["name"]}); drop "tier"')
            raise ActionError(f"{line} has tiers 1-{len(tiers)}: " + ", ".join(f"{t['tier']} {t['name']}" for t in tiers))
        if explicit and kind != line and want != tier:
            raise ActionError(f'kind {kind} is tier {tier} of the {line} line; drop "tier" or use kind "{line}"')
        tier = want
    spec = tiers[tier - 1]
    if spec.get("unlock"):
        _require(world, spec["unlock"])
    return spec


def _place_facility(world: World, p: dict, spec: dict) -> tuple[str, float]:
    x, y = _point(world, p)
    name = spec["name"].replace("_", " ")
    terrain = C.WATER if spec.get("on_river") else C.LAND
    try:
        tiles = world.footprint_tiles(spec["struct"], x, y, terrain, spec["tier"])
    except ValueError as e:
        raise ActionError(f"{e}" + (f" (a {name} dam goes on open river, not a bridge)" if spec.get("on_river") else ""))
    if spec.get("needs_river") and not any(world.is_riverbank(tx, ty) for tx, ty in tiles):
        raise ActionError(f"a {name} must be on the riverbank (a land tile next to water)")
    _spend(world, "utilities", spec["cost"])
    world.place_building(spec["struct"], x, y, terrain, spec["tier"])
    net = world.building_network(x, y)
    warn = "" if world.networks[net].outside_connected else \
        " WARNING: that road network is not connected to the highway"
    res = next(k for k in ("power", "water", "sewage") if spec.get(k))
    return (f"built {name} (tier {spec['tier']}) at ({x},{y}){_size(spec)} adding {spec[res]} {res} "
            f"to network {net}{warn}"), spec["cost"]


def build_power_plant(world: World, p: dict) -> tuple[str, float]:
    return _place_facility(world, p, _tier_spec(world, p.get("kind", ""), p, POWER_LINES))


def build_water(world: World, p: dict) -> tuple[str, float]:
    return _place_facility(world, p, _tier_spec(world, p.get("kind", ""), p, WATER_LINES))


def build_sewage(world: World, p: dict) -> tuple[str, float]:
    return _place_facility(world, p, _tier_spec(world, p.get("kind", ""), p, SEWAGE_LINES))


def demolish(world: World, p: dict, role: str = "utilities") -> tuple[str, float]:
    x, y = _point(world, p)
    st = int(world.struct[y, x])
    owned = OWNED_STRUCTS.get(role, set())
    if st not in owned:
        what = C.STRUCT_NAME.get(st, "nothing")
        raise ActionError(f"no {role} facility at ({x},{y}) (found {what}); you can only demolish your own buildings")
    spec = world.facility_spec(x, y)
    name = spec["name"].replace("_", " ") if spec else C.STRUCT_NAME[st]
    world.remove_building(x, y)
    return f"demolished {name} at ({x},{y})", 0.0


# ------------------------------------------------------------ transport
def upgrade_road(world: World, p: dict) -> tuple[str, float]:
    to = str(p.get("to", p.get("kind", "avenue"))).lower()
    if to not in ("avenue", "highway"):
        raise ActionError("to must be 'avenue' or 'highway'")
    _require(world, to)
    cls = C.ROAD_CLASSES[to]
    pts = _line(world, p)
    todo, cost = [], 0.0
    for x, y in pts:
        if world.struct[y, x] != C.ROAD:
            raise ActionError(f"tile ({x},{y}) is not a road; only existing roads can be upgraded")
        cur = int(world.road_class[y, x])
        if cur >= cls:
            continue
        per = C.ROAD_UPGRADE_COST[cls] - (C.ROAD_UPGRADE_COST.get(cur, 0) if cur > 1 else 0)
        cost += per * (3 if world.terrain[y, x] == C.WATER else 1)
        todo.append((x, y))
    if not todo:
        raise ActionError(f"every tile on that line is already {to} or better")
    _spend(world, "transport", cost)
    for x, y in todo:
        world.road_class[y, x] = cls
    warn = " Note: homes and shops next to a highway lose zone access." if to == "highway" else ""
    return f"upgraded {len(todo)} road tiles to {to}{warn}", cost


def create_bus_line(world: World, p: dict) -> tuple[str, float]:
    stops = p.get("stops")
    if not isinstance(stops, list) or len(stops) < 2:
        raise ActionError('stops must be a list of at least 2 [x, y] road tiles, e.g. "stops": [[5,24],[12,24]]')
    if len(stops) > C.MAX_STOPS:
        raise ActionError(f"at most {C.MAX_STOPS} stops per line")
    pts, nets = [], set()
    for st in stops:
        try:
            x, y = int(round(float(st[0]))), int(round(float(st[1])))
        except (TypeError, ValueError, IndexError, KeyError):
            raise ActionError(f"bad stop {st!r}; use [x, y]")
        if not world.in_bounds(x, y) or world.road_labels[y, x] < 0:
            raise ActionError(f"stop ({x},{y}) is not on a road")
        nets.add(int(world.road_labels[y, x]))
        pts.append((x, y))
    if len(nets) > 1:
        raise ActionError("all stops must be on the same road network")
    cost = C.BUS_LINE_COST
    _spend(world, "transport", cost)
    line_id = max([l["id"] for l in world.bus_lines] + [0]) + 1
    world.bus_lines.append({"id": line_id, "stops": pts})
    return (f"bus line {line_id} opened with {len(pts)} stops (upkeep "
            f"${C.BUS_LINE_UPKEEP + C.BUS_STOP_UPKEEP * len(pts):.0f}/mo)"), cost


def remove_bus_line(world: World, p: dict) -> tuple[str, float]:
    line_id = _int(p, "line")
    before = len(world.bus_lines)
    world.bus_lines = [l for l in world.bus_lines if l["id"] != line_id]
    if len(world.bus_lines) == before:
        raise ActionError(f"no bus line {line_id}; lines: {[l['id'] for l in world.bus_lines]}")
    return f"closed bus line {line_id}", 0.0


def set_fare(world: World, p: dict) -> tuple[str, float]:
    fare = _num(p, "fare")
    lo, hi = C.FARE_RANGE
    if not lo <= fare <= hi:
        raise ActionError(f"fare must be between ${lo:.2f} and ${hi:.2f}")
    world.fare = fare
    return f"bus fare set to ${fare:.2f}", 0.0


def build_train_station(world: World, p: dict) -> tuple[str, float]:
    _require(world, "train_station")
    x, y = _point(world, p)
    if not (x in (0, world.w - 1) or y in (0, world.h - 1)):
        raise ActionError("train stations must be on the map edge (x=0, x=max, y=0 or y=max)")
    if world.terrain[y, x] != C.LAND or world.struct[y, x] != C.NONE:
        raise ActionError(f"({x},{y}) is not an empty land tile")
    if world.adjacent_network(x, y) < 0:
        raise ActionError(f"({x},{y}) does not touch a road")
    cost = C.TRAIN_STATION["cost"]
    _spend(world, "transport", cost)
    world.struct[y, x] = C.TRAIN
    world.facility_state[(x, y)] = {"built": world.month}
    return f"built train station at ({x},{y}): a second link to the region", cost


# ------------------------------------------------------ trade & upgrades
def set_service_imports(world: World, p: dict) -> tuple[str, float]:
    on = p.get("enabled", True)
    on = on if isinstance(on, bool) else str(on).lower() in ("1", "true", "yes", "on")
    world.service_imports = on
    return (f"service imports {'allowed' if on else 'stopped'}: the region "
            f"{'fills' if on else 'no longer fills'} up to {int(C.IMPORT_SHARE_MAX * 100)}% of a power/water shortfall "
            f"(now ${world.service_import_price('power'):.2f}/power, ${world.service_import_price('water'):.2f}/water per unit)"), 0.0


UPGRADE_OWNER = {**{st: C.TIER_LINES[line][0]["dept"] for st, line in C.LINE_OF_STRUCT.items()},
                 C.TRAIN: "transport", C.HARBOUR: "transport"}


def upgrade_facility(world: World, p: dict, role: str = "utilities") -> tuple[str, float]:
    x, y = _point(world, p)
    st_code = int(world.struct[y, x])
    if UPGRADE_OWNER.get(st_code) != role:
        raise ActionError(f"no {role} facility at ({x},{y}) (found {C.STRUCT_NAME.get(st_code, 'nothing')})")
    module = str(p.get("module") or "").strip().lower()
    if module == "tier" or (not module and p.get("tier") is not None):
        return _upgrade_tier(world, x, y, role, None if p.get("tier") is None else _int(p, "tier"))
    if module not in C.MODULES:
        raise ActionError(f"module must be 'tier' or one of {list(C.MODULES)}")
    spec = world.facility_spec(x, y)
    kind = spec["line"] if spec else C.STRUCT_NAME[st_code]
    name = (spec["name"] if spec else kind).replace("_", " ")
    allowed = C.MODULES[module]["for"]
    if allowed != "any" and kind not in allowed:
        raise ActionError(f"{module} does not apply to a {name} (only {', '.join(allowed)})")
    x, y = world._anchor_of(x, y)
    state = world.facility_state.setdefault((x, y), {"built": world.month})
    mods = set(state.get("modules", ()))
    if module in mods and module != "overhaul":
        raise ActionError(f"the {name} at ({x},{y}) already has a {module}")
    base = spec["cost"] if spec else C.TRAIN_STATION["cost"]
    cost = round(base * C.MODULES[module]["cost_share"])
    _spend(world, role, cost)
    if module == "overhaul":
        state["built"] = world.month
        return f"overhauled the {name} at ({x},{y}): back to full output", cost
    mods.add(module)
    state["modules"] = sorted(mods)
    what = "halves its pollution" if module == "filter" else "+50% capacity"
    return f"added {module} to the {name} at ({x},{y}) ({what})", cost


def _upgrade_tier(world: World, x: int, y: int, role: str, want: int | None = None) -> tuple[str, float]:
    try:
        cur, nxt, (x0, y0), _ = world.next_tier(x, y)
    except ValueError as e:
        raise ActionError(str(e))
    if want is not None and want <= cur["tier"]:
        raise ActionError(f"the {cur['name'].replace('_', ' ')} at ({x},{y}) is already tier {cur['tier']}; "
                          "upgrades only go up")
    if nxt.get("unlock"):
        _require(world, nxt["unlock"])
    cost = C.upgrade_cost(cur, nxt)
    _spend(world, role, cost)
    world.upgrade_building(x, y)
    gain = ", ".join(f"+{v:g} {k}" for k, v in C.tier_gain(cur, nxt).items())
    return (f"upgraded the {cur['name'].replace('_', ' ')} to a {nxt['name'].replace('_', ' ')} (tier {nxt['tier']}) "
            f"at ({x0},{y0}){_size(nxt)}" + (f": {gain}" if gain else "")
            + (" (upgrades go one tier at a time)" if want is not None and want > nxt["tier"] else "")), cost


def build_harbour(world: World, p: dict) -> tuple[str, float]:
    _require(world, "harbour")
    x, y = _point(world, p)
    if y < world.h - C.HARBOUR_ROWS:
        raise ActionError(f"the harbour goes near the river mouth: within {C.HARBOUR_ROWS} rows of the south edge")
    if world.terrain[y, x] != C.LAND or world.struct[y, x] != C.NONE or not world.is_riverbank(x, y):
        raise ActionError(f"({x},{y}) is not an empty riverbank tile")
    if world.adjacent_network(x, y) < 0:
        raise ActionError(f"({x},{y}) does not touch a road")
    cost = C.HARBOUR_SPEC["cost"]
    _spend(world, "transport", cost)
    world.struct[y, x] = C.HARBOUR
    world.facility_state[(x, y)] = {"built": world.month}
    return f"built a harbour at ({x},{y}): +{C.FREIGHT_CAPACITY['harbour']:.0f} freight capacity and a regional link", cost


# ------------------------------------------------------ casinos, airports, links between cities
def build_casino(world: World, p: dict) -> tuple[str, float]:
    _require(world, "casino")
    spec = C.CASINO_SPEC
    x, y = _build(world, C.CASINO, "planning", spec["cost"], p)
    return (f"built a casino at ({x},{y}){_size(C.CASINO)}: {spec['jobs']} low-skill jobs, ~{spec['tourism']} gamblers a "
            f"month (gambling tax), and more crime within {spec['crime_radius']} tiles"), spec["cost"]


def build_airport(world: World, p: dict) -> tuple[str, float]:
    _require(world, "airport")
    spec = C.AIRPORT_SPEC
    x, y = _build(world, C.AIRPORT, "transport", spec["cost"], p)
    return (f"built an airport at ({x},{y}){_size(C.AIRPORT)}: +{spec['freight']:.0f} freight, "
            f"x{spec['tourism_boost']} tourists, and air links to every other city with an airport"), spec["cost"]


EDGE = {"north": "y=0", "south": "y=max", "west": "x=0", "east": "x=max"}
INLAND = {"north": (0, 1), "south": (0, -1), "west": (1, 0), "east": (-1, 0)}


def connector_feeder(world: World, side: str, x: int, y: int) -> list[tuple[int, int]] | None:
    """Road tiles needed to join an edge tile to the road network, going straight inland ([] if it already touches one)."""
    dx, dy = INLAND[side]
    touches = lambda px, py: any(world.in_bounds(px + a, py + b) and world.road_labels[py + b, px + a] >= 0
                                 for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    if touches(x, y):
        return []
    path, px, py = [], x, y
    for _ in range(C.FEEDER_MAX):
        px, py = px + dx, py + dy
        if not world.in_bounds(px, py) or world.struct[py, px] != C.NONE:
            return None
        path.append((px, py))
        if touches(px, py):
            return path
    return None


def build_connector(world: World, p: dict) -> tuple[str, float]:
    info = world.region_info
    if not info:
        raise ActionError("connectors link cities in region mode; in a single city use build_train_station / build_harbour")
    to = str(p.get("to", "")).lower()
    if to not in info["neighbors"]:
        raise ActionError(f"'to' must be a neighbouring city: {list(info['neighbors'])} (others: use harbours or airports)")
    kind = str(p.get("kind", "highway")).lower()
    if kind not in C.CONNECTOR:
        raise ActionError(f"kind must be one of {list(C.CONNECTOR)}")
    if any(c["to"] == to and c["kind"] == kind for c in world.connectors):
        raise ActionError(f"a {kind} towards {to} already exists")
    side = info["neighbors"][to]
    x, y = _point(world, p)
    on_edge = {"north": y == 0, "south": y == world.h - 1, "west": x == 0, "east": x == world.w - 1}[side]
    if not on_edge:
        raise ActionError(f"{to} lies to the {side}: build the connector on that map edge ({EDGE[side]})")
    if world.terrain[y, x] != C.LAND or world.struct[y, x] != C.NONE:
        raise ActionError(f"({x},{y}) is not an empty land tile")
    feeder = connector_feeder(world, side, x, y)
    if feeder is None:
        raise ActionError(f"no road within {C.FEEDER_MAX} tiles straight inland from ({x},{y}); pick another edge tile "
                          "or ask Planning for a road towards that edge")
    road_cost = sum(C.CATALOG["bridge" if world.terrain[fy, fx] == C.WATER else "road"]["cost"] for fx, fy in feeder)
    cost = C.CONNECTOR[kind]["cost"] + road_cost
    _spend(world, "transport", cost)
    for fx, fy in feeder:
        world.struct[fy, fx] = C.ROAD
    world.struct[y, x] = C.HIGHWAY if kind == "highway" else C.TRAIN
    world.facility_state[(x, y)] = {"built": world.month}
    world.connectors.append({"to": to, "kind": kind, "x": x, "y": y})
    return (f"built a {kind} connector towards {to} at ({x},{y})"
            + (f" with a {len(feeder)}-tile feeder road" if feeder else "") + f"; the link opens once {to} builds its side "
            f"(capacity {C.CONNECTOR[kind]['goods']:.0f} goods / {C.CONNECTOR[kind]['people']:.0f} people a month)"), cost


# ------------------------------------------------------ heritage & soil
def preserve_historic(world: World, p: dict) -> tuple[str, float]:
    _require(world, "historic_preservation")
    tiles = [(x, y) for x, y in _rect(world, p) if world.struct[y, x] in C.ZONES
             and world.level[y, x] >= C.HISTORIC_MIN_LEVEL and not world.historic[y, x]]
    if not tiles:
        raise ActionError(f"no unlisted buildings of level {C.HISTORIC_MIN_LEVEL}+ in that area")
    room = C.HISTORIC_MAX_TILES - int(world.historic.sum())
    if room <= 0:
        raise ActionError(f"at most {C.HISTORIC_MAX_TILES} listed tiles")
    for x, y in tiles[:room]:
        world.historic[y, x] = True
    return (f"listed {min(len(tiles), room)} historic building(s): they keep their level and add land value "
            "and tourists nearby"), 0.0


def delist_historic(world: World, p: dict) -> tuple[str, float]:
    tiles = [(x, y) for x, y in _rect(world, p) if world.historic[y, x]]
    if not tiles:
        raise ActionError("no listed buildings in that area")
    for x, y in tiles:
        world.historic[y, x] = False
    return f"delisted {len(tiles)} historic building(s)", 0.0


def clean_soil(world: World, p: dict) -> tuple[str, float]:
    tiles = [(x, y) for x, y in _rect(world, p) if world.soil[y, x] > 0.05]
    if not tiles:
        raise ActionError("no contaminated soil in that area")
    cost = len(tiles) * C.SOIL_CLEANUP_COST
    _spend(world, "health", cost)
    for x, y in tiles:
        world.soil[y, x] *= 1.0 - C.SOIL_CLEANUP_SHARE
    return f"cleaned the soil on {len(tiles)} tiles (-{int(C.SOIL_CLEANUP_SHARE * 100)}% contamination)", cost


# ------------------------------------------------------------ metro
def build_metro_station(world: World, p: dict) -> tuple[str, float]:
    _require(world, "metro")
    x, y = _build(world, C.METRO, "transport", C.METRO_STATION["cost"], p)
    return f"built a metro station at ({x},{y}); connect stations with create_metro_line", C.METRO_STATION["cost"]


def create_metro_line(world: World, p: dict) -> tuple[str, float]:
    _require(world, "metro")
    st = p.get("stations")
    if not isinstance(st, list) or len(st) < 2 or len(st) > C.MAX_STOPS:
        raise ActionError(f'stations must be a list of 2-{C.MAX_STOPS} metro station tiles, e.g. "stations": [[10,20],[25,30]]')
    pts = []
    for q in st:
        try:
            x, y = int(round(float(q[0]))), int(round(float(q[1])))
        except (TypeError, ValueError, IndexError, KeyError):
            raise ActionError(f"bad station {q!r}; use [x, y]")
        if not world.in_bounds(x, y) or world.struct[y, x] != C.METRO:
            raise ActionError(f"({x},{y}) is not a metro station; build stations first")
        pts.append((x, y))
    track = sum(((pts[i][0] - pts[i + 1][0]) ** 2 + (pts[i][1] - pts[i + 1][1]) ** 2) ** 0.5 for i in range(len(pts) - 1))
    cost = round(track * C.METRO_TRACK_COST)
    _spend(world, "transport", cost)
    line_id = max([l["id"] for l in world.metro_lines] + [0]) + 1
    world.metro_lines.append({"id": line_id, "stations": pts})
    return (f"metro line M{line_id} opened through {len(pts)} stations ({track:.0f} tiles of tunnel, upkeep "
            f"${C.METRO_LINE_UPKEEP}/mo)"), cost


def remove_metro_line(world: World, p: dict) -> tuple[str, float]:
    line_id = _int(p, "line")
    before = len(world.metro_lines)
    world.metro_lines = [l for l in world.metro_lines if l["id"] != line_id]
    if len(world.metro_lines) == before:
        raise ActionError(f"no metro line {line_id}")
    return f"closed metro line M{line_id}", 0.0


# ------------------------------------------------------ emergencies
def build_levee(world: World, p: dict) -> tuple[str, float]:
    pts = _line(world, p) if "x1" in p else [_point(world, p)]
    ok = [(x, y) for x, y in pts if world.terrain[y, x] == C.LAND and world.struct[y, x] == C.NONE
          and world.dist_water[y, x] == 1]
    if not ok:
        raise ActionError("levees go on empty riverbank tiles (land next to the river); none on that line")
    cost = len(ok) * C.CATALOG["levee"]["cost"]
    _spend(world, "fire", cost)
    for x, y in ok:
        world.struct[y, x] = C.LEVEE
    return (f"built {len(ok)} levee tile(s)" + (f" (skipped {len(pts) - len(ok)} tiles that are not empty riverbank)"
                                                 if len(ok) < len(pts) else "")), cost


# --------------------------------------------------- mayor: offers & districts
def accept_offer(world: World, p: dict) -> tuple[str, float]:
    try:
        return world.accept_offer(_int(p, "offer")), 0.0
    except ValueError as e:
        raise ActionError(str(e))


def decline_offer(world: World, p: dict) -> tuple[str, float]:
    try:
        return world.decline_offer(_int(p, "offer")), 0.0
    except ValueError as e:
        raise ActionError(str(e))


def set_district_tax(world: World, p: dict) -> tuple[str, float]:
    _check_district(world, p)
    if "district" not in p:
        raise ActionError("missing parameter 'district'")
    sector = str(p.get("sector", "")).upper()[:1]
    if sector not in ("R", "C", "I"):
        raise ActionError("sector must be 'R', 'C' or 'I'")
    delta = _num(p, "delta")
    if abs(delta) > C.DISTRICT_TAX_RANGE:
        raise ActionError(f"delta is in percentage points, between -{C.DISTRICT_TAX_RANGE:.0f} and +{C.DISTRICT_TAX_RANGE:.0f}")
    world.districts[str(p["district"])]["tax_mod"][sector] = delta / 100.0
    return f"district '{p['district']}' {sector} tax is now city rate {delta:+.1f} points", 0.0


# ---------------------------------------------------- planning: districts & landmarks
def paint_district(world: World, p: dict) -> tuple[str, float]:
    _require(world, "districts")
    name = str(p.get("name", "")).strip()
    if not name or len(name) > 20 or not all(c.isalnum() or c in " _-" for c in name):
        raise ActionError("name must be 1-20 letters, digits, spaces, _ or -")
    tiles = [(x, y) for x, y in _rect(world, p, max_tiles=600)]
    try:
        return world.paint_district(name, tiles), 0.0
    except ValueError as e:
        raise ActionError(str(e))


def specialize_district(world: World, p: dict) -> tuple[str, float]:
    from ..sim.economy import IND_TYPES
    _check_district(world, p)
    if "district" not in p:
        raise ActionError("missing parameter 'district'")
    ind = str(p.get("industry", "")).lower()
    if ind in ("none", "", "generic"):
        world.districts[str(p["district"])]["specialization"] = None
        return f"district '{p['district']}' has no industry specialisation", 0.0
    if ind not in IND_TYPES[1:]:
        raise ActionError(f"industry must be one of {list(IND_TYPES[1:])} or 'none'")
    world.districts[str(p["district"])]["specialization"] = ind
    return (f"new industry in district '{p['district']}' will be {ind}"
            + (" (needs the matching natural resource)" if ind != "hightech" else " (needs educated workers)")), 0.0


def _build(world: World, struct: int, dept: str, cost: float, p: dict, terrain: int = C.LAND,
           tier: int | None = None) -> tuple[int, int]:
    """Validate the footprint, charge the budget, place the building. Returns its top-left tile."""
    x, y = _point(world, p)
    try:
        world.footprint_tiles(struct, x, y, terrain, tier)
    except ValueError as e:
        raise ActionError(str(e))
    _spend(world, dept, cost)
    world.place_building(struct, x, y, terrain, tier)
    return x, y


def _size(what) -> str:
    """what: a struct code or a tier spec."""
    fw, fh = tuple(what["footprint"]) if isinstance(what, dict) else C.FOOTPRINT.get(what, (1, 1))
    return f" ({fw}x{fh} tiles from its top-left corner)" if fw * fh > 1 else ""


def build_landmark(world: World, p: dict) -> tuple[str, float]:
    kind = str(p.get("kind", "")).lower()
    if kind not in C.LANDMARKS:
        raise ActionError(f"kind must be one of {list(C.LANDMARKS)}")
    _require(world, kind)
    spec = C.LANDMARKS[kind]
    x, y = _build(world, spec["struct"], "planning", spec["cost"], p)
    return f"built {kind} at ({x},{y}){_size(spec['struct'])}", spec["cost"]


# ------------------------------------------------------------ services
def _place_service(world: World, p: dict, kind: str, explicit: bool = False) -> tuple[str, float]:
    """kind: the handler's building (e.g. 'hospital'); a "kind" param naming another tier of its line wins."""
    line = C.TIER_BY_NAME[kind][0]
    given = _kind_name(p.get("kind"))
    if not explicit and C.TIER_BY_NAME.get(given, ("",))[0] == line:
        kind, explicit = given, True
    spec = _tier_spec(world, kind, p, (line,), explicit)
    x, y = _build(world, spec["struct"], spec["dept"], spec["cost"], p, tier=spec["tier"])
    net = world.building_network(x, y)
    warn = "" if world.networks[net].outside_connected else \
        " WARNING: that road network is not connected to the highway"
    return f"built {spec['name'].replace('_', ' ')} at ({x},{y}){_size(spec)}{warn}", spec["cost"]


def _service_handler(kind: str):
    return lambda world, p: _place_service(world, p, kind)


SCHOOL_KINDS = {"elementary": "elementary_school", "high": "high_school", "university": "university"}


def build_school(world: World, p: dict) -> tuple[str, float]:
    kind = _kind_name(p.get("kind") or "elementary")
    kind = SCHOOL_KINDS.get(kind, kind)
    if kind not in C.TIER_BY_NAME and kind.endswith("_school"):     # 'large_elementary_school', 'university_school'
        short = kind[:-len("_school")]
        kind = SCHOOL_KINDS.get(short, short)
    if C.TIER_BY_NAME.get(kind, ("",))[0] not in SCHOOL_KINDS.values():
        raise ActionError(f"kind must be one of {list(SCHOOL_KINDS)} (or a tier name such as large_elementary)")
    return _place_service(world, p, kind, explicit=True)


def noop(world: World, p: dict) -> tuple[str, float]:
    return "no action", 0.0


HANDLERS = {
    "set_tax": set_tax, "allocate_budget": allocate_budget, "take_loan": take_loan,
    "set_funding": set_funding, "enact_policy": enact_policy, "repeal_policy": repeal_policy,
    "repay_loan": repay_loan, "build_road": build_road, "zone": zone, "dezone": dezone,
    "build_park": build_park, "bulldoze": bulldoze, "build_power_plant": build_power_plant,
    "build_water": build_water, "build_sewage": build_sewage, "demolish": demolish, "noop": noop,
    "build_police_station": _service_handler("police_station"), "build_jail": _service_handler("jail"),
    "build_fire_station": _service_handler("fire_station"),
    "build_landfill": _service_handler("landfill"), "build_incinerator": _service_handler("incinerator"),
    "build_clinic": _service_handler("clinic"), "build_hospital": _service_handler("hospital"),
    "build_cemetery": _service_handler("cemetery"),
    "build_school": build_school, "build_library": _service_handler("library"),
    "upgrade_road": upgrade_road, "create_bus_line": create_bus_line, "remove_bus_line": remove_bus_line,
    "set_fare": set_fare, "build_train_station": build_train_station,
    "build_levee": build_levee, "build_shelter": _service_handler("shelter"),
    "accept_offer": accept_offer, "decline_offer": decline_offer, "set_district_tax": set_district_tax,
    "paint_district": paint_district, "specialize_district": specialize_district, "build_landmark": build_landmark,
    "set_service_imports": set_service_imports, "upgrade_facility": upgrade_facility, "build_harbour": build_harbour,
    "preserve_historic": preserve_historic, "delist_historic": delist_historic, "clean_soil": clean_soil,
    "build_metro_station": build_metro_station, "create_metro_line": create_metro_line,
    "build_casino": build_casino, "build_airport": build_airport, "build_connector": build_connector,
    "remove_metro_line": remove_metro_line,
}
ROLE_AWARE = {"demolish", "upgrade_facility"}

# which structures each role may demolish
OWNED_STRUCTS = {
    "utilities": set(C.UTILITY_TYPES),
    **{d: set(C.STRUCTS_OF_DEPT[d]) for d in ("police", "fire", "health", "education")},
    "transport": {C.TRAIN, C.HARBOUR, C.METRO, C.AIRPORT},
    "planning": {*C.LANDMARK_BY_STRUCT, C.CASINO},
}
OWNED_STRUCTS["fire"] = OWNED_STRUCTS["fire"] | {C.LEVEE}

_GAIN_LABEL = {"power": "+{} power", "water": "+{} water", "sewage": "{} sewage", "capacity": "capacity {}",
               "seats": "{} seats", "collect": "collects {}/mo", "responses": "{} call-outs/mo", "rate": "{} deceased/mo",
               "holding": "{} cells", "radius": "radius {}"}


def _named(spec: dict, name: str | None = None, size: bool = False) -> str:
    """'water_works (10,000+ residents, 2x2)': a name with its unlock level (and footprint)."""
    fw, fh = spec["footprint"]
    tags = ([f"{C.UNLOCK_POP[spec['unlock']]:,}+ residents"] if spec.get("unlock") else []) + \
        ([f"{fw}x{fh}"] if size and fw * fh > 1 else [])
    return (name or spec["name"]) + (f" ({', '.join(tags)})" if tags else "")


def _tier_extras(t: dict, base: dict) -> list[str]:
    """What sets a tier apart beyond its numbers (read from the spec)."""
    out = []
    if t.get("pollution") and t["pollution"] != base.get("pollution"):
        out.append(f"pollution {t['pollution']:g}")
    if t.get("min_quality"):
        out.append("filters dirty river water")
    if t.get("discharge") is not None and t["discharge"] < 1 and t["discharge"] != base.get("discharge"):
        out.append(f"removes {100 - round(100 * t['discharge'])}% of the pollution")
    if base.get("fill") and not t.get("fill"):
        out.append("never fills up")
    if base.get("plots") and not t.get("plots"):
        out.append("no plots to fill")
    return out


def _tiers_doc(line: str, start: int = 2) -> str:
    """'2 pumping_station (1,000+ residents): $4,000, +650 water, upkeep $45/mo; ...' (tiers from `start` up)."""
    base, out = C.TIER_LINES[line][0], []
    for t in C.TIER_LINES[line][start - 1:]:
        bits = [f"${t['cost']:,}"] + [_GAIN_LABEL.get(k, k + " {}").format(f"{t[k]:,}") for k in C.GAIN_KEYS
                                       if t.get(k) and t[k] != base.get(k)]
        bits.append(f"upkeep ${t['upkeep']}/mo")
        out.append(f"{t['tier']} {_named(t, size=True)}: " + ", ".join(bits + _tier_extras(t, base)))
    return "; ".join(out)


def _higher_tiers(lines, start: int = 2) -> str:
    lines = [line for line in lines if len(C.TIER_LINES[line]) >= start]
    docs = [(f"{line}: " if len(lines) > 1 else "") + _tiers_doc(line, start) for line in lines]
    return (" Tiers: " + "; ".join(docs) + ".") if docs else ""


def _crowd_doc(*lines: str) -> str:
    """Crowding rule for production lines, e.g. _crowd_doc("pump", "tower")."""
    parts = [f"{line} within {g['radius']} tiles" for line in lines
             for g in C.CROWD_GROUPS.values() if line in g["lines"]]
    return (f" Crowding: each other {' or '.join(parts)} (any tier) cuts output {int(C.CROWD_STEP * 100)}%, "
            f"min {int(C.CROWD_FLOOR * 100)}%.") if parts else ""


def _downtown_doc() -> str:
    return (f" Downtown (level {C.DOWNTOWN_LEVEL}-5 buildings) gets only {int(C.T1_DOWNTOWN * 100)}% cover from "
            "tier-1 buildings; a tier-2+ building in range gives full cover.")


def _plant_doc(kind: str) -> str:
    sp = C.CATALOG[kind]
    res = next(k for k in ("power", "water", "sewage") if sp.get(k))
    fw, fh = sp["footprint"]
    bits = [f"${sp['cost']:,}", f"+{sp[res]} {res}" if res != "sewage" else f"{sp[res]} sewage",
            f"upkeep ${sp['upkeep']}/mo"]
    if sp.get("pollution"):
        bits.append(f"pollution {sp['pollution']:g}")
    if fw * fh > 1:
        bits.append(f"{fw}x{fh}")
    if sp.get("water_use"):
        bits.append(f"uses {sp['water_use']} water for cooling")
    return f"{_named(sp, kind)}: " + ", ".join(bits)


def _svc_doc(kind: str, head: bool = True) -> str:
    sp = C.SERVICES[kind]
    bits = [f"${sp['cost']:,}", f"upkeep ${sp['upkeep']}/mo"]
    if "radius" in sp and sp["radius"] < 60:
        bits.append(f"covers {sp['radius']} road steps")
    for key, label in (("capacity", "capacity"), ("seats", "seats"), ("collect", "collects/mo"),
                       ("responses", "call-outs/mo"), ("plots", "plots"), ("fill", "landfill size")):
        if key in sp and not (kind == "university" and key == "capacity"):
            bits.append(f"{label} {sp[key]:,}")
    if sp.get("power"):
        bits.append(f"+{sp['power']} power")
    if sp.get("pollution"):
        bits.append(f"pollution {sp['pollution']}")
    extra = _higher_tiers((sp["line"],), sp["tier"] + 1)
    if kind in C.DOWNTOWN_LINES:
        extra += _downtown_doc()
    lead = _named(sp, kind.replace("_", " ")) + (" next to a road" if head else "")
    return f"{lead}: " + ", ".join(bits) + "." + extra


# Human/LLM readable reference, rendered into prompts. Built on demand so that
# balance overrides from the config show up in the docs.
def action_docs() -> dict:
    return {
        "set_tax": ('{"type":"set_tax","sector":"R","rate":9}',
                    "Set tax % (0-20) for sector R, C or I. Default 9. Above ~9% slows growth; residential tax above 7% lowers happiness."),
        "allocate_budget": ('{"type":"allocate_budget","department":"planning","amount":5000}',
                            f"Move money from the treasury to a department ({', '.join(DEPARTMENTS)}). Negative amount takes money back."),
        "set_funding": ('{"type":"set_funding","department":"police","percent":110}',
                        f"Service funding level {C.FUNDING_MIN * 100:.0f}-{C.FUNDING_MAX * 100:.0f}% for {', '.join(C.FUNDED_DEPARTMENTS)}. "
                        "Upkeep scales with the percentage; effectiveness (coverage, capacity, plant output) with percent^0.7."),
        "enact_policy": ('{"type":"enact_policy","policy":"recycling"}',
                         "City-wide policy with a monthly cost: " + "; ".join(f"{k} ({v['desc']})" for k, v in C.POLICIES.items()) + "."),
        "repeal_policy": ('{"type":"repeal_policy","policy":"recycling"}', "End a policy."),
        "take_loan": ('{"type":"take_loan","amount":10000}', f"Borrow money (max total debt ${C.MAX_DEBT:,}). Interest {C.LOAN_INTEREST * 100:.1f}%/month."),
        "repay_loan": ('{"type":"repay_loan","amount":10000}', "Repay debt from the treasury."),
        "build_road": ('{"type":"build_road","x1":5,"y1":24,"x2":5,"y2":10}',
                       f"Road from (x1,y1) to (x2,y2): straight, or L-shaped (horizontal first). ${C.CATALOG['road']['cost']}/tile, bridge over water ${C.CATALOG['bridge']['cost']}/tile. Tiles that are already road are skipped for free, so a line may start on an existing road. Max {MAX_LINE} tiles."),
        "zone": ('{"type":"zone","zone":"R","density":"high","x1":2,"y1":21,"x2":12,"y2":23}',
                 f"Zone the empty land in a rectangle as R (homes), C (shops), I (industry), O (offices, need educated "
                 f"workers) or M (mixed use: shops below, flats above; always dense). density 'low' (${C.CATALOG['zone']['cost']}/tile, stops at level {C.LOW_DENSITY_MAX}; wealthy "
                 f"households prefer it) or 'high' (${C.CATALOG['zone_high']['cost']}/tile, up to level {C.MAX_LEVEL}). "
                 f"Re-zoning existing tiles of the same type changes their density. Max {MAX_RECT} tiles."),
        "dezone": ('{"type":"dezone","x1":2,"y1":21,"x2":4,"y2":23}', "Remove zoning (and buildings) in a rectangle. Free."),
        "build_park": ('{"type":"build_park","x":8,"y":20}', f"Park on an empty tile (or small rectangle, max 9). ${C.CATALOG['park']['cost']}/tile. Raises land value and happiness nearby."),
        "bulldoze": ('{"type":"bulldoze","x1":3,"y1":3,"x2":4,"y2":4}', f"Clear roads/parks/zones in a rectangle. ${C.CATALOG['bulldoze']['cost']}/tile."),
        "build_power_plant": ('{"type":"build_power_plant","kind":"wind","x":14,"y":25}',
                              "Power plant next to a road (hydro: on a river tile next to a road or bridge; nuclear: on "
                              "the riverbank). " + "; ".join(_plant_doc(k) for k in POWER_LINES) + "."
                              + _higher_tiers(POWER_LINES) + _crowd_doc("wind")),
        "build_water": ('{"type":"build_water","kind":"pump","x":29,"y":24}',
                        f"Water facility next to a road. {_plant_doc('pump')}, riverbank only; {_plant_doc('tower')}, "
                        "anywhere." + _higher_tiers(WATER_LINES) + _crowd_doc(*WATER_LINES)),
        "build_sewage": ('{"type":"build_sewage","kind":"outlet","x":29,"y":30}',
                         f"Sewage facility on the riverbank next to a road; serves its road network like water. "
                         f"{_plant_doc('outlet')}, dumps raw sewage into the river (flows SOUTH and dirties pumps "
                         f"downstream); tier 2 {_plant_doc('treatment')}, removes "
                         f"{100 - int(100 * C.CATALOG['treatment']['discharge'])}% of the pollution."
                         + _higher_tiers(SEWAGE_LINES, 3) + _crowd_doc("outlet")),
        "demolish": ('{"type":"demolish","x":14,"y":25}', "Remove one of your own facilities (no refund)."),
        "upgrade_road": ('{"type":"upgrade_road","to":"avenue","x1":5,"y1":24,"x2":20,"y2":24}',
                         f"Upgrade existing road tiles on a straight/L line. avenue: ${C.ROAD_UPGRADE_COST[2]}/tile, capacity "
                         f"{C.ROAD_CAPACITY[2]:.0f} vs {C.ROAD_CAPACITY[1]:.0f}, upkeep ${C.ROAD_UPKEEP[2]}/tile; highway: "
                         f"${C.ROAD_UPGRADE_COST[3]}/tile, capacity {C.ROAD_CAPACITY[3]:.0f}, upkeep ${C.ROAD_UPKEEP[3]}/tile, no zone access "
                         "next to it. x3 cost on bridges."),
        "create_bus_line": ('{"type":"create_bus_line","stops":[[5,24],[12,24],[18,20]]}',
                            f"Bus line through 2-{C.MAX_STOPS} stops on road tiles of one network. ${C.BUS_LINE_COST:,}, upkeep "
                            f"${C.BUS_LINE_UPKEEP:.0f}+${C.BUS_STOP_UPKEEP:.0f}/stop per month, carries up to {C.LINE_CAPACITY:.0f} "
                            f"riders/month. Trips from homes within {C.STOP_RADIUS} tiles of a stop to a job centre near another stop "
                            "of the same line may take the bus."),
        "remove_bus_line": ('{"type":"remove_bus_line","line":1}', "Close a bus line."),
        "set_fare": ('{"type":"set_fare","fare":2}', f"Bus fare ${C.FARE_RANGE[0]:.0f}-{C.FARE_RANGE[1]:.0f} per ride "
                                                    f"(default ${C.DEFAULT_FARE:.0f}); higher fares mean fewer riders but more income."),
        "build_train_station": ('{"type":"build_train_station","x":47,"y":30}',
                                f"Train station on an empty map-edge tile touching a road: ${C.TRAIN_STATION['cost']:,}, "
                                f"upkeep ${C.TRAIN_STATION['upkeep']}/mo. Links that road network to the region "
                                "(zones there develop even without the highway) and adds regional trips."),
        **{f"build_{k}": (f'{{"type":"build_{k}","x":14,"y":25}}', _svc_doc(k))
           for k in ("police_station", "jail", "fire_station", "landfill", "incinerator", "clinic",
                     "hospital", "cemetery", "library", "shelter")},
        "build_levee": ('{"type":"build_levee","x1":27,"y1":10,"x2":27,"y2":18}',
                        f"Levee on empty riverbank tiles along a line: ${C.CATALOG['levee']['cost']}/tile, upkeep "
                        f"${C.CATALOG['levee']['upkeep']}/tile. Protects land within {C.LEVEE_PROTECT} tiles from river floods."),
        "build_casino": ('{"type":"build_casino","x":20,"y":30}',
                         f"Casino{_size(C.CASINO)} next to a road: ${C.CASINO_SPEC['cost']:,}, upkeep ${C.CASINO_SPEC['upkeep']}/mo. "
                         f"{C.CASINO_SPEC['jobs']} low-skill jobs, ~{C.CASINO_SPEC['tourism']} gamblers a month paying "
                         f"${C.CASINO_SPEC['tax_per_tourist']} gambling tax each (more with airports and links to other cities), but "
                         f"+{int(C.CASINO_SPEC['crime'] * 100)}% crime within {C.CASINO_SPEC['crime_radius']} tiles. A gambling "
                         "specialisation suits a city with little education; high tech needs graduates instead."),
        "build_airport": ('{"type":"build_airport","x":35,"y":8}',
                          f"Airport{_size(C.AIRPORT)} touching a road: ${C.AIRPORT_SPEC['cost']:,}, upkeep "
                          f"${C.AIRPORT_SPEC['upkeep']}/mo, +{C.AIRPORT_SPEC['freight']:.0f} freight capacity, "
                          f"x{C.AIRPORT_SPEC['tourism_boost']} tourists, very noisy. In region mode: air links to every other "
                          "city with an airport."),
        "build_connector": ('{"type":"build_connector","kind":"highway","to":"south","x":20,"y":47}',
                            "Region mode only: link to a neighbouring city across the map edge that faces it. "
                            + "; ".join(f"{k}: ${v['cost']:,}, {v['goods']:.0f} goods + {v['people']:.0f} people a month"
                                        for k, v in C.CONNECTOR.items())
                            + ". Highways must continue an existing road; the link opens when the other city builds its side."),
        "preserve_historic": ('{"type":"preserve_historic","x1":10,"y1":20,"x2":14,"y2":22}',
                              f"List developed buildings (level {C.HISTORIC_MIN_LEVEL}+) in a rectangle as historic: they never "
                              f"level up or decline, and add +{int(C.HISTORIC_LV_BONUS * 100)}% land value within 3 tiles and "
                              f"{C.HISTORIC_TOURISTS} tourists/month each. Max {C.HISTORIC_MAX_TILES} tiles. Free."),
        "delist_historic": ('{"type":"delist_historic","x1":10,"y1":20,"x2":14,"y2":22}', "Remove the historic listing."),
        "clean_soil": ('{"type":"clean_soil","x1":2,"y1":20,"x2":8,"y2":24}',
                       f"Remediate contaminated soil in a rectangle: ${C.SOIL_CLEANUP_COST}/tile, removes "
                       f"{int(C.SOIL_CLEANUP_SHARE * 100)}% of the contamination (soil pollution hurts health, land value "
                       "and farms, and only fades slowly on its own)."),
        "build_metro_station": ('{"type":"build_metro_station","x":12,"y":22}',
                                f"Metro station on an empty tile touching a road: ${C.METRO_STATION['cost']:,}, upkeep "
                                f"${C.METRO_STATION['upkeep']}/mo. Serves homes and jobs within {C.METRO_RADIUS} tiles."),
        "create_metro_line": ('{"type":"create_metro_line","stations":[[12,22],[30,24],[20,36]]}',
                              f"Underground line through 2-{C.MAX_STOPS} metro stations: ${C.METRO_TRACK_COST}/tile of tunnel "
                              f"(straight-line distance), upkeep ${C.METRO_LINE_UPKEEP}/mo, up to {C.METRO_CAPACITY:.0f} riders/month, "
                              "about three times faster than driving and immune to traffic jams."),
        "remove_metro_line": ('{"type":"remove_metro_line","line":1}', "Close a metro line."),
        "set_service_imports": ('{"type":"set_service_imports","enabled":true}',
                                f"Allow or stop buying power and water from the region when short (up to "
                                f"{int(C.IMPORT_SHARE_MAX * 100)}% of demand, ${C.IMPORT_PRICE['power']}/power and "
                                f"${C.IMPORT_PRICE['water']}/water per unit, +{int(C.IMPORT_PRICE_GROWTH * 100)}% per 1,000 residents). On by default."),
        "upgrade_facility": ('{"type":"upgrade_facility","x":14,"y":25,"module":"expansion"}',
                             'Upgrade one of your own facilities. Every facility build action (power, water, sewage and '
                             'service buildings) takes "tier": n (absolute within the line, 1 = base) or a tier name as kind. module "tier": move the building one tier up '
                             f"for the price difference x{C.TIER_UPGRADE_PREMIUM:g}; bigger tiers "
                             "grow into free tiles around it; keeps age and modules. " + "; ".join(
                                 f"{k} ({int(v['cost_share'] * 100)}%" + (" of its build cost" if i == 0 else "") + ": " +
                                 {"filter": "halves pollution of coal/gas plants, incinerators, landfills",
                                  "expansion": "+50% capacity / output / coverage capacity",
                                  "overhaul": "resets an ageing power plant (plants lose 5% output per year after 5 years)"}[k] + ")"
                                 for i, (k, v) in enumerate(C.MODULES.items()))),
        "build_harbour": ('{"type":"build_harbour","x":30,"y":45}',
                          f"Harbour on an empty riverbank tile within {C.HARBOUR_ROWS} rows of the south edge (the river mouth), "
                          f"touching a road: ${C.HARBOUR_SPEC['cost']:,}, upkeep ${C.HARBOUR_SPEC['upkeep']}/mo, "
                          f"+{C.FREIGHT_CAPACITY['harbour']:.0f} freight capacity for exports, counts as a regional link."),
        "accept_offer": ('{"type":"accept_offer","offer":1}', "Accept an open offer from the events list (e.g. a big factory)."),
        "decline_offer": ('{"type":"decline_offer","offer":1}', "Decline an open offer."),
        "set_district_tax": ('{"type":"set_district_tax","district":"Docks","sector":"I","delta":-2}',
                             f"Tax modifier for one district: city rate +/- up to {C.DISTRICT_TAX_RANGE:.0f} points. "
                             "Also enact_policy/repeal_policy accept \"district\": name to apply a policy in one district."),
        "paint_district": ('{"type":"paint_district","name":"Docks","x1":2,"y1":20,"x2":12,"y2":30}',
                           f"Paint a named district over a rectangle (max {C.MAX_DISTRICTS} districts). The mayor can then "
                           "set district policies and taxes."),
        "specialize_district": ('{"type":"specialize_district","district":"Docks","industry":"hightech"}',
                                "New industry in the district becomes agriculture / forestry / ore / oil (on the matching "
                                "resource) or hightech (needs educated workers); 'none' to clear."),
        "build_landmark": ('{"type":"build_landmark","kind":"plaza","x":12,"y":22}',
                           "Landmark next to a road: " + "; ".join(
                               f"{k}: ${v['cost']:,}, upkeep ${v['upkeep']}/mo, +{int(100 * v['land_value'])}% land value within "
                               f"{v['radius']}, {v['tourism']} tourists/mo" for k, v in C.LANDMARKS.items())),
        "build_school": ('{"type":"build_school","kind":"elementary","x":14,"y":25}',
                         "School next to a road. " + " ".join(f"{k}: {_svc_doc(v, head=False)}" for k, v in SCHOOL_KINDS.items())),
        "noop": ('{"type":"noop"}', "Do nothing."),
    }


def apply_action(world: World, role: str, action: dict) -> ActionResult:
    if not isinstance(action, dict):
        return ActionResult({"raw": action}, False, "action must be a JSON object")
    kind = str(action.get("type", ""))
    allowed = ROLES[role]["actions"]
    if kind not in HANDLERS:
        return ActionResult(action, False, f"unknown action type '{kind}'. Allowed: {allowed}")
    if kind not in allowed:
        return ActionResult(action, False, f"the {role} role cannot use '{kind}'. Allowed: {allowed}")
    try:
        fn = HANDLERS[kind]
        msg, cost = fn(world, action, role) if kind in ROLE_AWARE else fn(world, action)
    except ActionError as e:
        return ActionResult(action, False, str(e))
    world.refresh()
    return ActionResult(action, True, msg, round(cost, 2))
