"""The city simulation: map, services, growth, economy.

`World` is the single source of truth for the game. Agents never touch it
directly; they go through `civitas.core` (observations + validated actions).
One call to `tick()` simulates one month.

The systems live in mixins: `population.PopulationMixin` (age x education
cohorts, job matching), `services.ServicesMixin` (road-network coverage,
crime, fire, garbage, health, deathcare) and `civics.CivicsMixin` (needs,
happiness, complaints, approval, funding levels, policies).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import catalog as C
from .grid import (box_sum, convolve, dilate, distance_to, label_components,
                   make_kernel)
from .civics import CivicsMixin
from .economy import EconomyMixin, TradeMixin
from .events import EventsMixin
from .progress import HeritageMixin, ProgressMixin
from .population import PopulationMixin
from .services import ServicesMixin
from .traffic import TrafficMixin

POLLUTION_KERNEL = make_kernel(radius=5, falloff=2.0)
DEPARTMENTS = ("planning", "utilities", "transport", "police", "fire", "health", "education")


@dataclass
class Network:
    id: int
    outside_connected: bool
    road_tiles: int
    power_supply: float = 0.0
    power_demand: float = 0.0
    water_supply: float = 0.0
    water_demand: float = 0.0
    water_quality: float = 1.0
    sewage_supply: float = 0.0
    sewage_demand: float = 0.0
    plants: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "outside_connected": self.outside_connected,
            "road_tiles": self.road_tiles,
            "power_supply": round(self.power_supply, 1),
            "power_demand": round(self.power_demand, 1),
            "water_supply": round(self.water_supply, 1),
            "water_demand": round(self.water_demand, 1),
            "water_quality": round(self.water_quality, 2),
            "sewage_supply": round(self.sewage_supply, 1),
            "sewage_demand": round(self.sewage_demand, 1),
            "plants": self.plants,
        }


class World(PopulationMixin, ServicesMixin, CivicsMixin, TrafficMixin, EconomyMixin, TradeMixin, EventsMixin,
            ProgressMixin, HeritageMixin):
    def __init__(self, width: int = 48, height: int = 48, seed: int = 0,
                 start_funds: float = 30000.0, river_width: int = 3, months: int = 120,
                 events=None, milestones: bool = False, river_start: float | None = None, seasons: bool = False):
        self.w, self.h = width, height
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.start_funds = float(start_funds)

        shape = (height, width)
        self.terrain = np.zeros(shape, np.uint8)
        self.struct = np.zeros(shape, np.uint8)
        self.level = np.zeros(shape, np.uint8)
        self.anchor = np.full(shape, -1, np.int32)     # multi-tile buildings: flat index of the top-left tile
        self.pollution = np.zeros(shape)
        self.plant_pollution = np.zeros(shape)
        self.land_value = np.zeros(shape)
        self.powered = np.zeros(shape, bool)
        self.watered = np.zeros(shape, bool)
        self.sewered = np.zeros(shape, bool)
        self.water_quality = np.zeros(shape)
        self.river_pollution = np.zeros(shape)     # 0..1 on river tiles
        self.river_row = np.zeros(height)          # 0..1 per row (north -> south)
        self.upstream_inflow = 0.0                 # raw pollution entering at the north edge
        self.river_outflow = 0.0                   # raw pollution leaving at the south edge
        self.sewage_discharge: list[tuple[int, int, float]] = []
        self.access = np.zeros(shape, bool)
        self.happiness = np.full(shape, 0.5)
        self.net_owner = np.full(shape, -1, np.int32)
        self.road_labels = np.full(shape, -1, np.int32)
        self.road_key = None
        self.networks: list[Network] = []

        self.month = 0
        self.treasury = self.start_funds
        self.dept_funds = {d: 0.0 for d in DEPARTMENTS}
        self.taxes = {"R": C.DEFAULT_TAX, "C": C.DEFAULT_TAX, "I": C.DEFAULT_TAX}
        self.debt = 0.0
        self.months_negative = 0
        self.bankrupt = False
        self.demand = {"R": 0.0, "R_low": 0.0, "R_high": 0.0, "C": 0.0, "I": 0.0, "O": 0.0}
        self.stats: dict = {}
        self.history: list[dict] = []
        self.events: list[str] = []          # events of the month in progress
        self.last_events: list[str] = []     # events of the last completed month
        self.ledger = self._new_ledger()
        self.last_ledger = self._new_ledger()
        self.version = 0                     # bumped on every change (metrics cache key)
        self._metrics_cache = (None, None)
        self._init_population()
        self._init_services()
        self._init_civics()
        self._init_traffic()
        self._init_economy()
        self._init_trade()
        self._init_events(events, months)
        self._init_progress(milestones)
        self._init_heritage(months, seasons)

        self.deal_cash = 0.0                 # region mode: money from neighbour deals this month
        self.region_info = None              # region mode: {"city": name, "neighbors": {city: side}}
        self.connectors: list[dict] = []     # region mode: highway / rail links built towards other cities
        self.regional = dict(C.REGIONAL_ZERO)   # region mode: last month's flows with the other cities
        self._generate_map(river_width, river_start)
        self.dist_water = distance_to(self.terrain == C.WATER)
        self._generate_resources()
        self.refresh()
        self._compute_environment()
        self._compute_population()
        self._compute_services()
        self._compute_demand()
        self._compute_happiness()

    # ------------------------------------------------------------------ map
    def _generate_map(self, river_width: int, river_start: float | None = None) -> None:
        """Flat plain with one meandering north-south river (continuing an upstream city's river if given)."""
        rng = self.rng
        x = self.w * 0.62 + rng.uniform(-2, 2)
        phase = rng.uniform(0, 2 * math.pi)
        if river_start is not None:
            phase = -math.pi / 2                   # sin term starts at its lowest: the river enters where it left
            x = river_start + 2.5
        self.river_path = []
        for y in range(self.h):
            x += rng.choice([-0.6, 0, 0, 0.6])
            x = min(max(x, self.w * 0.5), self.w * 0.75)
            cx = x + 2.5 * math.sin(phase + y / 7.0)
            width = river_width + (1 if rng.random() < 0.25 else 0)
            x0 = int(round(cx - width / 2))
            self.terrain[y, max(0, x0):min(self.w, x0 + width)] = C.WATER
            self.river_path.append(float(cx))

        # outside connection on the west edge + starter road + town hall
        yh = self.h // 2
        self.highway = (0, yh)
        self.struct[yh, 0] = C.HIGHWAY
        river_x = int(np.argmax(self.terrain[yh] == C.WATER))
        self.townhall = (max(6, river_x - 9), yh - 1)
        for xx in range(1, self.townhall[0] + 3):
            self.struct[yh, xx] = C.ROAD
        tx, ty = self.townhall
        self.struct[ty, tx] = C.TOWNHALL

    # ---------------------------------------------------------- utilities
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def is_riverbank(self, x: int, y: int) -> bool:
        if self.terrain[y, x] != C.LAND:
            return False
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if self.in_bounds(nx, ny) and self.terrain[ny, nx] == C.WATER:
                return True
        return False

    def adjacent_network(self, x: int, y: int) -> int:
        """Id of the road network touching (x, y) (8-neighbourhood), or -1."""
        best = -1
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if (dx or dy) and self.in_bounds(nx, ny):
                    lab = int(self.road_labels[ny, nx])
                    if lab >= 0 and (best < 0 or self._net_priority(lab) < self._net_priority(best)):
                        best = lab
        return best

    # ----------------------------------------------- multi-tile buildings
    def parts(self, x: int, y: int) -> list[tuple[int, int]]:
        """All tiles of the building at (x, y) (just (x, y) for one-tile buildings)."""
        a = int(self.anchor[y, x])
        if a < 0:
            return [(x, y)]
        ax, ay = a % self.w, a // self.w
        fw, fh = C.FOOTPRINT.get(int(self.struct[ay, ax]), (1, 1))
        return [(ax + i, ay + j) for j in range(fh) for i in range(fw)]

    def building_network(self, x: int, y: int) -> int:
        """Road network touching any tile of the building at (x, y), or -1."""
        best = -1
        for px, py in self.parts(x, y):
            lab = self.adjacent_network(px, py)
            if lab >= 0 and (best < 0 or self._net_priority(lab) < self._net_priority(best)):
                best = lab
        return best

    def footprint_tiles(self, struct: int, x: int, y: int, terrain: int = C.LAND) -> list[tuple[int, int]]:
        """Tiles a new building would cover; raises ValueError if they are not all free."""
        fw, fh = C.FOOTPRINT.get(struct, (1, 1))
        tiles = [(x + i, y + j) for j in range(fh) for i in range(fw)]
        for tx, ty in tiles:
            if not self.in_bounds(tx, ty):
                raise ValueError(f"a {fw}x{fh} building at ({x},{y}) does not fit on the map (x,y is its top-left tile)")
            if self.terrain[ty, tx] != terrain or self.struct[ty, tx] != C.NONE:
                raise ValueError(f"({tx},{ty}) is not an empty {'land' if terrain == C.LAND else 'river'} tile"
                                 + (f" (this building covers {fw}x{fh} tiles from ({x},{y}))" if fw * fh > 1 else ""))
        if not any(self.adjacent_network(tx, ty) >= 0 for tx, ty in tiles):
            raise ValueError(f"({x},{y}) does not touch a road; buildings must be next to a road")
        return tiles

    def place_building(self, struct: int, x: int, y: int, terrain: int = C.LAND) -> list[tuple[int, int]]:
        tiles = self.footprint_tiles(struct, x, y, terrain)
        for tx, ty in tiles:
            self.struct[ty, tx] = struct
            self.level[ty, tx] = 0
            self.anchor[ty, tx] = y * self.w + x if len(tiles) > 1 else -1
        self.facility_state[(x, y)] = {"built": self.month}
        return tiles

    def remove_building(self, x: int, y: int) -> None:
        tiles = self.parts(x, y)
        for tx, ty in tiles:
            self.struct[ty, tx] = C.NONE
            self.anchor[ty, tx] = -1
        self.facility_state.pop(tiles[0], None)

    def n_buildings(self, struct: int) -> int:
        return len(self.facilities(struct))

    # ------------------------------------------------------ people & jobs
    def residents_grid(self) -> np.ndarray:
        """Residents per tile: homes plus the flats above mixed-use shops."""
        lvl, s = self.level, self.struct
        return (np.array(C.RES_CAPACITY)[lvl] * (s == C.RES) + np.array(C.MIX_RES)[lvl] * (s == C.MIX)).astype(float)

    def homes_mask(self) -> np.ndarray:
        return np.isin(self.struct, C.HOMES) & (self.level > 0)

    def shop_jobs_grid(self) -> np.ndarray:
        lvl, s = self.level, self.struct
        return (np.array(C.COM_JOBS)[lvl] * (s == C.COM) + np.array(C.MIX_JOBS)[lvl] * (s == C.MIX)).astype(float)

    def _net_priority(self, lab: int) -> tuple:
        net = self.networks[lab]
        return (not net.outside_connected, -net.road_tiles, lab)

    def _new_ledger(self) -> dict:
        return {"spent": {d: 0.0 for d in DEPARTMENTS}, "allocated": {d: 0.0 for d in DEPARTMENTS},
                "income": {}, "expenses": {}}

    def event(self, text: str) -> None:
        self.events.append(text)

    def touch(self) -> None:
        """Mark the world as changed (invalidates cached metrics)."""
        self.version += 1



    # ----------------------------------------------------------- services
    def refresh(self) -> None:
        """Recompute road networks, access, and power / water distribution.

        Cheap enough to call after every agent's actions so everybody sees
        the up-to-date coverage.
        """
        self.touch()
        s = self.struct
        road_mask = (s == C.ROAD) | (s == C.HIGHWAY)
        rc = self.road_class
        rc[~road_mask] = 0
        rc[(s == C.ROAD) & (rc == 0)] = 1
        rc[s == C.HIGHWAY] = 3
        key = hash(road_mask.tobytes() + rc.tobytes())
        if key != self.road_key:
            self._road_labels, self._n_nets = label_components(road_mask)
            self.road_key = key
        labels, n = self._road_labels, self._n_nets
        self.road_labels = labels
        outside = {int(labels[y, x]) for y, x in zip(*np.nonzero(s == C.HIGHWAY))} - {-1}
        for x, y in self.facilities(C.TRAIN) + self.facilities(C.HARBOUR) + self.facilities(C.AIRPORT):
            # train stations, harbours and airports link their road network to the region
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for px, py in self.parts(x, y):
                        if self.in_bounds(px + dx, py + dy) and labels[py + dy, px + dx] >= 0:
                            outside.add(int(labels[py + dy, px + dx]))
        self.networks = [Network(i, i in outside, int((labels == i).sum())) for i in range(n)]

        connected_roads = np.isin(labels, list(outside)) & ((rc < 3) | (s == C.HIGHWAY))   # highways give no zone access
        self.access = dilate(connected_roads, C.ACCESS_RADIUS)

        # which network serves each tile (connected networks first, then largest)
        owner = np.full(s.shape, -1, np.int32)
        for lab in sorted(range(n), key=self._net_priority):
            area = dilate(labels == lab, C.ACCESS_RADIUS)
            owner[(owner == -1) & area] = lab
        self.net_owner = owner

        # consumers
        zone_mask = np.isin(s, C.ZONES)
        developed = zone_mask & (self.level > 0)
        p_need = np.zeros(s.shape)
        w_need = np.zeros(s.shape)
        for z in C.ZONES:
            m = developed & (s == z)
            p_need[m] = self.level[m] * C.POWER_PER_LEVEL[z]
            w_need[m] = self.level[m] * C.WATER_PER_LEVEL[z]
        tx, ty = self.townhall
        if s[ty, tx] == C.TOWNHALL:
            p_need[ty, tx] = C.TOWNHALL_POWER
            w_need[ty, tx] = C.TOWNHALL_WATER
        p_need *= self.event_factor("power_demand")
        w_need *= self.event_factor("water_demand")

        # producers
        plants_by_net: dict[int, list] = {i: [] for i in range(n)}
        ys, xs = np.nonzero(np.isin(s, list(C.UTILITY_TYPES) + [C.INCINERATOR]))
        u_eff = self.funding_eff("utilities")
        for y, x in zip(ys.tolist(), xs.tolist()):
            st = int(s[y, x])
            net = self.adjacent_network(x, y)
            if st == C.INCINERATOR:
                kind, spec = "incinerator", C.SERVICES["incinerator"]
            else:
                kind = C.KIND_BY_STRUCT[st]
                spec = C.CATALOG[kind]
            off = self.is_offline(x, y)
            f = 0.0 if off else u_eff * self.plant_output_factor(x, y, kind)
            info = {"kind": kind, "x": x, "y": y, "network": net,
                    "power": float(spec.get("power", 0)) * f, "water": float(spec.get("water", 0)) * f}
            if off:
                info["offline"] = True
            age = (self.month - self.facility_state.get((x, y), {}).get("built", self.month)) / 12.0
            if age >= 1:
                info["age_years"] = round(age, 1)
            info["sewage"] = float(spec.get("sewage", 0)) * f
            if kind == "pump":
                info["quality"] = round(float(max(0.2, 1.0 - 0.8 * max(self.pollution[y, x], self._river_near(x, y)))), 2)
            elif kind == "tower":
                info["quality"] = 1.0
            if net >= 0:
                plants_by_net[net].append(info)
        self.plant_list = [p for lst in plants_by_net.values() for p in lst]
        self.disconnected_plants = [
            {"kind": C.KIND_BY_STRUCT.get(int(s[y, x]), "incinerator"), "x": x, "y": y}
            for y, x in zip(ys.tolist(), xs.tolist()) if self.adjacent_network(x, y) < 0
        ]

        self.imports = {"power": 0.0, "water": 0.0}
        self.power_sold = 0.0
        powered = np.zeros(s.shape, bool)
        watered = np.zeros(s.shape, bool)
        sewered = np.zeros(s.shape, bool)
        wq = np.zeros(s.shape)
        discharge = []
        for lab in range(n):
            net = self.networks[lab]
            plants = plants_by_net[lab]
            net.plants = [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in p.items()
                           if k != "network" and not (k in ("power", "water", "sewage") and not v)} for p in plants]
            in_net = owner == lab
            for res_key, need, served_grid in (("power", p_need, powered), ("water", w_need, watered),
                                               ("sewage", w_need, sewered)):
                srcs = [p for p in plants if p[res_key] > 0]
                supply = sum(p[res_key] for p in srcs)
                cons = in_net & (need > 0)
                demand = float(need[cons].sum())
                if net.outside_connected and res_key in ("power", "water") and lab == self._main_net():
                    supply += self.external.get(res_key, 0.0)       # neighbour deals (+ bought, - sold)
                    supply = max(0.0, supply)
                    short = max(0.0, demand - supply)
                    if self.service_imports and short > 0:
                        imported = min(short, C.IMPORT_SHARE_MAX * demand)
                        supply += imported
                        self.imports[res_key] = imported
                        srcs = srcs + [{"x": self.highway[0], "y": self.highway[1], res_key: imported, "quality": 1.0,
                                        "kind": "import"}]
                    elif res_key == "power":
                        self.power_sold = max(0.0, supply - 1.1 * demand)
                if res_key == "power":
                    net.power_supply, net.power_demand = supply, demand
                elif res_key == "water":
                    net.water_supply, net.water_demand = supply, demand
                    if srcs:
                        net.water_quality = sum(p[res_key] * p["quality"] for p in srcs) / supply
                else:
                    net.sewage_supply, net.sewage_demand = supply, demand
                    handled = min(supply, demand)
                    for p in srcs:     # each facility treats its share of the collected sewage
                        factor = C.CATALOG[p["kind"]]["discharge"]
                        discharge.append((p["x"], p["y"], handled * p[res_key] / supply * factor))
                if supply <= 0:
                    continue
                cy, cx = np.nonzero(cons)
                if len(cy):
                    d = np.full(len(cy), 1e9)
                    for p in srcs:
                        d = np.minimum(d, np.abs(cy - p["y"]) + np.abs(cx - p["x"]))
                    order = np.argsort(d, kind="stable")
                    cum = np.cumsum(need[cy[order], cx[order]])
                    ok = cum <= supply + 1e-9
                    served_grid[cy[order][ok], cx[order][ok]] = True
                    leftover = supply - (cum[ok][-1] if ok.any() else 0.0)
                else:
                    leftover = supply
                if leftover > 0:
                    # undeveloped zones can hook up if there is spare capacity
                    idle = in_net & zone_mask & (self.level == 0)
                    served_grid[idle] = True
            if net.water_supply > 0:
                wq[in_net & watered] = net.water_quality
        self.powered, self.watered, self.water_quality = powered, watered, wq
        self.sewered = sewered
        self.sewage_discharge = discharge

    def _main_net(self) -> int:
        hx, hy = self.highway
        return int(self.road_labels[hy, hx])

    def _river_near(self, x: int, y: int) -> float:
        """Pollution of the river next to (x, y) (0 if not on the bank)."""
        return float(self.river_row[y]) if self.dist_water[y, x] <= 1 else 0.0

    # -------------------------------------------------------- environment
    def _compute_environment(self) -> None:
        s = self.struct
        src = np.zeros(s.shape)
        ind = s == C.IND
        src[ind] = self.level[ind] * C.IND_POLLUTION_PER_LEVEL * np.array(C.IND_POLLUTION_FACTOR)[self.ind_type[ind]]
        if self.policies_any("industry_filters"):
            src *= np.where(self.policy_mask("industry_filters"), C.POLICIES["industry_filters"]["pollution"], 1.0)
        plant_src = np.zeros(s.shape)
        for kind in ("coal", "gas"):
            plant_src[s == C.POWER_PLANTS[kind]] = C.CATALOG[kind]["pollution"]
        for kind, struct in (("landfill", C.LANDFILL), ("incinerator", C.INCINERATOR)):
            plant_src[s == struct] = C.SERVICES[kind]["pollution"]
        for (x, y), st in self.facility_state.items():          # filters halve plant pollution
            if "filter" in st.get("modules", ()) and plant_src[y, x] > 0:
                plant_src[y, x] *= C.MODULES["filter"]["pollution"]
        src[self.complex_mask & ind] *= C.COMPLEX_POLLUTION
        src = src + C.TRAFFIC_POLLUTION * self.traffic_load
        kernel = self.pollution_kernel()                      # drifts downwind
        raw = convolve(src + plant_src, kernel)
        self.pollution = 1.0 - np.exp(-raw / 18.0)
        self.plant_pollution = 1.0 - np.exp(-convolve(plant_src, kernel) / 18.0)

        self._river_flow()
        self._compute_noise()
        near_river = self.river_row[:, None] * (self.dist_water <= 3)
        water_bonus = 0.25 * np.clip(1.0 - self.dist_water / 4.0, 0, 1) * (1.0 - self.river_row[:, None])
        water_bonus[self.terrain == C.WATER] = 0
        parks = box_sum(s == C.PARK, 4)
        park_bonus = np.minimum(0.3, 0.08 * parks)
        tx, ty = self.townhall
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        center = 0.1 * np.clip(1.0 - (np.abs(xx - tx) + np.abs(yy - ty)) / 12.0, 0, 1)
        services = 0.1 * (self.powered & self.watered)
        safety = 0.05 * self.police_cov + 0.03 * self.fire_cov + 0.04 * self.health_cov + 0.04 * self.school_cov
        garbage = 0.15 * np.clip(self.garbage / C.GARBAGE_NORM, 0, 1)
        lv = (0.3 + water_bonus + park_bonus + center + services + safety + self.landmark_lv
              - 0.6 * self.pollution - 0.25 * self.crime - garbage - 0.15 * near_river - 0.1 * self.noise
              - 0.2 * self.soil_level() + C.HISTORIC_LV_BONUS * (box_sum(self.historic, 3) > 0))
        if self.policies_any("high_rise_ban"):
            lv = lv + C.POLICIES["high_rise_ban"]["land_value"] * self.policy_mask("high_rise_ban")
        self.land_value = np.clip(lv, 0, 1)

    def _river_flow(self) -> None:
        """Carry sewage and industrial runoff downstream (north -> south)."""
        inp = np.zeros(self.h)
        for x, y, amount in self.sewage_discharge:
            inp[y] += amount
        ind = (self.struct == C.IND) & (self.dist_water <= 2)
        if ind.any():
            ys = np.nonzero(ind)[0]
            inp += np.bincount(ys, weights=self.level[ind] * C.IND_RUNOFF, minlength=self.h)
        raw = np.zeros(self.h)
        carry = self.upstream_inflow
        for y in range(self.h):
            carry = carry * C.RIVER_DECAY + inp[y]
            raw[y] = carry
        self.river_outflow = carry
        self.river_row = 1.0 - np.exp(-raw / C.RIVER_NORM)
        self.river_pollution = np.where(self.terrain == C.WATER, self.river_row[:, None], 0.0)

    # ---------------------------------------------------------- population
    def _compute_population(self) -> None:
        s, lv = self.struct, self.level
        pop = float(self.residents_grid().sum())
        self._rescale_population(pop)
        jobs_c = float(self.shop_jobs_grid().sum())
        jobs_g = float(self.n_buildings(C.CASINO) * C.CASINO_SPEC["jobs"])
        jobs_o = float(np.array(C.OFF_JOBS)[lv][s == C.OFF].sum())
        ind_jobs = np.array(C.IND_JOBS)[lv] * (s == C.IND)
        sectors = {"C": jobs_c, "O": jobs_o, "G": jobs_g}
        for k in range(6):
            v = float(ind_jobs[self.ind_type == k].sum())
            if v > 0:
                sectors["I" if k == 0 else f"I{k}"] = v
        match = self._match_jobs(sectors)
        self._regional_commuting(match, sectors)
        jobs_i = float(ind_jobs.sum())
        workers = match["workers"]
        filled_i = sum(v for k, v in match["filled"].items() if k.startswith("I"))
        self.stats.update({
            "population": pop, "jobs_com": jobs_c, "jobs_ind": jobs_i, "jobs_off": jobs_o,
            "workers": workers, "jobs_filled_com": match["filled"]["C"],
            "jobs_filled_ind": filled_i, "jobs_filled_off": match["filled"]["O"],
            "jobs_casino": jobs_g, "jobs_filled_casino": match["filled"].get("G", 0.0),
            "unemployment": match["unemployment"],
            "unemployment_by_edu": match["unemployment_by_edu"],
            "skilled_fill": match["skilled_fill"], "sick_workers": match["sick_workers"],
            "education_index": self.education_index(),
        })
        self._household_budget(match)

    def dept_upkeep(self) -> dict[str, float]:
        """Monthly upkeep per department at the current funding levels."""
        s = self.struct
        out: dict[str, float] = {}
        water = self.terrain == C.WATER
        roads = float(((s == C.ROAD) & ~water).sum()) * C.upkeep_of(C.ROAD)
        bridges = float(((s == C.ROAD) & water).sum()) * C.upkeep_of(C.ROAD, True)
        out["roads"] = (roads + bridges) * self.funding.get("transport", 1.0)
        out["parks"] = float((s == C.PARK).sum()) * C.upkeep_of(C.PARK)
        util = [int(v) for v in s[np.isin(s, list(C.UTILITY_TYPES))].tolist()]
        out["utilities"] = sum(C.upkeep_of(v) for v in util) * self.funding.get("utilities", 1.0)
        out["landmarks"] = sum(C.upkeep_of(st) * self.n_buildings(st) for st in (*C.LANDMARK_BY_STRUCT, C.CASINO))
        out["transport"] = (self.transit_upkeep() + self.road_upgrade_upkeep()
                            + sum(C.upkeep_of(st) * self.n_buildings(st) for st in (C.TRAIN, C.HARBOUR, C.METRO, C.AIRPORT))
                            ) * self.funding.get("transport", 1.0)
        for dept in ("police", "fire", "health", "education"):
            structs = [v["struct"] for v in C.SERVICES.values() if v["dept"] == dept] + ([C.LEVEE] if dept == "fire" else [])
            out[dept] = sum(C.upkeep_of(st) * self.n_buildings(st) for st in structs) * self.funding.get(dept, 1.0)
        return out

    # ------------------------------------------------------ region mode
    def _regional_commuting(self, match: dict, sectors: dict) -> None:
        """Workers commuting to and from linked cities (set by the region each month)."""
        workers = match["workers"]
        unemployed = match["unemployment"] * workers
        out = min(self.regional.get("commuters_out", 0.0), unemployed)
        if out > 0 and workers > 0:
            match["unemployment"] = (unemployed - out) / workers
        jobs = sum(sectors.values())
        filled = sum(match["filled"].values())
        free = max(0.0, jobs - filled)
        cin = min(self.regional.get("commuters_in", 0.0), free)
        if cin > 0 and free > 0:
            for k, v in sectors.items():
                match["filled"][k] = match["filled"].get(k, 0.0) + cin * max(0.0, v - match["filled"].get(k, 0.0)) / free
        self.stats["unfilled_jobs"] = max(0.0, free - cin)
        self.stats["unemployed"] = match["unemployment"] * workers

    def specializations(self) -> dict[str, float]:
        """How specialised the economy is (0..1), SimCity 2013 style."""
        st, tr = self.stats, self.trade
        pop = max(1.0, st.get("population", 0.0))
        s, lvl = self.struct, self.level
        ind = (s == C.IND) & (lvl > 0)
        ij = np.array(C.IND_JOBS)[lvl] * ind
        jobs = max(1.0, st.get("jobs_com", 0) + st.get("jobs_ind", 0) + st.get("jobs_off", 0) + st.get("jobs_casino", 0))
        clip = lambda v: round(float(max(0.0, min(1.0, v))), 2)
        return {
            "mining & farming": clip(float(ij[(self.ind_type >= 1) & (self.ind_type <= 4)].sum()) / jobs / 0.2),
            "high tech": clip((float(ij[self.ind_type == 5].sum()) + st.get("jobs_off", 0)) / jobs / 0.3),
            "gambling": clip(st.get("jobs_casino", 0) / jobs / 0.06),
            "tourism": clip(st.get("tourists", 0) / pop / 0.12),
            "trade": clip((tr.get("exports", 0) + tr.get("imports", 0) + self.regional.get("goods_out", 0)
                           + self.regional.get("goods_in", 0)) / pop / 0.3),
            "education": clip((st.get("seats_university", 0) + 0.3 * st.get("seats_high", 0)) / pop / 0.2),
        }

    def _job_tax(self, sector: str, struct: int) -> float:
        """Average tax rate over the jobs of one zone type (district modifiers included)."""
        rate = self.tax_grid(sector)
        if np.isscalar(rate):
            return float(rate)
        if struct == C.COM:
            w = self.shop_jobs_grid()
        else:
            jobs = {C.OFF: C.OFF_JOBS, C.IND: C.IND_JOBS}[struct]
            w = np.array(jobs)[self.level] * (self.struct == struct)
        return float((rate * w).sum() / w.sum()) if w.sum() > 0 else self.taxes[sector]

    def _economy(self) -> None:
        st, t = self.stats, self.taxes
        income = {
            "residential_tax": self.residential_taxable(),
            "commercial_tax": self._job_tax("C", C.COM) * st["jobs_filled_com"] * C.TAXABLE_PER_JOB
            * (1.0 - 0.5 * self.trade.get("goods_shortage", 0.0)),
            "office_tax": self._job_tax("C", C.OFF) * st.get("jobs_filled_off", 0.0) * C.TAXABLE_PER_OFFICE_JOB,
            "industrial_tax": self._job_tax("I", C.IND) * st["jobs_filled_ind"] * C.TAXABLE_PER_JOB
            * (0.7 + 0.3 * self.trade.get("sold_share", 1.0)),
            "trade_fees": C.EXPORT_FEE * self.trade.get("exports", 0.0),
            "power_sales": C.POWER_SELL_PRICE * self.power_sold,
            "transit_fares": 0.0 if self.policy_on("free_transit") else st.get("transit_riders", 0.0) * self.fare,
            "tourism": st.get("tourists", 0.0) * C.TOURIST_SPEND,
            "gambling_tax": st.get("casino_tourists", 0.0) * C.CASINO_SPEC["tax_per_tourist"],
            "regional_trade": max(0.0, self.regional.get("cash", 0.0)),
            "neighbour_deals": max(0.0, self.deal_cash),
        }
        expenses = {k: v for k, v in self.dept_upkeep().items()}
        expenses.update({
            "town_hall": C.upkeep_of(C.TOWNHALL),
            "city_services": st["population"] * C.SERVICE_COST_PER_RESIDENT,
            "policies": self.policy_costs(),
            "service_imports": sum(v * self.service_import_price(k) for k, v in self.imports.items()),
            "neighbour_deals": max(0.0, -self.deal_cash),
            "regional_trade": max(0.0, -self.regional.get("cash", 0.0)),
            "loan_interest": self.debt * C.LOAN_INTEREST,
        })
        self.ledger["income"] = {k: round(v, 2) for k, v in income.items()}
        self.ledger["expenses"] = {k: round(v, 2) for k, v in expenses.items()}
        self.treasury += sum(income.values()) - sum(expenses.values())
        if self.treasury < 0:
            self.months_negative += 1
            self.event(f"Treasury is negative ({self.treasury:,.0f}) for {self.months_negative} month(s); "
                       f"bankruptcy after {C.BANKRUPTCY_MONTHS}")
            if self.months_negative >= C.BANKRUPTCY_MONTHS:
                self.bankrupt = True
                self.event("The city is BANKRUPT. Game over.")
        else:
            self.months_negative = 0

    # ---------------------------------------------------------------- tick
    def tick(self) -> dict:
        """Advance the simulation by one month; returns that month's metrics."""
        self.burning, self.flooded = [], []
        self.stats.update({"flood_damaged": 0, "flood_casualties": 0.0})
        self._run_events()
        self._compute_landmarks()
        self.refresh()
        self._compute_environment()
        self._compute_population()
        self._compute_demand()
        self._grow()
        self.refresh()
        self._compute_environment()
        self._step_cohorts()
        self._compute_population()
        self._compute_traffic()
        self._compute_services()
        self._update_crime()
        self._fire_events()
        self._garbage_and_health()
        self._deathcare()
        self._deplete_resources()
        self._update_soil()
        self._update_heritage()
        self._compute_trade()
        self._company_profits()
        self._compute_happiness()
        self._economy()
        self._update_approval()
        self._compute_demand()
        self._check_milestones()
        self.month += 1
        self.touch()
        metrics = self.metrics()
        self.history.append(metrics)
        self.last_events, self.events = self.events, []
        self.last_ledger, self.ledger = self.ledger, self._new_ledger()
        return metrics

    # ------------------------------------------------------------- metrics
    def metrics(self) -> dict:
        if self._metrics_cache[0] == self.version:
            return self._metrics_cache[1]
        m = self._metrics()
        self._metrics_cache = (self.version, m)
        return m

    def _metrics(self) -> dict:
        s, lvl = self.struct, self.level
        st = self.stats
        zone_mask = np.isin(s, C.ZONES)
        developed = zone_mask & (lvl > 0)
        p_need = np.zeros(s.shape)
        w_need = np.zeros(s.shape)
        for z in C.ZONES:
            m = developed & (s == z)
            p_need[m] = lvl[m] * C.POWER_PER_LEVEL[z]
            w_need[m] = lvl[m] * C.WATER_PER_LEVEL[z]
        tot_p, tot_w = p_need.sum(), w_need.sum()
        residents = self.residents_grid()
        rsum = residents.sum()
        wavg = lambda g: float((g * residents).sum() / rsum) if rsum > 0 else 0.0
        power_supply = sum(n.power_supply for n in self.networks)
        power_demand = sum(n.power_demand for n in self.networks)
        water_supply = sum(n.water_supply for n in self.networks)
        water_demand = sum(n.water_demand for n in self.networks)
        sewage_supply = sum(n.sewage_supply for n in self.networks)
        sewage_demand = sum(n.sewage_demand for n in self.networks)
        river = self.river_row
        zoned = {L: int((s == z).sum()) for L, z in C.ZONE_BY_LETTER.items()}
        dev = {L: int(((s == z) & (lvl > 0)).sum()) for L, z in C.ZONE_BY_LETTER.items()}
        income = sum(self.ledger["income"].values())
        expenses = sum(self.ledger["expenses"].values())
        util_structs = s[np.isin(s, list(C.UTILITY_TYPES))].tolist()
        clean = sum(float(C.CATALOG[C.KIND_BY_STRUCT[v]].get("power", 0)) for v in util_structs
                    if C.KIND_BY_STRUCT[v] in ("solar", "wind"))
        dept_total = sum(self.dept_funds.values())
        upkeep = self.dept_upkeep()
        pop = int(st["population"])
        return {
            "month": self.month,
            "population": pop,
            "jobs_com": int(st["jobs_com"]), "jobs_ind": int(st["jobs_ind"]), "jobs_off": int(st.get("jobs_off", 0)),
            "workers": int(st["workers"]),
            "unemployment": round(st["unemployment"], 4),
            "happiness": round(st.get("happiness", 0.5), 4),
            "treasury": round(self.treasury, 2),
            "dept_funds": {k: round(v, 2) for k, v in self.dept_funds.items()},
            "debt": round(self.debt, 2),
            "net_worth": round(self.treasury + dept_total - self.debt, 2),
            "income": round(income, 2), "expenses": round(expenses, 2),
            "income_breakdown": dict(self.ledger["income"]),
            "expense_breakdown": dict(self.ledger["expenses"]),
            "dept_spent": dict(self.ledger["spent"]),
            "dept_allocated": dict(self.ledger["allocated"]),
            "dept_upkeep": {k: round(v, 2) for k, v in upkeep.items()},
            "funding": dict(self.funding),
            "policies": sorted(self.policies),
            "taxes": dict(self.taxes),
            "demand": {k: round(v, 3) for k, v in self.demand.items()},
            "power_supply": round(power_supply, 1), "power_demand": round(power_demand, 1),
            "power_clean_capacity": clean,
            "utility_upkeep": round(sum(C.upkeep_of(v) for v in util_structs), 2),
            "water_supply": round(water_supply, 1), "water_demand": round(water_demand, 1),
            "power_coverage": round(float(p_need[self.powered].sum() / tot_p), 4) if tot_p > 0 else None,
            "water_coverage": round(float(w_need[self.watered].sum() / tot_w), 4) if tot_w > 0 else None,
            "sewage_supply": round(sewage_supply, 1), "sewage_demand": round(sewage_demand, 1),
            "sewage_coverage": round(float(w_need[self.sewered].sum() / tot_w), 4) if tot_w > 0 else None,
            "river_pollution": round(float(river.mean()), 4), "river_pollution_max": round(float(river.max()), 4),
            "river_outflow": round(float(self.river_outflow), 2),
            "zoned": zoned, "developed": dev,
            "zoned_total": int(zone_mask.sum()), "developed_total": int(developed.sum()),
            "zoned_no_access": int((zone_mask & ~self.access).sum()),
            "zoned_no_power": int((zone_mask & self.access & ~self.powered).sum()),
            "zoned_no_water": int((zone_mask & self.access & self.powered & ~self.watered).sum()),
            "zoned_no_sewage": int((zone_mask & self.access & self.watered & ~self.sewered).sum()),
            "road_tiles": int((s == C.ROAD).sum()),
            "parks": int((s == C.PARK).sum()),
            "pollution_res": round(wavg(self.pollution), 4),
            "plant_pollution_res": round(wavg(self.plant_pollution), 4),
            "water_quality_res": round(wavg(self.water_quality), 4),
            "land_value_dev": round(float(self.land_value[developed].mean()), 4) if developed.any() else 0.0,
            # services
            "crime_res": round(wavg(self.crime), 4),
            "police_coverage": round(st.get("police_coverage", 0.0), 4),
            "inmates": round(st.get("inmates", 0.0), 1), "jail_capacity": st.get("jail_capacity", 0),
            "fire_coverage": round(st.get("fire_coverage", 0.0), 4),
            "fires": st.get("fires", 0), "fire_tiles_lost": st.get("fire_tiles_lost", 0),
            "health": round(st.get("health", 0.75), 4),
            "health_coverage": round(st.get("health_coverage", 0.0), 4),
            "garbage_coverage": round(st.get("garbage_coverage", 0.0), 4),
            "garbage_backlog": round(st.get("garbage_backlog", 0.0), 1),
            "garbage_produced": round(st.get("garbage_produced", 0.0), 1),
            "garbage_homes_pct": round(st.get("garbage_homes_pct", 0.0), 4),
            "landfill_fill_pct": self._landfill_fill_pct(),
            "deaths": round(st.get("deaths", 0.0), 1), "death_backlog": round(st.get("death_backlog", 0.0), 1),
            "deathcare_capacity": round(st.get("deathcare_capacity", 0.0), 1),
            "school_coverage": round(st.get("school_coverage", 0.0), 4),
            "high_coverage": round(st.get("high_coverage", 0.0), 4),
            "enrollment": {k: round(v, 3) for k, v in self.enrollment.items()},
            "education_index": round(st.get("education_index", 0.0), 4),
            "skilled_fill": round(st.get("skilled_fill", 1.0), 4),
            "unemployment_by_edu": st.get("unemployment_by_edu", {}),
            "cohorts": self.cohort_summary(),
            "service_buildings": {k: len(self.facilities(v["struct"])) for k, v in C.SERVICES.items()},
            "noise_res": round(wavg(self.noise), 4),
            "soil_res": round(wavg(self.soil_level()), 4), "soil_max": round(float(self.soil_level().max()), 4),
            "historic_tiles": int(self.historic.sum()), "mixed_use": int(((s == C.MIX) & (lvl > 0)).sum()),
            "season": self.season(), "wind": self.wind_name(),
            "wealth": {k: int((self.homes_mask() & (self.wealth == v)).sum()) for k, v in (("low", 1), ("middle", 2), ("high", 3))},
            "industry_types": {name: int(((s == C.IND) & (lvl > 0) & (self.ind_type == k)).sum())
                               for k, name in enumerate(("generic", "agriculture", "forestry", "ore", "oil", "hightech"))},
            "density_high": int(((self.density == 1) & zone_mask).sum()),
            "levels_4_5": int((developed & (lvl >= 4)).sum()),
            "disposable_income": round(st.get("disposable_income", 0.0), 3),
            "income_per_resident": round(st.get("income_per_resident", 0.0), 3),
            "homeless_share": round(st.get("homeless_share", 0.0), 4),
            "demand_caps": dict(st.get("demand_caps", {})),
            "events_active": [e["kind"] for e in self.active_events()],
            "flood_damaged": st.get("flood_damaged", 0), "flood_casualties": round(st.get("flood_casualties", 0.0), 1),
            "levees": int((s == C.LEVEE).sum()), "shelter_coverage": round(st.get("shelter_coverage", 0.0), 4),
            "offers": [dict(o) for o in self.offers],
            "milestones": list(self.milestones_reached), "tourists": round(st.get("tourists", 0.0), 1),
            "landmarks": {k: self.n_buildings(v["struct"]) for k, v in C.LANDMARKS.items()},
            "districts": len(self.districts),
            "exports": round(self.trade.get("exports", 0.0), 1), "imports": round(self.trade.get("imports", 0.0), 1),
            "freight_capacity": round(self.trade.get("freight_capacity", 0.0), 1),
            "goods_shortage": round(self.trade.get("goods_shortage", 0.0), 4),
            "sold_share": round(self.trade.get("sold_share", 1.0), 4),
            "businesses_losing": st.get("businesses_losing", 0), "business_closures": st.get("business_closures", 0),
            "margins": {k: round(st.get(f"margin_{k}", 0.0), 3) for k in ("com", "ind", "off")},
            "service_imports": {k: round(v, 1) for k, v in self.imports.items()},
            "power_sold": round(self.power_sold, 1),
            "harbours": len(self.facilities(C.HARBOUR)),
            "airports": self.n_buildings(C.AIRPORT), "casinos": self.n_buildings(C.CASINO),
            "casino_tourists": round(st.get("casino_tourists", 0.0), 1),
            "specializations": self.specializations(),
            "regional": {k: round(v, 1) for k, v in self.regional.items()},
            "connectors": [dict(c) for c in self.connectors],
            "goods_produced": round(self.trade.get("goods_produced", 0.0), 1),
            "goods_needed": round(self.trade.get("goods_needed", 0.0), 1),
            "raw_produced": round(self.trade.get("raw_produced", 0.0), 1),
            "raw_needed": round(self.trade.get("raw_needed", 0.0), 1),
            "unfilled_jobs": round(st.get("unfilled_jobs", 0.0), 1), "unemployed": round(st.get("unemployed", 0.0), 1),
            "commute_minutes": round(st.get("commute_minutes", 0.0), 2),
            "congestion_index": round(st.get("congestion_index", 0.0), 4),
            "congestion_mean": round(st.get("congestion_mean", 0.0), 4),
            "transit_riders": round(st.get("transit_riders", 0.0), 1),
            "transit_share": round(st.get("transit_share", 0.0), 4),
            "trips": round(st.get("trips", 0.0), 1),
            "bus_lines": len(self.bus_lines), "fare": self.fare,
            "metro_lines": len(self.metro_lines), "metro_riders": round(st.get("metro_riders", 0.0), 1),
            "road_classes": {k: int((self.road_class == v).sum()) for k, v in C.ROAD_CLASSES.items()},
            "train_stations": len(self.facilities(C.TRAIN)),
            "approval": round(self.approval, 4), "low_approval_months": self.low_approval_months,
            "recalled": self.recalled,
            "complaints": list(self.complaints), "needs": dict(self.need_avg),
            "policy_cost": round(self.ledger["expenses"].get("policies", 0.0), 2),
            "months_negative": self.months_negative,
            "bankrupt": self.bankrupt,
        }

    def _landfill_fill_pct(self) -> float | None:
        fills = [self._fstate(x, y).get("fill", 0.0) for x, y in self.facilities(C.LANDFILL)]
        if not fills:
            return None
        return round(100 * sum(fills) / (len(fills) * C.SERVICES["landfill"]["fill"]), 1)

    # ---------------------------------------------------------- rendering
    def frame_layers(self) -> dict[str, np.ndarray]:
        """Per-tile 0..1 grids the viewer can show as map layers."""
        return {
            "crime": self.crime, "police": self.police_cov, "fire": self.fire_cov, "fire_risk": self.fire_risk,
            "health": np.where(np.isin(self.struct, C.ZONES) & (self.level > 0), self.health, 0.0),
            "garbage": np.clip(self.garbage / C.GARBAGE_NORM, 0, 1), "school": self.school_cov,
            "happiness": np.where(self.homes_mask(), self.happiness, 0.0),
            "affordability": np.where(self.homes_mask(), self.affordability, 0.0),
            "river": self.river_pollution, "noise": self.noise, "soil": self.soil_level(),
            "historic": self.historic.astype(float),
            "traffic": np.clip(self.congestion / 2.0, 0, 1),
            "transit": np.maximum(self.transit_cov.astype(float) * 0.6, self.metro_cov.astype(float)),
            "flood_risk": self._flood_risk_layer(),
            "profit": np.where(np.isin(self.struct, (C.COM, C.IND, C.OFF, C.MIX)) & (self.level > 0),
                               np.clip(0.5 + self.margin, 0, 1), 0.0),
        }

    def _flood_risk_layer(self) -> np.ndarray:
        nxt = next((e for e in self.schedule if e["kind"] == "flood" and 0 <= e["month"] - self.month <= 3), None)
        return self.flood_zone(nxt).astype(float) if nxt else np.zeros((self.h, self.w))

    def tile_char(self, x: int, y: int) -> str:
        s = int(self.struct[y, x])
        if s in C.ZONES:
            letter = C.LETTER_BY_ZONE[s]
            return letter if self.level[y, x] > 0 else letter.lower()
        if self.terrain[y, x] == C.WATER:
            return C.BRIDGE_CHAR if s == C.ROAD else (C.CHAR[s] if s in (C.DAM,) else C.WATER_CHAR)
        return C.CHAR[s]

    def tiles_string(self) -> str:
        return "".join(self.tile_char(x, y) for y in range(self.h) for x in range(self.w))

    def ascii_map(self, x0: int = 0, y0: int = 0, x1: int | None = None, y1: int | None = None) -> str:
        x1 = self.w if x1 is None else x1
        y1 = self.h if y1 is None else y1
        cols = range(x0, x1)
        tens = "    " + "".join(str((x // 10) % 10) for x in cols)
        ones = "    " + "".join(str(x % 10) for x in cols)
        rows = [tens, ones]
        for y in range(y0, y1):
            rows.append(f"{y:>3} " + "".join(self.tile_char(x, y) for x in cols))
        return "\n".join(rows)
