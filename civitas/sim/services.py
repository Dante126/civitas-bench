"""City services: one coverage framework shared by police, fire, health,
education, garbage and (later) transit.

Coverage is measured along the road network: a facility enters the network
through the road tiles it touches, the distance to every road tile is a BFS
in road steps, and a building is reached through any road within 2 tiles
(the same rule as zone access). Every facility has a capacity; if the demand
in its catchment (the tiles for which it is the nearest facility) is larger,
the coverage it gives is scaled down. BFS results are cached until the road
network or the set of facilities changes.
"""
from __future__ import annotations

import numpy as np

from . import catalog as C
from .grid import box_sum, road_bfs, shift, spread_to_tiles


class ServicesMixin:
    def _init_services(self) -> None:
        shape = (self.h, self.w)
        z = lambda: np.zeros(shape)
        self.police_cov, self.fire_cov, self.health_cov = z(), z(), z()
        self.school_cov, self.high_cov, self.univ_cov, self.library_cov = z(), z(), z(), z()
        self.garbage_cov, self.shelter_cov = z(), z()
        self.crime = z()
        self.fire_risk = z()
        self.garbage = z()                  # uncollected garbage stock per tile
        self.health = np.full(shape, 0.75)
        self.burning: list[int] = []        # tiles that burned this month (for the viewer)
        self.facility_state: dict[tuple[int, int], dict] = {}   # per-facility stock (landfill fill, plots...)
        self.inmates = 0.0
        self.death_backlog = 0.0
        self.service_info: dict[str, list] = {}
        self.downtown_gap: dict[str, np.ndarray] = {}   # level 4-5 tiles without tier-2 police / fire / health cover
        self._bfs_cache: dict = {}
        self._bfs_road_key = None

    # ---------------------------------------------------------- framework
    def facilities(self, struct: int) -> list[tuple[int, int]]:
        """Top-left tile of every building of this type (multi-tile buildings count once)."""
        m = self.struct == struct
        if not m.any():
            return []
        ys, xs = np.nonzero(m)
        a = self.anchor[ys, xs]
        keep = (a < 0) | (a == ys * self.w + xs)
        return list(zip(xs[keep].tolist(), ys[keep].tolist()))

    def _road_field(self, sources: tuple, radius: int):
        parts = [self.parts(x, y) for x, y in sources]
        key = (tuple(tuple(p) for p in parts), radius)  # tiles, not anchors: an upgrade can grow a building in place
        if self._bfs_road_key != self.road_key:
            self._bfs_cache.clear()
            self._bfs_road_key = self.road_key
        hit = self._bfs_cache.get(key)
        if hit is None:
            road = self.road_labels >= 0
            dist, lab = road_bfs(road, parts, max_d=radius + 1)
            td, tl = spread_to_tiles(dist, lab, C.ACCESS_RADIUS)
            for k, tiles in enumerate(parts):          # the facility's own tiles
                if any(lab[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max() >= 0 for x, y in tiles):
                    for x, y in tiles:
                        td[y, x], tl[y, x] = 0, k
            hit = (td, tl)
            self._bfs_cache[key] = hit
        return hit

    def coverage(self, struct: int, radius: int | None, capacity: float | None, demand: np.ndarray,
                 dept: str | None = None, full: float = C.COVERAGE_FULL,
                 skip: set | None = None, traffic: bool = False, cap_key: str = "capacity",
                 min_tier: int = 1) -> tuple[np.ndarray, list[dict]]:
        """Coverage grid (0..1) of all facilities of one type.

        Full strength up to `full` x radius (road steps), fading to 0 at the radius. Scaled by capacity / load of
        the nearest facility and by the department's funding level. radius / capacity None: each facility's own
        (tier) spec, capacity read from `cap_key`. min_tier: only facilities of at least this tier.
        """
        sources = [s for s in self.facilities(struct) if self.building_network(*s) >= 0 and not (skip and s in skip)
                   and self.facility_tier(*s) >= min_tier]
        if not sources:
            return np.zeros((self.h, self.w)), []
        eff = self.funding_eff(dept) if dept else 1.0
        specs = [self.facility_spec(x, y) for x, y in sources]
        radii = [int(radius if radius is not None else sp["radius"]) for sp in specs]
        caps = [(capacity if capacity is not None else sp[cap_key]) * eff * self.module_factor(x, y, "expansion")
                for (x, y), sp in zip(sources, specs)]
        # one pass per radius: nearest-source labelling across radii would leave holes where a tile is just outside
        # a small facility's radius but inside a farther big one's
        cov, load = None, [0.0] * len(sources)
        for r in sorted(set(radii)):
            idx = [k for k, rk in enumerate(radii) if rk == r]
            g, gl = self._cover_group([sources[k] for k in idx], r, np.array([caps[k] for k in idx]), demand, full)
            cov = g if cov is None else np.maximum(cov, g)
            for k, v in zip(idx, gl):
                load[k] = v
        cov = cov * min(1.0, eff)
        if traffic:                     # emergency vehicles stuck in traffic
            cov = cov * self.congestion_penalty
        info = []
        for k, ((x, y), sp) in enumerate(zip(sources, specs)):
            it = {"x": x, "y": y, "load": round(float(load[k]), 1), "capacity": round(float(caps[k]), 1)}
            if len(C.TIER_LINES.get(sp["line"], ())) > 1:
                it.update(name=sp["name"], tier=sp["tier"])
            info.append(it)
        return np.clip(cov, 0.0, 1.0), info

    def _cover_group(self, sources: list, radius: int, caps: np.ndarray, demand: np.ndarray,
                     full: float) -> tuple[np.ndarray, np.ndarray]:
        """Coverage (before funding / traffic) and load of facilities sharing one radius."""
        td, tl = self._road_field(tuple(sources), radius)
        reach = td <= radius
        load = np.bincount(tl[reach], weights=demand[reach], minlength=len(sources))
        ratio = np.where(load > caps, caps / np.maximum(load, 1e-9), 1.0)
        f = np.clip((radius - td) / max(1e-9, (1.0 - full) * radius), 0.0, 1.0) * reach if full < 1 else reach * 1.0
        return f * ratio[np.maximum(tl, 0)], load

    # ------------------------------------------------------------ compute
    def _people_grid(self) -> np.ndarray:
        s, lvl = self.struct, self.level
        g = self.residents_grid() + self.shop_jobs_grid() + np.array(C.IND_JOBS)[lvl] * (s == C.IND)
        if hasattr(C, "OFF_JOBS"):
            g = g + np.array(C.OFF_JOBS)[lvl] * (s == C.OFF)
        return g.astype(float)

    def _compute_services(self) -> None:
        s = self.struct
        people = self._people_grid()
        res = self.residents_grid()
        developed = np.isin(s, C.ZONES) & (self.level > 0)
        info = {}
        dev_hi = developed & (self.level >= C.DOWNTOWN_LEVEL)
        police, info["police_station"] = self.coverage(C.POLICE, None, None, people, "police", traffic=True)
        # tier-2-only passes: an HQ carries all demand in its radius on its own (tier-1 buildings don't share the
        # load there), so a downtown HQ must be big enough for the downtown
        police_hi, _ = self.coverage(C.POLICE, None, None, people, "police", traffic=True, min_tier=2)
        fire, info["fire_station"] = self.coverage(C.FIRE, None, None, developed.astype(float), "fire", traffic=True)
        fire_hi, _ = self.coverage(C.FIRE, None, None, developed.astype(float), "fire", traffic=True, min_tier=2)
        clinic, info["clinic"] = self.coverage(C.CLINIC, None, None, res, "health", traffic=True)
        hosp, info["hospital"] = self.coverage(C.HOSPITAL, None, None, res, "health", traffic=True)
        # downtown: tall buildings need tier-2 services (ladder trucks, HQ detectives, hospital care)
        down = lambda cov, hi: np.where(dev_hi, np.maximum(C.T1_DOWNTOWN * cov, hi), cov)
        self.police_cov, self.fire_cov = down(police, police_hi), down(fire, fire_hi)
        self.health_cov = down(np.maximum(clinic, hosp), hosp)
        gap = lambda hi: dev_hi & (hi < C.DOWNTOWN_COVERED)
        self.downtown_gap = {"police": gap(police_hi), "fire": gap(fire_hi), "health": gap(hosp)}
        self.stats["downtown_uncovered"] = {k: int(v.sum()) for k, v in self.downtown_gap.items()}
        # schools: demand is the school-age population living on each tile
        n = self.cohorts
        total = max(1.0, float(n.sum()))
        child_share, teen_share = n[0].sum() / total, n[1].sum() / total
        self.school_cov, info["elementary_school"] = self.coverage(
            C.SCHOOL, None, None, res * child_share, "education", cap_key="seats")
        self.high_cov, info["high_school"] = self.coverage(
            C.HIGHSCHOOL, None, None, res * teen_share, "education", cap_key="seats")
        self.univ_cov, info["university"] = self.coverage(
            C.UNIVERSITY, None, 1e9, res, "education")
        self.library_cov, info["library"] = self.coverage(C.LIBRARY, None, None, res, "education")
        # garbage collection: trucks from landfills (with room left) and incinerators
        self.garbage_cov = np.zeros((self.h, self.w))
        info["landfill"], info["incinerator"] = [], []
        full = self._full_landfills()
        for kind, struct in (("landfill", C.LANDFILL), ("incinerator", C.INCINERATOR)):
            cov, inf = self.coverage(struct, None, None, self._garbage_production(), "health", full=1.0,
                                     skip=full if kind == "landfill" else None, cap_key="collect")
            self.garbage_cov = np.maximum(self.garbage_cov, cov)
            info[kind] = inf
        for it in info["landfill"]:
            cap = self.facility_spec(it["x"], it["y"]).get("fill")     # recycling centres / waste complexes never fill
            if cap:
                it["fill_pct"] = round(100 * self._fstate(it["x"], it["y"]).get("fill", 0.0) / cap, 1)
        info["landfill"] += [{"x": x, "y": y, "fill_pct": 100.0, "full": True} for x, y in sorted(full)]
        self.shelter_cov, info["shelter"] = self.coverage(C.SHELTER, None, None, res, "fire")
        # jails, cemeteries and crematoria have capacity but no coverage area
        info["jail"] = [{"x": x, "y": y, "capacity": self.facility_spec(x, y)["capacity"]}
                        for x, y in self.facilities(C.JAIL)]
        info["cemetery"] = [{"x": x, "y": y,
                             "plots_left": int(self.facility_spec(x, y)["plots"] - self._fstate(x, y).get("used", 0))}
                            for x, y in self.facilities(C.CEMETERY)]
        info["crematorium"] = [{"x": x, "y": y, "rate": self.facility_spec(x, y)["rate"]}
                               for x, y in self.facilities(C.CREMATORIUM)]
        self.service_info = info

        rsum = res.sum()
        wavg = (lambda g: float((g * res).sum() / rsum)) if rsum > 0 else (lambda g: 0.0)
        covered = lambda g: float((res * (g > 0.05)).sum() / rsum) if rsum > 0 else 0.0
        seats = lambda struct: sum(self.facility_spec(x, y)["seats"] * self.module_factor(x, y, "expansion")
                                   for x, y in self.facilities(struct) if self.building_network(x, y) >= 0)
        self.stats.update({
            "police_coverage": wavg(self.police_cov), "fire_coverage": wavg(self.fire_cov),
            "health_coverage": wavg(self.health_cov), "school_coverage": covered(self.school_cov),
            "high_coverage": covered(self.high_cov), "library_coverage": covered(self.library_cov),
            "garbage_coverage": wavg(self.garbage_cov), "shelter_coverage": wavg(self.shelter_cov),
            "seats_elementary": seats(C.SCHOOL),
            "seats_high": seats(C.HIGHSCHOOL),
            "seats_university": seats(C.UNIVERSITY),
        })
        self._update_enrollment()

    def _full_landfills(self) -> set:
        out = set()
        for s in self.facilities(C.LANDFILL):
            cap = self.facility_spec(*s).get("fill")         # recycling centres / waste complexes never fill
            if cap and self._fstate(*s).get("fill", 0.0) >= cap:
                out.add(s)
        return out

    def _fstate(self, x: int, y: int) -> dict:
        return self.facility_state.setdefault((x, y), {"built": self.month})

    # -------------------------------------------------------------- crime
    def _update_crime(self) -> None:
        s, lvl = self.struct, self.level
        dev = np.isin(s, C.ZONES) & (lvl > 0)
        st = self.stats
        edu = self.education_index()
        homeless = st.get("homeless_share", 0.0)
        target = (C.CRIME_BASE + 0.03 * lvl + 0.55 * st.get("unemployment", 0.0) + 0.18 * (1.0 - edu)
                  + 0.15 * (1.0 - self.land_value) + 0.3 * homeless)
        police_eff = 0.8 * self.police_cov * (C.JAIL_OVERCROWD_EFF if st.get("jail_overcrowded") else 1.0)
        casinos = self.facilities(C.CASINO)
        if casinos:                                  # gambling brings crime to the neighbourhood
            spec = C.CASINO_SPEC
            near = np.zeros(s.shape)
            yy, xx = np.mgrid[0:self.h, 0:self.w]
            for cx, cy in casinos:
                d = np.hypot(xx + 0.5 - (cx + 1), yy + 0.5 - (cy + 1))      # from the centre of the 2x2 footprint
                near = np.maximum(near, np.clip(1.0 - d / spec["crime_radius"], 0, 1))
        else:
            near = None
        target = target * (1.0 - police_eff)
        if near is not None:                         # organised gambling crime: police only dent it
            target = target + spec["crime"] * near * (1.0 - 0.25 * police_eff)
        target *= np.where(self.policy_mask("neighborhood_watch"), C.POLICIES["neighborhood_watch"]["crime"], 1.0)
        casino_tiles = s == C.CASINO
        self.crime = np.where(dev | casino_tiles, 0.75 * self.crime + 0.25 * np.clip(target, 0, 1), 0.8 * self.crime)
        # arrests fill the jails; overcrowded jails make police less effective
        people = self._people_grid()
        arrests = float((self.crime * people * self.police_cov).sum()) * C.ARREST_RATE
        cap = sum(self.facility_spec(x, y)["capacity"] for x, y in self.facilities(C.JAIL)) + \
            sum(self.facility_spec(x, y)["holding"] for x, y in self.facilities(C.POLICE))
        self.inmates = self.inmates * (1.0 - C.RELEASE_RATE) + arrests
        st["jail_overcrowded"] = self.inmates > cap
        if self.inmates > cap * 1.3:
            self.inmates = cap * 1.3                 # the excess is released early
        st["inmates"], st["jail_capacity"] = self.inmates, cap

    # --------------------------------------------------------------- fire
    def _fire_events(self) -> None:
        s, lvl = self.struct, self.level
        dev = np.isin(s, C.ZONES) & (lvl > 0)
        base = np.zeros(s.shape)
        for z, r in C.FIRE_RISK.items():
            base[s == z] = r
        risk = base * (0.6 + 0.1 * lvl) * (1.0 - 0.3 * self.education_index())
        risk *= np.where(self.policy_mask("smoke_detectors"), C.POLICIES["smoke_detectors"]["fire"], 1.0)
        risk *= self.event_factor("fire_risk")
        self.fire_risk = np.where(dev, np.clip(risk, 0, 1), 0.0)
        p = C.FIRE_CHANCE * self.fire_risk * (1.0 - 0.85 * self.fire_cov)
        ign = np.nonzero((self.rng.random(s.shape) < p) & dev)
        responses = int(sum(self.facility_spec(x, y)["responses"] for x, y in self.facilities(C.FIRE))
                        * self.funding_eff("fire"))
        fires = sorted(zip(ign[0].tolist(), ign[1].tolist()), key=lambda t: -self.fire_cov[t])
        lost, contained, burned = 0, 0, []
        for n, (y, x) in enumerate(fires):
            covered = self.fire_cov[y, x] >= 0.3 and n < responses
            if covered:
                contained += 1
                if self.rng.random() < 0.5 and self.level[y, x] > 0:
                    self.level[y, x] -= 1
                burned.append(y * self.w + x)
                continue
            # uncontrolled: spreads to neighbouring buildings
            stack, seen = [(x, y)], {(x, y)}
            while stack and len(seen) <= C.FIRE_MAX_SPREAD:
                cx, cy = stack.pop()
                if self.level[cy, cx] > 0:
                    self.level[cy, cx] = 0
                    lost += 1
                    burned.append(cy * self.w + cx)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if (self.in_bounds(nx, ny) and (nx, ny) not in seen and dev[ny, nx]
                            and self.rng.random() < C.FIRE_SPREAD * (1.0 - self.fire_cov[ny, nx])):
                        seen.add((nx, ny))
                        stack.append((nx, ny))
        self.burning = burned
        if lost:
            self.event(f"Fire destroyed {lost} building(s)" + (f"; {contained} other fire(s) were put out" if contained else ""))
        elif contained:
            self.event(f"{contained} fire(s) were put out by the fire department")
        self.stats.update({"fires": len(fires), "fire_tiles_lost": lost, "fires_contained": contained})

    # ------------------------------------------------ garbage & health
    def _garbage_production(self) -> np.ndarray:
        s, lvl = self.struct, self.level
        g = np.zeros(s.shape)
        for z, rate in C.GARBAGE_PER_LEVEL.items():
            m = s == z
            g[m] = rate * lvl[m]
        if self.policies_any("recycling"):
            g *= np.where(self.policy_mask("recycling"), C.POLICIES["recycling"]["garbage"], 1.0)
        return g

    def _garbage_and_health(self) -> None:
        prod = self._garbage_production()
        stock = self.garbage + prod
        collected = stock * self.garbage_cov
        # landfills fill up; incinerators burn what they collect
        full = self._full_landfills()
        landfills = [s for s in self.facilities(C.LANDFILL) if self.building_network(*s) >= 0 and s not in full]
        if landfills and collected.sum() > 0:
            radius = max(self.facility_spec(*s)["radius"] for s in landfills)
            td, tl = self._road_field(tuple(landfills), radius)
            burn_share = 0.0
            if self.facilities(C.INCINERATOR):
                inc_cov, _ = self.coverage(C.INCINERATOR, None, None, prod, "health", full=1.0, cap_key="collect")
                burn_share = float((inc_cov * stock).sum() / max(1e-9, (self.garbage_cov * stock).sum()))
            per = np.bincount(tl[td <= radius], weights=collected[td <= radius], minlength=len(landfills))
            for k, (x, y) in enumerate(landfills):
                cap = self.facility_spec(x, y).get("fill")
                if not cap:
                    continue                                  # recycles what it collects
                st = self._fstate(x, y)
                before = st.get("fill", 0.0)
                st["fill"] = min(cap, before + float(per[k]) * (1.0 - min(1.0, burn_share)))
                if before < cap * 0.9 <= st["fill"]:
                    self.event(f"Landfill at ({x},{y}) is 90% full")
                if before < cap <= st["fill"]:
                    self.event(f"Landfill at ({x},{y}) is FULL and stopped collecting garbage")
        dev = np.isin(self.struct, C.ZONES) & (self.level > 0)
        self.garbage = np.where(dev, stock - collected, (stock - collected) * 0.9)
        g_norm = np.clip(self.garbage / C.GARBAGE_NORM, 0, 1)
        water_pen = np.where(self.watered, 1.0 - self.water_quality, 0.5)
        backlog_pen = min(0.3, self.death_backlog / max(50.0, self.stats.get("population", 0) / 100))
        unsewered = (self.level > 0) & ~self.sewered
        h = (C.HEALTH_BASE - 0.35 * self.pollution - 0.25 * water_pen - 0.25 * g_norm
             + 0.35 * self.health_cov - backlog_pen - C.UNSEWERED_HEALTH * unsewered - 0.15 * self.soil_level())
        self.health = np.clip(h, 0, 1)
        res = self.residents_grid()
        rsum = res.sum()
        self.stats["health"] = float((self.health * res).sum() / rsum) if rsum > 0 else 0.75
        self.stats["garbage_backlog"] = float(self.garbage[dev].sum()) if dev.any() else 0.0
        self.stats["garbage_produced"] = float(prod.sum())
        self.stats["garbage_collected"] = float(collected.sum())
        self.stats["garbage_homes_pct"] = float((res * (g_norm > 0.3)).sum() / rsum) if rsum > 0 else 0.0

    def _deathcare(self) -> None:
        deaths = self.deaths_month
        cap = 0.0
        cems = [s for s in self.facilities(C.CEMETERY) if self.building_network(*s) >= 0]
        eff = self.funding_eff("health")
        for x, y in cems:
            spec = self.facility_spec(x, y)
            cap += max(0.0, min(spec["rate"] * eff, spec["plots"] - self._fstate(x, y).get("used", 0.0)))
        # crematoria add capacity and are used before cemetery plots
        crems = [s for s in self.facilities(C.CREMATORIUM) if self.building_network(*s) >= 0]
        crem_cap = sum(self.facility_spec(x, y)["rate"] for x, y in crems) * eff
        morgue = len(self.facilities(C.HOSPITAL)) * C.HOSPITAL_MORGUE
        cap += crem_cap + morgue
        want = self.death_backlog + deaths
        handled = min(want, cap)
        left = handled - morgue - crem_cap                   # the rest goes to cemetery plots, in order
        for x, y in cems:
            if left <= 0:
                break
            spec = self.facility_spec(x, y)
            st = self._fstate(x, y)
            room = spec["plots"] - st.get("used", 0.0)
            take = min(room, left, spec["rate"] * eff)
            if take > 0:
                before = st.get("used", 0.0)
                st["used"] = before + take
                left -= take
                if before < spec["plots"] <= st["used"] + 1e-9:
                    self.event(f"Cemetery at ({x},{y}) is full")
        self.death_backlog = want - handled
        if self.death_backlog > 20 and self.month % 3 == 0:
            self.event(f"{self.death_backlog:.0f} deceased are waiting for deathcare (build cemeteries or a crematorium)")
        self.stats.update({"deaths": deaths, "death_backlog": self.death_backlog, "deathcare_capacity": cap})

    # ------------------------------------------------------ site hints
    def site_candidates(self, need: np.ndarray, radius: int = 6, k: int = 5, spacing: int = 4,
                        riverbank: bool = False, exclude: np.ndarray | None = None,
                        min_need: float = 1e-6, size: int = 1) -> list[dict]:
        """Empty tiles touching the highway-connected road network, ranked by the
        sum of `need` within `radius` (used for observations / site hints)."""
        s = self.struct
        empty = (s == C.NONE) & (self.terrain == C.LAND)
        road = self.road_labels >= 0
        near = np.zeros_like(road)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx or dy:
                    near |= shift(road, dy, dx, False)
        bank = self.dist_water == 1
        if size > 1:                               # top-left tiles whose whole size x size block is free
            fe, fn, fb = empty.copy(), near.copy(), bank.copy()
            for j in range(size):                  # (bank / road may touch any tile, as the build action allows)
                for i in range(size):
                    if i or j:
                        fe &= shift(empty, -j, -i, False)
                        fn |= shift(near, -j, -i, False)
                        fb |= shift(bank, -j, -i, False)
            empty, near, bank = fe, fn, fb
        ok = empty & near & (self.access | (size > 1))
        if riverbank:
            ok &= bank
        if exclude is not None:
            ok &= ~exclude
        score = box_sum(need, radius)
        ys, xs = np.nonzero(ok)
        order = np.argsort(-score[ys, xs], kind="stable")
        out = []
        for i in order.tolist():
            x, y = int(xs[i]), int(ys[i])
            if score[y, x] < min_need:
                break
            if any(abs(x - o["x"]) + abs(y - o["y"]) < spacing for o in out):
                continue
            net = self.adjacent_network(x, y) if size == 1 else self._block_network(x, y, size)
            if net < 0 or not self.networks[net].outside_connected:
                continue
            out.append({"x": x, "y": y, "need": round(float(score[y, x]), 1)})
            if len(out) >= k:
                break
        return out

    def _block_network(self, x: int, y: int, size: int) -> int:
        """Best road network touching any tile of the size x size block at (x, y) (as building_network)."""
        best = -1
        for j in range(size):
            for i in range(size):
                if self.in_bounds(x + i, y + j):
                    lab = self.adjacent_network(x + i, y + j)
                    if lab >= 0 and (best < 0 or self._net_priority(lab) < self._net_priority(best)):
                        best = lab
        return best

    def hotspots(self, grid: np.ndarray, k: int = 4, block: int = 6) -> list[dict]:
        """Top blocks of a per-tile 'problem' grid, with their centre tile."""
        h, w = grid.shape
        out = []
        for by in range(0, h, block):
            for bx in range(0, w, block):
                v = float(grid[by:by + block, bx:bx + block].sum())
                if v > 0:
                    out.append((v, bx + block // 2, by + block // 2))
        out.sort(reverse=True)
        return [{"x": min(w - 1, x), "y": min(h - 1, y), "amount": round(v, 1)} for v, x, y in out[:k]]
