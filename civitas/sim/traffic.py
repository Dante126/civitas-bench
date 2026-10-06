"""Traffic: an aggregate commuter model on the road network (no individual cars).

Every month workers travel from their home's road tile to job centres
("hubs"). Hubs are the busiest job areas plus the outside connections;
workers pick hubs with a gravity model (jobs x exp(-distance / L)). Each
flow follows the shortest-path tree towards its hub, so the load on a road
tile is the number of commuters passing through it. Load / capacity gives
congestion; congestion slows commutes (BPR-style), adds noise and exhaust,
and slows police, fire and ambulances. Bus lines carry a share of trips
between their stops and take them off the roads.

Shortest-path trees are cached per (hub, road network), so the model costs
a few numpy passes per month.
"""
from __future__ import annotations

import numpy as np

from . import catalog as C
from .grid import INF, shift

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def road_tree(road: np.ndarray, start: tuple[int, int]):
    """BFS from one road tile. Returns (dist, parent flat index, levels)."""
    h, w = road.shape
    dist = np.full((h, w), INF, np.int32)
    sx, sy = start
    dist[sy, sx] = 0
    frontier = np.zeros_like(road)
    frontier[sy, sx] = True
    step = 0
    while frontier.any():
        step += 1
        new = np.zeros_like(frontier)
        for dy, dx in _DIRS:
            m = shift(frontier, dy, dx, False) & road & (dist == INF) & ~new
            dist[m] = step
            new |= m
        frontier = new
    flat = np.arange(h * w).reshape(h, w)
    parent = np.full((h, w), -1, np.int64)
    for dy, dx in _DIRS:          # parent = a neighbour one step closer to the hub
        nd = shift(dist, -dy, -dx, INF)
        nidx = shift(flat, -dy, -dx, -1)
        m = (parent == -1) & (dist != INF) & (dist > 0) & (nd == dist - 1)
        parent[m] = nidx[m]
    reach = dist.ravel() != INF
    idx = np.nonzero(reach)[0]
    d = dist.ravel()[idx]
    order = np.argsort(d, kind="stable")
    idx, d = idx[order], d[order]
    cuts = np.nonzero(np.diff(d))[0] + 1
    levels = np.split(idx, cuts)
    return dist, parent.ravel(), levels


class TrafficMixin:
    def _init_traffic(self) -> None:
        shape = (self.h, self.w)
        self.road_class = np.zeros(shape, np.uint8)
        self.traffic_load = np.zeros(shape)
        self.congestion = np.zeros(shape)            # load / capacity on road tiles
        self.traffic_noise = np.zeros(shape)
        self.congestion_penalty = np.ones(shape)     # slows emergency services
        self.commute_minutes = np.zeros(shape)       # per home tile
        self.transit_cov = np.zeros(shape, bool)
        self.bus_lines: list[dict] = []
        self.metro_lines: list[dict] = []
        self.metro_cov = np.zeros(shape, bool)
        self.fare = C.DEFAULT_FARE
        self.hubs: list[dict] = []
        self._tree_cache: dict = {}
        self._tree_key = None
        self._anchor_key = None

    # ------------------------------------------------------------ helpers
    def road_capacity(self) -> np.ndarray:
        cap = np.zeros((self.h, self.w))
        for cls, c in C.ROAD_CAPACITY.items():
            cap[self.road_class == cls] = c
        return cap * (self.funding_eff("transport") ** 0.5) * self.event_factor("road_capacity")

    def _anchors(self) -> np.ndarray:
        """For every tile within 2 of a road: flat index of the nearest road tile (else -1)."""
        if self._anchor_key == self.road_key:
            return self._anchor
        road = (self.road_labels >= 0) & (self.road_class < 3)
        flat = np.where(road, np.arange(self.h * self.w).reshape(self.h, self.w), -1)
        anchor = flat.copy()
        for r in (1, 2):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dy), abs(dx)) != r:
                        continue
                    cand = shift(flat, dy, dx, -1)
                    m = (anchor < 0) & (cand >= 0)
                    anchor[m] = cand[m]
        self._anchor, self._anchor_key = anchor, self.road_key
        return anchor

    def _tree(self, node: tuple[int, int]):
        if self._tree_key != self.road_key:
            self._tree_cache.clear()
            self._tree_key = self.road_key
        t = self._tree_cache.get(node)
        if t is None:
            t = road_tree(self.road_labels >= 0, node)
            self._tree_cache[node] = t
        return t

    def outside_nodes(self) -> list[tuple[int, int]]:
        """Road tiles where traffic leaves the city: the highway and train stations."""
        nodes = [self.highway] + [(int(x), int(y)) for y, x in zip(*np.nonzero(self.struct == C.HIGHWAY))
                                  if (int(x), int(y)) != tuple(self.highway)]
        for x, y in self.facilities(C.TRAIN) + self.facilities(C.HARBOUR) + self.facilities(C.AIRPORT):
            near = [(px + dx, py + dy) for px, py in self.parts(x, y) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if self.in_bounds(px + dx, py + dy) and self.road_labels[py + dy, px + dx] >= 0]
            if near:
                nodes.append(near[0])
        return nodes

    def _pick_hubs(self, jobs: np.ndarray, anchor: np.ndarray) -> list[dict]:
        """Busiest job areas (one road tile per 12x12 block) + outside connections."""
        B = C.HUB_BLOCK
        dmass = np.bincount(anchor[anchor >= 0].ravel(), weights=jobs[anchor >= 0].ravel(), minlength=self.h * self.w)
        dm = dmass.reshape(self.h, self.w)
        blocks = []
        for by in range(0, self.h, B):
            for bx in range(0, self.w, B):
                sub = dm[by:by + B, bx:bx + B]
                tot = float(sub.sum())
                if tot <= 0:
                    continue
                y, x = np.unravel_index(int(np.argmax(sub)), sub.shape)
                blocks.append({"x": bx + int(x), "y": by + int(y), "jobs": tot, "kind": "jobs"})
        blocks.sort(key=lambda b: -b["jobs"])
        hubs = blocks[:C.MAX_HUBS]
        for b in blocks[C.MAX_HUBS:]:         # small job areas join the nearest hub
            near = min(hubs, key=lambda h: abs(h["x"] - b["x"]) + abs(h["y"] - b["y"]))
            near["jobs"] += b["jobs"]
        total_jobs = sum(h["jobs"] for h in hubs)
        outs = self.outside_nodes()
        for k, (x, y) in enumerate(outs):
            if self.road_labels[y, x] >= 0:
                share = C.REGIONAL_COMMUTE * (1.0 if k == 0 else C.TRAIN_REGIONAL_BOOST)
                hubs.append({"x": x, "y": y, "jobs": max(10.0, total_jobs * share), "kind": "outside"})
        return hubs

    # -------------------------------------------------------------- model
    # The pieces of _compute_traffic that don't write the world: the fleet's transit_estimate (fleet/tools.py) replays
    # the month's trip split with them for a bus line not yet built, so both compute it the same way.
    def _trip_origins(self, res: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        """Commuting workers per road node (flat index): residents x the worker share, on their anchor tile."""
        pop = max(1.0, float(res.sum()))
        wshare = self.stats.get("workers", 0.0) / pop
        ok = anchor >= 0
        return np.bincount(anchor[ok].ravel(), weights=(res * wshare)[ok].ravel(), minlength=self.h * self.w)

    def _gravity(self, hubs: list[dict]):
        """(trees, D, reach, P): each hub's shortest-path tree, distances (hubs x tiles), reachability and the share of
        each origin's trips that go to each hub (jobs x exp(-distance / COMMUTE_DECAY), normalised)."""
        trees = [self._tree((hb["x"], hb["y"])) for hb in hubs]
        D = np.stack([t[0].ravel() for t in trees])
        reach = D < INF
        A = np.array([hb["jobs"] for hb in hubs])[:, None] * np.exp(-np.where(reach, D, 0) / C.COMMUTE_DECAY) * reach
        tot = A.sum(0)
        P = np.divide(A, tot, out=np.zeros_like(A), where=tot > 0)
        return trees, D, reach, P

    def _split_trips(self, flows: np.ndarray, share: np.ndarray, m_share: np.ndarray, n_lines: int):
        """(metro, metro riders, bus flows, transit, bus riders): metro takes its share of the trips up to its lines'
        capacity, buses their share (at most 1) of the rest up to n_lines' seats; the rest drive."""
        metro = flows * m_share
        m_riders = float(metro.sum())
        m_cap = C.METRO_CAPACITY * self.funding_eff("transport") * len(self.metro_lines)
        if m_riders > m_cap > 0:
            metro *= m_cap / m_riders
            m_riders = m_cap
        flows_bus = flows - metro
        transit = flows_bus * share
        riders = float(transit.sum())
        cap_total = C.LINE_CAPACITY * self.funding_eff("transport") * n_lines
        if riders > cap_total > 0:                                # full buses: the rest drive
            transit *= cap_total / riders
            riders = cap_total
        return metro, m_riders, flows_bus, transit, riders

    def _compute_traffic(self) -> None:
        s = self.struct
        road = self.road_labels >= 0
        n_tiles = self.h * self.w
        cap = self.road_capacity()
        if not road.any():
            return
        anchor = self._anchors()
        lvl = self.level
        res = self.residents_grid()
        jobs = self._people_grid() - res
        ok = anchor >= 0
        omass = self._trip_origins(res, anchor)
        self.hubs = hubs = self._pick_hubs(jobs, anchor)
        load = np.zeros(n_tiles)
        riders = 0.0
        if omass.sum() <= 0 or not hubs:
            self._set_traffic(load.reshape(self.h, self.w), cap, np.zeros(n_tiles), 0.0, 0.0)
            return
        trees, D, reach, P = self._gravity(hubs)                 # D, P: hubs x tiles
        flows = omass[None, :] * P
        share, per_line = self._transit_share(hubs)               # hubs x tiles
        m_share = self._metro_share(hubs)
        share = np.minimum(share, 1.0)
        metro, m_riders, flows_bus, transit, riders = self._split_trips(flows, share, m_share, len(self.bus_lines))
        for line, ls in zip(self.bus_lines, per_line):
            line["riders"] = round(float((flows * ls).sum()) * (riders / max(1e-9, float((flows * share).sum()))), 1)
            line["capacity"] = round(C.LINE_CAPACITY * self.funding_eff("transport"))
        car = flows_bus - transit
        # freight: trucks from industry to the outside connections
        freight = float(self.stats.get("freight", 0.0))
        out_k = [k for k, hb in enumerate(hubs) if hb["kind"] == "outside"]
        if freight > 0 and out_k:
            ind = (s == C.IND) & (lvl > 0)
            fm = np.bincount(anchor[ok & ind].ravel(), weights=(np.array(C.IND_JOBS)[lvl] * ind)[ok & ind].ravel(),
                             minlength=n_tiles)
            if fm.sum() > 0:
                fm *= freight * C.FREIGHT_TRIPS * C.TRUCK_WEIGHT / fm.sum()
                Ao = np.stack([reach[k] * np.exp(-np.where(reach[k], D[k], 0) / C.COMMUTE_DECAY) for k in out_k])
                Po = np.divide(Ao, Ao.sum(0), out=np.zeros_like(Ao), where=Ao.sum(0) > 0)
                for j, k in enumerate(out_k):
                    car[k] += fm * Po[j]
        for k, (dist, parent, levels) in enumerate(trees):
            acc = car[k].copy()
            for lv in reversed(levels[1:]):
                np.add.at(acc, parent[lv], acc[lv])
            load += acc
        load2 = load.reshape(self.h, self.w)
        ratio = np.divide(load2, cap, out=np.zeros_like(load2), where=cap > 0)
        tile_time = 1.0 + C.BPR_ALPHA * np.clip(ratio, 0, 3) ** C.BPR_BETA
        tt = tile_time.ravel()
        T = np.zeros(n_tiles)
        T_metro = np.zeros(n_tiles)
        for k, (dist, parent, levels) in enumerate(trees):
            time = np.zeros(n_tiles)
            for lv in levels[1:]:
                time[lv] = time[parent[lv]] + tt[lv]
            T += P[k] * np.where(reach[k], time, 0.0)
            T_metro += P[k] * np.where(reach[k], D[k], 0.0)
        # metro riders: fast trains plus walking and waiting, no traffic jams
        car_min = C.COMMUTE_BASE_MIN + C.MIN_PER_TILE * T
        metro_min = C.COMMUTE_BASE_MIN + 3.0 + C.METRO_MIN_PER_TILE * T_metro
        sm = np.divide(metro.sum(0), flows.sum(0), out=np.zeros(n_tiles), where=flows.sum(0) > 0)
        minutes = (1.0 - sm) * car_min + sm * metro_min
        self.stats["metro_riders"] = m_riders
        self._set_traffic(load2, cap, minutes, riders, float(omass.sum()))

    def _set_traffic(self, load, cap, minutes_by_node, riders, trips) -> None:
        s = self.struct
        anchor = self._anchors()
        self.traffic_load = load
        self.congestion = np.divide(load, cap, out=np.zeros_like(load), where=cap > 0)
        road = self.road_labels >= 0
        noise_f = np.zeros_like(load)
        for cls, f in C.ROAD_NOISE.items():
            noise_f[self.road_class == cls] = f
        self.traffic_noise = np.where(road, np.clip(load / 150.0, 0, 4) * noise_f, 0.0)
        # emergency vehicles are slowed where the local roads are jammed
        cong_at = np.where(anchor >= 0, self.congestion.ravel()[np.maximum(anchor, 0)], 0.0)
        self.congestion_penalty = 1.0 - C.CONGESTION_SERVICE_PENALTY * np.clip(cong_at - 0.8, 0, 1)
        res = self.homes_mask()
        mins = np.where(anchor >= 0, np.asarray(minutes_by_node).ravel()[np.maximum(anchor, 0)] if np.size(minutes_by_node) else 0.0, 0.0)
        self.commute_minutes = np.where(res, mins, 0.0)
        self.commute_need = np.where(res, np.clip(1.0 - (mins - C.COMMUTE_OK_MIN) / C.COMMUTE_BAD_SPAN, 0, 1), 1.0)
        residents = self.residents_grid()
        rs = residents.sum()
        n_road = int(road.sum())
        self.stats.update({
            "commute_minutes": float((mins * residents).sum() / rs) if rs > 0 else 0.0,
            "congestion_index": float((self.congestion > 1.0).sum() / n_road) if n_road else 0.0,
            "congestion_mean": float(self.congestion[road].mean()) if n_road else 0.0,
            "transit_riders": riders + self.stats.get("metro_riders", 0.0), "bus_riders": riders, "trips": trips,
            "transit_share": (riders + self.stats.get("metro_riders", 0.0)) / trips if trips > 0 else 0.0,
        })

    # ------------------------------------------------------------ transit
    def _refresh_transit(self) -> None:
        cov = np.zeros((self.h, self.w), bool)
        for line in self.bus_lines:
            m = self._cover(line["stops"], C.STOP_RADIUS)
            line["_cov"] = m
            cov |= m
        self.transit_cov = cov

    def _cover(self, points, r: float) -> np.ndarray:
        """Tiles within walking distance r of any of `points` (bus stops, metro stations)."""
        m = np.zeros((self.h, self.w), bool)
        for x, y in points:
            m |= self._disk(x, y, r)
        return m

    def _bus_base(self) -> float:
        """Share of covered trips that ride the bus at the fare (free transit: no fare, more riders)."""
        free = self.policy_on("free_transit")
        fare_f = 1.0 if free else float(np.clip(1.25 - 0.25 * self.fare, 0.3, 1.2))
        boost = C.POLICIES["free_transit"]["ridership"] if free else 1.0
        return min(0.8, C.TRANSIT_BASE * fare_f * boost)

    @staticmethod
    def _hub_share(hubs: list[dict], cov: np.ndarray, base: float) -> np.ndarray:
        """hubs x tiles: `base` where a line (or metro line) covering `cov` reaches both the hub and the origin."""
        served = np.array([bool(cov[hb["y"], hb["x"]]) for hb in hubs], float)
        return np.outer(served, cov.ravel().astype(float)) * base

    def _transit_share(self, hubs: list[dict]):
        """Share of trips (origin road tile -> hub) that take the bus, overall and per line."""
        share = np.zeros((len(hubs), self.h * self.w))
        per_line = []
        if not self.bus_lines:
            return share, per_line
        self._refresh_transit()
        base = self._bus_base()
        for line in self.bus_lines:
            ls = self._hub_share(hubs, line["_cov"], base)
            per_line.append(ls)
            share = np.maximum(share, ls)
        return share, per_line

    def _metro_cover(self, hubs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """(share, cover): the share of trips that take the metro (hubs x tiles) and the tiles its stations reach."""
        share = np.zeros((len(hubs), self.h * self.w))
        cov_all = np.zeros((self.h, self.w), bool)
        for line in self.metro_lines:
            m = self._cover(line["stations"], C.METRO_RADIUS)
            cov_all |= m
            share = np.maximum(share, self._hub_share(hubs, m, C.METRO_BASE))
        return share, cov_all

    def _metro_share(self, hubs: list[dict]) -> np.ndarray:
        share, self.metro_cov = self._metro_cover(hubs)
        return share

    def _disk(self, x: int, y: int, r: float) -> np.ndarray:
        """Tiles within walking distance r of (x, y)."""
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        return (xx - x) ** 2 + (yy - y) ** 2 <= (r + 0.5) ** 2

    def transit_upkeep(self) -> float:
        return (sum(C.BUS_LINE_UPKEEP + C.BUS_STOP_UPKEEP * len(l["stops"]) for l in self.bus_lines)
                + C.METRO_LINE_UPKEEP * len(self.metro_lines))

    def road_upgrade_upkeep(self) -> float:
        """Extra upkeep of avenues / highway roads beyond a plain street."""
        extra = 0.0
        water = self.terrain == C.WATER
        for cls, up in C.ROAD_UPKEEP.items():
            if cls <= 1:
                continue
            m = (self.road_class == cls) & (self.struct == C.ROAD)
            extra += float((m & ~water).sum()) * (up - C.ROAD_UPKEEP[1]) + float((m & water).sum()) * 2 * (up - C.ROAD_UPKEEP[1])
        return extra
