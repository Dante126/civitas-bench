"""Seeded events and disasters.

The schedule comes from the map seed (its own RNG), so every agent line-up
faces the same floods, heatwaves, recessions and offers on a given seed.
Floods are forecast a few months ahead; levees protect the bank next to them
and shelters reduce casualties.
"""
from __future__ import annotations

import numpy as np

from . import catalog as C
from .grid import dilate

KINDS = ("flood", "heatwave", "recession", "factory_offer")


def make_schedule(seed: int, months: int, h: int) -> list[dict]:
    rng = np.random.default_rng([seed, 99])
    ev = []

    def month(lo, hi):
        return int(rng.integers(lo, hi)) if hi > lo else None

    m = month(30, months - 18)
    if m is not None:
        yc = int(rng.integers(8, h - 8))
        ev.append({"kind": "flood", "month": m, "duration": 1, "y1": max(0, yc - 7), "y2": min(h - 1, yc + 7)})
    m = month(18, months - 6)
    if m is not None:
        ev.append({"kind": "heatwave", "month": m, "duration": 3})
    m = month(42, months - 12)
    if m is not None:
        ev.append({"kind": "recession", "month": m, "duration": 8})
    m = month(20, months - 30)
    if m is not None:
        ev.append({"kind": "factory_offer", "month": m, "duration": 3, "jobs": 400, "bonus": 15000})
    for k, e in enumerate(sorted(ev, key=lambda e: e["month"])):
        e["id"] = k + 1
    return sorted(ev, key=lambda e: e["month"])


class EventsMixin:
    def _init_events(self, events, months: int) -> None:
        if events == "auto":
            self.schedule = make_schedule(self.seed, months, self.h)
        elif isinstance(events, list):
            self.schedule = [dict(e, id=e.get("id", k + 1)) for k, e in enumerate(events)]
        else:
            self.schedule = []
        self.offers: list[dict] = []           # open offers the mayor can accept
        self.complex_mask = np.zeros((self.h, self.w), bool)    # tiles of an accepted mega-factory
        self.flooded: list[int] = []

    # ----------------------------------------------------------- queries
    def active_events(self) -> list[dict]:
        return [e for e in self.schedule if e["month"] <= self.month < e["month"] + e["duration"]]

    def forecast(self, ahead: int = 3) -> list[str]:
        out = []
        for e in self.schedule:
            dt = e["month"] - self.month
            if 0 < dt <= ahead:
                if e["kind"] == "flood":
                    out.append(f"River flood expected in {dt} month(s) along rows {e['y1']}-{e['y2']}: "
                               "riverside buildings without a levee will be damaged")
                elif e["kind"] == "heatwave":
                    out.append(f"Heatwave expected in {dt} month(s): +20% power, +30% water demand, more fires")
                elif e["kind"] == "recession":
                    out.append(f"Economists expect a recession in {dt} month(s): weaker industry and exports")
        for e in self.active_events():
            if e["kind"] in ("heatwave", "recession"):
                left = e["month"] + e["duration"] - self.month
                out.append(f"{e['kind'].title()} under way ({left} month(s) left)")
        return out

    def event_factor(self, key: str) -> float:
        f = self.season_factor(key)
        for e in self.active_events():
            f *= C.EVENT_EFFECTS.get(e["kind"], {}).get(key, 1.0)
        return f

    # ----------------------------------------------------------- monthly
    def _run_events(self) -> None:
        for e in self.schedule:
            if e["month"] != self.month:
                continue
            if e["kind"] == "flood":
                self._flood(e)
            elif e["kind"] == "heatwave":
                self.event("HEATWAVE: power and water demand jump, fire risk rises")
            elif e["kind"] == "recession":
                self.event("RECESSION: industrial demand and export prices fall")
            elif e["kind"] == "factory_offer":
                offer = {"id": e["id"], "jobs": e["jobs"], "bonus": e["bonus"], "expires": self.month + e["duration"]}
                self.offers.append(offer)
                self.event(f"OFFER #{offer['id']}: a large factory wants to move to the city ({offer['jobs']} jobs, "
                           f"${offer['bonus']:,} signing bonus, heavy pollution). The mayor can accept or decline "
                           f"until month {offer['expires']}.")
        for o in list(self.offers):
            if self.month >= o["expires"]:
                self.offers.remove(o)
                self.event(f"Factory offer #{o['id']} expired")

    def flood_zone(self, e: dict) -> np.ndarray:
        band = np.zeros((self.h, self.w), bool)
        band[e["y1"]:e["y2"] + 1] = True
        zone = band & (self.terrain == C.LAND) & (self.dist_water <= C.FLOOD_REACH)
        levees = self.struct == C.LEVEE
        if levees.any():
            zone &= ~dilate(levees, C.LEVEE_PROTECT)
        return zone

    def _flood(self, e: dict) -> None:
        zone = self.flood_zone(e)
        dev = zone & np.isin(self.struct, C.ZONES) & (self.level > 0)
        res = self.residents_grid()
        displaced = float(res[dev].sum())
        casualties = displaced * C.FLOOD_CASUALTY_RATE * float(np.mean(1.0 - self.shelter_cov[dev])) if dev.any() else 0.0
        self.level[dev] = np.maximum(0, self.level[dev].astype(int) - 2).astype(np.uint8)
        self.flooded = [int(v) for v in np.flatnonzero(zone)]
        offline = 0
        for y, x in zip(*np.nonzero(zone & np.isin(self.struct, list(C.UTILITY_TYPES)))):
            self._fstate(int(x), int(y))["offline_until"] = self.month + C.FLOOD_OFFLINE_MONTHS
            offline += 1
        self.deaths_month += casualties
        self.death_backlog += casualties
        self.disaster_morale = C.DISASTER_MORALE if displaced > 0 else 0.0
        self.stats.update({"flood_damaged": int(dev.sum()), "flood_casualties": casualties})
        protected = int((self.struct == C.LEVEE).sum())
        self.event(f"FLOOD along rows {e['y1']}-{e['y2']}: {int(dev.sum())} buildings damaged, {displaced:.0f} residents "
                   f"displaced, {casualties:.0f} casualties, {offline} riverside facilities offline for "
                   f"{C.FLOOD_OFFLINE_MONTHS} months" + (f" ({protected} levee tiles held)" if protected else ""))

    def is_offline(self, x: int, y: int) -> bool:
        return self.facility_state.get((x, y), {}).get("offline_until", -1) > self.month

    # ------------------------------------------------------------ offers
    def accept_offer(self, offer_id: int) -> str:
        offer = next((o for o in self.offers if o["id"] == offer_id), None)
        if offer is None:
            raise ValueError(f"no open offer #{offer_id}")
        site = self._factory_site()
        if site is None:
            raise ValueError("no 3x3 block of empty land next to a connected road for the factory; "
                             "ask Planning to open up land")
        x0, y0 = site
        self.struct[y0:y0 + 3, x0:x0 + 3] = C.IND
        self.level[y0:y0 + 3, x0:x0 + 3] = 3
        self.density[y0:y0 + 3, x0:x0 + 3] = 1
        self.ind_type[y0:y0 + 3, x0:x0 + 3] = 0
        self.complex_mask[y0:y0 + 3, x0:x0 + 3] = True
        self.treasury += offer["bonus"]
        self.offers.remove(offer)
        return (f"accepted offer #{offer_id}: the factory opened at ({x0},{y0})-({x0 + 2},{y0 + 2}); "
                f"${offer['bonus']:,} added to the treasury")

    def decline_offer(self, offer_id: int) -> str:
        offer = next((o for o in self.offers if o["id"] == offer_id), None)
        if offer is None:
            raise ValueError(f"no open offer #{offer_id}")
        self.offers.remove(offer)
        return f"declined offer #{offer_id}"

    def _factory_site(self):
        s = self.struct
        empty = (s == C.NONE) & (self.terrain == C.LAND)
        homes = np.isin(s, C.HOMES).astype(float)
        best = None
        for y in range(0, self.h - 2):
            for x in range(0, self.w - 2):
                if not empty[y:y + 3, x:x + 3].all() or not self.access[y:y + 3, x:x + 3].any():
                    continue
                net = max((self.adjacent_network(xx, yy) for yy in range(y, y + 3) for xx in range(x, x + 3)), default=-1)
                if net < 0 or not self.networks[net].outside_connected:
                    continue
                near = float(homes[max(0, y - 4):y + 7, max(0, x - 4):x + 7].sum())
                if best is None or near < best[0]:
                    best = (near, x, y)
        return None if best is None else (best[1], best[2])
