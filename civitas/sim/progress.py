"""Progression: population milestones that unlock buildings (and pay a
reward), landmarks, and the tourism they bring.

Everything starts locked except the basics, so agents must notice when new
options appear and adapt their plans (the action docs mark what is locked).
"""
from __future__ import annotations

import numpy as np

from . import catalog as C
from .grid import box_sum


class ProgressMixin:
    def _init_progress(self, milestones: bool = True) -> None:
        self.milestones_on = milestones
        self.milestones_reached: list[str] = []
        self.landmark_bonus = np.zeros((self.h, self.w))
        self.landmark_lv = np.zeros((self.h, self.w))

    # ------------------------------------------------------------ unlocks
    def unlocked(self, key: str) -> bool:
        if not self.milestones_on:
            return True
        need = C.UNLOCK_POP.get(key)
        if need is None:
            return True
        return any(m["pop"] >= need for m in C.MILESTONES if m["name"] in self.milestones_reached) or need == 0

    def unlock_note(self, key: str) -> str:
        need = C.UNLOCK_POP.get(key, 0)
        name = next((m["name"] for m in C.MILESTONES if m["pop"] == need), "")
        return f"{key.replace('_', ' ')} unlocks when the city reaches {need:,} residents (milestone '{name}')"

    def locked_now(self) -> list[str]:
        return sorted(k for k in C.UNLOCK_POP if not self.unlocked(k))

    def next_milestone(self) -> dict | None:
        for m in C.MILESTONES:
            if m["name"] not in self.milestones_reached:
                return {**m, "unlocks": [k for k, v in C.UNLOCK_POP.items() if v == m["pop"]]}
        return None

    def _check_milestones(self) -> None:
        if not self.milestones_on:
            return
        pop = self.stats.get("population", 0)
        for m in C.MILESTONES:
            if m["name"] in self.milestones_reached or pop < m["pop"]:
                continue
            self.milestones_reached.append(m["name"])
            self.treasury += m["reward"]
            new = [k.replace("_", " ") for k, v in C.UNLOCK_POP.items() if v == m["pop"]]
            self.event(f"MILESTONE '{m['name']}' ({m['pop']:,} residents): ${m['reward']:,} reward"
                       + (f"; unlocked: {', '.join(new)}" if new else ""))

    # ---------------------------------------------------------- landmarks
    def _compute_landmarks(self) -> None:
        lb = np.zeros((self.h, self.w))
        lvb = np.zeros((self.h, self.w))
        visitors = 0.0
        for kind, spec in C.LANDMARKS.items():
            m = self.struct == spec["struct"]
            if not m.any():
                continue
            lb += spec["leisure"] * np.minimum(1.0, box_sum(m, spec["radius"]))
            lvb += spec["land_value"] * np.minimum(1.0, box_sum(m, spec["radius"]))
            visitors += spec["tourism"] * self.n_buildings(spec["struct"])
        self.landmark_bonus, self.landmark_lv = lb, lvb
        appeal = 0.5 + 0.5 * self.stats.get("happiness", 0.5) - 0.3 * self.stats.get("crime_city", 0.0)
        visitors += self.stats.get("heritage_tourists", 0.0)
        casino = C.CASINO_SPEC["tourism"] * self.n_buildings(C.CASINO)
        boost = C.AIRPORT_SPEC["tourism_boost"] if self.n_buildings(C.AIRPORT) else 1.0
        f = max(0.2, appeal) * self.event_factor("tourism") * boost
        regional = self.regional.get("tourists", 0.0)          # visitors from linked cities (region mode)
        share = casino / max(1.0, casino + visitors)
        self.stats["casino_tourists"] = casino * f + regional * share
        self.stats["tourists"] = (visitors + casino) * f + regional


class HeritageMixin:
    """Historic preservation, soil pollution, wind direction and seasons."""

    def _init_heritage(self, months: int, seasons: bool) -> None:
        shape = (self.h, self.w)
        self.historic = np.zeros(shape, bool)
        self.soil = np.zeros(shape)                 # accumulated contamination (raw units)
        self.seasons_on = seasons
        rng = np.random.default_rng([self.seed, 41])
        base = float(np.arctan2(C.PREVAILING_WIND[1], C.PREVAILING_WIND[0]))
        self._wind_angles = base + rng.normal(0.0, 0.8, size=months + 24)
        self._wind_kernel_key = None

    # ------------------------------------------------------------ seasons
    def season(self) -> str:
        return C.SEASONS[((self.month + 1) % 12) // 3]        # Dec-Feb winter, Mar-May spring, ...

    def season_factor(self, key: str) -> float:
        if not self.seasons_on:
            return 1.0
        return C.SEASON_EFFECTS[self.season()].get(key, 1.0)

    # --------------------------------------------------------------- wind
    def wind_dir(self) -> tuple[float, float]:
        a = float(self._wind_angles[min(self.month, len(self._wind_angles) - 1)])
        return float(np.cos(a)), float(np.sin(a))

    def wind_name(self) -> str:
        dx, dy = self.wind_dir()
        names = ["east", "south-east", "south", "south-west", "west", "north-west", "north", "north-east"]
        k = int(round(np.degrees(np.arctan2(dy, dx)) / 45.0)) % 8      # y grows south
        return names[k]

    def pollution_kernel(self):
        """Air pollution spreads ~5 tiles, mostly downwind."""
        if self._wind_kernel_key == self.month:
            return self._wind_kernel
        wx, wy = self.wind_dir()
        ker = []
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                d = (dx * dx + dy * dy) ** 0.5
                if d > 6.5:
                    continue
                cos = (dx * wx + dy * wy) / d if d > 0 else 1.0
                ker.append((dy, dx, float(np.exp(-d / 2.0) * max(0.15, 1.0 + 0.85 * cos))))
        self._wind_kernel, self._wind_kernel_key = ker, self.month
        return ker

    # --------------------------------------------------------------- soil
    def soil_level(self) -> np.ndarray:
        return 1.0 - np.exp(-self.soil / 3.0)

    def _update_soil(self) -> None:
        s, lvl = self.struct, self.level
        src = np.zeros(s.shape)
        ind = (s == C.IND) & (lvl > 0)
        src[ind] = lvl[ind] * np.array(C.SOIL_SOURCE["industry"])[self.ind_type[ind]]
        for struct, v in C.SOIL_SOURCE.items():
            if struct == "industry":
                continue
            for x, y in self.facilities(struct):   # per building, scaled by its tier's pollution, spread over its tiles
                spec = self.facility_spec(x, y)
                base = C.TIER_LINES[spec["line"]][0].get("pollution") if spec else None
                amt = v * spec["pollution"] / base if spec and spec.get("pollution") and base else v
                tiles = self.parts(x, y)
                for px, py in tiles:
                    src[py, px] += amt / len(tiles)
        soil = self.soil * C.SOIL_DECAY + src
        self.soil = (1.0 - C.SOIL_SPREAD) * soil + C.SOIL_SPREAD * box_sum(soil, 1) / 9.0

    # ----------------------------------------------------------- heritage
    def _update_heritage(self) -> None:
        self.historic &= np.isin(self.struct, C.ZONES) & (self.level > 0)
        n = int(self.historic.sum())
        self.stats["historic_tiles"] = n
        self.stats["heritage_tourists"] = n * C.HISTORIC_TOURISTS
