"""Civics: needs-based happiness, the complaints feed, mayoral approval,
service funding levels and policies.

Happiness on every home is a weighted sum of satisfied needs. The needs that
cost the most happiness city-wide become the complaints feed shown to every
agent ("40% of homes lack police coverage"). Approval follows happiness and
taxes; 12 months below 30% and the mayor loses the recall election.
"""
from __future__ import annotations

import numpy as np

from . import catalog as C
from .grid import box_sum

NEED_TEXT = {
    "power": "{p}% of homes have no power",
    "water": "{p}% of homes lack clean running water",
    "sewage": "{p}% of homes are not connected to sewage",
    "garbage": "garbage is piling up at {p}% of homes",
    "safety": "crime is high around {p}% of homes",
    "health": "{p}% of residents live in poor health",
    "education": "{p}% of homes have no school in reach",
    "commute": "commutes are long for {p}% of homes (average {commute:.0f} min)",
    "leisure": "{p}% of homes have no park or plaza nearby",
    "noise": "{p}% of homes are noisy",
    "pollution": "{p}% of homes suffer from air pollution",
    "tax": "residents say the {tax:.0f}% residential tax is too high",
    "affordability": "{p}% of households struggle to pay rent",
    "jobs": "{u:.0f}% of workers are unemployed",
}


class CivicsMixin:
    def _init_civics(self) -> None:
        shape = (self.h, self.w)
        self.funding = {d: 1.0 for d in C.FUNDED_DEPARTMENTS}      # 1.0 = 100%
        self.policies: set[str] = set()
        self.noise = np.zeros(shape)
        self.needs: dict[str, np.ndarray] = {}
        self.commute_need = np.ones(shape)        # replaced by the traffic model
        self.affordability = np.ones(shape)       # replaced by the household budget model
        self.complaints: list[dict] = []
        self.need_avg: dict[str, float] = {}
        self.approval = C.APPROVAL_START
        self.low_approval_months = 0
        self.recalled = False
        self._tax_shock = 0.0
        self.disaster_morale = 0.0
        self.districts: dict[str, dict] = {}
        self.district_id = np.zeros(shape, np.int8)      # 0 = no district

    # ---------------------------------------------------------- districts
    def paint_district(self, name: str, tiles: list[tuple[int, int]]) -> str:
        if name not in self.districts:
            if len(self.districts) >= C.MAX_DISTRICTS:
                raise ValueError(f"at most {C.MAX_DISTRICTS} districts")
            used = {d["id"] for d in self.districts.values()}
            did = next(i for i in range(1, C.MAX_DISTRICTS + 1) if i not in used)
            self.districts[name] = {"id": did, "policies": set(), "tax_mod": {"R": 0.0, "C": 0.0, "I": 0.0},
                                    "specialization": None}
        did = self.districts[name]["id"]
        for x, y in tiles:
            self.district_id[y, x] = did
        for n in [n for n, d in self.districts.items() if not (self.district_id == d["id"]).any()]:
            del self.districts[n]                           # fully painted over
        return f"district '{name}' now covers {int((self.district_id == did).sum())} tiles"

    def set_district_policy(self, name: str, policy: str, on: bool) -> str:
        d = self.districts[name]
        if on:
            d["policies"].add(policy)
            return f"enacted {policy} in district '{name}': {C.POLICIES[policy]['desc']}"
        d["policies"].discard(policy)
        return f"repealed {policy} in district '{name}'"

    def district_tax_mod(self, sector: str) -> np.ndarray:
        out = np.zeros((self.h, self.w))
        for d in self.districts.values():
            if d["tax_mod"].get(sector):
                out[self.district_id == d["id"]] = d["tax_mod"][sector]
        return out

    def district_specialization(self) -> np.ndarray | None:
        from .economy import IND_TYPES
        if not any(d["specialization"] for d in self.districts.values()):
            return None
        out = np.full((self.h, self.w), -1, np.int16)
        for d in self.districts.values():
            if d["specialization"]:
                out[self.district_id == d["id"]] = IND_TYPES.index(d["specialization"])
        return out

    def districts_summary(self) -> list[dict]:
        return [{"name": n, "tiles": int((self.district_id == d["id"]).sum()), "policies": sorted(d["policies"]),
                 "tax_mod_pct": {k: round(100 * v, 1) for k, v in d["tax_mod"].items() if v},
                 "specialization": d["specialization"]} for n, d in self.districts.items()]

    # ------------------------------------------------ funding & policies
    def funding_eff(self, dept: str | None) -> float:
        """Effectiveness of a department at its funding level: min(1.5, pct)^0.7."""
        if not dept or dept not in self.funding:
            return 1.0
        return float(min(1.5, max(0.0, self.funding[dept])) ** 0.7)

    def policy_on(self, name: str) -> bool:
        return name in self.policies

    def policies_any(self, name: str) -> bool:
        return name in self.policies or any(name in d["policies"] for d in getattr(self, "districts", {}).values())

    def policy_mask(self, name: str) -> np.ndarray:
        """Where a policy applies: everywhere if enacted city-wide, else in districts that enacted it."""
        if name in self.policies:
            return np.ones((self.h, self.w), bool)
        mask = np.zeros((self.h, self.w), bool)
        for d in getattr(self, "districts", {}).values():
            if name in d["policies"]:
                mask |= self.district_id == d["id"]
        return mask

    def policy_costs(self) -> float:
        st = self.stats
        pop = st.get("population", 0.0)
        total = 0.0
        for name in C.POLICIES:
            share = float(self.policy_mask(name).mean()) if self.policies_any(name) else 0.0
            if share <= 0:
                continue
            spec = C.POLICIES[name]
            base = spec.get("per_resident", 0.0) * pop + spec.get("per_ind_job", 0.0) * st.get("jobs_ind", 0.0)
            total += spec.get("fixed", 0.0) * (1.0 if name in self.policies else 0.5) + base * share
        return total

    def tax_changed(self, old: float, new: float) -> None:
        """Tax hikes cost approval right away."""
        if new > old:
            self._tax_shock += C.APPROVAL_TAX_SHOCK * (new - old) * 100

    # ----------------------------------------------------------- noise
    def _compute_noise(self) -> None:
        from .grid import convolve, make_kernel
        s, lvl = self.struct, self.level
        src = np.zeros(s.shape)
        src[s == C.IND] = lvl[s == C.IND] * 1.0
        src[s == C.COM] = lvl[s == C.COM] * 0.3
        src[s == C.MIX] = lvl[s == C.MIX] * 0.2
        src[s == C.ROAD] = 0.25
        for struct, v in C.NOISE_SOURCES.items():
            src[s == struct] = v
        src += getattr(self, "traffic_noise", 0.0)
        if not hasattr(self, "_noise_kernel"):
            self._noise_kernel = make_kernel(radius=2, falloff=1.2)
        self.noise = 1.0 - np.exp(-convolve(src, self._noise_kernel) / 6.0)

    # ------------------------------------------------------------ needs
    def _compute_needs(self) -> dict[str, np.ndarray]:
        s = self.struct
        g_norm = np.clip(self.garbage / C.GARBAGE_NORM, 0, 1)
        leisure = np.minimum(1.0, box_sum(s == C.PARK, 4) / 2.0 + getattr(self, "landmark_bonus", 0.0))
        t = self.tax_grid("R") if hasattr(self, "tax_grid") else self.taxes["R"]
        tax = np.clip(1.0 - (t - 0.04) / 0.14, 0, 1) * np.ones(s.shape)
        return {
            "power": self.powered.astype(float),
            "water": self.watered * self.water_quality,
            "sewage": self.sewered.astype(float),
            "garbage": 1.0 - g_norm,
            "safety": np.clip(1.0 - self.crime / 0.4, 0, 1),
            "health": np.clip((self.health - 0.3) / 0.55, 0, 1),
            "education": np.clip(self.school_cov * 1.2, 0, 1),
            "commute": self.commute_need,
            "leisure": leisure,
            "noise": 1.0 - self.noise,
            "pollution": 1.0 - self.pollution,
            "tax": tax,
            "affordability": self.affordability,
            "jobs": np.full(s.shape, 1.0 - min(1.0, 2.0 * self.stats.get("unemployment", 0.0))),
        }

    def _compute_happiness(self) -> None:
        s = self.struct
        res_dev = self.homes_mask()
        needs = self._compute_needs()
        W = C.NEED_WEIGHTS
        wealth = getattr(self, "wealth", None)
        h = np.zeros(s.shape)
        wsum = np.zeros(s.shape)
        for k, g in needs.items():
            w = W[k] * (np.where(wealth == 3, C.RICH_NEED_WEIGHT.get(k, 1.0), 1.0) if wealth is not None else 1.0)
            h += w * g
            wsum += w
        a, b = C.HAPPINESS_STRETCH
        self.happiness = np.clip(a * h / wsum + b, 0, 1)
        self.needs = needs
        residents = self.residents_grid()
        tot = residents.sum()
        self.stats["happiness"] = float((self.happiness * residents).sum() / tot) if tot > 0 else 0.5
        self._complaints(needs, residents, tot)

    def _complaints(self, needs: dict, residents: np.ndarray, tot: float) -> None:
        if tot <= 0:
            self.complaints, self.need_avg = [], {}
            return
        avg = {k: float((g * residents).sum() / tot) for k, g in needs.items()}
        self.need_avg = {k: round(v, 3) for k, v in avg.items()}
        drags = sorted(((C.NEED_WEIGHTS[k] * (1.0 - v), k) for k, v in avg.items()), reverse=True)
        out = []
        for drag, k in drags[:5]:
            if drag < 0.004:
                continue
            share = float((residents * (needs[k] < 0.6)).sum() / tot)
            text = NEED_TEXT[k].format(p=round(100 * share), u=100 * self.stats.get("unemployment", 0.0),
                                       tax=100 * self.taxes["R"], commute=self.stats.get("commute_minutes", 0.0))
            out.append({"need": k, "drag": round(drag, 3), "share_pct": round(100 * share, 1), "text": text})
        self.complaints = out

    # --------------------------------------------------------- approval
    def _update_approval(self) -> None:
        st = self.stats
        if st.get("population", 0) < C.APPROVAL_MIN_POP:
            self._tax_shock = 0.0
            return
        tax = float(np.mean(list(self.taxes.values())))
        basics = np.mean([self.need_avg.get(k, 1.0) for k in ("power", "water", "safety", "health")])
        target = (0.55 * st.get("happiness", 0.5) + 0.15 * float(np.clip(1.0 - (tax - 0.05) / 0.12, 0, 1))
                  + 0.15 * basics + 0.15 * (1.0 if self.treasury >= 0 else 0.0)) - self.disaster_morale
        self.disaster_morale *= 0.7
        self.approval = float(np.clip(0.85 * self.approval + 0.15 * target - self._tax_shock, 0, 1))
        self._tax_shock = 0.0
        if self.approval < C.RECALL_APPROVAL:
            self.low_approval_months += 1
            left = C.RECALL_MONTHS - self.low_approval_months
            if left > 0:
                self.event(f"Mayor's approval is {100 * self.approval:.0f}% (below {100 * C.RECALL_APPROVAL:.0f}%) "
                           f"for {self.low_approval_months} month(s); recall election in {left} month(s) if it stays low")
            else:
                self.recalled = True
                self.event("The mayor LOST the recall election. Game over.")
        else:
            self.low_approval_months = 0
