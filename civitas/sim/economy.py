"""Economy: demand, zone development (density, levels 1-5), wealth classes,
specialised industry on natural resources, and household budgets.

Levels need progressively more of the city:
  L1 road access + power          L2 + water
  L3 + sewage + garbage pickup + land value 0.40
  L4 + school in reach (homes) or skilled workers (jobs) + land value 0.58
  L5 + land value 0.75 + low crime + a park or plaza nearby
Low-density zones stop at L2; the high-rise ban policy stops everything at L3.
Industry stops at L3 unless it is high-tech (needs educated workers).
"""
from __future__ import annotations

import math

import numpy as np

from . import catalog as C
from .grid import box_sum, convolve, make_kernel
from .population import ADULT, HIGH, UNIV

RESOURCE_NAMES = ("none", "fertile", "forest", "ore", "oil")
IND_TYPES = ("generic", "agriculture", "forestry", "ore", "oil", "hightech")
IND_BY_RESOURCE = {1: 1, 2: 2, 3: 3, 4: 4}


class EconomyMixin:
    def _init_economy(self) -> None:
        shape = (self.h, self.w)
        self.density = np.zeros(shape, np.uint8)       # 0 low, 1 high (per zoned tile)
        self.wealth = np.zeros(shape, np.uint8)        # residential: 1 low, 2 middle, 3 high income
        self.ind_type = np.zeros(shape, np.uint8)      # index into IND_TYPES
        self.resource = np.zeros(shape, np.uint8)      # index into RESOURCE_NAMES
        self.resource_left = np.zeros(shape)           # 1.0 = untouched deposit
        self.rent = np.zeros(shape)
        self.disposable = np.zeros(shape)

    # ---------------------------------------------------------- resources
    def _generate_resources(self) -> None:
        """Fertile soil along the river, a few forests, ore and oil fields (own RNG: same map every run)."""
        rng = np.random.default_rng([self.seed, 17])
        land = self.terrain == C.LAND
        res = np.zeros((self.h, self.w), np.uint8)
        noise = convolve(rng.random((self.h, self.w)), make_kernel(3, 2.0))
        noise = (noise - noise.min()) / max(1e-9, np.ptp(noise))
        res[land & (self.dist_water >= 2) & (self.dist_water <= 6) & (noise > 0.55)] = 1
        for kind, n, r in ((2, 3, 3), (3, 2, 2), (4, 2, 2)):
            for _ in range(n):
                cx, cy = int(rng.integers(2, self.w - 2)), int(rng.integers(2, self.h - 2))
                yy, xx = np.mgrid[0:self.h, 0:self.w]
                blob = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r + rng.integers(0, 3)) & land
                res[blob] = kind
        tx, ty = self.townhall
        res[max(0, ty - 3):ty + 4, max(0, tx - 3):tx + 4] = 0      # keep the centre generic
        self.resource = res
        self.resource_left = (res > 0).astype(float)

    # ------------------------------------------------------------- demand
    def _compute_demand(self) -> None:
        st, t = self.stats, self.taxes
        pop, workers = st["population"], st["workers"]
        jobs = st["jobs_com"] + st["jobs_ind"] + st.get("jobs_off", 0.0)
        hap = st.get("happiness", 0.5)
        early = 0.6 * max(0.0, 1.0 - pop / 300.0)
        commute = st.get("commute_minutes", 0.0)
        d_r = (0.6 * math.tanh((jobs - workers) / max(50.0, 0.2 * workers)) + early
               + (C.DEFAULT_TAX - t["R"]) * 4.0 + (hap - 0.5) * 0.4 + 0.1
               - 0.4 * min(1.0, max(0.0, (commute - 25.0) / 30.0)))
        spend = st.get("spending_index", 1.0)
        target_c = 0.15 * pop * spend + 0.08 * st.get("tourists", 0.0)
        d_c = (0.8 * math.tanh((target_c - st["jobs_com"]) / max(20.0, 0.3 * target_c))
               + (C.DEFAULT_TAX - t["C"]) * 3.0 - 0.3 * max(0.0, 1.0 - pop / 150.0)
               + self.event_factor("commercial_demand") - 1.0)
        target_i = 0.28 * pop + 60.0 + st.get("export_demand", 0.0)
        d_i = (0.8 * math.tanh((target_i - st["jobs_ind"]) / max(30.0, 0.3 * target_i))
               + (C.DEFAULT_TAX - t["I"]) * 3.0
               + (C.POLICIES["industry_filters"]["demand"] if self.policy_on("industry_filters") else 0.0)
               + self.event_factor("industry_demand") - 1.0)
        # offices follow the educated workforce that is not yet in skilled jobs
        adults = self.cohorts[ADULT]
        skilled = float(adults[HIGH:].sum()) * C.LABOR_PARTICIPATION
        target_o = C.OFFICE_SHARE_OF_SKILLED * skilled
        d_o = (0.8 * math.tanh((target_o - st.get("jobs_off", 0.0)) / max(20.0, 0.3 * target_o))
               + (C.DEFAULT_TAX - t["C"]) * 3.0 - 0.4 * max(0.0, 1.0 - pop / 800.0))
        # demand caps: without links to the region the city stops attracting people and shoppers
        caps = self.demand_caps()
        clamp = lambda v, hi=1.0: float(max(-1.0, min(hi, v)))
        lvc = st.get("land_value_city", 0.4)
        wealthy = min(1.0, max(0.0, (lvc - 0.35) / 0.35))
        r = clamp(d_r, caps["R"])
        self.demand = {"R": r,
                       "R_low": clamp(d_r + 0.25 * (wealthy - 0.4), caps["R"]),
                       "R_high": clamp(d_r + 0.1 * min(1.0, pop / 2000.0) - 0.15 * wealthy, caps["R"]),
                       "C": clamp(d_c, caps["C"]), "I": clamp(d_i), "O": clamp(d_o, caps["C"])}

    def demand_caps(self) -> dict[str, float]:
        """SimCity 4 style caps: big cities need regional links (a train station) and parks."""
        pop = self.stats.get("population", 0.0)
        caps = {"R": 1.0, "C": 1.0}
        linked = bool(self.facilities(C.TRAIN)) or bool(self.facilities(C.HARBOUR))
        if pop >= C.CAP_POP_CONNECTION and not linked:
            caps["R"] = caps["C"] = C.CAP_LEVEL
        leisure = int((self.struct == C.PARK).sum()) + 6 * sum(self.n_buildings(st) for st in (C.PLAZA, C.STADIUM, C.MONUMENT))
        if pop >= C.CAP_POP_PARKS and leisure < pop / C.RESIDENTS_PER_PARK:
            caps["R"] = min(caps["R"], C.CAP_LEVEL)
        self.stats["demand_caps"] = {k: v for k, v in caps.items() if v < 1.0}
        return caps

    # --------------------------------------------------------- development
    def max_levels(self) -> np.ndarray:
        s, lv = self.struct, self.land_value
        res, com, ind, off = np.isin(s, C.HOMES), s == C.COM, s == C.IND, s == C.OFF
        ml = np.zeros(s.shape, np.int32)
        ok1 = self.access & self.powered
        ml[ok1] = 1
        ok2 = ok1 & self.watered
        ml[ok2] = 2
        ok3 = ok2 & self.sewered & (self.garbage_cov > 0.3) & ((lv >= C.LV_FOR_L3) | ind)
        ml[ok3] = 3
        skilled_ok = self.stats.get("skilled_fill", 1.0) >= 0.8
        ok4 = ok3 & (lv >= C.LV_FOR_L4) & np.where(res, self.school_cov > 0.1, skilled_ok)
        ml[ok4] = 4
        near_park = box_sum(np.isin(s, (C.PARK, C.PLAZA, C.STADIUM, C.MONUMENT)), 4) > 0
        ok5 = ok4 & (lv >= C.LV_FOR_L5) & (self.crime < 0.25) & (near_park | ind)
        ml[ok5] = 5
        # caps
        hightech = ind & (self.ind_type == 5)
        ml[ind & ~hightech] = np.minimum(ml[ind & ~hightech], 3)
        univ_share = float(self.cohorts[ADULT, UNIV].sum() / max(1.0, self.cohorts[ADULT].sum()))
        if univ_share < 0.12:
            ml[hightech] = np.minimum(ml[hightech], 4)
        low = (self.density == 0) & (res | com | off) & (s != C.MIX)     # mixed use is always dense
        ml[low] = np.minimum(ml[low], C.LOW_DENSITY_MAX)
        if self.policies_any("high_rise_ban"):
            ban = self.policy_mask("high_rise_ban") & (res | com | off)
            ml[ban] = np.minimum(ml[ban], C.POLICIES["high_rise_ban"]["max_level"])
        ml[res & (self.pollution > 0.7)] = np.minimum(ml[res & (self.pollution > 0.7)], 1)
        return ml

    def _zone_demand_grid(self) -> np.ndarray:
        s = self.struct
        d = np.zeros(s.shape)
        d[s == C.RES] = np.where(self.density[s == C.RES] == 1, self.demand["R_high"], self.demand["R_low"])
        d[s == C.MIX] = 0.5 * (self.demand["R_high"] + self.demand["C"]) + C.MIX_DEMAND_BONUS
        for letter in ("C", "I", "O"):
            d[s == C.ZONE_BY_LETTER[letter]] = self.demand[letter]
        if self.districts:                  # district tax modifiers shift local demand
            for letter, sector in (("R", "R"), ("C", "C"), ("O", "C"), ("I", "I")):
                m = s == C.ZONE_BY_LETTER[letter]
                d[m] -= 3.0 * self.district_tax_mod(sector)[m]
        return d

    def _grow(self) -> None:
        s, lvl, rng = self.struct, self.level, self.rng
        zone_mask = np.isin(s, C.ZONES)
        if not zone_mask.any():
            return
        lv = self.land_value
        ml = self.max_levels()
        d = self._zone_demand_grid()
        residents = self.residents_grid()
        customers = np.minimum(1.0, box_sum(residents, 6) / 400.0)
        desir = np.where(np.isin(s, C.HOMES), 0.4 * lv + 0.3 * (1 - self.pollution) + 0.2 * self.happiness
                         + 0.1 * (1 - self.crime),
                         np.where(s == C.COM, 0.5 * lv + 0.5 * customers,
                                  np.where(s == C.OFF, 0.6 * lv + 0.4 * (1 - self.noise), 0.7)))
        r = rng.random(s.shape)
        up = zone_mask & (lvl < ml) & (d > 0) & (r < 0.10 + 0.5 * d * desir) & ~self.historic
        r2 = rng.random(s.shape)
        homes = np.isin(s, C.HOMES)
        rich_flight = homes & (self.wealth == 3) & (self.happiness < 0.45)
        unaffordable = homes & (self.disposable < -0.5)
        closing = getattr(self, "closing", np.zeros(s.shape, bool))
        down = zone_mask & (lvl > 0) & (
            ((lvl > ml) & (r2 < 0.35))
            | ((d < -0.2) & (r2 < 0.15 * -d))
            | (homes & (self.happiness < 0.25) & (r2 < 0.10))
            | (rich_flight & (r2 < 0.08))
            | (unaffordable & (r2 < 0.05))
            | closing)
        down &= ~self.historic                  # listed buildings keep their level
        closing = closing & ~self.historic
        new = lvl.astype(np.int32) + up.astype(np.int32)
        new[down] = lvl[down].astype(np.int32) - 1
        if closing.any():
            new[closing & zone_mask] = 0
        self.level = np.clip(new, 0, C.MAX_LEVEL).astype(np.uint8)
        self._assign_variants(up & ~down)
        n_up, n_down = int((up & ~down).sum()), int(down.sum())
        if n_down >= 5:
            self.event(f"{n_down} zone tiles declined (lost services, low demand or unhappy residents)")
        self.stats["tiles_grew"], self.stats["tiles_declined"] = n_up, n_down

    def _assign_variants(self, grew: np.ndarray) -> None:
        """Wealth class of homes follows land value; new industry specialises on local resources."""
        s, lv = self.struct, self.land_value
        res = grew & np.isin(s, C.HOMES)
        self.wealth[res] = np.where(lv[res] >= C.WEALTH_LV[1], 3, np.where(lv[res] >= C.WEALTH_LV[0], 2, 1))
        new_ind = grew & (s == C.IND) & (self.level == 1)
        if new_ind.any():
            spec = self.district_specialization() if hasattr(self, "district_specialization") else None
            kind = np.zeros(s.shape, np.uint8)
            has_res = (self.resource > 0) & (self.resource_left > 0.1)
            kind[has_res] = [IND_BY_RESOURCE[int(v)] for v in self.resource[has_res]]
            edu = self.education_index()
            tech = (edu >= C.HIGHTECH_EDU) & (lv >= 0.45) & (self.rng.random(s.shape) < 0.5) & ~has_res
            kind[tech] = 5
            if spec is not None:
                forced = spec >= 0
                kind[forced] = spec[forced]
                bad = forced & (kind >= 1) & (kind <= 4) & ~((self.resource == kind) & has_res)
                kind[bad] = 0                         # a resource specialisation needs the resource
            self.ind_type[new_ind] = kind[new_ind]
        self.wealth[~np.isin(s, C.HOMES)] = 0
        self.ind_type[s != C.IND] = 0

    def _deplete_resources(self) -> None:
        ext = (self.struct == C.IND) & (self.ind_type >= 1) & (self.ind_type <= 4) & (self.level > 0)
        if not ext.any():
            return
        rate = np.array(C.RESOURCE_DEPLETION)[self.ind_type[ext]]
        self.resource_left[ext] = np.maximum(0.0, self.resource_left[ext] - rate * self.level[ext])
        gone = ext & (self.resource_left <= 0.05)
        if gone.any():
            self.ind_type[gone] = 0
            self.resource[gone] = 0
            self.event(f"{int(gone.sum())} industrial tile(s) used up their natural resource and became generic industry")

    # -------------------------------------------------------- households
    def _household_budget(self, match: dict) -> None:
        """Income from wages, rent from land value: what is left drives shopping and affordability."""
        s = self.struct
        res = self.homes_mask()
        pop = max(1.0, self.stats.get("population", 0.0))
        employed_e = match.get("employed_by_edu", np.zeros(4))
        wages = float((employed_e * np.array(C.WAGE_BY_EDU)).sum())
        unemployed = max(0.0, match["workers"] - float(employed_e.sum()))
        income_pc = (wages + unemployed * C.UNEMPLOYMENT_SUPPORT) / pop
        mult = np.array(C.WEALTH_INCOME)[self.wealth]
        income = income_pc * mult
        rent = (C.RENT_BASE + C.RENT_PER_LV * self.land_value) * np.where(self.density == 1, C.HIGH_DENSITY_RENT, 1.0)
        self.rent = np.where(res, rent * np.array(C.WEALTH_RENT)[self.wealth], 0.0)
        self.disposable = np.where(res, income - self.rent - C.LIVING_COST, 0.0)
        self.affordability = np.where(res, np.clip(0.5 + self.disposable / 3.0, 0, 1), 1.0)
        residents = self.residents_grid()
        rs = residents.sum()
        avg_disp = float((self.disposable * residents).sum() / rs) if rs > 0 else C.REF_DISPOSABLE
        self.stats.update({
            "income_per_resident": income_pc,
            "disposable_income": avg_disp,
            "spending_index": float(np.clip(0.4 + 0.6 * avg_disp / C.REF_DISPOSABLE, 0.7, 1.4)),
            "homeless_share": float((residents * (self.disposable < -0.5)).sum() / rs) * 0.2 if rs > 0 else 0.0,
            "land_value_city": float(self.land_value[np.isin(s, C.ZONES) & (self.level > 0)].mean())
            if (np.isin(s, C.ZONES) & (self.level > 0)).any() else 0.4,
        })

    def residential_taxable(self) -> float:
        residents = self.residents_grid()
        mult = np.array(C.WEALTH_TAX)[self.wealth]
        return float((residents * mult * self.tax_grid("R")).sum()) * C.TAXABLE_PER_RESIDENT

    def tax_grid(self, sector: str) -> np.ndarray | float:
        """Tax rate per tile (the city rate plus district modifiers)."""
        base = self.taxes[sector]
        mods = getattr(self, "district_tax_mod", None)
        if mods is None:
            return base
        return np.clip(base + mods(sector), 0.0, C.MAX_TAX)


class TradeMixin:
    """Goods, freight, trade with the region, company profits, service imports."""

    def _init_trade(self) -> None:
        shape = (self.h, self.w)
        self.margin = np.zeros(shape)              # business profit margin per C/I/O tile
        self.losing_months = np.zeros(shape, np.int16)
        self.closing = np.zeros(shape, bool)       # businesses that close this month
        self.service_imports = True
        self.imports = {"power": 0.0, "water": 0.0}
        self.power_sold = 0.0
        self.trade: dict = {}
        self.external = {"power": 0.0, "water": 0.0}         # neighbour deals: + bought, - sold
        rng = np.random.default_rng([self.seed, 23])
        wind = convolve(rng.random(shape), make_kernel(6, 3.0))
        wind = (wind - wind.min()) / max(1e-9, np.ptp(wind))
        self.wind = 0.6 + 0.8 * wind + 0.1 * (np.arange(self.w)[None, :] / self.w)   # windier to the east

    # -------------------------------------------------------------- goods
    def freight_capacity(self) -> float:
        cap = C.FREIGHT_CAPACITY["highway"]
        hx, hy = self.highway
        jam = float(self.congestion[max(0, hy - 1):hy + 2, 0:4].mean()) if hasattr(self, "congestion") else 0.0
        cap *= max(0.3, 1.0 - 0.5 * max(0.0, jam - 0.8))
        cap += C.FREIGHT_CAPACITY["train"] * len(self.facilities(C.TRAIN))
        cap += C.FREIGHT_CAPACITY["harbour"] * len(self.facilities(C.HARBOUR))
        cap += C.AIRPORT_SPEC["freight"] * self.n_buildings(C.AIRPORT)
        return cap

    def _compute_trade(self) -> None:
        s, lvl = self.struct, self.level
        st = self.stats
        ind = (s == C.IND) & (lvl > 0)
        jobs_i = st.get("jobs_ind", 0.0)
        fill_i = st.get("jobs_filled_ind", 0.0) / jobs_i if jobs_i > 0 else 0.0
        jobs_t = np.array(C.IND_JOBS)[lvl] * ind * fill_i
        out = np.array(C.GOODS_OUTPUT)[self.ind_type] * np.where(self.ind_type == 1, 1.0 - self.soil_level(), 1.0)
        prod_goods = float((jobs_t * out * np.isin(self.ind_type, (0, 5))).sum())
        prod_raw = float((jobs_t * out * ((self.ind_type >= 1) & (self.ind_type <= 4))).sum())
        need_goods = st.get("jobs_filled_com", 0.0) * C.GOODS_PER_SHOP_JOB
        need_raw = float((jobs_t * (self.ind_type == 0)).sum()) * C.RAW_PER_FACTORY_JOB \
            + float((jobs_t * (self.ind_type == 5)).sum()) * C.RAW_PER_TECH_JOB
        made_goods, made_raw = prod_goods, prod_raw
        # trade with linked cities comes first (region mode): their goods cover our shortfalls and vice versa
        rg = self.regional
        prod_goods += rg.get("goods_in", 0.0) - rg.get("goods_out", 0.0)
        prod_raw += rg.get("raw_in", 0.0) - rg.get("raw_out", 0.0)
        price = self.event_factor("export_price")
        cap = self.freight_capacity()
        surplus = max(0.0, prod_goods - need_goods) * C.PRICE["goods"] + max(0.0, prod_raw - need_raw) * C.PRICE["raw"]
        deficit_goods = max(0.0, need_goods - prod_goods)
        deficit_raw = max(0.0, need_raw - prod_raw)
        exports = min(surplus, cap)
        imports = min(deficit_goods + deficit_raw, max(0.0, cap - exports))
        short_goods = max(0.0, deficit_goods - imports) / need_goods if need_goods > 0 else 0.0
        sold = (min(prod_goods, need_goods) + min(prod_raw, need_raw)) + exports \
            + rg.get("goods_out", 0.0) + rg.get("raw_out", 0.0)
        produced = prod_goods + prod_raw + rg.get("goods_out", 0.0) + rg.get("raw_out", 0.0) \
            - rg.get("goods_in", 0.0) - rg.get("raw_in", 0.0)
        self.trade = {
            "goods_produced": prod_goods, "raw_produced": prod_raw, "goods_needed": need_goods, "raw_needed": need_raw,
            "goods_made": made_goods, "raw_made": made_raw,
            "exports": exports, "imports": imports, "freight_capacity": cap,
            "sold_share": min(1.0, sold / produced) if produced > 0 else 1.0,
            "goods_shortage": short_goods, "export_price": price,
            "freight_spare": max(0.0, cap - exports - imports),
        }
        room = min(self.trade["freight_spare"], C.EXPORT_DEMAND_CAP, 0.15 * st.get("population", 0.0))
        st["export_demand"] = C.EXPORT_JOBS_PER_SPARE * room * price
        st["freight"] = exports + imports

    # ------------------------------------------------------------ profits
    def _company_profits(self) -> None:
        s, lvl = self.struct, self.level
        st, tr = self.stats, self.trade
        biz = np.isin(s, (C.COM, C.IND, C.OFF, C.MIX)) & (lvl > 0)
        if not biz.any():
            self.margin[:] = 0.0
            self.losing_months[:] = 0
            self.closing[:] = False
            st.update({"businesses_losing": 0, "business_closures": 0})
            return
        residents = self.residents_grid()
        customers = np.minimum(1.0, box_sum(residents, 6) / 400.0)
        tax = lambda sec: self.tax_grid(sec) - C.DEFAULT_TAX
        cong = np.where(self._anchors() >= 0, self.congestion.ravel()[np.maximum(self._anchors(), 0)], 0.0) \
            if hasattr(self, "congestion") else 0.0
        m_c = (C.BASE_MARGIN + 0.5 * (customers - 0.5) + 0.3 * (st.get("spending_index", 1.0) - 1.0)
               - 2.0 * tax("C") - 0.5 * tr.get("goods_shortage", 0.0) - 0.15 * np.clip(cong - 1.0, 0, 2)
               + 0.1 * self.stats.get("tourists", 0.0) / max(1000.0, st.get("population", 0.0)))
        m_i = (C.BASE_MARGIN + 0.5 * (tr.get("sold_share", 1.0) - 0.85) + 0.4 * (tr.get("export_price", 1.0) - 1.0)
               - 2.0 * tax("I") - 0.2 * np.clip(cong - 1.0, 0, 2) + 0.05 * ((self.ind_type >= 1) & (self.ind_type <= 5)))
        m_o = (C.BASE_MARGIN + 0.6 * (st.get("skilled_fill", 1.0) - 0.8) - 2.0 * tax("C") + 0.2 * (self.land_value - 0.5)
               - 0.1 * np.clip(cong - 1.0, 0, 2))
        m = np.where(np.isin(s, C.SHOPS), m_c, np.where(s == C.IND, m_i, m_o)) - 0.3 * self.crime - 0.2 * (1.0 - self.powered)
        self.margin = np.where(biz, m, 0.0)
        self.losing_months = np.where(biz & (self.margin < 0), self.losing_months + 1, 0).astype(np.int16)
        self.closing = biz & (self.losing_months >= C.CLOSE_AFTER_MONTHS)
        n = int(self.closing.sum())
        if n:
            self.losing_months[self.closing] = 0
            self.event(f"{n} business(es) closed after {C.CLOSE_AFTER_MONTHS} months of losses (taxes, shortages, "
                       "traffic or lack of customers)")
        st.update({"businesses_losing": int((biz & (self.margin < 0)).sum()), "business_closures": n,
                   "margin_com": float(self.margin[(s == C.COM) & biz].mean()) if ((s == C.COM) & biz).any() else 0.0,
                   "margin_ind": float(self.margin[(s == C.IND) & biz].mean()) if ((s == C.IND) & biz).any() else 0.0,
                   "margin_off": float(self.margin[(s == C.OFF) & biz].mean()) if ((s == C.OFF) & biz).any() else 0.0})

    # --------------------------------------------------- utilities trade
    def plant_output_factor(self, x: int, y: int, kind: str) -> float:
        """Ageing (power plants lose output after a few years), wind strength, expansions."""
        f = self.module_factor(x, y, "expansion")
        if kind in ("coal", "gas", "solar", "wind", "hydro"):
            age = (self.month - self.facility_state.get((x, y), {}).get("built", self.month)) / 12.0
            f *= max(C.AGEING_FLOOR, 1.0 - C.AGEING_RATE * max(0.0, age - C.AGEING_START_YEARS))
        if kind == "wind":
            f *= float(self.wind[y, x])
        return f

    def module_factor(self, x: int, y: int, module: str) -> float:
        mods = self.facility_state.get((x, y), {}).get("modules", ())
        if module == "expansion":
            return C.MODULES["expansion"]["capacity"] if "expansion" in mods else 1.0
        if module == "filter":
            return C.MODULES["filter"]["pollution"] if "filter" in mods else 1.0
        return 1.0

    def service_import_price(self, res: str) -> float:
        return C.IMPORT_PRICE[res] * (1.0 + C.IMPORT_PRICE_GROWTH * self.stats.get("population", 0.0) / 1000.0)
