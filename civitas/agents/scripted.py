"""Rule-based baseline agents. They only read the observation (no privileged
access to the World), so they can also be run as remote agents and used to
generate demonstration data for fine-tuning LLMs."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..sim import catalog as C
from .base import Agent
from ..core.decision import Decision
from ..core.observation import GARBAGE_BUSY_RANK
from .mapview import MapView


# ====================================================================== mayor
class ScriptedMayor(Agent):
    kind = "scripted"

    def act(self, obs: dict) -> Decision:
        f, c = obs["finance"], obs["city"]
        treasury = f["treasury"]
        expenses = sum(f["last_month_expenses"].values()) if f["last_month_expenses"] else 50.0
        income = sum(f["last_month_income"].values()) if f["last_month_income"] else 0.0
        funds = f["department_funds"]
        acts, why = [], []

        requests = " ".join(m["text"].lower() for m in obs["council"]
                            if m["month"] >= obs["month"] - 1 and m["sender"] != "mayor")
        # --- loans
        if treasury < 3000 and f["debt"] < 30000 and obs["month"] < 96:
            acts.append({"type": "take_loan", "amount": 10000})
            treasury += 10000
            why.append("treasury low -> loan")
        elif f["debt"] > 0 and treasury > 25000:
            amt = min(f["debt"], treasury - 20000)
            acts.append({"type": "repay_loan", "amount": round(amt)})
            treasury -= amt
            why.append("repaying debt")

        # --- budget: keep each department near a working balance, most urgent first
        reserve = max(3000.0, (3 if income >= expenses else 6) * expenses)
        available = treasury - reserve
        pop = c["population"]
        power, water = c["power"], c["water"]
        tight = (power["supply"] < 1.3 * power["demand"] + 10 or water["supply"] < 1.3 * water["demand"] + 10
                 or "utilities" in requests and "fund" in requests)
        asks = _requested_amounts(obs["council"], obs["month"])
        targets = [("utilities", 7000 if tight else 3000, 0.45 if tight else 0.25), ("planning", 9000, 0.6)]
        if pop >= 150:
            targets = [targets[0], ("health", 3000, 0.35), ("fire", 3000, 0.35), ("police", 3000, 0.35),
                       ("education", 3000, 0.35), ("transport", 2000, 0.3), ("planning", 7000, 0.6)]
        given = {}
        for dept, target, share in targets:
            if available <= 500 or dept not in funds:
                continue
            target = max(target, asks.get(dept, 0) + 200)
            amt = min(available * share, max(0.0, target - funds[dept]))
            if amt >= 100:
                acts.append({"type": "allocate_budget", "department": dept, "amount": round(amt)})
                available -= amt
                treasury -= amt
                given[dept] = round(amt)
        if given:
            why.append(f"reserve {reserve:.0f}, gave " + ", ".join(f"{k} {v}" for k, v in given.items()))

        # --- policies: cheap fixes for the loudest complaints once the city can afford them
        complaints = " ".join(c.get("complaints", []))
        in_force = set(obs.get("policies", {}).get("in_force", []))
        rich = treasury > 15000 and income > expenses
        for policy, trigger in (("recycling", "garbage"), ("neighborhood_watch", "crime"), ("smoke_detectors", None)):
            if policy in in_force or not rich or pop < 2000:
                continue
            if (trigger and trigger in complaints) or (policy == "smoke_detectors" and c["services"]["fires_last_month"] >= 2):
                acts.append({"type": "enact_policy", "policy": policy})
                why.append(f"enact {policy}")
                break

        # --- offers: take the factory if people need jobs or the treasury needs the bonus
        for o in obs.get("offers", []):
            want = (c["unemployment_pct"] or 0) > 8 or treasury < 10000
            acts.append({"type": "accept_offer" if want else "decline_offer", "offer": o["id"]})
            why.append(f"{'accept' if want else 'decline'} offer #{o['id']}")
        # --- more money for the fire department when a flood is coming
        if any("flood" in f.lower() for f in obs.get("forecast", [])) and funds.get("fire", 0) < 4000 and treasury > 6000:
            acts.append({"type": "allocate_budget", "department": "fire", "amount": 3000})
            treasury -= 3000
            why.append("levee money before the flood")

        # --- taxes
        tax_r = c["tax_pct"]["R"]
        hap = c["happiness_pct"] or 50
        approval = c.get("approval_pct") or 60
        net = income - expenses
        if obs["month"] > 6:
            if net < 0 and hap > 55 and approval > 45 and tax_r < 12:
                acts.append({"type": "set_tax", "sector": "R", "rate": tax_r + 1})
                why.append("deficit -> raise R tax")
            elif (hap < 45 or net > 0.3 * max(income, 1)) and tax_r > 7:
                acts.append({"type": "set_tax", "sector": "R", "rate": tax_r - 1})
                why.append("lower R tax for happiness/growth")
        msgs = []
        if obs["month"] == 0:
            msgs.append({"to": "all", "text": "Budgets allocated. Planning: grow housing and jobs near the highway. "
                                              "Utilities: power and water first. Services: build once people move in. "
                                              "Ask me if you run short."})
        return Decision(actions=acts[: obs.get("max_actions", 8)], messages=msgs, thinking="; ".join(why))


def _requested_amounts(council: list[dict], month: int) -> dict[str, float]:
    """Parse '$1,234' amounts that departments asked the mayor for last month."""
    import re
    out: dict[str, float] = {}
    for m in council:
        if m["month"] < month - 1 or m["sender"] == "mayor" or m["to"] not in ("mayor", "all"):
            continue
        amounts = [float(a.replace(",", "")) for a in re.findall(r"\$([0-9][0-9,]*)", m["text"])]
        if amounts:
            out[m["sender"]] = max(out.get(m["sender"], 0.0), max(amounts))
    return out


# =================================================================== planning
class ScriptedPlanner(Agent):
    kind = "scripted"

    def reset(self, seed: int = 0) -> None:
        self.segments = None
        self.reserved = None

    def _plan(self, mv: MapView) -> None:
        tx, ty = mv.find("T")[0]
        yh = ty + 1
        sx = tx + 2
        self.center = (tx, ty)
        span = mv.river_span(yh)
        bank = (span[0] - 1) if span else mv.w - 2
        segs = [(sx, yh, bank, yh)]                       # main street to the river
        for k in range(1, 5):
            for sgn in (-1, 1):
                y = yh + sgn * 5 * k
                y_prev = yh + sgn * 5 * (k - 1)
                if not (1 <= y < mv.h - 1):
                    continue
                sp = mv.river_span(y)
                b = (sp[0] - 1) if sp else mv.w - 2
                segs.append((sx, y_prev, sx, y))          # spine
                segs.append((sx, y, b, y))                # row east of spine (river side)
                segs.append((sx, y, max(2, sx - 16), y))  # row west of spine
        # bridges + east bank grid. The east spine must stay on dry land for its whole length,
        # so it runs east of the river's widest meander; bridges cross the river straight.
        if span:
            rows = [yh + d for d in range(-20, 21) if 0 <= yh + d < mv.h]
            ends = [mv.river_span(y)[1] for y in rows if mv.river_span(y)]
            ex = min(mv.w - 3, max(ends) + 2)
            segs.append((bank, yh, ex, yh))
            for k in range(1, 5):
                for sgn in (-1, 1):
                    y = yh + sgn * 5 * k
                    y_prev = yh + sgn * 5 * (k - 1)
                    if 1 <= y < mv.h - 1:
                        segs.append((ex, y_prev, ex, y))
                        sp = mv.river_span(y)
                        if k == 2 and sp:                  # a second and third river crossing
                            segs.append(((sp[0] - 1), y, ex, y))
                        start = (sp[1] + 1) if sp else ex
                        segs.append((start, y, mv.w - 2, y))
            segs.append((ex, yh, mv.w - 2, yh))
        self.segments = segs
        res = np.zeros((mv.h, mv.w), bool)
        for x1, y1, x2, y2 in segs:
            for x, y in self._tiles(x1, y1, x2, y2):
                res[y, x] = True
        # civic lots: every 7th tile along each street stays empty for schools, clinics, stations...
        for x1, y1, x2, y2 in segs:
            if y1 == y2:
                for x in range(min(x1, x2) + 3, max(x1, x2), 7):
                    for yy in (y1 - 1, y1 + 1):
                        if 0 <= yy < mv.h and not res[yy, x]:
                            res[yy, x] = True
        self.reserved = res
        self.east_start = (span[1] + 1) if span else mv.w

    @staticmethod
    def _tiles(x1, y1, x2, y2):
        if y1 == y2:
            a, b = sorted((x1, x2))
            return [(x, y1) for x in range(a, b + 1)]
        a, b = sorted((y1, y2))
        return [(x1, y) for y in range(a, b + 1)]

    def _next_road(self, mv: MapView, budget: float):
        road = mv.mask("#=H")
        while self.segments:
            x1, y1, x2, y2 = self.segments[0]
            tiles = self._tiles(x1, y1, x2, y2)
            if all(road[y, x] for x, y in tiles):
                self.segments.pop(0)
                continue
            ordered = tiles if (x1, y1) == tiles[0] else tiles[::-1]
            if any(mv.g[y, x] not in ".~#=H" for x, y in ordered):
                # another department built on the planned line: route around it
                self.segments.pop(0)
                path = _route(mv, ordered[0], ordered[-1])
                if path:
                    self.segments[:0] = _runs(path)
                continue
            cost = sum(C.CATALOG["bridge" if mv.g[y, x] == "~" else "road"]["cost"]
                       for x, y in ordered if not road[y, x])
            if cost > budget:
                return None, cost
            self.segments.pop(0)
            (ax, ay), (bx, by) = ordered[0], ordered[-1]
            return {"type": "build_road", "x1": ax, "y1": ay, "x2": bx, "y2": by}, cost
        return None, 0

    def _region_ok(self, zone: str, x: int, y: int) -> bool:
        tx, ty = self.center
        near_center = abs(x - tx) + abs(y - ty) <= 7
        if zone == "I":
            wide = getattr(self, "wide_industry", False)
            return x < self.east_start and (x <= 9 or (wide and (x <= 13 or abs(y - ty) >= 12) and not near_center))
        if zone in ("C", "O"):
            return near_center or (x >= self.east_start and abs(y - (ty + 1)) <= 3)
        return x > 11 and not near_center or x >= self.east_start

    def _zone_runs(self, mv: MapView, zone: str, access, budget: float, max_runs: int):
        empty = (mv.g == ".") & access & ~self.reserved
        tx, ty = self.center
        ys, xs = np.nonzero(empty)
        cands = [(abs(x - tx) + abs(y - ty), x, y) for x, y in zip(xs.tolist(), ys.tolist())
                 if self._region_ok(zone, x, y)]
        cands.sort()
        acts, used, spent = [], set(), 0.0
        for _, x, y in cands:
            if len(acts) >= max_runs or (x, y) in used:
                continue
            x2 = x
            while x2 + 1 < mv.w and empty[y, x2 + 1] and x2 - x < 5 and self._region_ok(zone, x2 + 1, y):
                x2 += 1
            cost = (x2 - x + 1) * C.CATALOG["zone_high"]["cost"]
            if spent + cost > budget:
                break
            for xx in range(x, x2 + 1):
                used.add((xx, y))
            spent += cost
            tx, ty = self.center
            dense = (zone in ("C", "O") or abs(x - tx) + abs(y - ty) <= 14) and "high_density" not in self.locked
            acts.append({"type": "zone", "zone": zone, "density": "high" if dense else "low",
                         "x1": x, "y1": y, "x2": x2, "y2": y})
        return acts, spent

    def act(self, obs: dict) -> Decision:
        mv = MapView(obs["map"])
        self.locked = set(obs.get("progress", {}).get("locked", []))
        if getattr(self, "segments", None) is None:
            self._plan(mv)
        budget = obs["your_budget"] or 0.0
        c = obs["city"]
        access = mv.access()
        acts, why, msgs = [], [], []
        empty_access = int(((mv.g == ".") & access & ~self.reserved).sum())

        pending = {L: int((mv.g == L.lower()).sum()) for L in "RCIO"}
        demand = dict(c["demand"])
        demand["R"] = max(demand.get("R_low", demand["R"]), demand.get("R_high", demand["R"]))
        letters = "RCI" if "office_zone" in self.locked or c["population"] < 800 else "RCIO"
        starved = False
        for L in sorted(letters, key=lambda k: -demand.get(k, 0)):
            if demand[L] <= 0.05 or pending[L] >= 10:
                continue
            runs, spent = self._zone_runs(mv, L, access, budget, max_runs=3 if demand[L] > 0.4 else 1)
            if not runs and L == "I" and not getattr(self, "wide_industry", False):
                self.wide_industry = True       # the industrial strip is full: allow industry further out
                runs, spent = self._zone_runs(mv, L, access, budget, max_runs=2)
            if not runs and budget > 50:
                starved = True          # demand but nowhere to zone -> need roads
            acts.extend(runs)
            budget -= spent
            if runs:
                why.append(f"zone {L} (demand {demand[L]:+.2f})")

        if obs["month"] == 0 or starved or empty_access < 40:
            for _ in range(2 if starved else 1):
                a, cost = self._next_road(mv, budget * 0.8)
                if a:
                    acts.insert(0, a)
                    budget -= cost
                    why.append(f"new road ({cost:.0f})")
                else:
                    if cost:
                        msgs.append({"to": "mayor", "text": f"Planning needs about ${cost:,.0f} for the next road segment."})
                    break

        dev_r = c["developed_tiles"]["R"]
        if max(dev_r // 25, c["population"] // 350) > c["parks"] and budget > 400:
            rs = mv.find("R")
            empty = (mv.g == ".") & access & ~self.reserved
            best = None
            for x, y in rs:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < mv.w and 0 <= ny < mv.h and empty[ny, nx]:
                        best = (nx, ny)
                        break
                if best:
                    break
            if best:
                acts.append({"type": "build_park", "x": best[0], "y": best[1]})
                why.append("park for land value")
        # a plaza for every ~3,000 residents once unlocked (land value, leisure, tourists)
        if "plaza" not in self.locked and obs.get("landmarks", {}).get("plaza", 0) < c["population"] // 3000 \
                and budget > C.LANDMARKS["plaza"]["cost"] + 1000:
            rs = mv.find("R")
            empty = (mv.g == ".") & access & ~self.reserved
            for x, y in rs[len(rs) // 2:] + rs[:len(rs) // 2]:
                spot = next(((x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                             if 0 <= x + dx < mv.w and 0 <= y + dy < mv.h and empty[y + dy, x + dx]), None)
                if spot:
                    acts.append({"type": "build_landmark", "kind": "plaza", "x": spot[0], "y": spot[1]})
                    budget -= C.LANDMARKS["plaza"]["cost"]
                    why.append("plaza")
                    break
        # a low-education city can specialise in gambling: one casino (jobs + tourists, but crime)
        if ("casino" not in self.locked and "?" not in mv.g
                and (c["population"] > 6000 and (c["services"]["adults_high_school_plus_pct"] or 100) < 36
                     or c["population"] > 11000)):
            cost = C.CASINO_SPEC["cost"]
            sites = obs.get("casino_sites", [])
            if sites and budget >= cost:
                acts.append({"type": "build_casino", "x": sites[0]["x"], "y": sites[0]["y"]})
                budget -= cost
                why.append("casino (low-education city)")
            elif sites:
                msgs.append({"to": "mayor", "text": f"Planning needs ${cost:,} for a casino: jobs for low-skill workers."})
        if budget < 200 and obs["month"] > 0:
            msgs.append({"to": "mayor", "text": "Planning budget is nearly empty; zones are waiting for roads."})
        return Decision(actions=acts[: obs.get("max_actions", 8)], messages=msgs, thinking="; ".join(why))


def _route(mv: MapView, start, end):
    """Cheapest 4-connected path over empty land, water (as bridges) and roads,
    with a small penalty per turn so roads stay straight."""
    import heapq
    free = np.isin(mv.g, list(".~#=H"))
    (sx, sy), (ex, ey) = start, end
    if not (free[sy, sx] and free[ey, ex]):
        return None
    best = {(sx, sy, -1): 0.0}
    prev = {}
    heap = [(0.0, sx, sy, -1)]
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while heap:
        c, x, y, d = heapq.heappop(heap)
        if (x, y) == (ex, ey):
            path = [(x, y)]
            key = (x, y, d)
            while key in prev:
                key = prev[key]
                path.append((key[0], key[1]))
            return path[::-1]
        if c > best.get((x, y, d), 1e18):
            continue
        for k, (dx, dy) in enumerate(dirs):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < mv.w and 0 <= ny < mv.h and free[ny, nx]):
                continue
            ch = mv.g[ny, nx]
            if (ch == "~" or mv.g[y, x] == "~") and d not in (-1, k):
                continue                      # bridges only cross the river in a straight line
            step = 0.3 if ch in "#=H" else (30.0 if ch == "~" else 1.0)
            nc = c + step + (0.6 if d not in (-1, k) else 0.0)
            if nc < best.get((nx, ny, k), 1e18) and nc < 400:
                best[(nx, ny, k)] = nc
                prev[(nx, ny, k)] = (x, y, d)
                heapq.heappush(heap, (nc, nx, ny, k))
    return None


def _runs(path):
    """Split a tile path into straight (x1, y1, x2, y2) segments."""
    runs, i = [], 0
    while i < len(path) - 1:
        j = i + 1
        horiz = path[j][1] == path[i][1]
        while j + 1 < len(path) and (path[j + 1][1] == path[j][1]) == horiz and \
                (path[j + 1][0] == path[j][0]) != horiz:
            j += 1
        runs.append((path[i][0], path[i][1], path[j][0], path[j][1]))
        i = j
    return runs


# ================================================================== utilities
RESOURCES = ("power", "water", "sewage")


class Option(NamedTuple):
    """One way to add capacity: a tier upgrade in place or a new plant."""
    value: float            # net output per lifetime $ (price + ScriptedUtilities.LIFE months of extra upkeep)
    money_needed: float     # budget we want before acting (cost + reserves)
    cost: float
    action: dict
    note: str


def crowding(plants: list[dict], line: str, x: int, y: int) -> tuple[float, float]:
    """(output factor of a new plant of `line` at (x, y), output the plants around it would lose).

    plants: dicts with x, y, line, tier and (connected plants) their output and "crowd" factor."""
    g = next((g for g in C.CROWD_GROUPS.values() if line in g["lines"]), None)
    if g is None:
        return 1.0, 0.0
    n, lost = 0, 0.0
    for p in plants:
        if p["line"] not in g["lines"]:
            continue
        fw, fh = C.TIER_LINES[p["line"]][p["tier"] - 1]["footprint"]
        if (max(p["x"] - x, x - p["x"] - fw + 1, 0) <= g["radius"]
                and max(p["y"] - y, y - p["y"] - fh + 1, 0) <= g["radius"]):
            n += 1
            cf = p.get("crowd", 1.0)
            res = next(r for r in RESOURCES if C.TIER_LINES[p["line"]][0].get(r))
            lost += p.get(res, 0.0) / cf * max(0.0, min(C.CROWD_STEP, cf - C.CROWD_FLOOR))
    return max(C.CROWD_FLOOR, 1.0 - C.CROWD_STEP * n), lost


class ScriptedUtilities(Agent):
    kind = "scripted"
    LIFE = 36               # months of upkeep counted in the price when comparing options
    SMOG = 1.5              # units of output a point of plant pollution is worth giving up (dirty coal vs cleaner tiers)
    RESERVE = 1000          # cash kept back after a tier upgrade or a tier-2+ build
    COAL_MARGIN = 500       # extra cash kept back before building coal
    SHORTAGE = 1.05         # supply below this x demand is a real shortage: build what we can afford now ...
    GOOD_ENOUGH = 0.75      # ... as we also do when the affordable option is at least this good as the best one
    WIND_UNKNOWN = 0.7      # wind strength assumed at sites without a wind reading
    CLEAN_RIVER_POP = 3000  # from this population sewage options are weighed by how much they pollute the river
    DISCHARGE_WEIGHT = 0.5  # share of an option's value lost per unit of discharge (outlet 1.0, treatment 0.2)

    def act(self, obs: dict) -> Decision:
        c = obs["city"]
        budget = obs["your_budget"] or 0.0
        sites = obs.get("candidate_sites", {})
        net = next((n for n in obs["networks"] if n["outside_connected"]), None)
        acts, why, msgs = [], [], []
        if net is None:
            return Decision(thinking="no road network connected to the highway yet")
        pending_p = c["zoned_without_power"]
        pending_w = c["zoned_without_water"] + 0.5 * pending_p
        need = {"power": 1.25 * net["power_demand"] + 1.0 * min(pending_p, 60) + 5,
                "water": 1.25 * net["water_demand"] + 0.7 * min(pending_w, 60) + 5,
                "sewage": 1.2 * net["sewage_demand"] + 5}
        locked = set(obs.get("progress", {}).get("locked", []))
        own = {(p["x"], p["y"]) for p in net["plants"]}
        plants = [p for n in obs["networks"] for p in n["plants"]]
        for f in obs.get("disconnected_facilities", []):      # they crowd their neighbours too
            if "x" in f and "y" in f and f.get("kind") in C.TIER_BY_NAME:
                line, tier = C.TIER_BY_NAME[f["kind"]]
                plants.append({"x": f["x"], "y": f["y"], "line": line, "tier": tier})
        used = set()

        def best(line, left=float("inf")):
            """Highest unlocked 1x1 tier of a line we can afford (tiers above 1 keep a reserve)."""
            for t in reversed(C.TIER_LINES[line]):
                if (tuple(t["footprint"]) == (1, 1) and t.get("unlock") not in locked
                        and left >= t["cost"] + (self.RESERVE if t["tier"] > 1 else 0)):
                    return t
            return None

        def new_plant(res, t, cands):
            """(value, site) of the best candidate site for a new plant of tier t, or None."""
            top = None
            for s in cands:
                if (s["x"], s["y"]) in used:
                    continue
                cf, lost = crowding(plants, t["line"], s["x"], s["y"])
                wind = s.get("wind", self.WIND_UNKNOWN) if t["line"] == "wind" else 1.0
                out = t[res] * cf * wind - lost - self.SMOG * (t.get("pollution") or 0)
                v = out / max(1.0, t["cost"] + self.LIFE * t["upkeep"])
                if top is None or v > top[0]:
                    top = (v, s)
            return top

        # overhaul one ageing power plant a turn (output drops 5%/year after 5 years)
        for p in net["plants"]:
            if p.get("age_years", 0) >= 7 and p["line"] in ("coal", "gas", "solar", "wind"):
                cost = round(C.TIER_LINES[p["line"]][p["tier"] - 1]["cost"] * C.MODULES["overhaul"]["cost_share"])
                if budget >= cost + self.RESERVE:
                    acts.append({"type": "upgrade_facility", "x": p["x"], "y": p["y"], "module": "overhaul"})
                    budget -= cost
                    why.append(f"overhaul {p['kind']}")
                    break

        short = {"power": net["power_supply"] < need["power"], "water": net["water_supply"] < need["water"],
                 "sewage": net["water_supply"] > 0 and net["sewage_supply"] < need["sewage"]}
        asks = []           # (money, resource) for one message to the mayor
        upgraded = False
        for res in RESOURCES:
            if not short[res]:
                continue
            supply = net[f"{res}_supply"]
            state = f"{res} {supply:.0f}<{need[res]:.0f}"
            opts: list[Option] = []
            # one tier upgrade per turn: the offers were computed before this turn and may be stale after one
            if not upgraded:
                for u in obs.get("upgrades", []):
                    if u["gain"].get(res, 0) > 0 and (u["x"], u["y"]) in own:
                        cur, nxt = C.TIER_LINES[C.TIER_BY_NAME[u["to"]][0]][u["tier"] - 2:u["tier"]]
                        smog = (nxt.get("pollution") or 0) - (cur.get("pollution") or 0)
                        price = max(1.0, u["cost"] + self.LIFE * max(0, nxt["upkeep"] - cur["upkeep"]))
                        opts.append(Option((u["gain"][res] - self.SMOG * smog) / price, u["cost"] + self.RESERVE,
                                           u["cost"], {"type": "upgrade_facility", "x": u["x"], "y": u["y"],
                                                       "module": "tier"},
                                           f"{state}: upgrade {u['from']} -> {u['to']}"))
            if res == "power":
                builds = [("wind", sites.get("windy") or sites.get("near_homes", []), 0)]
                if c["population"] > 600:
                    builds.append(("coal", sites.get("far_from_homes") or sites.get("near_homes", []),
                                   self.COAL_MARGIN))
            elif res == "water":
                banks = [b for b in sites.get("riverbank_next_to_road", []) if b["network"] == net["id"]]
                builds = [("pump", banks, 0)] if banks else [("tower", sites.get("near_homes", []), 0)]
            else:
                banks = [b for b in sites.get("riverbank_for_sewage", []) if b["network"] == net["id"]]
                builds = [("outlet", banks, 0)]
            action = {"power": "build_power_plant", "water": "build_water", "sewage": "build_sewage"}[res]
            for line, cands, margin in builds:
                # the top unlocked tier, and the best one we can pay for now
                tiers = [best(line)]
                cheap = best(line, budget - margin)
                if cheap is not None and cheap is not tiers[0]:
                    tiers.append(cheap)
                for t in tiers:
                    o = new_plant(res, t, cands) if t else None
                    if o is None:
                        continue
                    v, site = o
                    if res == "sewage" and c["population"] > self.CLEAN_RIVER_POP:
                        v *= 1.0 - self.DISCHARGE_WEIGHT * t["discharge"]
                    money = t["cost"] + margin + (self.RESERVE if t["tier"] > 1 else 0)
                    opts.append(Option(v, money, t["cost"],
                                       {"type": action, "kind": t["name"], "x": site["x"], "y": site["y"]},
                                       f"{state}: {t['name']}"))
            opts = [o for o in opts if o.value > 0]
            if not opts:
                continue
            top = max(opts, key=lambda o: o.value)
            choice = top if top.money_needed <= budget else None
            if choice is None:
                asks.append((top.money_needed, res))
                # skip the cheaper option this turn unless something is actually short or it is nearly as good
                alt = max((o for o in opts if o.money_needed <= budget), key=lambda o: o.value, default=None)
                if alt and (supply < self.SHORTAGE * net[f"{res}_demand"] or alt.value >= self.GOOD_ENOUGH * top.value):
                    choice = alt
            if choice is None:
                continue
            acts.append(choice.action)
            budget -= choice.cost
            why.append(choice.note)
            if choice.action["type"] == "upgrade_facility":
                upgraded = True
            else:
                used.add((choice.action["x"], choice.action["y"]))
                if choice.action["kind"] in (t["name"] for t in C.TIER_LINES["tower"]):
                    msgs.append({"to": "planning",
                                 "text": "Please run a road to the riverbank so I can build cheap pumps."})
        if asks:
            money, res = max(asks)
            others = [r for _, r in asks if r != res]
            msgs.append({"to": "mayor", "text": f"Utilities needs ${money:,.0f} to add {res} (supply "
                                                f"{net[f'{res}_supply']:.0f}, need ~{need[res]:.0f})"
                                                + (f"; {' and '.join(others)} also short" if others else "")
                                                + ". Please fund."})
        return Decision(actions=acts, messages=msgs, thinking="; ".join(why) or "capacity OK")


# ============================================================ city services
class _ServiceAgent(Agent):
    """Shared helpers: build at the first suggested site, or ask the mayor for money."""
    kind = "scripted"
    UPGRADE_RESERVE = 500   # cash kept back after a tier upgrade

    def _build(self, obs, acts, msgs, why, action: dict, cost: float, sites: list, reason: str,
               spacing: int = 5) -> bool:
        key = {"build_jail": "jail", "build_hospital": "hospital", "build_library": "library",
               "build_incinerator": "incinerator"}.get(action["type"])
        if action["type"] == "build_school" and action.get("kind") in ("high", "university"):
            key = "high_school" if action["kind"] == "high" else "university"
        if key and key in obs.get("progress", {}).get("locked", []):
            return False
        budget = obs["your_budget"] or 0.0
        spent = sum(a.get("_cost", 0) for a in acts)
        own = {"build_police_station": "K", "build_fire_station": "F", "build_clinic": "A", "build_landfill": "L",
               "build_school": {"elementary": "N", "high": "B", "university": "Q"}.get(action.get("kind"), "N"),
               "build_library": "D", "build_cemetery": "Z", "build_hospital": "h", "build_jail": "J"}.get(action["type"])
        mine = MapView(obs["map"]).find(own) if own and "map" in obs else []
        site = next((s for s in sites if all((s["x"], s["y"]) != (a.get("x"), a.get("y")) for a in acts)
                     and all(abs(s["x"] - x) + abs(s["y"] - y) >= spacing for x, y in mine)), None)
        if site is None:
            return False
        if budget - spent < cost:
            if not any(m["to"] == "mayor" for m in msgs):
                msgs.append({"to": "mayor", "text": f"{obs['role'].title()} needs ${cost:,.0f} for a new building ({reason})."})
            return False
        acts.append({**action, "x": site["x"], "y": site["y"], "_cost": cost})
        why.append(reason)
        return True

    def _upgrade(self, obs, acts, why, ups, reason, min_need: float | None = 0.0, only=None, msgs=None) -> bool:
        """Upgrade the most needed facility one tier (optionally only those whose current tier is in `only`).

        Offers need more than min_need (None: any need). With msgs, ask the mayor for the money when the best offer is
        out of reach."""
        left = (obs["your_budget"] or 0.0) - sum(a.get("_cost", 0) for a in acts)
        busy = {(a.get("x"), a.get("y")) for a in acts}
        for u in ups or []:
            if ((only is None or u["from"] in only) and (min_need is None or u.get("need", 0.0) > min_need)
                    and (u["x"], u["y"]) not in busy):
                if left >= u["cost"] + self.UPGRADE_RESERVE:
                    acts.append({"type": "upgrade_facility", "x": u["x"], "y": u["y"], "module": "tier",
                                 "_cost": u["cost"]})
                    why.append(f"{reason}: {u['from']} -> {u['to']}")
                    return True
                if msgs is not None and not any(m["to"] == "mayor" for m in msgs):
                    msgs.append({"to": "mayor", "text": f"{obs['role'].title()} needs "
                                                        f"${u['cost'] + self.UPGRADE_RESERVE:,.0f} to "
                                                        f"upgrade a {u['from'].replace('_', ' ')} ({reason})."})
                return False
        return False

    @staticmethod
    def _done(acts, msgs, why, idle="coverage OK"):
        for a in acts:
            a.pop("_cost", None)
        return Decision(actions=acts, messages=msgs, thinking="; ".join(why) or idle)


class ScriptedTransport(_ServiceAgent):
    def act(self, obs: dict) -> Decision:
        d, c = obs["transport"], obs["city"]
        acts, msgs, why = [], [], []
        budget = obs["your_budget"] or 0.0
        locked = set(obs.get("progress", {}).get("locked", []))
        # widen the worst jammed street
        for run in d["congested_roads"][:2]:
            if "avenue" in locked or run["class"] != "street" or run["load_vs_capacity"] < 1.0:
                continue
            cost = C.ROAD_UPGRADE_COST[2] * run["tiles"]
            if cost <= budget:
                acts.append({"type": "upgrade_road", "to": "avenue", "x1": run["x1"], "y1": run["y1"],
                             "x2": run["x2"], "y2": run["y2"]})
                budget -= cost
                why.append(f"avenue on jammed stretch x{run['load_vs_capacity']}")
            else:
                msgs.append({"to": "mayor", "text": f"Transport needs ${cost:,.0f} to widen a jammed street."})
                break
        # a bus line once the city is big enough, another when buses are full
        full = all(l["riders"] >= 0.9 * l["capacity"] for l in d["bus_lines"])
        if c["population"] > 1500 and (not d["bus_lines"] or (full and len(d["bus_lines"]) < 4)):
            stops = d.get("suggested_stops", [])
            if len(stops) >= 2 and budget >= C.BUS_LINE_COST:
                acts.append({"type": "create_bus_line", "stops": stops})
                budget -= C.BUS_LINE_COST
                why.append("new bus line")
            elif len(stops) >= 2:
                msgs.append({"to": "mayor", "text": f"Transport needs ${C.BUS_LINE_COST:,} for a bus line."})
        fr = d.get("freight", {})
        if ("harbour" not in locked and not d.get("harbours") and d.get("harbour_sites")
                and fr.get("exports", 0) + fr.get("imports", 0) > 0.7 * max(1, fr.get("capacity", 1))):
            site = d["harbour_sites"][0]
            if budget >= C.HARBOUR_SPEC["cost"]:
                acts.append({"type": "build_harbour", "x": site["x"], "y": site["y"]})
                budget -= C.HARBOUR_SPEC["cost"]
                why.append("harbour for freight")
            else:
                msgs.append({"to": "mayor", "text": f"Transport needs ${C.HARBOUR_SPEC['cost']:,} for a harbour (freight is full)."})
        # region mode: link up with every neighbouring city (highway first, rail once trains unlock)
        for city, cs in d.get("connector_sites", {}).items():
            site = cs["site"]
            if site is None:
                continue
            for kind, pop in (("highway", 800), ("rail", 5000)):
                if kind in cs["built"] or c["population"] < pop or (kind == "rail" and "train_station" in locked):
                    continue
                cost = C.CONNECTOR[kind]["cost"] + 60 * site["feeder_tiles"]
                if budget >= cost:
                    acts.append({"type": "build_connector", "kind": kind, "to": city, "x": site["x"], "y": site["y"]})
                    budget -= cost
                    why.append(f"{kind} to {city}")
                else:
                    msgs.append({"to": "mayor", "text": f"Transport needs ${cost:,.0f} for a {kind} link to {city}."})
                break
        # metro once unlocked: two stations first, then a line between them
        if "metro" not in locked and not d.get("metro_lines"):
            stations = d.get("metro_stations", [])
            if len(stations) >= 2:
                pts = stations[:4]
                cost = C.METRO_TRACK_COST * sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(pts, pts[1:])) + 200
                if budget >= cost:
                    acts.append({"type": "create_metro_line", "stations": pts})
                    budget -= cost
                    why.append("metro line")
                else:
                    msgs.append({"to": "mayor", "text": f"Transport needs ${cost:,.0f} to open a metro line."})
            elif d.get("metro_station_sites") and budget >= C.METRO_STATION["cost"]:
                site = d["metro_station_sites"][0]
                acts.append({"type": "build_metro_station", "x": site["x"], "y": site["y"]})
                budget -= C.METRO_STATION["cost"]
                why.append("metro station")
            elif d.get("metro_station_sites"):
                msgs.append({"to": "mayor", "text": f"Transport needs ${C.METRO_STATION['cost']:,} per metro station."})
        if "train_station" not in locked and not d["train_stations"] and d.get("train_station_sites"):
            site = d["train_station_sites"][0]
            if budget >= C.TRAIN_STATION["cost"]:
                acts.append({"type": "build_train_station", "x": site["x"], "y": site["y"]})
                why.append("train station")
            else:
                msgs.append({"to": "mayor", "text": f"Transport needs ${C.TRAIN_STATION['cost']:,} for a train station."})
        return Decision(actions=acts, messages=msgs, thinking="; ".join(why) or "traffic OK")


class ScriptedPolice(_ServiceAgent):
    def act(self, obs: dict) -> Decision:
        c, d = obs["city"], obs["police"]
        acts, msgs, why = [], [], []
        S = C.SERVICES
        if d.get("downtown_uncovered", 0) > 0:
            self._upgrade(obs, acts, why, d.get("upgrades"), "downtown needs an HQ", only=("police_station",),
                          msgs=msgs)
        if c["population"] >= 150 and (c["services"]["police_coverage_pct"] or 0) < 85:
            self._build(obs, acts, msgs, why, {"type": "build_police_station"}, S["police_station"]["cost"],
                        d.get("candidate_sites", []), f"police cover {c['services']['police_coverage_pct']}%")
        if c["population"] > 1500 and d["inmates"] > 0.8 * max(1, d["jail_capacity"]):
            # an overflowing jail becomes a prison once they unlock (offered only when inmates exceed capacity)
            if not self._upgrade(obs, acts, why, d.get("upgrades"), "jail full", only=("jail",)):
                self._build(obs, acts, msgs, why, {"type": "build_jail"}, S["jail"]["cost"],
                            d.get("jail_sites", []), f"jail {d['inmates']}/{d['jail_capacity']}")
        return self._done(acts, msgs, why)


class ScriptedFire(_ServiceAgent):
    def act(self, obs: dict) -> Decision:
        c, d = obs["city"], obs["fire"]
        acts, msgs, why = [], [], []
        budget = obs["your_budget"] or 0.0
        for run in d.get("levee_lines", [])[:3]:
            cost = run["tiles"] * C.CATALOG["levee"]["cost"]
            if cost <= budget:
                acts.append({"type": "build_levee", "x1": run["x1"], "y1": run["y1"], "x2": run["x2"], "y2": run["y2"]})
                budget -= cost
                why.append(f"levee before the flood ({run['tiles']} tiles)")
            else:
                msgs.append({"to": "mayor", "text": f"FLOOD WARNING: Fire needs ${cost:,.0f} for levees now."})
                break
        if c["population"] > 3000 and d.get("shelter_sites") and not d["shelters"]:
            self._build(obs, acts, msgs, why, {"type": "build_shelter"}, C.SERVICES["shelter"]["cost"],
                        d["shelter_sites"], "emergency shelter")
        if d.get("downtown_uncovered", 0) > 0:
            self._upgrade(obs, acts, why, d.get("upgrades"), "downtown needs a fire HQ", only=("fire_station",),
                          msgs=msgs)
        if c["population"] >= 150 and ((c["services"]["fire_coverage_pct"] or 0) < 85 or d["uncovered_buildings"] > 40):
            self._build(obs, acts, msgs, why, {"type": "build_fire_station"}, C.SERVICES["fire_station"]["cost"],
                        d.get("candidate_sites", []), f"fire cover {c['services']['fire_coverage_pct']}%")
        return self._done(acts, msgs, why)


class ScriptedHealth(_ServiceAgent):
    RECYCLE_FILL = 60        # a landfill more than this % full becomes a recycling centre
    COLLECTION_SHORT = 0.9   # collecting less than this share of the garbage: busy landfills become recycling centres

    def act(self, obs: dict) -> Decision:
        c, d = obs["city"], obs["health"]
        sv = c["services"]
        acts, msgs, why = [], [], []
        S = C.SERVICES
        pop = c["population"]
        ups = d.get("upgrades", {})
        if d.get("downtown_uncovered", 0) > 0:
            # a clinic grows into a hospital, a hospital into a medical centre: both give downtown full cover
            self._upgrade(obs, acts, why, ups.get("care"), "downtown care", msgs=msgs)
        full = all(l.get("fill_pct", 0) > 85 for l in d["landfills"])
        # a filling landfill becomes a recycling centre (never fills, twice the collection) instead of a new landfill;
        # so does one with busy trucks (the observation ranks those at GARBAGE_BUSY_RANK + load %) when collection
        # falls short
        fill = {(l["x"], l["y"]): l.get("fill_pct", 0.0) for l in d["landfills"]}
        collect_short = ((sv["garbage_collection_pct"] or 0) >= 80
                         and d["garbage_collected"] < self.COLLECTION_SHORT * d["garbage_produced"])
        offers = [u for u in ups.get("garbage") or []
                  if fill.get((u["x"], u["y"]), 0.0) > self.RECYCLE_FILL
                  or (collect_short and u.get("need", 0.0) >= GARBAGE_BUSY_RANK)]
        recycled = self._upgrade(obs, acts, why, offers, "landfill filling up", min_need=None, msgs=msgs)
        if not recycled and pop >= 100 and ((sv["garbage_collection_pct"] or 0) < 80 or full or
                                            d["garbage_collected"] < 0.8 * d["garbage_produced"]):
            self._build(obs, acts, msgs, why, {"type": "build_landfill"}, S["landfill"]["cost"],
                        d.get("landfill_sites", []) or d.get("clinic_sites", []),
                        f"garbage collection {sv['garbage_collection_pct']}%")
        if pop >= 400 and (sv["health_coverage_pct"] or 0) < 80:
            self._build(obs, acts, msgs, why, {"type": "build_clinic"}, S["clinic"]["cost"],
                        d.get("clinic_sites", []), f"clinic cover {sv['health_coverage_pct']}%")
        # crematoria need no plots (and a cemetery upgraded to one leaves the cemetery list)
        low_plots = not d.get("crematoria") and sum(cm.get("plots_left", 0) for cm in d["cemeteries"]) < 300
        short = d["deathcare_capacity"] < 1.3 * d["deaths_per_month"] + 1 or d["deceased_waiting"] > 10
        # a cemetery running out of plots (or too slow) becomes a crematorium
        cremated = (low_plots or short) and self._upgrade(obs, acts, why, ups.get("deathcare"), "cemetery full",
                                                          min_need=None)
        if not cremated and pop >= 300 and (short or low_plots):
            self._build(obs, acts, msgs, why, {"type": "build_cemetery"}, S["cemetery"]["cost"],
                        d.get("landfill_sites", []) or d.get("clinic_sites", []), "deathcare")
        if pop >= 5000 and not d["hospitals"]:
            if not self._upgrade(obs, acts, why, ups.get("care"), "first hospital", min_need=None, only=("clinic",)):
                self._build(obs, acts, msgs, why, {"type": "build_hospital"}, S["hospital"]["cost"],
                            d.get("hospital_sites", []), "first hospital")
        return self._done(acts, msgs, why)


class ScriptedEducation(_ServiceAgent):
    SEATS_SHORT = 85   # homes in school range (%) above which low enrolment means too few seats, not too few schools

    def act(self, obs: dict) -> Decision:
        d = obs["education"]
        acts, msgs, why = [], [], []
        S = C.SERVICES
        e = d["enrollment_pct"]
        ups = d.get("upgrades")
        # low enrolment with homes in range means too few seats: grow an overloaded school instead of adding one
        el, hi = d["elementary"], d["high"]
        if el["students"] > 30 and (e["elementary"] or 0) < 90:
            if not ((el.get("homes_in_range_pct") or 0) >= self.SEATS_SHORT
                    and self._upgrade(obs, acts, why, ups, "more seats", only=("elementary_school",))):
                self._build(obs, acts, msgs, why, {"type": "build_school", "kind": "elementary"},
                            S["elementary_school"]["cost"], d.get("school_sites", []),
                            f"elementary enrolment {e['elementary']}%")
        if hi["students"] > 80 and (e["high"] or 0) < 85:
            if not ((hi.get("homes_in_range_pct") or 0) >= self.SEATS_SHORT
                    and self._upgrade(obs, acts, why, ups, "more seats", only=("high_school",))):
                self._build(obs, acts, msgs, why, {"type": "build_school", "kind": "high"},
                            S["high_school"]["cost"], d.get("high_school_sites", []),
                            f"high school enrolment {e['high']}%")
        if obs["city"]["population"] > 6000 and d["university"]["seats"] == 0:
            self._build(obs, acts, msgs, why, {"type": "build_school", "kind": "university"},
                        S["university"]["cost"], d.get("university_sites", []), "university")
        if obs["city"]["population"] > 2500 and not d["libraries"]:
            self._build(obs, acts, msgs, why, {"type": "build_library"}, S["library"]["cost"],
                        d.get("school_sites", []), "library")
        return self._done(acts, msgs, why, "enrolment OK")


SCRIPTED = {"mayor": ScriptedMayor, "planning": ScriptedPlanner, "utilities": ScriptedUtilities,
            "transport": ScriptedTransport,
            "police": ScriptedPolice, "fire": ScriptedFire, "health": ScriptedHealth, "education": ScriptedEducation}
