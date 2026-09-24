"""Rule-based baseline agents. They only read the observation (no privileged
access to the World), so they can also be run as remote agents and used to
generate demonstration data for fine-tuning LLMs."""
from __future__ import annotations

import numpy as np

from ..sim import catalog as C
from .base import Agent
from ..core.decision import Decision
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
class ScriptedUtilities(Agent):
    kind = "scripted"

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
        need_p = 1.25 * net["power_demand"] + 1.0 * min(pending_p, 60) + 5
        need_w = 1.25 * net["water_demand"] + 0.7 * min(pending_w, 60) + 5
        used = set()

        def pick(lst):
            for s in lst:
                if (s["x"], s["y"]) not in used:
                    used.add((s["x"], s["y"]))
                    return s
            return None

        # overhaul one ageing power plant a turn (output drops 5%/year after 5 years)
        for p in net["plants"]:
            if p.get("age_years", 0) >= 7 and p["kind"] in ("coal", "gas", "solar", "wind"):
                cost = round(C.CATALOG[p["kind"]]["cost"] * C.MODULES["overhaul"]["cost_share"])
                if budget >= cost + 1000:
                    acts.append({"type": "upgrade_facility", "x": p["x"], "y": p["y"], "module": "overhaul"})
                    budget -= cost
                    why.append(f"overhaul {p['kind']}")
                    break
        if net["power_supply"] < need_p:
            big = c["population"] > 600 and budget >= C.CATALOG["coal"]["cost"] + 500
            kind = "coal" if big else "wind"
            s = pick(sites.get("far_from_homes" if kind == "coal" else "windy", []) or sites.get("near_homes", []))
            if s and budget >= C.CATALOG[kind]["cost"]:
                acts.append({"type": "build_power_plant", "kind": kind, "x": s["x"], "y": s["y"]})
                budget -= C.CATALOG[kind]["cost"]
                why.append(f"power {net['power_supply']}<{need_p:.0f}: {kind}")
            else:
                msgs.append({"to": "mayor", "text": f"Utilities needs ${C.CATALOG['wind']['cost']:,}+ to add power "
                                                    f"(supply {net['power_supply']}, need ~{need_p:.0f}). Please fund."})
        if net["water_supply"] < need_w:
            banks = [b for b in sites.get("riverbank_next_to_road", []) if b["network"] == net["id"]]
            s = pick(banks)
            if s and budget >= C.CATALOG["pump"]["cost"]:
                acts.append({"type": "build_water", "kind": "pump", "x": s["x"], "y": s["y"]})
                budget -= C.CATALOG["pump"]["cost"]
                why.append("pump on riverbank")
            elif not banks and budget >= C.CATALOG["tower"]["cost"]:
                s = pick(sites.get("near_homes", []))
                if s:
                    acts.append({"type": "build_water", "kind": "tower", "x": s["x"], "y": s["y"]})
                    budget -= C.CATALOG["tower"]["cost"]
                    why.append("no riverbank road: water tower")
                msgs.append({"to": "planning", "text": "Please run a road to the riverbank so I can build cheap pumps."})
            else:
                msgs.append({"to": "mayor", "text": "Utilities needs ~$1,500 for a water pump."})

        need_s = 1.2 * net["sewage_demand"] + 5
        if net["water_supply"] > 0 and net["sewage_supply"] < need_s:
            locked = set(obs.get("progress", {}).get("locked", []))
            kind = "treatment" if (c["population"] > 3000 and "treatment_plant" not in locked
                                   and budget >= C.CATALOG["treatment"]["cost"] + 1000) else "outlet"
            banks = [b for b in sites.get("riverbank_for_sewage", []) if b["network"] == net["id"]]
            s = pick(banks)
            if s and budget >= C.CATALOG[kind]["cost"]:
                acts.append({"type": "build_sewage", "kind": kind, "x": s["x"], "y": s["y"]})
                budget -= C.CATALOG[kind]["cost"]
                why.append(f"sewage {net['sewage_supply']}<{need_s:.0f}: {kind}")
            elif s:
                msgs.append({"to": "mayor", "text": f"Utilities needs ${C.CATALOG[kind]['cost']:,} for sewage capacity."})
        return Decision(actions=acts, messages=msgs, thinking="; ".join(why) or "capacity OK")


# ============================================================ city services
class _ServiceAgent(Agent):
    """Shared helpers: build at the first suggested site, or ask the mayor for money."""
    kind = "scripted"

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
               "build_library": "D", "build_cemetery": "Z", "build_hospital": "M", "build_jail": "J"}.get(action["type"])
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
        if c["population"] >= 150 and (c["services"]["police_coverage_pct"] or 0) < 85:
            self._build(obs, acts, msgs, why, {"type": "build_police_station"}, S["police_station"]["cost"],
                        d.get("candidate_sites", []), f"police cover {c['services']['police_coverage_pct']}%")
        if c["population"] > 1500 and d["inmates"] > 0.8 * max(1, d["jail_capacity"]):
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
        if c["population"] >= 150 and ((c["services"]["fire_coverage_pct"] or 0) < 85 or d["uncovered_buildings"] > 40):
            self._build(obs, acts, msgs, why, {"type": "build_fire_station"}, C.SERVICES["fire_station"]["cost"],
                        d.get("candidate_sites", []), f"fire cover {c['services']['fire_coverage_pct']}%")
        return self._done(acts, msgs, why)


class ScriptedHealth(_ServiceAgent):
    def act(self, obs: dict) -> Decision:
        c, d = obs["city"], obs["health"]
        sv = c["services"]
        acts, msgs, why = [], [], []
        S = C.SERVICES
        pop = c["population"]
        full = all(l.get("fill_pct", 0) > 85 for l in d["landfills"])
        if pop >= 100 and ((sv["garbage_collection_pct"] or 0) < 80 or full or
                           d["garbage_collected"] < 0.8 * d["garbage_produced"]):
            self._build(obs, acts, msgs, why, {"type": "build_landfill"}, S["landfill"]["cost"],
                        d.get("landfill_sites", []) or d.get("clinic_sites", []), f"garbage collection {sv['garbage_collection_pct']}%")
        if pop >= 400 and (sv["health_coverage_pct"] or 0) < 80:
            self._build(obs, acts, msgs, why, {"type": "build_clinic"}, S["clinic"]["cost"],
                        d.get("clinic_sites", []), f"clinic cover {sv['health_coverage_pct']}%")
        low_plots = sum(cm.get("plots_left", 0) for cm in d["cemeteries"]) < 300
        if pop >= 300 and (d["deathcare_capacity"] < 1.3 * d["deaths_per_month"] + 1 or d["deceased_waiting"] > 10
                           or low_plots):
            self._build(obs, acts, msgs, why, {"type": "build_cemetery"}, S["cemetery"]["cost"],
                        d.get("landfill_sites", []) or d.get("clinic_sites", []), "deathcare")
        if pop >= 5000 and not d["hospitals"]:
            self._build(obs, acts, msgs, why, {"type": "build_hospital"}, S["hospital"]["cost"],
                        d.get("hospital_sites", []), "first hospital")
        return self._done(acts, msgs, why)


class ScriptedEducation(_ServiceAgent):
    def act(self, obs: dict) -> Decision:
        d = obs["education"]
        acts, msgs, why = [], [], []
        S = C.SERVICES
        e = d["enrollment_pct"]
        if d["elementary"]["students"] > 30 and (e["elementary"] or 0) < 90:
            self._build(obs, acts, msgs, why, {"type": "build_school", "kind": "elementary"},
                        S["elementary_school"]["cost"], d.get("school_sites", []), f"elementary enrolment {e['elementary']}%")
        if d["high"]["students"] > 80 and (e["high"] or 0) < 85:
            self._build(obs, acts, msgs, why, {"type": "build_school", "kind": "high"},
                        S["high_school"]["cost"], d.get("high_school_sites", []), f"high school enrolment {e['high']}%")
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
