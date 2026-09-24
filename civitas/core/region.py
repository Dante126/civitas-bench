"""A region of 2-6 cities, each run by its own team of agents.

Cities sit on a small grid (north, south, east, west, southeast, southwest).
North and south share the big river: the north city is upstream, so its
sewage and runoff arrive in the south city's water. Cities trade with each
other once they are linked:

  highway / rail  neighbouring cities; each builds a connector on the map edge that faces the other
  boat            any two cities with a harbour
  air             any two cities with an airport

Every month, over the capacity of those links, surplus goods and raw
materials go to cities that are short (at better prices than the outside
region), unemployed workers commute to open jobs, and residents visit the
casinos, landmarks and historic districts of their partners. So cities can
specialise: a mining town sells raw materials to a high-tech city, a
low-education city lives off gamblers from its neighbours (and their crime).

All teams share one council board (messages to "all", to a team - "south" -
or to one agent such as "south.mayor"), and mayors can agree power/water
deals. Roles are named "<city>.<role>". `RegionEnv` has the same interface as
`CivitasEnv`, so the runner, logging and viewer work unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from itertools import combinations

import numpy as np

from ..bench.scoring import ScoringConfig
from ..sim import catalog as C
from .council import Council
from .decision import Decision
from .env import CivitasEnv, GameConfig

LAYOUT = {"north": (0, 0), "south": (0, 1), "east": (1, 0), "west": (-1, 0), "southeast": (1, 1), "southwest": (-1, 1)}
SIDE = {(0, 1): "south", (0, -1): "north", (1, 0): "east", (-1, 0): "west"}
DEAL_ACTIONS = ("offer_deal", "accept_deal", "cancel_deal")


def split_role(role: str) -> tuple[str, str]:
    city, _, base = role.partition(".")
    return city, base


class RegionEnv:
    def __init__(self, game: GameConfig | None = None, scoring: ScoringConfig | None = None):
        self.cfg = game or GameConfig()
        self.scoring = scoring or ScoringConfig(start_funds=self.cfg.start_funds)
        cities = list(getattr(self.cfg, "cities", None) or ["north", "south"])
        bad = [c for c in cities if c not in LAYOUT]
        if bad or len(cities) < 2 or len(set(cities)) != len(cities):
            raise ValueError(f"cities must be 2-6 distinct names from {list(LAYOUT)}; got {cities}")
        self.cities = cities
        self.reset(self.cfg.seed)

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.cfg.seed = seed
        self.council = Council()
        self.envs: dict[str, CivitasEnv] = {}
        # the river column (x = 0) is one river flowing north -> south through its cities
        river = sorted([c for c in self.cities if LAYOUT[c][0] == 0], key=lambda c: LAYOUT[c][1])
        self.river_chain = river
        prev_exit = None
        for k, c in enumerate(sorted(self.cities, key=lambda c: (LAYOUT[c][0] != 0, LAYOUT[c][1], LAYOUT[c][0]))):
            start = prev_exit if c in river and river.index(c) > 0 else None
            env = CivitasEnv(replace(self.cfg, seed=self.cfg.seed + 1000 * self.cities.index(c)), self.scoring,
                             council=self.council, city=c, river_start=start)
            self.envs[c] = env
            if c in river:
                prev_exit = env.world.river_path[-1]
        self.envs = {c: self.envs[c] for c in self.cities}
        for c, env in self.envs.items():
            env.world.region_info = {"city": c, "neighbors": self.neighbors(c)}
        base_roles = self.envs[self.cities[0]].roles
        self.roles = [f"{c}.{r}" for r in base_roles for c in self.cities]    # interleaved: all mayors first, ...
        self.deals: list[dict] = []
        self.links: list[dict] = []
        self.flows: list[dict] = []
        self.done = False
        self.last_scores = {"city": 0.0}
        return {r: self.observe(r) for r in self.roles}

    # ---------------------------------------------------------- helpers
    @property
    def world(self):
        """The first city's world (for code that expects a single city)."""
        return self.envs[self.cities[0]].world

    @property
    def month(self) -> int:
        return self.world.month

    def acts_this_month(self, role: str) -> bool:
        return self.envs[split_role(role)[0]].acts_this_month(split_role(role)[1])

    def neighbors(self, city: str) -> dict[str, str]:
        """Cities next to this one on the region grid -> the map edge that faces them."""
        x, y = LAYOUT[city]
        out = {}
        for c in self.cities:
            d = (LAYOUT[c][0] - x, LAYOUT[c][1] - y)
            if d in SIDE:
                out[c] = SIDE[d]
        return out

    def others(self, city: str) -> list[str]:
        return [c for c in self.cities if c != city]

    # ---------------------------------------------------------------- links
    def compute_links(self) -> list[dict]:
        links = []
        w = {c: e.world for c, e in self.envs.items()}
        for a, b in combinations(self.cities, 2):
            if b in self.neighbors(a):
                for kind in C.CONNECTOR:
                    if any(x["to"] == b and x["kind"] == kind for x in w[a].connectors) and \
                            any(x["to"] == a and x["kind"] == kind for x in w[b].connectors):
                        links.append({"a": a, "b": b, "kind": kind, **C.LINK_CAPACITY[kind]})
            if w[a].facilities(C.HARBOUR) and w[b].facilities(C.HARBOUR):
                links.append({"a": a, "b": b, "kind": "boat", **C.LINK_CAPACITY["boat"]})
            if w[a].n_buildings(C.AIRPORT) and w[b].n_buildings(C.AIRPORT):
                links.append({"a": a, "b": b, "kind": "air", **C.LINK_CAPACITY["air"]})
        return links

    def _capacity(self, links: list[dict]) -> dict:
        cap = {}
        for l in links:
            key = tuple(sorted((l["a"], l["b"])))
            g, p = cap.get(key, (0.0, 0.0))
            cap[key] = (g + l["goods"] * self._link_eff(l), p + l["people"] * self._link_eff(l))
        return {k: {"goods": g, "people": p} for k, (g, p) in cap.items()}

    def _link_eff(self, link: dict) -> float:
        """Links run at the lower of the two transport departments' funding levels."""
        return min(self.envs[link["a"]].world.funding_eff("transport"), self.envs[link["b"]].world.funding_eff("transport"))

    # ---------------------------------------------------------------- flows
    def _run_flows(self) -> None:
        """Trade, commuting and tourism between linked cities (from last month's economies)."""
        self.links = links = self.compute_links()
        cap = self._capacity(links)
        W = {c: e.world for c, e in self.envs.items()}
        for w in W.values():
            w.regional = dict(C.REGIONAL_ZERO)
        flows = {k: {"a": k[0], "b": k[1], "goods": 0.0, "raw": 0.0, "commuters": 0.0, "tourists": 0.0} for k in cap}
        # goods and raw materials: sellers with a surplus -> buyers with a shortfall
        for com in ("goods", "raw"):
            surplus = {c: max(0.0, w.trade.get(f"{com}_made", 0.0) - w.trade.get(f"{com}_needed", 0.0)) for c, w in W.items()}
            deficit = {c: max(0.0, w.trade.get(f"{com}_needed", 0.0) - w.trade.get(f"{com}_made", 0.0)) for c, w in W.items()}
            for key in sorted(cap, key=lambda k: -cap[k]["goods"]):
                for s, b in (key, key[::-1]):
                    q = min(surplus[s], deficit[b], cap[key]["goods"])
                    if q <= 0:
                        continue
                    surplus[s] -= q
                    deficit[b] -= q
                    cap[key]["goods"] -= q
                    W[s].regional[f"{com}_out"] += q
                    W[b].regional[f"{com}_in"] += q
                    W[s].regional["cash"] += q * C.INTERCITY_PRICE[com] * C.INTERCITY_CITY_SHARE
                    flows[key][com] += q
        # commuting: unemployed workers take open jobs in a linked city
        for key in sorted(cap, key=lambda k: -cap[k]["people"]):
            for home, work in (key, key[::-1]):
                free = W[work].stats.get("unfilled_jobs", 0.0) * C.COMMUTE_SHARE - W[work].regional["commuters_in"]
                idle = W[home].stats.get("unemployed", 0.0) * C.COMMUTE_SHARE - W[home].regional["commuters_out"]
                q = min(free, idle, cap[key]["people"])
                if q <= 0:
                    continue
                cap[key]["people"] -= q
                W[home].regional["commuters_out"] += q
                W[work].regional["commuters_in"] += q
                flows[key]["commuters"] += q
        # tourism: residents visit the attractions of linked cities
        attraction = {c: max(0.0, w.stats.get("tourists", 0.0) - w.stats.get("regional_tourists_last", 0.0))
                      for c, w in W.items()}
        for key in cap:
            for home, dest in (key, key[::-1]):
                if attraction[dest] <= 0:
                    continue
                partners = [k[0] if k[1] == home else k[1] for k in cap if home in k]
                share = attraction[dest] / (sum(attraction[p] for p in partners) + 500.0)
                q = min(W[home].stats.get("population", 0.0) * C.TOURIST_SHARE * share, cap[key]["people"])
                if q <= 0:
                    continue
                cap[key]["people"] -= q
                W[dest].regional["tourists"] += q
                flows[key]["tourists"] += q
        for w in W.values():
            w.stats["regional_tourists_last"] = w.regional["tourists"]
        self.flows = [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in f.items()} for f in flows.values()]

    # ------------------------------------------------------------- observe
    def observe(self, role: str) -> dict:
        city, base = split_role(role)
        env = self.envs[city]
        obs = env.observe(base)
        w = env.world
        obs["city_name"] = city
        obs["region"] = self._region_view(city)
        obs["neighbour"] = {"river_inflow_pollution_pct": round(100 * (1 - np.exp(-w.upstream_inflow / C.RIVER_NORM)), 1),
                            "river_outflow_pollution_pct": round(100 * float(w.river_row[-1]), 1)}
        obs["text"] = self._region_header(city) + obs["text"] + self._region_footer(city, base, obs)
        return obs

    def _region_view(self, city: str) -> dict:
        nb = self.neighbors(city)
        mine = self.envs[city].world.connectors
        out = []
        for c in self.others(city):
            w = self.envs[c].world
            spec = w.specializations() if w.history else {}
            links = sorted({l["kind"] for l in self.links if {l["a"], l["b"]} == {city, c}})
            pend_me = sorted({x["kind"] for x in mine if x["to"] == c} - set(links))
            pend_them = sorted({x["kind"] for x in w.connectors if x["to"] == city} - set(links))
            fl = next((f for f in self.flows if {f["a"], f["b"]} == {city, c}), None)
            view = {"city": c, "population": int(w.stats.get("population", 0)),
                    "direction": nb.get(c, "not adjacent (boat or air only)"),
                    "strong_in": [k for k, v in spec.items() if v >= 0.5],
                    "links": links, "you_built_waiting_for_them": pend_me, "they_built_waiting_for_you": pend_them,
                    "surplus": {k: round(max(0.0, w.trade.get(f"{k}_made", 0) - w.trade.get(f"{k}_needed", 0)))
                                for k in ("goods", "raw")},
                    "shortfall": {k: round(max(0.0, w.trade.get(f"{k}_needed", 0) - w.trade.get(f"{k}_made", 0)))
                                  for k in ("goods", "raw")},
                    "unemployed": round(w.stats.get("unemployed", 0)), "open_jobs": round(w.stats.get("unfilled_jobs", 0))}
            if fl:
                sign = 1 if fl["a"] == city else -1
                view["last_month"] = {k: fl[k] for k in ("goods", "raw", "commuters", "tourists")}
                view["last_month"]["direction"] = "both ways (totals)"
            if city in self.river_chain and c in self.river_chain:
                view["river"] = "downstream of you" if self.river_chain.index(c) > self.river_chain.index(city) else "upstream of you"
            out.append(view)
        return {"cities": out, "my_specialisations": self.envs[city].world.specializations()}

    def _deal_view(self, d: dict, city: str) -> dict:
        role = "selling" if d["seller"] == city else "buying"
        partner = d["buyer"] if role == "selling" else d["seller"]
        return {"id": d["id"], "you_are": role, "partner": partner, "resource": d["resource"], "amount": d["amount"],
                "price": d["price"], "months_left": d["months_left"], "status": d["status"], "proposed_by": d["proposer"]}

    def _region_header(self, city: str) -> str:
        others = ", ".join(f"{c.upper()} ({self.neighbors(city).get(c, 'far')})" for c in self.others(city))
        river = ""
        if city in self.river_chain:
            i = self.river_chain.index(city)
            up = self.river_chain[:i]
            down = self.river_chain[i + 1:]
            river = (f" The river arrives from {', '.join(up)}" if up else "") + \
                    (f"; your sewage and runoff flow on to {', '.join(down)}" if down else "") + "."
        return (f"### REGION MODE: you work for the {city.upper()} city. Other cities: {others}.{river} Message your own "
                f"team with \"to\": \"mayor\" etc., another team with \"to\": \"<city>\" or \"<city>.mayor\", "
                "everyone with \"all\".\n\n")

    def _region_footer(self, city: str, base: str, obs: dict) -> str:
        r = obs["region"]
        L = ["", "## Region"]
        spec = r["my_specialisations"]
        L.append("Your specialisations (0-1): " + ", ".join(f"{k} {v}" for k, v in spec.items()))
        for v in r["cities"]:
            s = (f"- {v['city'].upper()} ({v['direction']}): pop {v['population']}, strong in "
                 f"{', '.join(v['strong_in']) or 'nothing yet'}; surplus goods {v['surplus']['goods']} raw {v['surplus']['raw']}, "
                 f"short goods {v['shortfall']['goods']} raw {v['shortfall']['raw']}; unemployed {v['unemployed']}, "
                 f"open jobs {v['open_jobs']}. Links: {', '.join(v['links']) or 'none'}")
            if v["you_built_waiting_for_them"]:
                s += f" (you built {', '.join(v['you_built_waiting_for_them'])}; waiting for them)"
            if v["they_built_waiting_for_you"]:
                s += f" (they built {', '.join(v['they_built_waiting_for_you'])}; build your side to open it)"
            if "last_month" in v:
                f = v["last_month"]
                s += (f". Last month between you: goods {f['goods']}, raw {f['raw']}, commuters {f['commuters']}, "
                      f"tourists {f['tourists']}")
            if "river" in v:
                s += f". River: {v['river']}"
            L.append(s)
        n = obs["neighbour"]
        L.append(f"River pollution arriving from upstream: {n['river_inflow_pollution_pct']}%; leaving downstream: "
                 f"{n['river_outflow_pollution_pct']}%")
        deals = [self._deal_view(d, city) for d in self.deals if d["status"] != "ended" and city in (d["seller"], d["buyer"])]
        if deals:
            L.append("Deals: " + "; ".join(f"#{d['id']} {d['status']}: you are {d['you_are']} {d['amount']} {d['resource']} "
                                           f"with {d['partner']} at ${d['price']}/unit/month, {d['months_left']} months left "
                                           f"(proposed by {d['proposed_by']})" for d in deals))
        L.append("Links carry trade, commuters and tourists: highway/rail to neighbours (transport: build_connector on the "
                 "facing map edge, both cities must build their side), boats between harbour cities, planes between "
                 "airport cities. Specialise: mining/farming needs resources, high tech needs graduates, casinos need "
                 "little education but bring crime.")
        if base == "mayor":
            L.append('Neighbour deals (mayor only): {"type":"offer_deal","to":"<city>","resource":"power","amount":100,'
                     '"price":0.8,"months":24,"sell":true} offers to sell (or buy with "sell":false) capacity; '
                     '{"type":"accept_deal","deal":1} accepts an offer; {"type":"cancel_deal","deal":1} ends a deal.')
        return "\n".join(L)

    # --------------------------------------------------------------- act
    def apply(self, role: str, decision: Decision) -> list[dict]:
        city, base = split_role(role)
        region_acts = [a for a in decision.actions if isinstance(a, dict) and a.get("type") in DEAL_ACTIONS]
        other = [a for a in decision.actions if not (isinstance(a, dict) and a.get("type") in DEAL_ACTIONS)]
        msgs = []
        for m in decision.messages:              # "mayor" means my mayor; "south" means their whole team
            to = str(m.get("to", "all"))
            if to != "all" and to not in self.cities and "." not in to:
                to = f"{city}.{to}"
            msgs.append({**m, "to": to})
        results = self.envs[city].apply(base, replace(decision, actions=other, messages=msgs))
        for a in region_acts:
            results.append(self._deal_action(city, base, a))
        env = self.envs[city]
        if region_acts and env.turns:
            env.turns[-1]["results"] = results
            env.turns[-1]["invalid"] = sum(1 for r in results if not r["ok"])
        return results

    def _deal_action(self, city: str, base: str, a: dict) -> dict:
        def res(ok, msg):
            return {"action": a, "ok": ok, "message": msg, "cost": 0.0}
        if base != "mayor":
            return res(False, "only mayors can make neighbour deals")
        kind = a["type"]
        try:
            if kind == "offer_deal":
                others = self.others(city)
                to = str(a.get("to", others[0] if len(others) == 1 else "")).lower()
                if to not in others:
                    return res(False, f"'to' must be one of {others}")
                resource = str(a.get("resource", "")).lower()
                if resource not in ("power", "water"):
                    return res(False, "resource must be 'power' or 'water'")
                amount, price = float(a.get("amount", 0)), float(a.get("price", 0))
                months = int(a.get("months", 24))
                if not (0 < amount <= 2000 and 0 <= price <= 10 and 1 <= months <= 120):
                    return res(False, "amount 1-2000 units, price $0-10 per unit per month, months 1-120")
                sell = bool(a.get("sell", True))
                deal = {"id": len(self.deals) + 1, "seller": city if sell else to, "buyer": to if sell else city,
                        "resource": resource, "amount": amount, "price": price, "months_left": months,
                        "status": "offered", "proposer": city}
                self.deals.append(deal)
                self.council.post(self.month, f"{city}.mayor",
                                  f"DEAL #{deal['id']} proposed: {deal['seller']} sells {amount:.0f} {resource} to "
                                  f"{deal['buyer']} at ${price}/unit/month for {months} months", to)
                return res(True, f"proposed deal #{deal['id']}; the {to} mayor must accept it")
            deal = next((d for d in self.deals if d["id"] == int(a.get("deal", -1))), None)
            if deal is None:
                return res(False, f"no deal #{a.get('deal')}")
            if kind == "accept_deal":
                if deal["status"] != "offered" or deal["proposer"] == city or city not in (deal["seller"], deal["buyer"]):
                    return res(False, "you can only accept an open offer made to your city")
                deal["status"] = "active"
                return res(True, f"deal #{deal['id']} is active")
            if city not in (deal["seller"], deal["buyer"]) or deal["status"] == "ended":
                return res(False, "that deal is not yours or already ended")
            deal["status"] = "ended"
            return res(True, f"deal #{deal['id']} cancelled")
        except (TypeError, ValueError) as e:
            return res(False, f"bad deal parameters: {e}")

    # ------------------------------------------------------------ advance
    def _run_deals(self) -> None:
        for w in (e.world for e in self.envs.values()):
            w.external = {"power": 0.0, "water": 0.0}
            w.deal_cash = 0.0
        for d in self.deals:
            if d["status"] != "active":
                continue
            buyer, seller = self.envs[d["buyer"]].world, self.envs[d["seller"]].world
            buyer.external[d["resource"]] += d["amount"]
            seller.external[d["resource"]] -= d["amount"]
            cash = d["amount"] * d["price"]
            buyer.deal_cash -= cash
            seller.deal_cash += cash
            d["months_left"] -= 1
            if d["months_left"] <= 0:
                d["status"] = "ended"

    def end_month(self) -> dict:
        for up, down in zip(self.river_chain, self.river_chain[1:]):    # last month's outflow reaches the next city
            self.envs[down].world.upstream_inflow = self.envs[up].world.river_outflow
        self._run_deals()
        self._run_flows()
        infos = {c: e.end_month() for c, e in self.envs.items()}
        scores, rewards = {}, {}
        for c, info in infos.items():
            for k, v in info["scores"].items():
                scores[f"{c}.{k}"] = v
            for r, v in info["rewards"].items():
                rewards[f"{c}.{r}"] = v
        scores["city"] = round(float(np.mean([infos[c]["scores"]["city"] for c in self.cities])), 3)
        self.last_scores = scores
        terminated = any(i["terminated"] for i in infos.values())
        truncated = all(i["truncated"] for i in infos.values())
        self.done = terminated or truncated
        turns = []
        for c, info in infos.items():
            for t in info["turns"]:
                turns.append({**t, "role": f"{c}.{t['role']}"})
        metrics = self._region_metrics(infos)
        return {"metrics": metrics, "scores": scores, "rewards": rewards, "terminated": terminated,
                "truncated": truncated, "turns": turns, "cities": infos}

    def _region_metrics(self, infos: dict) -> dict:
        m = {c: infos[c]["metrics"] for c in self.cities}
        tot = lambda k: round(sum(f[k] for f in self.flows), 1)
        return {"month": m[self.cities[0]]["month"], "population": sum(x["population"] for x in m.values()),
                "treasury": sum(x["treasury"] for x in m.values()),
                "happiness": float(np.mean([x["happiness"] for x in m.values()])),
                "bankrupt": any(x["bankrupt"] for x in m.values()), "links": len(self.links),
                "intercity": {k: tot(k) for k in ("goods", "raw", "commuters", "tourists")} if self.flows else {},
                "cities": m}

    def step(self, decisions: dict):
        for r in self.roles:
            if r in decisions and self.acts_this_month(r):
                self.apply(r, decisions[r])
        info = self.end_month()
        obs = {r: self.observe(r) for r in self.roles}
        return obs, info["rewards"], {r: info["terminated"] for r in self.roles}, \
            {r: info["truncated"] for r in self.roles}, {r: {"scores": info["scores"]} for r in self.roles}

    # ------------------------------------------------------------- replay
    def frame(self, turns: list[dict] | None = None, scores: dict | None = None) -> dict:
        by_city = {c: [dict(t, role=split_role(t["role"])[1]) for t in (turns or []) if t["role"].startswith(c + ".")]
                   for c in self.cities}
        sc = scores or self.last_scores
        cities = {}
        for c, e in self.envs.items():
            f = e.frame(turns=by_city[c], scores={k.split(".", 1)[1]: v for k, v in sc.items() if k.startswith(c + ".")})
            f.pop("council", None)
            cities[c] = f
        month = self.month
        return {"month": month, "region": True, "cities": cities, "scores": sc,
                "turns": [{k: v for k, v in t.items() if k != "meta"} for t in (turns or [])],
                "council": self.council.posted_in(month - 1) if month else [],
                "deals": [dict(d) for d in self.deals], "links": [dict(l) for l in self.links],
                "flows": [dict(f) for f in self.flows],
                "metrics": {"month": month, "population": sum(f["metrics"]["population"] for f in cities.values())}}

    def meta(self) -> dict:
        return {"frame_format": C.FRAME_FORMAT, "roles": self.roles, "region": True, "cities": list(self.cities),
                "layout": {c: LAYOUT[c] for c in self.cities}, "river_chain": self.river_chain,
                "city_meta": {c: e.meta() for c, e in self.envs.items()}}

    def config_dict(self) -> dict:
        return {"game": asdict(self.cfg), "scoring": asdict(self.scoring)}

    def final_metrics(self) -> dict:
        return {c: e.world.history[-1] for c, e in self.envs.items()}
