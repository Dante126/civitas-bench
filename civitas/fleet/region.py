"""Fleets in region mode: each city's fleet plays its own CivitasEnv; this module adds what the region brings.

A region (`core/region.py:RegionEnv`) runs 2-6 CivitasEnvs that share one council. A fleet agent for "north.mayor"
binds to the north city's env (`ToolAgent.bind`), so its FleetState (plan, department memory, history log), its
ToolBox and its briefings are north's alone: the north mayor plans for and funds north's departments. `link(region)`
remembers which region each city env belongs to (a weak map: nothing is added to the env or the FleetState, which
stay as in a single city); `region_of(env)` finds it and is None in a single-city game, where nothing here changes a
thing.

What the region adds:
- the system prompt's Region section (`region_rules`): the city, the other cities, the council's addresses, the
  river, the links and (mayor) the deals;
- a Region block in the mayor's, transport's and utilities' briefings (`region_lines`): the other cities (population,
  surplus and shortfall, unemployed and open jobs, links, last month's flows), connector sites, the river and the
  deals;
- tools (`region_tools`): `region_overview` for every role, the mayor's deal actions (`offer_deal`, `accept_deal`,
  `cancel_deal`, run through the region as it runs any mayor's deals), and `send_message` addressed to the city's
  own roles, a whole team ("south"), another city's agent ("south.mayor") or "all" (`message_targets`).
"""
from __future__ import annotations

import math
import weakref
from functools import partial

from ..core.actions import connector_feeder
from ..core.observation import _connector_sites
from ..sim import catalog as C
from .fmt import money, num
from .tools import Tool, ToolError, to_schema

_REGIONS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()     # city env -> its RegionEnv
MAX_DEALS = 4            # deals listed in a briefing's Region block (region_overview has them all)
REGION_BLOCK = ("mayor", "transport", "utilities")      # roles whose briefing has a Region block
RESOURCES = ("power", "water")


def link(region) -> None:
    """Remember the region of each of its city envs (ToolAgent.bind calls this; idempotent)."""
    for env in region.envs.values():
        _REGIONS[env] = region


def region_of(env):
    """The RegionEnv a city env belongs to (after `link`), else None."""
    return _REGIONS.get(env) if getattr(env, "city", None) else None


# ---------------------------------------------------------------- facts
def _others(region, city: str) -> list[str]:
    return [c for c in region.cities if c != city]


def _where(region, city: str, other: str) -> str:
    side = region.neighbors(city).get(other)
    return f"your {side} edge" if side else "far: boat or air only"


def river(region, city: str) -> dict | None:
    """Upstream and downstream cities and the river's pollution where it enters and leaves this city (as
    RegionEnv.observe reports them); None for a city off the river's chain."""
    chain = region.river_chain
    if city not in chain:
        return None
    w, i = region.envs[city].world, chain.index(city)
    return {"upstream": chain[:i], "downstream": chain[i + 1:],
            "inflow_pct": round(100 * (1 - math.exp(-w.upstream_inflow / C.RIVER_NORM)), 1),
            "outflow_pct": round(100 * float(w.river_row[-1]), 1)}


def connector_prices(world) -> dict[str, dict]:
    """Neighbour -> {"side", "built", "site", "price": {kind: what build_connector charges at that site}} for the best
    connector site towards it (observation._connector_sites: the shortest feeder road); price {} without a site."""
    out = {}
    for city, cs in _connector_sites(world).items():
        price = {}
        if cs["site"]:
            s = cs["site"]
            feeder = connector_feeder(world, cs["side"], s["x"], s["y"]) or []
            road = sum(C.CATALOG["bridge" if world.terrain[y, x] == C.WATER else "road"]["cost"] for x, y in feeder)
            price = {k: v["cost"] + road for k, v in C.CONNECTOR.items() if k not in cs["built"]}
        out[city] = {**cs, "price": price}
    return out


def _deals(region, city: str) -> list[dict]:
    return [region._deal_view(d, city) for d in region.deals if d["status"] != "ended"
            and city in (d["seller"], d["buyer"])]


def overview(region, city: str) -> dict:
    """The region_overview result: the other cities (RegionEnv._region_view), the river, this city's links,
    connectors and connector sites, last month's flows and its open deals."""
    view = region._region_view(city)
    w = region.envs[city].world
    return {"ok": True, "city": city, "cities": view["cities"], "my_specialisations": view["my_specialisations"],
            "river": river(region, city),
            "links": [dict(l) for l in region.links if city in (l["a"], l["b"])],
            "flows_last_month": [dict(f) for f in region.flows if city in (f["a"], f["b"])],
            "connectors": [dict(c) for c in w.connectors],
            "connector_sites": connector_prices(w),
            "deals": _deals(region, city)}


# ---------------------------------------------------------------- prompts
def region_rules(region, city: str, role: str) -> str:
    """The system prompt's Region section: fixed rules, the same every turn."""
    others = _others(region, city)
    near = ", ".join(f"{c.upper()} ({_where(region, city, c)})" for c in others)
    r = river(region, city)
    flow = ""
    if r:
        up = f"it arrives from {', '.join(c.upper() for c in r['upstream'])}" if r["upstream"] else "you are upstream"
        down = (f"your sewage and runoff flow on into {', '.join(c.upper() for c in r['downstream'])}'s water"
                if r["downstream"] else "nothing flows on from you")
        flow = f"\n- River: it flows north to south; {up}, and {down}."
    L = [f"# Region\nYou work for the {city.upper()} city, one of {len(region.cities)} cities in this region; the others: "
         f"{near}. Every city has its own team of agents and its own score (the region score is their average). "
         f"The plan, budgets, memory and history in your tools are {city.upper()}'s alone, and "
         + ("you manage" if role == "mayor" else "your mayor manages") + f" {city.upper()}'s departments only.",
         f"- Council: all teams share one board. send_message to a role of your city (\"mayor\"), a whole team "
         f"(\"{others[0]}\"; \"{city}\" is your own), one agent of another city (\"{others[0]}.mayor\") or \"all\" "
         f"(every city). Messages from other cities name the sender with its city (\"{others[0]}.mayor\")." + flow,
         "- Links: linked cities trade surplus goods and raw materials (the seller earns, the buyer avoids costly "
         "imports), unemployed workers commute to open jobs, and residents visit the other city's casinos, landmarks "
         "and historic districts. Transport links a neighbour with build_connector (highway or rail) on the map edge "
         "facing it; the link opens once both cities have built their side. Any two cities with a harbour are linked "
         "by boat, any two with an airport by air."]
    if role == "mayor":
        L.append("- Deals: offer_deal offers another mayor power or water capacity (sell: true) or asks to buy it "
                 "(sell: false) at a price per unit per month; it runs once that mayor calls accept_deal, and either "
                 "side can cancel_deal. Bought capacity adds to your supply, sold capacity comes off it; the money "
                 "moves every month.")
    L.append("- region_overview shows the other cities, the links, last month's flows"
             + (", connector sites" if role == "transport" else "") + (" and the deals" if role == "mayor" else "")
             + (" (your briefing's Region block has the summary)." if role in REGION_BLOCK else "."))
    return "\n".join(L)


def _nonzero(v: dict) -> str:
    return " ".join(f"{k} {num(x)}" for k, x in v.items() if x)


def _city_line(region, city: str, v: dict, prices: dict | None) -> str:
    """One other city on the mayor's Region block."""
    c = v["city"]
    parts = [f"pop {num(v['population'])}"]
    if v["strong_in"]:
        parts.append("strong in " + ", ".join(v["strong_in"]))
    if s := _nonzero(v["surplus"]):
        parts.append("surplus " + s)
    if s := _nonzero(v["shortfall"]):
        parts.append("short " + s)
    parts.append(f"unemployed {num(v['unemployed'])}, open jobs {num(v['open_jobs'])}")
    link = "links: " + (", ".join(v["links"]) or "none")
    if v["they_built_waiting_for_you"]:
        link += f" (they built {', '.join(v['they_built_waiting_for_you'])}: build your side)"
    if v["you_built_waiting_for_them"]:
        link += f" (you built {', '.join(v['you_built_waiting_for_them'])}: waiting for them)"
    if prices and c in prices and prices[c]["price"] and not v["links"] and not v["you_built_waiting_for_them"]:
        kind, cost = min(prices[c]["price"].items(), key=lambda kv: kv[1])
        link += f"; a {kind} connector costs transport {money(cost)}"
    head = f"- {c.upper()} ({_where(region, city, c)}" + (f", {v['river'].replace(' of you', '')}" if "river" in v else "")
    line = head + "): " + "; ".join(parts) + "; " + link
    if "last_month" in v:
        f = v["last_month"]
        line += f" | last month: goods {num(f['goods'])}, raw {num(f['raw'])}, commuters {num(f['commuters'])}, " \
                f"tourists {num(f['tourists'])}"
    return line


def _deal_line(d: dict, mayor: bool = True) -> str:
    """One deal: for the mayor what it pays or earns and what to do; for utilities what it does to supply."""
    sell = d["you_are"] == "selling"
    n = d["months_left"]
    if mayor:
        what = (f"you {'sell' if sell else 'buy'} {num(d['amount'])} {d['resource']} {'to' if sell else 'from'} "
                f"{d['partner']} at ${d['price']:g}/unit/month ({'+' if sell else '-'}"
                f"{money(d['amount'] * d['price'])}/month)")
    else:
        what = (f"the city {'sells' if sell else 'buys'} {num(d['amount'])} {d['resource']} {'to' if sell else 'from'} "
                f"{d['partner']} ({'-' if sell else '+'}{num(d['amount'])} {d['resource']} supply)")
    if d["status"] == "active":
        return f"- deal #{d['id']} active: {what}, {n} month{'s' * (n != 1)} left"
    if not mayor:
        return f"- deal #{d['id']} offered by {d['proposed_by']}: {what} for {n} months, if the mayors agree"
    if d["proposed_by"] == d["partner"]:
        return f"- deal #{d['id']} offered by {d['partner']}: {what} for {n} months: accept_deal to take it"
    return f"- deal #{d['id']} offered by you: {what} for {n} months, waiting for {d['partner']}.mayor to accept"


def _deal_lines(region, city: str, note: str, mayor: bool = True) -> list[str]:
    deals = _deals(region, city)
    if not deals:
        return [f"- deals: none{note}"]
    L = [_deal_line(d, mayor) for d in deals[:MAX_DEALS]]
    return L + ([f"- (+{len(deals) - MAX_DEALS} more deals: region_overview)"] if len(deals) > MAX_DEALS else [])


def _river_line(region, city: str) -> list[str]:
    r = river(region, city)
    if r is None:
        return []
    into = f"arrives from {', '.join(r['upstream'])} {r['inflow_pct']:g}% polluted" if r["upstream"] else ""
    out = f"leaves for {', '.join(r['downstream'])} {r['outflow_pct']:g}% polluted" if r["downstream"] else ""
    return [f"- river: {'; '.join(p for p in (into, out) if p)}"] if into or out else []


def region_lines(env, role: str) -> list[str]:
    """The briefing's Region block ([] in a single city and for roles outside REGION_BLOCK), with a blank line first.
    Mayor: every other city in one line, the specialisations, the river and the deals. Transport: each neighbour's
    link and best connector site with its price, and the boat / air links. Utilities: the river and the deals that
    move power or water capacity."""
    region = region_of(env)
    if region is None or role not in REGION_BLOCK:
        return []
    city = env.city
    L = ["", f"## Region ({city.upper()} and {len(region.cities) - 1} other cit"
         + ("y" if len(region.cities) == 2 else "ies") + "; details: region_overview)"]
    w = env.world
    if role == "mayor":
        view = region._region_view(city)
        prices = connector_prices(w)
        L += [_city_line(region, city, v, prices) for v in view["cities"]]
        spec = {k: v for k, v in view["my_specialisations"].items() if v >= 0.5}
        L.append("- your specialisations: " + (", ".join(f"{k} {v:g}" for k, v in spec.items()) or "none yet"))
        L += _river_line(region, city)
        L += _deal_lines(region, city, ": offer_deal sells or buys power / water capacity with another mayor")
    elif role == "transport":
        view = {v["city"]: v for v in region._region_view(city)["cities"]}
        for c, cs in connector_prices(w).items():
            v = view[c]
            s = f"- {c.upper()} ({cs['side']} edge): links: {', '.join(v['links']) or 'none'}"
            if v["they_built_waiting_for_you"]:
                s += f"; they built {', '.join(v['they_built_waiting_for_you'])}: build your side to open it"
            if v["you_built_waiting_for_them"]:
                s += f"; you built {', '.join(v['you_built_waiting_for_them'])}: waiting for them"
            if cs["site"] and cs["price"]:
                site = cs["site"]
                s += (f"; connector site ({site['x']},{site['y']}), {site['feeder_tiles']} feeder road tiles: "
                      + ", ".join(f"{k} {money(p)}" for k, p in cs["price"].items()))
            elif not cs["site"] and len(cs["built"]) < len(C.CONNECTOR):
                s += f"; no free edge tile within {C.FEEDER_MAX} tiles of a road: ask planning for a road towards " \
                     f"the {cs['side']} edge"
            L.append(s)
        harbours = {c: bool(e.world.facilities(C.HARBOUR)) for c, e in region.envs.items()}
        airports = {c: bool(e.world.n_buildings(C.AIRPORT)) for c, e in region.envs.items()}
        for kind, has, what in (("boat", harbours, "a harbour"), ("air", airports, "an airport")):
            partners = [c for c in _others(region, city) if has[c]]
            L.append(f"- {kind}: " + (f"linked with {', '.join(partners)}" if has[city] and partners else
                                      f"you have {what}; no other city has one yet" if has[city] else
                                      f"{what} here would link you with {', '.join(partners)}" if partners else
                                      f"needs {what} in two cities"))
        L.append("- a link carries per month: " + "; ".join(
            f"{k} {num(v['goods'])} goods + {num(v['people'])} people" for k, v in C.LINK_CAPACITY.items()))
    else:
        L += _river_line(region, city)
        L += _deal_lines(region, city, " (the mayor makes power / water deals; bought capacity adds to your supply, "
                                       "sold capacity comes off it)", mayor=False)
    return L


# ---------------------------------------------------------------- tools
def message_targets(env, role: str) -> list[str]:
    """send_message's addresses: the other roles of the city and "all", in a region also the city's whole team, the
    other teams and their agents."""
    own = [r for r in env.roles if r != role]
    region = region_of(env)
    if region is None:
        return [*own, "all"]
    others = _others(region, env.city)
    return [*own, "all", env.city, *others, *(f"{c}.{r}" for c in others for r in region.envs[c].roles)]


def _deal(box, region, name: str, /, **args) -> dict:
    """Run a deal action for the box's city as RegionEnv.apply runs a mayor's (it posts the offer on the council),
    logged like any other action."""
    w = box.world
    cap = box.env.cfg.max_actions_per_turn
    if sum(r["ok"] for r in box.executed) >= cap:
        raise ToolError(f"action limit reached for this turn ({cap} successful actions); call end_turn")
    res = region._deal_action(box.env.city, box.role, {"type": name, **args})
    box.executed.append(res)
    box.fleet.record(w.month, box.role, "action", action=res["action"], ok=res["ok"], message=res["message"],
                     cost=res["cost"])
    return {"ok": res["ok"], "message": res["message"], "cost": res["cost"], "budget_left": round(w.treasury, 2)}


def region_tools(box) -> list[Tool]:
    """Tools a ToolBox gets in a region: region_overview (a look-up), the region's send_message, and the mayor's
    deal actions. [] in a single city."""
    region = region_of(box.env)
    if region is None:
        return []
    city = box.env.city
    others = _others(region, city)
    tools = [
        Tool("region_overview", "The other cities of the region (population, specialisations, surplus and shortfall, "
             "unemployed and open jobs, links), the river, your links, connectors and connector sites, last month's "
             "flows of goods, commuters and tourists, and your city's deals.", lambda: overview(region, city)),
        Tool("send_message", "Post on the council board (delivered at the end of your turn): to a role of your city "
             f"(\"mayor\"), a whole team (\"{others[0]}\"; \"{city}\" is yours), one agent of another city "
             f"(\"{others[0]}.mayor\"), or \"all\" (every city). Several in one call (to: \"police,fire\") each get "
             "their own message.", box.send_message,
             {"to": to_schema(message_targets(box.env, box.role)), "text": {"type": "string"}}, ("to", "text"),
             lists=("to",)),
    ]
    if box.role == "mayor":
        deal = {"type": "integer", "description": "the deal number (#1 is 1)"}
        tools += [
            Tool("offer_deal", "Offer another city's mayor a power or water deal: sell (sell: true, the default) or buy "
                 "(sell: false) `amount` units of capacity (1-2000) at `price` dollars per unit per month (0-10) for "
                 "`months` (1-120, default 24). It runs once that mayor calls accept_deal; the offer is posted on the "
                 "council.", partial(_deal, box, region, "offer_deal"),
                 {"to": {"type": "string", "enum": others}, "resource": {"type": "string", "enum": list(RESOURCES)},
                  "amount": {"type": "number"}, "price": {"type": "number"}, "months": {"type": "integer"},
                  "sell": {"type": "boolean"}}, ("to", "resource", "amount", "price")),
            Tool("accept_deal", "Accept a deal another city's mayor offered your city.",
                 partial(_deal, box, region, "accept_deal"), {"deal": deal}, ("deal",)),
            Tool("cancel_deal", "End one of your city's deals (offered or active).",
                 partial(_deal, box, region, "cancel_deal"), {"deal": deal}, ("deal",)),
        ]
    return tools
