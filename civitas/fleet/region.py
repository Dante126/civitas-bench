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
  deals; the mayor's also has its deal opportunities (`RegionEnv.deal_opportunities`: it or a neighbour at its power /
  water limit while the other has spare, with the exact offer_deal call), its spare otherwise, and for each deal
  whether an offer to it is worth accepting, when an active one ends and whether to cancel it (`RegionEnv.deal_check`);
- tools (`region_tools`): `region_overview` for every role, the mayor's deal actions (`offer_deal`, `accept_deal`,
  `cancel_deal`, run through the region as it runs any mayor's deals), and `send_message` addressed to the city's
  own roles, a whole team ("south"), another city's agent ("south.mayor") or "all" (`message_targets`).
"""
from __future__ import annotations

import json
import math
import weakref
from functools import partial

from ..core.actions import connector_feeder
from ..core.observation import _connector_sites
from ..core.region import DEAL_AMOUNT_MAX, DEAL_BUSY, DEAL_MIN, DEAL_MONTHS_MAX, DEAL_PRICE_MAX, RegionEnv, \
    opportunity_text
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
    """The city's deals as its Region block lists them (RegionEnv.listed: open offers, active deals, offers that
    expired within the mayors' last turn)."""
    return [region._deal_view(d, city) for d in region.listed(city)]


def overview(region, city: str) -> dict:
    """The region_overview result: the other cities (RegionEnv._region_view), the river, this city's links,
    connectors and connector sites, last month's flows, its open deals, its power / water as a deal sees them
    (RegionEnv.supply_state) and its deal opportunities with their offer_deal arguments."""
    view = region._region_view(city)
    w = region.envs[city].world
    states = region.supply_states()
    return {"ok": True, "city": city, "cities": view["cities"], "my_specialisations": view["my_specialisations"],
            "river": river(region, city),
            "links": [dict(l) for l in region.links if city in (l["a"], l["b"])],
            "flows_last_month": [dict(f) for f in region.flows if city in (f["a"], f["b"])],
            "connectors": [dict(c) for c in w.connectors],
            "connector_sites": connector_prices(w),
            "deals": _deals(region, city),
            "supply": states[city],
            "deal_opportunities": [{**o, "offer_deal": region.offer_args(o, city)}
                                   for o in region.deal_opportunities(city, states)]}


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
                 "moves every month. No link is needed. A city short of power or water imports at most "
                 f"{round(100 * C.IMPORT_SHARE_MAX)}% of its demand at ${C.IMPORT_PRICE['power']:g} (power) / "
                 f"${C.IMPORT_PRICE['water']:g} (water) a unit a month, and while it imports no new lot connects, so "
                 "it stops growing; spare power earns only "
                 f"${C.POWER_SELL_PRICE:g} a unit and spare water nothing. A deal between those prices pays both "
                 "sides: your Region block names one when it fits, and says whether an offer to you is worth taking "
                 f"(and affordable). An offer not accepted within {region.offer_window()} months expires.")
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


def _use(s: dict, resource: str) -> str:
    """'power at 97% of your own supply' (supply_state)."""
    return (f"{resource} at {s['use_pct']}% of your own supply" if s["use_pct"] is not None else
            f"no {resource} supply of your own")


def _deal_line(d: dict, mayor: bool = True, check: dict | None = None) -> str:
    """One deal: for the mayor what it pays or earns and what to do (with `check`, RegionEnv.deal_check: whether an
    offer to it is worth taking, the last month of an active one, and whether to end it); for utilities what it does
    to supply."""
    sell = d["you_are"] == "selling"
    n = d["months_left"]
    call = lambda name: f'{name} {{"deal": {d["id"]}}}'
    if mayor:
        what = (f"you {'sell' if sell else 'buy'} {num(d['amount'])} {d['resource']} {'to' if sell else 'from'} "
                f"{d['partner']} at ${d['price']:g}/unit/month ({'+' if sell else '-'}"
                f"{money(d['amount'] * d['price'])}/month)")
    else:
        what = (f"the city {'sells' if sell else 'buys'} {num(d['amount'])} {d['resource']} {'to' if sell else 'from'} "
                f"{d['partner']} ({'-' if sell else '+'}{num(d['amount'])} {d['resource']} supply)")
    if d["status"] == "expired":
        by = "you" if d["proposed_by"] != d["partner"] else d["partner"]
        return (f"- deal #{d['id']} expired: {what} for {n} months, offered by {by}, was not accepted by month "
                f"{d['expires']}" + (": offer_deal again if it still fits" if mayor else ""))
    if d["status"] == "active":
        line = f"- deal #{d['id']} active: {what}, {n} month{'s' * (n != 1)} left"
        if check:
            line += f" (through month {check['ends']})"
            if check.get("cancel"):
                s = check["s"]
                line += (f"; you need it back ({_use(s, d['resource'])}): {call('cancel_deal')}" if sell else
                         f"; you have room without it: {call('cancel_deal')} saves "
                         f"{money(d['amount'] * d['price'])}/month")
            elif mayor:                  # the region late smoke's south cancelled a +$37/month sale with 243 spare
                line += "; keep it: it runs by itself, nothing to call"
        return line
    until = f" (open until month {d['expires']})" if d.get("expires") is not None else ""
    if not mayor:
        return f"- deal #{d['id']} offered by {d['proposed_by']}: {what} for {n} months, if the mayors agree{until}"
    if d["proposed_by"] == d["partner"]:
        line = f"- deal #{d['id']} offered by {d['partner']}: {what} for {n} months{until}"
        if not check:
            return line + ": accept_deal to take it"
        why = _offer_why(d, check)
        if check.get("reason") == "afford":
            return line + f"; {why}: leave it or {call('cancel_deal')}"
        return line + f"; {why}: " + (f"{call('accept_deal')} takes it" if check["worth"] else
                                      f"not worth it, leave it or {call('cancel_deal')}")
    return (f"- deal #{d['id']} offered by you: {what} for {n} months, waiting for {d['partner']}.mayor to accept"
            + until)


def _ids(ids) -> str:
    return ", ".join(f"#{i}" for i in ids)


def _offer_why(d: dict, check: dict) -> str:
    """Why an offer to the city is worth taking or not (RegionEnv.deal_check's `reason`), in the mayor's words."""
    s, r, reason = check["s"], d["resource"], check.get("reason")
    if d["you_are"] == "selling":
        left = check.get("spare_left", RegionEnv.spare(s))
        after = f" after deal {_ids(check['taken_by'])}" if check.get("taken_by") else ""
        return (f"below the ${s['floor']:g}/unit your spare earns now" if reason == "floor" else
                f"you have only {num(left)} spare{after}" if reason == "short" else
                f"you have {num(left)} spare{after}")
    if reason == "room":
        return f"you have room ({_use(s, r)})"
    if reason == "dearer":
        return f"dearer than imports (${s['import_price']:g}/unit)"
    if reason == "covered":
        return f"{_use(s, r)}, already covered by deal {_ids(check.get('covered_by', []))}"
    base = f"{_use(s, r)}, imports cost ${s['import_price']:g}/unit"
    if reason == "too_big":
        return base + f", but you need only {num(math.ceil(check.get('need_left', s['need'])))}"
    if reason == "seller_short":
        return base + f", but {d['partner']} has no {r} left to sell (its own supply only covers its demand)"
    if reason == "afford":
        return (f"{base}, but you can't afford it: {money(check['net'])}/month more than imports for "
                f"{d['months_left']} months is over the {money(check['budget'])} you have for it (above the reserve, "
                "with last month's surplus)")
    if check.get("worth") and check.get("seller_has", d["amount"]) < d["amount"]:
        return base + f" ({d['partner']} has only {num(check['seller_has'])} spare now and may end it early)"
    return base


def _deal_lines(region, city: str, note: str, mayor: bool = True, states: dict | None = None) -> list[str]:
    deals = _deals(region, city) if mayor else [d for d in _deals(region, city) if d["status"] != "expired"]
    if not deals:
        return [f"- deals: none{note}"]
    L = [_deal_line(d, mayor, region.deal_check(d, city, states) if mayor else None) for d in deals[:MAX_DEALS]]
    return L + ([f"- (+{len(deals) - MAX_DEALS} more deals: region_overview)"] if len(deals) > MAX_DEALS else [])


def _opportunity_lines(region, city: str, states: dict) -> list[str]:
    """The mayor's deal opportunities (RegionEnv.deal_opportunities), each with its offer_deal call; else, with spare
    beyond DEAL_BUSY of its own supply, one line saying so. `states`: RegionEnv.supply_states()."""
    L = []
    opps = region.deal_opportunities(city, states)
    for o in opps:
        why, what = opportunity_text(o, city)
        L.append(f"- deal opportunity: {why}: offer_deal {json.dumps(region.offer_args(o, city))} {what} "
                 "(no link needed)")
    told = {o["resource"] for o in opps} | {d["resource"] for d in region.deals if d["status"] == "offered"
                                            and city in (d["seller"], d["buyer"])}
    spare = {r: RegionEnv.spare(s) for r, s in states[city].items() if r not in told}
    spare = {r: v for r, v in spare.items() if v >= DEAL_MIN}
    if spare and region.deal_months():
        L.append("- spare: " + ", ".join(f"{r} {num(round(v))}" for r, v in spare.items())
                 + f" beyond {round(100 * DEAL_BUSY)}% use; no neighbour is short of "
                 + ("it" if len(spare) == 1 else "either") + " now (a deal opportunity shows here when one is)")
    return L


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
        states = region.supply_states()           # once for the deal and opportunity lines
        L += _deal_lines(region, city, ": offer_deal sells or buys power / water capacity with another mayor",
                         states=states)
        L += _opportunity_lines(region, city, states)
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
             "flows of goods, commuters and tourists, your city's deals, its power / water against its own supply and "
             "its deal opportunities.", lambda: overview(region, city)),
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
            Tool("offer_deal", "Offer another city's mayor a power or water deal: sell (sell: true) or buy "
                 f"(sell: false; always give it) `amount` units of capacity (1-{DEAL_AMOUNT_MAX}) at `price` dollars per unit per month "
                 f"(0-{DEAL_PRICE_MAX}) for `months` (1-{DEAL_MONTHS_MAX}, default 24). It runs once that mayor calls "
                 f"accept_deal within {region.offer_window()} months (then the offer expires); the offer is posted on "
                 "the council.", partial(_deal, box, region, "offer_deal"),
                 {"to": {"type": "string", "enum": others}, "resource": {"type": "string", "enum": list(RESOURCES)},
                  "amount": {"type": "number"}, "price": {"type": "number"}, "months": {"type": "integer"},
                  "sell": {"type": "boolean"}}, ("to", "resource", "amount", "price", "sell")),
            Tool("accept_deal", "Accept a deal another city's mayor offered your city.",
                 partial(_deal, box, region, "accept_deal"), {"deal": deal}, ("deal",)),
            Tool("cancel_deal", "End one of your city's deals (offered or active).",
                 partial(_deal, box, region, "cancel_deal"), {"deal": deal}, ("deal",)),
        ]
    return tools
