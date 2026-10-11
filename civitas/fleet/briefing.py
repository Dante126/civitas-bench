"""Fleet prompts: the system prompt (fixed rules + coach playbooks) and the per-turn briefing (first user message).

`system_prompt()` is the same every turn: how to play, the role's mission and the game rules, then the coach's
playbooks, so coaches steer strategy but cannot change rules or tools. `briefing()` is a compact status for the
turn: the mayor gets the city, finances, department money and reports, and the plan; a department gets its plan slice,
budget, key metrics, a "since your last turn" digest, the map, its ready sites and its memory. Everything else is a
tool call away.
"""
from __future__ import annotations

import calendar
import json
import math
from collections import Counter

from ..agents.prompts import GAME_RULES
from ..bench.scoring import ScoringConfig
from ..core.actions import zone_price
from ..core.roles import ROLES, TURN_ORDER
from ..sim import catalog as C
from .config import FleetAgentConfig
from .fmt import MESSAGE_PREVIEW_CHARS, a_or_an, cash, cut, money, months, num, pct, signed
from .memory import DEPARTMENTS, FleetState
from .planning import View, cap_line, parks_line, planning_next, planning_view, request_fix
from .region import region_lines
from .tools import MAP_PARTS, MAX_ENTRIES, MAX_SITES, MAYOR_MAP, NEED_KINDS, SITE_PURPOSES, SPEND_MONTHS, CashFlow, \
    DeptMoney, Funding, SupplyBuild, bank_keep, cash_flow, cheapest, \
    funding_gaps, land_blocked, land_fix, map_facts, map_lines, overview, plan_slice, ready_lines, ready_prices, \
    role_obs, scarce_bank, short_of, short_supplies, site_lists, site_price, spend_lines, unclaimed_spots
from .tools import REQUEST_CHARS, REQUESTS_SHOWN, OpenRequest, fund_cover, open_requests, request_groups, request_state
from .tools import Ask, _yours, keep_up_where, mayor_ask, needs_count
from .tools import AWAY_FROM_HOMES, _and, _wanted, _where, edge_site, road_price
from .tools import SITE_MIN_POP, TRANSIT_GOAL, FirstBuild, bus_seats, bus_stops, first_bus_line, transit_estimate
from .tools import TIER_UP, enrol_text, upgrade_line

# paths into build_observation(...)["city"]
KEY_METRICS = {
    "planning": ["population", "unemployment_pct", "demand", "avg_land_value_pct", "zoned_tiles", "developed_tiles",
                 "zoned_without_road_access", "pollution_at_homes_pct"],
    "utilities": ["power", "water", "sewage", "zoned_without_power", "zoned_without_water", "zoned_without_sewage",
                  "river_pollution_pct"],
    "transport": ["commute_minutes", "congested_road_pct", "trade.exports", "trade.freight_capacity",
                  "trade.goods_shortage_pct", "unemployment_pct", "population"],
    "police": ["services.crime_at_homes_pct", "services.police_coverage_pct", "unemployment_pct",
               "services.adults_high_school_plus_pct", "avg_land_value_pct", "population"],
    "fire": ["services.fire_coverage_pct", "services.fires_last_month", "developed_tiles", "jobs_industrial",
             "congested_road_pct", "services.adults_high_school_plus_pct"],
    "health": ["services.health_pct", "services.health_coverage_pct", "services.garbage_collection_pct",
               "services.homes_with_garbage_piling_pct", "pollution_at_homes_pct", "river_pollution_pct",
               "population"],
    "education": ["services.school_coverage_pct", "services.adults_high_school_plus_pct", "population",
                  "unemployment_pct", "jobs_offices", "services.crime_at_homes_pct"],
}
# unlock key -> the roles whose actions it opens (tier unlocks are derived from the tier lines' owners)
UNLOCK_ROLES = {
    **{s["unlock"]: tuple(d for d, structs in C.STRUCTS_OF_DEPT.items() if s["struct"] in structs)
       for specs in C.TIER_LINES.values() for s in specs if s.get("unlock")},
    **{k: ("planning",) for k in ("high_density", "office_zone", "mixed_use", "plaza", "stadium", "monument",
                                  "historic_preservation", "casino")},
    "districts": ("planning", "mayor"),
    **{k: ("transport",) for k in ("avenue", "highway", "train_station", "harbour", "metro", "airport")},
}
MAX_LINES = 6            # events / failed actions listed per briefing
MAX_UNLOCKS = 4          # unlock names on the milestone line
MAX_MESSAGES = 5         # new council messages quoted per briefing
# the mayor's own plan text, quoted short in its briefing (get_plan has it in full, the priorities too: the
# Departments lines dropped their "; priority: ..." suffix, about 385 characters at the limits)
GOAL_CHARS, NOTES_CHARS = 60, 150
BLOCKED_LINE_CHARS = 240   # planning's blocked-for-lack-of-land line: past this, department names only (then cut)
# the mayor's Departments block, once below any line with a blocked suffix: " - ask planning" on each blocked line took
# the worst case (2 idle, 5 blocked) to 7,445 of the 7,450 characters test_briefings_stay_bounded_at_the_limits allowed;
# said once here. Smoke v13: the mayor took a blocked department's money back instead of pushing planning
BLOCKED_NOTE = "- blocked for lack of land: money buys nothing there until planning frees lots (ask planning)"
# department reports quoted on the mayor's Departments block (history kind=report has them whole): shorter than
# REPORT_PREVIEW_CHARS, so the Finances split, the to spend / to fund lines and the debt line fit its 7,700 characters.
# 180 until review: with every department on a needs line (3 first, planning's road, 3 another) in a hold, debt at the
# cap and a 6x deficit, the worst mix took 7,694, over the plan's 7,650; at 170, 7,624. 165 since the City line shows
# imported supply apart ("power 1,046 own + 35 imported / 1,081"): at 170 that mix took 7,706
MAYOR_REPORT_CHARS = 165
# the Finances costs line names the dearest this many buildings lines, then "N more $X": with 3, the worst mix of
# test_briefings_stay_bounded_at_the_limits (needs lines on six departments, debt at the cap, a 6x deficit, seven
# buildings lines, a partial cover in a hold) took the mayor's briefing to 7,695 of its 7,700 characters; with 2, 7,677
# (the plan's fallback; its "top 3" is left for when there is room)
COSTS_SHOWN = 2
# GAME_RULES was written for the map-reading LLMAgent; fleet agents read the map with tools
RULE_EDITS = [("highway tile H on the west\n  edge", "highway on the west\n  edge"),
              ("ask for it on the council board.", "ask for it on the council board (post with send_message, read "
                                                   "with get_messages).")]

HOW_TO_PLAY = """\
# How to play
You are the {title} in a fleet of AI agents that runs a city. The mayor is the manager: it sets the city plan \
(goals, a budget target and a priority per department) and moves the money; the departments ({depts}) carry the \
plan out. {cadence}, and you start each turn from a short briefing.
- Each turn you work by calling tools. Your briefing covers most of what you need; look things up only for what \
it lacks ({lookups}). The Map block in your briefing usually suffices; call map_overview only when you need \
tile-level layout (at most once per turn); inspect for details.{coords}
- Action tools run immediately and report success or failure, the cost and the money left. When an action fails, \
read the error and fix the cause (another site, a cheaper option, wait for an unlock) instead of repeating it.
- {budget}
- {team}
- You remember nothing between turns except the plan, your memory (update_memory: commitments, open tasks, \
lessons) and the game's history log. Write down what you learn.
- When you are done, call end_turn with a short report (what you did, what you need).
- Limits per turn: at most {max_steps} tool calls; at that limit the turn ends and completed actions stay. \
{actions}
- Each tool result shows steps_left; repeating an identical call returns the earlier result (look-ups) or is \
refused (a failed action or an invalid call).
- The mode in your briefing header: autonomous = no coach will intervene during the game; supervised = your coach \
may revise the playbook between sessions."""
LOOKUPS = ("city_overview", "map_overview", "inspect", "find_sites", "my_facilities", "unlocks", "get_plan",
           "get_messages", "history")
LOOKUP_NOTES = {("mayor", "find_sites"): "find_sites (check which sites a department has; read-only)"}

BUDGET = {"mayor": "Stay solvent: upkeep is paid from the treasury every month and every building adds upkeep for "
                   "good; keep the reserve when you move money to departments (allocate_budget) or cut taxes.",
          "dept": "Stay within your budget: you can only spend your department's funds, which the mayor allocates."}
TEAM = {"mayor": "Write the plan with update_plan and fund it with allocate_budget; departments only see their "
                 "slice. Read their reports and adjust the plan when the numbers move. Answer every request to you "
                 "(your briefing's Open requests block) the turn you see it: a top-up, or one line by send_message to "
                 "that department saying why not or when (the reserve, a first building elsewhere comes first, its "
                 "funds already cover it); one send_message can answer several departments (to: \"police,fire\").",
        "dept": "Follow your slice of the plan (budget target, priority, city goals); the budget target is the "
                "mayor's ceiling for you, not money to ask for. Ask the mayor for money only "
                "when your briefing's budget block says to (a first building, supply, planning's road, or another "
                "building the mayor's money covers), for the amount it names; otherwise put what you need in your "
                "end_turn report, which the mayor reads. Ask other departments with send_message."}


def _value(key: str, v) -> str:
    if isinstance(v, dict) and "supply" in v:
        return f"supply {num(v['supply'])} / demand {num(v['demand'])}, coverage {pct(v['coverage_pct'])}"
    if isinstance(v, dict):
        fmt = (lambda x: f"{x:+.2f}") if key == "demand" else num
        return ", ".join(f"{k} {fmt(x)}" for k, x in v.items()) or "n/a"
    return pct(v) if key.endswith("_pct") else num(v)


def _metric(city: dict, path: str) -> str:
    v = city
    for k in path.split("."):
        v = v[k]
    key = path.split(".")[-1]
    return f"- {key.removesuffix('_pct').replace('_', ' ')}: {_value(key, v)}"


def _date(env) -> str:
    w = env.world
    season = f", {w.season()}" if w.seasons_on else ""
    return (f"month {w.month} of {env.cfg.months} (year {w.month // 12 + 1}, {calendar.month_name[w.month % 12 + 1]}"
            f"{season}): {months(env.cfg.months - w.month)} left")


def _action(e: dict) -> str:
    a = dict(e.get("action") or {})
    kind = a.pop("type", "?")
    args = json.dumps(a, separators=(",", ":"))[1:-1] if a else ""
    return cut(f"m{e['month']} {kind}({args}): {e.get('message', '')}")


def _listed(items: list[str], more: str) -> list[str]:
    extra = len(items) - MAX_LINES
    return [f"  - {x}" for x in items[-MAX_LINES:]] + ([f"  - (+{extra} earlier; {more})"] if extra > 0 else [])


def _digest(env, fleet: FleetState, role: str, moved: set = frozenset(), cap: int = MAX_MESSAGES) -> list[str]:
    """The "Since your last turn" block, quoting the `cap` newest new messages. `moved`: (month, sender, text) of
    messages to the role shown in another block instead (the mayor's Open requests), only counted here."""
    w = env.world
    d = fleet.digest(role, w.month, env.council, who=env._name(role))
    L = [f"## Since your last turn ({'month ' + str(d['since']) if d['since'] is not None else 'first turn'})"]
    fails = [_action(e) for e in d["failed_actions"]]
    L.append("- failed actions:" + ("" if fails else " none"))
    L.extend(_listed(fails, "history kind=failure"))
    events = [cut(t) for t in d["events"]]
    L.append("- events:" + ("" if events else " none"))
    L.extend(_listed(events, "history kind=event"))
    for f in w.forecast():
        L.append(f"- forecast: {cut(f)}")
    msgs = [env._local(m) for m in d["messages"]]
    away = [m for m in msgs if m["to"] == role and (m["month"], m["sender"], m["text"]) in moved]
    msgs = [m for m in msgs if m not in away]
    L.append("- new council messages (newest last):" if msgs or away else "- no new council messages")
    if away:
        L.append(f"  - ({len(away)} request{'s' * (len(away) != 1)} to you: under Open requests to you below)")
    older = msgs[:-cap]
    if older:
        L.append(f"  - ({len(older)} older{_older_reach(env, role, older)})")
    L.extend(f"  - m{m['month']} {m['sender']} -> {'you' if m['to'] == role else m['to']}: "
             f"{cut(m['text'], MESSAGE_PREVIEW_CHARS)}" for m in msgs[-cap:])
    return L


def _older_reach(env, role: str, older: list[dict]) -> str:
    """How the briefing's older new messages can be read: get_messages returns the newest MAX_ENTRIES messages
    visible within the council window (ToolBox.get_messages), so some may be past its cap or its window."""
    window = env.cfg.council_window
    since = env.world.month - window
    key = lambda m: (m["month"], m["sender"], m["to"], m["text"])
    got = {key(env._local(m)) for m in env.council.visible_to(env._name(role), since_month=since)[-MAX_ENTRIES:]}
    missing = [m for m in older if key(m) not in got]
    if not missing:
        return ": get_messages"
    if any(m["month"] >= since for m in missing):        # in the window but past the cap
        return f"; get_messages shows only the latest {MAX_ENTRIES}"
    beyond = "beyond" if len(missing) == len(older) else f"{len(missing)} of them beyond"
    return f", {beyond} get_messages' {window}-month window"


def _map(w, role: str) -> list[str]:
    """The Map block; the mayor's keeps only MAYOR_MAP: it has no map actions (its system prompt has no coordinates
    either), and its briefing needed the room for the departments' money."""
    return ["## Map"] + map_lines(map_facts(w), MAYOR_MAP if role == "mayor" else MAP_PARTS)


def _ready_sites(env, role: str, lists: dict, prices: dict, funds: float, keep_bank: dict | None = None,
                 view: View | None = None) -> list[str]:
    """One line per find_sites purpose (count, best site and price, or why there is none), so agents don't re-run
    find_sites for every purpose each turn. Planning's also names what funded departments are about to build
    (_leave_line; its zone strips spare those sites: spare_funded) and the first buildings blocked for lack of land
    (_land_line), and says high density is locked while it is (smoke v9-v11: 6 x zoning high density before 1,000
    residents, repeated across turns). Scarce riverbank lots are marked to keep (scarce_bank); utilities' say what
    for (`keep_bank`: bank_keep)."""
    w = env.world
    dense = [] if role != "planning" or w.unlocked("high_density") else [
        f"- high density: locked until {C.UNLOCK_POP['high_density']:,} residents (now "
        f"{w.metrics()['population']:,}): zone with density \"low\""]
    land, tail = [], ""
    keep = scarce_bank(w) if keep_bank is None else keep_bank
    if role == "planning":
        v = view or planning_view(env, lists, funds)
        lists = v.lists
        land = _leave_line(v.spots, unclaimed_spots(v.funded, v.claims)) + _land_line(env, v.depts, v.roads,
                                                                                       v.blocked)
        dense += parks_line(v.parks, v.park)
        tail = (f"; zoning {money(zone_price('low'))}/tile at low density, {money(zone_price('high'))} at high "
                "density and for I and M")
    return (["## Ready sites"] + ready_lines(role, lists, keep, prices, funds) + land + dense
            + [f"find_sites <purpose> lists up to {MAX_SITES} with details{tail}"])


def _supply_state(b: SupplyBuild) -> str:
    """One supply on utilities' supply-needed line, in the words the mayor's needs line uses (SupplyBuild.status) and
    with the numbers: "power 93% used (supply 111, demand 103)", "water 100% used (supply 1,235, demand 1,235, 85.10
    imported)", "sewage 0% covered (supply 0, demand 5; no site yet)"."""
    nums = (f"supply {num(b.have)}, demand {num(b.demand)}"
            + (f", {num(b.imported)} imported" if round(b.imported, 1) else ""))
    return f"{b.status} ({nums}{'' if b.purpose else '; no site yet'})"


def _supply_money(builds: list[SupplyBuild], funds: float) -> tuple[list[str], list[tuple[str, float]], float]:
    """(what `funds` pay for, what they don't with its price, what that still needs): greedy over the buildings
    short_supplies names, in SUPPLIES order (each "a coal plant at (0,23) ($4,000)")."""
    left = math.floor(funds)
    paid, unpaid = [], []
    for b in (b for b in builds if b.purpose):
        at = f"{a_or_an(b.price.building)} at ({b.site['x']},{b.site['y']}) ({money(b.price.cost)})"
        if short_of(left, b.price.cost):
            unpaid.append((at, b.price.cost))
        else:
            paid.append(at)
            left -= b.price.cost
    return paid, unpaid, math.ceil(sum(c for _, c in unpaid) - left) if unpaid else 0


def _short_line(builds: list[SupplyBuild], funds: float) -> list[str]:
    """Utilities' line under its funds while a supply is wanted (short_supplies: demand at 90% of it or more now or
    within 3 months at its growth, "water 84% used, 92% in 3 months", or a pump or outlet the city lacks), most urgent
    first (a lot develops at all only with power, to level 2 with water,
    to 3 with sewage), each as the mayor's needs line says it (_supply_state), and what its funds pay for, greedy in
    that order with the building its needs line names for each (the mayor's line asks the same), then what the rest
    needs: "- supply needed now: power 25% covered (supply 1.80, demand 7, 1.80 imported), sewage 0% covered (supply
    0, demand 7): your $4,400 pays for a coal plant at (0,23) ($4,000); a sewage outlet at (33,23) ($1,200) needs $800
    more". A supply with no ready site says so ("; no site yet"; its Ready sites line says why). It names, it doesn't
    choose: utilities decides. Smoke v20 seed 1, month 3: $4,400 held and the $4,000 coal plant ready, but 20 steps went
    on inspecting sewage sites and nothing was built; power stayed at 0-18% until month 10. Review: as "- short now:
    power (111 of 103)" it called a supply with room short, under Key metrics' "coverage 100%", while the mayor read
    "power 93% used" (fleet_scripted seed 0: 6 of 20 lines showed more supply than demand)."""
    if not builds:
        return []
    line = "- supply needed now: " + ", ".join(_supply_state(b) for b in builds)
    paid, unpaid, more = _supply_money(builds, funds)
    if not paid and not unpaid:
        return [line]
    line += f": your {money(math.floor(funds))} pays for " + (_and(paid) if paid else
                                                              "none of them" if len(unpaid) > 1 else "nothing yet")
    if unpaid:
        line += f"; {_and([at for at, _ in unpaid])} need{'s' * (len(unpaid) == 1)} {money(more)} more"
    return [line]


def _leave_line(spots: list, unclaimed: list = ()) -> list[str]:
    """Planning's line naming what funded departments are about to build and where (claim_spots, each its own site):
    "- leave free (funded): fire station (12,23), clinic (20,27): zone around them", then each funded building with no
    lot of its own (`unclaimed`: unclaimed_spots, at its best listed site): "; education (elementary school): no lot
    of its own - leave or free one near (38,25)". Past BLOCKED_LINE_CHARS whole items drop from the end, counted
    ("(+2 more)"): the claimed sites first, then the notes. Smoke v14 seed 0, months 3-6: the mayor funded fire,
    health and education for their ready sites near homes, planning (acting next) zoned those lots, and they read
    "nothing needs it right now" at coverage 0; fire held $2,500 to month 12. v15 seed 0, month 9: cut mid-word at 240
    ("zone arou"). v18 seed 1, month 6: education's only site was police's, the line named police's alone, and
    education never built."""
    found = [f"{sp.building} ({sp.xy[0]},{sp.xy[1]})" for sp in spots]
    notes = [f"{sp.dept} ({sp.building}): no lot of its own - leave or free one near ({sp.xy[0]},{sp.xy[1]})"
             for sp in unclaimed]
    if not found:                                  # an unclaimed building always sits under a claim
        return []
    head, tail = "- leave free (funded): ", ": zone around them"
    more = lambda items, n, sep: sep.join(items[:n]) + (f" (+{len(items) - n} more)" if n < len(items) else "")
    line = lambda n, k: head + more(found, n, ", ") + tail + (f"; {more(notes, k, '; ')}" if k else "")
    fits = (line(n, k) for k in (range(len(notes), 0, -1) if notes else [0]) for n in range(len(found), 0, -1)
            if len(line(n, k)) <= BLOCKED_LINE_CHARS)
    return [next(fits, line(1, min(1, len(notes))))]


def _land_line(env, depts: dict, roads: list[dict] = (), found: dict | None = None) -> list[str]:
    """Planning's line naming each department's cheapest first buildings blocked for lack of land or roads
    (land_blocked), by the names the mayor sees, each tagged with what it needs beyond a roadside lot (land_fix:
    "sewage outlet: a road to the river"); past BLOCKED_LINE_CHARS without the "away from homes" tags (the next line
    says it), then without tags, then departments only. Smoke v13 seed 0, months 6-9: planning zoned every roadside
    tile despite its playbook's "leave room", and utilities, police, fire, health and education all read "none - no
    free roadside land: ask planning for a lot". With a building that must stay away from homes, _away_line follows."""
    found = land_blocked(env, depts) if found is None else found       # planning's view passes its own
    if not found:
        return []
    w = env.world
    head, tail = "- blocked for lack of land: ", ": extend a road into open land, or free a lot, for them"

    def name(d: str, p: str, tag: int) -> str:            # tag 2: every tag, 1: all but away from homes, 0: none
        fix = land_fix(w, d, p) if tag > (p in AWAY_FROM_HOMES) else ""
        return site_price(w, d, p).building + (f": {fix}" if fix else "")
    forms = [", ".join(f"{d} ({', '.join(dict.fromkeys(name(d, p, tag) for p in ps))})" for d, ps in found.items())
             for tag in (2, 1, 0)] + [", ".join(found)]
    body = next((f for f in forms if len(head + f + tail) <= BLOCKED_LINE_CHARS), forms[-1])
    return [head + cut(body, BLOCKED_LINE_CHARS - len(head + tail)) + tail] + _away_line(w, found, roads)


def _away_line(w, found: dict[str, list[str]], roads: list[dict]) -> list[str]:
    """Planning's line under the blocked one when a blocked first building must stay away from homes
    (AWAY_FROM_HOMES: a landfill or jail): leave a lot free by the far end of a road_extension site (`roads`: a dead
    end's extension or a branch road) with under 3 homes within 4 tiles (edge_site), else by the end of a new road
    into open land at the town edge. Smoke v15 seed 1, months 6-9: health's landfill read "no roadside lot away from
    homes", planning never left a lot at the town edge, and garbage coverage stayed 0 all run."""
    away = [f"{d}'{'s' * (not d.endswith('s'))} {site_price(w, d, p).building}" for d, ps in found.items()
            for p in ps if p in AWAY_FROM_HOMES]
    if not away:
        return []
    head, site = f"- away from homes ({_and(away)}): ", edge_site(w, roads)
    if site is None:
        return [head + "extend a road into open land at the town edge and leave a lot free by its far end, with no "
                "homes within 4 tiles"]
    return [head + f"leave a lot free by ({site['x2']},{site['y2']}), the far end of road_extension {_where(site)}, "
            "with no homes within 4 tiles"]


def _needs(m: DeptMoney) -> str:
    """The mayor's Departments line suffix when a department's funds are below its cheapest ready build
    (first_build): "; needs $X for its first <building> (N sites)" or "for another <building> (coverage 74%)" (Need),
    else "". Every building at that price is named ("another clinic (coverage 21%) or another landfill (coverage
    83%)"): the mayor writes what it reads into the plan (smoke v9: "Build first shelter"). Smoke v16: "a police
    station (5 ready sites)" read the same with one built, and second stations were funded before a first school.
    Utilities names a building per short supply, bought together (FirstBuild.text: "its first coal plant and water
    tower ($110/month)"; one alone reads as before), then each short supply's building with no ready site and what it
    waits for (FirstBuild.waits: "; sewage outlet waits for a road to the river"). Review, month 0: the mayor read
    only "needs $6,200 for its first coal plant and water tower" while utilities read "sewage (0 of 5; no site yet)",
    and nothing told it sewage waited for planning's road to the river."""
    b = m.needs
    return "" if b is None else (f"; needs {money(b.cost)} for {b.text()}" + "".join(f"; {x}" for x in b.waits)
                                 + (f"; {m.stuck}" if m.count == 1 and m.stuck else ""))


def _dept_line(d: str, m: DeptMoney, f: Funding, budgets: dict) -> str:
    """One department on the mayor's Departments block: funds, the plan target ("<" when below it and it would use a
    top-up: funding_gaps), what it spent (the block's header says over how many months), then one suffix by its kind
    (DeptMoney.kind): its idle money and what it keeps ("; idle $1,508 (keeps $1,000)"), what its needs line buys
    (_needs), what is blocked (_blocked, with the unclaimed_firsts funding_gaps set; BLOCKED_NOTE below says what to
    do), else what its money does (DeptMoney.state: "; funds cover its builds", "; funds cover its next build",
    "; supply has room", "; only a library ready ($1,500)", "; nothing ready to buy"; planning none). Smoke
    v9-v12: nothing showed what a department spent, and planning, spending about $1,000 a quarter, read "funds $7,834 <
    target $9,000" and was topped up. v19 seed 1, month 6: take-backs left planning under what it keeps, unseen. A
    target above the funds of a department with nothing to buy (Funding.unusable) reads "target $7,000 (unusable
    now)", the suffix saying why ("; supply has room", "; nothing ready to buy"): region run 20261005,
    "funds $0 < target $7,000; supply has room" kept utilities asking for money it couldn't use."""
    kind = m.kind
    suffix = (f"; idle {money(m.idle)} (keeps {money(m.keep)})" if kind == "idle" else
              _needs(m) if kind in NEED_KINDS else
              _blocked(m, m.unclaimed) if kind == "blocked" else
              f"; {m.state}" if m.state else "")
    return (f"- {d}: funds {money(math.floor(m.funds))}"
            + (f"{' <' if d in f.gaps else ','} target {money(budgets[d])}" if d in budgets else "")
            + (" (unusable now)" if d in f.unusable and d in budgets else "")
            + (f", spent {money(m.spent)}" if m.months else "") + suffix)


def _blocked(m: DeptMoney, unclaimed: tuple[str, ...] = ()) -> str:
    """The mayor's line suffix when a department has no ready site but a first building blocked for lack of land or
    roads, or when every funded first building of it has no lot of its own (`unclaimed`: unclaimed_firsts, each of
    its sites claimed by an earlier department): "; blocked: a police station" (BLOCKED_NOTE, once below, says to
    ask planning; planning's Ready sites say why and what each needs). Smoke v13 seed 0, month 6: police, fire,
    health and education held their first building's price with every roadside lot zoned, and the mayor took it back
    instead of pushing planning. v18 seed 1, month 6: education's only site was police's too."""
    names = m.blocked or unclaimed
    return f"; blocked: {' or '.join(names)}" if names else ""


def _memory(fleet: FleetState, role: str) -> list[str]:
    mem = fleet.memory.get(role, {})
    return ["## Your memory"] + ([f"- {k.replace('_', ' ')}: " + "; ".join(v) for k, v in mem.items() if v] or [
        "- empty: save commitments, open tasks and lessons with update_memory"])


def _milestone(env, role: str) -> str:
    """One line: the next milestone, its reward and what it unlocks - for a department only its own options."""
    w = env.world
    nm = w.next_milestone()
    if not nm:
        return "Next milestone: none, all reached."
    keys = nm["unlocks"]
    mine = [k for k in keys if role in UNLOCK_ROLES.get(k, ())]
    shown = (mine + [k for k in keys if k not in mine])[:MAX_UNLOCKS] if role == "mayor" else mine[:MAX_UNLOCKS]
    names = ", ".join(k.replace("_", " ") for k in shown)
    rest = len(keys) - len(shown)
    if role == "mayor":
        what = f"unlocks {len(keys)} options" + (f" ({names}{f' and {rest} more' if rest else ''})" if keys else "")
    else:
        what = (f"unlocks for you: {names}" if shown else "unlocks nothing for you") + (
            f"; and {rest} more for other roles" if rest else "")
    return (f"Next milestone: '{nm['name']}' at {nm['pop']:,} residents (now {w.metrics()['population']:,}), "
            f"{money(nm['reward'])} reward: {what} (see unlocks).")


def _goals(goals: list[str], n: int | None = None) -> str:
    return "; ".join(f"{i}) {cut(g, n) if n else g}" for i, g in enumerate(goals, 1)) or "none"


def _header(env, role: str, mode: str) -> list[str]:
    s = env.last_scores
    who = ("Mayor" if role == "mayor" else ROLES[role]["title"]) + (f" of {env.city.upper()}" if env.city else "")
    return [f"# {who}: {_date(env)}",
            f"Scores: city {s.get('city', 0):.1f}, yours {s.get(role, 0):.1f} | mode: {mode} | "
            f"up to {env.cfg.max_actions_per_turn} successful actions this turn", ""]


REPAY_MIN = 500          # a repayment the debt line suggests: at least this much spare (in $100 steps)


def _finances(w, cf: CashFlow, spare: float = 0.0) -> list[str]:
    """The mayor's Finances lines (the WARNING line follows): the treasury and its change since overview's trend month
    split into operations, the mayor's top-ups, loans and the rest (cash_flow; parts that round to $0 left out),
    last month's income and expenses by kind (buildings, the dearest COSTS_SHOWN by name, then city services, imports
    when they round to $1 or more, and interest), and with debt what it costs and what is left to borrow:
    - "- treasury $7,922 (-$12.3k since month 3: operations +$977, top-ups -$13.3k), debt $0"
    - "- last month: income $597, expenses $469: buildings $312 (utilities $82, roads $75, 3 more $155), city services
      $157, interest $0" (city services: the rest of the expenses, the town hall and policies included, so the parts
      add up; "imports $173" before the interest when there are any: review, a region's $173 of service imports read
      as city services)
    - "- debt $18.8k: $94/month interest, $31.2k more can be borrowed" (the first line then leaves the debt out; what a
      repayment saves is in repay_loan's description and the playbook: review, the worst mix ran 27 characters over),
      and with `spare` (the treasury above the reserve after every top-up still to fund) of REPAY_MIN or more
      "; repay_loan 5000 now: that much is spare above the reserve and what waits to be funded (saves $25/month)":
      the 120-month region run ended with south at $55k treasury and $28k debt, north $137k and $5k, 2 repayments
      against 23 loans
    Smoke v13-v19: the treasury change mixed in the mayor's own transfers, so 30 of 81 turns showed a "loss" while
    operations ran a surplus; interest never showed in dollars, and there were 0 repay_loan calls in 96 turns."""
    debt = round(w.debt) > 0
    head = f"- treasury {cash(w.treasury)}"
    if cf.since is not None:
        parts = [f"{k} {signed(v, money)}" for k, v in (("operations", cf.ops), ("top-ups", -cf.top_ups),
                                                         ("loans", cf.loans), ("other", cf.other)) if round(v)]
        head += (f" ({signed(round(cf.change), money)} since month {cf.since}"
                 + (": " + ", ".join(parts) if parts else "") + ")")
    L = [head + ("" if debt else f", debt {money(w.debt)}")]
    if cf.since is None:
        L.append("- last month: no month played yet")
    else:
        costs = list(cf.buildings.items())
        n = len(costs) if len(costs) <= COSTS_SHOWN + 1 else COSTS_SHOWN      # one more: named ("1 more" as long)
        rest = [v for _, v in costs[n:]]
        detail = ", ".join(f"{k} {money(v)}" for k, v in costs[:n]) + (f", {len(rest)} more {money(sum(rest))}"
                                                                       if rest else "")
        L.append(f"- last month: income {money(cf.income)}, expenses {money(cf.expenses)}: buildings "
                 f"{money(sum(cf.buildings.values()))}" + (f" ({detail})" if detail else "")
                 + f", city services {money(cf.services)}"
                 + (f", imports {money(cf.imports)}" if round(cf.imports) else "") + f", interest {money(cf.interest)}")
    if debt:
        room = C.MAX_DEBT - w.debt
        pay = min(math.floor(w.debt), math.floor(max(0.0, spare) / 100) * 100)
        L.append(f"- debt {money(w.debt)}: {money(w.debt * C.LOAN_INTEREST)}/month interest, "
                 + (f"{money(room)} more can be borrowed" if round(room) > 0 else f"at the {money(C.MAX_DEBT)} limit")
                 + (f"; repay_loan {pay} now: that much is spare above the reserve and what waits to be funded "
                    f"(saves {money(pay * C.LOAN_INTEREST)}/month)" if pay >= REPAY_MIN else ""))
    return L


def _supply_parts(v: dict) -> str:
    """A supply on the mayor's City line: "1,081", or with service imports "1,046 own + 35 imported" (imports cover at
    most a quarter of demand and stop once the city's own output covers it; counted in, a city importing read
    "power 1,081 / 1,081" and looked served)."""
    imp = v.get("imported") or 0.0
    if round(imp) < 1:
        return num(v["supply"])
    return f"{num(round(v['supply'] - imp, 1))} own + {num(imp)} imported"


def _mayor(env, fleet: FleetState) -> list[str]:
    w = env.world
    o = overview(env)
    t = o["trend"]
    ch = lambda k, fmt=num: f" ({signed(t[k], fmt)})" if t else ""
    L = ["## City" + (f" (change since month {t['since_month']})" if t else ""),
         f"- population {num(o['population'])}{ch('population')}, happiness {pct(o['happiness_pct'])}"
         f"{ch('happiness_pct', lambda v: f'{v:.1f} pts')}, approval {pct(o['approval_pct'])}, "
         f"unemployment {pct(o['unemployment_pct'])}",
         "- jobs: " + ", ".join(f"{k} {num(v)}" for k, v in o["jobs"].items())
         + " | demand: " + _value("demand", o["demand"]),
         "- supply / demand (coverage): " + ", ".join(
             f"{k} {_supply_parts(o[k])} / {num(o[k]['demand'])} ({pct(o[k]['coverage_pct'])})"
             for k in ("power", "water", "sewage")),
         "- coverage: " + ", ".join(f"{k} {pct(v)}" for k, v in o["coverage_pct"].items()),
         "- taxes: " + ", ".join(f"{k} {v:g}%" for k, v in o["tax_pct"].items()),
         "", "## Finances"]
    f = funding_gaps(env, fleet)
    L += _finances(w, cash_flow(w), f.spare - sum(i.amount for i in f.to_fund))
    if w.months_negative or w.low_approval_months:
        L.append(f"- WARNING: negative treasury {months(w.months_negative)} ({C.BANKRUPTCY_MONTHS} = bankruptcy), "
                 f"approval below {100 * C.RECALL_APPROVAL:.0f}% {months(w.low_approval_months)} "
                 f"({C.RECALL_MONTHS} = recall)")
    reqs = open_requests(env, fleet)
    asks = _requests(reqs, f, w.month)
    # each department line of the Open requests block takes the place of a quoted message: the worst mix of
    # test_briefings_stay_bounded_at_the_limits stays within its bound with requests from every department
    cap = max(1, MAX_MESSAGES - max(0, len(asks) - 2))
    L += [""] + _digest(env, fleet, "mayor", {(r.month, r.dept, r.text) for r in reqs}, cap)
    for off in w.offers:
        L.append(f"- OPEN OFFER #{off['id']}: large factory, {off['jobs']} jobs, {money(off['bonus'])} bonus, heavy "
                 f"pollution (accept_offer / decline_offer before month {off['expires']})")
    p = plan_slice(fleet, "mayor")
    plan = bool(fleet.plan_versions)
    budgets = p["budgets"] if plan else {}
    latest = {e["role"]: e for e in fleet.log if e["kind"] == "report" and e["role"] in DEPARTMENTS}
    clipped = any(len(" ".join(e["text"].split())) > MAYOR_REPORT_CHARS for e in latest.values())
    spent = min(len(w.history), SPEND_MONTHS)
    notes = (["most urgent first"] + [f"spent = last {months(spent)}"] * bool(spent)
             + ["cut reports: history kind=report"] * clipped)
    # most urgent first (Funding.order): smoke v16-v19 funded in block order in 32 of 54 turns, and education, last
    # in turn order, was topped up last in 11
    L += ["", f"## Departments ({'; '.join(notes)})"]
    for d in f.order:
        L.append(_dept_line(d, f.depts[d], f, budgets))
        if d in latest:
            L.append(f"  report (m{latest[d]['month']}): {cut(latest[d]['text'], MAYOR_REPORT_CHARS)}")
    if not latest:
        L.append("- no reports yet")
    if any(m.kind == "blocked" for m in f.depts.values()):              # a line says "; blocked: ..."
        L.append(BLOCKED_NOTE)
    L += spend_lines(w, f)                     # the money to spend and the top-ups the needs lines ask, in order
    L += asks + [""]
    if not plan:
        L.append("## City plan: no plan yet - write one with update_plan (goals, budgets, priorities).")
    else:
        L += [f"## City plan (v{p['version']}, month {p['month']}; priorities and full text: get_plan)",
              f"- goals: {_goals(p['goals'], GOAL_CHARS)}"]
        if p["notes"]:
            L.append(f"- notes: {cut(p['notes'], NOTES_CHARS)}")
    return L + [""] + _map(w, "mayor") + [""] + _memory(fleet, "mayor")


def _requests(reqs: list[OpenRequest], f, now: int) -> list[str]:
    """The mayor's Open requests block (open_requests; [] without any): one line per department, the longest waiting
    first, at most REQUESTS_SHOWN (then "+N more"), its repeats merged: the months it asked, how long it has waited,
    "last turn to answer" when the next turn falls outside the window, the latest amount asked, what its money says
    (request_state: what to fund and whether the money covers it, or why it needs no top-up) and its latest request
    quoted. The digest counts these messages only. 120-month LLM runs (trinity v19 / v23): the mayor answered 59-81%
    of the requests sent to it (other departments ~100%), and replied to almost none of those it didn't fund."""
    if not reqs:
        return []
    covered = fund_cover(f)
    groups = request_groups(reqs)
    L = ["", "## Open requests to you (answer each: a top-up, or send_message on why not / when; one call answers "
              "several)"]
    for d, rs in groups[:REQUESTS_SHOWN]:
        n = Counter(r.month for r in rs)
        asked = ", ".join(f"m{m}" + (f" x{k}" if k > 1 else "") for m, k in n.items())
        last = "; last turn to answer" if any(r.last for r in rs) else ""
        amount = next((r.amount for r in reversed(rs) if r.amount), None)
        ask = f", asks {money(amount)}" if amount else ""
        L.append(f"- {d}: {asked} (waiting {months(now - rs[0].month)}{last}){ask}; {request_state(f, d, covered)}: "
                 f"\"{cut(rs[-1].text, REQUEST_CHARS)}\"")
    if len(groups) > REQUESTS_SHOWN:
        L.append(f"- +{len(groups) - REQUESTS_SHOWN} more: " + ", ".join(d for d, _ in groups[REQUESTS_SHOWN:]))
    return L


def _planning_requests(env, reqs: list[OpenRequest], v: View) -> list[str]:
    """Planning's Open requests block: one line per department asking it (open_requests to planning), the longest
    waiting first, at most REQUESTS_SHOWN: when it asked, how long it has waited, "last turn to answer" when its next
    turn falls outside the window, what planning can do (planning.request_fix) and the request quoted. 176 requests went
    to planning in the 120-month runs, 85 unanswered; in the late smokes all of them."""
    w = env.world
    groups = request_groups(reqs)
    L = ["", "## Open requests to you (answer each: do it, or send_message where it can build or why not; one call "
             "answers several)"]
    for d, rs in groups[:REQUESTS_SHOWN]:
        n = Counter(r.month for r in rs)
        asked = ", ".join(f"m{m}" + (f" x{k}" if k > 1 else "") for m, k in n.items())
        last = "; last turn to answer" if any(r.last for r in rs) else ""
        fix = request_fix(w, d, v, " ".join(r.text for r in rs))
        L.append(f"- {d}: {asked} (waiting {months(w.month - rs[0].month)}{last}): {fix}; "
                 f"\"{cut(rs[-1].text, REQUEST_CHARS)}\"")
    if len(groups) > REQUESTS_SHOWN:
        L.append(f"- +{len(groups) - REQUESTS_SHOWN} more: " + ", ".join(d for d, _ in groups[REQUESTS_SHOWN:]))
    return L


def _bus_first(bus, funds: float, ask: Ask) -> str:
    """Transport's priority line while the city has no bus line and its first is ready (first_bus_line): the line
    before any avenue, and what to ask the mayor for when short (only when it would fund it: mayor_ask). The 120-month
    LLM runs listed bus stops ($1,500) and congested roads ($30/tile) side by side; transport spent on avenues first
    and never afforded the line."""
    lack = "" if not short_of(funds, bus.cost) else f"; you have {money(math.floor(funds))}" + (
        f": ask the mayor for {money(math.ceil(bus.cost - funds))} (send_message)" if ask.ok else "")
    return (f"- First bus line ({bus.text()}) before any avenue: create_bus_line through all the bus_stops stops "
            f"below; upgrade_road is refused while it would leave you short of it{lack}")


# transport's supplementary builds -> their action (the Next line's call when first_build names one as `extra`)
STATION_ACTIONS = {"train_station": "build_train_station", "harbour": "build_harbour",
                   "metro_station": "build_metro_station"}


def _transport_next(env, lists: dict, funds: float, ask: Ask, b: FirstBuild | None) -> str:
    """Transport's Next line, every turn: the build first_build names (`b`, its DeptMoney.build: the one the mayor's needs line and mayor_ask
    name, so the price and the ask amount always agree), shown with its call. The first bus line (_bus_first); another
    one (first_build names it while its bus_stops list offers it), with transit now and with it (transit_estimate);
    an avenue stretch, priced whole, as one upgrade_road call on the most jammed stretch; the first link to the region
    once demand is capped without one ("your first train station"), or with nothing else ready the cheapest station or
    harbour its lists offer, and its build call; else that nothing is to build, and why (a
    region city's links are its Region block's). When funds are short, the ask amount when the mayor would fund it
    (mayor_ask), else to say so in the end_turn report. 120-month LLM runs: 156 of 240 transport turns did nothing, 142
    reasoned about "1,500 residents" past 1,500 (122 with it in memory), and in the region run 47 of 52 turns that
    listed another line with funds short of it read "funds cover your next build" ($30 avenue tiles) and never asked."""
    w = env.world
    purpose = b.purposes[0] if b is not None and b.purposes else None
    cost = (b.cost if b.cost is not None else b.extra[1]) if purpose else None

    def lack() -> str:
        if not short_of(funds, cost):
            return ""
        return f"; you have {money(math.floor(funds))}" + (
            f": ask the mayor for {money(ask.amount)} (send_message)" if ask.ok and ask.amount else
            ": say what you need in your end_turn report")

    goal = f"your score counts transit up to {pct(100 * TRANSIT_GOAL)}"
    if purpose == "bus_stops" and not w.bus_lines and (bus := first_bus_line(w, lists)):
        return _bus_first(bus, funds, ask)
    if purpose == "bus_stops":
        now, more = transit_estimate(w, bus_stops(lists["bus_stops"]))
        return (f"- Next: {b.named[0].label} ({money(cost)}): transit {pct(100 * now)} now, about "
                f"{pct(100 * more)} with it ({goal}): create_bus_line through all the bus_stops stops below, before "
                f"any avenue{lack()}")
    if purpose == "congested_roads":
        s = lists["congested_roads"].sites[0]
        call = json.dumps({"x1": s["x1"], "y1": s["y1"], "x2": s["x2"], "y2": s["y2"], "to": "avenue"},
                          separators=(",", ":"))
        return (f"- Next: {b.named[0].label} on the most jammed stretch, load {s.get('load_vs_capacity')}x capacity "
                f"({money(cost)}): upgrade_road {call}{lack()}")
    if purpose in STATION_ACTIONS:
        s = lists[purpose].sites[0]
        call = json.dumps({"x": s["x"], "y": s["y"]}, separators=(",", ":"))
        what = _yours(b.named[0].label) if b.named else a_or_an(b.extra[0])     # "your first train station"
        return (f"- Next: {what} ({money(cost)}): {STATION_ACTIONS[purpose]} {call}"
                + (" (a metro line needs 2 stations, then create_metro_line)" if purpose == "metro_station" else "")
                + lack())
    m = w.metrics()
    roads = lists["congested_roads"]
    jam = ("no road over capacity" if not roads.sites else "jammed roads are avenues already"
           if w.unlocked("avenue") else f"avenues unlock at {C.UNLOCK_POP['avenue']:,} residents")
    links = "; links to your neighbours: your Region block" if w.region_info else ""
    if not w.bus_lines and m["population"] < SITE_MIN_POP["bus_stops"]:
        return (f"- Next: no bus line or avenue to build: the first bus line comes at {SITE_MIN_POP['bus_stops']:,} "
                f"residents (now {m['population']:,}), {jam}{links}. End your turn")
    return (f"- Next: nothing to build: transit {pct(100 * transit_estimate(w)[0])} ({goal}), {jam}{links}. End "
            "your turn; there is no population threshold to wait for")


NEXT_ROLES = ("planning", "transport")      # the roles with a Next line (refreshed in their action results)
NEXT_REFRESH_ITEMS = 2                      # builds a refreshed Next line names (the briefing's: planning's NEXT_ITEMS)
NOTHING_LEFT = ("nothing to build", "no bus line or avenue to build")   # Next lines that name no build


def next_line(env, role: str, most: int = NEXT_REFRESH_ITEMS) -> str:
    """`role`'s Next line on the world as it is now, as _department builds it (the same site lists, funds, DeptMoney
    and mayor_ask; planning's view), with at most `most` of planning's builds; "" for a role without one. ToolBox
    puts it, shortened (next_clause), after each successful action of a model planning or transport agent: the late
    smokes' planning re-sent its briefing's Next-line calls after doing them, 15 calls later."""
    if role not in NEXT_ROLES:
        return ""
    w = env.world
    fleet = FleetState.of(env)
    funds = w.dept_funds[ROLES[role]["budget"]]
    obs = role_obs(env, role)
    lists = site_lists(w, role, obs)
    money_of = DeptMoney(env, ROLES[role]["budget"])
    ask = mayor_ask(env, fleet, role, money_of)
    if role == "transport":
        return _transport_next(env, lists, funds, ask, money_of.build)
    return planning_next(env, obs["city"], planning_view(env, lists, funds), funds, ask, most)


def next_clause(line: str) -> str:
    """A Next line as the clause a tool result ends with: " | next: <its builds and calls>", or " | next: nothing
    left: end_turn" when it names no build; "" for no line."""
    if not line:
        return ""
    text = line.removeprefix("- ").removeprefix("Next: ")
    if text.startswith(NOTHING_LEFT):
        return " | next: nothing left: end_turn"
    return f" | next: {text}"


def _ask_line(a: Ask) -> str:
    """A department's Your budget line on asking the mayor for money (mayor_ask): "- ask the mayor for $2,500 for your
    first police station (send_message, with a reason): the mayor funds first buildings, supply and planning's road
    first", or "- no money ask to the mayor now: another police station ($2,500 short) waits: the mayor's money above
    its reserve doesn't cover it after first buildings, supply and planning's road. Say what you need in your end_turn
    report" (the "can't pay" line says it instead when it shows). Region smoke 20261005: 28 of 46 asks to the mayor
    were for another building it couldn't cover or with nothing ready, and it spent its steps refusing them. A no-ask
    adds mayor_ask's plan note, when to ask and what to do this turn (_not_now): "- no money ask to the mayor now:
    supply has room; your $7,000 plan target is the mayor's ceiling for you, not money to ask for. Ask when a supply
    will be 90% used within 3 months at its growth (now water 51% used, 55% in 3 months). This turn: no supply needs
    building yet; say what you need in your end_turn report". Region run 20261005: 125 asks held back, each under a
    bare no-ask line; in 51 the plan target was above the funds."""
    if a.ok:
        return f"- ask the mayor for {money(a.amount)} for {a.what} (send_message, with a reason): {a.why}"
    return (f"- no money ask to the mayor now: {a.why}" + (f"; {a.plan}" if a.plan else "") + "."
            + (f" Ask when {a.when}." if a.when else "")
            + (f" This turn: {a.instead}; say" if a.instead else " Say") + " what you need in your end_turn report")


def _keep_up_line(m: DeptMoney, funds: float) -> list[str]:
    """A department's Your budget line while its coverage is under KEEP_UP_GOAL and its next build is another building
    of a KEEP_UP purpose: the buildings the mayor's count names (DeptMoney.count; one when the money covers no more),
    where (the sites keep_up picked: the most coverage, not the Ready line's best by need, which it says when they
    differ), what they bring coverage to, and to build all of them this turn when its funds cover them, else the ones
    they pay for now; when no ready site adds coverage, where the uncovered homes are and to ask planning (stuck); []
    otherwise. Late smoke 20261007 seed 2: fire read "build another fire station" every quarter while coverage went
    94 -> 86 -> 81 -> 78%; at month 99 it built at the Ready line's best (37,23), which served one building, where
    (36,30) would have brought coverage from 86% to ~92%.
    A tier upgrade its next build names (tier_up, school_upgrade: Need.call) gets the same line, with its call: "- tier
    upgrade: an upgrade of your police station at (12,20) to a police HQ ($7,800, +$75/month; coverage 81%, ~88% with
    it; 24 downtown tiles lack tier-2 cover; another police station at your best ready site adds under 1%):
    upgrade_facility {"x":12,"y":20,"module":"tier"} this turn: your funds cover it" (or "once your funds cover it (you
    have $2,129)"). v1: police and fire never read an upgrade, and the models built 23-26 tier-1 police stations."""
    b = m.build
    if b is not None and b.named and b.named[0].call and b.cost is not None:
        n = b.named[0]
        up = f", +{money(n.upkeep)}/month" if round(n.upkeep) else ""
        when = ("this turn: your funds cover it" if not short_of(funds, b.cost) else
                f"once your funds cover it (you have {money(math.floor(funds))})")
        return [f"- tier upgrade: {_yours(n.label)} ({money(b.cost)}{up}; {n.status}"
                + (f"; {n.note}" if n.note else "") + f"): {n.call} {when}"]
    if not m.keep_ups:
        if not m.stuck:
            return []
        n = m.build.named[0]
        return [f"- keep up with growth: {n.label} at your ready sites adds little ({n.status}): {m.stuck}; ask "
                f"planning for a lot or a road near ({m.gap[0]},{m.gap[1]}) before building another"]
    g, k = m.goal, m.count
    n = math.floor((funds + 1e-9) / (g.cost / k))
    status = "; ".join(f"{nd.label}: {nd.status}" for nd in g.named) if len(g.named) > 1 else g.named[0].status
    them, these = ("them", "both" if k == 2 else f"all {k}") if k > 1 else ("it", "it")
    now = (f"build {these} this turn: your funds cover {them} ({money(g.cost)})" if n >= k
           else f"build {these} this turn once your funds cover {them} ({money(g.cost)}, you "
                f"have {money(math.floor(funds))})" + (f"; they pay for {n} now: build {'it' if n == 1 else 'those'}"
                                                       if n else ""))
    b = m.build
    best = {p: spots[0] for p, spots in zip(b.purposes, b.sites) if spots}
    other = [f"({best[p][0]},{best[p][1]})" for p, spots in zip(g.purposes, g.sites)
             if p in best and best[p] not in spots and best[p] not in m.avoid]     # not one an earlier dept takes
    note = f"; the Ready line's best {', '.join(other)} adds less" if other else ""
    return [f"- keep up with growth: {keep_up_where(g)} ({status}): {now}{note}"]


def _service_metrics(w, role: str) -> list[str]:
    """Key metrics lines the coverage metrics don't show: police, fire and health's level 4-5 tiles without tier-2
    cover (tier-1 buildings cover them only to T1_DOWNTOWN: an upgrade covers them), education's enrolment per level
    (what its score counts; school coverage alone read 99% beside 78-85% high-school enrolment in v1)."""
    if role in TIER_UP:
        n = (w.stats.get("downtown_uncovered") or {}).get(TIER_UP[role][1], 0)
        return [f"- downtown (level 4-5) tiles without tier-2 cover: {n} (tier 1 covers them to "
                f"{pct(100 * C.T1_DOWNTOWN)} at most)"]
    if role == "education":
        return ["- " + "; ".join(enrol_text(w, lv) for lv in ("elementary", "high"))]
    return []


def _bus_riders(w) -> str:
    """The Key metrics line on transport's buses: riders against seats, and the share of trips they carry (what its
    score counts, up to TRANSIT_GOAL)."""
    if not w.bus_lines:
        return "- bus riders: no bus line yet"
    riders, seats = bus_seats(w)
    n = len(w.bus_lines)
    return (f"- bus riders: {riders:,} of {seats:,} seats ({n} line{'s' * (n != 1)}), transit "
            f"{pct(100 * w.metrics()['transit_share'])} of trips")


def _department(env, fleet: FleetState, role: str) -> list[str]:
    w = env.world
    dept = ROLES[role]["budget"]
    p = plan_slice(fleet, role)
    if not fleet.plan_versions:
        L = ["## Your plan slice", "- No city plan yet: follow your mission (your budget block says whether to ask the "
             "mayor for money)."]
        target = None
    else:
        target = p["budget"]
        L = [f"## Your plan slice (plan v{len(fleet.plan_versions)}, month {p['month']})",
             f"- budget target: {money(target) if target is not None else 'none set'}",
             f"- priority: {p['priority'] or 'none set'}", f"- city goals: {_goals(p['goals'])}"]
        if p.get("notes"):
            L.append(f"- mayor's notes: {cut(p['notes'], 200)}")
    spent = w.history[-1]["dept_spent"].get(dept, 0) if w.history else 0
    funds = w.dept_funds[dept]
    obs = role_obs(env, role)             # the one build: metrics and ready sites
    lists = site_lists(w, role, obs)
    prices = ready_prices(w, role, lists)
    L += ["", "## Your budget",
          f"- {money(math.floor(funds))} in department funds"      # rounded down, as against a price (short_of)
          + (f" (plan target {money(target)})" if target is not None else "")
          + f"; spent last month {money(spent)}"]
    supplies = short_supplies(w, lists) if role == "utilities" else []
    L += _short_line(supplies, funds)
    # smoke runs: builds tried with $0, stations funded $1,000 of $2,500. Only when funds are short of every ready
    # price (a supplementary one too, else it's false); it names the cheapest primary ones (all on a tie: health's
    # clinic or landfill), else a supplementary one
    # utilities: only purposes whose supply is wanted, and the amount its supply line names. Review, fleet_scripted
    # seed 0 month 24: "a coal plant at (0,23) ($4,000) needs $2,252 more", then "cheapest: near_homes $2,200" with
    # water at 141 of 250
    wanted = {p: pr for p, pr in prices.items() if role != "utilities" or _wanted(w, p)}
    purposes, low = cheapest(wanted) or cheapest(wanted, supplementary=True) or ((), None)
    more = _supply_money(supplies, funds)[2]
    # whether to ask the mayor for money (mayor_ask): for the needs line's top-up (utilities: its supply line's), or
    # else say what is needed in the end_turn report; one line, the "can't pay" one when it shows
    money_of = DeptMoney(env, dept)          # its next build (first_build), shared by the ask and transport's Next line
    if needs_count(money_of):                # more than one building to keep its coverage up: the mayor's count
        money_of = funding_gaps(env, fleet).depts.get(dept, money_of)
    ask = mayor_ask(env, fleet, role, money_of)
    if (low or more) and all(short_of(funds, pr.cost) for pr in prices.values()):
        L.append("- You can't pay for any ready site" + (" (the supply above needs " + money(more) + " more)" if more
                                                         else f" (cheapest: {' or '.join(purposes)} {low.text()})")
                 + ". Don't call build tools this turn: " + (
                     f"ask the mayor for {money(more or ask.amount)} with a reason (send_message) unless you already "
                     "did, then end_turn." if ask.ok else
                     f"no money ask to the mayor ({ask.why}" + (f"; {ask.plan}" if ask.plan else "") + ")"
                     + (f"; ask when {ask.when}" if ask.when else "") + ": say what you need in your end_turn report."))
    else:
        L.append(_ask_line(ask))
    L += _keep_up_line(money_of, funds)
    if role == "transport":
        L.append(_transport_next(env, lists, funds, ask, money_of.build))
    view = planning_view(env, lists, funds) if role == "planning" else None     # once a briefing
    if view is not None:
        L.append(planning_next(env, obs["city"], view, funds, ask))
    L += ["", "## Key metrics", *(_metric(obs["city"], k) for k in KEY_METRICS[role])]
    if role == "transport":
        L.append(_bus_riders(w))
    L += _service_metrics(w, role)
    if role == "planning":
        L += cap_line(w)
    reqs = open_requests(env, fleet, to="planning") if view is not None else []
    asks = _planning_requests(env, reqs, view) if reqs else []
    # each department line of the Open requests block takes the place of a quoted message, as on the mayor's
    cap = max(1, MAX_MESSAGES - max(0, len(asks) - 2))
    L += [""] + _digest(env, fleet, role, {(r.month, r.dept, r.text) for r in reqs}, cap) + asks
    if role == "planning" and (road := road_price(lists)):     # shown on its line only, not in the budget line above
        prices = {**prices, "road_extension": road}
    keep = bank_keep(w, supplies) if role == "utilities" else None
    ready = _ready_sites(env, role, lists, prices, funds, keep, view)
    ready[-1:-1] = upgrade_line(w, role, obs)                     # before the find_sites footer
    return L + [""] + _map(w, role) + [""] + ready + [""] + _memory(fleet, role)


def system_prompt(role: str, cfg: FleetAgentConfig, total_months: int, *, act_every: int = 1,
                  max_actions: int | None = None, target_pop: int | None = None, region: str = "") -> str:
    """Fixed "how to play" rules, the role's mission and the game rules (and in region mode the `region` section,
    fleet/region.py:region_rules), then the coach's playbooks (fleet, then the role's; empty ones left out). Callers
    pass `act_every=env.cfg.act_every`, `max_actions=env.cfg.max_actions_per_turn` (the successful-action cap) and
    `target_pop=env.scoring.target_population`."""
    spec = ROLES[role]
    side = "mayor" if role == "mayor" else "dept"
    rules = GAME_RULES
    for old, new in RULE_EDITS:
        rules = rules.replace(old, new)
    cap = (f"After {max_actions} successful actions further actions are refused" if max_actions else
           "After the successful-action cap in your briefing header further actions are refused")
    parts = [HOW_TO_PLAY.format(
                 title=spec["title"], depts=", ".join(DEPARTMENTS), budget=BUDGET[side], team=TEAM[side],
                 cadence="Each game month is one turn" if act_every <= 1 else
                 f"You act every {act_every} months (the city runs on its own in between)",
                 lookups=", ".join(LOOKUP_NOTES.get((role, t), t) for t in LOOKUPS
                                   if t != "find_sites" or SITE_PURPOSES[role] or role == "mayor"),
                 coords="" if role == "mayor" else " Coordinates: x = column (0 = west), y = row (0 = north).",
                 max_steps=cfg.max_steps, actions=cap + " (look-ups still work; call end_turn)."),
             f"# Your role: {spec['title']}\n{spec['mission']}",
             "# Game rules\n" + rules.format(turn_order=" -> ".join(TURN_ORDER), months=total_months,
                                             target_pop=target_pop or ScoringConfig.target_population).rstrip()]
    if region:
        parts.append(region.strip())
    books = [(h, b.strip()) for h, b in (("Fleet", cfg.fleet_playbook), (spec["title"], cfg.playbook)) if b.strip()]
    if books:
        parts.append("# Coach's playbook\n" + "\n\n".join(f"## {h}\n{b}" for h, b in books))
    return "\n\n".join(parts)


def briefing(env, role: str, *, mode: str = "autonomous", mark: bool = True) -> str:
    """The first user message of `role`'s turn. Syncs the fleet log with the latest month (idempotent) and, unless
    `mark=False`, marks the turn so the next briefing's "since your last turn" digest starts from here."""
    fleet = FleetState.of(env)
    fleet.sync(env)
    L = _header(env, role, mode)
    L += _mayor(env, fleet) if role == "mayor" else _department(env, fleet, role)
    L += region_lines(env, role)                  # region mode: neighbours, links, flows, deals ([] in one city)
    L += ["", _milestone(env, role)]
    if mark:
        fleet.mark_turn(role, env.world.month, env.council, who=env._name(role))
    return "\n".join(L)
