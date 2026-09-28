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

from ..agents.prompts import GAME_RULES
from ..bench.scoring import ScoringConfig
from ..core.roles import ROLES, TURN_ORDER
from ..sim import catalog as C
from .config import FleetAgentConfig
from .fmt import MESSAGE_PREVIEW_CHARS, REPORT_PREVIEW_CHARS, cut, money, months, num, pct, signed
from .memory import DEPARTMENTS, FleetState
from .tools import HOLD, MAP_PARTS, MAX_ENTRIES, MAX_SITES, MAYOR_MAP, SITE_PURPOSES, SPEND_MONTHS, DeptMoney, \
    Funding, cheapest, dept_lists, funded_firsts, funding_gaps, land_blocked, land_fix, map_facts, map_lines, \
    overview, plan_slice, ready_lines, ready_prices, role_obs, scarce_bank, short_of, site_lists, site_price

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
# the mayor's own plan text, quoted short in its briefing (get_plan has it in full)
GOAL_CHARS, PRIORITY_CHARS, NOTES_CHARS = 60, 40, 150
BLOCKED_LINE_CHARS = 240   # planning's blocked-for-lack-of-land line: past this, department names only (then cut)
# the mayor's Departments block, once below any line with a blocked suffix: " - ask planning" on each blocked line took
# the worst case (2 idle, 5 blocked) to 7,445 of the 7,450 characters test_briefings_stay_bounded_at_the_limits allows;
# said once here, 7,420
BLOCKED_NOTE = "- blocked for lack of land: ask planning for lots"
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

BUDGET = {"mayor": "Stay solvent: upkeep is paid from the treasury every month, so keep a reserve when you move money "
                   "to departments (allocate_budget) or cut taxes.",
          "dept": "Stay within your budget: you can only spend your department's funds, which the mayor allocates."}
TEAM = {"mayor": "Write the plan with update_plan and fund it with allocate_budget; departments only see their "
                 "slice. Read their reports and adjust the plan when the numbers move.",
        "dept": "Follow your slice of the plan (budget target, priority, city goals). To get something from the "
                "mayor or another department, ask with send_message."}


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


def _digest(env, fleet: FleetState, role: str) -> list[str]:
    w = env.world
    d = fleet.digest(role, w.month, env.council)
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
    L.append("- new council messages (newest last):" if msgs else "- no new council messages")
    older = msgs[:-MAX_MESSAGES]
    if older:
        L.append(f"  - ({len(older)} older{_older_reach(env, role, older)})")
    L.extend(f"  - m{m['month']} {m['sender']} -> {'you' if m['to'] == role else m['to']}: "
             f"{cut(m['text'], MESSAGE_PREVIEW_CHARS)}" for m in msgs[-MAX_MESSAGES:])
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


def _ready_sites(env, role: str, lists: dict, prices: dict, funds: float) -> list[str]:
    """One line per find_sites purpose (count, best site and price, or why there is none), so agents don't re-run
    find_sites for every purpose each turn. Planning's also names what funded departments are about to build
    (_leave_line) and the first buildings blocked for lack of land (_land_line), and says high density is locked while
    it is (smoke v9-v11: 6 x zoning high density before 1,000 residents, repeated across turns)."""
    w = env.world
    dense = [] if role != "planning" or w.unlocked("high_density") else [
        f"- high density: locked until {C.UNLOCK_POP['high_density']:,} residents (now "
        f"{w.metrics()['population']:,}): zone with density \"low\""]
    land = []
    if role == "planning":
        depts = dept_lists(env)                                    # one observation per priced department
        land = _leave_line(w, depts) + _land_line(env, depts)
    return (["## Ready sites"] + ready_lines(role, lists, scarce_bank(w), prices, funds) + land + dense
            + [f"find_sites <purpose> lists up to {MAX_SITES} with details"])


def _leave_line(w, depts: dict) -> list[str]:
    """Planning's line naming what funded departments are about to build (funded_firsts) and where, cut at
    BLOCKED_LINE_CHARS: "- leave free (funded): fire station (12,23), clinic (20,27): zone around them". Smoke v14
    seed 0, months 3-6: the mayor funded fire, health and education for their ready sites near homes, planning (acting
    next) zoned those lots, and they read "nothing needs it right now" at coverage 0; fire held $2,500 to month 12."""
    found = [f"{pr.building} ({lists[p].sites[0]['x']},{lists[p].sites[0]['y']})"
             for d, lists in depts.items() for p, pr in funded_firsts(w, d, lists).items()]
    if not found:
        return []
    head, tail = "- leave free (funded): ", ": zone around them"
    return [head + cut(", ".join(found), BLOCKED_LINE_CHARS - len(head + tail)) + tail]


def _land_line(env, depts: dict) -> list[str]:
    """Planning's line naming each department's cheapest first buildings blocked for lack of land or roads
    (land_blocked), by the names the mayor sees, each tagged with what it needs beyond a roadside lot (land_fix:
    "sewage outlet: a road to the river"); past BLOCKED_LINE_CHARS without the tags, then departments only. Smoke v13
    seed 0, months 6-9: planning zoned every roadside tile despite its playbook's "leave room", and utilities, police,
    fire, health and education all read "none - no free roadside land: ask planning for a lot"."""
    found = land_blocked(env, depts)
    if not found:
        return []
    w = env.world
    head, tail = "- blocked for lack of land: ", ": extend a road into open land, or free a lot, for them"
    name = lambda d, p, tag: site_price(w, d, p).building + (f": {fix}" if tag and (fix := land_fix(w, d, p)) else "")
    forms = [", ".join(f"{d} ({', '.join(dict.fromkeys(name(d, p, tag) for p in ps))})" for d, ps in found.items())
             for tag in (True, False)] + [", ".join(found)]
    body = next((f for f in forms if len(head + f + tail) <= BLOCKED_LINE_CHARS), forms[-1])
    return [head + cut(body, BLOCKED_LINE_CHARS - len(head + tail)) + tail]


def _needs(m: DeptMoney) -> str:
    """The mayor's Departments line suffix when a department's funds are below its cheapest ready build
    (first_build): "; needs $X for a <building> (N ready sites)", else "". Every building at that price is named ("a
    clinic (5 ready sites) or a landfill (4 ready sites)"): the mayor writes what it reads into the plan (smoke v9:
    "Build first shelter")."""
    b = m.needs
    return "" if b is None else f"; needs {money(b.cost)} for " + " or ".join(
        f"{what} ({n} ready site{'s' * (n != 1)})" for what, n in b.named.items())


def _dept_line(d: str, m: DeptMoney, f: Funding, budgets: dict) -> str:
    """One department on the mayor's Departments block: funds, the plan target ("<" when below it and it would use a
    top-up: funding_gaps), what it spent (the block's header says over how many months), then one of its idle money
    or what is blocked (summary lines say what to do), or what it needs. Smoke v9-v12: nothing showed what a
    department spent, and planning, spending about $1,000 a quarter, read "funds $7,834 < target $9,000" and was
    topped up."""
    return (f"- {d}: funds {money(math.floor(m.funds))}"
            + (f"{' <' if d in f.gaps else ','} target {money(budgets[d])}" if d in budgets else "")
            + (f", spent {money(m.spent)}" if m.months else "")
            + (f"; idle {money(m.idle)}" if m.idle else _needs(m) or _blocked(m)))


def _blocked(m: DeptMoney) -> str:
    """The mayor's line suffix when a department has no ready site but a first building blocked for lack of land or
    roads: "; blocked: a police station" (BLOCKED_NOTE, once below, says to ask planning; planning's Ready sites say
    why and what each needs). Smoke v13 seed 0, month 6: police, fire, health and education held their first
    building's price with every roadside lot zoned, and the mayor took it back instead of pushing planning."""
    return f"; blocked: {' or '.join(m.blocked)}" if m.blocked else ""


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
    who = "Mayor" if role == "mayor" else ROLES[role]["title"]
    return [f"# {who}: {_date(env)}",
            f"Scores: city {s.get('city', 0):.1f}, yours {s.get(role, 0):.1f} | mode: {mode} | "
            f"up to {env.cfg.max_actions_per_turn} successful actions this turn", ""]


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
             f"{k} {num(o[k]['supply'])} / {num(o[k]['demand'])} ({pct(o[k]['coverage_pct'])})"
             for k in ("power", "water", "sewage")),
         "- coverage: " + ", ".join(f"{k} {pct(v)}" for k, v in o["coverage_pct"].items()),
         "- taxes: " + ", ".join(f"{k} {v:g}%" for k, v in o["tax_pct"].items()),
         "", "## Finances",
         f"- treasury {money(o['treasury'])}{ch('treasury', money)}, debt {money(o['debt'])}; last month income "
         f"{money(o['last_month']['income'])}, expenses {money(o['last_month']['expenses'])}"]
    if w.months_negative or w.low_approval_months:
        L.append(f"- WARNING: negative treasury {months(w.months_negative)} ({C.BANKRUPTCY_MONTHS} = bankruptcy), "
                 f"approval below {100 * C.RECALL_APPROVAL:.0f}% {months(w.low_approval_months)} "
                 f"({C.RECALL_MONTHS} = recall)")
    L += [""] + _digest(env, fleet, "mayor")
    for off in w.offers:
        L.append(f"- OPEN OFFER #{off['id']}: large factory, {off['jobs']} jobs, {money(off['bonus'])} bonus, heavy "
                 f"pollution (accept_offer / decline_offer before month {off['expires']})")
    p = plan_slice(fleet, "mayor")
    plan = bool(fleet.plan_versions)
    budgets, priorities = (p["budgets"], p["priorities"]) if plan else ({}, {})
    latest = {e["role"]: e for e in fleet.log if e["kind"] == "report" and e["role"] in DEPARTMENTS}
    f = funding_gaps(env, fleet)
    clipped = any(len(" ".join(e["text"].split())) > REPORT_PREVIEW_CHARS for e in latest.values())
    spent = min(len(w.history), SPEND_MONTHS)
    notes = [f"spent = last {months(spent)}"] * bool(spent) + ["cut reports: history kind=report"] * clipped
    L += ["", "## Departments" + (f" ({'; '.join(notes)})" if notes else "")]
    for d, m in f.depts.items():
        L.append(_dept_line(d, m, f, budgets)
                 + (f"; priority: {cut(priorities[d], PRIORITY_CHARS)}" if priorities.get(d) else ""))
        if d in latest:
            L.append(f"  report (m{latest[d]['month']}): {cut(latest[d]['text'], REPORT_PREVIEW_CHARS)}")
    if not latest:
        L.append("- no reports yet")
    if f.idle:
        L.append(f"- {f.idle_note()}")
    if any(m.blocked for m in f.depts.values() if not m.idle):          # a line above says "; blocked: ..."
        L.append(BLOCKED_NOTE)
    if f.gaps:
        L.append(f"- {len(f.gaps)} below target ({money(f.total)} short; treasury {money(w.treasury)}"
                 + (f", at or under the {money(f.reserve)} reserve): {HOLD}" if f.hold else
                    f", {money(f.spare)} above the {money(f.reserve)} reserve): allocate_budget" if f.short else
                    "): allocate_budget"))
    L += [""]
    if not plan:
        L.append("## City plan: no plan yet - write one with update_plan (goals, budgets, priorities).")
    else:
        L += [f"## City plan (v{p['version']}, month {p['month']}; full text: get_plan)",
              f"- goals: {_goals(p['goals'], GOAL_CHARS)}"]
        if p["notes"]:
            L.append(f"- notes: {cut(p['notes'], NOTES_CHARS)}")
    return L + [""] + _map(w, "mayor") + [""] + _memory(fleet, "mayor")


def _department(env, fleet: FleetState, role: str) -> list[str]:
    w = env.world
    dept = ROLES[role]["budget"]
    p = plan_slice(fleet, role)
    if not fleet.plan_versions:
        L = ["## Your plan slice", "- No city plan yet: follow your mission and ask the mayor for a budget and "
             "priorities (send_message)."]
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
    # smoke runs: builds tried with $0, stations funded $1,000 of $2,500. Only when funds are short of every ready
    # price (a supplementary one too, else it's false); it names the cheapest primary ones (all on a tie: health's
    # clinic or landfill), else a supplementary one
    purposes, low = cheapest(prices) or cheapest(prices, supplementary=True) or ((), None)
    if low and all(short_of(funds, pr.cost) for pr in prices.values()):
        L.append(f"- You can't pay for any ready site (cheapest: {' or '.join(purposes)} {low.text()}). Don't call "
                 "build tools this turn: ask the mayor for the amount with a reason (send_message) unless you already "
                 "did, then end_turn.")
    L += ["", "## Key metrics", *(_metric(obs["city"], k) for k in KEY_METRICS[role])]
    L += [""] + _digest(env, fleet, role)
    return L + [""] + _map(w, role) + [""] + _ready_sites(env, role, lists, prices, funds) + [""] + _memory(fleet, role)


def system_prompt(role: str, cfg: FleetAgentConfig, total_months: int, *, act_every: int = 1,
                  max_actions: int | None = None, target_pop: int | None = None) -> str:
    """Fixed "how to play" rules, the role's mission and the game rules, then the coach's playbooks (fleet, then
    the role's; empty ones left out). Callers pass `act_every=env.cfg.act_every`,
    `max_actions=env.cfg.max_actions_per_turn` (the successful-action cap) and
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
    L += ["", _milestone(env, role)]
    if mark:
        fleet.mark_turn(role, env.world.month, env.council)
    return "\n".join(L)
