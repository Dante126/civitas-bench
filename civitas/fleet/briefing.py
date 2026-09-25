"""Fleet prompts: the system prompt (fixed rules + coach playbooks) and the per-turn briefing (first user message).

`system_prompt()` is the same every turn: how to play, the role's mission and the game rules, then the coach's
playbooks, so coaches steer strategy but cannot change rules or tools. `briefing()` is a compact status for the
turn: the mayor gets the city, finances, department reports and the plan; a department gets its plan slice,
budget, key metrics, a "since your last turn" digest and its memory. Everything else is a tool call away.
"""
from __future__ import annotations

import calendar
import json

from ..agents.prompts import GAME_RULES
from ..bench.scoring import ScoringConfig
from ..core.observation import build_observation
from ..core.roles import ROLES, TURN_ORDER
from ..sim import catalog as C
from .config import FleetAgentConfig
from .memory import DEPARTMENTS, FleetState
from .tools import MAX_ENTRIES, SITE_PURPOSES, overview, plan_slice

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
REPORT_CHARS = 200
LINE_CHARS = 120
MAX_LINES = 6            # events / failed actions listed per briefing
MAX_UNLOCKS = 4          # unlock names on the milestone line
# GAME_RULES was written for the map-reading LLMAgent; fleet agents read the map with tools
RULE_EDITS = [("highway tile H on the west\n  edge", "highway on the west\n  edge"),
              ("ask for it on the council board.", "ask for it on the council board (post with send_message, read "
                                                   "with get_messages).")]

HOW_TO_PLAY = """\
# How to play
You are the {title} in a fleet of AI agents that runs a city. The mayor is the manager: it sets the city plan \
(goals, a budget target and a priority per department) and moves the money; the departments ({depts}) carry the \
plan out. {cadence}, and you start each turn from a short briefing.
- Each turn you work by calling tools. Look things up before acting ({lookups}).{coords}
- Action tools run immediately and report success or failure, the cost and the money left. When an action fails, \
read the error and fix the cause (another site, a cheaper option, wait for an unlock) instead of repeating it.
- {budget}
- {team}
- You remember nothing between turns except the plan, your memory (update_memory: commitments, open tasks, \
lessons) and the game's history log. Write down what you learn.
- When you are done, call end_turn with a short report (what you did, what you need).
- Limits per turn: at most {max_steps} tool calls; at that limit the turn ends and completed actions stay. \
{actions}
- The mode in your briefing header: autonomous = no coach will intervene during the game; supervised = your coach \
may revise the playbook between sessions."""
LOOKUPS = ("city_overview", "inspect", "find_sites", "my_facilities", "unlocks", "get_plan", "get_messages", "history")

BUDGET = {"mayor": "Stay solvent: upkeep is paid from the treasury every month, so keep a reserve when you move money "
                   "to departments (allocate_budget) or cut taxes.",
          "dept": "Stay within your budget: you can only spend your department's funds, which the mayor allocates."}
TEAM = {"mayor": "Write the plan with update_plan and fund it with allocate_budget; departments only see their "
                 "slice. Read their reports and adjust the plan when the numbers move.",
        "dept": "Follow your slice of the plan (budget target, priority, city goals). To get something from the "
                "mayor or another department, ask with send_message."}


def _cut(text, n: int = LINE_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 3].rstrip() + "..."


def _num(v) -> str:
    if v is None:
        return "n/a"
    if abs(v) >= 10000:
        return f"{v / 1000:.1f}".removesuffix(".0") + "k"
    return f"{v:,.0f}" if float(v).is_integer() or abs(v) >= 100 else f"{v:,.2f}"


def _money(v) -> str:
    return ("-" if v < 0 else "") + "$" + _num(round(abs(v)))


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.0f}%" if abs(v) >= 10 or v == 0 else f"{v:.1f}%"


def _signed(v, fmt=_num) -> str:
    return ("+" if v >= 0 else "-") + fmt(abs(v))


def _value(key: str, v) -> str:
    if isinstance(v, dict) and "supply" in v:
        return f"supply {_num(v['supply'])} / demand {_num(v['demand'])}, coverage {_pct(v['coverage_pct'])}"
    if isinstance(v, dict):
        fmt = (lambda x: f"{x:+.2f}") if key == "demand" else _num
        return ", ".join(f"{k} {fmt(x)}" for k, x in v.items()) or "n/a"
    return _pct(v) if key.endswith("_pct") else _num(v)


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
            f"{season}): {env.cfg.months - w.month} months left")


def _action(e: dict) -> str:
    a = dict(e.get("action") or {})
    kind = a.pop("type", "?")
    args = json.dumps(a, separators=(",", ":"))[1:-1] if a else ""
    return _cut(f"m{e['month']} {kind}({args}): {e.get('message', '')}")


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
    events = [_cut(t) for t in d["events"]]
    L.append("- events:" + ("" if events else " none"))
    L.extend(_listed(events, "history kind=event"))
    for f in w.forecast():
        L.append(f"- forecast: {_cut(f)}")
    n = d["messages"]
    shows = f"get_messages shows the latest {MAX_ENTRIES}" if n > MAX_ENTRIES else "get_messages"
    L.append(f"- {n or 'no'} new council message{'' if n == 1 else 's'}" + (f" ({shows})" if n else ""))
    return L


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
            f"{_money(nm['reward'])} reward: {what} (see unlocks).")


def _goals(goals: list[str], n: int | None = None) -> str:
    return "; ".join(f"{i}) {_cut(g, n) if n else g}" for i, g in enumerate(goals, 1)) or "none"


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
    ch = lambda k, fmt=_num: f" ({_signed(t[k], fmt)})" if t else ""
    L = ["## City" + (f" (change since month {t['since_month']})" if t else ""),
         f"- population {_num(o['population'])}{ch('population')}, happiness {_pct(o['happiness_pct'])}"
         f"{ch('happiness_pct', lambda v: f'{v:.1f} pts')}, approval {_pct(o['approval_pct'])}, "
         f"unemployment {_pct(o['unemployment_pct'])}",
         "- jobs: " + ", ".join(f"{k} {_num(v)}" for k, v in o["jobs"].items())
         + " | demand: " + _value("demand", o["demand"]),
         *(f"- {k}: {_value(k, o[k])}" for k in ("power", "water", "sewage")),
         "- coverage: " + ", ".join(f"{k} {_pct(v)}" for k, v in o["coverage_pct"].items()),
         "- taxes: " + ", ".join(f"{k} {v:g}%" for k, v in o["tax_pct"].items()),
         "", "## Finances",
         f"- treasury {_money(o['treasury'])}{ch('treasury', _money)}, debt {_money(o['debt'])}; last month income "
         f"{_money(o['last_month']['income'])}, expenses {_money(o['last_month']['expenses'])}"]
    if w.months_negative or w.low_approval_months:
        L.append(f"- WARNING: {w.months_negative} month(s) with a negative treasury ({C.BANKRUPTCY_MONTHS} = "
                 f"bankruptcy), {w.low_approval_months} month(s) of approval below {100 * C.RECALL_APPROVAL:.0f}% "
                 f"({C.RECALL_MONTHS} = recall)")
    L += [""] + _digest(env, fleet, "mayor")
    for off in w.offers:
        L.append(f"- OPEN OFFER #{off['id']}: large factory, {off['jobs']} jobs, {_money(off['bonus'])} bonus, heavy "
                 f"pollution (accept_offer / decline_offer before month {off['expires']})")
    p = plan_slice(fleet, "mayor")
    plan = bool(fleet.plan_versions)
    budgets, priorities = (p["budgets"], p["priorities"]) if plan else ({}, {})
    latest = {e["role"]: e for e in fleet.log if e["kind"] == "report" and e["role"] in DEPARTMENTS}
    L += ["", "## Departments: funds, plan target, priority, latest report"]
    for d in (r for r in DEPARTMENTS if r in env.roles):
        L.append(f"- {d}: funds {_money(w.dept_funds[d])}" + (f", target {_money(budgets[d])}" if d in budgets else "")
                 + (f"; priority: {_cut(priorities[d], 60)}" if priorities.get(d) else ""))
        if d in latest:
            text = _cut(latest[d]["text"], REPORT_CHARS)
            cut = len(" ".join(latest[d]["text"].split())) > REPORT_CHARS
            L.append(f"  report (m{latest[d]['month']}): {text}" + (" (full: history kind=report)" if cut else ""))
    if not latest:
        L.append("- no reports yet")
    L += [""]
    if not plan:
        L.append("## City plan: no plan yet - write one with update_plan (goals, budgets, priorities).")
    else:
        L += [f"## City plan (v{p['version']}, month {p['month']}; full text: get_plan)",
              f"- goals: {_goals(p['goals'], 100)}"]
        if p["notes"]:
            L.append(f"- notes: {_cut(p['notes'], 200)}")
    return L + [""] + _memory(fleet, "mayor")


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
             f"- budget target: {_money(target) if target is not None else 'none set'}",
             f"- priority: {p['priority'] or 'none set'}", f"- city goals: {_goals(p['goals'])}"]
        if p.get("notes"):
            L.append(f"- mayor's notes: {_cut(p['notes'], 200)}")
    spent = w.history[-1]["dept_spent"].get(dept, 0) if w.history else 0
    L += ["", "## Your budget",
          f"- {_money(w.dept_funds[dept])} in department funds" + (f" (plan target {_money(target)})"
                                                                   if target is not None else "")
          + f"; spent last month {_money(spent)}"]
    city = build_observation(w, role, council_msgs=[], memory={}, scores={}, months_total=env.cfg.months,
                             include_map=False, site_hints=False)["city"]
    L += ["", "## Key metrics", *(_metric(city, k) for k in KEY_METRICS[role])]
    L += [""] + _digest(env, fleet, role)
    return L + [""] + _memory(fleet, role)


def system_prompt(role: str, cfg: FleetAgentConfig, months: int, *, act_every: int = 1,
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
                 lookups=", ".join(t for t in LOOKUPS if t != "find_sites" or SITE_PURPOSES[role]),
                 coords="" if role == "mayor" else " Coordinates: x = column (0 = west), y = row (0 = north).",
                 max_steps=cfg.max_steps, actions=cap + " (look-ups still work; call end_turn)."),
             f"# Your role: {spec['title']}\n{spec['mission']}",
             "# Game rules\n" + rules.format(turn_order=" -> ".join(TURN_ORDER), months=months,
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
