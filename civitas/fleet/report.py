"""End-of-run fleet report: scoreboard, the plan, per-role activity and coaching hints, as JSON and Markdown.

`run_episode` keeps one slim `turn_row(rec)` per fleet turn (the same rows can be rebuilt from
`trajectories.jsonl` with `rows_from_trajectories`) and writes `fleet_report.json` / `fleet_report.md`; a region run
gets one report with the region scoreboard and a section per city with fleet turns (`region_report`).
Hint rules: >= 3 failures of one role with one cause (failed actions and tool errors; CAUSES groups wordings); funds
above max(2x the plan budget, $1,000) for 3+ consecutive months; 2+ consecutive department turns alike without a
successful action spanning 6+ months (`_turn_kind`, from what the turn's briefing showed): funds that covered a ready
site something needed (idle: the department's hint), funds short of its cheapest such site (starved: one hint for
the mayor), none ready but some blocked for lack of land or roads (no_land: one hint for planning), nothing needed
(none); a negative city treasury; > 20% of turns ended on the step or token limit.
With the run folder (`run_dir`, which the runner passes) the report also scores how the departments worked together,
from frames.jsonl through bench/collab.py (`collaboration`: report only, never scored or shown to agents); the viewer's
Team tab shows it as of each month (`collab_score`, the collab endpoint's `score`).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from ..bench.collab import OPENERS, run_collab
from ..sim import catalog as C
from .fmt import REPORT_PREVIEW_CHARS, cut, money, pct
from .memory import DEPARTMENTS, FleetState
from .tools import ALL_ACTIONS, BANK_BUSY, OFF_TIER_SPECS, ROAD_NAME, SITE_BUILDS, SITE_PURPOSES, SUPPLEMENTARY, \
    SUPPLY_OF, building_name

LIMIT_REASONS = ("max_steps", "turn_tokens")
ERROR_REASONS = ("api_error", "briefing_error")     # turns cut short by an error, not the model
REPEATED_FAILURES = 3
FUNDS_MULTIPLE, FUNDS_MONTHS, UNUSED_FLOOR = 2.0, 3, 1000.0
IDLE_TURNS, IDLE_MONTHS = 2, 6
LIMIT_SHARE = 0.2
MSG_CHARS = 160
MAX_NAMED = 3               # departments detailed in the mayor's and planning's hints; the rest by name
# failure wordings -> one cause each (first match), with the lesson to write down. Scripted fleet, seeds 0-4: planning's
# failures gave 3-4 hints a run, e.g. seed 0: 15 x a plaza spot off the road, and 7 + 5 + 3 x zones, parks and a plaza
# on taken land in three wordings. Other wordings are a cause of their own (_key)
CAUSES = (("taken land", re.compile(r"is not an empty|no empty land|is occupied by"),
           "inspect a tile or use find_sites before building or zoning on it"),
          ("no road access", re.compile(r"does not touch a road|is not on a road"),
           "build only beside a connected road (find_sites lists such sites)"),
          ("money", re.compile(r"treasury only has|only holds \$|budget only has"),
           "check the money it holds (the treasury, for the mayor) before spending"))
LESSON = "add a lesson to its playbook (e.g. check find_sites / inspect before acting)"
# The idle rule: at or above COVERED (%) coverage a department has nothing it must build; a supply is short from
# BANK_BUSY used (what the leave-free line counts for utilities) or under COVERED coverage. 80: the playbooks build
# clinics under 80%, stations under 85%, schools under 90%. Scripted fleet, seed 0: fire held $2,567-3,000 for a ready
# station at 85-98% coverage, and read idle
COVERED = 80.0
# (role, purpose) -> the Key metrics lines that say whether its building is needed ((role, "*"): the role's other
# purposes); a purpose without one is needed whenever it lists sites (bus stops only list from 1,500 residents, levees
# with a flood forecast). Utilities' by the supply each adds to (SUPPLY_OF); a congested stretch while any road is
# congested (transport's list ranked stretches with 0% congested: scripted seed 0, months 42-66)
NEED = {**{("utilities", p): (supply,) for p, (supply, _) in SUPPLY_OF.items()},
        ("utilities", "riverbank_2x2"): ("power", "water", "sewage"),
        ("transport", "congested_roads"): ("congested road",), ("police", "*"): ("police coverage",),
        ("fire", "station"): ("fire coverage",), ("health", "clinic"): ("health coverage",),
        ("health", "hospital"): ("health coverage",), ("health", "landfill"): ("garbage collection",),
        ("education", "*"): ("school coverage",)}
NEED_LABELS = {label for labels in NEED.values() for label in labels}
NUMBER = r"\d[\d,]*(?:\.\d+)?k?"                   # num(): "55.80", "1,234", "12.5k"
MONEY = rf"-?\${NUMBER}"                            # money(): "$2,500", "$12.5k", "-$40"
BUDGET_LINE = re.compile(rf"- ({MONEY}) in department funds(?: \(plan target ({MONEY})\))?")
# a Ready sites line's price: "$2,500", "$30/tile", planning's road "$50 road + $75 zoning" (the two summed; older
# briefings priced the road alone), then " - you have $X" when short of it
PRICE_END = re.compile(rf"; ({MONEY})(?:/tile| road \+ ({MONEY}) zoning)?( - you have {MONEY})?$")
SUPPLY_LINE = re.compile(rf"supply ({NUMBER}) / demand ({NUMBER}), coverage ([\d.]+)%")
ASKED = re.compile(rf"\${NUMBER}")
# The collaboration score (collaboration()): a request thread's status (bench/collab.py) -> how far it was met; "open"
# threads (window not passed yet) are left out. A component with fewer than COLLAB_MIN cases is n/a
FOLLOW_WEIGHT = {"met": 1.0, "partial": 0.6, "acted": 0.5, "replied": 0.25, "unanswered": 0.0}
ANSWERED = ("met", "partial", "acted", "replied")
COLLAB_MIN = 3
UNANSWERED_SHOWN, UNANSWERED_CHARS = 3, 100


def _tool_error(call: dict) -> str | None:
    """A failed tool call that is not a failed action (those are in `results`): bad arguments, unknown tools, a
    look-up that failed, an action tool refused before the action ran. A repeated failed action or bad call and a
    guard refusal (a bulldoze or demolish that would cut roads off, a duplicate allocation, a take-back below what a
    department keeps, a top-up for another building under the reserve, a self-inflicted loan or one while only other
    buildings wait), refused without running, are counted in `refused` whatever its kind, not here; a call that
    overrode one (an earned confirm: true) ran, and is counted in `overrides` too."""
    res = call.get("result") or {}
    if res.get("ok") is not False or (call.get("tool") in ALL_ACTIONS and "message" in res) or "refused" in res:
        return None
    return str(res.get("error") or res.get("message") or "failed")[:MSG_CHARS]


def turn_row(rec: dict) -> dict:
    """The part of a trajectory record the report needs; `seen` is what its briefing showed (_seen), `asked` who its
    messages went to (_asked)."""
    usage = rec.get("usage") or {}
    trace = rec.get("tool_trace") or []
    errors = [_tool_error(c) for c in trace]
    marked = lambda k: sum(k in (c.get("result") or {}) for c in trace)
    return {"role": rec["role"], "month": rec["month"],
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "steps": rec.get("steps") or 0, "finish_reason": rec.get("finish_reason"),
            "length_stops": rec.get("length_stops") or 0, "ended": rec.get("ended"),
            "parse_error": rec.get("parse_error"),
            "results": [{"type": (r.get("action") or {}).get("type"), "ok": bool(r.get("ok")),
                         "message": str(r.get("message") or "")[:MSG_CHARS], "cost": r.get("cost") or 0}
                        for r in rec.get("results") or []],
            "tool_errors": [e for e in errors if e is not None],
            "repeats": marked("repeat"), "refused": marked("refused"),   # repeated look-ups / refused actions
            "overrides": marked("override"),       # guard refusals overridden by an earned confirm: true
            "seen": _safe(_seen, rec, None), "asked": _safe(_asked, rec, {})}


def _safe(read, rec: dict, default):
    """read(rec), or `default` on any error: the runner builds a row every fleet turn, and a briefing or message the
    readers don't expect must not end the run."""
    try:
        return read(rec)
    except Exception:
        return default


def _figure(s: str) -> float:
    """A money() or num() figure as a number: "$2,500" -> 2500, "$12.5k" -> 12500, "-$40" -> -40."""
    s = s.replace("$", "").replace(",", "")
    return float(s[:-1]) * 1000 if s.endswith("k") else float(s)


def _seen(rec: dict) -> dict | None:
    """What a department's briefing (the turn's first user message) showed, for the idle rule: its funds and plan
    target (Your budget), each find_sites purpose's Ready sites state and the Key metrics lines NEED reads. States:
    "ok" (a price its funds cover), "short" (a price with "- you have $X"), "ready" (sites, no price), "blocked" (none
    for lack of land or roads: every such _why_empty reason asks planning; `why` is the part before), "none" (none
    otherwise: nothing needs it, no flood forecast) and "locked". None without a briefing, and for the mayor's (no
    Your budget block)."""
    text = next((m.get("content") for m in rec.get("prompt") or [] if m.get("role") == "user"), None)
    if not isinstance(text, str) or "\n## Your budget\n" not in text:
        return None
    purposes = SITE_PURPOSES.get(str(rec["role"]).split(".")[-1], {})      # region runs: "north.police"
    out, section = {"funds": None, "target": None, "sites": {}, "metrics": {}}, None
    for ln in text.splitlines():
        if ln.startswith("## "):
            section = ln[3:]
        elif not ln.startswith("- "):
            continue
        elif section == "Your budget" and (m := BUDGET_LINE.match(ln)):
            out["funds"], out["target"] = _figure(m[1]), (_figure(m[2]) if m[2] else None)
        elif section == "Key metrics":
            label, _, value = ln[2:].partition(": ")
            if label in NEED_LABELS:
                if m := SUPPLY_LINE.fullmatch(value):
                    out["metrics"][label] = [_figure(m[1]), _figure(m[2]), float(m[3])]
                elif m := re.fullmatch(r"([\d.]+)%", value):
                    out["metrics"][label] = float(m[1])
        elif section == "Ready sites":
            purpose, _, rest = ln[2:].partition(": ")
            if purpose in purposes:
                out["sites"][purpose] = _site_state(rest)
    return out


def _site_state(rest: str) -> dict:
    """One Ready sites line after its "- purpose: " (briefing.ready_lines) as _seen reads it; a reason ending in a
    price (planning's "none - no dead end; branch: ...; $150 road + $375 zoning") is priced like a list."""
    if rest.startswith("locked"):
        return {"state": "locked"}
    m = PRICE_END.search(rest)
    if rest.startswith("none - ") and m is None:
        why = rest[len("none - "):]
        return {"state": "blocked", "why": why.split(": ask planning")[0]} if "ask planning" in why else {
            "state": "none"}
    return {"state": "ready"} if m is None else {"state": "short" if m[3] else "ok",
                                                 "price": _figure(m[1]) + (_figure(m[2]) if m[2] else 0)}


def _asked(rec: dict) -> dict[str, float]:
    """Who the turn's messages went to -> the largest dollar figure in them (0 for none): "Police needs $2,500 for a new
    building (police cover 0.0%)"."""
    out: dict[str, float] = {}
    for m in rec.get("messages") or []:
        to = str(m.get("to", ""))
        out[to] = max([out.get(to, 0.0), *(_figure(f) for f in ASKED.findall(str(m.get("text", ""))))])
    return out


def rows_from_trajectories(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if str(rec.get("agent", "")).startswith("fleet:"):
                    rows.append(turn_row(rec))
    return rows


def _key(msg: str) -> str:
    """Failure messages alike up to numbers and coordinates share a key (their first clause)."""
    msg = re.sub(r"^\s*[A-Za-z_][\w.]*(Error|Exception)\s*:\s*", "", msg)
    return re.split(r"[;:]", re.sub(r"-?\$?\d[\d,.]*", "#", msg.lower()), maxsplit=1)[0].strip()[:60]


def _failure_groups(msgs: list[str]) -> list[dict]:
    groups: dict[str, list[str]] = {}
    for m in msgs:
        groups.setdefault(_key(m), []).append(m)
    return sorted(({"message": v[0], "count": len(v)} for v in groups.values()), key=lambda g: -g["count"])


def _runs(items: list[tuple], flagged) -> list[list[tuple]]:
    """Runs of `(month, ...)` items (sorted by month) where `flagged(item)` holds on consecutive months."""
    out, cur = [], []
    for it in items:
        if cur and not (flagged(it) and it[0] == cur[-1][0] + 1):
            out.append(cur)
            cur = []
        if flagged(it):
            cur.append(it)
    return out + [cur] if cur else out


def _span(a: int, b: int) -> str:
    return f"month {a}" if a == b else f"months {a}-{b}"


def _role_stats(role: str, rows: list[dict], fleet: FleetState) -> dict:
    results = [r for row in rows for r in row["results"]]
    fails = [r["message"] for r in results if not r["ok"]]
    tool_errs = [e for row in rows for e in row["tool_errors"]]
    groups = _failure_groups(fails + tool_errs)
    return {"turns": len(rows), "tool_calls": sum(r["steps"] for r in rows), "actions": len(results),
            "failed": len(fails), "tool_errors": len(tool_errs),
            "repeats": sum(r["repeats"] for r in rows), "refused": sum(r["refused"] for r in rows),
            "overrides": sum(r.get("overrides", 0) for r in rows),
            "spent": round(sum(r["cost"] for r in results if r["ok"]), 2),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "limit_turns": sum(r["finish_reason"] in LIMIT_REASONS for r in rows),
            "length_stops": sum(r["length_stops"] for r in rows),
            "api_errors": sum(r["finish_reason"] in ERROR_REASONS for r in rows),
            "top_failures": groups[:max(3, sum(g["count"] >= REPEATED_FAILURES for g in groups))],
            "lessons": list((fleet.memory.get(role) or {}).get("lessons", [])),
            "last_report": cut(fleet.reports[role], REPORT_PREVIEW_CHARS) if fleet.reports.get(role) else None}


def _longest(turns: list[tuple[int, str]], kind: str, months: int) -> tuple[int, int, int] | None:
    """Longest (turns, first month, end month) run of consecutive `(month, kind)` turns that fires the rule; a run
    lasts until the department's next turn (or the end of the run)."""
    best, i = None, 0
    while i < len(turns):
        if turns[i][1] != kind:
            i += 1
            continue
        j = i
        while j + 1 < len(turns) and turns[j + 1][1] == kind:
            j += 1
        end = turns[j + 1][0] if j + 1 < len(turns) else months
        n, a = j - i + 1, turns[i][0]
        if n >= IDLE_TURNS and end - a >= IDLE_MONTHS and (best is None or end - a > best[2] - best[1]):
            best = (n, a, end)
        i = j + 1
    return best


def _short(label: str, v) -> bool:
    """A Key metrics value (_seen) saying its building is needed: a supply BANK_BUSY used or under COVERED coverage,
    any congested road, a coverage under COVERED."""
    if isinstance(v, list):
        supply, demand, coverage = v
        return demand >= BANK_BUSY * supply or coverage < COVERED
    return v > 0 if label == "congested road" else v < COVERED


def _needed(role: str, purpose: str, metrics: dict) -> bool:
    labels = NEED.get((role, purpose), NEED.get((role, "*"), ()))
    shown = [(k, metrics[k]) for k in labels if k in metrics]
    return not shown or any(_short(k, v) for k, v in shown)


def _turn_kind(role: str, row: dict, funds: float | None, target: float | None) -> tuple[str, dict]:
    """(kind, what) of a department turn: "acted" (a successful action), else from its briefing (row["seen"]): "idle"
    (its funds covered a ready site of a first or upgrade building it needed: `what` its purpose and price), "starved"
    (short of the cheapest such site: that one), "land" (none ready, some blocked for lack of land or roads: purpose ->
    why) or "none" (nothing needed it: coverage at COVERED+, "nothing needs it", locked); SUPPLEMENTARY
    purposes never count. Planning goes by its priced road_extension line (a road into open land and zoning it, the
    two summed: idle when its funds cover them, starved when short); without one planning, and a turn without a
    briefing (funds and target from the history log and the plan), go by the funds alone: idle with $1+ (it can still
    zone), under it starved against a plan target, else none. Live 120-month run, seed 0, months 9-12: planning at $0
    short of its $150 extension."""
    if any(r["ok"] for r in row["results"]):
        return "acted", {}
    seen = row.get("seen")
    if seen and seen["funds"] is not None:
        funds = seen["funds"]
    broke = funds is not None and funds < 1
    road = seen["sites"].get("road_extension", {}) if seen and role == "planning" else {}
    if "price" in road:
        return "idle" if road["state"] == "ok" else "starved", {"purpose": "road_extension", "price": road["price"]}
    if seen is None or role == "planning":
        return ("starved" if target else "none", {}) if broke else ("idle", {})
    wanted = {p: s for p, s in seen["sites"].items() if p not in SUPPLEMENTARY and _needed(role, p, seen["metrics"])}
    priced = lambda state: sorted((s.get("price", 0.0), p) for p, s in wanted.items() if s["state"] == state
                                  or (s["state"] == "ready" and (state == "short") == broke))
    for kind, state in (("idle", "ok"), ("starved", "short")):
        if found := priced(state):
            return kind, {"purpose": found[0][1], "price": wanted[found[0][1]].get("price")}
    if blocked := {p: s["why"] for p, s in wanted.items() if s["state"] == "blocked"}:
        return "land", {"blocked": blocked}
    return "none", {}


def _cost(name: str) -> float:
    if name in OFF_TIER_SPECS:
        return OFF_TIER_SPECS[name][0]()
    line, tier = C.TIER_BY_NAME[name]
    return C.TIER_LINES[line][tier - 1]["cost"]


def _building(role: str, purpose: str, price: float | None = None) -> str:
    """What a purpose's sites are for, by name: its SITE_BUILDS entry at `price`, else its cheapest (the purpose for
    those without one: planning's; its road_extension a road into open land and zoning it, ROAD_NAME)."""
    names = SITE_BUILDS.get(role, {}).get(purpose, ())
    if not names:
        return ROAD_NAME if (role, purpose) == ("planning", "road_extension") else purpose.replace("_", " ")
    return building_name(next((n for n in names if _cost(n) == price), None) or min(names, key=_cost))


def _held(values: list[float]) -> str:
    lo, hi = math.floor(min(values)), math.floor(max(values))
    return money(lo) if lo == hi else f"{money(lo)}-{money(hi)[1:]}"


def _range(values: list[float], fmt=pct) -> str:
    lo, hi = min(values), max(values)
    return fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo).rstrip('%')}-{fmt(hi)}"


def _metric_notes(role: str, purpose: str, run: list[dict]) -> list[str]:
    """The Key metrics that made `purpose` needed over a run, as ranges: "police coverage 30%", "power demand 93% of
    supply"."""
    out = []
    for k in NEED.get((role, purpose), NEED.get((role, "*"), ())):
        vals = [t["seen"]["metrics"][k] for t in run if k in t["seen"]["metrics"]]
        if vals and isinstance(vals[0], list):
            used = [100 * d / s for s, d, _ in vals if s > 0]
            out += [f"{k} demand {_range(used)} of supply"] if used else [f"no {k} supply"]
        elif vals:
            out.append(f"{k} {_range(vals)}")
    return out


def _dept_turns(dept: str, rows: list[dict], history: list[dict], target) -> list[dict]:
    """A department's turns in month order, each with its kind (_turn_kind), its funds and target at the turn: the
    briefing's, else the history log's (a turn in month t ends in history entry t + 1, after the mayor's turn, which
    comes first, and its own; only actions move dept funds, so after a turn without a success the funds there are the
    funds it had at its turn) and the plan in force."""
    after = {m["month"] - 1: (m.get("dept_funds") or {}).get(dept) for m in history}
    out = []
    for row in sorted((r for r in rows if r["role"] == dept), key=lambda r: r["month"]):
        t, seen = row["month"], row.get("seen")
        funds = seen["funds"] if seen and seen["funds"] is not None else after.get(t)
        tgt = seen["target"] if seen else target(dept, t + 1)
        kind, what = _turn_kind(dept, row, funds, tgt)
        out.append({"month": t, "kind": kind, "what": what, "funds": funds, "target": tgt, "seen": seen, "row": row})
    return out


def _run(turns: list[dict], kind: str, months: int) -> tuple[int, int, int, list[dict]] | None:
    found = _longest([(t["month"], t["kind"]) for t in turns], kind, months)
    return found and (*found, [t for t in turns if found[1] <= t["month"] < found[2]])


def _common(run: list[dict], key: str):
    """The most common `what[key]` over a run's turns (the latest on a tie), None when none has it."""
    vals = [t["what"][key] for t in reversed(run) if key in t["what"]]
    return Counter(vals).most_common(1)[0][0] if vals else None


def _idle_hint(dept: str, n: int, a: int, end: int, run: list[dict]) -> dict:
    head = f"{dept}: {n} turns / {end - a} months without "
    funds = [t["funds"] for t in run if t["funds"] is not None]
    purpose = _common(run, "purpose")
    notes = []
    held = f": it held {_held(funds)}" if funds else ""
    if purpose is not None:
        price = next(t["what"]["price"] for t in reversed(run) if t["what"].get("purpose") == purpose)
        head += (f"a build while it could pay for a ready site ({_span(a, end - 1)}){held}"
                 + (f", enough for its {money(price)} {_building(dept, purpose, price)}" if price is not None else ""))
        notes += _metric_notes(dept, purpose, [t for t in run if t["seen"]])
    else:
        head += f"a successful action ({_span(a, end - 1)}){held}"
    asks = [t["row"]["asked"]["mayor"] for t in run if "mayor" in t["row"]["asked"]]
    if asks:
        notes.append("asked the mayor" + (f" for up to {money(max(asks))}" if max(asks) else "")
                     + f" in {len(asks)} of them")
    fails = sum(not r["ok"] for t in run for r in t["row"]["results"])
    if fails:
        notes.append(f"{fails} failed action{'s' * (fails != 1)}")
    return {"kind": "idle", "role": dept, "turns": n, "months": end - a, "first": a, "last": end - 1,
            "text": head + "".join(f"; {x}" for x in notes) + " - "
                    + ("check its playbook and why it didn't build" if purpose is not None else
                       "check its playbook and whether it is blocked (funds, unlocks, failures)")}


def _starved_item(dept: str, n: int, a: int, end: int, run: list[dict]) -> tuple[dict, str]:
    purpose = _common(run, "purpose")
    price = purpose and next(t["what"]["price"] for t in reversed(run) if t["what"].get("purpose") == purpose)
    building = _building(dept, purpose, price) if purpose else None
    asked = sum("mayor" in t["row"]["asked"] for t in run)
    held = _held([t["funds"] for t in run if t["funds"] is not None] or [0.0])
    span = f"for {n} turns / {end - a} months ({_span(a, end - 1)}"
    if price is not None:
        text = (f"{dept} short of its {money(price)} {building} {span}; it held {held} and "
                + (f"asked {asked} time{'s' * (asked != 1)})" if asked else "never asked)"))
    else:
        tgt = next((t["target"] for t in reversed(run) if t["target"]), None)
        text = f"{dept} at {held}" + (f" against its {money(tgt)} target" if tgt else "") + f" {span})"
    return ({"role": dept, "turns": n, "months": end - a, "first": a, "last": end - 1, "building": building,
             "price": price, "asked": asked}, text)


def _land_item(dept: str, n: int, a: int, end: int, run: list[dict]) -> tuple[dict, str]:
    """The cheapest buildings blocked over the run, all on a tie, as planning's blocked line names them (smoke v17
    seed 0 read "police station or jail")."""
    costs = {p: min(map(_cost, SITE_BUILDS[dept][p]), default=0.0) if p in SITE_BUILDS.get(dept, {}) else 0.0
             for t in run for p in t["what"]["blocked"]}
    low = [p for p, c in costs.items() if c == min(costs.values())]
    names = list(dict.fromkeys(_building(dept, p) for p in low))
    why = next(t["what"]["blocked"][p] for t in reversed(run) for p in low if p in t["what"]["blocked"])
    funds = [t["funds"] for t in run if t["funds"] is not None]
    return ({"role": dept, "turns": n, "months": end - a, "first": a, "last": end - 1, "buildings": names, "why": why},
            f"{dept} without a lot for its {' or '.join(names)} for {n} turns / {end - a} months ({_span(a, end - 1)}: "
            f"{why}" + (f"; it held {_held(funds)}" if funds else "") + ")")


def _grouped(kind: str, role: str, items: list[tuple[dict, str]], advice: str) -> list[dict]:
    """One hint for `role` naming every department of `items`, the MAX_NAMED longest runs in full (scripted fleet,
    seed 0: in turn order, utilities' 2 turns were named and education's 13 were not)."""
    if not items:
        return []
    items = sorted(items, key=lambda it: -it[0]["months"])
    texts = [t for _, t in items[:MAX_NAMED]]
    rest = [d["role"] for d, _ in items[MAX_NAMED:]]
    return [{"kind": kind, "role": role, "depts": [d for d, _ in items],
             "text": f"{role}: left " + "; ".join(texts)
                     + (f"; and {len(rest)} more ({', '.join(rest)})" if rest else "") + f" - {advice}"}]


def _turn_hints(roles: dict, rows: list[dict], history: list[dict], months: int, target) -> list[dict]:
    """The idle rule over each department's turns (_dept_turns): its idle hint, and one for the mayor (starved) and
    one for planning (no_land) naming the departments whose runs fired."""
    idle, starved, land = [], [], []
    for dept in (d for d in DEPARTMENTS if d in roles):
        turns = _dept_turns(dept, rows, history, target)
        for kind, out, make in (("idle", idle, _idle_hint), ("starved", starved, _starved_item),
                                ("land", land, _land_item)):
            if found := _run(turns, kind, months):
                out.append(make(dept, *found))
    return (idle
            + _grouped("starved", "mayor", starved, "fund the first buildings departments ask for, or tell them why "
                                                     "not")
            + _grouped("no_land", "planning", land, "leave or free a roadside lot for each, or extend a road into "
                                                    "open land"))


def _failure_hints(roles: dict, rows: list[dict]) -> list[dict]:
    """REPEATED_FAILURES+ failed actions and tool errors of one role with one cause (CAUSES, else wordings alike:
    _key), one hint each, the most frequent first. A cause names its first wording per action (tool errors: per
    wording) with the count: smoke v17 seed 0, planning's build_road into a coal plant, shops, homes and industry read
    as four wordings."""
    hints = []
    for role in roles:
        groups: dict[str, dict] = {}
        fails = [(r["type"], r["message"]) for row in rows if row["role"] == role for r in row["results"] if not r["ok"]]
        fails += [(None, e) for row in rows if row["role"] == role for e in row["tool_errors"]]
        for kind, msg in fails:
            cause = next((c for c in CAUSES if c[1].search(msg.lower())), None)
            g = groups.setdefault(cause[0] if cause else "=" + _key(msg), {"cause": cause, "by": {}})
            g["by"].setdefault(kind or _key(msg), [kind, msg, 0])[2] += 1
        for g in sorted(groups.values(), key=lambda g: -sum(w[2] for w in g["by"].values())):
            ws = sorted(g["by"].values(), key=lambda w: -w[2])
            count, cause = sum(w[2] for w in ws), g["cause"]
            if count < REPEATED_FAILURES:
                continue
            if cause and len(ws) > 1:
                what = f"failures on {cause[0]} (" + ", ".join(
                    f"{n}x {kind + ' ' if kind else ''}\"{msg}\"" for kind, msg, n in ws) + ")"
            else:                                         # one wording, from one action or several
                acts = [f"{n}x {kind}" if len(ws) > 1 else kind for kind, _, n in ws if kind]
                what = f"failures like \"{ws[0][1]}\"" + (f" ({', '.join(acts)})" if acts else "")
            hints.append({"kind": "repeated_failures", "role": role, "cause": cause[0] if cause else None,
                          "count": count, "message": ws[0][1],
                          "text": f"{role}: {count} {what} - "
                                  + (f"add a lesson to its playbook: {cause[2]}" if cause else LESSON)})
    return hints


def _hints(roles: dict, rows: list[dict], fleet: FleetState, history: list[dict], months: int) -> list[dict]:
    hints = _failure_hints(roles, rows)
    versions = [v for v in fleet.plan_versions if v.get("month") is not None]
    def target(dept, m):
        live = [v for v in versions if v["month"] < m]          # history entry m = the state after month m - 1
        return live[-1]["budgets"].get(dept) if live else None
    for dept in (d for d in DEPARTMENTS if d in roles):
        entries = [(m["month"], (m.get("dept_funds") or {}).get(dept), target(dept, m["month"])) for m in history]
        over = lambda e: (e[1] is not None and e[2] is not None
                          and e[1] > max(FUNDS_MULTIPLE * e[2], UNUSED_FLOOR))
        runs = [r for r in _runs(entries, over) if len(r) >= FUNDS_MONTHS]
        if runs:
            run = max(runs, key=len)
            a, b = run[0][0], run[-1][0]
            hints.append({"kind": "unused_money", "role": dept, "months": len(run), "first": a, "last": b,
                          "text": f"{dept}: funds stayed above {FUNDS_MULTIPLE:g}x its plan budget "
                                  f"({money(run[-1][2])}) for {len(run)} months ({_span(a, b)}, up to "
                                  f"{money(max(e[1] for e in run))}) - have it spend, or lower its allocation"})
    hints += _turn_hints(roles, rows, history, months, target)
    neg = [m for m in history if m.get("treasury", 0) < 0]
    if neg:
        low = min(m["treasury"] for m in neg)
        hints.append({"kind": "negative_treasury", "role": "city", "months": len(neg), "first": neg[0]["month"],
                      "text": f"city treasury was negative in {len(neg)} month(s) (first month {neg[0]['month']}, "
                              f"low {money(low)})"
                              + ("; mayor: keep a reserve or allocate less" if "mayor" in roles else "")})
    for role, st in roles.items():
        if st["turns"] and st["limit_turns"] / st["turns"] > LIMIT_SHARE:
            hints.append({"kind": "limit", "role": role, "count": st["limit_turns"], "turns": st["turns"],
                          "text": f"{role}: {st['limit_turns']} of {st['turns']} turns "
                                  f"({100 * st['limit_turns'] / st['turns']:.0f}%) ended on the step or token limit - "
                                  "raise max_steps / turn_tokens or tell it to end_turn sooner"})
    return hints


def _share(part: float, whole: float, n: int) -> float | None:
    """100 x part / whole on 0-100 (1 decimal), None (n/a) with fewer than COLLAB_MIN cases."""
    return round(100 * part / whole, 1) if n >= COLLAB_MIN and whole > 0 else None


def collaboration(c: dict) -> dict:
    """How the departments worked together, from bench/collab.py's collab() of a run (or one city of a region): four
    components on 0-100, each n/a with fewer than COLLAB_MIN cases, and the score, their mean (n/a when all are).
    Only requests between two roles of the game count, and "open" ones (window not passed) are left out.
    - responsiveness: the share of requests answered in the window (met, partial, acted on or replied to);
    - follow-through: the requests weighted by how far they were met (FOLLOW_WEIGHT: met 1, partial 0.6, acted 0.5,
      replied 0.25, unanswered 0);
    - hand-offs: the share of planning's roads, freed lots and cleared land (OPENERS) another department built on
      within the window (collab's handoffs; openings whose window hasn't passed count only once used);
    - money aim: the share of the mayor's top-up dollars that answered the department's money request in the window
      before (part of its thread). Whether a top-up was for a first building or supply can't be told from frames, and
      "went into a build within the window" was ~100% on every 120-month LLM run, so neither counts.
    Per role: requests it sent (asked) and got answered, requests sent to it (asked_of) and it answered, its
    responsiveness (as the one asked) and follow-through (as the one asking), hand-offs (planning: its openings used;
    others: builds on land planning opened). `unanswered`: the UNANSWERED_SHOWN oldest unanswered requests."""
    roles, window, last = list(c["roles"]), c["window"], c["last_month"]
    threads = [t for t in c["threads"] if t["from"] in roles and t["to"] in roles and t["from"] != t["to"]]
    closed = [t for t in threads if t["status"] in FOLLOW_WEIGHT]
    answered = lambda ts: sum(t["status"] in ANSWERED for t in ts)
    weight = lambda ts: sum(FOLLOW_WEIGHT[t["status"]] for t in ts)
    events = c["events"]
    used = {h["event_from"] for h in c["handoffs"]}
    opens = [e for e in events if e["from"] == "planning" and e.get("type") in OPENERS and e.get("rect")
             and (e["id"] in used or e["month"] + window <= last)]
    n_used = sum(e["id"] in used for e in opens)
    gifts = [e for e in events if e["kind"] == "money" and e["from"] == "mayor" and not e.get("take_back")
             and e["to"] in roles and e["to"] != "mayor"]
    given = sum(g["amount"] for g in gifts)
    aimed = sum(g["amount"] for g in gifts if g.get("thread") is not None)
    comps = {
        "responsiveness": {"score": _share(answered(closed), len(closed), len(closed)), "n": len(closed),
                           "answered": answered(closed)},
        "follow_through": {"score": _share(weight(closed), len(closed), len(closed)), "n": len(closed),
                           "by_status": {s: sum(t["status"] == s for t in closed) for s in FOLLOW_WEIGHT}},
        "handoffs": {"score": _share(n_used, len(opens), len(opens)), "n": len(opens), "used": n_used},
        "money_aim": {"score": _share(aimed, given, len(gifts)), "n": len(gifts), "aimed": round(aimed, 2),
                      "given": round(given, 2)}}
    vals = [v["score"] for v in comps.values() if v["score"] is not None]
    per = {}
    for r in roles:
        sent, got = [t for t in threads if t["from"] == r], [t for t in threads if t["to"] == r]
        hand = (n_used if r == "planning" else sum(h["to"] == r for h in c["handoffs"]))
        if not (sent or got or hand):
            continue
        sent_c = [t for t in sent if t["status"] in FOLLOW_WEIGHT]
        got_c = [t for t in got if t["status"] in FOLLOW_WEIGHT]
        per[r] = {"asked": len(sent), "answered": answered(sent), "asked_of": len(got), "answered_for": answered(got),
                  "responsiveness": _share(answered(got_c), len(got_c), len(got_c)),
                  "follow_through": _share(weight(sent_c), len(sent_c), len(sent_c)), "handoffs": hand}
    waiting = sorted((t for t in threads if t["status"] == "unanswered"), key=lambda t: t["month"])
    return {"score": round(sum(vals) / len(vals), 1) if vals else None, "components": comps, "roles": per,
            "requests": len(threads), "open": len(threads) - len(closed), "window": window,
            "unanswered": [{"month": t["month"], "from": t["from"], "to": t["to"],
                            "text": cut(t["text"], UNANSWERED_CHARS)} for t in waiting[:UNANSWERED_SHOWN]]}


def collab_as_of(c: dict, month: int) -> dict:
    """collab()'s dict as a live run had it at frame `month` (a frame carries the turns of the month before it): the
    events up to month - 1, the requests made by then, each answered (its status) once its response is in, else
    unanswered once its window has passed (last_month = month), else open. collaboration() of it is the score the run
    showed at that frame, and at the last frame it is the dict itself. One gap: a money request partly funded by then
    and met later already counts as met."""
    if month >= c["last_month"]:
        return c
    seen, win = month - 1, c["window"]

    def status(t: dict) -> str:
        r = t.get("response")
        if r is not None and r["month"] <= seen:
            return t["status"]
        return "unanswered" if t["month"] + win <= month else "open"
    return {**c, "last_month": month, "threads": [{**t, "status": status(t)} for t in c["threads"] if t["month"] <= seen],
            "events": [e for e in c["events"] if e["month"] <= seen],
            "handoffs": [h for h in c["handoffs"] if h["month_to"] <= seen]}


def collab_score(c: dict) -> dict:
    """The viewer's collaboration score (/api/runs/<id>/collab's `score`): collaboration() of the collab dict, plus
    `series`, the score as of every frame month 0..last_month (collab_as_of), compact: `score` per month, per component
    [score, counts...] (responsiveness [score, answered, n]; follow_through [score, n, met, partial, acted, replied,
    unanswered]; handoffs [score, used, n]; money_aim [score, aimed, given, n]) and per department its responsiveness
    as the one asked [score, answered, n] (n: requests to it answered or past their window)."""
    out = {**collaboration(c), "min_cases": COLLAB_MIN, "follow_weight": FOLLOW_WEIGHT}
    roles = list(out["roles"])
    months = list(range(int(c["last_month"]) + 1))
    ser = {"month": months, "score": [], "responsiveness": [], "follow_through": [], "handoffs": [], "money_aim": [],
           "roles": {r: [] for r in roles}}
    for m in months:
        cm = collab_as_of(c, m)
        co = collaboration(cm)
        cp = co["components"]
        ser["score"].append(co["score"])
        ser["responsiveness"].append([cp["responsiveness"]["score"], cp["responsiveness"]["answered"],
                                      cp["responsiveness"]["n"]])
        ft = cp["follow_through"]
        ser["follow_through"].append([ft["score"], ft["n"], *(ft["by_status"][s] for s in FOLLOW_WEIGHT)])
        ser["handoffs"].append([cp["handoffs"]["score"], cp["handoffs"]["used"], cp["handoffs"]["n"]])
        ma = cp["money_aim"]
        ser["money_aim"].append([ma["score"], ma["aimed"], ma["given"], ma["n"]])
        closed = Counter(t["to"] for t in cm["threads"] if t["status"] in FOLLOW_WEIGHT and t["from"] != t["to"]
                         and t["from"] in c["roles"])
        for r in roles:
            x = co["roles"].get(r) or {}
            ser["roles"][r].append([x.get("responsiveness"), x.get("answered_for", 0), closed[r]])
    out["series"] = ser
    return out


def region_score(rc: dict, cities: dict[str, dict]) -> dict:
    """The Region view's scores (/api/runs/<id>/collab?region=1's `score`): each city's collab_score (`cities`: city ->
    its score dict) as its final score and per-month series, and between the cities, as of every frame month by
    collab_as_of's rule: `asks`, the share of asks to another city answered (met, acted on or replied to) among those
    answered or past their window [score, answered, n], and `deals`, the share of deal offers accepted among those
    accepted, declined or left unanswered past the window (withdrawn ones left out) [score, accepted, n]; each n/a
    under COLLAB_MIN cases."""
    win, last = rc["window"], int(rc["last_month"])
    months = list(range(last + 1))
    asks, deals = [], []
    for m in months:
        th = collab_as_of({**rc, "events": [], "handoffs": []}, m)["threads"]
        ask = [t for t in th if t["kind"] != "deal" and t["status"] != "open"]
        offer = [t for t in th if t["kind"] == "deal" and t["status"] not in ("open", "withdrawn")]
        ok = sum(t["status"] in ANSWERED for t in ask)
        acc = sum(t["status"] == "met" for t in offer)
        asks.append([_share(ok, len(ask), len(ask)), ok, len(ask)])
        deals.append([_share(acc, len(offer), len(offer)), acc, len(offer)])
    per = {c: {"score": s.get("score"), "requests": s.get("requests"),
               "components": {k: v["score"] for k, v in (s.get("components") or {}).items()},
               "series": (s.get("series") or {}).get("score", [])} for c, s in cities.items()}
    return {"cities": per, "window": win, "min_cases": COLLAB_MIN,
            "asks": dict(zip(("score", "answered", "n"), asks[-1])) if asks else None,
            "deals": dict(zip(("score", "accepted", "n"), deals[-1])) if deals else None,
            "series": {"month": months, "asks": asks, "deals": deals}}


def run_collaboration(run_dir: str | Path | None, city: str | None = None) -> dict | None:
    """collaboration() of a run folder's frames.jsonl (one city of a region with `city`); None without one. Works on
    any run folder, with or without the env (the runner's report, or a finished run read afterwards)."""
    if run_dir is None or not (Path(run_dir) / "frames.jsonl").exists():
        return None
    return collaboration(run_collab(run_dir, city))


def fleet_report(env, summary: dict, trajectories: list[dict], run_dir: str | Path | None = None) -> tuple[dict, str]:
    """(report, markdown) from the env after the run, the run summary and `turn_row`s of the fleet turns; with the run
    folder also its collaboration score (run_collaboration; None without). For a region (RegionEnv) one report with a
    section per city that had fleet turns (region_report)."""
    if hasattr(env, "envs"):
        return region_report(env, summary, trajectories, run_dir)
    fleet = getattr(env, "fleet", None) or FleetState()
    months = summary.get("months_played", env.world.month)
    roles = {r: _role_stats(r, [x for x in trajectories if x["role"] == r], fleet)
             for r in env.roles if any(x["role"] == r for x in trajectories)}
    scores = summary.get("final_scores") or {}
    fm = summary.get("final_metrics") or {}
    plan = fleet.plan
    rep = {"run_id": summary.get("run_id"), "seed": summary.get("seed"), "signature": summary.get("signature"),
           "months": months, "months_configured": getattr(getattr(env, "cfg", None), "months", None),
           "scoreboard": {"city": scores.get("city"), "roles": {r: scores.get(r) for r in env.roles},
                          "avg_city": summary.get("avg_city_score"), "population": fm.get("population"),
                          "treasury": fm.get("treasury"), "bankrupt": summary.get("bankrupt")},
           "plan": {**{k: plan.get(k) for k in ("goals", "budgets", "priorities", "notes", "month")},
                    "versions": len(fleet.plan_versions), "changed": [v["month"] for v in fleet.plan_versions]},
           "roles": roles,
           "hints": _hints(roles, trajectories, fleet, env.world.history, months)}
    try:
        rep["collaboration"] = run_collaboration(run_dir, getattr(env, "city", None))
    except Exception as e:                       # unreadable frames must not cost the run its report
        rep["collaboration"] = {"score": None, "error": f"{type(e).__name__}: {e}"}
    return rep, markdown(rep)


def _city_summary(summary: dict, city: str, env) -> dict:
    """The run summary as one city of a region saw it: its scores without the city prefix, its final metrics, and its
    own bankruptcy (the run's average city score is the region's, so none)."""
    pre = city + "."
    scores = {k[len(pre):]: v for k, v in (summary.get("final_scores") or {}).items() if k.startswith(pre)}
    fm = ((summary.get("final_metrics") or {}).get("by_city") or {}).get(city) or {}
    return {**summary, "final_scores": scores, "avg_city_score": None, "final_metrics": fm,
            "bankrupt": bool(env.world.bankrupt)}


def region_report(region, summary: dict, trajectories: list[dict],
                  run_dir: str | Path | None = None) -> tuple[dict, str]:
    """A region run's report: the region scoreboard, then for each city with fleet turns its own report (plan, roles,
    hints, the city's collaboration score) from its env's fleet state and its turns (`turn_row`s named "north.mayor"
    etc.). The top-level `hints` are every city's, each with its `city`."""
    cities = {}
    for c, env in region.envs.items():
        pre = c + "."
        rows = [{**r, "role": r["role"][len(pre):]} for r in trajectories if r["role"].startswith(pre)]
        if rows:
            cities[c] = fleet_report(env, _city_summary(summary, c, env), rows, run_dir)[0]
    scores, fm = summary.get("final_scores") or {}, summary.get("final_metrics") or {}
    by_city = fm.get("by_city") or {}
    rep = {"run_id": summary.get("run_id"), "seed": summary.get("seed"), "signature": summary.get("signature"),
           "months": summary.get("months_played", region.world.month), "months_configured": region.cfg.months,
           "region": list(region.cities),
           "scoreboard": {"city": scores.get("city"), "avg_city": summary.get("avg_city_score"),
                          "population": fm.get("population"), "treasury": fm.get("treasury"),
                          "bankrupt": summary.get("bankrupt"),
                          "cities": {c: {"city": scores.get(f"{c}.city"),
                                         "population": by_city.get(c, {}).get("population"),
                                         "treasury": by_city.get(c, {}).get("treasury")} for c in region.cities}},
           "cities": cities,
           "hints": [{**h, "city": c} for c, r in cities.items() for h in r["hints"]]}
    return rep, region_markdown(rep)


def region_markdown(rep: dict) -> str:
    sb = rep["scoreboard"]
    avg = f" (average {sb['avg_city']:.1f})" if sb.get("avg_city") is not None else ""
    L = [f"# Fleet report: {rep['run_id']}", "", _played(rep), "",
         f"Region of {len(rep['region'])} cities ({', '.join(rep['region'])}); fleet agents in "
         f"{', '.join(rep['cities']) or 'none'}.", "", "## Region scoreboard", "",
         f"Region score **{sb['city'] or 0:.1f}**{avg}, population {sb['population'] or 0:,}, treasury "
         f"{money(sb['treasury'] or 0)}" + (", **bankrupt**" if sb["bankrupt"] else ""), "",
         "| city | score | population | treasury |", "|---|---:|---:|---:|"]
    for c, s in sb["cities"].items():
        L.append(f"| {c} | {s['city'] or 0:.1f} | {s['population'] or 0:,} | {money(s['treasury'] or 0)} |")
    for c, r in rep["cities"].items():
        L += ["", f"## {c.upper()}", "", *_body(r, "###")]
    return "\n".join(L) + "\n"


def _cell(v) -> str:
    return str(v).replace("|", "/").replace("\n", " ")


def _played(rep: dict) -> str:
    played, conf = rep["months"], rep.get("months_configured")
    return (f"Seed {rep['seed']}, {played}" + (f" of {conf}" if conf else "") + " months played"
            + (" (ended early)" if conf and played < conf else "") + f". Agents: `{rep['signature']}`")


def markdown(rep: dict) -> str:
    return "\n".join([f"# Fleet report: {rep['run_id']}", "", _played(rep), "", *_body(rep)]) + "\n"


def _score(v) -> str:
    return "n/a" if v is None else f"{v:.1f}"


def _collab_lines(co: dict | None) -> list[str]:
    """The Collaboration section (collaboration()): score, components, per-department table, oldest unanswered."""
    if co is None:
        return ["n/a (no frames.jsonl to read)."]
    if "error" in co:
        return [f"n/a ({_cell(co['error'])})."]
    cp, w = co["components"], co["window"]
    st = cp["follow_through"]["by_status"]
    L = [f"Score **{_score(co['score'])}** / 100: the mean of the components with at least {COLLAB_MIN} cases "
         f"({co['requests']} requests between departments, {co['open']} still open; a request is answered within "
         f"{w} months, 2 x act_every).", "",
         "| component | score | from |", "|---|---:|---|",
         f"| responsiveness | {_score(cp['responsiveness']['score'])} | {cp['responsiveness']['answered']} of "
         f"{cp['responsiveness']['n']} requests met, partly met, acted on or replied to |",
         f"| follow-through | {_score(cp['follow_through']['score'])} | " + ", ".join(
             f"{st[s]} {s} x {FOLLOW_WEIGHT[s]:g}" for s in FOLLOW_WEIGHT) + " |",
         f"| hand-offs | {_score(cp['handoffs']['score'])} | {cp['handoffs']['used']} of {cp['handoffs']['n']} "
         f"planning roads / freed lots built on by another department |",
         f"| money aim | {_score(cp['money_aim']['score'])} | {money(cp['money_aim']['aimed'])} of "
         f"{money(cp['money_aim']['given'])} in {cp['money_aim']['n']} top-ups answered the department's request |"]
    if co["roles"]:
        L += ["", "| department | asked | answered by others | asked of it | answered for others | responsiveness "
                  "| follow-through | hand-offs |", "|---|" + "---:|" * 7]
        L += [f"| {r} | {x['asked']} | {x['answered']} | {x['asked_of']} | {x['answered_for']} "
              f"| {_score(x['responsiveness'])} | {_score(x['follow_through'])} | {x['handoffs']} |"
              for r, x in co["roles"].items()]
        L += ["", "Hand-offs: planning's openings another department built on; for the others, their builds on land "
                  "planning opened."]
    if co["unanswered"]:
        L += ["", "Longest unanswered:", ""]
        L += [f"- month {u['month']}, {u['from']} -> {u['to']}: \"{_cell(u['text'])}\"" for u in co["unanswered"]]
    return L


def _body(rep: dict, h: str = "##") -> list[str]:
    """Everything after the title and the seed line, headings at level `h` (a region report's city sections: ###)."""
    sb, plan = rep["scoreboard"], rep["plan"]
    avg = f" (average {sb['avg_city']:.1f})" if sb.get("avg_city") is not None else ""
    L = [f"{h} Scoreboard", "",
         f"City score **{sb['city'] or 0:.1f}**{avg}, population {sb['population'] or 0:,}, "
         f"treasury {money(sb['treasury'] or 0)}" + (", **bankrupt**" if sb["bankrupt"] else ""), "",
         "| role | score |", "|---|---:|"]
    L += [f"| {r} | {s:.1f} |" if s is not None else f"| {r} | - |" for r, s in sb["roles"].items()]
    n = plan["versions"]
    L += ["", f"{h} Plan ({n} version{'s' if n != 1 else ''}"
          + (f", saved in month{'s' if n != 1 else ''} {', '.join(map(str, plan['changed']))}" if n else "") + ")", ""]
    L += [f"- {g}" for g in plan["goals"]] or ["No goals set."]
    if plan["budgets"] or plan["priorities"]:
        L += ["", "| department | budget | priority |", "|---|---:|---|"]
        for d in DEPARTMENTS:
            if d in plan["budgets"] or d in plan["priorities"]:
                b = plan["budgets"].get(d)
                L.append(f"| {d} | {money(b) if b is not None else '-'} | {_cell(plan['priorities'].get(d, ''))} |")
    if plan["notes"]:
        L += ["", f"Notes: {_cell(plan['notes'])}"]
    L += ["", f"{h} Roles", "",
          "| role | turns | tool calls | actions | failed | tool errors | repeats (look-ups / refused / overrides) "
          "| spent | tokens (prompt / completion) | limit turns | length stops | API / briefing errors |",
          "|---|" + "---:|" * 11]
    for r, st in rep["roles"].items():
        L.append(f"| {r} | {st['turns']} | {st['tool_calls']} | {st['actions']} | {st['failed']} | {st['tool_errors']} "
                 f"| {st['repeats']} / {st['refused']} / {st.get('overrides', 0)} | {money(st['spent'])} "
                 f"| {st['prompt_tokens']:,} / {st['completion_tokens']:,} "
                 f"| {st['limit_turns']} | {st['length_stops']} | {st['api_errors']} |")
    L += ["", f"{h} Collaboration", "", *_collab_lines(rep.get("collaboration"))]
    fails =[(r, st["top_failures"]) for r, st in rep["roles"].items() if st["top_failures"]]
    if fails:
        L += ["", f"{h} Top failures", ""]
        L += [f"- {r}: " + "; ".join(f"{g['count']}x \"{_cell(g['message'])}\"" for g in gs) for r, gs in fails]
    L += ["", f"{h} Last reports", ""]
    L += [f"- {r}: {_cell(st['last_report'])}" for r, st in rep["roles"].items() if st["last_report"]] or ["None."]
    lessons = [(r, st["lessons"]) for r, st in rep["roles"].items() if st["lessons"]]
    L += ["", f"{h} Lessons in memory", ""]
    L += [f"- {r}: " + "; ".join(_cell(x) for x in ls) for r, ls in lessons] or ["None written."]
    L += ["", f"{h} Coaching hints", ""]
    L += [f"- {h['text']}" for h in rep["hints"]] or ["No coaching hints triggered."]
    return L
