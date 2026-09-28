"""End-of-run fleet report: scoreboard, the plan, per-role activity and coaching hints, as JSON and Markdown.

`run_episode` keeps one slim `turn_row(rec)` per fleet turn (the same rows can be rebuilt from
`trajectories.jsonl` with `rows_from_trajectories`) and writes `fleet_report.json` / `fleet_report.md`; a region run
gets one report with the region scoreboard and a section per city with fleet turns (`region_report`).
Hint rules: >= 3 failures alike (failed actions and tool errors); funds above max(2x the plan budget, $1,000) for
3+ consecutive months; 2+ consecutive turns without a successful action spanning 6+ months (a turn with no plan
budget and under $1 in funds at the turn is blocked, not idle, and splits the run); a negative city treasury; > 20%
of turns ended on the step or token limit.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .fmt import REPORT_PREVIEW_CHARS, cut, money
from .memory import DEPARTMENTS, FleetState
from .tools import ALL_ACTIONS

LIMIT_REASONS = ("max_steps", "turn_tokens")
ERROR_REASONS = ("api_error", "briefing_error")     # turns cut short by an error, not the model
REPEATED_FAILURES = 3
FUNDS_MULTIPLE, FUNDS_MONTHS, UNUSED_FLOOR = 2.0, 3, 1000.0
IDLE_TURNS, IDLE_MONTHS = 2, 6
LIMIT_SHARE = 0.2
MSG_CHARS = 160


def _tool_error(call: dict) -> str | None:
    """A failed tool call that is not a failed action (those are in `results`): bad arguments, unknown tools, a
    look-up that failed, an action tool refused before the action ran. A repeated failed action or bad call and a
    guard refusal (a bulldoze or demolish that would cut roads off, a duplicate allocation), refused without
    running, are counted in `refused`, not here."""
    res = call.get("result") or {}
    if res.get("ok") is not False or (call.get("tool") in ALL_ACTIONS and "message" in res) or "refused" in res:
        return None
    return str(res.get("error") or res.get("message") or "failed")[:MSG_CHARS]


def turn_row(rec: dict) -> dict:
    """The part of a trajectory record the report needs."""
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
            "repeats": marked("repeat"), "refused": marked("refused")}   # repeated look-ups / refused actions


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
            "spent": round(sum(r["cost"] for r in results if r["ok"]), 2),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "limit_turns": sum(r["finish_reason"] in LIMIT_REASONS for r in rows),
            "length_stops": sum(r["length_stops"] for r in rows),
            "api_errors": sum(r["finish_reason"] in ERROR_REASONS for r in rows),
            "top_failures": groups[:max(3, sum(g["count"] >= REPEATED_FAILURES for g in groups))],
            "lessons": list((fleet.memory.get(role) or {}).get("lessons", [])),
            "last_report": cut(fleet.reports[role], REPORT_PREVIEW_CHARS) if fleet.reports.get(role) else None}


def _idle(turns: list[tuple[int, bool]], months: int) -> tuple[int, int, int] | None:
    """Longest (turns, first month, end month) run of idle turns (`(month, False)`: no successful action and not
    blocked) that fires the rule; a run lasts until the department's next turn (or the end of the run)."""
    best, i = None, 0
    while i < len(turns):
        if turns[i][1]:
            i += 1
            continue
        j = i
        while j + 1 < len(turns) and not turns[j + 1][1]:
            j += 1
        end = turns[j + 1][0] if j + 1 < len(turns) else months
        n, a = j - i + 1, turns[i][0]
        if n >= IDLE_TURNS and end - a >= IDLE_MONTHS and (best is None or end - a > best[2] - best[1]):
            best = (n, a, end)
        i = j + 1
    return best


def _hints(roles: dict, rows: list[dict], fleet: FleetState, history: list[dict], months: int) -> list[dict]:
    hints = []
    for role, st in roles.items():
        for g in st["top_failures"]:
            if g["count"] >= REPEATED_FAILURES:
                hints.append({"kind": "repeated_failures", "role": role, "count": g["count"], "message": g["message"],
                              "text": f"{role}: {g['count']} failures like \"{g['message']}\" - add a lesson to its "
                                      "playbook (e.g. check find_sites / inspect before acting)"})
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
    for dept in (d for d in DEPARTMENTS if d in roles):
        # a turn in month t ends in history entry t + 1 (after the mayor's turn, which comes first, and its own). Only
        # actions move dept funds, so after a turn without a success the funds there are the funds it had at its turn;
        # under $1 with no plan budget either, it was blocked, not idle: such turns split idle runs like successful ones
        after = {m["month"] - 1: ((m.get("dept_funds") or {}).get(dept), target(dept, m["month"])) for m in history}
        broke = lambda t: t in after and after[t][0] is not None and after[t][0] < 1
        blocked = lambda t: broke(t) and not after[t][1]
        turns = sorted((row["month"], any(r["ok"] for r in row["results"]) or blocked(row["month"]))
                       for row in rows if row["role"] == dept)
        idle = _idle(turns, months)
        if idle:
            n, a, end = idle
            run = [t for t, _ in turns if a <= t < end]
            hints.append({"kind": "idle", "role": dept, "turns": n, "months": end - a, "first": a, "last": end - 1,
                          "text": f"{dept}: {n} turns / {end - a} months without a successful action "
                                  f"({_span(a, end - 1)}) - "
                                  + (f"its funds stayed at $0 against a {money(after[run[-1]][1])} target: check the "
                                     "mayor's allocations" if all(broke(t) for t in run) else
                                     "check its playbook and whether it is blocked (funds, unlocks, failures)")})
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


def fleet_report(env, summary: dict, trajectories: list[dict]) -> tuple[dict, str]:
    """(report, markdown) from the env after the run, the run summary and `turn_row`s of the fleet turns. For a region
    (RegionEnv) one report with a section per city that had fleet turns (region_report)."""
    if hasattr(env, "envs"):
        return region_report(env, summary, trajectories)
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
    return rep, markdown(rep)


def _city_summary(summary: dict, city: str, env) -> dict:
    """The run summary as one city of a region saw it: its scores without the city prefix, its final metrics, and its
    own bankruptcy (the run's average city score is the region's, so none)."""
    pre = city + "."
    scores = {k[len(pre):]: v for k, v in (summary.get("final_scores") or {}).items() if k.startswith(pre)}
    fm = ((summary.get("final_metrics") or {}).get("by_city") or {}).get(city) or {}
    return {**summary, "final_scores": scores, "avg_city_score": None, "final_metrics": fm,
            "bankrupt": bool(env.world.bankrupt)}


def region_report(region, summary: dict, trajectories: list[dict]) -> tuple[dict, str]:
    """A region run's report: the region scoreboard, then for each city with fleet turns its own report (plan, roles,
    hints) from its env's fleet state and its turns (`turn_row`s named "north.mayor" etc.). The top-level `hints` are
    every city's, each with its `city`."""
    cities = {}
    for c, env in region.envs.items():
        pre = c + "."
        rows = [{**r, "role": r["role"][len(pre):]} for r in trajectories if r["role"].startswith(pre)]
        if rows:
            cities[c] = fleet_report(env, _city_summary(summary, c, env), rows)[0]
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
          "| role | turns | tool calls | actions | failed | tool errors | repeats (look-ups / refused) | spent "
          "| tokens (prompt / completion) | limit turns | length stops | API / briefing errors |",
          "|---|" + "---:|" * 11]
    for r, st in rep["roles"].items():
        L.append(f"| {r} | {st['turns']} | {st['tool_calls']} | {st['actions']} | {st['failed']} | {st['tool_errors']} "
                 f"| {st['repeats']} / {st['refused']} | {money(st['spent'])} "
                 f"| {st['prompt_tokens']:,} / {st['completion_tokens']:,} "
                 f"| {st['limit_turns']} | {st['length_stops']} | {st['api_errors']} |")
    fails = [(r, st["top_failures"]) for r, st in rep["roles"].items() if st["top_failures"]]
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
