"""End-of-run fleet report: scoreboard, the plan, per-role activity and coaching hints, as JSON and Markdown.

`run_episode` keeps one slim `turn_row(rec)` per fleet turn (the same rows can be rebuilt from
`trajectories.jsonl` with `rows_from_trajectories`) and writes `fleet_report.json` / `fleet_report.md`.
Hint rules: >= 3 failures alike, funds above 2x the plan budget for 3+ consecutive months, no successful
action for 6+ consecutive months, a negative treasury, > 20% of turns ended on the step or token limit.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .briefing import _money
from .memory import DEPARTMENTS, FleetState

LIMIT_REASONS = ("max_steps", "turn_tokens")
REPEATED_FAILURES = 3
FUNDS_MULTIPLE, FUNDS_MONTHS = 2.0, 3
IDLE_MONTHS = 6
LIMIT_SHARE = 0.2
MSG_CHARS = 160


def turn_row(rec: dict) -> dict:
    """The part of a trajectory record the report needs."""
    usage = rec.get("usage") or {}
    return {"role": rec["role"], "month": rec["month"],
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "steps": rec.get("steps") or 0, "finish_reason": rec.get("finish_reason"),
            "length_stops": rec.get("length_stops") or 0, "ended": rec.get("ended"),
            "parse_error": rec.get("parse_error"),
            "results": [{"type": (r.get("action") or {}).get("type"), "ok": bool(r.get("ok")),
                         "message": str(r.get("message") or "")[:MSG_CHARS], "cost": r.get("cost") or 0}
                        for r in rec.get("results") or []],
            "tool_errors": [str(c["result"].get("error"))[:MSG_CHARS] for c in rec.get("tool_trace") or []
                            if not c["result"].get("ok") and "message" not in c["result"]]}


def rows_from_trajectories(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        if str(rec.get("agent", "")).startswith("fleet:"):
            rows.append(turn_row(rec))
    return rows


def _key(msg: str) -> str:
    """Failure messages alike up to numbers and coordinates share a key (their first clause)."""
    return re.split(r"[;:]", re.sub(r"-?\$?\d[\d,.]*", "#", msg.lower()), maxsplit=1)[0].strip()[:60]


def _failure_groups(msgs: list[str]) -> list[dict]:
    groups: dict[str, list[str]] = {}
    for m in msgs:
        groups.setdefault(_key(m), []).append(m)
    out = [{"message": v[0], "count": len(v)} for v in groups.values()]
    return sorted(out, key=lambda g: -g["count"])


def _runs(months: list[int], flagged) -> list[tuple[int, int]]:
    """(first, last) of each run of consecutive months where `flagged(month)` holds."""
    out, start = [], None
    for m in months:
        if flagged(m):
            start = m if start is None else start
        elif start is not None:
            out.append((start, prev))
            start = None
        prev = m
    if start is not None:
        out.append((start, months[-1]))
    return out


def _span(a: int, b: int) -> str:
    return f"month {a}" if a == b else f"months {a}-{b}"


def _role_stats(rows: list[dict], fleet: FleetState) -> dict:
    results = [r for row in rows for r in row["results"]]
    fails = [r["message"] for r in results if not r["ok"]]
    tool_errs = [e for row in rows for e in row["tool_errors"]]
    return {"turns": len(rows), "tool_calls": sum(r["steps"] for r in rows), "actions": len(results),
            "failed": len(fails), "tool_errors": len(tool_errs),
            "spent": round(sum(r["cost"] for r in results if r["ok"]), 2),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "limit_turns": sum(r["finish_reason"] in LIMIT_REASONS for r in rows),
            "length_stops": sum(r["length_stops"] for r in rows),
            "api_errors": sum(r["finish_reason"] == "api_error" for r in rows),
            "top_failures": _failure_groups(fails)[:3],
            "lessons": list((fleet.memory.get(rows[0]["role"]) or {}).get("lessons", []))}


def _hints(roles: dict, rows: list[dict], fleet: FleetState, history: list[dict], months: int) -> list[dict]:
    hints = []
    for role in roles:
        msgs = [r["message"] for row in rows if row["role"] == role for r in row["results"] if not r["ok"]]
        msgs += [e for row in rows if row["role"] == role for e in row["tool_errors"]]
        for g in _failure_groups(msgs):
            if g["count"] >= REPEATED_FAILURES:
                hints.append({"kind": "repeated_failures", "role": role, "count": g["count"], "message": g["message"],
                              "text": f"{role}: {g['count']} failures like \"{g['message']}\" - add a lesson to its "
                                      "playbook (e.g. check find_sites / inspect before acting)"})
    versions = fleet.plan_versions
    hist = [m for m in history if m.get("dept_funds")]
    for dept in (d for d in DEPARTMENTS if d in roles):
        def target(m):
            live = [v for v in versions if v["month"] is not None and v["month"] < m]
            return live[-1]["budgets"].get(dept) if live else None
        over = {m["month"]: (m["dept_funds"].get(dept, 0.0), target(m["month"])) for m in hist}
        runs = [(a, b) for a, b in _runs(sorted(over), lambda m: over[m][1] is not None
                                         and over[m][0] > FUNDS_MULTIPLE * over[m][1]) if b - a + 1 >= FUNDS_MONTHS]
        if runs:
            a, b = max(runs, key=lambda r: r[1] - r[0])
            peak = max(over[m][0] for m in range(a, b + 1))
            hints.append({"kind": "unused_money", "role": dept, "months": b - a + 1, "first": a, "last": b,
                          "text": f"{dept}: funds stayed above {FUNDS_MULTIPLE:g}x its plan budget "
                                  f"({_money(over[b][1])}) for {b - a + 1} months ({_span(a, b)}, up to "
                                  f"{_money(peak)}) - have it spend, or lower its allocation"})
    for dept in (d for d in DEPARTMENTS if d in roles):
        active = {row["month"] for row in rows if row["role"] == dept and any(r["ok"] for r in row["results"])}
        runs = [(a, b) for a, b in _runs(list(range(months)), lambda m: m not in active) if b - a + 1 >= IDLE_MONTHS]
        if runs:
            a, b = max(runs, key=lambda r: r[1] - r[0])
            hints.append({"kind": "idle", "role": dept, "months": b - a + 1, "first": a, "last": b,
                          "text": f"{dept}: no successful action for {b - a + 1} months ({_span(a, b)}) - check its "
                                  "playbook and whether it is blocked (funds, unlocks, failures)"})
    neg = [m for m in history if m.get("treasury", 0) < 0]
    if neg:
        low = min(m["treasury"] for m in neg)
        hints.append({"kind": "negative_treasury", "role": "mayor", "months": len(neg), "first": neg[0]["month"],
                      "text": f"treasury was negative in {len(neg)} month(s) (first month {neg[0]['month']}, low "
                              f"{_money(low)}) - the mayor should keep a reserve or allocate less"})
    for role, st in roles.items():
        if st["turns"] and st["limit_turns"] / st["turns"] > LIMIT_SHARE:
            hints.append({"kind": "limit", "role": role, "count": st["limit_turns"], "turns": st["turns"],
                          "text": f"{role}: {st['limit_turns']} of {st['turns']} turns "
                                  f"({100 * st['limit_turns'] / st['turns']:.0f}%) ended on the step or token limit - "
                                  "raise max_steps / turn_tokens or tell it to end_turn sooner"})
    return hints


def fleet_report(env, summary: dict, trajectories: list[dict]) -> tuple[dict, str]:
    """(report, markdown) from the env after the run, the run summary and `turn_row`s of the fleet turns."""
    fleet = getattr(env, "fleet", None) or FleetState()
    history = env.world.history
    months = summary.get("months_played", env.world.month)
    roles = {r: _role_stats([x for x in trajectories if x["role"] == r], fleet)
             for r in env.roles if any(x["role"] == r for x in trajectories)}
    scores = summary.get("final_scores") or {}
    fm = summary.get("final_metrics") or {}
    plan = fleet.plan
    rep = {"run_id": summary.get("run_id"), "seed": summary.get("seed"), "signature": summary.get("signature"),
           "months": months,
           "scoreboard": {"city": scores.get("city"), "roles": {r: scores.get(r) for r in env.roles},
                          "avg_city": summary.get("avg_city_score"), "population": fm.get("population"),
                          "treasury": fm.get("treasury"), "bankrupt": summary.get("bankrupt")},
           "plan": {**{k: plan.get(k) for k in ("goals", "budgets", "priorities", "notes", "month")},
                    "versions": len(fleet.plan_versions), "changed": [v["month"] for v in fleet.plan_versions]},
           "roles": roles,
           "hints": _hints(roles, trajectories, fleet, history, months)}
    return rep, markdown(rep)


def _cell(v) -> str:
    return str(v).replace("|", "/").replace("\n", " ")


def markdown(rep: dict) -> str:
    sb, plan = rep["scoreboard"], rep["plan"]
    L = [f"# Fleet report: {rep['run_id']}", "",
         f"Seed {rep['seed']}, {rep['months']} months. Agents: `{rep['signature']}`", "",
         "## Scoreboard", "",
         f"City score **{sb['city'] or 0:.1f}** (average {sb['avg_city'] or 0:.1f}), population {sb['population'] or 0:,}, "
         f"treasury {_money(sb['treasury'] or 0)}" + (", **bankrupt**" if sb["bankrupt"] else ""), "",
         "| role | score |", "|---|---:|"]
    L += [f"| {r} | {s:.1f} |" if s is not None else f"| {r} | - |" for r, s in sb["roles"].items()]
    L += ["", f"## Plan ({plan['versions']} version{'s' if plan['versions'] != 1 else ''}"
          + (f", last saved month {plan['month']}" if plan["month"] is not None else "") + ")", ""]
    L += [f"- {g}" for g in plan["goals"]] or ["No goals set."]
    if plan["budgets"] or plan["priorities"]:
        L += ["", "| department | budget | priority |", "|---|---:|---|"]
        for d in DEPARTMENTS:
            if d in plan["budgets"] or d in plan["priorities"]:
                b = plan["budgets"].get(d)
                L.append(f"| {d} | {_money(b) if b is not None else '-'} | {_cell(plan['priorities'].get(d, ''))} |")
    if plan["notes"]:
        L += ["", f"Notes: {_cell(plan['notes'])}"]
    L += ["", "## Roles", "",
          "| role | turns | tool calls | actions | failed | tool errors | spent | tokens (prompt / completion) "
          "| limit turns | length stops | API errors |", "|---|" + "---:|" * 10]
    for r, st in rep["roles"].items():
        L.append(f"| {r} | {st['turns']} | {st['tool_calls']} | {st['actions']} | {st['failed']} | {st['tool_errors']} "
                 f"| {_money(st['spent'])} | {st['prompt_tokens']:,} / {st['completion_tokens']:,} "
                 f"| {st['limit_turns']} | {st['length_stops']} | {st['api_errors']} |")
    fails = [(r, st["top_failures"]) for r, st in rep["roles"].items() if st["top_failures"]]
    if fails:
        L += ["", "## Top failures", ""]
        for r, gs in fails:
            L += [f"- {r}: " + "; ".join(f"{g['count']}x \"{_cell(g['message'])}\"" for g in gs)]
    lessons = [(r, st["lessons"]) for r, st in rep["roles"].items() if st["lessons"]]
    L += ["", "## Lessons in memory", ""]
    L += [f"- {r}: " + "; ".join(_cell(x) for x in ls) for r, ls in lessons] or ["None written."]
    L += ["", "## Coaching hints", ""]
    L += [f"- {h['text']}" for h in rep["hints"]] or ["No coaching hints triggered."]
    return "\n".join(L) + "\n"
