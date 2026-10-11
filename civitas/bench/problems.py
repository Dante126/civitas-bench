"""Problems in a run and how the agents handled them: the viewer's Problems tab (/api/runs/<id>/problems).

A problem is a tool call that failed (an action that ran and failed, or a tool error: bad arguments, an unknown
tool), a refused call (a guard, or a repeat of a failed call), or a turn that ended badly (the step limit, a reply
with no tool call, the token budget, an unreadable reply). What the agent did next is the point: fixed (a later
call of the same tool in the turn succeeded, with other arguments or, after another action changed what failed, the
same ones), heeded (a guard refused and the agent did something else), overridden
(it repeated the refused call with confirm), repeated (it sent the same failing call again), escalated (it gave up on
it but messaged another role, e.g. asked the mayor for money), dropped (it gave up on that tool this turn), and, across turns, recurred (the same role hit the same cause on its next turn). Rates over
these say how an agent solves problems, not only how often it makes them. A warm-up's turns (`"warmup": true`, played
by the scripted warm-up line-up before the configured agents take over) are left out.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..fleet.report import CAUSES, _key

BAD_ENDS = {"max_steps": "step limit", "no_tool_calls": "no tool call", "turn_tokens": "completion-token limit",
            "turn_budget": "token budget", "length": "cut at max_tokens"}
GUARDS_NOT_REPEATS = {"repeat"}
ACTION_LIMIT = "action limit reached"          # a refused repeat of a failed call is the agent's mistake, not a guard's


def cause(text: str) -> str:
    """A short cause shared by failures alike: the report's named causes, else the message up to numbers."""
    for name, pattern, _ in CAUSES:
        if pattern.search(text):
            return name
    return _key(text) or "unknown"


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def _plain(args) -> dict:
    """Arguments without confirm, so a confirmed repeat matches the refused call."""
    return {k: v for k, v in args.items() if k != "confirm"} if isinstance(args, dict) else args


def turn_problems(rec: dict) -> list[dict]:
    """The problems of one agent turn (a trajectories.jsonl record), each with how the turn handled it."""
    from ..server import trace_call               # the viewer's call outcomes; lazy: the server imports this module
    month, role = rec.get("month"), rec.get("role")
    out = []
    if "tool_trace" in rec:
        calls = [trace_call(i + 1, c) for i, c in enumerate(rec.get("tool_trace") or [])]
        for i, c in enumerate(calls):
            if c["outcome"] not in ("failed", "refused") or ACTION_LIMIT in c["text"]:
                continue                           # the turn's action cap: the agent used its actions, no mistake
            later = calls[i + 1:]
            guard = c["outcome"] == "refused" and c.get("refused") not in GUARDS_NOT_REPEATS
            kind = ("guard" if guard else "repeat" if c["outcome"] == "refused" else
                    "failed action" if c["ran"] or c["action"] else "tool error")
            same = [d for d in later if d["tool"] == c["tool"] and _same(_plain(d["args"]), _plain(c["args"]))]
            # fixed: a later call of the tool worked, other arguments or the same ones after something else changed
            # what failed (v19 planning: dezone the lots the tool result named, then the same build_road again)
            fixed = next((d for d in later if d["tool"] == c["tool"] and d["outcome"] == "ok"), None)
            if guard:
                over = next((d for d in same if d["outcome"] == "ok"), None)
                handled, at = ("overridden", over["n"]) if over else ("heeded", None)
            elif fixed:
                handled, at = "fixed", fixed["n"]
            elif any(d["outcome"] in ("failed", "refused") for d in same):
                handled, at = "repeated", next(d["n"] for d in same if d["outcome"] in ("failed", "refused"))
            elif (ask := next((d for d in later if d["tool"] == "send_message" and d["outcome"] == "ok"), None)):
                handled, at = "escalated", ask["n"]     # asked another role (the mayor for money, planning for a lot)
            else:
                handled, at = "dropped", None
            out.append({"month": month, "role": role, "n": c["n"], "tool": c["tool"], "args": c["args"],
                        "kind": kind, "refused": c.get("refused"), "text": c["text"][:300],
                        "cause": f"guard: {c.get('refused')}" if guard else cause(c["text"]),
                        "handled": handled, "at": at})
    else:                                          # one-shot LLM agents: failed actions, no calls to recover in
        for i, r in enumerate(rec.get("results") or []):
            if not r.get("ok"):
                out.append({"month": month, "role": role, "n": i + 1, "tool": (r.get("action") or {}).get("type"),
                            "args": r.get("action"), "kind": "failed action", "refused": None,
                            "text": str(r.get("message") or "")[:300], "cause": cause(str(r.get("message") or "")),
                            "handled": "dropped", "at": None})
    end = rec.get("finish_reason")
    if end in BAD_ENDS or rec.get("parse_error"):
        what = BAD_ENDS.get(end) or "unreadable reply"
        out.append({"month": month, "role": role, "n": None, "tool": None, "args": None, "kind": "turn end",
                    "refused": None, "text": str(rec.get("parse_error") or f"the turn ended on the {what}")[:300],
                    "cause": what, "handled": "ended", "at": None})
    return out


def run_problems(path: str | Path, report: str | Path | None = None) -> dict:
    """Every problem in a run's trajectories, recurrences across turns, per-role rates, and the fleet report's
    coaching hints when there is one."""
    recs = []
    with open(path) as fh:
        for line in fh:
            if line.endswith("\n"):
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    items, turns = [], {}
    for rec in recs:
        if rec.get("warmup"):                      # a warm-up's scripted turns (bench/runner.py) aren't under test
            continue
        turns.setdefault(rec.get("role"), []).append(rec.get("month"))
        items.extend(turn_problems(rec))
    # recurred: the same role hit the same cause on its next turn (guards are the harness's, not a recurrence)
    by_turn = {}
    for p in items:
        by_turn.setdefault((p["role"], p["month"]), set()).add(p["cause"])
    for p in items:
        months = turns.get(p["role"], [])
        nxt = next((m for m in months if m is not None and p["month"] is not None and m > p["month"]), None)
        p["recurred"] = p["kind"] not in ("guard",) and nxt is not None and p["cause"] in by_turn.get((p["role"], nxt), set())
    roles = {}
    for role, months in turns.items():
        ps = [p for p in items if p["role"] == role]
        fixable = [p for p in ps if p["kind"] in ("failed action", "tool error", "repeat")]
        guards = [p for p in ps if p["kind"] == "guard"]
        roles[role] = {"turns": len(months), "problems": len(ps),
                       "failed": sum(p["kind"] == "failed action" for p in ps),
                       "tool_errors": sum(p["kind"] == "tool error" for p in ps),
                       "repeats": sum(p["kind"] == "repeat" for p in ps),
                       "fixed": sum(p["handled"] == "fixed" for p in fixable),
                       "repeated": sum(p["handled"] == "repeated" for p in fixable),
                       "escalated": sum(p["handled"] == "escalated" for p in fixable),
                       "dropped": sum(p["handled"] == "dropped" for p in fixable),
                       "fixable": len(fixable), "recurred": sum(p["recurred"] for p in ps),
                       "guards": len(guards), "heeded": sum(p["handled"] == "heeded" for p in guards),
                       "overridden": sum(p["handled"] == "overridden" for p in guards),
                       "bad_ends": sum(p["kind"] == "turn end" for p in ps)}
    total = {k: sum(r[k] for r in roles.values()) for k in next(iter(roles.values()), {})}
    hints = []
    if report and Path(report).exists():
        try:
            hints = json.loads(Path(report).read_text()).get("hints") or []
        except ValueError:
            pass
    return {"problems": items, "roles": roles, "total": total, "hints": hints}
