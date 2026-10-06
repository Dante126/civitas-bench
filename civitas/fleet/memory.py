"""Fleet memory: the mayor's city plan, per-department memory and an automatic history log.

Everything lives in one `FleetState` stored on the env (`env.fleet`). Tools write the plan and memory
(`update_plan`, `update_memory`), action tools and `end_turn` `record()` into the log, `sync(env)` copies the
world's metrics snapshot and events into it (every month, from `CivitasEnv.end_month`), and briefings read `history()` / `digest()`.
"""
from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field

from ..core.roles import TURN_ORDER

DEPARTMENTS = tuple(r for r in TURN_ORDER if r != "mayor")
LIMITS = {"goals": 5, "item_chars": 200, "notes_chars": 1500, "list_items": 8}
PLAN_FIELDS = ("goals", "budgets", "priorities", "notes")
MEMORY_FIELDS = ("commitments", "open_tasks", "lessons")
KINDS = ("action", "event", "report", "metrics")
HISTORY_KINDS = (*KINDS, "failure")       # failure = action entries with ok False
MAX_HISTORY = 40
log = logging.getLogger(__name__)


def _empty_plan() -> dict:
    return {"goals": [], "budgets": {}, "priorities": {}, "notes": "", "month": None}


def _empty_memory() -> dict:
    return {k: [] for k in MEMORY_FIELDS}


def _items(name: str, value, n: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of strings")
    return [str(v)[:LIMITS["item_chars"]] for v in value if str(v).strip()][:n]


def _dept_dict(name: str, value) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object keyed by department")
    bad = [k for k in value if k not in DEPARTMENTS]
    if bad:
        raise ValueError(f"{name}: unknown department(s) {', '.join(map(str, bad))}; "
                         f"valid departments: {', '.join(DEPARTMENTS)}")
    return value


def _budget(dept: str, v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"budgets.{dept} must be a number of dollars, got {v!r}")
    if v < 0:
        raise ValueError(f"budgets.{dept} must not be negative, got {v}")
    return round(v, 2)


@dataclass
class FleetState:
    plan: dict = field(default_factory=_empty_plan)
    plan_versions: list = field(default_factory=list)
    memory: dict = field(default_factory=lambda: {r: _empty_memory() for r in ("mayor", *DEPARTMENTS)})
    log: list = field(default_factory=list)
    reports: dict = field(default_factory=dict)
    last_turn_month: dict = field(default_factory=dict)
    seen_messages: dict = field(default_factory=dict)   # {role: council messages visible to it at its last turn}
    synced_month: int | None = None
    needs_failures: int = 0     # months needs() failed (logged once per run; collab counts the months missing a record)

    @classmethod
    def of(cls, env) -> "FleetState":
        if getattr(env, "fleet", None) is None:
            env.fleet = cls()
        return env.fleet

    # ------------------------------------------------------------------ plan
    def update_plan(self, month: int, **fields) -> str:
        """Validate and merge a (partial) plan: goals / notes replace, budgets / priorities merge per department."""
        bad = [k for k in fields if k not in PLAN_FIELDS]
        if bad:
            raise ValueError(f"unknown plan field(s) {', '.join(bad)}; valid fields: {', '.join(PLAN_FIELDS)}")
        fields = {k: v for k, v in fields.items() if v is not None}
        if not fields:
            raise ValueError(f"nothing to update; give at least one of: {', '.join(PLAN_FIELDS)}")
        new = copy.deepcopy(self.plan)
        if "goals" in fields:
            new["goals"] = _items("goals", fields["goals"], LIMITS["goals"])
        if "budgets" in fields:
            new["budgets"].update({d: _budget(d, v) for d, v in _dept_dict("budgets", fields["budgets"]).items()})
        if "priorities" in fields:
            new["priorities"].update({d: str(v)[:LIMITS["item_chars"]]
                                      for d, v in _dept_dict("priorities", fields["priorities"]).items()})
        if "notes" in fields:
            new["notes"] = str(fields["notes"])[:LIMITS["notes_chars"]]
        new["month"] = month
        self.plan = new
        self.plan_versions.append(copy.deepcopy(new))
        return f"plan v{len(self.plan_versions)} saved (month {month}): updated {', '.join(fields)}"

    # ---------------------------------------------------------------- memory
    def update_memory(self, role: str, **fields) -> str:
        """Replace the given lists (commitments / open_tasks / lessons), each capped at 8 items of 200 chars."""
        bad = [k for k in fields if k not in MEMORY_FIELDS]
        if bad:
            raise ValueError(f"unknown memory field(s) {', '.join(bad)}; valid fields: {', '.join(MEMORY_FIELDS)}")
        fields = {k: v for k, v in fields.items() if v is not None}
        if not fields:
            raise ValueError(f"nothing to update; give at least one of: {', '.join(MEMORY_FIELDS)}")
        new = {k: _items(k, v, LIMITS["list_items"]) for k, v in fields.items()}
        self.memory.setdefault(role, _empty_memory()).update(new)
        return "memory saved: " + ", ".join(f"{k} ({len(v)})" for k, v in new.items())

    # --------------------------------------------------------------- history
    def record(self, month: int, role: str, kind: str, **data) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown history kind {kind!r}; valid: {', '.join(KINDS)}")
        self.log.append({"month": month, "role": role, "kind": kind, **data})

    def sync(self, env) -> None:
        """Log the latest monthly metrics snapshot and last month's world events. `CivitasEnv.end_month` calls this
        every month once `env.fleet` exists; calling it again in the same month does nothing."""
        w = env.world
        if not w.history or self.synced_month == w.month:
            return
        self.synced_month = w.month
        m = w.history[-1]
        self.record(m["month"], "city", "metrics", population=m["population"], treasury=round(m["treasury"]),
                    happiness=round(m["happiness"], 3), city_score=round(env.last_scores.get("city", 0.0), 1))
        for text in w.last_events:
            self.record(w.month - 1, "city", "event", text=text)

    def needs(self, env) -> dict | None:
        """The department needs the mayor's Departments block shows this month (tools.funding_gaps), for the frame's
        `needs` key (CivitasEnv.frame): {dept: [kind, top_up, covered]} per department in the game, kind one of
        tools.KINDS, top_up what its needs line lacks (0 without one), covered 1 when the to-fund line's cover clause
        (fund_cover) names it. A frame of month M is written before month M's turns, so it is what the mayor sees
        that month (the mayor acts first). Only in months the agents act (act_every) and with a mayor in the game;
        None otherwise or on any error (a frame must never end a run): the first error of a run is logged with its
        traceback, later ones only counted (needs_failures), and the frame goes without a record, which the
        collaboration block counts (bench/collab.py: needs_missing) and the fleet report notes. Read-only:
        report-only (the collaboration score's money aim), never shown to agents. About 5-10 ms a call."""
        if "mayor" not in env.roles or not env.acts_this_month("mayor"):
            return None
        from .tools import fund_cover, funding_gaps      # tools imports this module
        try:
            f = funding_gaps(env, self)
            ups, cover = {i.dept: i.amount for i in f.to_fund}, set(fund_cover(f))
            return {d: [m.kind, ups.get(d, 0), int(d in cover)] for d, m in f.depts.items()}
        except Exception:
            self.needs_failures = getattr(self, "needs_failures", 0) + 1
            if self.needs_failures == 1:
                log.warning("FleetState.needs failed at month %s (the frame goes without a needs record; later "
                            "failures this run are not logged)", env.world.month, exc_info=True)
            return None

    def history(self, kind: str | None = None, months: int | None = 3, role=None,
                now: int | None = None) -> list[dict]:
        """Log entries from months `> now - months` (`now` defaults to the newest logged month), newest last,
        at most the 40 newest. `role` is one role or a collection of roles (filtered before the cap);
        `kind="failure"` selects failed actions; `months=None` means all months."""
        roles = None if role is None else {role} if isinstance(role, str) else set(role)
        if kind is not None and kind not in HISTORY_KINDS:
            raise ValueError(f"unknown history kind {kind!r}; valid: {', '.join(HISTORY_KINDS)}")
        if now is None:
            now = max((e["month"] for e in self.log), default=0)
        out = [e for e in self.log
               if (months is None or e["month"] > now - months)
               and (roles is None or e["role"] in roles)
               and (kind is None or e["kind"] == kind
                    or (kind == "failure" and e["kind"] == "action" and not e.get("ok", True)))]
        out.sort(key=lambda e: e["month"])
        return out[-MAX_HISTORY:]

    def mark_turn(self, role: str, month: int, council=None, who: str | None = None) -> None:
        """Remember that `role` took a turn in `month` (and how many council messages it could see then);
        call after building the turn's digest. `who` is its name on the council (region mode: "north.police";
        default `role`)."""
        self.last_turn_month[role] = month
        if council is not None:
            self.seen_messages[role] = len(council.visible_to(who or role, since_month=0))

    def digest(self, role: str, now: int, council=None, who: str | None = None) -> dict:
        """Since the role's previous turn (inclusive of that month): its failed actions and world events, plus the
        council messages from others it can see that were posted after that turn was marked, oldest first. `since`
        is None before its first turn (all months). `who` as for mark_turn."""
        since = self.last_turn_month.get(role)
        span = None if since is None else now - since + 1
        msgs = []
        if council is not None:
            who = who or role
            new = council.visible_to(who, since_month=0)[self.seen_messages.get(role, 0):]
            msgs = [m for m in new if m["sender"] != who]
        return {"since": since,
                "failed_actions": self.history("failure", span, role=role, now=now),
                "events": [e["text"] for e in self.history("event", span, now=now)],
                "messages": msgs}
