"""ScriptedFleetAgent: the scripted baselines playing through the fleet tools (`kind = "fleet"`, `model = "scripted"`).

A zero-cost fleet line-up and an end-to-end test of the tool layer. Each turn it builds the role's briefing (logged
as the turn's first user message, as an LLM fleet agent gets it), reads what the policy needs through look-ups
(`map_overview` for the roles whose policy reads the map, `get_messages` for the mayor's council requests) on top of
`role_obs()`, asks the role's `agents/scripted.py` policy for a decision and runs it through `ToolBox.call`: the mayor
first writes its working balances into the plan (`update_plan`), then every action runs as its own tool call (a
duplicate allocation, a take-back below a department's keep or a loan the guards refuse, which the policy means, is
repeated with `confirm: true`: the refusal earned it),
messages go out with `send_message`, new requests and failures go into memory (`update_memory`), and `end_turn`
carries the report. Arguments are sent as JSON text, as a model sends them. The turn is logged like a ToolAgent turn
(transcript, tool trace, no tokens), so the runner, history log and fleet report treat it as any fleet turn.
"""
from __future__ import annotations

import json

from ..agents.base import Agent
from ..agents.scripted import SCRIPTED
from ..core.decision import Decision
from ..core.region import RegionEnv
from .agent import _budget, _slim
from .briefing import briefing, system_prompt
from .config import FleetAgentConfig
from .fmt import TOOL_MESSAGE_CHARS
from .memory import DEPARTMENTS, LIMITS, FleetState
from .report import _key
from .tools import ALL_ACTIONS, ToolBox, role_obs

MODEL = "scripted"              # [fleet] / [agents.<role>] model that selects this agent
MAP_ROLES = {"planning", "police", "fire", "health", "education"}    # policies that read obs["map"]
GOALS = ["Grow: roads and zones where demand is, toward the target population",
         "Power, water and sewage about 25% ahead of demand; services once people move in",
         "Stay solvent: keep $3,000 or 3 months of expenses (6 in a deficit) in the treasury"]
PRIORITIES = {           # what each scripted department does (agents/scripted.py), one line each
    "planning": "roads from the highway, zones by demand; parks, plazas, a casino later",
    "utilities": "power, water and sewage ahead of demand; upgrade and overhaul plants",
    "transport": "bus lines from 1,500 residents; avenues on jammed streets; rail and harbour later",
    "police": "stations to 85% coverage; a jail when inmates pass 80% of capacity",
    "fire": "stations to 85% coverage; levees before a flood; a shelter from 3,000",
    "health": "landfill first, clinics to 80% coverage, deathcare; a hospital from 5,000",
    "education": "schools to 90% enrolment; a library from 2,500, a university from 6,000",
}
PLAN_NOTES = ("Budgets are working balances: each turn departments below theirs are topped up, most urgent first, "
              "from the treasury above the reserve. Ask with a dollar amount to raise yours.")


def _plain(o):
    """JSON for numpy scalars the policies hand over (tool arguments go out as text, as a model sends them)."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


class _Turn:
    """One turn's tool calls through the ToolBox, logged as ToolAgent logs them: the transcript (one assistant tool
    call and its result per step, the result with steps_left) and the tool trace (slim results). The last step is
    kept for end_turn."""

    def __init__(self, box: ToolBox, max_steps: int):
        self.box, self.max_steps = box, max_steps
        self.trace: list[dict] = []
        self.messages: list[dict] = []
        self.skipped: list[str] = []          # calls left out for lack of steps

    def call(self, name: str, args: dict | None = None, *, last: bool = False) -> dict | None:
        """Run one tool call; None when it can't run (the turn ended, or only end_turn's step is left)."""
        if self.box.ended or len(self.trace) >= self.max_steps - (0 if last else 1):
            self.skipped.append(name)
            return None
        raw = json.dumps(args or {}, default=_plain)
        cid = f"call_{len(self.trace)}"
        res = self.box.call(name, raw)
        self.trace.append({"step": len(self.trace), "finish_reason": MODEL, "tool": name, "args": raw,
                           "result": _slim(res)})
        shown = res if self.box.ended else _budget(res, self.max_steps - len(self.trace))
        self.messages += [{"role": "assistant", "content": "",
                           "tool_calls": [{"id": cid, "type": "function",
                                           "function": {"name": name, "arguments": raw}}]},
                          {"role": "tool", "tool_call_id": cid,
                           "content": json.dumps(shown, separators=(",", ":"))[:TOOL_MESSAGE_CHARS]}]
        return res


class ScriptedFleetAgent(Agent):
    kind = "fleet"

    def __init__(self, role: str, *, cfg: FleetAgentConfig | None = None, **kw):
        super().__init__(role, **kw)
        self.cfg = cfg or FleetAgentConfig(model=MODEL)
        self.policy = SCRIPTED[role](role)
        self.env = None
        self.system = ""
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "tool_calls": 0, "api_errors": 0,
                      "latency_s": 0.0}

    def reset(self, seed: int = 0) -> None:
        self.policy = SCRIPTED[self.role](self.role)       # policies keep state (the planner's road plan)
        self.policy.reset(seed)

    def bind(self, env) -> None:
        if isinstance(env, RegionEnv) or getattr(env, "city", None):
            raise ValueError("fleet agents support single-city games only")
        self.env = env
        FleetState.of(env)
        c = env.cfg
        self.system = system_prompt(self.role, self.cfg, c.months, act_every=c.act_every,
                                    max_actions=c.max_actions_per_turn, target_pop=env.scoring.target_population)

    def describe(self) -> str:
        return f"fleet:{self.cfg.model}"

    # ------------------------------------------------------------------ turn
    def act(self, obs: dict) -> Decision:
        if self.env is None:
            raise RuntimeError("ScriptedFleetAgent.act before bind(env)")
        env, role = self.env, self.role
        box = ToolBox(env, role)
        turn = _Turn(box, self.cfg.max_steps)
        brief = briefing(env, role, mode=self.cfg.mode)
        view = self._view(turn)
        d = self.policy.act(view)
        if role == "mayor":
            self._plan(turn)
        self._actions(turn, d.actions)
        for m in d.messages:
            turn.call("send_message", {"to": m.get("to", "all"), "text": m["text"]})
        self._memory(turn)
        turn.call("end_turn", {"report": self._report(d, box, turn)}, last=True)
        self.usage["tool_calls"] += len(turn.trace)
        out = Decision(executed=box.executed, messages=box.messages, thinking=box.report or "(no report)")
        out.meta = {"model": self.cfg.model,
                    "messages": [{"role": "system", "content": f"(system prompt: meta.json fleet_prompts.{role})"},
                                 {"role": "user", "content": brief}, *turn.messages],
                    "tool_trace": turn.trace, "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "steps": len(turn.trace), "ended": box.ended,
                    "finish_reason": "end_turn" if box.ended else "max_steps", "finish_reasons": [],
                    "length_stops": 0, "recovered_calls": 0, "reasoning": d.thinking, "response": box.report,
                    "latency_s": 0.0}
        return out

    def _view(self, turn: _Turn) -> dict:
        """The policy's observation: role_obs, with the map from map_overview and the council from get_messages
        where the policy reads them (as env.observe gives them to the scripted baselines). A map too big for one
        tool result comes downsampled (48x48 always fits); the policies parse single tiles, so they then read the
        observation's map."""
        env, role = self.env, self.role
        view = role_obs(env, role)
        view["max_actions"] = env.cfg.max_actions_per_turn
        if role in MAP_ROLES:
            res = turn.call("map_overview") or {}
            full = res.get("ok") and "scale" not in res
            view["map"] = "\n".join(res["map"]) if full else env.world.ascii_map()
        if role == "mayor":
            res = turn.call("get_messages") or {}
            view["council"] = res.get("messages", [])
        return view

    def _plan(self, turn: _Turn) -> None:
        """The mayor's plan update: the policy's working balances as budget targets (goals, notes and a priority
        per department with the first plan; a department that lost its target gets 0), when the budgets changed."""
        env, fleet = self.env, FleetState.of(self.env)
        targets = {d: round(float(v), 2) for d, v in getattr(self.policy, "targets", {}).items() if d in DEPARTMENTS}
        budgets = {**{d: 0.0 for d, v in fleet.plan["budgets"].items() if v and d not in targets}, **targets}
        args = {}
        if not fleet.plan_versions:
            args = {"goals": GOALS, "notes": PLAN_NOTES,
                    "priorities": {d: PRIORITIES[d] for d in DEPARTMENTS if d in env.roles and d in PRIORITIES}}
        if any(fleet.plan["budgets"].get(d) != v for d, v in budgets.items()):
            args["budgets"] = budgets
        if args:
            turn.call("update_plan", args)

    def _actions(self, turn: _Turn, actions: list[dict]) -> None:
        """Each action as its own tool call, in the policy's order, until the turn's successful-action cap. A money
        call the mayor's guards refuse (a duplicate allocation, a take-back below a department's keep, a loan while
        the treasury holds $3,000 or its own top-ups emptied it) is one the policy means: it is repeated at once with
        confirm: true, which that refusal earned (the scripted mayor sends no take-backs and borrows only at a turn's
        start under $3,000, so only duplicates happen)."""
        box, cap = turn.box, self.env.cfg.max_actions_per_turn
        for a in actions:
            args = dict(a)
            name = args.pop("type", None)
            if name not in ALL_ACTIONS or sum(r["ok"] for r in box.executed) >= cap:
                continue
            res = turn.call(name, args)
            if res is None:
                break
            if res.get("refused") in ("duplicate", "take_back", "loan"):     # the policy means it
                turn.call(name, {**args, "confirm": True})

    def _memory(self, turn: _Turn) -> None:
        """open_tasks: this turn's requests to other roles; lessons: one per kind of failed action, newest last.
        Written only when they change."""
        box, fleet = turn.box, FleetState.of(self.env)
        mem = fleet.memory.get(self.role, {})
        n, chars = LIMITS["list_items"], LIMITS["item_chars"]
        tasks = [f"asked {m['to']}: {m['text']}"[:chars] for m in box.messages if m["to"] != "all"][:n]
        lessons = list(mem.get("lessons", []))
        for r in box.executed:
            if not r["ok"]:
                text = f"{r['action'].get('type')}: {r['message']}"[:chars]
                lessons = [x for x in lessons if _key(x) != _key(text)] + [text]
        fields = {k: v for k, v in (("open_tasks", tasks), ("lessons", lessons[-n:])) if v != mem.get(k, [])}
        if fields:
            turn.call("update_memory", fields)

    @staticmethod
    def _report(d: Decision, box: ToolBox, turn: _Turn) -> str:
        """The policy's rationale, what it asks the mayor for, and what the actions did. A policy that built nothing
        says its idle note ("coverage OK"), also when it gave up for lack of money: next to a request that is left
        out (scripted police asked for $2,500 at 0% coverage for 36 months of seed 0 reading "coverage OK")."""
        ok = sum(r["ok"] for r in box.executed)
        failed = [r for r in box.executed if not r["ok"]]
        asks = [f"need: {m['text']}" for m in box.messages if m["to"] == "mayor"]
        parts = ([d.thinking or "nothing to do"] if d.actions or not asks else []) + asks
        tally = f"{ok} action{'s' * (ok != 1)} done"
        if failed:
            tally += f", {len(failed)} failed ({failed[0]['action'].get('type')}: {failed[0]['message']})"
        if turn.skipped:
            tally += f", {len(turn.skipped)} call(s) skipped at the step limit"
        return "; ".join(parts + [tally])
