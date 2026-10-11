"""Hand-made policies for a TaskEpisode (no model): what `civitas rl check` scores to show the reward means something.

- noop: end_turn at once (again if the first one is refused: the mayor's and planning's open-requests reminder);
- scripted: the scripted fleet agent's own decision for the turn (fleet/scripted.py: `decide`, the mayor's
  `plan_args`), sent one call at a time through the episode's model toolbox, a guard refusal the policy means repeated
  with confirm: true, as ScriptedFleetAgent does. The decision is made on a copy of the policy, so the branch's
  scripted agent for the role plays on from where it was, as in a model's branch. The baseline plays the same
  decision through the scripted fleet's plain toolbox, so this differs where the model toolbox does (branch roads
  that dezone, held-back money asks, end_turn reminders) and where the policy's own state carries the turn on (the
  planner's road plan): ~0, not exactly 0;
- degraded: scripted with about half its actions dropped (at least one, seeded per task);
- random: 1-4 action calls of the role's tools with random arguments of the right types, then end_turn;
- oracle: the guided briefing's own calls first (a department's "Your budget" block: its Next / This turn / tier
  upgrade calls; the mayor: a top-up of each department the to-fund line covers), then the scripted calls.
The "baseline" policy is not an episode policy: check.py plays the scripted fleet agent's whole turn (its own act)
on the prepared state, which must score exactly 0 apart from penalties for its own failed calls.
"""
from __future__ import annotations

import copy
import json
import random
import re
import zlib

from ..fleet.scripted import CONFIRMED, _plain
from ..fleet.tools import ALL_ACTIONS

POLICIES = ("noop", "scripted", "degraded", "random", "oracle")
REPORT = "turn done"


def rng_for(ep, name: str, seed: int = 0) -> random.Random:
    """A generator seeded by the task and the policy (zlib.crc32: stable across processes, unlike hash())."""
    return random.Random(zlib.crc32(f"{ep.row.get('task_id') or ep.row['state_fp']}|{name}|{seed}".encode()))


def end(ep, report: str = REPORT) -> None:
    """end_turn, once more if the first is refused (the open-requests reminder), while the turn isn't done."""
    for _ in range(2):
        if ep.done:
            return
        ep.step("end_turn", {"report": report})


def run_calls(ep, calls: list[tuple[str, dict]], *, confirm: bool = True, report: str = REPORT) -> None:
    """Send the calls in order (one step kept for end_turn; actions past the action cap skipped, as the scripted
    fleet skips them), repeating a refusal the policy means with confirm: true; then end the turn."""
    for name, args in calls:
        if ep.done or len(ep.trace) >= ep.max_steps - 1:
            break
        if name in ALL_ACTIONS and ep.capped:
            continue
        ep.step(name, json.dumps(args, default=_plain))
        res = ep.last_result
        if (confirm and name in ALL_ACTIONS and res.get("refused") in CONFIRMED and not args.get("confirm")
                and not ep.done and len(ep.trace) < ep.max_steps - 1):
            ep.step(name, json.dumps({**args, "confirm": True}, default=_plain))
    end(ep, report)


def scripted_calls(ep) -> tuple[list[tuple[str, dict]], str]:
    """(calls, report): the scripted fleet agent's decision for this turn as tool calls (its look-ups run on a
    scratch toolbox: they only read), the mayor's plan update first."""
    from ..fleet.tools import ToolBox
    agent = ep.agents[ep.role]
    scratch = ToolBox(ep.env, ep.role, style=agent.cfg.briefing)
    kept = agent.policy                   # decide on a copy of the policy: the branch's scripted agent for this role
    agent.policy = copy.deepcopy(kept)    # plays on from where it was, as in a model's branch (it never decided now)
    try:
        d = agent.decide(lambda name: scratch.call(name, "{}"))
        plan = agent.plan_args() if ep.role == "mayor" else {}
    finally:
        agent.policy = kept
    calls: list[tuple[str, dict]] = [("update_plan", plan)] if plan else []
    for a in d.actions:
        args = dict(a)
        name = args.pop("type", None)
        if name in ALL_ACTIONS:
            calls.append((name, args))
    calls += [("send_message", {"to": m.get("to", "all"), "text": m["text"]}) for m in d.messages]
    return calls, (d.thinking or REPORT)[:300]


_CALL = re.compile(r"\b([a-z_]+) (?=\{)")


def text_calls(text: str, names) -> list[tuple[str, dict]]:
    """`name {json}` calls written in a briefing, in order, for tool names in `names`."""
    dec, out = json.JSONDecoder(), []
    for m in _CALL.finditer(text):
        if m.group(1) not in names:
            continue
        try:
            args, _ = dec.raw_decode(text, m.end())
        except ValueError:
            continue
        if isinstance(args, dict) and (m.group(1), args) not in out:
            out.append((m.group(1), args))
    return out


def briefing_calls(ep) -> list[tuple[str, dict]]:
    """The guided briefing's own calls for this turn (see the module doc), whatever the task's style."""
    from ..fleet.briefing import briefing
    from ..fleet.memory import FleetState
    from ..fleet.tools import fund_cover, funding_gaps
    if ep.role == "mayor":
        f = funding_gaps(ep.env, FleetState.of(ep.env))
        cover = set(fund_cover(f))
        return [("allocate_budget", {"department": i.dept, "amount": int(i.amount)})
                for i in f.to_fund if i.dept in cover]
    text = briefing(ep.env, ep.role, mode=ep.fleet_cfg.mode, mark=False, style="guided")
    block = text.split("## Your budget", 1)[-1].split("\n## ", 1)[0] if "## Your budget" in text else ""
    return text_calls(block, set(ep.box.tools) & set(ALL_ACTIONS))


def _xy(rng, w, h) -> dict:
    x, y = rng.randrange(w), rng.randrange(h)
    return {"x": x, "y": y}


def _rect(rng, w, h) -> dict:
    x1, y1 = rng.randrange(w), rng.randrange(h)
    return {"x1": x1, "y1": y1, "x2": min(w - 1, x1 + rng.randint(0, 4)), "y2": min(h - 1, y1 + rng.randint(0, 4))}


KINDS = {"build_power_plant": ["coal", "gas", "solar", "wind"], "build_water": ["pump", "tower"],
         "build_sewage": ["outlet", "treatment"], "build_school": ["elementary", "high"],
         "build_landmark": ["stadium", "monument", "plaza"], "build_connector": ["highway", "rail"]}


def random_args(ep, name: str, rng: random.Random) -> dict:
    """Arguments of the right types for `name` (its schema's properties), drawn at random over the map."""
    from ..fleet.memory import DEPARTMENTS
    from ..sim import catalog as C
    h, w = ep.env.world.struct.shape
    props = ep.box.tools[name].properties
    args: dict = {}
    if {"x1", "y1"} <= set(props):
        args.update(_rect(rng, w, h))
        if "x" in props:                          # build_park: one tile or a block
            args = {k: v for k, v in args.items() if rng.random() < 0.5} or _xy(rng, w, h)
    elif {"x", "y"} <= set(props):
        args.update(_xy(rng, w, h))
    picks = {"kind": KINDS.get(name, ["basic"]), "zone": ["R", "C", "I", "O"], "density": ["low", "high"],
             "department": list(DEPARTMENTS), "sector": ["R", "C", "I"], "module": ["tier"], "to": ["avenue"],
             "policy": sorted(C.POLICIES), "industry": ["farming", "forest", "oil", "ore"]}
    for k in props:
        if k in args or k == "confirm" or k in ("tier", "x", "y", "x1", "y1", "x2", "y2"):
            continue
        if k in picks:
            args[k] = rng.choice(picks[k])
        elif k == "amount":
            args[k] = rng.choice([500, 1000, 2500, 5000]) * (1 if name != "allocate_budget" or rng.random() < 0.8
                                                              else -1)
        elif k == "rate":
            args[k] = rng.randint(5, 12)
        elif k == "percent":
            args[k] = rng.choice([80, 100, 120])
        elif k == "delta":
            args[k] = rng.choice([-1, 1])
        elif k == "fare":
            args[k] = rng.choice([0.5, 1.0, 2.0])
        elif k in ("stops", "stations"):
            args[k] = [[rng.randrange(w), rng.randrange(h)] for _ in range(rng.randint(2, 4))]
        elif k in ("line", "offer"):
            args[k] = rng.randint(0, 2)
        elif k == "enabled":
            args[k] = rng.random() < 0.5
        elif k in ("name", "district"):
            args[k] = "D1"
    return args


def play(ep, name: str, seed: int = 0) -> None:
    """Play policy `name` on a reset episode until its turn is done."""
    if name == "noop":
        end(ep, "nothing to do")
    elif name == "scripted":
        calls, report = scripted_calls(ep)
        run_calls(ep, calls, report=report)
    elif name == "degraded":
        calls, report = scripted_calls(ep)
        rng = rng_for(ep, name, seed)
        acts = [i for i, (n, _) in enumerate(calls) if n in ALL_ACTIONS]
        drop = {i for i in acts if rng.random() < 0.5}
        if acts and not drop:
            drop = {rng.choice(acts)}
        run_calls(ep, [c for i, c in enumerate(calls) if i not in drop], report=report)
    elif name == "random":
        rng = rng_for(ep, name, seed)
        tools = sorted(t for t in ep.box.tools if t in ALL_ACTIONS)
        calls = [(t, random_args(ep, t, rng)) for t in (rng.choice(tools) for _ in range(rng.randint(1, 4)))]
        run_calls(ep, calls, confirm=False, report="random")
    elif name == "oracle":
        first = briefing_calls(ep)
        calls, report = scripted_calls(ep)
        run_calls(ep, first + [c for c in calls if c not in first], report=report)
    else:
        raise ValueError(f"unknown policy {name!r}; policies: {', '.join(POLICIES)}")
