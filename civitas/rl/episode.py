"""TaskEpisode: one department's turn of a bank task through the real fleet tools.

    ep = TaskEpisode(bank.get("fleet_smoke_late/s100/m48/planning"))
    obs = ep.reset()          # {"system_prompt", "briefing", "tools", "messages", ...}
    text = ep.step("find_sites", '{"purpose": "zone_R"}')        # the tool message content
    ...
    ep.step("end_turn", '{"report": "zoned R"}')                  # ep.done
    r = ep.finish()           # the reward dict (reward.py), the branch's scores under "series"

What it mirrors of fleet/agent.py:ToolAgent.act, and how:
- the system prompt is a ToolAgent's own (one is built and bound to the task's env; it never gets a client), the
  briefing is `fleet/briefing.py:briefing` in the task config's style and mode, and the tools are the schemas of a
  `ToolBox(env, role, branch_dezone=True, max_steps=..., next_line=True, style=...)`: the model's toolbox, with its
  guards, refusals, hints, refreshed Next lines, end_turn reminders and held-back money asks;
- a tool result is what ToolAgent puts in the tool message: `_budget` (steps_left, wrap-up notes) unless the call
  ended the turn, as compact JSON cut at TOOL_MESSAGE_CHARS; calls after end_turn or past max_steps get ToolAgent's
  "not run" results without running;
- `reply(content, tool_calls, ...)` takes one whole assistant reply as ToolAgent's loop does: calls recovered from
  the text when there are none (`_recover`), NUDGE for a reply without a call (once), LENGTH_NUDGE for one cut at the
  token limit (up to LENGTH_NUDGES), the turn ending on a further reply without a call; `messages` then holds what a
  ToolAgent would send next. `replay(messages)` feeds a whole OpenAI-format conversation through `reply`, and with
  `check=True` reports any tool message that differs from what the episode returned.
- done when end_turn succeeds, after max_steps tool calls, or (reply only) on a reply without a call past the nudge;
  `end_at_action_cap=True` also ends the turn once the action cap is reached (ToolAgent doesn't: after the cap the
  model can still message and report, and a message to the mayor changes what the scripted mayor funds).
- the token budget (civitas-v3, fleet/budget.py): `turn_budget` (None: the task config's `[fleet] turn_budget`, 0 =
  off for every v2 bank) gives the system prompt its rule and the briefing its line, and then each reply counts
  against it in the budget's units (the request's characters / 4, and the completion from the reply's `usage` as the
  provider reported it, or its characters / 4 when none is given), every tool result ends with ToolAgent's "budget
  left" line, end_turn skips its open-requests reminder when the next reply may not fit (ToolBox.budget_low), and the
  turn is done ("turn_budget", a refused end_turn's report, else ToolAgent's cut report) once the next
  reply wouldn't fit: check `done` before sampling it, as ToolAgent checks before each call. A ToolAgent trajectory's
  `call_usage` replays a logged turn's cutoff exactly (`replay(messages, usage=...)`).
What is the trainer's: turn_tokens and max_tokens, sampling, and whether to send the nudge messages `reply` appends
(`step` sends none).
"""
from __future__ import annotations

import json
import pickle
import threading
from collections import OrderedDict

from ..core.decision import Decision
from ..fleet.agent import LENGTH_NUDGE, LENGTH_NUDGES, NUDGE, _budget, _norm_call, _recover, _reply_chars, _slim
from ..fleet.budget import CUT_REPORT, MESSAGE_CHARS, TurnBudget
from ..fleet.fmt import TOOL_MESSAGE_CHARS
from . import bank as B
from . import reward as R

NO_MODEL = {"base_url": "http://127.0.0.1:9/civitas-rl-task-has-no-model", "api_key_env": "", "max_retries": 0,
            "timeout": 1.0, "deadline": 1.0, "stream": False}
_PREPARED: OrderedDict[tuple, bytes] = OrderedDict()   # (snapshot sha, role, fp) -> pickled state at the role's turn
_LOCK = threading.Lock()                               # evaluate.py plays tasks in threads


class TaskEpisode:
    def __init__(self, task: B.Task | None = None, *, end_at_action_cap: bool = False, check: bool = True,
                 row: dict | None = None, cfg: dict | None = None, blob: bytes | None = None,
                 turn_budget: int | None = None):
        if task is not None:
            row, cfg = task.row, task.cfg
        if row is None or cfg is None:
            raise ValueError("TaskEpisode needs a bank task (or a row, its config and its snapshot bytes)")
        self.task, self.row, self.cfg, self._blob = task, row, cfg, blob
        self.end_at_action_cap, self.check = end_at_action_cap, check
        if turn_budget is not None and (isinstance(turn_budget, bool) or not isinstance(turn_budget, int)
                                        or turn_budget < 0):
            raise ValueError(f"turn_budget must be a whole number of tokens (0 = off), got {turn_budget!r}")
        self.turn_budget = turn_budget          # None: the task config's [fleet] turn_budget
        self.budget: TurnBudget | None = None
        self.env = self.agents = self.box = None
        self.trace: list[dict] = []
        self.messages: list[dict] = []
        self.reason: str | None = None
        self.replies = self.cuts = 0
        self.nudged = self.stopped = self.finished = False

    @classmethod
    def from_blob(cls, blob: bytes, row: dict, cfg: dict, **kw) -> "TaskEpisode":
        """An episode of a task row whose bank isn't written yet (bank.build's noop branches)."""
        return cls(row=row, cfg=cfg, blob=blob, **kw)

    # ---------------------------------------------------------------------------------------------- state
    @property
    def role(self) -> str:
        return self.row["role"]

    @property
    def fleet_cfg(self):
        from ..fleet.config import fleet_agent_config
        fc = fleet_agent_config(self.cfg, self.role)
        if self.turn_budget is not None:
            fc.turn_budget = self.turn_budget
        return fc

    def prepare(self):
        """The env (and scripted agents) at the role's turn: the snapshot, the roles before it this month played by
        the scripted fleet (bank.advance), its memory emptied (bank.settle); checked against the task's fingerprint.
        Cached per process, so later episodes of the task unpickle it. Returns the env; no briefing is built (a
        ToolAgent playing the turn builds its own: evaluate.py)."""
        key = (self.row["snapshot_sha256"], self.role, self.row["state_fp"])
        with _LOCK:
            data = _PREPARED.get(key)
            if data is not None:
                _PREPARED.move_to_end(key)
        if data is not None:
            self.env, self.agents = pickle.loads(data)          # our own pickle of a checked state
            B._balance(self.env)
            return self.env
        blob = self._blob if self._blob is not None else self.task.blob()
        env, agents = B.load_state(blob, self.row["snapshot_sha256"])
        if env.world.month != self.row["month"]:
            raise B.BankError(f"task {self.row.get('task_id')}: snapshot at month {env.world.month}, task month "
                              f"{self.row['month']}")
        B.advance(env, agents, self.role)
        B.settle(env, self.role)
        if self.check and B.state_fp(env) != self.row["state_fp"]:
            raise B.BankError(f"task {self.row.get('task_id')}: the city at the role's turn isn't the one the bank "
                              "recorded (the sim, the scripted agents or the tools changed since the bank was built): "
                              "rebuild the bank")
        data = pickle.dumps((env, agents), protocol=pickle.HIGHEST_PROTOCOL)
        with _LOCK:
            _PREPARED[key] = data
            while len(_PREPARED) > B.CACHE_STATES:
                _PREPARED.popitem(last=False)
        self.env, self.agents = env, agents
        return env

    def reset(self) -> dict:
        """Prepare the state and open the turn as ToolAgent.act does: its toolbox, then the briefing (which marks the
        turn), then the tool schemas. Returns the first two messages and what they're made of."""
        from ..agents.llm_client import ProviderConfig
        from ..fleet.agent import ToolAgent
        from ..fleet.briefing import briefing
        from ..fleet.tools import ToolBox
        if self.box is not None:
            raise RuntimeError("TaskEpisode.reset: an episode plays one turn; make a new one")
        if self.env is None:
            self.prepare()
        fc = self.fleet_cfg
        agent = ToolAgent(self.role, cfg=fc, provider=ProviderConfig(**NO_MODEL))
        agent.client = None                               # prompts only: this agent never calls a model
        agent.bind(self.env)
        self.system = agent.system
        self.box = ToolBox(self.env, self.role, branch_dezone=True, max_steps=fc.max_steps, next_line=True,
                           style=fc.briefing)
        self.briefing = briefing(self.env, self.role, mode=fc.mode, style=fc.briefing, turn_budget=fc.turn_budget)
        self.tools = self.box.schemas()
        self.max_steps = fc.max_steps
        self.budget = TurnBudget(fc.turn_budget, fc.budget_reserve, self.tools) if fc.turn_budget else None
        if self.budget is not None:                       # ToolAgent's hook: no end_turn reminder past the budget
            b = self.budget
            self.box.budget_low = lambda: b.low(b.chars(self.messages))
        self.messages = [{"role": "system", "content": self.system}, {"role": "user", "content": self.briefing}]
        out = {"task_id": self.row.get("task_id"), "role": self.role, "month": self.row["month"],
               "style": fc.briefing, "system_prompt": self.system, "briefing": self.briefing, "tools": self.tools,
               "messages": [dict(m) for m in self.messages], "max_steps": fc.max_steps,
               "max_actions": self.env.cfg.max_actions_per_turn}
        if self.budget is not None:
            out["turn_budget"] = self.budget.limit
        return out

    # ---------------------------------------------------------------------------------------------- turn
    @property
    def capped(self) -> bool:
        return sum(1 for r in self.box.executed if r.get("ok")) >= self.env.cfg.max_actions_per_turn

    @property
    def done(self) -> bool:
        return bool(self.box is not None and (self.box.ended or len(self.trace) >= self.max_steps or self.stopped
                                               or (self.end_at_action_cap and self.capped)))

    def _run(self, name: str, raw: str) -> str:
        """One tool call as ToolAgent runs it (see the module doc); the tool message content."""
        if self.box is None:
            raise RuntimeError("TaskEpisode: call reset() first")
        if self.box.ended:
            res = {"ok": False, "error": "not run: the turn already ended (end_turn)"}
        elif len(self.trace) >= self.max_steps:
            res = {"ok": False, "error": f"not run: tool-call limit reached ({self.max_steps} per turn)"}
        elif self.end_at_action_cap and self.capped:
            res = {"ok": False, "error": "not run: the action limit ended the turn"}
        elif self.budget is not None and self.budget.cut:     # ToolAgent makes no further call
            res = {"ok": False, "error": "not run: the turn's token budget ran out"}
        else:
            res = self.box.call(name, raw)
            self.trace.append({"step": self.replies, "finish_reason": None, "tool": name, "args": raw,
                               "result": _slim(res)})
            if not self.box.ended:
                res = _budget(res, self.max_steps - len(self.trace))
            if self.budget is not None:
                self.trace[-1]["tokens_left"] = max(0, self.budget.left)
        self.last_result = res
        self._reason()
        content = json.dumps(res, separators=(",", ":"))[:TOOL_MESSAGE_CHARS]
        if self.budget is not None and not self.box.ended:     # ToolAgent's "budget left" line
            content += self.budget.suffix(self.budget.chars(self.messages) + len(content) + MESSAGE_CHARS)
        return content

    def _count(self, usage: dict | None, out: dict) -> None:
        """Count one reply against the token budget (before its messages join the transcript)."""
        if self.budget is not None:
            self.budget.record(usage, self.budget.chars(self.messages), _reply_chars(out))

    def _budget_check(self) -> None:
        """ToolAgent's check before its next call: done ("turn_budget", the cut report) when that reply won't fit."""
        b = self.budget
        if b is not None and not self.done and not b.fits(b.chars(self.messages)):
            b.cut, self.stopped, self.reason = True, True, "turn_budget"
            self.box.close(self.box.refused_report or CUT_REPORT)

    @property
    def budget_used(self) -> int | None:
        """Tokens counted against the budget so far (None without one)."""
        return self.budget.used if self.budget is not None else None

    @property
    def budget_over(self) -> int | None:
        """Tokens counted beyond the budget (None without one): ToolAgent's `budget_over`."""
        return self.budget.over if self.budget is not None else None

    def _reason(self) -> None:
        if self.box.ended:
            self.reason = "end_turn"
        elif len(self.trace) >= self.max_steps:
            self.reason = "max_steps"
        elif self.end_at_action_cap and self.capped:
            self.reason = "action_cap"

    def step(self, name: str, arguments: str | dict | None = None, *, usage: dict | None = None) -> str:
        """Run one tool call (arguments as the model sent them: JSON text, or a dict) and return the tool message
        content; the transcript in `messages` gets the call and its result as a one-call reply. `usage`: the reply's
        tokens, for the token budget."""
        raw = arguments if isinstance(arguments, str) else json.dumps(arguments if arguments is not None else {})
        cid = f"call_{len(self.trace)}_{self.replies}"
        call = {"id": cid, "type": "function", "function": {"name": name, "arguments": raw}}
        self._count(usage, {"tool_calls": [call]})
        self.messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        content = self._run(name, raw)
        self.messages.append({"role": "tool", "tool_call_id": cid, "content": content})
        self.replies += 1
        self._budget_check()
        return content

    def reply(self, content: str = "", tool_calls: list | None = None, *, reasoning: str = "",
              finish_reason: str | None = None, nudges: bool = True, usage: dict | None = None) -> list[str]:
        """One assistant reply, handled as one iteration of ToolAgent's loop (module doc). Returns the tool message
        contents in call order; `messages` gets the reply, the tool messages and any nudge (when `nudges`). `usage`:
        the reply's prompt / completion tokens as the provider reported them, for the token budget."""
        if self.box is None:
            raise RuntimeError("TaskEpisode: call reset() first")
        out = {"content": content or "", "reasoning": reasoning or "", "finish_reason": finish_reason}
        self._count(usage, {**out, "tool_calls": list(tool_calls or [])})
        raw_calls = list(tool_calls or []) or _recover(out)
        pairs = [_norm_call(c, self.replies, i) for i, c in enumerate(raw_calls)]
        calls = [c for c, _ in pairs]
        self.messages.append({"role": "assistant", "content": content or ("" if calls else "(no reply)"),
                              **({"tool_calls": calls} if calls else {})})
        nudge = None
        if not calls:
            if finish_reason == "length" and self.cuts < LENGTH_NUDGES:
                self.cuts, nudge = self.cuts + 1, LENGTH_NUDGE
            elif not self.nudged:
                self.nudged, nudge = True, NUDGE
            else:
                self.stopped, self.reason = True, "no_tool_calls"
        results = []
        for c, raw in pairs:
            text = self._run(c["function"]["name"], raw)
            self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": text})
            results.append(text)
        self.replies += 1
        if nudge and nudges and not self.done:
            self.messages.append({"role": "user", "content": nudge})
        self._budget_check()
        return results

    def replay(self, messages: list[dict], *, check: bool = False, usage: list | None = None) -> list[dict]:
        """Play a whole OpenAI-format conversation's assistant replies (system / user / tool messages are skipped:
        the episode makes its own) until the turn is done. Each call: {id, name, content}, plus with `check`
        `match`: whether the conversation's tool message for that id says the same (False when it differs, None when
        it has none). With `check` a differing system prompt or briefing raises ValueError. `usage`: one entry per
        assistant reply, in order (a ToolAgent trajectory's `call_usage`: the provider's usage, and in civitas-v3 the
        units the budget counted, which are taken as they are), so a v3 turn is cut where it was."""
        if self.box is None:
            self.reset()
        given = {m.get("tool_call_id"): m.get("content") for m in messages if m.get("role") == "tool"}
        if check:
            for want, (i, role) in ((self.system, (0, "system")), (self.briefing, (1, "user"))):
                if len(messages) > i and messages[i].get("role") == role and messages[i].get("content") != want:
                    raise ValueError(f"replay: message {i} ({role}) differs from this task's")
        out, k = [], 0
        for m in messages:
            if m.get("role") != "assistant" or self.done:
                continue
            calls = m.get("tool_calls") or []
            n0 = len(self.messages)
            u = usage[k] if usage is not None and k < len(usage) else None
            k += 1
            self.reply(m.get("content") or "", calls, reasoning=m.get("reasoning_content") or m.get("reasoning") or "",
                       usage=u)
            ran = [x for x in self.messages[n0:] if x.get("role") == "tool"]
            for c, t in zip(calls or [None] * len(ran), ran):
                cid = (c or {}).get("id") or t["tool_call_id"]
                rec = {"id": cid, "name": ((c or {}).get("function") or {}).get("name"), "content": t["content"]}
                if check:
                    rec["match"] = None if cid not in given else given[cid] == t["content"]
                out.append(rec)
        return out

    def decision(self) -> Decision:
        """The turn as the runner logs it (what ToolAgent.act returns, without the transcript)."""
        return Decision(executed=list(self.box.executed), messages=list(self.box.messages),
                        thinking=self.box.report or "(no report)")

    # ---------------------------------------------------------------------------------------------- reward
    def finish(self, decision: Decision | None = None, *, horizon: int | None = None,
               reward_cfg: R.RewardConfig | None = None, series_only: bool = False) -> dict:
        """End the turn and score it: the turn's decision applied as the runner applies it (the episode's own, or a
        ToolAgent's `decision` on the prepared env: evaluate.py), the branch played on `horizon` months (default the
        reward's), and the reward against the task's baseline (reward.compute) with the branch under "series" and
        the turn's counts. `series_only`: just the branch's scores."""
        if self.finished:
            raise RuntimeError("TaskEpisode.finish: already finished")
        cfg = reward_cfg or R.RewardConfig()
        if decision is None:
            if self.box is None:
                raise RuntimeError("TaskEpisode.finish: no turn played (reset() and step(), or pass a decision)")
            decision, trace, ended = self.decision(), self.trace, self.box.ended
        else:
            if self.env is None:
                raise RuntimeError("TaskEpisode.finish: prepare() first")
            meta = decision.meta or {}
            trace, ended = meta.get("tool_trace") or [], bool(meta.get("ended"))
        self.finished = True
        self.env.apply(self.role, decision)
        series = R.play_on(self.env, self.agents, self.role, max(int(horizon or 0), int(cfg.horizon)))
        if series_only:
            return series
        n = R.counts(decision.executed, trace, ended)
        return {**R.compute(self.row, series, n, cfg), "series": series, "counts": n}
