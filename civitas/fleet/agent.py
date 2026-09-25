"""ToolAgent: one turn = a tool-calling loop over the fleet tools, starting from the briefing.

Actions run inside `act()` (through `ToolBox`), so the returned `Decision` carries their results in `executed`
and the council messages to post; `env.apply` only logs them. In `game.simultaneous` mode fleet agents therefore
act on the state left by the roles before them.
"""
from __future__ import annotations

import json

from ..agents.base import Agent
from ..agents.llm_client import ChatClient, LLMError, ProviderConfig
from ..core.decision import Decision
from ..core.region import RegionEnv
from .briefing import briefing, system_prompt
from .config import FleetAgentConfig
from .memory import FleetState
from .tools import ToolBox

RESULT_CHARS = 4000
NUDGE = "Use the tools to act, then call end_turn with a short report."


def _norm_call(c: dict, step: int, i: int) -> dict:
    fn = c.get("function") or {}
    args = fn.get("arguments")
    if not isinstance(args, str):
        args = json.dumps(args if args is not None else {})
    return {"id": c.get("id") or f"call_{step}_{i}", "type": "function",
            "function": {"name": fn.get("name") or "", "arguments": args}}


class ToolAgent(Agent):
    kind = "fleet"

    def __init__(self, role: str, *, cfg: FleetAgentConfig | None = None, provider: ProviderConfig | None = None,
                 **kw):
        super().__init__(role, **kw)
        self.cfg = cfg or FleetAgentConfig()
        self.provider = provider or ProviderConfig()
        self.client = ChatClient(self.provider)
        self.env = None
        self.system = ""
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "tool_calls": 0, "api_errors": 0,
                      "latency_s": 0.0}

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

    def act(self, obs: dict) -> Decision:
        if self.env is None:
            raise RuntimeError("ToolAgent.act before bind(env)")
        env, role, cfg = self.env, self.role, self.cfg
        box = ToolBox(env, role)
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": briefing(env, role, mode=cfg.mode)}]
        tools, trace, reasoning = box.schemas(), [], []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        nudged, error, reason, latency, step = False, None, "max_steps", 0.0, 0
        while len(trace) < cfg.max_steps:
            try:
                out = self.client.chat(cfg.model, messages, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                                       extra={"tools": tools, **cfg.extra_body})
            except LLMError as e:
                self.usage["api_errors"] += 1
                error, reason = f"api_error: {e}", "api_error"
                break
            self.usage["calls"] += 1
            latency += out.get("latency_s") or 0.0
            for k in usage:
                usage[k] += int((out.get("usage") or {}).get(k) or 0)
            if out.get("reasoning"):
                reasoning.append(f"[step {step}] {out['reasoning']}")
            calls = [_norm_call(c, step, i) for i, c in enumerate(out.get("tool_calls") or [])]
            messages.append({"role": "assistant", "content": out.get("content") or "",
                             **({"tool_calls": calls} if calls else {})})
            if not calls:
                if nudged:
                    reason = "no_tool_calls"
                    break
                nudged = True
                messages.append({"role": "user", "content": NUDGE})
                step += 1
                continue
            for c in calls:
                name, args = c["function"]["name"], c["function"]["arguments"]
                if box.ended:
                    res = {"ok": False, "error": "not run: the turn already ended (end_turn)"}
                elif len(trace) >= cfg.max_steps:
                    res = {"ok": False, "error": f"not run: tool-call limit reached ({cfg.max_steps} per turn)"}
                else:
                    res = box.call(name, args)
                    trace.append({"step": step, "tool": name, "args": args, "result": res})
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": json.dumps(res, separators=(",", ":"))[:RESULT_CHARS]})
            step += 1
            if box.ended:
                reason = "end_turn"
                break
            if usage["completion_tokens"] >= cfg.turn_tokens:
                reason = "turn_tokens"
                break
        for k in usage:
            self.usage[k] += usage[k]
        self.usage["tool_calls"] += len(trace)
        self.usage["latency_s"] = round(self.usage["latency_s"] + latency, 3)
        d = Decision(executed=box.executed, messages=box.messages, thinking=box.report or "(no report)")
        d.parse_error = error
        d.meta = {"model": cfg.model, "messages": messages, "tool_trace": trace, "usage": usage,
                  "steps": len(trace), "ended": box.ended, "finish_reason": reason,
                  "reasoning": "\n\n".join(reasoning),
                  "response": messages[-1]["content"] if messages[-1]["role"] == "assistant" else "",
                  "latency_s": round(latency, 3)}
        return d
