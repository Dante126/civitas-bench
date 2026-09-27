"""ToolAgent: one turn = a tool-calling loop over the fleet tools, starting from the briefing.

Actions run inside `act()` (through `ToolBox`), so the returned `Decision` carries their results in `executed`
and the council messages to post; `env.apply` only logs them. In `game.simultaneous` mode fleet agents therefore
act on the state left by the roles before them.
"""
from __future__ import annotations

import json
import re

from ..agents.base import Agent
from ..agents.llm_client import ChatClient, ProviderConfig
from ..core.decision import Decision
from ..core.region import RegionEnv
from .briefing import briefing, system_prompt
from .config import FleetAgentConfig
from .fmt import TOOL_MESSAGE_CHARS
from .memory import FleetState
from .tools import ToolBox

TRACE_CHARS = 300
WRAP_UP_AT = 3               # tool results carry a wrap-up note from this many calls left
NUDGE = "Use the tools to act, then call end_turn with a short report."


def _norm_call(c: dict, step: int, i: int) -> tuple[dict, str]:
    """(call for the history, raw arguments). Servers such as vLLM parse the arguments of past tool calls, so the
    history gets "{}" for arguments that are not a JSON object; the tool still sees the raw text and can say why."""
    fn = c.get("function") or {}
    raw = fn.get("arguments")
    raw = raw if isinstance(raw, str) else json.dumps(raw if raw is not None else {})
    try:
        ok = isinstance(json.loads(raw), dict)
    except ValueError:
        ok = False
    return ({"id": c.get("id") or f"call_{step}_{i}", "type": "function",
             "function": {"name": fn.get("name") or "invalid_tool", "arguments": raw if ok else "{}"}}, raw)


_OPEN, _CLOSE = "<tool_call>", "</tool_call>"
_XML_FN = re.compile(r"<function=([\w.-]+)>(.*?)(?:</function>|\Z)", re.S)
_XML_PARAM = re.compile(r"<parameter=([\w.-]+)>(.*?)</parameter>", re.S)


def _blocks(text: str) -> list[str]:
    """Bodies of the trailing run of <tool_call> blocks: only whitespace after the last one and between them, so
    calls drafted earlier and then abandoned don't run. A block ends at </tool_call>, at the next <tool_call>
    (an unclosed block doesn't swallow the next call) or at the end of the text."""
    spans, i = [], text.find(_OPEN)
    while i >= 0:
        body_start = i + len(_OPEN)
        nxt, close = text.find(_OPEN, body_start), text.find(_CLOSE, body_start)
        if close >= 0 and (nxt < 0 or close < nxt):
            spans.append((i, body_start, close, close + len(_CLOSE)))
        else:
            stop = nxt if nxt >= 0 else len(text)
            spans.append((i, body_start, stop, stop))
        i = nxt
    run = []
    tail = len(text)
    for start, body_start, body_end, end in reversed(spans):
        if text[end:tail].strip():
            break
        run.append(text[body_start:body_end])
        tail = start
    return run[::-1]


def _text_calls(text: str) -> list[dict]:
    """Tool calls written out as text: `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`
    or a JSON {"name", "arguments"} inside <tool_call>. trinity-large-thinking sometimes leaves its calls in the
    reasoning and stops without parsed `tool_calls`; the agent runs them rather than losing the step. XML values
    stay strings (ToolBox coerces them by the tool's schema); identical calls are kept once."""
    calls, seen = [], set()
    for body in _blocks(text or ""):
        fn = _XML_FN.search(body)
        if fn:
            name, args = fn.group(1), {k: v.strip() for k, v in _XML_PARAM.findall(fn.group(2))}
        else:
            try:
                obj = json.loads(body.strip())
            except ValueError:
                continue
            if not isinstance(obj, dict) or not obj.get("name"):
                continue
            name, args = str(obj["name"]), obj.get("arguments") or {}
        raw = args if isinstance(args, str) else json.dumps(args, sort_keys=True)
        if (name, raw) not in seen:
            seen.add((name, raw))
            calls.append({"type": "function", "function": {"name": name, "arguments": raw}})
    return calls


def _recover(out: dict) -> list[dict]:
    """Text tool calls from the reply's content, else from its reasoning; none from a truncated reply."""
    if out.get("finish_reason") == "length":
        return []
    return _text_calls(out.get("content") or "") or _text_calls(out.get("reasoning") or "")


def _slim(res: dict) -> dict:
    out = {"ok": res.get("ok")}
    for k in ("error", "message"):
        if res.get(k):
            out[k] = str(res[k])[:TRACE_CHARS]
    out.update({k: res[k] for k in ("cost", "budget_left", "repeat", "refused") if k in res})
    return out


def _budget(res: dict, left: int) -> dict:
    """The result with the tool calls left this turn, and a wrap-up note when few are."""
    out = {**res, "steps_left": left}
    if left == 0:                # trace-only: the loop stops here, so the model never reads it (the log does)
        out["wrap_up"] = "no tool calls left: the turn ends now without a report; completed actions stay"
    elif left == 1:
        out["wrap_up"] = "1 tool call left: call end_turn with your report now"
    elif left <= WRAP_UP_AT:
        out["wrap_up"] = f"{left} tool calls left: finish what matters, then call end_turn with your report"
    return out


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
        try:
            box = ToolBox(env, role)
            messages = [{"role": "system", "content": self.system},
                        {"role": "user", "content": briefing(env, role, mode=cfg.mode)}]
            tools = box.schemas()
        except Exception as e:                       # a briefing bug costs the role this turn, not the run
            return self._failed_turn(f"briefing_error: {type(e).__name__}: {e}", "briefing_error")
        trace, reasoning, finishes = [], [], []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        nudged, error, reason, latency, step, recovered = False, None, "max_steps", 0.0, 0, 0
        while len(trace) < cfg.max_steps:
            try:
                out = self.client.chat(cfg.model, messages, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                                       extra={**cfg.extra_body, "tools": tools})
            except Exception as e:                   # completed actions stay; the turn just ends
                self.usage["api_errors"] += 1
                error, reason = f"api_error: {type(e).__name__}: {e}", "api_error"
                break
            self.usage["calls"] += 1
            latency += out.get("latency_s") or 0.0
            for k in usage:
                usage[k] += int((out.get("usage") or {}).get(k) or 0)
            if out.get("reasoning"):
                reasoning.append(f"[step {step}] {out['reasoning']}")
            finish = out.get("finish_reason")
            finishes.append(finish)
            raw_calls = out.get("tool_calls") or []
            from_text = not raw_calls
            if from_text:
                raw_calls = _recover(out)
            pairs = [_norm_call(c, step, i) for i, c in enumerate(raw_calls)]
            calls, raws = [c for c, _ in pairs], [r for _, r in pairs]
            messages.append({"role": "assistant", "content": out.get("content") or ("" if calls else "(no reply)"),
                             **({"tool_calls": calls} if calls else {})})
            if not calls:
                if nudged:
                    reason = "no_tool_calls"
                    break
                nudged = True
                messages.append({"role": "user", "content": NUDGE})
                step += 1
                continue
            for c, args in zip(calls, raws):
                name = c["function"]["name"]
                if box.ended:
                    res = {"ok": False, "error": "not run: the turn already ended (end_turn)"}
                elif len(trace) >= cfg.max_steps:
                    res = {"ok": False, "error": f"not run: tool-call limit reached ({cfg.max_steps} per turn)"}
                else:
                    res = box.call(name, args)
                    recovered += from_text
                    trace.append({"step": step, "finish_reason": finish, "tool": name, "args": args,
                                  "result": _slim(res)})
                    if not box.ended:                    # end_turn's own result needs no budget
                        res = _budget(res, cfg.max_steps - len(trace))
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": json.dumps(res, separators=(",", ":"))[:TOOL_MESSAGE_CHARS]})
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
        logged = [{"role": "system", "content": f"(system prompt: meta.json fleet_prompts.{role})"}, *messages[1:]]
        said = [m["content"] for m in messages if m["role"] == "assistant" and m.get("content")
                and m["content"] != "(no reply)"]
        d.meta = {"model": cfg.model, "messages": logged, "tool_trace": trace, "usage": usage,
                  "steps": len(trace), "ended": box.ended, "finish_reason": reason, "finish_reasons": finishes,
                  "length_stops": finishes.count("length"), "recovered_calls": recovered,
                  "reasoning": "\n\n".join(reasoning),
                  "response": box.report or (said[-1] if said else ""),   # the end_turn report, else the last reply
                  "latency_s": round(latency, 3)}
        return d

    def _failed_turn(self, error: str, reason: str) -> Decision:
        """A turn that ended before the first model call: nothing executed, the error logged."""
        d = Decision(thinking="(no report)")
        d.parse_error = error
        d.meta = {"model": self.cfg.model, "messages": [], "tool_trace": [],
                  "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "steps": 0, "ended": False,
                  "finish_reason": reason, "finish_reasons": [], "length_stops": 0, "recovered_calls": 0,
                  "reasoning": "", "response": "", "latency_s": 0.0}
        return d
