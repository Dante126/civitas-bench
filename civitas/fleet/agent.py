"""ToolAgent: one turn = a tool-calling loop over the fleet tools, starting from the briefing.

Actions run inside `act()` (through `ToolBox`), so the returned `Decision` carries their results in `executed`
and the council messages to post; `env.apply` only logs them. In `game.simultaneous` mode fleet agents therefore
act on the state left by the roles before them. In region mode the agent binds to its own city's env (`bind`), so
the region runs it like any other agent of that city (fleet/region.py). With `[fleet] turn_budget` (civitas-v3) the
turn also ends before a model call that wouldn't fit the token budget (fleet/budget.py). Every turn logs each model
call's provider usage in `call_usage` (with the budget's units under "budget" in v3), so TaskEpisode.replay
reproduces a v3 turn's cutoff exactly.
"""
from __future__ import annotations

import json
import re

from ..agents.base import Agent
from ..bench import live
from ..agents.llm_client import ChatClient, ProviderConfig
from ..core.decision import Decision
from ..core.region import RegionEnv
from .briefing import briefing, system_prompt
from .budget import CUT_REPORT, MESSAGE_CHARS, TurnBudget, call_record
from .config import FleetAgentConfig
from .fmt import TOOL_MESSAGE_CHARS, briefing_style
from .memory import FleetState
from .region import link, region_of, region_rules
from .tools import ToolBox

TRACE_CHARS = 300
WRAP_UP_AT = 3               # tool results carry a wrap-up note from this many calls left
NUDGE = "Use the tools to act, then call end_turn with a short report."
# smoke v9: seed 0's planning lost its month-0 turn to two replies cut at max_tokens (16k) mid-reasoning, with the
# one generic nudge spent between them. A cut reply without a call gets this instead, up to LENGTH_NUDGES a turn;
# after those it counts as any reply without a call (the generic nudge if unused, else the turn ends). No nudge goes
# out past turn_tokens; at most 4 replies without a call (64k tokens at the default 16k) fit in its default 80k
LENGTH_NUDGE = ("Your reply was cut off at the token limit before any tool call. Keep your reasoning short: pick one "
                "action and call its tool now.")
LENGTH_NUDGES = 2


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
    out.update({k: res[k] for k in ("cost", "budget_left", "repeat", "refused", "override") if k in res})
    return out


def _reply_chars(out: dict) -> int:
    """A reply's characters (content, reasoning, tool calls): the token budget's estimate when usage is missing."""
    return (len(out.get("content") or "") + len(out.get("reasoning") or "")
            + len(json.dumps(out.get("tool_calls") or [], separators=(",", ":"))))


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
                 city: str | None = None, **kw):
        super().__init__(role, **kw)
        self.city = city                 # region mode: the city this agent works for (role is the base role)
        self.cfg = cfg or FleetAgentConfig()
        self.provider = provider or ProviderConfig()
        self.client = ChatClient(self.provider)
        self.env = None
        self.system = ""
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "tool_calls": 0, "api_errors": 0,
                      "latency_s": 0.0}

    @property
    def name(self) -> str:
        """The role as the run names it: "north.mayor" in region mode, else the role."""
        return f"{self.city}.{self.role}" if self.city else self.role

    def bind(self, env) -> None:
        """Bind to a single-city env, or in a region (a RegionEnv, or one of its city envs) to this agent's city env:
        its fleet state, tools and briefings are that city's alone (fleet/region.py)."""
        region = None
        if isinstance(env, RegionEnv):
            if self.city not in env.envs:
                raise ValueError(f"fleet agent {self.role!r} in a region needs its city (one of {', '.join(env.envs)}), "
                                 f"got {self.city!r}: name the role as '<city>.{self.role}'")
            region, env = env, env.envs[self.city]
            link(region)
        elif getattr(env, "city", None):
            if self.city not in (None, env.city):
                raise ValueError(f"fleet agent {self.name!r} bound to the {env.city} city's env")
            region, self.city = region_of(env), env.city
        self.env = env
        FleetState.of(env)
        c = env.cfg
        with briefing_style(self.cfg.briefing):          # the region section's deal rule reads the style
            self.system = system_prompt(self.role, self.cfg, c.months, act_every=c.act_every,
                                        max_actions=c.max_actions_per_turn, target_pop=env.scoring.target_population,
                                        region=region_rules(region, env.city, self.role) if region else "")

    def describe(self) -> str:
        return f"fleet:{self.cfg.model}"

    def act(self, obs: dict) -> Decision:
        if self.env is None:
            raise RuntimeError("ToolAgent.act before bind(env)")
        env, role, cfg = self.env, self.role, self.cfg
        try:
            # the scripted fleet runs its own dezones and gets no end_turn reminder of open requests, nor refreshed
            # Next lines in its action results
            box = ToolBox(env, role, branch_dezone=True, max_steps=cfg.max_steps, next_line=True, style=cfg.briefing)
            messages = [{"role": "system", "content": self.system},
                        {"role": "user", "content": briefing(env, role, mode=cfg.mode, style=cfg.briefing,
                                                             turn_budget=cfg.turn_budget)}]
            tools = box.schemas()
        except Exception as e:                       # a briefing bug costs the role this turn, not the run
            return self._failed_turn(f"briefing_error: {type(e).__name__}: {e}", "briefing_error")
        trace, reasoning, finishes = [], [], []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        nudged, cuts, error, reason, latency, step, recovered = False, 0, None, "max_steps", 0.0, 0, 0
        tb = TurnBudget(cfg.turn_budget, cfg.budget_reserve, tools) if cfg.turn_budget else None   # civitas-v3
        if tb is not None:                           # end_turn skips its open-requests reminder when the reply after
                                                     # this one may not fit (ToolBox.budget_low)
            box.budget_low = lambda: tb.low(tb.chars(messages))
        calls_usage = []                             # per model call: the provider's usage (call_record) and, with a
                                                     # budget, its units: logged as call_usage (TaskEpisode.replay)
        while len(trace) < cfg.max_steps:
            sent = tb.chars(messages) if tb is not None else 0
            if tb is not None and not tb.fits(sent):    # the next reply wouldn't fit: the turn ends before it
                tb.cut, reason = True, "turn_budget"
                break
            live.emit(env, role, "think", step=step)
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
            calls_usage.append(call_record(out.get("usage")))
            if tb is not None:
                tb.record(out.get("usage"), sent, _reply_chars(out))
                calls_usage[-1]["budget"] = dict(tb.log[-1])
            if out.get("reasoning"):
                reasoning.append(f"[step {step}] {out['reasoning']}")
            finish = out.get("finish_reason")
            finishes.append(finish)
            raw_calls = out.get("tool_calls") or []
            from_text = not raw_calls
            if from_text:
                raw_calls = _recover(out)
            said = (out.get("content") or "").strip() or (out.get("reasoning") or "").strip()
            live.emit(env, role, "reply", step=step, text=live._clip(said, live.TEXT_CHARS),
                      latency_s=round(out.get("latency_s") or 0.0, 1), finish=finish)
            pairs = [_norm_call(c, step, i) for i, c in enumerate(raw_calls)]
            calls, raws = [c for c, _ in pairs], [r for _, r in pairs]
            messages.append({"role": "assistant", "content": out.get("content") or ("" if calls else "(no reply)"),
                             **({"tool_calls": calls} if calls else {})})
            nudge = None                             # a reply without a call gets one, sent below within turn_tokens
            if not calls:
                if finish == "length" and cuts < LENGTH_NUDGES:
                    cuts, nudge = cuts + 1, LENGTH_NUDGE
                elif not nudged:
                    nudged, nudge = True, NUDGE
                else:
                    reason = "no_tool_calls"
                    break
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
                    if tb is not None:
                        trace[-1]["tokens_left"] = max(0, tb.left)
                content = json.dumps(res, separators=(",", ":"))[:TOOL_MESSAGE_CHARS]
                if tb is not None and not box.ended:     # "budget left: ~N tokens" (and a wrap-up note when low)
                    content += tb.suffix(tb.chars(messages) + len(content) + MESSAGE_CHARS)
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": content})
            step += 1
            if box.ended:
                reason = "end_turn"
                break
            if usage["completion_tokens"] >= cfg.turn_tokens:
                reason = "turn_tokens"
                break
            if nudge:
                messages.append({"role": "user", "content": nudge})
        if tb is not None and tb.cut and not box.ended:
            box.close(box.refused_report or CUT_REPORT)   # end_turn's record of a report (a refused end_turn's),
                                                          # without its refusals
        for k in usage:
            self.usage[k] += usage[k]
        self.usage["tool_calls"] += len(trace)
        self.usage["latency_s"] = round(self.usage["latency_s"] + latency, 3)
        d = Decision(executed=box.executed, messages=box.messages, thinking=box.report or "(no report)")
        d.parse_error = error
        logged = [{"role": "system", "content": f"(system prompt: meta.json fleet_prompts.{self.name})"},
                  *messages[1:]]
        said = [m["content"] for m in messages if m["role"] == "assistant" and m.get("content")
                and m["content"] != "(no reply)"]
        d.meta = {"model": cfg.model, "messages": logged, "tool_trace": trace, "usage": usage,
                  "steps": len(trace), "ended": box.ended, "finish_reason": reason, "finish_reasons": finishes,
                  "length_stops": finishes.count("length"), "recovered_calls": recovered,
                  "reasoning": "\n\n".join(reasoning),
                  "response": box.report or (said[-1] if said else ""),   # the end_turn report, else the last reply
                  "latency_s": round(latency, 3), "call_usage": calls_usage}
        if tb is not None:                           # civitas-v3 only: a v2 turn plays as it always did
            d.meta.update({"turn_budget": tb.limit, "budget_used": tb.used, "budget_cut": tb.cut,
                           "budget_over": tb.over})
        return d

    def _failed_turn(self, error: str, reason: str) -> Decision:
        """A turn that ended before the first model call: nothing executed, the error logged."""
        d = Decision(thinking="(no report)")
        d.parse_error = error
        d.meta = {"model": self.cfg.model, "messages": [], "tool_trace": [],
                  "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "steps": 0, "ended": False,
                  "finish_reason": reason, "finish_reasons": [], "length_stops": 0, "recovered_calls": 0,
                  "reasoning": "", "response": "", "latency_s": 0.0, "call_usage": []}
        return d
