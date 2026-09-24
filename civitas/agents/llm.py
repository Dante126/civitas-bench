"""LLM agent: reads the rendered observation, replies with JSON actions."""
from __future__ import annotations

import json
import re

from ..core.decision import Decision
from .base import Agent
from .llm_client import ChatClient, LLMError, ProviderConfig
from .prompts import system_prompt

THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def extract_json(text: str) -> dict:
    """Pull the decision object out of a model reply (tolerates prose, fences, <think>)."""
    text = THINK_RE.sub("", text or "").strip()
    candidates = [m.group(1) for m in FENCE_RE.finditer(text)]
    # balanced-brace scan for top-level objects
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
    parsed = []
    for c in candidates:
        for variant in (c, re.sub(r"//[^\n]*", "", c), re.sub(r",\s*([}\]])", r"\1", re.sub(r"//[^\n]*", "", c))):
            try:
                obj = json.loads(variant)
                if isinstance(obj, dict):
                    parsed.append(obj)
                    break
            except json.JSONDecodeError:
                continue
    if not parsed:
        raise ValueError("no JSON object found in the reply")
    with_actions = [p for p in parsed if "actions" in p]
    return (with_actions or parsed)[-1]


def _from_reasoning(text: str) -> dict:
    """Last resort: a decision object in the reasoning, else the action objects in its final part."""
    try:
        d = extract_json(text)
        if "actions" in d:
            return d
    except ValueError:
        d = None
    from ..core.actions import HANDLERS
    tail = text[int(len(text) * 0.75):]
    acts, seen = [], set()
    for m in re.finditer(r"\{[^{}]*\}", tail):
        try:
            obj = json.loads(re.sub(r",\s*}", "}", m.group(0)))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("type") in HANDLERS:
            key = json.dumps(obj, sort_keys=True)
            if key not in seen:
                seen.add(key)
                acts.append(obj)
    if acts:
        return {"actions": acts[-8:], "thinking": "(recovered from the model's reasoning)"}
    if d is not None:
        return d
    raise ValueError("no JSON object found in the reply")


class LLMAgent(Agent):
    kind = "llm"

    def __init__(self, role: str, *, model: str, provider: ProviderConfig | None = None,
                 temperature: float = 0.4, max_tokens: int = 12000, history_turns: int = 0,
                 months: int = 120, target_pop: int = 30000, max_actions: int = 8,
                 extra_body: dict | None = None, city: str | None = None, **kw):
        super().__init__(role, **kw)
        self.model = model
        self.provider = provider or ProviderConfig()
        self.client = ChatClient(self.provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history_turns = history_turns
        self.extra_body = extra_body
        self.system = system_prompt(role, months=months, target_pop=target_pop, max_actions=max_actions, city=city)
        self.history: list[dict] = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "parse_failures": 0,
                      "api_errors": 0, "latency_s": 0.0}

    def describe(self) -> str:
        return f"llm:{self.model}"

    def reset(self, seed: int = 0) -> None:
        self.history = []

    def _call(self, messages: list[dict]) -> dict:
        out = self.client.chat(self.model, messages, temperature=self.temperature,
                               max_tokens=self.max_tokens, extra=self.extra_body)
        u = out["usage"]
        self.usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        self.usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
        self.usage["calls"] += 1
        self.usage["latency_s"] = round(self.usage["latency_s"] + out["latency_s"], 3)
        return out

    @staticmethod
    def _parse(out: dict) -> dict:
        """JSON from the final answer; fall back to the reasoning channel."""
        try:
            return extract_json(out["content"])
        except ValueError:
            if out.get("reasoning"):
                return _from_reasoning(out["reasoning"])
            raise

    def act(self, obs: dict) -> Decision:
        messages = [{"role": "system", "content": self.system}]
        messages += self.history[-2 * self.history_turns:] if self.history_turns else []
        messages.append({"role": "user", "content": obs["text"]})
        meta = {"model": self.model, "messages": messages}
        try:
            out = self._call(messages)
            reply = out["content"]
            try:
                d = Decision.from_dict(self._parse(out))
                parse_error = None
            except ValueError as e:
                self.usage["parse_failures"] += 1
                hint = " Your reply was cut off (token limit): think less and answer sooner." \
                    if out.get("finish_reason") == "length" else ""
                cut = out.get("finish_reason") == "length"
                prior = reply[-4000:] or (f"(my reasoning so far, cut off) ...{out.get('reasoning', '')[-3000:]}"
                                          if cut and out.get("reasoning") else "(empty)")
                ask = ("You ran out of tokens while thinking. Stop deliberating: output ONLY the final JSON object now, "
                       "based on the plan above." if cut else
                       f"Your reply could not be parsed ({e}).{hint} Reply ONLY with the JSON object described in the instructions.")
                retry = messages + [{"role": "assistant", "content": prior}, {"role": "user", "content": ask}]
                out = self._call(retry)
                reply = out["content"]
                try:
                    d = Decision.from_dict(self._parse(out))
                    parse_error = "recovered_after_retry"
                except ValueError as e2:
                    d = Decision(thinking="(unparseable reply)")
                    parse_error = f"unparseable: {e2}"
        except LLMError as e:
            self.usage["api_errors"] += 1
            d, reply, out = Decision(thinking="(api error)"), "", {"usage": {}, "latency_s": 0}
            parse_error = f"api_error: {e}"
        d.parse_error = parse_error
        meta.update({"response": reply, "reasoning": out.get("reasoning", ""), "usage": out.get("usage"),
                     "latency_s": out.get("latency_s"), "finish_reason": out.get("finish_reason")})
        d.meta = meta
        if self.history_turns:
            self.history += [{"role": "user", "content": obs["text"]}, {"role": "assistant", "content": reply}]
        return d
