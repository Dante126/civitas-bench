"""What an agent returns each turn."""
from __future__ import annotations

from dataclasses import dataclass, field


def _normalize(a: dict) -> dict:
    """Accept common LLM variations: {"action": ...}, {"name": ...}, nested params/args."""
    a = dict(a)
    if "type" not in a:
        for k in ("action", "name", "action_type", "tool"):
            if isinstance(a.get(k), str):
                a["type"] = a.pop(k)
                break
    for k in ("params", "parameters", "args", "arguments"):
        if isinstance(a.get(k), dict):
            nested = a.pop(k)
            a = {**nested, **a}
    return a


@dataclass
class Decision:
    actions: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)  # [{"to": "all"|role, "text": "..."}]
    notes: str | None = None          # persisted scratchpad shown next turn
    thinking: str = ""                # short rationale (logged + shown in viewer)
    parse_error: str | None = None    # set when an LLM reply could not be parsed
    meta: dict = field(default_factory=dict)  # prompt/response/tokens/latency
    executed: list[dict] = field(default_factory=list)  # results of actions an agent already ran (tool agents)

    @classmethod
    def from_dict(cls, d: dict) -> "Decision":
        msgs = d.get("messages") or []
        if isinstance(d.get("message"), str) and d["message"].strip():
            msgs = [*msgs, {"to": "all", "text": d["message"]}]
        norm = []
        for m in msgs:
            if isinstance(m, str):
                norm.append({"to": "all", "text": m})
            elif isinstance(m, dict) and m.get("text"):
                norm.append({"to": str(m.get("to", "all")), "text": str(m["text"])})
        actions = d.get("actions") or []
        if isinstance(actions, dict):
            actions = [actions]
        return cls(actions=[_normalize(a) for a in actions if isinstance(a, dict)] if isinstance(actions, list) else [],
                   messages=norm, notes=d.get("notes"), thinking=str(d.get("thinking", ""))[:2000])
