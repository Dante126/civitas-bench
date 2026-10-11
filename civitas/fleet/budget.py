"""The turn's token budget (civitas-v3, `[fleet] turn_budget`): prompt + completion tokens summed over a turn's model
calls, as the provider reports usage. Off (0) in civitas-v2, which then plays exactly as before.

Why: on civitas-v2 every model had the same caps (20 tool calls, 80k completion tokens a turn, 16k a call), but
token use per department turn differed 3x on the open track (trinity-large-thinking 92k with one tool call per
reply and 8 look-ups, kimi-k3 34k) at the same ~2.3 actions. Every reply re-sends the whole conversation, so
prompt tokens grow with the number of replies; a budget on the sum makes calls per reply and look-ups count.

How the turn uses it (fleet/agent.py:ToolAgent.act; rl/episode.py:TaskEpisode mirrors it):
- the first model call always runs (the briefing must be read);
- before every further call, the call's prompt is estimated from the request's characters (system prompt, tool
  schemas and every message so far) at the tokens-per-character of the last reported call, plus `reserve`
  completion tokens; when `used` plus that would pass the budget, the call isn't made and the turn ends
  ("turn_budget": completed actions stay, the report reads CUT_REPORT, end_turn's record without its refusals);
- every tool result the model reads ends with "budget left: ~N tokens", and with a wrap-up note once the next reply
  is about the last that fits (`suffix`), so a model that heeds it always ends with end_turn;
- a provider that reports no usage gets the characters / CHARS_PER_TOKEN estimate instead.
A reply may still overshoot by what it writes beyond `reserve` (max_tokens bounds it).
"""
from __future__ import annotations

import json

from .fmt import TOOL_MESSAGE_CHARS

CHARS_PER_TOKEN = 3.0        # before the first reported call, and for a provider that reports no usage
MESSAGE_CHARS = 20           # per message: role and framing the characters don't show
GROWTH_CHARS = TOOL_MESSAGE_CHARS + 600   # what the next reply adds to the request after it: a tool result (cut at
                                          # TOOL_MESSAGE_CHARS) and the reply's own call, for the wrap-up note
CUT_REPORT = "(no report: the turn's token budget ran out before end_turn; completed actions stay)"


def budget_rule(limit: int) -> str:
    """The system prompt's rule (fleet/briefing.py:system_prompt), "" when off."""
    if not limit:
        return ""
    return (f"\n- Token budget: {limit:,} tokens per turn, prompt plus completion summed over all your replies, as the "
            "model provider counts them. Every reply re-sends the whole conversation (this prompt, the briefing and "
            "every tool result so far), so a turn of many short replies costs the most: put several tool calls in "
            "one reply and look up only what your briefing lacks. Each tool result ends with the budget left; when "
            "the next reply would not fit, the turn ends there (completed actions stay), so call end_turn before.")


def briefing_line(limit: int) -> str:
    """The briefing header's line (fleet/briefing.py:briefing), "" when off."""
    if not limit:
        return ""
    return (f"Token budget: {limit:,} tokens this turn (prompt + completion; every reply re-sends the conversation, "
            "so look-ups and single tool calls per reply cost the most)")


def message_chars(m: dict) -> int:
    calls = m.get("tool_calls")
    return len(m.get("content") or "") + (len(json.dumps(calls, separators=(",", ":"))) if calls else 0) + MESSAGE_CHARS


class TurnBudget:
    """One turn's budget: `limit` tokens, `reserve` kept for the next reply's completion."""

    def __init__(self, limit: int, reserve: int, tools: list | None = None):
        self.limit, self.reserve = int(limit), int(reserve)
        self.tool_chars = len(json.dumps(tools or [], separators=(",", ":")))
        self.used = 0                 # prompt + completion tokens of this turn's calls so far
        self.calls = 0
        self.ratio = 1.0 / CHARS_PER_TOKEN   # prompt tokens per request character, from the last reported call
        self.cut = False

    def chars(self, messages: list) -> int:
        """The request's characters: the tool schemas and every message."""
        return self.tool_chars + sum(message_chars(m) for m in messages)

    def record(self, usage: dict | None, request_chars: int, reply_chars: int) -> int:
        """Count one call (its `usage` as the provider reported it; estimated from characters when it reports none);
        returns its tokens."""
        u = usage or {}
        prompt, completion = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
        if prompt > 0 and request_chars > 0:
            self.ratio = prompt / request_chars
        if prompt <= 0 and completion <= 0:
            prompt, completion = round(request_chars / CHARS_PER_TOKEN), round(reply_chars / CHARS_PER_TOKEN)
        self.calls += 1
        self.used += prompt + completion
        return prompt + completion

    @property
    def left(self) -> int:
        return self.limit - self.used

    def next_cost(self, request_chars: int) -> int:
        """The next call's tokens as the cutoff counts them: its estimated prompt plus the reserve."""
        return round(self.ratio * request_chars) + self.reserve

    def fits(self, request_chars: int) -> bool:
        """Whether the next call (on a request of these characters) fits; the first call always does."""
        return self.calls == 0 or self.used + self.next_cost(request_chars) <= self.limit

    def suffix(self, request_chars: int) -> str:
        """The line a tool result ends with: the budget left (to the nearest 100), and a wrap-up note when the next
        reply, on a request of `request_chars`, is about the last that fits: the one after it, on a request
        GROWTH_CHARS longer, wouldn't (or when none does: the model never reads that one, the turn ends first)."""
        left = max(0, self.left)
        nxt = self.next_cost(request_chars)
        text = f"\nbudget left: ~{round(left, -2):,} tokens"
        if left < nxt:
            return text + ": not enough for another reply, so the turn ends now; completed actions stay"
        if left < nxt + self.next_cost(request_chars + GROWTH_CHARS):
            return text + ": room for about one more reply: call end_turn with your report now"
        return text
