"""The turn's token budget (civitas-v3, `[fleet] turn_budget`), in tokenizer-neutral units summed over a turn's model
calls. Off (0) in civitas-v2, which then plays exactly as before.

Why: on civitas-v2 every model had the same caps (20 tool calls, 80k completion tokens a turn, 16k a call), but
token use per department turn differed 3x on the open track (trinity-large-thinking 92k with one tool call per
reply and 8 look-ups, kimi-k3 34k) at the same ~2.3 actions. Every reply re-sends the whole conversation, so
prompt tokens grow with the number of replies; a budget on the sum makes calls per reply and look-ups count.

The units (the same conversation costs the same for every model, whatever its tokenizer):
- a call's prompt counts its request's characters / CHARS_PER_TOKEN (4): the tool schemas and every message as sent
  (`chars`), never the provider's prompt tokens;
- its completion counts the provider's completion tokens (what the model chose to write and think), hardened:
  the largest of completion_tokens, completion_tokens_details.reasoning_tokens and total_tokens - prompt_tokens
  (`completion`); the reply's characters / CHARS_PER_TOKEN when the provider reports none.
The provider's own usage is still logged apart (the cost); `budget_used` is these units.

How the turn uses it (fleet/agent.py:ToolAgent.act; rl/episode.py:TaskEpisode mirrors it):
- the first model call always runs (the briefing must be read);
- before every further call, its cost is the request's units plus max(`reserve`, the largest completion this turn)
  (`next_cost`); when `used` plus that would pass the budget, the call isn't made and the turn ends ("turn_budget":
  completed actions stay, the report is the one a refused end_turn carried, else CUT_REPORT; end_turn's record
  without its refusals). The largest completion, not the mean: the cutoff admits a reply only when it fits, and a
  model's next reply is bounded best by the longest it has written this turn (a mean admits the long half of a
  verbose model's replies, which then overshoot); a verbose model is cut a reply earlier instead of writing past the
  budget, and a terse one keeps the `reserve` floor;
- every tool result the model reads ends with "budget left: ~N tokens (~K more replies at the current size)", and
  with a wrap-up note once the next reply is about the last that fits (`suffix`), so a model that heeds it always
  ends with end_turn; end_turn's once-a-turn open-requests reminder (fleet/tools.py:ToolBox.end_turn) is skipped
  then (`low`, ToolBox.budget_low), as it is with fewer than REMIND_MIN_LEFT tool calls left.
A reply may still overshoot by what it writes beyond that (max_tokens bounds it); trajectories log it as
`budget_over`.
"""
from __future__ import annotations

import json

from .fmt import TOOL_MESSAGE_CHARS

CHARS_PER_TOKEN = 4          # the units: characters per token, for the prompt always, for a completion without usage
MESSAGE_CHARS = 20           # per message: role and framing the characters don't show
GROWTH_CHARS = TOOL_MESSAGE_CHARS + 600   # what the next reply adds to the request after it: a tool result (cut at
                                          # TOOL_MESSAGE_CHARS) and the reply's own call, for the wrap-up note
CUT_REPORT = "(no report: the turn's token budget ran out before end_turn; completed actions stay)"


def budget_rule(limit: int) -> str:
    """The system prompt's rule (fleet/briefing.py:system_prompt), "" when off."""
    if not limit:
        return ""
    return (f"\n- Token budget: {limit:,} tokens per turn, summed over all your replies: each reply's prompt (counted "
            f"as its characters / {CHARS_PER_TOKEN}, the same for every model) plus the tokens it writes, reasoning "
            "included. Every reply re-sends the whole conversation (this prompt, the briefing and every tool result "
            "so far), so a turn of many short replies costs the most: put several tool calls in one reply and look "
            "up only what your briefing lacks. Each tool result ends with the budget left; when the next reply would "
            "not fit, the turn ends there (completed actions stay), so call end_turn before.")


def briefing_line(limit: int) -> str:
    """The briefing header's line (fleet/briefing.py:briefing), "" when off."""
    if not limit:
        return ""
    return (f"Token budget: {limit:,} tokens this turn (prompt + completion; every reply re-sends the conversation, "
            "so look-ups and single tool calls per reply cost the most)")


def message_chars(m: dict) -> int:
    calls = m.get("tool_calls")
    return len(m.get("content") or "") + (len(json.dumps(calls, separators=(",", ":"))) if calls else 0) + MESSAGE_CHARS


def units(chars: int) -> int:
    """Characters in budget units (tokens)."""
    return round(chars / CHARS_PER_TOKEN)


def completion(usage: dict | None, reply_chars: int = 0) -> int:
    """A reply's completion in budget units: the provider's completion tokens, or more when its reasoning tokens or
    total_tokens - prompt_tokens show more; the reply's characters / CHARS_PER_TOKEN when it reports none. A logged
    call's own count (`call_usage` entries' "budget": {"completion"}) is taken as it is (a replay: TaskEpisode)."""
    u = usage or {}
    logged = u.get("budget")
    if isinstance(logged, dict) and logged.get("completion") is not None:
        return int(logged["completion"])
    prompt, done = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
    reasoning = int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    total = int(u.get("total_tokens") or 0)
    out = max(done, reasoning, total - prompt if total > 0 and prompt > 0 else 0)
    return out if out > 0 else units(reply_chars)


def call_record(usage: dict | None) -> dict:
    """One model call's provider usage as a trajectory logs it (`call_usage`): prompt / completion tokens, and
    total / reasoning tokens when reported."""
    u = usage or {}
    out = {"prompt_tokens": int(u.get("prompt_tokens") or 0), "completion_tokens": int(u.get("completion_tokens") or 0)}
    if u.get("total_tokens") is not None:
        out["total_tokens"] = int(u["total_tokens"] or 0)
    r = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if r is not None:
        out["reasoning_tokens"] = int(r or 0)
    return out


class TurnBudget:
    """One turn's budget: `limit` units; the next reply's completion is reckoned at max(`reserve`, the largest so far)."""

    def __init__(self, limit: int, reserve: int, tools: list | None = None):
        self.limit, self.reserve = int(limit), int(reserve)
        self.tool_chars = len(json.dumps(tools or [], separators=(",", ":")))
        self.used = 0                 # prompt + completion units of this turn's calls so far
        self.calls = 0
        self.largest = 0              # the largest completion of this turn so far
        self.log: list[dict] = []     # per call: {"prompt", "completion"} units
        self.wrapped = False          # a tool result carried the wrap-up note
        self.cut = False

    def chars(self, messages: list) -> int:
        """The request's characters: the tool schemas and every message."""
        return self.tool_chars + sum(message_chars(m) for m in messages)

    def record(self, usage: dict | None, request_chars: int, reply_chars: int) -> int:
        """Count one call: its request's characters in units, and its completion (`completion`: from the provider's
        `usage`, from the reply's characters when it reports none); returns its units."""
        p, c = units(request_chars), completion(usage, reply_chars)
        self.calls += 1
        self.used += p + c
        self.largest = max(self.largest, c)
        self.log.append({"prompt": p, "completion": c})
        return p + c

    @property
    def left(self) -> int:
        return self.limit - self.used

    @property
    def over(self) -> int:
        """Units used beyond the budget (a reply that wrote more than the cutoff reckoned)."""
        return max(0, self.used - self.limit)

    def next_cost(self, request_chars: int) -> int:
        """The next call's units as the cutoff counts them: its prompt plus max(reserve, the largest completion)."""
        return units(request_chars) + max(self.reserve, self.largest)

    def fits(self, request_chars: int) -> bool:
        """Whether the next call (on a request of these characters) fits; the first call always does."""
        return self.calls == 0 or self.used + self.next_cost(request_chars) <= self.limit

    def low(self, request_chars: int) -> bool:
        """Whether the reply after the one being handled may not fit: the wrap-up note went out this turn, or a
        request GROWTH_CHARS longer than this one (`request_chars`: the conversation so far) wouldn't fit. end_turn
        skips its open-requests reminder then (ToolBox.budget_low): the model may not get a reply to call it again."""
        return self.wrapped or not self.fits(request_chars + GROWTH_CHARS)

    def suffix(self, request_chars: int) -> str:
        """The line a tool result ends with: the budget left (to the nearest 100) and how many replies of the next
        one's cost it holds, and a wrap-up note when the next reply, on a request of `request_chars`, is about the
        last that fits: the one after it, on a request GROWTH_CHARS longer, wouldn't (or when none does: the model
        never reads that one, the turn ends first)."""
        left = max(0, self.left)
        nxt = self.next_cost(request_chars)
        k = left // nxt if nxt > 0 else 0
        text = f"\nbudget left: ~{round(left, -2):,} tokens (~{k} more repl{'y' if k == 1 else 'ies'} at the current size)"
        if left < nxt:
            return text + ": not enough for another reply, so the turn ends now; completed actions stay"
        if left < nxt + self.next_cost(request_chars + GROWTH_CHARS):
            self.wrapped = True
            return text + ": room for about one more reply: call end_turn with your report now"
        return text
