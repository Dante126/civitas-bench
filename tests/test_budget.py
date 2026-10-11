"""civitas-v3's token budget per turn (fleet/budget.py, `[fleet] turn_budget`): a model whose calls report usage is
told its budget (system prompt, briefing, every tool result) and its turn ends before a reply that wouldn't fit, a
model that heeds the wrap-up note always ends with end_turn, budget 0 (civitas-v2) plays byte for byte as before,
and over a real HTTP chat server the counts come from the provider's usage. No model is called."""
import contextlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from civitas.agents.llm_client import ProviderConfig
from civitas.core.env import CivitasEnv, GameConfig
from civitas.fleet import budget as TB
from civitas.fleet.agent import ToolAgent
from civitas.fleet.config import FleetAgentConfig, fleet_agent_config
from civitas.fleet.fmt import TOOL_MESSAGE_CHARS

COMPLETION = 300
LAST = "room for about one more reply"


def _usage(messages, tools) -> dict:
    """A provider's usage, deterministic: request characters / 4, and COMPLETION tokens a reply."""
    return {"prompt_tokens": (len(json.dumps(messages)) + len(json.dumps(tools))) // 4,
            "completion_tokens": COMPLETION}


def _call(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class UsageClient:
    """A ChatClient stand-in: `reply(messages, n)` gives the tool calls; usage as `_usage` counts it."""

    def __init__(self, reply):
        self.reply, self.requests, self.tools = reply, [], []

    def chat(self, model, messages, *, temperature=0.4, max_tokens=1500, extra=None):
        tools = self.tools = (extra or {}).get("tools") or []
        self.requests.append([dict(m) for m in messages])
        calls = self.reply(messages, len(self.requests))
        return {"content": "", "reasoning": "", "tool_calls": calls,
                "finish_reason": "tool_calls" if calls else "stop", "usage": _usage(messages, tools),
                "latency_s": 0.0}


def _agent(reply, role="police", **cfg):
    env = CivitasEnv(GameConfig(seed=0, months=12))
    a = ToolAgent(role, cfg=FleetAgentConfig(model="fake", **cfg),
                  provider=ProviderConfig(base_url="http://127.0.0.1:9", api_key_env=""))
    a.client = UsageClient(reply)
    a.bind(env)
    return a, env


def _looping(messages, n):
    """One look-up per reply, a new one each time (no cached repeat), never end_turn: the wasteful pattern."""
    return [_call("inspect", {"x": n % 30, "y": n // 30}, f"c{n}")]


def _heeding(messages, n):
    """The same, until a tool result says the next reply is about the last: then end_turn."""
    last = messages[-1]
    if last["role"] == "tool" and LAST in last["content"]:
        return [_call("end_turn", {"report": "out of budget, stopping"}, f"e{n}")]
    return _looping(messages, n)


def _budget_for(calls: float, role="police", reserve: int = 4000) -> int:
    """A budget of `calls` model calls of this turn's first size, plus the reserve the cutoff keeps."""
    a, env = _agent(lambda m, n: [], role=role)
    a.act(env.observe(role))
    return int(reserve + calls * (_usage(a.client.requests[0], a.client.tools)["prompt_tokens"] + COMPLETION))


# ------------------------------------------------------------------------------------------------- the pieces
def test_turn_budget_counts_estimates_and_notes():
    b = TB.TurnBudget(10_000, 1_000, tools=[{"x": "y" * 96}])
    assert b.tool_chars == len('[{"x":"' + "y" * 96 + '"}]') and b.fits(10 ** 9)           # the first call always
    assert b.record({"prompt_tokens": 2_000, "completion_tokens": 500}, 8_000, 40) == 2_500
    assert b.ratio == 0.25 and b.left == 7_500 and b.next_cost(8_400) == 2_100 + 1_000
    assert b.fits(8_400) and b.fits(26_000) and not b.fits(26_100)   # 2,500 + 6,525 + 1,000 > 10,000
    assert b.suffix(8_400) == "\nbudget left: ~7,500 tokens"          # 7,500 >= 2 x 3,100
    assert b.suffix(14_000).endswith(f"~7,500 tokens: {LAST}: call end_turn with your report now")   # < 2 x 4,500
    assert "the turn ends now" in b.suffix(30_000)
    b.record(None, 3_000, 300)                                        # no usage reported: characters / 3
    assert b.used == 2_500 + 1_000 + 100 and b.calls == 2
    assert TB.budget_rule(0) == "" == TB.briefing_line(0)
    assert "90,000 tokens this turn" in TB.briefing_line(90_000) and "90,000 tokens per turn" in TB.budget_rule(90_000)


def test_config_takes_and_checks_the_budget():
    cfg = {"fleet": {"turn_budget": 90000}, "agents": {"police": {"kind": "fleet", "budget_reserve": 2500}}}
    fc = fleet_agent_config(cfg, "police")
    assert (fc.turn_budget, fc.budget_reserve) == (90000, 2500)
    assert fleet_agent_config({}, "police").turn_budget == 0                          # off: civitas-v2
    for bad in (-1, 1.5, True, "90k"):
        with pytest.raises(ValueError, match="turn_budget"):
            fleet_agent_config({"fleet": {"turn_budget": bad}}, "police")


# ------------------------------------------------------------------------------------------------- the turn
def test_the_turn_ends_at_the_budget_and_the_model_is_told():
    limit = _budget_for(4.5)
    a, env = _agent(_looping, turn_budget=limit)
    d = a.act(env.observe("police"))
    reqs = a.client.requests
    assert d.meta["finish_reason"] == "turn_budget" and d.meta["budget_cut"] and not d.meta["ended"]
    assert d.meta["turn_budget"] == limit and d.meta["budget_used"] <= limit
    assert d.meta["budget_used"] == d.meta["usage"]["prompt_tokens"] + d.meta["usage"]["completion_tokens"]
    assert 2 <= len(reqs) < 20 and d.meta["steps"] == len(reqs)       # cut well before the step cap
    assert f"Token budget: {limit:,} tokens per turn" in reqs[0][0]["content"]          # the rule
    assert f"Token budget: {limit:,} tokens this turn" in reqs[0][1]["content"]         # the briefing line
    tools = [m["content"] for m in d.meta["messages"] if m["role"] == "tool"]
    lefts = [int(re.search(r"budget left: ~([\d,]+) tokens", t).group(1).replace(",", "")) for t in tools]
    assert len(lefts) == len(tools) and lefts == sorted(lefts, reverse=True) and lefts[0] < limit
    assert all(t.split("\nbudget left:")[0].startswith('{"ok"') for t in tools)    # the JSON, then the line
    assert LAST in tools[-2] or LAST in tools[-1]                     # told before the cut
    assert [x["tokens_left"] for x in d.meta["tool_trace"]] == [max(0, limit - u) for u in _used_after(a)]
    assert d.thinking == TB.CUT_REPORT and env.fleet.reports["police"] == TB.CUT_REPORT     # end_turn's record
    assert env.fleet.log[-1]["kind"] == "report" and env.fleet.log[-1]["text"] == TB.CUT_REPORT


def _used_after(a) -> list[int]:
    """Tokens used after each call of the turn, as the client reported them."""
    out, used = [], 0
    for r in a.client.requests:
        u = _usage(r, a.client.tools)
        used += u["prompt_tokens"] + u["completion_tokens"]
        out.append(used)
    return out


def test_a_model_that_heeds_the_note_ends_with_end_turn():
    """Whatever the budget, once it admits a second reply: the note comes while a reply is left (with the default
    reserve, and with a reserve no larger than what each reply writes)."""
    for reserve in (4000, COMPLETION):
        for tenths in range(22, 90, 3):
            limit = _budget_for(tenths / 10, reserve=reserve)
            a, env = _agent(_heeding, turn_budget=limit, budget_reserve=reserve)
            d = a.act(env.observe("police"))
            assert d.meta["finish_reason"] == "end_turn" and d.meta["ended"] and not d.meta["budget_cut"], \
                (reserve, tenths)
            assert d.meta["budget_used"] <= limit and d.thinking == "out of budget, stopping"
            assert d.meta["tool_trace"][-1]["tool"] == "end_turn" and len(d.meta["tool_trace"]) >= 2


def test_a_budget_under_two_replies_ends_after_the_first():
    """The first call always runs (the briefing must be read); its result already says the turn ends."""
    limit = _budget_for(1.2)
    a, env = _agent(_heeding, turn_budget=limit)
    d = a.act(env.observe("police"))
    assert d.meta["finish_reason"] == "turn_budget" and len(a.client.requests) == 1 and d.meta["steps"] == 1
    assert "not enough for another reply" in d.meta["messages"][-1]["content"]


def test_the_step_cap_stays_a_hard_cap():
    a, env = _agent(_looping, turn_budget=10 ** 7, max_steps=3)
    d = a.act(env.observe("police"))
    assert d.meta["finish_reason"] == "max_steps" and d.meta["steps"] == 3 and not d.meta["budget_cut"]
    assert d.thinking == "(no report)"


def test_budget_zero_is_civitas_v2_byte_for_byte():
    """Off (0, the default): no rule, no briefing line, no suffix, no budget keys; the same messages and results."""
    runs = []
    for cfg in ({}, {"turn_budget": 0}):
        a, env = _agent(lambda m, n: _looping(m, n) if n < 4 else [_call("end_turn", {"report": "ok"}, "e")],
                        **cfg)
        d = a.act(env.observe("police"))
        runs.append((a.client.requests, d.meta, d.executed, d.thinking))
        assert "Token budget" not in json.dumps(a.client.requests) and "budget left" not in json.dumps(d.meta)
        assert not {"turn_budget", "budget_used", "budget_cut"} & set(d.meta)
        assert all("tokens_left" not in x for x in d.meta["tool_trace"])
        tool = [m["content"] for m in d.meta["messages"] if m["role"] == "tool"]
        assert all(json.loads(t) for t in tool if len(t) < TOOL_MESSAGE_CHARS)          # plain JSON, as in v2
    assert runs[0] == runs[1]


# ------------------------------------------------------------------------------------------------- over HTTP
class FakeUsageLLM(BaseHTTPRequestHandler):
    """An OpenAI-compatible chat server: one inspect per reply until a result says the next reply is about the
    last, then end_turn; usage = request characters / 4."""
    requests: list[dict] = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeUsageLLM.requests.append(body)
        n = sum(m["role"] == "assistant" for m in body["messages"]) + 1
        calls = _heeding(body["messages"], n)
        msg = {"role": "assistant", "content": "", "tool_calls": calls}
        out = json.dumps({"choices": [{"message": msg, "finish_reason": "tool_calls"}],
                          "usage": _usage(body["messages"], body.get("tools") or [])}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@contextlib.contextmanager
def _server():
    FakeUsageLLM.requests = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeUsageLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv
    finally:
        srv.shutdown()


def test_over_http_the_provider_usage_counts(tmp_path):
    from civitas.bench import runner
    from civitas.config import load_config
    from civitas.core.roles import TURN_ORDER
    limit = _budget_for(3.5, role="police")
    with _server() as srv:
        cfg = load_config(None)
        cfg["name"] = "budget_fake"
        cfg["game"]["months"] = 1
        cfg["providers"]["fake"] = {"base_url": f"http://127.0.0.1:{srv.server_port}/v1", "api_key_env": "",
                                    "stream": False, "max_retries": 0}
        cfg["fleet"] = {"provider": "fake", "model": "fake", "turn_budget": limit}
        cfg["agents"] = {r: {"kind": "fleet"} for r in TURN_ORDER}
        s = runner.run_episode(cfg, 0, tmp_path, verbose=False)
    traj = [json.loads(x) for x in (tmp_path / s["run_id"] / "trajectories.jsonl").read_text().splitlines()]
    assert len(traj) == 8 and FakeUsageLLM.requests
    for t in traj:
        u = t["usage"]
        assert t["turn_budget"] == limit and t["budget_used"] == u["prompt_tokens"] + u["completion_tokens"]
        assert t["budget_cut"] == (t["finish_reason"] == "turn_budget")
        assert t["budget_used"] <= limit or t["finish_reason"] != "turn_budget"
    assert any(t["finish_reason"] == "end_turn" for t in traj)
    assert all(f"{limit:,} tokens this turn" in b["messages"][1]["content"] for b in FakeUsageLLM.requests)


def test_the_v3_configs_are_v2_plus_the_budget():
    """fleet_smoke_late_v3 / fleet_open_late_v3 = the v2 fleet configs + turn_budget and one fleet playbook line."""
    from pathlib import Path
    from civitas.config import load_config
    root = Path(__file__).resolve().parents[1]
    for v2, v3 in (("fleet_smoke_late", "fleet_smoke_late_v3"), ("fleet_open_late", "fleet_open_late_v3")):
        a, b = load_config(root / f"configs/{v2}.toml"), load_config(root / f"configs/{v3}.toml")
        assert b["fleet"].pop("turn_budget") == 90000 and b.pop("name") == v3 and a.pop("name") == v2
        extra = b["fleet"]["playbook"].replace(a["fleet"]["playbook"], "")
        assert extra.startswith("- Token budget: 90,000 tokens a turn") and extra.count("\n") == 1
        b["fleet"]["playbook"] = a["fleet"]["playbook"]
        assert a == b
