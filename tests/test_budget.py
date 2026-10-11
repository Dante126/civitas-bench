"""civitas-v3's token budget per turn (fleet/budget.py, `[fleet] turn_budget`): it counts the prompt as the request's
characters / 4 (the same for every model) and the completion as the provider reports it; a model is told its budget
(system prompt, briefing, every tool result) and its turn ends before a reply that wouldn't fit (reckoned at
max(reserve, the turn's largest completion), so a verbose model doesn't write past it), a model that heeds the
wrap-up note always ends with end_turn and its report (the mayor's and planning's open-requests reminder waits),
budget 0 (civitas-v2) plays byte for byte as before, and over a real HTTP chat server the completion comes from the
provider's usage. No model is called."""
import contextlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from civitas.agents.llm_client import ProviderConfig
from civitas.core.env import CivitasEnv, GameConfig
from civitas.fleet import budget as TB
from civitas.fleet.agent import ToolAgent
from civitas.fleet.briefing import briefing
from civitas.fleet.config import FleetAgentConfig, fleet_agent_config
from civitas.fleet.fmt import TOOL_MESSAGE_CHARS
from civitas.fleet.tools import ToolBox

COMPLETION = 300
LAST = "room for about one more reply"
BUDGET = 100_000                 # civitas-v3's (configs/fleet_*_v3.toml)


def _usage(messages, tools, ratio=4, completion=COMPLETION) -> dict:
    """A provider's usage, deterministic: request characters / `ratio` (its tokenizer), `completion` tokens a reply."""
    return {"prompt_tokens": (len(json.dumps(messages)) + len(json.dumps(tools))) // ratio,
            "completion_tokens": completion}


def _call(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class UsageClient:
    """A ChatClient stand-in: `reply(messages, n)` gives the tool calls; usage as `_usage` counts it (`completion`:
    tokens per reply, or a function of the reply's number)."""

    def __init__(self, reply, ratio=4, completion=COMPLETION):
        self.reply, self.requests, self.tools = reply, [], []
        self.ratio, self.completion = ratio, completion

    def chat(self, model, messages, *, temperature=0.4, max_tokens=1500, extra=None):
        tools = self.tools = (extra or {}).get("tools") or []
        self.requests.append([dict(m) for m in messages])
        calls = self.reply(messages, len(self.requests))
        c = self.completion(len(self.requests)) if callable(self.completion) else self.completion
        return {"content": "", "reasoning": "", "tool_calls": calls,
                "finish_reason": "tool_calls" if calls else "stop", "usage": _usage(messages, tools, self.ratio, c),
                "latency_s": 0.0}


def _agent(reply, role="police", env=None, ratio=4, completion=COMPLETION, **cfg):
    env = env or CivitasEnv(GameConfig(seed=0, months=12))
    a = ToolAgent(role, cfg=FleetAgentConfig(model="fake", **cfg),
                  provider=ProviderConfig(base_url="http://127.0.0.1:9", api_key_env=""))
    a.client = UsageClient(reply, ratio, completion)
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


def _first_units(role="police") -> int:
    """The units of the turn's first request (its characters / 4), as the budget counts it."""
    a, env = _agent(lambda m, n: [], role=role, turn_budget=10 ** 7)
    return a.act(env.observe(role)).meta["call_usage"][0]["budget"]["prompt"]


def _budget_for(calls: float, role="police", reserve: int = 4000) -> int:
    """A budget of `calls` model calls of this turn's first size, plus the reserve the cutoff keeps."""
    return int(reserve + calls * (_first_units(role) + COMPLETION))


def _units(d) -> list[dict]:
    return [c["budget"] for c in d.meta["call_usage"]]


# ------------------------------------------------------------------------------------------------- the pieces
def test_turn_budget_counts_units_and_notes():
    b = TB.TurnBudget(10_000, 1_000, tools=[{"x": "y" * 96}])
    assert b.tool_chars == len('[{"x":"' + "y" * 96 + '"}]') and b.fits(10 ** 9)           # the first call always
    # the prompt is the request's characters / 4 whatever the provider says; the completion is the provider's
    assert b.record({"prompt_tokens": 9_999, "completion_tokens": 500}, 8_000, 40) == 2_000 + 500
    assert b.left == 7_500 and b.largest == 500 and b.next_cost(8_400) == 2_100 + 1_000      # reserve floor
    assert b.fits(8_400) and b.fits(25_996) and not b.fits(26_004)   # 2,500 + 6,499 + 1,000 <= 10,000 < ... 6,501
    assert b.suffix(8_400) == "\nbudget left: ~7,500 tokens (~2 more replies at the current size)"
    assert b.suffix(14_000).endswith(f"(~1 more reply at the current size): {LAST}: call end_turn with your report now")
    assert b.wrapped and b.low(0)                                     # the note went out: no end_turn reminder now
    assert "(~0 more replies at the current size): not enough for another reply" in b.suffix(30_000)
    b.record(None, 3_000, 300)                                        # no usage: the reply's characters / 4 too
    assert b.used == 2_500 + 750 + 75 and b.calls == 2 and b.log == [{"prompt": 2_000, "completion": 500},
                                                                     {"prompt": 750, "completion": 75}]
    b.record({"completion_tokens": 3_000}, 400, 10)                   # a long reply raises what the next is reckoned at
    assert b.largest == 3_000 and b.next_cost(400) == 100 + 3_000 and b.over == 0
    assert TB.budget_rule(0) == "" == TB.briefing_line(0)
    assert "90,000 tokens this turn" in TB.briefing_line(90_000) and "90,000 tokens per turn" in TB.budget_rule(90_000)
    assert "characters / 4" in TB.budget_rule(90_000)


def test_completion_is_hardened_and_falls_back_to_characters():
    c = TB.completion
    assert c({"prompt_tokens": 100, "completion_tokens": 40}) == 40
    assert c({"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 900}) == 800        # hidden output
    assert c({"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 120}) == 40         # cached / odd totals
    assert c({"completion_tokens": 40, "completion_tokens_details": {"reasoning_tokens": 300}}) == 300
    assert c({"completion_tokens": 40, "total_tokens": 900}) == 40     # no prompt count: total can't be split
    assert c({}, 400) == c(None, 400) == c({"prompt_tokens": 0, "completion_tokens": 0}, 400) == 100
    assert c({"completion_tokens": 7, "budget": {"prompt": 1, "completion": 55}}, 400) == 55     # a logged call
    assert TB.call_record({"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 9,
                           "completion_tokens_details": {"reasoning_tokens": 1}}) == \
        {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 9, "reasoning_tokens": 1}
    assert TB.call_record(None) == {"prompt_tokens": 0, "completion_tokens": 0}


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
    assert d.meta["turn_budget"] == limit and d.meta["budget_used"] <= limit and d.meta["budget_over"] == 0
    units = _units(d)
    assert d.meta["budget_used"] == sum(u["prompt"] + u["completion"] for u in units)
    assert [u["completion"] for u in units] == [COMPLETION] * len(reqs)                # the provider's completion
    assert [c["prompt_tokens"] for c in d.meta["call_usage"]] == \
        [_usage(r, a.client.tools)["prompt_tokens"] for r in reqs]                     # its usage logged as it was
    assert d.meta["usage"]["prompt_tokens"] == sum(c["prompt_tokens"] for c in d.meta["call_usage"])
    assert 2 <= len(reqs) < 20 and d.meta["steps"] == len(reqs)       # cut well before the step cap
    assert f"Token budget: {limit:,} tokens per turn" in reqs[0][0]["content"]          # the rule
    assert f"Token budget: {limit:,} tokens this turn" in reqs[0][1]["content"]         # the briefing line
    tools = [m["content"] for m in d.meta["messages"] if m["role"] == "tool"]
    lefts = [int(re.search(r"budget left: ~([\d,]+) tokens \(~\d+ more repl", t).group(1).replace(",", ""))
             for t in tools]
    assert len(lefts) == len(tools) and lefts == sorted(lefts, reverse=True) and lefts[0] < limit
    assert all(t.split("\nbudget left:")[0].startswith('{"ok"') for t in tools)    # the JSON, then the line
    assert LAST in tools[-2] or LAST in tools[-1]                     # told before the cut
    used, after = 0, []
    for u in units:
        used += u["prompt"] + u["completion"]
        after.append(max(0, limit - used))
    assert [x["tokens_left"] for x in d.meta["tool_trace"]] == after
    assert d.thinking == TB.CUT_REPORT and env.fleet.reports["police"] == TB.CUT_REPORT     # end_turn's record
    assert env.fleet.log[-1]["kind"] == "report" and env.fleet.log[-1]["text"] == TB.CUT_REPORT


def test_the_units_are_the_same_for_every_tokenizer():
    """Two providers whose tokenizers count the same request 2x apart: the same budget, cut, messages and units."""
    limit = _budget_for(4.5)
    runs = []
    for ratio in (2, 5):
        a, env = _agent(_looping, ratio=ratio, turn_budget=limit)
        d = a.act(env.observe("police"))
        runs.append((d.meta["messages"], d.meta["budget_used"], d.meta["finish_reason"], _units(d)))
        assert d.meta["usage"]["prompt_tokens"] == sum(_usage(r, a.client.tools, ratio)["prompt_tokens"]
                                                       for r in a.client.requests)    # its cost, logged apart
    assert runs[0] == runs[1]


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


def test_a_verbose_model_stays_within_the_budget():
    """The next reply is reckoned at the turn's largest completion: a model writing 12k tokens a reply (3x the
    reserve, under max_tokens 16k) is cut a reply earlier instead of writing past the budget; with a fixed 4k reserve
    it would overshoot by up to 8k."""
    base = _first_units()
    for completion in (12_000, lambda n: 12_000 if n == 1 else 9_000 + 500 * (n % 3)):
        for k in range(3, 10):
            limit = 4000 + k * (base + 9_000)
            a, env = _agent(_looping, completion=completion, turn_budget=limit, max_tokens=16_000,
                            turn_tokens=10 ** 7)
            d = a.act(env.observe("police"))
            assert d.meta["finish_reason"] == "turn_budget" and d.meta["budget_over"] == 0, (k, d.meta["budget_used"])
            assert d.meta["budget_used"] <= limit and len(a.client.requests) >= 2
    # a model whose replies grow overshoots by at most the growth over the largest so far (max_tokens bounds it)
    a, env = _agent(_looping, completion=lambda n: 2_000 * n, turn_budget=4000 + 5 * (base + 4_000),
                    turn_tokens=10 ** 7)
    d = a.act(env.observe("police"))
    assert d.meta["budget_over"] == max(0, d.meta["budget_used"] - d.meta["turn_budget"]) <= 2_000


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
    """Off (0, the default): no rule, no briefing line, no suffix, no budget keys; the same messages and results.
    Each call's provider usage is logged (call_usage) without budget units."""
    runs = []
    for cfg in ({}, {"turn_budget": 0}):
        a, env = _agent(lambda m, n: _looping(m, n) if n < 4 else [_call("end_turn", {"report": "ok"}, "e")],
                        **cfg)
        d = a.act(env.observe("police"))
        runs.append((a.client.requests, d.meta, d.executed, d.thinking))
        assert "Token budget" not in json.dumps(a.client.requests) and "budget left" not in json.dumps(d.meta)
        assert not {"turn_budget", "budget_used", "budget_cut", "budget_over"} & set(d.meta)
        assert all("tokens_left" not in x for x in d.meta["tool_trace"])
        assert len(d.meta["call_usage"]) == len(a.client.requests) == 4
        assert all(set(c) == {"prompt_tokens", "completion_tokens"} for c in d.meta["call_usage"])
        tool = [m["content"] for m in d.meta["messages"] if m["role"] == "tool"]
        assert all(json.loads(t) for t in tool if len(t) < TOOL_MESSAGE_CHARS)          # plain JSON, as in v2
    assert runs[0] == runs[1]


# ------------------------------------------------------------------------------------------------- reports
def _open_request(role: str):
    """A city at its second month with a department's money ask to `role` still open (end_turn's reminder)."""
    env = CivitasEnv(GameConfig(seed=0, months=12, act_every=3))
    briefing(env, role)
    env.council.post(0, "police", "Requesting $2,500 for a station", role)
    env.end_month()
    return env


def _heeds_or_reminded(messages, n):
    """The reviewer's model: inspects until the wrap-up note (or end_turn's reminder), then end_turn with its report."""
    last = messages[-1]
    if last["role"] == "tool" and ("one more reply" in last["content"] or "open_requests" in last["content"]):
        return [_call("end_turn", {"report": "MY REAL REPORT"}, f"e{n}")]
    return _looping(messages, n)


def test_heeding_the_note_keeps_the_report_with_open_requests():
    """The mayor and planning, with a request to them still open, heed the wrap-up note: end_turn's once-a-turn
    reminder would take the reply that no longer fits, so it waits (ToolBox.budget_low) and the turn ends with the
    model's own report, at every budget and reply size (the review's scenario: 0 of 258 reports lost)."""
    lost = []
    for role in ("mayor", "planning"):
        for completion in (300, 1500, 3000):
            for limit in range(30_000, 60_000, 700):
                a, env = _agent(_heeds_or_reminded, role=role, env=_open_request(role), completion=completion,
                                turn_budget=limit)
                d = a.act(env.observe(role))
                refused = [x for x in d.meta["tool_trace"] if (x["result"] or {}).get("refused") == "open_requests"]
                if d.meta["finish_reason"] != "end_turn" or d.thinking != "MY REAL REPORT" or refused:
                    lost.append((role, completion, limit, d.meta["finish_reason"], d.thinking[:20]))
    assert lost == []


def test_a_cut_turn_keeps_a_refused_end_turns_report():
    """end_turn refused for open requests while the budget had room; the model goes on until the budget cuts the
    turn: it closes with the report that end_turn carried, not CUT_REPORT."""
    def model(messages, n):
        return [_call("end_turn", {"report": "the report I wrote"}, f"e{n}")] if n == 1 else _looping(messages, n)
    a, env = _agent(model, role="mayor", env=_open_request("mayor"), turn_budget=_budget_for(4.5, role="mayor"))
    d = a.act(env.observe("mayor"))
    assert d.meta["tool_trace"][0]["result"]["refused"] == "open_requests"
    assert d.meta["finish_reason"] == "turn_budget" and d.thinking == "the report I wrote"
    assert env.fleet.reports["mayor"] == "the report I wrote" and env.fleet.log[-1]["text"] == "the report I wrote"


def test_the_reminder_waits_only_when_the_budget_is_low():
    for low, refused in ((None, True), (lambda: False, True), (lambda: True, False)):
        env = _open_request("mayor")
        box = ToolBox(env, "mayor", max_steps=20)
        box.budget_low = low
        res = box.call("end_turn", {"report": "r"})
        assert (res.get("refused") == "open_requests") == refused and box.ended != refused
        assert box.refused_report == ("r" if refused else "")


# ------------------------------------------------------------------------------------------------- over HTTP
class FakeUsageLLM(BaseHTTPRequestHandler):
    """An OpenAI-compatible chat server: one inspect per reply until a result says the next reply is about the
    last, then end_turn; usage = request characters / 3 (another tokenizer) and a total with hidden reasoning."""
    requests: list[dict] = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeUsageLLM.requests.append(body)
        n = sum(m["role"] == "assistant" for m in body["messages"]) + 1
        calls = _heeding(body["messages"], n)
        msg = {"role": "assistant", "content": "", "tool_calls": calls}
        u = _usage(body["messages"], body.get("tools") or [], ratio=3)
        u["total_tokens"] = u["prompt_tokens"] + u["completion_tokens"] + 200      # reasoning not in completion
        out = json.dumps({"choices": [{"message": msg, "finish_reason": "tool_calls"}], "usage": u}).encode()
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


def test_over_http_the_provider_completion_counts(tmp_path):
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
        cu = t["call_usage"]
        assert len(cu) == len(t["finish_reasons"]) and all("total_tokens" in c for c in cu)
        assert all(c["budget"]["completion"] == COMPLETION + 200 for c in cu)         # total - prompt: the hidden part
        assert t["budget_used"] == sum(c["budget"]["prompt"] + c["budget"]["completion"] for c in cu)
        assert t["usage"]["prompt_tokens"] == sum(c["prompt_tokens"] for c in cu)    # the provider's, logged apart
        assert t["turn_budget"] == limit and t["budget_over"] == max(0, t["budget_used"] - limit)
        assert t["budget_cut"] == (t["finish_reason"] == "turn_budget")
        assert t["budget_used"] <= limit or t["finish_reason"] != "turn_budget"
    assert any(t["finish_reason"] == "end_turn" for t in traj)
    assert all(f"{limit:,} tokens this turn" in b["messages"][1]["content"] for b in FakeUsageLLM.requests)
    from civitas.fleet.report import rows_from_trajectories, _budget_stats
    st = _budget_stats(rows_from_trajectories(tmp_path / s["run_id"] / "trajectories.jsonl"))
    assert st["budget_turns"] == 8 and st["budget_cuts"] == sum(t["budget_cut"] for t in traj)
    md = (tmp_path / s["run_id"] / "fleet_report.md").read_text()
    assert "| step-limit ends | budget cuts |" in md and "## Token budget" in md


def test_the_v3_configs_are_v2_plus_the_budget():
    """fleet_smoke_late_v3 / fleet_open_late_v3 = the v2 fleet configs + turn_budget and one fleet playbook line."""
    from civitas.config import load_config
    root = Path(__file__).resolve().parents[1]
    for v2, v3 in (("fleet_smoke_late", "fleet_smoke_late_v3"), ("fleet_open_late", "fleet_open_late_v3")):
        a, b = load_config(root / f"configs/{v2}.toml"), load_config(root / f"configs/{v3}.toml")
        assert b["fleet"].pop("turn_budget") == BUDGET and b.pop("name") == v3 and a.pop("name") == v2
        extra = b["fleet"]["playbook"].replace(a["fleet"]["playbook"], "")
        assert extra.startswith(f"- Token budget: {BUDGET:,} tokens a turn") and extra.count("\n") == 1
        b["fleet"]["playbook"] = a["fleet"]["playbook"]
        assert a == b
