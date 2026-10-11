import json
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from civitas.agents.llm import LLMAgent, extract_json
from civitas.agents.llm_client import ProviderConfig
from civitas.agents.remote import LiveHub
from civitas.agents.scripted import SCRIPTED
from civitas.bench.runner import run_episode
from civitas.config import apply_cli_overrides, load_config
from civitas.core.env import CivitasEnv, GameConfig
from civitas.core.roles import TURN_ORDER
from civitas.server import serve


def test_extract_json_variants():
    assert extract_json('{"actions": []}') == {"actions": []}
    assert extract_json('<think>{"x":1}</think> sure: ```json\n{"actions":[{"type":"noop"}]}\n```')["actions"][0]["type"] == "noop"
    txt = 'Plan:\n{"thinking": "a {b}", "actions": [ {"type":"noop"}, ], // trailing\n "notes":"n"}'
    assert extract_json(txt)["notes"] == "n"


class FakeLLM(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLLM.calls.append(body)
        # first call of each turn returns junk to exercise the retry path
        junk = len(FakeLLM.calls) == 1
        content = "I think we should build." if junk else json.dumps({
            "thinking": "zone some homes", "notes": "keep going",
            "actions": [{"type": "zone", "zone": "R", "x1": 2, "y1": 25, "x2": 6, "y2": 25}],
            "messages": [{"to": "mayor", "text": "need money"}]})
        out = json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 100, "completion_tokens": 20}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def test_llm_agent_end_to_end_with_fake_server(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("FAKE_KEY", "k")
    prov = ProviderConfig(base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key_env="FAKE_KEY")
    agent = LLMAgent("planning", model="fake", provider=prov)
    env = CivitasEnv(GameConfig(seed=0, months=6))
    env.world.dept_funds["planning"] = 500
    d = agent.act(env.observe("planning"))
    assert d.parse_error == "recovered_after_retry"
    res = env.apply("planning", d)
    assert res[0]["ok"], res
    assert env.memory["planning"]["notes"] == "keep going"
    assert env.council.messages[0].to == "mayor"
    assert agent.usage["calls"] == 2 and agent.usage["parse_failures"] == 1
    assert FakeLLM.calls[0]["model"] == "fake"
    srv.shutdown()


def test_cli_overrides():
    cfg = apply_cli_overrides(load_config("configs/baseline.toml"),
                              ["planning=llm:arcee/trinity-mini"], ["game.months=6"])
    assert cfg["agents"]["planning"] == {"kind": "llm", "provider": "arcee", "model": "trinity-mini"}
    assert cfg["game"]["months"] == 6


def test_scripted_episode_logs_everything(tmp_path):
    cfg = apply_cli_overrides(load_config("configs/baseline.toml"), [], ["game.months=36"])
    s = run_episode(cfg, seed=2, runs_dir=tmp_path, verbose=False)
    d = tmp_path / s["run_id"]
    frames = (d / "frames.jsonl").read_text().splitlines()
    assert len(frames) == 37
    assert s["final_metrics"]["population"] > 500
    assert s["final_scores"]["city"] > 30
    traj = [json.loads(line) for line in (d / "trajectories.jsonl").read_text().splitlines()]
    assert {t["role"] for t in traj} == set(TURN_ORDER)
    assert all("reward" in t for t in traj)


def test_scripted_mayor_reads_the_requests_since_its_previous_turn():
    """Acting every 3 months, a request posted right after the mayor's turn (departments act after it) is read at
    its next turn; acting monthly the window is the month before, as it always was."""
    env = CivitasEnv(GameConfig(seed=0, months=12))
    obs = env.observe("mayor")
    city = {**obs["city"], "power": {"supply": 500, "demand": 10}, "water": {"supply": 500, "demand": 10}}
    mayor = SCRIPTED["mayor"]("mayor")
    ask = lambda month: {"month": month, "sender": "planning", "to": "mayor", "text": "Planning needs $12,000."}
    fund = lambda month: {"month": month, "sender": "utilities", "to": "mayor", "text": "Utilities needs funds."}

    def turn(month, *council):          # the working balances the mayor aims for (planning 9,000, utilities 3,000)
        mayor.act({**obs, "month": month, "city": city, "council": list(council)})
        return mayor.targets["planning"], mayor.targets["utilities"]

    mayor.reset(0)                                                  # every 3 months
    assert turn(0) == (9000, 3000)
    assert turn(3, ask(0), fund(0)) == (12200, 7000)                # posted after its month-0 turn: read
    assert turn(6, ask(0), fund(0)) == (9000, 3000)                 # read last turn
    assert turn(9, ask(6), fund(6)) == (12200, 7000)
    mayor.reset(0)                                                  # monthly: the month before, as before
    assert turn(0) == (9000, 3000)
    assert turn(1, ask(0), fund(0)) == (12200, 7000)
    assert turn(2, ask(0), fund(0)) == (9000, 3000)
    mayor.reset(0)                                                  # a new episode: no previous turn
    assert turn(5, ask(3), fund(3)) == (9000, 3000)
    assert turn(6, ask(5)) == (12200, 3000)
    assert turn(3, ask(2)) == (12200, 3000)                         # a turn before the last one: the month before


def _mayor_turns(cls):
    """turn(month, funds, *council, forecast=(), supply=900) -> (decision, targets) of one scripted mayor on a city of
    500 with ample supply (power and water demand 100; supply=120 is tight) and a rich treasury; funds left out hold
    what the mayor's last turn left them (nothing spent)."""
    env = CivitasEnv(GameConfig(seed=0, months=24))
    base = env.observe("mayor")
    mayor = cls("mayor")
    mayor.reset(0)

    def turn(month, funds=None, *council, forecast=(), supply=900):
        city = {**base["city"], "population": 500, "power": {"supply": supply, "demand": 100},
                "water": {"supply": supply, "demand": 100}}
        held = {d: 0.0 for d in base["finance"]["department_funds"]} | dict(getattr(mayor, "held", None) or {})
        fin = {**base["finance"], "treasury": 200000, "department_funds": held | dict(funds or {})}
        d = mayor.act({**base, "month": month, "city": city, "finance": fin, "council": list(council),
                       "forecast": list(forecast), "offers": []})
        return d, dict(mayor.targets)
    return turn


def _takes(d) -> dict:
    return {a["department"]: -a["amount"] for a in d.actions if a["type"] == "allocate_budget" and a["amount"] < 0}


def test_scripted_mayor_stops_funding_an_ask_a_department_sits_on():
    """v1's scripted planning held $20k for a casino through seed 0's last years: funded in full, it zoned first, came
    up short and asked again. Holding (nearly) all of an ask it was fully funded for and asking the same again, the
    ask goes unfunded while it repeats it and the money above its working balance comes back, before any top-up."""
    from civitas.agents.scripted import ScriptedMayor, ScriptedMayorV1
    casino = lambda month, amount=20000: {"month": month, "sender": "planning", "to": "mayor",
                                          "text": f"Planning needs ${amount:,} for a casino: jobs for low-skill workers."}
    turn = _mayor_turns(ScriptedMayor)
    d, t = turn(3, None, casino(2))
    assert t["planning"] == 20200 and {"type": "allocate_budget", "department": "planning", "amount": 20200} in d.actions
    d, t = turn(6, {"planning": 20050}, casino(3))                  # spent $150 of it, asks the same again
    assert t["planning"] == 7000 and _takes(d) == {"planning": 13050}
    assert d.actions[0] == {"type": "allocate_budget", "department": "planning", "amount": -13050}
    assert "took back idle planning 13050" in d.thinking
    d, t = turn(9, {"planning": 6900}, casino(6))                   # still asking it: still unfunded
    assert t["planning"] == 7000 and not _takes(d)
    d, t = turn(12, {"planning": 7000}, casino(9, 25000))           # a new ask is funded
    assert t["planning"] == 25200
    v1 = _mayor_turns(ScriptedMayorV1)                              # civitas-v1: every ask funded, nothing taken
    v1(3, None, casino(2))
    d, t = v1(6, {"planning": 20050}, casino(3))
    assert t["planning"] == 20200 and not _takes(d)


def test_scripted_mayor_keeps_a_service_balance_only_while_it_spends():
    """A service gets its usual working balance until it builds its first building; after that it keeps twice what
    it spent over the last 6 months (at least $1,000, at most the usual balance) and asks for a build it can't pay.
    A take-back waits two turns after its balance falls, and the money for an ask stays while it asked in this or
    the last two turns (taken back, it would be given again for the same build). Fire isn't touched while a flood
    is forecast."""
    from civitas.agents.scripted import ScriptedMayor
    turn = _mayor_turns(ScriptedMayor)
    d, t = turn(3)
    assert (t["police"], t["fire"], t["transport"]) == (3000, 3000, 2000)
    d, t = turn(6, {"police": 500})                                 # police built its first station
    assert t["police"] == 3000 and not _takes(d)
    d, t = turn(9)
    assert t["police"] == 3000 and not _takes(d)                    # spent $2,500 within 6 months
    d, t = turn(12)
    assert t["police"] == 1000 and not _takes(d)                    # nothing spent since month 6: 3,000 last turn
    d, t = turn(15)
    assert t["police"] == 1000 and not _takes(d)                    # ... and the turn before
    d, t = turn(18)
    assert t["police"] == 1000 and _takes(d) == {"police": 2000}
    assert t["fire"] == 3000                                        # no first building yet: the full balance
    station = {"month": 18, "sender": "police", "to": "mayor", "text": "Police needs $2,500 for a new building."}
    d, t = turn(21, None, station)
    assert t["police"] == 2700 and {"type": "allocate_budget", "department": "police", "amount": 1700} in d.actions
    d, t = turn(24)                                                 # not built yet, not asking again
    assert t["police"] == 1000 and not _takes(d)                    # asked last turn: it keeps the money
    d, t = turn(27)
    assert not _takes(d)                                            # asked two turns ago
    d, t = turn(30)
    assert _takes(d) == {"police": 1700}                            # no ask in the last two turns: idle
    d, t = turn(33, {"fire": 0})                                    # fire built: the full balance while it spends
    d, t = turn(39, {"fire": 3000})
    d, t = turn(42, {"fire": 4000}, forecast=["River flood expected in 2 month(s) along rows 3-17: ..."])
    assert "fire" not in _takes(d)


def test_scripted_mayor_keeps_utilities_money_while_supply_swings():
    """Utilities' working balance is 7,000 while supply is under 1.3x demand and 3,000 otherwise. The money for the
    tight balance stays for two turns after supply recovers (the first cut took $4,000 back and gave it again the
    next quarter as the margin crossed 1.3x); after that the money above 3,000 comes back."""
    from civitas.agents.scripted import ScriptedMayor
    turn = _mayor_turns(ScriptedMayor)
    d, t = turn(3, {"utilities": 3000}, supply=120)                 # tight: topped up to 7,000
    assert t["utilities"] == 7000 and {"type": "allocate_budget", "department": "utilities",
                                       "amount": 4000} in d.actions
    d, t = turn(6)
    assert t["utilities"] == 3000 and not _takes(d)                 # ample again: kept
    d, t = turn(9, supply=120)                                      # tight again: nothing to give back
    assert t["utilities"] == 7000 and not [a for a in d.actions if a.get("department") == "utilities"]
    d, t = turn(12)
    d, t = turn(15)
    assert not _takes(d)
    d, t = turn(18)
    assert _takes(d) == {"utilities": 4000}


def test_remote_agent_over_http(tmp_path):
    hub = LiveHub()
    httpd = serve(tmp_path, port=0, hub=hub, background=True)
    port = httpd.server_address[1]
    cfg = apply_cli_overrides(load_config("configs/remote.toml"), [], ["game.months=2"])
    result = {}
    t = threading.Thread(target=lambda: result.update(run_episode(cfg, 0, tmp_path, hub=hub, verbose=False)))
    t.start()
    base = f"http://127.0.0.1:{port}/api/live"
    turns, deadline = 0, time.time() + 60       # a missed turn fails here instead of polling forever (one suite run
    while t.is_alive() and turns < 2:           # under heavy load hung 17 minutes in a test with a server socket)
        assert time.time() < deadline, f"no planning turn to answer after 60 s ({turns} answered)"
        turn = json.loads(urllib.request.urlopen(f"{base}/turn?role=planning", timeout=30).read())
        if not turn["waiting"]:
            time.sleep(0.05)
            continue
        body = json.dumps({"role": "planning", "turn_id": turn["turn_id"],
                           "decision": {"actions": [{"type": "noop"}], "thinking": "remote hi"}}).encode()
        req = urllib.request.Request(f"{base}/act", data=body, headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req, timeout=30).read())["ok"]
        turns += 1
    t.join(timeout=30)
    httpd.shutdown()
    assert result["months_played"] == 2 and turns == 2


def test_llm_episode_through_runner(tmp_path, monkeypatch):
    FakeLLM.calls = [None]  # skip the junk reply
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("FAKE_KEY", "k")
    cfg = load_config("configs/smoke.toml")
    cfg["providers"]["fake"] = {"base_url": f"http://127.0.0.1:{srv.server_port}/v1", "api_key_env": "FAKE_KEY"}
    for r in ("mayor", "planning", "utilities"):
        cfg["agents"][r]["provider"] = "fake"
    cfg["game"]["months"] = 3
    s = run_episode(cfg, 0, tmp_path, verbose=False)
    srv.shutdown()
    assert s["agent_stats"]["planning"]["llm_usage"]["calls"] >= 1
    traj = [json.loads(x) for x in (tmp_path / s["run_id"] / "trajectories.jsonl").read_text().splitlines()]
    assert traj and "prompt" in traj[0] and traj[0]["response"]
    # mayor can't zone -> the fake reply's zone action is rejected with a clear message
    mayor = [t for t in traj if t["role"] == "mayor"][0]
    assert not mayor["results"][0]["ok"] and "cannot use" in mayor["results"][0]["message"]


def test_decision_normalizes_llm_variants():
    from civitas.core.decision import Decision
    d = Decision.from_dict({"actions": [{"action": "zone", "params": {"zone": "R", "x": 1, "y": 2}}],
                            "message": "hello"})
    assert d.actions[0] == {"type": "zone", "zone": "R", "x": 1, "y": 2}
    assert d.messages == [{"to": "all", "text": "hello"}]


def test_thinking_model_answer_in_reasoning_channel():
    out = {"content": "", "reasoning": 'Let me think... final: {"actions": [{"type": "noop"}]}', "finish_reason": "stop"}
    assert LLMAgent._parse(out)["actions"][0]["type"] == "noop"
    out = {"content": '<think>zone at {x}</think>{"actions": []}', "reasoning": ""}
    assert LLMAgent._parse(out) == {"actions": []}


def test_dotenv_loading(tmp_path, monkeypatch):
    import os
    from civitas.cli import load_dotenv
    monkeypatch.delenv("CIVITAS_TEST_KEY", raising=False)
    (tmp_path / ".env").write_text("# comment\nexport CIVITAS_TEST_KEY='abc'\n")
    load_dotenv(str(tmp_path / ".env"))
    assert os.environ["CIVITAS_TEST_KEY"] == "abc"


def test_server_configs_launch_validation(tmp_path):
    import urllib.error
    httpd = serve(tmp_path, port=0, background=True, configs_dir="configs")
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    cfgs = json.loads(urllib.request.urlopen(f"{base}/api/configs").read())
    assert any(c["file"] == "baseline.toml" for c in cfgs)
    assert urllib.request.urlopen(f"{base}/").status == 200            # 3D interface
    assert urllib.request.urlopen(f"{base}/vendor/three-bundle.min.js").status == 200

    def post(body):
        req = urllib.request.Request(f"{base}/api/launch", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(req).read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())
    assert not post({"config": "../pyproject.toml"})["ok"]
    assert not post({"config": "baseline.toml", "agents": {"mayor": "llm:x; rm -rf /"}})["ok"]
    r = post({"config": "baseline.toml", "seed": 1, "months": 6})
    assert r["ok"]
    for _ in range(100):
        runs = json.loads(urllib.request.urlopen(f"{base}/api/runs").read())
        if runs and runs[0]["status"] == "finished":
            break
        time.sleep(0.2)
    assert runs[0]["run_id"] == r["run_id"] and runs[0]["status"] == "finished"
    frames = json.loads(urllib.request.urlopen(f"{base}/api/runs/{r['run_id']}/frames").read())
    assert len(frames["frames"]) == 7
    turns = json.loads(urllib.request.urlopen(f"{base}/api/runs/{r['run_id']}/turns?month=0").read())
    assert {t["role"] for t in turns} == set(TURN_ORDER)
    assert not any("trace" in t for t in turns)                         # scripted turns have no tool trace
    httpd.shutdown()


def _turn_rec(month, role, trace=None, **kw):
    """A trajectories.jsonl record with the runner's key order; a fleet turn when `trace` is given."""
    rec = {"run_id": "r1", "seed": 0, "month": month, "role": role,
           "agent": "fleet:fake" if trace is not None else "scripted", "results": [], **kw}
    if trace is not None:
        rec.update(prompt=[{"role": "system", "content": "sys"}, {"role": "user", "content": f"briefing {month}"}],
                   response="report", reasoning="thinking", finish_reason="end_turn", tool_trace=trace,
                   steps=len(trace), ended=True, finish_reasons=["tool_calls"] * 3, length_stops=0, recovered_calls=0)
    else:
        rec["observation_text"] = "obs"
    return json.dumps(rec) + "\n"


def _get(url, gzip_ok=False):
    import gzip
    import urllib.error
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"} if gzip_ok else {})
    try:
        r = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), None
    body, enc = r.read(), r.headers.get("Content-Encoding")
    return r.status, json.loads(gzip.decompress(body) if enc == "gzip" else body), enc


def test_server_fleet_tool_trace(tmp_path):
    """/trace returns one fleet turn's calls with outcomes, found through a byte-offset index that follows a live
    run's appends; /turns carries the same summary."""
    d = tmp_path / "r1"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"name": "t", "roles": ["mayor", "planning"]}))
    trace = [
        {"step": 0, "finish_reason": "tool_calls", "tool": "map_overview", "args": "{}", "result": {"ok": True}},
        {"step": 0, "finish_reason": "tool_calls", "tool": "zone", "args": '{"zone": "R", "x1": 1, "y1": 2}',
         "result": {"ok": True, "message": "zoned 4 tiles R", "cost": 40.0, "budget_left": 960.0}},
        {"step": 1, "finish_reason": "tool_calls", "tool": "build_road", "args": '{"x1": 0, "y1": 0}',
         "result": {"ok": False, "error": "tile (0,0) is taken; route around", "message": "tile (0,0) is taken",
                    "cost": 0.0, "budget_left": 960.0}},
        {"step": 1, "finish_reason": "tool_calls", "tool": "find_sites", "args": '{"purpose": "x"}',
         "result": {"ok": False, "error": "argument 'purpose' must be one of ..."}},
        {"step": 2, "finish_reason": "tool_calls", "tool": "build_road", "args": '{"x1": 0, "y1": 0}',
         "result": {"ok": False, "refused": "repeat", "error": "this exact call failed at call 3: tile taken"}},
        {"step": 2, "finish_reason": "tool_calls", "tool": "map_overview", "args": "{}",
         "result": {"ok": True, "repeat": "same as call 1 (nothing changed since)"}},
        {"step": 2, "finish_reason": "stop", "tool": "zone", "args": "zone R please", "result": {"ok": False,
         "error": "arguments must be a JSON object"}},
        {"step": 3, "finish_reason": "tool_calls", "tool": "end_turn", "args": '{"report": "done"}',
         "result": {"ok": True, "message": "turn ended"}},
    ]
    big = [{"step": i, "finish_reason": "tool_calls", "tool": "inspect", "args": json.dumps({"x": i, "y": i}),
            "result": {"ok": True, "message": f"tile {i}: " + "x" * 280}} for i in range(120)]
    traj = d / "trajectories.jsonl"
    traj.write_text(_turn_rec(0, "mayor", trace) + _turn_rec(0, "planning") + _turn_rec(1, "planning", big))
    httpd = serve(tmp_path, port=0, background=True, allow_launch=False)
    base = f"http://127.0.0.1:{httpd.server_address[1]}/api/runs/r1"
    try:
        code, t, _ = _get(f"{base}/trace?month=0&role=mayor")
        assert code == 200 and t["agent"] == "fleet:fake" and t["month"] == 0 and t["role"] == "mayor"
        calls = t["calls"]
        assert [c["n"] for c in calls] == list(range(1, 9))
        assert [c["outcome"] for c in calls] == ["ok", "ok", "failed", "failed", "refused", "repeat", "failed", "ok"]
        assert [c["action"] for c in calls] == [False, True, True, False, True, False, True, False]
        assert [c["ran"] for c in calls] == [False, True, True, False, False, False, False, False]
        assert calls[1]["args"] == {"zone": "R", "x1": 1, "y1": 2} and calls[1]["cost"] == 40.0
        assert calls[2]["text"] == "tile (0,0) is taken; route around"          # the error the model read
        assert calls[4]["refused"] == "repeat" and calls[5]["text"].startswith("same as call 1")
        assert calls[6]["args"] == "zone R please" and calls[6]["finish_reason"] == "stop"
        assert t["summary"] == {"calls": 8, "actions": 2, "failed_actions": 1, "errors": 2, "refused": 1,
                                "repeats": 1, "spent": 40.0, "model_calls": 3, "finish_reason": "end_turn",
                                "ended": True, "length_stops": 0, "recovered_calls": 0}
        code, turns, _ = _get(f"{base}/turns?month=0")
        assert [x["role"] for x in turns] == ["mayor", "planning"]
        assert turns[0]["trace"] == t["summary"] and turns[0]["observation"] == "briefing 0"
        assert "trace" not in turns[1] and turns[1]["observation"] == "obs"
        # a large trace comes gzipped
        code, t1, enc = _get(f"{base}/trace?month=1&role=planning", gzip_ok=True)
        assert code == 200 and enc == "gzip" and len(t1["calls"]) == 120
        # errors
        assert _get(f"{base}/trace?month=0&role=planning")[0] == 404                # scripted: no tool trace
        assert _get(f"{base}/trace?month=5&role=mayor")[0] == 404
        assert _get(f"{base}/trace?month=x&role=mayor")[0] == 400
        assert _get(f"{base}/trace?month=0")[0] == 400
        assert _get(f"http://127.0.0.1:{httpd.server_address[1]}/api/runs/nope/trace?month=0&role=mayor")[0] == 404
        # a live run appends: complete lines are indexed on the next request, a line still being written is not
        tail = _turn_rec(2, "mayor", trace[:2])
        with open(traj, "a") as fh:
            fh.write(_turn_rec(1, "mayor", trace[:1]) + tail[:40])
        assert _get(f"{base}/trace?month=1&role=mayor")[1]["summary"]["calls"] == 1
        assert _get(f"{base}/trace?month=2&role=mayor")[0] == 404
        with open(traj, "a") as fh:
            fh.write(tail[40:])
        code, t2, _ = _get(f"{base}/trace?month=2&role=mayor")
        assert code == 200 and [c["tool"] for c in t2["calls"]] == ["map_overview", "zone"]
        assert [x["role"] for x in _get(f"{base}/turns?month=1")[1]] == ["planning", "mayor"]
        # a rewritten file is indexed again
        traj.write_text(_turn_rec(0, "mayor", trace[:3]))
        assert _get(f"{base}/trace?month=0&role=mayor")[1]["summary"]["calls"] == 3
        assert _get(f"{base}/trace?month=1&role=planning")[0] == 404
    finally:
        httpd.shutdown()


def test_turn_index_reads_only_the_lines_it_returns(tmp_path):
    from civitas.server import TurnIndex
    f = tmp_path / "trajectories.jsonl"
    odd = json.dumps({"role": "fire", "month": 4, "run_id": "r1"}) + "\n"      # not the runner's key order
    f.write_text(_turn_rec(3, "north.police", []) + odd + "not json\n" + _turn_rec(4, "fire", []))
    ix = TurnIndex()
    assert [r["role"] for r in ix.records(f, 3)] == ["north.police"]
    assert ix.records(f, 4, "fire")[0] == json.loads(odd)          # the first record of a (month, role) wins
    assert [r.get("agent") for r in ix.records(f, 4)] == [None, "fleet:fake"]
    assert ix.records(f, 9) == [] and ix.records(tmp_path / "missing.jsonl", 0) == []
    entry = ix.files[str(f)]
    assert entry["end"] == f.stat().st_size and sorted(entry["month"]) == [3, 4]


def test_simultaneous_mode_runs(tmp_path):
    cfg = apply_cli_overrides(load_config("configs/baseline.toml"), [], ["game.months=6", "game.simultaneous=true"])
    s = run_episode(cfg, seed=1, runs_dir=tmp_path, verbose=False)
    assert s["months_played"] == 6


def test_actions_recovered_from_cut_off_reasoning():
    text = ("Long deliberation... maybe {\"type\": \"noop\"}? " * 3 + "x" * 2000 +
            ' Final plan: {"type": "zone", "zone": "R", "x1": 1, "y1": 2, "x2": 5, "y2": 3} and '
            '{"type": "build_road", "x1": 19, "y1": 24, "x2": 30, "y2": 24}.')
    d = LLMAgent._parse({"content": "", "reasoning": text, "finish_reason": "length"})
    assert [a["type"] for a in d["actions"]] == ["zone", "build_road"]


class _DripThenAnswer(BaseHTTPRequestHandler):
    """First request: headers, then one body byte every 0.2 s (never silent long enough for a socket
    timeout, like a hung API call). Later requests: a normal answer."""
    calls = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).calls += 1
        out = json.dumps({"choices": [{"message": {"content": '{"actions": []}'}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out) + 1000))
        self.end_headers()
        if type(self).calls == 1:
            try:
                for _ in range(100):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.2)
            except OSError:
                pass
            return
        self.wfile.write(out + b" " * 1000)


def test_chat_client_gives_up_on_a_hung_request_and_retries():
    from civitas.agents.llm_client import ChatClient
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _DripThenAnswer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    client = ChatClient(ProviderConfig(base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key_env="",
                                       timeout=5.0, deadline=1.0, max_retries=1))
    t0 = time.time()
    out = client.chat("fake", [{"role": "user", "content": "hi"}])
    assert out["content"] == '{"actions": []}'
    assert _DripThenAnswer.calls == 2
    assert time.time() - t0 < 6          # the drip would take 20 s; the deadline cuts it at ~1 s
    srv.shutdown()


class _StreamingLLM(BaseHTTPRequestHandler):
    """Answers with server-sent events, as Arcee does for "stream": true."""
    bodies = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        type(self).bodies.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [{"choices": [{"delta": {"reasoning_content": "think "}}]},
                  {"choices": [{"delta": {"reasoning_content": "more"}}]},
                  {"choices": [{"delta": {"content": '{"actions": '}}]},
                  {"choices": [{"delta": {"content": "[]}"}, "finish_reason": "stop"}]},
                  {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 5}}]
        for c in chunks:
            self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


def test_chat_client_streams_and_reassembles_the_answer():
    from civitas.agents.llm_client import ChatClient
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _StreamingLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    client = ChatClient(ProviderConfig(base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key_env=""))
    out = client.chat("fake", [{"role": "user", "content": "hi"}], max_tokens=100)
    assert out["content"] == '{"actions": []}' and out["reasoning"] == "think more"
    assert out["finish_reason"] == "stop" and out["usage"]["completion_tokens"] == 5
    assert _StreamingLLM.bodies[0]["stream"] is True
    srv.shutdown()


class _FlakyStream(BaseHTTPRequestHandler):
    """1st request: one data chunk, then only keep-alive pings (a stalled generation).
    2nd: a stream that stops without a finish reason. 3rd: a complete answer."""
    calls = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).calls += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        send = lambda c: (self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n"), self.wfile.flush())
        try:
            send({"choices": [{"delta": {"reasoning_content": "hmm"}}]})
            if type(self).calls == 1:
                for _ in range(100):
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    time.sleep(0.2)
                return
            if type(self).calls == 2:
                return
            send({"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n")
        except OSError:
            pass


def test_chat_client_retries_stalled_and_truncated_streams(capsys):
    from civitas.agents.llm_client import ChatClient
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FlakyStream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    client = ChatClient(ProviderConfig(base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key_env="",
                                       timeout=1.0, max_retries=2))
    t0 = time.time()
    out = client.chat("fake", [{"role": "user", "content": "hi"}])
    assert out["content"] == "done" and out["finish_reason"] == "stop"
    assert _FlakyStream.calls == 3
    assert time.time() - t0 < 12         # the ping stream would last 20 s; it is cut after ~1 s of no data
    log = capsys.readouterr().err.splitlines()     # nothing slept: the retry lines carry no sleep note
    assert re.fullmatch(r"\[llm\] fake: attempt 1 failed after \d s \(TimeoutError: stream stalled: "
                        r"no data for 1 s\); retrying", log[0])
    assert re.fullmatch(r"\[llm\] fake: attempt 2 failed after \d s \(ConnectionError: stream ended "
                        r"before the answer was finished\); retrying", log[1])
    assert len(log) == 2
    srv.shutdown()


class _ToolCallLLM(BaseHTTPRequestHandler):
    """Streams a tool call split across chunks, as OpenAI-style servers do."""
    stream = True

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        assert body["tools"][0]["function"]["name"] == "get_budget"
        chunks = [{"choices": [{"delta": {"content": "checking", "tool_calls": [
                      {"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_budget", "arguments": ""}}]}}]},
                  {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"dep'}}]}}]},
                  {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'artment":"health"}'}}]},
                                "finish_reason": "tool_calls"}]}]
        if type(self).stream:
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            for c in chunks:
                self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            out = json.dumps({"choices": [{"message": {"content": "checking", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_budget", "arguments": '{"department":"health"}'}}]},
                "finish_reason": "tool_calls"}], "usage": {}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)


def test_chat_client_returns_tool_calls_streamed_and_plain():
    from civitas.agents.llm_client import ChatClient
    tools = [{"type": "function", "function": {"name": "get_budget", "parameters": {"type": "object", "properties": {}}}}]
    for stream in (True, False):
        _ToolCallLLM.stream = stream
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _ToolCallLLM)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        c = ChatClient(ProviderConfig(base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key_env="", stream=stream))
        out = c.chat("fake", [{"role": "user", "content": "hi"}], extra={"tools": tools})
        assert out["finish_reason"] == "tool_calls"
        assert out["tool_calls"] == [{"id": "call_1", "type": "function",
                                      "function": {"name": "get_budget", "arguments": '{"department":"health"}'}}]
        srv.shutdown()


class _SleepyClock:
    """Stands in for llm_client's `time` module. The clocks are the real ones, except that `nap(s)`
    puts the wall clock s seconds ahead of the monotonic one, as a laptop that slept for s seconds
    finds on waking. `sleep` (the retry backoff) is cut short to keep the tests fast."""

    def __init__(self):
        self.napped = 0.0

    def nap(self, s):
        self.napped += s

    def time(self):
        return time.time() + self.napped

    def monotonic(self):
        return time.monotonic()

    def sleep(self, s):
        time.sleep(min(s, 0.01))


class _SleepsMidAnswer(BaseHTTPRequestHandler):
    """With `nap` set, the 1st request starts its answer, the client machine then sleeps for `nap` s
    and the connection goes silent (it died in the sleep) until the test ends. Other requests: a
    normal answer. `stream` picks server-sent events or a plain JSON body."""
    clock, nap, stream, calls, release = None, 0.0, True, 0, None

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        cls = type(self)
        cls.calls += 1
        stall = cls.nap and cls.calls == 1
        body = json.dumps({"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
                           "usage": {"completion_tokens": 1}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if cls.stream else "application/json")
        if not cls.stream:
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            if cls.stream:
                self.wfile.write(b'data: {"choices": [{"delta": {"content": "do"}}]}\n\n')
            self.wfile.flush()
            if stall:
                time.sleep(0.2)              # the client has read what came so far
                cls.clock.nap(cls.nap)
                cls.release.wait(10)
                return
            if cls.stream:
                self.wfile.write(b'data: {"choices": [{"delta": {"content": "ne"}, "finish_reason": "stop"}], '
                                 b'"usage": {"completion_tokens": 1}}\n\ndata: [DONE]\n\n')
            else:
                self.wfile.write(body)
        except OSError:
            pass


def _sleepy_chat(monkeypatch, *, nap, stream, max_retries=1):
    """chat() against _SleepsMidAnswer on the sleepy clock; returns (answer or LLMError, real seconds)."""
    from civitas.agents import llm_client
    clock = _SleepyClock()
    monkeypatch.setattr(llm_client, "time", clock)
    monkeypatch.setattr(llm_client, "JOIN_SLICE", 0.05)
    h = _SleepsMidAnswer
    h.clock, h.nap, h.stream, h.calls, h.release = clock, nap, stream, 0, threading.Event()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    client = llm_client.ChatClient(ProviderConfig(base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                                                  api_key_env="", stream=stream, max_retries=max_retries))
    t0 = time.time()
    try:
        out = client.chat("fake", [{"role": "user", "content": "hi"}])
    except llm_client.LLMError as e:
        out = e
    finally:
        h.release.set()
        srv.shutdown()
    return out, time.time() - t0


@pytest.mark.parametrize("stream", [True, False])
def test_chat_client_deadline_runs_on_the_wall_clock(monkeypatch, capsys, stream):
    # 35 minutes asleep: past the 1,800 s deadline on the wall clock, though the monotonic clock
    # (and so Thread.join and the socket timeout) saw well under a second
    out, took = _sleepy_chat(monkeypatch, nap=2100, stream=stream)
    assert out["content"] == "done" and _SleepsMidAnswer.calls == 2
    assert took < 3                      # ended on waking, not 10 s later when the server let go
    assert re.fullmatch(r"\[llm\] fake: attempt 1 failed after 210\d s \(system slept ~2100 s; "
                        r"TimeoutError: no complete response after 1800 s\); retrying\n", capsys.readouterr().err)


def test_chat_client_idle_check_runs_on_the_wall_clock(monkeypatch, capsys):
    # 10 minutes asleep: inside the deadline, but past the 300 s idle limit while no line arrives,
    # so _read_events' own check never runs
    out, took = _sleepy_chat(monkeypatch, nap=600, stream=True)
    assert out["content"] == "done" and _SleepsMidAnswer.calls == 2 and took < 3
    assert re.fullmatch(r"\[llm\] fake: attempt 1 failed after 60\d s \(system slept ~600 s; "
                        r"TimeoutError: stream stalled: no data for 300 s\); retrying\n", capsys.readouterr().err)


def test_chat_client_notes_the_sleep_when_giving_up(monkeypatch):
    out, took = _sleepy_chat(monkeypatch, nap=2100, stream=True, max_retries=0)
    assert str(out) == ("giving up after 1 attempts: system slept ~2100 s; "
                        "TimeoutError: no complete response after 1800 s") and took < 3


@pytest.mark.parametrize("stream", [True, False])
def test_chat_client_answers_as_before_when_nothing_slept(monkeypatch, capsys, stream):
    out, took = _sleepy_chat(monkeypatch, nap=0, stream=stream)
    assert (out["content"], out["finish_reason"], out["usage"]) == ("done", "stop", {"completion_tokens": 1})
    assert _SleepsMidAnswer.calls == 1 and took < 1 and capsys.readouterr().err == ""
