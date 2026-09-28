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


def test_remote_agent_over_http(tmp_path):
    hub = LiveHub()
    httpd = serve(tmp_path, port=0, hub=hub, background=True)
    port = httpd.server_address[1]
    cfg = apply_cli_overrides(load_config("configs/remote.toml"), [], ["game.months=2"])
    result = {}
    t = threading.Thread(target=lambda: result.update(run_episode(cfg, 0, tmp_path, hub=hub, verbose=False)))
    t.start()
    base = f"http://127.0.0.1:{port}/api/live"
    turns = 0
    while t.is_alive() and turns < 2:
        turn = json.loads(urllib.request.urlopen(f"{base}/turn?role=planning").read())
        if not turn["waiting"]:
            time.sleep(0.05)
            continue
        body = json.dumps({"role": "planning", "turn_id": turn["turn_id"],
                           "decision": {"actions": [{"type": "noop"}], "thinking": "remote hi"}}).encode()
        req = urllib.request.Request(f"{base}/act", data=body, headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req).read())["ok"]
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
    httpd.shutdown()


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
