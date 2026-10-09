"""The live feed (bench/live.py): events and per-build frames while agents play, read by the viewer's live mode."""
import json
import urllib.request
from collections import Counter
from pathlib import Path

from civitas.agents.llm_client import ProviderConfig  # noqa: F401  (the fleet imports it too)
from civitas.bench import live
from civitas.bench.runner import run_episode
from civitas.config import apply_cli_overrides, load_config
from civitas.core.env import CivitasEnv, GameConfig
from civitas.fleet.agent import ToolAgent
from civitas.fleet.config import FleetAgentConfig
from civitas.server import serve

ROOT = Path(__file__).resolve().parents[1]


def _cfg(name, months):
    return apply_cli_overrides(load_config(ROOT / "configs" / name), sets=[f"game.months={months}"])


def _events(d):
    return [json.loads(x) for x in (d / live.EVENTS).read_text().splitlines()]


def _logs(d):
    trajs = [{k: v for k, v in json.loads(x).items() if k not in ("run_id", "latency_s")}
             for x in (d / "trajectories.jsonl").read_text().splitlines()]
    return (d / "frames.jsonl").read_text(), trajs


def test_a_frame_after_every_build_leaves_the_game_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "FRAME_EVERY_S", 0.0)
    for name, months in (("fleet_scripted.toml", 12), ("region.toml", 6)):
        on = run_episode(_cfg(name, months), 1, tmp_path, verbose=False, run_id=f"on_{name}", yardstick=False)
        off = run_episode(_cfg(name, months), 1, tmp_path, verbose=False, run_id=f"off_{name}", live=False,
                          yardstick=False)
        assert on["final_scores"] == off["final_scores"]
        assert _logs(tmp_path / f"on_{name}") == _logs(tmp_path / f"off_{name}")
        ev = _events(tmp_path / f"on_{name}")
        builds = sum(e["kind"] == "call" and e["action"] and e["ok"] for e in ev)
        assert builds and sum(e["kind"] == "frame" for e in ev) >= builds // 2   # turn after turn, nearly each
        assert not (tmp_path / f"off_{name}" / live.EVENTS).exists()
        assert not (tmp_path / f"on_{name}" / live.FRAME).exists()        # the month's own frame replaces it


def test_the_feed_tells_each_turn_its_calls_and_the_months(tmp_path):
    run_episode(_cfg("fleet_scripted.toml", 6), 0, tmp_path, verbose=False, run_id="f", yardstick=False)
    ev = _events(tmp_path / "f")
    assert [e["i"] for e in ev] == list(range(1, len(ev) + 1))
    kinds = Counter(e["kind"] for e in ev)
    assert kinds["turn"] == kinds["done"] == 16 and kinds["month"] == 6        # 8 roles at months 0 and 3
    trajs = [json.loads(x) for x in (tmp_path / "f" / "trajectories.jsonl").read_text().splitlines()]
    for t in trajs:                                  # each turn's actions on the feed, in order, between its start and end
        start = next(i for i, e in enumerate(ev) if e["kind"] == "turn" and e["role"] == t["role"]
                     and e["month"] == t["month"])
        end = next(i for i, e in enumerate(ev) if i > start and e["kind"] == "done")
        calls = [e for e in ev[start:end] if e["kind"] == "call" and e["action"]]
        assert [e["tool"] for e in calls] == [r["action"]["type"] for r in t["results"]]
        assert ev[end]["actions"] == sum(r["ok"] for r in t["results"]) and ev[end]["role"] == t["role"]
    say = [e for e in ev if e["kind"] == "call" and e["tool"] == "send_message"]
    assert say and all(e["args"]["text"] and e["action"] is False for e in say)


def test_plain_agents_and_region_roles_are_reported_after_their_turn(tmp_path):
    run_episode(_cfg("region.toml", 3), 0, tmp_path, verbose=False, run_id="r", yardstick=False)
    ev = _events(tmp_path / "r")
    roles = {e["role"] for e in ev if e["kind"] == "turn"}
    assert "north.planning" in roles and "south.mayor" in roles
    built = [e for e in ev if e["kind"] == "call" and e["action"] and e["ok"]]
    assert built and all("." in e["role"] and "type" not in e["args"] for e in built)


def test_a_model_turn_streams_its_steps(tmp_path):
    env = CivitasEnv(GameConfig(seed=0, months=12))
    log = live.LiveLog(tmp_path, env, frame_every=0.0)
    a = ToolAgent("planning", cfg=FleetAgentConfig(model="fake"))
    steps = iter([[{"id": "1", "type": "function", "function": {"name": "find_sites", "arguments": '{"purpose": "road_extension"}'}}],
                  [{"id": "2", "type": "function", "function": {"name": "end_turn", "arguments": '{"report": "looked"}'}}]])

    class Client:
        def chat(self, model, messages, **kw):
            return {"content": "", "reasoning": "first I look for land " * 40, "tool_calls": next(steps),
                    "finish_reason": "tool_calls", "usage": {}, "latency_s": 1.23}
    a.client = Client()
    a.bind(env)
    a.act(env.observe("planning"))
    log.close()
    ev = [(e["kind"], e.get("step", e.get("tool"))) for e in _events(tmp_path)]
    assert ev == [("think", 0), ("reply", 0), ("call", "find_sites"), ("think", 1), ("reply", 1), ("call", "end_turn")]
    reply = _events(tmp_path)[1]
    assert reply["role"] == "planning" and len(reply["text"]) == live.TEXT_CHARS and reply["latency_s"] == 1.2
    assert live.sink(env) is None                    # closed: nothing reports into it any more


def test_reading_the_feed_by_offset_and_tail(tmp_path):
    p = tmp_path / live.EVENTS
    p.write_text('{"i":1}\n{"i":2}\n{"i":3')        # the last line still being written
    r = live.read_events(tmp_path, 0)
    assert [e["i"] for e in r["events"]] == [1, 2] and r["offset"] == len('{"i":1}\n{"i":2}\n')
    with open(p, "a") as fh:
        fh.write("}\n")
    r2 = live.read_events(tmp_path, r["offset"])
    assert [e["i"] for e in r2["events"]] == [3] and live.read_events(tmp_path, r2["offset"])["events"] == []
    p.write_text("".join(json.dumps({"i": i, "pad": "x" * 100}) + "\n" for i in range(2000)))
    tail = live.read_events(tmp_path, -1)
    assert tail["events"][-1]["i"] == 1999 and tail["offset"] == p.stat().st_size
    assert 0 < len(tail["events"]) < 2000 and all("pad" in e for e in tail["events"])
    assert live.read_events(tmp_path / "missing", -1) == {"offset": 0, "events": []}


def test_the_server_serves_the_feed_and_the_live_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "FRAME_EVERY_S", 0.0)
    env = CivitasEnv(GameConfig(seed=0, months=12))
    d = tmp_path / "r"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"name": "x", "seed": 0}))
    log = live.LiveLog(d, env)
    log.emit("turn", "planning", agent="scripted")
    log.emit("call", "planning", tool="build_road", args={}, ok=True, action=True)
    httpd = serve(tmp_path, port=0, background=True, allow_launch=False)
    base = f"http://127.0.0.1:{httpd.server_address[1]}/api/runs/r"
    feed = json.loads(urllib.request.urlopen(f"{base}/live?offset=-1").read())
    assert [e["kind"] for e in feed["events"]] == ["turn", "call", "frame"] and feed["status"] == "incomplete"
    frame = json.loads(urllib.request.urlopen(f"{base}/liveframe").read())
    assert frame["live"] == 1 and frame["month"] == 0 and "needs" not in frame and frame["tiles"]
    assert json.loads(urllib.request.urlopen(f"{base}/live?offset={feed['offset']}").read())["events"] == []
    log.month(1)
    assert json.loads(urllib.request.urlopen(f"{base}/liveframe").read()) is None
    v = json.loads(urllib.request.urlopen(base.replace("/runs/r", "/version")).read())["viewer"]
    assert v and v == json.loads(urllib.request.urlopen(base.replace("/runs/r", "/version")).read())["viewer"]
    log.close()
    httpd.shutdown()
