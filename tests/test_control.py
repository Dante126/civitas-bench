"""The viewer's Pause / Resume / Stop (bench/control.py) and the New run dialog's launches (server.Launcher.compose)."""
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from civitas.bench import control
from civitas.bench.runner import run_episode
from civitas.config import _coerce, apply_cli_overrides, game_config, load_config
from civitas.core.roles import TURN_ORDER
from civitas.server import AGENT_SPEC, LAUNCH_BASE, Launcher, serve

ROOT = Path(__file__).resolve().parents[1]


def _cfg(months=4):
    return apply_cli_overrides(load_config(ROOT / "configs" / "baseline.toml"), sets=[f"game.months={months}"])


def _wait_for(cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_a_held_game_waits_before_its_next_turn_and_plays_on_the_same(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "HEARTBEAT_S", 0.02)
    ref = run_episode(_cfg(), 0, tmp_path / "ref", verbose=False)
    d = tmp_path / "runs" / "held"
    d.mkdir(parents=True)
    control.pause(d)                                   # asked before the first turn
    out = {}
    t = threading.Thread(target=lambda: out.update(run_episode(_cfg(), 0, tmp_path / "runs", verbose=False,
                                                               run_id="held")))
    t.start()
    assert _wait_for(lambda: control.held(d) is not None)
    h = control.held(d)
    assert h["month"] == 0 and h["role"] == "mayor"
    time.sleep(0.1)
    assert (d / "trajectories.jsonl").read_text() == "" and not (d / "summary.json").exists()
    assert control.release(d)
    t.join(60)
    assert not t.is_alive() and out["final_scores"] == ref["final_scores"]
    assert not any((d / m).exists() for m in (control.HOLD, control.HELD, control.STOP))
    assert out["wall_time_s"] < 5                     # the wait isn't playing time


def test_stop_ends_the_game_at_the_next_turn_without_a_summary(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    control.stop(d)
    out = run_episode(_cfg(), 0, tmp_path, verbose=False, run_id="s")
    assert out == {"run_id": "s", "stopped": True, "month": 0}
    assert (d / control.STOPPED).exists() and not (d / control.STOP).exists()
    assert not (d / "summary.json").exists() and (d / "frames.jsonl").read_text().count("\n") == 1


def test_a_stale_held_marker_is_not_a_held_game(tmp_path):
    (tmp_path / control.HELD).write_text(json.dumps({"month": 3, "role": "fire", "since": 0}))
    assert control.held(tmp_path)["role"] == "fire"
    old = time.time() - control.STALE_S - 5
    os.utime(tmp_path / control.HELD, (old, old))
    assert control.held(tmp_path) is None              # its process died while paused


def test_set_takes_json_lists():
    assert _coerce('["north","south"]') == ["north", "south"] and _coerce("[x") == "[x" and _coerce("") == ""
    cfg = apply_cli_overrides(load_config(ROOT / "configs" / "baseline.toml"),
                              sets=["game.region=true", 'game.cities=["north","south","east"]'])
    assert game_config(cfg).cities == ["north", "south", "east"]


def _args(cmd):
    """--set / --agent values of a launch command."""
    out = {"set": [], "agent": []}
    for flag, val in zip(cmd, cmd[1:]):
        if flag in ("--set", "--agent"):
            out[flag[2:]].append(val)
    return out


def test_the_new_run_dialog_composes_map_length_and_team(tmp_path):
    launcher = Launcher(tmp_path, ROOT / "configs")
    o = launcher.options()
    assert o["base"] == LAUNCH_BASE and set(o["maps"]) == {"city", "two", "three"} and o["roles"] == TURN_ORDER
    assert o["fleet"]["model"] in o["models"] and o["act_every"] == 3
    team = {r: "fleet:arcee/trinity-mini" for r in TURN_ORDER}
    run_id, cmd = launcher.compose({"map": "three", "seed": 2, "months": 12, "act_every": 3, "agents": team})
    a = _args(cmd)
    assert run_id.endswith("_region3_trinity-mini_s2_ui") and cmd[cmd.index("-c") + 1].endswith(LAUNCH_BASE)
    assert "game.region=true" in a["set"] and 'game.cities=["north","south","east"]' in a["set"]
    assert "game.months=12" in a["set"] and len(a["agent"]) == 8 and all(AGENT_SPEC.match(v.split("=")[1]) for v in a["agent"])
    # what `civitas run` makes of it: a three-city region of trinity-mini fleets with the base config's playbooks
    cfg = load_config(ROOT / "configs" / LAUNCH_BASE)
    cfg = apply_cli_overrides(cfg, a["agent"], a["set"])
    assert cfg["name"] == "region3_trinity-mini" and game_config(cfg).cities == ["north", "south", "east"]
    assert cfg["agents"]["mayor"]["model"] == "trinity-mini" and cfg["agents"]["mayor"]["playbook"].strip()
    run_id, cmd = launcher.compose({"map": "city", "months": 24, "act_every": 1,
                                    "agents": {r: "scripted" for r in TURN_ORDER}})
    assert "_city_scripted_s0_ui" in run_id and not any("region" in v for v in _args(cmd)["set"])
    mixed = {**{r: "scripted" for r in TURN_ORDER}, "planning": "llm:arcee/trinity-large-thinking"}
    assert "_city_mixed_" in launcher.compose({"map": "city", "agents": mixed})[0]


@pytest.mark.parametrize("body", [
    {"map": "moon", "agents": {r: "scripted" for r in TURN_ORDER}},
    {"map": "city", "agents": {"mayor": "scripted"}},                                   # every role named
    {"map": "city", "agents": {**{r: "scripted" for r in TURN_ORDER}, "fire": "fleet:x; rm -rf /"}},
    {"map": "city", "months": 0, "agents": {r: "scripted" for r in TURN_ORDER}},
])
def test_the_new_run_dialog_refuses_bad_launches(tmp_path, body):
    with pytest.raises(ValueError):
        Launcher(tmp_path, ROOT / "configs").launch(body)


def test_server_pause_resume_stop_and_statuses(tmp_path):
    httpd = serve(tmp_path, port=0, background=True, configs_dir=str(ROOT / "configs"))
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(req).read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())

    def status(rid):
        return {r["run_id"]: r["status"] for r in json.loads(urllib.request.urlopen(f"{base}/api/runs").read())}[rid]

    d = tmp_path / "r"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"name": "x", "seed": 0}))
    (d / "frames.jsonl").write_text("{}\n")             # fresh frames, no process of this server: running
    assert status("r") == "running"
    assert post("/api/pause", {"run_id": "r"})["ok"] and status("r") == "pausing"
    (d / control.HELD).write_text(json.dumps({"month": 6, "role": "planning", "since": time.time()}))
    assert status("r") == "held"
    frames = json.loads(urllib.request.urlopen(f"{base}/api/runs/r/frames?since=0").read())
    assert frames["held"]["role"] == "planning" and frames["control"]
    assert post("/api/resume", {"run_id": "r"})["ok"] and not (d / control.HOLD).exists()
    (d / control.HELD).unlink()
    assert post("/api/stop", {"run_id": "r"})["ok"] and (d / control.STOP).exists()   # not ours: at its next turn
    for bad in ("..", "missing"):
        assert not post("/api/pause", {"run_id": bad})["ok"] and not post("/api/stop", {"run_id": bad})["ok"]
    (d / "summary.json").write_text("{}")
    assert not post("/api/pause", {"run_id": "r"})["ok"] and status("r") == "finished"
    opts = json.loads(urllib.request.urlopen(f"{base}/api/launch").read())
    assert opts["roles"] == TURN_ORDER
    httpd.shutdown()


def test_the_dialog_launches_a_game_that_runs_and_can_be_stopped(tmp_path):
    launcher = Launcher(tmp_path, ROOT / "configs")
    r = launcher.launch({"map": "city", "seed": 0, "months": 600, "act_every": 1,
                         "agents": {r: "scripted" for r in TURN_ORDER}})
    d = tmp_path / r["run_id"]
    assert _wait_for(lambda: (d / "frames.jsonl").exists() and (d / "frames.jsonl").read_text().count("\n") >= 3, 60), \
        (tmp_path / "_logs" / f"{r['run_id']}.log").read_text()
    meta = json.loads((d / "meta.json").read_text())
    assert meta["config"]["game"]["act_every"] == 1 and set(meta["agents"].values()) == {"scripted"}
    assert launcher.stop(r["run_id"]) and (d / control.STOPPED).exists()
    launcher.procs[r["run_id"]]["proc"].wait(30)
