"""Supervised sessions: a fleet run pauses at a session boundary, the coach edits the playbooks, it resumes."""
import contextlib
import gzip
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from civitas.bench import runner
from civitas.bench import session as S
from civitas.config import load_config
from civitas.core.roles import ROLES

ROOT = Path(__file__).resolve().parents[1]
FLEET = ("utilities", "police")
KEY_ENV, KEY = "CIVITAS_FAKE_KEY", "sk-fake-never-in-a-checkpoint"


def _call(name, args, i):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _role(body) -> str:
    """The role whose action tools the request carries (the most action tools matched)."""
    names = {t["function"]["name"] for t in body.get("tools") or []}
    return max(ROLES, key=lambda r: sum(a in names for a in ROLES[r]["actions"]))


def _script(role: str, messages: list[dict]) -> list[dict]:
    """Deterministic tool calls that depend on the game: find a site, build on the first one, end the turn."""
    step = sum(m["role"] == "assistant" for m in messages)
    purpose, action, kind = {"utilities": ("wind", "build_power_plant", "wind"),
                             "police": ("station", "build_police_station", None)}[role]
    if step == 0:
        return [_call("find_sites", {"purpose": purpose}, 0)]
    if step == 1:
        found = json.loads([m for m in messages if m["role"] == "tool"][-1]["content"])
        sites = found.get("sites") or []
        if sites:
            args = {"x": sites[0]["x"], "y": sites[0]["y"], **({"kind": kind} if kind else {})}
            return [_call(action, args, 1)]
    return [_call("end_turn", {"report": f"{role} step {step}"}, 2)]


class FakeLLM(BaseHTTPRequestHandler):
    requests: list[tuple[str, dict]] = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        role = _role(body)
        FakeLLM.requests.append((role, body))
        calls = _script(role, body["messages"])
        msg = {"role": "assistant", "content": "", **({"tool_calls": calls} if calls else {})}
        out = json.dumps({"choices": [{"message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                          "usage": {"prompt_tokens": 50, "completion_tokens": 10}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def port(monkeypatch):
    monkeypatch.setenv(KEY_ENV, KEY)
    FakeLLM.requests = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        yield srv.server_port
    finally:
        srv.shutdown()


def _cfg(port: int, *, months: int = 12, session: int = 6, mode: str = "supervised") -> dict:
    """Fleet utilities and police on the fake model, scripted mayor, planner and the rest (so the city grows)."""
    cfg = load_config(None)
    cfg["name"] = "sessions"
    cfg["game"].update(months=months, act_every=2)
    cfg["providers"]["fake"] = {"base_url": f"http://127.0.0.1:{port}/v1", "api_key_env": KEY_ENV, "stream": False,
                                "max_retries": 0}
    cfg["fleet"] = {"mode": mode, "provider": "fake", "model": "fake", "max_steps": 6, "session_months": session,
                    "playbook": "Grow steadily."}
    cfg["agents"] = {"utilities": {"kind": "fleet", "playbook": "Power first."},
                     "police": {"kind": "fleet", "playbook": "- Cover the homes.\n- Ask the mayor for money.\n"}}
    return cfg


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def _system(role: str) -> list[str]:
    return [b["messages"][0]["content"] for r, b in FakeLLM.requests if r == role and len(b["messages"]) == 2]


# ------------------------------------------------------------------ pause
def test_a_supervised_run_pauses_at_the_session_boundary(tmp_path, port):
    out = runner.run_episode(_cfg(port), 0, tmp_path, verbose=False, run_id="r")
    d = tmp_path / "r"
    assert out["paused"] and (out["session"], out["month"]) == (1, 6)
    assert out["resume"] == f"civitas resume r --runs-dir {tmp_path}"
    assert not (d / "summary.json").exists() and not (d / "fleet_report.md").exists()
    assert json.loads((d / S.PAUSED).read_text()) == out
    frames = _lines(d / "frames.jsonl")
    assert [f["month"] for f in frames] == list(range(7))          # month 0 and the 6 months played
    traj = _lines(d / "trajectories.jsonl")
    assert sorted({t["month"] for t in traj}) == [0, 2, 4] and {t["session"] for t in traj} == {1}
    assert any(r["ok"] for t in traj if t["role"] == "utilities" for r in t["results"])     # the fake built
    s, = json.loads((d / "meta.json").read_text())["sessions"]
    assert (s["session"], s["start_month"], s["end_month"], s["status"], s["session_months"]) == (1, 0, 6, "paused", 6)
    assert set(s["playbooks"]) == {"fleet", *FLEET} and s["fleet_version"]

    state = S.load_checkpoint(d)
    env = state["env"]
    assert env.world.month == 6 and len(env.world.history) == 6 and not env.done
    assert state["month"] == 6 and env.fleet.log and env.fleet.last_turn_month == {"utilities": 4, "police": 4}
    assert set(state["agents"]) == set(env.roles) and state["agents"]["police"].client is None
    assert state["agents"]["police"].env is env                    # one pickle: shared references stay shared
    assert state["agents"]["planning"].segments                    # scripted agents keep their state
    assert state["stats"]["police"]["turns"] == 3 and len(state["city_scores"]) == 6
    assert len(state["fleet_rows"]) == 6
    assert state["log_offsets"] == {n: (d / n).stat().st_size for n in S.LOGS}
    assert KEY.encode() not in gzip.decompress((d / S.CHECKPOINT).read_bytes())   # no API key on disk

    pb = tomllib.loads((d / S.PLAYBOOKS).read_text())
    assert pb["fleet"] == {"session_months": 6, "playbook": "Grow steadily."}
    assert pb["agents"] == {r: {"playbook": _cfg(0)["agents"][r]["playbook"]} for r in FLEET}


def test_autonomous_runs_ignore_session_months(tmp_path, port, capsys):
    s = runner.run_episode(_cfg(port, mode="autonomous", session=4), 0, tmp_path, run_id="a")
    d = tmp_path / "a"
    assert "paused" not in s and s["months_played"] == 12 and (d / "summary.json").exists()
    assert not any((d / n).exists() for n in (S.CHECKPOINT, S.PAUSED, S.PLAYBOOKS))
    assert "sessions" not in json.loads((d / "meta.json").read_text()) and "sessions" not in s
    assert all("session" not in t for t in _lines(d / "trajectories.jsonl"))
    assert "only pauses supervised runs" in capsys.readouterr().out


def test_bench_plays_supervised_runs_through(tmp_path, port):
    agg = runner.run_benchmark(_cfg(port, months=4, session=2), [0], tmp_path, verbose=False)
    assert agg["n"] == 1 and not list(tmp_path.glob(f"*/{S.CHECKPOINT}"))


# ----------------------------------------------------------------- resume
def test_resume_plays_each_session_and_finishes(tmp_path, port):
    assert runner.run_episode(_cfg(port, session=4), 0, tmp_path, verbose=False, run_id="r")["month"] == 4
    assert runner.resume_episode("r", runs_dir=tmp_path, verbose=False)["month"] == 8
    s = runner.resume_episode(tmp_path / "r", verbose=False)                 # the run's folder works too
    d = tmp_path / "r"
    assert s["months_played"] == 12 and s["sessions"] == 3 and (d / "summary.json").exists()
    assert not (d / S.CHECKPOINT).exists() and not (d / S.PAUSED).exists()
    assert [f["month"] for f in _lines(d / "frames.jsonl")] == list(range(13))
    traj = _lines(d / "trajectories.jsonl")
    assert [(t["month"], t["session"]) for t in traj if t["role"] == "police"] == \
        [(0, 1), (2, 1), (4, 2), (6, 2), (8, 3), (10, 3)]
    meta = json.loads((d / "meta.json").read_text())
    assert [(x["start_month"], x["end_month"], x["status"]) for x in meta["sessions"]] == \
        [(0, 4, "paused"), (4, 8, "paused"), (8, 12, "finished")]
    assert all(x["changed"] == [] and "fleet_prompts" not in x for x in meta["sessions"][1:])
    assert s["fleet_version"] == meta["fleet_version"] and "fleet_versions" not in s
    assert s["agent_stats"]["police"]["turns"] == 6 and s["agent_stats"]["police"]["llm_usage"]["calls"] >= 6
    assert (d / "fleet_report.md").exists() and "12 of 12 months played" in (d / "fleet_report.md").read_text()
    with pytest.raises(ValueError, match="finished"):
        runner.resume_episode("r", runs_dir=tmp_path, verbose=False)


def test_an_edited_playbook_is_in_the_next_system_prompt(tmp_path, port):
    runner.run_episode(_cfg(port), 0, tmp_path, verbose=False, run_id="r")
    d = tmp_path / "r"
    before = {r: _system(r) for r in FLEET}
    assert all("Cover the homes." in p for p in before["police"])
    pb = d / S.PLAYBOOKS
    pb.write_text(pb.read_text().replace("Cover the homes.", "Build by the school."))
    FakeLLM.requests = []
    s = runner.resume_episode("r", runs_dir=tmp_path, verbose=False)
    police, util = _system("police"), _system("utilities")
    assert police and all("Build by the school." in p and "Cover the homes." not in p for p in police)
    assert util and set(util) == set(before["utilities"])                 # an unchanged playbook, the same prompt
    meta = json.loads((d / "meta.json").read_text())
    s1, s2 = meta["sessions"]
    assert s2["changed"] == ["police"] and s2["fleet_prompts"]["police"] == police[0]
    assert s2["config_file"] == str(pb) and s2["fleet_version"] != s1["fleet_version"]
    assert s2["playbooks"]["police"] != s1["playbooks"]["police"] and s2["playbooks"]["fleet"] == s1["playbooks"]["fleet"]
    assert "Build by the school." in s2["fleet_config"]["agents"]["police"]["playbook"]
    assert meta["fleet_prompts"]["police"] == before["police"][0]          # session 1's prompt stays where it was
    refs = {(t["role"], t["session"]): t["prompt"][0]["content"] for t in _lines(d / "trajectories.jsonl")
            if t["role"] in FLEET}
    assert refs[("police", 1)] == "(system prompt: meta.json fleet_prompts.police)"
    assert refs[("police", 2)] == "(system prompt: meta.json sessions[1].fleet_prompts.police)"
    assert refs[("utilities", 2)] == "(system prompt: meta.json fleet_prompts.utilities)"
    assert s["fleet_versions"] == [s1["fleet_version"], s2["fleet_version"]]
    assert s["fleet_version"] not in s["fleet_versions"]


def test_resume_with_an_edited_config_and_overrides(tmp_path, port):
    runner.run_episode(_cfg(port), 0, tmp_path, verbose=False, run_id="r")
    edited = tmp_path / "edited.toml"                                     # a full config copy, as a coach edits it
    edited.write_text('name = "other"\n[game]\nmonths = 12\nact_every = 2\n[fleet]\nmode = "supervised"\n'
                      '[agents.police]\nkind = "fleet"\nplaybook = "Only the -c file."\n')
    FakeLLM.requests = []
    s = runner.resume_episode("r", edited, runs_dir=tmp_path, verbose=False, sets=["fleet.max_steps=2"])
    assert all("Only the -c file." in p for p in _system("police")) and s["name"] == "sessions"
    later = [t for t in _lines(tmp_path / "r" / "trajectories.jsonl") if t["role"] in FLEET and t["month"] >= 6]
    assert later and all(len(t["tool_trace"]) <= 2 for t in later)          # --set applies on top of -c
    assert "max_steps" in {t["finish_reason"] for t in later}


def test_resume_refuses_a_changed_game_line_up_or_mode(tmp_path, port):
    runner.run_episode(_cfg(port), 0, tmp_path, verbose=False, run_id="r")
    bad = tmp_path / "bad.toml"
    for text, match in (("[game]\nmonths = 24\n", r"\[game\] months: 12 -> 24"),
                        ("[scoring]\ntarget_population = 5\n", r"\[scoring\] target_population"),
                        ('[agents.police]\nkind = "scripted"\n', r"\[agents\] police: 'fleet' -> 'scripted'"),
                        ('[fleet]\nmode = "autonomous"\n', r"mode: 'supervised' -> 'autonomous'"),
                        ("[fleet]\nsession_months = -1\n", "session_months")):
        bad.write_text(text)
        with pytest.raises(ValueError, match=match):
            runner.resume_episode("r", bad, runs_dir=tmp_path, verbose=False)
    with pytest.raises(ValueError, match="months"):
        runner.resume_episode("r", runs_dir=tmp_path, verbose=False, sets=["game.months=24"])
    with pytest.raises(ValueError, match="no config file"):
        runner.resume_episode("r", tmp_path / "nope.toml", runs_dir=tmp_path, verbose=False)
    assert (tmp_path / "r" / S.PAUSED).exists()                          # refusals leave the pause as it was
    assert runner.resume_episode("r", runs_dir=tmp_path, verbose=False)["months_played"] == 12
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "meta.json").write_text("{}")
    with pytest.raises(ValueError, match="no checkpoint"):
        runner.resume_episode("a", runs_dir=tmp_path, verbose=False)
    with pytest.raises(ValueError, match="no run at"):
        runner.resume_episode("missing", runs_dir=tmp_path, verbose=False)


def test_resume_drops_log_lines_of_an_interrupted_session(tmp_path, port, capsys):
    runner.run_episode(_cfg(port), 0, tmp_path, verbose=False, run_id="r")
    d = tmp_path / "r"
    with open(d / "frames.jsonl", "a") as f:                              # a session that died after one month
        f.write('{"month": 7}\n')
    s = runner.resume_episode("r", runs_dir=tmp_path, verbose=False)
    assert [f["month"] for f in _lines(d / "frames.jsonl")] == list(range(13)) and s["months_played"] == 12
    assert "frames.jsonl: dropped lines logged after the checkpoint" in capsys.readouterr().err


# ------------------------------------------------------------ determinism
def _played(d: Path) -> dict:
    s = json.loads((d / "summary.json").read_text())
    keep = ("final_scores", "avg_city_score", "final_metrics", "months_played", "bankrupt", "fleet_version")
    traj = [{k: t.get(k) for k in ("month", "role", "actions", "results", "messages", "thinking", "reward",
                                   "scores_after", "tool_trace")} for t in _lines(d / "trajectories.jsonl")]
    stats = {r: {k: v for k, v in st.items() if k != "llm_usage"} for r, st in s["agent_stats"].items()}
    return {"frames": _lines(d / "frames.jsonl"), "traj": traj, "stats": stats, **{k: s.get(k) for k in keep}}


@pytest.mark.parametrize("session, pauses", [(6, 1), (5, 2)])     # 5: boundaries between turns (act_every 2)
def test_a_paused_and_resumed_run_plays_like_an_uninterrupted_one(tmp_path, port, session, pauses):
    """Each resume runs in a fresh process, so nothing the game needs may live outside the checkpoint."""
    runner.run_episode(_cfg(port, session=0), 0, tmp_path, verbose=False, run_id="whole")
    assert runner.run_episode(_cfg(port, session=session), 0, tmp_path, verbose=False, run_id="split")["paused"]
    for _ in range(pauses):
        res = subprocess.run([sys.executable, "-m", "civitas.cli", "resume", "split", "--runs-dir", str(tmp_path),
                              "--quiet"], cwd=ROOT, capture_output=True, text=True, timeout=300)
        assert res.returncode == 0, res.stderr
    assert json.loads((tmp_path / "split" / "summary.json").read_text())["sessions"] == pauses + 1
    whole, split = _played(tmp_path / "whole"), _played(tmp_path / "split")
    assert len(whole["frames"]) == 13 and whole["final_metrics"]["population"] > 0
    for k in whole:
        assert split[k] == whole[k], k


# -------------------------------------------------------------- helpers / CLI
@pytest.mark.parametrize("text", ["", "one line", 'it\'s "quoted" \\ with a backslash', "- a\n- b\n", "ends in '\n",
                                  "last char '", "has ''' in it\n", "\nleading newline", "tab\tand \x01 control\n",
                                  "emoji \U0001f600\nand \x7f del", 'triple """ quotes\n'])
def test_playbook_strings_read_back_exactly(text):
    assert tomllib.loads(f"x = {S.toml_string(text)}")["x"] == text


def test_overlay_merges_tables_and_role_entries():
    base = {"name": "run", "game": {"months": 24, "balance": {"A": 1, "B": 2}},
            "fleet": {"mode": "supervised", "extra_body": {"top_p": 0.9, "seed": 1}},
            "agents": {"mayor": {"kind": "fleet", "playbook": "old"}, "police": {"kind": "scripted"}},
            "providers": {"arcee": {"timeout": 300}}}
    out = S.overlay(base, {"name": "x", "game": {"balance": {"A": 5}}, "fleet": {"extra_body": {"top_k": 5}},
                           "agents": {"mayor": {"playbook": "new"}, "health": {"kind": "fleet"}},
                           "providers": {"arcee": {"max_retries": 1}}})
    assert out["name"] == "run" and out["game"] == {"months": 24, "balance": {"A": 5}}
    assert out["fleet"] == {"mode": "supervised", "extra_body": {"top_k": 5}}
    assert out["agents"] == {"mayor": {"kind": "fleet", "playbook": "new"}, "police": {"kind": "scripted"},
                             "health": {"kind": "fleet"}}
    assert out["providers"] == {"arcee": {"timeout": 300, "max_retries": 1}}
    assert base["agents"]["mayor"]["playbook"] == "old"                   # the base is not touched


def test_session_months_is_validated_and_left_out_of_the_fleet_version():
    from civitas.fleet.config import fleet_version, session_months
    cfg = _cfg(1, session=0)
    assert session_months(cfg) == 0 and session_months(load_config(None)) == 0
    v = fleet_version(cfg)
    cfg["fleet"]["session_months"] = 12
    assert session_months(cfg) == 12 and fleet_version(cfg) == v
    for bad in (-1, 1.5, "12", True):
        cfg["fleet"]["session_months"] = bad
        with pytest.raises(ValueError, match="session_months"):
            session_months(cfg)


def test_the_server_lists_paused_runs_and_resumes_them(tmp_path, port):
    import urllib.error
    import urllib.request
    from civitas.server import serve
    runner.run_episode(_cfg(port), 0, tmp_path, verbose=False, run_id="r")
    httpd = serve(tmp_path, 0, background=True, configs_dir=ROOT / "configs")
    base = f"http://127.0.0.1:{httpd.server_address[1]}/api"

    def status():
        return {r["run_id"]: r["status"] for r in json.loads(urllib.request.urlopen(f"{base}/runs").read())}

    def resume(run_id):
        req = urllib.request.Request(f"{base}/resume", data=json.dumps({"run_id": run_id}).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())
    try:
        assert status() == {"r": "paused"}
        for bad in ("../r", "missing"):
            with pytest.raises(urllib.error.HTTPError) as e:
                resume(bad)
            assert e.value.code == 400 and "no paused run" in json.loads(e.value.read())["error"]
        assert resume("r")["ok"]
        assert httpd.launcher.procs["r"]["proc"].wait(timeout=300) == 0, (tmp_path / "_logs" / "r.log").read_text()
        assert status() == {"r": "finished"}
    finally:
        httpd.shutdown()


def test_cli_resume(tmp_path, port, capsys):
    from civitas.cli import main
    out = runner.run_episode(_cfg(port), 0, tmp_path, verbose=False, run_id="r")
    printed = capsys.readouterr().out
    assert "paused at month 6 after session 1" in printed and out["resume"] in printed and S.PLAYBOOKS in printed
    main(["resume", "r", "--runs-dir", str(tmp_path)])
    printed = capsys.readouterr().out
    assert "resumed at month 6, session 2 (new system prompts: none)" in printed and "done: city" in printed
    with pytest.raises(SystemExit, match="civitas resume: run r has finished"):
        main(["resume", "r", "--runs-dir", str(tmp_path)])
