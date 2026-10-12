"""Scripted yardsticks of warm-up runs (bench/yardstick.py): the same game from the same hand-over, played by
scripted agents only, never a model."""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from test_warmup import FakeChat, _fleet

from civitas.agents import llm as llm_mod
from civitas.agents import llm_client
from civitas.agents.base import NoopAgent
from civitas.agents.llm import LLMAgent
from civitas.agents.remote import LiveHub, RemoteAgent
from civitas.agents.scripted import SCRIPTED, ScriptedPlanner
from civitas.bench import runner
from civitas.bench import yardstick as Y
from civitas.config import apply_cli_overrides, build_agents, game_config, load_config, scoring_config
from civitas.core.env import make_env
from civitas.fleet import agent as agent_mod
from civitas.fleet.agent import ToolAgent
from civitas.fleet.config import FleetAgentConfig
from civitas.fleet.scripted import ScriptedFleetAgent

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake(monkeypatch):
    FakeChat.briefings = []
    monkeypatch.setattr(agent_mod, "ChatClient", FakeChat)
    monkeypatch.setattr(llm_client, "ChatClient", FakeChat)
    return FakeChat


class Boom:
    """A ChatClient that must never be built: the yardstick builds no model-backed agent."""

    def __init__(self, *a, **k):
        raise AssertionError("the yardstick built a model client")


def _built(cfg: dict) -> dict:
    ycfg = Y.yardstick_config(cfg)
    game = game_config(ycfg, 0)
    env = make_env(game, scoring_config(ycfg, game))
    return build_agents(ycfg, env.roles, game, env.scoring)


def _lines(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines()]


# ------------------------------------------------------------------ safety
def test_an_llm_line_up_gets_an_all_scripted_yardstick(monkeypatch):
    monkeypatch.setattr(agent_mod, "ChatClient", Boom)
    monkeypatch.setattr(llm_mod, "ChatClient", Boom)
    late = load_config(str(ROOT / "configs/fleet_smoke_late.toml"))                # the trinity fleet, all 8 roles
    late["agents"]["police"] = {"kind": "llm", "model": "trinity-mini"}
    late["agents"]["fire"] = {"kind": "remote"}
    assert len(Y.model_roles(late)) == 8                                            # the run's own line-up
    assert Y.model_roles(Y.yardstick_config(late)) == []
    agents = _built(late)
    Y.check_agents(agents)
    assert {type(a) for r, a in agents.items() if r not in ("police", "fire")} == {ScriptedFleetAgent}
    assert all(a.cfg.model == "scripted" for a in agents.values() if a.kind == "fleet")
    assert type(agents["police"]) is SCRIPTED["police"] and type(agents["fire"]) is SCRIPTED["fire"]
    # a region: plain scripted agents in both cities
    region = _built(load_config(str(ROOT / "configs/fleet_region_late.toml")))
    Y.check_agents(region)
    assert len(region) == 16 and all(type(a) is SCRIPTED[r.split(".")[1]] for r, a in region.items())
    # and a short yardstick of the LLM config plays without building a model client
    cfg = apply_cli_overrides(late, sets=["game.months=9", "game.warmup_months=6"])
    ys = Y.play(cfg, 0)
    assert ys["handed_over"] and set(ys["agents"].values()) <= {"fleet:scripted", "scripted"}


def test_the_check_trips_on_a_model_agent_before_a_month_is_played(monkeypatch):
    monkeypatch.setattr(agent_mod, "ChatClient", FakeChat)
    monkeypatch.setattr(llm_mod, "ChatClient", FakeChat)
    ok = {"planning": ScriptedPlanner("planning"), "mayor": ScriptedFleetAgent("mayor"), "fire": NoopAgent("fire")}
    Y.check_agents(ok)

    class Sneaky(ScriptedPlanner):                       # not the exact class: refused
        pass

    sneaks = {"tool agent": ToolAgent("planning"), "llm": LLMAgent("planning", model="m"),
              "remote": RemoteAgent("planning", hub=LiveHub()), "subclass": Sneaky("planning"),
              "fleet on a model": ScriptedFleetAgent("planning", cfg=FleetAgentConfig(model="trinity"))}
    for name, bad in sneaks.items():
        with pytest.raises(Y.ModelAgentError, match="could call a model"):
            Y.check_agents({**ok, "planning": bad})
    planner = ScriptedPlanner("planning")
    planner.client = object()                            # holds a client: refused
    with pytest.raises(Y.ModelAgentError):
        Y.check_agents({"planning": planner})
    # a model agent sneaking into the built line-up stops play() before any agent binds or a month passes
    cfg = _fleet(12, 6)
    real = Y.build_agents
    envs = []
    monkeypatch.setattr(Y, "make_env", lambda *a, **k: envs.append(make_env(*a, **k)) or envs[-1])
    monkeypatch.setattr(Y, "build_agents", lambda *a, **k: {**real(*a, **k), "transport": ToolAgent("transport")})
    with pytest.raises(Y.ModelAgentError, match="transport: ToolAgent"):
        Y.play(cfg, 0)
    assert envs[-1].world.month == 0 and getattr(envs[-1], "fleet", None) is None
    # and a line-up config that would keep a model agent is refused before anything is built
    monkeypatch.setattr(Y, "warmup_config", lambda c: c)
    with pytest.raises(Y.ModelAgentError, match="would call a model"):
        Y.yardstick_config(cfg)


# ------------------------------------------------------------------ equality
def test_a_scripted_run_is_its_own_yardstick(tmp_path):
    """An all-scripted warm-up run (the scripted fleet before and after the hand-over) and its yardstick are one game:
    the same change, average, final scores and per-month city score and population as the run's frames."""
    cfg = apply_cli_overrides(load_config(str(ROOT / "configs/fleet_scripted.toml")),
                              sets=["game.months=18", "game.warmup_months=9"])
    s = runner.run_episode(cfg, 0, tmp_path, verbose=False, run_id="r")
    ys = json.loads((tmp_path / "r" / "yardstick.json").read_text())
    w = s["warmup"]
    assert ys["matches_run"] and ys["at_handover"] == w["at_handover"] and ys["change"] == w["change"]
    assert ys["avg_city_score"] == w["avg_city_score"] and ys["final_scores"] == pytest.approx(s["final_scores"])
    assert ys["run"]["change"] == w["change"] and ys["seed"] == 0 and ys["months"] == 9
    frames = [f for f in _lines(tmp_path / "r" / "frames.jsonl") if f["month"] >= 9]
    assert ys["series"]["month"] == [f["month"] for f in frames] == list(range(9, 19))
    assert ys["series"]["city"] == pytest.approx([f["scores"]["city"] for f in frames], abs=1e-3)
    assert ys["series"]["population"] == [f["metrics"]["population"] for f in frames]


def test_an_llm_run_shares_the_warmup_and_differs_after(tmp_path, fake):
    """A fake-model fleet run's yardstick: the run's game up to the hand-over (as test_warmup checks for the frames),
    then a plain fleet_scripted game: the yardstick equals a direct all-scripted run of the same [game]."""
    s = runner.run_episode(_fleet(12, 6), 0, tmp_path, verbose=False, run_id="llm")
    ys = json.loads((tmp_path / "llm" / "yardstick.json").read_text())
    assert ys["matches_run"] and ys["at_handover"] == s["warmup"]["at_handover"]
    assert set(ys["agents"].values()) == {"fleet:scripted"} and ys["run"]["change"] == s["warmup"]["change"]
    plain = apply_cli_overrides(load_config(str(ROOT / "configs/fleet_scripted.toml")), sets=["game.months=12"])
    p = runner.run_episode(plain, 0, tmp_path, verbose=False, run_id="plain")
    frames = [f for f in _lines(tmp_path / "plain" / "frames.jsonl") if f["month"] >= 6]
    assert ys["series"]["city"] == pytest.approx([f["scores"]["city"] for f in frames], abs=1e-3)
    assert ys["series"]["population"] == [f["metrics"]["population"] for f in frames]
    assert ys["final_scores"] == pytest.approx(p["final_scores"])
    assert ys["at_end"]["population"] == p["final_metrics"]["population"]
    llm_frames = _lines(tmp_path / "llm" / "frames.jsonl")
    assert llm_frames[-1]["scores"]["city"] != pytest.approx(ys["at_end"]["city"], abs=1e-6)  # the fake played after


def test_a_region_yardstick_per_city(tmp_path, fake):
    cfg = apply_cli_overrides(load_config(str(ROOT / "configs/fleet_region_late.toml")),
                              sets=["game.months=9", "game.warmup_months=6"])
    cfg["providers"]["fake"] = {"base_url": "http://127.0.0.1:9/v1", "api_key_env": ""}
    cfg["fleet"].update(provider="fake", model="fake", max_steps=4)
    s = runner.run_episode(cfg, 0, tmp_path, verbose=False, run_id="r")
    ys = json.loads((tmp_path / "r" / "yardstick.json").read_text())
    assert ys["matches_run"] and set(ys["agents"].values()) == {"scripted"} and len(ys["agents"]) == 16
    assert set(ys["by_city"]) == {"north", "south"}
    assert sum(b["at_end"]["population"] for b in ys["by_city"].values()) == ys["at_end"]["population"]
    assert ys["at_handover"]["population"] == s["warmup"]["at_handover"]["population"]


# ------------------------------------------------------------------ runner, CLI
def test_the_runner_line_and_the_switches(tmp_path, capsys):
    cfg = apply_cli_overrides(load_config(str(ROOT / "configs/fleet_scripted.toml")),
                              sets=["game.months=8", "game.warmup_months=4"])
    runner.run_episode(cfg, 0, tmp_path, run_id="v")
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith("[v] after the warm-up (months 4-7)") and "; scripted from the same start: " in last
    runner.run_episode(cfg, 0, tmp_path, verbose=False, run_id="off", yardstick=False)
    assert not (tmp_path / "off" / "yardstick.json").exists()
    runner.run_benchmark(cfg, [0], tmp_path, verbose=False)                              # bench skips it
    assert [p.parent.name for p in tmp_path.glob("*/yardstick.json")] == ["v"]         # only run "v" has one
    plain = apply_cli_overrides(cfg, sets=["game.warmup_months=0"])
    runner.run_episode(plain, 0, tmp_path, verbose=False, run_id="nowarm")
    assert not (tmp_path / "nowarm" / "yardstick.json").exists()


def test_the_cli_computes_a_runs_yardstick(tmp_path, capsys):
    from civitas.cli import main
    cfg = apply_cli_overrides(load_config(str(ROOT / "configs/fleet_scripted.toml")),
                              sets=["game.months=10", "game.warmup_months=6"])
    s = runner.run_episode(cfg, 1, tmp_path, verbose=False, run_id="late", yardstick=False)
    main(["yardstick", "late", "--runs-dir", str(tmp_path), "--quiet"])
    out = capsys.readouterr().out
    c = s["warmup"]["change"]
    assert "late: months 6-9" in out and f"scripted yardstick:  {c['city']:+.1f} city / {c['population']:+,}" in out
    ys = json.loads((tmp_path / "late" / "yardstick.json").read_text())
    assert ys["seed"] == 1 and ys["matches_run"] and ys["change"] == c
    main(["yardstick", str(tmp_path / "late"), "--dry-run", "--json"])               # a folder works too
    assert json.loads(capsys.readouterr().out)["change"] == c
    runner.run_episode(apply_cli_overrides(cfg, sets=["game.warmup_months=0"]), 0, tmp_path, verbose=False,
                       run_id="early")
    with pytest.raises(SystemExit, match="has no warm-up"):
        main(["yardstick", "early", "--runs-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="no run at"):
        main(["yardstick", "nope", "--runs-dir", str(tmp_path)])


# ------------------------------------------------------------------ server
def _get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_the_yardstick_endpoint_and_compare(tmp_path):
    from civitas.server import serve
    cfg = apply_cli_overrides(load_config(str(ROOT / "configs/fleet_scripted.toml")),
                              sets=["game.months=8", "game.warmup_months=4"])
    runner.run_episode(cfg, 0, tmp_path, verbose=False, run_id="has")
    runner.run_episode(cfg, 0, tmp_path, verbose=False, run_id="lacks", yardstick=False)
    runner.run_episode(apply_cli_overrides(cfg, sets=["game.warmup_months=0"]), 0, tmp_path, verbose=False,
                       run_id="early")
    srv = serve(tmp_path, port=0, background=True)
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}/api"
        code, ys = _get(f"{base}/runs/has/yardstick")
        assert code == 200 and ys == json.loads((tmp_path / "has" / "yardstick.json").read_text())
        assert _get(f"{base}/runs/early/yardstick") == (200, {"missing": True, "reason": "not a warm-up run"})
        assert _get(f"{base}/runs/nope/yardstick")[0] == 404
        # a finished warm-up run without one: computed in the background (a `civitas yardstick` process)
        code, first = _get(f"{base}/runs/lacks/yardstick")
        assert code == 200 and first["missing"] and first["computing"]
        deadline = time.time() + 90
        while time.time() < deadline:
            code, d = _get(f"{base}/runs/lacks/yardstick")
            if not d.get("missing") or not d.get("computing"):
                break
            time.sleep(0.5)
        assert not d.get("missing"), d
        assert d["change"] == ys["change"] and d["matches_run"]
        # the Compare view's runs carry it
        _, cmp = _get(f"{base}/compare?runs=has,early")
        has, early = cmp["runs"]
        assert has["yardstick"]["change"] == ys["change"] and early["yardstick"] is None
        assert has["yardstick"]["series"]["city"] == ys["series"]["city"]
    finally:
        srv.shutdown()
