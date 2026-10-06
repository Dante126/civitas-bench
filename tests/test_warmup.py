"""Warm-ups: scripted agents play the first N months, then the configured agents take the same game over
(bench/runner.py: lineups, warm_up, _hand_over; config.warmup_config / warmup_months)."""
import json
from pathlib import Path

import pytest

from civitas.agents import llm_client
from civitas.bench import runner
from civitas.bench import session as S
from civitas.bench.collab import run_collab
from civitas.bench.problems import run_problems
from civitas.config import apply_cli_overrides, game_config, load_config, warmup_config, warmup_months
from civitas.fleet import agent as agent_mod
from civitas.fleet.memory import FleetState

ROOT = Path(__file__).resolve().parents[1]


def _call(name, args, i=0):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class FakeChat:
    """Stands in for ChatClient (no network): a look-up, then end_turn. Records each turn's first briefing."""
    briefings: list[str] = []

    def __init__(self, provider):
        self.p = provider

    def chat(self, model, messages, **kw):
        step = sum(m["role"] == "assistant" for m in messages)
        if step == 0:
            FakeChat.briefings.append(messages[1]["content"])
        calls = [_call("city_overview", {})] if step == 0 else [_call("end_turn", {"report": f"fake step {step}"}, 1)]
        return {"content": "", "reasoning": "", "tool_calls": calls, "finish_reason": "tool_calls",
                "usage": {"prompt_tokens": 7, "completion_tokens": 3}, "latency_s": 0.0}


@pytest.fixture
def fake(monkeypatch):
    FakeChat.briefings = []
    monkeypatch.setattr(agent_mod, "ChatClient", FakeChat)
    monkeypatch.setattr(llm_client, "ChatClient", FakeChat)          # session.restore_agents' import
    return FakeChat


def _fleet(months=12, warm=6, *sets) -> dict:
    """configs/fleet_scripted.toml's line-up on a fake model (all eight roles), with a warm-up."""
    cfg = apply_cli_overrides(load_config(str(ROOT / "configs/fleet_scripted.toml")),
                              sets=[f"game.months={months}", f"game.warmup_months={warm}", *sets])
    cfg["providers"]["fake"] = {"base_url": "http://127.0.0.1:9/v1", "api_key_env": "", "max_retries": 0}
    cfg["fleet"].update(provider="fake", model="fake", max_steps=6)
    return cfg


def _lines(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines()]


# ------------------------------------------------------------------ config
def test_warmup_months_is_validated():
    cfg = load_config(str(ROOT / "configs/fleet_smoke.toml"))
    assert warmup_months(cfg) == 0 and game_config(cfg).warmup_months == 0
    for bad, msg in ((12, "less than game.months"), (20, "less than game.months"), (-1, "whole number"),
                     (True, "whole number"), (2.5, "whole number")):
        cfg["game"]["warmup_months"] = bad
        with pytest.raises(ValueError, match=msg):
            warmup_months(cfg)
        with pytest.raises(ValueError, match=msg):
            runner.run_episode(cfg, 0, "/nonexistent", verbose=False)
    cfg["game"]["warmup_months"] = 11
    assert warmup_months(cfg) == 11


def test_the_cli_takes_warmup_and_refuses_a_bad_one(tmp_path, capsys):
    from civitas.cli import main
    with pytest.raises(SystemExit, match="warmup_months \\(12\\) must be less than game.months \\(12\\)"):
        main(["run", "-c", "configs/fleet_scripted.toml", "--set", "game.months=12", "--warmup", "12",
              "--runs-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="whole number"):
        main(["bench", "-c", "configs/baseline.toml", "--warmup", "-3", "--runs-dir", str(tmp_path)])


def test_the_warmup_line_up_makes_costly_agents_scripted():
    cfg = load_config(str(ROOT / "configs/fleet_smoke.toml"))
    cfg["game"]["warmup_months"] = 6
    cfg["agents"]["police"] = {"kind": "llm", "model": "m"}
    cfg["agents"]["fire"] = {"kind": "noop"}
    w = warmup_config(cfg)
    assert "warmup_months" not in w["game"] and cfg["game"]["warmup_months"] == 6          # the input is kept
    assert w["agents"]["mayor"]["kind"] == "fleet" and w["agents"]["mayor"]["model"] == "scripted"
    assert w["agents"]["mayor"]["playbook"] == cfg["agents"]["mayor"]["playbook"]          # fleet settings kept
    assert w["agents"]["police"] == {"kind": "scripted"} and w["agents"]["fire"] == {"kind": "noop"}
    assert w["fleet"] == cfg["fleet"]
    # a region: plain scripted agents (the scripted fleet plays single cities only), per city
    reg = load_config(str(ROOT / "configs/fleet_region_late.toml"))
    wr = warmup_config(reg)
    assert all(wr["agents"][f"{c}.{r}"] == {"kind": "scripted"} for c in ("north", "south")
               for r in ("mayor", "planning", "education"))
    # the signature marks a warm-up run (leaderboards keep it apart), the warm-up's own doesn't
    roles = list(game_config(cfg).roles)
    assert runner.agent_signature(cfg, roles).endswith("warmup:6 [autonomous]")
    assert "warmup" not in runner.agent_signature(w, roles)


def test_the_late_configs_keep_the_smoke_fleets():
    smoke, late = load_config(str(ROOT / "configs/fleet_smoke.toml")), load_config(str(ROOT / "configs/fleet_smoke_late.toml"))
    assert late["fleet"] == smoke["fleet"] and late["agents"] == smoke["agents"]
    assert late["providers"] == smoke["providers"] and late["scoring"] == smoke["scoring"]
    g = game_config(late)
    assert (g.months, g.warmup_months, g.act_every) == (108, 96, 3) and warmup_months(late) == 96
    both, rlate = (load_config(str(ROOT / f"configs/{n}.toml")) for n in ("fleet_region_both_120", "fleet_region_late"))
    assert rlate["fleet"] == both["fleet"] and rlate["agents"] == both["agents"]
    g = game_config(rlate)
    assert g.region and (g.months, g.warmup_months) == (72, 60) and g.cities == ["north", "south"]


# ------------------------------------------------------------- determinism
def test_a_warmup_plays_the_scripted_fleet_game_up_to_the_hand_over(tmp_path, fake):
    """Up to month N a warm-up run is a plain fleet_scripted run with the same game settings: the same frames line for
    line (tiles, metrics, scores, turns, council, needs); from N on the fake model plays."""
    plain = apply_cli_overrides(load_config(str(ROOT / "configs/fleet_scripted.toml")), sets=["game.months=12"])
    runner.run_episode(plain, 0, tmp_path, verbose=False, run_id="plain")
    runner.run_episode(_fleet(12, 6), 0, tmp_path, verbose=False, run_id="warm")
    a = (tmp_path / "plain" / "frames.jsonl").read_text().splitlines()
    b = (tmp_path / "warm" / "frames.jsonl").read_text().splitlines()
    assert len(a) == len(b) == 13
    assert a[:7] == b[:7]                          # frames 0-6: month 6's frame carries the warm-up's last turns
    assert a[7] != b[7]                            # month 6's turns were the fake model's


# ------------------------------------------------------------------ hand-over
def test_the_configured_agents_take_over_with_the_fleet_state(tmp_path, fake, monkeypatch):
    states = []
    monkeypatch.setattr(runner, "settle_fleet", lambda agents, warm=None: states.append(
        {r: (a.env.fleet.plan_versions[:], dict(a.env.fleet.last_turn_month)) for r, a in agents.items()}))
    s = runner.run_episode(_fleet(12, 6), 0, tmp_path, verbose=False, run_id="r")
    d = tmp_path / "r"
    traj = _lines(d / "trajectories.jsonl")
    warm, after = [t for t in traj if t.get("warmup")], [t for t in traj if not t.get("warmup")]
    assert {t["month"] for t in warm} == {0, 3} and {t["month"] for t in after} == {6, 9}
    assert {t["agent"] for t in warm} == {"fleet:scripted"} and {t["agent"] for t in after} == {"fleet:fake"}
    assert all(t["tool_trace"][0]["tool"] == "city_overview" for t in after)              # the fake model played
    # the fake's briefings: its first at month 6, with the plan, memory and digest the scripted fleet left
    assert len(fake.briefings) == 16 and fake.briefings[0].startswith("# Mayor: month 6 of 12")
    assert "Since your last turn (month 3)" in fake.briefings[0] and "City plan (v" in fake.briefings[0]
    st, = states                                    # the hand-over found the warm-up's plan and turns
    assert st["mayor"][0] and all(lt["mayor"] == 3 for _, lt in st.values())
    # meta, summary, report
    meta = json.loads((d / "meta.json").read_text())
    assert meta["warmup_months"] == 6 and meta["warmup"]["months"] == 6
    assert set(meta["warmup"]["agents"].values()) == {"fleet:scripted"} and set(meta["agents"].values()) == {"fleet:fake"}
    assert meta["signature"].endswith("warmup:6 [autonomous]") and "warmup" not in meta["warmup"]["signature"]
    w = s["warmup"]
    assert w["handed_over"] and w["at_handover"]["month"] == 6 and w["at_end"]["month"] == 12
    assert w["change"]["city"] == pytest.approx(s["final_scores"]["city"] - w["at_handover"]["city"], abs=1e-3)
    assert w["change"]["population"] == s["final_metrics"]["population"] - w["at_handover"]["population"]
    assert w["at_handover"]["city"] == pytest.approx(
        next(f for f in _lines(d / "frames.jsonl") if f["month"] == 6)["scores"]["city"])
    assert all(v["turns"] == 2 for v in s["agent_stats"].values())                        # months 6 and 9 only
    assert all(v["turns"] == 2 for v in w["agent_stats"].values())                        # months 0 and 3
    rep = json.loads((d / "fleet_report.json").read_text())
    assert all(r["turns"] == 2 for r in rep["roles"].values()) and rep["warmup"]["months"] == 6
    md = (d / "fleet_report.md").read_text()
    assert "Warm-up: months 0-5 played by `mayor=fleet:scripted" in md and "cover months 6-11 only" in md
    # the Team tab and the collaboration score count months 6 on; problems skip the warm-up's turns
    c = run_collab(d)
    assert c["since"] == 6 and all(e["month"] >= 6 for e in c["events"])
    assert all(p["month"] >= 6 for p in run_problems(d / "trajectories.jsonl")["problems"])


def test_problems_and_report_rows_skip_warmup_turns(tmp_path):
    from civitas.fleet.report import rows_from_trajectories
    bad = {"action": {"type": "build_road"}, "ok": False, "message": "no money"}
    recs = [{"month": 0, "role": "planning", "agent": "fleet:scripted", "warmup": True, "results": [bad]},
            {"month": 3, "role": "planning", "agent": "fleet:fake", "results": [bad]}]
    p = tmp_path / "trajectories.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = run_problems(p)
    assert [x["month"] for x in out["problems"]] == [3] and out["roles"]["planning"]["turns"] == 1
    assert [r["month"] for r in rows_from_trajectories(p)] == [3]


def test_a_supervised_warmup_run_pauses_after_the_warmup(tmp_path, fake, monkeypatch):
    monkeypatch.setenv(S.KEY_DIR_ENV, str(tmp_path / "key"))
    cfg = _fleet(12, 6)
    cfg["fleet"].update(mode="supervised", session_months=3)
    out = runner.run_episode(cfg, 0, tmp_path, verbose=False, run_id="r")
    assert out["paused"] and out["month"] == 9                                            # 3 months after the warm-up
    s, = json.loads((tmp_path / "r" / "meta.json").read_text())["sessions"]
    assert (s["start_month"], s["end_month"]) == (6, 9)
    done = runner.resume_episode("r", runs_dir=tmp_path, verbose=False)
    assert done["months_played"] == 12 and done["warmup"]["at_handover"]["month"] == 6
    assert all(v["turns"] == 2 for v in done["agent_stats"].values())


# ------------------------------------------------------------------ region
def test_a_region_warmup_hands_each_city_over(tmp_path, fake):
    cfg = apply_cli_overrides(load_config(str(ROOT / "configs/fleet_region_late.toml")),
                              sets=["game.months=9", "game.warmup_months=6"])
    cfg["providers"]["fake"] = {"base_url": "http://127.0.0.1:9/v1", "api_key_env": ""}
    cfg["fleet"].update(provider="fake", model="fake", max_steps=4)
    envs = []
    real = runner.make_env
    runner.make_env = lambda *a, **k: envs.append(real(*a, **k)) or envs[-1]
    try:
        s = runner.run_episode(cfg, 0, tmp_path, verbose=False, run_id="r")
    finally:
        runner.make_env = real
    traj = _lines(tmp_path / "r" / "trajectories.jsonl")
    assert {t["agent"] for t in traj if t.get("warmup")} == {"scripted"}
    assert {t["agent"] for t in traj if not t.get("warmup")} == {"fleet:fake"} and {t["month"] for t in traj
                                                                                     if not t.get("warmup")} == {6}
    region = envs[-1]
    for c, env in region.envs.items():
        fleet = FleetState.of(env)
        assert fleet.plan_versions and fleet.plan_versions[0]["month"] == 3               # the warm-up mayor's plan
        assert fleet.last_turn_month["mayor"] == 6 and any(e["kind"] == "metrics" and e["month"] == 1 for e in fleet.log)
    mayor = next(b for b in FakeChat.briefings if b.startswith("# Mayor of NORTH"))
    assert "Since your last turn (month 3)" in mayor and "City plan (v1, month 3" in mayor
    assert s["warmup"]["handed_over"] and "Warm-up: months 0-5" in (tmp_path / "r" / "fleet_report.md").read_text()


# ------------------------------------------------------------------ prompt
def test_prompt_shows_the_first_turn_after_the_warmup(capsys):
    from civitas.cli import main
    main(["prompt", "-c", "configs/fleet_smoke_late.toml", "--role", "transport", "--set", "game.months=12",
          "--warmup", "6"])
    out = capsys.readouterr().out
    assert "after a warm-up of 6 months" in out and "# Transport Department: month 6 of 12" in out
    assert "Since your last turn (month 3)" in out and "build_train_station" in out
    main(["prompt", "-c", "configs/fleet_smoke_late.toml", "--role", "transport", "--set", "game.months=12",
          "--warmup", "0"])
    assert "month 0 of 12" in capsys.readouterr().out
