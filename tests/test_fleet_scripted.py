"""The scripted fleet: the scripted baselines playing through the fleet tools (civitas/fleet/scripted.py)."""
import json
import re

import pytest

from civitas.agents import scripted
from civitas.agents.scripted import SCRIPTED
from civitas.bench import runner
from civitas.config import apply_cli_overrides, build_agents, game_config, load_config, scoring_config
from civitas.core import env as env_mod
from civitas.core.decision import Decision
from civitas.core.env import CivitasEnv, GameConfig, make_env
from civitas.core.roles import TURN_ORDER
from civitas.fleet.agent import ToolAgent
from civitas.fleet.config import FleetAgentConfig, fleet_version
from civitas.fleet.report import fleet_report, rows_from_trajectories
from civitas.fleet.scripted import ScriptedFleetAgent, _Turn
from civitas.fleet.tools import ALL_ACTIONS, ToolBox

CONFIG = "configs/fleet_scripted.toml"
MAP_TOOL_ROLES = {"planning", "police", "fire", "health", "education"}


def _cfg(months: int = 12, *sets: str) -> dict:
    return apply_cli_overrides(load_config(CONFIG), sets=[f"game.months={months}", *sets])


def _run(tmp_path, monkeypatch, cfg: dict, seed: int = 0):
    """(summary, env, run dir, trajectory records) of one episode."""
    envs = []
    monkeypatch.setattr(runner, "make_env", lambda *a, **k: envs.append(make_env(*a, **k)) or envs[-1])
    s = runner.run_episode(cfg, seed, tmp_path, verbose=False)
    d = tmp_path / s["run_id"]
    return s, envs[-1], d, [json.loads(x) for x in (d / "trajectories.jsonl").read_text().splitlines()]


def _agent(role: str, env=None, **cfg) -> ScriptedFleetAgent:
    a = ScriptedFleetAgent(role, cfg=FleetAgentConfig(model="scripted", **cfg))
    a.reset(0)
    a.bind(env or CivitasEnv(GameConfig(seed=0, months=12)))
    return a


def test_the_config_builds_a_scripted_fleet_without_a_provider():
    cfg = load_config(CONFIG)
    game = game_config(cfg, 0)
    assert (game.months, game.act_every, game.max_actions_per_turn) == (120, 3, 8)
    big = game_config(load_config("configs/fleet_trinity.toml"), 0)
    assert (big.months, big.act_every, big.max_actions_per_turn) == (game.months, game.act_every,
                                                                      game.max_actions_per_turn)
    agents = build_agents(cfg, list(TURN_ORDER), game, scoring_config(cfg, game))
    assert set(agents) == set(TURN_ORDER) and all(isinstance(a, ScriptedFleetAgent) for a in agents.values())
    assert all(a.kind == "fleet" and a.describe() == "fleet:scripted" and a.cfg.max_steps == 20
               for a in agents.values())
    sig = runner.agent_signature(cfg, list(TURN_ORDER))
    assert sig.count("=fleet:scripted") == 8 and sig.endswith(" [autonomous]")
    assert fleet_version(cfg) != fleet_version(load_config("configs/fleet_trinity.toml"))


def test_one_role_can_be_a_scripted_fleet_agent_in_any_line_up():
    cfg = apply_cli_overrides(load_config("configs/fleet_smoke.toml"), ["police=fleet:scripted"])
    game = game_config(cfg, 0)
    agents = build_agents(cfg, list(TURN_ORDER), game, scoring_config(cfg, game))
    assert isinstance(agents["police"], ScriptedFleetAgent) and agents["police"].cfg.playbook.strip()
    assert all(isinstance(a, ToolAgent) for r, a in agents.items() if r != "police")
    cfg = apply_cli_overrides(load_config("configs/baseline.toml"), ["mayor=fleet:scripted"])
    agents = build_agents(cfg, list(TURN_ORDER), game, scoring_config(cfg, game))
    assert isinstance(agents["mayor"], ScriptedFleetAgent) and agents["planning"].kind == "scripted"
    assert "mayor=fleet:scripted" in runner.agent_signature(cfg, list(TURN_ORDER))


def test_a_short_episode_runs_end_to_end(tmp_path, monkeypatch):
    s, env, d, traj = _run(tmp_path, monkeypatch, _cfg(12))
    assert len(traj) == 4 * 8 and s["months_played"] == 12 and not s["bankrupt"]
    assert all(t["agent"] == "fleet:scripted" and t["ended"] and t["finish_reason"] == "end_turn" for t in traj)
    assert all(t["prompt"][0]["content"] == f"(system prompt: meta.json fleet_prompts.{t['role']})"
               and t["prompt"][1]["role"] == "user" for t in traj)                  # the briefing
    assert all(t["tool_trace"][-1]["tool"] == "end_turn" and t["steps"] == len(t["tool_trace"]) <= 20 for t in traj)
    assert all(t["usage"] == {"prompt_tokens": 0, "completion_tokens": 0} for t in traj)
    # the transcript pairs every call with its result, as a ToolAgent logs it
    for t in traj:
        calls = [m["tool_calls"][0] for m in t["prompt"] if m["role"] == "assistant"]
        assert [c["function"]["name"] for c in calls] == [c["tool"] for c in t["tool_trace"]]
        assert [c["id"] for c in calls] == [m["tool_call_id"] for m in t["prompt"] if m["role"] == "tool"]
    meta = json.loads((d / "meta.json").read_text())
    assert set(meta["fleet_prompts"]) == set(TURN_ORDER) and meta["fleet_version"] == s["fleet_version"]
    rep = json.loads((d / "fleet_report.json").read_text())
    assert list(rep["roles"]) == list(TURN_ORDER) and s["fleet"]["report"] == "fleet_report.md"
    assert all(st["tool_errors"] == 0 and st["refused"] == 0 and st["api_errors"] == 0 and st["limit_turns"] == 0
               for st in rep["roles"].values())
    assert rep["scoreboard"]["city"] == s["final_scores"]["city"] and rep["plan"]["versions"] >= 1
    assert "fleet:scripted" in (d / "fleet_report.md").read_text()
    # the collaboration score, from frames.jsonl (report only): the scripted departments ask the mayor for money
    co = rep["collaboration"]
    assert co["requests"] > 0 and co["components"]["responsiveness"]["n"] > 0 and "## Collaboration" in (
        d / "fleet_report.md").read_text()
    rebuilt, _ = fleet_report(env, s, rows_from_trajectories(d / "trajectories.jsonl"), d)
    assert json.loads(json.dumps(rebuilt)) == rep


def test_every_action_goes_through_the_toolbox(tmp_path, monkeypatch):
    def direct(*a, **k):
        raise AssertionError("env.apply ran an action itself")
    monkeypatch.setattr(env_mod, "apply_action", direct)          # env.apply's own path: never taken
    s, env, d, traj = _run(tmp_path, monkeypatch, _cfg(12))
    results = [(t["month"], t["role"], r) for t in traj for r in t["results"]]
    assert len(results) > 30
    for t in traj:
        ran = [c for c in t["tool_trace"] if c["tool"] in ALL_ACTIONS and "refused" not in c["result"]]
        assert [r["action"]["type"] for r in t["results"]] == [c["tool"] for c in ran]
        assert [r["ok"] for r in t["results"]] == [c["result"]["ok"] for c in ran]
        assert [r["cost"] for r in t["results"]] == [c["result"]["cost"] for c in ran]
    # the history log has each action, with its result and cost; the fleet report counts them
    log = [(e["month"], e["role"], e["action"], e["ok"], e["cost"]) for e in env.fleet.log if e["kind"] == "action"]
    assert log == [(m, role, r["action"], r["ok"], r["cost"]) for m, role, r in results]
    reports = [(e["month"], e["role"]) for e in env.fleet.log if e["kind"] == "report"]
    assert reports == [(t["month"], t["role"]) for t in traj]
    rep = json.loads((d / "fleet_report.json").read_text())
    for role in TURN_ORDER:
        mine = [r for m, who, r in results if who == role]
        assert rep["roles"][role]["actions"] == len(mine) == s["agent_stats"][role]["actions"]
        assert rep["roles"][role]["spent"] == pytest.approx(sum(r["cost"] for r in mine if r["ok"]))
    # look-ups where the policies read them: the map for the map-reading departments, the council for the mayor
    for t in traj:
        tools = [c["tool"] for c in t["tool_trace"]]
        assert ("map_overview" in tools) == (t["role"] in MAP_TOOL_ROLES)
        assert ("get_messages" in tools) == (t["role"] == "mayor")
    # the mayor writes its working balances into the plan before it allocates, and the budgets are its targets
    first = traj[0]
    assert first["role"] == "mayor" and [c["tool"] for c in first["tool_trace"]][:3] == [
        "get_messages", "update_plan", "allocate_budget"]
    plan, v1 = env.fleet.plan, env.fleet.plan_versions[0]
    assert plan["goals"] and set(plan["priorities"]) == set(TURN_ORDER) - {"mayor"}
    assert set(v1["budgets"]) == {"utilities", "planning"} and v1["month"] == 0      # under 150 residents
    for m, who, r in results:
        if who == "mayor" and m == 0:
            assert r["action"]["type"] == "allocate_budget" and 0 < r["action"]["amount"] <= v1["budgets"][
                r["action"]["department"]]
    # messages go out with send_message and reach the council; memory holds the requests
    sent = [(t["month"], t["role"], m["to"], m["text"]) for t in traj for m in t["messages"]]
    assert sent and sent == [(m.month, m.sender, m.to, m.text) for m in env.council.messages]
    assert all(sum(c["tool"] == "send_message" for c in t["tool_trace"]) == len(t["messages"]) for t in traj)
    wrote = [t for t in traj if any(c["tool"] == "update_memory" for c in t["tool_trace"])]
    assert wrote
    for role in {t["role"] for t in wrote}:
        mem = env.fleet.memory[role]
        assert mem["open_tasks"] or mem["lessons"]


def test_it_is_deterministic_per_seed(tmp_path, monkeypatch):
    key = lambda traj: [(t["month"], t["role"], t["tool_trace"], t["results"], t["messages"], t["thinking"],
                         t["prompt"][1:]) for t in traj]
    a = _run(tmp_path / "a", monkeypatch, _cfg(12))
    b = _run(tmp_path / "b", monkeypatch, _cfg(12))
    c = _run(tmp_path / "c", monkeypatch, _cfg(12), seed=1)
    assert key(a[3]) == key(b[3]) and a[0]["final_scores"] == b[0]["final_scores"]
    assert a[1].fleet.plan_versions == b[1].fleet.plan_versions and a[1].fleet.log == b[1].fleet.log
    assert key(a[3]) != key(c[3])


def test_it_plays_exactly_as_the_scripted_baseline(tmp_path, monkeypatch):
    """The tool layer changes nothing the scripted policies do: acting monthly, every turn has the same actions,
    results, costs and council messages as the baselines reading the observation (seeds 0-4, 120 months: the same
    scores to the decimal at act_every 1 and 3). In a 24-month game of seed 7 the mayor's council requests
    (get_messages) change an allocation from month 17 (the event schedule depends on the game's length)."""
    months, seed = 24, 7
    base = apply_cli_overrides(load_config("configs/baseline.toml"), sets=[f"game.months={months}"])
    fleet = _cfg(months, "game.act_every=1")
    b, f = _run(tmp_path / "b", monkeypatch, base, seed)[3], _run(tmp_path / "f", monkeypatch, fleet, seed)[3]
    key = lambda traj: [(t["month"], t["role"], [(r["action"], r["ok"], r["message"], r["cost"]) for r in t["results"]],
                         t["messages"]) for t in traj]
    assert len(b) == len(f) == months * 8
    assert key(b) == key(f)
    assert sum(len(t["results"]) for t in f) > 100


def test_the_mayor_reads_every_request_since_its_previous_turn(tmp_path, monkeypatch):
    """At act_every 3 the departments' requests go out right after the mayor's turn; its next turn reads them all
    (reading only the month before its turn, it read none of them)."""
    read = []
    real = scripted._requested_amounts
    monkeypatch.setattr(scripted, "_requested_amounts", lambda *a, **k: read.append(real(*a, **k)) or read[-1])
    s, env, d, traj = _run(tmp_path, monkeypatch, _cfg(24))
    turns = [t["month"] for t in traj if t["role"] == "mayor"]
    assert turns == list(range(0, 24, 3)) and len(read) == len(turns)
    for since, month, got in zip([None, *turns], turns, read):
        want = {}
        for m in env.council.messages:
            amounts = [float(a.replace(",", "")) for a in re.findall(r"\$([0-9][0-9,]*)", m.text)]
            if since is not None and since <= m.month < month and m.sender != "mayor" and m.to in ("mayor", "all") \
                    and amounts:
                want[m.sender] = max(want.get(m.sender, 0.0), *amounts)
        assert got == want, month
    assert sum(map(len, read)) >= 10


def test_a_duplicate_allocation_the_policy_means_is_confirmed(monkeypatch):
    a = _agent("mayor")
    a.env.world.treasury = 50000
    alloc = {"type": "allocate_budget", "department": "fire", "amount": 1000}
    monkeypatch.setattr(a.policy, "act", lambda obs: Decision(actions=[dict(alloc), dict(alloc)], thinking="twice"))
    d = a.act({})
    tr = [(c["tool"], c["result"].get("refused")) for c in d.meta["tool_trace"]]
    assert tr == [("get_messages", None), ("update_plan", None), ("allocate_budget", None),
                  ("allocate_budget", "duplicate"), ("allocate_budget", None), ("end_turn", None)]
    assert json.loads(d.meta["tool_trace"][4]["args"])["confirm"] is True
    assert a.env.world.dept_funds["fire"] == 2000 and [r["ok"] for r in d.executed] == [True, True]


def test_a_take_back_the_policy_means_is_confirmed(monkeypatch):
    a = _agent("mayor")
    w = a.env.world
    w.dept_funds["fire"] = 1500                        # not this turn's top-up: under the $1,000 it keeps if taken
    alloc = {"type": "allocate_budget", "department": "fire", "amount": -1500}
    monkeypatch.setattr(a.policy, "act", lambda obs: Decision(actions=[dict(alloc)], thinking="take it"))
    d = a.act({})
    tr = [(c["tool"], c["result"].get("refused")) for c in d.meta["tool_trace"]]
    assert tr == [("get_messages", None), ("update_plan", None), ("allocate_budget", "take_back"),
                  ("allocate_budget", None), ("end_turn", None)]
    assert json.loads(d.meta["tool_trace"][3]["args"])["confirm"] is True
    assert w.dept_funds["fire"] == 0 and [r["ok"] for r in d.executed] == [True]


def test_a_loan_refusal_the_policy_means_is_confirmed(monkeypatch):
    """The scripted mayor borrows only at turn start with the treasury under $3,000, so the loan guard never meets it;
    a loan it means anyway is repeated with confirm: true, which the refusal earned."""
    a = _agent("mayor")
    w = a.env.world
    w.treasury = 20000
    loan = {"type": "take_loan", "amount": 10000}
    monkeypatch.setattr(a.policy, "act", lambda obs: Decision(actions=[dict(loan)], thinking="borrow"))
    d = a.act({})
    tr = [(c["tool"], c["result"].get("refused"), c["result"].get("override")) for c in d.meta["tool_trace"]]
    assert tr == [("get_messages", None, None), ("update_plan", None, None), ("take_loan", "loan", None),
                  ("take_loan", None, "loan"), ("end_turn", None, None)]
    assert json.loads(d.meta["tool_trace"][3]["args"])["confirm"] is True
    assert w.debt == 10000 and [r["ok"] for r in d.executed] == [True]


def test_the_step_limit_keeps_a_step_for_end_turn():
    a = _agent("planning", max_steps=3)
    a.env.world.dept_funds["planning"] = 9000                # a road and zones to run
    d = a.act({})
    assert [c["tool"] for c in d.meta["tool_trace"]] == ["map_overview", "build_road", "end_turn"]
    assert d.meta["ended"] and d.meta["finish_reason"] == "end_turn" and d.meta["steps"] == 3
    assert "skipped at the step limit" in d.thinking and len(d.executed) == 1


def test_the_successful_action_cap_stops_the_actions():
    a = _agent("mayor")
    a.env.cfg.max_actions_per_turn = 1                        # month 0: utilities and planning to fund
    d = a.act({})
    assert [r["action"]["department"] for r in d.executed] == ["utilities"] and d.executed[0]["ok"]
    assert not any("action limit" in str(c["result"].get("error")) for c in d.meta["tool_trace"])


def test_a_request_is_not_reported_as_coverage_ok():
    env = CivitasEnv(GameConfig(seed=0, months=12))
    box = ToolBox(env, "police")
    box.messages.append({"to": "mayor", "text": "Police needs $2,500 for a new building (police cover 0.0%)."})
    text = ScriptedFleetAgent._report(Decision(thinking="coverage OK"), box, _Turn(box, 20))
    assert text == "need: Police needs $2,500 for a new building (police cover 0.0%).; 0 actions done"
    box.messages.clear()
    assert ScriptedFleetAgent._report(Decision(thinking="coverage OK"), box, _Turn(box, 20)).startswith("coverage OK")


def test_a_map_too_big_for_map_overview_falls_back_to_the_observation_map():
    env = CivitasEnv(GameConfig(seed=0, months=12, width=64, height=64))
    a = _agent("planning", env)
    assert "scale" in ToolBox(env, "planning").call("map_overview")               # downsampled
    turn = _Turn(ToolBox(env, "planning"), 20)
    view = a._view(turn)
    assert view["map"] == env.world.ascii_map() and [c["tool"] for c in turn.trace] == ["map_overview"]
    small = _agent("planning")
    assert small._view(_Turn(ToolBox(small.env, "planning"), 20))["map"] == small.env.world.ascii_map()


def test_region_games_are_refused():
    with pytest.raises(ValueError, match="single-city"):
        _agent("planning", CivitasEnv(GameConfig(seed=0, months=12), city="north"))


def test_reset_starts_a_fresh_policy():
    a = _agent("planning")
    a.act({})
    assert a.policy.segments is not None
    a.reset(0)
    assert a.policy.segments is None and type(a.policy) is SCRIPTED["planning"]


def test_prompt_command_prints_the_scripted_fleet_prompt(capsys):
    from civitas.cli import main
    main(["prompt", "-c", CONFIG, "--role", "mayor"])
    out = capsys.readouterr().out
    assert "# How to play" in out and "USER (briefing)" in out and "update_plan" in out
