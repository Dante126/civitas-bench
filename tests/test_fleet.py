"""Agent fleets: tool-using manager + departments."""
import pytest

from civitas.core.actions import apply_action
from civitas.core.decision import Decision
from civitas.core.env import CivitasEnv, GameConfig


def _env(**kw):
    return CivitasEnv(GameConfig(seed=0, months=12, **kw))


def test_env_apply_logs_already_executed_results():
    env = _env()
    env.world.treasury = 50000
    res = apply_action(env.world, "mayor", {"type": "allocate_budget", "department": "health", "amount": 1000}).to_dict()
    bad = apply_action(env.world, "mayor", {"type": "set_tax", "sector": "Q", "rate": 9}).to_dict()
    out = env.apply("mayor", Decision(executed=[res, bad]))
    assert out == [res, bad]
    assert env.invalid_counts["mayor"] == 1
    assert env.turns[-1]["results"] == [res, bad]
    assert env.world.dept_funds["health"] == 1000          # not applied a second time


def test_agents_get_bound_to_the_env():
    from civitas.agents.base import Agent
    seen = {}

    class Probe(Agent):
        def bind(self, env):
            seen["env"] = env
        def act(self, obs):
            return Decision()
    env = _env()
    Probe("mayor").bind(env)
    assert seen["env"] is env


# ------------------------------------------------------------------ memory
from civitas.fleet.memory import FleetState


def test_fleet_state_lives_on_the_env():
    env = _env()
    fs = FleetState.of(env)
    assert env.fleet is fs and FleetState.of(env) is fs
    assert fs.plan["goals"] == [] and fs.plan_versions == []
    assert fs.memory["mayor"] == {"commitments": [], "open_tasks": [], "lessons": []}


def test_plan_validation():
    fs = FleetState()
    with pytest.raises(ValueError, match="parks.*valid.*planning"):
        fs.update_plan(0, budgets={"parks": 100})
    with pytest.raises(ValueError, match="negative"):
        fs.update_plan(0, budgets={"health": -5})
    for bad in ("100", True, None):
        with pytest.raises(ValueError, match="number"):
            fs.update_plan(0, budgets={"health": bad})
    with pytest.raises(ValueError, match="list"):
        fs.update_plan(0, goals="grow")
    with pytest.raises(ValueError, match="valid.*goals"):
        fs.update_plan(0, targets=["x"])
    with pytest.raises(ValueError, match="mayor"):
        fs.update_plan(0, priorities={"mayor": "x"})
    assert fs.plan_versions == [] and fs.plan["budgets"] == {}   # nothing accepted
    fs.update_plan(0, goals=[f"goal {i}" for i in range(7)] + ["x" * 500], notes="n" * 2000)
    assert fs.plan["goals"] == [f"goal {i}" for i in range(5)]
    assert len(fs.plan["notes"]) == 1500
    fs.update_plan(0, goals=["y" * 500])
    assert fs.plan["goals"] == ["y" * 200]


def test_plan_merges_and_keeps_versions():
    fs = FleetState()
    msg = fs.update_plan(2, goals=["reach 5k people"], budgets={"health": 1000, "fire": 500.5},
                         priorities={"police": "cover downtown"})
    assert "goals" in msg and "budgets" in msg
    fs.update_plan(5, budgets={"fire": 800}, notes="tight quarter")
    p = fs.plan
    assert p["goals"] == ["reach 5k people"] and p["priorities"] == {"police": "cover downtown"}
    assert p["budgets"] == {"health": 1000, "fire": 800} and p["notes"] == "tight quarter" and p["month"] == 5
    assert [v["month"] for v in fs.plan_versions] == [2, 5]
    assert fs.plan_versions[0]["budgets"] == {"health": 1000, "fire": 500.5}   # copies, not live
    fs.plan["goals"].append("mutated")
    assert fs.plan_versions[1]["goals"] == ["reach 5k people"]


def test_memory_caps_and_validation():
    fs = FleetState()
    fs.update_memory("police", lessons=[f"l{i}" for i in range(12)], open_tasks=["z" * 300])
    m = fs.memory["police"]
    assert m["lessons"] == [f"l{i}" for i in range(8)] and m["open_tasks"] == ["z" * 200]
    fs.update_memory("police", lessons=["only one"])
    assert m["lessons"] == ["only one"] and m["open_tasks"] == ["z" * 200]   # other lists kept
    with pytest.raises(ValueError, match="valid.*commitments"):
        fs.update_memory("police", notes=["x"])
    with pytest.raises(ValueError, match="list"):
        fs.update_memory("police", lessons="x")


def test_history_filters():
    fs = FleetState()
    for month in range(4):
        fs.record(month, "fire", "action", action={"type": "build"}, ok=month != 2, message="", cost=10)
        fs.record(month, "police", "action", action={"type": "build"}, ok=False, message="no", cost=0)
    fs.record(3, "fire", "report", text="done")
    with pytest.raises(ValueError, match="kind"):
        fs.record(3, "fire", "gossip")
    recent = fs.history(kind="action", months=1)
    assert [(e["month"], e["role"]) for e in recent] == [(3, "fire"), (3, "police")]
    assert [e["month"] for e in fs.history(kind="action", months=2, role="fire")] == [2, 3]
    assert [e["month"] for e in fs.history(kind="failure", months=4, role="fire")] == [2]
    assert fs.history(kind="report", months=1)[0]["text"] == "done"
    assert len(fs.history(months=4)) == 9
    for i in range(50):
        fs.record(3, "fire", "action", action={"type": "build", "i": i}, ok=True, message="", cost=0)
    newest = fs.history(kind="action", months=1)
    assert len(newest) == 40 and newest[-1]["action"]["i"] == 49


def test_sync_logs_metrics_and_events_once():
    env = _env()
    fs = FleetState.of(env)
    fs.sync(env)
    assert fs.log == []                               # month 0: nothing happened yet
    env.world.event("A tornado hit")
    env.end_month()
    fs.sync(env)
    fs.sync(env)
    snap, = fs.history(kind="metrics")
    assert snap["month"] == 1 and set(snap) >= {"population", "treasury", "happiness", "city_score"}
    ev, = fs.history(kind="event")
    assert ev["month"] == 0 and ev["text"] == "A tornado hit"


def test_digest_covers_only_this_role_since_its_last_turn():
    env = _env()
    fs = FleetState.of(env)
    fs.record(0, "fire", "action", action={"type": "a"}, ok=False, message="old", cost=0)
    env.council.post(0, "mayor", "seen at month 0", "fire")
    fs.mark_turn("fire", 0, env.council)
    env.world.event("Flood")
    env.end_month()
    env.council.post(1, "mayor", "seen at month 1", "all")
    fs.record(1, "fire", "action", action={"type": "b"}, ok=False, message="blocked", cost=0)
    fs.record(1, "fire", "action", action={"type": "c"}, ok=True, message="", cost=5)
    fs.record(1, "police", "action", action={"type": "d"}, ok=False, message="theirs", cost=0)
    fs.mark_turn("fire", 1, env.council)
    env.council.post(1, "mayor", "fire: build a station", "fire")
    env.council.post(1, "police", "hello all", "all")
    env.council.post(1, "mayor", "police only", "police")
    env.council.post(1, "fire", "my own", "all")
    env.world.event("Fire downtown")
    env.end_month()
    d = fs.digest("fire", 2, council=env.council)
    assert d["since"] == 1
    assert [e["message"] for e in d["failed_actions"]] == ["blocked"]
    assert d["events"] == ["Fire downtown"]
    assert d["messages"] == 2
    assert fs.digest("police", 2)["since"] is None and fs.digest("police", 2)["messages"] == 0
    assert fs.digest("police", 2, council=env.council)["messages"] == 3   # never marked: all from others
    assert fs.digest("fire", 2)["failed_actions"] == d["failed_actions"]
    first = FleetState()
    first.record(0, "fire", "event", text="x")
    assert first.digest("fire", 1)["since"] is None


def test_end_month_syncs_the_fleet_every_month():
    env = _env(act_every=3)
    assert env.fleet is None
    fs = FleetState.of(env)
    for month in range(5):
        env.world.event(f"event {month}")
        env.end_month()                                  # no agent turns at all
    assert [e["month"] for e in fs.history(kind="metrics", months=None)] == [1, 2, 3, 4, 5]
    assert [(e["month"], e["text"]) for e in fs.history(kind="event", months=None)
            if e["text"].startswith("event ")] == [(m, f"event {m}") for m in range(5)]
    env.reset()
    assert env.fleet is None and FleetState.of(env) is not fs


# ------------------------------------------------------------------- tools
import json

import numpy as np

from civitas.agents.scripted import SCRIPTED
from civitas.core.roles import ROLES, TURN_ORDER
from civitas.fleet.tools import SITE_PURPOSES, ToolBox
from civitas.sim import catalog as C


@pytest.fixture(scope="module")
def grown_env():
    env = CivitasEnv(GameConfig(seed=0, months=24))
    agents = {r: SCRIPTED[r](r) for r in env.roles}
    for _ in range(24):
        for r in env.roles:
            env.apply(r, agents[r].act(env.observe(r)))
        env.end_month()
    return env


def test_tool_schemas_cover_every_action():
    env = _env()
    for role in TURN_ORDER:
        schemas = ToolBox(env, role).schemas()
        names = [s["function"]["name"] for s in schemas]
        assert len(names) == len(set(names))
        assert set(ROLES[role]["actions"]) - {"noop"} <= set(names) and "noop" not in names
        assert {"city_overview", "inspect", "end_turn", "send_message", "history"} <= set(names)
        assert ("update_plan" in names) == (role == "mayor")
        for s in schemas:
            assert s["type"] == "function"
            p = s["function"]["parameters"]
            assert p["type"] == "object" and "type" not in p["properties"]
            assert all(set(v) & {"type", "enum"} for v in p["properties"].values())
        json.dumps(schemas)
    tools = {s["function"]["name"]: s["function"] for s in ToolBox(env, "utilities").schemas()}
    props = tools["build_power_plant"]["parameters"]["properties"]
    assert props["x"]["type"] == "integer" and props["kind"]["type"] == "string" and props["tier"]["type"] == "integer"
    assert "tier" not in tools["demolish"]["parameters"]["properties"]
    assert set(tools["find_sites"]["parameters"]["properties"]["purpose"]["enum"]) == set(SITE_PURPOSES["utilities"])
    mayor = {s["function"]["name"]: s["function"] for s in ToolBox(env, "mayor").schemas()}
    assert mayor["set_service_imports"]["parameters"]["properties"]["enabled"]["type"] == "boolean"
    transport = {s["function"]["name"]: s["function"] for s in ToolBox(env, "transport").schemas()}
    assert transport["create_bus_line"]["parameters"]["properties"]["stops"] == {"type": "array", "items": {}}


def test_action_tool_builds_and_records():
    env = _env()
    env.world.dept_funds["utilities"] = 10000
    box = ToolBox(env, "utilities")
    sites = box.call("find_sites", {"purpose": "wind"})
    assert sites["ok"] and sites["sites"]
    s = sites["sites"][0]
    res = box.call("build_power_plant", json.dumps({"kind": "wind", "x": str(s["x"]), "y": s["y"]}))
    assert res["ok"], res
    assert res["cost"] > 0 and res["budget_left"] == pytest.approx(10000 - res["cost"])
    assert env.world.struct[s["y"], s["x"]] == C.WIND
    assert box.executed[-1]["ok"] and box.executed[-1]["action"]["type"] == "build_power_plant"
    entry, = FleetState.of(env).history(kind="action", role="utilities")
    assert entry["ok"] and entry["action"]["kind"] == "wind"


def test_failing_action_is_reported_and_logged():
    env = _env()
    env.world.dept_funds["utilities"] = 0
    box = ToolBox(env, "utilities")
    res = box.call("build_power_plant", {"kind": "wind", "x": 0, "y": 0})
    assert res["ok"] is False and res["message"] and res["error"] == res["message"] and res["cost"] == 0
    assert box.executed[-1]["message"] == res["message"]
    fail, = FleetState.of(env).history(kind="failure", role="utilities")
    assert fail["message"] == res["message"]
    res = box.call("build_power_plant", {"kind": "wind", "x": "twelve", "y": 3})
    assert res["ok"] is False and "x" in res["error"] and "integer" in res["error"]
    assert len(box.executed) == 1                      # type errors never reach the handler


def test_bad_calls_return_errors():
    env = _env()
    box = ToolBox(env, "police")
    for name, args, hint in [("build_power_plant", {}, "police"), ("fly", {}, "unknown tool"),
                             ("inspect", "{not json", "JSON"), ("inspect", "[1, 2]", "object"),
                             ("update_plan", {"goals": ["x"]}, "mayor"), ("inspect", {"x": 3}, "y"),
                             ("inspect", {"x": 3, "y": 3, "zoom": 2}, "zoom"),
                             ("history", {"kind": "gossip"}, "failure"),
                             ("send_message", {"to": "parks", "text": "hi"}, "mayor"),
                             ("find_sites", {"purpose": "wind"}, "station")]:
        res = box.call(name, args)
        assert res["ok"] is False and hint in res["error"], (name, res)
    mayor = ToolBox(env, "mayor")
    res = mayor.call("update_plan", {"budgets": {"parks": 5}})
    assert res["ok"] is False and "planning" in res["error"]
    assert mayor.call("update_plan", {"goals": ["grow"], "budgets": {"fire": 500}, "notes": None})["ok"]
    assert env.fleet.plan["budgets"] == {"fire": 500}
    plan = ToolBox(env, "fire").call("get_plan", {})
    assert plan["ok"] and plan["budget"] == 500 and plan["goals"] == ["grow"] and "budgets" not in plan


def test_inspect_clips_radius_and_names_tiers():
    env = _env()
    w = env.world
    site = ToolBox(env, "utilities").call("find_sites", {"purpose": "near_homes"})["sites"][0]
    x, y = site["x"], site["y"]
    w.place_building(C.POLICE, x, y, tier=2)
    w.refresh()
    res = ToolBox(env, "police").call("inspect", {"x": x, "y": y, "radius": 99})
    assert res["ok"] and res["radius"] == 4
    tiles = res["tiles"]
    assert all(abs(t["x"] - x) <= 4 and abs(t["y"] - y) <= 4 for t in tiles)
    hq, = [t for t in tiles if (t["x"], t["y"]) == (x, y)]
    assert hq["what"] == "police_hq" and hq["tier"] == 2
    assert any(t["what"] == "road" for t in tiles) and isinstance(res["empty_land"], int)
    assert ToolBox(env, "police").call("inspect", {"x": x, "y": y, "radius": -3})["radius"] == 0
    spot = w.site_candidates(np.ones(w.struct.shape), radius=1, k=1, size=2)[0]
    w.place_building(C.HOSPITAL, spot["x"], spot["y"], tier=2)
    w.refresh()
    tiles = ToolBox(env, "health").call("inspect", {"x": spot["x"] + 1, "y": spot["y"] + 1, "radius": 1})["tiles"]
    hosp = [t for t in tiles if t["what"] == "hospital"]
    assert len(hosp) == 1 and (hosp[0]["x"], hosp[0]["y"]) == (spot["x"], spot["y"]) and hosp[0]["size"] == "2x2"


CONDITIONAL_SITES = {"riverbank_2x2", "casino", "metro_station", "levee"}   # lists only there once unlocked / forecast


def _check_sites(env):
    from civitas.core.observation import build_observation
    for role, purposes in SITE_PURPOSES.items():
        box = ToolBox(env, role)
        obs = build_observation(env.world, role, council_msgs=[], memory={}, scores={}, months_total=24,
                                include_map=False, site_hints=True)
        for purpose, path in purposes.items():
            found = obs
            for key in path:
                found = found.get(key) if isinstance(found, dict) else None
            assert isinstance(found, list) or purpose in CONDITIONAL_SITES, (role, purpose, path)
            res = box.call("find_sites", {"purpose": purpose, "near_x": 10, "near_y": 10})
            assert res["ok"], (role, purpose, res)
            assert len(res["sites"]) <= 6 and all(isinstance(s["x"], int) and isinstance(s["y"], int)
                                                  for s in res["sites"])
            assert res["sites"] or res.get("note"), (role, purpose)
            json.dumps(res)


def test_find_sites_on_a_fresh_world():
    env = _env()
    _check_sites(env)
    res = ToolBox(env, "transport").call("find_sites", {"purpose": "metro_station"})
    assert res["sites"] == [] and "10,000" in res["note"]


def test_find_sites_and_lookups_on_a_grown_world(grown_env):
    env = grown_env
    _check_sites(env)
    near = ToolBox(env, "utilities").call("find_sites", {"purpose": "near_homes", "near_x": 40, "near_y": 40})["sites"]
    d = [abs(s["x"] - 40) + abs(s["y"] - 40) for s in near]
    assert d == sorted(d)
    for role in TURN_ORDER:
        box = ToolBox(env, role)
        for name in ("city_overview", "my_facilities", "unlocks", "get_plan", "get_messages", "history"):
            res = box.call(name, {})
            assert res["ok"], (role, name, res)
            assert len(json.dumps(res)) <= 4000, (role, name)
        assert len(json.dumps(box.call("inspect", {"x": 24, "y": 24, "radius": 4}))) <= 4000
    ov = ToolBox(env, "mayor").call("city_overview", {})
    assert ov["population"] > 0 and ov["trend"]["since_month"] == env.world.month - 3
    util = ToolBox(env, "utilities").call("my_facilities", {})
    assert any(n["plants"] for n in util["networks"])
    health = ToolBox(env, "health").call("my_facilities", {})
    assert health["facilities"]


def test_turn_control_tools():
    env = _env()
    box = ToolBox(env, "fire")
    assert box.call("send_message", {"to": "mayor", "text": "need $2k"})["ok"]
    assert box.messages == [{"to": "mayor", "text": "need $2k"}]
    assert box.call("update_memory", {"lessons": ["stations need roads"]})["ok"]
    assert env.fleet.memory["fire"]["lessons"] == ["stations need roads"]
    assert not box.ended
    assert box.call("end_turn", {"report": "built nothing"})["ok"]
    assert box.ended and box.report == "built nothing"
    assert env.fleet.reports["fire"] == "built nothing"
    rep, = box.call("history", {"kind": "report"})["entries"]
    assert rep["text"] == "built nothing"
    env.apply("fire", Decision(messages=box.messages))
    assert ToolBox(env, "mayor").call("get_messages", {})["messages"][-1]["text"] == "need $2k"


def test_history_filters_by_several_roles_before_the_cap():
    fs = FleetState()
    fs.record(0, "fire", "action", action={"type": "a"}, ok=True, message="", cost=0)
    for i in range(45):
        fs.record(0, "police", "action", action={"type": "b"}, ok=True, message="", cost=0)
    assert [e["role"] for e in fs.history(role=("fire", "city"))] == ["fire"]
    assert len(fs.history(role={"fire", "police"})) == 40


def test_department_history_keeps_its_own_entries():
    env = _env()
    fs = FleetState.of(env)
    fs.record(0, "fire", "action", action={"type": "build_fire_station"}, ok=False, message="no road", cost=0)
    for i in range(45):
        fs.record(0, "police", "action", action={"type": "build_police_station"}, ok=True, message="", cost=0)
    entries = ToolBox(env, "fire").call("history", {})["entries"]
    assert [e["message"] for e in entries] == ["no road"]
    assert len(ToolBox(env, "mayor").call("history", {})["entries"]) == 20


def test_successful_actions_are_capped_per_turn():
    env = _env(max_actions_per_turn=2)
    box = ToolBox(env, "mayor")
    assert box.call("set_tax", {"sector": "R", "rate": 8})["ok"]
    assert not box.call("set_tax", {"sector": "Q", "rate": 8})["ok"]      # failures don't count
    assert box.call("set_tax", {"sector": "C", "rate": 8.5})["ok"]
    res = box.call("set_tax", {"sector": "I", "rate": 8})
    assert res["ok"] is False and "limit" in res["error"] and "end_turn" in res["error"]
    assert len(box.executed) == 3 and env.world.taxes["I"] != 0.08
    assert box.call("city_overview", {})["ok"] and box.call("end_turn", {"report": "taxes"})["ok"]
    assert ToolBox(env, "mayor").call("set_tax", {"sector": "I", "rate": 8})["ok"]     # a new turn


def test_float_handler_params_are_numbers():
    import inspect as pyinspect
    import re
    from civitas.core.actions import HANDLERS, action_docs
    from civitas.fleet.tools import NUMBER_PARAMS
    env = _env()
    docs = action_docs()
    floats = set()
    for name, fn in HANDLERS.items():
        for k in re.findall(r'_num\(p, "(\w+)"\)', pyinspect.getsource(fn)):
            if isinstance(json.loads(docs[name][0]).get(k), int):
                floats.add((name, k))
    assert floats == NUMBER_PARAMS
    for name, k in NUMBER_PARAMS:
        role = next(r for r, spec in ROLES.items() if name in spec["actions"])
        tool = {s["function"]["name"]: s["function"] for s in ToolBox(env, role).schemas()}[name]
        assert tool["parameters"]["properties"][k]["type"] == "number"


def test_type_argument_cannot_switch_the_action():
    env = _env()
    w = env.world
    box = ToolBox(env, "mayor")
    debt = w.debt
    res = box.call("set_tax", {"type": "take_loan", "amount": 5000, "sector": "R", "rate": 8})
    assert res["ok"] is False and "kind" in res["error"] and w.debt == debt and box.executed == []
    assert box.call("set_tax", {"type": "set_tax", "sector": "R", "rate": 8})["ok"]
    assert box.executed[-1]["action"]["type"] == "set_tax"
    res = ToolBox(env, "utilities").call("build_water", {"type": "pump", "x": 3, "y": 3})
    assert res["ok"] is False and "did you mean kind" in res["error"]
    res = box.call("noop", {})
    assert res["ok"] is False and "end_turn" in res["error"]


def test_trimming_keeps_the_newest_log_entries_and_messages():
    env = _env()
    fs = FleetState.of(env)
    for i in range(40):
        fs.record(0, "fire", "action", action={"type": "build_fire_station", "i": i}, ok=False,
                  message="x" * 300, cost=0)
    res = ToolBox(env, "fire").call("history", {})
    assert res["entries"][-1]["action"]["i"] == 39 and len(res["entries"]) < 20
    assert "older" in res["truncated"] and "fewer months" in res["truncated"]
    for i in range(20):
        env.council.post(0, "mayor", f"{i} " + "y" * 400, "all")
    res = ToolBox(env, "fire").call("get_messages", {})
    assert res["messages"][-1]["text"].startswith("19 ") and "older" in res["truncated"]
    assert "radius" not in res["truncated"] and "months" not in res["truncated"]
    res = ToolBox(env, "fire").call("inspect", {"x": 24, "y": 24, "radius": 4})
    assert "truncated" not in res or "radius" in res["truncated"]


def test_role_specific_history_description_and_fresh_lookups():
    env = _env()
    desc = lambda role: next(s["function"]["description"] for s in ToolBox(env, role).schemas()
                             if s["function"]["name"] == "history")
    assert "all roles" in desc("mayor") and "your department" in desc("police")
    for role in TURN_ORDER:
        box = ToolBox(env, role)
        for name in ("city_overview", "my_facilities", "unlocks", "history", "get_plan", "get_messages"):
            res = box.call(name, {})
            assert res["ok"], (role, name, res)
            json.dumps(res)
    assert ToolBox(env, "mayor").call("city_overview", {})["trend"] is None


# ---------------------------------------------------------------- briefing
import copy

from civitas.fleet.briefing import KEY_METRICS, briefing, system_prompt
from civitas.fleet.config import FleetAgentConfig

DEPTS = [r for r in TURN_ORDER if r != "mayor"]


def _words(text: str) -> int:
    return len(text.split())


@pytest.fixture(scope="module")
def env12():
    env = CivitasEnv(GameConfig(seed=0, months=24))
    agents = {r: SCRIPTED[r](r) for r in env.roles}
    for _ in range(12):
        for r in env.roles:
            env.apply(r, agents[r].act(env.observe(r)))
        env.end_month()
    return env


def test_system_prompt_sections_in_order():
    cfg = FleetAgentConfig(fleet_playbook="FLEET-PB: grow first", playbook="ROLE-PB: stations by homes", max_steps=17)
    for role in TURN_ORDER:
        text = system_prompt(role, cfg, 36, max_actions=5)
        marks = ["How to play", ROLES[role]["title"], ROLES[role]["mission"][:40], "36 months", "Coach's playbook",
                 "FLEET-PB", "ROLE-PB"]
        pos = [text.find(m) for m in marks]
        assert -1 not in pos and pos == sorted(pos), (role, list(zip(marks, pos)))
        assert "end_turn" in text and "17 tool calls" in text and "5 successful actions" in text
        assert "{" not in text.replace("{}", "")                         # every placeholder filled
    assert ("update_plan" in system_prompt("mayor", cfg, 36)) and "update_plan" not in system_prompt("fire", cfg, 36)
    bare = system_prompt("fire", FleetAgentConfig(), 36)
    assert "Coach's playbook" not in bare and "20 tool calls" in bare
    only_fleet = system_prompt("fire", FleetAgentConfig(fleet_playbook="FLEET-PB"), 36)
    assert "FLEET-PB" in only_fleet and "Fire" not in only_fleet.split("Coach's playbook")[1]


def test_key_metrics_come_from_the_observation():
    from civitas.core.observation import build_observation
    env = _env()
    city = build_observation(env.world, "police", council_msgs=[], memory={}, scores={}, months_total=12,
                             include_map=False, site_hints=False)["city"]
    assert set(KEY_METRICS) == set(DEPTS)
    for role, paths in KEY_METRICS.items():
        assert 5 <= len(paths) <= 8, role
        for path in paths:
            v = city
            for k in path.split("."):
                v = v[k]


@pytest.mark.parametrize("grown", [False, True])
def test_briefings_render_for_every_role(grown, env12):
    env = copy.deepcopy(env12) if grown else CivitasEnv(GameConfig(seed=0, months=24))
    w = env.world
    for role in TURN_ORDER:
        text = briefing(env, role, mark=False)
        assert _words(text) < 700, (role, _words(text))
        if role == "planning":
            dev, = [ln for ln in text.splitlines() if "developed tiles" in ln]
            assert "R " in dev and "+" not in dev                        # counts, not signed demand values
        assert f"{24 - w.month} months left" in text and "autonomous" in text
        if role == "mayor":
            assert "no plan yet" in text and "update_plan" in text and "treasury" in text
        else:
            assert "No city plan yet" in text and "budget" in text.lower()
    assert FleetState.of(env).history("metrics") or not grown          # the briefing synced the month's snapshot
    w.dept_funds["police"] = 1234
    box = ToolBox(env, "mayor")
    assert box.call("update_plan", {"goals": ["reach 5k residents"], "notes": "save for bridges",
                                    "budgets": {d: 1000 + i for i, d in enumerate(DEPTS)},
                                    "priorities": {d: f"PRIORITY-{d}" for d in DEPTS}})["ok"]
    for i, role in enumerate(DEPTS):
        text = briefing(env, role, mode="supervised", mark=False)
        assert f"PRIORITY-{role}" in text and "reach 5k residents" in text and "save for bridges" in text
        assert f"${1000 + i:,}" in text                                  # budget target
        assert "PRIORITY-" + ("fire" if role != "fire" else "police") not in text
        assert "supervised" in text and _words(text) < 700
    assert "$1,234" in briefing(env, "police", mark=False)
    mayor = briefing(env, "mayor", mark=False)
    assert "reach 5k residents" in mayor and "PRIORITY-health" in mayor and "no plan yet" not in mayor


def test_mayor_briefing_lists_department_reports():
    env = _env()
    assert "no reports yet" in briefing(env, "mayor").lower()
    ToolBox(env, "health").call("end_turn", {"report": "HEALTH-REPORT: built a clinic, need $3k"})
    ToolBox(env, "fire").call("end_turn", {"report": "FIRE-REPORT " + "z" * 600})
    env.end_month()
    ToolBox(env, "health").call("end_turn", {"report": "HEALTH-REPORT-2: all good"})
    text = briefing(env, "mayor")
    assert "HEALTH-REPORT-2" in text and "HEALTH-REPORT:" not in text     # latest per department
    fire, = [ln for ln in text.splitlines() if "FIRE-REPORT" in ln]
    assert len(fire) < 240
    assert "no reports yet" not in text.lower()


def test_department_briefing_digests_its_own_last_turn():
    env = _env()
    fs = FleetState.of(env)
    briefing(env, "police")
    assert fs.last_turn_month["police"] == 0
    env.world.dept_funds["police"] = 0
    bad = ToolBox(env, "police").call("build_police_station", {"x": 0, "y": 0})
    theirs = ToolBox(env, "fire").call("build_fire_station", {"x": 1, "y": 1, "kind": "FIRE-KIND"})
    assert not bad["ok"] and not theirs["ok"]
    env.council.post(0, "mayor", "police: cover downtown", "police")
    env.world.event("A tornado hit")
    env.end_month()
    assert "failed actions: none" in briefing(env, "health", mark=False)
    text = briefing(env, "police", mark=False)
    assert "build_police_station" in text and bad["message"][:40] in text and "failed actions: none" not in text
    assert "events: none" not in text
    assert "build_fire_station" not in text and "FIRE-KIND" not in text
    assert "A tornado hit" in text and "1 new council message" in text
    assert fs.last_turn_month["police"] == 0                          # mark=False leaves the turn unmarked
    briefing(env, "police")
    assert fs.last_turn_month["police"] == 1 and fs.digest("police", 1, env.council)["messages"] == 0
