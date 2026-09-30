"""Agent fleets: tool-using manager + departments."""
import copy

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
    assert [m["text"] for m in d["messages"]] == ["fire: build a station", "hello all"]
    assert fs.digest("police", 2)["since"] is None and fs.digest("police", 2)["messages"] == []
    assert len(fs.digest("police", 2, council=env.council)["messages"]) == 3   # never marked: all from others
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
from civitas.fleet.memory import DEPARTMENTS
from civitas.fleet.tools import SITE_PURPOSES, SITE_UNLOCKS, ToolBox, free_bank
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
    pairs = {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}   # [[x, y], ...]
    assert transport["create_bus_line"]["parameters"]["properties"]["stops"] == pairs
    assert transport["create_metro_line"]["parameters"]["properties"]["stations"] == pairs


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
    assert res["ok"] and res["radius"] == 4 and "clipped to 4" in res["note"]
    assert "note" not in ToolBox(env, "police").call("inspect", {"x": x, "y": y, "radius": 4})
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


def test_inspect_points_to_map_overview_from_the_4th_call_in_a_turn():
    """Smoke v9-v11: planning made 5.4 inspects a turn (other roles 0.4-1.2) and 12 of 24 turns hit the step limit.
    From the 4th inspect in a turn the result says map_overview shows the whole map, unless it ran this turn; an
    identical repeat returns the earlier result and isn't counted."""
    env = _env()
    note = lambda n: f"inspect #{n} this turn: map_overview shows the whole map in one call"
    for role in TURN_ORDER:                                          # every role, counted per turn (per ToolBox)
        box = ToolBox(env, role)
        assert ["note" in box.call("inspect", {"x": x, "y": 24}) for x in (4, 8, 12)] == [False] * 3, role
        again = box.call("inspect", {"x": 12, "y": 24})
        assert again["repeat"] == "same as call 3 (nothing changed since)" and "note" not in again, role
        assert box.call("inspect", {"x": 16, "y": 24})["note"] == note(4), role     # the repeat didn't count
        again = box.call("inspect", {"x": 16, "y": 24})                             # the look-up, not the advice
        assert again["repeat"] == "same as call 5 (nothing changed since)" and "note" not in again, role
        assert box.call("inspect", {"x": 20, "y": 24})["note"] == note(5), role
    box = ToolBox(env, "planning")
    for x in (4, 8, 12):
        box.call("inspect", {"x": x, "y": 24})
    res = box.call("inspect", {"x": 16, "y": 24, "radius": 9})                  # the radius note gives the advice
    assert res["note"] == "radius 9 clipped to 4 (0-4); map_overview shows the whole map; inspect #4 this turn", res
    assert box.call("map_overview", {})["ok"]
    assert "note" not in box.call("inspect", {"x": 20, "y": 24}) and "note" not in box.call("inspect", {"x": 4, "y": 4})
    box = ToolBox(env, "planning")
    assert box.call("map_overview", {})["ok"]                                         # first: never this note
    assert not any(box.call("inspect", {"x": x, "y": 20}).get("note", "").endswith("in one call")
                   for x in range(0, 40, 5))
    box = ToolBox(env, "planning")                                                    # an off-map inspect counts
    assert box.call("inspect", {"x": 99, "y": 24})["ok"] is False
    assert ["note" in box.call("inspect", {"x": x, "y": 24}) for x in (4, 8)] == [False] * 2
    assert box.call("inspect", {"x": 12, "y": 24})["note"] == note(4)
    env.world.dept_funds["planning"] = 1000                                           # after a successful action an
    box = ToolBox(env, "planning")                                                    # identical inspect runs again
    assert ["note" in box.call("inspect", {"x": x, "y": 24}) for x in (4, 8, 12)] == [False] * 3
    assert box.call("zone", {"zone": "R", "x1": 3, "y1": 25, "x2": 8, "y2": 25})["ok"]
    res = box.call("inspect", {"x": 4, "y": 24})
    assert "repeat" not in res and res["note"] == note(4), res


def test_inspects_after_map_overview_say_its_map_shows_the_layout():
    """Smoke v12-v16: planning called map_overview on 35 of its 40 turns and then inspected 134 times (3.8 a turn, up
    to 10), and the note, which only pointed to map_overview, stayed silent. From the 3rd inspect after map_overview
    (counted from its first run; repeats don't count) the result says its map already shows the layout."""
    env = _env()
    note = lambda n: (f"inspect #{n} since map_overview: its map already shows the layout; inspect only a tile you are "
                      "about to act on")
    box = ToolBox(env, "planning")
    assert ["note" in box.call("inspect", {"x": x, "y": 24}) for x in (4, 8, 12)] == [False] * 3
    assert box.call("map_overview", {})["ok"]
    assert ["note" in box.call("inspect", {"x": x, "y": 30}) for x in (4, 8)] == [False] * 2    # counted afresh
    assert box.call("inspect", {"x": 8, "y": 30})["repeat"]
    assert box.call("inspect", {"x": 12, "y": 30})["note"] == note(3)
    res = box.call("inspect", {"x": 16, "y": 30, "radius": 9})
    assert res["note"] == "radius 9 clipped to 4 (0-4); map_overview shows the whole map; " + note(4), res
    env.world.dept_funds["planning"] = 1000
    assert box.call("zone", {"zone": "R", "x1": 3, "y1": 25, "x2": 8, "y2": 25})["ok"]
    assert box.call("map_overview", {})["ok"]                                           # runs again: no fresh count
    assert box.call("inspect", {"x": 4, "y": 30})["note"] == note(5)
    for role in TURN_ORDER:                                                             # every role
        box = ToolBox(env, role)
        box.call("map_overview", {})
        assert [box.call("inspect", {"x": x, "y": 20}).get("note") for x in (0, 5, 10)] == [None, None, note(3)], role


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
            if "stops" in res:                         # bus_stops: one line through every stop
                assert purpose == "bus_stops" and 2 <= len(res["stops"]) <= C.MAX_STOPS, res
                assert res["line"].startswith(f"{len(res['stops'])} stops for one line: create_bus_line stops [[")
                json.dumps(res)
                continue
            assert len(res["sites"]) <= 6 and all(isinstance(s["x"], int) and isinstance(s["y"], int)
                                                  for s in res["sites"])
            assert res["sites"] or res.get("note"), (role, purpose)
            json.dumps(res)


def test_find_sites_on_a_fresh_world():
    env = _env()
    _check_sites(env)
    res = ToolBox(env, "transport").call("find_sites", {"purpose": "metro_station"})
    assert res["sites"] == [] and "10,000" in res["note"]


def test_riverbank_sites_say_a_road_must_reach_the_river():
    env = _env()
    w = env.world
    box = ToolBox(env, "utilities")
    res = box.call("find_sites", {"purpose": "pump"})
    assert res["sites"] == [] and "ask planning" in res["note"] and "inspect the map" not in res["note"]
    assert "a road has to reach the river" in res["note"] and "taken" not in res["note"]
    ys = np.nonzero((w.struct == C.ROAD).any(axis=1))[0]
    cols = np.nonzero((w.terrain[ys.min():ys.max() + 1] == C.WATER).any(axis=0))[0]
    assert f"x {cols.min()}-{cols.max()} in rows {ys.min()}-{ys.max()}" in res["note"]
    y = int(ys.max())
    x_end = int(np.nonzero(w.struct[y] == C.ROAD)[0].max())
    bank = int(np.nonzero(w.terrain[y] == C.WATER)[0].min()) - 1
    w.dept_funds["planning"] = 1000
    assert apply_action(w, "planning", {"type": "build_road", "x1": x_end, "y1": y, "x2": bank, "y2": y}).ok
    res = ToolBox(env, "utilities").call("find_sites", {"purpose": "pump"})     # utilities' next turn
    assert res["sites"] and "note" not in res


def _bridge_and_take_the_bank(env):
    """Seed 0: the highway road on row 24 ends at x 19, the river runs at x 26-28 there. Bridge it and zone every
    riverbank tile beside the connected roads, as planning did in the v5 smoke run."""
    from civitas.fleet.tools import _connected_roads
    w = env.world
    w.dept_funds["planning"] = 5000
    assert apply_action(w, "planning", {"type": "build_road", "x1": 19, "y1": 24, "x2": 38, "y2": 24}).ok
    assert (w.terrain[24, 26:29] == C.WATER).all() and (w.terrain[24, [25, 29]] == C.LAND).all()
    road = _connected_roads(w)
    taken = []
    for y, x in zip(*np.nonzero((w.struct == C.NONE) & (w.terrain == C.LAND))):
        near_road = road[max(0, y - 1):y + 2, max(0, x - 1):x + 2].any()
        if near_road and w.is_riverbank(x, y):
            w.struct[y, x] = C.RES
            taken.append((int(x), int(y)))
    w.refresh()
    assert taken
    return taken


def _blocked(env, role: str, purpose: str) -> bool:
    """Whether `role`'s `purpose` list is empty for lack of land or roads (SiteList.blocked)."""
    from civitas.fleet.tools import role_obs, site_lists
    return site_lists(env.world, role, role_obs(env, role))[purpose].blocked


def test_riverbank_note_says_the_bank_beside_the_road_is_taken():
    env = _env()
    w = env.world
    taken = _bridge_and_take_the_bank(env)
    box = ToolBox(env, "utilities")
    for purpose in ("pump", "sewage"):
        res = box.call("find_sites", {"purpose": purpose})
        note = res["note"]
        assert res["sites"] == [] and "taken" in note and "a road has to reach" not in note, note
        assert "(25,24)" in note and "(29,24)" in note and "ask planning" in note      # the bridgeheads
        assert _blocked(env, "utilities", purpose)
    line, = [ln for ln in _ready_sites(briefing(env, "utilities", mark=False)) if ln.startswith("- sewage:")]
    assert "taken" in line and "(25,24)" in line and len(line) <= 120, line
    w.milestones_on = False                              # riverbank_2x2 unlocked: same reason
    res = box.call("find_sites", {"purpose": "riverbank_2x2"})
    assert res["sites"] == [] and "taken" in res["note"] and _blocked(env, "utilities", "riverbank_2x2"), res
    x, y = taken[0]                                      # free one lot, hemmed in: the pump list finds it
    w.struct[y, x] = C.NONE
    for ny in range(y - 1, y + 2):
        for nx in range(x - 1, x + 2):
            if (nx, ny) != (x, y) and w.struct[ny, nx] == C.NONE and w.terrain[ny, nx] == C.LAND:
                w.struct[ny, nx] = C.PARK
    w.refresh()
    box = ToolBox(env, "utilities")                      # a new turn: this one's look-ups would repeat
    res = box.call("find_sites", {"purpose": "pump"})
    assert [(s["x"], s["y"]) for s in res["sites"]] == [(x, y)] and "note" not in res
    res = box.call("find_sites", {"purpose": "riverbank_2x2"})    # one free tile is no 2x2 block
    assert res["sites"] == [] and "2x2" in res["note"] and "taken" not in res["note"], res
    assert _blocked(env, "utilities", "riverbank_2x2")


def test_scarce_riverbank_lots_are_marked_for_pump_and_sewage():
    from civitas.fleet.tools import SiteList, ready_lines, scarce_bank
    env = _env()
    w = env.world
    taken = _bridge_and_take_the_bank(env)
    assert scarce_bank(w) == set()                       # no free bank lot at all
    for x, y in taken[:2]:
        w.struct[y, x] = C.NONE
    w.refresh()
    assert scarce_bank(w) == set(taken[:2])
    x, y = taken[2]
    w.struct[y, x] = C.NONE
    w.refresh()
    assert scarce_bank(w) == set()                       # three free lots: plenty
    bank, dry = {"x": 5, "y": 6, "homes_within_5": 0}, {"x": 9, "y": 9, "homes_within_5": 1}
    lists = {"coal_or_gas": SiteList([bank, dry], None, None), "near_homes": SiteList([bank, dry], None, None),
             "pump": SiteList([{"x": 5, "y": 6, "pollution": 0.1}], None, None)}
    lines = ready_lines("utilities", lists, {(5, 6)})
    assert lines == ["- coal_or_gas: 2 sites, best (5,6), 0 homes within 5 (riverbank: keep for pump/sewage)",
                     "- near_homes: 2 sites, best as coal_or_gas; next (9,9), 1 homes within 5",
                     "- pump: 1 site, same site as coal_or_gas"]
    assert "riverbank" not in ready_lines("utilities", lists)[0]
    assert ready_lines("health", {"landfill": SiteList([{"x": 5, "y": 6, "need": 3}], None, None)}, {(5, 6)}) == [
        "- landfill: 1 site, best (5,6), need 3 (riverbank: keep for pump/sewage)"]
    assert ready_lines("transport", {"harbour": SiteList([{"x": 5, "y": 6}], None, None)}, {(5, 6)}) == [
        "- harbour: 1 site, best (5,6)"]                  # a harbour belongs on the river


def test_sites_away_from_homes_say_when_every_roadside_lot_is_near_homes():
    from civitas.fleet.tools import _connected_roads, _touching, _why_empty
    env = _env()
    w = env.world
    assert ToolBox(env, "health").call("find_sites", {"purpose": "landfill"})["sites"]
    free = (w.struct == C.NONE) & (w.terrain == C.LAND)
    w.struct[free & ~_touching(_connected_roads(w))] = C.RES        # homes behind every roadside lot
    w.refresh()
    box = ToolBox(env, "health")
    for purpose in ("landfill", "cemetery"):
        res = box.call("find_sites", {"purpose": purpose})
        note = res["note"]
        assert res["sites"] == [] and "3 or more homes within 4 tiles" in note and "edge of town" in note, note
        assert "nothing needs" not in note and f"a {purpose} must stay away" in note
    lines = [ln for ln in _ready_sites(briefing(env, "health", mark=False)) if ln.startswith(("- landfill", "- cem"))]
    assert len(lines) == 2
    for line in lines:
        assert "away from homes" in line and "town edge" in line and len(line) <= 110, line
    short, note, blocked = _why_empty(w, "police", "jail")
    assert "away from homes" in short and "a jail must stay away" in note and blocked
    assert _why_empty(w, "health", "clinic")[::2] == ("nothing needs it right now", False)   # other purposes: unchanged


def _isolated_riverside_road(w):
    """A road at (24,15)-(24,17), off the highway network, touching the riverbank tile (25,17) on seed 0."""
    assert w.is_riverbank(25, 17) and w.struct[17, 25] == C.NONE
    w.struct[15:18, 24] = C.ROAD
    w.refresh()
    assert not w.networks[w.adjacent_network(25, 17)].outside_connected


def test_riverbank_sites_beside_unconnected_roads_are_not_listed():
    env = _env()
    w = env.world
    _isolated_riverside_road(w)
    from civitas.core.observation import build_observation
    obs = build_observation(w, "utilities", council_msgs=[], memory={}, scores={}, months_total=12,
                            include_map=False, site_hints=True)
    assert (25, 17) in [(s["x"], s["y"]) for s in obs["candidate_sites"]["riverbank_next_to_road"]]  # the hint has it
    box = ToolBox(env, "utilities")
    for purpose in ("pump", "sewage"):
        res = box.call("find_sites", {"purpose": purpose})
        assert res["sites"] == [] and "a road has to reach the river" in res["note"], res
    line, = [ln for ln in _ready_sites(briefing(env, "utilities", mark=False)) if ln.startswith("- pump:")]
    assert "no connected road reaches the river yet" in line, line


def test_listed_riverbank_sites_build_on_the_highway_network():
    env = _env()
    w = env.world
    _isolated_riverside_road(w)
    w.dept_funds["planning"] = w.dept_funds["utilities"] = 50000
    assert apply_action(w, "planning", {"type": "build_road", "x1": 19, "y1": 24, "x2": 38, "y2": 24}).ok
    box = ToolBox(env, "utilities")
    for purpose, action, kind in (("pump", "build_water", "pump"), ("sewage", "build_sewage", "outlet")):
        sites = box.call("find_sites", {"purpose": purpose})["sites"]
        assert sites and (25, 17) not in [(s["x"], s["y"]) for s in sites]
        s = sites[0]
        res = apply_action(w, "utilities", {"type": action, "kind": kind, "x": s["x"], "y": s["y"]})
        assert res.ok and "not connected" not in res.message, res.message


def _two_bridges(w):
    """Seed 0: highway-connected bridges on rows 24 and 34, with 8 free riverbank lots beside them."""
    w.dept_funds["planning"] = w.dept_funds["utilities"] = 50000
    for road in ({"x1": 19, "y1": 24, "x2": 38, "y2": 24}, {"x1": 20, "y1": 25, "x2": 20, "y2": 34},
                 {"x1": 21, "y1": 34, "x2": 38, "y2": 34}):
        assert apply_action(w, "planning", {"type": "build_road", **road}).ok
    return {(25, 23), (29, 23), (26, 25), (30, 25), (29, 33), (33, 33), (28, 35), (33, 35)}


def _bank_stubs(w, rows):
    """A road tile two west of the river on each row, off the highway: a riverbank hint beside it."""
    for y in rows:
        x = int(np.nonzero(w.terrain[y] == C.WATER)[0].min()) - 2
        assert apply_action(w, "planning", {"type": "build_road", "x1": x, "y1": y, "x2": x, "y2": y}).ok


def _where_all(sites):
    return [(s["x"], s["y"]) for s in sites]


def test_riverbank_lists_top_up_when_the_hints_name_unconnected_roads():
    """The observation cuts its riverbank lists to 6 before site_lists drops the sites beside roads off the highway:
    road stubs along the bank north and south left pump and sewage empty ("all taken") with 8 free lots beside the
    connected roads."""
    from civitas.fleet.tools import _site, role_obs
    env = _env()
    w = env.world
    lots = _two_bridges(w)
    hints = lambda key: role_obs(env, "utilities")["candidate_sites"][key]
    where = _where_all
    find = lambda purpose: ToolBox(env, "utilities").call("find_sites", {"purpose": purpose})
    for purpose, key in (("pump", "riverbank_next_to_road"), ("sewage", "riverbank_for_sewage")):
        sites = find(purpose)["sites"]                   # nothing dropped: the hints as they are
        assert sites == [_site(s) for s in hints(key)] and len(sites) == 6 and set(where(sites)) < lots
    _bank_stubs(w, range(2))                             # 3 unconnected lots north: the pump hints' first 3
    assert where(hints("riverbank_next_to_road"))[:3] == [(30, 0), (30, 1), (30, 2)]
    assert where(find("pump")["sites"]) == [(25, 23), (29, 23), (26, 25), (30, 25), (29, 33), (33, 33)]
    assert find("sewage")["sites"] == [_site(s) for s in hints("riverbank_for_sewage")]     # south first: untouched
    _bank_stubs(w, [*range(2, 10), *range(38, 48)])
    w.river_row[23] = 0.3                                # the river is dirty by row 23, and so is (30,25)
    w.pollution[25, 30] = 0.12
    for key in ("riverbank_next_to_road", "riverbank_for_sewage"):
        assert len(hints(key)) == 6 and not any(w.networks[s["network"]].outside_connected for s in hints(key))
    pump = find("pump")
    assert "note" not in pump and where(pump["sites"]) == [(26, 25), (29, 33), (33, 33), (28, 35), (33, 35), (30, 25)]
    assert [s["pollution"] for s in pump["sites"]] == [0, 0, 0, 0, 0, 0.12]           # cleanest, then north first
    sewage = find("sewage")["sites"]
    assert where(sewage) == [(28, 35), (33, 35), (29, 33), (33, 33), (26, 25), (30, 25)]     # south first
    ready = _ready_sites(briefing(env, "utilities", mark=False))
    assert next(ln for ln in ready if ln.startswith("- pump:")).startswith("- pump: 6 sites, best (26,25), pollution 0;")
    assert next(ln for ln in ready if ln.startswith("- sewage:")).startswith("- sewage: 6 sites, best (28,35);")
    res = apply_action(w, "utilities", {"type": "build_water", "kind": "pump", "x": 26, "y": 25})
    assert res.ok and "not connected" not in res.message, res.message
    res = apply_action(w, "utilities", {"type": "build_sewage", "kind": "outlet", "x": 28, "y": 35})
    assert res.ok and "not connected" not in res.message, res.message


def test_topped_up_bank_sites_rank_as_the_hints_do():
    """Topped-up lots are ranked as observation._candidate_sites ranks its hints: pumps +0.5 at or below an outlet,
    crowded spots last with "crowd" (the pump at (26,25) makes (25,23) lose 15%: it was the best pump site)."""
    env = _env()
    w = env.world
    _two_bridges(w)
    for action in ({"type": "build_water", "kind": "pump", "x": 26, "y": 25},
                   {"type": "build_sewage", "kind": "outlet", "x": 28, "y": 35}):
        assert apply_action(w, "utilities", action).ok
    _bank_stubs(w, [*range(10), *range(38, 48)])
    w.pollution[33, 33] = 0.2                            # dirtier than (33,35), which is below the outlet (+0.5)
    box = ToolBox(env, "utilities")
    pump = box.call("find_sites", {"purpose": "pump"})["sites"]
    assert _where_all(pump) == [(29, 33), (33, 33), (33, 35), (25, 23), (29, 23), (30, 25)]
    assert [s.get("crowd") for s in pump] == [None, None, None, 1, 1, 1]              # the pump's stretch last
    sewage = box.call("find_sites", {"purpose": "sewage"})["sites"]
    assert _where_all(sewage) == [(33, 35), (33, 33), (30, 25), (25, 23), (29, 23), (29, 33)]
    assert [s.get("crowd") for s in sewage] == [None] * 5 + [1]                       # the outlet's stretch last
    ready = _ready_sites(briefing(env, "utilities", mark=False))
    assert next(ln for ln in ready if ln.startswith("- pump:")).startswith("- pump: 6 sites, best (29,33), pollution 0;")


def test_riverbank_note_when_only_roads_line_the_bank():
    """A road that bridges the river counts as reaching it even when every bank tile beside the connected roads
    is itself road: the note must not send planning after a road that exists."""
    from civitas.fleet.tools import _touching
    env = _env()
    w = env.world
    w.dept_funds["planning"] = 5000
    assert apply_action(w, "planning", {"type": "build_road", "x1": 19, "y1": 24, "x2": 38, "y2": 24}).ok
    bank = (w.terrain == C.LAND) & _touching(w.terrain == C.WATER, diagonal=False)
    while True:                                     # pave every empty bank tile beside a road
        lots = bank & _touching(w.road_labels >= 0) & (w.struct == C.NONE)
        if not lots.any():
            break
        w.struct[lots] = C.ROAD
        w.refresh()
    res = ToolBox(env, "utilities").call("find_sites", {"purpose": "sewage"})
    assert res["sites"] == [] and "taken" in res["note"] and "a road has to reach" not in res["note"], res
    assert "(25,24)" in res["note"] and _blocked(env, "utilities", "sewage"), res


def _fill_roadside(w):
    ys, xs = np.nonzero(w.struct == C.ROAD)
    for x, y in zip(xs.tolist(), ys.tolist()):
        for nx, ny in ((x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            if w.in_bounds(nx, ny) and w.struct[ny, nx] == C.NONE and w.terrain[ny, nx] != C.WATER:
                w.struct[ny, nx] = C.PARK
    w.refresh()


def test_empty_site_lists_say_whether_land_or_need_is_missing():
    env = _env()                                       # free land, nobody living there yet: nothing needs a clinic
    res = ToolBox(env, "health").call("find_sites", {"purpose": "clinic"})
    assert res["sites"] == [] and "nothing needs this right now" in res["note"] and "planning" not in res["note"]
    res = ToolBox(env, "transport").call("find_sites", {"purpose": "bus_stops"})
    assert res["sites"] == [] and "too few homes" in res["note"]
    env = CivitasEnv(GameConfig(seed=0, months=12))    # residents, but every roadside tile taken
    agents = {r: SCRIPTED[r](r) for r in env.roles}
    for _ in range(6):
        for r in env.roles:
            env.apply(r, agents[r].act(env.observe(r)))
        env.end_month()
    w = env.world
    assert w.metrics()["population"] > 0
    _fill_roadside(w)
    from civitas.fleet.tools import _connected_roads
    ys, xs = np.nonzero(_connected_roads(w))
    span = f"highway-connected roads: x {xs.min()}-{xs.max()}, y {ys.min()}-{ys.max()}, {len(xs)} tiles"
    res = ToolBox(env, "education").call("find_sites", {"purpose": "school"})
    assert res["sites"] == [] and "ask planning" in res["note"] and span in res["note"]
    res = ToolBox(env, "planning").call("find_sites", {"purpose": "zone_R"})
    assert res["sites"] == [] and "find_sites road_extension" in res["note"] and "ask planning" not in res["note"]
    from civitas.fleet.tools import _why_empty
    note = _why_empty(w, "planning", "road_extension")[1]
    assert "find_sites" not in note and "build_road" in note                   # not circular
    res = ToolBox(env, "mayor").call("find_sites", {"purpose": "school"})     # the mayor names the department
    assert not res["ok"] and "education.school" in res["error"] and "utilities.sewage" in res["error"]


def test_empty_site_lists_say_whether_land_or_roads_block_them():
    """A list empty for lack of land or roads is blocked (planning can fix it: DeptMoney keeps money for it, the mayor's
    line and planning's Ready sites say so); nothing needing it, a SITE_NOTES reason, no river or a lock isn't."""
    from civitas.fleet.tools import _why_empty, role_obs, site_lists
    env = _env()
    blocked = lambda role: sorted(p for p, lst in site_lists(env.world, role, role_obs(env, role)).items()
                                  if lst.blocked)
    assert blocked("utilities") == ["pump", "sewage"]            # a fresh map: no road reaches the river
    assert blocked("police") == blocked("fire") == blocked("health") == blocked("transport") == []   # need, notes,
                                                                                                      # locks
    env.world.dept_funds["planning"] = 5000
    assert apply_action(env.world, "planning", {"type": "build_road", "x1": 19, "y1": 24, "x2": 38, "y2": 24}).ok
    assert blocked("utilities") == []                            # across the river: free bank lots by the road
    env = CivitasEnv(GameConfig(seed=0, months=12))              # residents, then every roadside tile taken
    agents = {r: SCRIPTED[r](r) for r in env.roles}
    for _ in range(6):
        for r in env.roles:
            env.apply(r, agents[r].act(env.observe(r)))
        env.end_month()
    _fill_roadside(env.world)
    assert blocked("police") == ["station"] and blocked("education") == ["school"]     # jail, library: locked
    assert blocked("health") == ["cemetery", "clinic", "landfill"] and blocked("fire") == ["shelter", "station"]
    assert blocked("transport") == []                            # bus stops: a SITE_NOTES reason (population)
    w = _road_world((1, 24, 20, 24))
    w.terrain[:] = C.LAND
    assert _why_empty(w, "utilities", "pump") == ("no river on this map", "this map has no river", False)


def test_a_riverbank_plant_the_city_has_room_for_is_not_blocked(env12, grown_env, monkeypatch):
    """Smoke v14 seed 1, months 6 and 9: pump and outlet built, water 370-490 and sewage 250 against demand of 99-116,
    the bank lots beside the roads taken; planning read "blocked for lack of land: utilities (pump, sewage) ... dezone
    a roadside lot for them", and its playbook frees a bank lot when asked. A pump or outlet counts while the city
    lacks one or uses 90% of its supply; utilities' own lists still say the bank is taken."""
    import civitas.fleet.tools as T
    from civitas.fleet.tools import land_blocked, role_obs, site_lists
    env = copy.deepcopy(env12)                         # month 12: a pump and an outlet, 250 each against 125
    w = env.world
    m = w.metrics()
    assert w.facilities(C.PUMP) and w.facilities(C.OUTLET) and not free_bank(w).any()
    assert m["water_demand"] < 0.9 * m["water_supply"] and m["sewage_demand"] < 0.9 * m["sewage_supply"]
    lists = site_lists(w, "utilities", role_obs(env, "utilities"))
    for p in ("pump", "sewage"):
        assert lists[p].blocked and lists[p].short.startswith("riverbank beside roads all taken"), lists[p]
    assert land_blocked(env) == {}
    real = T.site_lists                                # nothing else to build: nothing kept for the bank
    monkeypatch.setattr(T, "site_lists", lambda w, role, obs: {
        p: lst if p in ("pump", "sewage") else T.SiteList([], "nothing needs it right now", "", False)
        for p, lst in real(w, role, obs).items()})
    assert T.first_build(env, "utilities") is None     # was the $1,500 pump kept, "blocked: a sewage outlet"
    monkeypatch.undo()
    (x, y), = w.facilities(C.OUTLET)                   # no outlet: the bank is needed again
    w.remove_building(x, y)
    w.struct[y, x] = C.RES
    w.refresh()
    assert land_blocked(env) == {"utilities": ["sewage"]}
    m = grown_env.world.metrics()                      # month 24: water at capacity, sewage 313 of 500
    assert m["water_demand"] >= 0.9 * m["water_supply"] and m["sewage_demand"] < 0.9 * m["sewage_supply"]
    assert land_blocked(grown_env) == {"utilities": ["pump"]}


def _one_roadside_lot(w):
    """Every roadside tile taken but one, hemmed in: free roadside land, but no free 2x2 block beside a road."""
    from civitas.fleet.tools import _connected_roads, _touching
    _fill_roadside(w)
    ys, xs = np.nonzero((w.struct == C.PARK) & _touching(_connected_roads(w)))
    x, y = int(xs[0]), int(ys[0])
    w.struct[y, x] = C.NONE
    for ny in range(y - 1, y + 2):
        for nx in range(x - 1, x + 2):
            if w.in_bounds(nx, ny) and (nx, ny) != (x, y) and w.struct[ny, nx] == C.NONE:
                w.struct[ny, nx] = C.PARK
    w.refresh()


def test_two_by_two_lists_say_when_no_free_block_is_left(open_env):
    """The hospital and university lists rank free 2x2 roadside blocks with need +0.01 everywhere (observation's
    size=2 site_candidates), so they go empty only for lack of a block, but read "nothing needs it right now". Review
    probe, month 60 (6,564 residents): education's keep fell from its $12,000 university to its $5,000 high school,
    and $7,000 read idle."""
    from civitas.fleet.tools import DeptMoney, _free_roadside, _why_empty, blocked_firsts, role_obs, site_lists
    env = copy.deepcopy(open_env)                      # every milestone reached
    w = env.world
    assert _why_empty(w, "education", "university")[::2] == ("nothing needs it right now", False)   # a block is free
    _one_roadside_lot(w)
    assert _free_roadside(w) == 1 and not w.site_candidates(np.ones(w.struct.shape), radius=1, k=1, size=2)
    lists = {r: site_lists(w, r, role_obs(env, r)) for r in ("health", "education")}
    for role, purpose in (("health", "hospital"), ("education", "university")):
        lst = lists[role][purpose]
        assert (lst.sites, lst.short, lst.blocked) == ([], "no free 2x2 roadside block: ask planning for one", True)
        assert "2x2" in lst.note and "ask planning (send_message)" in lst.note, lst.note
    assert "hospital" not in blocked_firsts(w, "health", lists["health"])       # the clinic's tier 2: an upgrade
    assert "university" in blocked_firsts(w, "education", lists["education"])
    w.school_cov[:] = w.high_cov[:] = 1.0              # schools need no lot: the university is the cheapest blocked
    line, = [ln for ln in _ready_sites(briefing(env, "planning", mark=False)) if ln.startswith("- blocked")]
    assert "education (university: a free 2x2 lot)" in line, line
    _held(env, education=(12000, 0))
    m = DeptMoney(env, "education")
    assert (m.keep, m.idle) == (12000, 0)


def _hem_homes(env, keep=(), road=True):
    """Seed 0, month 12 (no service built yet): a road south into open land (x=19, y 30-46; `road`), then every
    roadside tile within 8 of a developed one parked but `keep`: free roadside land far from the homes, none near
    them."""
    from civitas.fleet.tools import _roadside
    from civitas.sim.grid import box_sum
    w = env.world
    w.dept_funds["planning"] += 5000
    if road:
        assert apply_action(w, "planning", {"type": "build_road", "x1": 19, "y1": 30, "x2": 19, "y2": 46}).ok
    near = _roadside(w) & (box_sum(np.isin(w.struct, C.ZONES) & (w.level > 0), 8) > 0)
    for x, y in keep:
        near[y, x] = False
    w.struct[near] = C.PARK
    w.refresh()


NEAR = "no free roadside lot near the homes that need it (around ("
NEED_LISTS = (("police", "station"), ("fire", "station"), ("health", "clinic"), ("education", "school"))
COV_GRIDS = ("police_cov", "fire_cov", "health_cov", "school_cov", "high_cov")


def test_need_ranked_lists_say_when_the_lots_near_homes_are_taken(env12):
    """Smoke v14 seed 0, months 3-12: the mayor read "fire: needs $2,500 for a fire station (4 ready sites)" and funded
    fire, health and education; planning, acting next, zoned the lots near homes; they read "none - nothing needs it
    right now" at coverage 0 and burned 13-20 calls a turn searching, and fire held $2,500 to month 12 unbuilt. These
    lists only rank free roadside tiles near uncovered need: empty with free land elsewhere, the need is real."""
    import re
    from civitas.fleet.tools import DeptMoney, _free_roadside, role_obs, site_lists
    from civitas.sim.grid import box_sum
    env = copy.deepcopy(env12)                         # 916 residents, no station, clinic or school
    _hem_homes(env)
    w = env.world
    assert _free_roadside(w) > 0
    for role, purpose in NEED_LISTS:
        lst = site_lists(w, role, role_obs(env, role))[purpose]
        assert lst.sites == [] and lst.blocked and lst.short.startswith(NEAR), (role, purpose, lst)
        assert lst.short.endswith("): ask planning for one") and "ask planning (send_message)" in lst.note, lst
    line, = [ln for ln in _ready_sites(briefing(env, "fire", mark=False)) if ln.startswith("- station:")]
    x, y = map(int, re.search(r"around \((\d+),(\d+)\)", line).groups())
    dev = np.isin(w.struct, C.ZONES) & (w.level > 0)
    assert box_sum(dev, 2)[y, x] > 0 and len(line) <= 110, line                  # among the homes it would cover
    _held(env, fire=(2500, 0))
    m = DeptMoney(env, "fire")                         # was idle $1,500 above the $1,000 floor
    assert (m.keep, m.idle, m.blocked) == (2500, 0, ("a fire station",))
    plan = lambda: [ln for ln in _ready_sites(briefing(env, "planning", mark=False)) if ln.startswith("- blocked")]
    tail = ": extend a road into open land, or free a lot, for them"
    assert plan() == ["- blocked for lack of land: police (police station: near homes), fire (fire station: near "
                      "homes), health (clinic: near homes), education (elementary school: near homes)" + tail]
    assert len(plan()[0]) == 223
    import civitas.fleet.briefing as B
    B.BLOCKED_LINE_CHARS, keep = 200, B.BLOCKED_LINE_CHARS
    try:                                               # too long with the tags: without them
        assert plan() == ["- blocked for lack of land: police (police station), fire (fire station), health (clinic), "
                          "education (elementary school)" + tail]
    finally:
        B.BLOCKED_LINE_CHARS = keep


def test_need_ranked_lists_with_little_need_left_need_nothing(env12):
    """Everyone covered, or all but one building of 61: still "nothing needs it right now", not blocked. Real need is
    15% of what the building serves under 0.3 coverage (the observation's uncovered_buildings test): review, scripted
    month 120 (fire score 87.4) with the lots near homes taken, 1 - coverage summed to 11.5% for fire, 15.2% for
    clinics and 15.9% for police (coverage fades from 60% of the radius and with congestion), and all three read
    blocked. The shelter is supplementary: never asked for."""
    from civitas.fleet.tools import NEAR_NEED, NEED_SHARE, _why_empty
    assert NEED_SHARE == 0.15 and ("fire", "shelter") not in NEAR_NEED
    env = copy.deepcopy(env12)
    _hem_homes(env)
    w = env.world
    for value in (1.0, 0.3):                           # covered, or thinly everywhere (70% "uncovered" before)
        for grid in COV_GRIDS:
            getattr(w, grid)[:] = value
        for role, purpose in NEED_LISTS:
            short, note, blocked = _why_empty(w, role, purpose)
            assert (short, blocked) == ("nothing needs it right now", False), (role, purpose)
            assert "under 15% of the" in note and "poorly covered" in note, note
    dev = np.isin(w.struct, C.ZONES) & (w.level > 0)
    ys, xs = np.nonzero(dev)
    w.fire_cov[ys[:9], xs[:9]] = 0.0                   # 9 buildings of 61: under 15%
    assert _why_empty(w, "fire", "station")[::2] == ("nothing needs it right now", False)
    w.fire_cov[ys[:10], xs[:10]] = 0.29                # 10 of 61: 16%
    assert _why_empty(w, "fire", "station")[0].startswith(NEAR)


def test_a_well_covered_department_with_no_free_land_is_not_blocked(env12):
    """The same need gate on the "no free roadside land" path: a well-covered department isn't blocked for want of a
    lot (it keeps no money for one, and planning isn't told to free lots)."""
    from civitas.fleet.tools import _why_empty
    env = copy.deepcopy(env12)
    _fill_roadside(env.world)
    w = env.world
    for role, purpose in NEED_LISTS:                   # coverage 0: blocked
        assert _why_empty(w, role, purpose)[::2] == ("no free roadside land: ask planning for a lot", True)
    for grid in COV_GRIDS:
        getattr(w, grid)[:] = 0.5
    for role, purpose in NEED_LISTS:
        assert _why_empty(w, role, purpose)[::2] == ("nothing needs it right now", False), (role, purpose)
    assert _why_empty(w, "health", "landfill")[2]      # not a need-ranked list: blocked as before


@pytest.mark.slow
def test_a_mature_city_with_the_lots_near_homes_taken():
    """Review probe, scripted seed 0 month 120 (14,092 residents; fire 87.4, health 86.8, police 69.9) with every
    roadside lot within 8 of a building taken: police, fire and clinics read blocked near homes. Now only the schools,
    with 22% of children and teenagers under 0.3 coverage."""
    from civitas.fleet.tools import role_obs, site_lists
    env = CivitasEnv(GameConfig(seed=0, months=120))
    agents = {r: SCRIPTED[r](r) for r in env.roles}
    for _ in range(120):
        for r in env.roles:
            env.apply(r, agents[r].act(env.observe(r)))
        env.end_month()
    _hem_homes(env, road=False)
    blocked = {(d, p) for d, p in NEED_LISTS + (("education", "high_school"),)
               if site_lists(env.world, d, role_obs(env, d))[p].blocked}
    assert blocked == {("education", "school"), ("education", "high_school")}


def test_road_span_ignores_unconnected_roads_and_no_river_maps():
    from civitas.fleet.tools import _river_span, _roads, _roads_text, _why_empty
    env = _env()
    w = env.world
    before = _roads_text(_roads(w))
    w.struct[2, 44] = w.struct[2, 45] = C.ROAD           # a stray road far from the network
    w.refresh()
    assert _roads_text(_roads(w)) == before and "x 44" not in before
    w.terrain[w.terrain == C.WATER] = C.LAND
    assert _river_span(w) is None
    assert _why_empty(w, "utilities", "pump")[1] == "this map has no river"


def test_map_facts_on_a_fresh_world():
    from civitas.fleet.tools import map_facts, map_lines
    env = _env()
    w = env.world
    f = map_facts(w)
    assert f["size"] == [48, 48]
    (hx, hy), = [(int(x), int(y)) for y, x in zip(*np.nonzero(w.struct == C.HIGHWAY))]
    assert hx == 0 and f["highway"] == [{"x": 0, "y": hy, "edge": "west"}]
    assert len(f["river"]) == 3
    for band in f["river"]:
        for x0, x1, y in band:                                                 # the real columns at the band's ends
            cols = np.nonzero(w.terrain[y] == C.WATER)[0]
            assert (x0, x1) == (cols.min(), cols.max())
    ends = [e[2] for band in f["river"] for e in band]
    assert ends[0] == 0 and ends[-1] == 47 and ends == sorted(ends)
    assert all(b[0][2] == a[1][2] + 1 for a, b in zip(f["river"], f["river"][1:]))   # bands tile the rows
    ys, xs = np.nonzero((w.struct == C.ROAD) | (w.struct == C.HIGHWAY))        # only the starter road
    assert f["roads"] == {"x": [0, int(xs.max())], "y": [hy, hy], "tiles": len(xs)}
    tx, ty = w.facilities(C.TOWNHALL)[0]
    assert f["built"] == {"x": [tx, tx], "y": [ty, ty], "zoned": 0, "developed": 0, "buildings": 1}
    lines = map_lines(f)
    text = "\n".join(lines)
    assert len(lines) <= 6 and len(text) <= 400, (len(lines), len(text))
    assert f"(0,{hy}) on the west edge" in text and "x = column 0-47" in text
    (a0, a1, _), (b0, b1, by) = f["river"][0]
    assert f"river: x {a0}-{a1} at row 0 to x {b0}-{b1} at row {by}," in text
    assert f"highway-connected roads: x 0-{int(xs.max())}, y {hy}, {len(xs)} tiles" in text
    assert f["road_lines"] == [{"y": hy, "x": [0, int(xs.max())]}]                  # the highway tile starts it
    assert lines[lines.index(f"- highway-connected roads: x 0-{int(xs.max())}, y {hy}, {len(xs)} tiles") + 1] == \
        f"- road lines: y={hy} x 0-{int(xs.max())}"


def test_map_facts_on_an_empty_map():
    from civitas.fleet.tools import map_facts, map_lines
    env = _env()
    w = env.world
    w.struct[:] = C.NONE
    w.anchor[:] = -1
    w.refresh()
    f = map_facts(w)
    assert f["highway"] == [] and f["roads"] is None and f["built"] is None and f["road_lines"] == []
    assert map_lines(f)[1:] == ["- highway entry: none", map_lines(f)[2], "- no roads connected to the highway yet",
                                "- built-up area: none yet"]


def test_map_facts_grow_and_without_a_river(grown_env):
    from civitas.fleet.tools import map_facts, map_lines
    fresh, grown = map_facts(_env().world), map_facts(grown_env.world)
    assert grown["roads"]["tiles"] > fresh["roads"]["tiles"]
    r = grown["roads"]
    assert r["x"][1] - r["x"][0] >= fresh["roads"]["x"][1] - fresh["roads"]["x"][0] and r["y"][0] < r["y"][1]
    b = grown["built"]
    assert b["zoned"] > 0 and b["developed"] > 0 and b["buildings"] > 1 and b["x"][0] < b["x"][1]
    assert len(grown["road_lines"]) > len(fresh["road_lines"])
    assert len("\n".join(map_lines(grown))) <= 550             # 400 + the road lines (8 named at most: ~135 chars)
    env = _env()
    env.world.terrain[env.world.terrain == C.WATER] = C.LAND
    f = map_facts(env.world)
    assert f["river"] == [] and "- no river" in map_lines(f)


def _road_world(*rects):
    """Seed 0's map cleared to its highway tile (0,24) and these road rectangles (x1, y1, x2, y2, inclusive)."""
    w = _env().world
    w.struct[:] = C.NONE
    w.anchor[:] = -1
    w.struct[24, 0] = C.HIGHWAY
    for x1, y1, x2, y2 in rects:
        w.struct[y1:y2 + 1, x1:x2 + 1] = C.ROAD
    w.refresh()
    return w


def _line_tiles(ln: dict) -> set:
    if isinstance(ln["x"], list):
        return {(x, ln["y"]) for x in range(ln["x"][0], ln["x"][1] + 1)}
    return {(ln["x"], y) for y in range(ln["y"][0], ln["y"][1] + 1)}


def test_map_block_names_the_road_lines():
    """Smoke v9-v11: the Map block gave the network's box ("x 0-25, y 19-29") and planning rebuilt existing road 13
    times ("every tile on that line is already road") and made 5.4 inspect calls a turn; the network was three
    lines, y=19, y=24 and x=19."""
    from civitas.fleet.tools import _connected_roads, map_facts, map_lines
    w = _road_world((1, 24, 47, 24),                      # the main road to the east edge, bridging the river
                    (40, 25, 40, 47),                     # a street to the south edge
                    (19, 19, 19, 29), (3, 19, 25, 19),    # a cross street, and a road it links to the network
                    (5, 25, 5, 26), (7, 25, 7, 25),       # stubs of 2 and 1 tiles off the main road
                    (2, 40, 12, 40))                      # a road off the network
    w.road_class[19, 3:10] = C.ROAD_CLASSES["avenue"]
    w.refresh()
    assert (w.terrain[24, 26:29] == C.WATER).all() and (w.struct[24, 26:29] == C.ROAD).all()
    f = map_facts(w)
    main, cross = {"y": 24, "x": [0, 47]}, {"x": 19, "y": [19, 29]}
    assert f["road_lines"] == [main, {"x": 40, "y": [24, 47]}, {"y": 19, "x": [3, 25]}, cross,
                               {"x": 5, "y": [24, 26]}]                                       # 3 with its junction
    assert [ln for ln in f["road_lines"] if (19, 24) in _line_tiles(ln)] == [main, cross]      # a crossing: both
    named = set().union(*map(_line_tiles, f["road_lines"]))
    assert all(_connected_roads(w)[y, x] for x, y in named)
    assert (7, 25) not in named and not named & {(x, 40) for x in range(2, 13)} and f["unconnected"]["tiles"] == 11
    lines = map_lines(f)
    i = next(i for i, ln in enumerate(lines) if ln.startswith("- highway-connected roads: x 0-47, y 19-47, "))
    assert lines[i + 1:i + 3] == ["- road lines: y=24 x 0-47; x=40 y 24-47; y=19 x 3-25; x=19 y 19-29; x=5 y 24-26",
                                  "- roads not connected to the highway: 11 tiles (x 2-12, y 40) - reconnect them"]


def test_road_lines_are_capped_longest_first():
    from civitas.fleet.tools import ROAD_LINE_MIN, ROAD_LINES, map_facts, map_lines
    assert (ROAD_LINE_MIN, ROAD_LINES) == (3, 8)
    streets = [(3, y, 6, y) for y in range(4, 45, 4) if y != 24]             # 10 side streets off a spine down x 2
    f = map_facts(_road_world((1, 24, 10, 24), (2, 4, 2, 44), *streets))
    assert len(f["road_lines"]) == 12
    road_lines, = [ln for ln in map_lines(f) if ln.startswith("- road lines: ")]
    assert road_lines == ("- road lines: x=2 y 4-44; y=24 x 0-10; y=4 x 2-6; y=8 x 2-6; y=12 x 2-6; y=16 x 2-6; "
                          "y=20 x 2-6; y=28 x 2-6 +4 more")
    # equal lengths: the first tile's y, then x, then rows before columns
    f = map_facts(_road_world((1, 24, 9, 24), (4, 15, 4, 23), (9, 20, 13, 20), (9, 21, 9, 23), (13, 21, 13, 24)))
    assert f["road_lines"] == [{"x": 4, "y": [15, 24]}, {"y": 24, "x": [0, 9]}, {"y": 20, "x": [9, 13]},
                               {"x": 9, "y": [20, 24]}, {"x": 13, "y": [20, 24]}]


def _map_body(res):
    return [row[4:] for row in res["map"][2:]]


def test_map_overview_fits_and_has_rulers_and_legend(grown_env):
    from civitas.fleet.tools import RESULT_CHARS, map_facts, map_lines
    for role in TURN_ORDER:
        res = ToolBox(grown_env, role).call("map_overview", {})
        assert res["ok"] and len(json.dumps(res)) <= RESULT_CHARS, (role, len(json.dumps(res)))
        assert "scale" not in res and "truncated" not in res                     # 48x48 fits at full size
    rows = res["map"]
    assert rows[0].strip() == "".join(str(x // 10) for x in range(48))
    assert rows[1].strip() == "".join(str(x % 10) for x in range(48))
    assert rows[2].startswith("  0 ") and rows[-1].startswith(" 47 ") and len(rows) == 50
    assert _map_body(res) == [grown_env.world.ascii_map().split("\n")[y + 2][4:] for y in range(48)]
    assert res["legend"].startswith("Legend: ")
    keys = {c for item in res["legend"][8:].split("  ") for c in item.split(" ", 1)[0].split("/")}
    used = set("".join(_map_body(res)))
    assert used <= keys and {"#", "~", "H", "T", "R"} <= used
    assert "E coal" not in res["legend"] or "E" in used                       # only what is on the map
    assert res["summary"] == [ln[2:] for ln in map_lines(map_facts(grown_env.world))]
    assert "higher tiers keep the letter" in res["read"] and "higher tiers" not in res["legend"]
    from civitas.fleet.tools import map_overview
    tight = map_overview(grown_env.world, limit=len(json.dumps(res)) - 1)     # a full legend: the summary shrinks
    assert tight["summary"] != res["summary"] and tight["map"] == res["map"] and "scale" not in tight
    tight = map_overview(grown_env.world, limit=len(json.dumps({**res, "summary": ["see the Map block in your "
                                                                                    "briefing"]})))
    assert tight["summary"] == ["see the Map block in your briefing"] and tight["map"] == res["map"], tight["summary"]


def test_the_mayor_map_overview_fallback_lists_what_its_briefing_leaves_out(grown_env, monkeypatch):
    """When the summary doesn't fit, map_overview points to the briefing's Map block; the mayor's leaves out size,
    highway entry, river and road lines (MAYOR_MAP), so its fallback lists those."""
    import civitas.fleet.tools as T
    from civitas.fleet.tools import MAP_PARTS, MAYOR_MAP, map_facts, map_lines, map_overview
    w = grown_env.world
    facts = map_facts(w)
    lines = lambda parts: [ln[2:] for ln in map_lines(facts, parts)]
    full = map_overview(w, parts=MAYOR_MAP)
    assert full["summary"] == lines(MAP_PARTS)                               # it fits: everything
    rest = "the rest: see the Map block in your briefing"
    tight = map_overview(w, limit=len(json.dumps(full)) - 1, parts=MAYOR_MAP)
    assert tight["summary"] == lines(("size", "highway", "river", "road_lines")) + [rest]
    assert tight["map"] == full["map"] and "scale" not in tight
    tighter = map_overview(w, limit=len(json.dumps(tight)) - 1, parts=MAYOR_MAP)     # then fewer road lines
    n = min(len(facts["road_lines"]), T.ROAD_LINES)
    assert n > 1 and tighter["summary"] == [ln[2:] for ln in map_lines(facts, ("size", "highway", "river",
                                                                               "road_lines"), n - 1)] + [rest]
    none = [ln[2:] for ln in map_lines(facts, ("size", "highway", "river"))] + [rest]
    limit = len(json.dumps({**tighter, "summary": none}))                           # room for none of them
    assert map_overview(w, limit=limit, parts=MAYOR_MAP)["summary"] == none
    seen, real = [], T.map_overview
    monkeypatch.setattr(T, "map_overview", lambda w, **k: seen.append(k.get("parts")) or real(w, **k))
    ToolBox(grown_env, "mayor").call("map_overview", {})
    ToolBox(grown_env, "police").call("map_overview", {})
    assert seen == [MAYOR_MAP, MAP_PARTS]


@pytest.fixture(scope="module")
def env72():
    """Scripted seed 0 at month 72 of a 120-month game (about 2 s)."""
    env = CivitasEnv(GameConfig(seed=0, months=120))
    agents = {r: SCRIPTED[r](r) for r in env.roles}
    for _ in range(72):
        for r in env.roles:
            env.apply(r, agents[r].act(env.observe(r)))
        env.end_month()
    return env


def test_map_overview_drops_road_lines_before_the_summary(env72):
    """Scripted seed 0: from month 66 the map, legend and summary no longer fit in one result (3,803 of 3,800
    characters at month 72, 3,894 at month 120: the legend grows with the city), and a department's map_overview gave
    only "see the Map block in your briefing". Now the road lines go first, the longest kept while they fit (month 72:
    7 of 12, 3,790), then all of them (month 120: 3,768), and only then the summary."""
    from civitas.fleet.tools import RESULT_CHARS, ROAD_LINES, map_facts, map_lines, map_overview
    w = env72.world
    facts = map_facts(w)
    full = [ln[2:] for ln in map_lines(facts)]
    road = lambda summary: [ln for ln in summary if ln.startswith("road lines: ")]
    rest = [ln for ln in full if not road([ln])]
    res = ToolBox(env72, "planning").call("map_overview", {})
    assert res["ok"] and len(json.dumps(res)) <= RESULT_CHARS and "scale" not in res
    assert len(json.dumps({**res, "summary": full})) > RESULT_CHARS                  # the whole summary doesn't fit
    kept, = road(res["summary"])
    n = kept.count(";") + 1
    assert 0 < n < ROAD_LINES < len(facts["road_lines"]), kept
    assert res["summary"] == [ln[2:] for ln in map_lines(facts, road_lines=n)]      # in place, "+N more" at the end
    assert kept.endswith(f" +{len(facts['road_lines']) - n} more")
    steps, limit = [], RESULT_CHARS                                                  # tighter and tighter: the steps
    while "scale" not in (r := map_overview(w, limit=limit)):
        steps.append(r["summary"])
        limit = len(json.dumps(r)) - 1
    assert steps[0] == res["summary"]
    assert [road(s)[0].count(";") + 1 for s in steps[:-2]] == list(range(n, 0, -1))  # one road line fewer each time
    assert steps[-2:] == [rest, ["see the Map block in your briefing"]]


def test_map_overview_downsamples_big_maps():
    from civitas.fleet.tools import RESULT_CHARS
    env = CivitasEnv(GameConfig(seed=0, months=12, width=96, height=80))
    res = ToolBox(env, "planning").call("map_overview", {})
    assert res["ok"] and len(json.dumps(res)) <= RESULT_CHARS and "truncated" not in res
    k = int(res["scale"].split("x")[0].split()[-1])
    assert k >= 2 and "inspect" in res["scale"]
    rows = res["map"]
    assert rows[2].startswith("  0 ") and rows[3].startswith(f"{k:>3} ")
    assert len(rows) - 2 == -(-80 // k) and len(rows[2]) - 4 == -(-96 // k)
    assert any("~" in r for r in rows) and any("H" in r for r in rows)


def test_downsampled_blocks_break_ties_by_reading_order():
    from civitas.fleet.tools import _ascii
    w = _env().world
    w.struct[0, 0] = w.struct[1, 1] = C.COM                    # c r / r c: a 2-2 tie, 'c' comes first
    w.struct[0, 1] = w.struct[1, 0] = C.RES
    w.struct[0, 2] = w.struct[0, 3] = C.ROAD                   # # # / . c: the road wins 2-1
    w.struct[1, 3] = C.COM
    assert _ascii(w, 2)[2][4:6] == "c#"
    w.struct[0, 0], w.struct[0, 1] = C.RES, C.COM              # r c / r c: now 'r' comes first
    assert _ascii(w, 2)[2][4] == "r"


def _check_mayor_sites(env):
    mayor = ToolBox(env, "mayor")
    for dept in DEPARTMENTS:
        box = ToolBox(env, dept)
        for purpose in SITE_PURPOSES[dept]:
            for near in ({}, {"near_x": 30, "near_y": 12}):
                mine = mayor.call("find_sites", {"purpose": f"{dept}.{purpose}", **near})
                theirs = box.call("find_sites", {"purpose": purpose, **near})
                assert mine["ok"] and mine["purpose"] == f"{dept}.{purpose}", (dept, purpose, mine)
                assert mine.get("sites") == theirs.get("sites") and mine.get("stops") == theirs.get("stops"), \
                    (dept, purpose)
                assert mine.get("note") == (theirs["note"] and f"({dept} department's note) {theirs['note']}"
                                            if "note" in theirs else None), (dept, purpose)


def test_mayor_find_sites_is_every_department_s_list():
    env = _env()
    tools = {s["function"]["name"]: s["function"] for s in ToolBox(env, "mayor").schemas()}
    fs = tools["find_sites"]
    assert fs["parameters"]["properties"]["purpose"]["enum"] == [f"{d}.{p}" for d in DEPARTMENTS
                                                                 for p in SITE_PURPOSES[d]]
    assert "read-only" in fs["description"]
    _check_mayor_sites(env)
    res = ToolBox(env, "mayor").call("find_sites", {"purpose": "utilities.sewage", "near_x": 5})
    assert not res["ok"] and "near_y" in res["error"]
    for bad in ("sewage", "utilities.school", "mayor.x"):
        res = ToolBox(env, "mayor").call("find_sites", {"purpose": bad})
        assert not res["ok"] and "planning.zone_R" in res["error"] and "education.school" in res["error"], bad


def test_find_sites_and_lookups_on_a_grown_world(grown_env):
    env = grown_env
    _check_sites(env)
    _check_mayor_sites(env)
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


def test_update_plan_lists_the_departments_below_their_target():
    env = _env()
    w = env.world
    w.treasury = 30000
    w.dept_funds.update(planning=9500, utilities=2000)
    mayor = ToolBox(env, "mayor")
    res = mayor.call("update_plan", {"goals": ["grow"]})
    assert res["ok"] and "below_target" not in res                                  # no targets yet
    res = mayor.call("update_plan", {"budgets": {"planning": 9000, "utilities": 7000, "fire": 2500, "police": 0}})
    assert res["ok"] and res["message"].startswith("plan v2 saved")
    assert res["below_target"] == {"utilities": {"funds": 2000, "target": 7000, "gap": 5000},
                                   "fire": {"funds": 0, "target": 2500, "gap": 2500}}      # planning is funded
    assert res["treasury"] == 30000
    assert res["next"].startswith("to spend: $27k above the $3,000 reserve; ")    # needs-based: test_fleet_money
    assert mayor.call("allocate_budget", {"department": "utilities", "amount": 5000})["ok"]
    res = mayor.call("update_plan", {"priorities": {"fire": "stations by homes"}})
    assert res["below_target"] == {"fire": {"funds": 0, "target": 2500, "gap": 2500}}
    assert res["treasury"] == 25000


def test_update_plan_says_what_there_is_to_spend():
    env = _env()
    w = env.world
    w.treasury = 12000                                   # the treasury above the $3,000 reserve, whatever the gaps
    res = ToolBox(env, "mayor").call("update_plan", {"budgets": {"planning": 6000, "utilities": 3000}})
    assert res["to_spend"] == 9000 and res["next"].startswith("to spend: $9,000 above the $3,000 reserve; ")
    w.treasury = 11000
    res = ToolBox(env, "mayor").call("update_plan", {"budgets": {"planning": 6000, "utilities": 3000}})
    assert res["to_spend"] == 8000 and res["next"].startswith("to spend: $8,000 above the $3,000 reserve; ")
    assert res["below_target"] == {"planning": {"funds": 0, "target": 6000, "gap": 6000},
                                   "utilities": {"funds": 0, "target": 3000, "gap": 3000}}


def test_health_finds_cemetery_sites():
    env = _env()
    w = env.world
    w.dept_funds["health"] = 5000
    box = ToolBox(env, "health")
    res = box.call("find_sites", {"purpose": "cemetery"})
    assert res["ok"] and res["sites"] == box.call("find_sites", {"purpose": "landfill"})["sites"]
    s = res["sites"][0]
    assert box.call("build_cemetery", {"x": s["x"], "y": s["y"]})["ok"]
    line, = [ln for ln in briefing(env, "health", mark=False).splitlines() if ln.startswith("- cemetery:")]
    assert "same list as landfill" in line


def test_education_finds_library_sites(grown_env):
    line = lambda env: next(ln for ln in _ready_sites(briefing(env, "education", mark=False))
                            if ln.startswith("- library:"))
    assert line(_env()) == f"- library: locked (unlocks at {C.UNLOCK_POP['library']:,} residents)"
    env = copy.deepcopy(grown_env)                     # 2,6xx residents: unlocked
    w = env.world
    box = ToolBox(env, "education")
    res = box.call("find_sites", {"purpose": "library"})
    sites = res["sites"]
    assert sites and sites == box.call("find_sites", {"purpose": "school"})["sites"] and "note" not in res
    assert _building_cost(w, "education", "library") == C.TIER_LINES["library"][0]["cost"] == 1500
    w.dept_funds["education"] = 1500
    assert line(env) == f"- library: {len(sites)} sites, same list as school; $1,500"
    assert box.call("build_library", {"x": sites[0]["x"], "y": sites[0]["y"]})["ok"]
    assert w.struct[sites[0]["y"], sites[0]["x"]] == C.LIBRARY and w.dept_funds["education"] == 0


def test_update_plan_says_hold_under_the_reserve():
    env = _env()
    from civitas.fleet.fmt import money
    w = env.world
    for treasury in (3000, 1200, -500):                  # at, under and below zero: nothing above the $3,000 reserve
        w.treasury = treasury
        res = ToolBox(env, "mayor").call("update_plan", {"budgets": {"planning": 6000}})
        assert res["to_spend"] == 0 and res["next"].startswith("to spend: none (hold): the treasury is ")
        text = briefing(env, "mayor", mark=False)
        line, = [ln for ln in text.splitlines() if ln.startswith("- to spend:")]
        assert line.startswith("- to spend: none (hold): the treasury is ") and line.endswith(
            "; nothing above $3,000"), line                 # nothing for first buildings either
    w.treasury = 3001                                    # $1 above: short, but something to allocate
    assert ToolBox(env, "mayor").call("update_plan", {"budgets": {"planning": 6000}})["next"].startswith(
        "to spend: $1 above the $3,000 reserve")


def test_mayor_tools_in_a_mixed_line_up():
    env = _env(roles=["mayor", "planning", "utilities"])
    mayor = ToolBox(env, "mayor")
    fs, = [s["function"] for s in mayor.schemas() if s["function"]["name"] == "find_sites"]
    assert fs["parameters"]["properties"]["purpose"]["enum"] == [f"{d}.{p}" for d in ("planning", "utilities")
                                                                 for p in SITE_PURPOSES[d]]
    assert not mayor.call("find_sites", {"purpose": "education.school"})["ok"]
    res = mayor.call("update_plan", {"budgets": {"planning": 5000, "fire": 2500}})
    assert res["ok"] and set(res["below_target"]) == {"planning"}                 # no fire department in the game
    text = briefing(env, "mayor", mark=False)
    assert "- planning: funds $0 < target $5,000" in text and "- fire:" not in text


def test_reserve_follows_the_mayor_playbook():
    from types import SimpleNamespace
    from civitas.fleet.tools import reserve
    world = lambda *h: SimpleNamespace(history=[{"income": i, "expenses": e} for i, e in h])
    assert reserve(world()) == 3000                                      # before the first month
    assert reserve(world((5000, 800))) == 3000
    assert reserve(world((5000, 2000))) == 6000                          # 3 months of expenses
    assert reserve(world((1000, 2000))) == 12000                         # 6 while running a deficit
    assert reserve(world((1999.6, 2000.4))) == 6000                      # in rounded dollars: $2,000 both, no deficit


def _months_spent(env, *spent: dict) -> None:
    """A month passes per dict, with what the departments spent in it (the ledger the history's dept_spent records)."""
    for s in spent:
        env.world.ledger["spent"].update(s)
        env.end_month()


def _no_spending(env) -> None:
    """No department spent anything over the last 3 months (the mayor's Departments block shows spending)."""
    for h in env.world.history[-3:]:
        h["dept_spent"] = {d: 0 for d in DEPARTMENTS}


def test_idle_money_is_what_a_department_holds_beyond_its_use():
    """Smoke v9-v12: planning held $7,000-10,700 all game in 7 of 8 runs, spending about $1,000 a quarter, and the
    mayor topped it up because it read "below target"."""
    from civitas.fleet.tools import IDLE_MIN, IDLE_MULT, SPEND_MONTHS, DeptMoney, funding_gaps
    assert (SPEND_MONTHS, IDLE_MULT, IDLE_MIN) == (3, 2, 1000)
    env = _env()
    w = env.world
    fleet = FleetState.of(env)
    w.dept_funds.update(planning=7834, transport=2000)
    assert funding_gaps(env, fleet).idle == {}                  # month 0: no department has had a turn yet
    _months_spent(env, {"planning": 500, "utilities": 1200}, {"planning": 400}, {"planning": 266, "police": 99})
    w.treasury = 30000
    w.dept_funds.update(utilities=3000, police=1999)
    fleet.update_plan(w.month, budgets={"planning": 9000, "utilities": 7000})
    m = DeptMoney(env, "planning")
    assert (m.spent, m.months, m.idle) == (1166, 3, 5502)       # keeps 2 x $1,166 (its purposes aren't priced)
    f = funding_gaps(env, fleet)
    assert f.idle == {"planning": 5502, "transport": 1000}      # transport: no ready priced site, keeps $1,000
    assert set(f.gaps) == {"utilities"}                         # planning is $1,166 short of $9,000, but idle
    assert f.depts["utilities"].idle == 0                       # keeps 2 x $1,200: $600 above is under $1,000
    assert f.depts["police"].idle == 0                          # keeps $1,000: $999 above
    mayor = ToolBox(env, "mayor")
    assert mayor.call("allocate_budget", {"department": "health", "amount": 5000})["ok"]
    assert "health" not in funding_gaps(env, fleet).idle        # moved this turn: no department turn since
    assert mayor.call("allocate_budget", {"department": "planning", "amount": -5502})["ok"]
    assert mayor.call("allocate_budget", {"department": "transport", "amount": 1500})["ok"]
    assert funding_gaps(env, fleet).idle == {"transport": 1000}     # what it held through its turn


def test_saving_for_the_cheapest_ready_site_is_not_idle(grown_env):
    from civitas.fleet.tools import DeptMoney
    env = copy.deepcopy(grown_env)
    for h in env.world.history[-3:]:
        h["dept_spent"]["fire"] = 0
    fire = lambda funds: (env.world.dept_funds.update(fire=funds), DeptMoney(env, "fire"))[1]
    assert fire(1500).build.cost == 2500 and fire(1500).idle == 0       # saving toward a $2,500 station
    assert fire(2400).idle == 0                                         # $1,400 over $1,000, not over the station
    assert fire(3600).idle == 1100


def _held(env, **held: tuple) -> None:
    """Over the last 3 months each department held `held` at its turn (the first month) and spent `spent` then:
    dept: (held, spent). It holds the rest now."""
    w = env.world
    have = {d: held.get(d, (0, 0)) for d in DEPARTMENTS}
    for i, h in enumerate(w.history[-3:]):
        h["dept_funds"] = {d: got - spent for d, (got, spent) in have.items()}
        h["dept_spent"] = {d: spent if i == 0 else 0 for d, (got, spent) in have.items()}
    w.dept_funds.update({d: got - spent for d, (got, spent) in have.items()})


def test_below_target_only_when_it_used_its_money_or_never_had_enough(grown_env):
    from civitas.fleet.tools import USED_SHARE, funding_gaps
    assert USED_SHARE == 0.5
    env = copy.deepcopy(grown_env)
    w = env.world
    w.treasury = 50000
    fleet = FleetState.of(env)
    fleet.update_plan(w.month, budgets={"planning": 9000, "police": 2500, "utilities": 7000, "fire": 2500,
                                        "health": 5000})
    _held(env, planning=(9000, 1166), utilities=(7000, 5000), fire=(1500, 0), health=(2900, 0))
    f = funding_gaps(env, fleet)
    assert f.idle == {"planning": 5502}
    assert f.gaps == {"police": {"funds": 0, "target": 2500, "gap": 2500},        # never funded
                      "utilities": {"funds": 2000, "target": 7000, "gap": 5000},  # used $5,000 of $7,000
                      "fire": {"funds": 1500, "target": 2500, "gap": 1000}}       # saving for a $2,500 station
    # health: held $2,900, more than its $2,000 clinic, and used none of it: $2,100 short of $5,000, but no gap
    assert ToolBox(env, "mayor").call("allocate_budget", {"department": "planning", "amount": -5502})["ok"]
    assert "planning" not in funding_gaps(env, fleet).gaps      # $2,332 of $9,000, but it used $1,166 of $9,000


def test_a_department_keeps_its_dearest_ready_first_building(grown_env, open_env):
    """Review replay (seed 0, act_every 3, month 9): utilities held $3,000, spent nothing, power at capacity; its
    cheapest ready site was a $1,200 outlet (sewage had spare capacity), so the mayor read "idle $1,800" and took it
    back, and utilities, holding its $1,200 keep, was no gap and never reached the $2,500 turbine it needed. It keeps
    its dearest ready first building now (utilities, since v20, the building it names per short supply, together)."""
    from civitas.fleet.tools import DeptMoney, funding_gaps
    env = copy.deepcopy(grown_env)
    w = env.world
    fleet = FleetState.of(env)
    fleet.update_plan(w.month, budgets={"utilities": 5000})
    _held(env, utilities=(3000, 0))
    m = DeptMoney(env, "utilities")                    # power and water at capacity: a coal plant and a tower
    assert (m.build.cost, m.build.dearest, m.keep, m.idle) == (6200, 6200, 6200, 0)
    assert "utilities" in funding_gaps(env, fleet).gaps                 # it never held its $6,200 keep
    _held(env, utilities=(1200, 0))
    assert funding_gaps(env, fleet).gaps["utilities"] == {"funds": 1200, "target": 5000, "gap": 3800}
    line = next(ln for ln in briefing(env, "mayor", mark=False).splitlines() if ln.startswith("- utilities:"))
    assert line.startswith("- utilities: funds $1,200 < target $5,000, spent $0; needs $6,200 for another coal plant "
                           "and water tower (")
    env = copy.deepcopy(open_env)                      # every milestone reached, every priced purpose has sites
    dearest = {d: DeptMoney(env, d).build.dearest for d in ("utilities", "health", "transport")}
    assert dearest == {"utilities": 5500,              # a coal plant and a pump, not the $12k water works (tier 3)
                       "health": 2000,                 # not the $8,000 hospital (the clinic's tier 2)
                       "transport": 1500}              # not a train station, harbour or metro (supplementary)


def test_keep_never_drops_below_the_needs_line(open_env, monkeypatch):
    """With only an upgrade ready (health: the $8,000 hospital, the clinic's tier 2) the dearest first building was
    $0, so money saved for the hospital read as idle while the needs line named it."""
    import civitas.fleet.briefing as B
    import civitas.fleet.tools as T
    from civitas.fleet.tools import MAX_SITES, DeptMoney, SiteList
    real = T.site_lists
    none = SiteList([], "nothing needs it right now", "")

    def only_hospital(w, role, obs):
        lists = real(w, role, obs)
        return {p: lst if role != "health" or p == "hospital" else none for p, lst in lists.items()}
    monkeypatch.setattr(T, "site_lists", only_hospital)
    monkeypatch.setattr(B, "site_lists", only_hospital)
    env = copy.deepcopy(open_env)                      # every milestone reached: the hospital is unlocked
    _held(env, health=(6000, 0))
    m = DeptMoney(env, "health")
    assert (m.build.cost, m.build.dearest, m.keep, m.idle) == (8000, 8000, 8000, 0)
    n = len(real(env.world, "health", T.role_obs(env, "health"))["hospital"].sites[:MAX_SITES])
    line = next(ln for ln in briefing(env, "mayor", mark=False).splitlines() if ln.startswith("- health:"))
    assert n and line == f"- health: funds $6,000, spent $0; needs $8,000 for its first hospital ({n} sites, $120/month)"


def _empty_lists(monkeypatch, dept: str, short: str, blocked: bool) -> None:
    """`dept`'s unlocked site lists come back empty for `short`: blocked for lack of land or roads, or not."""
    import civitas.fleet.briefing as B
    import civitas.fleet.tools as T
    real = T.site_lists

    def empty(w, role, obs):
        lists = real(w, role, obs)
        return lists if role != dept else {p: lst if (lst.short or "").startswith("locked") else
                                           T.SiteList([], short, short, blocked) for p, lst in lists.items()}
    monkeypatch.setattr(T, "site_lists", empty)
    monkeypatch.setattr(B, "site_lists", empty)


LAND = "no free roadside land: ask planning for a lot"
BLOCKED_NOTE = "- blocked for lack of land: money buys nothing there until planning frees lots (ask planning)"   # once


def test_money_for_a_blocked_first_building_is_not_idle(grown_env, monkeypatch):
    """Smoke v13 seed 0, month 6: planning had zoned every roadside lot; police, fire, health and education held
    their first building's price with nothing spent, their lists read "no free roadside land", the keep fell to the
    $1,000 floor, and the mayor took back "idle $1,500 / $1,000". Police and fire built at month 9; health and
    education never did."""
    from civitas.fleet.tools import DeptMoney
    env = copy.deepcopy(grown_env)
    _held(env, police=(2500, 0))
    _empty_lists(monkeypatch, "police", LAND, blocked=True)
    m = DeptMoney(env, "police")                       # keeps $4,000 for the jail, its dearest; names the station
    assert (m.build.cost, m.build.dearest, m.keep, m.idle, m.blocked) == (None, 4000, 4000, 0, ("a police station",))
    lines = lambda: briefing(env, "mayor", mark=False).splitlines()
    line = lambda: next(ln for ln in lines() if ln.startswith("- police:"))
    assert line() == "- police: funds $2,500, spent $0; blocked: a police station"
    assert lines().count(BLOCKED_NOTE) == 1                              # what to do, said once
    _empty_lists(monkeypatch, "police", "nothing needs it right now", blocked=False)
    m = DeptMoney(env, "police")                       # not needed: as before, idle above the $1,000 floor
    assert (m.build, m.keep, m.idle, m.blocked) == (None, 1000, 1500, ())
    assert line() == "- police: funds $2,500, spent $0; idle $1,500 (keeps $1,000)" and BLOCKED_NOTE not in lines()


def test_blocked_plants_count_in_the_keep(grown_env, monkeypatch):
    from civitas.fleet.tools import DeptMoney
    env = copy.deepcopy(grown_env)
    _held(env, utilities=(7000, 0))
    _empty_lists(monkeypatch, "utilities", LAND, blocked=True)
    m = DeptMoney(env, "utilities")                    # coal $4,000 (riverbank_2x2 is locked, and an upgrade)
    assert (m.keep, m.idle, m.blocked) == (4000, 3000, ("a water pump",))    # not the outlet: sewage 313 of 500
    line = next(ln for ln in briefing(env, "mayor", mark=False).splitlines() if ln.startswith("- utilities:"))
    assert line == "- utilities: funds $7,000, spent $0; idle $3,000 (keeps $4,000)"   # one suffix: idle first


def test_a_blocked_department_that_never_held_its_keep_is_a_gap(grown_env, monkeypatch):
    from civitas.fleet.tools import funding_gaps
    env = copy.deepcopy(grown_env)
    fleet = FleetState.of(env)
    fleet.update_plan(env.world.month, budgets={"police": 2500})
    _held(env, police=(1500, 0))
    _empty_lists(monkeypatch, "police", "nothing needs it right now", blocked=False)
    assert "police" not in funding_gaps(env, fleet).gaps          # held $1,500 above the floor and used none of it
    _empty_lists(monkeypatch, "police", LAND, blocked=True)
    assert funding_gaps(env, fleet).gaps["police"] == {"funds": 1500, "target": 2500, "gap": 1000}   # keep $4,000


def _quarter(env, spent: dict) -> None:
    """Departments spend at their turn (funds and ledger, as actions._spend does), then 3 months pass."""
    for d, v in spent.items():
        env.world.dept_funds[d] -= v
        env.world.ledger["spent"][d] += v
    for _ in range(3):
        env.end_month()


def test_a_take_back_starts_no_top_up_cycle():
    """After a take-back planning holds what it keeps, 2 x its spending, below its $9,000 target: with the plain
    target rule it was a gap again at once, got topped up, and was idle the quarter after."""
    from civitas.fleet.tools import funding_gaps
    env = _env()
    w = env.world
    w.treasury = 30000
    fleet = FleetState.of(env)
    mayor = ToolBox(env, "mayor")
    mayor.call("update_plan", {"budgets": {"planning": 9000}})
    assert set(funding_gaps(env, fleet).gaps) == {"planning"}          # month 0: the plain target rule
    assert mayor.call("allocate_budget", {"department": "planning", "amount": 9000})["ok"]
    _quarter(env, {"planning": 1166})
    f = funding_gaps(env, fleet)                                       # month 3
    assert f.idle == {"planning": 5502} and not f.gaps
    assert mayor.call("allocate_budget", {"department": "planning", "amount": -5502})["ok"]
    assert not funding_gaps(env, fleet).gaps
    _quarter(env, {"planning": 1166})
    f = funding_gaps(env, fleet)                                       # month 6: spent $1,166 of the $2,332 it held
    assert not f.gaps and not f.idle
    text = briefing(env, "mayor", mark=False)
    assert "- planning: funds $1,166, target $9,000, spent $1,166\n" in text and "below target" not in text


def test_update_plan_says_to_take_idle_money_back():
    env = _env()
    _months_spent(env, {}, {}, {})
    w = env.world
    w.treasury = 30000
    w.dept_funds.update(planning=7834, transport=2000)
    mayor = ToolBox(env, "mayor")
    note = "to spend: $27k above the $3,000 reserve + $7,834 idle to take back = $34.8k; "
    res = mayor.call("update_plan", {"budgets": {"planning": 9000}})
    assert res["idle"] == {"planning": 6834, "transport": 1000} and "below_target" not in res
    assert res["to_spend"] == 34834 and res["next"].startswith(note)
    res = mayor.call("update_plan", {"budgets": {"fire": 2500}})
    assert res["below_target"] == {"fire": {"funds": 0, "target": 2500, "gap": 2500}}
    assert res["next"].startswith(note)


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


# ---------------------------------------------------------------- guards
CUT_OFF = ("this would cut 17 road tiles (e.g. around (3,24)) off the highway; build a new link first or pick tiles "
           "that aren't on the only route; to do it anyway repeat with confirm: true")
UNCONNECTED = "- roads not connected to the highway: 17 tiles (x 3-19, y 24) - reconnect them"


def test_bulldozing_the_only_link_to_the_highway_is_refused_until_confirmed():
    """Seed 1, v7: planning bulldozed (0,22)-(2,26) for a road along x 0, taking the 2 road tiles between the highway
    (0,24) and the main road; the city emptied. Seed 0's starter road has the same neck."""
    from civitas.fleet.agent import _slim
    from civitas.fleet.briefing import briefing
    from civitas.fleet.report import _tool_error, turn_row
    from civitas.fleet.tools import map_facts, map_lines
    env = _env()
    w = env.world
    w.dept_funds["planning"] = 1000
    assert w.highway == (0, 24) and (w.struct[24, 1:20] == C.ROAD).all()
    assert map_facts(w)["unconnected"] is None and "not connected" not in "\n".join(map_lines(map_facts(w)))
    struct, funds = w.struct.copy(), dict(w.dept_funds)
    box = ToolBox(env, "planning")
    rect = {"x1": 0, "y1": 22, "x2": 2, "y2": 26}
    for n in (1, 2):                                     # an identical repeat is refused the same way, not as a repeat
        res = box.call("bulldoze", rect)
        assert res == {"ok": False, "refused": "disconnect", "error": CUT_OFF}, res
        assert box.n_calls == n
    assert (w.struct == struct).all() and w.dept_funds == funds
    assert box.executed == [] and box._failed == {} and FleetState.of(env).history(months=12) == []
    call = {"tool": "bulldoze", "args": rect, "result": _slim(res)}
    assert call["result"]["refused"] == "disconnect" and _tool_error(call) is None
    assert turn_row({"role": "planning", "month": 0, "tool_trace": [call]})["refused"] == 1
    props = next(s["function"] for s in box.schemas() if s["function"]["name"] == "bulldoze")["parameters"]
    assert props["properties"]["confirm"]["type"] == "boolean"
    res = box.call("bulldoze", {**rect, "confirm": "true"})
    assert res["ok"] and res["message"] == "bulldozed 2 tiles", res
    assert (w.struct[24, 1:3] == C.NONE).all() and w.struct[24, 0] == C.HIGHWAY
    assert box.executed[-1]["action"] == {**rect, "type": "bulldoze"}              # confirm is not an action param
    assert UNCONNECTED in map_lines(map_facts(w))
    for role in ("mayor", "planning", "police"):                                   # every role sees the broken link
        assert UNCONNECTED in briefing(env, role, mark=False)


def test_bulldozing_a_dead_end_or_a_building_is_not_refused():
    env = _env()
    w = env.world
    w.dept_funds["planning"] = 1000
    assert apply_action(w, "planning", {"type": "build_road", "x1": 5, "y1": 25, "x2": 5, "y2": 28}).ok
    box = ToolBox(env, "planning")
    res = box.call("bulldoze", {"x": 5, "y": 26})                 # the middle of the spur strands its tail
    assert res["refused"] == "disconnect" and "cut 2 road tiles (e.g. around (5,27))" in res["error"]
    res = box.call("bulldoze", {"x": 5, "y": 27})                 # strands the last tile only
    assert res["refused"] == "disconnect" and "cut 1 road tile (e.g. around (5,28)) off" in res["error"]
    res = box.call("bulldoze", {"x1": 5, "y1": 25, "x2": 5, "y2": 28})      # the whole spur: nothing left behind
    assert res["ok"] and res["message"] == "bulldozed 4 tiles" and "refused" not in res
    assert apply_action(w, "planning", {"type": "zone", "zone": "R", "x1": 4, "y1": 25, "x2": 4, "y2": 26}).ok
    assert ToolBox(env, "planning").call("bulldoze", {"x1": 4, "y1": 25, "x2": 4, "y2": 26})["ok"]   # zones only
    w.dept_funds["utilities"] = 10000
    s = ToolBox(env, "utilities").call("find_sites", {"purpose": "wind"})["sites"][0]
    assert apply_action(w, "utilities", {"type": "build_power_plant", "kind": "wind", "x": s["x"], "y": s["y"]}).ok
    res = ToolBox(env, "utilities").call("demolish", {"x": s["x"], "y": s["y"]})
    assert res["ok"] and "demolished" in res["message"], res


def _linked_road(env, road: dict, action: dict):
    """An off-highway road (`road`, planning's build_road) that transport links to the region with `action`."""
    w = env.world
    w.milestones_on = False
    w.dept_funds["planning"] = w.dept_funds["transport"] = 100000
    assert apply_action(w, "planning", {"type": "build_road", **road}).ok
    x, y = road["x1"], road["y1"]
    assert not w.networks[w.road_labels[y, x]].outside_connected
    res = apply_action(w, "transport", action)
    assert res.ok, res.message
    assert w.networks[w.road_labels[y, x]].outside_connected


def test_demolishing_the_only_rail_link_is_refused_until_confirmed():
    """A train station, harbour or airport links the roads it touches to the region (World.refresh): demolishing
    the only link of highway-connected roads cuts them off, as bulldozing the link to the highway does."""
    from civitas.fleet.report import turn_row
    env = _env()
    w = env.world
    _linked_road(env, {"x1": 10, "y1": 0, "x2": 10, "y2": 5}, {"type": "build_train_station", "x": 11, "y": 0})
    struct, funds = w.struct.copy(), dict(w.dept_funds)
    box = ToolBox(env, "transport")
    for n in (1, 2):                                     # an identical repeat is refused the same way, not as a repeat
        res = box.call("demolish", {"x": 11, "y": 0})
        assert res == {"ok": False, "refused": "disconnect", "error": (
            "this would cut 6 road tiles (e.g. around (10,0)) off the region: the train station is their only link; "
            "ask planning to connect them to the highway by road first, or build another link; to do it anyway repeat "
            "with confirm: true")}
        assert box.n_calls == n
    assert (w.struct == struct).all() and w.dept_funds == funds
    assert box.executed == [] and box._failed == {} and FleetState.of(env).history(months=12) == []
    trace = [{"tool": "demolish", "args": {"x": 11, "y": 0}, "result": res}]
    assert turn_row({"role": "transport", "month": 0, "tool_trace": trace})["refused"] == 1
    props = lambda role, tool: next(s["function"]["parameters"]["properties"] for s in ToolBox(env, role).schemas()
                                    if s["function"]["name"] == tool)
    assert props("transport", "demolish")["confirm"]["type"] == "boolean"
    assert "demolish" in props("transport", "demolish")["confirm"]["description"]
    assert "bulldoze" in props("planning", "bulldoze")["confirm"]["description"]
    for role in ("utilities", "police", "fire", "health", "education", "planning"):    # can't demolish a link
        assert "confirm" not in props(role, "demolish"), role
    res = box.call("demolish", {"x": 11, "y": 0, "confirm": "true"})
    assert res["ok"] and res["message"] == "demolished train_station at (11,0)", res
    assert box.executed[-1]["action"] == {"x": 11, "y": 0, "type": "demolish"}    # confirm is not an action param
    assert w.struct[0, 11] == C.NONE and not w.networks[w.road_labels[0, 10]].outside_connected


def test_demolishing_an_airport_checks_the_whole_building_and_other_links(monkeypatch):
    """Any tile of a multi-tile link names the building; a link the roads don't need (another link, the highway)
    and a demolish of anything else pass without relabelling."""
    import civitas.fleet.tools as tools
    env = _env()
    w = env.world
    _linked_road(env, {"x1": 40, "y1": 1, "x2": 40, "y2": 8}, {"type": "build_airport", "x": 41, "y": 3})
    box = ToolBox(env, "transport")
    res = box.call("demolish", {"x": 43, "y": 5})               # the 3x3 airport's far corner
    assert res["refused"] == "disconnect" and res["error"].startswith(
        "this would cut 8 road tiles (e.g. around (40,3)) off the region: the airport is their only link;"), res
    assert apply_action(w, "transport", {"type": "build_train_station", "x": 40, "y": 0}).ok   # a second link
    assert box.call("demolish", {"x": 43, "y": 5})["ok"] and w.struct[5, 43] == C.NONE
    assert apply_action(w, "transport", {"type": "build_train_station", "x": 0, "y": 25}).ok   # beside the highway
    assert box.call("demolish", {"x": 0, "y": 25})["ok"]
    assert apply_action(w, "transport", {"type": "build_metro_station", "x": 5, "y": 23}).ok
    monkeypatch.setattr(tools, "label_components", None)       # a metro station links nothing: no relabelling
    assert box.call("demolish", {"x": 5, "y": 23})["ok"]
    assert box.call("demolish", {"x": 30, "y": 30})["message"].startswith("no transport facility at (30,30)")


def test_a_repeated_allocation_in_one_turn_is_refused_until_confirmed():
    """v12 seed 0, month 3: the mayor sent utilities +400, planning +1,112, health +2,000 three times in one turn
    (v10 seed 0: utilities +5,500 and health +2,000 twice). Allocations add up and a success clears the repeat
    cache, so nothing stopped it; the treasury drained and loans followed."""
    from civitas.fleet.agent import _slim
    from civitas.fleet.report import _tool_error, turn_row
    env = _env()
    w = env.world
    box = ToolBox(env, "mayor")
    batch = [{"department": "utilities", "amount": 400}, {"department": "planning", "amount": 1112},
             {"department": "health", "amount": 2000}]
    for a in batch:
        assert box.call("allocate_budget", a)["ok"]
    funds, treasury = dict(w.dept_funds), w.treasury
    assert funds["utilities"] == 400 and treasury == 30000 - 3512
    refusal = lambda dept, amount: {"ok": False, "refused": "duplicate", "error": (
        f"you already moved {amount} to {dept} this turn (it now has {amount}); allocations add up - repeat with "
        "confirm: true only to add it again")}
    again = ['{"amount": 400.0, "department": "Utilities"}', {"department": "planning", "amount": "1112"}, batch[2]]
    trace = []
    for args, (dept, amount) in zip(again, [("utilities", "$400"), ("planning", "$1,112"), ("health", "$2,000")]):
        res = box.call("allocate_budget", args)                 # the same after the handler's normalisation
        assert res == refusal(dept, amount), res
        trace.append({"tool": "allocate_budget", "args": args, "result": _slim(res)})
    assert box.call("allocate_budget", batch[0]) == refusal("utilities", "$400")    # checked, and refused, again
    assert w.dept_funds == funds and w.treasury == treasury and box.n_calls == 7
    assert len(box.executed) == 3 and box._failed == {}
    assert len(FleetState.of(env).history(kind="action", months=12)) == 3
    assert all(_tool_error(c) is None for c in trace)
    assert turn_row({"role": "mayor", "month": 3, "tool_trace": trace})["refused"] == 3
    props = lambda tool: next(s["function"]["parameters"]["properties"] for s in box.schemas()
                              if s["function"]["name"] == tool)
    assert props("allocate_budget")["confirm"]["type"] == "boolean"
    assert "refused this turn with the same department and amount" in props("allocate_budget")["confirm"][
        "description"]
    assert "confirm" not in props("set_tax") and props("take_loan")["confirm"]["type"] == "boolean"   # the loan guard
    res = box.call("allocate_budget", {**batch[0], "confirm": "true"})     # a repeat of a refused call: earned
    assert res["ok"] and res["message"].startswith("transferred $400 to utilities (now $800)"), res
    assert res["override"] == "duplicate"
    assert box.executed[-1]["action"] == {**batch[0], "type": "allocate_budget"}   # confirm is not an action param
    assert box.call("allocate_budget", batch[0])["error"].startswith(               # counts as moved: still refused
        "you already moved $400 to utilities this turn (it now has $800);")


def test_other_allocations_pass_the_duplicate_guard():
    """Another amount or department, another take-back (an identical one is a duplicate:
    test_identical_take_backs_are_duplicates), and a retry of an allocation that failed all run."""
    env = _env()
    w = env.world
    box = ToolBox(env, "mayor")
    ok = lambda dept, amount: box.call("allocate_budget", {"department": dept, "amount": amount})
    assert ok("utilities", 400)["ok"]
    for dept, amount in (("utilities", 500), ("fire", 400), ("utilities", -100), ("utilities", -200)):
        res = ok(dept, amount)
        assert res["ok"] and "refused" not in res, (dept, amount, res)
    assert w.dept_funds["utilities"] == 600 and w.dept_funds["fire"] == 400
    big = ok("health", 40000)
    assert big["ok"] is False and "treasury only has" in big["error"]
    assert box.call("set_tax", {"sector": "R", "rate": 9})["ok"]           # clears the failed-action cache
    w.treasury = 50000
    res = ok("health", 40000)
    assert res["ok"] and "refused" not in res and w.dept_funds["health"] == 40000, res
    assert len(box.executed) == 8


def test_a_take_back_below_what_a_department_keeps_is_refused_until_confirmed():
    """Smoke v18 seed 0, month 6: the mayor read "planning: funds $2,508, target $3,000, spent $492; idle $1,508" and
    took back all $2,508; planning had $0 at months 9 and 12. Seed 1, month 6: planning held $1,479 (nothing idle),
    and the mayor took $1,182, then its last $297, to fund health."""
    from civitas.fleet.report import _tool_error, turn_row
    from civitas.fleet.tools import DeptMoney
    env = _env()
    w = env.world
    _months_spent(env, {"planning": 492}, {}, {})
    w.treasury, w.dept_funds["planning"] = 30000, 2508
    m = DeptMoney(env, "planning")
    assert (m.keep, m.idle) == (1000, 1508)
    box = ToolBox(env, "mayor")
    take = lambda amount, **kw: box.call("allocate_budget", {"department": "planning", "amount": amount, **kw})
    import civitas.fleet.tools as T
    assert T.road_build(env).cost == 600               # what it would keep less than: its road into open land
    tail = (", so it can't pay for a road into open land and zoning it ($600); {}. To take it anyway, repeat this "
            "exact call with confirm: true.")
    res = take(-2508)
    assert res == {"ok": False, "refused": "take_back", "error": (
        "that takes planning below the $1,000 it keeps (the $1,000 minimum)"
        + tail.format("its idle money is $1,508: take back $1,508"))}, res
    assert take(-2508) == res                          # checked, and refused, again: not remembered as a failure
    assert w.dept_funds["planning"] == 2508 and box.executed == [] and box._failed == {}
    assert _tool_error({"tool": "allocate_budget", "result": res}) is None
    assert turn_row({"role": "mayor", "month": 3, "tool_trace": [{"tool": "allocate_budget", "result": res}]})[
        "refused"] == 1
    assert "refused this turn" in next(s["function"]["parameters"]["properties"]["confirm"]["description"]
                                       for s in box.schemas() if s["function"]["name"] == "allocate_budget")
    res = take(-1508)                                  # its idle money: taken
    assert res["ok"] and w.dept_funds["planning"] == 1000, res
    res = take(-500)                                   # nothing idle left
    assert res == {"ok": False, "refused": "take_back", "error": (
        "that takes planning below the $1,000 it keeps (the $1,000 minimum)"
        + tail.format("you already took back $1,508 from it this turn; it has no idle money: leave its funds"))}, res
    assert take(-5000)["error"].startswith("planning only holds $1,000")    # more than it holds: the action says so
    res = take(-500, confirm=True)                     # a repeat of the refused call: earned
    assert res["ok"] and res["message"].startswith("reclaimed $500 from planning") and res["override"] == "take_back"
    assert w.dept_funds["planning"] == 500, res
    assert box.executed[-1]["action"] == {"department": "planning", "amount": -500, "type": "allocate_budget"}
    w.dept_funds["planning"] = 1479                    # seed 1: $479 over the $1,000, nothing idle
    assert take(-1182)["refused"] == "take_back" and take(-479)["ok"] and w.dept_funds["planning"] == 1000


def test_take_backs_of_this_turn_s_top_ups_and_of_unkept_money_pass():
    """This turn's top-ups (DeptMoney.held leaves them out) can go back whatever the department keeps, and a
    department that isn't in the game keeps nothing."""
    env = _env(roles=["mayor", "planning", "utilities"])
    w = env.world
    _months_spent(env, {}, {}, {})
    w.treasury = 30000
    w.dept_funds.update(planning=500, police=3000)     # planning under its $1,000; police not in the game
    box = ToolBox(env, "mayor")
    move = lambda dept, amount: box.call("allocate_budget", {"department": dept, "amount": amount})
    assert move("planning", 700)["ok"] and move("planning", -300)["ok"] and move("planning", -400)["ok"]
    assert w.dept_funds["planning"] == 500
    assert move("planning", 800)["ok"]
    tail = " To take it anyway, repeat this exact call with confirm: true."
    res = move("planning", -801)
    road = ", so it can't pay for a road into open land and zoning it ($600)"      # $499 left
    assert res["error"] == ("that takes planning below the $1,000 it keeps (the $1,000 minimum)" + road + "; you "
                            "already took back $700 from it this turn; it has no idle money, only the $800 you moved "
                            "to it this turn: take back $800." + tail), res
    assert move("police", -3000)["ok"] and w.dept_funds["police"] == 0       # nothing kept
    _months_spent(env, {})
    w.treasury, w.dept_funds["planning"] = 30000, 2500
    box = ToolBox(env, "mayor")
    assert move("planning", 1000)["ok"]                # $2,500 held: $1,500 idle, plus this turn's $1,000
    res = move("planning", -3000)
    assert res["error"] == ("that takes planning below the $1,000 it keeps (the $1,000 minimum)" + road + "; its idle "
                            "money is $1,500, plus the $1,000 you moved to it this turn: take back $2,500." + tail), res
    assert move("planning", -2500)["ok"] and w.dept_funds["planning"] == 1000


def test_zoning_the_last_riverbank_lot_warns_while_a_pump_or_outlet_is_missing():
    env = _env()
    w = env.world
    taken = _bridge_and_take_the_bank(env)
    w.dept_funds["planning"] = w.dept_funds["utilities"] = 10000
    warning = (" WARNING: that zoned the last free riverbank lot(s) beside roads; utilities still needs one for its "
               "{} - dezone one near ({},{})")

    def free(*lots):
        for tx, ty in lots:
            w.struct[ty, tx] = C.NONE
        w.refresh()

    def zone(x, y):
        return ToolBox(env, "planning").call("zone", {"zone": "R", "x1": x, "y1": y, "x2": x, "y2": y})

    (x, y), (x2, y2), (x3, y3) = taken[:3]
    free((x, y), (x2, y2))
    res = zone(x, y)                                     # one free lot left: no warning
    assert res["ok"] and "WARNING" not in res["message"], res
    res = zone(x2, y2)                                   # the last one, with neither pump nor outlet
    assert res["ok"] and res["message"].endswith(warning.format("pump and sewage outlet", x2, y2)), res
    assert "WARNING" not in FleetState.of(env).history(kind="action", role="planning")[-1]["message"]   # result only
    free((x, y), (x2, y2))                               # a pump takes one: the outlet still needs the other
    assert apply_action(w, "utilities", {"type": "build_water", "kind": "pump", "x": x, "y": y}).ok
    res = zone(x2, y2)
    assert res["ok"] and res["message"].endswith(warning.format("sewage outlet", x2, y2)), res
    free((x2, y2), (x3, y3))                             # an outlet takes one: nothing left to warn about
    assert apply_action(w, "utilities", {"type": "build_sewage", "kind": "outlet", "x": x2, "y": y2}).ok
    res = zone(x3, y3)
    assert res["ok"] and "WARNING" not in res["message"], res


def test_a_water_tower_counts_as_water_in_the_riverbank_warning():
    """A city on a water tower needs no pump: zoning the last riverbank lot warns about the sewage outlet only."""
    env = _env()
    w = env.world
    (x, y), *_ = _bridge_and_take_the_bank(env)
    w.dept_funds["planning"] = w.dept_funds["utilities"] = 10000
    assert apply_action(w, "utilities", {"type": "build_water", "kind": "tower", "x": 10, "y": 25}).ok
    assert w.facilities(C.TOWER) and not w.facilities(C.PUMP)
    w.struct[y, x] = C.NONE
    w.refresh()
    res = ToolBox(env, "planning").call("zone", {"zone": "R", "x1": x, "y1": y, "x2": x, "y2": y})
    assert res["ok"] and res["message"].endswith(" WARNING: that zoned the last free riverbank lot(s) beside roads; "
                                                 "utilities still needs one for its sewage outlet - dezone one near "
                                                 f"({x},{y})"), res


# ------------------------------------------------------------ branch roads
def _boxed_in(lots=(), free=(), rows=(23, 25)):
    """Smoke v17 seed 0, month 6: one road across the map (y=24 x 0-47, bridging the river at x 26-28) and a level-1
    home on every land tile of `rows` (the two beside it), but the empty zoned `lots` and the `free` tiles: no dead
    end to extend (the road ends at the map edge) and, without `free`, no free lot beside the road."""
    env = _env()
    w = env.world
    w.struct[:], w.anchor[:], w.level[:] = C.NONE, -1, 0
    w.struct[24, 0], w.struct[24, 1:] = C.HIGHWAY, C.ROAD
    for y in rows:
        land = w.terrain[y] == C.LAND
        w.struct[y, land], w.level[y, land] = C.RES, 1
    for x, y in lots:
        w.level[y, x] = 0
    for x, y in free:
        w.struct[y, x], w.level[y, x] = C.NONE, 0
    w.refresh()
    w.dept_funds["planning"] = 1000
    return env


def _road_line(env) -> str:
    line, = [ln for ln in _ready_sites(briefing(env, "planning", mark=False)) if ln.startswith("- road_extension: ")]
    return line


def test_boxed_in_planning_is_offered_branch_roads():
    """Smoke v17 seed 0, months 6-9: the only road ran across the map with every tile beside it zoned or built;
    planning read "no straight extension from a dead end: build_road from any road into free land" and spent 3 turns
    on roads into buildings and past the map edge. A branch starts from a free lot, else an empty lot (dezone: free),
    else the cheapest building to bulldoze, ranked by land opened."""
    from civitas.fleet.tools import BRANCH_APART, branch_sites
    head = "- road_extension: none - no dead end; branch: "
    env = _boxed_in(free=[(12, 23)], lots=[(40, 25)])                                # priced: planning holds $1,000
    assert _road_line(env) == head + ("build_road (12,23)-(12,9), opens 75 land tiles; $150 road + $375 "
                                      "zoning")                                     # nothing to clear
    env = _boxed_in(lots=[(12, 23)])
    assert _road_line(env) == head + ("dezone (12,23) (empty lot, free), then build_road (12,23)-(12,9), opens 75 "
                                      "land tiles; $150 road + $375 zoning")
    res = ToolBox(env, "planning").call("find_sites", {"purpose": "road_extension"})
    assert res["sites"][0] == {"x": 12, "y": 23, "x1": 12, "y1": 23, "x2": 12, "y2": 9, "tiles": 15,
                               "bridge_tiles": 0, "opens_land": 75, "branch_from": [12, 24], "cost": 150,
                               "first": "dezone (12,23) (empty lot, free)"}, res
    assert res["note"].startswith("no straight extension from a dead end into open land, so these are branch roads")
    assert "first" in res["note"] and len(res["sites"]) == 6
    assert res["sites"][1]["first"] == "bulldoze (17,23) (a level-1 home, $5)"       # then buildings, nearest the
    near = ToolBox(env, "planning").call("find_sites", {"purpose": "road_extension", "near_x": 12, "near_y": 15})
    assert near["sites"][0]["x1"] == 12 and near["sites"][0]["distance"] == 0        # on it (it runs north)
    for i, a in enumerate(branch_sites(env.world)):                                  # town hall on a tie
        for b in branch_sites(env.world)[:i]:
            assert a.way != b.way or abs(a.start[0] - b.start[0]) + abs(a.start[1] - b.start[1]) > BRANCH_APART
    env = _boxed_in()                                                                # only buildings beside the road
    assert _road_line(env) == head + ("bulldoze (17,23) (a level-1 home, $5), then build_road (17,23)-(17,9), "
                                      "opens 75 land tiles; $155 road + $375 zoning")   # the bulldozing counted
    deep = (22, 23, 25, 26)                                                          # zoned 2 deep, as v17 did
    assert _road_line(_boxed_in(lots=[(12, 22), (12, 23)], rows=deep)) == head + (
        "dezone (12,22)-(12,23) (empty lots, free), then build_road (12,23)-(12,9), opens 75 land tiles; $150 road + "
        "$375 zoning")
    assert _road_line(_boxed_in(lots=[(12, 23)], rows=deep)) == head + (               # the fewest levels first
        "bulldoze (12,22)-(12,23) (a level-1 home and an empty lot, $10), then build_road (12,23)-(12,9), opens 75 "
        "land tiles; $160 road + $375 zoning")
    assert _road_line(_boxed_in(rows=(21, 22, 23, 25, 26, 27))) == (                 # 3 deep: none within reach
        "- road_extension: none - no straight extension from a dead end: build_road from any road into free land")
    env = _boxed_in(lots=[(12, 23)])
    env.world.struct[22:24, 40] = C.ROAD                                             # a dead end at (40,22)
    env.world.refresh()
    assert _road_line(env).startswith("- road_extension: 1 site, best (40,22)-(40,") and "branch" not in _road_line(env)
    assert "note" not in ToolBox(env, "planning").call("find_sites", {"purpose": "road_extension"})


def test_a_road_into_a_building_points_to_a_branch_while_no_dead_end_extends():
    """Smoke v17 seed 0, months 6-9: 6 roads into buildings ("occupied by coal_plant / commercial / residential ...
    route around") and 2 past the map edge ("(52,24) is outside the 48x48 map"), with no dead end left to extend."""
    env = _boxed_in(lots=[(12, 23)])
    box = ToolBox(env, "planning")
    branch = ("no dead end to extend: branch instead, e.g. dezone (12,23) (empty lot, free), then build_road "
              "(12,23)-(12,9) (find_sites road_extension)")
    res = box.call("build_road", {"x1": 17, "y1": 24, "x2": 17, "y2": 20})
    assert res["message"] == ("tile (17,23) is occupied by residential; bulldoze it or route around - in the way: "
                              f"building (17,23){AROUND}; {branch}"), res
    res = box.call("build_road", {"x1": 47, "y1": 24, "x2": 52, "y2": 24})
    assert res["message"] == f"(52,24) is outside the 48x48 map - {branch}", res
    env.world.struct[22:24, 40] = C.ROAD                                             # a dead end: no branch hint
    env.world.refresh()
    box = ToolBox(env, "planning")
    assert box.call("build_road", {"x1": 47, "y1": 24, "x2": 52, "y2": 24})["message"] == \
        "(52,24) is outside the 48x48 map"
    res = box.call("build_road", {"x1": 17, "y1": 24, "x2": 17, "y2": 20})
    assert res["message"].endswith(f"building (17,23){AROUND}"), res


def test_zoning_the_last_lot_a_road_could_branch_from_warns():
    """Smoke v17 seed 0, months 0-3: planning zoned both sides of its only road nearly end to end (R at y=22 and y=23,
    I at y=25) with no dead end left to extend, and the last free lots beside it went too; nothing said so."""
    warning = " WARNING: that zoned the last lot a new road could branch from; leave one free, e.g. dezone ({},{})"
    zone = lambda env, x, y: ToolBox(env, "planning").call("zone", {"zone": "R", "x1": x, "y1": y, "x2": x, "y2": y})
    env = _boxed_in(free=[(12, 23)])
    res = zone(env, 12, 23)
    assert res["ok"] and res["message"] == "zoned 1 tiles R (low density)" + warning.format(12, 23), res
    assert "WARNING" not in FleetState.of(env).history(kind="action", role="planning")[-1]["message"]   # result only
    res = zone(env, 5, 22)                                           # boxed in already: nothing more to warn about
    assert res["ok"] and "WARNING" not in res["message"], res
    env = _boxed_in(free=[(12, 23), (40, 25)])
    res = zone(env, 12, 23)                                          # one lot left: no warning
    assert res["ok"] and "WARNING" not in res["message"], res
    res = zone(env, 40, 25)
    assert res["ok"] and res["message"].endswith(warning.format(40, 25)), res
    env = _boxed_in(free=[(12, 23)])
    res = zone(env, 12, 22)                                          # the lot free, the way through it zoned
    assert res["ok"] and res["message"].endswith(warning.format(12, 22)), res
    env = _boxed_in(free=[(12, 23)])
    res = zone(env, 12, 21)                                          # after the game's own warning: one WARNING
    assert res["message"] == ("zoned 1 tiles R (low density) WARNING: 1 of them have no road access yet; it also "
                              "zoned the last lot a new road could branch from (dezone (12,21) to leave one)"), res
    env = _boxed_in(free=[(12, 23)])
    env.world.struct[22:24, 40] = C.ROAD                             # a dead end still extends: no warning
    env.world.refresh()
    res = zone(env, 12, 23)
    assert res["ok"] and "WARNING" not in res["message"], res
    env = _boxed_in(free=[(25, 23)])                                 # the last lot is also the last riverbank lot
    res = zone(env, 25, 23)
    assert res["ok"] and res["message"].count("WARNING") == 1, res
    assert res["message"].endswith("utilities still needs one for its pump and sewage outlet - dezone one near "
                                   "(25,23); it also zoned the last lot a new road could branch from"), res


def test_a_landfill_blocked_away_from_homes_is_tagged_with_where_to_leave_its_lot():
    """Smoke v15 seed 1, months 6-9: health's landfill read "blocked ... no roadside lot away from homes", planning
    never made a lot at the town edge, and garbage coverage stayed 0 all run."""
    from civitas.fleet.tools import edge_site, land_fix
    env = _boxed_in(free=[(12, 23)])                               # its one free lot has homes all round
    block = _ready_sites(briefing(env, "planning", mark=False))
    i, = [i for i, ln in enumerate(block) if ln.startswith("- blocked for lack of land: ")]
    assert "health (landfill: away from homes)" in block[i], block[i]
    assert block[i + 1] == ("- away from homes (health's landfill): leave a lot free by (12,9), the far end of "
                            "road_extension (12,23)-(12,9), with no homes within 4 tiles")
    env = _boxed_in(rows=(21, 22, 23, 25, 26, 27))                 # no branch within reach, no dead end
    block = _ready_sites(briefing(env, "planning", mark=False))
    assert ("- away from homes (health's landfill): extend a road into open land at the town edge and leave a lot free "
            "by its far end, with no homes within 4 tiles") in block, block
    w = env.world
    assert land_fix(w, "police", "jail") == land_fix(w, "health", "landfill") == "away from homes"
    among, clear = {"x2": 12, "y2": 23}, {"x2": 12, "y2": 9}         # a far end among the homes is skipped
    assert edge_site(w, [among, clear]) is clear and edge_site(w, [among]) is None


def _planning_line(env) -> str:
    return next(ln for ln in briefing(env, "mayor", mark=False).splitlines() if ln.startswith("- planning: "))


def test_the_mayor_sees_what_planning_needs_for_its_road():
    """Live 120-month run, seed 0, month 9: the mayor read planning "idle $1,243", took that and $1,000 more, and
    planning's $150 extension (24,20)-(24,5) failed twice on "the planning budget only has $0"; with nothing priced,
    its line on the mayor's Departments block never showed the need. A road is priced as the actions charge it, with
    zoning the land it opens at low density."""
    from civitas.fleet.tools import IDLE_MIN, DeptMoney, road_max
    env = _env()                                       # seed 0: the highway road's dead end (19,24), the river ahead
    w = env.world
    _months_spent(env, {}, {}, {})
    w.dept_funds["planning"] = 0
    assert _planning_line(env) == ("- planning: funds $0, spent $0; needs $600 for a road into open land and zoning it "
                                   "(opens 60 tiles)")
    road = [ln for ln in _ready_sites(briefing(env, "planning", mark=False)) if ln.startswith("- road_extension: ")]
    assert road == ["- road_extension: 1 site, best (19,24)-(34,24), opens 60 land tiles; $300 road + $300 zoning - "
                    "you have $0"]
    assert 12 * C.CATALOG["road"]["cost"] + 3 * C.CATALOG["bridge"]["cost"] == 300       # 12 land tiles, 3 bridge
    assert 60 * C.CATALOG["zone"]["cost"] == 300                                           # zoned at low density
    w.dept_funds["planning"] = 599.5
    assert _planning_line(env).endswith("; needs $600 for a road into open land and zoning it (opens 60 tiles)")
    for funds in (600, 1000):                          # enough for it: no needs line
        w.dept_funds["planning"] = funds
        assert _planning_line(env) == f"- planning: funds ${funds:,}, spent $0", funds
    assert DeptMoney(env, "planning").keep == IDLE_MIN                     # the $1,000 floor covers this one
    assert road_max(w) == 15 * C.CATALOG["bridge"]["cost"] + 2 * C.CATALOG["bulldoze"]["cost"] + 100 * C.CATALOG[
        "zone"]["cost"] == 1410                        # not the dearest: 15 bridge tiles, 2 to bulldoze, 100 to zone
    env = _boxed_in(lots=[(12, 23)])                   # no dead end: its branch road, dezoning free
    env.world.dept_funds["planning"] = 0
    assert _planning_line(env) == ("- planning: funds $0; needs $525 for a road into open land and zoning it (opens 75 "
                                   "tiles)")
    env = _boxed_in()                                  # only buildings beside the road: $5 to bulldoze one
    env.world.dept_funds["planning"] = 0
    assert _planning_line(env) == ("- planning: funds $0; needs $530 for a road into open land and zoning it (opens 75 "
                                   "tiles)")
    road, = [ln for ln in _ready_sites(briefing(env, "planning", mark=False)) if ln.startswith("- road_extension: ")]
    assert road.endswith(", opens 75 land tiles; $155 road + $375 zoning - you have $0"), road
    env = _boxed_in(rows=(21, 22, 23, 25, 26, 27))     # no road to build: no needs line
    env.world.dept_funds["planning"] = 0
    assert _planning_line(env) == "- planning: funds $0"


def test_planning_s_needs_line_covers_zoning_the_land_its_road_opens():
    """Smoke v18 seed 0, month 9: the mayor read "needs $50 for a road into open land (opens 15 tiles)", moved exactly
    $50, and planning had nothing left to zone the land the road opened. Zoning is priced as zone charges it at low
    density (a balance override reaches it), and planning keeps its road and zoning when they top the $1,000 floor."""
    from civitas.fleet.tools import DeptMoney, road_max
    env = _env()
    w = env.world
    _months_spent(env, {}, {}, {})
    w.dept_funds["planning"] = 400                     # the road's $300, not its zoning
    assert _planning_line(env).endswith("; needs $600 for a road into open land and zoning it (opens 60 tiles)")
    w.dept_funds["planning"] = 2600
    assert DeptMoney(env, "planning").idle == 1600     # its $600 road and zoning within the $1,000 it keeps
    was = C.CATALOG["zone"]["cost"]
    try:
        C.CATALOG["zone"]["cost"] = 8
        w.dept_funds["planning"] = 400
        assert _planning_line(env).endswith("; needs $780 for a road into open land and zoning it (opens 60 tiles)")
        road, = [ln for ln in _ready_sites(briefing(env, "planning", mark=False)) if ln.startswith("- road_extension")]
        assert road.endswith("; $300 road + $480 zoning - you have $400"), road
        assert road_max(w) == 910 + 800
        C.CATALOG["zone"]["cost"] = 20                 # $300 road + $1,200 zoning: over the floor
        w.dept_funds["planning"] = 2600
        m = DeptMoney(env, "planning")
        assert (m.keep, m.idle) == (1500, 1100)
    finally:
        C.CATALOG["zone"]["cost"] = was


def test_planning_s_road_is_no_building_for_the_other_lines(grown_env):
    """Planning's road is priced on its own line and the mayor's needs line only: never a funded building to leave
    a lot for (funded_firsts, claim_spots), a blocked one (land_blocked), a first building to fund (its kind is
    growth), or its budget line's "can't pay"."""
    from civitas.fleet.tools import dept_lists, funded_sites, funding_gaps, land_blocked
    env = copy.deepcopy(grown_env)
    w = env.world
    w.dept_funds.update({d: 50000 for d in DEPTS})     # every department funded, planning too
    depts = dept_lists(env)
    assert "planning" not in depts and not any(d == "planning" for d, _ in funded_sites(w, depts))
    assert "planning" not in land_blocked(env, depts)
    block = _ready_sites(briefing(env, "planning", mark=False))
    leave, = [ln for ln in block if ln.startswith("- leave free (funded): ")]
    assert "road" not in leave and "planning" not in leave, leave
    road, = [ln for ln in block if ln.startswith("- road_extension: ")]
    assert re.search(r"opens \d+ land tiles; \$[\d,]+ road \+ \$[\d,]+ zoning$", road), road   # on its own line
    w.dept_funds.update({d: 0 for d in DEPTS})         # every department short
    need, = funding_gaps(env, FleetState.of(env)).depts["planning"].needs.named
    assert (need.what, need.have, need.first) == ("a road into open land and zoning it", None, False)
    f = funding_gaps(env, FleetState.of(env))
    assert f.depts["planning"].kind == "growth" and "planning" not in f.firsts_at_zero
    assert "can't pay" not in briefing(env, "planning", mark=False)


# ------------------------------------------------------------ explained failures
AROUND = "; route around the building (the Map block has the road extent)"


def test_road_through_empty_zoned_lots_says_to_dezone_them_first():
    """Seed 0, v8 month 0: planning ran the highway road on to x 28, zoned R along row 23 and built roads north
    through the lots it had just zoned (still empty; dezoning is free), then re-built the row-24 road."""
    env = _env()
    w = env.world
    w.dept_funds["planning"] = 1000
    box = ToolBox(env, "planning")
    row = {"x1": 19, "y1": 24, "x2": 28, "y2": 24}
    assert box.call("build_road", row)["ok"]
    res = box.call("build_road", row)
    assert res["ok"] is False and res["message"] == (
        "every tile on that line is already road - the road line y=24 x 0-28 is already built (you built y=24 x 19-28 "
        "at call 1 this turn); branch off it into free land (find_sites road_extension)"), res
    assert box.call("zone", {"zone": "R", "x1": 19, "y1": 21, "x2": 28, "y2": 23})["ok"]
    core = "tile (19,23) is occupied by residential; bulldoze it or route around"
    road = {"x1": 19, "y1": 24, "x2": 19, "y2": 20}
    res = box.call("build_road", road)
    assert res["ok"] is False and res["message"] == res["error"] == (
        core + " - in the way: empty zoned lots (19,23), (19,22), (19,21); dezone the lots first (free): "
        "dezone x1=19,y1=21,x2=19,y2=23, then retry"), res
    assert box.executed[-1]["message"] == core                    # the log and the trajectories keep the core message
    assert FleetState.of(env).history(kind="failure", role="planning")[-1]["message"] == core
    again = box.call("build_road", road)                          # the refused repeat quotes the advice
    assert again["refused"] == "repeat" and again["error"].startswith(f"this exact call failed at call 4: {core} - ")
    assert box.call("dezone", {"x1": 19, "y1": 21, "x2": 19, "y2": 23})["message"] == "de-zoned 3 tiles"
    res = box.call("build_road", road)
    assert res["ok"] and res["message"] == "built 4 road tiles", res          # a success gets no advice
    res = box.call("build_road", {"x1": 21, "y1": 24, "x2": 21, "y2": 23})
    assert res["message"].endswith(" - in the way: empty zoned lot (21,23); dezone the lot first (free): "
                                   "dezone x1=21,y1=23,x2=21,y2=23, then retry"), res
    res = box.call("build_road", {"x1": 20, "y1": 22, "x2": 25, "y2": 22})   # names 3 lots, the box covers all
    assert res["message"].endswith(" - in the way: empty zoned lots (20,22), (21,22), (22,22) +3 more; dezone the "
                                   "lots first (free): dezone x1=20,y1=22,x2=25,y2=22, then retry"), res
    w.dept_funds["planning"] = 0                                  # other failures are left as they are
    res = box.call("build_road", {"x1": 19, "y1": 20, "x2": 10, "y2": 20})
    assert res["message"] == "costs $90 but the planning budget only has $0; ask the mayor for funds", res


def test_zoning_high_density_before_town_says_zone_low():
    env = _env()
    w = env.world
    w.dept_funds["planning"] = 1000
    box = ToolBox(env, "planning")
    rect = {"x1": 3, "y1": 25, "x2": 8, "y2": 25}
    res = box.call("zone", {"zone": "R", "density": "high", **rect})
    assert res["ok"] is False and res["message"] == (
        "high density unlocks when the city reaches 1,000 residents (milestone 'Town') - zone with density \"low\" now")
    assert box.executed[-1]["message"] == w.unlock_note("high_density")
    assert box.call("zone", {"zone": "O", **rect})["message"] == w.unlock_note("office_zone")    # not about density
    res = box.call("zone", {"zone": "R", "density": "low", **rect})
    assert res["ok"] and res["message"] == "zoned 6 tiles R (low density)", res
    assert box.call("zone", {"zone": "C", **rect})["message"] == (   # row 24 is road, row 26 off the road, and the
        "no empty land tiles in that area - all 6 tiles are zoned R; you zoned them at call 3 this turn")  # 1-tile end
                                                                     # caps aren't offered as single lots


NO_LAND = "no empty land tiles in that area"


def test_zoning_the_road_just_built_says_so_and_where_to_zone_beside_it():
    """Smoke v9, seed 0 month 9: planning built (35,23)-(35,8) north off the main road, then zoned R x=35 y 9-22, the
    road itself, and was told only "no empty land tiles in that area" (11 times in v9-v11)."""
    env = _env()
    w = env.world
    w.dept_funds["planning"] = 10000
    box = ToolBox(env, "planning")
    assert box.call("build_road", {"x1": 19, "y1": 24, "x2": 47, "y2": 24})["ok"]      # the main road, as in v9
    assert box.call("build_road", {"x1": 35, "y1": 23, "x2": 35, "y2": 8})["ok"]
    assert (w.terrain[8:24, 34:37] == C.LAND).all() and (w.struct[8:24, 35] == C.ROAD).all()
    rect = {"x1": 35, "y1": 9, "x2": 35, "y2": 22}
    west, east = {"x1": 34, "y1": 9, "x2": 34, "y2": 22}, {"x1": 36, "y1": 9, "x2": 36, "y2": 22}
    res = box.call("zone", {"zone": "R", **rect})
    assert res["ok"] is False and res["message"] == res["error"] == (
        f"{NO_LAND} - all 14 tiles are road; zone beside it instead: x1=34,y1=9,x2=34,y2=22 or x1=36,y1=9,x2=36,y2=22")
    assert box.executed[-1]["message"] == NO_LAND                 # the log and the trajectories keep the core message
    assert FleetState.of(env).history(kind="failure", role="planning")[-1]["message"] == NO_LAND
    res = box.call("zone", {"zone": "R", **west})                 # the suggestion works, and a success gets no advice
    assert res["ok"] and res["message"] == "zoned 14 tiles R (low density)", res
    assert box.call("zone", {"zone": "R", "x1": 36, "y1": 9, "x2": 36, "y2": 12})["ok"]
    res = box.call("zone", {"zone": "C", **rect})                 # west taken; east trimmed to its free roadside land
    assert res["message"] == f"{NO_LAND} - all 14 tiles are road; zone beside it instead: x1=36,y1=13,x2=36,y2=22", res
    res = box.call("zone", {"zone": "R", "x1": 35, "y1": 21, "x2": 35, "y2": 23})   # west: 1 free tile of 3 (too few)
    assert res["message"] == f"{NO_LAND} - all 3 tiles are road; zone beside it instead: x1=36,y1=21,x2=36,y2=23", res
    assert box.call("zone", {"zone": "R", **east})["message"] == "zoned 10 tiles R (low density)"


def test_failed_zone_counts_a_mixed_area_and_ranks_the_sides():
    env = _env()
    w = env.world
    w.dept_funds["planning"] = 10000
    box = ToolBox(env, "planning")
    assert box.call("build_road", {"x1": 19, "y1": 24, "x2": 47, "y2": 24})["ok"]      # bridges x 26-28
    res = box.call("zone", {"zone": "R", "x1": 25, "y1": 23, "x2": 29, "y2": 23})      # the river between
    assert res["message"] == "zoned 2 tiles R (low density)", res
    w.level[23, 25] = 1
    assert (w.terrain[23, 26:29] == C.WATER).all() and (w.terrain[25, 27:30] == C.WATER).all()
    res = box.call("zone", {"zone": "C", "x1": 25, "y1": 23, "x2": 29, "y2": 24})
    assert res["message"] == (f"{NO_LAND} - 5 road, 3 river, 2 zoned R (1 developed); you zoned 2 of them at call 2 "
                              "this turn; zone beside it instead: x1=24,y1=23,x2=24,y2=23 or x1=30,y1=23,x2=30,y2=23"
                              ), res                                                          # 2-tile sides: 1 lot is half
    assert box.call("zone", {"zone": "C", "x1": 24, "y1": 23, "x2": 24, "y2": 23})["ok"]   # (south: only (25,25) is
    hall = {"x1": 17, "y1": 23, "x2": 17, "y2": 23}                                        # free, off the bank)
    res = box.call("zone", {"zone": "R", **hall})                  # one tile: the town hall, and 1-tile sides
    assert res["message"] == (f"{NO_LAND} - the tile is a town hall; zone beside it instead: x1=16,y1=23,x2=16,y2=23 "
                              "or x1=18,y1=23,x2=18,y2=23"), res
    assert box.call("zone", {"zone": "R", "x1": 16, "y1": 23, "x2": 18, "y2": 23})["ok"]    # around the town hall
    assert box.call("zone", {"zone": "C", **hall})["message"] == f"{NO_LAND} - the tile is a town hall"   # none free,
    res = box.call("zone", {"zone": "C", "x1": 16, "y1": 23, "x2": 18, "y2": 24})                      # none its own
    assert res["message"] == (f"{NO_LAND} - 3 road, 2 zoned R, 1 town hall; you zoned 2 of them at call 6 this turn; "
                              "zone beside it instead: x1=16,y1=25,x2=18,y2=25 or x1=15,y1=23,x2=15,y2=23"), res


def test_zoning_over_its_own_zone_names_the_call():
    """Smoke v14 seed 0 month 6: planning zoned C (45,27)-(46,30) at call 17 and I on the same rectangle at call 18
    ("all 8 tiles are zoned C"); in v12-v16, 14 of the 25 zones on land that wasn't free came after the role's own
    zone of some of those tiles that turn. The hint names the call(s) and how many of the tiles they zoned (those
    still zoned; "them" for all), and keeps the strips beside it; the log keeps the core message."""
    env = _env()
    w = env.world
    w.dept_funds["planning"] = 1000
    box = ToolBox(env, "planning")
    rect = {"x1": 3, "y1": 22, "x2": 8, "y2": 23}
    assert box.call("zone", {"zone": "C", **rect})["ok"]                                        # call 1
    res = box.call("zone", {"zone": "I", **rect})
    assert res["message"] == (f"{NO_LAND} - all 12 tiles are zoned C; you zoned them at call 1 this turn; zone beside "
                              "it instead: x1=2,y1=23,x2=2,y2=23 or x1=9,y1=23,x2=9,y2=23"), res
    assert box.executed[-1]["message"] == NO_LAND
    assert FleetState.of(env).history(kind="failure", role="planning")[-1]["message"] == NO_LAND
    assert box.call("zone", {"zone": "R", "x1": 9, "y1": 23, "x2": 12, "y2": 23})["ok"]       # call 3
    res = box.call("zone", {"zone": "I", "x1": 7, "y1": 23, "x2": 10, "y2": 24})              # two calls, and road
    assert res["message"] == (f"{NO_LAND} - 4 road, 2 zoned C, 2 zoned R; you zoned 2 of them at call 1 and 2 at call "
                              "3 this turn; zone beside it instead: x1=7,y1=25,x2=10,y2=25"), res
    assert box.call("dezone", {"x1": 5, "y1": 23, "x2": 6, "y2": 23})["ok"]                   # call 5
    assert box.call("build_road", {"x1": 5, "y1": 23, "x2": 6, "y2": 23})["ok"]               # call 6: road now
    res = box.call("zone", {"zone": "I", "x1": 5, "y1": 22, "x2": 6, "y2": 23})
    assert res["message"] == f"{NO_LAND} - 2 zoned C, 2 road; you zoned 2 of them at call 1 this turn", res
    box = ToolBox(env, "planning")                                                              # a new turn: not its own
    res = box.call("zone", {"zone": "I", **rect})
    assert res["message"].startswith(f"{NO_LAND} - 10 zoned C, 2 road; zone beside it instead: "), res


def test_failed_zone_names_the_structures_and_caps_the_kinds():
    env = _env()
    w = env.world
    w.dept_funds["planning"] = 10000
    for a in ({"type": "zone", "zone": "I", "x1": 15, "y1": 22, "x2": 16, "y2": 22},
              {"type": "build_park", "x1": 17, "y1": 22, "x2": 18, "y2": 22},
              {"type": "zone", "zone": "C", "x1": 19, "y1": 22, "x2": 19, "y2": 22},
              {"type": "zone", "zone": "R", "x1": 15, "y1": 23, "x2": 16, "y2": 23},
              {"type": "zone", "zone": "C", "x1": 18, "y1": 23, "x2": 18, "y2": 23},
              {"type": "build_park", "x": 19, "y": 23}):
        assert apply_action(w, "planning", a).ok, a
    res = ToolBox(env, "planning").call("zone", {"zone": "R", "x1": 15, "y1": 22, "x2": 19, "y2": 24})
    assert res["message"] == (f"{NO_LAND} - 5 road, 3 park, 2 zoned I, 2 zoned C, 2 zoned R +1 other tile; zone "
                              "beside it instead: x1=15,y1=25,x2=19,y2=25 or x1=20,y1=23,x2=20,y2=24"), res
    assert w.struct[23, 17] == C.TOWNHALL                          # the other tile; west: 1 lot of 3 is too few


def test_failed_zone_names_one_structure_with_its_article():
    from civitas.fleet.tools import _contents
    w = _env().world
    for struct, what in ((C.UNIVERSITY, "a university"), (C.AIRPORT, "an airport"), (C.SCHOOL, "an elementary school"),
                         (C.CLINIC, "a clinic")):
        w.struct[5, 5] = struct
        assert _contents(w, [(5, 5)]) == f"the tile is {what}"


def test_failed_zone_with_no_roadside_land_beside_it_gets_the_counts_only():
    """Land beside the rectangle counts only when it touches a highway-connected road."""
    env = _env()
    w = env.world
    w.dept_funds["planning"] = 1000
    assert (w.terrain[29:31, 28:32] == C.WATER).all() and w.terrain[29, 27] == w.terrain[30, 32] == C.LAND
    res = ToolBox(env, "planning").call("zone", {"zone": "R", "x1": 28, "y1": 29, "x2": 31, "y2": 30})
    assert res["ok"] is False and res["message"] == f"{NO_LAND} - all 8 tiles are river", res


def test_zone_hints_keep_the_riverbank_lots_while_a_pump_or_outlet_is_missing():
    """v7: zoning took the last free riverbank lots beside roads and utilities had nowhere for its pump or outlet. The
    free bank lots here are (25,23) and (29,23) north of the bridge road, (26,25) and (30,25) south of it."""
    env = _env()
    w = env.world
    w.dept_funds["planning"] = w.dept_funds["utilities"] = 10000
    box = ToolBox(env, "planning")
    assert box.call("build_road", {"x1": 19, "y1": 24, "x2": 47, "y2": 24})["ok"]
    assert {(int(x), int(y)) for y, x in zip(*np.nonzero(free_bank(w)))} == {(25, 23), (29, 23), (26, 25), (30, 25)}
    road = {"zone": "R", "x1": 24, "y1": 24, "x2": 30, "y2": 24}    # over the bridge
    skip = f"{NO_LAND} - all 7 tiles are road; zone beside it instead: x1=24,y1=25,x2=25,y2=25"
    assert box.call("zone", road)["message"] == skip     # north's rect would cover (25,23), (29,23); south: trimmed
    assert apply_action(w, "utilities", {"type": "build_water", "kind": "tower", "x": 10, "y": 25}).ok
    assert box.call("zone", {**road, "zone": "C"})["message"] == skip       # the outlet still needs one
    assert apply_action(w, "utilities", {"type": "build_sewage", "kind": "outlet", "x": 30, "y": 25}).ok
    box = ToolBox(env, "planning")                                   # a new turn: the same call isn't a repeat
    res = box.call("zone", road)                                     # both exist: the bank is land like any other
    assert res["message"] == (f"{NO_LAND} - all 7 tiles are road; zone beside it instead: x1=24,y1=23,x2=30,y2=23 or "
                              "x1=24,y1=25,x2=26,y2=25"), res
    res = box.call("zone", {"zone": "R", "x1": 24, "y1": 23, "x2": 30, "y2": 23})
    assert res["ok"] and res["message"] == "zoned 4 tiles R (low density)", res


def test_zone_hints_stay_on_the_map():
    env = _env()
    w = env.world
    w.dept_funds["planning"] = 10000
    box = ToolBox(env, "planning")
    assert box.call("build_road", {"x1": 19, "y1": 24, "x2": 47, "y2": 24})["ok"]
    assert box.call("build_road", {"x1": 47, "y1": 23, "x2": 47, "y2": 15})["ok"]      # up the last column
    res = box.call("zone", {"zone": "R", "x1": 47, "y1": 16, "x2": 47, "y2": 22})      # east of it: off the map
    assert res["message"] == f"{NO_LAND} - all 7 tiles are road; zone beside it instead: x1=46,y1=16,x2=46,y2=22", res
    res = box.call("zone", {"zone": "R", "x1": 0, "y1": 24, "x2": 0, "y2": 24})        # the highway, on column 0
    assert res["message"] == (f"{NO_LAND} - the tile is highway; zone beside it instead: x1=0,y1=23,x2=0,y2=23 or "
                              "x1=0,y1=25,x2=0,y2=25"), res


def test_failed_dezone_says_what_is_there():
    env = _env()
    w = env.world
    w.dept_funds["planning"] = 1000
    box = ToolBox(env, "planning")
    core = "no zoned tiles in that area"
    res = box.call("dezone", {"x1": 0, "y1": 23, "x2": 3, "y2": 25})
    assert res["ok"] is False and res["message"] == f"{core} - 8 empty land, 3 road, 1 highway", res
    assert box.executed[-1]["message"] == core
    assert FleetState.of(env).history(kind="failure", role="planning")[-1]["message"] == core
    assert box.call("dezone", {"x1": 5, "y1": 24, "x2": 9, "y2": 24})["message"] == f"{core} - all 5 tiles are road"
    res = box.call("dezone", {"x1": 17, "y1": 23, "x2": 17, "y2": 23})      # one tile: the town hall
    assert res["message"] == f"{core} - the tile is a town hall", res
    assert box.call("zone", {"zone": "R", "x1": 1, "y1": 25, "x2": 3, "y2": 25})["ok"]
    res = box.call("dezone", {"x1": 0, "y1": 23, "x2": 3, "y2": 25})
    assert res["ok"] and res["message"] == "de-zoned 3 tiles", res
    res = box.call("dezone", {"x1": 0, "y1": 0, "x2": 47, "y2": 47})       # other failures are left as they are
    assert res["message"] == "rectangle has 2304 tiles; max 150 per action", res


def test_rebuilding_a_road_names_the_road_line_it_is_on():
    """Smoke v9-v11: 13 x "every tile on that line is already road", e.g. rebuilding (0,24)-(47,24), the main road,
    told only the network's extent."""
    env = _env()
    w = env.world
    w.dept_funds["planning"] = 1000
    box = ToolBox(env, "planning")
    tail = "; branch off {} into free land (find_sites road_extension)"
    res = box.call("build_road", {"x1": 0, "y1": 24, "x2": 19, "y2": 24})
    assert res["ok"] is False and res["message"] == (
        "every tile on that line is already road - the road line y=24 x 0-19 is already built" + tail.format("it")), res
    assert box.executed[-1]["message"] == "every tile on that line is already road"
    road = apply_action(w, "planning", {"type": "build_road", "x1": 10, "y1": 23, "x2": 10, "y2": 18})  # not this turn's
    assert road.message == "built 6 road tiles"
    res = box.call("build_road", {"x1": 3, "y1": 24, "x2": 10, "y2": 18})          # an L along both lines
    assert res["message"].endswith(" - the road lines y=24 x 0-19 and x=10 y 18-24 are already built"
                                   + tail.format("them")), res
    assert len(res["message"]) <= 200
    res = box.call("build_road", {"x1": 10, "y1": 24, "x2": 10, "y2": 20})         # crosses y=24: not along it
    assert res["message"].endswith(" - the road line x=10 y 18-24 is already built" + tail.format("it")), res
    res = box.call("build_road", {"x1": 5, "y1": 24, "x2": 5, "y2": 24})           # a single tile
    assert res["message"].endswith(" - the road line y=24 x 0-19 is already built" + tail.format("it")), res
    stub = {"x1": 7, "y1": 25, "x2": 7, "y2": 25}                                  # a 1-tile stub: on no road line
    assert apply_action(w, "planning", {"type": "build_road", **stub}).ok
    res = box.call("build_road", stub)
    assert res["message"] == ("every tile on that line is already road - highway-connected roads: x 0-19, y 18-25, "
                              "27 tiles; start on the road and run into free land (find_sites road_extension)"), res


def test_rebuilding_its_own_new_road_names_the_call():
    """Smoke v12-v16, 4 of the 13 "already road": v15 seed 0 month 3, planning built (19,43)-(28,43) at call 5 and
    again at call 10; v16 seed 0 month 0, (19,24)-(34,24) at call 2, then (0,24)-(19,24), the old road up to it, at
    call 6; told only the road line, it took its own new road for someone else's. The hint names this turn's
    build_road calls along the failed line (a road it crosses isn't along it); the log keeps the core message."""
    env = _env()
    w = env.world
    w.dept_funds["planning"] = 5000
    box = ToolBox(env, "planning")
    core = "every tile on that line is already road"
    tail = " into free land (find_sites road_extension)"
    assert box.call("build_road", {"x1": 19, "y1": 24, "x2": 40, "y2": 24})["ok"]              # call 1
    assert box.call("build_road", {"x1": 35, "y1": 23, "x2": 35, "y2": 15})["ok"]              # call 2: north off it
    assert box.call("city_overview", {})["ok"]
    res = box.call("build_road", {"x1": 0, "y1": 24, "x2": 19, "y2": 24})                      # the old road up to it
    assert res["message"] == (f"{core} - the road line y=24 x 0-40 is already built (you built y=24 x 19-40 at call 1 "
                              "this turn); branch off it" + tail), res
    assert box.executed[-1]["message"] == core
    assert FleetState.of(env).history(kind="failure", role="planning")[-1]["message"] == core
    res = box.call("build_road", {"x1": 35, "y1": 24, "x2": 35, "y2": 18})                     # crosses call 1's road
    assert res["message"] == (f"{core} - the road line x=35 y 15-24 is already built (you built x=35 y 15-23 at call 2 "
                              "this turn); branch off it" + tail), res
    res = box.call("build_road", {"x1": 30, "y1": 24, "x2": 35, "y2": 20})                     # an L along both
    assert res["message"] == (f"{core} - the road lines y=24 x 0-40 and x=35 y 15-24 are already built (you built "
                              "y=24 x 19-40 at call 1 and x=35 y 15-23 at call 2 this turn); branch off them" + tail), res
    assert box.call("build_road", {"x1": 7, "y1": 25, "x2": 7, "y2": 25})["ok"]                # call 7: a 1-tile stub
    res = box.call("build_road", {"x1": 7, "y1": 25, "x2": 7, "y2": 25})
    assert res["message"] == (f"{core} - you built (7,25) at call 7 this turn; highway-connected roads: x 0-40, y "
                              "15-25, 51 tiles; start on the road and run" + tail), res
    box = ToolBox(env, "planning")                                                              # a new turn: not its own
    res = box.call("build_road", {"x1": 0, "y1": 24, "x2": 19, "y2": 24})
    assert res["message"] == f"{core} - the road line y=24 x 0-40 is already built; branch off it" + tail, res


def _developed_below_free_land(w):
    """A developed zone tile (x, y) of the grown city with a zoned tile south of it and free land on the 3 x 3 block
    north of it."""
    zoned = np.isin(w.struct, list(C.ZONES))
    free = (w.struct == C.NONE) & (w.terrain == C.LAND)
    return next((int(x), int(y)) for y, x in zip(*np.nonzero(zoned & (w.level > 0)))
                if 0 < x < w.w - 1 and 3 <= y < w.h - 1 and free[y - 3:y, x - 1:x + 2].all() and zoned[y + 1, x])


def test_road_into_buildings_says_to_route_around_them(grown_env):
    from civitas.fleet.tools import REPEAT_ERROR_CHARS
    env = copy.deepcopy(grown_env)
    w = env.world
    w.dept_funds["planning"] = 1000
    x, y = _developed_below_free_land(w)
    box = ToolBox(env, "planning")
    road = {"x1": x, "y1": y - 3, "x2": x, "y2": y}                # free land, then the building
    res = box.call("build_road", road)
    assert res["message"] == (f"tile ({x},{y}) is occupied by {C.STRUCT_NAME[int(w.struct[y, x])]}; bulldoze it or "
                              f"route around - in the way: building ({x},{y}){AROUND}"), res
    assert box.call("zone", {"zone": "R", "x1": x - 1, "y1": y - 2, "x2": x + 1, "y2": y - 1})["ok"]
    res = box.call("build_road", road)                             # lots, then the building
    assert res["message"] == (f"tile ({x},{y - 2}) is occupied by residential; bulldoze it or route around - in the "
                              f"way: empty zoned lots ({x},{y - 2}), ({x},{y - 1}) and building ({x},{y}); dezone the "
                              f"lots first (free): dezone x1={x},y1={y - 2},x2={x},y2={y - 1}{AROUND}"), res
    assert len(res["message"]) < REPEAT_ERROR_CHARS
    res = box.call("build_road", {"x1": x, "y1": y - 1, "x2": x, "y2": y})
    assert res["message"].endswith(f" - in the way: empty zoned lot ({x},{y - 1}) and building ({x},{y}); dezone the lot "
                                   f"first (free): dezone x1={x},y1={y - 1},x2={x},y2={y - 1}{AROUND}"), res


def _fits(w, struct, x, y) -> bool:
    try:
        return bool(w.footprint_tiles(struct, x, y))
    except ValueError:
        return False


def test_a_multi_tile_building_is_named_once(grown_env):
    env = copy.deepcopy(grown_env)
    w = env.world
    w.dept_funds["planning"] = 1000
    x, y = next((x, y) for y in range(w.h) for x in range(w.w) if _fits(w, C.HOSPITAL, x, y))
    w.place_building(C.HOSPITAL, x, y)
    w.refresh()
    res = ToolBox(env, "planning").call("build_road", {"x1": x, "y1": y + 1, "x2": x + 1, "y2": y + 1})   # bottom row
    assert res["message"] == (f"tile ({x},{y + 1}) is occupied by hospital; bulldoze it or route around - in the way: "
                              f"building ({x},{y + 1}){AROUND}"), res           # its first tile on the line, not (x,y)


def test_dezone_advice_covers_both_legs_of_an_l():
    env = _env()
    w = env.world
    w.dept_funds["planning"] = 1000
    box = ToolBox(env, "planning")
    assert box.call("zone", {"zone": "R", "x1": 3, "y1": 20, "x2": 10, "y2": 23})["ok"]
    road = {"x1": 4, "y1": 21, "x2": 8, "y2": 23}                   # row 21, then down column 8 to the road on row 24
    res = box.call("build_road", road)
    assert res["message"] == (
        "tile (4,21) is occupied by residential; bulldoze it or route around - in the way: empty zoned lots (4,21), "
        "(5,21), (6,21) +4 more; dezone the lots first (free): dezone x1=4,y1=21,x2=8,y2=21; dezone x1=8,y1=22,x2=8,"
        "y2=23, then retry"), res                                   # not the 15-tile box: rects stay on the line
    for rect in ({"x1": 4, "y1": 21, "x2": 8, "y2": 21}, {"x1": 8, "y1": 22, "x2": 8, "y2": 23}):
        assert box.call("dezone", rect)["ok"]
    assert box.call("build_road", road)["message"] == "built 7 road tiles"
    assert (w.struct[20, 3:11] == C.RES).all() and w.struct[22, 7] == C.RES and w.struct[22, 9] == C.RES


def test_dezone_advice_stays_between_buildings(grown_env):
    env = copy.deepcopy(grown_env)
    w = env.world
    w.dept_funds["planning"] = 1000
    x, y = _developed_below_free_land(w)
    box = ToolBox(env, "planning")
    south = {"x1": x, "y1": y + 1, "x2": x, "y2": y + 1}
    assert box.call("dezone", south)["ok"] and box.call("zone", {"zone": "R", **south})["ok"]     # an empty lot
    assert box.call("zone", {"zone": "R", "x1": x, "y1": y - 2, "x2": x, "y2": y - 1})["ok"]
    road = {"x1": x, "y1": y + 1, "x2": x, "y2": y - 2}            # lot, building, lot, lot
    res = box.call("build_road", road)
    rects = [{"x1": x, "y1": y + 1, "x2": x, "y2": y + 1}, {"x1": x, "y1": y - 2, "x2": x, "y2": y - 1}]
    assert res["message"].endswith(
        f" - in the way: empty zoned lots ({x},{y + 1}), ({x},{y - 1}), ({x},{y - 2}) and building ({x},{y}); dezone "
        f"the lots first (free): dezone x1={x},y1={y + 1},x2={x},y2={y + 1}; dezone x1={x},y1={y - 2},x2={x},"
        f"y2={y - 1}{AROUND}"), res
    for rect in rects:
        assert box.call("dezone", rect)["ok"]
    assert w.level[y, x] > 0 and all(w.struct[r["y2"], x] == C.NONE for r in rects)      # the building survives
    box = ToolBox(env, "planning")                                 # a new turn: the action cap is per turn
    assert box.call("build_park", {"x": x, "y": y - 2})["ok"]      # lot, building, lot, park, lot: a third run
    assert box.call("zone", {"zone": "R", **south})["ok"]
    assert box.call("zone", {"zone": "R", "x1": x, "y1": y - 3, "x2": x, "y2": y - 1})["ok"]
    res = box.call("build_road", {"x1": x, "y1": y + 1, "x2": x, "y2": y - 3})
    assert res["message"].endswith(
        f" and buildings ({x},{y}), ({x},{y - 2}); dezone the lots first (free): dezone x1={x},y1={y + 1},x2={x},"
        f"y2={y + 1}; dezone x1={x},y1={y - 1},x2={x},y2={y - 1} +1 more; route around the buildings (the Map block "
        "has the road extent)"), res


def test_a_build_on_a_taken_tile_says_what_took_it_and_the_next_site():
    """Smoke v14 seed 1 month 0: utilities built its pump at (29,23) at call 5 and its outlet there at call 6, "(29,23)
    is not an empty land tile", then asked the mayor to free a riverbank lot. In v12-v16, 15 of the 19 builds on a
    taken tile (utilities 11, health 2, education 2) went where the role's own build had gone that turn. The hint
    says what holds the tile (this turn's own call, the role's own building, anything else) and names the next free
    site of that purpose from the role's list, or why there is none; the log keeps the core message."""
    from civitas.fleet.tools import role_obs, site_lists
    env = _env()
    w = env.world
    w.dept_funds["planning"] = w.dept_funds["utilities"] = w.dept_funds["police"] = 20000
    assert apply_action(w, "planning", {"type": "build_road", "x1": 19, "y1": 24, "x2": 47, "y2": 24}).ok
    box = ToolBox(env, "utilities")
    assert box.call("build_water", {"kind": "pump", "x": 25, "y": 23})["ok"]                 # call 1
    core = "(25,23) is not an empty land tile"
    res = box.call("build_sewage", {"kind": "outlet", "x": 25, "y": 23})
    assert res["ok"] is False and res["message"] == res["error"] == (
        f"{core} - (25,23) now holds your water pump (call 1 this turn); next free sewage site: (26,25)"), res
    assert box.executed[-1]["message"] == core
    assert FleetState.of(env).history(kind="failure", role="utilities")[-1]["message"] == core
    res = box.call("build_power_plant", {"kind": "wind", "x": 25, "y": 23})
    assert res["message"] == f"{core} - (25,23) now holds your water pump (call 1 this turn); next free wind site: " \
                             f"(41,25)", res
    res = box.call("build_water", {"kind": "tower", "x": 20, "y": 24})                       # a road
    assert res["message"] == ("(20,24) is not an empty land tile - (20,24) is road; next free near_homes site: (0,23)"
                              ), res
    assert apply_action(w, "planning", {"type": "zone", "zone": "R", "x1": 3, "y1": 25, "x2": 4, "y2": 25}).ok
    res = box.call("build_water", {"kind": "tower", "x": 3, "y": 25})
    assert res["message"].startswith("(3,25) is not an empty land tile - (3,25) is zoned R; next free near_homes "), res
    for x, y in ((29, 23), (26, 25), (30, 25)):                                               # the other bank lots
        assert apply_action(w, "planning", {"type": "zone", "zone": "R", "x": x, "y": y}).ok
    short = site_lists(w, "utilities", role_obs(env, "utilities"))["sewage"].short
    assert short.startswith("riverbank beside roads all taken")
    res = box.call("build_sewage", {"kind": "outlet", "x": 25, "y": 23, "tier": 1})
    assert res["message"] == f"{core} - (25,23) now holds your water pump (call 1 this turn); no free sewage site " \
                             f"listed ({short})", res
    res = ToolBox(env, "utilities").call("build_sewage", {"kind": "outlet", "x": 25, "y": 23})     # a new turn
    assert res["message"] == f"{core} - (25,23) holds your water pump; no free sewage site listed ({short})", res
    res = ToolBox(env, "police").call("build_police_station", {"x": 25, "y": 23})                 # not its own
    assert res["message"].startswith(f"{core} - (25,23) holds a water pump; "), res
    res = ToolBox(env, "police").call("build_police_station", {"x": 5, "y": 5})                  # other failures:
    assert res["message"] == "(5,5) does not touch a road; buildings must be next to a road", res   # as they are


@pytest.mark.parametrize("role, action, args, purpose", [
    ("utilities", "build_power_plant", {"kind": "coal"}, "coal_or_gas"),
    ("utilities", "build_power_plant", {"kind": "wind"}, "wind"),              # its own list before near_homes'
    ("utilities", "build_power_plant", {"kind": "Solar"}, "near_homes"),
    ("utilities", "build_power_plant", {"kind": "hydro"}, None),               # a dam goes on the river: no list
    ("utilities", "build_water", {"kind": "pumping station"}, "pump"),         # a higher tier: its line's list
    ("utilities", "build_water", {"kind": "pump", "tier": 3}, "riverbank_2x2"),   # water works
    ("utilities", "build_water", {"kind": "tower"}, "near_homes"),
    ("utilities", "build_sewage", {"kind": "outlet"}, "sewage"),
    ("police", "build_police_station", {}, "station"), ("police", "build_jail", {}, "jail"),
    ("fire", "build_fire_station", {}, "station"), ("fire", "build_shelter", {}, "shelter"),
    ("health", "build_clinic", {}, "clinic"), ("health", "build_hospital", {}, "hospital"),
    ("health", "build_cemetery", {}, "cemetery"),
    ("education", "build_school", {}, "school"), ("education", "build_school", {"kind": "high"}, "high_school"),
    ("education", "build_school", {"kind": "university"}, "university"), ("education", "build_library", {}, "library"),
    ("transport", "build_harbour", {}, "harbour"), ("transport", "build_train_station", {}, "train_station"),
    ("planning", "build_casino", {}, "casino"), ("planning", "build_landmark", {"kind": "plaza"}, None)])
def test_build_purpose_is_the_list_the_build_uses(role, action, args, purpose):
    from civitas.fleet.tools import build_purpose
    assert build_purpose(role, action, args) == purpose


# ---------------------------------------------------------------- briefing
from civitas.agents.prompts import GAME_RULES
from civitas.fleet.briefing import KEY_METRICS, RULE_EDITS, UNLOCK_ROLES, briefing, system_prompt
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
        assert "steps_left" in text and "identical call" in text
        assert "{" not in text.replace("{}", "")                         # every placeholder filled
    assert ("update_plan" in system_prompt("mayor", cfg, 36)) and "update_plan" not in system_prompt("fire", cfg, 36)
    bare = system_prompt("fire", FleetAgentConfig(), 36)
    assert "Coach's playbook" not in bare and "20 tool calls" in bare
    only_fleet = system_prompt("fire", FleetAgentConfig(fleet_playbook="FLEET-PB"), 36)
    assert "FLEET-PB" in only_fleet and "Fire" not in only_fleet.split("Coach's playbook")[1]


def test_system_prompt_fits_the_fleet():
    cfg = FleetAgentConfig()
    for old, new in RULE_EDITS:
        assert old in GAME_RULES, old                                      # a GAME_RULES edit must not no-op this
    mayor, fire = system_prompt("mayor", cfg, 36), system_prompt("fire", cfg, 36)
    for text in (mayor, fire):
        assert "tile H" not in text and "get_messages" in text.split("# Game rules")[1]
        assert "refused" in text and "autonomous = " in text and "supervised = " in text
        assert "Each game month is one turn" in text
    assert "find_sites (check which sites a department has; read-only)" in mayor and "x = column" not in mayor
    assert "find_sites" in fire and "x = column" in fire
    assert all("map_overview" in t and "map_overview only when you need tile-level layout (at most once per turn)" in t for t in (mayor, fire))
    slow = system_prompt("fire", cfg, 36, act_every=3)
    assert "every 3 months" in slow and "Each game month is one turn" not in slow


def test_every_unlock_has_an_owner():
    assert set(UNLOCK_ROLES) == set(C.UNLOCK_POP)
    for key, roles in UNLOCK_ROLES.items():
        assert roles and set(roles) <= set(ROLES), key


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
    from civitas.fleet.tools import MAP_PARTS, MAYOR_MAP, map_facts, map_lines
    env = copy.deepcopy(env12) if grown else CivitasEnv(GameConfig(seed=0, months=24))
    w = env.world
    for role in TURN_ORDER:
        text = briefing(env, role, mark=False)
        assert _words(text) < 700, (role, _words(text))
        if role == "planning":
            dev, = [ln for ln in text.splitlines() if "developed tiles" in ln]
            assert "R " in dev and "+" not in dev                        # counts, not signed demand values
        assert f"{24 - w.month} months left" in text and "autonomous" in text
        parts = MAYOR_MAP if role == "mayor" else MAP_PARTS
        assert "\n## Map\n" + "\n".join(map_lines(map_facts(w), parts)) + "\n" in text     # every role
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
    assert "reach 5k residents" in mayor and "no plan yet" not in mayor
    assert "PRIORITY-" not in mayor and "priorities and full text: get_plan" in mayor       # get_plan has them


def _ready_sites(text: str) -> list[str]:
    block = text[text.index("## Ready sites"):].split("\n\n")[0].splitlines()
    return block[1:]


@pytest.mark.parametrize("grown", [False, True])
def test_ready_sites_in_department_briefings(grown, grown_env):
    env = grown_env if grown else CivitasEnv(GameConfig(seed=0, months=24))
    w = env.world
    assert "## Ready sites" not in briefing(env, "mayor", mark=False)
    for role in DEPTS:
        text = briefing(env, role, mark=False)
        assert text.index("## Map") < text.index("## Ready sites") < text.index("## Your memory")
        lines = _ready_sites(text)
        purposes = list(SITE_PURPOSES[role])
        dense = role == "planning" and not w.unlocked("high_density")                 # + the high density line
        land = sum(role == "planning" and ln.startswith(("- blocked for lack of land: ", "- leave free (funded): ",
                                                          "- away from homes ("))
                   for ln in lines)
        assert len(lines) == len(purposes) + 1 + dense + land
        assert lines[-1] == "find_sites <purpose> lists up to 6 with details"
        box = ToolBox(env, role)
        for purpose, line in zip(purposes, lines):
            # bus_stops names every stop of its one line (up to C.MAX_STOPS)
            assert line.startswith(f"- {purpose}: ") and len(line) <= (170 if purpose == "bus_stops" else 110), \
                (role, line)
            res = box.call("find_sites", {"purpose": purpose})
            sites, note = res.get("sites", []), res.get("note", "")
            if "stops" in res and not note:
                assert line.startswith(f"- bus_stops: {res['line']}"), (role, line)
            elif note.startswith("locked"):
                pop = min(C.UNLOCK_POP[k] for k in SITE_UNLOCKS[purpose])
                assert f"locked (unlocks at {pop:,} residents)" in line, (role, line)
            elif sites:
                n = len(sites)
                assert line.startswith(f"- {purpose}: {n} site{'s' * (n != 1)}, "), (role, line)
                assert (f"best ({sites[0]['x']},{sites[0]['y']})" in line
                        or re.search(r"(best as|same (strip|site|list) as) \w+", line)), (role, line)
            else:
                assert ": none - " in line and note, (role, line)
    if not grown:                                       # fresh seed 0: the road stops short of the river
        util = _ready_sites(briefing(env, "utilities", mark=False))
        assert any("no connected road reaches the river" in ln for ln in util)
        assert any(ln.startswith("- wind: 6 sites, best (") and "wind 1." in ln for ln in util)
    assert w.month == (24 if grown else 0)


def test_ready_sites_never_repeat_a_best_site(grown_env):
    """Zoning ranks one strip first for several zone types: later lines show the next distinct strip, or say it's
    the same; zone_M reads zone_C's list by design."""
    fresh = CivitasEnv(GameConfig(seed=0, months=24))
    plan = _ready_sites(briefing(fresh, "planning", mark=False))
    assert plan[0].startswith("- zone_R: 2 sites, best (0,25)-(19,26)"), plan
    for line in plan[1:3]:                              # zone_C, zone_I: same top strip as zone_R
        assert "best (0,25)-(19,26)" not in line and ("best as zone_R; next (" in line
                                                     or "same strip as zone_R" in line), line
    for env in (fresh, grown_env):
        for role in DEPTS:
            shown = [m.group(1) for ln in _ready_sites(briefing(env, role, mark=False))
                     for m in [re.search(r"(?:best|next) (\(\d+,\d+\)(?:-\(\d+,\d+\))?)", ln)] if m]
            assert len(shown) == len(set(shown)), (role, shown)
    zone_m, = [ln for ln in _ready_sites(briefing(grown_env, "planning", mark=False)) if ln.startswith("- zone_M:")]
    assert zone_m.endswith("same list as zone_C"), zone_m


def test_planning_ready_sites_say_high_density_is_locked(env12):
    """Smoke v9-v11: 6 x zoning high density before 1,000 residents, repeated across turns. Planning's Ready sites
    say so while it's locked, last before the find_sites line; no other role's do."""
    w = env12.world
    pop = w.metrics()["population"]
    assert 0 < pop < 1000 and not w.unlocked("high_density")
    plan = _ready_sites(briefing(env12, "planning", mark=False))
    assert plan[-2:] == [f'- high density: locked until 1,000 residents (now {pop:,}): zone with density "low"',
                         "find_sites <purpose> lists up to 6 with details"], plan
    assert plan[-4 if plan[-3].startswith("- blocked") else -3].startswith("- casino: "), plan   # the last purpose
    for role in DEPTS[1:]:                                           # the other departments
        assert not any("high density" in ln for ln in _ready_sites(briefing(env12, role, mark=False))), role
    env = _env()
    env.world.milestones_reached.append("Town")                       # the 1,000 milestone
    assert env.world.unlocked("high_density")
    plan = _ready_sites(briefing(env, "planning", mark=False))
    assert not any("high density" in ln for ln in plan) and plan[-1].startswith("find_sites"), plan


def test_ready_sites_extra_fields():
    from civitas.fleet.tools import SiteList, ready_lines
    lists = {"near_homes": SiteList([{"x": 1, "y": 2, "homes_within_5": 0}], None, None),
             "sewage": SiteList([{"x": 3, "y": 4, "pollution": 0.2}], None, None),
             "pump": SiteList([{"x": 5, "y": 6, "pollution": 0.2}], None, None)}
    assert ready_lines("utilities", lists) == ["- near_homes: 1 site, best (1,2)", "- sewage: 1 site, best (3,4)",
                                               "- pump: 1 site, best (5,6), pollution 0.2"]


# find_sites purpose -> the build action for the cheapest building its sites are for, with every milestone reached
BUILD_ACTIONS = {
    ("utilities", "coal_or_gas"): {"type": "build_power_plant", "kind": "coal"},
    ("utilities", "wind"): {"type": "build_power_plant", "kind": "wind"},
    ("utilities", "pump"): {"type": "build_water", "kind": "pump"},
    ("utilities", "sewage"): {"type": "build_sewage", "kind": "outlet"},
    ("utilities", "near_homes"): {"type": "build_water", "kind": "tower"},
    ("utilities", "riverbank_2x2"): {"type": "build_water", "kind": "water_works"},
    ("transport", "train_station"): {"type": "build_train_station"},
    ("transport", "harbour"): {"type": "build_harbour"},
    ("transport", "metro_station"): {"type": "build_metro_station"},
    ("transport", "bus_stops"): {"type": "create_bus_line"},                # stops: the first two suggested
    ("police", "station"): {"type": "build_police_station"}, ("police", "jail"): {"type": "build_jail"},
    ("fire", "station"): {"type": "build_fire_station"}, ("fire", "shelter"): {"type": "build_shelter"},
    ("health", "clinic"): {"type": "build_clinic"}, ("health", "hospital"): {"type": "build_hospital"},
    ("health", "landfill"): {"type": "build_landfill"}, ("health", "cemetery"): {"type": "build_cemetery"},
    ("education", "school"): {"type": "build_school", "kind": "elementary"},
    ("education", "high_school"): {"type": "build_school", "kind": "high"},
    ("education", "university"): {"type": "build_school", "kind": "university"},
    ("education", "library"): {"type": "build_library"},
}


def _building_cost(w, role: str, purpose: str) -> int | None:
    """site_price for a whole building (or bus line), rounded; None per tile (levees, avenue tiles) or unpriced."""
    from civitas.fleet.tools import site_price
    p = site_price(w, role, purpose)
    return None if p is None or p.per_tile else round(p.cost)


def _build_action(role: str, purpose: str, sites: list[dict]) -> dict:
    """BUILD_ACTIONS' action on the listed site (a bus line: from the first two suggested stops)."""
    a = BUILD_ACTIONS[role, purpose]
    if a["type"] == "create_bus_line":
        return {**a, "stops": [[t["x"], t["y"]] for t in sites[:2]]}
    return {**a, "x": sites[0]["x"], "y": sites[0]["y"]}


@pytest.fixture(scope="module")
def open_env(grown_env):
    """The grown city with every milestone reached and a road to the riverbank near the south edge: every priced
    find_sites purpose has sites."""
    env = copy.deepcopy(grown_env)
    w = env.world
    w.milestones_on = False
    w.dept_funds["planning"] = 50000
    for line in ({"x1": 3, "y1": 29, "x2": 3, "y2": 46}, {"x1": 3, "y1": 46, "x2": 31, "y2": 46}):
        assert apply_action(w, "planning", {"type": "build_road", **line}).ok
    return env


@pytest.mark.parametrize("role,purpose", list(BUILD_ACTIONS))
def test_build_cost_is_what_the_build_charges(open_env, role, purpose):
    env = copy.deepcopy(open_env)
    w = env.world
    cost = _building_cost(w, role, purpose)
    res = ToolBox(env, role).call("find_sites", {"purpose": purpose})
    sites = res["sites"] if "sites" in res else [{"x": x, "y": y} for x, y in res["stops"]]    # bus_stops
    dept = ROLES[role]["budget"]
    action = _build_action(role, purpose, sites)
    w.dept_funds[dept] = cost - 1
    res = apply_action(w, role, action)
    assert not res.ok and res.message.startswith(f"costs ${cost:,} but the {dept} budget only has ${cost - 1:,}"), \
        res.message
    w.dept_funds[dept] = cost                          # the listed site takes it, for exactly that
    res = apply_action(w, role, action)
    assert res.ok and res.cost == cost and w.dept_funds[dept] == 0, res.message


def test_build_cost_prices_the_cheapest_unlocked_building(grown_env, open_env):
    from civitas.fleet.tools import Price, site_price
    build_cost = _building_cost
    w = grown_env.world                                # 2,6xx residents: hospitals, stations, 2x2 blocks locked
    cost = lambda line, tier=1: C.TIER_LINES[line][tier - 1]["cost"]
    assert build_cost(w, "utilities", "coal_or_gas") == cost("coal") < cost("gas")
    assert build_cost(w, "utilities", "near_homes") == cost("tower") < cost("wind") < cost("solar")
    assert build_cost(w, "education", "high_school") == cost("high_school")
    assert build_cost(w, "health", "cemetery") == cost("cemetery")
    for role, purpose in (("health", "hospital"), ("education", "university"), ("utilities", "riverbank_2x2"),
                          ("transport", "train_station"), ("transport", "harbour"), ("transport", "metro_station")):
        assert build_cost(w, role, purpose) is None, (role, purpose)
    w = open_env.world
    assert build_cost(w, "health", "hospital") == cost("clinic", 2)          # build_hospital: the clinic line's tier 2
    assert build_cost(w, "utilities", "riverbank_2x2") == cost("pump", 3)     # water works
    assert build_cost(w, "transport", "harbour") == C.HARBOUR_SPEC["cost"]
    priced = {(r, p) for r in SITE_PURPOSES for p in SITE_PURPOSES[r] if build_cost(w, r, p) is not None}
    assert priced == set(BUILD_ACTIONS)                # not planning, nor levees and avenues (per tile)
    line = C.BUS_LINE_UPKEEP + 2 * C.BUS_STOP_UPKEEP     # a line and the 2 stops it needs at least
    assert site_price(w, "transport", "bus_stops") == Price(C.BUS_LINE_COST, "bus line", name="bus_line", upkeep=line)
    assert site_price(w, "transport", "congested_roads") == Price(C.ROAD_UPGRADE_COST[2], "avenue tile", True,
                                                                  name="avenue_tile",
                                                                  upkeep=C.ROAD_UPKEEP[2] - C.ROAD_UPKEEP[1])
    fresh = _env().world                               # avenues unlock at 1,000 residents; bus lines need no unlock
    assert site_price(fresh, "transport", "congested_roads") is None
    assert site_price(fresh, "transport", "bus_stops") == Price(C.BUS_LINE_COST, "bus line", name="bus_line",
                                                                 upkeep=line)


def test_ready_sites_show_the_price_and_what_is_short():
    from civitas.fleet.tools import Price, SiteList, ready_lines
    lists = {"clinic": SiteList([{"x": 1, "y": 2, "need": 5}], None, None),
             "hospital": SiteList([{"x": 3, "y": 4}], "locked (unlocks at 5,000 residents)", "locked"),
             "landfill": SiteList([{"x": 5, "y": 6, "need": 3}], None, None),
             "cemetery": SiteList([{"x": 5, "y": 6, "need": 3}], None, None)}
    prices = {"clinic": Price(2000, "clinic"), "hospital": Price(8000, "hospital"), "landfill": Price(2000, "landfill"),
              "cemetery": Price(1500, "cemetery")}
    assert ready_lines("health", lists, prices=prices, funds=1800) == [
        "- clinic: 1 site, best (1,2), need 5; $2,000 - you have $1,800",
        "- hospital: locked (unlocks at 5,000 residents)",
        "- landfill: 1 site, best (5,6), need 3; $2,000 - you have $1,800",
        "- cemetery: 1 site, same list as landfill; $1,500"]
    clinic = lambda funds: ready_lines("health", lists, prices=prices, funds=funds)[0]
    assert clinic(2000) == clinic(2000 - 1e-10) == "- clinic: 1 site, best (1,2), need 5; $2,000"   # _spend's tolerance
    assert clinic(1999.6) == "- clinic: 1 site, best (1,2), need 5; $2,000 - you have $1,999"       # rounded down
    assert ready_lines("health", lists)[0] == "- clinic: 1 site, best (1,2), need 5"
    levee = {"levee": SiteList([{"x": 25, "y": 15, "x1": 25, "y1": 15, "x2": 25, "y2": 16, "tiles": 2}], None, None)}
    assert ready_lines("fire", levee, prices={"levee": Price(300, "levee tile", per_tile=True)}, funds=299) == [
        "- levee: 1 site, best (25,15)-(25,16), 2 tiles; $300/tile - you have $299"]


def _block(text: str, head: str) -> list[str]:
    return text[text.index(head):].split("\n\n")[0].splitlines()[1:]


def test_department_briefings_price_ready_sites_and_say_when_none_is_affordable(grown_env):
    env = copy.deepcopy(grown_env)
    w = env.world
    w.dept_funds.update(police=1000, fire=1499, health=0, transport=0, planning=0)
    police = briefing(env, "police", mark=False)
    station, jail = _ready_sites(police)[:2]
    assert station.endswith("; $2,500 - you have $1,000") and jail.endswith("; $4,000 - you have $1,000"), police
    assert _block(police, "## Your budget")[1] == (
        "- You can't pay for any ready site (cheapest: station $2,500). Don't call build tools this turn: ask the "
        "mayor for the amount with a reason (send_message) unless you already did, then end_turn.")
    assert "(cheapest: station $2,500)" in briefing(env, "fire", mark=False)       # never the $1,500 shelter
    assert "(cheapest: clinic or landfill $2,000)" in briefing(env, "health", mark=False)     # a tie names both
    planning = briefing(env, "planning", mark=False)  # only its road priced, and never on the budget line
    priced = [ln for ln in _ready_sites(planning) if "$" in ln]
    assert "can't pay" not in planning and len(priced) == 1 and priced[0].startswith("- road_extension: "), priced
    assert re.search(r"; \$[\d,]+ road \+ \$[\d,]+ zoning - you have \$0$", priced[0]), priced
    transport = briefing(env, "transport", mark=False)
    assert _ready_sites(transport)[0].endswith("; $1,500 - you have $0") and "(cheapest: bus_stops $1,500)" in transport
    w.dept_funds["police"] = 3000
    police = briefing(env, "police", mark=False)
    station, jail = _ready_sites(police)[:2]
    assert re.search(r", need [\d,]+; \$2,500$", station) and jail.endswith("; $4,000 - you have $3,000"), police
    assert len(_block(police, "## Your budget")) == 1
    for funds in (2500, 2500 - 1e-10):                 # exactly the cost, or short by float dust (_spend lets it pass)
        w.dept_funds["police"] = funds
        police = briefing(env, "police", mark=False)
        assert len(_block(police, "## Your budget")) == 1 and _ready_sites(police)[0].endswith("; $2,500"), police
    w.dept_funds["police"] = 2499.6                    # short: funds shown rounded down, never "$2,500"
    police = briefing(env, "police", mark=False)
    assert _block(police, "## Your budget")[0].startswith("- $2,499 in department funds"), police
    assert "(cheapest: station $2,500)" in police and _ready_sites(police)[0].endswith("; $2,500 - you have $2,499")
    w.dept_funds["police"] = 2500 - 1e-10
    s = ToolBox(env, "police").call("find_sites", {"purpose": "station"})["sites"][0]
    assert apply_action(w, "police", {"type": "build_police_station", "x": s["x"], "y": s["y"]}).ok


def test_a_library_never_counts_as_the_cheapest_ready_site(grown_env):
    """A library ($1,500) only multiplies school graduation: it keeps its price on its Ready line, and the mayor's
    needs line and the budget line name the $2,500 school; the budget line still only shows when no ready site,
    the library included, is affordable (else it would be false)."""
    from civitas.fleet.tools import Price, cheapest
    school, library = Price(2500, "elementary school"), Price(1500, "library")
    assert cheapest({"library": library, "school": school}) == (("school",), school)
    assert cheapest({"library": library}) is None and cheapest({"library": library}, supplementary=True) == (
        ("library",), library)
    assert cheapest({"library": library, "school": school}, supplementary=True) == (("library",), library)
    env = copy.deepcopy(grown_env)
    _no_spending(env)
    w = env.world
    ready = lambda: {ln.split(":")[0][2:]: ln for ln in _ready_sites(briefing(env, "education", mark=False))}
    budget = lambda: _block(briefing(env, "education", mark=False), "## Your budget")
    needs = lambda: next(ln for ln in briefing(env, "mayor", mark=False).splitlines() if ln.startswith("- education:"))
    w.dept_funds["education"] = 2500                   # the school is affordable: no advice
    assert len(budget()) == 1 and needs() == "- education: funds $2,500, spent $0; funds cover its next build"
    assert ready()["library"].endswith("same list as school; $1,500") and ready()["school"].endswith("; $2,500")
    w.dept_funds["education"] = 2000                   # a library is affordable (no budget line), the school isn't
    assert len(budget()) == 1 and needs().startswith("- education: funds $2,000, spent $0; needs $2,500 for its "
                                                     "first elementary school (")
    assert ready()["library"].endswith("; $1,500") and ready()["school"].endswith("; $2,500 - you have $2,000")
    w.dept_funds["education"] = 1000                   # short of both: the library's line says so too
    assert budget()[1].startswith("- You can't pay for any ready site (cheapest: school $2,500).")
    assert "$2,500 for its first elementary school" in needs()
    assert ready()["library"].endswith("; $1,500 - you have $1,000")


def test_shelter_and_cemetery_are_never_named_as_the_first_build(grown_env):
    """Smoke v9: the mayor read "fire: funds $0; needs $1,500 for a shelter", gave fire $1,500, and fire never reached
    the $2,500 station (0% coverage). Shelter and cemetery are supplementary like the library: while a station, clinic
    or landfill has sites, the mayor's line names that, and an affordable one still keeps the budget line away."""
    from civitas.fleet.tools import SUPPLEMENTARY, Price, cheapest
    assert SUPPLEMENTARY == {"library", "shelter", "cemetery", "train_station", "harbour", "metro_station"}
    station, shelter = Price(2500, "fire station"), Price(1500, "shelter")
    clinic, landfill, cemetery = Price(2000, "clinic"), Price(2000, "landfill"), Price(1500, "cemetery")
    assert cheapest({"station": station, "shelter": shelter}) == (("station",), station)
    assert cheapest({"clinic": clinic, "landfill": landfill, "cemetery": cemetery}) == (("clinic", "landfill"), clinic)
    assert cheapest({"shelter": shelter}) is None and cheapest({"cemetery": cemetery}) is None
    env = copy.deepcopy(grown_env)
    _no_spending(env)
    w = env.world
    n = {p: len(ToolBox(env, d).call("find_sites", {"purpose": p})["sites"])
         for d, p in (("fire", "station"), ("health", "clinic"), ("health", "landfill"))}
    line = lambda d: next(ln for ln in briefing(env, "mayor", mark=False).splitlines() if ln.startswith(f"- {d}:"))
    for funds in (0, 1000, 1499, 1500, 1999):
        w.dept_funds.update(fire=funds, health=funds)
        assert line("fire") == (f"- fire: funds ${funds:,}, spent $0; needs $2,500 for another fire station "
                                f"({_cover(w, 'fire_coverage')}, $55/month)")
        assert line("health") == (f"- health: funds ${funds:,}, spent $0; needs $2,000 for another clinic "
                                  f"({_cover(w, 'health_coverage')}, $40/month) or another landfill "
                                  f"({_cover(w, 'garbage_coverage')}, $30/month)")                                    # a tie
    w.dept_funds.update(fire=1500, health=1500)        # v9: a shelter / cemetery affordable, the station / landfill not
    for dept, cheap, dear in (("fire", "shelter", "station"), ("health", "cemetery", "landfill")):
        text = briefing(env, dept, mark=False)
        ready = {ln.split(":")[0][2:]: ln for ln in _ready_sites(text)}
        assert len(_block(text, "## Your budget")) == 1 and "can't pay" not in text, dept
        assert ready[cheap].endswith("; $1,500") and "- you have $1,500" in ready[dear], ready
    w.dept_funds.update(fire=2000, health=2000)        # the landfill and clinic affordable: health needs nothing
    assert line("health") == "- health: funds $2,000, spent $0; funds cover its builds" and \
        "needs $2,500 for another fire station" in \
        line("fire")


def test_the_budget_line_falls_back_to_a_supplementary_price(grown_env, monkeypatch):
    """With no primary site ready (fire: no station site, a shelter site), the mayor's line names nothing, since no
    first build is in reach, while fire's budget line names the shelter: nothing it has ready is affordable."""
    import civitas.fleet.briefing as B
    import civitas.fleet.tools as T
    from civitas.fleet.tools import SiteList
    real = B.site_lists

    def no_station(w, role, obs):
        lists = real(w, role, obs)
        return {**lists, "station": SiteList([], "nothing needs it right now", "")} if role == "fire" else lists
    monkeypatch.setattr(B, "site_lists", no_station)   # station sites never run out in the grown city
    monkeypatch.setattr(T, "site_lists", no_station)   # the mayor's line (first_build)
    env = copy.deepcopy(grown_env)
    _no_spending(env)
    env.world.dept_funds["fire"] = 0
    fire = briefing(env, "fire", mark=False)
    assert _ready_sites(fire)[0] == "- station: none - nothing needs it right now"
    assert _block(fire, "## Your budget")[1].startswith("- You can't pay for any ready site (cheapest: shelter "
                                                        "$1,500).")
    mayor = briefing(env, "mayor", mark=False).splitlines()     # the needs line skips it; the state names it
    assert next(ln for ln in mayor if ln.startswith("- fire:")) == "- fire: funds $0, spent $0; only a shelter ready " \
                                                                    "($1,500)"


def test_levees_are_priced_per_tile_in_a_flood(grown_env):
    from civitas.fleet.tools import site_price
    env = copy.deepcopy(grown_env)
    _no_spending(env)
    w = env.world
    w.schedule.append({"kind": "flood", "month": w.month + 2, "duration": 1, "y1": 15, "y2": 32})
    tile = C.CATALOG["levee"]["cost"]
    assert _building_cost(w, "fire", "levee") is None and site_price(w, "fire", "levee").per_tile  # not a building
    fire_line = lambda: next(ln for ln in briefing(env, "mayor", mark=False).splitlines() if ln.startswith("- fire:"))
    sites = ToolBox(env, "fire").call("find_sites", {"purpose": "levee"})["sites"]
    s, n = sites[0], sites[0]["tiles"]                 # the mayor's line prices the best run whole
    assert n > 1
    up = n * C.CATALOG["levee"]["upkeep"]
    needs = f"needs ${n * tile:,} for {n} levee tiles ({len(sites)} sites, ${up:g}/month)"
    w.dept_funds["fire"] = tile                        # one levee tile, less than a shelter
    fire = briefing(env, "fire", mark=False)
    levee, = [ln for ln in _ready_sites(fire) if ln.startswith("- levee:")]
    assert levee.endswith(f"; ${tile:,}/tile") and "can't pay" not in fire, fire     # one tile is buildable
    spent = ", spent $0"
    assert fire_line() == f"- fire: funds ${tile:,}{spent}; {needs}"       # cheaper than the $2,500 station
    w.dept_funds["fire"] = n * tile                    # the whole run
    assert fire_line() == f"- fire: funds ${n * tile:,}{spent}; funds cover its next build"
    w.dept_funds["fire"] = tile - 1
    fire = briefing(env, "fire", mark=False)
    levee, = [ln for ln in _ready_sites(fire) if ln.startswith("- levee:")]
    assert levee.endswith(f"; ${tile:,}/tile - you have ${tile - 1:,}") and f"(cheapest: levee ${tile:,}/tile)" in fire
    assert fire_line() == f"- fire: funds ${tile - 1:,}{spent}; {needs}"
    res = apply_action(w, "fire", {"type": "build_levee", "x": s["x"], "y": s["y"]})       # a single tile
    assert not res.ok and res.message.startswith(f"costs ${tile:,} but the fire budget only has ${tile - 1:,}")


def test_congested_roads_are_priced_per_avenue_tile(open_env, monkeypatch):
    """upgrade_road charges C.ROAD_UPGRADE_COST[2] per street tile on land for an avenue: the listed stretch costs its
    street tiles times the Ready line's per-tile price."""
    import civitas.fleet.tools as T
    from civitas.fleet.tools import site_price
    env = copy.deepcopy(open_env)
    _no_spending(env)
    w = env.world
    w.congestion[40:44, 3] = 1.5                       # the new street down x=3 (all land): one congested stretch
    s = ToolBox(env, "transport").call("find_sites", {"purpose": "congested_roads"})["sites"][0]
    assert (s["x1"], s["y1"], s["x2"], s["y2"]) == (3, 40, 3, 43), s
    tile = site_price(w, "transport", "congested_roads")
    cost = 4 * tile.cost
    line, = [ln for ln in _ready_sites(briefing(env, "transport", mark=False)) if ln.startswith("- congested_roads:")]
    assert line.endswith(f"; ${tile.cost:,}/tile - you have $0") or line.endswith(f"; ${tile.cost:,}/tile"), line
    action = {"type": "upgrade_road", "to": "avenue", "x1": 3, "y1": 40, "x2": 3, "y2": 43}
    w.dept_funds["transport"] = cost - 1
    res = apply_action(w, "transport", action)
    assert not res.ok and res.message.startswith(f"costs ${cost:,} but the transport budget only has ${cost - 1:,}")
    fleet_line = lambda: next(ln for ln in briefing(env, "mayor", mark=False).splitlines()
                              if ln.startswith("- transport:"))
    # without a bus line its first one comes first, dearer or not (review: seed 0, months 42-69)
    assert fleet_line().startswith(f"- transport: funds ${cost - 1:,}, spent $0; needs $1,500 for its first bus line (")
    real = T.buildings_of
    monkeypatch.setattr(T, "buildings_of", lambda w, name: 1 if name == "bus_line" else real(w, name))
    assert fleet_line() == (f"- transport: funds ${cost - 1:,}, spent $0; needs ${cost:,} for 4 avenue "
                            f"tiles (1 site, ${4 * (C.ROAD_UPKEEP[2] - C.ROAD_UPKEEP[1]):g}/month)")
    w.dept_funds["transport"] = cost                   # the whole stretch: cheaper than another $1,500 bus line
    assert fleet_line() == f"- transport: funds ${cost:,}, spent $0; funds cover its next build"
    res = apply_action(w, "transport", action)
    assert res.ok and res.cost == cost and w.dept_funds["transport"] == 0, res.message


def test_a_congested_bridge_is_priced_as_upgrade_road_charges():
    """The mayor's line prices a congested stretch whole: its tiles below avenue class, a bridge tile 3 times (as
    upgrade_road charges it); the first listed stretch with something to charge (an avenue stretch has nothing)."""
    from civitas.fleet.tools import best_stretch, site_price, site_tiles
    env = _env()
    w = env.world
    w.milestones_on = False                            # avenues unlocked
    w.dept_funds["planning"] = 5000
    assert apply_action(w, "planning", {"type": "build_road", "x1": 19, "y1": 24, "x2": 38, "y2": 24}).ok
    w.congestion[24, 24:31] = 1.5                      # x 24-30: 4 land tiles and the 3-tile bridge
    s = ToolBox(env, "transport").call("find_sites", {"purpose": "congested_roads"})["sites"][0]
    assert (s["x1"], s["x2"], s["y1"]) == (24, 30, 24), s
    per = site_price(w, "transport", "congested_roads").cost
    assert site_tiles(w, "congested_roads", s) == (7, 4 + 3 * 3)
    cost = (4 + 3 * 3) * per
    mayor = briefing(env, "mayor", mark=False).splitlines()
    assert next(ln for ln in mayor if ln.startswith("- transport:")) == (
        f"- transport: funds $0; needs ${cost:,} for 7 avenue tiles (1 site, "
        f"${(4 + 2 * 3) * (C.ROAD_UPKEEP[2] - C.ROAD_UPKEEP[1]):g}/month)")     # no bus stops yet; bridges x2
    assert f"(cheapest: congested_roads ${per:,}/tile)" in briefing(env, "transport", mark=False)
    w.dept_funds["transport"] = cost - 1
    res = apply_action(w, "transport", {"type": "upgrade_road", "to": "avenue", "x1": 24, "y1": 24, "x2": 30, "y2": 24})
    assert not res.ok and res.message.startswith(f"costs ${cost:,} but"), res.message
    w.congestion[24, 5:9] = 1.2                       # a second stretch, x 5-8 on land
    w.road_class[24, 24:31] = C.ROAD_CLASSES["avenue"]    # the first one avenues already: nothing to charge there
    sites = ToolBox(env, "transport").call("find_sites", {"purpose": "congested_roads"})["sites"]
    assert [(t["x1"], t["x2"]) for t in sites] == [(5, 8), (24, 30)], sites           # ranked last, not dropped
    assert site_tiles(w, "congested_roads", sites[1]) == (0, 0) and best_stretch(w, "congested_roads", sites) == (4, 4)
    w.dept_funds["transport"] = 0
    ready, = [ln for ln in _ready_sites(briefing(env, "transport", mark=False)) if ln.startswith("- congested_roads:")]
    assert ready.startswith("- congested_roads: 2 sites, best (5,24)-(8,24)"), ready
    mayor = briefing(env, "mayor", mark=False).splitlines()
    assert next(ln for ln in mayor if ln.startswith("- transport:")) == (
        f"- transport: funds $0; needs ${4 * per:,} for 4 avenue tiles (2 sites, "
        f"${4 * (C.ROAD_UPKEEP[2] - C.ROAD_UPKEEP[1]):g}/month)")
    w.road_class[24, 5:9] = C.ROAD_CLASSES["avenue"]      # none left to charge: one tile, as when a site doesn't say
    assert best_stretch(w, "congested_roads", sites) == (1, 1)


def test_no_bus_stops_before_1500_residents():
    """observation._suggest_stops has no population floor: from month 1 bus stops were listed, and with bus lines
    priced transport at $0 read "can't pay (cheapest: bus_stops $1,500)" and the mayor "transport: needs $1,500 for
    a bus line", against both playbooks ("not before ~1,500 residents"). Below SITE_MIN_POP the list is empty."""
    from civitas.fleet.tools import SITE_MIN_POP, SITE_NOTES
    from civitas.core.observation import build_observation
    env = CivitasEnv(GameConfig(seed=0, months=24))
    agents = {r: SCRIPTED[r](r) for r in env.roles}
    for _ in range(3):
        for r in env.roles:
            env.apply(r, agents[r].act(env.observe(r)))
        env.end_month()
    w = env.world
    assert w.metrics()["population"] < SITE_MIN_POP["bus_stops"] == 1500
    assert build_observation(w, "transport", council_msgs=[], memory={}, scores={}, months_total=24,
                             include_map=False, site_hints=True)["transport"]["suggested_stops"]   # the hint has some
    w.dept_funds["transport"] = 0
    text = briefing(env, "transport", mark=False)
    assert _ready_sites(text)[0] == f"- bus_stops: none - {SITE_NOTES['bus_stops'][0]}", _ready_sites(text)
    assert len(_block(text, "## Your budget")) == 1 and "can't pay" not in text
    res = ToolBox(env, "transport").call("find_sites", {"purpose": "bus_stops"})
    assert res["sites"] == [] and res["note"] == SITE_NOTES["bus_stops"][1]
    _no_spending(env)
    mayor = briefing(env, "mayor", mark=False).splitlines()
    transport = next(ln for ln in mayor if ln.startswith("- transport:"))
    assert transport == "- transport: funds $0, spent $0; nothing ready to buy"


def test_transport_is_priced_by_its_bus_lines_past_5000(open_env):
    """From 5,000 residents a train station ($8k) and a harbour ($10k) are priced; with only them priced, transport
    holding $3,000 read "You can't pay for any ready site (cheapest: harbour $10k)" and the mayor "needs $10k for a
    harbour" while bus stops were ready. Bus lines are priced now, and stations, harbours and metro stations are
    supplementary (never the first build: bus lines come first)."""
    from civitas.fleet.fmt import money
    env = copy.deepcopy(open_env)                      # every milestone reached; bus stops, stations, harbour ready
    _no_spending(env)
    w = env.world
    ready = {ln.split(":")[0][2:]: ln for ln in _ready_sites(briefing(env, "transport", mark=False))}
    assert all(f"; {money(c)}" in ready[p] for p, c in (
        ("bus_stops", C.BUS_LINE_COST), ("harbour", C.HARBOUR_SPEC["cost"]), ("train_station", C.TRAIN_STATION["cost"]),
        ("metro_station", C.METRO_STATION["cost"]))), ready
    fleet_line = lambda: next(ln for ln in briefing(env, "mayor", mark=False).splitlines()
                              if ln.startswith("- transport:"))
    w.dept_funds["transport"] = 3000
    text = briefing(env, "transport", mark=False)
    budget = _block(text, "## Your budget")          # the funds line, then the first bus line's priority line
    assert len(budget) == 2 and budget[1].startswith("- First bus line ($1,500) before any avenue") and \
        "can't pay" not in text, text
    assert fleet_line() == "- transport: funds $3,000, spent $0; idle $1,500 (keeps $1,500)"     # one bus line, not
    # a harbour
    w.dept_funds["transport"] = 0
    n = min(6, len(ToolBox(env, "transport").call("find_sites", {"purpose": "bus_stops"})["stops"]))
    assert "(cheapest: bus_stops $1,500)" in briefing(env, "transport", mark=False)
    assert fleet_line() == f"- transport: funds $0, spent $0; needs $1,500 for its first bus line ({n} sites, $76/month)"


def test_bank_top_up_ranks_as_the_observation_does():
    """With every free bank lot beside the highway network, _bank_top_up must list what the observation's riverbank
    lists do (its ranking is a copy of observation._candidate_sites'), crowding and plants included."""
    from civitas.fleet.tools import _bank_top_up, free_bank, role_obs
    env = _env()
    w = env.world
    w.dept_funds.update(planning=50000, utilities=50000)
    for line in ({"x1": 19, "y1": 24, "x2": 38, "y2": 24}, {"x1": 22, "y1": 25, "x2": 22, "y2": 46},
                 {"x1": 23, "y1": 46, "x2": 31, "y2": 46}):
        assert apply_action(w, "planning", {"type": "build_road", **line}).ok

    def same():
        hints = role_obs(env, "utilities")["candidate_sites"]
        for line, key in (("pump", "riverbank_next_to_road"), ("outlet", "riverbank_for_sewage")):
            want = [{k: v for k, v in t.items() if k != "network"} for t in hints[key]]
            n = min(6, len(want))
            assert n >= 4 and _bank_top_up(w, [], line)[:n] == want[:n], (line, want)
        return hints
    assert free_bank(w).sum() == 6
    hints = same()
    pump, outlet = hints["riverbank_next_to_road"][0], hints["riverbank_for_sewage"][0]
    assert apply_action(w, "utilities", {"type": "build_water", "kind": "pump", "x": pump["x"], "y": pump["y"]}).ok
    assert apply_action(w, "utilities", {"type": "build_sewage", "kind": "outlet", "x": outlet["x"],
                                         "y": outlet["y"]}).ok
    same()                                             # ranked around the new plants, a crowded lot last


def test_mayor_briefing_says_what_short_departments_need(grown_env):
    env = copy.deepcopy(grown_env)
    _no_spending(env)
    w = env.world
    w.dept_funds.update(police=1000, fire=5000, health=0, education=0, utilities=0, transport=0, planning=0)
    n = len(ToolBox(env, "police").call("find_sites", {"purpose": "station"})["sites"])
    lines = lambda: {ln.split(":")[0][2:]: ln.replace(", spent $0", "")                 # as before spending showed
                     for ln in briefing(env, "mayor", mark=False).splitlines()
                     if ln.startswith("- ") and ": funds " in ln}
    got = lines()                                      # the building, not the find_sites purpose
    assert got["police"] == (f"- police: funds $1,000; needs $2,500 for its first police station ({n} sites, "
                             "$55/month)")
    assert re.fullmatch(r"- health: funds \$0; needs \$2,000 for another clinic \(coverage \d+%, \$40/month\) or "
                        r"another landfill \(coverage \d+%, \$30/month\)", got["health"])   # $2,000 each: a tie names both, not the cemetery
    assert got["education"].startswith("- education: funds $0; needs $2,500 for its first elementary school (")
    assert got["utilities"].startswith("- utilities: funds $0; needs $6,200 for another coal plant and water tower (")
    assert got["fire"] == "- fire: funds $5,000; idle $2,500 (keeps $2,500)"          # the station's $2,500
    assert got["transport"].startswith("- transport: funds $0; needs $1,500 for its first bus line (")
    assert re.fullmatch(r"- planning: funds \$0; needs \$[\d,]+ for a road into open land and zoning it "
                        r"\(opens \d+ tiles\)",
                        got["planning"]), got["planning"]              # no building: neither its first nor another
    ToolBox(env, "mayor").call("update_plan", {"budgets": {"police": 2500}, "priorities": {"police": "cover downtown"}})
    assert lines()["police"] == (f"- police: funds $1,000 < target $2,500; needs $2,500 for its first police station "
                                 f"({n} sites, $55/month)")                  # the priority: get_plan
    w.dept_funds["police"] = 2500
    assert lines()["police"] == "- police: funds $2,500, target $2,500; funds cover its next build"   # not a jail


def _dept_lines(env) -> dict[str, str]:
    return {ln.split(":")[0][2:]: ln for ln in briefing(env, "mayor", mark=False).splitlines()
            if ln.startswith("- ") and ": funds " in ln}


def _cover(w, key: str) -> str:
    from civitas.fleet.fmt import pct
    return f"coverage {pct(100 * w.metrics()[key])}"


def test_the_needs_line_says_first_or_another(grown_env):
    """Smoke v16: "police: funds $0 < target $2,500, spent $2,500; needs $2,500 for a police station (5 ready sites)"
    read the same with a station built at month 3; the mayor funded second police and fire stations and a second
    clinic, and education never got its school (33.2). The line says which, with the coverage for another one."""
    env = copy.deepcopy(grown_env)                     # month 24: a fire station, clinic and landfill; no police
    _no_spending(env)                                  # station, school or bus line
    w = env.world
    w.dept_funds.update({d: 0 for d in DEPTS})
    n = {p: min(6, len(res.get("sites") or res["stops"]))          # bus_stops: its line's stops, at most 6 counted
         for d, p in (("police", "station"), ("education", "school"), ("transport", "bus_stops"))
         for res in [ToolBox(env, d).call("find_sites", {"purpose": p})]}
    m = w.metrics()
    got = _dept_lines(env)
    assert got["police"] == (f"- police: funds $0, spent $0; needs $2,500 for its first police station "
                             f"({n['station']} sites, $55/month)")
    assert got["fire"] == (f"- fire: funds $0, spent $0; needs $2,500 for another fire station "
                           f"({_cover(w, 'fire_coverage')}, $55/month)")
    assert got["health"] == (f"- health: funds $0, spent $0; needs $2,000 for another clinic "
                             f"({_cover(w, 'health_coverage')}, $40/month) or another landfill "
                             f"({_cover(w, 'garbage_coverage')}, $30/month)")
    assert got["education"].endswith(f"; needs $2,500 for its first elementary school ({n['school']} sites, $45/month)")
    assert got["transport"].endswith(f"; needs $1,500 for its first bus line ({n['bus_stops']} sites, $76/month)")
    assert m["water_demand"] == m["water_supply"] == 313            # the pump counts: another one
    assert m["power_demand"] == m["power_supply"]                   # imports fill both gaps: at capacity
    assert got["utilities"].endswith("; needs $6,200 for another coal plant and water tower ($110/month)")
    s = ToolBox(env, "police").call("find_sites", {"purpose": "station"})["sites"][0]
    w.dept_funds["police"] = 2500
    assert apply_action(w, "police", {"type": "build_police_station", "x": s["x"], "y": s["y"]}).ok
    assert _dept_lines(env)["police"] == (f"- police: funds $0, spent $0; needs $2,500 for another police station "
                                          f"({_cover(w, 'police_coverage')}, $55/month)")
    (x, y), = w.facilities(C.LANDFILL)                 # at a tie a first one is named alone: the landfill first
    w.remove_building(x, y)
    w.refresh()
    health = _dept_lines(env)["health"]
    assert re.fullmatch(r"- health: funds \$0, spent \$0; needs \$2,000 for its first landfill "
                        r"\(\d sites?, \$30/month\)", health)


def test_buildings_are_counted_on_their_line(open_env):
    """What the needs line calls a first or another: a line's buildings at that tier or above (a hospital is another
    clinic, a clinic no hospital); utilities by the supply (the pump makes a water tower another one); bus lines by
    count; tiles and off-tier stations not at all."""
    from civitas.fleet.tools import buildings_of
    env = copy.deepcopy(open_env)                      # the grown city, every milestone reached
    w = env.world
    assert [buildings_of(w, n) for n in ("tower", "pump", "wind", "coal", "outlet", "treatment")] == [1, 1, 6, 6, 2, 2]
    assert [buildings_of(w, n) for n in ("clinic", "hospital", "landfill", "police_station", "elementary_school")] == [
        1, 0, 1, 0, 0]
    assert buildings_of(w, "bus_line") == 0 and buildings_of(w, "avenue_tile") is None
    assert buildings_of(w, "levee_tile") is None and buildings_of(w, "train_station") is None
    w.dept_funds["health"] = 50000
    sites = ToolBox(env, "health").call("find_sites", {"purpose": "hospital"})["sites"]
    assert apply_action(w, "health", _build_action("health", "hospital", sites)).ok     # the clinic line's tier 2
    assert (buildings_of(w, "clinic"), buildings_of(w, "hospital")) == (2, 1)


def test_first_buildings_come_before_second_ones(grown_env):
    """Smoke v16 seed 0, months 6-9: the treasury was short; the mayor funded second police and fire stations and a
    second clinic first, and education's allocations were refused ("treasury only has $500"): education asked every
    quarter and never got its school (33.2). v17: education, last in turn order and in the block, again got none. The
    Departments block and the to-fund line put first buildings and supply first (Funding.order), then planning's road,
    then another building, turn order within a kind, and the cover clause fills them in that order."""
    from civitas.fleet.fmt import money
    from civitas.fleet.tools import reserve
    env = copy.deepcopy(grown_env)
    _no_spending(env)
    w = env.world
    w.dept_funds.update({d: 0 for d in DEPTS}, police=2500, transport=1500)
    s = ToolBox(env, "police").call("find_sites", {"purpose": "station"})["sites"][0]
    assert apply_action(w, "police", {"type": "build_police_station", "x": s["x"], "y": s["y"]}).ok
    w.treasury = reserve(w) + 9000
    text = briefing(env, "mayor", mark=False)          # police and fire want another station, education its first
    to_fund = lambda text: next(ln for ln in text.splitlines() if ln.startswith("- to fund: "))
    order = lambda text: [ln.split(":")[0][2:] for ln in _block(text, "## Departments") if ": funds " in ln]
    assert "; needs $2,500 for another police station (" in text and "; needs $2,500 for another fire station (" in text
    assert order(text) == ["utilities", "education", "planning", "police", "fire", "health", "transport"]
    # first/supply and growth may use the treasury above $3,000 ($9,000 above the reserve, which is above $3,000):
    # the road fits after them, the other stations don't (utilities: a coal plant and a tower, v20 fixes)
    dip = money(w.treasury - 3000)
    assert f"; first/supply and growth may use {dip} (treasury above $3,000)" in text
    assert re.fullmatch(r"- to fund: first/supply: utilities \$6,200, education \$2,500 \(\$8,700, "
                        r"upkeep \+\$155/month\) \| growth: planning \$[\d,]+ \| another: police \$2,500, "
                        r"fire \$2,500, health \$2,000 "
                        rf"\(\$7,000, upkeep \+\$150/month\): {re.escape(dip)} covers utilities, education, planning; "
                        r"the other \$7,000 would go under the reserve", to_fund(text)), to_fund(text)
    ToolBox(env, "mayor").call("update_plan", {"budgets": {"police": 5000}})
    text = briefing(env, "mayor", mark=False)          # below its target: marked, still another station
    assert "- police: funds $0 < target $5,000" in text and order(text).index("education") < order(text).index("police")
    w.dept_funds["transport"] = 0                      # a first bus line too, in turn order
    assert to_fund(briefing(env, "mayor", mark=False)).startswith(
        "- to fund: first/supply: utilities $6,200, transport $1,500, education $2,500 ($10.2k, upkeep +$231/month) | "
        "growth: ")                                    # the bus line's upkeep: its line and 2 stops
    (x, y), = w.facilities(C.LANDFILL)                 # a third: health's first landfill, the clinic left out
    w.remove_building(x, y)
    w.refresh()
    text = briefing(env, "mayor", mark=False)
    assert to_fund(text).startswith("- to fund: first/supply: utilities $6,200, transport $1,500, health $2,000, "
                                    "education $2,500 ($12.2k, upkeep +$261/month) | growth: planning ")
    assert "| another: police $2,500, fire $2,500 ($5,000, upkeep +$110/month): " in to_fund(text)
    w.dept_funds.update(transport=1500, health=2000, education=2500)     # every first building funded
    text = briefing(env, "mayor", mark=False)
    assert to_fund(text).startswith("- to fund: first/supply: utilities $6,200 (upkeep +$110/month) | growth: "
                                    "planning ")
    assert order(text)[-3:] == ["transport", "health", "education"]      # funds cover their builds: last
    env = _env()                                       # nothing built yet: every needs line is for a first one
    text = briefing(env, "mayor", mark=False)
    assert "its first coal plant and water tower" in text and to_fund(text).startswith(
        "- to fund: first/supply: utilities $6,200, health $2,000 ($8,200, upkeep +$140/month) | growth: planning ")


def test_utilities_needs_line_names_only_a_short_supply(env12):
    """Smoke v14 and v15 seed 1: utilities' needs line named a water tower with water never short, and towers were
    built until water was 4.7-4.9x demand. first_build's prices count a utilities building only while its supply is
    short (_wanted), a riverbank 2x2 one (water works) by its own supply."""
    from civitas.fleet.fmt import pct
    from civitas.fleet.tools import DeptMoney, _wanted
    env = copy.deepcopy(env12)                         # month 12: water 250 of 125, sewage 250 of 125, power 220 of 203
    _no_spending(env)
    w = env.world
    w.dept_funds.update({d: 0 for d in DEPTS})
    m = w.metrics()
    assert m["water_demand"] < 0.9 * m["water_supply"] and m["power_demand"] >= 0.9 * m["power_supply"]
    used = pct(100 * m["power_demand"] / m["power_supply"])
    assert _dept_lines(env)["utilities"] == (f"- utilities: funds $0, spent $0; needs $4,000 for another coal plant "
                                             f"(power {used} used, $80/month)")           # not the $2,200 tower
    # (v20 fixes: a turbine's 54 at its site's wind falls short of the 84 power wants: half again its 203, less 220)
    w.milestones_on = False                            # water works unlocked: only while water is short
    assert not _wanted(w, "riverbank_2x2")
    w.dept_funds["utilities"] = 50000
    s = ToolBox(env, "utilities").call("find_sites", {"purpose": "coal_or_gas"})["sites"][0]
    assert apply_action(w, "utilities", {"type": "build_power_plant", "kind": "coal", "x": s["x"], "y": s["y"]}).ok
    m = w.metrics()
    assert m["power_demand"] < 0.9 * m["power_supply"]
    w.dept_funds["utilities"] = 0                      # spare water, sewage and power: nothing to name, and it says why
    assert _dept_lines(env)["utilities"] == "- utilities: funds $0, spent $0; supply has room"
    b = DeptMoney(env, "utilities").build
    assert (b.cost, b.named, b.room) == (None, (), True)


def test_mayor_names_the_cheapest_building():
    from civitas.fleet.fmt import a_or_an
    from civitas.fleet.tools import SITE_BUILDS, building_name
    names = {n: a_or_an(building_name(n)) for purposes in SITE_BUILDS.values() for ns in purposes.values() for n in ns}
    assert (names["tower"], names["outlet"], names["coal"], names["pump"]) == (
        "a water tower", "a sewage outlet", "a coal plant", "a water pump")                  # base tiers: struct names
    assert (names["water_works"], names["advanced_treatment"], names["hospital"]) == (
        "a water works", "an advanced treatment", "a hospital")                               # higher tiers: their own
    assert (names["elementary_school"], names["university"], names["harbour"], names["metro_station"]) == (
        "an elementary school", "a university", "a harbour", "a metro station")
    assert not any("_" in v for v in names.values()), names


def test_department_briefing_builds_one_observation(monkeypatch):
    """One observation per department briefing; planning's blocked line and the mayor's Departments block add one per
    department with a priced purpose, never two; so does funding_gaps alone (update_plan): each department's kind
    needs its needs line or blocked marker (mayor money, Unit A), and planning its road while short of the dearest
    one."""
    import civitas.fleet.tools as T
    from civitas.fleet.tools import funding_gaps
    env = _env()
    calls = []
    real = T.build_observation

    def counted(*a, **k):
        calls.append(k.get("site_hints"))
        return real(*a, **k)
    monkeypatch.setattr(T, "build_observation", counted)           # briefings build it through tools.role_obs
    n = sum(any(T.site_price(env.world, d, p) for p in SITE_PURPOSES[d]) for d in DEPTS)
    assert n == 6                                      # every department but planning
    for role in DEPTS:
        calls.clear()
        briefing(env, role, mark=False)
        assert calls == [True] * (1 + n * (role == "planning")), (role, calls)
    fleet = FleetState.of(env)
    for funds in (0, 50000):
        env.world.dept_funds.update({d: funds for d in DEPTS})
        calls.clear()                                  # the blocked marker needs every priced department's sites, and
        briefing(env, "mayor", mark=False)             # planning's needs line its road while short of the dearest one
        assert calls == [True] * (n + (funds < T.road_max(env.world))), funds
        calls.clear()                                  # month 0, no plan: nothing idle, no gap, but every kind
        funding_gaps(env, fleet)
        assert calls == [True] * (n + (funds < T.road_max(env.world))), funds
    _months_spent(env, {}, {}, {})
    calls.clear()                                      # rich after a turn: maybe idle, so each priced one looks once,
    funding_gaps(env, fleet)                           # and planning, whose road and zoning may top its $1,000 floor
    assert calls == [True] * (n + 1)
    env.world.dept_funds.update({d: 1999 for d in DEPTS})
    calls.clear()                                      # can't be idle ($999 over $1,000), no targets: still each once
    funding_gaps(env, fleet)
    assert calls == [True] * (n + (1999 < T.road_max(env.world)))


def test_planning_ready_sites_name_the_first_buildings_blocked_for_lack_of_land():
    """Smoke v13 seed 0, months 6-9: planning zoned every roadside tile despite its playbook's "leave room"; utilities,
    police, fire, health and education all read "none - no free roadside land: ask planning for a lot"."""
    land = lambda env: [ln for ln in _ready_sites(briefing(env, "planning", mark=False)) if ln.startswith("- blocked")]
    head, tail = "- blocked for lack of land: ", ": extend a road into open land, or free a lot, for them"
    env = _env()                                       # a fresh map: the road stops short of the river
    assert land(env) == [head + "utilities (sewage outlet: a road to the river)" + tail]   # $1,200, not the pump
    env.world.dept_funds["planning"] = 5000
    assert apply_action(env.world, "planning", {"type": "build_road", "x1": 19, "y1": 24, "x2": 38, "y2": 24}).ok
    assert land(env) == []                             # across the river, free land by the roads: nothing blocked
    env = _env()
    _bridge_and_take_the_bank(env)                     # the road reaches the river, its bank lots zoned
    assert land(env) == [head + "utilities (sewage outlet: a free riverbank lot)" + tail]
    env = CivitasEnv(GameConfig(seed=0, months=12))
    agents = {r: SCRIPTED[r](r) for r in env.roles}
    for _ in range(6):
        for r in env.roles:
            env.apply(r, agents[r].act(env.observe(r)))
        env.end_month()
    _fill_roadside(env.world)                          # every roadside tile taken; pump and outlet have room
    block = _ready_sites(briefing(env, "planning", mark=False))
    line = (head + "police (police station), fire (fire station), health (clinic, landfill: away from homes), "
            "education (elementary school)" + tail)    # the cheapest first buildings, all on a tie; water has room
    assert len(line) == 202
    assert block[-4] == line and block[-2].startswith("- high density: locked")          # before the high density
    assert block[-3] == ("- away from homes (health's landfill): leave a lot free by (40,19), the far end of "
                         "road_extension (25,19)-(40,19), with no homes within 4 tiles")      # a dead end's extension
    import civitas.fleet.briefing as B
    B.BLOCKED_LINE_CHARS, keep = 150, B.BLOCKED_LINE_CHARS
    try:                                               # too long: the departments alone
        assert land(env) == [head + "police, fire, health, education" + tail]
    finally:
        B.BLOCKED_LINE_CHARS = keep
    assert not any(ln.startswith("- blocked") for ln in _ready_sites(briefing(env, "police", mark=False)))


def test_planning_blocked_line_fits_in_full_past_1000_residents(grown_env):
    """Review probes, months 24, 60 and 120: naming every blocked first building, planning's line always fell back to
    the department names alone. The "away from homes" tag goes first (248 characters with it): the next line says it."""
    from civitas.fleet.briefing import BLOCKED_LINE_CHARS
    env = copy.deepcopy(grown_env)                     # 2,6xx residents, water at capacity
    _fill_roadside(env.world)
    block = _ready_sites(briefing(env, "planning", mark=False))
    i, = [i for i, ln in enumerate(block) if ln.startswith("- blocked")]
    assert block[i] == ("- blocked for lack of land: utilities (water pump: a free riverbank lot), police (police "
                        "station), fire (fire station), health (clinic, landfill), education (elementary school): "
                        "extend a road into open land, or free a lot, for them") and len(block[i]) == 231 <= \
        BLOCKED_LINE_CHARS
    assert block[i + 1].startswith("- away from homes (health's landfill): "), block[i + 1]


def _best(env, role: str, purpose: str) -> str:
    from civitas.fleet.tools import role_obs, site_lists
    s = site_lists(env.world, role, role_obs(env, role))[purpose].sites[0]
    return f"({s['x']},{s['y']})"


def test_planning_ready_sites_name_what_funded_departments_will_build(env12):
    """Smoke v14 seed 0, months 3-6: the mayor funded fire, health and education for their ready sites near homes, and
    planning, acting next, zoned those lots. Its Ready sites now name each funded department's cheapest ready first
    building and best site; the mayor's top-ups this month count."""
    env = copy.deepcopy(env12)                         # police $796, utilities $1,283, education $518: short
    w = env.world
    w.dept_funds.update(fire=2500, health=2000)
    leave = lambda: [ln for ln in _ready_sites(briefing(env, "planning", mark=False)) if ln.startswith("- leave")]
    fire, clinic, landfill = ("fire", "station", "fire station"), ("health", "clinic", "clinic"), ("health", "landfill",
                                                                                                  "landfill")
    items = _claimed(env, fire, clinic, landfill)      # the clinic's best lot is the fire station's: its next one
    assert _best(env, "fire", "station") == _best(env, "health", "clinic") and not items[1].endswith(
        _best(env, "health", "clinic"))
    block = _ready_sites(briefing(env, "planning", mark=False))
    assert block[-3] == "- leave free (funded): " + ", ".join(items) + ": zone around them"
    assert block[-2].startswith("- high density: locked"), block                         # nothing blocked here
    w.treasury = 10000
    assert apply_action(w, "mayor", {"type": "allocate_budget", "department": "police", "amount": 1704}).ok
    assert leave() == ["- leave free (funded): " + ", ".join(_claimed(env, ("police", "station", "police station"),
                                                                      fire, clinic, landfill)) + ": zone around them"]
    w.dept_funds.update({d: 0 for d in DEPTS})
    assert leave() == []
    env = copy.deepcopy(env12)
    w = env.world
    w.dept_funds.update(fire=2500, health=2000)
    _hem_homes(env)                                    # nothing ready near homes: the fire station is blocked
    assert leave() == [f"- leave free (funded): landfill {_best(env, 'health', 'landfill')}: zone around them"]
    assert "fire (fire station: near homes)" in briefing(env, "planning", mark=False)


def test_planning_zone_strips_spare_funded_sites(env12):
    """Planning zones what it is suggested: next to "leave free (funded): fire station (20,22), clinic (20,22)" its
    best zone_R and zone_C strip was (20,20)-(21,22). Strips covering a funded site now come after those that don't,
    marked, and a failed zone's suggestions skip the site, as they skip free riverbank lots while the bank is needed."""
    from civitas.fleet.tools import SiteList, Spot, ready_lines, spare_funded
    env = copy.deepcopy(env12)
    w = env.world
    zone = lambda: ToolBox(env, "planning").call("zone", {"zone": "R", "x1": 19, "y1": 21, "x2": 19, "y2": 23})
    road = "no empty land tiles in that area - all 3 tiles are road"
    assert zone()["message"] == road + "; zone beside it instead: x1=20,y1=21,x2=20,y2=22"    # nothing funded
    w.dept_funds.update(fire=2500, health=2000)
    assert _best(env, "fire", "station") == _best(env, "health", "clinic") == "(20,22)"
    ready = _ready_sites(briefing(env, "planning", mark=False))
    assert next(ln for ln in ready if ln.startswith("- zone_R:")) == "- zone_R: 2 sites, best (0,22)-(3,23), 8 tiles"
    assert next(ln for ln in ready if ln.startswith("- zone_C:")) == (       # the other strip has the landfill's
        "- zone_C: 2 sites, best (20,20)-(21,22), 6 tiles (leave (20,22) for the fire station)")    # the clinic's
    sites = ToolBox(env, "planning").call("find_sites", {"purpose": "zone_R"})["sites"]            # is (22,20)
    assert [s.get("leave") for s in sites] == [None, "(20,22) for the fire station"], sites
    assert zone()["message"] == road                   # (20,21)-(20,22) takes the funded lot
    only = {"zone_R": SiteList([{"x": 20, "y": 20, "x1": 20, "y1": 20, "x2": 21, "y2": 22, "tiles": 6}], None, None)}
    fire, clinic = Spot("fire", "station", "fire station", (20, 22)), Spot("health", "clinic", "clinic", (21, 20))
    assert ready_lines("planning", spare_funded(only, [fire])) == [
        "- zone_R: 1 site, best (20,20)-(21,22), 6 tiles (leave (20,22) for the fire station)"]   # no other strip
    assert ready_lines("planning", spare_funded(only, [fire, clinic])) == [
        "- zone_R: 1 site, best (20,20)-(21,22), 6 tiles (leave (20,22), (21,20) free)"]


def _zone(env, x: int, y: int) -> dict:
    return ToolBox(env, "planning").call("zone", {"zone": "R", "x1": x, "y1": y, "x2": x, "y2": y})


def test_zoning_the_last_site_of_a_funded_building_warns(env12):
    """Smoke v14 seed 0: planning zoned the lots near homes fire held $2,500 for. A zone that takes the last ready site
    of a funded department's cheapest first building warns, like the riverbank warning; the log keeps the core text."""
    from civitas.fleet.tools import role_obs, site_lists
    fire = lambda env: [(s["x"], s["y"]) for s in site_lists(env.world, "fire", role_obs(env, "fire"))["station"].sites]
    sites = fire(env12)
    env = copy.deepcopy(env12)
    _hem_homes(env, keep=sites[:1])                    # one lot near homes left
    assert fire(env) == sites[:1]
    env.world.dept_funds["fire"] = 2500
    (x, y), = sites[:1]
    res = _zone(env, x, y)
    warn = (f" WARNING: that zoned the last site for fire's fire station near homes (fire holds $2,500 for it): "
            f"dezone one, e.g. ({x},{y})")
    assert res["ok"] and res["message"] == "zoned 1 tiles R (low density)" + warn, res
    assert FleetState.of(env).log[-1]["message"] == "zoned 1 tiles R (low density)"
    env = copy.deepcopy(env12)
    _hem_homes(env, keep=sites[:1])
    env.world.dept_funds.update(police=2500, fire=2500, health=2000)   # one lot for three: police, first, claims it
    assert _zone(env, x, y)["message"].endswith(
        f" WARNING: that zoned the last site for police's police station near homes (police holds $2,500 for it): "
        f"dezone one, e.g. ({x},{y})")
    env = copy.deepcopy(env12)
    _hem_homes(env, keep=sites[:1])
    env.world.dept_funds["fire"] = 2499                # not funded: no warning
    assert _zone(env, x, y)["message"] == "zoned 1 tiles R (low density)"
    env = copy.deepcopy(env12)
    _hem_homes(env, keep=sites[:2])                    # another lot near homes left: no warning
    env.world.dept_funds["fire"] = 2500
    assert fire(env) == sites[:2]
    assert _zone(env, x, y)["message"] == "zoned 1 tiles R (low density)"


def _claimed(env, *buildings) -> list[str]:
    """(role, purpose, building) in turn order -> "building (x,y)" at each one's first site no earlier one took."""
    taken, out = set(), []
    for d, p, b in buildings:
        xy = next(xy for xy in _sites(env, d, p) if xy not in taken)
        taken.add(xy)
        out.append(f"{b} ({xy[0]},{xy[1]})")
    return out


def _sites(env, role: str, purpose: str) -> list[tuple[int, int]]:
    from civitas.fleet.tools import role_obs, site_lists
    return [(s["x"], s["y"]) for s in site_lists(env.world, role, role_obs(env, role))[purpose].sites]


def _leave(env) -> list[str]:
    return [ln for ln in _ready_sites(briefing(env, "planning", mark=False)) if ln.startswith("- leave")]


def test_utilities_are_left_a_lot_only_while_their_supply_is_short(env12, grown_env):
    """Smoke v15: all 8 leave-free lines named a water tower with water never short (e.g. 125 of 250): utilities'
    near_homes list ranks every roadside lot by the homes around it, so planning held back its best residential lot.
    A tower counts while water demand is at least 90% of supply, coal, gas and wind on power, as pump and outlet do."""
    from civitas.fleet.tools import blocked_firsts, funded_firsts, role_obs, site_lists
    env = copy.deepcopy(env12)                         # water 125 of 250, power 203 of 220
    w = env.world
    m = w.metrics()
    assert m["water_demand"] < 0.9 * m["water_supply"] and m["power_demand"] >= 0.9 * m["power_supply"]
    w.dept_funds["utilities"] = 2200                   # a tower's price, short of a turbine's
    lists = site_lists(w, "utilities", role_obs(env, "utilities"))
    assert lists["near_homes"].sites and funded_firsts(w, "utilities", lists) == {} and _leave(env) == []
    w.dept_funds["utilities"] = 2500
    assert _leave(env) == [f"- leave free (funded): wind turbine {_best(env, 'utilities', 'wind')}: zone around them"]
    env = copy.deepcopy(grown_env)                     # month 24: water 313 of 313
    w = env.world
    w.dept_funds.update({d: 0 for d in DEPTS}, utilities=2200)
    assert _leave(env) == [f"- leave free (funded): water tower {_best(env, 'utilities', 'near_homes')}: zone "
                           "around them"]
    env = copy.deepcopy(env12)                         # blocked: kept for only while wanted too
    _fill_roadside(env.world)
    lists = site_lists(env.world, "utilities", role_obs(env, "utilities"))
    assert lists["near_homes"].blocked and set(blocked_firsts(env.world, "utilities", lists)) == {"coal_or_gas", "wind"}


def test_funded_buildings_claim_distinct_sites_in_turn_order(env12):
    """Smoke v15 seed 1, month 6: "leave free (funded): water tower (22,23), police station (34,25), fire station
    (34,25), clinic (34,25), elementary school (34,25)": one lot for four buildings, and with the rest zoned the zone
    warning never fired (each still had a site). Each funded building now claims its first site no earlier one did."""
    env = copy.deepcopy(env12)
    w = env.world
    w.dept_funds.update(police=2500, fire=2500, education=2500)
    want = _claimed(env, ("police", "station", "police station"), ("fire", "station", "fire station"),
                    ("education", "school", "elementary school"))
    assert _sites(env, "police", "station")[0] == _sites(env, "fire", "station")[0] == (20, 22) and len(set(want)) == 3
    assert _leave(env) == ["- leave free (funded): " + ", ".join(want) + ": zone around them"]
    spots = [tuple(map(int, t.strip("()").split(","))) for t in re.findall(r"\(\d+,\d+\)", _leave(env)[0])]
    _hem_homes(env, keep=spots)                        # planning zoned around them, as the line asks
    (fx, fy) = spots[1]
    res = _zone(env, fx, fy)                           # fire's lot: its other sites are police's and the school's
    assert res["message"].endswith(f" WARNING: that zoned the last site for fire's fire station near homes (fire "
                                   f"holds $2,500 for it): dezone one, e.g. ({fx},{fy})"), res


def test_the_leave_free_line_drops_whole_items_at_its_cap():
    """Smoke v15 seed 0, month 9: "...elementary school (28,42): zone arou", cut at 240 characters."""
    from civitas.fleet.briefing import BLOCKED_LINE_CHARS, _leave_line
    from civitas.fleet.tools import Spot
    spots = [Spot("education", "school", "elementary school", (10 + i, 40), (1, 1)) for i in range(12)]
    line, = _leave_line(spots)
    assert len(line) <= BLOCKED_LINE_CHARS and line.endswith(" (+5 more): zone around them"), line
    assert line.startswith("- leave free (funded): elementary school (10,40), ") and "(16,40) (+5" in line
    assert _leave_line(spots[:3]) == ["- leave free (funded): elementary school (10,40), elementary school (11,40), "
                                      "elementary school (12,40): zone around them"]


def test_a_2x2_funded_site_is_spared_whole():
    """A funded university's site is its 2x2 block's top-left tile: strips and claims check the whole block."""
    from civitas.fleet.tools import SiteList, Spot, claim_spots, ready_lines, spare_funded, Price
    uni = Spot("education", "university", "university", (20, 20), (2, 2))
    strip = {"zone_R": SiteList([{"x": 21, "y": 21, "x1": 21, "y1": 21, "x2": 22, "y2": 25, "tiles": 10}], None, None)}
    assert ready_lines("planning", spare_funded(strip, [uni])) == [
        "- zone_R: 1 site, best (21,21)-(22,25), 10 tiles (leave (20,20) for the university)"]   # (21,21) is its
    spots = claim_spots({("education", "university"): (Price(12000, "university", size=(2, 2)), [(20, 20)]),
                         ("fire", "station"): (Price(2500, "fire station"), [(21, 21), (23, 20)])})
    assert [(s.building, s.xy) for s in spots.values()] == [("university", (20, 20)), ("fire station", (23, 20))]


def test_kept_claims_try_their_own_site_first():
    """Review: a kept building was only processed first, taking its first unclaimed site in list order, so police,
    kept on (5,5), took (3,3) once that was listed ahead of it and moved fire off its kept (3,3). Kept buildings now try
    their kept site first: only a building whose site went is re-seated, and the warning names that one."""
    from civitas.fleet.tools import Price, claim_spots
    P, Q, R = (3, 3), (5, 5), (7, 7)
    pol, fire = Price(2500, "police station"), Price(2500, "fire station")
    was = claim_spots({("police", "station"): (pol, [Q]), ("fire", "station"): (fire, [P])})
    now = claim_spots({("police", "station"): (pol, [P, Q]), ("fire", "station"): (fire, [P, R])}, keep=was)
    assert {k: s.xy for k, s in now.items()} == {("police", "station"): Q, ("fire", "station"): P}
    gone = claim_spots({("police", "station"): (pol, [P, R]), ("fire", "station"): (fire, [P, R])}, keep=was)
    assert {k: s.xy for k, s in gone.items()} == {("fire", "station"): P, ("police", "station"): R}


def _warn_after(monkeypatch, env, before: dict, after: dict[str, list]) -> str:
    """planning's _funded_warning for `before` ((dept, purpose) -> (Price, sites)) once the claimed sites of `before`
    are zoned (the first claim's tiles) and each department's "station" list reads `after[dept]`."""
    import civitas.fleet.tools as T
    from civitas.fleet.tools import SiteList, claim_spots
    w = env.world
    for sp in claim_spots(before).values():
        if sp.xy not in [xy for sites in after.values() for xy in sites]:
            w.struct[sp.xy[1], sp.xy[0]] = C.RES
    monkeypatch.setattr(T, "site_lists", lambda w, role, obs: {"station": SiteList(
        [{"x": x, "y": y} for x, y in after[role]], None, None)})
    return ToolBox(env, "planning")._funded_warning(before)


def test_a_zone_warns_only_when_a_fresh_claim_leaves_a_building_out(monkeypatch):
    """Review: police lists P and Q, fire Q and R, and a zone takes P. Sticky claims kept fire on Q, left police with
    nothing and warned that police lost its last site; a fresh claim in turn order seats police on Q and fire on R,
    which is how play goes. The warning now needs a fresh claim that seats fewer buildings; the sticky one only
    names them."""
    from civitas.fleet.tools import Price
    P, Q, R = (3, 3), (5, 5), (7, 7)
    pol, fire = Price(2500, "police station"), Price(2500, "fire station")
    env = _env()
    env.world.dept_funds.update(police=2500, fire=2500)
    before = {("police", "station"): (pol, [P, Q]), ("fire", "station"): (fire, [Q, R])}
    assert _warn_after(monkeypatch, env, before, {"police": [Q], "fire": [Q, R]}) == ""
    env = _env()
    env.world.dept_funds.update(police=2500, fire=2500)
    before = {("police", "station"): (pol, [P]), ("fire", "station"): (fire, [Q, R])}
    assert _warn_after(monkeypatch, env, before, {"police": [], "fire": [Q, R]}) == (
        " WARNING: that zoned the last site for police's police station near homes (police holds $2,500 for it): "
        "dezone one, e.g. (3,3)")


def test_the_zone_warning_never_fails_a_zone(env12, monkeypatch):
    """The funded look-ups run before and after apply_action: an error in them means no warning, not a failed zone;
    their result is kept until an action succeeds."""
    import civitas.fleet.tools as T
    from civitas.fleet.tools import role_obs, site_lists
    env = copy.deepcopy(env12)
    env.world.dept_funds["fire"] = 2500
    box = ToolBox(env, "planning")
    funded = box._funded_sites()
    assert ("fire", "station") in funded
    res = box.call("zone", {"zone": "R", "x1": 19, "y1": 21, "x2": 19, "y2": 23})    # road: fails
    assert not res["ok"] and box._funded is funded
    (x, y) = _sites(env, "fire", "station")[0]
    assert box.call("zone", {"zone": "R", "x1": x, "y1": y, "x2": x, "y2": y})["ok"] and box._funded is None
    real = T.site_lists
    monkeypatch.setattr(T, "site_lists", lambda w, role, obs: (1 / 0) if role == "fire" else real(w, role, obs))
    box = ToolBox(env, "planning")
    s = real(env.world, "fire", role_obs(env, "fire"))["station"].sites[0]
    res = box.call("zone", {"zone": "R", "x1": s["x"], "y1": s["y"], "x2": s["x"], "y2": s["y"]})
    assert res["ok"] and res["message"] == "zoned 1 tiles R (low density)", res


def test_riverbank_and_funded_warnings_come_once():
    """The last free bank lot, funded for utilities' outlet: the riverbank warning alone."""
    env = _env()
    w = env.world
    taken = _bridge_and_take_the_bank(env)
    x, y = taken[0]
    w.struct[y, x] = C.NONE
    w.refresh()
    w.dept_funds["utilities"] = 7000
    res = _zone(env, x, y)
    assert res["message"].count("WARNING") == 1 and "riverbank lot" in res["message"], res


def test_a_funded_building_whose_every_site_is_claimed_is_named_and_blocked(env12):
    """Smoke v18 seed 1, month 6: the mayor funded education's first school, but its only site (38,25) was the police
    station's too, claimed first in turn order. Planning read only "leave free (funded): police station (38,25)";
    education was on neither that line nor the blocked one (its list wasn't empty), read "no free roadside land" in
    month 6 and never built (33.4). Planning's line now says it has no lot of its own, and the mayor's line counts it as
    blocked, its money kept (not idle)."""
    from civitas.fleet.briefing import BLOCKED_NOTE
    from civitas.fleet.tools import DeptMoney
    env = copy.deepcopy(env12)
    w = env.world
    _hem_homes(env, keep=[(20, 22)])                   # one lot near homes: every station's and the school's only one
    w.dept_funds.update({d: 0 for d in DEPTS if d != "planning"}, police=2500, education=2500)
    assert _sites(env, "police", "station") == _sites(env, "education", "school") == [(20, 22)]
    note = lambda d, b: f"; {d} ({b}): no lot of its own - leave or free one near (20,22)"
    assert _leave(env) == ["- leave free (funded): police station (20,22): zone around them"
                           + note("education", "elementary school")]
    lines = _dept_lines(env)
    assert lines["education"] == "- education: funds $2,500, spent $0; blocked: an elementary school"
    assert lines["police"] == "- police: funds $2,500, spent $0; funds cover its builds"
    assert briefing(env, "mayor", mark=False).count(BLOCKED_NOTE) == 1
    m = DeptMoney(env, "education")
    assert m.held == 2500 and m.keep == 2500 and m.idle == 0      # held through a turn, kept for the school
    w.dept_funds["health"] = 2000                      # a tie: the clinic's only lot is police's, the landfill has its own
    assert _leave(env) == ["- leave free (funded): police station (20,22), landfill (18,40): zone around them"
                           + note("health", "clinic") + note("education", "elementary school")]
    lines = _dept_lines(env)
    assert lines["health"] == "- health: funds $2,000, spent $0; funds cover its builds"   # its landfill can go up
    w.dept_funds["police"] = 0                         # the school claims the lot itself
    assert _leave(env) == ["- leave free (funded): clinic (20,22), landfill (18,40): zone around them"
                           + note("education", "elementary school")]
    env = copy.deepcopy(env12)                         # a lot of its own: nothing to add
    env.world.dept_funds.update({d: 0 for d in DEPTS if d != "planning"}, police=2500, education=2500)
    assert _leave(env) == ["- leave free (funded): " + ", ".join(_claimed(
        env, ("police", "station", "police station"), ("education", "school", "elementary school")))
        + ": zone around them"]
    assert "blocked" not in _dept_lines(env)["education"]


def test_the_leave_free_line_keeps_its_unclaimed_notes_at_its_cap():
    """The notes for funded buildings with no lot of their own come after the claimed sites; past BLOCKED_LINE_CHARS
    the claimed sites drop first, whole and counted, then the notes."""
    from civitas.fleet.briefing import BLOCKED_LINE_CHARS, _leave_line
    from civitas.fleet.tools import Spot
    spots = [Spot("education", "school", "elementary school", (10 + i, 40), (1, 1)) for i in range(12)]
    notes = [Spot("health", "clinic", "clinic", (10, 40)), Spot("fire", "station", "fire station", (11, 40))]
    line, = _leave_line(spots, notes)
    assert len(line) <= BLOCKED_LINE_CHARS and line.endswith(
        " more): zone around them; health (clinic): no lot of its own - leave or free one near (10,40); fire (fire "
        "station): no lot of its own - leave or free one near (11,40)"), line
    assert line.startswith("- leave free (funded): elementary school (10,40) (+11 more)"), line
    line, = _leave_line(spots, notes * 2)
    assert len(line) <= BLOCKED_LINE_CHARS and line.endswith("near (11,40) (+2 more)") and "(+11 more)" in line, line
    assert _leave_line(spots[:1], notes[:1]) == [
        "- leave free (funded): elementary school (10,40): zone around them; health (clinic): no lot of its own - "
        "leave or free one near (10,40)"]


def test_a_zone_that_leaves_several_funded_buildings_no_site_names_a_tile_for_each(env12):
    """Smoke v18 seed 1, month 9: each of planning's two high-density strips warned "that zoned the last site for
    fire's fire station near homes ($2,500 held) and health's landfill ($2,000 held): dezone one, e.g. (46,29)" (then
    police's, the clinic's and the school's), and planning dezoned exactly that tile, which seats one building. Each
    building now gets a tile of its own from its list among those the zone took; a landfill one that is away from
    homes once dezoned, else a lot at the town edge."""
    env = copy.deepcopy(env12)
    _hem_homes(env, keep=[(20, 22), (22, 20)])         # two lots near homes: police claims one, fire the other
    env.world.dept_funds.update({d: 0 for d in DEPTS if d != "planning"}, police=2500, fire=2500)
    assert _sites(env, "police", "station") == _sites(env, "fire", "station") == [(20, 22), (22, 20)]
    res = ToolBox(env, "planning").call("zone", {"zone": "R", "x1": 20, "y1": 20, "x2": 22, "y2": 22})
    assert res["message"] == (
        "zoned 5 tiles R (low density) WARNING: that zoned the last site for police's police station near homes "
        "($2,500 held) and fire's fire station near homes ($2,500 held): dezone (20,22) for the police station and "
        "(22,20) for the fire station"), res
    head = ("WARNING: that zoned the last site for fire's fire station near homes ($2,500 held) and health's landfill "
            "($2,000 held): dezone (20,22) for the fire station")
    edge = ("the landfill needs a lot at the town edge: leave one by (40,19), the far end of road_extension "
            "(25,19)-(40,19)")
    for zone, fix in (("R", "; " + edge), ("C", " and (18,40) for the landfill")):   # homes around it, or shops
        env = copy.deepcopy(env12)
        _hem_homes(env, keep=[(20, 22)])               # fire claims (20,22), the landfill (18,40) by the new road
        env.world.dept_funds.update({d: 0 for d in DEPTS if d != "planning"}, fire=2500, health=2000)
        assert _sites(env, "health", "landfill")[0] == (18, 40)
        res = ToolBox(env, "planning").call("zone", {"zone": zone, "x1": 18, "y1": 22, "x2": 20, "y2": 47})
        assert res["message"].endswith(head + fix), (zone, res)


def test_a_zone_names_each_building_a_distinct_tile_from_its_own_list(monkeypatch):
    """Police and fire both list P and Q, police claims P and fire Q, and a zone takes both: each gets its own claimed
    tile, not the first of its list (P twice)."""
    from civitas.fleet.tools import Price
    P, Q = (3, 3), (5, 5)
    pol, fire = Price(2500, "police station"), Price(2500, "fire station")
    env = _env()
    env.world.dept_funds.update(police=2500, fire=2500)
    before = {("police", "station"): (pol, [P, Q]), ("fire", "station"): (fire, [P, Q])}
    assert _warn_after(monkeypatch, env, before, {"police": [], "fire": []}) == (
        " WARNING: that zoned the last site for police's police station near homes ($2,500 held) and fire's fire "
        "station near homes ($2,500 held): dezone (3,3) for the police station and (5,5) for the fire station")


def test_a_landfill_losing_its_last_site_alone_is_told_where_it_can_go(env12):
    """One building: a station keeps "dezone one, e.g. (x,y)" (test_zoning_the_last_site_of_a_funded_building_warns);
    a landfill's tile must be away from homes once dezoned, else it needs a lot at the town edge: smoke v18's (46,35)
    sat in a strip of new homes."""
    env = copy.deepcopy(env12)
    _hem_homes(env, keep=[(20, 22)])
    env.world.dept_funds.update({d: 0 for d in DEPTS if d != "planning"}, health=2000)   # clinic (20,22), landfill
    res = ToolBox(env, "planning").call("zone", {"zone": "R", "x1": 18, "y1": 40, "x2": 18, "y2": 46})    # (18,40)
    assert res["message"] == (
        "zoned 7 tiles R (low density) WARNING: that zoned the last site for health's landfill (health holds $2,000 "
        "for it): it needs a lot at the town edge: leave one by (40,19), the far end of road_extension "
        "(25,19)-(40,19)"), res
    env = copy.deepcopy(env12)
    _hem_homes(env, keep=[(20, 22)])
    env.world.dept_funds.update({d: 0 for d in DEPTS if d != "planning"}, health=2000)
    res = ToolBox(env, "planning").call("zone", {"zone": "C", "x1": 18, "y1": 40, "x2": 20, "y2": 47})
    assert res["message"].endswith(" WARNING: that zoned the last site for health's landfill (health holds $2,000 "
                                   "for it): dezone one, e.g. (18,40)"), res


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
    assert len(fire) < 270
    assert "no reports yet" not in text.lower()


def test_mayor_briefing_marks_departments_below_their_target():
    """The "<" marks a plan-target gap (Funding.gaps); what to fund is needs-based whatever the targets: the to-fund
    line lists what the needs lines lack, and a department below its target with nothing ready isn't on it. Smoke
    v16-v19: utilities stayed at a $7,000+ target after month 0 and got $113k of its $135k top-ups beyond its needs."""
    env = _env()
    w = env.world
    w.dept_funds.update(planning=9500, utilities=2000)
    ToolBox(env, "mayor").call("update_plan", {"budgets": {"planning": 9000, "utilities": 7000, "fire": 2500}})
    lines = _dept_lines(env)
    assert lines["planning"] == "- planning: funds $9,500, target $9,000"
    # a building per short supply (v20 fixes; the pump and outlet wait for a road to the river)
    assert lines["utilities"] == ("- utilities: funds $2,000 < target $7,000; needs $6,200 for its first coal plant "
                                  "and water tower ($110/month); sewage outlet waits for a road to the river")
    assert lines["fire"] == "- fire: funds $0 < target $2,500; nothing ready to buy"
    assert lines["police"] == "- police: funds $0; nothing ready to buy"
    to_fund = lambda: next(ln for ln in briefing(env, "mayor", mark=False).splitlines() if ln.startswith("- to fund:"))
    assert to_fund() == ("- to fund: first/supply: utilities $4,200, health $2,000 ($6,200, upkeep +$140/month): "
                         "$27k covers utilities, health")   # what the needs lines lack, not the $5,000 and $2,500 gaps
    w.treasury = 7200
    assert to_fund().endswith(": $4,200 covers utilities; the other $2,000 would go under the reserve")
    text = briefing(env, "mayor", mark=False)
    assert text.index("- no reports yet") < text.index("- to spend: $4,200 above") < text.index("- to fund:")
    w.dept_funds.update(utilities=6400, fire=2350)                  # within 10% of the target counts as funded
    lines = _dept_lines(env)
    assert lines["fire"] == "- fire: funds $2,350, target $2,500; nothing ready to buy"
    assert lines["utilities"] == "- utilities: funds $6,400, target $7,000; funds cover its next build"
    assert "utilities" not in to_fund()
    w.dept_funds.update(fire=2399)                                  # $101 short of $2,500: max($100, 10%) = $250
    assert _dept_lines(env)["fire"].startswith("- fire: funds $2,399, target $2,500")
    w.dept_funds.update(fire=0)
    ToolBox(env, "mayor").call("update_plan", {"budgets": {"fire": 500}})
    w.dept_funds.update(fire=401)                                   # $99 short: under the $100 floor
    assert _dept_lines(env)["fire"].startswith("- fire: funds $401, target $500")
    w.dept_funds.update(fire=399)
    assert _dept_lines(env)["fire"].startswith("- fire: funds $399 < target $500")
    assert "below target" not in briefing(env, "mayor", mark=False)


def test_mayor_departments_show_spending_and_idle_money():
    """Smoke v9-v12: nothing in the mayor's briefing showed what a department spent; planning read "funds $7,834 <
    target $9,000" and got topped up to $10,721."""
    env = _env()
    w = env.world
    w.dept_funds.update(planning=9500)
    text = briefing(env, "mayor", mark=False)
    assert "spent" not in text and "\n## Departments (most urgent first)\n" in text   # month 0: nothing spent or idle
    _months_spent(env, {"planning": 500}, {"planning": 400})
    assert "\n## Departments (most urgent first; spent = last 2 months)\n" in briefing(env, "mayor", mark=False)
    _months_spent(env, {"planning": 266, "police": 300})
    w.treasury = 30000
    w.dept_funds.update(planning=7834, utilities=2000, transport=2000)
    ToolBox(env, "mayor").call("update_plan", {"budgets": {"planning": 9000, "utilities": 7000}})
    text = briefing(env, "mayor", mark=False)
    lines = {ln.split(":")[0][2:]: ln for ln in text.splitlines() if ln.startswith("- ") and ": funds " in ln}
    assert "\n## Departments (most urgent first; spent = last 3 months)\n" in text
    assert lines["planning"] == "- planning: funds $7,834, target $9,000, spent $1,166; idle $5,502 (keeps $2,332)"
    assert lines["transport"] == "- transport: funds $2,000, spent $0; idle $1,000 (keeps $1,000)"
    assert lines["utilities"] == ("- utilities: funds $2,000 < target $7,000, spent $0; needs $6,200 for its first "
                                  "coal plant and water tower ($110/month); sewage outlet waits for a road to the "
                                  "river")
    assert lines["police"] == "- police: funds $0, spent $300; nothing ready to buy"
    assert ("- no reports yet\n- to spend: $27k above the $3,000 reserve + $6,502 idle to take back = $33.5k; "
            "operations ") in text                     # the idle money counts in the money to spend


def test_the_mayor_map_block_keeps_what_it_manages_by():
    """The mayor has no map actions; its briefing needed the room for the departments' spending and idle money.
    Departments keep the whole block, road lines included."""
    from civitas.fleet.tools import MAP_PARTS, MAYOR_MAP, map_facts, map_lines
    assert MAYOR_MAP == ("roads", "unconnected", "built")
    env = _env()
    w = env.world
    w.struct[24, 1:21] = C.ROAD
    w.struct[25:36, 10] = C.ROAD
    w.struct[40, 2:8] = C.ROAD
    w.refresh()
    full = map_lines(map_facts(w))
    assert len(full) == 7 and full == map_lines(map_facts(w), MAP_PARTS)
    block = lambda text: text[text.index("\n## Map\n") + 1:].split("\n\n")[0].splitlines()[1:]
    for role in TURN_ORDER:
        assert block(briefing(env, role, mark=False)) == (full if role != "mayor" else [
            ln for ln in full if ln.startswith(("- highway-connected roads:", "- roads not connected", "- built-up"))])


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
    assert "A tornado hit" in text and "- m0 mayor -> you: police: cover downtown" in text
    assert fs.last_turn_month["police"] == 0                          # mark=False leaves the turn unmarked
    briefing(env, "police")
    assert fs.last_turn_month["police"] == 1 and fs.digest("police", 1, env.council)["messages"] == []
    assert "no new council messages" in briefing(env, "police", mark=False)


def test_briefing_lists_new_council_messages_for_this_role():
    from civitas.fleet.tools import MAX_ENTRIES
    env = _env()
    c = env.council
    c.post(0, "mayor", "OLD-MSG before the last turn", "planning")
    briefing(env, "planning")
    c.post(0, "utilities", "TO-PLANNING need road access at the river " + "r" * 300, "planning")
    c.post(0, "mayor", "TO-ALL keep the treasury positive", "all")
    c.post(0, "fire", "TO-POLICE share the corner lot", "police")
    c.post(0, "planning", "OWN-POST zoning the east side", "all")
    env.end_month()
    text = briefing(env, "planning", mark=False)
    msgs = text[text.index("new council messages"):].split("\n\n")[0]
    line, = [ln for ln in msgs.splitlines() if "TO-PLANNING" in ln]
    assert line.startswith("  - m0 utilities -> you: ") and line.endswith("...") and len(line) < 240
    assert "  - m0 mayor -> all: TO-ALL keep the treasury positive" in msgs
    assert msgs.index("TO-PLANNING") < msgs.index("TO-ALL")                 # newest last
    assert "TO-POLICE" not in msgs and "OWN-POST" not in msgs and "OLD-MSG" not in text
    assert "older" not in msgs
    for i in range(6):
        c.post(1, "health", f"REQ-{i} clinic lot please", "all")
    text = briefing(env, "planning", mark=False)
    msgs = text[text.index("new council messages"):].split("\n\n")[0]
    assert [f"REQ-{i}" in msgs for i in range(6)] == [False] + [True] * 5
    assert "TO-PLANNING" not in msgs and "(3 older: get_messages)" in msgs
    env.cfg.council_window = 0                             # two of the older ones (month 0) are out of the window
    assert "(3 older, 2 of them beyond get_messages' 0-month window)" in briefing(env, "planning", mark=False)
    env.world.month = 2
    assert "(3 older, beyond get_messages' 0-month window)" in briefing(env, "planning", mark=False)
    assert "- m1 health -> all: REQ-5" in briefing(env, "mayor", mark=False)


def test_briefing_dates_and_milestones():
    env = _env()
    police = briefing(env, "police", mark=False)
    assert "(year 1, January, winter)" in police
    line, = [ln for ln in police.splitlines() if ln.startswith("Next milestone")]
    assert "jail" in line and "high density" not in line and "more" in line and "unlocks" in line
    line, = [ln for ln in briefing(env, "mayor", mark=False).splitlines() if ln.startswith("Next milestone")]
    assert "districts" in line and "$5,000" in line and "options" in line


def test_mayor_warning_and_offer_lines():
    env = _env()
    w = env.world
    assert "WARNING" not in briefing(env, "mayor", mark=False)
    w.months_negative, w.low_approval_months = 3, 2
    w.offers = [{"id": 7, "jobs": 250, "bonus": 8000, "expires": 5}]
    text = briefing(env, "mayor", mark=False)
    warn, = [ln for ln in text.splitlines() if "WARNING" in ln]
    assert f"{C.BANKRUPTCY_MONTHS} = bankruptcy" in warn and f"{C.RECALL_MONTHS} = recall" in warn
    assert f"{100 * C.RECALL_APPROVAL:.0f}%" in warn and "treasury 3 months (" in warn and "% 2 months (" in warn
    w.months_negative, w.low_approval_months = 1, 0
    warn, = [ln for ln in briefing(env, "mayor", mark=False).splitlines() if "WARNING" in ln]
    assert "negative treasury 1 month (" in warn and "% 0 months (" in warn
    w.month = env.cfg.months - 1
    assert "): 1 month left" in briefing(env, "mayor", mark=False).splitlines()[0]
    offer, = [ln for ln in text.splitlines() if "OPEN OFFER #7" in ln]
    assert "250 jobs" in offer and "$8,000" in offer and "accept_offer" in offer and "month 5" in offer


def test_briefings_stay_bounded_at_the_limits(monkeypatch):
    import civitas.fleet.tools as T
    from civitas.fleet.memory import LIMITS
    from civitas.fleet.tools import MAX_ENTRIES, REPORT_MAX_CHARS
    # characters without the memory block (shown in full). The mayor's is higher: its needs lines ("its first ..." /
    # "another ... (coverage 74%, $55/month)"), the Finances split and the to spend / to fund lines are what it was
    # missing in smoke v13-v19, when it funded second stations before education's school and borrowed after its own
    # top-ups; the worst mix below (needs lines on all seven departments, debt at the cap, a 6x deficit, seven
    # buildings lines in the costs, a partial cover in a hold, in a game long enough for "close that in N months")
    # takes 7,624, under the plan's 7,650, with MAYOR_REPORT_CHARS at 170; with utilities short of all three supplies
    # (v20 fixes: a building per short supply on one line), 7,666 at its longest labels; with two and the third's
    # building waiting for a lot (review), 7,683
    bound = lambda role: 7700 if role == "mayor" else 7450
    env = _env()
    w = env.world
    spend = {d: 1111 for d in DEPTS}                     # $3,333 over the 3 months on every Departments line
    _months_spent(env, spend, spend)
    item = lambda tag: (tag + " ") * (LIMITS["item_chars"] // (len(tag) + 1))
    full = lambda tag: [item(tag)] * LIMITS["list_items"]
    ToolBox(env, "mayor").call("update_plan", {"goals": [item("goal")] * LIMITS["goals"],
                                               "budgets": {d: 12345 for d in DEPTS},
                                               "priorities": {d: item("prio") for d in DEPTS},
                                               "notes": "note " * LIMITS["notes_chars"]})
    for role in TURN_ORDER:
        ToolBox(env, role).call("update_memory", {"commitments": full("MEM-C"), "open_tasks": full("MEM-T"),
                                                  "lessons": full("MEM-L")})
        if role != "mayor":
            ToolBox(env, role).call("end_turn", {"report": "rep " * REPORT_MAX_CHARS})
        for i in range(10):
            FleetState.of(env).record(0, role, "action", action={"type": "build_x", "x": i, "y": i}, ok=False,
                                      message="m" * 300, cost=0)
    for i in range(MAX_ENTRIES + 5):
        env.council.post(0, "press", f"msg {i} " + "y" * 300, "all")
    for i in range(12):
        w.event(f"event {i} " + "e" * 300)
    w.struct[36:48, 37:47] = C.ROAD                     # 121 road tiles off the highway, spanning x 10-46, y 10-47
    w.struct[10, 10] = C.ROAD
    w.struct[24, 1:26] = C.ROAD                         # 9 road lines: the main road to the river and 8 streets off it
    w.struct[25:41, 10:25:2] = C.ROAD
    w.refresh()
    _months_spent(env, spend)
    w.dept_funds.update(planning=9876, transport=9876, police=9876, fire=9876, education=9876)   # idle: keep $6,666
    w.months_negative, w.low_approval_months, w.treasury = 11, 11, -123456.78        # every warning, below the reserve
    w.offers.append({"id": 12345, "jobs": 1234, "bonus": 123456, "expires": 123})
    road_lines = "- road lines: y=24 x 0-25; " + "; ".join(f"x={x} y 24-40" for x in range(10, 23, 2)) + " +1 more\n"
    for role in TURN_ORDER:
        text = briefing(env, role, mark=False)
        memory = text[text.index("## Your memory"):].split("\n\n")[0]
        assert len(text) - len(memory) < bound(role), (role, len(text) - len(memory))
        assert (road_lines in text) == (role != "mayor"), role                   # a full road lines line
        assert ("spent $3,333" in text) == (role == "mayor"), role
        assert len(text) < 13000, (role, len(text))
        assert "MEM-C" in text and "MEM-L" in text                            # memory shown to every role
        assert f"({MAX_ENTRIES} older; get_messages shows only the latest {MAX_ENTRIES})" in text and f"msg {MAX_ENTRIES + 4} " in text
    sizes = {}

    def check(tag: str) -> str:
        """Every role's briefing within its bound; the mayor's (its size recorded under `tag`) returned."""
        for role in TURN_ORDER:
            text = briefing(env, role, mark=False)
            memory = text[text.index("## Your memory"):].split("\n\n")[0]
            assert len(text) - len(memory) < bound(role), (tag, role, len(text) - len(memory))
            sizes[tag] = len(text) - len(memory) if role == "mayor" else sizes.get(tag, 0)
        mayor = briefing(env, "mayor", mark=False)
        assert "- to spend: " in mayor and "- to fund: " in mayor, tag
        assert not any(s in mayor for s in ("; priority: ", "first buildings still unfunded", "below target (")), tag
        return mayor

    def money_at_the_limits(treasury: float) -> None:
        """Debt at the cap, a 6x deficit last month with operations positive over the window (the hold's "close that
        in N months", in a game long enough to close it), seven buildings lines in last month's costs, and the
        treasury at `treasury`."""
        env.cfg.months = 9999
        prev, last = w.history[-2:]
        prev.update(income=99999.0, expenses=12345.0)
        costs = {d: 12345.0 for d in ("utilities", "roads", "transport", "police", "fire", "health", "education")}
        spent = {**costs, "town_hall": 20.0, "city_services": 12345.0, "loan_interest": 250.0}
        last.update(income=12000.0, expenses=sum(spent.values()), expense_breakdown=spent, dept_upkeep=costs)
        w.debt, w.treasury = C.MAX_DEBT, treasury

    mayor = check("every warning")
    assert "WARNING" in mayor and "OPEN OFFER" in mayor and "under the $3,000 reserve" in mayor
    assert "- roads not connected to the highway: 121 tiles (x 10-46, y 10-47) - reconnect them" in mayor
    assert "\n## Departments (most urgent first; spent = last 3 months; cut reports: history kind=report)\n" in mayor
    assert mayor.count("; idle $3,210 (keeps $6,666)") == 5
    # the idle money doesn't lift the treasury above $3,000: nothing for first/supply either (review: the old dip
    # counted it with the treasury at -$123k)
    assert "- to spend: none (hold): " in mayor and "; nothing above $3,000\n" in mayor
    # the needs-heavy mix: 2 idle, 5 needs lines, 4 of them for a first building, each as long as it gets: 6 sites,
    # its upkeep, health's another clinic and landfill (the longest tie); the to-fund line lists all three kinds
    N, FB = T.Need, T.FirstBuild
    mix = {"utilities": FB(1200, (N("sewage outlet", 6, 0, first=True, upkeep=15),), 1200, ()),
           "police": FB(2500, (N("police station", 6, 0, first=True, upkeep=55),), 2500, ()),
           "fire": FB(2500, (N("fire station", 6, 0, first=True, upkeep=55),), 2500, ()),
           "health": FB(2000, (N("clinic", 6, 1, "coverage 21%", upkeep=40),
                               N("landfill", 6, 1, "coverage 83%", upkeep=30)), 2000, ()),
           "education": FB(2500, (N("elementary school", 6, 0, first=True, upkeep=45),), 2500, ())}
    w.dept_funds.update(utilities=1000, police=2000, fire=2000, health=1500, education=2000)
    with monkeypatch.context() as mp:
        real = T.first_build
        mp.setattr(T, "first_build", lambda env, dept: mix.get(dept) or real(env, dept))
        mayor = check("needs")
        assert mayor.count("; idle $3,210 (keeps $6,666)") == 2 and mayor.count("; needs $") == 5
        assert mayor.count("its first ") == 4
        assert ("- to fund: first/supply: utilities $200, police $500, fire $500, education $500 ($1,700, "
                "upkeep +$170/month) | another: health $500 (upkeep +$40/month)") in mayor
        dear = T.road_max(w)                           # planning short of the dearest road and zoning, not idle
        road = N(T.a_or_an(T.ROAD_NAME), 6, None, "opens 999 tiles")
        mp.setattr(T, "road_build", lambda env: FB(dear, (road,), dear, ()))
        w.dept_funds["planning"] = dear - 1
        mayor = check("needs + road")
        assert mayor.count("; needs $") == 6
        assert "- planning: funds $1,409 < target $12.3k, spent $3,333; needs $1,410 for a road into open land and " \
               "zoning it (opens 999 tiles)\n" in mayor
        # and the money at its limits: a partial cover in a hold (first/supply and growth, from above $3,000 and
        # idle), then above the reserve with everything covered
        w.dept_funds.update(utilities=0, police=0, fire=0, health=0, education=0)
        money_at_the_limits(3500.0)
        mayor = check("needs + money, hold")
        assert "- debt $50k: $250/month interest, at the $50k limit\n" in mayor
        assert "(6 x $99k expenses: last month's deficit was $87k)" in mayor and "close that in " in mayor
        assert "buildings $86.4k (utilities $12.3k, roads $12.3k, 5 more $61.7k)" in mayor
        assert (": $3,710 covers utilities, police, planning; the other $5,000 would go under $3,000\n"   # greedy: the
                in mayor)                                                   # $1 road fits after a skipped station
        money_at_the_limits(999999.0)
        check("needs + money, above the reserve")
        # every department on a needs line (review: 3 first, planning's road, 3 another with coverage), in the hold
        seven = {"utilities": mix["utilities"], "education": mix["education"],
                 "transport": FB(1500, (N("bus line", 6, 0, first=True, upkeep=76),), 1500, ()),
                 "police": FB(2500, (N("police station", 6, 1, "coverage 74%", upkeep=55),), 2500, ()),
                 "fire": FB(2500, (N("fire station", 6, 1, "coverage 74%", upkeep=55),), 2500, ()),
                 "health": mix["health"]}
        mp.setattr(T, "first_build", lambda env, dept: seven.get(dept) or real(env, dept))
        w.dept_funds.update(transport=0)
        money_at_the_limits(3500.0)
        mayor = check("seven needs + money, hold")
        assert mayor.count("; needs $") == 7 and mayor.count("its first ") == 3, mayor
        # utilities short of all three supplies (v20 fixes: a building per short supply, bought together), the labels
        # as long as they get (review): alternating heads, and past 10,000 residents an advanced treatment where no
        # outlet site is ready
        seven["utilities"] = FB(19500, (N("wind turbine", 6, 1, "power 93% used", upkeep=20),
                                        N("water pump", 6, 0, "water 25% covered", first=True, upkeep=25),
                                        N("advanced treatment", 6, 1, "sewage 99% used", upkeep=110)),
                                19500, (), all_of=True)
        mayor = check("seven needs, three supplies + money, hold")
        assert ("; needs $19.5k for another wind turbine, its first water pump and another advanced treatment "
                "($155/month)\n") in mayor
        # two named and the third supply's building with no site (review: the mayor's line says what it waits for)
        seven["utilities"] = FB(4000, (N("wind turbine", 6, 1, "power 93% used", upkeep=20),
                                       N("water pump", 6, 0, "water 25% covered", first=True, upkeep=25)),
                                5200, (), all_of=True, waits=("sewage outlet waits for a free riverbank lot",))
        mayor = check("seven needs, two supplies and a wait + money, hold")
        assert ("; needs $4,000 for another wind turbine and its first water pump ($45/month); sewage outlet waits for "
                "a free riverbank lot\n") in mayor
        w.dept_funds.update(planning=9876, transport=9876)
    w.debt = 0.0
    _fill_roadside(w)                                  # every roadside lot taken: a blocked line outruns an idle one
    monkeypatch.setattr(T, "_uncovered_need", lambda w, role, purpose: ((0, 0), 1.0) if (role, purpose) in T.NEAR_NEED
                        else None)                     # as if the (empty) map had people to serve
    # the blocked mix: 2 idle, 5 blocked, each holding its first building's price (4 digits, like the idle ones)
    w.dept_funds.update(utilities=4000, police=2500, fire=2500, health=2000, education=2500)
    mayor = check("blocked")
    assert mayor.count("; idle $3,210 (keeps $6,666)") == 2 and mayor.count("; blocked: ") == 5, mayor
    assert "\n- police: funds $2,500 < target $12.3k, spent $3,333; blocked: a police station\n" in mayor
    assert "; blocked: a clinic or a landfill\n" in mayor
    assert mayor.count("ask planning") == mayor.count(BLOCKED_NOTE) == 1
    assert mayor.index("- planning: ") < mayor.index(BLOCKED_NOTE) < mayor.index("- to spend: ")   # idle lines last
    money_at_the_limits(3500.0)
    check("blocked + money")
    assert "\n- blocked for lack of land: utilities (" in briefing(env, "planning", mark=False)
    # the plan's budget was 7,650 (7,624 at "seven needs + money, hold"); utilities' line naming a building per short
    # supply (v20 fixes) took the worst mix to 7,661, its longest labels to 7,666, and a supply's building waiting for
    # a lot (review) to 7,683, under the bound
    assert max(sizes.values()) < 7690, sizes
    assert max(sizes, key=sizes.get) == "seven needs, two supplies and a wait + money, hold", sizes



# ------------------------------------------------------------------ tool agent + fleet configs
import contextlib
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from civitas.agents.llm_client import LLMError
from pathlib import Path

from civitas.config import apply_cli_overrides, build_agents, game_config, load_config, scoring_config
from civitas.fleet.agent import LENGTH_NUDGE, LENGTH_NUDGES, NUDGE, ToolAgent
from civitas.fleet.config import FleetAgentConfig, fleet_agent_config, fleet_version

TITLES = {spec["title"]: r for r, spec in ROLES.items()}


def _call(name, args, cid="auto"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _script(role: str, messages: list[dict]) -> list[dict]:
    """The fake model's tool calls for this role at this step (step = assistant messages so far)."""
    step = sum(m["role"] == "assistant" for m in messages)
    if role == "mayor":
        seq = [_call("update_plan", {"budgets": {"utilities": 3000}, "goals": ["power the city"]}),
               _call("allocate_budget", {"department": "utilities", "amount": 3000}),
               _call("end_turn", {"report": "plan set"})]
        return [{**seq[step], "id": f"m{step}"}] if step < len(seq) else []
    if role == "utilities":
        if step == 0:
            return [_call("find_sites", {"purpose": "wind"}, "u0")]
        if step == 1:
            last = json.loads([m for m in messages if m["role"] == "tool"][-1]["content"])
            s = last["sites"][0]
            return [_call("build_power_plant", {"kind": "wind", "x": s["x"], "y": s["y"]}, "u1")]
        return [_call("end_turn", {"report": "built wind"}, "u2")] if step == 2 else []
    return [_call("end_turn", {"report": "nothing to do"}, None)] if step == 0 else []   # no id: agent fills one in


class FakeToolLLM(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeToolLLM.requests.append(body)
        title = re.search(r"You are the (.+?) in a fleet", body["messages"][0]["content"]).group(1)
        calls = _script(TITLES[title], body["messages"])
        msg = {"role": "assistant", "content": "", "reasoning_content": "hmm, let me think"}
        if calls:
            msg["tool_calls"] = calls
        out = json.dumps({"choices": [{"message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                          "usage": {"prompt_tokens": 50, "completion_tokens": 10}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def _fleet_cfg(port: int, months: int = 2) -> dict:
    cfg = load_config(None)
    cfg["name"] = "fleet_fake"
    cfg["game"]["months"] = months
    cfg["providers"]["fake"] = {"base_url": f"http://127.0.0.1:{port}/v1", "api_key_env": "", "stream": False,
                                "max_retries": 0}
    cfg["fleet"] = {"mode": "autonomous", "provider": "fake", "model": "fake", "max_steps": 6,
                    "playbook": "Grow steadily."}
    cfg["agents"] = {r: {"kind": "fleet"} for r in TURN_ORDER}
    cfg["agents"]["mayor"]["playbook"] = "Fund utilities first."
    return cfg


@contextlib.contextmanager
def _fake_llm():
    FakeToolLLM.requests = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeToolLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv
    finally:
        srv.shutdown()


def test_fleet_episode_over_http(tmp_path, monkeypatch):
    from civitas.bench import runner
    envs = []
    real = runner.make_env
    monkeypatch.setattr(runner, "make_env", lambda *a, **k: envs.append(real(*a, **k)) or envs[-1])
    with _fake_llm() as srv:
        s = runner.run_episode(_fleet_cfg(srv.server_port), 0, tmp_path, verbose=False)
    env, = envs
    d = tmp_path / s["run_id"]
    start = real(runner.game_config(_fleet_cfg(0), 0)).world.dept_funds["utilities"]
    traj = [json.loads(x) for x in (d / "trajectories.jsonl").read_text().splitlines()]
    assert len(traj) == 16 and all("tool_trace" in t and "steps" in t and "ended" in t for t in traj)
    assert all(t["ended"] and t["finish_reason"] == "end_turn" for t in traj)
    mayor = [t for t in traj if t["role"] == "mayor"][0]
    assert [c["tool"] for c in mayor["tool_trace"]] == ["update_plan", "allocate_budget", "end_turn"]
    assert mayor["actions"] == [{"type": "allocate_budget", "department": "utilities", "amount": 3000}]
    assert mayor["results"][0]["ok"] and mayor["thinking"] == "plan set"
    util = [t for t in traj if t["role"] == "utilities"][0]
    assert util["actions"][0]["type"] == "build_power_plant" and util["results"][0]["ok"], util["results"]
    assert util["usage"]["completion_tokens"] == 30 and util["steps"] == 3
    allocs = [r for t in traj if t["role"] == "mayor" for r in t["results"] if r["ok"]]
    builds = [r for t in traj if t["role"] == "utilities" for r in t["results"] if r["ok"]]
    assert len(allocs) == 2 and builds and all(r["cost"] == 2500 for r in builds)
    assert env.world.dept_funds["utilities"] == pytest.approx(start + 3000 * len(allocs) - 2500 * len(builds))
    assert util["tool_trace"][1]["result"]["budget_left"] == pytest.approx(start + 500)
    slim = {"ok", "error", "message", "cost", "budget_left"}
    assert all(set(c["result"]) <= slim and c["finish_reason"] == "tool_calls" for t in traj for c in t["tool_trace"])
    assert "sites" not in util["tool_trace"][0]["result"]
    assert env.world.facilities(C.WIND)
    assert env.fleet.plan["budgets"] == {"utilities": 3000}
    assert s["agent_stats"]["utilities"]["actions"] == 2 and s["agent_stats"]["utilities"]["invalid"] == 0
    assert "fleet:fake" in s["signature"] and s["signature"].endswith("[autonomous]")
    assert s["signature"].count("[autonomous]") == 1
    meta = json.loads((d / "meta.json").read_text())
    assert meta["fleet_version"] == fleet_version(_fleet_cfg(srv.server_port))
    assert set(meta["fleet_prompts"]) == set(TURN_ORDER) and "Fund utilities first." in meta["fleet_prompts"]["mayor"]
    assert all(t["prompt"][0] == {"role": "system", "content": f"(system prompt: meta.json fleet_prompts.{t['role']})"}
               for t in traj)
    assert FakeToolLLM.requests[0]["messages"][0]["content"] == meta["fleet_prompts"]["mayor"]
    assert meta["agents"]["police"] == "fleet:fake"
    first = FakeToolLLM.requests[0]
    assert first["model"] == "fake" and {t["function"]["name"] for t in first["tools"]} >= {"end_turn", "update_plan"}
    assert "Fund utilities first." in first["messages"][0]["content"] and "Grow steadily." in first["messages"][0]["content"]
    for body in FakeToolLLM.requests:
        for m in body["messages"]:
            if m["role"] == "assistant":
                assert set(m) <= {"role", "content", "tool_calls"}
                assert all(c["id"] for c in m.get("tool_calls", []))
    police = [t for t in traj if t["role"] == "police"][0]
    ids = [c["id"] for m in police["prompt"] if m["role"] == "assistant" for c in m.get("tool_calls", [])]
    assert ids and ids == [m["tool_call_id"] for m in police["prompt"] if m["role"] == "tool"]
    assert all(t["finish_reasons"] and t["length_stops"] == 0 for t in traj)
    assert mayor["finish_reasons"] == ["tool_calls"] * 3
    # end-of-run fleet report
    rep = json.loads((d / "fleet_report.json").read_text())
    md = (d / "fleet_report.md").read_text()
    assert {"scoreboard", "plan", "roles", "hints"} <= set(rep)
    assert all(r in md for r in TURN_ORDER) and list(rep["roles"]) == list(TURN_ORDER)
    assert s["run_id"] in md and rep["plan"]["versions"] == 2 and rep["plan"]["budgets"] == {"utilities": 3000}
    assert rep["roles"]["utilities"]["tool_calls"] == 6 and rep["roles"]["utilities"]["actions"] == 2
    assert rep["roles"]["mayor"]["completion_tokens"] == 60 and rep["scoreboard"]["city"] == s["final_scores"]["city"]
    assert s["fleet"] == {"report": "fleet_report.md", "hints": len(rep["hints"])}
    assert "2 of 2 months played" in md and "saved in months 0, 1" in md and "- mayor: plan set" in md
    rebuilt, _ = fleet_report(env, s, rows_from_trajectories(d / "trajectories.jsonl"))
    assert json.loads(json.dumps(rebuilt)) == rep


def test_a_failing_fleet_report_keeps_the_summary(tmp_path, monkeypatch, capsys):
    from civitas.bench import runner

    def boom(*a):
        raise KeyError("oops")
    monkeypatch.setattr(runner, "fleet_report", boom)
    with _fake_llm() as srv:
        s = runner.run_episode(_fleet_cfg(srv.server_port, months=1), 0, tmp_path, verbose=False)
    d = tmp_path / s["run_id"]
    assert s["fleet"] == {"error": "KeyError: 'oops'"}
    assert json.loads((d / "summary.json").read_text())["fleet"] == s["fleet"]
    assert not (d / "fleet_report.md").exists() and "fleet report failed: KeyError" in capsys.readouterr().err


class FakeClient:
    """Stands in for ChatClient: `reply(messages)` gives the tool calls (or raises)."""

    def __init__(self, reply, tokens: int = 10, content: str = "thinking aloud", finish: str = "stop"):
        self.reply, self.tokens, self.content, self.finish, self.calls = reply, tokens, content, finish, []

    def chat(self, model, messages, *, temperature=0.4, max_tokens=1500, extra=None):
        self.calls.append({"messages": [dict(m) for m in messages], "extra": extra, "max_tokens": max_tokens})
        calls = self.reply(messages)
        return {"content": self.content, "reasoning": "secret", "tool_calls": calls, "finish_reason": self.finish,
                "usage": {"prompt_tokens": 5, "completion_tokens": self.tokens}, "latency_s": 0.01}


def _agent(role="police", reply=lambda m: [], tokens=10, content="thinking aloud", finish="stop", **cfg):
    env = _env()
    a = ToolAgent(role, cfg=FleetAgentConfig(model="fake", **cfg))
    a.client = FakeClient(reply, tokens, content, finish)
    a.bind(env)
    return a, env


def test_tool_agent_ends_after_one_nudge_without_tool_calls():
    a, env = _agent()
    d = a.act(env.observe("police"))
    assert len(a.client.calls) == 2
    assert "Use the tools" in a.client.calls[1]["messages"][-1]["content"]
    assert d.meta["finish_reason"] == "no_tool_calls" and not d.meta["ended"] and d.meta["steps"] == 0
    assert d.executed == [] and d.thinking == "(no report)"
    assert d.meta["messages"][2]["content"] == "thinking aloud" and _nudges(d) == [NUDGE]
    a, env = _agent(content="")
    d = a.act(env.observe("police"))
    asst = [m for m in d.meta["messages"] if m["role"] == "assistant"]
    assert [m["content"] for m in asst] == ["(no reply)", "(no reply)"]


def test_tool_agent_stops_a_looping_model_at_max_steps():
    a, env = _agent(reply=lambda m: [_call("city_overview", {}, None)], max_steps=3)
    d = a.act(env.observe("police"))
    assert len(d.meta["tool_trace"]) == 3 and d.meta["finish_reason"] == "max_steps"
    assert len(a.client.calls) == 3
    assert a.usage["tool_calls"] == 3 and a.usage["calls"] == 3
    assert d.meta["finish_reasons"] == ["stop"] * 3 and d.meta["length_stops"] == 0
    assert [c["finish_reason"] for c in d.meta["tool_trace"]] == ["stop"] * 3
    a, env = _agent(reply=lambda m: [_call("city_overview", {})], finish="length", max_steps=2)
    d = a.act(env.observe("police"))
    assert d.meta["length_stops"] == 2 and d.meta["finish_reasons"] == ["length", "length"]


def test_tool_agent_caps_parallel_calls_and_skips_calls_after_end_turn():
    a, env = _agent(reply=lambda m: [_call("city_overview", {}, None)] * 5, max_steps=3)
    d = a.act(env.observe("police"))
    tools = [m for m in d.meta["messages"] if m["role"] == "tool"]
    assert len(d.meta["tool_trace"]) == 3 and len(tools) == 5 and "not run" in tools[-1]["content"]
    assert len({m["tool_call_id"] for m in tools}) == 5
    a, env = _agent(reply=lambda m: [_call("end_turn", {"report": "done"}), _call("city_overview", {})])
    d = a.act(env.observe("police"))
    assert d.meta["finish_reason"] == "end_turn" and [c["tool"] for c in d.meta["tool_trace"]] == ["end_turn"]
    assert len(a.client.calls) == 1 and d.thinking == "done"


def test_tool_agent_messages_ids_and_truncation():
    a, env = _agent(reply=lambda m: [_call("inspect", {"x": 20, "y": 20, "radius": 4}, None),
                                     _call("send_message", {"to": "mayor", "text": "need funds"}, "")]
                    if len(m) == 2 else [_call("end_turn", {"report": "r"}, "x9")])
    d = a.act(env.observe("police"))
    asst = [m for m in d.meta["messages"] if m["role"] == "assistant"]
    assert all(set(m) == {"role", "content", "tool_calls"} for m in asst)
    ids = [c["id"] for m in asst for c in m["tool_calls"]]
    assert ids == ["call_0_0", "call_0_1", "x9"]
    assert ids == [m["tool_call_id"] for m in d.meta["messages"] if m["role"] == "tool"]
    assert all(len(m["content"]) <= 4000 and ": " not in m["content"][:20]
               for m in d.meta["messages"] if m["role"] == "tool")
    assert d.messages == [{"to": "mayor", "text": "need funds"}]
    assert d.meta["finish_reason"] == "end_turn" and d.meta["reasoning"] == "[step 0] secret\n\n[step 1] secret"
    assert d.meta["usage"] == {"prompt_tokens": 10, "completion_tokens": 20}
    env.apply("police", d)
    assert env.council.messages[-1].to == "mayor"


def test_tool_agent_turn_tokens_and_api_errors():
    a, env = _agent(reply=lambda m: [_call("city_overview", {})], tokens=600, turn_tokens=1000)
    d = a.act(env.observe("police"))
    assert d.meta["finish_reason"] == "turn_tokens" and len(a.client.calls) == 2

    def boom(m):
        if len(m) > 2:
            raise LLMError("HTTP 500")
        return [_call("city_overview", {})]
    a, env = _agent(reply=boom)
    d = a.act(env.observe("police"))
    assert d.meta["finish_reason"] == "api_error" and d.parse_error == "api_error: LLMError: HTTP 500"
    assert d.meta["steps"] == 1 and a.usage["api_errors"] == 1


def test_tool_agent_prompt_and_request_settings():
    a, env = _agent(role="mayor", reply=lambda m: [_call("end_turn", {"report": "ok"})], max_tokens=777,
                    extra_body={"top_p": 0.9}, mode="supervised", playbook="Keep a reserve.")
    env.cfg.act_every = 3
    a.bind(env)
    d = a.act(env.observe("mayor"))
    call = a.client.calls[0]
    assert call["max_tokens"] == 777 and call["extra"]["top_p"] == 0.9 and call["extra"]["tools"]
    a.cfg.extra_body = {"tools": [], "top_p": 0.5}                     # extra_body cannot replace the tool schemas
    a.act(env.observe("mayor"))
    assert a.client.calls[-1]["extra"]["tools"] == call["extra"]["tools"] and a.client.calls[-1]["extra"]["top_p"] == 0.5
    system, user = call["messages"][0]["content"], call["messages"][1]["content"]
    assert "Keep a reserve." in system and "every 3 months" in system
    assert "mode: supervised" in user
    assert a.describe() == "fleet:fake" and d.meta["model"] == "fake"


def test_tool_agent_needs_its_city_in_a_region():
    from civitas.core.region import RegionEnv
    a = ToolAgent("mayor", cfg=FleetAgentConfig())
    with pytest.raises(ValueError, match="needs its city"):          # region games: tests/test_fleet_region.py
        a.bind(RegionEnv(GameConfig(seed=0, months=2, region=True)))
    with pytest.raises(RuntimeError, match="bind"):
        ToolAgent("mayor", cfg=FleetAgentConfig()).act({})


def test_fleet_agent_config_merges_fleet_and_role():
    cfg = load_config(None)
    cfg["fleet"] = {"mode": "supervised", "playbook": "fleet book", "model": "m1", "max_steps": 9, "temperature": 0.1}
    cfg["agents"] = {"mayor": {"kind": "fleet", "playbook": "mayor book", "model": "m2", "max_tokens": 500},
                     "police": {"kind": "fleet"}}
    m = fleet_agent_config(cfg, "mayor")
    assert (m.mode, m.fleet_playbook, m.playbook, m.model, m.max_steps, m.max_tokens, m.temperature) == \
        ("supervised", "fleet book", "mayor book", "m2", 9, 500, 0.1)
    p = fleet_agent_config(cfg, "police")
    assert (p.playbook, p.model, p.provider, p.turn_tokens) == ("", "m1", "arcee", 80000)
    cfg["fleet"]["mode"] = "wild"
    with pytest.raises(ValueError, match="mode"):
        fleet_agent_config(cfg, "police")
    cfg["fleet"]["mode"] = "autonomous"
    cfg["fleet"]["extra_body"] = {"top_p": 0.9, "seed": 1}
    cfg["agents"]["mayor"]["extra_body"] = {"top_k": 5}
    assert fleet_agent_config(cfg, "mayor").extra_body == {"top_k": 5}       # replaces, does not merge
    assert fleet_agent_config(cfg, "police").extra_body == {"top_p": 0.9, "seed": 1}
    for k in ("max_steps", "max_tokens", "turn_tokens"):
        bad = {**cfg, "agents": {"police": {"kind": "fleet", k: 0}}}
        with pytest.raises(ValueError, match=k):
            fleet_agent_config(bad, "police")
    bad = {**cfg, "agents": {"police": "fleet"}}
    for fn in (lambda: fleet_agent_config(bad, "police"), lambda: fleet_version(bad)):
        with pytest.raises(ValueError, match=r"agents\.police.*table"):
            fn()
    with pytest.raises(ValueError, match=r"agents\.police.*table"):
        build_agents(bad, ["police"], GameConfig(), scoring_config(bad, GameConfig()))


def test_fleet_version_tracks_fleet_settings():
    v = fleet_version(_fleet_cfg(1))
    assert re.fullmatch(r"[0-9a-f]{12}", v) and fleet_version(_fleet_cfg(1)) == v
    other = _fleet_cfg(1)
    other["game"]["months"] = 99                                     # not part of the fleet
    other["providers"]["fake"]["base_url"] = "http://elsewhere/v1"
    assert fleet_version(other) == v
    other["agents"]["mayor"]["playbook"] = "changed"
    assert fleet_version(other) != v
    other = _fleet_cfg(1)
    other["fleet"]["max_steps"] = 7
    assert fleet_version(other) != v
    other = _fleet_cfg(1)
    other["agents"]["planning"] = {"kind": "scripted"}               # the fleet's line-up changed
    assert fleet_version(other) != v


def test_build_agents_and_signature_for_fleets():
    from civitas.bench.runner import agent_signature
    cfg = load_config("configs/baseline.toml")
    cfg["fleet"] = {"model": "big", "provider": "ollama"}
    cfg["agents"]["mayor"] = {"kind": "fleet"}
    cfg["agents"]["health"] = {"kind": "fleet", "model": "small"}
    game = game_config(cfg, 0)
    agents = build_agents(cfg, list(TURN_ORDER), game, scoring_config(cfg, game))
    assert isinstance(agents["mayor"], ToolAgent) and agents["health"].cfg.model == "small"
    assert agents["mayor"].provider.base_url.startswith("http://localhost:11434")
    sig = agent_signature(cfg, list(TURN_ORDER))
    assert "mayor=fleet:big" in sig and "health=fleet:small" in sig and "planning=scripted" in sig
    assert sig.endswith(" [autonomous]") and sig.count("[") == 1
    assert "[" not in agent_signature(load_config("configs/baseline.toml"), list(TURN_ORDER))
    north = build_agents(cfg, ["north.mayor"], game, scoring_config(cfg, game))["north.mayor"]
    assert (north.role, north.city, north.name) == ("mayor", "north", "north.mayor")
    cfg["agents"]["health"]["mode"] = "supervised"
    with pytest.raises(ValueError, match="mode"):
        build_agents(cfg, list(TURN_ORDER), game, scoring_config(cfg, game))


@pytest.mark.parametrize("path, months", [("configs/fleet_trinity.toml", 120), ("configs/fleet_smoke.toml", 12)])
def test_example_fleet_configs(path, months):
    cfg = load_config(path)
    game = game_config(cfg, 0)
    assert (game.months, game.act_every) == (months, 3)
    agents = build_agents(cfg, list(TURN_ORDER), game, scoring_config(cfg, game))     # no network needed
    assert all(isinstance(a, ToolAgent) for a in agents.values()) and set(agents) == set(TURN_ORDER)
    for role, a in agents.items():
        assert a.cfg.model == "trinity-large-thinking" and a.cfg.mode == "autonomous"
        assert a.provider.api_key_env == "ARCEE_API_KEY"
        assert a.cfg.fleet_playbook.strip() and a.cfg.playbook.strip(), role
        assert "\\" not in a.cfg.fleet_playbook + a.cfg.playbook          # TOML line continuations resolved
    big = load_config("configs/fleet_trinity.toml")
    assert cfg["fleet"] == big["fleet"] and cfg["agents"] == big["agents"]          # the same fleet
    assert fleet_version(cfg) == fleet_version(big)


def test_prompt_command_prints_the_fleet_prompt(capsys):
    from civitas.cli import main
    main(["prompt", "-c", "configs/fleet_smoke.toml", "--role", "police"])
    out = capsys.readouterr().out
    assert "# How to play" in out and "## Police" in out and "# Police" in out and "find_sites" in out
    assert out.index("USER (briefing)") < out.index("TOOLS") and "build_police_station" in out


def test_tool_agent_sanitises_malformed_calls():
    raw = ["", "{bad", {"x": 20, "y": 20}, "[1, 2]"]
    calls = [{"id": f"c{i}", "function": {"name": "inspect" if i else "city_overview", "arguments": a}}
             for i, a in enumerate(raw)] + [{"id": "c9", "function": {"name": "", "arguments": "{}"}}]
    a, env = _agent(reply=lambda m: calls if len(m) == 2 else [_call("end_turn", {"report": "r"})])
    d = a.act(env.observe("police"))
    sent = a.client.calls[1]["messages"][2]["tool_calls"]
    assert [c["function"]["arguments"] for c in sent] == ["{}", "{}", '{"x": 20, "y": 20}', "{}", "{}"]
    assert [c["function"]["name"] for c in sent] == ["city_overview", "inspect", "inspect", "inspect", "invalid_tool"]
    tr = d.meta["tool_trace"]
    assert [c["args"] for c in tr[:5]] == ["", "{bad", '{"x": 20, "y": 20}', "[1, 2]", "{}"]
    assert tr[0]["result"]["ok"] and tr[2]["result"]["ok"]
    assert "not valid JSON" in tr[1]["result"]["error"] and "JSON object" in tr[3]["result"]["error"]
    assert "unknown tool 'invalid_tool'" in tr[4]["result"]["error"]
    assert len(tr[4]["result"]["error"]) <= 300
    assert d.meta["finish_reason"] == "end_turn"


def _xml_call(name, **params):
    ps = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{ps}</function>\n</tool_call>"


class _ScriptedClient(FakeClient):
    """Replies are (content, reasoning, tool_calls[, finish_reason]) tuples, one per model call."""

    def __init__(self, replies):
        super().__init__(None)
        self.replies = iter(replies)

    def chat(self, model, messages, **kw):
        self.calls.append({"messages": [dict(m) for m in messages]})
        content, reasoning, calls, *finish = next(self.replies)
        return {"content": content, "reasoning": reasoning, "tool_calls": calls,
                "finish_reason": finish[0] if finish else "stop", "usage": {}, "latency_s": 0.0}


def _run_scripted(replies, role="police", **cfg):
    a, env = _agent(role=role, **cfg)
    a.client = _ScriptedClient(replies)
    return a, a.act(env.observe(role))


END = ("done", "", [_call("end_turn", {"report": "r"})])


def test_tool_agent_runs_tool_calls_left_in_the_reasoning():
    xml = ("Let me look.\n" + _xml_call("find_sites", purpose="station", near_x=20, near_y=20) + "\n"
           + '<tool_call>{"name": "send_message", "arguments": {"to": "mayor", "text": "hi"}}</tool_call>\n')
    a, d = _run_scripted([("", xml, []), END])
    tr = d.meta["tool_trace"]
    assert [c["tool"] for c in tr] == ["find_sites", "send_message", "end_turn"] and all(c["result"]["ok"] for c in tr)
    assert json.loads(tr[0]["args"]) == {"purpose": "station", "near_x": "20", "near_y": "20"}   # coerced by schema
    assert d.meta["recovered_calls"] == 2 and d.meta["finish_reason"] == "end_turn"
    assert [c["id"] for c in a.client.calls[1]["messages"][2]["tool_calls"]] == ["call_0_0", "call_0_1"]
    assert d.messages == [{"to": "mayor", "text": "hi"}]


def test_text_calls_prefer_content_dedupe_and_skip_native_replies():
    call = _xml_call("city_overview")
    a, d = _run_scripted([(call + "\n" + call, call + "\n" + _xml_call("unlocks"), []), END])
    assert [c["tool"] for c in d.meta["tool_trace"]] == ["city_overview", "end_turn"]   # content only, once
    assert d.meta["recovered_calls"] == 1
    a, d = _run_scripted([("", _xml_call("unlocks"), [_call("city_overview", {})]), END])
    assert [c["tool"] for c in d.meta["tool_trace"]] == ["city_overview", "end_turn"]   # native calls win
    assert d.meta["recovered_calls"] == 0


def test_text_calls_only_the_trailing_run_and_not_truncated_replies():
    drafted = _xml_call("send_message", to="mayor", text="draft") + "\nOn second thought, no.\n"
    a, d = _run_scripted([("", drafted + _xml_call("unlocks") + "\n  ", []), END])
    assert [c["tool"] for c in d.meta["tool_trace"]] == ["unlocks", "end_turn"] and d.messages == []
    a, d = _run_scripted([("", drafted, []), ("", "", []), END])            # nothing after the draft but prose
    assert d.meta["tool_trace"] == [] and d.meta["finish_reason"] == "no_tool_calls"
    a, d = _run_scripted([("", _xml_call("unlocks"), [], "length"), ("", "", []), END])
    assert [c["tool"] for c in d.meta["tool_trace"]] == ["end_turn"] and d.meta["recovered_calls"] == 0


def _nudges(d) -> list[str]:
    """The user messages the loop added after the briefing."""
    return [m["content"] for m in d.meta["messages"][2:] if m["role"] == "user"]


CUT = ("", "reasoning on and on", [], "length")         # a reply cut at max_tokens before any tool call


def test_tool_agent_gives_a_reply_cut_at_max_tokens_its_own_nudge():
    """Smoke v9: planning lost its month-0 turn to two replies cut at max_tokens mid-reasoning, with the one generic
    nudge spent between them. A cut reply without a call gets LENGTH_NUDGE instead, and the generic nudge stays."""
    a, d = _run_scripted([CUT] * LENGTH_NUDGES + [("", "", [_call("city_overview", {})]), END])
    assert [c["tool"] for c in d.meta["tool_trace"]] == ["city_overview", "end_turn"]
    assert d.meta["finish_reason"] == "end_turn" and _nudges(d) == [LENGTH_NUDGE] * LENGTH_NUDGES
    assert d.meta["finish_reasons"] == ["length"] * LENGTH_NUDGES + ["stop", "stop"]
    assert d.meta["length_stops"] == LENGTH_NUDGES and "short" in LENGTH_NUDGE and LENGTH_NUDGES == 2
    a, d = _run_scripted([CUT] * LENGTH_NUDGES + [("", "", []), END])        # then a plain reply without a call
    assert _nudges(d) == [LENGTH_NUDGE] * LENGTH_NUDGES + [NUDGE] and d.meta["finish_reason"] == "end_turn"


def test_tool_agent_ends_the_turn_once_its_nudges_are_spent():
    a, env = _agent(finish="length")                   # every reply cut, none with a call
    d = a.act(env.observe("police"))
    assert _nudges(d) == [LENGTH_NUDGE] * LENGTH_NUDGES + [NUDGE] and len(a.client.calls) == LENGTH_NUDGES + 2
    assert d.meta["finish_reason"] == "no_tool_calls" and d.meta["steps"] == 0
    assert d.meta["length_stops"] == LENGTH_NUDGES + 2
    a, d = _run_scripted([("", "", [])] + [CUT] * (LENGTH_NUDGES + 1))       # generic nudge spent first
    assert _nudges(d) == [NUDGE] + [LENGTH_NUDGE] * LENGTH_NUDGES and d.meta["finish_reason"] == "no_tool_calls"
    assert len(a.client.calls) == LENGTH_NUDGES + 2
    a, d = _run_scripted([("", "", []), CUT, ("", "", [])])                  # still one generic nudge a turn
    assert _nudges(d) == [NUDGE, LENGTH_NUDGE] and d.meta["finish_reason"] == "no_tool_calls"


def test_tool_agent_sends_no_nudge_past_turn_tokens():
    a, env = _agent(finish="length", tokens=600, turn_tokens=1000)
    d = a.act(env.observe("police"))
    assert d.meta["finish_reason"] == "turn_tokens" and len(a.client.calls) == 2 and _nudges(d) == [LENGTH_NUDGE]
    assert d.meta["usage"]["completion_tokens"] == 1200
    a, env = _agent(tokens=1000, turn_tokens=1000)     # the generic nudge too
    d = a.act(env.observe("police"))
    assert d.meta["finish_reason"] == "turn_tokens" and len(a.client.calls) == 1 and _nudges(d) == []
    a, env = _agent(finish="length", tokens=600, turn_tokens=1203)    # completion tokens only (+5 prompt per call)
    d = a.act(env.observe("police"))
    assert d.meta["finish_reason"] == "turn_tokens" and len(a.client.calls) == 3
    assert _nudges(d) == [LENGTH_NUDGE] * 2 and d.meta["usage"] == {"prompt_tokens": 15, "completion_tokens": 1800}


def test_text_calls_are_strings_for_the_schema_and_tolerate_broken_tags():
    a, d = _run_scripted([("", _xml_call("send_message", to="mayor", text="true")
                           + _xml_call("end_turn", report="null"), [])])
    assert d.messages == [{"to": "mayor", "text": "true"}] and d.thinking == "null"
    broken = ("<tool_call>\n<function=city_overview>\n</tool_call>"         # no </function>
              "<tool_call>\n<function=unlocks>\n"                            # never closed
              + _xml_call("end_turn", report="ok"))
    a, d = _run_scripted([("", broken, [])])
    assert [c["tool"] for c in d.meta["tool_trace"]] == ["city_overview", "unlocks", "end_turn"]
    assert d.meta["recovered_calls"] == 3
    a, d = _run_scripted([("", _xml_call("city_overview") + _xml_call("unlocks"), [])], max_steps=1)
    assert len(d.meta["tool_trace"]) == 1 and d.meta["recovered_calls"] == 1   # only calls that ran count


def test_tool_agent_keeps_actions_when_the_client_crashes():
    def reply(m):
        if len(m) > 2:
            raise ConnectionResetError("socket gone")
        return [_call("allocate_budget", {"department": "health", "amount": 1000})]
    a, env = _agent(role="mayor", reply=reply)
    d = a.act(env.observe("mayor"))
    assert d.meta["finish_reason"] == "api_error" and "ConnectionResetError: socket gone" in d.parse_error
    assert len(d.executed) == 1 and d.executed[0]["ok"]
    assert env.apply("mayor", d) == d.executed and env.world.dept_funds["health"] == 1000


# ------------------------------------------------------------------ step budget and repeated calls
def _tool_msgs(d) -> list[dict]:
    return [json.loads(m["content"]) for m in d.meta["messages"] if m["role"] == "tool"]


def test_tool_results_carry_steps_left_and_a_wrap_up_note():
    a, env = _agent(reply=lambda m: [_call("inspect", {"x": 10 + len(m), "y": 20, "radius": 0})], max_steps=6)
    d = a.act(env.observe("police"))
    res = _tool_msgs(d)
    assert [r["steps_left"] for r in res] == [5, 4, 3, 2, 1, 0]
    assert "wrap_up" not in res[0] and "wrap_up" not in res[1]
    assert res[2]["wrap_up"].startswith("3 tool calls left") and "end_turn" in res[2]["wrap_up"]
    assert res[3]["wrap_up"].startswith("2 tool calls left")
    assert res[4]["wrap_up"].startswith("1 tool call left") and "end_turn" in res[4]["wrap_up"]
    assert "turn ends now" in res[5]["wrap_up"]
    assert all("steps_left" not in c["result"] for c in d.meta["tool_trace"])      # the trace stays slim
    a, env = _agent(reply=lambda m: [_call("unlocks", {}), _call("end_turn", {"report": "done"})], max_steps=5)
    d = a.act(env.observe("police"))
    unlocks, end = _tool_msgs(d)
    assert unlocks["steps_left"] == 4 and "wrap_up" not in unlocks
    assert end == {"ok": True, "message": "turn ended"}                             # end_turn's own result: no budget


def test_steps_left_and_notes_fit_in_the_tool_message():
    a, env = _agent(role="planning", reply=lambda m: [_call("map_overview", {}), _call("history", {"months": 12}),
                                                     _call("inspect", {"x": 24, "y": 24, "radius": 4})], max_steps=6)
    d = a.act(env.observe("planning"))
    msgs = [m["content"] for m in d.meta["messages"] if m["role"] == "tool"]
    assert len(msgs) == 6 and all(len(m) <= 4000 for m in msgs)
    res = [json.loads(m) for m in msgs]                                              # nothing cut mid-JSON
    assert res[3]["repeat"] == "same as call 1 (nothing changed since)" and res[3]["wrap_up"]
    assert max(len(m) for m in msgs) > 3000                                         # a big result was in the mix


def test_fit_leaves_room_for_its_truncation_note():
    from civitas.fleet.tools import _fit
    res = _fit({"ok": True, "items": [{"x": i, "y": i} for i in range(500)]}, limit=600, hint="ask for less")
    assert len(json.dumps(res)) <= 600 and "truncated" in res


def test_repeated_look_up_returns_the_earlier_result():
    env = _env()
    box = ToolBox(env, "police")
    first = box.call("inspect", {"x": 20, "y": 20, "radius": 1})
    again = box.call("inspect", '{"y": "20", "radius": 1, "x": 20.0}')              # same after coercion, any order
    assert again == {**first, "repeat": "same as call 1 (nothing changed since)"}
    assert "repeat" not in first
    other = box.call("inspect", {"x": 20, "y": 20})                                   # radius left out: a new call
    assert "repeat" not in other
    assert box.call("unlocks", {})["ok"]
    assert box.call("unlocks", None)["repeat"] == "same as call 4 (nothing changed since)"


def test_repeated_look_up_after_a_successful_action_runs_again():
    env = _env()
    env.world.treasury = 50000
    box = ToolBox(env, "mayor")
    before = box.call("city_overview", {})
    assert box.call("allocate_budget", {"department": "health", "amount": 1000})["ok"]
    after = box.call("city_overview", {})
    assert "repeat" not in after and after != before
    assert box.call("city_overview", {})["repeat"] == "same as call 3 (nothing changed since)"
    plan = box.call("get_plan", {})                                                   # the plan is state too
    assert box.call("update_plan", {"goals": ["grow"]})["ok"]
    assert "repeat" not in box.call("get_plan", {}) and box.call("get_plan", {})["goals"] == ["grow"]
    assert plan["plan"] is None


def test_failed_action_refreshes_history_but_not_other_look_ups():
    env = _env()
    env.world.dept_funds["utilities"] = 0
    box = ToolBox(env, "utilities")
    box.call("history", {"kind": "failure"})
    box.call("find_sites", {"purpose": "wind"})
    assert box.call("build_power_plant", {"kind": "wind", "x": 0, "y": 0})["ok"] is False
    hist = box.call("history", {"kind": "failure"})
    assert "repeat" not in hist and len(hist["entries"]) == 1                        # shows the new failure
    assert box.call("find_sites", {"purpose": "wind"})["repeat"] == "same as call 2 (nothing changed since)"


def test_repeated_failed_action_is_refused():
    env = _env()
    env.world.dept_funds["utilities"] = 0
    box = ToolBox(env, "utilities")
    first = box.call("build_power_plant", {"kind": "wind", "x": 0, "y": 0})
    assert first["ok"] is False
    struct = env.world.struct.copy()
    again = box.call("build_power_plant", '{"y": "0", "x": 0, "kind": "wind"}')
    assert again == {"ok": False, "refused": "repeat",
                     "error": f"this exact call failed at call 1: {first['error']}; change the arguments or do "
                              "something else"}
    assert len(box.executed) == 1 and (env.world.struct == struct).all()
    assert len(FleetState.of(env).history(kind="failure", role="utilities")) == 1   # not charged twice
    moved = box.call("build_power_plant", {"kind": "wind", "x": 1, "y": 0})           # other arguments run
    assert "refused" not in moved and len(box.executed) == 2
    env.apply("utilities", Decision(executed=box.executed))
    assert env.invalid_counts["utilities"] == 2


def test_failed_action_repeated_after_a_success_runs_again():
    env = _env()
    env.world.treasury = 500
    box = ToolBox(env, "mayor")
    big = {"department": "health", "amount": 1000}
    assert box.call("allocate_budget", big)["ok"] is False
    assert box.call("allocate_budget", big)["refused"] == "repeat"
    assert box.call("set_tax", {"sector": "R", "rate": 9})["ok"]
    res = box.call("allocate_budget", big)
    assert res["ok"] is False and "refused" not in res and "treasury only has" in res["error"]
    assert len(box.executed) == 3
    assert len(FleetState.of(env).history(kind="failure", role="mayor")) == 2


def test_refused_repeat_after_the_action_cap_says_end_turn():
    env = _env(max_actions_per_turn=1)
    env.world.treasury = 50000
    box = ToolBox(env, "mayor")
    assert box.call("set_tax", {"sector": "R", "rate": 9})["ok"]
    capped = box.call("set_tax", {"sector": "C", "rate": 9})
    assert capped["ok"] is False and capped["error"].endswith("call end_turn")
    again = box.call("set_tax", {"sector": "C", "rate": 9})
    assert again == {"ok": False, "refused": "repeat", "error": f"this exact call failed at call 2: {capped['error']}"}
    assert "change the arguments" not in again["error"]


def test_repeated_bad_call_is_refused():
    """An identical repeat of a call with bad arguments, bad JSON or an unknown tool (the arguments as sent, keys in
    any order) is refused like a failed action's, until a successful action; the report counts it as refused."""
    from civitas.fleet.agent import _slim
    from civitas.fleet.report import turn_row
    env = _env()
    env.world.treasury = 50000
    box = ToolBox(env, "mayor")
    refused = lambda n, err, tail="; change the arguments or do something else": {
        "ok": False, "refused": "repeat", "error": f"this exact call failed at call {n}: {err}{tail}"}
    trace = []

    def call(name, args=None):
        res = box.call(name, args)
        trace.append({"tool": name, "args": args, "result": _slim(res)})
        return res

    first = call("set_tax", {"sector": "R", "rate": "high"})
    assert first["ok"] is False and first["error"].startswith("argument 'rate' must be") and "refused" not in first
    assert call("set_tax", '{"rate": "high", "sector": "R"}') == refused(1, first["error"])
    other = call("set_tax", {"sector": "C", "rate": "high"})                  # other arguments run
    assert other["ok"] is False and "refused" not in other
    bad_json = call("set_tax", '{"sector": "R", "rate": 9')
    assert "not valid JSON" in bad_json["error"] and "refused" not in bad_json
    assert call("set_tax", ' {"sector": "R", "rate": 9\n') == refused(4, bad_json["error"])   # the text, stripped
    unknown = call("raise_taxes", {})
    assert unknown["error"].startswith("unknown tool 'raise_taxes'") and "refused" not in unknown
    assert call("raise_taxes", None)["refused"] == "repeat" and "refused" not in call("raise_taxes", {"x": 1})
    noop = "there is no noop tool; to do nothing, call end_turn"
    assert call("noop") == {"ok": False, "error": noop} and call("noop", "{}") == refused(9, noop, "")
    assert call("end_turn", {})["error"] == "end_turn needs argument(s) report"
    assert call("end_turn", {}) == refused(11, "end_turn needs argument(s) report")
    assert box.n_calls == 12 and not box.ended and box.executed == []
    row = turn_row({"role": "mayor", "month": 0, "tool_trace": trace})
    assert len(row["tool_errors"]) == 7 and row["refused"] == 5 and row["repeats"] == 0
    assert call("allocate_budget", {"department": "fire", "amount": 10**9})["ok"] is False    # a failed action and
    assert call("unlocks", {})["ok"]                                                          # a look-up keep them
    assert call("set_tax", {"sector": "C", "rate": "high"}) == refused(3, other["error"])
    assert call("set_tax", {"sector": "R", "rate": 9})["ok"]                 # a success clears the refusals
    assert call("set_tax", {"sector": "C", "rate": "high"}) == other


def test_bad_call_sent_as_json_null_is_not_a_call_without_arguments():
    box = ToolBox(_env(), "police")
    null = box.call("end_turn", "null")
    assert null == {"ok": False, "error": "arguments must be a JSON object of named parameters, got NoneType"}
    assert box.call("end_turn", "{}") == {"ok": False, "error": "end_turn needs argument(s) report"}   # not refused
    assert box.call("end_turn", " null ")["error"].startswith("this exact call failed at call 1: arguments must")
    assert box.call("end_turn", None)["error"].startswith("this exact call failed at call 2: end_turn needs")


def test_repeated_look_up_is_a_copy():
    env = _env()
    box = ToolBox(env, "police")
    first = box.call("inspect", {"x": 20, "y": 20, "radius": 1})
    want = copy.deepcopy(first)
    first["tiles"].clear()
    first["ok"] = "changed"
    again = box.call("inspect", {"x": 20, "y": 20, "radius": 1})
    assert again == {**want, "repeat": "same as call 1 (nothing changed since)"}
    again["tiles"].clear()
    assert box.call("inspect", {"x": 20, "y": 20, "radius": 1})["tiles"] == want["tiles"]


def test_turn_control_tools_are_not_deduplicated():
    env = _env()
    box = ToolBox(env, "police")
    for _ in range(2):
        res = box.call("send_message", {"to": "mayor", "text": "need funds"})
        assert res["ok"] and "repeat" not in res
        res = box.call("update_memory", {"lessons": ["roads first"]})
        assert res["ok"] and "repeat" not in res
    assert len(box.messages) == 2


def test_refused_repeats_in_the_agent_trace():
    a, env = _agent(role="mayor", reply=lambda m: [_call("allocate_budget", {"department": "fire", "amount": 10**9}),
                                                   _call("allocate_budget", {"amount": "1000000000",
                                                                             "department": "fire"}),
                                                   _call("unlocks", {}), _call("unlocks", {}),
                                                   _call("end_turn", {"report": "r"})])
    d = a.act(env.observe("mayor"))
    tr = d.meta["tool_trace"]
    assert tr[1]["result"]["refused"] == "repeat" and "failed at call 1" in tr[1]["result"]["error"]
    assert tr[3]["result"]["repeat"] == "same as call 3 (nothing changed since)"
    assert len(d.executed) == 1 and d.meta["steps"] == 5
    res = _tool_msgs(d)
    assert res[1]["steps_left"] == 18 and res[1]["refused"] == "repeat"
    assert env.apply("mayor", d) == d.executed and env.invalid_counts["mayor"] == 1


# ------------------------------------------------------------------ end-of-run report
from civitas.fleet.report import fleet_report, rows_from_trajectories, turn_row


def _rec(role, month, results=(), finish="end_turn", steps=3, tokens=(100, 20), trace=(), length_stops=0):
    return {"role": role, "month": month, "agent": "fleet:fake", "prompt": [{"role": "user", "content": "x" * 999}],
            "results": [{"action": {"type": t}, "ok": ok, "message": msg, "cost": cost} for t, ok, msg, cost in results],
            "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1]}, "steps": steps,
            "finish_reason": finish, "finish_reasons": ["tool_calls"], "length_stops": length_stops, "ended": True,
            "tool_trace": list(trace), "parse_error": None}


def _report_env(months=12, funds=500.0, treasury=20000.0, budget=1000.0):
    env = _env()
    fs = FleetState.of(env)
    fs.update_plan(0, goals=["grow"], budgets={d: budget for d in DEPTS})
    env.world.month = months
    env.world.history = [{"month": m + 1, "population": 100 * m, "treasury": treasury,
                          "dept_funds": {d: funds for d in DEPTS}} for m in range(months)]
    return env


def _clean_rows(months=12):
    return [turn_row(_rec(r, m, [("noop", True, "ok", 10)])) for m in range(months) for r in TURN_ORDER]


def _summary(env):
    return {"run_id": "r1", "seed": 0, "signature": "sig", "months_played": env.world.month, "bankrupt": False,
            "final_scores": {"city": 50.0, **{r: 40.0 for r in TURN_ORDER}}, "avg_city_score": 45.0,
            "final_metrics": {"population": 1234, "treasury": 5678.0}}


def _hints(rep, kind):
    return [h for h in rep["hints"] if h["kind"] == kind]


def test_turn_row_is_slim():
    trace = [{"tool": "inspect", "result": {"ok": False, "error": "bad args"}},
             {"tool": "zone", "result": {"ok": False, "error": "no land", "message": "no land"}},   # a failed action
             {"tool": "zone", "result": {"ok": False, "error": "action limit reached"}},           # refused: tool error
             {"tool": "get_plan", "result": {"ok": False, "error": "no plan", "message": "no plan"}},
             {"tool": "city_overview", "result": {"ok": True}}, {"tool": "inspect"},
             {"tool": "zone", "result": {"ok": False, "error": "this exact call failed at call 2: no land; change "
                                         "the arguments or do something else", "refused": "repeat"}},
             {"tool": "city_overview", "result": {"ok": True, "repeat": "same as call 5 (nothing changed since)"}}]
    row = turn_row(_rec("police", 3, [("build_service", False, "tile (3,4) is occupied", 0)], trace=trace,
                        length_stops=2))
    assert row == {"role": "police", "month": 3, "prompt_tokens": 100, "completion_tokens": 20, "steps": 3,
                   "finish_reason": "end_turn", "length_stops": 2, "ended": True, "parse_error": None,
                   "results": [{"type": "build_service", "ok": False, "message": "tile (3,4) is occupied", "cost": 0}],
                   "tool_errors": ["bad args", "action limit reached", "no plan"], "repeats": 1, "refused": 1,
                   "overrides": 0, "seen": None, "asked": {}}


def test_turn_row_and_rebuild_tolerate_minimal_records(tmp_path):
    row = turn_row({"role": "fire", "month": 2})
    assert row["steps"] == 0 and row["prompt_tokens"] == 0 and row["results"] == [] and row["tool_errors"] == []
    assert row["seen"] is None and row["asked"] == {}
    p = tmp_path / "trajectories.jsonl"
    p.write_text(json.dumps({"role": "fire", "month": 2, "agent": "fleet:x"}) + "\n\n"
                 + json.dumps({"role": "mayor", "month": 2, "agent": "scripted"}) + "\n")
    assert rows_from_trajectories(p) == [row]


def test_fleet_report_on_a_clean_run_has_no_hints():
    env = _report_env()
    env.fleet.update_memory("police", lessons=["check roads first"])
    env.fleet.reports["police"] = "patrolled " * 40
    rep, md = fleet_report(env, _summary(env), _clean_rows())
    assert rep["hints"] == [] and "No coaching hints" in md
    pol = rep["roles"]["police"]
    assert pol == {**pol, "turns": 12, "tool_calls": 36, "actions": 12, "failed": 0, "spent": 120,
                   "prompt_tokens": 1200, "completion_tokens": 240, "limit_turns": 0, "length_stops": 0,
                   "api_errors": 0, "top_failures": [], "lessons": ["check roads first"]}
    assert pol["last_report"].startswith("patrolled") and len(pol["last_report"]) <= 200
    assert rep["roles"]["fire"]["last_report"] is None and pol["last_report"] in md
    assert rep["months"] == rep["months_configured"] == 12 and "12 of 12 months played." in md
    assert rep["scoreboard"] == {"city": 50.0, "roles": {r: 40.0 for r in TURN_ORDER}, "avg_city": 45.0,
                                 "population": 1234, "treasury": 5678.0, "bankrupt": False}
    assert rep["plan"]["goals"] == ["grow"] and rep["plan"]["versions"] == 1
    assert all(r in md for r in TURN_ORDER) and "check roads first" in md and "r1" in md


def test_fleet_report_repeated_failure_hint():
    env = _report_env()
    rows = _clean_rows()
    fails = [("build_service", False, f"tile ({x},{x + 3}) is occupied by road; pick an empty tile", 0)
             for x in (3, 14, 27, 5)]
    rows.append(turn_row(_rec("police", 4, fails + [("build_service", False, "costs $9,000 but the police "
                                                     "budget only has $500; ask the mayor", 0)])))
    rows.append(turn_row(_rec("fire", 4, fails[:2])))            # two alike: below the threshold
    rep, md = fleet_report(env, _summary(env), rows)
    h, = _hints(rep, "repeated_failures")
    assert h["role"] == "police" and h["count"] == 4 and "occupied by road" in h["text"] and h["text"] in md
    assert h["cause"] == "taken land" and h["text"] == (
        "police: 4 failures like \"tile (3,6) is occupied by road; pick an empty tile\" (build_service) - add a "
        "lesson to its playbook: inspect a tile or use find_sites before building or zoning on it")
    pol = rep["roles"]["police"]
    assert pol["failed"] == 5 and pol["top_failures"][0]["count"] == 4
    assert pol["top_failures"] == [{"message": fails[0][2], "count": 4}, {"message": "costs $9,000 but the police "
                                   "budget only has $500; ask the mayor", "count": 1}]


def test_repeated_failures_group_by_role_and_cause():
    """Scripted fleet, seed 0: planning's failures gave 4 hints, 15 for a plaza spot off the road and 7 + 5 + 3 in
    three wordings for zones, parks and a plaza on taken land. One hint per role and cause, with the wordings."""
    env = _report_env()
    fails = ([("zone", False, "no empty land tiles in that area", 0)] * 7
             + [("build_park", False, "no empty land tiles there", 0)] * 5
             + [("build_landmark", False, f"({x},26) is not an empty land tile", 0) for x in (21, 22, 23)]
             + [("build_landmark", False, f"({x},27) does not touch a road; buildings must be next to a road", 0)
                for x in range(11, 26)]
             + [("build_road", False, f"tile ({x},23) is occupied by {s}; bulldoze it or route around", 0)
                for x, s in ((35, "coal_plant"), (30, "residential"))])      # smoke v17: one action, two wordings
    rows = _clean_rows() + [turn_row(_rec("planning", m, fails[m::3])) for m in range(3)]
    rows += [turn_row(_rec("mayor", m, [("allocate_budget", False, f"treasury only has ${m}", 0)])) for m in range(2)]
    rep, _ = fleet_report(env, _summary(env), rows)
    hs = _hints(rep, "repeated_failures")
    assert [(h["role"], h["cause"], h["count"]) for h in hs] == [("planning", "taken land", 17),
                                                                  ("planning", "no road access", 15)]
    assert hs[0]["text"] == (
        "planning: 17 failures on taken land (7x zone \"no empty land tiles in that area\", 5x build_park \"no empty "
        "land tiles there\", 3x build_landmark \"(21,26) is not an empty land tile\", 2x build_road \"tile (35,23) is "
        "occupied by coal_plant; bulldoze it or route around\") - add a lesson to its playbook: inspect a tile or use "
        "find_sites before building or zoning on it")
    assert hs[1]["text"] == (
        "planning: 15 failures like \"(11,27) does not touch a road; buildings must be next to a road\" "
        "(build_landmark) - add a lesson to its playbook: build only beside a connected road (find_sites lists such "
        "sites)")
    assert len(rep["roles"]["planning"]["top_failures"]) == 4           # still one entry per wording (3 or more)


def test_fleet_report_tool_errors_count_towards_repeated_failures():
    env = _report_env()
    trace = [{"tool": "noop", "result": {"ok": False, "error": "there is no noop tool; to do nothing, call end_turn"}}]
    crash = [{"tool": "inspect", "result": {"ok": False, "error": f"{e}: bad radius {n}"}}
             for e, n in (("ValueError", 3), ("TypeError", 40), ("ValueError", 7))]
    rows = _clean_rows() + [turn_row(_rec("health", m, trace=trace)) for m in (1, 2, 3)]
    rows += [turn_row(_rec("fire", 1, trace=crash))]
    rep, _ = fleet_report(env, _summary(env), rows)
    hs = _hints(rep, "repeated_failures")
    assert [(h["role"], h["count"]) for h in hs] == [("fire", 3), ("health", 3)] and "no noop tool" in hs[1]["text"]
    assert rep["roles"]["health"]["tool_errors"] == 3 and rep["roles"]["health"]["failed"] == 0
    assert rep["roles"]["health"]["top_failures"] == [{"message": trace[0]["result"]["error"], "count": 3}]


def test_fleet_report_counts_repeats_apart_from_failures():
    env = _report_env()
    refused = {"tool": "build_service", "result": {"ok": False, "refused": "repeat",
                                                   "error": "this exact call failed at call 1: occupied; change the "
                                                            "arguments or do something else"}}
    seen = {"tool": "find_sites", "result": {"ok": True, "repeat": "same as call 2 (nothing changed since)"}}
    rows = _clean_rows() + [turn_row(_rec("fire", m, [("build_service", False, "occupied", 0)],
                                          trace=[refused, refused, seen])) for m in (1, 2)]
    rep, md = fleet_report(env, _summary(env), rows)
    fire = rep["roles"]["fire"]
    assert fire["failed"] == 2 and fire["tool_errors"] == 0
    assert fire["repeats"] == 2 and fire["refused"] == 4
    assert "repeats (look-ups / refused / overrides)" in md and "| 2 / 4 / 0 |" in md
    assert _hints(rep, "repeated_failures") == []


def test_every_repeated_failure_hint_is_in_top_failures():
    env = _report_env()
    fails = [("zone", False, f"reason {c} for tile ({i},{i})", 0) for c in "abcd" for i in range(3)]
    rep, _ = fleet_report(env, _summary(env), _clean_rows() + [turn_row(_rec("planning", 2, fails))])
    assert len(_hints(rep, "repeated_failures")) == 4 and len(rep["roles"]["planning"]["top_failures"]) == 4


def test_fleet_report_unused_money_hint():
    env = _report_env(funds=1500.0)                  # below 2x the 1000 budget everywhere
    hist = env.world.history
    for m in (3, 4, 5):                               # three months in a row above 2x for police
        hist[m]["dept_funds"] = {**hist[m]["dept_funds"], "police": 2500.0}
    for m in (3, 4, 6):                               # fire: not consecutive
        hist[m]["dept_funds"] = {**hist[m]["dept_funds"], "fire": 2500.0}
    rep, _ = fleet_report(env, _summary(env), _clean_rows())
    h, = _hints(rep, "unused_money")
    assert h["role"] == "police" and h["months"] == 3 and "$2,500" in h["text"] and "$1,000" in h["text"]
    del hist[4]                                       # a missing month breaks the run
    rep, _ = fleet_report(env, _summary(env), _clean_rows())
    assert _hints(rep, "unused_money") == []


def test_unused_money_with_a_zero_budget_needs_the_floor():
    env = _report_env(funds=900.0, budget=0.0)
    rep, _ = fleet_report(env, _summary(env), _clean_rows())
    assert _hints(rep, "unused_money") == []
    env = _report_env(funds=1200.0, budget=0.0)
    rep, _ = fleet_report(env, _summary(env), _clean_rows())
    assert len(_hints(rep, "unused_money")) == len(DEPTS)


def test_unused_money_uses_the_plan_in_effect_and_skips_depts_without_a_target():
    env = _env()
    fs = FleetState.of(env)
    fs.update_plan(0, budgets={"police": 1000})
    fs.update_plan(5, budgets={"police": 5000})      # later version raises the target: no longer unused
    env.world.month = 10
    env.world.history = [{"month": m + 1, "treasury": 100.0,
                          "dept_funds": {d: 3000.0 for d in DEPTS}} for m in range(10)]
    rep, _ = fleet_report(env, _summary(env), _clean_rows(10))
    h, = _hints(rep, "unused_money")                 # months 1-5 under the 1000 target; no other dept has one
    assert h["role"] == "police" and h["months"] == 5


def _idle_hints(turns: dict, months: int = 12):
    env = _report_env(months=months)
    rows = [turn_row(_rec("fire", m, [("build_service", ok, "ok" if ok else "nope", 0)])) for m, ok in turns.items()]
    rep, _ = fleet_report(env, _summary(env), rows)
    return _hints(rep, "idle")


def test_fleet_report_idle_department_hint():
    h, = _idle_hints({m: not 1 <= m <= 6 for m in range(12)})            # 6 idle turns, next turn in month 7
    assert h["role"] == "fire" and (h["turns"], h["months"]) == (6, 6) and "6 turns / 6 months" in h["text"]
    assert "months 1-6" in h["text"]
    assert _idle_hints({m: not 1 <= m <= 5 for m in range(12)}) == []     # 5 months
    h, = _idle_hints({m: m < 6 for m in range(12)})                       # idle until the end of the run
    assert (h["turns"], h["months"], h["first"], h["last"]) == (6, 6, 6, 11)
    h, = _idle_hints({0: True, 3: False, 6: False, 9: True})              # act_every=3: two idle turns = 6 months
    assert (h["turns"], h["months"]) == (2, 6)
    assert _idle_hints({0: False, 7: True}, months=14) == []              # act_every=7: one failed turn
    assert len(_idle_hints({0: False, 7: False}, months=14)) == 1


def test_idle_hint_ignores_the_mayor():
    env = _report_env()
    rows = [turn_row(_rec("mayor", m, [("set_tax", False, "nope", 0)])) for m in range(12)]
    rep, _ = fleet_report(env, _summary(env), rows)
    assert _hints(rep, "idle") == []


def _transport_idle(funds, plans=(), turns=(0, 3, 6, 9), months=12, kind="idle"):
    """Hints of `kind` for a transport department that never acts and whose turns carry no briefing (so the history
    log gives its funds); `funds` maps history month -> its funds (default 0), `plans` are (month, budgets) plan
    versions."""
    env = _env()
    fs = FleetState.of(env)
    fs.update_plan(0, goals=["grow"])
    for m, b in plans:
        fs.update_plan(m, budgets=b)
    env.world.month = months
    env.world.history = [{"month": m, "treasury": 100.0, "dept_funds": {**{d: 500.0 for d in DEPTS},
                                                                       "transport": funds.get(m, 0.0)}}
                         for m in range(1, months + 1)]
    rows = [turn_row(_rec("transport", m, trace=[{"tool": "inspect", "result": {"ok": True}}])) for m in turns]
    rep, _ = fleet_report(env, _summary(env), rows)
    return _hints(rep, kind)


def test_idle_hint_skips_a_department_without_money_or_a_target():
    assert _transport_idle({}) == []                                      # $0 and no plan budget all run
    assert _transport_idle({}, kind="starved") == []
    assert _transport_idle({}, plans=[(0, {"transport": 0})]) == []       # a zero budget counts as none
    assert _transport_idle({m: 0.6 for m in range(13)}) == []             # under $1
    h, = _transport_idle({m: 500.0 for m in range(13)})                   # money but no target: idle
    assert (h["turns"], h["months"]) == (4, 12) and "mayor" not in h["text"]
    plan = [(0, {"transport": 1000})]
    assert _transport_idle({}, plans=plan) == []                          # a target but no money: the mayor's
    h, = _transport_idle({}, plans=plan, kind="starved")
    assert h["role"] == "mayor" and h["depts"] == [{"role": "transport", "turns": 4, "months": 12, "first": 0,
                                                     "last": 11, "building": None, "price": None, "asked": 0}]
    assert h["text"] == ("mayor: left transport at $0 against its $1,000 target for 4 turns / 12 months (months "
                         "0-11) - fund the first buildings departments ask for, or tell them why not")
    assert _transport_idle({4: 500.0}, plans=plan) == []                  # money at the month-3 turn splits the run
    h, = _transport_idle({4: 500.0}, plans=plan, kind="starved")
    assert (h["depts"][0]["turns"], h["depts"][0]["first"], h["depts"][0]["last"]) == (2, 6, 11)


def test_unfunded_turns_split_idle_runs_and_read_the_month_after_the_turn():
    # a turn in month t ends in history entry t + 1: the mayor's month-6 allocation shows there, not in entry 6
    h, = _transport_idle({m: 5000.0 for m in range(7, 13)})
    assert (h["turns"], h["first"], h["last"]) == (2, 6, 11)
    assert _transport_idle({m: 5000.0 for m in range(8, 13)}) == []       # funded after the month-6 turn: one idle turn
    # the plan saved in month 6 (before the departments' turns) is in force for the month-6 turn
    h, = _transport_idle({}, plans=[(6, {"transport": 2000})], kind="starved")
    d, = h["depts"]
    assert (d["turns"], d["first"], d["last"]) == (2, 6, 11) and "$2,000 target" in h["text"]
    assert _transport_idle({4: 500.0, 10: 500.0}) == []                   # idle turns at 3 and 9, blocked between
    assert _transport_idle({}, turns=range(12)) == []


def _briefed(role, month, funds, ready, metrics=(), target=3000.0, results=(), asks=()):
    """A fleet turn row whose briefing showed `funds` (and the plan `target`), these Key metrics and Ready sites lines
    (without their "- "), and which sent the mayor `asks`."""
    from civitas.fleet.fmt import money
    text = "\n".join([f"# {role}: month {month}", "", "## Your budget",
                      f"- {money(funds)} in department funds"
                      + (f" (plan target {money(target)})" if target is not None else "") + "; spent last month $0",
                      "", "## Key metrics", *(f"- {m}" for m in metrics), "", "## Since your last turn (month 0)",
                      "- failed actions: none", "", "## Ready sites", *(f"- {r}" for r in ready),
                      "find_sites <purpose> lists up to 6 with details", "", "## Your memory", "- empty"])
    rec = _rec(role, month, results)
    rec["prompt"] = [{"role": "system", "content": "(system prompt: meta.json fleet_prompts.x)"},
                     {"role": "user", "content": text}]
    rec["messages"] = [{"to": "mayor", "text": a} for a in asks]
    return turn_row(rec)


STATION = "station: 5 sites, best (20,20), need 270; $2,500"
JAIL = "jail: locked (unlocks at 1,000 residents)"


def _turn_hints(rows, kind):
    env = _report_env()
    return _hints(fleet_report(env, _summary(env), rows)[0], kind)


def test_turn_row_reads_the_briefing():
    """The idle rule's view of a turn: the briefing's funds, Ready sites (price, "- you have", blocked for land or
    nothing to build) and the Key metrics that say whether a building is needed, plus what it asked the mayor for."""
    from civitas.fleet.tools import role_obs, site_lists
    row = _briefed("utilities", 3, 12500, ["coal_or_gas: 6 sites, best (0,23), 0 homes within 5; $4,000",
                                           "wind: 6 sites, best (18,25), wind 1.38; $12.5k - you have $12.5k",
                                           "pump: none - riverbank beside roads all taken: ask planning to free a lot "
                                           "near (25,24)", "sewage: none - no river on this map",
                                           "near_homes: 2 sites, same list as wind", "riverbank_2x2: locked (unlocks "
                                           "at 10,000 residents)", "leave free (funded): x (1,1): zone around them"],
                   ["power: supply 1.2k / demand 58, coverage 100%", "water: n/a", "river pollution: 3.6%"],
                   target=None, asks=["Utilities needs $2,500 to add power", "and $1.5k"])
    assert row["seen"] == {"funds": 12500.0, "target": None, "metrics": {"power": [1200.0, 58.0, 100.0]},
                           "sites": {"coal_or_gas": {"state": "ok", "price": 4000.0},
                                     "wind": {"state": "short", "price": 12500.0},
                                     "pump": {"state": "blocked", "why": "riverbank beside roads all taken"},
                                     "sewage": {"state": "none"}, "near_homes": {"state": "ready"},
                                     "riverbank_2x2": {"state": "locked"}}}
    assert row["asked"] == {"mayor": 2500.0}
    assert turn_row({**_rec("mayor", 0), "prompt": [{"role": "user", "content": "## City\n- x"}]})["seen"] is None
    odd = turn_row({**_rec("police", 0), "messages": [{"to": "mayor", "text": "costs $, or $1,500, or $x"}],
                    "prompt": [{"role": "user", "content": ["not text"]}]})
    assert odd["seen"] is None and odd["asked"] == {"mayor": 1500.0}
    assert turn_row({**_rec("police", 0), "messages": "not a list of messages"})["asked"] == {}   # never raises
    env = _env()                                       # a real briefing, then every roadside lot taken
    env.world.dept_funds["police"] = 1500
    row = turn_row({**_rec("police", 0), "prompt": [{"role": "user", "content": briefing(env, "police", mark=False)}]})
    assert row["seen"]["funds"] == 1500 and row["seen"]["metrics"] == {"police coverage": 0.0}
    for fill in (False, True):                         # the parsed states follow SiteList.blocked, not the words
        if fill:
            _fill_roadside(env.world)
        for d in DEPTS[1:]:
            lists = site_lists(env.world, d, role_obs(env, d))
            seen = turn_row({**_rec(d, 0), "prompt": [{"role": "user", "content": briefing(env, d, mark=False)}]})
            for p, lst in lists.items():
                assert (seen["seen"]["sites"][p]["state"] == "blocked") == lst.blocked, (d, p, lst)


def test_idle_needs_money_for_a_ready_site_it_needs():
    """Scripted fleet, seeds 0-4: the idle rule fired for 6-7 departments a run, lumping departments waiting for money
    they asked for, departments with nothing left to do (fire at 83-98% coverage) and the real case: money and a ready
    site but no build. Only that case is an idle hint now, with what the briefing showed."""
    rows = [_briefed("police", m, 3000, [STATION, JAIL], ["police coverage: 30%"],
                     asks=["Police needs $8,300 to upgrade a police station"] * (m > 3)) for m in (0, 3, 6, 9)]
    h, = _turn_hints(rows, "idle")
    assert h["role"] == "police" and (h["turns"], h["months"], h["first"], h["last"]) == (4, 12, 0, 11)
    assert h["text"] == ("police: 4 turns / 12 months without a build while it could pay for a ready site (months "
                         "0-11): it held $3,000, enough for its $2,500 police station; police coverage 30%; asked the "
                         "mayor for up to $8,300 in 2 of them - check its playbook and why it didn't build")
    for ready, cov in (([STATION], "80%"),                               # covered: nothing needed
                       (["station: none - nothing needs it right now"], "30%"),
                       ([JAIL, "station: 5 sites, best (20,20); $2,500 - you have $2,000"], "30%")):
        rows = [_briefed("police", m, 2000, ready, [f"police coverage: {cov}"]) for m in (0, 3, 6, 9)]
        assert _turn_hints(rows, "idle") == [], ready
    rows = [_briefed("fire", m, 2000, ["station: 5 sites, best (1,1); $2,500 - you have $2,000",
                                       "shelter: 3 sites, best (2,2); $1,500"], ["fire coverage: 30%"])
            for m in (0, 3, 6)]                                          # a shelter only: short of the station
    assert _turn_hints(rows, "idle") == [] and len(_turn_hints(rows, "starved")) == 1
    rows = [_briefed("police", m, 3000, [STATION], ["police coverage: 30%"],
                     results=[("build_police_station", False, "tile (20,20) is occupied by road", 0)])
            for m in (0, 3, 6)]
    assert _turn_hints(rows, "idle")[0]["text"].endswith("; 3 failed actions - check its playbook and why it didn't "
                                                         "build")


def test_utilities_and_transport_are_idle_only_while_something_needs_building():
    """Utilities' lists price a turbine with power at a fifth of supply (scripted seed 0, months 3-6: $3,000 held,
    read idle); transport's ranks congested stretches with no road congested. A supply line counts from 90% used (the
    leave-free line's BANK_BUSY) or under 80% coverage, roads once any is congested."""
    wind = "wind: 6 sites, best (18,25), wind 1.38; $2,500"
    idle = lambda role, ready, metrics: _turn_hints([_briefed(role, m, 3000, ready, metrics) for m in (0, 3, 6)],
                                                    "idle")
    assert idle("utilities", [wind], ["power: supply 250 / demand 58, coverage 100%"]) == []
    assert len(idle("utilities", [wind], ["power: supply 111 / demand 103, coverage 100%"])) == 1
    assert len(idle("utilities", [wind], ["power: supply 250 / demand 58, coverage 70%"])) == 1
    assert "power demand 93% of supply" in idle("utilities", [wind], ["power: supply 111 / demand 103, coverage "
                                                                     "100%"])[0]["text"]
    road = "congested_roads: 6 sites, best (3,24)-(9,24), 7 tiles; $30/tile"
    assert idle("transport", [road], ["congested road: 0%"]) == []
    assert "its $30 avenue tile; congested road 1.2%" in idle("transport", [road], ["congested road: 1.2%"])[0]["text"]
    assert len(idle("transport", ["bus_stops: 4 sites, best (9,24); $1,500"], ["congested road: 0%"])) == 1


def test_departments_short_of_their_first_building_make_one_mayor_hint():
    """Scripted fleet, seed 0: police held $1,494-2,436 against a $2,500 station in months 6-33 and asked every turn,
    and read idle. A department short of its cheapest ready site is the mayor's to fund: one hint for all of them."""
    short = lambda p, funds: f"{p}: 5 sites, best (20,20); $2,500 - you have ${funds:,}"
    rows = [_briefed("police", m, 1494 + m, [short("station", 1494 + m), JAIL], ["police coverage: 0%"],
                     asks=["Police needs $2,500 for a new building"] * bool(m)) for m in range(0, 12, 3)]
    rows += [_briefed("education", m, 0, [short("school", 0)], ["school coverage: 0%"]) for m in (3, 6, 9)]
    rows += [_briefed("fire", 9, 0, [short("station", 0)], ["fire coverage: 0%"])]       # one turn: below the rule
    rows += [_briefed("health", m, 0, [short("clinic", 0)], ["health coverage: 90%"]) for m in (3, 6, 9)]  # covered
    assert _turn_hints(rows, "idle") == []
    h, = _turn_hints(rows, "starved")
    assert h["role"] == "mayor" and [(d["role"], d["turns"], d["asked"]) for d in h["depts"]] == [
        ("police", 4, 3), ("education", 3, 0)]
    assert h["text"] == ("mayor: left police short of its $2,500 police station for 4 turns / 12 months (months 0-11; "
                         "it held $1,494-1,503 and asked 3 times); education short of its $2,500 elementary school for "
                         "3 turns / 9 months (months 3-11; it held $0 and never asked) - fund the first buildings "
                         "departments ask for, or tell them why not")
    rows = [_briefed(d, m, 0, [short(p, 0)], [f"{c} coverage: 0%"]) for m in (3, 6)
            for d, p, c in (("police", "station", "police"), ("fire", "station", "fire"),
                            ("health", "clinic", "health"), ("education", "school", "school"))]
    rows += [_briefed("education", 0, 0, [short("school", 0)], ["school coverage: 0%"])]
    h, = _turn_hints(rows, "starved")                  # the longest runs in full: education's 12 months first
    assert [d["role"] for d in h["depts"]] == ["education", "police", "fire", "health"]
    assert h["text"].startswith("mayor: left education short of its $2,500 elementary school for 3 turns / 12 months")
    assert h["text"].count(" short of its ") == 3 and "; and 1 more (health) - fund" in h["text"]


def test_departments_blocked_for_land_make_one_planning_hint():
    """Smoke v17 seed 0: planning zoned every roadside lot by month 6, and police, fire, health and education each read
    idle holding their first building's money. No lot is planning's to fix: one hint for all of them. Utilities'
    blocked pump doesn't count with water at a fifth of supply."""
    blocked = "station: none - no free roadside land: ask planning for a lot"
    jail = "jail: none - no roadside lot away from homes (3+ within 4 tiles): ask planning for one at the town edge"
    rows = [_briefed("police", m, 2500, [blocked, jail if m > 3 else JAIL], ["police coverage: 23%"])
            for m in (3, 6, 9)]                        # the cheapest blocked building is named, as planning's line does
    rows += [_briefed("utilities", m, 4800, [
        "pump: none - riverbank beside roads all taken: ask planning to free a lot near (25,24)",
        "sewage: none - riverbank beside roads all taken: ask planning to free a lot near (25,24)"],
        ["water: supply 250 / demand 50, coverage 100%", "sewage: supply 0 / demand 17, coverage 0%"]) for m in (3, 6, 9)]
    assert _turn_hints(rows, "idle") == [] and _turn_hints(rows, "starved") == []
    h, = _turn_hints(rows, "no_land")
    assert h["role"] == "planning" and [d["role"] for d in h["depts"]] == ["utilities", "police"]
    assert h["text"] == ("planning: left utilities without a lot for its sewage outlet for 3 turns / 9 months (months "
                         "3-11: riverbank beside roads all taken; it held $4,800); police without a lot for its police "
                         "station for 3 turns / 9 months (months 3-11: no free roadside land; it held $2,500) - leave "
                         "or free a roadside lot for each, or extend a road into open land")


def test_planning_is_idle_with_money_and_starved_without():
    """Planning's lists empty out when the roadside is zoned (smoke v17 seed 0, months 6-9: $1,000-6,000 held, no
    action), but a road into open land is always there to build: with money and no successful action it is idle."""
    full = ["zone_R: none - no free roadside strip: extend the roads (find_sites road_extension)",
            "road_extension: none - no straight extension from a dead end: build_road from any road into free land"]
    h, = _turn_hints([_briefed("planning", m, 1000 + 5000 * (m > 6), full) for m in (6, 9)], "idle")
    assert h["text"] == ("planning: 2 turns / 6 months without a successful action (months 6-11): it held "
                         "$1,000-6,000 - check its playbook and whether it is blocked (funds, unlocks, failures)")
    h, = _turn_hints([_briefed("planning", m, 0, full) for m in (6, 9)], "starved")
    assert h["depts"][0]["role"] == "planning" and "planning at $0 against its $3,000 target" in h["text"]


def test_planning_is_starved_short_of_its_road_and_idle_with_it():
    """Live 120-month run, seed 0, months 9-12: planning at $0 read "road_extension: 1 site, best (24,20)-(24,5), opens
    71 land tiles" and failed on its $150 twice. With the road priced, its turns read by that price (a branch road's
    line too), not by the funds alone ($1+ was idle): the road and zoning the land it opens (smoke v18 seed 0, month 9:
    the mayor moved the $50 road alone), and a road-only price in older briefings."""
    from civitas.fleet.report import _site_state
    zone = "zone_R: none - no free roadside strip: extend the roads (find_sites road_extension)"
    ext = "road_extension: 1 site, best (24,20)-(24,5), opens 75 land tiles; $150 road + $375 zoning"
    branch = ("road_extension: none - no dead end; branch: dezone (13,22)-(13,23) (empty lots, free), then build_road "
              "(13,23)-(13,9), opens 75 land tiles; $150 road + $375 zoning")
    assert _briefed("planning", 9, 0, [zone, branch + " - you have $0"])["seen"]["sites"]["road_extension"] == {
        "state": "short", "price": 525.0}
    assert _site_state("1 site, best (24,20)-(24,5), opens 71 land tiles; $150 - you have $0") == {
        "state": "short", "price": 150.0}                                    # an older briefing: the road alone
    assert _site_state("1 site, best (24,20)-(24,5), opens 999 land tiles; $1,395 road + $5k zoning") == {
        "state": "ok", "price": 6395.0}
    for road in (ext, branch):
        rows = [_briefed("planning", m, 200, [zone, road + " - you have $200"], asks=["We need $525"]) for m in (6, 9)]
        assert _turn_hints(rows, "idle") == [], road
        h, = _turn_hints(rows, "starved")
        assert h["text"].startswith("mayor: left planning short of its $525 road into open land and zoning it for 2 "
                                    "turns / 6 months (months 6-11; it held $200 and asked 2 times)"), h["text"]
    h, = _turn_hints([_briefed("planning", m, 1000, [zone, ext]) for m in (6, 9)], "idle")
    assert "enough for its $525 road into open land and zoning it" in h["text"], h["text"]


def test_fleet_report_negative_treasury_hint():
    env = _report_env()
    env.world.history[4]["treasury"] = -2500.0
    env.world.history[7]["treasury"] = -100.0
    rep, _ = fleet_report(env, _summary(env), _clean_rows())
    h, = _hints(rep, "negative_treasury")
    assert h["months"] == 2 and "-$2,500" in h["text"] and "month 5" in h["text"]
    assert h["role"] == "city" and h["text"].startswith("city treasury") and "mayor:" in h["text"]
    rep, _ = fleet_report(env, _summary(env), [r for r in _clean_rows() if r["role"] != "mayor"])
    h, = _hints(rep, "negative_treasury")
    assert "mayor" not in h["text"]


def test_fleet_report_limit_hint():
    env = _report_env()
    rows = [r for r in _clean_rows() if r["role"] not in ("police", "fire")]
    finishes = ["max_steps", "turn_tokens", "end_turn", "end_turn", "end_turn", "end_turn", "end_turn", "end_turn",
                "end_turn", "end_turn", "end_turn", "end_turn"]
    rows += [turn_row(_rec("police", m, [("x", True, "ok", 0)], finish=f)) for m, f in enumerate(finishes)]
    rows += [turn_row(_rec("police", 0, [("x", True, "ok", 0)], finish="max_steps"))]             # 3 of 13 > 20%
    rows += [turn_row(_rec("fire", m, [("x", True, "ok", 0)], finish="api_error" if m < 6 else "end_turn",
                           length_stops=1)) for m in range(12)]                                 # not limit turns
    rep, _ = fleet_report(env, _summary(env), rows)
    h, = _hints(rep, "limit")
    assert h["role"] == "police" and "3 of 13" in h["text"]
    assert rep["roles"]["police"]["limit_turns"] == 3
    assert rep["roles"]["fire"]["api_errors"] == 6 and rep["roles"]["fire"]["length_stops"] == 12


def test_limit_hint_threshold():
    env = _report_env()
    rows = [turn_row(_rec("police", m, [("x", True, "ok", 0)], finish="max_steps" if m < 2 else "end_turn"))
            for m in range(10)]                                                        # exactly 20%
    rep, _ = fleet_report(env, _summary(env), rows)
    assert _hints(rep, "limit") == []
    rows.append(turn_row(_rec("police", 10, [("x", True, "ok", 0)], finish="turn_tokens")))   # 3 of 11
    rep, _ = fleet_report(env, _summary(env), rows)
    assert len(_hints(rep, "limit")) == 1


# ------------------------------------------------------------------ review fixes: launcher, leaderboard, export
def test_history_accepts_kind_all_and_plan_budgets_are_not_quarterly():
    env = _env()
    box = ToolBox(env, "mayor")
    FleetState.of(env).record(0, "police", "report", text="hello")
    everything = box.call("history", {})
    res = box.call("history", {"kind": "all"})
    assert res["ok"] and res["entries"] == everything["entries"] and res["entries"]
    assert not box.call("history", {"kind": "nonsense"})["ok"]
    desc = {t["function"]["name"]: t["function"]["description"] for t in box.schemas()}
    assert "quarter" not in desc["update_plan"]


def test_tool_agent_response_is_the_report_or_the_last_reply():
    a, env = _agent(reply=lambda m: [_call("end_turn", {"report": "all quiet"})])
    d = a.act(env.observe("police"))
    assert d.meta["messages"][-1]["role"] == "tool" and d.meta["response"] == "all quiet"
    a, env = _agent(content="nothing to add")                   # never calls a tool, never ends
    assert a.act(env.observe("police")).meta["response"] == "nothing to add"
    a, env = _agent(content="")
    assert a.act(env.observe("police")).meta["response"] == ""


def test_tool_agent_turns_a_briefing_crash_into_a_no_op(monkeypatch):
    from civitas.fleet import agent as agent_mod

    def boom(*a, **k):
        raise KeyError("population")
    monkeypatch.setattr(agent_mod, "briefing", boom)
    a, env = _agent(reply=lambda m: [_call("end_turn", {"report": "x"})])
    d = a.act(env.observe("police"))
    assert a.client.calls == [] and d.executed == [] and d.messages == []
    assert d.parse_error == "briefing_error: KeyError: 'population'"
    assert d.meta["finish_reason"] == "briefing_error" and d.meta["steps"] == 0 and d.meta["tool_trace"] == []
    assert env.apply("police", d) == []
    rec = {"role": "police", "month": 0, "agent": "fleet:fake", "results": [], "parse_error": d.parse_error,
           **{k: d.meta[k] for k in ("usage", "steps", "finish_reason", "length_stops", "ended", "tool_trace")}}
    renv = _report_env()
    rep, md = fleet_report(renv, _summary(renv), [turn_row(rec)])
    assert rep["roles"]["police"]["api_errors"] == 1 and "API / briefing errors" in md


def test_tool_message_and_report_limits_keep_their_values():
    from civitas.fleet import agent as agent_mod, briefing as briefing_mod, fmt, report as report_mod, tools
    assert fmt.TOOL_MESSAGE_CHARS == 4000 and tools.RESULT_CHARS == 3800 and tools.REPORT_MAX_CHARS == 1000
    assert fmt.REPORT_PREVIEW_CHARS == 200
    for mod in (agent_mod, briefing_mod, report_mod, tools):
        assert not hasattr(mod, "REPORT_CHARS")
    assert not hasattr(agent_mod, "RESULT_CHARS")


def test_launcher_accepts_fleet_specs_and_lists_the_fleet_config(tmp_path):
    from civitas.server import AGENT_SPEC, Launcher
    for ok in ("fleet", "fleet:arcee/trinity-large-thinking", "fleet:big", "llm:arcee/x", "scripted"):
        assert AGENT_SPEC.match(ok), ok
    for bad in ("fleet:", "fleet:x; rm -rf /", "fleetx", "remote"):
        assert not AGENT_SPEC.match(bad), bad
    cfgs = {c["file"]: c for c in Launcher(tmp_path, Path("configs")).configs()}
    fleet = cfgs["fleet_smoke.toml"]
    assert all(a["kind"] == "fleet" and a["model"] == "trinity-large-thinking" and a["fleet"]["max_steps"] == 20
               for a in fleet["agents"].values()) and not fleet["region"]
    base = cfgs["baseline.toml"]["agents"]["police"]
    assert base["kind"] == "scripted" and base["fleet"]["model"] == FleetAgentConfig().model
    assert cfgs["region.toml"]["region"]
    # what the launcher sends for a changed fleet model, as the CLI parses it: playbooks stay
    cfg = apply_cli_overrides(load_config("configs/fleet_smoke.toml"), ["police=fleet:ollama/small"])
    fc = fleet_agent_config(cfg, "police")
    assert (fc.provider, fc.model) == ("ollama", "small") and fc.playbook.strip()


def _summary_file(d, sig, city, fv=None):
    d.mkdir()
    s = {"name": "x", "signature": sig, "agents": {"mayor": "m"}, "final_scores": {"city": city, "mayor": city},
         "avg_city_score": city, "agent_stats": {"mayor": {"invalid_rate": 0.0}},
         "final_metrics": {"population": 100}, "bankrupt": False, **({"fleet_version": fv} if fv else {})}
    (d / "summary.json").write_text(json.dumps(s))


def test_leaderboard_groups_fleets_by_version(tmp_path, capsys):
    from civitas.bench.runner import leaderboard, print_table
    sig = "mayor=fleet:trinity-large-thinking planning=fleet:trinity-large-thinking utilities=fleet:x [supervised]"
    _summary_file(tmp_path / "a", sig, 50, "aaaaaaaaaaaa")
    _summary_file(tmp_path / "b", sig, 52, "aaaaaaaaaaaa")
    _summary_file(tmp_path / "c", sig, 40, "bbbbbbbbbbbb")
    _summary_file(tmp_path / "d", "mayor=scripted", 70)
    _summary_file(tmp_path / "e", "mayor=scripted", 72)
    rows = leaderboard(tmp_path)
    assert [(r["n"], r["fleet_version"]) for r in rows] == [(2, None), (2, "aaaaaaaaaaaa"), (1, "bbbbbbbbbbbb")]
    print_table(rows)
    out = capsys.readouterr().out.splitlines()
    assert "fleet" in out[0] and sum("[supervised]" in ln for ln in out) == 2
    assert any("aaaaaaaaaaaa" in ln for ln in out) and len({len(ln) for ln in out[2:]}) == 1
    print_table([r for r in rows if r["fleet_version"] is None])
    assert "fleet" not in capsys.readouterr().out.splitlines()[0]


def test_fleet_runs_record_fleet_version_in_the_summary(tmp_path):
    from civitas.bench import runner
    with _fake_llm() as srv:
        cfg = _fleet_cfg(srv.server_port, months=1)
        s = runner.run_episode(cfg, 0, tmp_path, verbose=False)
    assert s["fleet_version"] == fleet_version(cfg)
    assert json.loads((tmp_path / s["run_id"] / "summary.json").read_text())["fleet_version"] == s["fleet_version"]


def test_export_sft_skips_fleet_turns(tmp_path):
    import subprocess
    import sys
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({"months": 12}))
    llm = {"run_id": "r1", "month": 0, "role": "police", "agent": "llm:x", "reward": 1.0, "results": [],
           "thinking": "t", "actions": [], "messages": [],
           "prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]}
    fleet = {**llm, "agent": "fleet:x", "prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "b"},
                                                    {"role": "assistant", "content": ""}, {"role": "tool", "content": "{}"}]}
    (run / "trajectories.jsonl").write_text("\n".join(json.dumps(r) for r in (llm, fleet, fleet)) + "\n")
    out = tmp_path / "sft.jsonl"
    res = subprocess.run([sys.executable, "examples/export_sft.py", str(tmp_path / "runs"), "--out", str(out)],
                         capture_output=True, text=True, check=True)
    assert len(out.read_text().splitlines()) == 1 and "2 fleet turns skipped" in res.stdout


# --- transport: the first bus line before avenues, one line through every stop, another only near full
# (tools.py: first_bus_line, bus_room, ToolBox._bus_line_refusal / _stops_hint; briefing.py: _bus_first, _bus_riders)
def _street(w) -> dict:
    """An upgrade_road call for one street tile on land: $30 to an avenue."""
    ys, xs = np.nonzero((w.struct == C.ROAD) & (w.road_class == 1) & (w.terrain != C.WATER))
    x, y = int(xs[0]), int(ys[0])
    return {"x1": x, "y1": y, "x2": x, "y2": y, "to": "avenue"}


def test_the_first_bus_line_comes_before_avenues(open_env):
    """120-month LLM runs: transit 0.00 (no bus line in 120 months) and 0.28; seed 0, month 78: of $2,000 transport
    put $690 into avenues, then create_bus_line failed at $1,310. While the city has no bus line and its first is
    ready, an upgrade_road that leaves transport short of it is refused (confirm: true runs it); the briefing says
    the line comes first, and a failed create_bus_line names the suggested stops as one call."""
    import civitas.fleet.tools as T
    env = copy.deepcopy(open_env)
    _no_spending(env)
    w = env.world
    assert not w.bus_lines
    lst = T.site_lists(w, "transport", T.role_obs(env, "transport"))["bus_stops"]
    call = T.bus_line_call(T.bus_stops(lst))
    w.dept_funds["transport"] = 1520                   # $30 of avenue leaves $1,490: short of the $1,500 line
    box = ToolBox(env, "transport")
    up = _street(w)
    res = box.call("upgrade_road", up)
    assert res["refused"] == "bus_line" and res["error"] == (
        "this upgrade ($30) would leave transport $1,490, short of the $1,500 first bus line, and the city has none "
        f"yet (transit is a fifth of your score): build the line first, {call}, asking the mayor for what you lack; "
        "to upgrade anyway repeat with confirm: true"), res
    assert w.dept_funds["transport"] == 1520
    text = briefing(env, "transport", mark=False)
    assert "- First bus line ($1,500) before any avenue: create_bus_line through all the bus_stops stops" in text
    assert f"- bus_stops: {len(T.bus_stops(lst))} stops for one line: {call}" in text
    assert "- bus riders: no bus line yet" in text
    assert box.call("upgrade_road", {**up, "confirm": True})["ok"] and w.dept_funds["transport"] == 1490
    # short of the line: create_bus_line's failure asks for the gap, then names the call
    res = box.call("create_bus_line", {"stops": T.bus_stops(lst)})
    assert res["ok"] is False and res["message"].endswith(f" - ask the mayor for $10 (send_message), then {call}"), res
    # enough for both: the upgrade runs unrefused
    w.dept_funds["transport"] = 5000
    assert ToolBox(env, "transport").call("upgrade_road", _street(w))["ok"]


def test_another_bus_line_only_when_buses_run_full(open_env):
    """The mayor read "needs $1,500 for another bus line (transit 4.6%)" with buses at ~210 of 1,500 seats, and s1 ran
    four lines on the same stops (riders take the best line). With a line under BUS_FULL of its seats the bus_stops
    list says the buses have room and prices nothing; near full another line is ready again."""
    import civitas.fleet.tools as T
    env = copy.deepcopy(open_env)
    _no_spending(env)
    w = env.world
    stops = T.bus_stops(T.site_lists(w, "transport", T.role_obs(env, "transport"))["bus_stops"])
    w.dept_funds["transport"] = 1500
    assert apply_action(w, "transport", {"type": "create_bus_line", "stops": stops}).ok and len(w.bus_lines) == 1
    w.bus_lines[0].update(riders=210, capacity=1500)
    lst = T.site_lists(w, "transport", T.role_obs(env, "transport"))["bus_stops"]
    assert lst.short == "buses have room (210 of 1,500 seats used; another line pays once buses run 90% full)", lst
    assert "bus_stops" not in T.ready_prices(w, "transport", {"bus_stops": lst})
    text = briefing(env, "transport", mark=False)
    assert "- bus riders: 210 of 1,500 seats (1 line)" in text and "First bus line" not in text
    w.bus_lines[0]["riders"] = 1400
    lst = T.site_lists(w, "transport", T.role_obs(env, "transport"))["bus_stops"]
    assert lst.short is None and "bus_stops" in T.ready_prices(w, "transport", {"bus_stops": lst})
    # an upgrade with a bus line in place: never refused for it
    w.dept_funds["transport"] = 30
    assert ToolBox(env, "transport").call("upgrade_road", _street(w))["ok"]
