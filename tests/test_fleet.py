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
from civitas.fleet.tools import SITE_PURPOSES, SITE_UNLOCKS, ToolBox
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
    line, = [ln for ln in _ready_sites(briefing(env, "utilities", mark=False)) if ln.startswith("- sewage:")]
    assert "taken" in line and "(25,24)" in line and len(line) <= 120, line
    w.milestones_on = False                              # riverbank_2x2 unlocked: same reason
    res = box.call("find_sites", {"purpose": "riverbank_2x2"})
    assert res["sites"] == [] and "taken" in res["note"], res
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
    short, note = _why_empty(w, "police", "jail")
    assert "away from homes" in short and "a jail must stay away" in note
    assert _why_empty(w, "health", "clinic")[0] == "nothing needs it right now"     # other purposes: unchanged


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
    assert "(25,24)" in res["note"], res


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


def test_map_facts_on_an_empty_map():
    from civitas.fleet.tools import map_facts, map_lines
    env = _env()
    w = env.world
    w.struct[:] = C.NONE
    w.anchor[:] = -1
    w.refresh()
    f = map_facts(w)
    assert f["highway"] == [] and f["roads"] is None and f["built"] is None
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
    assert len("\n".join(map_lines(grown))) <= 400
    env = _env()
    env.world.terrain[env.world.terrain == C.WATER] = C.LAND
    f = map_facts(env.world)
    assert f["river"] == [] and "- no river" in map_lines(f)


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
    tight = map_overview(grown_env.world, limit=len(json.dumps(res)) - 1)     # a full legend: summary goes first
    assert "briefing" in tight["summary"][0] and tight["map"] == res["map"] and "scale" not in tight


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
                assert mine["sites"] == theirs["sites"], (dept, purpose)
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
    assert res["ok"] and "below_target" not in res and "next" not in res          # no targets yet
    res = mayor.call("update_plan", {"budgets": {"planning": 9000, "utilities": 7000, "fire": 2500, "police": 0}})
    assert res["ok"] and res["message"].startswith("plan v2 saved")
    assert res["below_target"] == {"utilities": {"funds": 2000, "target": 7000, "gap": 5000},
                                   "fire": {"funds": 0, "target": 2500, "gap": 2500}}      # planning is funded
    assert res["treasury"] == 30000
    assert res["next"] == "call allocate_budget to fund these (treasury $30k)"
    assert mayor.call("allocate_budget", {"department": "utilities", "amount": 5000})["ok"]
    res = mayor.call("update_plan", {"priorities": {"fire": "stations by homes"}})
    assert res["below_target"] == {"fire": {"funds": 0, "target": 2500, "gap": 2500}}
    assert res["treasury"] == 25000


def test_update_plan_says_when_the_gaps_exceed_the_treasury():
    env = _env()
    w = env.world
    w.treasury = 12000                                   # gaps $9,000 = treasury - the $3,000 reserve: enough
    res = ToolBox(env, "mayor").call("update_plan", {"budgets": {"planning": 6000, "utilities": 3000}})
    assert res["next"] == "call allocate_budget to fund these (treasury $12k)"
    w.treasury = 11000
    res = ToolBox(env, "mayor").call("update_plan", {"budgets": {"planning": 6000, "utilities": 3000}})
    assert res["next"] == ("call allocate_budget to fund these (treasury $11k); the gaps total $9,000, more than the "
                           "treasury above a $3,000 reserve ($8,000): fund the most urgent first or lower targets")


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


def test_update_plan_says_hold_under_the_reserve():
    env = _env()
    from civitas.fleet.fmt import money
    w = env.world
    for treasury in (3000, 1200, -500):                  # at, under and below zero: nothing above the $3,000 reserve
        w.treasury = treasury
        res = ToolBox(env, "mayor").call("update_plan", {"budgets": {"planning": 6000}})
        assert res["next"] == (f"the treasury ({money(treasury)}) is at or under the $3,000 reserve: hold; lower "
                               "targets or wait for income")
        text = briefing(env, "mayor", mark=False)
        line, = [ln for ln in text.splitlines() if "below target" in ln]
        assert line.endswith(f"treasury {money(treasury)}, at or under the $3,000 reserve): hold; lower targets "
                             "or wait for income")
        assert "allocate_budget" not in line
    w.treasury = 3001                                    # $1 above: short, but something to allocate
    assert ToolBox(env, "mayor").call("update_plan", {"budgets": {"planning": 6000}})["next"].startswith(
        "call allocate_budget")


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
    assert "- 1 below target ($5,000 short" in text and "- fire:" not in text


def test_reserve_follows_the_mayor_playbook():
    from types import SimpleNamespace
    from civitas.fleet.tools import reserve
    world = lambda *h: SimpleNamespace(history=[{"income": i, "expenses": e} for i, e in h])
    assert reserve(world()) == 3000                                      # before the first month
    assert reserve(world((5000, 800))) == 3000
    assert reserve(world((5000, 2000))) == 6000                          # 3 months of expenses
    assert reserve(world((1000, 2000))) == 12000                         # 6 while running a deficit


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
    res = box.call("bulldoze", {"x1": 5, "y1": 25, "x2": 5, "y2": 28})      # the whole spur: nothing left behind
    assert res["ok"] and res["message"] == "bulldozed 4 tiles" and "refused" not in res
    assert apply_action(w, "planning", {"type": "zone", "zone": "R", "x1": 4, "y1": 25, "x2": 4, "y2": 26}).ok
    assert ToolBox(env, "planning").call("bulldoze", {"x1": 4, "y1": 25, "x2": 4, "y2": 26})["ok"]   # zones only
    w.dept_funds["utilities"] = 10000
    s = ToolBox(env, "utilities").call("find_sites", {"purpose": "wind"})["sites"][0]
    assert apply_action(w, "utilities", {"type": "build_power_plant", "kind": "wind", "x": s["x"], "y": s["y"]}).ok
    res = ToolBox(env, "utilities").call("demolish", {"x": s["x"], "y": s["y"]})
    assert res["ok"] and "demolished" in res["message"], res


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
        "every tile on that line is already road - highway-connected roads: x 0-28, y 24, 29 tiles; start on the road "
        "and run into free land (find_sites road_extension)"), res
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
    assert box.call("zone", {"zone": "C", **rect})["message"] == "no empty land tiles in that area"


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
    from civitas.fleet.tools import map_facts, map_lines
    env = copy.deepcopy(env12) if grown else CivitasEnv(GameConfig(seed=0, months=24))
    w = env.world
    for role in TURN_ORDER:
        text = briefing(env, role, mark=False)
        assert _words(text) < 700, (role, _words(text))
        if role == "planning":
            dev, = [ln for ln in text.splitlines() if "developed tiles" in ln]
            assert "R " in dev and "+" not in dev                        # counts, not signed demand values
        assert f"{24 - w.month} months left" in text and "autonomous" in text
        assert "\n## Map\n" + "\n".join(map_lines(map_facts(w))) + "\n" in text     # every role, mayor included
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
        assert len(lines) == len(purposes) + 1 and lines[-1] == "find_sites <purpose> lists up to 6 with details"
        box = ToolBox(env, role)
        for purpose, line in zip(purposes, lines):
            assert line.startswith(f"- {purpose}: ") and len(line) <= 110, (role, line)
            res = box.call("find_sites", {"purpose": purpose})
            sites, note = res["sites"], res.get("note", "")
            if note.startswith("locked"):
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
    ("police", "station"): {"type": "build_police_station"}, ("police", "jail"): {"type": "build_jail"},
    ("fire", "station"): {"type": "build_fire_station"}, ("fire", "shelter"): {"type": "build_shelter"},
    ("health", "clinic"): {"type": "build_clinic"}, ("health", "hospital"): {"type": "build_hospital"},
    ("health", "landfill"): {"type": "build_landfill"}, ("health", "cemetery"): {"type": "build_cemetery"},
    ("education", "school"): {"type": "build_school", "kind": "elementary"},
    ("education", "high_school"): {"type": "build_school", "kind": "high"},
    ("education", "university"): {"type": "build_school", "kind": "university"},
}


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
    from civitas.fleet.tools import build_cost
    env = copy.deepcopy(open_env)
    w = env.world
    cost = build_cost(w, role, purpose)
    s = ToolBox(env, role).call("find_sites", {"purpose": purpose})["sites"][0]
    dept = ROLES[role]["budget"]
    action = {**BUILD_ACTIONS[role, purpose], "x": s["x"], "y": s["y"]}
    w.dept_funds[dept] = cost - 1
    res = apply_action(w, role, action)
    assert not res.ok and res.message.startswith(f"costs ${cost:,} but the {dept} budget only has ${cost - 1:,}"), \
        res.message
    w.dept_funds[dept] = cost                          # the listed site takes it, for exactly that
    res = apply_action(w, role, action)
    assert res.ok and res.cost == cost and w.dept_funds[dept] == 0, res.message


def test_build_cost_prices_the_cheapest_unlocked_building(grown_env, open_env):
    from civitas.fleet.tools import build_cost
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
    assert priced == set(BUILD_ACTIONS)                # not planning, levees (per tile), bus stops, congested roads


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
    assert "(cheapest: shelter $1,500)" in briefing(env, "fire", mark=False)
    assert "(cheapest: cemetery $1,500)" in briefing(env, "health", mark=False)     # cheaper than a clinic
    for role in ("transport", "planning"):             # only bus stops ready / nothing priced
        text = briefing(env, role, mark=False)
        assert "can't pay" not in text and "$" not in "".join(_ready_sites(text)), role
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


def test_levees_are_priced_per_tile_in_a_flood(grown_env):
    from civitas.fleet.tools import build_cost, site_price
    env = copy.deepcopy(grown_env)
    w = env.world
    w.schedule.append({"kind": "flood", "month": w.month + 2, "duration": 1, "y1": 15, "y2": 32})
    tile = C.CATALOG["levee"]["cost"]
    assert build_cost(w, "fire", "levee") is None and site_price(w, "fire", "levee").per_tile      # not a building
    fire_line = lambda: next(ln for ln in briefing(env, "mayor", mark=False).splitlines() if ln.startswith("- fire:"))
    w.dept_funds["fire"] = tile                        # one levee tile, less than a shelter
    fire = briefing(env, "fire", mark=False)
    levee, = [ln for ln in _ready_sites(fire) if ln.startswith("- levee:")]
    assert levee.endswith(f"; ${tile:,}/tile") and "can't pay" not in fire, fire
    assert fire_line() == f"- fire: funds ${tile:,}"
    w.dept_funds["fire"] = tile - 1
    fire = briefing(env, "fire", mark=False)
    levee, = [ln for ln in _ready_sites(fire) if ln.startswith("- levee:")]
    assert levee.endswith(f"; ${tile:,}/tile - you have ${tile - 1:,}") and f"(cheapest: levee ${tile:,}/tile)" in fire
    assert fire_line().startswith(f"- fire: funds ${tile - 1:,}; needs ${tile:,} for a levee tile (")
    s = ToolBox(env, "fire").call("find_sites", {"purpose": "levee"})["sites"][0]
    res = apply_action(w, "fire", {"type": "build_levee", "x": s["x"], "y": s["y"]})       # a single tile
    assert not res.ok and res.message.startswith(f"costs ${tile:,} but the fire budget only has ${tile - 1:,}")


def test_mayor_briefing_says_what_short_departments_need(grown_env):
    env = copy.deepcopy(grown_env)
    w = env.world
    w.dept_funds.update(police=1000, fire=5000, health=0, education=0, utilities=0, transport=0, planning=0)
    n = len(ToolBox(env, "police").call("find_sites", {"purpose": "station"})["sites"])
    lines = lambda: {ln.split(":")[0][2:]: ln for ln in briefing(env, "mayor", mark=False).splitlines()
                     if ln.startswith("- ") and ": funds " in ln}
    got = lines()                                      # the building, not the find_sites purpose
    assert got["police"] == f"- police: funds $1,000; needs $2,500 for a police station ({n} ready sites)"
    assert got["health"].startswith("- health: funds $0; needs $1,500 for a cemetery (")
    assert got["education"].startswith("- education: funds $0; needs $2,500 for an elementary school (")
    assert got["utilities"].startswith("- utilities: funds $0; needs $2,200 for a water tower (")
    assert got["fire"] == "- fire: funds $5,000"
    assert got["transport"] == "- transport: funds $0" and got["planning"] == "- planning: funds $0"
    ToolBox(env, "mayor").call("update_plan", {"budgets": {"police": 2500}, "priorities": {"police": "cover downtown"}})
    assert lines()["police"] == (f"- police: funds $1,000 < target $2,500; needs $2,500 for a police station ({n} "
                                 "ready sites); priority: cover downtown")
    w.dept_funds["police"] = 2500
    assert lines()["police"] == "- police: funds $2,500, target $2,500; priority: cover downtown"


def test_mayor_names_the_cheapest_building():
    from civitas.fleet.briefing import _a
    from civitas.fleet.tools import SITE_BUILDS, building_name
    names = {n: _a(building_name(n)) for purposes in SITE_BUILDS.values() for ns in purposes.values() for n in ns}
    assert (names["tower"], names["outlet"], names["coal"], names["pump"]) == (
        "a water tower", "a sewage outlet", "a coal plant", "a water pump")                  # base tiers: struct names
    assert (names["water_works"], names["advanced_treatment"], names["hospital"]) == (
        "a water works", "an advanced treatment", "a hospital")                               # higher tiers: their own
    assert (names["elementary_school"], names["university"], names["harbour"], names["metro_station"]) == (
        "an elementary school", "a university", "a harbour", "a metro station")
    assert not any("_" in v for v in names.values()), names


def test_department_briefing_builds_one_observation(monkeypatch):
    import civitas.fleet.tools as T
    env = _env()
    calls = []
    real = T.build_observation

    def counted(*a, **k):
        calls.append(k.get("site_hints"))
        return real(*a, **k)
    monkeypatch.setattr(T, "build_observation", counted)           # briefings build it through tools.role_obs
    for role in DEPTS:
        calls.clear()
        briefing(env, role, mark=False)
        assert calls == [True], (role, calls)
    calls.clear()                                      # the mayor: one per department that may lack money for a site
    briefing(env, "mayor", mark=False)
    n = sum(any(T.site_price(env.world, d, p) for p in SITE_PURPOSES[d]) for d in DEPTS)
    assert n and calls == [True] * n
    env.world.dept_funds.update({d: 50000 for d in DEPTS})
    calls.clear()
    briefing(env, "mayor", mark=False)
    assert calls == []


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
    env = _env()
    w = env.world
    w.dept_funds.update(planning=9500, utilities=2000)
    ToolBox(env, "mayor").call("update_plan", {"budgets": {"planning": 9000, "utilities": 7000, "fire": 2500}})
    lines = {ln.split(":")[0][2:]: ln for ln in briefing(env, "mayor", mark=False).splitlines()
             if ln.startswith("- ") and ": funds " in ln}
    assert lines["planning"] == "- planning: funds $9,500, target $9,000"
    assert lines["utilities"] == ("- utilities: funds $2,000 < target $7,000; needs $2,200 for a water tower (6 ready "
                                  "sites)")                          # near_homes: its cheapest ready build
    assert lines["fire"] == "- fire: funds $0 < target $2,500"
    assert lines["police"] == "- police: funds $0"
    text = briefing(env, "mayor", mark=False)
    from civitas.fleet.fmt import money
    assert f"- 2 below target ($7,500 short; treasury {money(w.treasury)}): allocate_budget" in text
    w.treasury = 9000
    assert ("- 2 below target ($7,500 short; treasury $9,000, $6,000 above the $3,000 reserve): "
            "allocate_budget") in briefing(env, "mayor", mark=False)
    text = briefing(env, "mayor", mark=False)
    assert text.index("- no reports yet") < text.index("- 2 below target")
    w.dept_funds.update(utilities=6400, fire=2350)                  # within 10% of the target counts as funded
    text = briefing(env, "mayor", mark=False)
    assert "below target" not in text and "- fire: funds $2,350, target $2,500" in text
    w.dept_funds.update(fire=2399)                                  # $101 short of $2,500: max($100, 10%) = $250
    assert "below target" not in briefing(env, "mayor", mark=False)
    w.dept_funds.update(fire=0)
    ToolBox(env, "mayor").call("update_plan", {"budgets": {"fire": 500}})
    w.dept_funds.update(fire=401)                                   # $99 short: under the $100 floor
    assert "below target" not in briefing(env, "mayor", mark=False)
    w.dept_funds.update(fire=399)
    assert "- 1 below target ($101 short" in briefing(env, "mayor", mark=False)


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


def test_briefings_stay_bounded_at_the_limits():
    from civitas.fleet.memory import LIMITS
    from civitas.fleet.tools import MAX_ENTRIES, REPORT_MAX_CHARS
    env = _env()
    w = env.world
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
    w.refresh()
    env.end_month()
    w.months_negative, w.low_approval_months, w.treasury = 11, 11, -123456.78        # every warning, below the reserve
    w.offers.append({"id": 12345, "jobs": 1234, "bonus": 123456, "expires": 123})
    for role in TURN_ORDER:
        text = briefing(env, role, mark=False)
        memory = text[text.index("## Your memory"):].split("\n\n")[0]
        assert len(text) - len(memory) < 7500 - 200, (role, len(text) - len(memory))   # memory is shown in full
        assert len(text) < 13000, (role, len(text))
        assert "MEM-C" in text and "MEM-L" in text                            # memory shown to every role
        assert f"({MAX_ENTRIES} older; get_messages shows only the latest {MAX_ENTRIES})" in text and f"msg {MAX_ENTRIES + 4} " in text
    mayor = briefing(env, "mayor", mark=False)
    assert "WARNING" in mayor and "OPEN OFFER" in mayor and "under the $3,000 reserve" in mayor
    assert "- roads not connected to the highway: 121 tiles (x 10-46, y 10-47) - reconnect them" in mayor
    assert "(cut reports: history kind=report)" in mayor



# ------------------------------------------------------------------ tool agent + fleet configs
import contextlib
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from civitas.agents.llm_client import LLMError
from pathlib import Path

from civitas.config import apply_cli_overrides, build_agents, game_config, load_config, scoring_config
from civitas.fleet.agent import ToolAgent
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
    assert d.meta["messages"][2]["content"] == "thinking aloud"
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


def test_tool_agent_refuses_region_envs():
    from civitas.core.region import RegionEnv
    a = ToolAgent("mayor", cfg=FleetAgentConfig())
    with pytest.raises(ValueError, match="single-city"):
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
    with pytest.raises(ValueError, match="single-city"):
        build_agents(cfg, ["north.mayor"], game, scoring_config(cfg, game))
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
    assert d.meta["tool_trace"] == [] and d.meta["recovered_calls"] == 0


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
    bad = [box.call("end_turn", {}) for _ in range(2)]                              # a failed end_turn is not refused
    assert all(r["ok"] is False and "refused" not in r for r in bad)


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
                   "tool_errors": ["bad args", "action limit reached", "no plan"], "repeats": 1, "refused": 1}


def test_turn_row_and_rebuild_tolerate_minimal_records(tmp_path):
    row = turn_row({"role": "fire", "month": 2})
    assert row["steps"] == 0 and row["prompt_tokens"] == 0 and row["results"] == [] and row["tool_errors"] == []
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
    pol = rep["roles"]["police"]
    assert pol["failed"] == 5 and pol["top_failures"][0]["count"] == 4
    assert pol["top_failures"] == [{"message": fails[0][2], "count": 4}, {"message": "costs $9,000 but the police "
                                   "budget only has $500; ask the mayor", "count": 1}]


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
    assert "repeats (look-ups / refused)" in md and "| 2 / 4 |" in md
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


def _transport_idle(funds, plans=(), turns=(0, 3, 6, 9), months=12):
    """Idle hints for a transport department that never acts; `funds` maps history month -> its funds (default 0),
    `plans` are (month, budgets) plan versions."""
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
    return _hints(rep, "idle")


def test_idle_hint_skips_a_department_without_money_or_a_target():
    assert _transport_idle({}) == []                                      # $0 and no plan budget all run
    assert _transport_idle({}, plans=[(0, {"transport": 0})]) == []       # a zero budget counts as none
    assert _transport_idle({m: 0.6 for m in range(13)}) == []             # under $1
    h, = _transport_idle({m: 500.0 for m in range(13)})                   # money but no target: idle
    assert (h["turns"], h["months"]) == (4, 12) and "mayor" not in h["text"]
    h, = _transport_idle({}, plans=[(0, {"transport": 1000})])            # a target but no money: the mayor's fault
    assert (h["turns"], h["months"]) == (4, 12)
    assert "its funds stayed at $0 against a $1,000 target: check the mayor's allocations" in h["text"]
    h, = _transport_idle({4: 500.0}, plans=[(0, {"transport": 1000})])    # money at one turn of the run: no note
    assert (h["turns"], h["months"]) == (4, 12) and "mayor" not in h["text"]


def test_unfunded_turns_split_idle_runs_and_read_the_month_after_the_turn():
    # a turn in month t ends in history entry t + 1: the mayor's month-6 allocation shows there, not in entry 6
    h, = _transport_idle({m: 5000.0 for m in range(7, 13)})
    assert (h["turns"], h["first"], h["last"]) == (2, 6, 11)
    assert _transport_idle({m: 5000.0 for m in range(8, 13)}) == []       # funded after the month-6 turn: one idle turn
    # the plan saved in month 6 (before the departments' turns) is in force for the month-6 turn
    h, = _transport_idle({}, plans=[(6, {"transport": 2000})])
    assert (h["turns"], h["first"], h["last"]) == (2, 6, 11) and "$2,000 target" in h["text"]
    assert _transport_idle({4: 500.0, 10: 500.0}) == []                   # idle turns at 3 and 9, blocked between
    assert _transport_idle({}, turns=range(12)) == []


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
