import numpy as np

from civitas.core.actions import apply_action
from civitas.core.env import CivitasEnv, GameConfig
from civitas.sim import catalog as C
from civitas.sim.world import World


def test_map_is_deterministic_and_has_river():
    a, b = World(seed=5), World(seed=5)
    assert (a.terrain == b.terrain).all()
    assert (a.terrain == C.WATER).sum() > 100
    # every row has river
    assert all((a.terrain[y] == C.WATER).any() for y in range(a.h))
    assert a.networks[a.road_labels[a.highway[1], a.highway[0]]].outside_connected


def test_power_is_served_nearest_first_and_limited_by_capacity():
    w = World(seed=0)
    y = w.highway[1]
    w.struct[y + 1, 2:22] = C.RES
    w.level[y + 1, 2:22] = 3                     # 20 tiles * 3 power = 60 demand
    w.struct[y - 1, 3] = C.WIND                  # 40 capacity
    w.refresh()
    served = w.powered[y + 1, 2:22]
    assert 0 < served.sum() < 20
    assert served[0] and not served[-1]          # nearest to the plant first


def test_tick_runs_and_records_history():
    w = World(seed=1)
    for _ in range(3):
        m = w.tick()
    assert m["month"] == 3 and len(w.history) == 3
    assert m["treasury"] < w.start_funds         # upkeep, no income yet


def _env():
    env = CivitasEnv(GameConfig(seed=0, months=12))
    env.world.dept_funds["planning"] = 1000
    env.world.dept_funds["utilities"] = 10000
    return env


def test_action_validation():
    env = _env()
    w = env.world
    assert not apply_action(w, "planning", {"type": "build_power_plant", "kind": "coal", "x": 1, "y": 1}).ok
    assert not apply_action(w, "planning", {"type": "zone", "zone": "X", "x1": 0, "y1": 0, "x2": 1, "y2": 1}).ok
    assert not apply_action(w, "planning", {"type": "zone", "zone": "R", "x1": 0, "y1": 99, "x2": 1, "y2": 1}).ok
    r = apply_action(w, "planning", {"type": "zone", "zone": "R", "x1": 2, "y1": 25, "x2": 6, "y2": 25})
    assert r.ok and r.cost == 25
    assert w.dept_funds["planning"] == 975


def test_budget_is_enforced():
    env = _env()
    r = apply_action(env.world, "planning", {"type": "build_road", "x1": 1, "y1": 5, "x2": 1, "y2": 40})
    assert r.ok  # 36 tiles * 10 = 360 <= 1000
    r = apply_action(env.world, "planning", {"type": "build_park", "x1": 3, "y1": 3, "x2": 5, "y2": 5})
    assert not r.ok and "budget" in r.message


def test_bridge_costs_more_and_pump_needs_riverbank():
    env = _env()
    w = env.world
    y = w.highway[1]
    river = np.nonzero(w.terrain[y] == C.WATER)[0]
    x_end = int(river.max()) + 1
    start_x = int(np.nonzero(w.struct[y] == C.ROAD)[0].max())
    w.dept_funds["planning"] = 5000
    r = apply_action(w, "planning", {"type": "build_road", "x1": start_x, "y1": y, "x2": x_end, "y2": y})
    assert r.ok and "bridge" in r.message
    bank_x = int(river.min()) - 1
    assert apply_action(w, "utilities", {"type": "build_water", "kind": "pump", "x": bank_x, "y": y - 1}).ok
    assert not apply_action(w, "utilities", {"type": "build_water", "kind": "pump", "x": 3, "y": y - 1}).ok


def test_facility_must_touch_road():
    env = _env()
    r = apply_action(env.world, "utilities", {"type": "build_power_plant", "kind": "wind", "x": 3, "y": 3})
    assert not r.ok and "road" in r.message


def test_tensor_observation_shapes():
    env = CivitasEnv(GameConfig(seed=0, months=12))
    t = env.tensor_observation()
    assert t["grid"].shape == (14, 48, 48)
    assert t["vector"].ndim == 1
