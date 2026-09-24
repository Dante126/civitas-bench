"""Mixed use, historic preservation, soil pollution, wind, seasons, metro, multi-tile buildings."""
import numpy as np

from civitas.core.actions import apply_action
from civitas.core.env import CivitasEnv, GameConfig
from civitas.sim import catalog as C
from civitas.sim.world import World


def _env():
    env = CivitasEnv(GameConfig(seed=0, months=24, milestones=False, events="none"))
    w = env.world
    y = w.highway[1]
    w.struct[y, 1:30] = C.ROAD
    w.refresh()
    for d in w.dept_funds:
        w.dept_funds[d] = 100000
    return env, w, y


def test_mixed_use_has_residents_and_shop_jobs():
    env, w, y = _env()
    assert apply_action(w, "planning", {"type": "zone", "zone": "M", "x1": 2, "y1": y + 1, "x2": 6, "y2": y + 1}).ok
    w.level[y + 1, 2:7] = 3
    assert w.residents_grid()[y + 1, 3] == C.MIX_RES[3] and w.shop_jobs_grid()[y + 1, 3] == C.MIX_JOBS[3]
    w._compute_population()
    assert w.stats["population"] == 5 * C.MIX_RES[3] and w.stats["jobs_com"] == 5 * C.MIX_JOBS[3]


def test_historic_buildings_keep_their_level():
    env, w, y = _env()
    w.struct[y + 1, 2:6] = C.RES
    w.level[y + 1, 2:6] = 3
    assert apply_action(w, "planning", {"type": "preserve_historic", "x1": 2, "y1": y + 1, "x2": 5, "y2": y + 1}).ok
    w.demand = {k: -1.0 for k in w.demand}           # everyone wants to leave
    for _ in range(10):
        w._grow()
    assert (w.level[y + 1, 2:6] == 3).all()
    w._update_heritage()
    assert w.stats["heritage_tourists"] == 4 * C.HISTORIC_TOURISTS


def test_soil_builds_up_under_industry_and_can_be_cleaned():
    env, w, y = _env()
    w.struct[y + 1, 2:6] = C.IND
    w.level[y + 1, 2:6] = 3
    w.ind_type[y + 1, 2:6] = 4                       # oil field
    for _ in range(60):
        w._update_soil()
    dirty = w.soil_level()[y + 1, 3]
    assert dirty > 0.2 and w.soil_level()[5, 40] < 0.01
    assert apply_action(w, "health", {"type": "clean_soil", "x1": 2, "y1": y + 1, "x2": 5, "y2": y + 1}).ok
    assert w.soil_level()[y + 1, 3] < dirty


def test_air_pollution_drifts_downwind():
    w = World(seed=0)
    w.struct[20, 20] = C.COAL
    w._wind_angles[:] = 0.0                          # wind blowing east
    w._wind_kernel_key = None
    w._compute_environment()
    assert w.pollution[20, 23] > w.pollution[20, 17]


def test_seasons_change_power_demand():
    w = World(seed=0, seasons=True)
    w.month = 0                                      # January: winter
    winter = w.event_factor("power_demand")
    w.month = 3                                      # April: spring
    assert winter > w.event_factor("power_demand") == 1.0
    assert World(seed=0).event_factor("power_demand") == 1.0      # seasons off by default in bare worlds


def test_multi_tile_buildings_cover_their_footprint_and_count_once():
    env, w, y = _env()
    r = apply_action(w, "health", {"type": "build_hospital", "x": 5, "y": y - 2})
    assert r.ok, r.message
    assert (w.struct[y - 2:y, 5:7] == C.HOSPITAL).all()
    assert w.facilities(C.HOSPITAL) == [(5, y - 2)] and w.n_buildings(C.HOSPITAL) == 1
    assert w.dept_upkeep()["health"] == C.SERVICES["hospital"]["upkeep"]
    assert not apply_action(w, "health", {"type": "build_hospital", "x": 6, "y": y - 3}).ok      # overlaps
    assert apply_action(w, "health", {"type": "demolish", "x": 6, "y": y - 1}).ok                # any tile works
    assert (w.struct[y - 2:y, 5:7] == C.NONE).all()


def test_metro_carries_riders_faster_than_cars():
    env, w, y = _env()
    w.struct[y + 1, 15:22] = C.RES
    w.level[y + 1, 15:22] = 3
    w.struct[y + 1, 2:8] = C.IND
    w.level[y + 1, 2:8] = 3
    w.refresh()
    w._compute_population()
    w._compute_traffic()
    slow = w.stats["commute_minutes"]
    assert apply_action(w, "transport", {"type": "build_metro_station", "x": 4, "y": y - 1}).ok
    assert apply_action(w, "transport", {"type": "build_metro_station", "x": 18, "y": y - 1}).ok
    assert apply_action(w, "transport", {"type": "create_metro_line", "stations": [[4, y - 1], [18, y - 1]]}).ok
    assert not apply_action(w, "transport", {"type": "create_metro_line", "stations": [[4, y - 1], [9, 9]]}).ok
    w._compute_traffic()
    assert w.stats["metro_riders"] > 0 and w.stats["commute_minutes"] < slow
