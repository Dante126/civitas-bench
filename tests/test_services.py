"""City services: road-network coverage, crime, fire, garbage, health, education."""
import numpy as np

from civitas.core.actions import apply_action
from civitas.core.env import CivitasEnv, GameConfig
from civitas.sim import catalog as C
from civitas.sim.grid import road_bfs, spread_to_tiles
from civitas.sim.world import World


def _street_world(seed=0, length=30):
    """A world with a long straight street east of the highway and homes along it."""
    w = World(seed=seed)
    y = w.highway[1]
    w.struct[y, 1:length] = C.ROAD
    w.struct[y + 1, 2:length] = C.RES
    w.level[y + 1, 2:length] = 3
    w.refresh()
    return w, y


def test_road_bfs_measures_distance_along_roads():
    road = np.zeros((10, 10), bool)
    road[5, :] = True            # horizontal road
    road[0:6, 9] = True          # goes up the east edge
    dist, lab = road_bfs(road, [(0, 4)], max_d=50)
    assert dist[5, 0] == 1 and dist[5, 1] == 1 and dist[5, 9] == 9 and dist[0, 9] == 14
    assert lab[0, 9] == 0
    td, _ = spread_to_tiles(dist, lab, 2)
    assert td[7, 4] == dist[5, 2] + 2          # best road within 2 tiles (here diagonally)
    assert td[8, 4] > 1000                     # 3 rows from the road: unreachable


def test_coverage_follows_roads_and_fades_with_distance():
    w, y = _street_world()
    w.struct[y - 1, 3] = C.POLICE
    w.refresh()
    w._compute_services()
    near, far = w.police_cov[y + 1, 4], w.police_cov[y + 1, 25]
    assert near == 1.0 and far == 0.0
    # a straight-line neighbour with no road between gets nothing
    w2, y2 = _street_world()
    w2.struct[y2 - 6, 3] = C.POLICE        # not touching any road
    w2.refresh()
    w2._compute_services()
    assert w2.police_cov.max() == 0.0


def test_capacity_limits_coverage():
    w, y = _street_world()
    w.struct[y - 1, 10] = C.CLINIC
    w.refresh()
    w._compute_services()
    full = w.health_cov[y + 1, 10]
    C.SERVICES["clinic"]["capacity"] = 100          # far fewer than the residents nearby
    try:
        w._compute_services()
        assert w.health_cov[y + 1, 10] < 0.5 * full
    finally:
        C.SERVICES["clinic"]["capacity"] = 2500


def test_police_lowers_crime():
    def crime_after(with_police):
        w, y = _street_world()
        if with_police:
            w.struct[y - 1, 12] = C.POLICE
        w.refresh()
        w._compute_services()
        for _ in range(12):
            w._update_crime()
        return float(w.crime[y + 1, 5:20].mean())
    assert crime_after(True) < 0.5 * crime_after(False)


def test_uncovered_fires_spread_and_covered_fires_are_put_out():
    def lost(with_station):
        w, y = _street_world(length=40)
        w.struct[y + 1, 2:40] = C.IND
        if with_station:
            w.struct[y - 1, 20] = C.FIRE
        w.refresh()
        w._compute_services()
        total = 0
        for _ in range(24):
            w.level[y + 1, 2:40] = 3
            w._fire_events()
            total += w.stats["fire_tiles_lost"]
        return total
    assert lost(False) > lost(True)


def test_garbage_collection_and_landfill_filling():
    w, y = _street_world()
    w._garbage_and_health()
    piled = w.stats["garbage_backlog"]
    assert piled > 0
    w.struct[y - 1, 10] = C.LANDFILL
    w.refresh()
    w._compute_services()
    w._garbage_and_health()
    assert w.stats["garbage_collected"] > 0
    assert w.stats["garbage_backlog"] < piled
    assert w._fstate(10, y - 1)["fill"] > 0


def test_full_landfill_stops_collecting():
    w, y = _street_world()
    w.struct[y - 1, 10] = C.LANDFILL
    w.refresh()
    w._fstate(10, y - 1)["fill"] = C.SERVICES["landfill"]["fill"]
    w._compute_services()
    assert w.garbage_cov.max() == 0.0


def test_schools_graduate_only_students_with_seats():
    def adult_high_school_share(with_schools):
        w, y = _street_world()
        if with_schools:
            w.struct[y - 1, 8] = C.SCHOOL
            w.struct[y - 1, 16] = C.HIGHSCHOOL
            w.refresh()
        w._compute_population()
        for _ in range(96):
            w._compute_services()
            w._step_cohorts()
            w._compute_population()
        teens = w.cohorts[1]
        return float(teens[2:].sum() / teens.sum())
    assert adult_high_school_share(True) > adult_high_school_share(False) + 0.1


def test_job_matching_over_educated_take_lower_jobs():
    w, _ = _street_world()
    w.cohorts[:] = 0
    w.cohorts[2, 3] = 1000                    # only university-educated adults
    m = w._match_jobs({"I": 300})
    assert abs(m["filled"]["I"] - 300) < 1e-6     # they take unskilled jobs
    w.cohorts[:] = 0
    w.cohorts[2, 0] = 1000                    # only uneducated adults
    m = w._match_jobs({"O": 300})
    assert m["filled"]["O"] < 1.0                 # offices need educated workers


def test_service_actions_use_the_right_budget_and_validate():
    env = CivitasEnv(GameConfig(seed=0, months=12, milestones=False))
    w = env.world
    y = w.highway[1]
    w.dept_funds["police"] = 3000
    assert not apply_action(w, "police", {"type": "build_fire_station", "x": 3, "y": y - 1}).ok   # not police's
    assert not apply_action(w, "police", {"type": "build_police_station", "x": 3, "y": 3}).ok      # no road
    r = apply_action(w, "police", {"type": "build_police_station", "x": 3, "y": y - 1})
    assert r.ok and w.dept_funds["police"] == 500
    assert not apply_action(w, "fire", {"type": "demolish", "x": 3, "y": y - 1}).ok             # not fire's
    assert apply_action(w, "police", {"type": "demolish", "x": 3, "y": y - 1}).ok
    w.dept_funds["education"] = 20000
    r = apply_action(w, "education", {"type": "build_school", "kind": "high", "x": 5, "y": y - 1})
    assert r.ok and w.struct[y - 1, 5] == C.HIGHSCHOOL
    assert not apply_action(w, "education", {"type": "build_school", "kind": "kindergarten", "x": 6, "y": y - 1}).ok


def test_noop_city_scores_zero():
    from civitas.config import apply_cli_overrides, load_config
    from civitas.bench.runner import run_episode
    import tempfile
    cfg = load_config("configs/baseline.toml")
    cfg = apply_cli_overrides(cfg, [f"{r}=noop" for r in ("mayor", "planning", "utilities", "police", "fire",
                                                          "health", "education")], ["game.months=24"])
    with tempfile.TemporaryDirectory() as d:
        s = run_episode(cfg, 0, d, verbose=False)
    assert s["final_scores"]["city"] < 1.0
    assert all(v < 1.0 for v in s["final_scores"].values())


# ------------------------------------------------------------- sewage & river
def _river_world():
    """Homes on the main street, which runs east to the river bank."""
    w = World(seed=0)
    y = w.highway[1]
    bank = int(np.nonzero(w.terrain[y] == C.WATER)[0].min()) - 1
    w.struct[y, 1:bank + 1] = C.ROAD
    w.struct[y + 1, 2:bank - 2] = C.RES
    w.level[y + 1, 2:bank - 2] = 3
    w.refresh()
    return w, y, bank


def test_sewage_flows_downstream_and_treatment_cleans_it():
    def river(kind):
        w, y, bank = _river_world()
        w.struct[y - 1, bank] = C.SEWAGE_PLANTS[kind]
        w.refresh()
        assert w.sewered[y + 1, 5]
        w._compute_environment()
        return w.river_row, y
    raw, y = river("outlet")
    assert raw[y - 5] == 0.0 and raw[y] > 0.05       # nothing upstream, polluted downstream
    assert raw[-1] < raw[y]                          # dilutes further south
    assert river("treatment")[0][y] < 0.35 * raw[y]


def test_pump_downstream_of_outlet_gets_dirty_water():
    def pump_quality(outlet_row):
        w, y, bank = _river_world()
        w.struct[y - 1, bank] = C.PUMP
        w.refresh()
        w.sewage_discharge = [(bank, outlet_row, 400.0)]
        w._compute_environment()
        w.refresh()
        return next(p["quality"] for p in w.plant_list if p["kind"] == "pump")
    w, y, _ = _river_world()
    assert pump_quality(y - 10) < pump_quality(y + 10) == 1.0     # upstream outlet dirties the pump


def test_sewage_facilities_need_the_riverbank():
    env = CivitasEnv(GameConfig(seed=0, months=12))
    w = env.world
    y = w.highway[1]
    w.dept_funds["utilities"] = 20000
    assert not apply_action(w, "utilities", {"type": "build_sewage", "kind": "outlet", "x": 3, "y": y - 1}).ok
    assert not apply_action(w, "utilities", {"type": "build_sewage", "kind": "septic", "x": 3, "y": y - 1}).ok
