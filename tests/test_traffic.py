"""Traffic model, road classes, bus lines, train stations, transport actions."""
import numpy as np

from civitas.core.actions import apply_action
from civitas.core.env import CivitasEnv, GameConfig
from civitas.sim import catalog as C
from civitas.sim.traffic import road_tree
from civitas.sim.world import World


def _commuter_world(homes=3):
    """Homes at the east end of a long street, jobs at the west end."""
    w = World(seed=0)
    y = w.highway[1]
    w.struct[y, 1:30] = C.ROAD
    w.struct[y + 1, 22:30] = C.RES
    w.level[y + 1, 22:30] = homes
    w.struct[y + 1, 2:8] = C.IND
    w.level[y + 1, 2:8] = 3
    w.refresh()
    w._compute_population()
    return w, y


def test_road_tree_parents_point_towards_the_hub():
    road = np.zeros((5, 8), bool)
    road[2, :] = True
    dist, parent, levels = road_tree(road, (0, 2))
    assert dist[2, 7] == 7
    assert parent[2 * 8 + 7] == 2 * 8 + 6
    assert [len(l) for l in levels] == [1] * 8


def test_commuters_load_the_road_between_homes_and_jobs():
    w, y = _commuter_world()
    w._compute_traffic()
    assert w.stats["trips"] > 0
    mid = w.traffic_load[y, 15]
    assert mid > 0.5 * w.stats["trips"]           # most commuters pass the middle of the street
    assert w.traffic_load[y, 29] < mid
    assert w.stats["commute_minutes"] > C.COMMUTE_BASE_MIN


def test_avenues_relieve_congestion():
    w, y = _commuter_world()
    C.ROAD_CAPACITY[1], old = 50.0, C.ROAD_CAPACITY[1]
    try:
        w._compute_traffic()
        jammed, t_jam = w.congestion[y, 15], w.stats["commute_minutes"]
        w.road_class[y, 1:30] = 2
        w.refresh()
        w._compute_traffic()
        assert w.congestion[y, 15] < jammed and w.stats["commute_minutes"] < t_jam
    finally:
        C.ROAD_CAPACITY[1] = old


def test_bus_line_carries_riders_and_takes_cars_off_the_road():
    w, y = _commuter_world()
    w._compute_traffic()
    cars = w.traffic_load[y, 15]
    hub = next(h for h in w.hubs if h["kind"] == "jobs")
    w.bus_lines = [{"id": 1, "stops": [(hub["x"], hub["y"]), (25, y)]}]
    w._compute_traffic()
    assert w.stats["transit_riders"] > 0
    assert w.traffic_load[y, 15] < cars


def test_highway_roads_give_no_zone_access():
    w, y = _commuter_world()
    assert w.access[y + 2, 15]
    w.road_class[y, 10:20] = 3
    w.refresh()
    assert not w.access[y + 2, 15]


def test_transport_actions_validate_and_charge_the_budget():
    env = CivitasEnv(GameConfig(seed=0, months=12, milestones=False))
    w = env.world
    y = w.highway[1]
    w.dept_funds["transport"] = 5000
    r = apply_action(w, "transport", {"type": "upgrade_road", "to": "avenue", "x1": 1, "y1": y, "x2": 10, "y2": y})
    assert r.ok and r.cost == 10 * C.ROAD_UPGRADE_COST[2] and w.road_class[y, 5] == 2
    assert not apply_action(w, "transport", {"type": "upgrade_road", "to": "avenue", "x1": 1, "y1": y, "x2": 10, "y2": y}).ok
    assert not apply_action(w, "transport", {"type": "upgrade_road", "to": "avenue", "x1": 1, "y1": 3, "x2": 5, "y2": 3}).ok
    assert not apply_action(w, "transport", {"type": "create_bus_line", "stops": [[1, y], [3, 3]]}).ok     # (3,3) off-road
    assert apply_action(w, "transport", {"type": "create_bus_line", "stops": [[1, y], [8, y]]}).ok
    assert apply_action(w, "transport", {"type": "set_fare", "fare": 3}).ok and w.fare == 3
    assert not apply_action(w, "transport", {"type": "set_fare", "fare": 30}).ok
    assert not apply_action(w, "transport", {"type": "build_train_station", "x": 5, "y": y - 1}).ok   # not the edge
    assert apply_action(w, "transport", {"type": "remove_bus_line", "line": 1}).ok


def test_train_station_links_a_separate_network_to_the_region():
    w = World(seed=0)
    w.struct[0, 5:12] = C.ROAD            # a road on the north edge, not connected to the highway
    w.refresh()
    assert not w.access[1, 8]
    w.struct[0, 4] = C.TRAIN
    w.refresh()
    assert w.access[1, 8]
