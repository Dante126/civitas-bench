"""Seeded events and disasters, milestones and unlocks, districts, landmarks."""
import numpy as np

from civitas.core.actions import apply_action
from civitas.core.env import CivitasEnv, GameConfig
from civitas.sim import catalog as C
from civitas.sim.events import make_schedule
from civitas.sim.world import World


def test_schedule_is_seeded():
    assert make_schedule(3, 120, 48) == make_schedule(3, 120, 48)
    assert make_schedule(3, 120, 48) != make_schedule(4, 120, 48)
    kinds = {e["kind"] for e in make_schedule(0, 120, 48)}
    assert kinds == {"flood", "heatwave", "recession", "factory_offer"}
    assert make_schedule(0, 24, 48) == []           # short games: no room for events


def _riverside_world(levee=False):
    """Homes on the riverbank one row below the main street."""
    w = World(seed=0)
    y = w.highway[1]
    road_end = int(np.nonzero(w.terrain[y] == C.WATER)[0].min()) - 1
    w.struct[y, 1:road_end + 1] = C.ROAD
    bank = int(np.nonzero(w.terrain[y + 1] == C.WATER)[0].min()) - 1     # bank of the homes' row
    w.struct[y + 1, bank - 1:bank + 1] = C.RES
    w.level[y + 1, bank - 1:bank + 1] = 3
    if levee:
        by = y + 2
        bx = int(np.nonzero(w.terrain[by] == C.WATER)[0].min()) - 1
        w.struct[by, bx] = C.LEVEE
    w.refresh()
    return w, y, bank


def test_flood_damages_riverside_buildings_unless_a_levee_protects_them():
    flood = {"kind": "flood", "month": 0, "duration": 1, "y1": 10, "y2": 40}
    w, y, bank = _riverside_world()
    w._flood(flood)
    assert w.level[y + 1, bank] == 1 and w.stats["flood_damaged"] > 0
    w2, y2, bank2 = _riverside_world(levee=True)
    w2._flood(flood)
    assert w2.level[y2 + 1, bank2] == 3


def test_heatwave_raises_power_demand():
    w, y, bank = _riverside_world()
    w.refresh()
    base = sum(n.power_demand for n in w.networks)
    w.schedule = [{"kind": "heatwave", "month": 0, "duration": 3, "id": 1}]
    w.refresh()
    assert sum(n.power_demand for n in w.networks) > 1.15 * base


def test_factory_offer_accept_and_decline():
    env = CivitasEnv(GameConfig(seed=0, months=24, events=[{"kind": "factory_offer", "month": 0, "duration": 3,
                                                           "jobs": 400, "bonus": 15000}]))
    w = env.world
    y = w.highway[1]
    w.struct[y, 1:30] = C.ROAD
    w.refresh()
    w._run_events()
    assert w.offers
    t0 = w.treasury
    r = apply_action(w, "mayor", {"type": "accept_offer", "offer": 1})
    assert r.ok and w.treasury == t0 + 15000 and w.complex_mask.sum() == 9
    assert not apply_action(w, "mayor", {"type": "decline_offer", "offer": 1}).ok


def test_milestones_unlock_buildings_and_pay_a_reward():
    env = CivitasEnv(GameConfig(seed=0, months=24))
    w = env.world
    y = w.highway[1]
    w.dept_funds["education"] = 20000
    r = apply_action(w, "education", {"type": "build_school", "kind": "high", "x": 5, "y": y - 1})
    assert not r.ok and "1,000 residents" in r.message
    t0 = w.treasury
    w.stats["population"] = 1200
    w._check_milestones()
    assert "Town" in w.milestones_reached and w.treasury == t0 + 5000
    assert apply_action(w, "education", {"type": "build_school", "kind": "high", "x": 5, "y": y - 1}).ok


def test_districts_policies_and_taxes():
    env = CivitasEnv(GameConfig(seed=0, months=24, milestones=False))
    w = env.world
    assert apply_action(w, "planning", {"type": "paint_district", "name": "Docks", "x1": 0, "y1": 0, "x2": 9, "y2": 9}).ok
    assert apply_action(w, "mayor", {"type": "enact_policy", "policy": "neighborhood_watch", "district": "Docks"}).ok
    mask = w.policy_mask("neighborhood_watch")
    assert mask[5, 5] and not mask[20, 20]
    assert apply_action(w, "mayor", {"type": "set_district_tax", "district": "Docks", "sector": "R", "delta": 3}).ok
    assert abs(w.tax_grid("R")[5, 5] - (w.taxes["R"] + 0.03)) < 1e-9 and w.tax_grid("R")[20, 20] == w.taxes["R"]
    assert not apply_action(w, "mayor", {"type": "set_district_tax", "district": "Nowhere", "sector": "R", "delta": 1}).ok
    assert apply_action(w, "planning", {"type": "specialize_district", "district": "Docks", "industry": "hightech"}).ok


def test_landmarks_raise_land_value_and_bring_tourists():
    env = CivitasEnv(GameConfig(seed=0, months=24, milestones=False))
    w = env.world
    y = w.highway[1]
    w.dept_funds["planning"] = 5000
    w._compute_environment()
    before = w.land_value[y - 2, 5]
    assert apply_action(w, "planning", {"type": "build_landmark", "kind": "plaza", "x": 5, "y": y - 1}).ok
    w._compute_landmarks()
    w._compute_environment()
    assert w.land_value[y - 2, 5] > before and w.stats["tourists"] > 0
