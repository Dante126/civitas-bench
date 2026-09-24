"""Density, levels 1-5, office zones, wealth classes, specialised industry, households."""
import numpy as np

from civitas.core.actions import apply_action
from civitas.core.env import CivitasEnv, GameConfig
from civitas.sim import catalog as C
from civitas.sim.world import World


def _serviced(w, x, y, lv=0.9):
    """Give one tile every service so only density / policy caps limit it."""
    for g in (w.access, w.powered, w.watered, w.sewered):
        g[y, x] = True
    w.garbage_cov[y, x] = 1.0
    w.school_cov[y, x] = 1.0
    w.land_value[y, x] = lv
    w.crime[y, x] = 0.0
    w.struct[y + 1, x] = C.PARK


def test_density_and_policy_caps_on_levels():
    w = World(seed=0)
    x, y = 5, 5
    w.struct[y, x] = C.RES
    _serviced(w, x, y)
    w.density[y, x] = 1
    assert w.max_levels()[y, x] == 5
    w.density[y, x] = 0
    assert w.max_levels()[y, x] == C.LOW_DENSITY_MAX
    w.density[y, x] = 1
    w.policies.add("high_rise_ban")
    assert w.max_levels()[y, x] == 3
    w.policies.clear()
    w.land_value[y, x] = 0.5              # enough for L3, not L4
    assert w.max_levels()[y, x] == 3
    w.sewered[y, x] = False               # L3 needs sewage
    assert w.max_levels()[y, x] == 2


def test_generic_industry_stops_at_level_3_but_high_tech_goes_higher():
    w = World(seed=0)
    x, y = 5, 5
    w.struct[y, x] = C.IND
    w.density[y, x] = 1
    _serviced(w, x, y)
    assert w.max_levels()[y, x] == 3
    w.ind_type[y, x] = 5
    assert w.max_levels()[y, x] >= 4


def test_wealth_follows_land_value_and_industry_specialises_on_resources():
    w = World(seed=0)
    y = 5
    w.struct[y, 2:5] = C.RES
    w.land_value[y, 2], w.land_value[y, 3], w.land_value[y, 4] = 0.3, 0.55, 0.8
    w.level[y, 2:5] = 1
    grew = np.zeros_like(w.access)
    grew[y, 2:5] = True
    fx, fy = map(int, np.argwhere(w.resource == 1)[0][::-1])
    w.struct[fy, fx] = C.IND
    w.level[fy, fx] = 1
    grew[fy, fx] = True
    w._assign_variants(grew)
    assert list(w.wealth[y, 2:5]) == [1, 2, 3]
    assert w.ind_type[fy, fx] == 1          # farm on fertile land


def test_zone_density_and_office_zone():
    env = CivitasEnv(GameConfig(seed=0, months=12, milestones=False))
    w = env.world
    y = w.highway[1]
    w.dept_funds["planning"] = 1000
    r = apply_action(w, "planning", {"type": "zone", "zone": "O", "density": "high", "x1": 2, "y1": y + 1, "x2": 5, "y2": y + 1})
    assert r.ok and r.cost == 4 * C.CATALOG["zone_high"]["cost"]
    assert w.struct[y + 1, 3] == C.OFF and w.density[y + 1, 3] == 1
    r = apply_action(w, "planning", {"type": "zone", "zone": "O", "density": "low", "x1": 2, "y1": y + 1, "x2": 5, "y2": y + 1})
    assert r.ok and w.density[y + 1, 3] == 0          # re-zoned to low density
    assert not apply_action(w, "planning", {"type": "zone", "zone": "R", "density": "medium", "x1": 2, "y1": y + 2, "x2": 3, "y2": y + 2}).ok


def test_rent_rises_with_land_value():
    w = World(seed=0)
    y = w.highway[1]
    w.struct[y, 1:20] = C.ROAD
    w.struct[y + 1, 2:6] = C.RES
    w.level[y + 1, 2:6] = 2
    w.wealth[y + 1, 2:6] = 1
    w.refresh()
    w._compute_population()
    w.land_value[y + 1, 2], w.land_value[y + 1, 3] = 0.2, 0.9
    w._household_budget({"workers": 50.0, "employed_by_edu": np.array([20.0, 20.0, 5.0, 0.0])})
    assert w.rent[y + 1, 3] > w.rent[y + 1, 2]
    assert w.disposable[y + 1, 3] < w.disposable[y + 1, 2]
