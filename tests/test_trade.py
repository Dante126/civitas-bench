"""Goods & freight, business profitability, service imports, plant ageing, upgrades, harbour."""
import numpy as np

from civitas.core.actions import apply_action
from civitas.core.env import CivitasEnv, GameConfig
from civitas.sim import catalog as C
from civitas.sim.world import World


def _industrial_world():
    w = World(seed=0)
    y = w.highway[1]
    w.struct[y, 1:30] = C.ROAD
    w.struct[y + 1, 2:20] = C.IND
    w.level[y + 1, 2:20] = 3
    w.struct[y - 1, 2:30] = C.RES
    w.level[y - 1, 2:30] = 3
    w.refresh()
    w._compute_population()
    return w, y


def test_industry_exports_through_limited_freight_capacity():
    w, _ = _industrial_world()
    w._compute_trade()
    t = w.trade
    assert t["goods_produced"] > t["goods_needed"] and t["exports"] > 0
    C.FREIGHT_CAPACITY["highway"], old = 10.0, C.FREIGHT_CAPACITY["highway"]
    try:
        w._compute_trade()
        assert w.trade["exports"] <= 10.0 and w.trade["sold_share"] < t["sold_share"]
    finally:
        C.FREIGHT_CAPACITY["highway"] = old


def test_businesses_close_after_months_of_losses():
    w, y = _industrial_world()
    w._compute_trade()
    w.taxes["I"] = 0.20                       # crushing industrial tax
    for _ in range(C.CLOSE_AFTER_MONTHS):
        w._company_profits()
    assert w.closing[y + 1, 5]
    w._grow()
    assert w.level[y + 1, 5] == 0


def test_service_imports_cover_a_shortfall_at_a_price():
    w, y = _industrial_world()                # no power plants at all
    w.service_imports = False
    w.refresh()
    assert w.imports["power"] == 0 and not w.powered[y - 1, 5]
    w.service_imports = True
    w.refresh()
    assert w.imports["power"] > 0 and w.powered.sum() > 0
    assert w.service_import_price("power") > 0


def test_power_plants_age_and_overhaul_restores_them():
    env = CivitasEnv(GameConfig(seed=0, months=12, milestones=False))
    w = env.world
    y = w.highway[1]
    w.dept_funds["utilities"] = 20000
    assert apply_action(w, "utilities", {"type": "build_power_plant", "kind": "coal", "x": 3, "y": y - 1}).ok
    w.month = 12 * 9                          # nine years later
    old = w.plant_output_factor(3, y - 1, "coal")
    assert old < 0.9
    r = apply_action(w, "utilities", {"type": "upgrade_facility", "x": 3, "y": y - 1, "module": "overhaul"})
    assert r.ok and w.plant_output_factor(3, y - 1, "coal") == 1.0
    assert apply_action(w, "utilities", {"type": "upgrade_facility", "x": 3, "y": y - 1, "module": "expansion"}).ok
    assert w.plant_output_factor(3, y - 1, "coal") == C.MODULES["expansion"]["capacity"]
    assert not apply_action(w, "utilities", {"type": "upgrade_facility", "x": 3, "y": y - 1, "module": "expansion"}).ok
    assert not apply_action(w, "police", {"type": "upgrade_facility", "x": 3, "y": y - 1, "module": "filter"}).ok


def test_filter_halves_plant_pollution():
    w, y = _industrial_world()
    w.struct[y - 3, 10] = C.COAL
    w._compute_environment()
    dirty = w.plant_pollution[y - 3, 10]
    w.facility_state[(10, y - 3)] = {"built": 0, "modules": ["filter"]}
    w._compute_environment()
    assert w.plant_pollution[y - 3, 10] < dirty


def test_wind_turbines_depend_on_the_wind_map():
    w = World(seed=0)
    assert w.wind.min() >= 0.5 and w.wind.max() <= 1.6
    (y1, x1), (y2, x2) = np.unravel_index(np.argmin(w.wind), w.wind.shape), np.unravel_index(np.argmax(w.wind), w.wind.shape)
    assert w.plant_output_factor(int(x1), int(y1), "wind") < w.plant_output_factor(int(x2), int(y2), "wind")


def test_harbour_must_be_near_the_river_mouth():
    env = CivitasEnv(GameConfig(seed=0, months=12, milestones=False))
    w = env.world
    w.dept_funds["transport"] = 20000
    y = w.highway[1]
    assert not apply_action(w, "transport", {"type": "build_harbour", "x": 3, "y": y - 1}).ok
