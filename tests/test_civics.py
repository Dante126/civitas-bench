"""Needs-based happiness, complaints, approval / recall, funding levels, policies."""
import numpy as np

from civitas.core.actions import apply_action
from civitas.core.env import CivitasEnv, GameConfig
from civitas.sim import catalog as C
from civitas.sim.world import World


def _homes_world():
    w = World(seed=0)
    y = w.highway[1]
    w.struct[y, 1:30] = C.ROAD
    w.struct[y + 1, 2:30] = C.RES
    w.level[y + 1, 2:30] = 2
    w.refresh()
    w._compute_population()
    return w, y


def test_complaints_name_the_biggest_unmet_need():
    w, _ = _homes_world()          # no power plant: nobody has power
    w.service_imports = False
    w.refresh()
    w._compute_happiness()
    assert w.complaints and w.complaints[0]["need"] in ("power", "water")
    assert any("no power" in c["text"] for c in w.complaints)


def test_funding_scales_coverage_and_upkeep():
    env = CivitasEnv(GameConfig(seed=0, months=12))
    w = env.world
    y = w.highway[1]
    w.struct[y, 1:30] = C.ROAD
    w.struct[y + 1, 2:30] = C.RES
    w.level[y + 1, 2:30] = 3
    w.struct[y - 1, 15] = C.POLICE
    w.refresh()
    w._compute_population()
    w._compute_services()
    full = w.police_cov.sum()
    upkeep = w.dept_upkeep()["police"]
    assert apply_action(w, "mayor", {"type": "set_funding", "department": "police", "percent": 50}).ok
    w._compute_services()
    assert w.police_cov.sum() < full and w.dept_upkeep()["police"] == 0.5 * upkeep
    assert not apply_action(w, "mayor", {"type": "set_funding", "department": "police", "percent": 300}).ok
    assert not apply_action(w, "mayor", {"type": "set_funding", "department": "army", "percent": 100}).ok


def test_policies_have_effects_and_costs():
    w, _ = _homes_world()
    w.stats["population"] = 1000
    before = w._garbage_production().sum()
    w.policies.add("recycling")
    assert w._garbage_production().sum() < before
    assert w.policy_costs() > 0
    env = CivitasEnv(GameConfig(seed=0, months=12))
    assert apply_action(env.world, "mayor", {"type": "enact_policy", "policy": "smoke_detectors"}).ok
    assert not apply_action(env.world, "mayor", {"type": "enact_policy", "policy": "smoke_detectors"}).ok
    assert not apply_action(env.world, "mayor", {"type": "enact_policy", "policy": "martial_law"}).ok
    assert apply_action(env.world, "mayor", {"type": "repeal_policy", "policy": "smoke_detectors"}).ok


def test_tax_hike_costs_approval():
    def approval(rate):
        w, _ = _homes_world()
        w.stats["population"] = 1000
        w._compute_happiness()
        w.tax_changed(w.taxes["R"], rate)
        w.taxes["R"] = rate
        w._update_approval()
        return w.approval
    assert approval(0.15) < approval(0.09)


def test_low_approval_for_a_year_ends_the_game():
    env = CivitasEnv(GameConfig(seed=0, months=60))
    w = env.world
    w.stats["population"] = 1000
    w.approval = 0.1
    for _ in range(C.RECALL_MONTHS):
        w.stats["population"] = 1000
        w.stats["happiness"] = 0.0
        w.taxes = {"R": 0.2, "C": 0.2, "I": 0.2}
        w._update_approval()
    assert w.recalled
    info = env.end_month()
    assert info["terminated"] and env.done
