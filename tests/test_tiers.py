"""Building tiers: ladders, crowding, downtown cover, upgrades in place."""
import numpy as np
import pytest

from civitas.core.actions import apply_action
from civitas.core.env import CivitasEnv, GameConfig
from civitas.sim import catalog as C
from civitas.sim.grid import distance_to
from civitas.sim.world import World


def test_lines_start_with_the_existing_specs_and_are_tagged():
    assert C.TIER_LINES["pump"][0] is C.CATALOG["pump"]
    assert C.TIER_LINES["clinic"][1] is C.SERVICES["hospital"]
    assert C.TIER_LINES["outlet"][1] is C.CATALOG["treatment"]
    assert C.TIER_BY_NAME["treatment"] == ("outlet", 2)
    assert C.TIER_BY_NAME["pumping_station"] == ("pump", 2)
    assert C.BASE_TIER[C.TREATMENT] == 2 and C.BASE_TIER[C.HOSPITAL] == 2 and C.BASE_TIER[C.PUMP] == 1
    for line, tiers in C.TIER_LINES.items():
        for n, t in enumerate(tiers, 1):
            assert t["tier"] == n and t["line"] == line and t["struct"] in C.CHAR
            assert t["dept"] in ("utilities", "police", "fire", "health", "education")
            if n > 1:
                assert t["unlock"] in C.UNLOCK_POP


@pytest.mark.parametrize("line,key", [("pump", "water"), ("tower", "water"), ("coal", "power"), ("gas", "power"),
                                      ("wind", "power"), ("solar", "power"), ("landfill", "collect"),
                                      ("police_station", "capacity"), ("fire_station", "responses"),
                                      ("elementary_school", "seats"), ("high_school", "seats"), ("clinic", "capacity")])
def test_higher_tiers_give_more_per_dollar_of_upkeep(line, key):
    tiers = C.TIER_LINES[line]
    per = [t[key] / t["upkeep"] for t in tiers]
    assert all(b > a for a, b in zip(per, per[1:])), per


def test_footprint_depends_on_tier():
    assert C.footprint(C.PUMP) == (1, 1) and C.footprint(C.PUMP, 3) == (2, 2)
    assert C.footprint(C.HOSPITAL) == (2, 2) and C.footprint(C.CASINO) == (2, 2)


def test_tier_and_old_overrides():
    old2, old1 = C.TIER_LINES["pump"][1]["water"], C.CATALOG["pump"]["water"]
    try:
        C.apply_overrides({"TIERS.pump.2.water": 999, "CATALOG.pump.water": 300})
        assert C.TIER_LINES["pump"][1]["water"] == 999 and C.TIER_LINES["pump"][0]["water"] == 300
    finally:
        C.TIER_LINES["pump"][1]["water"], C.CATALOG["pump"]["water"] = old2, old1


def test_upgrade_cost_and_gain():
    t1, t2 = C.TIER_LINES["pump"][:2]
    assert C.upgrade_cost(t1, t2) == round((t2["cost"] - t1["cost"]) * C.TIER_UPGRADE_PREMIUM)
    assert C.tier_gain(t1, t2) == {"water": t2["water"] - t1["water"]}


def test_nested_overrides_keep_catalog_and_bad_tier_raises():
    old2, old1, cat = C.TIER_LINES["pump"][1]["water"], C.CATALOG["pump"]["water"], C.CATALOG
    try:
        C.apply_overrides({"TIERS": {"pump": {"2": {"water": 777}}}})
        C.apply_overrides({"CATALOG": {"pump": {"water": 301}}})
        assert C.CATALOG is cat and C.TIER_LINES["pump"][0] is C.CATALOG["pump"]
        assert C.TIER_LINES["pump"][1]["water"] == 777 and C.CATALOG["pump"]["water"] == 301
        with pytest.raises(KeyError):
            C.apply_overrides({"TIERS.pump.0.water": 1})
    finally:
        C.TIER_LINES["pump"][1]["water"], C.CATALOG["pump"]["water"] = old2, old1


def test_tier_invariants():
    assert len(C.TIER_BY_NAME) == sum(len(t) for t in C.TIER_LINES.values())
    for s in C.TIERED_STRUCTS:
        assert C.footprint(s) == C.FOOTPRINT.get(s, (1, 1)), s
    for tiers in C.TIER_LINES.values():
        for t in tiers:
            assert t["unlock"] is None or t["unlock"] in C.UNLOCK_POP, t["name"]
    assert C.tier_gain(*C.TIER_LINES["police_station"])["radius"] == 8


def _bank_world(w=None):
    """Straight river at x=31-33; the east bank x=34 is riverbank.

    Roads: the main street (row y) crosses the river to x=36; a road along x=35 north of the street (1x1 bank sites at
    x=34, rows 2..y-1); a road along x=36 south of the street (free 2x2 bank blocks at x=34-35); a stub on row y+8
    from x=34 to 36 (a bank tile at (34, y+7) that touches it).
    """
    w = w or World(seed=0)
    w.terrain[:] = C.LAND
    w.terrain[:, 31:34] = C.WATER
    w.dist_water = distance_to(w.terrain == C.WATER)
    y = w.highway[1]
    w.struct[y, 1:37] = C.ROAD
    w.struct[2:y, 35] = C.ROAD
    w.struct[y + 1:46, 36] = C.ROAD
    w.struct[y + 8, 34:37] = C.ROAD
    w.refresh()
    return w, y


def test_facility_spec_defaults_and_stored_tiers():
    w, y = _bank_world()
    w.struct[5, 34] = C.TREATMENT                 # placed by hand, no facility_state
    assert w.facility_tier(34, 5) == 2 and w.facility_spec(34, 5)["name"] == "treatment"
    w.place_building(C.PUMP, 34, y + 2, tier=3)
    assert w.facility_tier(35, y + 3) == 3        # any tile of the building
    assert w.facility_spec(35, y + 3)["name"] == "water_works"
    assert sorted(w.parts(35, y + 3)) == [(34, y + 2), (34, y + 3), (35, y + 2), (35, y + 3)]
    assert w.facility_spec(0, 0) is None


def test_upgrade_building_moves_up_the_line_and_keeps_state():
    w, y = _bank_world()
    w.place_building(C.PUMP, 34, 5)
    w.facility_state[(34, 5)].update(built=-30, modules=["expansion"])
    nxt = w.upgrade_building(34, 5)
    assert nxt["name"] == "pumping_station"
    assert w.facility_state[(34, 5)] == {"built": -30, "modules": ["expansion"], "tier": 2}
    with pytest.raises(ValueError, match="2x2"):
        w.upgrade_building(34, 5)                 # the x=35 road and the river leave no room


def test_upgrade_grows_into_free_tiles_and_can_move_the_anchor():
    w, y = _bank_world()
    w.place_building(C.PUMP, 34, y + 7, tier=2)
    w.upgrade_building(34, y + 7)                 # row y+8 is road, so it grows north
    assert (34, y + 6) in w.facility_state and (34, y + 7) not in w.facility_state
    assert w._anchor_of(34, y + 7) == (34, y + 6)
    assert w.facility_tier(34, y + 6) == 3
    assert sorted(w.parts(35, y + 7)) == [(34, y + 6), (34, y + 7), (35, y + 6), (35, y + 7)]
    with pytest.raises(ValueError, match="top tier"):
        w.upgrade_building(34, y + 6)


def test_upgrade_can_change_the_building_type():
    w, y = _bank_world()
    w.place_building(C.OUTLET, 34, 8)
    w.upgrade_building(34, 8)
    assert w.struct[8, 34] == C.TREATMENT and w.facility_tier(34, 8) == 2


def test_remove_and_multi_buildings():
    w, y = _bank_world()
    w.place_building(C.PUMP, 34, y + 2, tier=3)
    i = (y + 2) * w.w + 34
    assert [i, 2, 2] in w.multi_buildings()
    lv = w.frame_levels()
    assert lv[y + 2, 34] == lv[y + 3, 35] == 3 and w.level[y + 2, 34] == 0
    w.remove_building(35, y + 3)
    assert (w.struct[y + 2:y + 4, 34:36] == C.NONE).all() and (34, y + 2) not in w.facility_state


def test_place_building_rejects_a_tier_of_another_struct():
    w, y = _bank_world()
    with pytest.raises(ValueError, match="no tier 1"):
        w.place_building(C.TREATMENT, 34, 5, tier=1)       # tier 1 of the outlet line is the outlet
    with pytest.raises(ValueError, match="no tier 4"):
        w.place_building(C.PUMP, 34, 5, tier=4)
    assert w.struct[5, 34] == C.NONE


def test_parts_come_from_the_anchor_grid():
    w, y = _bank_world()
    w.place_building(C.PUMP, 34, y + 2, tier=3)
    w.facility_state[(34, y + 2)] = {"built": 0}          # lost tier: parts() must not shrink
    assert len(w.parts(35, y + 3)) == 4
    w.remove_building(35, y + 3)
    assert (w.struct[y + 2:y + 4, 34:36] == C.NONE).all() and (w.anchor[y + 2:y + 4, 34:36] < 0).all()
    w.place_building(C.CASINO, 37, y + 1)
    w.place_building(C.AIRPORT, 37, y + 4)
    assert len(w.parts(38, y + 2)) == 4 and len(w.parts(39, y + 6)) == 9


def test_road_field_follows_an_in_place_upgrade():
    w, y = _bank_world()
    w.place_building(C.LANDFILL, 34, y + 1, tier=2)
    radius = C.SERVICES["landfill"]["radius"]
    assert w._road_field(((34, y + 1),), radius)[0][y + 2, 35] > 0
    w.upgrade_building(34, y + 1)                          # waste complex: same struct and anchor, 2x2
    assert (34, y + 1) in w.facility_state and w.facility_tier(34, y + 1) == 3
    assert w._road_field(((34, y + 1),), radius)[0][y + 2, 35] == 0


def test_flood_marks_a_multi_tile_utility_once():
    w, y = _bank_world()
    w.place_building(C.PUMP, 34, y + 2, tier=3)
    w._flood({"y1": y + 1, "y2": y + 4})
    off = [k for k, v in w.facility_state.items() if "offline_until" in v]
    assert off == [(34, y + 2)] and w.is_offline(34, y + 2)
    assert " 1 riverside facilities offline" in w.events[-1]


def _street(length=40):
    """A straight street east of the highway (row y). Row y+1 is free; on row y-1 the town hall sits somewhere in
    x=11..27, so tests only use x<=10 there."""
    w = World(seed=0)
    y = w.highway[1]
    w.struct[y, 1:length] = C.ROAD
    w.refresh()
    return w, y


def _out(w, x, y, key="water"):
    return next(p[key] for p in w.plant_list if (p["x"], p["y"]) == (x, y))


def test_crowded_pumps_lose_output():
    w, y = _street()
    full = C.CATALOG["pump"]["water"]
    w.place_building(C.PUMP, 5, y + 1)
    w.refresh()
    assert _out(w, 5, y + 1) == pytest.approx(full)
    for x in (7, 9):
        w.place_building(C.PUMP, x, y + 1)
    w.refresh()
    assert _out(w, 7, y + 1) == pytest.approx(full * (1 - 2 * C.CROWD_STEP))
    for x in (11, 13, 15):
        w.place_building(C.PUMP, x, y + 1)
    w.refresh()
    assert _out(w, 9, y + 1) == pytest.approx(full * C.CROWD_FLOOR)       # 4 neighbours within 4 tiles
    p = next(p for p in w.plant_list if (p["x"], p["y"]) == (9, y + 1))
    assert p["crowd"] == pytest.approx(C.CROWD_FLOOR) and p["neighbours"] == 4
    for x in (6, 8, 10):                                   # 7 neighbours: 1 - 7 * step < floor, so the clamp bites
        w.place_building(C.PUMP, x, y - 1)
    w.refresh()
    p = next(p for p in w.plant_list if (p["x"], p["y"]) == (9, y + 1))
    assert p["neighbours"] == 7 and 1 - 7 * C.CROWD_STEP < C.CROWD_FLOOR
    assert p["water"] == pytest.approx(full * C.CROWD_FLOOR) and p["line"] == "pump"


def test_spread_out_pumps_and_one_big_plant_are_not_crowded():
    w, y = _street()
    w.place_building(C.PUMP, 5, y + 1)
    w.place_building(C.PUMP, 15, y + 1)
    w.place_building(C.PUMP, 25, y + 1, tier=3)
    w.refresh()
    assert _out(w, 5, y + 1) == pytest.approx(C.CATALOG["pump"]["water"])
    assert _out(w, 25, y + 1) == pytest.approx(C.TIER_LINES["pump"][2]["water"])
    assert all("crowd" not in p for p in w.plant_list)


def test_water_works_filter_dirty_river_water():
    w, y = _bank_world()
    w.river_row[:] = 1.0
    w.place_building(C.PUMP, 34, 5)
    w.place_building(C.PUMP, 34, y + 2, tier=3)
    w.refresh()
    q = {(p["x"], p["y"]): p["quality"] for p in w.plant_list if "quality" in p}
    assert q[(34, 5)] == pytest.approx(0.2) and q[(34, y + 2)] == pytest.approx(0.85)


def test_nuclear_plant_needs_cooling_water():
    w, y = _bank_world()
    w.place_building(C.NUCLEAR, 34, y + 2)
    w.refresh()
    w.refresh()
    assert _out(w, 34, y + 2, "power") == 0.0             # no water, no power
    w.place_building(C.PUMP, 34, 5, tier=2)
    w.refresh()
    w.refresh()
    assert _out(w, 34, y + 2, "power") == pytest.approx(C.CATALOG["nuclear"]["power"])
    assert sum(n.water_demand for n in w.networks) >= C.CATALOG["nuclear"]["water_use"]


def test_upkeep_and_pollution_follow_the_tier():
    w, y = _street()
    base = w.dept_upkeep()["utilities"]
    w.place_building(C.COAL, 5, y - 1)
    t1, t2 = C.TIER_LINES["coal"]
    assert w.dept_upkeep()["utilities"] - base == pytest.approx(t1["upkeep"])
    w._compute_environment()
    dirty = w.plant_pollution[y - 1, 5]
    w.upgrade_building(5, y - 1)
    assert w.dept_upkeep()["utilities"] - base == pytest.approx(t2["upkeep"])
    w._compute_environment()
    assert w.plant_pollution[y - 1, 5] < dirty
    assert w.metrics()["tiers"]["coal"] == [0, 1]


def test_a_multi_tile_plant_is_one_entry():
    w, y = _street()
    w.place_building(C.PUMP, 25, y + 1, tier=3)
    w.refresh()
    [p] = w.plant_list
    assert (p["x"], p["y"], p["kind"], p["line"], p["tier"]) == (25, y + 1, "water_works", "pump", 3)
    assert sum(n.water_supply for n in w.networks) == pytest.approx(p["water"])
    assert p["water"] == pytest.approx(C.TIER_LINES["pump"][2]["water"])


def test_cooling_water_is_not_sewage():
    w, y = _bank_world()
    w.place_building(C.NUCLEAR, 34, y + 2)
    w.place_building(C.PUMP, 34, 5, tier=2)
    w.refresh()
    w.refresh()
    assert _out(w, 34, y + 2, "power") > 0                # watered, so running
    water = sum(n.water_demand for n in w.networks)
    sewage = sum(n.sewage_demand for n in w.networks)
    assert water - sewage == pytest.approx(C.CATALOG["nuclear"]["water_use"])


def test_service_upkeep_follows_the_tier():
    w, y = _street()
    base = w.dept_upkeep()["health"]
    w.place_building(C.CLINIC, 5, y + 1)                   # row y+2 is free, so it can grow to a 2x2 hospital
    clinic, hospital = C.TIER_LINES["clinic"][:2]
    assert w.dept_upkeep()["health"] - base == pytest.approx(clinic["upkeep"])
    w.upgrade_building(5, y + 1)
    assert w.struct[y + 2, 6] == C.HOSPITAL and w.facility_tier(5, y + 1) == 2
    assert w.dept_upkeep()["health"] - base == pytest.approx(hospital["upkeep"])


def test_a_multi_tile_polluter_spreads_its_pollution():
    w, y = _street()
    w.place_building(C.LANDFILL, 25, y + 1, tier=3)
    w.pollution_kernel = lambda: [(0, 0, 1.0)]             # no drift: plant_pollution shows each tile's own source
    w._compute_environment()
    share = C.TIER_LINES["landfill"][2]["pollution"] / 4
    tiles = w.plant_pollution[y + 1:y + 3, 25:27]
    assert tiles == pytest.approx(np.full((2, 2), 1.0 - np.exp(-share / 18.0)))
    assert w.plant_pollution.sum() == pytest.approx(tiles.sum())


def test_downtown_needs_tier_two_cover():
    w, y = _street(30)
    w.struct[y + 1, 8:14] = C.RES
    w.level[y + 1, 8:14] = 5
    w.place_building(C.POLICE, 10, y - 1)
    w.refresh()
    w._compute_services()
    assert w.police_cov[y + 1, 10] == pytest.approx(C.T1_DOWNTOWN)
    assert w.stats["downtown_uncovered"]["police"] == 6
    w.level[y + 1, 8:14] = 3                            # not downtown: full tier-1 cover
    w._compute_services()
    assert w.police_cov[y + 1, 10] == pytest.approx(1.0)
    w.level[y + 1, 8:14] = 5
    w.upgrade_building(10, y - 1)                        # police HQ
    w._compute_services()
    assert w.police_cov[y + 1, 10] == pytest.approx(1.0)
    assert w.stats["downtown_uncovered"]["police"] == 0


def test_service_capacity_radius_and_responses_follow_the_tier():
    w, y = _street()
    w.place_building(C.FIRE, 10, y + 1, tier=2)
    w.place_building(C.JAIL, 30, y + 1)
    w.upgrade_building(30, y + 1)                        # prison (2x2)
    w.refresh()
    w._compute_services()
    fire = w.service_info["fire_station"][0]
    assert fire["capacity"] == C.TIER_LINES["fire_station"][1]["capacity"] and fire["name"] == "fire_hq"
    w._update_crime()
    assert w.stats["jail_capacity"] == C.TIER_LINES["jail"][1]["capacity"]
    # radius: 16-19 road steps away the fire HQ still reaches, a tier-1 station (radius 14) does not
    t1, _ = _street()
    t1.place_building(C.FIRE, 10, y + 1)
    t1.refresh()
    t1._compute_services()
    assert (w.fire_cov[y + 1, 26:30] > 0).all() and (t1.fire_cov[y + 1, 26:30] == 0).all()
    # responses: with a fire on every developed tile, the HQ puts out 15 (a station 5)
    w.struct[y + 1, 12:29] = C.RES
    w.level[y + 1, 12:29] = 3
    w._compute_services()
    w.rng = _AlwaysZero()
    w._fire_events()
    assert w.stats["fires"] == 17
    assert w.stats["fires_contained"] == C.TIER_LINES["fire_station"][1]["responses"]


class _AlwaysZero:
    """Stand-in rng: every roll is 0, so every tile at risk catches fire."""
    def random(self, size=None):
        return 0.0 if size is None else np.zeros(size)


def test_a_farther_bigger_facility_fills_the_gap_past_a_small_ones_radius():
    def city(station):
        w, y = _street(60)
        w.place_building(C.POLICE, 2, y + 1, tier=2)     # HQ, radius 22
        if station:
            w.place_building(C.POLICE, 38, y + 1)       # tier-1 station, radius 14
        w.refresh()
        w._compute_services()
        return w.police_cov[y + 1, 20:24]
    hq_only, both = city(False), city(True)
    assert (hq_only > 0).all() and (both >= hq_only - 1e-9).all()


def test_downtown_health_needs_a_hospital():
    w, y = _street(30)
    w.struct[y + 1, 8:14] = C.RES
    w.level[y + 1, 8:14] = 5
    w.place_building(C.CLINIC, 10, y - 1)
    w.refresh()
    w._compute_services()
    assert w.health_cov[y + 1, 10] == pytest.approx(C.T1_DOWNTOWN)
    assert w.stats["downtown_uncovered"]["health"] == 6
    w.place_building(C.HOSPITAL, 3, y - 2)              # 2x2, tier 2 of the clinic line
    w.refresh()
    w._compute_services()
    assert w.health_cov[y + 1, 10] == pytest.approx(1.0)
    assert w.stats["downtown_uncovered"]["health"] == 0


def test_landfill_fill_pct_ignores_recycling_centres():
    w, y = _street()
    w.place_building(C.LANDFILL, 5, y - 1, tier=2)
    w.refresh()
    assert w._landfill_fill_pct() is None
    w.place_building(C.LANDFILL, 8, y - 1)
    w._fstate(8, y - 1)["fill"] = C.SERVICES["landfill"]["fill"] / 2
    assert w._landfill_fill_pct() == pytest.approx(50.0)


def test_a_nearly_full_cemetery_only_takes_what_fits():
    w, y = _street()
    w.place_building(C.CEMETERY, 5, y - 1)
    w.refresh()
    spec = C.SERVICES["cemetery"]
    w._fstate(5, y - 1)["used"] = spec["plots"] - 5
    w.deaths_month = 40.0
    w._deathcare()
    assert w.stats["deathcare_capacity"] == pytest.approx(5.0)
    assert w.death_backlog == pytest.approx(35.0)


def test_recycling_centres_never_fill_and_crematoria_need_no_plots():
    w, y = _street(30)
    w.struct[y + 1, 2:30] = C.RES
    w.level[y + 1, 2:30] = 3
    w.place_building(C.LANDFILL, 5, y - 1, tier=2)
    w.place_building(C.CREMATORIUM, 8, y - 1)
    w.place_building(C.CEMETERY, 3, y - 1)
    w.refresh()
    w._compute_services()
    for _ in range(6):
        w._garbage_and_health()
    assert w.stats["garbage_collected"] > 0
    assert w._fstate(5, y - 1).get("fill", 0.0) == 0.0 and w._full_landfills() == set()
    w.deaths_month = 50.0
    w._deathcare()
    assert w.stats["deathcare_capacity"] >= C.TIER_LINES["cemetery"][1]["rate"]
    assert w.death_backlog == pytest.approx(0.0)
    assert w._fstate(3, y - 1).get("used", 0.0) == 0.0     # the crematorium handled everyone, no plots used


def _funded(w):
    for d in w.dept_funds:
        w.dept_funds[d] = 200000
    return w


def test_higher_tiers_are_gated_by_milestones_and_cost_their_price():
    w, y = _bank_world(World(seed=0, milestones=True))
    _funded(w)
    r = apply_action(w, "utilities", {"type": "build_water", "kind": "pump", "tier": 2, "x": 34, "y": 5})
    assert not r.ok and "1,000 residents" in r.message
    w.milestones_reached.append("Town")
    r = apply_action(w, "utilities", {"type": "build_water", "kind": "pumping_station", "x": 34, "y": 5})
    assert r.ok and r.cost == C.TIER_LINES["pump"][1]["cost"]
    assert w.facility_spec(34, 5)["name"] == "pumping_station"


def test_two_by_two_riverbank_build_and_riverbank_rule():
    w, y = _bank_world()
    _funded(w)
    assert apply_action(w, "utilities", {"type": "build_water", "kind": "water_works", "x": 34, "y": y + 2}).ok
    r = apply_action(w, "utilities", {"type": "build_water", "kind": "pump", "x": 37, "y": y + 6})
    assert not r.ok and "riverbank" in r.message


def test_upgrade_action_charges_the_difference_and_reports_blocks():
    w, y = _bank_world()
    _funded(w)
    assert apply_action(w, "utilities", {"type": "build_water", "kind": "pump", "x": 34, "y": 5}).ok
    r = apply_action(w, "utilities", {"type": "upgrade_facility", "x": 34, "y": 5, "module": "tier"})
    t1, t2 = C.TIER_LINES["pump"][:2]
    assert r.ok and r.cost == C.upgrade_cost(t1, t2) and "pumping station" in r.message
    r = apply_action(w, "utilities", {"type": "upgrade_facility", "x": 34, "y": 5, "module": "tier"})
    assert not r.ok and "2x2" in r.message
    r = apply_action(w, "police", {"type": "upgrade_facility", "x": 34, "y": 5, "module": "tier"})
    assert not r.ok


def test_old_kinds_and_service_actions_still_work():
    w, y = _bank_world()
    _funded(w)
    assert apply_action(w, "utilities", {"type": "build_sewage", "kind": "treatment", "x": 34, "y": 11}).ok
    assert w.facility_tier(34, 11) == 2
    r = apply_action(w, "health", {"type": "build_hospital", "x": 37, "y": y + 2})
    assert r.ok and w.facility_spec(37, y + 2)["name"] == "hospital"
    r = apply_action(w, "health", {"type": "build_hospital", "tier": 3, "x": 37, "y": y + 5})
    assert r.ok and w.facility_spec(37, y + 5)["name"] == "medical_centre"
    r = apply_action(w, "education", {"type": "build_school", "kind": "elementary", "tier": 2, "x": 37, "y": y + 9})
    assert r.ok and w.facility_spec(37, y + 9)["name"] == "large_elementary"
    assert apply_action(w, "police", {"type": "build_police_station", "tier": 2, "x": 37, "y": y + 11}).ok


def test_health_upgrades_landfills_and_cemeteries():
    w, y = _bank_world()
    _funded(w)
    assert apply_action(w, "health", {"type": "build_cemetery", "x": 37, "y": y + 2}).ok
    assert apply_action(w, "health", {"type": "upgrade_facility", "x": 37, "y": y + 2, "module": "tier"}).ok
    assert w.struct[y + 2, 37] == C.CREMATORIUM
    assert apply_action(w, "health", {"type": "demolish", "x": 37, "y": y + 2}).ok


def test_action_docs_list_tiers_and_crowding():
    from civitas.core.actions import action_docs
    docs = action_docs()
    assert "pumping_station" in docs["build_water"][1] and "crowd" in docs["build_water"][1].lower()
    assert '"tier": n' in docs["upgrade_facility"][1] and 'module "tier"' in docs["upgrade_facility"][1]
    assert "police_hq" in docs["build_police_station"][1]


def test_messy_kind_and_tier_inputs():
    w, y = _bank_world()
    _funded(w)
    assert apply_action(w, "utilities", {"type": "build_water", "kind": "pump", "tier": None, "x": 34, "y": 3}).ok
    assert w.facility_tier(34, 3) == 1
    assert apply_action(w, "utilities", {"type": "build_water", "kind": "pump", "tier": "2", "x": 34, "y": 5}).ok
    assert w.facility_spec(34, 5)["name"] == "pumping_station"
    assert apply_action(w, "utilities", {"type": "build_water", "kind": "Pumping Station", "x": 34, "y": 7}).ok
    assert w.facility_spec(34, 7)["name"] == "pumping_station"
    r = apply_action(w, "utilities", {"type": "build_sewage", "kind": "treatment", "tier": 1, "x": 34, "y": 9})
    assert not r.ok and 'drop "tier"' in r.message
    r = apply_action(w, "utilities", {"type": "build_water", "kind": "pump", "tier": 1e400, "x": 34, "y": 9})
    assert not r.ok and "tier" in r.message
    r = apply_action(w, "utilities", {"type": "build_power_plant", "kind": "hydro", "tier": 2, "x": 32, "y": 9})
    assert not r.ok and "single tier" in r.message and "1-1" not in r.message
    r = apply_action(w, "health", {"type": "build_hospital", "kind": "medical_centre", "x": 37, "y": y + 2})
    assert r.ok and w.facility_tier(37, y + 2) == 3
    for i, kind in enumerate(("large elementary", "Large Elementary School", "high school ")):
        r = apply_action(w, "education", {"type": "build_school", "kind": kind, "x": 37, "y": y + 5 + 2 * i})
        assert r.ok, (kind, r.message)
    assert w.facility_spec(37, y + 5)["name"] == w.facility_spec(37, y + 7)["name"] == "large_elementary"
    assert w.facility_spec(37, y + 9)["name"] == "high_school"


def test_hydro_errors_keep_the_real_reason():
    w, y = _bank_world()
    _funded(w)
    r = apply_action(w, "utilities", {"type": "build_power_plant", "kind": "hydro", "x": 32, "y": 5})
    assert not r.ok and "does not touch a road" in r.message and "open river" in r.message
    r = apply_action(w, "utilities", {"type": "build_power_plant", "kind": "hydro", "x": 32, "y": y})   # the bridge
    assert not r.ok and "not an empty river tile" in r.message


def test_a_locked_tier_upgrade_changes_nothing():
    w, y = _bank_world(World(seed=0, milestones=True))
    _funded(w)
    assert apply_action(w, "utilities", {"type": "build_water", "kind": "pump", "x": 34, "y": 5}).ok
    before = w.dept_funds["utilities"]
    r = apply_action(w, "utilities", {"type": "upgrade_facility", "x": 34, "y": 5, "module": "tier"})
    assert not r.ok and "1,000 residents" in r.message
    assert w.dept_funds["utilities"] == before and w.facility_tier(34, 5) == 1


def test_tier_without_module_upgrades_one_step_and_modules_cost_the_tier_price():
    w, y = _bank_world()
    _funded(w)
    assert apply_action(w, "utilities", {"type": "build_water", "kind": "pump", "x": 34, "y": 5}).ok
    r = apply_action(w, "utilities", {"type": "upgrade_facility", "x": 34, "y": 5, "tier": 3})
    assert r.ok and w.facility_tier(34, 5) == 2 and "one tier at a time" in r.message
    r = apply_action(w, "utilities", {"type": "upgrade_facility", "x": 34, "y": 5, "module": "expansion"})
    assert r.ok and r.cost == round(C.TIER_LINES["pump"][1]["cost"] * C.MODULES["expansion"]["cost_share"])


def test_a_two_by_two_riverbank_build_has_one_anchor():
    w, y = _bank_world()
    _funded(w)
    assert apply_action(w, "utilities", {"type": "build_water", "kind": "water_works", "x": 34, "y": y + 2}).ok
    tiles = w.parts(35, y + 3)
    assert sorted(tiles) == [(34, y + 2), (34, y + 3), (35, y + 2), (35, y + 3)]
    assert {int(w.anchor[ty, tx]) for tx, ty in tiles} == {(y + 2) * w.w + 34}
    assert (34, y + 2) in w.facility_state and not any(k in w.facility_state for k in tiles[1:])


def test_frames_carry_tiers_and_multi_tile_upgrades():
    env = CivitasEnv(GameConfig(seed=0, months=12, milestones=False))
    w, y = _bank_world(env.world)
    w.place_building(C.PUMP, 34, y + 2, tier=3)
    w.place_building(C.POLICE, 37, y + 5, tier=2)
    f = env.frame()
    i = (y + 2) * w.w + 34
    assert f["levels"][i] == "3" and f["levels"][i + w.w + 1] == "3"
    assert f["levels"][(y + 5) * w.w + 37] == "2"
    assert [i, 2, 2] in f["multi"]


def test_prompts_show_tiers_crowding_upgrades_and_downtown():
    env = CivitasEnv(GameConfig(seed=0, months=12, milestones=False))
    w, y = _bank_world(env.world)
    for yy in (3, 5, 7):
        w.place_building(C.PUMP, 34, yy)
    w.place_building(C.POLICE, 37, y + 1)
    w.struct[y + 3:y + 8, 37] = C.RES                # a column of towers east of the x=36 road
    w.level[y + 3:y + 8, 37] = 5
    w.refresh()
    w._compute_services()
    text = env.observe("utilities")["text"]
    assert "crowded plants (output kept): 0.7 x3" in text and "pumping station" in text.lower()
    obs = env.observe("police")
    assert obs["police"]["downtown_uncovered"] == 5
    assert obs["police"]["upgrades"][0]["to"] == "police_hq"
    assert "police hq" in obs["text"].lower()


def test_upgrade_options_filters_orders_and_collapses():
    w, y = _bank_world(World(seed=0, milestones=True))
    w.place_building(C.PUMP, 34, 5)
    assert w.upgrade_options([C.PUMP]) == []                      # pumping station still locked
    w, y = _bank_world()
    w.place_building(C.PUMP, 34, 5, tier=2)                       # x=35 road + river: no room for water works
    w.place_building(C.PUMP, 35, y + 3, tier=2)                   # room to grow into x=34
    assert [(o["x"], o["y"], o["to"]) for o in w.upgrade_options([C.PUMP])] == [(35, y + 3, "water_works")]
    spots = [(37, y + 1), (37, y + 3), (37, y + 5)]
    for x, yy in spots:
        w.place_building(C.POLICE, x, yy)
    need = dict(zip(spots, (1, 5, 3)))
    got = lambda **kw: [(o["y"], o.get("similar")) for o in w.upgrade_options([C.POLICE], need=need, **kw)]
    assert got(per_pair=False) == [(y + 3, None), (y + 5, None), (y + 1, None)]
    assert got(per_pair=False, min_need=1) == [(y + 3, None), (y + 5, None)]     # strictly greater
    assert got(per_pair=False, min_need=3) == [(y + 3, None)]
    assert got() == [(y + 3, 2)]


def test_a_crowded_plant_offers_its_effective_gain():
    w, y = _bank_world()
    for yy in (3, 5, 7):
        w.place_building(C.PUMP, 34, yy)
    w.refresh()
    o = next(o for o in w.upgrade_options([C.PUMP], per_pair=False) if o["y"] == 5)
    f = w.crowd[(34, 5)][0] * w.plant_output_factor(34, 5, "pump") * w.funding_eff("utilities")
    assert o["nameplate"] == {"water": 400} and o["gain"] == {"water": round(400 * f)} and o["gain"]["water"] < 400


def test_health_upgrades_shape_and_education_skips_schools_with_room():
    env = CivitasEnv(GameConfig(seed=0, months=12, milestones=False))
    w, y = _bank_world(env.world)
    assert set(env.observe("health")["health"]["upgrades"]) == {"care", "garbage", "deathcare"}
    w.place_building(C.SCHOOL, 37, y + 1)
    w.place_building(C.SCHOOL, 37, y + 6)
    w.refresh()
    w._compute_services()
    for s, extra in zip(w.service_info["elementary_school"], (-100, 100)):
        s["load"] = s["capacity"] + extra
    ups = env.observe("education")["education"]["upgrades"]
    full = w.service_info["elementary_school"][1]
    assert [(o["x"], o["y"], o["need"]) for o in ups] == [(full["x"], full["y"], 100.0)]


def test_busy_landfills_get_recycling_offers_before_they_fill():
    from civitas.core import observation as O
    env = CivitasEnv(GameConfig(seed=0, months=12, milestones=False))
    w, y = _bank_world(env.world)
    w.place_building(C.LANDFILL, 37, y + 1)
    w.place_building(C.LANDFILL, 37, y + 6)
    w.refresh()
    w._compute_services()
    busy, idle = w.service_info["landfill"][:2]
    busy["load"], idle["load"] = O.GARBAGE_BUSY_LOAD * busy["capacity"], 0.1 * idle["capacity"]
    busy["fill_pct"] = idle["fill_pct"] = 5.0
    ups = env.observe("health")["health"]["upgrades"]["garbage"]
    assert [(o["x"], o["y"], o["to"]) for o in ups] == [(busy["x"], busy["y"], "recycling_centre")]
    assert ups[0]["need"] == pytest.approx(O.GARBAGE_BUSY_RANK + 100 * O.GARBAGE_BUSY_LOAD, abs=0.1)




def _crowd_ranked(w, before, after, line, target):
    """Ranking the old and new lists together puts the target after the uncrowded sites with crowd >= 1; the new
    list starts uncrowded and has every uncrowded site before every crowded one (the target may drop out of it)."""
    from civitas.core import observation as O
    pool = [s for s in before if w.struct[s["y"], s["x"]] == C.NONE] + [s for s in after if s not in before]
    ranked = O._uncrowded_first(w, [{k: v for k, v in s.items() if k != "crowd"} for s in pool], line, k=len(pool))
    at = [(s["x"], s["y"]) for s in ranked].index(target)
    assert ranked[at]["crowd"] >= 1 and at > 0
    for lst in (ranked, after):
        counts = [s.get("crowd", 0) for s in lst]
        assert counts == sorted(counts) and counts[0] == 0
        assert all(s["crowd"] > 0 for s in lst if "crowd" in s)
    assert (after[0]["x"], after[0]["y"]) != target
    assert all(s["crowd"] >= 1 for s in after if (s["x"], s["y"]) == target)
    return ranked


def _free_near(w, x, y):
    """An empty land tile touching a road next to (x, y)."""
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            tx, ty = x + dx, y + dy
            if ((dx or dy) and w.in_bounds(tx, ty) and w.struct[ty, tx] == C.NONE
                    and w.terrain[ty, tx] == C.LAND and w.adjacent_network(tx, ty) >= 0):
                return tx, ty
    raise AssertionError("no free tile")


def test_windy_sites_rank_crowded_spots_last():
    from civitas.core import observation as O
    env = CivitasEnv(GameConfig(seed=0, months=12, milestones=False))
    w, y = _street()
    env.world = w
    before = O._candidate_sites(w)["windy"]
    top = before[0]
    assert "crowd" not in top
    w.place_building(C.WIND, *_free_near(w, top["x"], top["y"]))
    w.refresh()
    ranked = _crowd_ranked(w, before, O._candidate_sites(w)["windy"], "wind", (top["x"], top["y"]))
    obs = env.observe("utilities")
    obs["candidate_sites"]["windy"] = ranked
    n = next(s["crowd"] for s in ranked if (s["x"], s["y"]) == (top["x"], top["y"]))
    assert f"({top['x']},{top['y']}) wind={top['wind']} near {n} turbine" in O.render_text(obs)


@pytest.mark.parametrize("key,struct,line,noun", [("riverbank_next_to_road", C.PUMP, "pump", "intake"),
                                                  ("riverbank_for_sewage", C.OUTLET, "outlet", "outfall")])
def test_riverbank_sites_rank_crowded_spots_last(key, struct, line, noun):
    from civitas.core import observation as O
    env = CivitasEnv(GameConfig(seed=0, months=12, milestones=False))
    w, y = _bank_world(env.world)
    before = O._candidate_sites(w)[key]
    top = before[0]
    assert "crowd" not in top
    w.place_building(struct, *_free_near(w, top["x"], top["y"]))
    w.refresh()
    ranked = _crowd_ranked(w, before, O._candidate_sites(w)[key], line, (top["x"], top["y"]))
    obs = env.observe("utilities")
    obs["candidate_sites"][key] = ranked
    assert f"({top['x']},{top['y']}) pollution={top['pollution']} near 1 {noun}" in O.render_text(obs)


def test_two_by_two_site_hints_test_the_whole_block():
    """West bank (x=30): a 2x2 block there has its top-left at x=29, off the bank. Above the main street the
    top-left doesn't touch the road either; the action accepts both, so the site hints must too."""
    w, y = _bank_world()
    _funded(w)
    for (nx, ny), site in (((29, y - 3), (29, y - 2)), ((29, y + 2), (29, y + 1))):
        need = np.zeros(w.struct.shape)
        need[ny, nx] = 1.0
        got = w.site_candidates(need, radius=1, k=3, riverbank=True, size=2)
        assert [(s["x"], s["y"]) for s in got] == [site]
        assert w.dist_water[site[1], site[0]] > 1
    assert w.adjacent_network(29, y - 2) < 0
    r = apply_action(w, "utilities", {"type": "build_water", "kind": "water_works", "x": 29, "y": y - 2})
    assert r.ok, r.message
    one = w.site_candidates(np.ones(w.struct.shape), radius=1, k=50, riverbank=True)
    assert one and all(w.dist_water[s["y"], s["x"]] == 1 and w.adjacent_network(s["x"], s["y"]) >= 0 for s in one)


def test_soil_is_per_building_and_follows_the_tier():
    base = C.SOIL_SOURCE[C.LANDFILL]
    l1, l2, l3 = (t["pollution"] for t in C.TIER_LINES["landfill"])
    for tier, want in ((1, base), (2, base * l2 / l1), (3, base * l3 / l1)):
        w, y = _bank_world()
        w.place_building(C.LANDFILL, 37, y + 2, tier=tier)
        w.soil[:] = 0.0
        w._update_soil()
        assert w.soil.sum() == pytest.approx(want), tier
    w, y = _bank_world()
    w.place_building(C.OUTLET, 34, 5)
    w.soil[:] = 0.0
    w._update_soil()
    assert w.soil.sum() == pytest.approx(C.SOIL_SOURCE[C.OUTLET])
