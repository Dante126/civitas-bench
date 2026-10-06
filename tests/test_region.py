"""Two cities on one river: shared council, river coupling, neighbour deals."""
from civitas.core.decision import Decision
from civitas.core.env import GameConfig
from civitas.core.region import RegionEnv


def _env(months=6):
    return RegionEnv(GameConfig(seed=0, months=months, region=True, events="none"))


def test_roles_are_namespaced_and_rivers_connect():
    env = _env()
    assert env.roles[:2] == ["north.mayor", "south.mayor"] and len(env.roles) == 16
    n, s = env.envs["north"].world, env.envs["south"].world
    assert abs(n.river_path[-1] - s.river_path[0]) <= 2.5          # the river runs on into the south city
    obs = env.observe("south.utilities")
    assert "REGION MODE" in obs["text"] and obs["region"]["cities"][0]["city"] == "north"


def test_upstream_pollution_reaches_the_south_city():
    env = _env()
    env.envs["north"].world.river_outflow = 500.0
    env.end_month()
    assert env.envs["south"].world.upstream_inflow == 500.0
    assert env.envs["south"].world.river_row[0] > 0.5


def test_team_messages_and_cross_city_messages():
    env = _env()
    env.apply("north.planning", Decision(messages=[{"to": "mayor", "text": "fund roads"},
                                                   {"to": "south", "text": "hello neighbours"}]))
    north_mayor = [m["text"] for m in env.observe("north.mayor")["council"]]
    south_mayor = [m["text"] for m in env.observe("south.mayor")["council"]]
    assert "fund roads" in north_mayor and "fund roads" not in south_mayor
    assert "hello neighbours" in south_mayor


def _coal(env, city):
    """A coal plant (250 power) in `city`, on the first site utilities' find_sites names."""
    from civitas.fleet.tools import ToolBox
    e = env.envs[city]
    e.world.dept_funds["utilities"] = 10000
    box = ToolBox(e, "utilities")
    site = box.call("find_sites", {"purpose": "coal_or_gas"})["sites"][0]
    assert box.call("build_power_plant", {"kind": "coal", "x": site["x"], "y": site["y"]})["ok"]


def test_neighbour_deal_moves_capacity_and_money():
    env = _env()
    r = env.apply("north.mayor", Decision(actions=[{"type": "offer_deal", "resource": "power", "amount": 50,
                                                     "price": 1.0, "months": 3, "sell": True}]))
    assert r[-1]["ok"] and r[-1]["message"] == ("proposed deal #1 (you sell 50 power to south at $1.0/unit/month for 3 "
                                                  "months); the south mayor must accept it by month 1")
    # north has no plant yet: the deal would sell capacity it doesn't have
    no = env.apply("south.mayor", Decision(actions=[{"type": "accept_deal", "deal": 1}]))[-1]
    assert not no["ok"] and no["message"].startswith("north no longer has any power to sell: its own supply 0")
    _coal(env, "north")
    assert not env.apply("north.mayor", Decision(actions=[{"type": "accept_deal", "deal": 1}]))[-1]["ok"]
    assert not env.apply("south.planning", Decision(actions=[{"type": "accept_deal", "deal": 1}]))[-1]["ok"]
    assert env.apply("south.mayor", Decision(actions=[{"type": "accept_deal", "deal": 1}]))[-1]["ok"]
    env.end_month()
    n, s = env.envs["north"].world, env.envs["south"].world
    assert s.external["power"] == 50 and n.external["power"] == -50
    assert n.history[-1]["income_breakdown"]["neighbour_deals"] == 50.0


def test_region_episode_through_the_runner(tmp_path):
    from civitas.bench.runner import run_episode
    from civitas.config import apply_cli_overrides, load_config
    cfg = apply_cli_overrides(load_config("configs/region.toml"), [], ["game.months=12"])
    s = run_episode(cfg, 0, tmp_path, verbose=False)
    assert s["months_played"] == 12
    assert {"north.city", "south.city", "city", "north.mayor", "south.education"} <= set(s["final_scores"])



def _linked_pair():
    """North and south with a highway link between them (both built their side)."""
    from civitas.core.actions import apply_action
    from civitas.sim import catalog as C
    env = RegionEnv(GameConfig(seed=0, months=24, region=True, events="none", milestones=False))
    n, s = env.envs["north"].world, env.envs["south"].world
    for w, y_edge, y_road in ((n, 47, 46), (s, 0, 1)):
        w.struct[y_road, 5:10] = C.ROAD
        w.refresh()
        w.dept_funds["transport"] = 10000
    r1 = apply_action(n, "transport", {"type": "build_connector", "kind": "highway", "to": "south", "x": 6, "y": 47})
    assert r1.ok, r1.message
    assert env.compute_links() == []                               # only one side built so far
    r2 = apply_action(s, "transport", {"type": "build_connector", "kind": "highway", "to": "north", "x": 6, "y": 0})
    assert r2.ok, r2.message
    return env, n, s


def test_links_open_when_both_cities_build_their_side():
    env, n, s = _linked_pair()
    links = env.compute_links()
    assert [(l["a"], l["b"], l["kind"]) for l in links] == [("north", "south", "highway")]


def test_connector_must_face_the_partner_city():
    from civitas.core.actions import apply_action
    env = RegionEnv(GameConfig(seed=0, months=24, region=True, events="none", milestones=False))
    n = env.envs["north"].world
    n.dept_funds["transport"] = 10000
    assert not apply_action(n, "transport", {"type": "build_connector", "kind": "highway", "to": "south", "x": 6, "y": 0}).ok
    assert not apply_action(n, "transport", {"type": "build_connector", "kind": "highway", "to": "mars", "x": 6, "y": 47}).ok


def test_goods_and_commuters_flow_over_links():
    env, n, s = _linked_pair()
    n.trade.update({"goods_made": 500.0, "goods_needed": 100.0})
    s.trade.update({"goods_made": 0.0, "goods_needed": 300.0})
    n.stats.update({"unemployed": 200.0, "unfilled_jobs": 0.0})
    s.stats.update({"unemployed": 0.0, "unfilled_jobs": 300.0})
    env._run_flows()
    assert n.regional["goods_out"] == 300.0 and s.regional["goods_in"] == 300.0 and n.regional["cash"] > 0
    assert n.regional["commuters_out"] > 0 and s.regional["commuters_in"] == n.regional["commuters_out"]


def test_three_city_region_layout():
    env = RegionEnv(GameConfig(seed=0, months=3, region=True, events="none", cities=["north", "south", "east"]))
    assert env.neighbors("north") == {"south": "south", "east": "east"}
    assert env.neighbors("east") == {"north": "west"}
    assert env.river_chain == ["north", "south"] and len(env.roles) == 24
    env.end_month()


def test_casino_brings_gamblers_and_crime():
    from civitas.core.actions import apply_action
    from civitas.core.env import CivitasEnv
    from civitas.sim import catalog as C
    env = CivitasEnv(GameConfig(seed=0, months=12, milestones=False, events="none"))
    w = env.world
    y = w.highway[1]
    w.struct[y + 1, 2:20] = C.RES
    w.level[y + 1, 2:20] = 3
    w.refresh()
    w._compute_population()
    w._compute_services()
    for _ in range(6):
        w._update_crime()
    calm = w.crime[y + 1, 12]
    w.dept_funds["planning"] = 50000
    r = apply_action(w, "planning", {"type": "build_casino", "x": 10, "y": y - 2})
    assert r.ok, r.message
    w._compute_population()
    for _ in range(6):
        w._update_crime()
    w._compute_landmarks()
    assert w.crime[y + 1, 12] > calm + 0.05
    assert w.stats["casino_tourists"] > 0 and w.stats["jobs_casino"] == C.CASINO_SPEC["jobs"]


def test_connector_builds_a_feeder_road_inland():
    from civitas.core.actions import apply_action
    from civitas.sim import catalog as C
    env = RegionEnv(GameConfig(seed=0, months=24, region=True, events="none", milestones=False))
    n = env.envs["north"].world
    n.struct[40, 5:10] = C.ROAD                 # road 7 tiles from the south edge
    n.refresh()
    n.dept_funds["transport"] = 10000
    r = apply_action(n, "transport", {"type": "build_connector", "kind": "highway", "to": "south", "x": 6, "y": 47})
    assert r.ok, r.message
    assert (n.struct[41:47, 6] == C.ROAD).all() and n.struct[47, 6] == C.HIGHWAY
