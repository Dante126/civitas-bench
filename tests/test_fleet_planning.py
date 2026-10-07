"""Planning's tool hints (v19, 120-month fleet runs): zones over budget, one zone type per strip, branch roads through
empty lots in one call."""
import json
import re

from civitas.core.env import CivitasEnv, GameConfig
from civitas.fleet.briefing import briefing
from civitas.fleet.tools import ToolBox, zone_fit
from civitas.sim import catalog as C


def _env():
    return CivitasEnv(GameConfig(seed=0, months=12))


def _ready(env) -> list[str]:
    text = briefing(env, "planning", mark=False)
    return text[text.index("## Ready sites"):].split("\n\n")[0].splitlines()[1:]


def _boxed_in(lots=()):
    """One road across the map (y=24), a level-1 home on every land tile beside it but the empty zoned `lots`."""
    env = _env()
    w = env.world
    w.struct[:], w.anchor[:], w.level[:] = C.NONE, -1, 0
    w.struct[24, 0], w.struct[24, 1:] = C.HIGHWAY, C.ROAD
    for y in (23, 25):
        land = w.terrain[y] == C.LAND
        w.struct[y, land], w.level[y, land] = C.RES, 1
    for x, y in lots:
        w.level[y, x] = 0
    w.refresh()
    w.dept_funds["planning"] = 1000
    return env


def test_a_zone_over_budget_names_the_price_and_a_smaller_zone_that_fits():
    """v19: 43 zones failed on "costs $48 but the planning budget only has $46"; 41 had money for some tiles."""
    env = _env()
    w = env.world
    w.dept_funds["planning"] = 46
    box = ToolBox(env, "planning")
    rect = {"x1": 0, "y1": 25, "x2": 19, "y2": 26}
    res = box.call("zone", {"zone": "I", "density": "low", **rect})
    assert not res["ok"] and res["message"].startswith("costs $320 but the planning budget only has $46"), res
    hint = res["message"][res["message"].index(" - zone I"):]
    assert "costs $8/tile (industry and mixed use are always charged the high-density price)" in hint, hint
    assert "$46 pays for 5 new tiles: zone x1=" in hint and "(5 tiles, $40) now" in hint, hint
    # v23 seed 0: "zone fewer tiles rather than waiting for funds" halved planning's requests; it asks for the rest
    assert hint.endswith("; then ask the mayor for $274 for the rest (send_message)"), hint
    m = re.search(r"zone x1=(\d+),y1=(\d+),x2=(\d+),y2=(\d+)", hint)
    sub = dict(zip(("x1", "y1", "x2", "y2"), map(int, m.groups())))
    assert all(rect[k[0] + "1"] <= sub[k] <= rect[k[0] + "2"] for k in sub)             # inside the request
    assert " - zone I" not in box.executed[-1]["message"]                               # the log keeps the core text
    ok = box.call("zone", {"zone": "I", "density": "low", **sub})
    assert ok["ok"] and ok["cost"] == 40, ok
    w.dept_funds["planning"] = 7
    assert zone_fit(w, {"zone": "I", **rect}, 7) == ""                                   # not one tile


def test_a_zone_over_budget_offers_low_density():
    env = _env()
    w = env.world
    w.unlocked = lambda key: True                                                        # high density open
    w.dept_funds["planning"] = 60
    hint = zone_fit(w, {"zone": "R", "density": "high", "x1": 0, "y1": 25, "x2": 5, "y2": 26}, 60)
    assert "zone R costs $8/tile (high density)" in hint and 'or zone it all with density "low" for $60' in hint, hint


def test_zone_lines_show_one_strip_per_type():
    """v19: "zone_C: best as zone_R; next ...", "zone_I: same strip as zone_R"; planning zoned R, then I on the same
    rectangle 19 times."""
    plan = _ready(_env())
    assert plan[0] == "- zone_R: 2 sites, best (0,25)-(19,26), 40 tiles; $200 - you have $0"    # each strip priced
    assert plan[1] == ("- zone_C: 2 sites, next free (0,22)-(16,23), 34 tiles (one zone type per strip); $170 - you "
                       "have $0")
    assert plan[2] == ("- zone_I: 2 sites, all zone_R's and zone_C's strips (one zone type per strip): extend the "
                       "roads")
    assert not any("same strip as" in ln or "best as" in ln for ln in plan)


def test_a_zone_on_this_turns_own_strip_names_the_next_free_strip():
    env = _env()
    box = ToolBox(env, "planning")
    rect = {"x1": 0, "y1": 25, "x2": 19, "y2": 26}
    env.world.dept_funds["planning"] = 1000
    assert box.call("zone", {"zone": "R", **rect})["ok"]
    res = box.call("zone", {"zone": "C", **rect})
    assert "you zoned them at call 1 this turn; one zone type per strip: next free zone_C strip x1=0,y1=22,x2=16," \
           "y2=23 (34 tiles" in res["message"], res


def test_a_branch_road_through_empty_lots_dezones_them_in_the_same_call():
    """v19: 10 of planning's 29 roads into zoned lots were the briefing's branch road sent without its dezone."""
    road = {"x1": 12, "y1": 23, "x2": 12, "y2": 9}
    env = _boxed_in(lots=[(12, 23)])
    assert any("dezone (12,23) (empty lot, free), then build_road (12,23)-(12,9)" in ln for ln in _ready(env))
    res = ToolBox(env, "planning").call("build_road", road)                           # the scripted fleet's toolbox
    assert not res["ok"] and "dezone x1=12,y1=23,x2=12,y2=23" in res["message"], res
    box = ToolBox(env, "planning", branch_dezone=True)                                 # ToolAgent's
    res = box.call("build_road", road)
    assert res["ok"] and res["message"].startswith("de-zoned 1 tiles at (12,23) first (the branch road's empty "
                                                   "lots, free); built 15 road tiles"), res
    assert [r["action"]["type"] for r in box.executed] == ["dezone", "build_road"]
    assert env.world.struct[23, 12] == C.ROAD and res["cost"] == 150
    env = _boxed_in()                                                                  # a home in the way: no
    res = ToolBox(env, "planning", branch_dezone=True).call("build_road", {"x1": 17, "y1": 23, "x2": 17, "y2": 9})
    assert not res["ok"] and env.world.struct[23, 17] == C.RES
    env = _boxed_in(lots=[(12, 23)])                                                   # short of the road: no
    env.world.dept_funds["planning"] = 100
    res = ToolBox(env, "planning", branch_dezone=True).call("build_road", road)
    assert not res["ok"] and env.world.struct[23, 12] == C.RES


# ------------------------------------------------------------------ planning's Next line (fleet/planning.py)
def _no_ask():
    from civitas.fleet.tools import Ask
    return Ask(False, 0, "", "", "", "", "")


def _lists(env):
    from civitas.fleet.tools import role_obs, site_lists
    return site_lists(env.world, "planning", role_obs(env, "planning"))


def test_a_park_goes_where_it_raises_most_homes_off_the_roadside_lots():
    """120-month runs: 1 park tile in 244 planning turns, no home within 4 tiles of a park, land value 14-19%."""
    from civitas.fleet import planning as P
    env = _boxed_in()
    w = env.world
    site = P.park_site(w)
    assert site is not None and site.tiles == 4 and site.cost == 4 * C.CATALOG["park"]["cost"], site
    assert site.y2 <= 22 or site.y1 >= 26                       # off the roadside lots (rows 23 and 25)
    assert site.homes > 0 and all(w.struct[y, x] == C.NONE for x in range(site.x1, site.x2 + 1)
                                  for y in range(site.y1, site.y2 + 1))
    assert P.park_site(w, funds=200).tiles == 1                 # what the funds pay for, one tile at least
    assert P.park_site(w, funds=0).tiles == 1
    taken = {(x, y) for x in range(site.x1, site.x2 + 1) for y in range(site.y1, site.y2 + 1)}
    other = P.park_site(w, avoid=taken)
    assert not taken & {(x, y) for x in range(other.x1, other.x2 + 1) for y in range(other.y1, other.y2 + 1)}
    assert P.park_site(_env().world) is None                    # no developed home: nothing to raise
    line, = [ln for ln in _ready(env) if ln.startswith("- parks: ")]
    assert line.startswith("- parks: 0 of ") and line.endswith(
        f'build_park {{"x1":{site.x1},"y1":{site.y1},"x2":{site.x2},"y2":{site.y2}}}'), line


def test_the_next_line_puts_parks_first_while_r_demand_is_capped_for_lack_of_them(monkeypatch):
    from civitas.fleet import planning as P
    from civitas.fleet.tools import role_obs
    env = _boxed_in()
    w = env.world
    lists, city = _lists(env), role_obs(env, "planning")["city"]

    def nxt(funds=1000, ask=None):
        v = P.planning_view(env, lists, funds)
        return v, P.planning_next(env, city, v, funds, ask or _no_ask())
    monkeypatch.setattr(P, "parks", lambda w: P.Parks(2, 10, 10, True, False))
    v, line = nxt()
    assert "4 park tiles by " in line and "R demand capped at +0.25 until 10" in line, line
    park, = P.parks_line(v.parks, v.park)                        # one park choice for the parks line and the Next line
    assert park.endswith(P.call("build_park", v.park.args())) and P.call("build_park", v.park.args()) in line
    v, line = nxt(funds=200)                                     # sized to the funds, the same call on both
    assert v.park.tiles == 1 and P.call("build_park", v.park.args()) in line
    assert P.parks_line(v.parks, v.park)[0].endswith(P.call("build_park", v.park.args()))
    v, line = nxt(funds=0)                                       # short: report it, or ask what the items lack
    assert line.endswith("; you have $0: say what you need in your end_turn report"), line
    from civitas.fleet.tools import Ask
    v, line = nxt(funds=0, ask=Ask(True, 600, "a road", "", "", "", ""))
    total = sum(int(c) for c in __import__("re").findall(r"\(\$(\d+)\): ", line))
    assert line.endswith(f"; you have $0: ask the mayor for ${total:,} for these (send_message)") and total != 600, line
    monkeypatch.setattr(P, "parks", lambda w: P.Parks(10, 10, 10, False, False))   # enough parks: none named
    assert "park" not in nxt()[1]
    w.stats["demand_caps"] = {"R": 0.25}
    monkeypatch.setattr(P, "parks", lambda w: P.Parks(2, 10, 10, True, False))
    assert P.cap_line(w) == ["- demand capped: R +0.25: parks 2 of 10 tiles (yours: build_park)"]


def test_the_parks_cap_is_the_worlds_own():
    """Planning reads World.parks_capped / leisure_tiles, the helpers demand_caps uses, on the sim's population."""
    from civitas.fleet import planning as P
    env = _boxed_in()
    w = env.world
    for pop, parks in ((1999, 0), (2000, 4), (2000, 5), (4001, 10), (4001, 11)):
        w.stats["population"] = float(pop)
        w.struct[w.struct == C.PARK] = C.NONE
        free = list(zip(*((w.struct == C.NONE) & (w.terrain == C.LAND)).nonzero()))[:parks]
        for y, x in free:
            w.struct[y, x] = C.PARK
        assert w.leisure_tiles() == parks
        assert P.parks(w).capped == w.parks_capped() == (w.demand_caps()["R"] < 1), (pop, parks)
        assert P.parks(w).by_pop == -(-pop // 400)


def test_the_next_line_zones_the_largest_demand_with_room_on_its_strip_within_the_funds():
    """v19-v23: 134 zones failed on money (131 after the turn's other actions spent it); the zone lines had no price."""
    from civitas.fleet import planning as P
    env = _env()
    w = env.world
    lists = _lists(env)
    assert set(P.strips_shown(lists)) == {"zone_R", "zone_C"}   # zone_I has no strip of its own (one type per strip)
    w.demand.update(R=0.3, C=0.6, I=0.9, O=0.95)
    city = {"zoned_tiles": {"R": 0, "C": 0}, "developed_tiles": {}}
    item, blocked = P.zone_item(w, city, lists, 1000)
    assert blocked == ["I"] and item.call.startswith('zone {"zone":"C","density":"low","x1":0,"y1":22'), item
    assert lists["zone_O"].short.startswith("locked")            # offices locked: neither zoned nor "no free strip"
    assert item.cost == 34 * 5 and "demand +0.60, 0 zoned tiles waiting" in item.text
    city["zoned_tiles"]["C"] = 12                                # 10+ waiting: the next type
    item, _ = P.zone_item(w, city, lists, 1000)
    assert '"zone":"R"' in item.call
    item, _ = P.zone_item(w, city, lists, 23)                    # sized to the funds
    assert item.cost <= 23 and item.text.startswith("zone 4 tiles R"), item
    keep = {(x, 25) for x in range(0, 20)}
    item, _ = P.zone_item(w, city, lists, 1000, avoid=keep)      # sparing funded lots: the other row only
    assert '"y1":26,' in item.call and '"y2":26}' in item.call, item
    city["zoned_tiles"]["C"] = 0                                 # C's whole strip spared: R next, C has no strip
    spared = {(x, y) for x in range(0, 17) for y in (22, 23)}
    item, blocked = P.zone_item(w, city, lists, 1000, avoid=spared)
    assert '"zone":"R"' in item.call and blocked == ["C", "I"], (item, blocked)
    w.demand.update(R=-0.1, C=-0.1, I=-0.1)
    assert P.zone_item(w, city, lists, 1000) == (None, [])       # O locked: no road "for O" either


def test_the_next_line_frees_the_riverbank_lot_utilities_waits_for():
    """v23 seed 0: utilities' outlet waited for a bank lot in 36 of planning's 40 turns; planning inspected 417
    times."""
    from civitas.fleet import planning as P
    from civitas.fleet.briefing import briefing
    from civitas.fleet.tools import dept_lists
    env = _boxed_in()
    w = env.world
    depts = dept_lists(env)
    assert P.bank_purposes(w, depts) == ["sewage", "pump"]
    item = P.bank_item(w, depts)
    assert item.cost == 0 and item.call.startswith("dezone ") and "razes a level-1 home" in item.text, item
    w.level[23, 25] = 0                                           # an empty zoned bank lot goes first
    item = P.bank_item(w, depts)
    assert item.call == 'dezone {"x1":25,"y1":23,"x2":25,"y2":23}' and "(an empty lot)" in item.text, item
    text = briefing(env, "planning", mark=False)
    assert '- Next: free a riverbank lot for utilities\' sewage outlet (an empty lot) (free): dezone {"x1":25,' in text
    res = ToolBox(env, "planning").call("dezone", {"x1": 3, "y1": 3, "x2": 3, "y2": 3})
    assert res["message"].endswith('; the riverbank lot utilities waits for: dezone {"x1":25,"y1":23,"x2":25,"y2":23}')
    assert ToolBox(env, "planning").call("dezone", {"x1": 25, "y1": 23, "x2": 25, "y2": 23})["ok"]
    assert P.bank_item(w, dept_lists(env)) is None               # a free bank lot: nothing waits
    fresh = _env()                                                # no road at the river: the main road reaches it
    item = P.bank_item(fresh.world, dept_lists(fresh), _lists(fresh))
    assert item.road and item.call == 'build_road {"x1":19,"y1":24,"x2":34,"y2":24}' and "leave (" in item.text, item


def test_requests_to_planning_show_with_what_to_do_and_hold_its_end_turn_once():
    """Late smokes: utilities' bank lots, police's jail and health's landfill sites went to planning unanswered."""
    from civitas.fleet.briefing import briefing
    from civitas.fleet.memory import FleetState
    from civitas.fleet.tools import open_requests
    env = _boxed_in()
    w = env.world
    w.level[23, 25] = 0
    env.council.post(0, "utilities", "Please free a riverbank lot near (25,23) for our sewage outlet", "planning")
    env.council.post(0, "health", "We need a landfill site away from homes", "planning")
    env.council.post(0, "fire", "station built, thanks", "planning")                      # no request
    env.end_month()
    fleet = FleetState.of(env)
    assert [r.dept for r in open_requests(env, fleet, to="planning")] == ["utilities", "health"]
    text = briefing(env, "planning", mark=False)
    block = text[text.index("## Open requests to you"):].split("\n\n")[0].splitlines()
    assert block[1].startswith('- utilities: m0 (waiting 1 month): dezone {"x1":') and block[1].endswith(
        '"Please free a riverbank lot near (25,23) for our sewage outlet"'), block
    assert block[2].startswith("- health: m0 (waiting 1 month): "), block
    assert "(2 requests to you: under Open requests to you below)" in text and "station built" in text
    box = ToolBox(env, "planning", max_steps=20)                                # the model's toolbox: held once
    res = box.call("end_turn", {"report": "done"})
    assert res["refused"] == "open_requests" and "utilities (since m0), health (since m0)" in res["error"], res
    assert box.call("end_turn", {"report": "done"})["ok"]
    box = ToolBox(env, "planning", max_steps=20)
    lot = {"x1": 25, "y1": 23, "x2": 25, "y2": 23}
    assert w.struct[23, 25] in C.ZONES and box.call("dezone", lot)["ok"]      # acting at the tile answers it
    assert [r.dept for r in open_requests(env, fleet, box.messages, to="planning")] == ["health"]
    box.call("send_message", {"to": "health", "text": "a lot at the town edge comes with the next road"})
    assert open_requests(env, fleet, box.messages, to="planning") == []
    assert box.call("end_turn", {"report": "done"})["ok"]
    assert ToolBox(env, "planning").call("end_turn", {"report": "x"})["ok"]     # the scripted fleet's: never held


def test_a_road_blocked_by_a_building_names_a_detour():
    """s1 v23: planning sent build_road through the water pump at (29,24) 16 times over 10 turns."""
    env = _env()
    w = env.world
    w.dept_funds["planning"] = 1000
    w.struct[24, 22] = C.PUMP
    box = ToolBox(env, "planning")
    res = box.call("build_road", {"x1": 19, "y1": 24, "x2": 25, "y2": 24})
    assert not res["ok"] and "; around it: build_road {" in res["message"], res
    calls = re.findall(r"build_road (\{[^}]*\})", res["message"].split("around it: ")[1].split(" ($")[0])
    assert len(calls) == 3, calls
    for c in calls:
        assert box.call("build_road", json.loads(c))["ok"]
    assert w.struct[24, 25] == C.ROAD and w.struct[24, 22] == C.PUMP


def test_a_zone_that_leaves_planning_nearly_broke_says_what_the_rest_pays_for():
    env = _env()
    env.world.dept_funds["planning"] = 60
    res = ToolBox(env, "planning").call("zone", {"zone": "R", "x1": 0, "y1": 25, "x2": 3, "y2": 25})
    assert res["ok"] and res["message"].endswith(" ($40 left: 8 more tiles at low density, 5 at high density or "
                                                 "for I and M)"), res
    env.world.dept_funds["planning"] = 1000
    res = ToolBox(env, "planning").call("zone", {"zone": "R", "x1": 4, "y1": 25, "x2": 5, "y2": 25})
    assert res["message"] == "zoned 2 tiles R (low density)"


def test_the_hand_over_from_the_scripted_fleet_clears_its_memory():
    """Late smoke, month 96: planning's memory read "lessons: ... build_park: no empty land tiles there", and the
    scripted fleet's "asked mayor: ..." open tasks are requests the new agent never sent."""
    from types import SimpleNamespace
    from civitas.bench.runner import settle_fleet
    from civitas.fleet.config import FleetAgentConfig
    from civitas.fleet.memory import FleetState
    from civitas.fleet.scripted import ScriptedFleetAgent
    env = _env()
    fleet = FleetState.of(env)
    fleet.update_memory("planning", lessons=["build_park: no empty land tiles there"], open_tasks=["asked mayor"])
    fleet.update_memory("police", lessons=["kept"])
    agent = SimpleNamespace(kind="fleet", role="planning", env=env)
    scripted = ScriptedFleetAgent("planning", cfg=FleetAgentConfig(model="scripted"))
    settle_fleet({"planning": agent}, {"planning": scripted})
    assert fleet.memory["planning"] == {"commitments": [], "open_tasks": [], "lessons": []}
    assert fleet.memory["police"]["lessons"] == ["kept"]
    fleet.update_memory("planning", lessons=["mine"])
    settle_fleet({"planning": agent}, {"planning": agent})        # played as configured: its own memory stays
    assert fleet.memory["planning"]["lessons"] == ["mine"]


def test_failed_dezones_read_the_departments_lists_once_until_an_action_succeeds(monkeypatch):
    import civitas.fleet.tools as T
    env = _boxed_in()
    calls = []
    real = T.dept_lists
    monkeypatch.setattr(T, "dept_lists", lambda e: calls.append(1) or real(e))
    box = ToolBox(env, "planning")
    for x in (3, 4):
        res = box.call("dezone", {"x1": x, "y1": 3, "x2": x, "y2": 3})
        assert "the riverbank lot utilities waits for" in res["message"], res
    assert len(calls) == 1
    assert box.call("dezone", {"x1": 26, "y1": 25, "x2": 26, "y2": 25})["ok"]     # a success: read afresh
    box.call("dezone", {"x1": 5, "y1": 3, "x2": 5, "y2": 3})
    assert len(calls) == 2
