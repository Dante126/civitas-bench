"""Planning's tool hints (v19, 120-month fleet runs): zones over budget, one zone type per strip, branch roads through
empty lots in one call."""
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
    assert plan[0] == "- zone_R: 2 sites, best (0,25)-(19,26), 40 tiles"
    assert plan[1] == "- zone_C: 2 sites, next free (0,22)-(16,23), 34 tiles (one zone type per strip)"
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
