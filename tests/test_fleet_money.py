"""Mayor money (docs/plans/2026-09-29-mayor-money.md): the money model, the mayor's briefing, the money guards and
the money results. Each unit's tests sit under its own header."""
from types import SimpleNamespace

from civitas.core.actions import apply_action
from civitas.core.env import CivitasEnv, GameConfig
from civitas.fleet import tools as T
from civitas.fleet.memory import FleetState
from civitas.fleet.tools import (RESERVE_MIN, ROAD_NAME, DeptMoney, FundItem, Funding, Need, cash_flow,
                                 funding_gaps, reserve, reserve_why, site_price, spend_lines, still_to_fund,
                                 upkeep_of)
from civitas.sim import catalog as C


def _env(**kw):
    return CivitasEnv(GameConfig(seed=0, months=12, **kw))


def _months_spent(env, *spent: dict) -> None:
    """A month passes per dict, with what the departments spent in it (the ledger the history's dept_spent records)."""
    for s in spent:
        env.world.ledger["spent"].update(s)
        env.end_month()


# --- Unit A: money model (tools.py: upkeep, DeptMoney, Funding, reserve, cash_flow, spend_lines)
N, FB = Need, T.FirstBuild
ROAD = N(T.a_or_an(ROAD_NAME), 1, None, "opens 20 tiles")


def _builds(monkeypatch, mix: dict, road=None) -> None:
    """first_build / road_build as `mix` says (dept -> FirstBuild); a department left out has nothing ready."""
    monkeypatch.setattr(T, "first_build", lambda env, dept: mix.get(dept))
    monkeypatch.setattr(T, "road_build", lambda env: road)


def test_needs_lines_carry_upkeep():
    env = _env()
    w = env.world
    assert [upkeep_of(w, d, n) for d, n in (("police", "police_station"), ("health", "clinic"),
                                            ("education", "elementary_school"), ("health", "landfill"))] == [
        55, 40, 45, 30]
    assert upkeep_of(w, "utilities", "water_works") == C.TIER_LINES["pump"][2]["upkeep"]        # a tier 3
    assert upkeep_of(w, "transport", "bus_line") == C.BUS_LINE_UPKEEP
    assert upkeep_of(w, "transport", "train_station") == C.TRAIN_STATION["upkeep"]
    assert upkeep_of(w, "fire", "levee_tile") == C.CATALOG["levee"]["upkeep"]                   # per tile
    assert upkeep_of(w, "transport", "avenue_tile") == C.ROAD_UPKEEP[2] - C.ROAD_UPKEEP[1]      # on the street's
    assert site_price(w, "police", "station").upkeep == 55
    old = C.TIER_LINES["police_station"][0]["upkeep"]
    try:
        C.apply_overrides({"TIERS.police_station.1.upkeep": 70})        # read at call time: overrides reach it
        assert upkeep_of(w, "police", "police_station") == 70
        w.funding["police"] = 0.5                                     # as World.dept_upkeep applies funding
        assert upkeep_of(w, "police", "police_station") == 35 and site_price(w, "police", "station").upkeep == 35
        assert upkeep_of(w, "fire", "fire_station") == 55
    finally:
        C.TIER_LINES["police_station"][0]["upkeep"] = old
        w.funding["police"] = 1.0
    # inside the parentheses; the road (no building) and an upkeep of 0 add nothing
    assert N("police station", 5, 0, first=True, upkeep=55).text() == "its first police station (5 sites, $55/month)"
    assert N("clinic", 6, 1, "coverage 21%", upkeep=40).text() == "another clinic (coverage 21%, $40/month)"
    assert N("clinic", 1, 1, upkeep=40).text() == "another clinic (1 site, $40/month)"
    assert N("3 avenue tiles", 1, None, upkeep=4.5).text() == "3 avenue tiles (1 site, $4/month)"
    assert N("police station", 5, 0, first=True).text() == "its first police station (5 sites)"
    assert ROAD.text() == "a road into open land and zoning it (opens 20 tiles)"
    assert [n.label for n in (N("police station", 5, 0), N("clinic", 6, 1), ROAD)] == [
        "its first police station", "another clinic", "a road into open land and zoning it"]
    # the fresh map: first_build fills Need.upkeep, road_build doesn't (the road Need gets 0)
    tower, = T.first_build(env, "utilities").named
    road, = T.road_build(env).named
    assert (tower.what, tower.upkeep, road.upkeep) == ("water tower", C.TIER_LINES["tower"][0]["upkeep"], 0)
    f = funding_gaps(env, FleetState.of(env))
    assert f.to_fund == [FundItem("utilities", "first/supply", 2200, 30, "its first water tower"),
                         FundItem("health", "first/supply", 2000, 30, "its first landfill"),
                         FundItem("planning", "growth", 600, 0, "a road into open land and zoning it")]


def test_need_kinds_and_order(monkeypatch):
    env = _env()
    w = env.world
    _months_spent(env, {"health": 1000})                # health keeps $2,000 of $9,000: idle
    w.dept_funds["health"] = 9000
    _builds(monkeypatch, {
        "utilities": FB(1200, (N("sewage outlet", 2, 1, "sewage 95% used", upkeep=15),), 1200, ()),  # supply busy
        "education": FB(2500, (N("elementary school", 3, 0, first=True, upkeep=45),), 2500, ()),
        "police": FB(2500, (N("police station", 4, 1, "coverage 40%", upkeep=55),), 2500, ()),
        "fire": FB(None, (), 2500, ("a fire station",))},                                             # blocked
        road=FB(315, (ROAD,), 315, ()))
    f = funding_gaps(env, FleetState.of(env))
    assert {d: m.kind for d, m in f.depts.items()} == {
        "planning": "growth", "utilities": "first/supply", "transport": "nothing", "police": "another",
        "fire": "blocked", "health": "idle", "education": "first/supply"}
    assert f.order == ("utilities", "education", "planning", "police", "fire", "transport", "health")
    assert f.to_fund == [FundItem("utilities", "first/supply", 1200, 15, "another sewage outlet"),
                         FundItem("education", "first/supply", 2500, 45, "its first elementary school"),
                         FundItem("planning", "growth", 315, 0, "a road into open land and zoning it"),
                         FundItem("police", "another", 2500, 55, "another police station")]
    assert f.firsts_at_zero == ("utilities", "education")                   # no plan yet: no targets
    assert f.unclaimed == {} and f.idle == {"health": 7000}
    assert f.spendable == f.spare + 7000 and not f.hold
    assert [ln[:12] for ln in spend_lines(w, f)] == ["- to spend: ", "- to fund: f"]
    m = f.depts["education"]
    assert (m.top_up, m.upkeep, f.depts["transport"].top_up, f.depts["transport"].upkeep) == (2500, 45, 0, 0)
    # a department whose funded first buildings all go unclaimed is blocked too (in turn order within the kind),
    # and a levee is a first/supply need like a first building
    monkeypatch.setattr(T, "unclaimed_firsts", lambda depts: {"transport": ("a bus line",)})
    mix ={"fire": FB(900, (N("3 levee tiles", 1, None, upkeep=3),), 900, ()),
           "police": FB(2500, (N("police station", 4, 1, "coverage 40%", upkeep=55),), 2500, ())}
    _builds(monkeypatch, mix, road=FB(315, (ROAD,), 315, ()))
    f = funding_gaps(env, FleetState.of(env))
    assert f.unclaimed == {"transport": ("a bus line",)} and f.depts["transport"].unclaimed == ("a bus line",)
    assert {d: f.depts[d].kind for d in ("fire", "transport", "utilities")} == {
        "fire": "first/supply", "transport": "blocked", "utilities": "nothing"}
    assert f.order == ("fire", "planning", "police", "transport", "utilities", "education", "health")


def test_dept_money_says_what_it_keeps_and_its_state(monkeypatch):
    env = _env()
    w = env.world
    _months_spent(env, {"police": 658, "fire": 100})
    _builds(monkeypatch, {"fire": FB(2500, (N("fire station", 3, 0, first=True, upkeep=55),), 2500, ())},
            road=FB(1410, (ROAD,), 1410, ()))
    m = {d: DeptMoney(env, d) for d in ("planning", "police", "fire", "health")}
    assert {d: (x.keep, x.keep_why) for d, x in m.items()} == {
        "planning": (1410, f"its {ROAD_NAME}"), "police": (1316, "2 x its $658 spending"),
        "fire": (2500, "its dearest first building"), "health": (1000, "the $1,000 minimum")}
    w.dept_funds.update(police=3000, fire=2500)
    m = {d: DeptMoney(env, d) for d in ("planning", "police", "fire", "health")}
    assert {d: x.state for d, x in m.items()} == {
        "planning": "",                                  # never for planning
        "police": "funds cover its builds",              # $3,000 >= its dearest price (a $2,500 station)
        "fire": "funds cover its next build",            # levees (per tile) leave no bound: its $2,500 station
        "health": "nothing ready to buy"}
    w.dept_funds["fire"] = 2000.5
    m = DeptMoney(env, "fire")
    assert (m.kind, m.top_up, m.upkeep) == ("first/supply", 500, 55)       # rounded up: $499.50 buys nothing


def test_to_fund_is_needs_based_not_target_based(monkeypatch):
    """Smoke v16-v19: education was at a $0 target in 15 of 15 plans, and utilities stayed at a $7,000+ target
    after month 0; the old to-do line counted target gaps only."""
    env = _env()
    w = env.world
    fleet = FleetState.of(env)
    fleet.update_plan(0, budgets={"utilities": 7000, "education": 0, "police": 2500})
    w.dept_funds.update(utilities=800, police=2500)
    _builds(monkeypatch, {"utilities": FB(1200, (N("sewage outlet", 2, 0, first=True, upkeep=15),), 1200, ()),
                          "education": FB(2500, (N("elementary school", 3, 0, first=True, upkeep=45),), 2500, ())})
    f = funding_gaps(env, fleet)
    assert f.to_fund == [FundItem("utilities", "first/supply", 400, 15, "its first sewage outlet"),
                         FundItem("education", "first/supply", 2500, 45, "its first elementary school")]
    assert f.firsts_at_zero == ("education",)
    assert f.gaps == {"utilities": {"funds": 800, "target": 7000, "gap": 6200}} and f.total == 6200   # as before
    assert f.depts["police"].kind == "nothing"          # at its target with nothing ready


def _fake(*months, treasury: float = 0.0, debt: float = 0.0):
    """A stand-in world for reserve, cash_flow and spend_lines: `months` are the (income, expenses) of the months
    played, the treasury steady."""
    hist = [{"month": i + 1, "income": inc, "expenses": exp, "treasury": treasury, "debt": debt,
             "dept_allocated": {}, "expense_breakdown": {}, "dept_upkeep": {}} for i, (inc, exp) in enumerate(months)]
    return SimpleNamespace(history=hist, month=len(hist), treasury=treasury, debt=debt, ledger={"allocated": {}},
                           funding={})


def _funding(w, treasury: float, items: list, idle: dict | None = None) -> Funding:
    keep = reserve(w)
    return Funding({}, 0, keep, treasury - keep, idle or {}, {}, to_fund=items,
                   order=tuple(i.dept for i in items))


ITEMS = [FundItem("education", "first/supply", 5000, 75, "its first high school"),
         FundItem("planning", "growth", 315, 0, "a road into open land and zoning it"),
         FundItem("police", "another", 2500, 55, "another police station"),
         FundItem("health", "another", 1200, 40, "another clinic")]
TO_FUND = ("- to fund: first/supply education $5,000 (+$75/month) | growth planning $315 | another police $2,500, "
           "health $1,200 ($3,700, +$95/month)")


def test_cover_clause_is_greedy():
    """An item that doesn't fit is skipped, not the end: a $5,000 first high school must not hide a $315 road."""
    w = _fake(*[(1300, 1100)] * 4)                      # reserve 3 x $1,100; operations +$200 a month
    assert reserve(w) == 3300
    assert spend_lines(w, _funding(w, 5300, ITEMS)) == [
        "- to spend: $2,000 above the $3,300 reserve (3 x $1,100 expenses); operations +$200/month",
        TO_FUND + ": $2,000 covers planning, health; the other $7,500 would go under the reserve"]
    assert spend_lines(w, _funding(w, 5300, ITEMS, {"transport": 1508})) == [
        "- to spend: $2,000 above the $3,300 reserve (3 x $1,100 expenses) + $1,508 idle to take back = $3,508; "
        "operations +$200/month",
        TO_FUND + ": $3,508 covers planning, police; the other $6,200 would go under the reserve"]
    assert spend_lines(w, _funding(w, 3000, ITEMS, {"transport": 1508}))[0] == (
        "- to spend: $1,208: $1,508 idle to take back, less the $300 the treasury lacks for the $3,300 reserve "
        "(3 x $1,100 expenses); operations +$200/month")
    assert spend_lines(w, _funding(w, 50000, ITEMS))[1] == (
        TO_FUND + ": $46.7k covers education, planning, police, health")
    assert spend_lines(w, _funding(w, 3400, ITEMS))[1] == (
        TO_FUND + ": $100 covers none of them; all $9,015 would go under the reserve")
    assert spend_lines(w, _funding(w, 5300, [])) == [
        "- to spend: $2,000 above the $3,300 reserve (3 x $1,100 expenses); operations +$200/month",
        "- to fund: nothing ready"]


def test_the_hold_covers_first_buildings_and_growth_from_above_3000():
    w = _fake(*[(1500, 1300)] * 4)                      # reserve $3,900; operations +$200 a month
    f = _funding(w, 3500, ITEMS)
    assert f.hold and f.dip == 500                     # treasury above $3,000
    assert spend_lines(w, f) == [
        "- to spend: none (hold): the treasury is $400 under the $3,900 reserve (3 x $1,300 expenses); operations "
        "+$200/month close that in 2 months; first/supply and growth may use $500 (treasury above $3,000)",
        TO_FUND + ": $500 covers planning; the other $5,000 would go under $3,000"]
    f = _funding(w, 3500, ITEMS, {"transport": 300})     # idle, still in the hold
    assert f.hold and f.dip == 800
    assert spend_lines(w, f)[0].endswith("close that in 2 months; first/supply and growth may use $800 (treasury "
                                         "above $3,000 + idle)")
    w = _fake(*[(1100, 1300)] * 4)                      # a deficit: 6 x expenses, and operations don't close it
    f = _funding(w, 2500, ITEMS)
    assert f.hold and f.dip == 0
    assert spend_lines(w, f) == [
        "- to spend: none (hold): the treasury is $5,300 under the $7,800 reserve (6 x $1,300 expenses: last "
        "month's deficit was $200); operations -$200/month; nothing above $3,000", TO_FUND]
    w = _fake()                                         # month 0: the minimum, no operations yet
    f = _funding(w, RESERVE_MIN, ITEMS)
    assert f.hold and spend_lines(w, f)[0] == (
        "- to spend: none (hold): the treasury is at the $3,000 reserve; operations +$0/month; nothing above $3,000")


def test_still_to_fund():
    w = _fake()
    assert still_to_fund(_funding(w, 0, ITEMS)) == ("education $5,000 (first/supply), planning $315 (growth), "
                                                    "police $2,500 (another), health $1,200 (another)")
    assert still_to_fund(_funding(w, 0, ITEMS), n=2) == ("education $5,000 (first/supply), planning $315 (growth), "
                                                         "+2 more")
    assert still_to_fund(_funding(w, 0, [])) == "none"


def test_reserve_rounds_the_deficit():
    """v16 seed 1, month 6: a $0.70 deficit doubled the reserve."""
    world = lambda *h: SimpleNamespace(history=[{"income": i, "expenses": e} for i, e in h])
    assert reserve(world((675.7, 676.4))) == 3000       # $676 both: 3 x $676, under the minimum
    assert reserve(world((675, 677))) == 6 * 677
    assert reserve(world((2000, 1236.8))) == 3 * 1237   # rounded dollars, as the text shows them
    assert reserve_why(world()) == reserve_why(world((675.7, 676.4))) == ""               # the minimum
    assert reserve_why(world((2000, 1236.8))) == " (3 x $1,237 expenses)"
    assert reserve_why(world((674.4, 676.4))) == " (6 x $676 expenses: last month's deficit was $2)"


def test_cash_flow_identity(monkeypatch):
    """The treasury change splits into operations, the mayor's top-ups (take-backs net), loans (repayments net) and
    the rest (rewards, offer bonuses): smoke v13-v19 showed a "loss" in 30 of 81 turns that ran an operating
    surplus, the mayor's own transfers mixed in."""
    env = _env()
    w = env.world
    mayor = lambda **a: apply_action(w, "mayor", a).ok
    assert mayor(type="allocate_budget", department="police", amount=2000) and mayor(type="take_loan", amount=5000)
    env.end_month()                                     # month 1: the window's start
    assert mayor(type="allocate_budget", department="police", amount=-500) and mayor(type="repay_loan", amount=2000)
    env.end_month()
    assert mayor(type="allocate_budget", department="fire", amount=1000)
    env.end_month()
    assert mayor(type="allocate_budget", department="health", amount=300) and mayor(type="take_loan", amount=1000)
    cf = cash_flow(w)                                   # month 3, mid-turn: this turn's moves count
    past = w.history[0]
    after = w.history[1:]
    assert (cf.since, cf.months) == (1, 2)
    assert abs(cf.ops - sum(h["income"] - h["expenses"] for h in after)) < 1e-6 and cf.ops < 0
    assert cf.ops_per_month == cf.ops / 2
    assert (cf.top_ups, cf.loans) == (800, -1000)
    assert abs(cf.change - (w.treasury - past["treasury"])) < 1e-6
    assert abs(cf.ops - cf.top_ups + cf.loans + cf.other - cf.change) < 1e-6 and abs(cf.other) < 0.5
    last = w.history[-1]
    assert (cf.income, cf.expenses, cf.interest) == (last["income"], last["expenses"],
                                                     last["expense_breakdown"]["loan_interest"])
    assert cf.interest == 3000 * C.LOAN_INTEREST        # the month's debt: $5,000 less $2,000 repaid
    assert list(cf.buildings) == ["roads"] and cf.buildings["roads"] == last["expense_breakdown"]["roads"]
    assert abs(sum(cf.buildings.values()) + cf.services + cf.interest - cf.expenses) < 1e-6
    env = _env()                                        # a Town reward in the window: "other"
    env.end_month()
    env.end_month()
    monkeypatch.setitem(C.MILESTONES[0], "pop", 0)
    env.end_month()
    cf = cash_flow(env.world)
    assert C.MILESTONES[0]["name"] in env.world.milestones_reached and abs(cf.other - 5000) < 0.5
    assert cash_flow(_env().world) == T.CashFlow(None, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, 0.0, 0.0)


# --- Unit D: result notes and look-ups (tools.py: ToolBox._money_note, MONEY_DOCS, update_plan, overview)
from civitas.fleet.tools import ToolBox, overview  # noqa: E402

STATION = FB(2500, (N("police station", 4, 1, "coverage 40%", upkeep=55),), 2500, ())
SCHOOL = FB(2500, (N("elementary school", 3, 0, first=True, upkeep=45),), 2500, ())
NO_LOT = FB(None, (), 2500, ("a fire station",))
ROAD_315 = FB(315, (ROAD,), 315, ())


def _money(box, name: str, confirm: bool = False, **args) -> str:
    """Run one of the mayor's money actions as the model does, and return its result note: what the hook in _act
    appends to the model's message (Unit C), and only there; `executed` and the history log keep the core text. With
    `confirm`, a refused call is repeated once with confirm: true, as an earned override takes."""
    sent = {**args, "confirm": True} if confirm else args
    res = box.call(name, sent)
    if confirm and res.get("refused"):
        res = box.call(name, sent)
    core = box.executed[-1]
    assert core["action"]["type"] == name and core["ok"] == res["ok"]
    note = box._money_note(name, args, core)
    assert res["message"] in (core["message"], core["message"] + note)        # with or without the hook
    assert box.fleet.log[-1]["message"] == core["message"] and " | " not in core["message"]
    return note


def test_money_results_show_the_treasury_and_what_is_left(monkeypatch):
    env = _env()
    w = env.world
    _months_spent(env, {"planning": 620})               # planning keeps 2 x $620
    assert reserve(w) == RESERVE_MIN and reserve_why(w) == ""
    _builds(monkeypatch, {"police": STATION, "education": SCHOOL, "fire": NO_LOT}, road=ROAD_315)
    w.treasury, w.dept_funds["planning"] = 5288, 2748     # planning: $1,508 idle
    box = ToolBox(env, "mayor")
    school = "still to fund: education $2,500 (first/supply)"
    # top-ups: what the department can buy now, then the treasury against the reserve and what is still to fund
    assert _money(box, "allocate_budget", department="police", amount=2500) == (
        " | police: can pay for another police station (+$55/month); treasury $2,788, $212 under the $3,000 "
        f"reserve; {school}")
    school = "still to fund: education $1,000 (first/supply)"
    assert _money(box, "allocate_budget", department="education", amount=1500) == (
        " | education: still $1,000 short of its first elementary school: less buys nothing; treasury $1,288, "
        f"$1,712 under the $3,000 reserve; {school}")
    assert _money(box, "allocate_budget", department="health", amount=200) == (
        f" | health: had nothing ready to buy; treasury $1,088, $1,912 under the $3,000 reserve; {school}")
    assert _money(box, "allocate_budget", department="fire", amount=88) == (
        " | fire: blocked for lack of land: this buys nothing until planning frees a lot; treasury $1,000, $2,000 "
        f"under the $3,000 reserve; {school}")
    # the treasury can't pay: where it stands, the idle money to take back instead and what is still to fund
    assert _money(box, "allocate_budget", department="education", amount=2000) == (
        f" | treasury $1,000, $2,000 under the $3,000 reserve; idle to take back: planning $1,508; {school}")
    # take-backs: what the department holds against what it keeps
    assert _money(box, "allocate_budget", department="planning", amount=-1000) == (
        f" | planning now $1,748, $508 over what it keeps; treasury $2,000, $1,000 under the $3,000 reserve; {school}")
    assert _money(box, "allocate_budget", department="planning", amount=-508) == (
        " | planning now $1,240, what it keeps (nothing idle); treasury $2,508, $492 under the $3,000 reserve; "
        f"{school}")
    assert _money(box, "allocate_budget", department="planning", amount=-1240, confirm=True) == (
        " | planning now $0, $1,240 below what it keeps (2 x its $620 spending): it can't build next quarter; "
        f"treasury $3,748, $748 above the $3,000 reserve; {school}, planning $315 (growth)")
    # other failures and other actions: no note
    assert _money(box, "allocate_budget", department="parks", amount=100) == ""
    assert _money(box, "repay_loan", amount=100) == ""                      # nothing to repay
    assert box._money_note("set_tax", {"zone": "R", "rate": 0.1}, {"ok": True, "message": "taxes set"}) == ""


def test_money_results_show_the_reserve_reason_and_an_idle_top_up(monkeypatch):
    env = _env()
    w = env.world
    _months_spent(env, {"police": 700})
    w.history[-1].update(income=2000, expenses=1237)     # last month: 3 x $1,237
    _builds(monkeypatch, {"police": STATION})
    w.treasury, w.dept_funds["police"] = 9711, 5000       # police keeps $2,500 of $5,000: idle
    box = ToolBox(env, "mayor")
    assert _money(box, "allocate_budget", department="police", amount=1000) == (
        " | police: can pay for another police station (+$55/month); treasury $8,711, $5,000 above the $3,711 "
        "reserve (3 x $1,237 expenses); still to fund: none")
    assert _money(box, "allocate_budget", department="police", amount=-3500) == (
        " | police now $2,500, what it keeps (nothing idle); treasury $12.2k, $8,500 above the $3,711 reserve "
        "(3 x $1,237 expenses); still to fund: none")
    w.dept_funds["police"] = 4000
    assert box._money_note("allocate_budget", {"department": "police", "amount": -1},
                           {"ok": True, "message": "reclaimed $1 from police"}).startswith(
        " | police now $4,000, $1,500 over what it keeps: still idle; treasury")


def test_loan_results_show_the_interest(monkeypatch):
    env = _env()
    w = env.world
    _builds(monkeypatch, {"police": STATION, "education": SCHOOL})
    still = "still to fund: education $2,500 (first/supply), police $2,500 (another)"
    w.treasury = 1000                                   # a turn that starts under the $3,000 minimum
    box = ToolBox(env, "mayor")
    assert _money(box, "take_loan", amount=3000) == (
        f" | interest now $15/month (+$15); $47k more can be borrowed; treasury $4,000, $1,000 above the $3,000 "
        f"reserve; {still}")
    w.treasury, w.debt = 500, 48000
    assert _money(box, "take_loan", amount=5000) == " | only $2,000 more can be borrowed (interest $240/month)"
    assert _money(box, "take_loan", amount=2000) == (
        f" | interest now $250/month (+$10); no more can be borrowed; treasury $2,500, $500 under the $3,000 "
        f"reserve; {still}")
    assert _money(box, "take_loan", amount=1000) == (
        " | no more credit: repay_loan from money above the reserve cuts the $250/month interest")
    assert _money(box, "repay_loan", amount=3000) == (
        f" | treasury $2,500, $500 under the $3,000 reserve; idle to take back: none; {still}")   # treasury only has
    w.treasury, w.debt = 5000, 3000
    assert _money(box, "repay_loan", amount=1000) == (
        " | interest now $10/month (saves $5/month); treasury $4,000, $1,000 above the $3,000 reserve")
    assert _money(box, "repay_loan", amount=5000) == (
        " | no debt left: no more interest; treasury $2,000, $1,000 under the $3,000 reserve")
    assert w.debt == 0


def test_a_money_note_never_fails_an_action(monkeypatch):
    env = _env()
    box = ToolBox(env, "mayor")
    res = apply_action(env.world, "mayor", {"type": "allocate_budget", "department": "police", "amount": 100})

    def boom(*a, **k):
        raise RuntimeError("no")
    monkeypatch.setattr(T, "funding_gaps", boom)
    assert res.ok and box._money_note("allocate_budget", {"department": "police", "amount": 100}, res.to_dict()) == ""


def test_money_docs_say_what_a_loan_costs():
    assert T.MONEY_DOCS == {
        "take_loan": " Interest is $5 a month per $1,000 until repaid; the finance score reads net worth (treasury + "
                     "department funds - debt).",
        "repay_loan": " Each $1,000 repaid saves $5 a month in interest."}


def test_update_plan_reports_what_to_fund(monkeypatch):
    """Needs-based, whatever the targets: to_spend, to_fund (most urgent first) and the first buildings at a $0
    target, with the briefing's to spend / to fund lines as `next`."""
    env = _env()
    w = env.world
    fleet = FleetState.of(env)
    _builds(monkeypatch, {"police": STATION, "education": SCHOOL, "fire": NO_LOT}, road=ROAD_315)
    w.dept_funds["education"] = 500
    w.treasury = 29500
    mayor = ToolBox(env, "mayor")
    res = mayor.call("update_plan", {"budgets": {"police": 2500, "planning": 3000}})
    f = funding_gaps(env, fleet)
    assert res["to_spend"] == 26500 and "idle" not in res
    assert res["to_fund"] == [
        {"department": "education", "kind": "first/supply", "amount": 2000, "upkeep_per_month": 45},
        {"department": "planning", "kind": "growth", "amount": 315, "upkeep_per_month": 0},
        {"department": "police", "kind": "another", "amount": 2500, "upkeep_per_month": 55}]
    assert res["first_buildings"] == ["education"]
    assert res["next"] == "; ".join(ln.removeprefix("- ") for ln in spend_lines(w, f)) + (
        "; first buildings with a $0 target: education")
    assert res["next"] == (
        "to spend: $26.5k above the $3,000 reserve; operations +$0/month; to fund: first/supply education $2,000 "
        "(+$45/month) | growth planning $315 | another police $2,500 (+$55/month): $26.5k covers education, planning, "
        "police; first buildings with a $0 target: education")
    assert set(res["below_target"]) == {"police", "planning"} and res["treasury"] == 29500     # targets, as before
    res = mayor.call("update_plan", {"budgets": {"education": 2500}})
    assert "first_buildings" not in res and "$0 target" not in res["next"]
    w.treasury = 2000                                   # a hold
    res = mayor.call("update_plan", {"notes": "hold"})
    assert res["to_spend"] == 0 and res["next"].startswith("to spend: none (hold): the treasury is $1,000 under")
    w.treasury = 30000
    _builds(monkeypatch, {})
    res = mayor.call("update_plan", {"notes": "nothing ready"})
    assert res["to_fund"] == [] and res["next"] == (
        "to spend: $27k above the $3,000 reserve; operations +$0/month; to fund: nothing ready")


def test_city_overview_splits_the_treasury_change():
    env = _env()
    w = env.world
    ov = overview(env)
    assert ov["operations_3_months"] is None and ov["costs_last_month"] == {}          # before the first month
    mayor = lambda **a: apply_action(w, "mayor", a).ok
    assert mayor(type="allocate_budget", department="police", amount=2000)
    env.end_month()
    env.end_month()
    assert mayor(type="allocate_budget", department="fire", amount=500) and mayor(type="take_loan", amount=1000)
    cf = cash_flow(w)
    ov = ToolBox(env, "police").call("city_overview", {})
    assert ov["operations_3_months"] == {
        "since_month": 1, "months": 1, "treasury_change": round(cf.change), "operations": round(cf.ops),
        "top_ups": 500, "loans": 1000, "other": 0}
    assert ov["operations_3_months"]["since_month"] == ov["trend"]["since_month"]
    spent = w.history[-1]["expense_breakdown"]
    assert list(ov["costs_last_month"].items()) == sorted(
        ((k, round(v)) for k, v in spent.items() if round(v)), key=lambda kv: -kv[1])
    assert ov["costs_last_month"]["town_hall"] == 20 and "police" not in ov["costs_last_month"]    # $0 left out
