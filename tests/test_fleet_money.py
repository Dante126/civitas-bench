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


# --- Unit B: briefing (briefing.py: the mayor's Finances and Departments blocks, the system prompt)
from civitas.fleet.briefing import BLOCKED_NOTE, COSTS_SHOWN, MAYOR_REPORT_CHARS, briefing, system_prompt  # noqa: E402
from civitas.fleet.config import FleetAgentConfig  # noqa: E402
from civitas.fleet.fmt import money, signed  # noqa: E402
from civitas.fleet.tools import ToolBox  # noqa: E402


def _block(text: str, header: str) -> list[str]:
    """The lines of the briefing block whose header line starts with `header`, the header left out."""
    at = text.index("\n" + header) + 1
    return text[text.index("\n", at) + 1:].split("\n\n")[0].splitlines()


def _mayor_block(env, header: str) -> list[str]:
    return _block(briefing(env, "mayor", mark=False), header)


def test_mayor_finances_lines(monkeypatch):
    """Smoke v13-v19: the treasury change mixed in the mayor's own transfers (30 of 81 turns showed a "loss" while
    operations ran a surplus), and interest never showed in dollars."""
    env = _env()
    w = env.world
    fin = lambda: _mayor_block(env, "## Finances")
    assert fin() == ["- treasury $30k, debt $0", "- last month: no month played yet"]      # month 0
    mayor = lambda **a: apply_action(w, "mayor", a).ok
    assert mayor(type="allocate_budget", department="police", amount=2500)
    env.end_month()                                     # month 1: the window's start
    env.end_month()
    ops = lambda: sum(h["income"] - h["expenses"] for h in w.history[1:])
    change = lambda: signed(round(w.treasury - w.history[0]["treasury"]), money)
    assert fin()[0] == (f"- treasury {money(w.treasury)} ({change()} since month 1: operations "
                        f"{signed(ops(), money)}), debt $0")          # no top-ups since month 1: left out
    assert mayor(type="allocate_budget", department="fire", amount=2500)
    env.end_month()                                     # month 3
    assert fin()[0] == (f"- treasury {money(w.treasury)} ({change()} since month 1: operations {signed(ops(), money)}, "
                        "top-ups -$2,500), debt $0")
    last = w.history[-1]
    b = {k: v for k, v in last["expense_breakdown"].items() if k in last["dept_upkeep"] and v > 0}
    rest = last["expenses"] - sum(b.values()) - last["expense_breakdown"]["loan_interest"]
    dear = sorted(b.items(), key=lambda kv: -kv[1])
    assert len(dear) <= 3 and fin()[1] == (
        f"- last month: income {money(last['income'])}, expenses {money(last['expenses'])}: buildings "
        f"{money(sum(b.values()))} (" + ", ".join(f"{k} {money(v)}" for k, v in dear)
        + f"), city services {money(rest)}, interest $0")
    assert len(fin()) == 2                              # no debt: no debt line
    # the costs line names the dearest COSTS_SHOWN (2), then how many more and what they cost together; city services
    # are the rest (the town hall included), so the parts add up to the expenses
    assert COSTS_SHOWN == 2
    last.update(income=597, expenses=469, expense_breakdown={
        "utilities": 82, "roads": 75, "police": 55, "fire": 55, "health": 45, "town_hall": 20, "city_services": 137,
        "loan_interest": 0}, dept_upkeep={"utilities": 82, "roads": 75, "police": 55, "fire": 55, "health": 45,
                                          "education": 0})
    assert fin()[1] == ("- last month: income $597, expenses $469: buildings $312 (utilities $82, roads $75, 3 more "
                        "$155), city services $157, interest $0")
    last["expense_breakdown"].update(fire=0, health=0, city_services=237)       # one more: named ("1 more" is as long)
    assert fin()[1] == ("- last month: income $597, expenses $469: buildings $212 (utilities $82, roads $75, police "
                        "$55), city services $257, interest $0")
    # a loan: the split shows it, the first line drops the debt, and a debt line says what it costs
    assert mayor(type="take_loan", amount=18800)
    lines = fin()
    assert "loans +$18.8k" in lines[0] and lines[0].endswith(")") and ", debt" not in lines[0], lines[0]
    assert lines[2] == ("- debt $18.8k: $94/month interest, $31.2k more can be borrowed; each $1,000 repaid saves "
                        "$5/month")                        # 0.5% of the debt a month
    assert 18800 * C.LOAN_INTEREST == 94
    env.end_month()
    assert fin()[1].endswith(", interest $94")          # last month's, from the ledger
    w.debt = C.MAX_DEBT
    assert fin()[2] == "- debt $50k: $250/month interest, at the $50k limit; each $1,000 repaid saves $5/month"
    # a milestone reward is the rest of the change: "other"
    env = _env()
    env.end_month()
    env.end_month()
    monkeypatch.setitem(C.MILESTONES[0], "pop", 0)
    env.end_month()
    line = _mayor_block(env, "## Finances")[0]
    assert ", other +$5,000)" in line and "top-ups" not in line and "loans" not in line, line


def test_mayor_departments_most_urgent_first(monkeypatch):
    """Smoke v16-v19: block order became funding order (32 of 54 turns followed it), and education, last in turn
    order, was topped up last in 11 turns. The block follows Funding.order, each line with one suffix."""
    env = _env()
    w = env.world
    _months_spent(env, {"health": 1000})                # health keeps $2,000 of $9,000: idle
    w.dept_funds["health"] = 9000
    fleet = FleetState.of(env)
    ToolBox(env, "mayor").call("update_plan", {"budgets": {"police": 5000},
                                               "priorities": {d: "PRIO " * 8 for d in ("police", "fire")}})
    ToolBox(env, "fire").call("end_turn", {"report": "FIRE-REPORT " + "z" * 600})
    _builds(monkeypatch, {
        "utilities": FB(1200, (N("sewage outlet", 2, 1, "sewage 95% used", upkeep=15),), 1200, ()),
        "education": FB(2500, (N("elementary school", 3, 0, first=True, upkeep=45),), 2500, ()),
        "police": FB(2500, (N("police station", 4, 1, "coverage 40%", upkeep=55),), 2500, ()),
        "fire": FB(None, (), 2500, ("a fire station",))},
        road=FB(315, (ROAD,), 315, ()))
    text = briefing(env, "mayor", mark=False)
    header = next(ln for ln in text.splitlines() if ln.startswith("## Departments"))
    assert header == "## Departments (most urgent first; spent = last 1 month; cut reports: history kind=report)"
    block = _block(text, "## Departments")
    lines = [ln for ln in block if ": funds " in ln]
    assert [ln.split(":")[0][2:] for ln in lines] == list(funding_gaps(env, fleet).order) == [
        "utilities", "education", "planning", "police", "fire", "transport", "health"]
    assert lines == [
        "- utilities: funds $0, spent $0; needs $1,200 for another sewage outlet (sewage 95% used, $15/month)",
        "- education: funds $0, spent $0; needs $2,500 for its first elementary school (3 sites, $45/month)",
        "- planning: funds $0, spent $0; needs $315 for a road into open land and zoning it (opens 20 tiles)",
        "- police: funds $0 < target $5,000, spent $0; needs $2,500 for another police station (coverage 40%, "
        "$55/month)",
        "- fire: funds $0, spent $0; blocked: a fire station",
        "- transport: funds $0, spent $0; nothing ready to buy",
        "- health: funds $9,000, spent $1,000; idle $7,000 (keeps $2,000)"]
    report, = [ln for ln in block if "FIRE-REPORT" in ln]
    assert block.index(report) == block.index(lines[4]) + 1                  # under its department's line
    assert report == f"  report (m1): FIRE-REPORT {'z' * (MAYOR_REPORT_CHARS - 15)}..."
    assert "priority" not in text and "PRIO" not in text                   # get_plan has them
    assert block.count(BLOCKED_NOTE) == 1 and BLOCKED_NOTE == (
        "- blocked for lack of land: money buys nothing there until planning frees lots (ask planning)")
    assert block[-2:] == spend_lines(w, funding_gaps(env, fleet))           # after the notes, in their place
    assert block.index(BLOCKED_NOTE) == len(block) - 3
    assert not any(s in text for s in ("below target", "first buildings still unfunded", "idle in"))
    assert "\n## City plan (v1, month 1; priorities and full text: get_plan)\n" in text
    # a department with nothing to name reads what its money does; planning then reads nothing more
    _builds(monkeypatch, {"police": FB(2500, (N("police station", 4, 1, "coverage 40%", upkeep=55),), 2500, ()),
                          "fire": FB(2500, (N("fire station", 3, 0, first=True, upkeep=55),), 2500, ())})
    w.dept_funds.update(police=2500, fire=2500, health=0)
    lines = {ln.split(":")[0][2:]: ln for ln in _mayor_block(env, "## Departments") if ": funds " in ln}
    assert lines["planning"] == "- planning: funds $0, spent $0"
    assert lines["police"] == "- police: funds $2,500 < target $5,000, spent $0; funds cover its builds"
    assert lines["fire"] == "- fire: funds $2,500, spent $0; funds cover its next build"    # levees: no dearest price
    assert lines["health"] == "- health: funds $0, spent $1,000; nothing ready to buy"
    assert BLOCKED_NOTE not in _mayor_block(env, "## Departments")
    env = _env()                                        # month 0: nothing spent yet
    assert "\n## Departments (most urgent first)\n" in briefing(env, "mayor", mark=False)


def test_mayor_system_prompt_says_upkeep_is_for_good():
    text = system_prompt("mayor", FleetAgentConfig(), 36)
    assert ("Stay solvent: upkeep is paid from the treasury every month and every building adds upkeep for good; keep "
            "the reserve when you move money to departments (allocate_budget) or cut taxes.") in text
    assert "Stay within your budget" in system_prompt("police", FleetAgentConfig(), 36)


# --- Unit C: money guards (tools.py: earned confirm, duplicate take-backs, the loan guard, the money-note hook;
# agent._slim and report.py: overrides)
EARNED = " (confirm: true counts only when repeating this exact call after it was refused)"
BORROW = " To borrow anyway, repeat this exact call with confirm: true."


def _mayor(env, treasury: float = 30000.0, **funds) -> T.ToolBox:
    """The mayor's toolbox for a turn starting with this treasury and these department funds."""
    env.world.treasury = treasury
    env.world.dept_funds.update(funds)
    return T.ToolBox(env, "mayor")


def _props(box, tool: str) -> dict:
    return next(s["function"]["parameters"]["properties"] for s in box.schemas() if s["function"]["name"] == tool)


def test_confirm_counts_only_after_a_refusal():
    """v19: all 3 take-back refusals were overridden at once, and 3 of 7 duplicate refusals; confirm: true was a free
    pass. It now counts only on a repeat of the exact call refused this turn, one override per refusal."""
    from civitas.fleet.agent import _slim
    from civitas.fleet.report import _tool_error, turn_row
    env = _env()
    w = env.world
    _months_spent(env, {"planning": 658}, {}, {})
    box = _mayor(env, planning=2508)
    m = DeptMoney(env, "planning")
    assert (m.keep, m.keep_why, m.idle) == (1316, "2 x its $658 spending", 1192)
    take = lambda amount, **kw: box.call("allocate_budget", {"department": "planning", "amount": amount, **kw})
    res = take(-1000, confirm=True)                    # within what it keeps: confirm or not, it just runs
    assert res["ok"] and "override" not in res and w.dept_funds["planning"] == 1508, res
    why = ("that takes planning below the $1,316 it keeps (2 x its $658 spending), so it can't build next quarter; you "
           "already took back $1,000 from it this turn; it has no idle money: leave its funds. To take it anyway, "
           "repeat this exact call with confirm: true")
    first = take(-1000, confirm=True)                  # confirm on a call not refused before: refused, with the note
    assert first == {"ok": False, "refused": "take_back", "error": why + EARNED + "."}, first
    assert take(-1000) == {"ok": False, "refused": "take_back", "error": why + "."}       # no confirm: no note
    again = take(-1000, confirm=True)                  # the identical repeat: earned (the take-back guard ran first)
    assert again["ok"] and again["override"] == "take_back" and w.dept_funds["planning"] == 508, again
    assert again["message"].startswith("reclaimed $1,000 from planning")
    assert box.executed[-1]["action"] == {"department": "planning", "amount": -1000, "type": "allocate_budget"}
    assert _slim(again)["override"] == "take_back" and _tool_error({"tool": "allocate_budget", "result": again}) is None
    row = turn_row({"role": "mayor", "month": 3, "tool_trace": [{"tool": "allocate_budget", "result": _slim(r)}
                                                                for r in (first, again)]})
    assert (row["refused"], row["overrides"]) == (1, 1)
    # the same call as the handler reads it (department case, a number as text), and one refusal per override
    assert box.call("allocate_budget", '{"department": "fire", "amount": 400}')["ok"]
    dup = lambda **kw: box.call("allocate_budget", {"department": "Fire", "amount": "400", **kw})
    assert dup()["refused"] == "duplicate"
    assert dup(confirm="true")["override"] == "duplicate" and w.dept_funds["fire"] == 800
    res = dup(confirm=True)                            # that refusal earned one override: refused again, with the note
    assert res["refused"] == "duplicate" and res["error"].endswith("only to add it again" + EARNED), res
    assert dup(confirm=True)["override"] == "duplicate" and w.dept_funds["fire"] == 1200
    assert [r["ok"] for r in box.executed] == [True, True, True, True, True]
    assert _props(box, "allocate_budget")["confirm"] == {"type": "boolean", "description": T.CONFIRM["allocate_budget"]}
    assert T.CONFIRM["allocate_budget"] == (
        "true only to repeat an allocate_budget refused this turn with the same department and amount, once you are "
        "sure; on any other call it doesn't count")


def test_identical_take_backs_are_duplicates():
    """v19 seed 1, month 6: the mayor repeated its take-back pairs; the duplicate guard let negative amounts through.
    The take-back guard runs first, then the duplicate guard."""
    env = _env()
    w = env.world
    _months_spent(env, {}, {}, {})
    box = _mayor(env, planning=5000)
    assert DeptMoney(env, "planning").idle == 4000
    take = lambda amount, **kw: box.call("allocate_budget", {"department": "planning", "amount": amount, **kw})
    assert take(-1140)["ok"]
    res = take(-1140)
    assert res == {"ok": False, "refused": "duplicate", "error": (
        "you already took $1,140 back from planning this turn (it now has $3,860); take-backs add up - repeat with "
        "confirm: true only to take it back again")}, res
    assert box.executed[-1]["ok"] and len(box.executed) == 1 and w.dept_funds["planning"] == 3860
    assert take(-1000)["ok"] and w.dept_funds["planning"] == 2860                      # another amount
    res = take(-1140, confirm=True)                    # refused before this turn: earned
    assert res["ok"] and res["override"] == "duplicate" and w.dept_funds["planning"] == 1720, res
    assert take(-1720)["refused"] == "take_back"       # below what it keeps: that guard first


def test_a_self_inflicted_loan_is_refused_until_confirmed():
    """Smoke v13-v19: 10 of 15 loans ($34.8k of $60.0k) came in turns that started at $3,000 or more, and 14 of 15
    after the mayor's own top-ups had emptied the treasury; none was repaid."""
    env = _env()
    w = env.world
    box = _mayor(env, treasury=3598)
    loan = lambda amount, **kw: box.call("take_loan", {"amount": amount, **kw})
    over = ("the treasury holds $3,598, over the $3,000 minimum: a $5,000 loan costs $25/month until repaid (debt would "
            "be $5,000 of $50k); borrow only for a first building or supply need the treasury can't cover." + BORROW)
    assert loan(5000) == {"ok": False, "refused": "loan", "error": over}
    assert w.debt == 0 and box.executed == [] and box._failed == {}
    res = loan("5000", confirm=True)                   # the earned repeat
    assert res["ok"] and res["override"] == "loan" and (w.debt, w.treasury) == (5000, 8598), res
    assert res["message"].startswith("borrowed $5,000 at 0.5%/month (debt $5,000)")
    # started at $3,000 or more, and this turn's own top-ups took it under
    w.debt = 0
    box = _mayor(env, treasury=5698)
    move = lambda dept, amount: box.call("allocate_budget", {"department": dept, "amount": amount})
    assert move("police", 4698)["ok"] and move("fire", 1000)["ok"] and w.treasury == 0
    assert box.call("take_loan", {"amount": 3000}) == {"ok": False, "refused": "loan", "error": (
        "the treasury started this turn at $5,698 and this turn's top-ups ($5,698 net) took it under $3,000: take back "
        "a top-up that isn't a first building or supply (free) instead of a $3,000 loan at $15/month." + BORROW)}
    assert move("fire", -1000)["ok"]                   # take-backs net off
    res = box.call("take_loan", {"amount": 3000})
    assert res["refused"] == "loan" and "this turn's top-ups ($4,698 net) took it under $3,000" in res["error"]
    res = box.call("take_loan", {"amount": 3000, "confirm": True})
    assert res["ok"] and res["override"] == "loan" and w.debt == 3000
    # a turn that started under $3,000 borrows at once, top-ups or not
    w.debt = 0
    box = _mayor(env, treasury=2400)
    assert move("police", 400)["ok"]
    res = box.call("take_loan", {"amount": 10000})
    assert res["ok"] and "refused" not in res and "override" not in res and w.treasury == 12000, res
    res = box.call("take_loan", {"amount": 1000, "confirm": True})      # now over $3,000; confirm not earned yet
    assert res["refused"] == "loan" and res["error"].endswith(" with confirm: true" + EARNED + "."), res
    # what the action refuses (at the cap, nothing to borrow) is left to it
    w.debt = 50000
    res = box.call("take_loan", {"amount": 1000})
    assert res["ok"] is False and "refused" not in res and res["error"].startswith("debt limit is $50,000"), res
    assert box.call("take_loan", {"amount": 0})["error"].startswith("amount must be positive")
    assert _props(box, "take_loan")["confirm"] == {"type": "boolean", "description": T.CONFIRM["take_loan"]}
    assert T.CONFIRM["take_loan"] == ("true only to repeat a take_loan refused this turn with the same amount, once "
                                      "you are sure; on any other call it doesn't count")
    assert T.guarded("mayor", "take_loan") and not T.guarded("mayor", "repay_loan")
    assert not T.guarded("utilities", "take_loan")


def test_money_results_carry_the_money_note(monkeypatch):
    """The mayor's allocate_budget, take_loan and repay_loan results end with ToolBox._money_note on success and on a
    core failure, instead of _explain's fix; a guard refusal gets none, and executed and the log keep the core text.
    MONEY_DOCS extends those tools' descriptions."""
    from civitas.core.actions import action_docs
    env = _env()
    w = env.world
    notes = []
    monkeypatch.setattr(T.ToolBox, "_money_note",
                        lambda self, name, args, res: notes.append((name, dict(args), res["ok"])) or f" | {name}")
    monkeypatch.setattr(T.ToolBox, "_explain", lambda self, name, args, message: " | explained")
    monkeypatch.setitem(T.MONEY_DOCS, "take_loan", " Interest is due monthly.")
    box = _mayor(env, treasury=2000)
    desc = {s["function"]["name"]: s["function"]["description"] for s in box.schemas()}
    docs = action_docs()
    assert desc["take_loan"] == docs["take_loan"][1] + " Interest is due monthly." and desc["set_tax"] == docs[
        "set_tax"][1]
    assert box.call("allocate_budget", {"department": "police", "amount": 500})["message"] == (
        "transferred $500 to police (now $500) | allocate_budget")
    res = box.call("allocate_budget", {"department": "police", "amount": 5000})        # a core failure
    assert res["ok"] is False and res["message"] == "treasury only has $1,500 | allocate_budget"
    assert box.call("take_loan", {"amount": 1000})["message"] == "borrowed $1,000 at 0.5%/month (debt $1,000) | take_loan"
    assert box.call("repay_loan", {"amount": 400})["message"] == "repaid $400 (debt $600) | repay_loan"
    res = box.call("allocate_budget", {"department": "police", "amount": 500})         # a guard refusal: no note
    assert res["refused"] == "duplicate" and " | " not in res["error"]
    assert notes == [("allocate_budget", {"department": "police", "amount": 500}, True),
                     ("allocate_budget", {"department": "police", "amount": 5000}, False),
                     ("take_loan", {"amount": 1000}, True), ("repay_loan", {"amount": 400}, True)]
    assert box.call("set_tax", {"sector": "Q", "rate": 9})["message"].endswith(" | explained")   # other actions
    core = ["transferred $500 to police (now $500)", "treasury only has $1,500", "borrowed $1,000 at 0.5%/month (debt "
            "$1,000)", "repaid $400 (debt $600)"]
    assert [r["message"] for r in box.executed][:4] == core
    assert [e["message"] for e in FleetState.of(env).log if e["kind"] == "action"][:4] == core
    monkeypatch.setattr(T.ToolBox, "_money_note", lambda self, *a: 1 / 0)          # advice never fails an action
    res = box.call("allocate_budget", {"department": "fire", "amount": 100})
    assert res["ok"] and res["message"] == "transferred $100 to fire (now $100)" and w.dept_funds["fire"] == 100


def test_the_report_counts_overrides():
    from civitas.fleet.report import fleet_report, turn_row
    env = _env()
    FleetState.of(env).update_plan(0, goals=["grow"])
    trace = [{"tool": "allocate_budget", "result": {"ok": False, "refused": "take_back", "error": "below its keep"}},
             {"tool": "allocate_budget", "result": {"ok": True, "message": "reclaimed $900", "override": "take_back"}},
             {"tool": "take_loan", "result": {"ok": True, "message": "borrowed $3,000", "override": "loan"}}]
    rows = [turn_row({"role": "mayor", "month": m, "tool_trace": trace}) for m in (0, 3)]
    assert [(r["refused"], r["overrides"]) for r in rows] == [(1, 2), (1, 2)]
    assert turn_row({"role": "fire", "month": 2})["overrides"] == 0
    rep, md = fleet_report(env, {"run_id": "r", "seed": 0, "months_played": 6}, rows)
    assert (rep["roles"]["mayor"]["refused"], rep["roles"]["mayor"]["overrides"]) == (2, 4)
    assert "| repeats (look-ups / refused / overrides) |" in md and "| 0 / 2 / 4 |" in md
