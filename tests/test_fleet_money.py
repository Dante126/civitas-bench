"""Mayor money (docs/plans/2026-09-29-mayor-money.md): the money model, the mayor's briefing, the money guards and
the money results. Each unit's tests sit under its own header."""
from types import SimpleNamespace

import pytest

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
    assert upkeep_of(w, "transport", "bus_line") == C.BUS_LINE_UPKEEP + 2 * C.BUS_STOP_UPKEEP    # its 2 stops at least
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
    # the fresh map: first_build fills Need.upkeep, road_build doesn't (the road Need gets 0); utilities names a
    # building per short supply, their upkeep summed (v20 fixes)
    coal, tower = T.first_build(env, "utilities").named
    road, = T.road_build(env).named
    assert (coal.what, coal.upkeep, tower.what, tower.upkeep, road.upkeep) == (
        "coal plant", C.TIER_LINES["coal"][0]["upkeep"], "water tower", C.TIER_LINES["tower"][0]["upkeep"], 0)
    f = funding_gaps(env, FleetState.of(env))
    assert f.to_fund == [FundItem("utilities", "first/supply", 6200, 110, "its first coal plant and water tower"),
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
    _builds(monkeypatch, {"fire": FB(2500, (N("fire station", 3, 0, first=True, upkeep=55),), 2500, (),
                                     keep_for="its first fire station")},
            road=FB(1410, (ROAD,), 1410, ()))
    m = {d: DeptMoney(env, d) for d in ("planning", "police", "fire", "health")}
    assert {d: (x.keep, x.keep_why) for d, x in m.items()} == {
        "planning": (1410, f"for its {ROAD_NAME}"), "police": (1316, "2 x its $658 spending"),
        "fire": (2500, "for its first fire station"), "health": (1000, "the $1,000 minimum")}
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
TO_FUND = ("- to fund: first/supply: education $5,000 (upkeep +$75/month) | growth: planning $315 | another: police "
           "$2,500, health $1,200 ($3,700, upkeep +$95/month)")


def test_cover_clause_is_greedy():
    """An item that doesn't fit is skipped, not the end: a $5,000 first high school must not hide a $315 road.
    First/supply and growth may use the treasury and idle money above $3,000 (Funding.dip), another only what is
    above the reserve."""
    w = _fake(*[(1300, 1100)] * 4)                      # reserve 3 x $1,100; operations +$200 a month
    assert reserve(w) == 3300
    assert spend_lines(w, _funding(w, 5300, ITEMS)) == [
        "- to spend: $2,000 above the $3,300 reserve (3 x $1,100 expenses); operations +$200/month; first/supply and "
        "growth may use $2,300 (treasury above $3,000)",
        TO_FUND + ": $2,300 covers planning, health; the other $7,500 would go under the reserve"]
    assert spend_lines(w, _funding(w, 5300, ITEMS, {"transport": 1508})) == [
        "- to spend: $2,000 above the $3,300 reserve (3 x $1,100 expenses) + $1,508 idle to take back = $3,508; "
        "operations +$200/month; first/supply and growth may use $3,808 (treasury + idle above $3,000)",
        TO_FUND + ": $3,808 covers planning, police; the other $6,200 would go under the reserve"]
    assert spend_lines(w, _funding(w, 3000, ITEMS, {"transport": 1508}))[0] == (
        "- to spend: $1,208: $1,508 idle to take back, less the $300 the treasury lacks for the $3,300 reserve "
        "(3 x $1,100 expenses); operations +$200/month; first/supply and growth may use $1,508 (treasury + idle above "
        "$3,000)")
    assert spend_lines(w, _funding(w, 50000, ITEMS))[1] == (
        TO_FUND + ": $47k covers education, planning, police, health")
    assert spend_lines(w, _funding(w, 3400, ITEMS))[1] == (    # $100 above the reserve, the road $300 below it
        TO_FUND + ": $400 covers planning; the other $8,700 would go under the reserve")
    assert spend_lines(w, _funding(w, 3310, ITEMS))[1] == (
        TO_FUND + ": $310 covers none of them; all $9,015 would go under the reserve")
    assert spend_lines(w, _funding(w, 5300, [])) == [     # nothing first/supply or growth: no allowance to name
        "- to spend: $2,000 above the $3,300 reserve (3 x $1,100 expenses); operations +$200/month",
        "- to fund: nothing ready"]
    only = [i for i in ITEMS if i.kind == "first/supply"]
    assert spend_lines(w, _funding(w, 3310, only))[1] == (    # only first/supply left: they may go to $3,000
        "- to fund: first/supply: education $5,000 (upkeep +$75/month): $310 covers none of them; all $5,000 would go "
        "under $3,000")


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
                                         "+ idle above $3,000)")
    f = _funding(w, 2000, ITEMS, {"transport": 1200})    # idle money counts only once the treasury reaches $3,000
    assert f.hold and f.dip == 200
    w = _fake(*[(1100, 1300)] * 4)                      # a deficit: 6 x expenses, and operations don't close it
    f = _funding(w, 2500, ITEMS)
    assert f.hold and f.dip == 0
    assert spend_lines(w, f) == [
        "- to spend: none (hold): the treasury is $5,300 under the $7,800 reserve (6 x $1,300 expenses: last "
        "month's deficit was $200); operations -$200/month; nothing above $3,000", TO_FUND]
    w = _fake()                                         # month 0: the minimum, no operations yet
    f = _funding(w, RESERVE_MIN, ITEMS)
    assert f.hold and spend_lines(w, f)[0] == (
        "- to spend: none (hold): the treasury is at the $3,000 reserve; nothing above $3,000")


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
    assert cf.imports == sum(last["expense_breakdown"][k] for k in T.IMPORT_COSTS)
    assert abs(sum(cf.buildings.values()) + cf.services + cf.imports + cf.interest - cf.expenses) < 1e-6
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
from civitas.fleet.fmt import cash, money, signed  # noqa: E402
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
    assert fin()[0] == (f"- treasury {cash(w.treasury)} ({change()} since month 1: operations "
                        f"{signed(ops(), money)}), debt $0")          # no top-ups since month 1: left out
    assert mayor(type="allocate_budget", department="fire", amount=2500)
    env.end_month()                                     # month 3
    assert fin()[0] == (f"- treasury {cash(w.treasury)} ({change()} since month 1: operations {signed(ops(), money)}, "
                        "top-ups -$2,500), debt $0")
    last = w.history[-1]
    b = {k: v for k, v in last["expense_breakdown"].items() if k in last["dept_upkeep"] and v > 0}
    imports = sum(last["expense_breakdown"][k] for k in T.IMPORT_COSTS)
    rest = last["expenses"] - sum(b.values()) - last["expense_breakdown"]["loan_interest"] - imports
    dear = sorted(b.items(), key=lambda kv: -kv[1])
    assert len(dear) <= 3 and fin()[1] == (
        f"- last month: income {money(last['income'])}, expenses {money(last['expenses'])}: buildings "
        f"{money(sum(b.values()))} (" + ", ".join(f"{k} {money(v)}" for k, v in dear)
        + f"), city services {money(rest)}" + (f", imports {money(imports)}" if round(imports) else "")
        + ", interest $0")
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
    assert lines[2] == "- debt $18.8k: $94/month interest, $31.2k more can be borrowed"   # 0.5% of the debt a month
    assert 18800 * C.LOAN_INTEREST == 94                # what a repayment saves: repay_loan's description, the playbook
    env.end_month()
    assert fin()[1].endswith(", interest $94")          # last month's, from the ledger
    w.debt = C.MAX_DEBT
    assert fin()[2] == "- debt $50k: $250/month interest, at the $50k limit"
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
    # its target above funds that cover its builds: no gap, flagged (Funding.unusable)
    assert lines["police"] == "- police: funds $2,500, target $5,000 (unusable now), spent $0; funds cover its builds"
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


def _topup(box, dept: str, amount: float, **kw) -> dict:
    """allocate_budget as the model does, a reserve refusal repeated with confirm: true (an earned override): the loan
    and note tests top up departments with nothing ready, which the reserve guard refuses once under the reserve."""
    res = box.call("allocate_budget", {"department": dept, "amount": amount, **kw})
    if res.get("refused") == "reserve":
        res = box.call("allocate_budget", {"department": dept, "amount": amount, **kw, "confirm": True})
    return res


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
    road = T.road_build(env).cost                       # the $508 it would keep is short of its road and zoning
    assert 508 < road
    why = (f"that takes planning below the $1,316 it keeps (2 x its $658 spending), so it can't pay for a road into "
           f"open land and zoning it ({money(road)}); you already took back $1,000 from it this turn; it has no idle "
           "money: leave its funds. To take it anyway, repeat this exact call with confirm: true")
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


def test_a_self_inflicted_loan_is_refused_until_confirmed(monkeypatch):
    """Smoke v13-v19: 10 of 15 loans ($34.8k of $60.0k) came in turns that started at $3,000 or more, and 14 of 15
    after the mayor's own top-ups had emptied the treasury; none was repaid. (The fresh map's first buildings cost
    more than the $598 above $3,000, which lets a loan pass since v22: the first part is checked with nothing ready.)"""
    env = _env()
    w = env.world
    box = _mayor(env, treasury=3598)
    loan = lambda amount, **kw: box.call("take_loan", {"amount": amount, **kw})
    over = ("the treasury holds $3,598, over the $3,000 minimum: a $5,000 loan costs $25/month until repaid (debt would "
            "be $5,000 of $50k); borrow only for a first building or supply need the treasury can't cover. Also, no "
            "first building, supply or planning road is waiting (nothing is ready to fund): the loan would sit in the "
            "treasury." + BORROW)
    with monkeypatch.context() as m:
        _builds(m, {})
        assert loan(5000) == {"ok": False, "refused": "loan", "error": over}
        assert w.debt == 0 and box.executed == [] and box._failed == {}
        res = loan("5000", confirm=True)                   # the earned repeat
        assert res["ok"] and res["override"] == "loan" and (w.debt, w.treasury) == (5000, 8598), res
        assert res["message"].startswith("borrowed $5,000 at 0.5%/month (debt $5,000)")
    # started at $3,000 or more, and this turn's own top-ups took it under
    w.debt = 0
    box = _mayor(env, treasury=5698)
    move = lambda dept, amount: _topup(box, dept, amount)     # nothing ready on the fresh map: earned overrides
    assert move("police", 4698)["ok"] and move("fire", 1000)["ok"] and w.treasury == 0
    assert box.call("take_loan", {"amount": 3000}) == {"ok": False, "refused": "loan", "error": (
        "the treasury started this turn at $5,698; this turn's top-ups ($5,698 net) took it under $3,000: take back "
        "(free) a top-up that isn't a first building or supply (police $4,698, fire $1,000) instead of a $3,000 loan "
        "at $15/month." + BORROW)}          # neither has a first building ready on the fresh map
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
    money_docs extends those tools' descriptions."""
    from civitas.core.actions import action_docs
    env = _env()
    w = env.world
    notes = []
    monkeypatch.setattr(T.ToolBox, "_money_note",
                        lambda self, name, args, res: notes.append((name, dict(args), res["ok"])) or f" | {name}")
    monkeypatch.setattr(T.ToolBox, "_explain", lambda self, name, args, message: " | explained")
    monkeypatch.setattr(T, "money_docs", lambda: {"take_loan": " Interest is due monthly."})
    box = _mayor(env, treasury=2000)
    desc = {s["function"]["name"]: s["function"]["description"] for s in box.schemas()}
    docs = action_docs()
    assert desc["take_loan"] == docs["take_loan"][1] + " Interest is due monthly." and desc["set_tax"] == docs[
        "set_tax"][1]
    assert _topup(box, "police", 500)["message"] == (         # police: nothing ready, under the reserve
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
    res = _topup(_mayor(env, treasury=w.treasury), "fire", 100)     # a new turn: no loan, so the confirm counts
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


# --- Unit D: result notes and look-ups (tools.py: ToolBox._money_note, money_docs, update_plan, overview)
from civitas.fleet.tools import ToolBox, overview  # noqa: E402

STATION = FB(2500, (N("police station", 4, 1, "coverage 40%", upkeep=55),), 2500, ())
SCHOOL = FB(2500, (N("elementary school", 3, 0, first=True, upkeep=45),), 2500, ())
NO_LOT = FB(None, (), 2500, ("a fire station",))
ROAD_315 = FB(315, (ROAD,), 315, ())


def _money(box, name: str, confirm: bool = False, **args) -> str:
    """Run one of the mayor's money actions as the model does, and return its result note: what the hook in _act
    appends to the model's message, and only there; `executed` and the history log keep the core text. With
    `confirm`, a refused call is repeated once with confirm: true, as an earned override takes."""
    sent = {**args, "confirm": True} if confirm else args
    res = box.call(name, sent)
    if confirm and res.get("refused"):
        res = box.call(name, sent)
    core = box.executed[-1]
    assert core["action"]["type"] == name and core["ok"] == res["ok"]
    note = box._money_note(name, args, core)
    assert res["message"] == core["message"] + note        # the hook in _act appends it to the model's message
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
    # top-ups: what the department can buy now, then the treasury against the reserve and what is still to fund (another
    # police station under the reserve: refused once, then earned)
    assert _money(box, "allocate_budget", confirm=True, department="police", amount=2500) == (
        " | police: can pay for another police station (upkeep +$55/month); treasury $2,788, $212 under the $3,000 "
        f"reserve; {school}")
    school = "still to fund: education $1,000 (first/supply)"
    assert _money(box, "allocate_budget", department="education", amount=1500) == (
        " | education: still $1,000 short of its first elementary school: less buys nothing; treasury $1,288, "
        f"$1,712 under the $3,000 reserve; {school}")
    assert _money(box, "allocate_budget", confirm=True, department="health", amount=200) == (
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
        " | planning now $0, $1,240 below what it keeps (2 x its $620 spending): it can't pay for a road into open "
        f"land and zoning it ($315); treasury $3,748, $748 above the $3,000 reserve; {school}, planning $315 (growth)")
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
        " | police: can pay for another police station (upkeep +$55/month); treasury $8,711, $5,000 above the $3,711 "
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


def test_the_treasury_shown_can_be_allocated():
    """Treasury figures round down: $1,937.60 read as $1,938 and the mayor's "allocate 1938" failed."""
    env = _env()
    box = _mayor(env, treasury=1937.6)
    res = box.call("allocate_budget", {"department": "police", "amount": 5000})
    assert res["ok"] is False and res["message"].startswith("treasury only has $1,937 ")
    assert "treasury $1,937, " in res["message"] and box.call("city_overview", {})["treasury"] == 1937
    assert box.call("allocate_budget", {"department": "utilities", "amount": 1937})["ok"]     # first/supply: no guard
    assert cash(12399.9) == "$12.3k" and cash(1937.6) == "$1,937" and cash(-5.4) == "-$5"


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
        "; first/supply at a $0 target: education")
    assert res["next"] == (           # month 0: no operations yet
        "to spend: $26.5k above the $3,000 reserve; to fund: first/supply: education $2,000 (upkeep +$45/month) | "
        "growth: planning $315 | another: police $2,500 (upkeep +$55/month): $26.5k covers education, planning, "
        "police; first/supply at a $0 target: education")
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
        "to spend: $27k above the $3,000 reserve; to fund: nothing ready; nothing to fund now: police (nothing ready to "
        "buy), education (nothing ready to buy)")       # targets above their funds with nothing to buy


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


# --- Unit E: the mayor's playbook (configs/fleet_smoke.toml, fleet_trinity.toml, fleet_region_smoke.toml)
MAYOR_CONFIGS = ("configs/fleet_smoke.toml", "configs/fleet_trinity.toml", "configs/fleet_region_smoke.toml")


def test_the_mayor_playbook_follows_the_money_lines():
    """The mayor's playbook reads the briefing's money lines (to spend / to fund, needs lines with upkeep), funds
    first buildings by need rather than target, borrows only for first buildings or supply, repays, and treats a
    refusal as advice; its numbers match the code (the reserve, the loan interest)."""
    from civitas.config import load_config
    books = [load_config(p)["agents"]["mayor"]["playbook"] for p in MAYOR_CONFIGS]
    assert books[0] == books[1] == books[2]                             # one fleet
    book = " ".join(books[0].split())
    for s in ("utilities $7,000 for its first plants",
              "fund every first building a needs line names",
              "whatever its target",
              "After month 0 give utilities what its needs line asks, not its target",
              "every building adds upkeep for good (its needs line says how much)",
              "the Departments block (most urgent first)",
              "the idle money the 'to spend' line counts",
              "then fund what the 'to fund' line says the money covers, in its order",     # greedy: not stop at a misfit
              "A department with no needs line needs no money now, whatever its target",
              "each result shows the treasury and what is still to fund",
              f"keep at least {money(RESERVE_MIN)} or 3 months of expenses in the treasury (6 after a deficit month)",
              "First buildings, supply and planning's road may use the treasury and idle money above $3,000; anything "
              "else only the money above the reserve",                                     # Funding.dip, in a hold or not
              "Loans only for a first building or supply need the money to spend can't cover",
              "positive after its upkeep and the interest (its needs line and Finances show them)",
              "never after this turn's own top-ups emptied the treasury (take one back instead)",
              f"each $1,000 repaid saves {money(1000 * C.LOAN_INTEREST)} a month",
              "A refused call says why and what to do instead: do that. confirm: true counts only on a repeat of "
              "the exact refused call"):
        assert s in book, s
    for s in ("allocate nothing new", "repay once the treasury is above $25,000", "top up the departments below "
              "target", "first buildings before second ones", "stop where the money to spend runs out",
              "has nothing ready to buy", "(Finances shows both)"):
        assert s not in book, s


# --- Review fixes (unit 14): take-back wording, first buildings ahead of cheaper builds, the dip, the loan guard's
# advice, and polish (docs/plans/2026-09-29-mayor-money.md, "As built")
from civitas.fleet.tools import SiteList  # noqa: E402


def test_take_back_texts_say_what_the_money_left_buys(monkeypatch):
    """Review: "so it can't build next quarter" was false whenever the keep came from the $1,000 minimum, 2 x the
    spending or the dearest first building while the money left still paid for the cheapest ready build (month 12:
    planning left $950 with a $640 road). The clause now follows the numbers, and the keep names its building."""
    env = _env()
    _months_spent(env, {}, {}, {})
    station = N("police station", 4, 1, "coverage 40%", upkeep=55)
    _builds(monkeypatch, {"police": FB(2500, (station,), 4000, (), keep_for="its first jail"),
                          "fire": FB(2500, (N("fire station", 3, 2, "coverage 77%", upkeep=55),), 2500, (),
                                     keep_for="another fire station")},
            road=FB(640, (ROAD,), 640, (), keep_for=T.a_or_an(ROAD_NAME)))
    box = _mayor(env, planning=1900, police=4900, fire=2500)
    m = {d: DeptMoney(env, d) for d in ("planning", "police", "fire")}
    assert {d: (x.keep, x.keep_why, x.idle) for d, x in m.items()} == {
        "planning": (1000, "the $1,000 minimum", 0), "police": (4000, "for its first jail", 0),
        "fire": (2500, "for another fire station", 0)}
    take = lambda dept, amount, **kw: box.call("allocate_budget", {"department": dept, "amount": amount, **kw})
    tail = "; it has no idle money: leave its funds. To take it anyway, repeat this exact call with confirm: true."
    # left $950: the $640 road still fits, so nothing is said it can't do
    assert take("planning", -950)["error"] == "that takes planning below the $1,000 it keeps (the $1,000 minimum)" + tail
    assert take("planning", -1400)["error"] == (
        "that takes planning below the $1,000 it keeps (the $1,000 minimum), so it can't pay for a road into open "
        "land and zoning it ($640)" + tail)
    assert take("police", -1400)["error"] == "that takes police below the $4,000 it keeps (for its first jail)" + tail
    assert take("police", -3000)["error"] == (
        "that takes police below the $4,000 it keeps (for its first jail), so it can't pay for another police "
        "station ($2,500)" + tail)
    assert take("fire", -500)["error"] == "that takes fire below the $2,500 it keeps (for another fire station)" + tail
    # the result after an override says the same
    assert take("police", -1400, confirm=True)["ok"]
    assert box._money_note("allocate_budget", {"department": "police", "amount": -1400},
                           {"ok": True, "message": "reclaimed"}).startswith(
        " | police now $3,500, $500 below what it keeps (for its first jail); treasury")
    assert take("planning", -1400, confirm=True)["ok"]
    assert box._money_note("allocate_budget", {"department": "planning", "amount": -1400},
                           {"ok": True, "message": "reclaimed"}).startswith(
        " | planning now $500, $500 below what it keeps (the $1,000 minimum): it can't pay for a road into open "
        "land and zoning it ($640); treasury")


def _lists(**purposes) -> dict:
    """Fake site lists: purpose -> SiteList of that many ready sites on a free lot."""
    return {p: SiteList([{"x": 1 + i, "y": 1, "x1": 1, "y1": 1, "x2": 5, "y2": 1} for i in range(n)], None, None)
            for p, n in purposes.items()}


def test_a_first_building_the_city_lacks_comes_before_a_cheaper_build(monkeypatch):
    """Review: with a $30 avenue stretch ready, transport's first $1,500 bus line dropped off the needs line (seed 0,
    months 42-69: "funds $738 < target $2,000, spent $0; funds cover its next build", 0 bus lines), and education's
    first high school hid behind another elementary school. A ready first building the city lacks is named first."""
    env = _env()
    w = env.world
    fake = {"transport": _lists(bus_stops=3, congested_roads=2),
            "education": _lists(school=4, high_school=2, university=1)}
    real_lists, real_have = T.site_lists, T.buildings_of
    monkeypatch.setattr(T, "site_lists", lambda w, role, obs: fake[role] if role in fake else real_lists(w, role, obs))
    monkeypatch.setattr(T, "buildings_of", lambda w, name: 2 if name == "elementary_school" else real_have(w, name))
    monkeypatch.setattr(T, "best_stretch", lambda w, purpose, sites: (5, 5))
    monkeypatch.setattr(w, "unlocked", lambda key: True)
    bus = T.first_build(env, "transport")
    assert (bus.cost, bus.dearest, bus.keep_for) == (C.BUS_LINE_COST, C.BUS_LINE_COST, "its first bus line")
    assert [(n.what, n.have, n.first) for n in bus.named] == [("bus line", 0, True)]
    w.dept_funds["transport"] = 738
    m = DeptMoney(env, "transport")
    assert (m.kind, m.top_up, m.what) == ("first/supply", C.BUS_LINE_COST - 738, "its first bus line")
    assert FundItem("transport", "first/supply", C.BUS_LINE_COST - 738, m.upkeep, "its first bus line") in \
        funding_gaps(env, FleetState.of(env)).to_fund
    w.dept_funds["transport"] = C.BUS_LINE_COST
    assert DeptMoney(env, "transport").state == "funds cover its next build"
    w.bus_lines.append({"id": 1, "stops": [(1, 1), (5, 1)]})     # it has one: another line while its list offers one
    again = T.first_build(env, "transport")
    assert again.cost == C.BUS_LINE_COST and [(n.what, n.have, n.first) for n in again.named] == [("bus line", 1, False)]
    fake["transport"]["bus_stops"] = fake["transport"]["bus_stops"]._replace(short="buses have room")
    avenue = T.first_build(env, "transport")                     # buses have room: the avenue stretch is next again
    assert avenue.cost == 5 * C.ROAD_UPGRADE_COST[C.ROAD_CLASSES["avenue"]] and avenue.named[0].what == "5 avenue tiles"
    school = T.first_build(env, "education")                      # another elementary school is cheaper
    assert [(n.what, n.have) for n in school.named] == [("high school", 0)] and school.cost == 5000
    assert (school.dearest, school.keep_for) == (12000, "its first university")
    w.dept_funds["education"] = 3291
    m = DeptMoney(env, "education")
    assert (m.kind, m.top_up, m.keep_why) == ("first/supply", 5000 - 3291, "for its first university")


def test_a_line_without_a_needs_line_says_why(monkeypatch):
    """Review: "nothing ready to buy" was shown for utilities with a pump and an outlet ready but their supply spare
    (seed 0, month 108), and for a department whose only ready sites are supplementary."""
    env = _env()
    fake = {"utilities": _lists(pump=2, sewage=1), "fire": _lists(shelter=2)}
    real_lists = T.site_lists
    monkeypatch.setattr(T, "site_lists", lambda w, role, obs: fake[role] if role in fake else real_lists(w, role, obs))
    monkeypatch.setattr(T, "_wanted", lambda w, purpose: purpose not in T.SUPPLY_OF)     # every supply has room
    b = T.first_build(env, "utilities")
    assert (b.cost, b.named, b.dearest, b.room) == (None, (), 0, True)
    m = DeptMoney(env, "utilities")
    assert (m.state, m.keep, m.kind) == ("supply has room", T.IDLE_MIN, "nothing")
    b = T.first_build(env, "fire")
    assert (b.cost, b.extra) == (None, ("shelter", C.TIER_LINES["shelter"][0]["cost"]))
    assert DeptMoney(env, "fire").state == "only a shelter ready ($1,500)"
    assert T.topped_up(DeptMoney(env, "utilities")) == "needs nothing now: supply has room"
    assert T.topped_up(DeptMoney(env, "fire")) == "had only a shelter ready ($1,500)"
    fake["fire"] = {}
    assert T.first_build(env, "fire") is None and DeptMoney(env, "fire").state == "nothing ready to buy"


def test_first_buildings_may_use_more_as_money_rises():
    """Review: a hold gave first/supply the treasury above $3,000 plus idle, but $1 more idle money (or treasury past
    the reserve) left the hold and covered them against the money above the reserve only: a richer city was told to
    stop funding the first school it had just been told to fund. First/supply and growth now always may use the
    treasury and idle money above $3,000, another only what is above the reserve."""
    w = _fake(*[(2000, 1535)] * 4)                      # reserve 3 x $1,535
    assert reserve(w) == 4605
    items = [FundItem("education", "first/supply", 1380, 45, "its first elementary school"),
             FundItem("police", "another", 2379, 55, "another police station"),
             FundItem("fire", "another", 254, 55, "another fire station"),
             FundItem("health", "another", 643, 40, "another clinic")]
    covers = lambda f: "education" in T.fund_cover(f)
    for sweep in ([(t, 0) for t in range(0, 9001, 7)], [(1000, i) for i in range(0, 6001, 7)]):
        last_dip, covered = -1.0, False
        for treasury, idle in sweep:
            f = _funding(w, treasury, items, {"planning": idle} if idle else None)
            assert f.dip == max(0, treasury + idle - RESERVE_MIN) and f.dip >= f.spendable, (treasury, idle)
            assert f.dip >= last_dip and (covers(f) or not covered), (treasury, idle)
            last_dip, covered = f.dip, covers(f)
        assert covered
    # the review's states, either side of the idle boundary and of the reserve
    f = _funding(w, 1000, items, {"planning": 3605})
    assert f.hold and spend_lines(w, f)[0].endswith(
        "; first/supply and growth may use $1,605 (treasury + idle above $3,000)")
    assert spend_lines(w, f)[1].endswith(": $1,605 covers education")
    f = _funding(w, 1000, items, {"planning": 3606})
    assert not f.hold and spend_lines(w, f)[0] == (
        "- to spend: $1: $3,606 idle to take back, less the $3,605 the treasury lacks for the $4,605 reserve (3 x "
        "$1,535 expenses); operations +$465/month; first/supply and growth may use $1,606 (treasury + idle above "
        "$3,000)")
    assert spend_lines(w, f)[1].endswith(": $1,606 covers education; the other $3,276 would go under the reserve")
    for treasury in (4604, 4606, 5900):
        assert covers(_funding(w, treasury, items)), treasury
    f = _funding(w, 5900, items)
    assert spend_lines(w, f) == [
        "- to spend: $1,295 above the $4,605 reserve (3 x $1,535 expenses); operations +$465/month; first/supply and "
        "growth may use $2,900 (treasury above $3,000)",
        "- to fund: first/supply: education $1,380 (upkeep +$45/month) | another: police $2,379, fire $254, health "
        "$643 ($3,276, upkeep +$150/month): $2,900 covers education; the other $3,276 would go under the reserve"]
    # the reserve at the minimum: first/supply may use what there is to spend, and the line says nothing more
    w = _fake(*[(1300, 900)] * 4)
    f = _funding(w, 5000, items)
    assert f.dip == f.spendable == 2000 and "may use" not in spend_lines(w, f)[0]
    assert spend_lines(w, f)[1].endswith(": $2,000 covers education, fire; the other $3,022 would go under the reserve")


def test_the_hold_names_what_it_covers():
    """Review: "covers none of them; all $540 would go under $3,000" counted only first/supply and growth while the
    same line listed $9,620 of another."""
    w = _fake(*[(1500, 1300)] * 4)                      # reserve $3,900
    items = [FundItem("planning", "growth", 540, 0, "a road into open land and zoning it"),
             FundItem("police", "another", 2500, 55, "another police station")]
    f = _funding(w, 3400, items)
    assert f.hold and spend_lines(w, f)[1] == (
        "- to fund: growth: planning $540 | another: police $2,500 (upkeep +$55/month): $400 covers no first/supply "
        "or growth; their $540 would go under $3,000")


def test_a_loan_after_the_turn_s_own_moves_names_what_to_undo(monkeypatch):
    """Review: refusal (b) always advised taking back "a top-up that isn't a first building or supply", even when
    every top-up was one, and counted only allocations, so a repayment that drained the treasury was blamed on a $100
    top-up."""
    env = _env()
    w = env.world
    clinic = FB(2000, (N("clinic", 3, 0, first=True, upkeep=40),), 2000, ())     # a first building still waits
    _builds(monkeypatch, {"police": STATION, "education": SCHOOL, "health": clinic})
    box = _mayor(env, treasury=5698)
    move = lambda dept, amount, **kw: box.call("allocate_budget", {"department": dept, "amount": amount, **kw})
    assert move("education", 2500)["ok"] and move("police", 2500)["refused"] == "reserve"   # another, under $3,000
    assert move("police", 2500, confirm=True)["ok"] and _topup(box, "fire", 698)["ok"]   # fire: nothing ready
    assert box.call("take_loan", {"amount": 3000}) == {"ok": False, "refused": "loan", "error": (
        "the treasury started this turn at $5,698; this turn's top-ups ($5,698 net) took it under $3,000: take back "
        "(free) a top-up that isn't a first building or supply (police $2,500, fire $698) instead of a $3,000 loan "
        "at $15/month." + BORROW)}
    # every top-up for a first building: the loan is the playbook's own case, so it runs
    box = _mayor(env, treasury=4000, education=0)
    assert move("education", 2500)["ok"]
    res = box.call("take_loan", {"amount": 2000})
    assert res["ok"] and "override" not in res, res
    # a repayment that drained it is named, and a loan after it only borrows it back
    w.debt = 10000
    box = _mayor(env, treasury=5000)
    assert move("police", 100)["ok"] and box.call("repay_loan", {"amount": 3000})["ok"] and w.treasury == 1900
    assert box.call("take_loan", {"amount": 3000})["error"] == (
        "the treasury started this turn at $5,000; this turn's top-ups ($100 net) and repaying $3,000 took it under "
        "$3,000: a $3,000 loan at $15/month borrows back what you just repaid; take back (free) a top-up that isn't "
        "a first building or supply (police $100) instead." + BORROW)
    box = _mayor(env, treasury=5000)
    assert box.call("repay_loan", {"amount": 4000})["ok"]
    assert box.call("take_loan", {"amount": 4000})["error"] == (
        "the treasury started this turn at $5,000; repaying $4,000 took it under $3,000: a $4,000 loan at $20/month "
        "borrows back what you just repaid." + BORROW)


def test_money_docs_read_the_rate_when_the_toolbox_is_built():
    """Review: the take_loan text told fleet mayors what the finance score reads (other LLM agents never see it) and
    read the rate at import, before a LOAN_INTEREST override."""
    assert T.money_docs() == {"take_loan": " Interest is $5 a month per $1,000 until repaid.",
                              "repay_loan": " Each $1,000 repaid saves $5 a month in interest."}
    old = C.LOAN_INTEREST
    try:
        env = _env(balance={"LOAN_INTEREST": 0.01})
        desc = {s["function"]["name"]: s["function"]["description"] for s in ToolBox(env, "mayor").schemas()}
        assert desc["take_loan"].endswith(" Interest is $10 a month per $1,000 until repaid.")
        assert desc["repay_loan"].endswith(" Each $1,000 repaid saves $10 a month in interest.")
        assert "score" not in desc["take_loan"]
    finally:
        C.LOAN_INTEREST = old


def test_money_polish(monkeypatch):
    """Review polish: a bus line's upkeep counts its two stops; month 0 has no operations; "close that" stays within
    the game; top-ups name upkeep as upkeep and don't say less buys nothing where it buys some; the $0-target list
    says first/supply; imports show apart from city services."""
    env = _env()
    w = env.world
    assert upkeep_of(w, "transport", "bus_line") == C.BUS_LINE_UPKEEP + 2 * C.BUS_STOP_UPKEEP
    f = funding_gaps(env, FleetState.of(env))
    assert spend_lines(w, f)[0] == "- to spend: $27k above the $3,000 reserve"          # month 0: no operations yet
    fk = _fake(*[(1500, 1300)] * 4)                     # reserve $3,900, operations +$200 a month: 2 months to close
    f = _funding(fk, 3500, ITEMS)._replace(months_left=1)
    assert "close that" not in spend_lines(fk, f)[0] and "operations +$200/month;" in spend_lines(fk, f)[0]
    assert "close that in 2 months" in spend_lines(fk, f._replace(months_left=2))[0]
    # top-ups: planning's road and per-tile needs buy part with less
    _builds(monkeypatch, {"fire": FB(900, (N("3 levee tiles", 1, None, upkeep=3),), 900, ()), "police": STATION},
            road=ROAD_315)
    w.dept_funds.update(fire=600, planning=100, police=1000)
    assert T.topped_up(DeptMoney(env, "fire")) == "still $300 short of 3 levee tiles: less buys fewer tiles"
    assert T.topped_up(DeptMoney(env, "planning")) == "still $215 short of a road into open land and zoning it"
    assert T.topped_up(DeptMoney(env, "police")) == "still $1,500 short of another police station: less buys nothing"
    w.dept_funds["police"] = 2500
    assert T.topped_up(DeptMoney(env, "police")) == "can pay for another police station (upkeep +$55/month)"
    # the $0-target list is first/supply, not only first buildings (utilities' supply is on it)
    _builds(monkeypatch, {"utilities": FB(1200, (N("sewage outlet", 2, 1, "sewage 95% used", upkeep=15),), 1200, ())})
    res = ToolBox(env, "mayor").call("update_plan", {"goals": ["grow"]})
    assert res["first_buildings"] == ["utilities"] and res["next"].endswith("; first/supply at a $0 target: utilities")
    # imports apart from city services
    env.end_month()
    last = w.history[-1]
    last["expense_breakdown"].update(service_imports=173.4, regional_trade=0.0)
    last["expenses"] += 173.4
    cf = cash_flow(w)
    assert round(cf.imports) == 173 and abs(sum(cf.buildings.values()) + cf.services + cf.imports + cf.interest
                                            - cf.expenses) < 1e-6
    fin = _mayor_block(env, "## Finances")[1]
    assert fin.endswith(f", city services {money(cf.services)}, imports $173, interest $0"), fin


# --- v20 fixes: utilities (tools.py: short_supplies, first_build's supply builds, DeptMoney's bound, bank_keep;
# briefing.py: utilities' short-now line and its Ready sites' riverbank wording, the mayor's needs line)
def _to_the_river(env) -> None:
    """Planning's month-0 road: the main road on to the river and over it (free bank lots (25,23), (29,23), (26,25),
    (30,25))."""
    w = env.world
    w.dept_funds["planning"] = 5000
    assert apply_action(w, "planning", {"type": "build_road", "x1": 20, "y1": 24, "x2": 38, "y2": 24}).ok
    w.dept_funds["planning"] = 0


def _one_bank_lot(env) -> None:
    """Smoke v20 seed 1, month 3: a pump built, one free riverbank lot left (29,23), power and sewage short."""
    w = env.world
    _to_the_river(env)
    w.dept_funds["utilities"] = 1500
    assert apply_action(w, "utilities", {"type": "build_water", "kind": "pump", "x": 25, "y": 23}).ok
    for x, y in ((26, 25), (30, 25)):
        w.struct[y, x] = C.RES
    w.refresh()
    assert T.scarce_bank(w) == {(29, 23)}


def test_a_short_supply_wants_a_quarter_of_growth_or_half_again_its_demand():
    """What a short supply wants added (supply_need): max(1.5 x demand, demand + a young city's quarter of growth)
    less what the city's own buildings make (imports left out). Fresh map: power 6 (the town hall in winter), 1.5 of
    it imported."""
    assert (T.SUPPLIES, T.SUPPLY_MARGIN, T.SUPPLY_GROWTH) == (("power", "water", "sewage"), 1.5,
                                                              {"power": 60, "water": 50, "sewage": 50})
    env = _env()
    w = env.world
    m = w.metrics()
    assert (m["power_supply"], m["service_imports"]["power"]) == (1.5, 1.5)
    assert [T.supply_need(w, s) for s in T.SUPPLIES] == [66, 55, 55]
    big = SimpleNamespace(metrics=lambda: {"power_demand": 1000.0, "power_supply": 950.0,
                                           "service_imports": {"power": 0.0}})
    assert T.supply_need(big, "power") == 550                           # half again the demand, less 950
    coal, wind = T.Price(4000, "coal plant", name="coal"), T.Price(2500, "wind turbine", name="wind")
    opts = [("coal_or_gas", coal, 250.0, {"x": 0, "y": 23}), ("wind", wind, 58.8, {"x": 39, "y": 25})]
    pick = lambda need: T.pick_supply(need, opts)[0]
    assert (pick(66), pick(58.8), pick(20), pick(550)) == ("coal_or_gas", "wind", "wind", "coal_or_gas")
    assert T.pick_supply(66, []) is None
    # its output: the tier's, a wind turbine's at its site's wind strength
    assert (T.supply_output("coal", "power", {}), T.supply_output("wind", "power", {"wind": 1.47})) == (250, 40 * 1.47)
    assert (T.supply_output("pump", "water", {}), T.supply_output("tower", "water", {}),
            T.supply_output("outlet", "sewage", {})) == (250, 120, 250)


def test_the_needs_line_names_a_building_for_every_short_supply():
    """Smoke v20 seed 1, month 0: the needs line named only the cheapest wanted building ("its first water tower
    $2,200"), the mayor gave exactly that instead of the playbook's $7,000, utilities built a pump and could pay for
    no power plant: power stayed at 0-18% until month 10 and the city ended at 504 residents. One building per short
    supply now, each the cheapest ready one that covers what the supply wants (supply_need; wind's 59 falls short of
    power's 66, the coal plant's 250 doesn't); a supply with no ready site (the pump and outlet before the road reaches
    the river) stays blocked, and its building still counts in what utilities keeps."""
    env = _env()
    w = env.world
    b = T.first_build(env, "utilities")
    assert b.all_of and b.cost == 6200
    assert [(n.what, n.have, n.first, n.status, n.upkeep) for n in b.named] == [
        ("coal plant", 0, True, "power 25% covered", 80), ("water tower", 0, True, "water 24% covered", 30)]
    assert (b.what, b.upkeep) == ("its first coal plant and water tower", 110)
    assert b.text() == "its first coal plant and water tower ($110/month)"     # the City block has each supply
    assert T.labels_and(["another coal plant", "its first water pump", "its first sewage outlet"]) == (
        "another coal plant, its first water pump and sewage outlet")
    assert (b.dearest, b.keep_for) == (7400, "its first coal plant, water tower and sewage outlet")   # + the outlet
    line = next(ln for ln in briefing(env, "mayor", mark=False).splitlines() if ln.startswith("- utilities:"))
    assert line == ("- utilities: funds $0; needs $6,200 for its first coal plant and water tower ($110/month); sewage "
                    "outlet waits for a road to the river")               # review: the outlet's wait named too
    assert b.waits == ("sewage outlet waits for a road to the river",)
    f = funding_gaps(env, FleetState.of(env))
    assert f.to_fund[0] == FundItem("utilities", "first/supply", 6200, 110, "its first coal plant and water tower")
    m = f.depts["utilities"]
    assert (m.bound, m.keep, m.keep_why) == (7400, 7400, "for its first coal plant, water tower and sewage outlet")
    _to_the_river(env)                                  # the road reaches the river: the pump and outlet are ready
    b = T.first_build(env, "utilities")
    assert [(n.what, n.status) for n in b.named] == [("coal plant", "power 25% covered"),
                                                      ("water pump", "water 24% covered"),
                                                      ("sewage outlet", "sewage 0% covered")]
    assert (b.cost, b.upkeep, b.dearest) == (6700, 120, 6700)
    assert b.text() == "its first coal plant, water pump and sewage outlet ($120/month)"
    # v20 seed 1, month 6: $4,400 was above the $4,000 coal plant (the old bound: the dearest priced purpose), so the
    # line read "funds cover its builds" with power at 17% and sewage at 0%
    w.dept_funds["utilities"] = 4400
    m = DeptMoney(env, "utilities")
    assert (m.kind, m.top_up, m.upkeep, m.what) == ("first/supply", 2300, 120,
                                                    "its first coal plant, water pump and sewage outlet")
    assert T.topped_up(m) == ("still $2,300 short of its first coal plant, water pump and sewage outlet: less buys "
                              "fewer of them")
    w.dept_funds["utilities"] = 6700
    m = DeptMoney(env, "utilities")
    assert m.needs is None and m.state == "funds cover its next build"
    assert T.topped_up(m) == "can pay for its first coal plant, water pump and sewage outlet (upkeep +$120/month)"
    assert m.cant_pay(4400) == ""                       # what it keeps names them already
    assert DeptMoney(_env(), "utilities").cant_pay(4000) == "its first coal plant and water tower ($6,200)"


def test_supply_builds_share_a_scarce_riverbank_lot():
    """One free riverbank lot and both water and sewage short: the outlet, which has nowhere else to go, gets the lot,
    and water a tower near homes (at month 0 the coal plant and the tower, both ranked on (0,23), don't share it
    either: test_utilities_briefing_says_what_is_short_and_what_its_funds_build)."""
    env = _env()
    w = env.world
    _to_the_river(env)
    for x, y in ((25, 23), (26, 25), (30, 25)):
        w.struct[y, x] = C.RES
    w.refresh()
    lists = T.site_lists(w, "utilities", T.role_obs(env, "utilities"))
    builds = T.short_supplies(w, lists)
    assert [(b.supply, b.purpose, (b.site["x"], b.site["y"])) for b in builds] == [
        ("power", "coal_or_gas", (0, 23)), ("water", "near_homes", (30, 23)), ("sewage", "sewage", (29, 23))]
    assert lists["pump"].sites[0] == {"x": 29, "y": 23, "pollution": 0.0} and builds[1].output == 120
    assert T.bank_keep(w, builds) == {(29, 23): "sewage"}
    assert [n.what for n in T.first_build(env, "utilities").named] == ["coal plant", "water tower", "sewage outlet"]


def test_utilities_briefing_says_what_is_short_and_what_its_funds_build():
    """Smoke v20 seed 1, month 3: utilities had $4,400, enough for the $4,000 coal plant, but its one riverbank lot read
    "(riverbank: keep for pump/sewage)" on the wind line and "same site as wind" on the pump and sewage lines; it spent
    all 20 steps inspecting sewage sites and built nothing. Its budget block now says what is short and what its funds
    pay for, most urgent first (power lets a lot develop at all, water to level 2, sewage to 3), and the Ready sites
    lines say which purpose the lot is kept for."""
    env = _env()
    w = env.world
    _one_bank_lot(env)
    w.dept_funds["utilities"] = 4400
    text = briefing(env, "utilities", mark=False)
    budget = _block(text, "## Your budget")
    power = "power 25% covered (supply 1.50, demand 6, 1.50 imported)"    # as the mayor's needs line says it, and
    sewage = "sewage 0% covered (supply 0, demand 5)"   # the supply as Key metrics give it; imports aren't the city's
    assert budget[1] == (f"- supply needed now: {power}, {sewage}: your $4,400 pays for a coal plant at (0,23) "
                         "($4,000); a sewage outlet at (29,23) ($1,200) needs $800 more")
    ready = {ln.split(":")[0][2:]: ln for ln in _block(text, "## Ready sites")[:-1]}
    assert ready["pump"] == "- pump: 1 site, best (29,23), pollution 0 (riverbank: kept for sewage); $1,500"
    assert ready["sewage"] == "- sewage: 1 site, best (29,23) (riverbank: kept for sewage); $1,200"
    w.dept_funds["utilities"] = 5200                    # every one paid for
    assert _block(briefing(env, "utilities", mark=False), "## Your budget")[1] == (
        f"- supply needed now: {power}, {sewage}: your $5,200 pays for a coal plant at (0,23) ($4,000) and a sewage "
        "outlet at (29,23) ($1,200)")
    w.dept_funds["utilities"] = 400                     # none: the can't-pay line follows
    budget = _block(briefing(env, "utilities", mark=False), "## Your budget")
    assert budget[1] == (f"- supply needed now: {power}, {sewage}: your $400 pays for none of them; a coal plant at "
                         "(0,23) ($4,000) and a sewage outlet at (29,23) ($1,200) need $4,800 more")
    assert budget[2].startswith("- You can't pay for any ready site (the supply above needs $4,800 more). Don't call "
                                "build tools this turn: ask the mayor for $4,800 with a reason")
    env = _env()                                        # month 0, no road to the river: the outlet has no site yet
    w = env.world
    w.dept_funds["utilities"] = 7000
    assert _block(briefing(env, "utilities", mark=False), "## Your budget")[1] == (
        f"- supply needed now: {power}, water 24% covered (supply 1.20, demand 5, 1.20 imported), sewage 0% covered "
        "(supply 0, demand 5; no site yet): your $7,000 pays for a coal plant at (0,23) ($4,000) and a water tower at "
        "(1,23) ($2,200)")                              # the two ranked (0,23) first
    w.dept_funds["utilities"] = 50000
    _to_the_river(env)                                  # every supply built: nothing short, no line
    for a in ({"type": "build_power_plant", "kind": "coal", "x": 0, "y": 23},
              {"type": "build_water", "kind": "pump", "x": 25, "y": 23},
              {"type": "build_sewage", "kind": "outlet", "x": 30, "y": 25}):
        assert apply_action(w, "utilities", a).ok, a
    assert "- supply needed now" not in briefing(env, "utilities", mark=False)


def test_the_mayor_playbook_says_what_the_month_0_needs_line_leaves_out():
    """Month 0: the mayor acts before planning's road reaches the river, so utilities' needs line names a plant and a
    tower ($6,200), and the pump and outlet it builds on its own turn wait; the playbook's $7,000 covers both."""
    from civitas.config import load_config
    books = [" ".join(load_config(p)["agents"]["mayor"]["playbook"].split()) for p in MAYOR_CONFIGS]
    assert books[0] == books[1] == books[2]
    assert ("utilities $7,000 for its first plants (its month-0 needs line leaves out the pump and outlet until "
            "planning's road reaches the river)") in books[0]


def test_ready_sites_say_which_purpose_a_scarce_bank_lot_is_kept_for():
    """Smoke v20 seed 1, month 3: "wind: 6 sites, best (33,23), wind 1.39 (riverbank: keep for pump/sewage)", "pump: 1
    site, same site as wind", "sewage: 1 site, same site as wind". With utilities' claims (bank_keep: the lot its
    short-now line builds the outlet on), each line says which purpose the lot is kept for, and a purpose that doesn't
    need the river is shown its next site."""
    lot, dry = {"x": 33, "y": 23, "wind": 1.39}, {"x": 34, "y": 23, "wind": 1.39}
    lists = {"wind": SiteList([lot, dry], None, None),
             "pump": SiteList([{"x": 33, "y": 23, "pollution": 0.0}], None, None),
             "sewage": SiteList([{"x": 33, "y": 23, "pollution": 0.0}], None, None),
             "near_homes": SiteList([{"x": 33, "y": 23, "homes_within_5": 4}], None, None)}
    prices = {"wind": T.Price(2500, "wind turbine"), "pump": T.Price(1500, "water pump"),
              "sewage": T.Price(1200, "sewage outlet"), "near_homes": T.Price(2200, "water tower")}
    assert T.ready_lines("utilities", lists, {(33, 23): "sewage"}, prices, 4400) == [
        "- wind: 2 sites, best (34,23), wind 1.39 (not (33,23): riverbank kept for sewage); $2,500",
        "- pump: 1 site, best (33,23), pollution 0 (riverbank: kept for sewage); $1,500",
        "- sewage: 1 site, best (33,23) (riverbank: kept for sewage); $1,200",
        "- near_homes: 1 site, best (33,23), 4 homes within 5 (riverbank: kept for sewage); $2,200"]
    # two lots, one for each: the pump's line moves to its own
    lists["pump"] = SiteList([{"x": 33, "y": 23, "pollution": 0.0}, {"x": 29, "y": 23, "pollution": 0.0}], None, None)
    assert T.ready_lines("utilities", {"pump": lists["pump"]}, {(33, 23): "sewage", (29, 23): "pump"}) == [
        "- pump: 2 sites, best (29,23), pollution 0 (riverbank: kept for pump) (not (33,23): riverbank kept for "
        "sewage)"]
    # a lot no short supply claims: kept for either, as before (only purposes that don't need the river are moved)
    assert T.ready_lines("utilities", {"wind": lists["wind"], "sewage": lists["sewage"]},
                         {(33, 23): "pump/sewage"}) == [
        "- wind: 2 sites, best (34,23), wind 1.39 (not (33,23): riverbank kept for pump/sewage)",
        "- sewage: 1 site, best (33,23)"]


# --- v20 fixes: mayor guards (tools.py: ToolBox._reserve_refusal, _loan_refusal; fleet/scripted.py's confirm tuple)
FIRE_2 = FB(2500, (N("fire station", 4, 1, "coverage 69%", upkeep=55),), 2500, ())
CLINIC_2 = FB(2000, (N("clinic", 3, 1, "coverage 68%", upkeep=40),), 2000, ())
FUND_ANYWAY = " To fund it anyway, repeat this exact call with confirm: true."
DEFICIT = " (6 x $536 expenses: last month's deficit was $22)"


def test_another_building_under_the_reserve_is_refused_until_confirmed(monkeypatch):
    """Smoke v20 seed 0, month 6: the to-fund line read "$4,305 covers planning, police; the other $6,500 would go
    under the reserve", and the mayor funded another fire station and another clinic anyway, leaving $303 of a $3,216
    reserve. A top-up for another building that would leave the treasury under the reserve is refused once; the exact
    repeat with confirm: true is an override."""
    env = _env()
    w = env.world
    env.end_month()
    w.history[-1].update(income=514, expenses=536)       # a $22 deficit: 6 x $536
    assert reserve(w) == 3216 and reserve_why(w) == DEFICIT
    _builds(monkeypatch, {"police": STATION, "fire": FIRE_2, "health": CLINIC_2, "education": SCHOOL}, road=ROAD_315)
    box = _mayor(env, treasury=6803)
    move = lambda dept, amount, **kw: box.call("allocate_budget", {"department": dept, "amount": amount, **kw})
    assert move("police", 2500)["ok"] and w.treasury == 4303          # another, and the reserve stays: it runs
    why = ("this top-up for another fire station leaves the treasury at $1,803, $1,413 under the $3,216 reserve"
           + DEFICIT + "; first/supply and growth come first: education $2,500 (first/supply), planning $315 (growth) "
           "still wait." + FUND_ANYWAY)
    assert move("fire", 2500) == {"ok": False, "refused": "reserve", "error": why}
    assert w.treasury == 4303 and len(box.executed) == 1 and box._failed == {}   # nothing ran or is remembered
    # first/supply and growth may go under the reserve
    assert move("education", 2500)["ok"] and move("planning", 315)["ok"] and round(w.treasury) == 1488
    why = ("this top-up for another clinic leaves the treasury at $488, $2,728 under the $3,216 reserve" + DEFICIT
           + "; the reserve keeps the city solvent and pays for first buildings, supply and planning's road when they "
           "come (none waits now)." + FUND_ANYWAY)
    first = move("health", 1000, confirm=True)       # confirm on a call not refused before: refused, with the note
    assert first == {"ok": False, "refused": "reserve", "error": why[:-1] + EARNED + "."}, first
    res = move("health", 1000, confirm=True)         # the identical repeat: earned
    assert res["ok"] and res["override"] == "reserve" and w.dept_funds["health"] == 1000, res
    assert res["message"].startswith("transferred $1,000 to health (now $1,000) | health: still $1,000 short of "
                                     "another clinic")
    assert move("health", 400)["refused"] == "reserve"          # still another, still under: another amount asks again
    # not refused: more than the treasury holds (the action says so), a blocked first building
    res = move("fire", 2500)
    assert res["ok"] is False and "refused" not in res and res["message"].startswith("treasury only has $488"), res
    _builds(monkeypatch, {"fire": NO_LOT, "health": CLINIC_2})
    assert DeptMoney(env, "fire").kind == "blocked" and move("fire", 300)["ok"]
    assert [r["ok"] for r in box.executed] == [True, True, True, True, False, True]


def test_the_reserve_refusal_names_idle_money_and_spares_dust(monkeypatch):
    """The refusal names idle money to take back first (the playbook's first step); a top-up that leaves the treasury
    at the reserve, or less than $1 under it, runs."""
    env = _env()
    w = env.world
    _months_spent(env, {}, {}, {})
    assert reserve(w) == RESERVE_MIN
    _builds(monkeypatch, {"police": STATION, "fire": FIRE_2})
    box = _mayor(env, treasury=5500, planning=5000)          # planning keeps $1,000 of $5,000: $4,000 idle
    assert DeptMoney(env, "planning").idle == 4000
    assert box.call("allocate_budget", {"department": "police", "amount": 2500})["ok"] and w.treasury == 3000
    res = box.call("allocate_budget", {"department": "fire", "amount": 2500})
    assert res["error"] == ("this top-up for another fire station leaves the treasury at $500, $2,500 under the "
                            "$3,000 reserve; the reserve keeps the city solvent and pays for first buildings, supply "
                            "and planning's road when they come (none waits now); take back idle money first "
                            "(planning $4,000)." + FUND_ANYWAY), res
    res = _mayor(env, treasury=5999.6).call("allocate_budget", {"department": "fire", "amount": 3000})
    assert res["ok"] and "override" not in res, res             # $0.40 under: rounds to the reserve


def test_levee_money_before_a_flood_is_not_another_building(monkeypatch):
    """Fleet_scripted acting monthly, seeds 3 and 4 (months 43, 65, 66): the playbook's flood rule ("give fire ~$3,000
    for levees at once"), which the scripted mayor follows, met the reserve guard, since fire's needs line named its
    cheaper next fire station (and at seed 4 month 43 no levee site was listed: the forecast rows had nothing to
    protect by a road). Levees are first/supply: while the forecast the briefing shows names a flood, a fire top-up
    runs; a heatwave doesn't count."""
    env = _env()
    w = env.world
    _builds(monkeypatch, {"fire": FIRE_2, "police": STATION})
    forecast = []
    monkeypatch.setattr(w, "forecast", lambda ahead=3: list(forecast))
    fire = {"department": "fire", "amount": 3000}
    assert _mayor(env, treasury=3500).call("allocate_budget", fire)["refused"] == "reserve"
    forecast.append("Heatwave expected in 1 month(s): +20% power, +30% water demand, more fires")
    assert _mayor(env, treasury=3500).call("allocate_budget", fire)["refused"] == "reserve"
    forecast.append("River flood expected in 2 month(s) along rows 8-22: riverside buildings without a levee will be "
                    "damaged")
    box = _mayor(env, treasury=3500)
    res = box.call("allocate_budget", fire)
    assert res["ok"] and "override" not in res and w.treasury == 500, res
    assert box.call("allocate_budget", {"department": "police", "amount": 400})["refused"] == "reserve"  # fire only


def test_a_loan_while_only_extra_buildings_wait_is_refused_until_confirmed(monkeypatch):
    """Smoke v20 seed 0, month 9: in a hold that started at $550, the mayor took loans of $5,000, $3,000 and $2,500
    (confirm: true on each first call, which passed since no guard fired: the turn started under $3,000) to fund
    another police, fire and health building. A loan is refused once when no first building, supply or planning road
    waits (Funding.to_fund), whatever the treasury at the turn's start."""
    env = _env()
    w = env.world
    _builds(monkeypatch, {"police": STATION, "fire": FIRE_2})
    box = _mayor(env, treasury=550)
    only = ("no first building, supply or planning road is waiting (only another: police $2,500, fire $2,500); this "
            "$3,000 loan would pay for another building at $15/month interest until repaid (debt would be $3,000 of "
            "$50k): fund another building only from the money above the reserve." + BORROW)
    first = box.call("take_loan", {"amount": 3000, "confirm": True})      # v20: confirm on the first call
    assert first == {"ok": False, "refused": "loan", "error": only[:-1] + EARNED + "."}, first
    res = box.call("take_loan", {"amount": 3000, "confirm": True})
    assert res["ok"] and res["override"] == "loan" and w.debt == 3000 and w.treasury == 3550, res
    # nothing to fund at all: the loan would sit in the treasury
    _builds(monkeypatch, {})
    box = _mayor(env, treasury=550)
    assert box.call("take_loan", {"amount": 2000})["error"] == (
        "no first building, supply or planning road is waiting (nothing is ready to fund); this $2,000 loan would sit "
        "in the treasury at $10/month interest until repaid (debt would be $5,000 of $50k)." + BORROW)
    # below $0 (the WARNING line's months in the red) a loan pays the bills: it runs
    res = _mayor(env, treasury=-500).call("take_loan", {"amount": 2000})
    assert res["ok"] and "override" not in res and w.debt == 5000, res
    # a first building, supply or planning road waiting: the loan runs
    outlet = FB(1200, (N("sewage outlet", 2, 1, "sewage 95% used", upkeep=15),), 1200, ())
    for mix, road in (({"police": STATION, "education": SCHOOL}, None), ({"police": STATION}, ROAD_315),
                      ({"utilities": outlet}, None)):
        _builds(monkeypatch, mix, road=road)
        w.debt = 0
        res = _mayor(env, treasury=550).call("take_loan", {"amount": 1000})
        assert res["ok"] and "override" not in res, (mix, res)


def test_the_loan_refusals_stack(monkeypatch):
    """Only extra buildings waiting stacks with the other loan refusals (the treasury at $3,000 or more, or this turn's
    own moves took it under) into one refusal text, which one override passes."""
    env = _env()
    w = env.world
    _builds(monkeypatch, {"police": STATION, "fire": FIRE_2})
    box = _mayor(env, treasury=3598)
    assert box.call("take_loan", {"amount": 5000})["error"] == (
        "the treasury holds $3,598, over the $3,000 minimum: a $5,000 loan costs $25/month until repaid (debt would "
        "be $5,000 of $50k); borrow only for a first building or supply need the treasury can't cover. Also, no first "
        "building, supply or planning road is waiting (only another: police $2,500, fire $2,500): the loan would pay "
        "for another building." + BORROW)
    box = _mayor(env, treasury=5698)
    move = lambda dept, amount, **kw: box.call("allocate_budget", {"department": dept, "amount": amount, **kw})
    assert move("police", 2500)["ok"] and move("fire", 2500)["refused"] == "reserve"
    assert move("fire", 2500, confirm=True)["override"] == "reserve" and round(w.treasury) == 698
    assert box.call("take_loan", {"amount": 3000})["error"] == (
        "the treasury started this turn at $5,698; this turn's top-ups ($5,000 net) took it under $3,000: take back "
        "(free) a top-up that isn't a first building or supply (police $2,500, fire $2,500) instead of a $3,000 loan "
        "at $15/month. Also, no first building, supply or planning road is waiting (nothing is ready to fund): the "
        "loan would sit in the treasury." + BORROW)
    res = box.call("take_loan", {"amount": 3000, "confirm": True})
    assert res["ok"] and res["override"] == "loan" and w.debt == 3000


def test_the_new_guards_never_block_an_action_by_failing(monkeypatch):
    """Like the money notes, the guards read funding_gaps; if that fails the call runs as if they said nothing."""
    env = _env()
    w = env.world
    _builds(monkeypatch, {"police": STATION})

    def boom(*a, **k):
        raise RuntimeError("no")
    monkeypatch.setattr(T, "funding_gaps", boom)
    box = _mayor(env, treasury=2000)
    assert box.call("allocate_budget", {"department": "police", "amount": 1500})["ok"]
    assert box.call("take_loan", {"amount": 1000})["ok"] and w.debt == 1000


def test_the_report_counts_the_new_refusals():
    """fleet_report counts refusals and overrides whatever their kind: the reserve refusal and its override too."""
    from civitas.fleet.agent import _slim
    from civitas.fleet.report import _tool_error, turn_row
    refused = {"ok": False, "refused": "reserve", "error": "that leaves the treasury at $303, ..."}
    over = {"ok": True, "message": "transferred $1,500 to health", "override": "reserve"}
    trace = [{"tool": "allocate_budget", "result": _slim(r)} for r in (refused, over)]
    trace.append({"tool": "take_loan", "result": {"ok": False, "refused": "loan", "error": "no first building ..."}})
    row = turn_row({"role": "mayor", "month": 6, "tool_trace": trace})
    assert (row["refused"], row["overrides"], row["tool_errors"]) == (2, 1, [])
    assert _tool_error(trace[0]) is None and _slim(over)["override"] == "reserve"


def test_the_scripted_mayor_confirms_the_new_refusals(monkeypatch):
    """The scripted fleet's mayor repeats a refusal its policy means with confirm: true, the new kinds included; after
    a loan this turn a reserve refusal stays (the scripted mayor borrows only at a turn's start under $3,000 and tops
    up only above its own reserve, never on seeds 0-4)."""
    from civitas.core.decision import Decision
    from civitas.fleet.scripted import ScriptedFleetAgent
    env = _env()
    w = env.world
    _builds(monkeypatch, {"police": STATION, "fire": FIRE_2})
    a = ScriptedFleetAgent("mayor", cfg=FleetAgentConfig(model="scripted"))
    a.reset(0)
    a.bind(env)
    w.treasury = 2000
    acts = [{"type": "take_loan", "amount": 10000},
            {"type": "allocate_budget", "department": "police", "amount": 11000}]
    monkeypatch.setattr(a.policy, "act", lambda obs: Decision(actions=[dict(x) for x in acts], thinking="spend"))
    d = a.act({})
    tr = [(c["tool"], c["result"].get("refused"), c["result"].get("override")) for c in d.meta["tool_trace"]
          if c["tool"] in ("take_loan", "allocate_budget")]
    assert tr == [("take_loan", "loan", None), ("take_loan", None, "loan"), ("allocate_budget", "reserve", None),
                  ("allocate_budget", "reserve", None)]
    assert w.debt == 10000 and w.dept_funds["police"] == 0 and [r["ok"] for r in d.executed] == [True]


# --- v20 fixes: integration (briefing.py: _short_line's wording with one unpaid building)
def test_the_short_now_line_says_nothing_is_paid_for_one_building():
    """The merged scripted fleet at month 24 read "your $1,748 pays for none of them; a coal plant at (0,23) ($4,000)
    needs $2,252 more": one building named, so "none of them" had nothing plural to point at."""
    from civitas.fleet.briefing import _short_line
    w = _env().world
    coal = T.site_price(w, "utilities", "coal_or_gas")
    b = T.SupplyBuild("power", 229, 229, 60, "coal_or_gas", coal, 250, {"x": 0, "y": 23}, 6, imported=9.7)
    assert _short_line([b], 1748) == ["- supply needed now: power 100% used (supply 229, demand 229, 9.70 imported): "
                                      "your $1,748 pays for nothing yet; a coal plant at (0,23) ($4,000) needs $2,252 "
                                      "more"]
    assert _short_line([b], 4000) == ["- supply needed now: power 100% used (supply 229, demand 229, 9.70 imported): "
                                      "your $4,000 pays for a coal plant at (0,23) ($4,000)"]


# --- v20 fixes: review (tools.py: pick_supply's one-tile first buildings, the reserve guard's kinds, the loan guard's
# excess, bills and flood; briefing.py: the supply-needed line's wording, the can't-pay line, the mayor's blocked supply)
def _big_city(env, monkeypatch, **metrics) -> None:
    """Past 10,000 residents (the water works, advanced treatment and nuclear plant unlocked), water at capacity with
    imports filling the gap (review: scripted seed 0 month 90 with pumps removed, 1,235 of 1,235, 85.10 imported),
    power and sewage with room, or as `metrics` say."""
    w = env.world
    w.milestones_reached += [m["name"] for m in C.MILESTONES if m["pop"] <= 10000]
    real = w.metrics()
    big = {**real, "population": 11788, "water_supply": 1235.1, "water_demand": 1235.1, "power_supply": 2000.0,
           "power_demand": 1000.0, "sewage_supply": 2000.0, "sewage_demand": 1000.0,
           "service_imports": {**real["service_imports"], "water": 85.1, "power": 0.0}, **metrics}
    monkeypatch.setattr(w, "metrics", lambda: big)


def test_a_big_city_short_of_water_names_a_pump_not_a_water_works(monkeypatch):
    """Review, scripted seed 0 month 90 (11,788 residents, water 1,235 of 1,235): water wanted 703 more, which only the
    $12,000 water works (2,000) covered, so utilities' lines and the mayor's named it over six ready $1,500 pumps
    ("needs $12k for another water works", first on the to-fund line; main named a pump). A one-tile first building
    is named while one is ready (the one with the most output when none covers the need); a 2x2 one only when none
    is, and then its whole footprint is taken."""
    env = _env()
    w = env.world
    _big_city(env, monkeypatch)
    assert T.site_price(w, "utilities", "riverbank_2x2").name == "water_works"
    bank = lambda *xy: SiteList([{"x": x, "y": y, "pollution": 0.0} for x, y in xy], None, None)
    lists = {"pump": bank((25, 23), (29, 23)), "near_homes": SiteList([{"x": 1, "y": 23}], None, None),
             "riverbank_2x2": SiteList([{"x": 25, "y": 22}], None, None), "sewage": bank((26, 23), (30, 25))}
    water = next(b for b in T.short_supplies(w, lists) if b.supply == "water")
    assert (round(water.need), water.purpose, water.price.cost, water.output) == (703, "pump", 1500, 250)
    coal, works = T.Price(4000, "coal plant", name="coal"), T.Price(12000, "water works", first=False, size=(2, 2))
    nuclear = T.Price(40000, "nuclear plant", size=(2, 2))
    opts = [("coal_or_gas", coal, 250.0, {}), ("riverbank_2x2", nuclear, 2000.0, {})]
    assert T.pick_supply(700, opts)[0] == "coal_or_gas" and T.pick_supply(700, opts[1:])[0] == "riverbank_2x2"
    assert T.pick_supply(700, [("riverbank_2x2", works, 2000.0, {})])[1] is works
    # only the 2x2 lot ready for water: the works is named, and the outlet goes on past its footprint
    lists = {"riverbank_2x2": lists["riverbank_2x2"], "sewage": lists["sewage"]}
    _big_city(env, monkeypatch, sewage_supply=1000.0)
    builds = T.short_supplies(w, lists)
    assert [(b.supply, b.purpose, (b.site["x"], b.site["y"])) for b in builds] == [
        ("water", "riverbank_2x2", (25, 22)), ("sewage", "sewage", (30, 25))]
    assert T.footprint_tiles({"x": 25, "y": 22}, (2, 2)) == {(25, 22), (26, 22), (25, 23), (26, 23)}


def test_the_supply_needed_line_says_used_or_covered_as_the_mayor_reads_it():
    """Review, fleet_scripted seed 0 month 12: "- short now: power (111 of 103)" called a supply with room short, under
    Key metrics' "power: supply 111 / demand 103, coverage 100%", while the mayor read "another wind turbine (power 93%
    used)". Each supply now reads as the mayor's line says it (SupplyBuild.status), with the numbers."""
    from civitas.fleet.briefing import _short_line
    w = _env().world
    wind = T.site_price(w, "utilities", "wind")
    b = T.SupplyBuild("power", 111, 103, 44.5, "wind", wind, 58.8, {"x": 20, "y": 25}, 6)
    assert b.status == "power 93% used"
    assert _short_line([b], 1559) == ["- supply needed now: power 93% used (supply 111, demand 103): your $1,559 pays "
                                      "for nothing yet; a wind turbine at (20,25) ($2,500) needs $941 more"]
    none = T.SupplyBuild("sewage", 0, 0, 50)
    assert _short_line([none], 0) == ["- supply needed now: no sewage demand yet (supply 0, demand 0; no site yet)"]


def test_utilities_can_t_pay_line_names_what_its_supply_needs():
    """Review, fleet_scripted seed 0 month 24: the supply line named a $4,000 coal plant needing $2,252 more, then the
    can't-pay line read "cheapest: near_homes $2,200" and "ask the mayor for the amount", with water at 141 of 250 (a
    tower would buy nothing needed). It names the supply line's amount now (test_utilities_briefing_says_what_is_short_
    and_what_its_funds_build), and with every supply roomy there is no line: utilities needs no money."""
    env = _env()
    w = env.world
    w.dept_funds["utilities"] = 50000
    _to_the_river(env)
    for a in ({"type": "build_power_plant", "kind": "coal", "x": 0, "y": 23},
              {"type": "build_water", "kind": "pump", "x": 25, "y": 23},
              {"type": "build_sewage", "kind": "outlet", "x": 30, "y": 25}):
        assert apply_action(w, "utilities", a).ok, a
    w.dept_funds["utilities"] = 100
    text = briefing(env, "utilities", mark=False)
    assert "- supply needed now" not in text and "You can't pay" not in text, _block(text, "## Your budget")
    assert "- you have $100" in text                                    # the Ready sites lines still price them


def test_a_top_up_that_buys_nothing_now_is_refused_under_the_reserve(monkeypatch):
    """Review probe, treasury $3,500: $3,000 to transport with nothing ready ran and left $500 of a $3,000 reserve,
    then $400 for another police station was refused. Money a department has no use for now (nothing ready, or idle
    already) meets the reserve guard like another building; with nothing waiting the refusal says what the reserve is
    for."""
    env = _env()
    w = env.world
    _months_spent(env, {}, {}, {})
    _builds(monkeypatch, {"police": STATION})
    box = _mayor(env, treasury=3500, planning=5000)
    assert DeptMoney(env, "transport").kind == "nothing" and DeptMoney(env, "planning").kind == "idle"
    assert box.call("allocate_budget", {"department": "transport", "amount": 3000})["error"] == (
        "this top-up buys nothing now (transport: nothing ready to buy) and leaves the treasury at $500, $2,500 under "
        "the $3,000 reserve; the reserve keeps the city solvent and pays for first buildings, supply and planning's "
        "road when they come (none waits now); take back idle money first (planning $4,000)." + FUND_ANYWAY)
    assert box.call("allocate_budget", {"department": "planning", "amount": 1000})["error"] == (
        "this top-up buys nothing now (planning holds $4,000 idle) and leaves the treasury at $2,500, $500 under the "
        "$3,000 reserve; the reserve keeps the city solvent and pays for first buildings, supply and planning's road "
        "when they come (none waits now)." + FUND_ANYWAY)                  # its own idle money isn't named twice
    res = box.call("allocate_budget", {"department": "transport", "amount": 3000, "confirm": True})
    assert res["ok"] and res["override"] == "reserve" and w.treasury == 500
    assert _mayor(env, treasury=9000).call("allocate_budget", {"department": "transport", "amount": 3000})["ok"]


def test_a_loan_is_refused_only_beyond_what_first_buildings_growth_and_the_reserve_need(monkeypatch):
    """Review replay of smoke v20 seed 0, month 9 (treasury $550): with utilities' $900 and planning's $500 waiting, a
    $5,000 loan ran, though $3,850 funded those and brought the treasury back to the reserve. A loan while first/supply
    or growth waits is refused once when it is LOAN_SLACK or more over that; rounding up to the next $1,000 runs."""
    assert T.LOAN_SLACK == 1000
    env = _env()
    w = env.world
    tower = FB(900, (N("water tower", 3, 1, "water 95% used", upkeep=30),), 900, ())
    _builds(monkeypatch, {"utilities": tower, "police": STATION}, road=FB(500, (ROAD,), 500, ()))
    box = _mayor(env, treasury=550)
    assert box.call("take_loan", {"amount": 5000})["error"] == (
        "only $3,850 of this $5,000 loan is needed ($1,400 for first/supply and growth, $2,450 to keep the treasury at "
        "the $3,000 reserve): the other $1,150 would pay for another building at $6/month interest until repaid (debt "
        "would be $5,000 of $50k): borrow $3,850 or less." + BORROW)
    res = box.call("take_loan", {"amount": 4000})
    assert res["ok"] and "override" not in res and w.debt == 4000, res
    # nothing but first/supply waits: the rest would sit in the treasury; stacked after the first part as one text
    _builds(monkeypatch, {"utilities": tower})
    w.debt = 0
    # $900 waiting, more than the $100 above $3,000: only the part over the need is refused
    assert _mayor(env, treasury=3100).call("take_loan", {"amount": 5000})["error"] == (
        "only $900 of this $5,000 loan is needed ($900 for first/supply and growth): the other $4,100 would sit in "
        "the treasury at $20/month interest until repaid (debt would be $5,000 of $50k): borrow $900 or less." + BORROW)
    assert _mayor(env, treasury=4100).call("take_loan", {"amount": 5000})["error"] == (   # the treasury covers it
        "the treasury holds $4,100, over the $3,000 minimum: a $5,000 loan costs $25/month until repaid (debt would "
        "be $5,000 of $50k); borrow only for a first building or supply need the treasury can't cover. Also, only $900 "
        "of this $5,000 loan is needed ($900 for first/supply and growth): the other $4,100 would sit in the "
        "treasury." + BORROW)


def test_a_loan_for_levees_before_a_flood_is_not_for_another_building(monkeypatch):
    """Review probe, treasury $1,000 and a flood forecast: take_loan 3000 for fire's levees (the playbook's flood rule;
    allocate_budget fails over the treasury) was refused as "would pay for another building". Fire's item counts as
    first/supply while flood_ahead, as the reserve guard already had it."""
    env = _env()
    w = env.world
    _builds(monkeypatch, {"police": STATION, "fire": FIRE_2})
    forecast = []
    monkeypatch.setattr(w, "forecast", lambda ahead=3: list(forecast))
    assert _mayor(env, treasury=1000).call("take_loan", {"amount": 3000})["refused"] == "loan"
    forecast.append("River flood expected in 2 month(s) along rows 8-22: riverside buildings without a levee will be "
                    "damaged")
    assert T.flood_ahead(w)
    res = _mayor(env, treasury=1000).call("take_loan", {"amount": 3000})
    assert res["ok"] and "override" not in res and w.debt == 3000, res


def test_a_loan_that_pays_the_bills_before_the_next_turn_runs(monkeypatch):
    """Review probe: operations at -$800/month, treasury $300, only another police station waiting: the loan was
    refused as "would pay for another building", though the treasury goes below $0 before the mayor's next turn.
    Operations over act_every months (bills_ahead) count as a need: the loan runs up to them and the reserve."""
    env = _env(act_every=3)
    w = env.world
    env.end_month()
    w.history[-1].update(income=100, expenses=900)
    assert cash_flow(w).ops_per_month == -800 and T.bills_ahead(w, 3) == 2400 and reserve(w) == 5400
    _builds(monkeypatch, {"police": STATION})
    assert _mayor(env, treasury=3000).call("take_loan", {"amount": 2000})["refused"] == "loan"   # $3,000 covers them
    res = _mayor(env, treasury=300).call("take_loan", {"amount": 7500})
    assert res["ok"] and "override" not in res and w.debt == 7500, res
    w.debt = 0
    assert _mayor(env, treasury=300).call("take_loan", {"amount": 9000})["error"] == (
        "only $7,500 of this $9,000 loan is needed ($7,500 to keep the treasury at the $5,400 reserve through "
        "operations of -$2,400 to your next turn): the other $1,500 would pay for another building at $8/month "
        "interest until repaid (debt would be $9,000 of $50k): borrow $7,500 or less." + BORROW)
    w.history[-1].update(income=900, expenses=100)                     # a surplus: no bills
    assert T.bills_ahead(w, 3) == 0


# --- v22 fixes: the loan guard's treasury part and the reserve guard after a loan (tools.py: ToolBox._own_loan_refusal,
# _reserve_refusal, _act)
HIGH_SCHOOL = FB(5000, (N("high school", 3, 0, first=True, upkeep=75),), 5000, ())
CLINIC_2B = FB(2000, (N("landfill", 3, 1, "garbage 70% full", upkeep=30),), 2000, ())


def test_a_loan_for_a_first_building_the_treasury_can_t_cover_runs(monkeypatch):
    """Smoke v22 seed 1, month 9: treasury $3,607 and education's $5,000 first high school waiting; take_loan 4500 was
    refused as "the treasury holds $3,607, over the $3,000 minimum ... borrow only for a first building or supply need
    the treasury can't cover", the very case, and the mayor overrode it. Over $3,000 the treasury part stays silent
    while the first/supply and growth waiting come to more than what they may use (Funding.dip)."""
    env = _env()
    w = env.world
    _months_spent(env, {}, {}, {})
    _builds(monkeypatch, {"education": HIGH_SCHOOL, "police": STATION})
    res = _mayor(env, treasury=3607).call("take_loan", {"amount": 4500})
    assert res["ok"] and "override" not in res and w.debt == 4500, res
    # what the treasury covers: still refused (the $4,000 above $3,000 pays the $2,500 school)
    _builds(monkeypatch, {"education": SCHOOL, "police": STATION})
    w.debt = 0
    assert _mayor(env, treasury=7000).call("take_loan", {"amount": 2000})["error"].startswith(
        "the treasury holds $7,000, over the $3,000 minimum")


def test_after_a_loan_a_top_up_under_the_reserve_can_t_be_confirmed(monkeypatch):
    """Smoke v22 seed 1, month 9: after borrowing $4,500 for a first high school the mayor funded another landfill with
    confirm: true, leaving $1,107 of a $3,000 reserve. After a loan this turn the reserve refusal says so and no
    confirm overrides it; without a loan the earned confirm still works."""
    env = _env()
    w = env.world
    _months_spent(env, {}, {}, {})
    _builds(monkeypatch, {"education": HIGH_SCHOOL, "health": CLINIC_2B})
    box = _mayor(env, treasury=3607)
    assert box.call("take_loan", {"amount": 4500})["ok"] and box.borrowed == 4500
    assert box.call("allocate_budget", {"department": "education", "amount": 5000})["ok"] and round(w.treasury) == 3107
    health = {"department": "health", "amount": 2000}
    why = ("this top-up for another landfill leaves the treasury at $1,107, $1,893 under the $3,000 reserve; the "
           "reserve keeps the city solvent and pays for first buildings, supply and planning's road when they come "
           "(none waits now). You borrowed $4,500 this turn, and a loan pays only for first buildings, supply and "
           "planning's road: fund this from money above the reserve next turn (confirm: true doesn't count after a "
           "loan).")
    assert box.call("allocate_budget", health) == {"ok": False, "refused": "reserve", "error": why}
    assert box.call("allocate_budget", {**health, "confirm": True}) == {"ok": False, "refused": "reserve", "error": why}
    assert w.dept_funds["health"] == 0
    box = _mayor(env, treasury=3107)                                   # a new turn, no loan: earned confirm
    assert box.call("allocate_budget", health)["refused"] == "reserve"
    assert box.call("allocate_budget", {**health, "confirm": True})["override"] == "reserve"


# --- Unit D: requests to the mayor (tools.py: open_requests, request_state, ToolBox.end_turn's reminder; briefing.py:
# the Open requests block; bench/collab.py: answer, shared with the collaboration score)
import json  # noqa: E402

from civitas.bench.collab import collab  # noqa: E402
from civitas.core.decision import Decision  # noqa: E402
from civitas.fleet.agent import ToolAgent  # noqa: E402

ASKS = [("police", "Requesting $2,500 for our first police station"),
        ("health", "Mayor, I need $2,000 for a landfill"),
        ("transport", "Transport reporting: nothing to build yet, please allocate budget"),
        ("fire", "Fire station built, all good"),                     # no ask: not a request
        ("education", "need a school")]                                # to all: a broadcast


def _asked(env):
    """act_every 3: the mayor's month-0 turn (marked), then the departments' messages, then months to month 3."""
    briefing(env, "mayor")
    for d, text in ASKS:
        env.council.post(0, d, text, "all" if d == "education" else "mayor")
    for _ in range(3):
        env.end_month()


def test_open_requests_until_a_top_up_or_a_reply():
    """v23 seed 0: the mayor answered 59% of the requests sent to it. A request stays open until a top-up to the
    department (any share of what it asked: met or partial) or a message back to it, inside 2 x act_every months."""
    env = _env(act_every=3)
    _asked(env)
    fleet = FleetState.of(env)
    reqs = T.open_requests(env, fleet)
    assert [(r.month, r.dept, r.amount, r.last) for r in reqs] == [(0, "transport", None, False),
                                                                  (0, "police", 2500, False),
                                                                  (0, "health", 2000, False)]
    box = _mayor(env, treasury=30000)
    assert box.call("allocate_budget", {"department": "police", "amount": 2500})["ok"]
    assert box.call("send_message", {"to": "health", "text": "Not this quarter: the reserve comes first"})["ok"]
    assert box.call("send_message", {"to": "all", "text": "Budget review next quarter"})["ok"]   # no reply to anyone
    assert [r.dept for r in T.open_requests(env, fleet, box.messages)] == ["transport"]
    env2 = _env(act_every=3)
    _asked(env2)
    assert _mayor(env2).call("allocate_budget", {"department": "health", "amount": 500})["ok"]     # partial
    assert "health" not in [r.dept for r in T.open_requests(env2, FleetState.of(env2))]
    # month 6 is the last turn to answer a month-0 request; at month 7 it no longer counts
    for _ in range(3):
        env.end_month()
    assert [(r.dept, r.last) for r in T.open_requests(env, fleet)] == [   # the top-up is in the log; the month-3
        ("transport", True), ("health", True)]                             # reply was never posted
    env.end_month()
    assert T.open_requests(env, fleet) == []


def test_open_requests_agree_with_the_collaboration_score():
    """The same requests and the mayor's month-3 answers as frames: collab's threads say what open_requests says
    (a money ask with only a reply back is "replied", no longer unanswered)."""
    env = _env(act_every=3)
    _asked(env)
    box = _mayor(env, treasury=30000)
    box.call("allocate_budget", {"department": "police", "amount": 2500})
    box.call("send_message", {"to": "health", "text": "Not this quarter: the reserve comes first"})
    still = {r.dept for r in T.open_requests(env, FleetState.of(env), box.messages)}
    msg = lambda m, s, to, text: {"month": m, "sender": s, "to": to, "text": text}
    frames = [{"month": 1, "council": [msg(0, d, "all" if d == "education" else "mayor", t) for d, t in ASKS],
               "turns": []},
              {"month": 4, "council": [msg(3, "mayor", "health", "Not this quarter: the reserve comes first")],
               "turns": [{"month": 3, "role": "mayor", "results": list(box.executed)}]},
              {"month": 7, "council": [], "turns": []}]
    th = {t["from"]: t["status"] for t in collab(frames, list(env.roles), act_every=3)["threads"]}
    assert th == {"police": "met", "health": "replied", "transport": "unanswered"}
    assert still == {d for d, s in th.items() if s == "unanswered"}


def test_the_mayor_s_briefing_lists_open_requests(monkeypatch):
    _builds(monkeypatch, {"police": FB(2500, (N("police station", 5, 0, first=True, upkeep=55),), 2500, ())})
    env = _env(act_every=3)
    _asked(env)
    text = briefing(env, "mayor", mark=False)
    assert "  - (3 requests to you: under Open requests to you below)\n" in text
    assert "Requesting $2,500" not in text.split("## Open requests")[0]          # quoted once, in the block
    assert "- m0 education -> all: need a school" in text and "Fire station built" in text   # the others stay
    lines = _block(text, "## Open requests to you")
    assert lines[0].startswith("- transport: m0 (waiting 3 months); no top-up needed (")
    assert lines[1].startswith("- police: m0 (waiting 3 months), asks $2,500; to fund $2,500 (first/supply), ")
    assert lines[1].endswith(': "Requesting $2,500 for our first police station"')
    assert lines[2].startswith("- health: m0 (waiting 3 months), asks $2,000; ") and len(lines) == 3
    assert text.index("- to fund: ") < text.index("## Open requests to you") < text.index("## City plan")
    # repeats from one department merge; past REQUESTS_SHOWN departments the rest are named
    env.council.post(3, "health", "Again: need $2,000 for the landfill please", "mayor")
    for d in ("utilities", "planning"):
        env.council.post(3, d, f"{d} needs $900, please", "mayor")
    env.end_month()
    lines = _mayor_block(env, "## Open requests to you")
    assert lines[2].startswith("- health: m0, m3 (waiting 4 months), asks $2,000; ")
    assert lines[2].endswith(': "Again: need $2,000 for the landfill please"')
    assert len(lines) == T.REQUESTS_SHOWN + 1 and lines[-1] == "- +1 more: utilities"   # planning first in turn order
    # every one answered: no block
    box = _mayor(env, treasury=30000)
    for d in ("transport", "police", "health", "utilities", "planning"):
        box.call("send_message", {"to": d, "text": "Noted: not this month, the reserve first"})
    env.apply("mayor", Decision(executed=box.executed, messages=box.messages))
    assert "## Open requests to you" not in briefing(env, "mayor", mark=False)


def test_request_state_says_what_the_mayor_can_answer(monkeypatch):
    env = _env()
    _months_spent(env, {}, {}, {})
    _builds(monkeypatch, {"police": FB(2500, (N("police station", 5, 1, "coverage 66%", upkeep=55),), 2500, ()),
                          "education": FB(2500, (N("elementary school", 5, 0, first=True, upkeep=45),), 2500, ())})
    env.world.treasury = 1000.0                                           # a hold, nothing above $3,000
    f = funding_gaps(env, FleetState.of(env))
    cov = T.fund_cover(f)
    assert T.request_state(f, "police", cov) == "to fund $2,500 (another), not covered (hold)"
    assert T.request_state(f, "fire", cov) == "no top-up needed (nothing ready to buy)"
    env.world.treasury = 30000.0
    f = funding_gaps(env, FleetState.of(env))
    assert T.request_state(f, "education", T.fund_cover(f)) == "to fund $2,500 (first/supply), covered"
    env.world.dept_funds["fire"] = 9000
    f = funding_gaps(env, FleetState.of(env))
    assert T.request_state(f, "fire", T.fund_cover(f)).startswith("no top-up needed (holds $")


def test_end_turn_reminds_the_model_s_mayor_once():
    env = _env(act_every=3)
    _asked(env)
    assert T.ToolBox(env, "mayor").call("end_turn", {"report": "done"})["ok"]          # scripted fleet: no reminder
    box = T.ToolBox(env, "mayor", max_steps=12)
    res = box.call("end_turn", {"report": "done"})
    assert res["ok"] is False and res["refused"] == "open_requests" and not box.ended
    assert res["error"].startswith("not ended: 3 requests to you still open: transport (since m0), police (asks "
                                   "$2,500, since m0), health (asks $2,000, since m0). Answer each department now")
    assert box.call("end_turn", {"report": "done"})["ok"] and box.ended              # once a turn
    box = T.ToolBox(env, "mayor", max_steps=12)                       # all answered: it ends at once
    box.call("allocate_budget", {"department": "police", "amount": 2500})
    for d in ("transport", "health"):
        box.call("send_message", {"to": d, "text": "Not this quarter"})
    assert box.call("end_turn", {"report": "done"})["ok"] and box.ended
    box = T.ToolBox(env, "mayor", max_steps=2)                        # no call left for a reply: no reminder
    box.call("city_overview")
    assert box.call("end_turn", {"report": "done"})["ok"]
    assert T.ToolBox(env, "police", max_steps=12).call("end_turn", {"report": "done"})["ok"]   # the mayor only


def test_the_tool_agent_s_mayor_gets_the_reminder():
    class Fake:
        n = 0

        def chat(self, model, messages, **kw):
            self.n += 1
            seq = [("end_turn", {"report": "done"}), ("send_message", {"to": "police", "text": "Not yet: reserve"}),
                   ("end_turn", {"report": "done"})]
            name, args = seq[min(self.n, len(seq)) - 1]
            return {"content": "", "tool_calls": [{"id": f"c{self.n}", "type": "function",
                                                   "function": {"name": name, "arguments": json.dumps(args)}}],
                    "finish_reason": "tool_calls", "usage": {}, "latency_s": 0.0}

    env = _env(act_every=3)
    briefing(env, "mayor")
    env.council.post(0, "police", "Requesting $2,500 for a station", "mayor")
    env.end_month()
    a = ToolAgent("mayor", cfg=FleetAgentConfig(model="fake"))
    a.client = Fake()
    a.bind(env)
    d = a.act(env.observe("mayor"))
    trace = d.meta["tool_trace"]
    assert [c["tool"] for c in trace] == ["end_turn", "send_message", "end_turn"] and d.meta["ended"]
    assert trace[0]["result"]["refused"] == "open_requests"
    assert d.messages == [{"to": "police", "text": "Not yet: reserve"}]


# --- Unit E: departments ask the mayor only for what it can fund now (tools.py: mayor_ask, ToolBox._ask_refusal;
# briefing.py: _ask_line), and one send_message reaches several roles (tools.py: recipients)
POLICE_2ND = FB(2500, (N("police station", 5, 1, "coverage 66%", upkeep=55),), 2500, ())
SCHOOL_1ST = FB(2500, (N("elementary school", 5, 0, first=True, upkeep=45),), 2500, ())


def test_mayor_ask_by_kind(monkeypatch):
    """Region smoke 20261005: 28 of 46 asks to the mayor were for another building it couldn't cover or with nothing
    ready. First/supply and growth always ask (the needs line's top-up); another only when the money covers it."""
    env = _env()
    _months_spent(env, {}, {}, {})
    _builds(monkeypatch, {"police": POLICE_2ND, "education": SCHOOL_1ST, "fire": FB(None, (), 2500, ("a fire station",))},
            road=FB(315, (ROAD,), 315, ()))
    fleet = FleetState.of(env)
    env.world.treasury, env.world.dept_funds["education"] = 30000.0, 1000
    assert T.mayor_ask(env, fleet, "education") == T.Ask(
        True, 1500, "your first elementary school", "the mayor funds first buildings, supply and planning's road first")
    assert T.mayor_ask(env, fleet, "planning")[:3] == (True, 315, "a road into open land and zoning it")
    assert T.mayor_ask(env, fleet, "police") == T.Ask(True, 2500, "another police station",
                                                      "the mayor's money above its reserve covers it now")
    env.world.treasury = 6000.0                       # $3,000 above the reserve: the school and road, then no station
    a = T.mayor_ask(env, fleet, "police")
    assert not a.ok and a.amount == 0 and a.why == (
        "another police station ($2,500 short) waits: the mayor's money above its reserve doesn't cover it after first "
        "buildings, supply and planning's road")
    env.world.treasury = 1000.0                       # a hold: the first school still asks
    assert T.mayor_ask(env, fleet, "police").why.endswith(": the mayor is in a hold (the treasury is at or under its "
                                                          "reserve) and funds only first buildings, supply and "
                                                          "planning's road")
    assert T.mayor_ask(env, fleet, "education").ok
    assert T.mayor_ask(env, fleet, "transport") == T.Ask(False, 0, "", "nothing ready to buy", "",
                                                         "a site is ready for you", "nothing is ready to build")
    assert T.mayor_ask(env, fleet, "fire").why == ("a fire station is blocked for lack of land: ask planning for a "
                                                   "lot, not the mayor for money")
    env.world.dept_funds["transport"] = 9000
    assert T.mayor_ask(env, fleet, "transport").why == "you hold $8,000 more than you need now"
    env.world.dept_funds["planning"] = 900
    assert T.mayor_ask(env, fleet, "planning").why == "your funds cover your road"


def test_department_briefings_say_whether_to_ask_the_mayor(monkeypatch):
    env = _env()
    _months_spent(env, {}, {}, {})
    _builds(monkeypatch, {"police": POLICE_2ND, "education": SCHOOL_1ST})
    env.world.treasury = 1000.0                                           # a hold
    budget = lambda d: _block(briefing(env, d, mark=False), "## Your budget")
    assert budget("education")[1] == ("- ask the mayor for $2,500 for your first elementary school (send_message, with "
                                      "a reason): the mayor funds first buildings, supply and planning's road first")
    assert budget("police")[1] == (
        "- no money ask to the mayor now: another police station ($2,500 short) waits: the mayor is in a hold (the "
        "treasury is at or under its reserve) and funds only first buildings, supply and planning's road. This turn: "
        "build only what your funds pay for; say what you need in your end_turn report")
    assert budget("transport")[1] == ("- no money ask to the mayor now: nothing ready to buy. Ask when a site is ready "
                                      "for you. This turn: nothing is ready to build; say what you need in your "
                                      "end_turn report")
    dept = system_prompt("police", FleetAgentConfig(model="x"), 12)
    assert "Ask the mayor for money only when your briefing's budget block says to (a first building, supply, " in dept
    assert "one send_message can answer several departments (to: \"police,fire\")" in \
        system_prompt("mayor", FleetAgentConfig(model="x"), 12)


def test_a_held_back_ask_still_reaches_the_other_recipients(monkeypatch):
    """Review: "mayor,planning" with an ask the mayor can't fund lost planning's copy too."""
    env = _env()
    _months_spent(env, {}, {}, {})
    _builds(monkeypatch, {"police": POLICE_2ND, "education": SCHOOL_1ST})
    env.world.treasury = 1000.0                                           # a hold
    box = T.ToolBox(env, "police", max_steps=12)
    res = box.call("send_message", {"to": "mayor,planning", "text": "Requesting $2,500 and a lot near (5,5)"})
    assert res["ok"] and "; not to the mayor: not sent: " in res["message"]
    assert box.messages == [{"to": "planning", "text": "Requesting $2,500 and a lot near (5,5)"}]


def test_send_message_holds_back_a_money_ask_the_mayor_cant_fund_once(monkeypatch):
    env = _env()
    _months_spent(env, {}, {}, {})
    _builds(monkeypatch, {"police": POLICE_2ND, "education": SCHOOL_1ST})
    env.world.treasury = 1000.0                                           # a hold
    ask = {"to": "mayor", "text": "Requesting $2,500 for another police station"}
    box = T.ToolBox(env, "police", max_steps=12)
    assert box.call("send_message", ask) == {"ok": False, "refused": "ask", "error": (
        "not sent: another police station ($2,500 short) waits: the mayor is in a hold (the treasury is at or under "
        "its reserve) and funds only first buildings, supply and planning's road, so a money ask now gets a no. "
        "Instead this turn: build only what your funds pay for; put what you need in your end_turn report (the mayor "
        "reads it). To post this message anyway, send it to the mayor again.")}
    assert box.messages == []
    assert box.call("send_message", ask)["ok"] and box.messages == [ask]           # once a turn: then it goes
    for to, text in (("planning", "need $500 for a lot"), ("mayor", "Station built, coverage 80%"),   # no money ask
                     ("mayor", "please free a lot near (3,4)")):
        assert T.ToolBox(env, "police", max_steps=12).call("send_message", {"to": to, "text": text})["ok"], text
    assert T.ToolBox(env, "education", max_steps=12).call(              # a first building: it asks
        "send_message", {"to": "mayor", "text": "Need $2,500 for our first school"})["ok"]
    assert T.ToolBox(env, "police").call("send_message", ask)["ok"]        # the scripted fleet: never held
    tbox = T.ToolBox(env, "transport", max_steps=12)                     # nothing ready: the mayor's copy held
    held = tbox.call("send_message", {"to": "mayor,planning", "text": "please allocate $1,000 for future upgrades"})
    assert held["ok"] and "not to the mayor: not sent: nothing ready to buy, so a money ask now gets a no" in \
        held["message"] and [m["to"] for m in tbox.messages] == ["planning"]


ROOM = FB(None, (), 0.0, (), room=True)           # utilities: ready sites left out, their supply has room


def _supplies(monkeypatch, env, **sd) -> dict:
    """Pin the live metrics' supply and demand (power=(250, 100), ...); the history keeps the real months."""
    m = dict(env.world.metrics())
    for s, (have, want) in sd.items():
        m[f"{s}_supply"], m[f"{s}_demand"] = float(have), float(want)
    monkeypatch.setattr(env.world, "metrics", lambda: m)
    return m


def test_a_supply_is_wanted_ahead_of_its_growth(monkeypatch):
    """Region run 20261005: 28 asks held back on "supply has room"; 5 of those supplies reached 99%+ before
    utilities' next turn, all 5 with demand plus the last quarter's growth at 90% of the supply (_busy, demand_ahead)."""
    env = _env(act_every=3)
    _months_spent(env, {}, {}, {}, {}, {})
    w = env.world
    m = _supplies(monkeypatch, env, power=(250, 180))
    m["demand_factor"] = {"power": 1.0, "water": 1.0, "sewage": 1.0}
    for h, d in zip(w.history[-4:], (120, 140, 160, 180)):
        h["power_demand"], h["demand_factor"] = d, {"power": 1.0}
    assert T.demand_ahead(w, "power") == 240                        # +60 over the last 3 months
    assert T._busy(w, "power") and T._wanted(w, "coal_or_gas")      # 240 >= 90% of 250, though 72% used now
    assert T.supply_ahead(w, "power") == "power 72% used, 96% in 3 months"
    for h in w.history:
        h["power_demand"] = 180                                     # flat: as before, under 90% has room
    assert T.demand_ahead(w, "power") == 180 and not T._busy(w, "power") and not T._wanted(w, "coal_or_gas")
    w.history[-4]["power_demand"] = 220                             # falling: no growth counted
    assert T.demand_ahead(w, "power") == 180
    w.history[:] = w.history[-2:]                                   # one month of growth, scaled to 3
    w.history[-2]["power_demand"] = 170
    assert T.demand_ahead(w, "power") == 210
    w.history[:] = w.history[-1:]                                   # one month recorded: demand now
    assert T.demand_ahead(w, "power") == 180
    b = T.SupplyBuild("power", 250, 180, 0, ahead=240)              # wanted for its growth: says when
    assert b.status == "power 72% used, 96% in 3 months"
    assert b._replace(demand=230).status == "power 92% used"        # busy now: as before
    assert b._replace(ahead=None).status == "power 72% used"


def test_the_supply_forecast_leaves_out_seasons_and_events(monkeypatch):
    """demand_ahead projects growth that persists: a review found summer water (x1.15) or a winter / heatwave power
    jump (x1.2-1.3, World.event_factor) read as growth, so utilities could build for a temporary spike. Each month's
    demand is taken without the multiplier it was computed under (metrics' demand_factor)."""
    env = _env(act_every=3)
    _months_spent(env, {}, {}, {}, {}, {})
    w = env.world
    m = _supplies(monkeypatch, env, water=(250, 207))
    m["demand_factor"] = {"power": 1.0, "water": 1.15, "sewage": 1.15}           # summer now
    for h, d, f in zip(w.history[-4:], (180, 180, 180, 207), (1.0, 1.0, 1.0, 1.15)):
        h["water_demand"], h["demand_factor"] = d, {"water": f}
    assert T.demand_ahead(w, "water") == 207 and not T._busy(w, "water")       # 83% used: the jump is the season
    assert T.supply_ahead(w, "water") == "water 83% used, 83% in 3 months"
    for h, d, f in zip(w.history[-4:], (180, 190, 200, 241.5), (1.0, 1.0, 1.0, 1.15)):
        h["water_demand"], h["demand_factor"] = d, {"water": f}                # +10 a month underneath
    assert T.demand_ahead(w, "water") == pytest.approx(207 + 30 * 1.15)        # at today's multiplier: 241.5
    assert T._busy(w, "water")
    # the record: refresh notes the multipliers it applied, metrics carry them into the history
    fresh = _env(events="none")
    _months_spent(fresh, {}, {}, {}, {}, {})
    h = fresh.world.history                     # month 1's record was computed in January (winter), month 5's in May
    assert h[0]["demand_factor"] == {"power": 1.2, "water": 1.0, "sewage": 1.0}
    assert h[-1]["demand_factor"] == {"power": 1.0, "water": 1.0, "sewage": 1.0}


def test_a_target_with_nothing_to_buy_is_flagged_not_a_gap(monkeypatch):
    """Region run 20261005: the Departments block read "utilities: funds $0 < target $7,000; supply has room", its
    briefing "budget target: $7,000" beside "supply has room", and utilities asked 28 times against it. A target above
    the funds of a department with nothing to buy (kind "nothing", not planning) is no gap but flagged; a blocked one
    keeps its gap (money for its first building once planning frees a lot)."""
    env = _env()
    _months_spent(env, {}, {}, {})
    _builds(monkeypatch, {"utilities": ROOM, "fire": FB(None, (), 2500, ("a fire station",))},
            road=FB(315, (ROAD,), 315, ()))
    w = env.world
    w.treasury, w.dept_funds["planning"] = 30000.0, 900                # planning's funds cover its road
    mayor = ToolBox(env, "mayor")
    res = mayor.call("update_plan", {"budgets": {"utilities": 7000, "fire": 2500, "planning": 9000},
                                     "priorities": {"utilities": "expand water capacity", "police": "patrols",
                                                    "fire": "a station by the homes"}})
    f = funding_gaps(env, FleetState.of(env))
    assert f.unusable == {"utilities": "supply has room"} and set(f.gaps) == {"fire", "planning"}
    assert "utilities" not in res["below_target"] and res["nothing_to_fund"] == {
        "utilities": "target $7,000 above its $0 funds and priority \"expand water capacity\", but supply has room: no "
                     "top-up buys anything now",
        "police": "priority \"patrols\", but nothing ready to buy: no top-up buys anything now"}   # fire: blocked
    assert res["next"].endswith("; nothing to fund now: utilities (supply has room), police (nothing ready to buy)")
    lines = {ln.split(":")[0][2:]: ln for ln in _mayor_block(env, "## Departments") if ": funds " in ln}
    assert lines["utilities"] == "- utilities: funds $0, target $7,000 (unusable now), spent $0; supply has room"
    assert lines["fire"] == "- fire: funds $0 < target $2,500, spent $0; blocked: a fire station"
    assert lines["planning"] == "- planning: funds $900 < target $9,000, spent $0"
    w.dept_funds["utilities"] = 6400                                   # within 10% of the target: nothing flagged
    assert "utilities" not in funding_gaps(env, FleetState.of(env)).unusable


def test_a_no_ask_reconciles_the_plan_and_says_what_to_do_instead(monkeypatch):
    """Region run 20261005: 125 money asks held back, every one under a budget line saying not to ask, 51 with the
    plan target above the funds; 12 went again. The no-ask line and the hold-back say the target is a ceiling, when an
    ask would be valid and what to do this turn instead."""
    env = _env(act_every=3)
    _months_spent(env, {}, {}, {})
    _builds(monkeypatch, {"utilities": ROOM, "police": POLICE_2ND})
    _supplies(monkeypatch, env, power=(250, 100), water=(250, 150), sewage=(250, 120))
    fleet = FleetState.of(env)
    fleet.update_plan(env.world.month, budgets={"utilities": 7000}, priorities={"utilities": "expand water capacity"})
    a = T.mayor_ask(env, fleet, "utilities")
    plan = "your $7,000 plan target is the mayor's ceiling for you, not money to ask for"
    when = "a supply will be 90% used within 3 months at its growth (now water 60% used, 60% in 3 months)"
    assert a == T.Ask(False, 0, "", "supply has room", plan, when, "no supply needs building yet")
    budget = _block(briefing(env, "utilities", mark=False), "## Your budget")      # (its supply line: no pump yet)
    assert budget[-1] == f"- no money ask to the mayor now: supply has room; {plan}. Ask when {when}. This turn: no " \
                        "supply needs building yet; say what you need in your end_turn report"
    box = T.ToolBox(env, "utilities", max_steps=12)
    assert box.call("send_message", {"to": "mayor", "text": "Requesting $3,000 to expand water capacity"})["error"] == (
        f"not sent: supply has room, so a money ask now gets a no; {plan}. Ask when {when}. Instead this turn: no "
        "supply needs building yet; put what you need in your end_turn report (the mayor reads it). To post this "
        "message anyway, send it to the mayor again.")
    fleet.update_plan(env.world.month, budgets={"utilities": 0})       # no target above the funds: no plan note
    assert T.mayor_ask(env, fleet, "utilities").plan == ""
    env.world.dept_funds["police"] = 3000                              # its funds cover another station: build it
    a = T.mayor_ask(env, fleet, "police")
    assert (a.ok, a.when, a.instead) == (False, "", "build another police station ($2,500): your funds cover it")


def test_one_send_message_answers_several_departments():
    """The south mayor spent all 20 steps of month 9 on one refusal per department. A comma list (or a JSON list)
    queues one council message per role, so the digests, collab threads and the Open requests block see each."""
    env = _env(act_every=3)
    _asked(env)
    fleet = FleetState.of(env)
    box = _mayor(env, treasury=30000)
    res = box.call("send_message", {"to": "transport, Police,police", "text": "Not this quarter: the reserve first"})
    assert res == {"ok": True, "message": "message to transport and police (one message each) will be posted at the "
                                          "end of your turn"}
    assert box.call("send_message", {"to": ["health"], "text": "Not this quarter: the reserve first"})["ok"]  # a list
    assert [m["to"] for m in box.messages] == ["transport", "police", "health"]
    assert T.open_requests(env, fleet, box.messages) == []
    assert box.call("send_message", {"to": "health", "text": "Not this quarter: the reserve first"})["ok"]  # no cache
    bad = box.call("send_message", {"to": "police,mayor", "text": "x"})
    assert not bad["ok"] and bad["error"].startswith("argument 'to' must be one of planning, utilities, ")
    assert box.call("send_message", {"to": "all,police", "text": "x"})["error"].startswith("'all' already reaches")
    env.apply("mayor", Decision(executed=box.executed, messages=box.messages))
    assert [(m["sender"], m["to"]) for m in env.council.posted_in(3)] == [
        ("mayor", "transport"), ("mayor", "police"), ("mayor", "health"), ("mayor", "health")]
    assert "## Open requests to you" not in briefing(env, "mayor", mark=False)
    assert "one send_message can answer several (to: \"police,fire\")" in T.requests_reminder(
        [T.OpenRequest(0, "police", "x", None, False)])
