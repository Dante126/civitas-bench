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
    assert res["to_fund"] == [] and res["next"] == "to spend: $27k above the $3,000 reserve; to fund: nothing ready"


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
    w.bus_lines.append({"id": 1, "stops": [(1, 1), (5, 1)]})     # it has one: the avenue stretch is next again
    avenue = T.first_build(env, "transport")
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
    assert move("police", 2500, confirm=True)["ok"] and move("fire", 698)["ok"]
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
    why = ("that leaves the treasury at $1,803, $1,413 under the $3,216 reserve" + DEFICIT + ", for another fire "
           "station; first/supply and growth come first: education $2,500 (first/supply), planning $315 (growth) "
           "still wait." + FUND_ANYWAY)
    assert move("fire", 2500) == {"ok": False, "refused": "reserve", "error": why}
    assert w.treasury == 4303 and len(box.executed) == 1 and box._failed == {}   # nothing ran or is remembered
    # first/supply and growth may go under the reserve
    assert move("education", 2500)["ok"] and move("planning", 315)["ok"] and round(w.treasury) == 1488
    why = ("that leaves the treasury at $488, $2,728 under the $3,216 reserve" + DEFICIT + ", for another clinic; "
           "first/supply and growth come first and none is waiting." + FUND_ANYWAY)
    first = move("health", 1000, confirm=True)       # confirm on a call not refused before: refused, with the note
    assert first == {"ok": False, "refused": "reserve", "error": why[:-1] + EARNED + "."}, first
    res = move("health", 1000, confirm=True)         # the identical repeat: earned
    assert res["ok"] and res["override"] == "reserve" and w.dept_funds["health"] == 1000, res
    assert res["message"].startswith("transferred $1,000 to health (now $1,000) | health: still $1,000 short of "
                                     "another clinic")
    assert move("health", 400)["refused"] == "reserve"          # still another, still under: another amount asks again
    # not refused: more than the treasury holds (the action says so), a blocked first building, nothing ready
    res = move("fire", 2500)
    assert res["ok"] is False and "refused" not in res and res["message"].startswith("treasury only has $488"), res
    _builds(monkeypatch, {"fire": NO_LOT, "health": CLINIC_2})
    assert DeptMoney(env, "fire").kind == "blocked" and move("fire", 300)["ok"]
    assert DeptMoney(env, "transport").kind == "nothing" and move("transport", 100)["ok"]
    assert [r["ok"] for r in box.executed] == [True, True, True, True, False, True, True]


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
    assert res["error"] == ("that leaves the treasury at $500, $2,500 under the $3,000 reserve, for another fire "
                            "station; first/supply and growth come first and none is waiting; take back idle money "
                            "first (planning $4,000)." + FUND_ANYWAY), res
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
    """The scripted fleet's mayor repeats a refusal its policy means with confirm: true, the new kinds included."""
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
                  ("allocate_budget", None, "reserve")]
    assert w.debt == 10000 and w.dept_funds["police"] == 11000 and [r["ok"] for r in d.executed] == [True, True]
