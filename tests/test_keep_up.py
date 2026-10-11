"""Keeping coverage up: a department short of KEEP_UP_GOAL names the buildings it takes to get back (tools.keep_up),
as many as the money covers (keep_up_counts), and every line agrees on that count. Late smoke 20261007 seed 2: the
LLM fleet grew the city 30% in a year while fire coverage went 94 -> 86 -> 81 -> 78%, one "another" building a
quarter, with $35-39k above the reserve."""
import copy
import re

import pytest

from civitas.agents.scripted import SCRIPTED
from civitas.core.env import CivitasEnv, GameConfig
from civitas.fleet import tools as T
from civitas.fleet.briefing import briefing
from civitas.fleet.fmt import money
from civitas.fleet.memory import DEPARTMENTS, FleetState
from civitas.fleet.tools import KEEP_UP_GAIN, KEEP_UP_MAX, DeptMoney, ToolBox, funding_gaps, keep_up, mayor_ask
from civitas.sim import catalog as C


@pytest.fixture(scope="module")
def short_env():
    """Seed 0 after 48 scripted months (about 6,500 residents) with every police station and clinic but the first
    removed and the services recomputed: police and health coverage far under the goal, every department at $0."""
    env = CivitasEnv(GameConfig(seed=0, months=60))
    agents = {r: SCRIPTED[r](r) for r in env.roles}
    for _ in range(48):
        for r in env.roles:
            env.apply(r, agents[r].act(env.observe(r)))
        env.end_month()
    w = env.world
    for struct in (C.POLICE, C.CLINIC):
        for x, y in w.facilities(struct)[1:]:
            w.remove_building(x, y)
    w.refresh()
    w._compute_services()
    w.refresh()
    w.dept_funds.update({d: 0 for d in DEPARTMENTS})
    return env


@pytest.fixture
def tier1(monkeypatch):
    """No tier upgrade on offer (tools.tier_up): these tests are about tier-1 buildings; short_env's fire has one
    downtown tile without tier-2 cover, and a fire HQ would be its next build (tests/test_tier_up.py)."""
    monkeypatch.setattr(T, "tier_up", lambda *a, **k: None)


def _copy(env, treasury: float):
    env = copy.deepcopy(env)
    env.world.treasury = treasury
    return env


def _line(text: str, head: str) -> str:
    return next(ln for ln in text.splitlines() if ln.startswith(head))


def test_service_coverage_is_the_metric_and_writes_nothing(short_env):
    env = copy.deepcopy(short_env)
    w = env.world
    m = w.metrics()
    for service, key in (("police", "police_coverage"), ("fire", "fire_coverage"), ("health", "health_coverage"),
                         ("garbage", "garbage_coverage"), ("school", "school_coverage"), ("high", "high_coverage")):
        assert T.service_coverage(w, service) == pytest.approx(m[key], abs=1e-4), service
    before = (w.version, len(w._bfs_cache), set(w.facility_state), w.struct.copy())
    sites = DeptMoney(env, "police").build.sites[0]
    assert keep_up(w, "police", "station", sites).n >= 2
    assert (w.version, len(w._bfs_cache), set(w.facility_state)) == before[:3] and (w.struct == before[3]).all()


def test_keep_up_counts_the_buildings_back_to_the_goal_capped_by_sites(short_env, tier1):
    env = copy.deepcopy(short_env)
    w = env.world
    b = DeptMoney(env, "police").build
    assert b.purposes == ("station",) and b.named[0].have == 1
    sites = b.sites[0]
    ku = keep_up(w, "police", "station", sites)
    assert ku.now == w.metrics()["police_coverage"] < 0.5
    assert 2 <= ku.n <= min(KEEP_UP_MAX, len(sites)) and len(set(ku.sites)) == ku.n and set(ku.sites) <= set(sites)
    assert all(b - a >= KEEP_UP_GAIN for a, b in zip(ku.levels, ku.levels[1:]))       # each one adds coverage
    spec = C.TIER_LINES["police_station"][0]
    for k in range(ku.n + 1):                          # each level is the sim's coverage with those stations built
        assert ku.levels[k] == pytest.approx(T.service_coverage(w, "police", ku.sites[:k], spec))
    assert keep_up(w, "police", "station", sites[:1]).n == 1                             # one site: one building
    assert keep_up(w, "fire", "station", DeptMoney(env, "fire").build.sites[0]) is None   # fire is above the goal


def test_the_money_above_the_reserve_caps_the_count(short_env):
    fleet = lambda env: FleetState.of(env)
    rich = _copy(short_env, 1_000_000)
    f = funding_gaps(rich, fleet(rich))
    want = {d: m.want for d, m in f.depts.items()}
    assert want["police"] >= 2 and want["health"] >= 2
    assert {d: m.count for d, m in f.depts.items()} == want                  # ample money: all of them
    assert set(T.fund_cover(f)) == {i.dept for i in f.to_fund}
    police = next(i for i in f.to_fund if i.dept == "police")
    assert police.count == want["police"] and police.amount == want["police"] * 2500
    # in a hold (nothing above the reserve): the one-building lines as before
    poor = _copy(short_env, 0)
    f = funding_gaps(poor, fleet(poor))
    assert all(m.count == 1 for m in f.depts.values())
    base = sum(i.amount for i in f.to_fund)
    assert _line(briefing(poor, "mayor", mark=False), "- police:").count("another police station (") == 1
    # money for every one-building top-up and one more station: police (first in turn order) gets it
    env = _copy(short_env, 0)
    env.world.treasury = T.reserve(env.world) + base + 2500
    f = funding_gaps(env, fleet(env))
    assert f.depts["police"].count == 2 and f.depts["health"].count == 1
    assert sum(i.amount for i in f.to_fund) == base + 2500 and set(T.fund_cover(f)) == {i.dept for i in f.to_fund}


def test_every_line_agrees_on_the_count(short_env):
    env = _copy(short_env, 1_000_000)
    fleet = FleetState.of(env)
    m = funding_gaps(env, fleet).depts["police"]
    n, top = m.count, m.top_up
    assert n >= 2 and top == n * 2500 and m.what == f"{n} more police stations"
    mayor = briefing(env, "mayor", mark=False)
    assert re.search(rf"- police: funds \$0[^\n]*; needs {re.escape(money(top))} for {n} more police stations "
                     r"\(coverage [^()]*, ~\d+% with them, \$\d+/month\)\n", mayor)
    assert f"police {money(top)} ({n} buildings)" in _line(mayor, "- to fund:")
    ask = mayor_ask(env, fleet, "police")
    assert ask.ok and ask.amount == top and ask.what == f"{n} more police stations"
    police = briefing(env, "police", mark=False)
    assert f"ask the mayor for {money(top)} with a reason" in police     # its $0 pays for no ready site
    keep = _line(police, "- keep up with growth: ")
    assert keep.startswith(f"- keep up with growth: {n} more police stations at (")
    assert f"once your funds cover them ({money(top)}, you have $0)" in keep
    # the mayor tops it up: its note, then the department's lines say to build all of them now
    res = ToolBox(env, "mayor").call("allocate_budget", {"department": "police", "amount": top})
    assert res["ok"] and f"police: can pay for {n} more police stations" in res["message"]
    m = funding_gaps(env, fleet).depts["police"]
    assert m.count == n and m.kind == "nothing" and m.state == f"funds cover {n} more police stations"
    police = briefing(env, "police", mark=False)
    assert f"This turn: build {n} more police stations at (" in police
    assert f"build {'both' if n == 2 else f'all {n}'} this turn: your funds cover them" in police
    # what a take-back leaves is what it was: the guards read the same keep
    assert m.keep == DeptMoney(env, "police").keep


def test_another_building_still_meets_the_reserve_guard(short_env):
    env = _copy(short_env, 0)
    w = env.world
    w.treasury = T.reserve(w) + 1000                   # a top-up of $2,500 would go under the reserve
    box = ToolBox(env, "mayor")
    res = box.call("allocate_budget", {"department": "police", "amount": 2500})
    assert res.get("refused") == "reserve" and "for another police station" in res["error"]


def test_the_needs_line_says_when_coverage_is_falling(short_env, tier1):
    env = copy.deepcopy(short_env)
    w = env.world
    now = round(100 * w.metrics()["fire_coverage"])
    w.history[-4]["fire_coverage"], w.history[-7]["fire_coverage"] = (now + 5) / 100, (now + 9) / 100
    assert T.coverage_trend(w, "fire_coverage") == f"falling: {now + 9} -> {now + 5} -> {now}%"
    w.history[-7]["fire_coverage"] = (now + 2) / 100                  # it rose, then fell: the fall only
    assert T.coverage_trend(w, "fire_coverage") == f"falling: {now + 5} -> {now}%"
    w.history[-4]["fire_coverage"] = now / 100                        # flat over the last quarter: nothing
    assert T.coverage_trend(w, "fire_coverage") == ""
    assert T.coverage_trend(w, "transit_share") == ""
    w.history[-4]["fire_coverage"] = (now + 5) / 100
    line = _line(briefing(env, "mayor", mark=False), "- fire:")
    assert f"needs $2,500 for another fire station (coverage falling: {now + 5} -> {now}%" in line


def test_ready_sites_that_add_nothing_say_where_the_homes_are(short_env, monkeypatch, tier1):
    """Late smoke seed 2, month 102: no ready station site reached the uncovered homes (another road), and the fire
    station built at the Ready line's best served one building."""
    env = copy.deepcopy(short_env)
    env.world.stats["fire_coverage"] = now = 0.8      # under the goal (metrics reread after the refresh)
    env.world.refresh()
    monkeypatch.setattr(T, "keep_up", lambda w, dept, purpose, sites: T.KeepUp(
        "fire_coverage", now, (now,), (), (7, 9)) if dept == "fire" else None)
    line = _line(briefing(env, "mayor", mark=False), "- fire:")
    assert line.endswith("; ready sites add under 1%: the uncovered homes near (7,9) are out of their road reach")
    fire = briefing(env, "fire", mark=False)
    assert ("the uncovered homes near (7,9) are out of their road reach; ask planning for a lot or a road near (7,9) "
            "before building another") in fire
    env.world.dept_funds["fire"] = 2500                # funds for a station: it isn't told to build one there
    fire = briefing(env, "fire", mark=False)
    assert "This turn: ask planning for a lot or a road near the uncovered homes at (7,9)" in fire
    assert "This turn: build" not in fire
    env.world.dept_funds["fire"] = 0
    # most of what is missing downtown, where tier 1 covers 60% at most: it says so
    monkeypatch.setattr(T, "keep_up", lambda w, dept, purpose, sites: T.KeepUp(
        "fire_coverage", now, (now,), (), (7, 9), 0.07) if dept == "fire" else None)
    assert _line(briefing(env, "mayor", mark=False), "- fire:").endswith(
        "out of their road reach; 7 points are lost downtown (level 4-5), where tier 1 covers 60% at most: "
        "upgrade_facility to tier 2")
