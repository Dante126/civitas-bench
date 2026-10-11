"""civitas-v2 guided fixes (docs/plans/2026-10-10-civitas-v2-plan.md, Track A) on the late game the models take over:
fleet_smoke_late's warm-up (the scripted fleet, months 0-95) at the hand-over, seed 4 (6,052 residents, eight tier-1
police and fire stations each, 24 downtown tiles without tier-2 police / fire / health cover, no hospital, no train
station or harbour). v1 (4 models x 8 seeds): education's line named a $12k university ahead of 78-85% high-school
enrolment and then another elementary school at ~99%; police and fire named another tier-1 station every turn and never
an upgrade; transport's station or harbour never reached the mayor's to-fund line."""
import copy
import json
import math
import re
from pathlib import Path

import pytest

from civitas.agents.scripted import SCRIPTED, SCRIPTED_V1
from civitas.bench.runner import warm_up
from civitas.config import load_config
from civitas.core.actions import apply_action
from civitas.fleet import planning as P
from civitas.fleet import tools as T
from civitas.fleet.briefing import briefing
from civitas.fleet.fmt import money
from civitas.fleet.memory import FleetState
from civitas.fleet.tools import DeptMoney, ToolBox, first_build, funding_gaps, mayor_ask
from civitas.sim import catalog as C

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = {"police": "police_coverage", "fire": "fire_coverage", "health": "health_coverage"}


@pytest.fixture(scope="module")
def late():
    """Seed 4 at month 96, before anyone acts (as `civitas prompt -c configs/fleet_smoke_late.toml --role mayor
    --seed 4` shows it); the [game] the warm-up plays pinned here."""
    cfg = load_config(str(ROOT / "configs/fleet_smoke_late.toml"))
    cfg["game"].update(months=108, warmup_months=96, act_every=3, max_actions_per_turn=8)
    with pytest.MonkeyPatch.context() as mp:          # the city these tests were written on: civitas-v1's mayor
        mp.setitem(SCRIPTED, "mayor", SCRIPTED_V1["mayor"])
        env, _ = warm_up(cfg, 4)
    return env


def _line(text: str, head: str) -> str:
    return next(ln for ln in text.splitlines() if ln.startswith(head))


def _args(call: str) -> dict:
    return json.loads(call.split(" ", 1)[1])


def _recompute(w) -> None:
    w.refresh()
    w._compute_services()
    w.refresh()


# ------------------------------------------------------------------ education: by enrolment
def test_school_levels_go_by_enrolment(late):
    env = copy.deepcopy(late)
    w = env.world
    w.stats.update(seats_university=0)
    w.enrollment.update(elementary=0.99, high=0.8, university=0.0)
    w.stats.update(high_coverage=0.8)                  # short of schools in reach, not of seats: a new high school
    w.refresh()
    assert not T.seat_bound(w, "high") and T.enrol_short(w, "high") and not T.enrol_short(w, "elementary")
    b = first_build(env, "education")
    assert b.purposes == ("high_school",) and b.named[0].label in ("another high school", "its first high school")
    assert not T._wanted(w, "university") and not T._wanted(w, "school")
    edu = briefing(env, "education", mark=False)
    assert "- school: none - elementary enrolment 99% (goal 90%)" in edu
    assert "- university: none - high-school enrolment 80%: schools first (a university counts in no enrolment" in edu
    assert "high-school enrolment 80%: " in _line(edu, "- elementary enrolment 99%")       # Key metrics
    # both school levels enrolled: a university, another building, never a first one (v1 named it first whatever
    # enrolment; test_a_university_is_never_a_first_building)
    w.enrollment.update(high=0.95)
    w.stats.update(high_coverage=0.95)
    w.refresh()
    b = first_build(env, "education")
    assert [(n.what, n.have, n.first) for n in b.named] == [("a university", None, False)]
    # and with a university enrolling its students, nothing to build: the line says enrolment is covered
    w.stats.update(seats_university=1500)
    w.enrollment.update(university=1.0)
    w.refresh()
    m = DeptMoney(env, "education")
    assert m.needs is None and m.state.startswith("enrolment covered: elementary 99%, high-school 95%, university 100%")
    w.dept_funds["education"] = 0                      # nothing idle: the line says why it needs nothing
    assert _line(briefing(env, "mayor", mark=False), "- education:").endswith(f"; {DeptMoney(env, 'education').state}")


def test_a_level_short_of_seats_names_its_bigger_school(late):
    """High-school enrolment short with the teens' homes in reach but too few seats (seed 3 at month 96: 86%, 700 seats
    for 817 teens, 92% in reach): the bigger tier of an overloaded high school (upgrade_facility), not another one."""
    env = copy.deepcopy(late)
    w = env.world
    (hx, hy), = w.facilities(C.HIGHSCHOOL)
    teens = float(w.cohorts[1].sum())
    seats = round(0.7 * teens)
    w.stats.update(seats_high=seats, high_coverage=0.95)
    w.enrollment.update(elementary=0.99, high=seats / teens)
    for s in w.service_info["high_school"]:
        s.update(load=round(teens), capacity=seats)       # overloaded: the observation offers its next tier
    w.refresh()
    assert T.seat_bound(w, "high")
    b = first_build(env, "education")
    up = b.named[0]
    cur, nxt = C.TIER_LINES["high_school"][:2]
    assert up.call == f'upgrade_facility {{"x":{hx},"y":{hy},"module":"tier"}}' and not up.first
    assert b.cost == C.upgrade_cost(cur, nxt) == 7200
    assert up.label == f"an upgrade of its high school at ({hx},{hy}) to a large high school"
    w.dept_funds["education"] = 3000
    mayor = briefing(env, "mayor", mark=False)
    assert f"; needs $7,200 for {up.label} (high-school enrolment {round(100 * seats / teens)}%: " in mayor
    assert DeptMoney(env, "education").kind == "another"
    edu = briefing(env, "education", mark=False)
    assert f"- tier upgrade: an upgrade of your high school at ({hx},{hy}) to a large high school ($7,200" in edu
    assert up.call in edu
    w.dept_funds["education"] = b.cost                 # the call is what the action takes, for that price
    res = apply_action(w, "education", {"type": "upgrade_facility", **_args(up.call)})
    assert res.ok and res.cost == b.cost and w.facility_tier(hx, hy) == 2


# ------------------------------------------------------------------ police, fire, health: tier upgrades
def test_the_upgrade_estimate_is_the_sim_after_it(late):
    """The coverage a tier upgrade (or health's fresh hospital) is named with is the sim's after building it, and its
    price is what the action charges."""
    named = 0
    for dept in ("police", "fire", "health"):
        env = copy.deepcopy(late)
        w = env.world
        b = first_build(env, dept)
        n = b.named[0]
        if not n.call:
            continue
        named += 1
        assert not n.first and n.have is None and b.purposes and len(b.named) == 1
        est = int(re.search(r"~(\d+)% with it", n.status)[1])
        assert f"{w.stats['downtown_uncovered'][dept]} downtown tile" in n.status
        w.dept_funds[dept] = b.cost
        res = apply_action(w, dept, {"type": n.call.split(" ")[0], **_args(n.call)})
        assert res.ok and res.cost == b.cost, res
        _recompute(w)
        assert round(100 * w.metrics()[COVERAGE[dept]]) == est, (dept, n.status)
    assert named >= 2                                   # seed 4: police and fire HQs, health's first hospital


def test_every_line_prices_the_upgrade_the_same(late):
    env = copy.deepcopy(late)
    w = env.world
    fleet = FleetState.of(env)
    f = funding_gaps(env, fleet)
    m = f.depts["police"]
    b = m.build
    up = b.named[0]
    x, y = _args(up.call)["x"], _args(up.call)["y"]
    cur, nxt = w.facility_spec(x, y), C.TIER_LINES["police_station"][1]
    assert b.cost == C.upgrade_cost(cur, nxt) and up.upkeep == pytest.approx(nxt["upkeep"] - cur["upkeep"])
    # what it holds is kept for the upgrade (v1: "idle $3,095 (keeps $4,000)", its $4,000 jail the dearest site)
    assert m.idle == 0 and m.keep >= b.cost and m.kind == "another"
    assert m.top_up == math.ceil(b.cost - m.funds) > 0
    assert T.FundItem("police", "another", m.top_up, up.upkeep, up.label) in f.to_fund
    mayor = briefing(env, "mayor", mark=False)
    assert _line(mayor, "- police:").endswith(f"; needs {money(b.cost)} for {up.label} ({up.status}, "
                                              f"{money(up.upkeep)}/month)")
    ask = mayor_ask(env, fleet, "police")
    assert (ask.ok, ask.amount) == ("police" in T.fund_cover(f), m.top_up if ask.ok else 0)
    police = briefing(env, "police", mark=False)
    assert f"- tier upgrade: {T._yours(up.label)} ({money(b.cost)}, +$75/month; " in police and up.call in police
    assert (f"- downtown (level 4-5) tiles without tier-2 cover: {w.stats['downtown_uncovered']['police']} (tier 1 "
            "covers them to 60% at most)") in police
    assert _line(police, "- upgrade: ").endswith(f"; {money(b.cost)} (upgrade_facility, module \"tier\")")
    # its funds cover it: this turn's instruction is the exact call
    w.dept_funds["police"] = b.cost
    assert f"This turn: {up.call} ({T._yours(up.label)}, {money(b.cost)}): your funds cover it" in \
        briefing(env, "police", mark=False)
    # an upgrade is another building to the guards: a top-up that leaves the treasury under the reserve is refused
    w.dept_funds["police"] = 0
    w.treasury = T.reserve(w) + 100
    res = ToolBox(env, "mayor").call("allocate_budget", {"department": "police", "amount": 1000})
    assert res.get("refused") == "reserve", res


def test_no_upgrade_without_a_downtown_gap_or_when_a_station_adds_more(late, monkeypatch):
    env = copy.deepcopy(late)
    w = env.world
    assert first_build(env, "police").named[0].call
    w.stats["downtown_uncovered"] = {**w.stats["downtown_uncovered"], "police": 0}
    b = first_build(env, "police")
    assert not any(n.call for n in b.named)
    # a downtown gap, but a tier-1 station at a ready site adds more: the station
    w.stats["downtown_uncovered"]["police"] = 24
    real = T._Coverage.value
    monkeypatch.setattr(T._Coverage, "value", lambda self, extra=(), up=None: (
        (0.99, None) if extra and up is None else real(self, extra, up)))
    T._KEEP_UPS.pop(w, None)
    assert not any(n.call for n in first_build(env, "police").named)


# ------------------------------------------------------------------ transport: stations reach the to-fund line
def test_the_first_link_is_a_first_building_once_demand_is_capped(late):
    """Seed 4: 6,052 residents and no train station or harbour, so R and C demand are capped (World.demand_caps), and no
    free map-edge lot by a road: before v2 the list read "nothing needs it right now"."""
    env = copy.deepcopy(late)
    w = env.world
    assert T.capped_unlinked(w)
    lst = T.site_lists(w, "transport", T.role_obs(env, "transport"))["train_station"]
    assert (lst.sites, lst.blocked) == ([], True) and "ask planning for a road to the map edge" in lst.short
    assert "train_station" in T.land_blocked(env)["transport"]
    planning = briefing(env, "planning", mark=False)
    args, (lx, ly), _ = P.edge_road(w)
    assert "a road to the map edge for transport's first train station" in _line(planning, "- Next: ")
    assert f'build_road {json.dumps(args, separators=(",", ":"))}' in _line(planning, "- Next: ")
    assert "transport (train station: a road to the map edge)" in _line(planning, "- blocked for lack of land: ")
    v = P.planning_view(env, T.site_lists(w, "planning", T.role_obs(env, "planning")), w.dept_funds["planning"])
    assert P.request_fix(w, "transport", v, "a lot for our train station") == (
        f'its first train station needs a road to the map edge: build_road {json.dumps(args, separators=(",", ":"))} '
        f"({money(v.link.cost)})")
    # planning builds it: the station is transport's next build, a first building, before another bus line
    w.dept_funds["planning"] += 1000
    assert apply_action(w, "planning", {"type": "build_road", **args}).ok
    b = first_build(env, "transport")
    assert b.purposes == ("train_station",) and [(n.label, n.first) for n in b.named] == [("its first train station",
                                                                                          True)]
    assert b.cost == C.TRAIN_STATION["cost"]
    w.dept_funds["transport"] = 2000
    f = funding_gaps(env, FleetState.of(env))
    assert T.FundItem("transport", "first/supply", 6000, C.TRAIN_STATION["upkeep"], "its first train station") \
        in f.to_fund
    nxt = _line(briefing(env, "transport", mark=False), "- Next: ")
    assert nxt.startswith("- Next: your first train station ($8,000): build_train_station {")
    s = _args(nxt.split("): ", 1)[1].split(";")[0])
    assert apply_action(w, "transport", {"type": "build_train_station", **s}).ok is False      # $2,000: the ask
    w.dept_funds["transport"] = 8000
    assert apply_action(w, "transport", {"type": "build_train_station", **s}).ok and not T.capped_unlinked(w)


def test_a_station_is_priced_another_and_only_when_it_pays(late, monkeypatch):
    """Once the city has a link: another train station or harbour is offered only while freight runs FREIGHT_FULL of
    its capacity (v1 seed 3, month 96: "Next: a harbour ($10k)" at 15%), and then it is a priced "another" build the
    to-fund line counts (it was a cost-less extra: kind "nothing", no to-fund item, never funded)."""
    env = copy.deepcopy(late)
    w = env.world
    args, _, _ = P.edge_road(w)
    w.dept_funds.update(planning=w.dept_funds["planning"] + 1000, transport=8000)
    assert apply_action(w, "planning", {"type": "build_road", **args}).ok
    site = T.site_lists(w, "transport", T.role_obs(env, "transport"))["train_station"].sites[0]
    assert apply_action(w, "transport", {"type": "build_train_station", "x": site["x"], "y": site["y"]}).ok
    w.refresh()
    lists = T.site_lists(w, "transport", T.role_obs(env, "transport"))
    lst = lists["train_station"]
    if lst.sites:                                       # an edge lot left: another station waits for freight
        assert lst.short.startswith("freight runs ") and "another train station pays from 70%" in lst.short
    m = w.metrics()
    m["exports"], m["imports"] = 0.8 * m["freight_capacity"], 0.0      # freight past 70% of capacity
    lists = T.site_lists(w, "transport", T.role_obs(env, "transport"))
    assert lists["train_station"].sites and lists["train_station"].short is None
    for p in ("bus_stops", "congested_roads"):          # nothing else ready: the station is the next build
        lists[p] = lists[p]._replace(sites=[], short="nothing")
    real = T.site_lists
    monkeypatch.setattr(T, "site_lists", lambda w_, role, obs: lists if role == "transport" else real(w_, role, obs))
    b = first_build(env, "transport")
    assert b.cost == C.TRAIN_STATION["cost"] and [(n.label, n.first) for n in b.named] == [("a train station", False)]
    w.dept_funds["transport"] = 0
    m = DeptMoney(env, "transport")
    assert (m.kind, m.top_up, m.keep) == ("another", C.TRAIN_STATION["cost"], C.TRAIN_STATION["cost"])


# ------------------------------------------------------------------ civitas-v2 review: what biased the guided evals
def test_a_university_is_never_a_first_building(late):
    """eval_v2 seeds 4, 5, 12, 18, 20 at month 96: with both school levels at 90% the needs line read "needs $12k for
    its first university", first/supply (the treasury down to $3,000, past the loan guard). A university counts in no
    enrolment the score reads: it is "a university", another building, funded only from money above the reserve."""
    env = copy.deepcopy(late)
    w = env.world
    w.stats.update(seats_university=0, high_coverage=0.95)
    w.enrollment.update(elementary=0.99, high=0.95, university=0.0)
    w.refresh()
    assert not T.site_price(w, "education", "university").first and T.site_price(w, "education", "school").first
    b = first_build(env, "education")
    assert b.purposes == ("university",) and [(n.label, n.first) for n in b.named] == [("a university", False)]
    w.dept_funds["education"] = 0
    fleet = FleetState.of(env)
    f = funding_gaps(env, fleet)
    m = f.depts["education"]
    assert (m.kind, m.what) == ("another", "a university") and m.keep >= b.cost
    assert [i.kind for i in f.to_fund if i.dept == "education"] == ["another"]
    mayor = briefing(env, "mayor", mark=False)
    assert f"; needs {money(b.cost)} for a university (" in _line(mayor, "- education:")
    assert "first university" not in mayor
    ask = mayor_ask(env, fleet, "education")
    assert ask.ok == ("education" in T.fund_cover(f)) and "first" not in ask.what
    # another building to the guards: a top-up that leaves the treasury under the reserve is refused
    w.treasury = T.reserve(w) + 100
    res = ToolBox(env, "mayor").call("allocate_budget", {"department": "education", "amount": 1000})
    assert res.get("refused") == "reserve", res


def test_kept_up_coverage_names_no_station_nor_upgrade(late):
    """eval_v2 seeds at month 96: "This turn: build another fire station ($2,500): your funds cover it" at 96.9% fire
    coverage, where the best ready site added 0.00; seed 18: a $7,800 police HQ for 97 -> 98%. At KEEP_UP_GOAL or more
    the station list reads covered, nothing is named, the mayor's to-fund line leaves the department out and its money
    above the $1,000 floor is idle."""
    env = copy.deepcopy(late)
    w = env.world
    assert first_build(env, "fire").named[0].call                  # seed 4: the fire HQ, coverage under the goal
    w.metrics()["fire_coverage"] = 0.95
    assert T.at_goal(w, "fire", "station")
    lst = T.site_lists(w, "fire", T.role_obs(env, "fire"))["station"]
    assert lst.sites and lst.short == "fire coverage 95% (goal 90%)" and not lst.blocked
    offers = T._offers(T.role_obs(env, "fire"), "fire")
    assert offers and T.tier_up(w, "fire", [(s["x"], s["y"]) for s in lst.sites], offers) is None
    b = first_build(env, "fire")
    assert b.cost is None and not b.named and b.covered.startswith("fire coverage 95% (goal 90%)")
    w.dept_funds["fire"] = 3000
    d = funding_gaps(env, FleetState.of(env)).depts["fire"]
    assert d.keep == d.floor and d.needs is None
    w.dept_funds["fire"] = 0
    f = funding_gaps(env, FleetState.of(env))
    d = f.depts["fire"]
    assert d.kind == "nothing" and d.state.startswith("fire coverage 95% (goal 90%)")
    assert "fire" not in [i.dept for i in f.to_fund]
    assert _line(briefing(env, "mayor", mark=False), "- fire:").endswith(f"; {d.state}")
    w.dept_funds["fire"] = 2000                         # its shelter paid for: the ask line, not "can't pay"
    fire = briefing(env, "fire", mark=False)
    assert "- station: none - fire coverage 95% (goal 90%)" in fire
    assert "This turn: nothing needs building now" in _line(fire, "- no money ask to the mayor now: ")
    assert "another fire station" not in fire
    assert "- tier upgrade:" not in fire and "upgrade_facility {" not in fire


def test_a_station_no_ready_site_lifts_is_not_named(late, monkeypatch):
    """Under the goal, another station whose best ready site adds under KEEP_UP_GAIN (keep_up picks none) is not named
    either: the list says so, and the department reads where the uncovered homes are (FirstBuild.gap); a jail, the
    cheapest ready site left, only while the cells are busy."""
    env = copy.deepcopy(late)
    w = env.world
    w.metrics()["police_coverage"] = 0.8
    w.stats["downtown_uncovered"] = {**w.stats["downtown_uncovered"], "police": 0}     # no tier upgrade to name
    real = T.keep_up
    monkeypatch.setattr(T, "keep_up", lambda w_, dept, p, sites: (
        T.KeepUp("police_coverage", 0.8, (0.8,), (), (3, 4), 0.05) if (dept, p) == ("police", "station")
        else real(w_, dept, p, sites)))
    lst = T.site_lists(w, "police", T.role_obs(env, "police"))["station"]
    assert lst.sites and lst.short == "ready sites add under 1% (police coverage 80%)"
    assert "the uncovered homes near (3,4) are out of their road reach; 5 points are lost downtown" in lst.note
    assert not T._cells_busy(w)
    b = first_build(env, "police")
    assert b.cost is None and not b.named and b.gap == (3, 4) and b.covered.startswith("ready sites add under 1%")
    w.dept_funds["police"] = 0
    d = DeptMoney(env, "police")
    assert d.kind in ("nothing", "blocked") and d.gap == (3, 4)
    assert ("ask planning for a lot or a road near the uncovered homes at (3,4)"
            in mayor_ask(env, FleetState.of(env), "police").instead)
    assert "cheapest: jail" not in briefing(env, "police", mark=False)


def test_the_first_link_counts_before_the_cap(late, monkeypatch):
    """eval_v2 seed 20 at month 96: 5,932 residents, no link, a harbour ready at $10k, transport holding $5,194; the
    needs line named avenue tiles, so the mayor read "idle $4,194 ... no top-up needed". From LINK_AHEAD of
    CAP_POP_CONNECTION an unlinked city's first link is a first building: kept for while a bus line comes first, then
    named before avenues."""
    env = copy.deepcopy(late)
    w = env.world
    args, _, _ = P.edge_road(w)
    w.dept_funds["planning"] += 1000
    assert apply_action(w, "planning", {"type": "build_road", **args}).ok      # an edge lot for a train station
    w.refresh()
    w.metrics()["population"] = round(0.9 * C.CAP_POP_CONNECTION)
    assert T.link_due(w) and not T.capped_unlinked(w)
    lists = T.site_lists(w, "transport", T.role_obs(env, "transport"))
    assert lists["train_station"].sites and not lists["train_station"].short
    real = T.site_lists
    monkeypatch.setattr(T, "site_lists", lambda w_, role, obs: lists if role == "transport" else real(w_, role, obs))
    if lists["bus_stops"].sites and not lists["bus_stops"].short:              # a bus line first, the link kept for
        b = first_build(env, "transport")
        assert b.purposes == ("bus_stops",) and b.dearest == C.TRAIN_STATION["cost"]
        assert b.keep_for == "its first train station"
    for p in ("bus_stops", "congested_roads"):
        lists[p] = lists[p]._replace(sites=[], short="nothing")
    b = first_build(env, "transport")
    assert b.purposes == ("train_station",)
    assert [(n.label, n.first) for n in b.named] == [("its first train station", True)]
    w.dept_funds["transport"] = 5000
    d = DeptMoney(env, "transport")
    assert (d.kind, d.idle, d.top_up) == ("first/supply", 0, C.TRAIN_STATION["cost"] - 5000)
    # below LINK_AHEAD it is a station like any: another
    w.metrics()["population"] = round(0.8 * C.CAP_POP_CONNECTION)
    assert not T.link_due(w)
    b = first_build(env, "transport")
    assert [(n.label, n.first) for n in b.named] == [("a train station", False)]
