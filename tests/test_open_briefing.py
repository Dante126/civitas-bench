"""The open track (docs/plans/2026-10-10-civitas-v2-plan.md, Track C): `[fleet] briefing = "open"` keeps the game,
tools, guards and facts and drops every prescription from briefings and tool results (exact calls, "This turn", Next
lines, the to-fund / to-spend lines, "ask the mayor for $X", fixes after a failure). The guided style, the default,
stays byte for byte what it was: tests/data/guided_seed4.json.gz holds the guided briefings and a script of tool
calls at fleet_smoke_late's seed-4 hand-over as the code before the switch rendered them (`snapshot()` regenerates it
after an intended guided change)."""
import copy
import gzip
import json
import re
import sys
from pathlib import Path

import pytest

from civitas.bench.runner import warm_up
from civitas.config import load_config
from civitas.fleet.briefing import briefing, system_prompt
from civitas.fleet.config import FleetAgentConfig, fleet_agent_config
from civitas.fleet.tools import ALL_ACTIONS, SITE_PURPOSES, ToolBox

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "tests/data/guided_seed4.json.gz"
LATE = ROOT / "configs/fleet_smoke_late.toml"
OPEN = ROOT / "configs/fleet_open_late.toml"
ROLES = ("mayor", "planning", "utilities", "transport", "police", "fire", "health", "education")
_ACTS = "|".join(sorted(ALL_ACTIONS, key=len, reverse=True))
# what the open style never shows: an action followed by its arguments (JSON, a stops list, x1=..., a tile), a
# "This turn" instruction, a Next line or refreshed next clause, an ask amount, the to-fund / to-spend lines, a
# repayment to make, "ask planning", "route around", "... instead"
FORBIDDEN = {
    "exact call": re.compile(rf"\b({_ACTS})\s*(\{{|\(\"|stops \[\[|\(?x1=|\(\d)"),
    "This turn": re.compile(r"\bThis turn\b|\bthis turn: "),
    "Next line": re.compile(r"\bNext:|\| next:"),
    "ask the mayor for $": re.compile(r"ask the mayor for (\$|funds)"),
    "route around": re.compile(r"route around"),
    "to fund / to spend": re.compile(r"- to (fund|spend):|still to fund|\"to_fund\"|\"next\":"),
    "repay_loan suggestion": re.compile(r"repay_loan \d"),
    "ask planning": re.compile(r"ask planning"),
    "instead": re.compile(r"\binstead\b"),
}


def _late_cfg() -> dict:
    cfg = load_config(str(LATE))
    cfg["game"].update(months=108, warmup_months=96, act_every=3, max_actions_per_turn=8)
    return cfg


def _handover(seed: int):
    env, _ = warm_up(_late_cfg(), seed)
    return env


@pytest.fixture(scope="module")
def late():
    """fleet_smoke_late's seed 4 at month 96, before anyone acts."""
    return _handover(4)


def tool_script(env, role: str, **kw) -> list:
    """A fixed script of `role`'s tool calls on `env` (changed: pass a copy) with the model's toolbox (branch dezones,
    max_steps, refreshed Next lines; `kw`: the style), as [role, tool, args, result]: every find_sites list, then
    actions that succeed, fail and meet the guards, then my_facilities. Only what the code before the briefing switch
    had too, so snapshot() can run it there."""
    box = ToolBox(env, role, branch_dezone=True, max_steps=20, next_line=True, **kw)
    w, out = env.world, []

    def call(name, args=None):
        res = box.call(name, json.dumps(args or {}))
        out.append([role, name, args or {}, res])
        return res

    hall = {"x": int(w.townhall[0]), "y": int(w.townhall[1])}         # a taken tile
    first = lambda p: ((call("find_sites", {"purpose": p}).get("sites") or [None])[0])
    if role == "mayor":
        call("find_sites", {"purpose": "utilities.pump"})
        call("update_plan", {"budgets": {"police": 9000}, "priorities": {"utilities": "more water"}})
        call("allocate_budget", {"department": "planning", "amount": -11000})       # below what it keeps
        call("allocate_budget", {"department": "transport", "amount": 630})
        call("allocate_budget", {"department": "transport", "amount": 630})         # a duplicate
        call("allocate_budget", {"department": "fire", "amount": 20000})            # under the reserve
        call("take_loan", {"amount": 5000})
        call("allocate_budget", {"department": "police", "amount": 999999})         # more than the treasury
        call("repay_loan", {"amount": 100})
        call("end_turn", {"report": "done"})                                        # open requests
    else:
        sites = {p: first(p) for p in SITE_PURPOSES[role]}
        if role == "planning":
            if (z := sites.get("zone_I")) and "x2" in z:
                args = {"zone": "I", "x1": z["x1"], "y1": z["y1"], "x2": z["x2"], "y2": z["y2"]}
                call("zone", args)
                call("zone", {**args, "zone": "C"})                                 # on its own zoning
            if (r := sites.get("road_extension")) and "x2" in r:
                call("build_road", {k: r[k] for k in ("x1", "y1", "x2", "y2")})
                call("build_road", {k: r[k] for k in ("x1", "y1", "x2", "y2")})     # already road
            call("build_road", {"x1": hall["x"] - 3, "y1": hall["y"], "x2": hall["x"] + 3, "y2": hall["y"]})
            call("build_park", hall)
            call("dezone", {"x1": 0, "y1": 0, "x2": 1, "y2": 1})
            call("zone", {"zone": "R", "density": "high", "x1": 0, "y1": 0, "x2": 47, "y2": 47})
        elif role == "transport":
            call("create_bus_line", {"stops": [[3, 24]]})
            if (c := sites.get("congested_roads")) and "x2" in c:
                call("upgrade_road", {"to": "avenue", **{k: c[k] for k in ("x1", "y1", "x2", "y2")}})
        else:
            build = {"utilities": ("build_power_plant", {"kind": "wind"}), "police": ("build_police_station", {}),
                     "fire": ("build_fire_station", {}), "health": ("build_clinic", {}),
                     "education": ("build_school", {})}[role]
            call(build[0], {**build[1], **hall})                                    # a taken tile
            offers = call("my_facilities").get("upgrades") or []
            if isinstance(offers, dict):                                            # health's by line
                offers = next((v for v in offers.values() if v), [])
            if offers:
                call("upgrade_facility", {"x": offers[0]["x"], "y": offers[0]["y"], "module": "tier"})
        call("send_message", {"to": "mayor", "text": "We need $5,000 for a new building."})
        call("end_turn", {"report": "done"})
    call("my_facilities")
    return out


def run_script(env, **kw) -> dict:
    """Every role's briefing at `env` (none marked) and its tool_script on a copy of `env`."""
    briefs = {r: briefing(env, r, mark=False, **kw) for r in ROLES}
    calls = [row for r in ROLES for row in tool_script(copy.deepcopy(env), r, **kw)]
    return json.loads(json.dumps({"briefings": briefs, "calls": calls}, default=str))


def snapshot(path: Path = SNAPSHOT) -> None:
    """Write the guided snapshot from the code on sys.path (python tests/test_open_briefing.py)."""
    data = run_script(_handover(4))
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(data, f, sort_keys=True)


def _texts(data: dict) -> list[tuple[str, str]]:
    return ([(f"briefing {r}", t) for r, t in data["briefings"].items()]
            + [(f"{r} {name} {json.dumps(args)}", json.dumps(res)) for r, name, args, res in data["calls"]])


def _forbidden(text: str) -> list[str]:
    return [f"{k}: ...{text[max(0, m.start() - 60):m.end() + 40]}..." for k, rx in FORBIDDEN.items()
            for m in rx.finditer(text)]


# ------------------------------------------------------------------ the switch
def test_briefing_style_config():
    """[fleet] briefing: "guided" by default, per role like every fleet key, validated."""
    from civitas.fleet.fmt import advice, briefing_style, is_open
    assert FleetAgentConfig().briefing == "guided"
    cfg = {"fleet": {"briefing": "open"}, "agents": {"mayor": {"kind": "fleet", "briefing": "guided"},
                                                     "police": {"kind": "fleet"}}}
    assert fleet_agent_config(cfg, "mayor").briefing == "guided"
    assert fleet_agent_config(cfg, "police").briefing == "open"
    with pytest.raises(ValueError, match="briefing must be one of guided, open"):
        fleet_agent_config({"fleet": {"briefing": "terse"}}, "police")
    with pytest.raises(ValueError):
        with briefing_style("terse"):
            pass
    assert not is_open()
    with briefing_style("open"):
        assert is_open() and advice("do it", "a fact") == "a fact"
        with briefing_style(None):                     # None keeps the style in force
            assert is_open()
    assert not is_open() and advice("do it", "a fact") == "do it"
    opened = load_config(str(OPEN))
    late = load_config(str(LATE))
    assert opened["game"] == late["game"] and opened["scoring"] == late["scoring"]
    for role in ROLES:
        assert fleet_agent_config(opened, role).briefing == "open"
        assert fleet_agent_config(late, role).briefing == "guided"
        for k in ("model", "provider", "max_steps", "max_tokens", "turn_tokens", "temperature", "mode"):
            assert getattr(fleet_agent_config(opened, role), k) == getattr(fleet_agent_config(late, role), k)
    for role in ROLES:                                 # the open playbooks name no briefing line that tells what to do
        book = fleet_agent_config(opened, role).playbook + fleet_agent_config(opened, role).fleet_playbook
        assert not re.search(r"Next line|This turn|to-fund|to fund|needs line|budget block says|exact call", book), role
        assert not [h for h in _forbidden(book) if h.split(":")[0] in ("exact call", "This turn", "Next line")], role
    # the system prompt: the departments' team rule points at no budget-block instruction in the open style
    assert "budget block says to" in system_prompt("police", fleet_agent_config(late, "police"), 108)
    assert "budget block says to" not in system_prompt("police", fleet_agent_config(opened, "police"), 108)


def test_guided_is_unchanged(late):
    """The guided style (the default) renders the briefings and tool results the code before the switch rendered at
    the seed-4 hand-over, byte for byte, and is what a call without a style gets."""
    with gzip.open(SNAPSHOT, "rt") as f:
        want = json.load(f)
    got = run_script(late, style="guided")
    for r in ROLES:
        assert got["briefings"][r] == want["briefings"][r], r
        assert briefing(late, r, mark=False) == got["briefings"][r]
    assert len(got["calls"]) == len(want["calls"])
    for g, w in zip(got["calls"], want["calls"]):
        assert g == w, (g[:3], g[3], w[3])
    assert ToolBox(late, "planning").style == "guided"


def test_open_states_facts_not_calls(late):
    """The open style at the seed-4 hand-over: no exact call, This turn, Next line, ask amount or to-fund line in any
    briefing or in the script's tool results, and the facts are all there: the reserve, operations and debt, each
    department's funds and priced options, sites with prices, coverage and enrolment, the transit estimate."""
    data = run_script(late, style="open")
    bad = [(where, hit) for where, text in _texts(data) for hit in _forbidden(text)]
    assert not bad, bad[:5]
    b = data["briefings"]
    mayor = b["mayor"]
    assert "## Finances" in mayor
    assert re.search(r"- reserve \$[\d.,k]+ \(3 x \$[\d,]+ expenses\): the treasury is", mayor)
    assert re.search(r"operations [+-]\$", mayor) and "debt $0" in mayor
    for d in ROLES[1:]:
        assert re.search(rf"- {d}: funds \$[\d.,k]+, target \$[\d.,k]+, spent \$[\d.,k]+, keeps \$[\d.,k]+; options: ",
                         mayor), d
    # (no police station: its ready sites add under 1% to police coverage, civitas-v2's keep_up_room)
    assert "jail $4,000 (+$" in mayor and "upgrade police station (" in mayor
    assert "- station: none - ready sites add under 1% (police coverage 76%)" in b["police"]
    assert "## Open requests to you" in mayor and "asks $8,300" in mayor
    for d in ROLES[1:]:
        assert re.search(r"- \$[\d.,k]+ in department funds", b[d]), d
        assert re.search(r"## Ready sites\n(- .*\n)*- \w+: \d+ sites?, best .*; \$[\d,.k]+", b[d]), d
    assert "police coverage: 76%" in b["police"]
    assert "downtown (level 4-5) tiles without tier-2 cover: 28" in b["police"]
    assert re.search(r"coverage estimate: an upgrade of your police station at \(\d+,\d+\) to a police HQ "
                     r"\(\$7,800, \+\$75/month\): coverage falling: 77 -> 76%, ~97% with it", b["police"])
    assert "elementary enrolment 92%" in b["education"] and "university: 3 sites, best" in b["education"]
    assert re.search(r"transit estimate: 1.1% of trips now, about \d+% with a line through the 7 bus_stops stops",
                     b["transport"])
    assert "bus_stops: 7 stops for one line: (3,24), " in b["transport"]
    assert "train_station: none - blocked: no free map-edge lot by a connected road" in b["transport"]
    assert "pump: none - blocked: riverbank beside roads all taken" in b["utilities"]
    assert "road_extension: 4 sites, best (32,24)-(47,24), opens 56 land tiles; $350 road + $280 zoning" in \
        b["planning"]
    parks = next(ln for ln in b["planning"].splitlines() if ln.startswith("- parks: "))
    assert parks.startswith("- parks: 21 of 18 park tiles") and parks.endswith("8 homes near; $600")
    # the tools stay the same: identical schemas, and the guards refuse what they refused (open text, same kinds)
    guided = run_script(late, style="guided")
    assert [(r, n, a, res.get("ok"), res.get("refused")) for r, n, a, res in data["calls"]] == \
           [(r, n, a, res.get("ok"), res.get("refused")) for r, n, a, res in guided["calls"]]
    assert ToolBox(late, "mayor", style="open").schemas() == ToolBox(late, "mayor").schemas()
    refused = {res.get("refused") for _, _, _, res in data["calls"]} - {None}
    assert {"take_back", "duplicate", "reserve", "loan", "open_requests", "ask"} <= refused
    # failures keep their reason as a fact
    calls = {(r, n): res for r, n, a, res in data["calls"]}
    taken = calls["police", "build_police_station"]["message"]
    assert taken.endswith("is not an empty land tile - (24,23) holds a town hall"), taken      # no "next free site"
    short = calls["police", "upgrade_facility"]["message"]
    assert short == "costs $7,800 but the police budget only has $6,632", short                 # no "ask the mayor"
    zoned = next(res for r, n, a, res in data["calls"] if (r, n) == ("planning", "zone"))
    assert zoned["ok"] and zoned["message"] == "zoned 42 tiles I (industry has no density setting)", zoned
    loan = calls["mayor", "take_loan"]["error"]
    assert loan.startswith("the treasury holds $27.1k, over the $3,000 minimum: a $5,000 loan costs $25/month") and \
        loan.endswith("To borrow anyway, repeat this exact call with confirm: true."), loan


def test_open_region_states_deals_without_calls():
    """Region mode, open style: a deal opportunity says who is short and who has spare, an offer to the city what it
    is worth, an active deal its supply, and none names the offer_deal / accept_deal / cancel_deal call; region_overview
    leaves the offer_deal arguments out. The guided lines (tests/test_fleet_region.py) name them."""
    from civitas.fleet.fmt import briefing_style
    from civitas.fleet.region import region_lines
    from test_fleet_region import _short_north
    region, n, s = _short_north()
    with briefing_style("open"):
        buy = next(x for x in region_lines(n, "mayor") if x.startswith("- deal opportunity"))
        sell = next(x for x in region_lines(s, "mayor") if x.startswith("- deal opportunity"))
    assert buy == ("- deal opportunity: power: your power has no supply of your own (imports 35 at $1.5/unit, 104 "
                   "unserved); SOUTH has 243 spare (a deal needs no link)")
    assert sell.startswith("- deal opportunity: power: NORTH's power has no supply of its own") and \
        sell.endswith("you have 243 spare; unsold, your spare power earns $0.25/unit (a deal needs no link)")
    nb, sb = ToolBox(n, "mayor", style="open"), ToolBox(s, "mayor", style="open")
    ov = nb.call("region_overview", {})
    assert ov["deal_opportunities"] and "offer_deal" not in ov["deal_opportunities"][0]
    assert "offer_deal" in ToolBox(n, "mayor").call("region_overview", {})["deal_opportunities"][0]
    assert nb.call("offer_deal", {"to": "south", "resource": "power", "amount": 160, "price": 0.88, "months": 3,
                                  "sell": False})["ok"]
    with briefing_style("open"):
        offer = next(x for x in region_lines(s, "mayor") if x.startswith("- deal #1"))
    assert offer == ("- deal #1 offered by north: you sell 160 power to north at $0.88/unit/month (+$141/month) for 3 "
                     "months (open until month 1); you have 243 spare")
    for city_env in (n, s):
        for role in ("mayor", "utilities", "transport"):
            text = briefing(city_env, role, mark=False, style="open")
            assert not _forbidden(text), (role, _forbidden(text)[:3])


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 5, 11])
def test_open_on_more_hand_overs(seed):
    """Seeds 0, 5 and 11 at their hand-over (seed 4 above): the open briefings and script results show no prescription,
    and the guards and actions do what the guided ones do."""
    env = _handover(seed)
    data = run_script(env, style="open")
    bad = [(where, hit) for where, text in _texts(data) for hit in _forbidden(text)]
    assert not bad, bad[:5]
    for d in ROLES[1:]:
        assert re.search(r"- \$[\d.,k]+ in department funds", data["briefings"][d])
        assert "## Ready sites" in data["briefings"][d] and "## Key metrics" in data["briefings"][d]
    assert "- reserve $" in data["briefings"]["mayor"] and "; options: " in data["briefings"]["mayor"]
    guided = run_script(env, style="guided")
    assert [(r, n, res.get("ok"), res.get("refused")) for r, n, _, res in data["calls"]] == \
           [(r, n, res.get("ok"), res.get("refused")) for r, n, _, res in guided["calls"]]


@pytest.mark.slow
def test_prompt_prints_the_open_briefing(capsys):
    """`civitas prompt -c configs/fleet_open_late.toml --role <role> --seed 4` prints the open briefing (no model)."""
    from civitas.cli import main
    for role in ("mayor", "planning", "education", "transport"):
        main(["prompt", "-c", str(OPEN), "--role", role, "--seed", "4"])
        out = capsys.readouterr().out
        brief = out.split("USER (briefing)", 1)[1].split("=" * 30 + " TOOLS", 1)[0]
        assert f"month 96 of 108" in brief and not _forbidden(brief), (role, _forbidden(brief)[:3])
        assert ("- reserve $" in brief) if role == "mayor" else ("in department funds" in brief)


if __name__ == "__main__":
    snapshot(Path(sys.argv[1]) if len(sys.argv) > 1 else SNAPSHOT)
