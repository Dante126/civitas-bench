"""Fleets in region mode: each city's fleet on its own env, cross-city messages, region briefings, tools and reports."""
import json
import re

import pytest

from civitas.config import build_agents, game_config, load_config, scoring_config
from civitas.core.actions import apply_action
from civitas.core.decision import Decision
from civitas.core.env import CivitasEnv, GameConfig
from civitas.core.region import RegionEnv
from civitas.core.roles import ROLES, TURN_ORDER
from civitas.fleet import agent as agent_mod
from civitas.fleet.agent import ToolAgent
from civitas.fleet.briefing import briefing
from civitas.fleet.config import FleetAgentConfig
from civitas.fleet.fmt import money
from civitas.fleet.memory import FleetState
from civitas.fleet.region import MAX_DEALS, connector_prices, link, message_targets, region_lines, region_of
from civitas.fleet.report import fleet_report, rows_from_trajectories
from civitas.fleet.tools import ToolBox, to_schema
from civitas.sim import catalog as C

TITLES = {spec["title"]: r for r, spec in ROLES.items()}


def _region(months=6, **kw):
    return RegionEnv(GameConfig(seed=0, months=months, region=True, events="none", **kw))


def _linked(months=6, **kw):
    region = _region(months, **kw)
    link(region)
    return region, region.envs["north"], region.envs["south"]


def _call(name, args, cid=None):
    return {"id": cid or f"c_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _reply(calls):
    return {"content": "", "reasoning": "", "tool_calls": calls, "finish_reason": "tool_calls" if calls else "stop",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}, "latency_s": 0.0}


class Replies:
    """Stands in for ChatClient: one list of tool calls per model call, then end_turn."""

    def __init__(self, *steps):
        self.steps, self.calls = list(steps), []

    def chat(self, model, messages, **kw):
        self.calls.append([dict(m) for m in messages])
        return _reply(self.steps.pop(0) if self.steps else [_call("end_turn", {"report": "done"})])


def _agent(role, city, region, *steps):
    a = ToolAgent(role, cfg=FleetAgentConfig(model="fake"), city=city)
    a.client = Replies(*steps)
    a.bind(region)
    return a


# ------------------------------------------------------------ an episode
def _script(city: str, role: str, messages: list[dict]) -> list[dict]:
    """The fake model's calls for (city, role) at this step (assistant messages so far)."""
    step = sum(m["role"] == "assistant" for m in messages)
    other = "south" if city == "north" else "north"
    if role == "mayor":
        seq = [[_call("update_plan", {"budgets": {"utilities": 3000}, "goals": [f"power {city}"]})],
               [_call("allocate_budget", {"department": "utilities", "amount": 3000})],
               [_call("send_message", {"to": f"{other}.mayor", "text": f"hello {other}, want power?"}),
                _call("offer_deal", {"to": other, "resource": "power", "amount": 20, "price": 0.5, "months": 6})],
               [_call("end_turn", {"report": "plan set, deal offered"})]]
        return seq[step] if step < len(seq) else []
    if role == "utilities":
        if step == 0:
            return [_call("find_sites", {"purpose": "wind"})]
        if step == 1:
            last = json.loads([m for m in messages if m["role"] == "tool"][-1]["content"])
            s = last["sites"][0]
            return [_call("build_power_plant", {"kind": "wind", "x": s["x"], "y": s["y"]})]
        return [_call("end_turn", {"report": "built wind"})] if step == 2 else []
    if role == "police" and step == 0:
        return [_call("send_message", {"to": other, "text": f"{city} police says hi"})]
    return [_call("end_turn", {"report": "nothing to do"})] if step <= 1 else []


class ScriptedChat:
    """ChatClient for a whole run: the role and city come from the agent's system prompt."""
    requests: list[dict] = []

    def __init__(self, provider=None):
        pass

    def chat(self, model, messages, **kw):
        ScriptedChat.requests.append({"messages": messages, **kw})
        system = messages[0]["content"]
        role = TITLES[re.search(r"You are the (.+?) in a fleet", system).group(1)]
        city = re.search(r"You work for the (\w+) city", system).group(1).lower()
        return _reply(_script(city, role, messages))


def _region_cfg(months=3, cities=("north",)) -> dict:
    cfg = load_config("configs/region.toml")
    cfg["name"] = "fleet_region_fake"
    cfg["game"].update(months=months, events="none")
    cfg["fleet"] = {"mode": "autonomous", "model": "fake", "max_steps": 6, "playbook": "Grow steadily."}
    cfg["agents"] = {f"{c}.{r}": {"kind": "fleet"} for c in cities for r in TURN_ORDER}
    return cfg


def _run(tmp_path, monkeypatch, cfg):
    from civitas.bench import runner
    envs = []
    real = runner.make_env
    monkeypatch.setattr(runner, "make_env", lambda *a, **k: envs.append(real(*a, **k)) or envs[-1])
    monkeypatch.setattr(agent_mod, "ChatClient", ScriptedChat)
    ScriptedChat.requests = []
    s = runner.run_episode(cfg, 0, tmp_path, verbose=False)
    region, = envs
    return s, region, tmp_path / s["run_id"]


def test_a_fleet_city_and_a_scripted_city_play_a_region_episode(tmp_path, monkeypatch):
    s, region, d = _run(tmp_path, monkeypatch, _region_cfg())
    n, south = region.envs["north"], region.envs["south"]
    assert s["months_played"] == 3 and not s["bankrupt"]
    traj = [json.loads(x) for x in (d / "trajectories.jsonl").read_text().splitlines()]
    fleet = [t for t in traj if t["role"].startswith("north.")]
    scripted = [t for t in traj if t["role"].startswith("south.")]
    assert len(fleet) == len(scripted) == 24                       # 8 roles x 3 months each
    assert all("tool_trace" in t and t["finish_reason"] == "end_turn" for t in fleet)
    assert all("tool_trace" not in t and "observation_text" in t for t in scripted)
    # north's fleet state is north's alone; south (scripted) has none
    assert isinstance(n.fleet, FleetState) and south.fleet is None
    assert n.fleet.plan["goals"] == ["power north"] and n.fleet.plan["budgets"] == {"utilities": 3000}
    assert [e["month"] for e in n.fleet.history("metrics", months=None)] == [1, 2, 3]     # end_month synced it
    built = [r for t in fleet if t["role"] == "north.utilities" for r in t["results"]]
    assert [r["action"]["type"] for r in built] == ["build_power_plant"] * 3 and all(r["ok"] for r in built)
    assert len(n.world.facilities(C.WIND)) == 3                     # in north's world; south's plants are its own
    mayor =[t for t in fleet if t["role"] == "north.mayor"][0]
    assert [c["tool"] for c in mayor["tool_trace"]] == ["update_plan", "allocate_budget", "send_message", "offer_deal",
                                                        "end_turn"]
    assert [r["action"]["type"] for r in mayor["results"]] == ["allocate_budget", "offer_deal"]
    assert all(r["ok"] for r in mayor["results"]) and mayor["prompt"][0]["content"].endswith("fleet_prompts.north.mayor)")
    # messages reach the other city's mayor and team; deals reach the region
    council = [(m.month, m.sender, m.to, m.text) for m in region.council.messages]
    assert (0, "north.mayor", "south.mayor", "hello south, want power?") in council
    assert (0, "north.police", "south", "north police says hi") in council
    seen = [m["text"] for m in south.council.visible_to("south.fire", since_month=0)]
    assert "north police says hi" in seen and "hello south, want power?" not in seen
    assert [(x["proposer"], x["seller"], x["buyer"], x["status"]) for x in region.deals] == [("north", "north", "south",
                                                                                               "offered")] * 3
    # the system prompts name the city; the run records them by the region's role names
    meta = json.loads((d / "meta.json").read_text())
    assert set(meta["fleet_prompts"]) == {f"north.{r}" for r in TURN_ORDER}
    assert "You work for the NORTH city" in meta["fleet_prompts"]["north.police"]
    assert "north.mayor=fleet:fake" in s["signature"] and "south.mayor=scripted" in s["signature"]
    # one report, one section per fleet city
    rep = json.loads((d / "fleet_report.json").read_text())
    md = (d / "fleet_report.md").read_text()
    assert rep["region"] == ["north", "south"] and list(rep["cities"]) == ["north"]
    north = rep["cities"]["north"]
    assert list(north["roles"]) == list(TURN_ORDER) and north["plan"]["budgets"] == {"utilities": 3000}
    assert north["scoreboard"]["city"] == s["final_scores"]["north.city"]
    assert north["scoreboard"]["roles"]["police"] == s["final_scores"]["north.police"]
    assert north["roles"]["mayor"]["actions"] == 6 and north["roles"]["utilities"]["tool_calls"] >= 3
    assert rep["scoreboard"]["city"] == s["final_scores"]["city"]
    assert set(rep["scoreboard"]["cities"]) == {"north", "south"}
    assert s["fleet"] == {"report": "fleet_report.md", "hints": len(rep["hints"])}
    assert "## NORTH" in md and "### Scoreboard" in md and "### Roles" in md and "## SOUTH" not in md
    assert "| south |" in md and "fleet agents in north" in md
    assert "collaboration" in north and "### Collaboration" in md      # per city, from frames.jsonl
    rebuilt, _ = fleet_report(region, s, rows_from_trajectories(d / "trajectories.jsonl"), d)
    assert json.loads(json.dumps(rebuilt)) == rep


def test_fleets_in_both_cities_each_play_their_own(tmp_path, monkeypatch):
    s, region, d = _run(tmp_path, monkeypatch, _region_cfg(months=1, cities=("north", "south")))
    n, south = region.envs["north"], region.envs["south"]
    assert n.fleet is not south.fleet
    assert n.fleet.plan["goals"] == ["power north"] and south.fleet.plan["goals"] == ["power south"]
    for env in (n, south):
        assert {e["role"] for e in env.fleet.log if e["kind"] == "report"} == set(TURN_ORDER)
        assert [e["action"]["type"] for e in env.fleet.log if e["kind"] == "action" and e["role"] == "mayor"] == \
            ["allocate_budget", "offer_deal"]
    assert [(x["proposer"], x["seller"], x["buyer"]) for x in region.deals] == [("north", "north", "south"),
                                                                                  ("south", "south", "north")]
    council = [(m.sender, m.to, m.text) for m in region.council.messages]
    assert ("south.mayor", "north.mayor", "hello north, want power?") in council
    assert ("south.police", "north", "south police says hi") in council
    rep = json.loads((d / "fleet_report.json").read_text())
    assert list(rep["cities"]) == ["north", "south"]
    assert rep["cities"]["south"]["plan"]["goals"] == ["power south"]
    assert all(h["city"] in ("north", "south") for h in rep["hints"])
    assert set(json.loads((d / "meta.json").read_text())["fleet_prompts"]) == {f"{c}.{r}" for c in ("north", "south")
                                                                              for r in TURN_ORDER}


# ------------------------------------------------------------ state
def test_each_city_keeps_its_own_fleet_state():
    region = _region()
    agents = {c: _agent("mayor", c, region) for c in region.cities}
    n, s = region.envs["north"], region.envs["south"]
    assert agents["north"].env is n and agents["south"].env is s and region_of(n) is region_of(s) is region
    assert isinstance(n.fleet, FleetState) and n.fleet is not s.fleet
    assert ToolBox(n, "mayor").call("update_plan", {"goals": ["north first"], "budgets": {"police": 2500}})["ok"]
    assert ToolBox(s, "police").call("update_memory", {"lessons": ["south lesson"]})["ok"]
    assert n.fleet.plan["goals"] == ["north first"] and s.fleet.plan_versions == []
    assert s.fleet.memory["police"]["lessons"] == ["south lesson"] and n.fleet.memory["police"]["lessons"] == []
    # each mayor manages its own city's departments and money
    n.world.dept_funds["police"] = 1234
    assert "- police: funds $1,234, target $2,500" in briefing(n, "mayor", mark=False)     # nothing ready yet
    assert "$1,234" not in briefing(s, "mayor", mark=False)
    r = ToolBox(s, "mayor").call("allocate_budget", {"department": "fire", "amount": 1000})
    assert r["ok"] and s.world.dept_funds["fire"] == 1000 and n.world.dept_funds["fire"] == 0
    assert [e["kind"] for e in s.fleet.log] == ["action"] and not n.fleet.log
    region.end_month()
    for env in (n, s):
        metrics = env.fleet.history("metrics")
        assert [e["month"] for e in metrics] == [1] and metrics[0]["population"] == env.world.history[-1]["population"]


def test_a_region_agent_binds_to_its_city():
    region = _region()
    with pytest.raises(ValueError, match="needs its city"):
        ToolAgent("mayor", cfg=FleetAgentConfig()).bind(region)
    with pytest.raises(ValueError, match="needs its city"):
        ToolAgent("mayor", cfg=FleetAgentConfig(), city="east").bind(region)
    a = _agent("police", "south", region)
    assert a.env is region.envs["south"] and a.name == "south.police"
    with pytest.raises(ValueError, match="north city"):
        ToolAgent("police", cfg=FleetAgentConfig(), city="south").bind(region.envs["north"])
    b = ToolAgent("police", cfg=FleetAgentConfig())
    b.bind(region.envs["north"])                          # a linked city env: its region's rules still apply
    assert b.city == "north" and "You work for the NORTH city" in b.system
    single = ToolAgent("police", cfg=FleetAgentConfig())
    single.bind(CivitasEnv(GameConfig(seed=0, months=3)))
    assert "# Region" not in single.system and single.name == "police"


# ------------------------------------------------------------ messages
def test_a_message_to_another_city_arrives():
    region = _region()
    police = _agent("police", "north", region,
                    [_call("send_message", {"to": "south.mayor", "text": "a road link, please"}, "a"),
                     _call("send_message", {"to": "south", "text": "hello south"}, "b"),
                     _call("send_message", {"to": "mayor", "text": "for my own mayor"}, "c")])
    d = police.act(region.observe("north.police"))
    assert [c["result"]["ok"] for c in d.meta["tool_trace"][:3]] == [True] * 3
    region.apply("north.police", d)
    south_mayor = _agent("mayor", "south", region)
    text = briefing(south_mayor.env, "mayor", mark=False)
    assert "- m0 north.police -> you: a road link, please" in text
    assert "- m0 north.police -> south: hello south" in text and "for my own mayor" not in text
    north_mayor = _agent("mayor", "north", region)
    text = briefing(north_mayor.env, "mayor")
    assert "- m0 police -> you: for my own mayor" in text and "a road link" not in text
    assert "no new council messages" in briefing(north_mayor.env, "mayor")          # marked by the council name
    # a scripted (non-fleet) mayor reads the same board
    assert "a road link, please" in [m["text"] for m in region.observe("south.mayor")["council"]]
    # the other city's reply comes back the same way
    reply = _agent("mayor", "south", region, [_call("send_message", {"to": "north.police", "text": "on it"})])
    region.apply("south.mayor", reply.act(region.observe("south.mayor")))
    assert "- m0 south.mayor -> you: on it" in briefing(police.env, "police", mark=False)
    got = ToolBox(police.env, "police").call("get_messages", {})["messages"]
    assert {"month": 0, "sender": "south.mayor", "to": "police", "text": "on it"} in got


def test_message_addresses_in_a_region():
    region, n, s = _linked()
    assert message_targets(n, "police") == ["mayor", "planning", "utilities", "transport", "fire", "health",
                                            "education", "all", "north", "south",
                                            *(f"south.{r}" for r in TURN_ORDER)]
    box = ToolBox(n, "police")
    assert box.tools["send_message"].properties["to"] == to_schema(message_targets(n, "police"))
    assert not box.call("send_message", {"to": "north.police", "text": "x"})["ok"]          # own name: not a target
    single = ToolBox(CivitasEnv(GameConfig(seed=0, months=3)), "police")
    assert single.tools["send_message"].properties["to"] == to_schema([r for r in TURN_ORDER if r != "police"] + ["all"])
    assert "region_overview" not in single.tools and "offer_deal" not in ToolBox(single.env, "mayor").tools


def test_one_message_to_several_addresses_in_a_region():
    """send_message's comma list in a region: own roles, another city's team and agent, each its own council message
    under its council name (RegionEnv.apply prefixes the own city's roles)."""
    region, n, s = _linked()
    box = ToolBox(n, "mayor")
    res = box.call("send_message", {"to": "police, fire,south,south.mayor", "text": "Noted: next quarter"})
    assert res["ok"] and res["message"] == ("message to police, fire, south and south.mayor (one message each) will be "
                                            "posted at the end of your turn")
    region.apply("north.mayor", Decision(executed=box.executed, messages=box.messages))
    assert [(m["sender"], m["to"]) for m in region.council.posted_in(0)] == [
        ("north.mayor", "north.police"), ("north.mayor", "north.fire"), ("north.mayor", "south"),
        ("north.mayor", "south.mayor")]
    assert "Noted" in region.council.visible_to("south.planning", 0)[0]["text"]          # the south team's copy
    assert not box.call("send_message", {"to": "police,west", "text": "x"})["ok"]        # every name must be a target


# ------------------------------------------------------------ deals
def test_mayors_make_deals_with_their_tools():
    region, n, s = _linked()
    nb, sb = ToolBox(n, "mayor"), ToolBox(s, "mayor")
    offer = {"to": "south", "resource": "power", "amount": 50, "price": 1, "months": 3}
    r = nb.call("offer_deal", offer)
    assert r == {"ok": True, "message": "proposed deal #1; the south mayor must accept it", "cost": 0.0,
                 "budget_left": n.world.treasury}
    assert region.council.messages[-1].sender == "north.mayor" and region.council.messages[-1].to == "south"
    assert not nb.call("accept_deal", {"deal": 1})["ok"]                          # not its own offer
    assert nb.call("accept_deal", {"deal": 1})["refused"] == "repeat"
    assert "must be one of" in nb.call("offer_deal", {**offer, "resource": "gold"})["error"]
    assert "the police role cannot use 'offer_deal'" in ToolBox(n, "police").call("offer_deal", offer)["error"]
    assert ("- deal #1 offered by north: you buy 50 power from north at $1/unit/month (-$50/month) for 3 months: "
            "accept_deal to take it") in briefing(s, "mayor", mark=False)
    assert ("- deal #1 offered by you: you sell 50 power to south at $1/unit/month (+$50/month) for 3 months, waiting "
            "for south.mayor to accept") in briefing(n, "mayor", mark=False)
    assert ("- deal #1 offered by north: the city buys 50 power from north (+50 power supply) for 3 months, if the "
            "mayors agree") in region_lines(s, "utilities")
    assert sb.call("accept_deal", {"deal": 1})["ok"]
    assert "- deal #1 active: you sell 50 power to south at $1/unit/month (+$50/month), 3 months left" \
        in briefing(n, "mayor", mark=False)
    assert "- deal #1 active: the city sells 50 power to south (-50 power supply), 3 months left" in \
        region_lines(n, "utilities")
    # the executed deals are logged like any action: in the city's history and the region's turn log
    assert [e["action"]["type"] for e in n.fleet.log if e["kind"] == "action"] == ["offer_deal", "accept_deal"]
    res = region.apply("north.mayor", Decision(executed=nb.executed))
    assert [x["ok"] for x in res] == [True, False] and n.invalid_counts["mayor"] == 1
    region.apply("south.mayor", Decision(executed=sb.executed))
    region.end_month()
    assert s.world.external["power"] == 50 and n.world.external["power"] == -50
    assert sb.call("cancel_deal", {"deal": 1})["ok"] and region.deals[0]["status"] == "ended"


def test_deals_count_towards_the_action_cap():
    region, n, s = _linked()
    n.cfg.max_actions_per_turn = 1
    box = ToolBox(n, "mayor")
    assert box.call("offer_deal", {"to": "south", "resource": "water", "amount": 10, "price": 0.2})["ok"]
    assert "action limit" in box.call("cancel_deal", {"deal": 1})["error"]


# ------------------------------------------------------------ briefings
def test_region_blocks_in_the_briefings():
    region, n, s = _linked()
    for role in TURN_ORDER:
        text = briefing(n, role, mark=False)
        assert text.startswith(f"# {'Mayor' if role == 'mayor' else ROLES[role]['title']} of NORTH: month 0")
        assert ("## Region (NORTH and 1 other city; details: region_overview)" in text) == (role in (
            "mayor", "transport", "utilities")), role
    mayor = region_lines(n, "mayor")
    assert mayor[2] == "- SOUTH (your south edge, downstream): pop 0; unemployed 0, open jobs 0; links: none"
    assert "- river: leaves for south 0% polluted" in mayor
    assert "- river: arrives from north 0% polluted" in region_lines(s, "utilities")
    single = CivitasEnv(GameConfig(seed=0, months=3))
    FleetState.of(single)
    assert region_lines(single, "mayor") == [] and "## Region" not in briefing(single, "mayor", mark=False)
    assert briefing(single, "police", mark=False).startswith("# Police Department: month 0")


def _roads_to_the_edge(region):
    """North and south each with a road 2 tiles from their facing edges, as in tests/test_region.py."""
    for env, y_road in ((region.envs["north"], 45), (region.envs["south"], 2)):
        env.world.struct[y_road, 5:10] = C.ROAD
        env.world.refresh()
        env.world.dept_funds["transport"] = 20000


def test_transport_sees_the_connector_site_and_its_price():
    region, n, s = _linked()
    _roads_to_the_edge(region)
    cs = connector_prices(n.world)["south"]
    site = cs["site"]
    line = next(x for x in region_lines(n, "transport") if x.startswith("- SOUTH"))
    assert line == (f"- SOUTH (south edge): links: none; connector site ({site['x']},{site['y']}), "
                    f"{site['feeder_tiles']} feeder road tiles: highway {money(cs['price']['highway'])}, rail "
                    f"{money(cs['price']['rail'])}")
    assert cs["price"] == {k: v["cost"] + 10 * site["feeder_tiles"] for k, v in C.CONNECTOR.items()}
    r = apply_action(n.world, "transport", {"type": "build_connector", "kind": "highway", "to": "south", **site})
    assert r.ok and r.cost == cs["price"]["highway"]                       # the price is what the action charges
    assert "you built highway: waiting for them" in next(x for x in region_lines(n, "transport") if "SOUTH" in x)
    assert "they built highway: build your side to open it" in next(
        x for x in region_lines(s, "transport") if "NORTH" in x)
    assert "(they built highway: build your side); a highway connector costs transport $" in next(
        x for x in region_lines(s, "mayor") if "NORTH" in x)
    ssite = connector_prices(s.world)["north"]["site"]
    assert apply_action(s.world, "transport", {"type": "build_connector", "kind": "highway", "to": "north",
                                                **ssite}).ok
    region.end_month()
    assert "links: highway;" in next(x for x in region_lines(n, "transport") if "SOUTH" in x)
    line = next(x for x in region_lines(n, "mayor") if "SOUTH" in x)
    assert "links: highway | last month: goods" in line and "connector costs" not in line
    ov = ToolBox(n, "transport").call("region_overview", {})
    assert ov["ok"] and ov["links"][0]["kind"] == "highway" and ov["connectors"][0]["to"] == "south"
    assert ov["flows_last_month"] and ov["cities"][0]["city"] == "south"


def test_the_region_block_stays_bounded():
    cities = ["north", "south", "east", "west", "southeast", "southwest"]
    region = RegionEnv(GameConfig(seed=0, months=3, region=True, events="none", cities=cities))
    link(region)
    n = region.envs["north"]
    for i, c in enumerate(cities[1:] * 2):
        region.deals.append({"id": i + 1, "seller": "north", "buyer": c, "resource": "water", "amount": 1999.0,
                             "price": 9.99, "months_left": 120, "status": "active" if i % 2 else "offered",
                             "proposer": c})
    for w in (e.world for e in region.envs.values()):             # the longest city lines: every field shown
        w.trade.update(goods_made=12345.0, goods_needed=0.0, raw_made=0.0, raw_needed=23456.0)
        w.stats.update(unemployed=12345.0, unfilled_jobs=23456.0, population=123456.0)
        w.history = [{}]
        w.specializations = lambda: {k: 1.0 for k in ("mining & farming", "high tech", "gambling", "tourism", "trade",
                                                      "education")}
    others = cities[1:]
    region.links = [{"a": "north", "b": c, "kind": k} for c in others for k in ("highway", "rail", "boat", "air")]
    region.flows = [{"a": "north", "b": c, "goods": 12345.6, "raw": 12345.6, "commuters": 1234.5, "tourists": 1234.5}
                    for c in others]
    block = region_lines(n, "mayor")
    deals = [x for x in block if x.startswith("- deal #")]
    assert len(deals) == MAX_DEALS and block[-1] == f"- (+{2 * 5 - MAX_DEALS} more deals: region_overview)"
    assert "links: air, boat, highway, rail | last month: goods 12.3k" in block[2]
    assert len("\n".join(block)) <= 2800, len("\n".join(block))
    assert all(len(x) <= 330 for x in block), max(map(len, block))
    for w in (e.world for e in region.envs.values()):
        w.history = []
    text = briefing(n, "mayor", mark=False)
    assert "## Region (NORTH and 5 other cities" in text
    assert "SOUTHWEST (far: boat or air only)" in text and "EAST (your east edge)" in text


def test_region_rules_in_the_system_prompt():
    region = _region()
    mayor, police = _agent("mayor", "south", region), _agent("police", "south", region)
    rules = mayor.system[mayor.system.index("# Region"):]
    assert "You work for the SOUTH city, one of 2 cities in this region; the others: NORTH (your north edge)" in rules
    assert "it arrives from NORTH" in rules and "nothing flows on from you" in rules
    assert "offer_deal" in rules and "offer_deal" not in police.system
    assert mayor.system.index("# Game rules") < mayor.system.index("# Region")
    playbook = ToolAgent("mayor", cfg=FleetAgentConfig(playbook="Sell power."), city="south")
    playbook.bind(region)
    assert playbook.system.index("# Region") < playbook.system.index("# Coach's playbook")


# ------------------------------------------------------------ config and CLI
def test_the_region_smoke_config():
    cfg, smoke = load_config("configs/fleet_region_smoke.toml"), load_config("configs/fleet_smoke.toml")
    game = game_config(cfg, 0)
    assert game.region and game.cities == ["north", "south"] and (game.months, game.act_every) == (12, 3)
    assert cfg["fleet"] == smoke["fleet"]                                           # the same fleet as fleet_smoke
    assert {k: v for k, v in cfg["agents"].items() if "." not in k} == smoke["agents"]
    region = RegionEnv(game, scoring_config(cfg, game))
    agents = build_agents(cfg, region.roles, game, region.scoring)                 # no network needed
    for role, a in agents.items():
        city, base = role.split(".")
        assert isinstance(a, ToolAgent) == (city == "north"), role
        if city == "north":
            assert (a.city, a.role, a.cfg.model) == ("north", base, "trinity-large-thinking") and a.cfg.playbook


def test_prompt_command_for_a_region_fleet_role(capsys):
    from civitas.cli import main
    main(["prompt", "-c", "configs/fleet_region_smoke.toml", "--role", "north.mayor"])
    out = capsys.readouterr().out
    system, rest = out.split("USER (briefing)")
    briefing_text, tools = rest.split("TOOLS")
    assert "# Region" in system and "You work for the NORTH city" in system and "## Mayor" in system
    assert briefing_text.lstrip("= \n").startswith("# Mayor of NORTH: month 0 of 12")
    assert "## Region (NORTH and 1 other city" in briefing_text and "- SOUTH (your south edge" in briefing_text
    assert {"offer_deal", "accept_deal", "cancel_deal", "region_overview", "allocate_budget"} <= set(
        tools.strip("= \n").split(", "))
    main(["prompt", "-c", "configs/fleet_region_smoke.toml", "--role", "south.mayor"])       # scripted: LLM prompt
    out = capsys.readouterr().out
    assert "REGION MODE" in out and "SOUTH" in out
    with pytest.raises(SystemExit, match="north.mayor"):
        main(["prompt", "-c", "configs/fleet_region_smoke.toml", "--role", "mayor"])
