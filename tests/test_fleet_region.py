"""Fleets in region mode: each city's fleet on its own env, cross-city messages, region briefings, tools and reports."""
import json
import math
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
    # south's scripted mayor (agents/scripted.py:_deal_actions) turns down the first offer (north has no power of its
    # own to sell yet: its utilities build after the mayors' turns, RegionEnv.deal_check's "seller_short") and the
    # later ones (south's own plant gives it room)
    assert [(x["proposer"], x["seller"], x["buyer"], x["status"]) for x in region.deals] == [("north", "north", "south",
                                                                                               "ended")] * 3
    assert [(t["month"], a["type"], a["deal"]) for t in scripted if t["role"] == "south.mayor" for a in t["actions"]
            if a["type"].endswith("_deal")] == [(0, "cancel_deal", 1), (1, "cancel_deal", 2), (2, "cancel_deal", 3)]
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
    assert r == {"ok": True, "message": "proposed deal #1; the south mayor must accept it by month 1", "cost": 0.0,
                 "budget_left": n.world.treasury}
    assert region.council.messages[-1].sender == "north.mayor" and region.council.messages[-1].to == "south"
    assert not nb.call("accept_deal", {"deal": 1})["ok"]                          # not its own offer
    assert nb.call("accept_deal", {"deal": 1})["refused"] == "repeat"
    assert "must be one of" in nb.call("offer_deal", {**offer, "resource": "gold"})["error"]
    assert "the police role cannot use 'offer_deal'" in ToolBox(n, "police").call("offer_deal", offer)["error"]
    assert ("- deal #1 offered by north: you buy 50 power from north at $1/unit/month (-$50/month) for 3 months (open "
            "until month 1); no power supply of your own, imports cost $1.5/unit, but you need only 7: not worth it, "
            'leave it or cancel_deal {"deal": 1}') in briefing(s, "mayor", mark=False)   # a fresh map's power: 6 of demand
    assert ("- deal #1 offered by you: you sell 50 power to south at $1/unit/month (+$50/month) for 3 months, waiting "
            "for south.mayor to accept (open until month 1)") in briefing(n, "mayor", mark=False)
    assert ("- deal #1 offered by north: the city buys 50 power from north (+50 power supply) for 3 months, if the "
            "mayors agree (open until month 1)") in region_lines(s, "utilities")
    _coal(n)                                        # north needs power of its own to sell (RegionEnv._accept_refusal)
    assert sb.call("accept_deal", {"deal": 1})["ok"]
    assert ("- deal #1 active: you sell 50 power to south at $1/unit/month (+$50/month), 3 months left (through month "
            "2)") in briefing(n, "mayor", mark=False)
    assert ("- deal #1 active: you buy 50 power from north at $1/unit/month (-$50/month), 3 months left (through month "
            "2)") in region_lines(s, "mayor")
    assert "- deal #1 active: the city sells 50 power to south (-50 power supply), 3 months left" in \
        region_lines(n, "utilities")
    # the executed deals are logged like any action: in the city's history and the region's turn log
    assert [e["action"]["type"] for e in n.fleet.log if e["kind"] == "action" and e["role"] == "mayor"] == \
        ["offer_deal", "accept_deal"]
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


# ------------------------------------------------------------ deal opportunities
def _coal(env):
    """A coal plant (250 power) in env's city, on the first site utilities' find_sites names."""
    env.world.dept_funds["utilities"] = env.world.dept_funds.get("utilities", 0) + 10000
    box = ToolBox(env, "utilities")
    site = box.call("find_sites", {"purpose": "coal_or_gas"})["sites"][0]
    assert box.call("build_power_plant", {"kind": "coal", "x": site["x"], "y": site["y"]})["ok"]


def _short_north(months=24, cities=None):
    """North with 37 level-3 homes along its road and no plant (power and water imported, the rest unserved); south
    (and every other city of `cities`) with a coal plant and nobody living there (power: 243 spare beyond DEAL_BUSY of
    its 250)."""
    import numpy as np
    region = _region(months, **({"cities": list(cities)} if cities else {}))
    link(region)
    n, s = region.envs["north"], region.envs["south"]
    w = n.world
    for y, x in zip(*np.nonzero(w.struct == C.ROAD)):
        for yy in (y - 1, y + 1):
            if w.struct[yy, x] == C.NONE and w.terrain[yy, x] != C.WATER:
                w.struct[yy, x], w.level[yy, x] = C.RES, 3
    w.refresh()
    for c in region.cities[1:]:
        _coal(region.envs[c])
    return region, n, s


def _call_in(line: str, tool: str) -> dict:
    """The JSON arguments of `tool {...}` named in a briefing line."""
    return json.loads(re.search(tool + r" (\{.*?\})", line).group(1))


def test_a_deal_opportunity_when_one_city_is_short_and_the_other_has_spare():
    from civitas.core.region import DEAL_BUSY
    from civitas.fleet.tools import BANK_BUSY
    assert DEAL_BUSY == BANK_BUSY                                  # the limit utilities' supply lines use
    region, n, s = _short_north()
    st = {c: region.supply_state(c)["power"] for c in ("north", "south")}
    assert st["north"]["own"] == 0 and st["north"]["imported"] > 0 and st["north"]["unserved"] > 0
    assert st["south"]["own"] == 250 and region.spare(st["south"]) == pytest.approx(250 - 6 / DEAL_BUSY, abs=0.1)
    price = round((C.POWER_SELL_PRICE + n.world.service_import_price("power")) / 2, 2)     # 0.88
    need = st["north"]["need"]                                     # 139.2 demand / 0.9: 155 -> 160
    buy = next(x for x in region_lines(n, "mayor") if x.startswith("- deal opportunity"))
    assert buy == (f"- deal opportunity: your power has no supply of your own (imports 35 at $1.5/unit, 104 unserved); "
                   f"SOUTH has 243 spare: offer_deal {{\"to\": \"south\", \"resource\": \"power\", "
                   f"\"amount\": 160, \"price\": {price:g}, \"months\": 3, \"sell\": false}} buys 160 at "
                   f"${price:g}/unit/month (-$141/month) for 3 months (no link needed)")
    assert math.ceil(need / 10) * 10 == 160
    sell = next(x for x in region_lines(s, "mayor") if x.startswith("- deal opportunity"))
    assert sell.startswith("- deal opportunity: NORTH's power has no supply of its own (imports 35 at") and \
        sell.endswith("sells 160 at $0.88/unit/month (+$141/month) for 3 months; unsold, spare power earns $0.25/unit "
                      "(no link needed)")
    assert _call_in(sell, "offer_deal") == {**_call_in(buy, "offer_deal"), "to": "north", "sell": True}
    assert not any("water" in x for x in region_lines(n, "mayor") if "opportunity" in x)     # south has no water
    assert "Deal opportunity: your power has no supply of your own" in region.observe("north.mayor")["text"]
    assert '{"type": "offer_deal", "to": "south", "resource": "power", "amount": 160' in \
        region.observe("north.mayor")["text"]
    # the call as named works; the offer then shows as an accept line on the other side and the opportunity goes
    nb, sb = ToolBox(n, "mayor"), ToolBox(s, "mayor")
    ov = nb.call("region_overview", {})
    assert ov["deal_opportunities"][0]["offer_deal"] == _call_in(buy, "offer_deal") and ov["supply"]["power"]["own"] == 0
    assert nb.call("offer_deal", _call_in(buy, "offer_deal"))["ok"]
    assert not any("opportunity" in x for x in region_lines(n, "mayor") + region_lines(s, "mayor"))
    accept = next(x for x in region_lines(s, "mayor") if x.startswith("- deal #1"))
    assert accept == ("- deal #1 offered by north: you sell 160 power to north at $0.88/unit/month (+$141/month) for "
                      '3 months (open until month 1); you have 243 spare: accept_deal {"deal": 1} takes it')
    assert sb.call("accept_deal", _call_in(accept, "accept_deal"))["ok"]
    region.apply("north.mayor", Decision(executed=nb.executed))
    region.apply("south.mayor", Decision(executed=sb.executed))
    t0 = {c: e.world.treasury for c, e in region.envs.items()}
    region.end_month()
    assert n.world.external["power"] == 160 and s.world.external["power"] == -160
    assert n.world.history[-1]["expense_breakdown"]["neighbour_deals"] == pytest.approx(160 * 0.88) == \
        s.world.history[-1]["income_breakdown"]["neighbour_deals"]
    after = region.supply_state("north")["power"]
    assert after["own"] == 160 and after["imported"] < st["north"]["imported"]
    assert "- deal #1 active: you buy 160 power from south at $0.88/unit/month (-$141/month), 2 months left (through " \
           "month 2)" in region_lines(n, "mayor")
    assert t0["north"] != n.world.treasury


def test_offers_to_a_city_say_whether_they_are_worth_taking():
    region, n, s = _short_north()
    offer = lambda box, **kw: box.call("offer_deal", {"to": "north", "resource": "power", "months": 3, **kw})
    sb = ToolBox(s, "mayor")
    assert offer(sb, amount=160, price=2.0)["ok"] and offer(sb, amount=1000, price=0.5, months=4)["ok"]
    lines = [x for x in region_lines(n, "mayor") if x.startswith("- deal #")]
    assert lines[0].endswith('dearer than imports ($1.5/unit): not worth it, leave it or cancel_deal {"deal": 1}')
    assert lines[1].endswith(", but you need only 155: not worth it, leave it or cancel_deal {\"deal\": 2}")
    nb = ToolBox(n, "mayor")
    assert nb.call("offer_deal", {"to": "south", "resource": "power", "amount": 500, "price": 1, "sell": False})["ok"]
    line = next(x for x in region_lines(s, "mayor") if x.startswith("- deal #3"))
    assert line.endswith('; you have only 0 spare after deal #1, #2: not worth it, leave it or cancel_deal {"deal": 3}')
    # the scripted mayor follows the same checks
    from civitas.agents.scripted import _deal_actions
    assert _deal_actions(region.observe("north.mayor")) == [{"type": "cancel_deal", "deal": 1},
                                                            {"type": "cancel_deal", "deal": 2}]
    assert _deal_actions(region.observe("south.mayor")) == [{"type": "cancel_deal", "deal": 3}]


def test_the_scripted_region_mayor_makes_the_deal():
    from civitas.agents.scripted import ScriptedMayor
    region, n, s = _short_north()
    north, south = ScriptedMayor("mayor"), ScriptedMayor("mayor")
    d = north.act(region.observe("north.mayor"))
    offer = [a for a in d.actions if a["type"] == "offer_deal"]
    assert offer == [{"type": "offer_deal", "to": "south", "resource": "power", "amount": 160, "price": 0.88,
                      "months": 3, "sell": False}]
    assert region.apply("north.mayor", d)[-1]["ok"]
    d = south.act(region.observe("south.mayor"))
    assert [a for a in d.actions if a["type"].endswith("_deal")] == [{"type": "accept_deal", "deal": 1}]
    assert region.apply("south.mayor", d)[-1]["message"] == "deal #1 is active"
    region.end_month()
    assert n.world.external["power"] == 160
    # once north has room without it (its own coal plant), its mayor ends the deal; the city's action cap doesn't hold
    # deals back
    n.world.dept_funds["utilities"] = 10000
    box = ToolBox(n, "utilities")
    site = box.call("find_sites", {"purpose": "coal_or_gas"})["sites"][0]
    assert box.call("build_power_plant", {"kind": "coal", "x": site["x"], "y": site["y"]})["ok"]
    assert north.act(region.observe("north.mayor")).actions[-1] == {"type": "cancel_deal", "deal": 1}
    obs = region.observe("north.mayor")
    obs["max_actions"] = 0
    assert north.act(obs).actions == [{"type": "cancel_deal", "deal": 1}]
    single = CivitasEnv(GameConfig(seed=0, months=3))                # a single city: no deal actions
    assert not any(a["type"].endswith("_deal") for a in ScriptedMayor("mayor").act(single.observe("mayor")).actions)


def test_two_sellers_cannot_both_cover_one_need():
    """Review: with a third city two sellers could each offer the buyer its whole need, both offers read "worth it"
    against the same need (supply moves only at the next tick), and the buyer took both and paid twice."""
    from civitas.agents.scripted import _deal_actions
    region, n, s = _short_north(cities=("north", "south", "east"))
    e = region.envs["east"]
    sb, eb, nb = ToolBox(s, "mayor"), ToolBox(e, "mayor"), ToolBox(n, "mayor")
    sell = next(x for x in region_lines(s, "mayor") if x.startswith("- deal opportunity"))
    assert sb.call("offer_deal", _call_in(sell, "offer_deal"))["ok"]              # south offers north its whole need
    # east's opportunity counts south's open offer: north's need is covered, so it names none
    assert region.deal_opportunities("east") == [] and \
        not any(x.startswith("- deal opportunity") for x in region_lines(e, "mayor"))
    # offered anyway, each offer alone is worth taking; the first accepted, the second is covered
    assert eb.call("offer_deal", {"to": "north", "resource": "power", "amount": 160, "price": 0.88, "months": 3})["ok"]
    assert [region.deal_check(d, "north")["worth"] for d in region.deals] == [True, True]
    obs = region.observe("north.mayor")
    assert _deal_actions(obs) == [{"type": "accept_deal", "deal": 1}, {"type": "cancel_deal", "deal": 2}]
    assert obs["region"]["deal_opportunities"] == []                              # the offers it takes cover it
    assert nb.call("accept_deal", {"deal": 1})["ok"]
    check = region.deal_check(region.deals[1], "north")
    assert (check["worth"], check["reason"], check["covered_by"]) == (False, "covered", [1])
    assert any(x.startswith("- deal #2 offered by east") and ", already covered by deal #1: not worth it" in x
               for x in region_lines(n, "mayor"))
    r = nb.call("accept_deal", {"deal": 2})                                      # the region refuses it too
    assert not r["ok"] and r["message"].startswith("deal #1 accepted this month already covers your power need")


def test_a_deal_the_buyer_cannot_afford_is_not_worth_it():
    """Review: "worth it" checked only the price against imports; imports cover at most a quarter of demand, so a
    deal topping supply up to its limit could cost several times the import bill, with no money check."""
    from civitas.agents.scripted import _deal_actions
    from civitas.core.money import reserve
    region, n, s = _short_north()
    w = n.world
    w.treasury = reserve(w) + 30.0                  # $30 above the reserve, no surplus on record (month 0)
    st = region.supply_state("north")["power"]
    # north affords a deal that only replaces its imports, plus $30 over 3 months: (30 / 3 + 35 x 1.5) / 0.88 -> 70
    opp, = region.deal_opportunities("north")
    assert (opp["amount"], opp["capped"]) == (math.floor((10 + st["imported"] * 1.5) / 0.88 / 10) * 10, "afford")
    line = next(x for x in region_lines(n, "mayor") if x.startswith("- deal opportunity"))
    assert "for 3 months, as much as you can afford (no link needed)" in line
    assert ToolBox(s, "mayor").call("offer_deal", {"to": "north", "resource": "power", "amount": 160, "price": 0.88,
                                                   "months": 3})["ok"]
    check = region.deal_check(region.deals[0], "north")
    assert (check["worth"], check["reason"]) == (False, "afford") and check["budget"] == 30
    assert check["net"] == pytest.approx(160 * 0.88 - st["imported"] * 1.5)
    line = next(x for x in region_lines(n, "mayor") if x.startswith("- deal #1"))
    assert (f"but you can't afford it: {money(check['net'])}/month more than imports for 3 months is over the $30 you "
            "have for it (above the reserve, with last month's surplus)") in line
    assert line.endswith('leave it or cancel_deal {"deal": 1}') and "worth" not in line
    assert _deal_actions(region.observe("north.mayor")) == [{"type": "cancel_deal", "deal": 1}]
    w.treasury += 10000                                                          # now it can
    assert region.deal_check(region.deals[0], "north")["worth"]


def test_a_deal_opportunity_stays_within_offer_deals_limit(monkeypatch):
    """Review: amounts had no cap, but offer_deal refuses more than 2,000 units, so a big city's named call failed."""
    from civitas.core.region import DEAL_AMOUNT_MAX
    region, n, s = _linked()
    n.world.treasury = 1e7
    base = {"imported": 0.0, "unserved": 0.0, "use_pct": 100, "import_price": 1.5, "floor": 0.25, "demand": 9000.0}
    states = {"north": {"power": {**base, "own": 9000.0, "need": 1000.0}, "water": {**base, "own": 0.0, "need": 0.0}},
              "south": {"power": {**base, "own": 20000.0, "demand": 1000.0, "need": -18888.9},
                        "water": {**base, "own": 0.0, "need": 0.0}}}
    states["north"]["power"]["need"] = 5000.0
    monkeypatch.setattr(region, "supply_state", lambda c: states[c])
    opp, = region.deal_opportunities("north")
    assert (opp["amount"], opp["capped"]) == (DEAL_AMOUNT_MAX, "max")
    assert "offer_deal's 2,000 limit" in next(x for x in region_lines(n, "mayor") if "deal opportunity" in x)
    assert ToolBox(n, "mayor").call("offer_deal", region.offer_args(opp, "north"))["ok"]


def test_an_unanswered_offer_expires(tmp_path):
    """Review: an ignored offer stayed open for the rest of the game, blocking that pair and resource, and accepted
    months later could make the seller sell capacity it no longer had. It now lapses after offer_window months
    (two mayor turns) as status "expired", shown in the Region block and the frames' deals."""
    from civitas.bench.collab import deal_history, region_collab
    region, n, s = _short_north()
    assert region.offer_window() == 2                                            # act_every 1
    nb, sb = ToolBox(n, "mayor"), ToolBox(s, "mayor")
    buy = next(x for x in region_lines(n, "mayor") if x.startswith("- deal opportunity"))
    assert nb.call("offer_deal", _call_in(buy, "offer_deal"))["message"].endswith("must accept it by month 1")
    frames = [region.frame()]
    region.end_month()
    assert region.deals[0]["status"] == "offered"                                # still open in month 1
    region.end_month()
    assert region.deals[0]["status"] == "expired" and region.month == 2
    frames.append(region.frame())
    assert frames[-1]["deals"][0]["status"] == "expired"
    assert any(x.startswith("- deal #1 expired: you sell 160 power to north") and "not accepted by month 1" in x
               for x in region_lines(s, "mayor"))
    assert "#1 expired: you are buying 160 power with south" in region.observe("north.mayor")["text"]
    r = ToolBox(s, "mayor").call("accept_deal", {"deal": 1})
    assert not r["ok"] and r["message"].startswith("deal #1 expired after month 1")
    # the pair is free again: the opportunity is back
    assert any(x.startswith("- deal opportunity") for x in region_lines(n, "mayor"))
    # replays: the deal ended "expired" without being accepted, its offer unanswered
    d, = deal_history(frames, region.cities)
    assert (d["end"], d["accepted"], d["ended"]) == ("expired", None, 1)
    rc = region_collab(frames, region.roles, 1, region.cities)
    assert any(ev.get("what") == "expired" and ev["text"] == "deal #1 offer expired, not accepted"
               for ev in rc["events"])
    assert next(t for t in rc["threads"] if t["kind"] == "deal")["status"] == "unanswered"
    # a month later it leaves the Region block
    region.end_month()
    assert not any("deal #1" in x for x in region_lines(s, "mayor"))


def test_supply_states_are_computed_once_per_view(monkeypatch):
    """Review: supply_state ran for every role's observation (only mayors use the others') and twice in the mayor's
    Region block. DEAL_BUSY and BANK_BUSY are one constant (catalog.SUPPLY_BUSY)."""
    from civitas.core.region import DEAL_BUSY
    from civitas.fleet.tools import BANK_BUSY
    assert DEAL_BUSY == BANK_BUSY == C.SUPPLY_BUSY
    region, n, s = _short_north()
    calls = []
    real = region.supply_state
    monkeypatch.setattr(region, "supply_state", lambda c: calls.append(c) or real(c))
    region.observe("north.police")
    assert calls == ["north"]
    calls.clear()
    region_lines(n, "mayor")
    assert sorted(calls) == ["north", "south"]


def test_the_city_block_shows_imported_supply_apart():
    """Review: the mayor's City line counted service imports in ("power 1,081 / 1,081"), hiding the shortage."""
    region, n, s = _short_north()
    m = n.world.metrics()
    imp = m["service_imports"]["power"]
    assert imp > 0
    from civitas.fleet.fmt import num
    assert (f"- supply / demand (coverage): power {num(round(m['power_supply'] - imp, 1))} own + {num(imp)} imported "
            f"/ {num(m['power_demand'])}") in briefing(n, "mayor", mark=False)
    assert "sewage 0 / " in briefing(n, "mayor", mark=False)                    # no imports: as before
    obs = region.observe("north.mayor")
    assert obs["city"]["power"]["supply"] == m["power_supply"] and obs["city"]["power"]["imported"] == round(imp, 1)
    assert f"Power: supply {round(m['power_supply'] - imp, 1)} own + {round(imp, 1)} imported, demand" in obs["text"]


def test_spare_without_a_short_neighbour_and_the_longest_opportunity_line():
    from civitas.core.region import opportunity_text
    region, n, s = _linked()
    s.world.dept_funds["utilities"] = 10000
    box = ToolBox(s, "utilities")
    site = box.call("find_sites", {"purpose": "coal_or_gas"})["sites"][0]
    assert box.call("build_power_plant", {"kind": "coal", "x": site["x"], "y": site["y"]})["ok"]
    assert region_lines(s, "mayor")[-1] == ("- spare: power 243 beyond 90% use; no neighbour is short of it now (a deal "
                                            "opportunity shows here when one is)")
    assert not any("spare" in x for x in region_lines(n, "mayor"))
    # the longest line: long city names, 4-5 digit amounts, a price to the cent (deal_months is at most 6 at act_every 3)
    o = {"resource": "power", "buyer": "southwest", "seller": "southeast", "amount": 1990, "price": 1.99, "months": 6,
         "buyer_use_pct": 999, "buyer_imported": 9999.0, "buyer_unserved": 9999.0, "import_price": 3.99,
         "floor": 0.25, "seller_spare": 99999}
    args = {"to": "southwest", "resource": "power", "amount": 1990, "price": 1.99, "months": 6, "sell": True}
    why, what = opportunity_text(o, "southeast")
    assert len(f"- deal opportunity: {why}: offer_deal {json.dumps(args)} {what} (no link needed)") <= 370


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
    # a deal opportunity per resource (the most there can be), at their longest
    longest = lambda r: {"resource": r, "buyer": "north", "seller": "southwest", "amount": 1990, "price": 1.99,
                         "months": 6, "buyer_use_pct": 999, "buyer_imported": 9999.0, "buyer_unserved": 9999.0,
                         "import_price": 3.99, "floor": 0.25, "seller_spare": 99999}
    region.deal_opportunities = lambda city, states=None: [longest("power"), longest("water")]
    block = region_lines(n, "mayor")
    deals = [x for x in block if x.startswith("- deal #")]
    assert len(deals) == MAX_DEALS and block[-3] == f"- (+{2 * 5 - MAX_DEALS} more deals: region_overview)"
    assert [x[:20] for x in block[-2:]] == ["- deal opportunity: "] * 2
    assert "links: air, boat, highway, rail | last month: goods 12.3k" in block[2]
    assert len("\n".join(block)) <= 3200, len("\n".join(block))           # 3,110
    assert all(len(x) <= 340 for x in block), max(map(len, block))
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
