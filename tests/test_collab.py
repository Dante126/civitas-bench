"""The Team tab's backend (bench/collab.py): requests, their responses and outcomes, from frames alone."""
import json

from civitas.bench.collab import _amount, collab, run_collab

ROLES = ["mayor", "planning", "utilities", "transport", "police", "fire", "health", "education"]


def msg(month, sender, to, text):
    return {"month": month, "sender": sender, "to": to, "text": text}


def res(typ, ok=True, cost=0.0, **args):
    return {"action": {"type": typ, **args}, "ok": ok, "message": f"{typ} done", "cost": cost}


def turn(month, role, *results):
    return {"month": month, "role": role, "results": list(results)}


def frame(month, council=(), turns=()):
    return {"month": month, "council": list(council), "turns": list(turns)}


def test_money_requests_met_partial_unanswered_and_open():
    """act_every 3: a request at month 0 is answered by the mayor's top-ups in months 1-6 (the mayor acts first in
    each round, so its month-0 top-up came before the request)."""
    frames = [
        frame(1, [msg(0, "health", "mayor", "I need $2,000 for a landfill"),
                  msg(0, "police", "mayor", "Please fund $2,500 for a station"),
                  msg(0, "fire", "mayor", "Requesting $2,000 for a fire station"),
                  msg(0, "planning", "all", "Roads done; need nothing")],
              [turn(0, "mayor", res("allocate_budget", department="police", amount=2500))]),
        frame(4, [], [turn(3, "mayor", res("allocate_budget", department="health", amount=2000),
                                   res("allocate_budget", department="police", amount=1000)),
                      turn(3, "health", res("build_landfill", cost=2000, x=21, y=30))]),
        frame(7, [msg(6, "education", "mayor", "need $2,500 for a school")],
              [turn(6, "mayor", res("allocate_budget", department="fire", amount=500))]),
    ]
    d = collab(frames, ROLES, act_every=3)
    th = {t["from"]: t for t in d["threads"]}
    assert set(th) == {"health", "police", "fire", "education"}          # the broadcast starts no thread
    assert th["health"]["status"] == "met" and th["health"]["response"]["amount"] == 2000
    assert th["health"]["outcome"] == {"month": 3, "what": "built landfill", "tile": [21, 30]}
    assert th["police"]["status"] == "partial" and th["police"]["response"]["amount"] == 1000   # month 0 came before
    assert th["fire"]["status"] == "partial" and th["fire"]["outcome"] is None                  # $500 of $2,000
    assert th["education"]["status"] == "open"                         # month 6 + 6 > last month 6: still open
    assert all(t["kind"] == "money" for t in d["threads"])
    assert d["stats"]["mayor"]["asked_of"] == 4 and d["stats"]["mayor"]["answered_by"] == 3
    assert d["stats"]["health"]["money_in"] == 2000 and d["stats"]["health"]["builds"] == 1
    months = [e["month"] for e in d["events"]]
    assert months == sorted(months) and [e["id"] for e in d["events"]] == list(range(len(d["events"])))
    # nothing came within the window: unanswered once it has passed
    d = collab(frames + [frame(13)], ROLES, act_every=3)
    assert {t["from"]: t["status"] for t in d["threads"]}["education"] == "unanswered"


def test_a_money_request_the_mayor_replies_to_is_answered():
    """A no with why or when is an answer: a money ask with no top-up but a message back from the mayor is replied
    to (the mayor's Open requests block decides the same way, through collab.answer); a top-up still wins."""
    frames = [frame(1, [msg(0, "police", "mayor", "Requesting $2,500 for a station"),
                        msg(0, "fire", "mayor", "Need $2,500 for a station")]),
              frame(4, [msg(3, "mayor", "police", "Not this quarter: the reserve comes first"),
                        msg(3, "mayor", "fire", "Next quarter")],
                    [turn(3, "mayor", res("allocate_budget", department="fire", amount=2500))]),
              frame(13)]
    th = {t["from"]: t for t in collab(frames, ROLES, act_every=3)["threads"]}
    assert th["police"]["status"] == "replied" and th["police"]["kind"] == "money"
    assert th["police"]["response"]["text"] == "Not this quarter: the reserve comes first"
    assert th["fire"]["status"] == "met"


def test_land_requests_acted_and_replies():
    frames = [
        frame(1, [msg(0, "utilities", "planning", "Please free a riverbank lot near (33,23) for the pump"),
                  msg(0, "health", "planning", "Can you leave a lot for a clinic?")],
              [turn(0, "planning", res("dezone", x1=33, y1=22, x2=34, y2=22))]),     # planning acts before utilities?
        frame(4, [msg(3, "planning", "health", "Left (20,22) free for you")],
              [turn(3, "planning", res("dezone", x1=32, y1=23, x2=33, y2=23)),
               turn(3, "utilities", res("build_water", cost=900, x=33, y=23))]),
    ]
    d = collab(frames, ROLES, act_every=3)
    th = {t["from"]: t for t in d["threads"]}
    # planning comes before utilities in turn order, so its month-0 dezone preceded the request: the month-3 one counts
    assert th["utilities"]["kind"] == "land" and th["utilities"]["status"] == "acted"
    assert th["utilities"]["response"]["month"] == 3 and th["utilities"]["outcome"]["what"] == "built water"
    assert th["health"]["status"] == "replied" and th["health"]["kind"] == "land"


def test_amounts_asked_for():
    assert _amount("Budget remaining: $61. Requesting additional funds: $2,000 for roads") == 2000
    assert _amount("I need $1,200 additional funds") == 1200
    assert _amount("please send $4k now") == 4000
    assert _amount("we have $0 left") is None and _amount("no money talk") is None


def test_a_status_report_that_needs_nothing_is_no_request():
    """Region smoke 20261005 and the 120-month trinity runs: transport's "No budget needed at this time (current $0
    matches target)" counted as a request to the mayor. A negated need and $0 aren't asks; any other ask word is."""
    from civitas.bench.collab import is_request
    assert not is_request("mayor", "Transport Department reporting: below the ~1,500 threshold. No congested roads "
                                   "detected. No budget needed at this time (current $0 matches target).")
    assert not is_request("mayor", "Police Department ready. No immediate needs.")
    assert not is_request("mayor", "Education: no schools needed yet. Current budget: $0.")
    assert not is_request("mayor", "Utilities: water at 66% of capacity, so no expansion is required.")
    assert is_request("mayor", "No budget needed now; next quarter we need $1,500 for a bus line")
    assert is_request("mayor", "Station built for $2,500") and is_request("mayor", "Holding $0, need $2,000")
    assert is_request("mayor", "Please allocate $0.5k") and not is_request("all", "We need $2,000")


def test_a_scripted_episode(tmp_path):
    from civitas.bench.runner import run_episode
    from civitas.config import load_config
    cfg = load_config("configs/baseline.toml")
    cfg["game"]["months"] = 24
    s = run_episode(cfg, 0, tmp_path, verbose=False, log_observations=False)
    d = run_collab(tmp_path / s["run_id"])
    json.dumps(d)
    assert d["last_month"] == 24 and d["events"] and d["roles"][0] == "mayor"
    assert any(e["kind"] == "money" for e in d["events"]) and any(e["kind"] == "build" for e in d["events"])
    assert all(t["status"] in ("met", "partial", "acted", "replied", "unanswered", "open") for t in d["threads"])


def test_collab_endpoint(tmp_path):
    import urllib.request
    from civitas.server import serve
    run = tmp_path / "r1"; run.mkdir()
    (run / "meta.json").write_text(json.dumps({"roles": ROLES, "config": {"game": {"act_every": 3}}}))
    (run / "frames.jsonl").write_text(json.dumps(frame(1, [msg(0, "health", "mayor", "need $2,000")])) + "\n")
    srv = serve(tmp_path, port=0, background=True)
    try:
        port = srv.server_address[1]
        d = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs/r1/collab"))
        assert d["act_every"] == 3 and d["threads"][0]["status"] == "open" and d["threads"][0]["amount"] == 2000
    finally:
        srv.shutdown()


def test_handoffs_link_planning_openers_to_buildings_nearby():
    """A department's build within 2 tiles of planning's earlier road, dezone or clearance, inside the window, is a
    hand-off from the latest such opener; planning's own builds and builds far away are not."""
    frames = [
        frame(1, [], [turn(0, "planning", res("build_road", x1=10, y1=20, x2=18, y2=20), res("build_road", x1=30, y1=5, x2=30, y2=9))]),
        frame(4, [], [turn(3, "planning", res("dezone", x1=16, y1=21, x2=17, y2=21), res("build_park", x=12, y=21)),
                      turn(3, "health", res("build_clinic", cost=2000, x=17, y=22)),
                      turn(3, "police", res("build_police_station", cost=2500, x=40, y=40))]),
        frame(13, [], [turn(12, "fire", res("build_fire_station", cost=2500, x=12, y=19))]),    # past the window
    ]
    d = collab(frames, ROLES, act_every=3)
    assert len(d["handoffs"]) == 1, d["handoffs"]
    h = d["handoffs"][0]
    assert (h["from"], h["what_from"], h["month"], h["rect_from"]) == ("planning", "freed lot", 3, [16, 21, 17, 21])
    assert (h["to"], h["what"], h["month_to"], h["tile_to"], h["tile_from"]) == ("health", "built clinic", 3, [17, 22], [17, 21])


# ------------------------------------------------------------------ region runs: deals and messages between cities
def deal_res(typ, message, **args):
    return {"action": {"type": typ, **args}, "ok": True, "message": message, "cost": 0.0}


def snap(i, seller, buyer, resource, amount, price, left, status, proposer):
    return {"id": i, "seller": seller, "buyer": buyer, "resource": resource, "amount": amount, "price": price,
            "months_left": left, "status": status, "proposer": proposer}


def rframe(month, council=(), turns=(), deals=()):
    """A region frame: prefixed roles in the council and the top-level turns, the deal snapshot after the month."""
    return {"month": month, "region": True, "council": list(council), "turns": list(turns), "deals": list(deals),
            "cities": {"north": {"turns": []}, "south": {"turns": []}}}


def region_frames():
    """north asks south for water (month 0); south offers deal #1 (month 1), north accepts it (month 2), it runs out
    after 3 paid months (month 4). north offers deal #2 (month 3) and south cancels it (month 4)."""
    d1 = lambda left, st: snap(1, "south", "north", "water", 100.0, 0.5, left, st, "south")
    d2 = lambda left, st: snap(2, "north", "south", "power", 50.0, 1.0, left, st, "north")
    return [
        rframe(1, [msg(0, "north.mayor", "south.mayor", "Can you sell us water? We are short."),
                   msg(0, "north.mayor", "north.planning", "Please zone more housing"),
                   msg(0, "south.planning", "north", "Our highway connector is done")]),
        rframe(2, [msg(1, "south.mayor", "north", "DEAL #1 proposed: south sells 100 water to north at $0.5/unit/month for 3 months")],
               [turn(1, "south.mayor", deal_res("offer_deal", "proposed deal #1; the north mayor must accept it",
                                                to="north", resource="water", amount=100, price=0.5, months=3))],
               [d1(3, "offered")]),
        rframe(3, [], [turn(2, "north.mayor", deal_res("accept_deal", "deal #1 is active", deal=1))], [d1(2, "active")]),
        rframe(4, [msg(3, "north.mayor", "south", "DEAL #2 proposed: north sells 50 power to south at $1.0/unit/month for 12 months")],
               [turn(3, "north.mayor", deal_res("offer_deal", "proposed deal #2; the south mayor must accept it",
                                                to="south", resource="power", amount=50, price=1.0, months=12))],
               [d1(1, "active"), d2(12, "offered")]),
        rframe(5, [], [turn(4, "south.mayor", deal_res("cancel_deal", "deal #2 cancelled", deal=2))],
               [d1(0, "ended"), d2(12, "ended")]),
    ]


def test_region_collab_deals_messages_and_threads():
    from civitas.bench.collab import region_collab
    d = region_collab(region_frames(), ROLES, act_every=1, cities=["north", "south"])
    json.dumps(d)
    assert d["region"] and d["cities"] == ["north", "south"] and d["last_month"] == 5
    d1, d2 = d["deals"]
    assert (d1["proposer"], d1["partner"], d1["offered"], d1["accepted"], d1["ended"], d1["end"]) == \
        ("south", "north", 1, 2, 4, "expired")
    assert (d1["months"], d1["months_paid"], d1["monthly"], d1["paid"]) == (3, 3, 50.0, 150.0)
    assert (d2["offered"], d2["accepted"], d2["ended"], d2["end"], d2["ended_by"], d2["paid"]) == \
        (3, None, 4, "cancelled", "south", 0)
    deal_ev = [(e["month"], e["deal"], e["what"], e["from_city"]) for e in d["events"] if e["kind"] == "deal"]
    assert deal_ev == [(1, 1, "offered", "south"), (2, 1, "accepted", "north"), (3, 2, "offered", "north"),
                       (4, 2, "cancelled", "south"), (4, 1, "expired", "south")]
    msgs = [e for e in d["events"] if e["kind"] == "message"]
    assert len(msgs) == 4 and all(e["from_city"] != e["to_city"] for e in msgs)      # the local message is not here
    assert [e["deal"] for e in msgs if e["deal"]] == [1, 2] and not any(e["ask"] for e in msgs if e["deal"])
    th = {(t["kind"], t["deal"]): t for t in d["threads"]}
    ask = th[("trade", None)]
    assert (ask["from"], ask["to"], ask["status"]) == ("north.mayor", "south.mayor", "acted")
    assert ask["response"]["deal"] == 1 and ask["outcome"]["deal"] == 1 and ask["outcome"]["month"] == 2
    assert th[("deal", 1)]["status"] == "met" and th[("deal", 1)]["response"]["month"] == 2
    assert th[("deal", 2)]["status"] == "declined"
    assert len(d["threads"]) == 3                                  # the connector news asks nothing
    n, s = d["stats"]["north"], d["stats"]["south"]
    assert (n["sent"], n["received"], s["sent"], s["received"]) == (2, 2, 2, 2)
    assert (n["asks_out"], n["asks_out_answered"], s["asks_in"], s["asks_in_answered"]) == (1, 1, 1, 1)
    assert (n["deals_proposed"], n["deals_accepted"], s["deals_proposed"], s["deals_offered"]) == (1, 1, 1, 1)
    assert (n["bought"], s["sold"], n["requests"], n["answered"]) == (150.0, 150.0, 1, 0)


def test_city_view_of_a_region_run_keeps_other_cities_out():
    """collab(city=C) keeps C's own team: a message to another city is not one of its requests but an `external`
    event, beside C's deals and its threads with other cities."""
    d = collab(region_frames(), ROLES, act_every=1, city="north")
    assert [(t["from"], t["to"]) for t in d["threads"]] == [("mayor", "planning")]
    assert d["stats"]["mayor"]["messages"] == 1
    assert [x["id"] for x in d["deals"]] == [1, 2]
    assert all("north" in (e["from_city"], e["to_city"]) for e in d["external"]) and len(d["external"]) == 9
    assert {t["kind"] for t in d["external_threads"]} == {"trade", "deal"}
    single = collab([frame(1, [msg(0, "health", "mayor", "need $2,000")])], ROLES, act_every=3)
    assert "deals" not in single and "external" not in single


def test_region_collab_endpoint(tmp_path):
    import urllib.request
    from civitas.server import serve
    run = tmp_path / "r2"; run.mkdir()
    (run / "meta.json").write_text(json.dumps({"roles": [f"{c}.{r}" for r in ROLES for c in ("north", "south")],
                                               "region": True, "cities": ["north", "south"]}))
    (run / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in region_frames()))
    srv = serve(tmp_path, port=0, background=True)
    try:
        port = srv.server_address[1]
        d = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs/r2/collab?region=1"))
        assert d["region"] and len(d["deals"]) == 2 and d["stats"]["south"]["sold"] == 150.0
        c = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs/r2/collab?city=south"))
        assert "region" not in c and len(c["deals"]) == 2 and c["roles"][0] == "mayor"
    finally:
        srv.shutdown()


def test_a_negation_stays_in_its_clause():
    """Review: "no clinic and need one" lost its only ask word to the negated-need strike."""
    from civitas.bench.collab import is_request
    assert is_request("mayor", "We have no clinic and need one near (12,4)")
    assert is_request("planning", "Nothing built yet, we need land near 3,4")
    assert not is_request("mayor", "Below 1,500 residents. No budget needed at this time (current $0 matches target)")
    assert not is_request("mayor", "No schools needed yet")


def test_replies_are_not_requests_of_their_own():
    """A mayor's answer ("Approved ...", "unable to fund ...") is no request; a money ask back to the mayor still is."""
    from civitas.bench.collab import replies
    ev = [{"kind": "message", "month": 3, "from": "police", "to": "mayor", "text": "Need $2,500 for our first station"},
          {"kind": "message", "month": 3, "from": "mayor", "to": "police", "text": "Please hold on, next quarter"},
          {"kind": "message", "month": 6, "from": "mayor", "to": "health", "text": "I'm unable to fund a clinic now"},
          {"kind": "message", "month": 6, "from": "mayor", "to": "planning", "text": "Please build a road at (3,4)"},
          {"kind": "message", "month": 6, "from": "planning", "to": "mayor", "text": "Need $300 to build it"},
          {"kind": "message", "month": 9, "from": "planning", "to": "mayor", "text": "Could you check the plan?"}]
    assert replies(ev, 6) == {1, 2, 5}
