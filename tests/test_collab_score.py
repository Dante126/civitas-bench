"""The viewer's collaboration score (fleet/report.py: collab_as_of, collab_score, region_score; the collab endpoint's
`score`): the fleet report's score as a run stood at every frame month."""
import json

from civitas.bench.collab import collab, region_collab
from civitas.fleet.report import COLLAB_MIN, collab_as_of, collab_score, collaboration, region_score

ROLES = ["mayor", "planning", "utilities", "transport", "police", "fire", "health", "education"]


def msg(month, sender, to, text):
    return {"month": month, "sender": sender, "to": to, "text": text}


def res(typ, ok=True, cost=0.0, **args):
    return {"action": {"type": typ, **args}, "ok": ok, "message": f"{typ} done", "cost": cost}


def turn(month, role, *results):
    return {"month": month, "role": role, "results": list(results)}


def monthly(last, council=None, turns=None):
    """One frame per month 0..last, as a run writes them: frame F carries the council and turns of month F - 1."""
    council, turns = council or {}, turns or {}
    return [{"month": f, "council": council.get(f - 1, []), "turns": turns.get(f - 1, [])} for f in range(last + 1)]


def _frames():
    """act_every 3 (a 6-month window). Month 0: health, police, fire and education ask the mayor; month 1 the mayor
    funds health in full (met); month 4 it funds police in full (met, late in the window) and gives transport $500
    unasked; fire's request is replied to in month 2; education's is never answered (unanswered from frame 6).
    Planning builds 3 roads in month 0; utilities builds beside one in month 2 (a hand-off)."""
    return monthly(12, council={
        0: [msg(0, "health", "mayor", "I need $2,000 for a landfill"), msg(0, "police", "mayor", "Please fund $2,500"),
            msg(0, "fire", "mayor", "Can you tell me when the station can be built?"),
            msg(0, "education", "mayor", "need $1,000 for a school")],
        2: [msg(2, "mayor", "fire", "Next quarter, once taxes come in")]},
        turns={
        0: [turn(0, "planning", res("build_road", cost=50, x1=9, y1=10, x2=12, y2=10),
                 res("build_road", cost=50, x1=30, y1=30, x2=33, y2=30),
                 res("build_road", cost=50, x1=30, y1=40, x2=33, y2=40))],
        1: [turn(1, "mayor", res("allocate_budget", department="health", amount=2000))],
        2: [turn(2, "utilities", res("build_water_tower", cost=2200, x=11, y=11))],
        4: [turn(4, "mayor", res("allocate_budget", department="police", amount=2500),
                 res("allocate_budget", department="transport", amount=500))]})


def test_series_ends_on_the_report_score():
    c = collab(_frames(), ROLES, act_every=3)
    sc = collab_score(c)
    co = collaboration(c)
    assert {k: sc[k] for k in co} == co                       # the final numbers are the report's
    assert sc["min_cases"] == COLLAB_MIN and sc["follow_weight"]["met"] == 1.0
    ser = sc["series"]
    assert ser["month"] == list(range(13)) and all(len(ser[k]) == 13 for k in ("score", "responsiveness", "money_aim"))
    assert ser["score"][-1] == co["score"] and ser["responsiveness"][-1] == [75.0, 3, 4]
    assert collab_as_of(c, 12) is c and collab_as_of(c, 99) is c
    json.dumps(sc)


def test_as_of_counts_a_request_once_answered_or_past_its_window():
    """At frame F the run had seen the events up to month F - 1: a request counts once its answer is in, an unanswered
    one only once its 6-month window has passed (frame 6); open ones count nowhere, like the report's rule."""
    sc = collab_score(collab(_frames(), ROLES, act_every=3))
    resp = sc["series"]["responsiveness"]
    assert [r[1:] for r in resp[:8]] == [[0, 0], [0, 0], [1, 1], [2, 2], [2, 2], [3, 3], [3, 4], [3, 4]]
    assert resp[5][0] == 100.0 and resp[6][0] == 75.0 and resp[2][0] is None      # n/a under 3 cases
    ft = sc["series"]["follow_through"]
    assert ft[6] == [56.2, 4, 2, 0, 0, 1, 1]                   # (1 + 1 + 0.25 + 0) / 4: met 2, replied 1, unanswered 1
    money = sc["series"]["money_aim"]
    assert money[2] == [None, 2000.0, 2000.0, 1] and money[5] == [90.0, 4500.0, 5000.0, 3]
    hand = sc["series"]["handoffs"]
    assert hand[2] == [None, 0, 0] and hand[3] == [None, 1, 1] and hand[6] == [33.3, 1, 3]     # roads unused by frame 6
    assert sc["series"]["roles"]["mayor"][4] == [None, 2, 2] and sc["series"]["roles"]["mayor"][6] == [75.0, 3, 4]
    assert sc["series"]["score"][1] is None and sc["series"]["score"][6] == round((75 + 56.2 + 33.3 + 90) / 4, 1)


def test_as_of_matches_the_live_run_at_every_frame():
    """The score at frame F is what the server computed for the live run when frame F was its latest."""
    frames = _frames()
    c = collab(frames, ROLES, act_every=3)
    for f in range(len(frames)):
        live = collaboration(collab(frames[:f + 1], ROLES, act_every=3))
        then = collaboration(collab_as_of(c, f))
        assert {k: then[k] for k in ("score", "components", "roles", "requests", "open")} == \
            {k: live[k] for k in ("score", "components", "roles", "requests", "open")}, f


# ------------------------------------------------------------------ region runs
def snap(i, seller, buyer, resource, amount, price, left, status, proposer):
    return {"id": i, "seller": seller, "buyer": buyer, "resource": resource, "amount": amount, "price": price,
            "months_left": left, "status": status, "proposer": proposer}


def deal_res(typ, message, **args):
    return {"action": {"type": typ, **args}, "ok": True, "message": message, "cost": 0.0}


def rframe(month, council=(), turns=(), deals=()):
    return {"month": month, "region": True, "council": list(council), "turns": list(turns), "deals": list(deals),
            "cities": {"north": {"turns": []}, "south": {"turns": []}}}


def region_frames():
    """north asks south for water (month 0); south offers deal #1 (month 1), north accepts it (month 2); north offers
    deal #2 (month 3) and south cancels it (month 4)."""
    d1 = lambda left, st: snap(1, "south", "north", "water", 100.0, 0.5, left, st, "south")
    d2 = lambda left, st: snap(2, "north", "south", "power", 50.0, 1.0, left, st, "north")
    return [
        rframe(0),
        rframe(1, [msg(0, "north.mayor", "south.mayor", "Can you sell us water? We are short.")]),
        rframe(2, [], [turn(1, "south.mayor", deal_res("offer_deal", "proposed deal #1", to="north", resource="water",
                                                       amount=100, price=0.5, months=3))], [d1(3, "offered")]),
        rframe(3, [], [turn(2, "north.mayor", deal_res("accept_deal", "deal #1 is active", deal=1))], [d1(2, "active")]),
        rframe(4, [], [turn(3, "north.mayor", deal_res("offer_deal", "proposed deal #2", to="south", resource="power",
                                                       amount=50, price=1.0, months=12))],
               [d1(1, "active"), d2(12, "offered")]),
        rframe(5, [], [turn(4, "south.mayor", deal_res("cancel_deal", "deal #2 cancelled", deal=2))],
               [d1(0, "ended"), d2(12, "ended")]),
    ]


def test_region_score_asks_and_deals_as_of_each_month():
    frames = region_frames()
    rc = region_collab(frames, ROLES, act_every=1, cities=["north", "south"])
    cities = {c: collab_score(collab(frames, ROLES, 1, c, external=False)) for c in ("north", "south")}
    rs = region_score(rc, cities)
    json.dumps(rs)
    assert set(rs["cities"]) == {"north", "south"} and rs["cities"]["north"]["series"] == cities["north"]["series"]["score"]
    asks, deals = rs["series"]["asks"], rs["series"]["deals"]
    assert asks[1] == [None, 0, 0] and asks[2] == [None, 1, 1]       # answered by the deal offer in month 1
    assert deals[2] == [None, 0, 0] and deals[3] == [None, 1, 1]      # deal #1 accepted in month 2, seen in frame 3
    assert deals[5] == [None, 1, 2]                                   # deal #2 declined (cancelled by the partner)
    assert rs["asks"] == {"score": None, "answered": 1, "n": 1} and rs["deals"] == {"score": None, "accepted": 1, "n": 2}


def test_collab_endpoint_carries_the_score(tmp_path):
    import urllib.request
    from civitas.server import serve
    run = tmp_path / "r1"; run.mkdir()
    (run / "meta.json").write_text(json.dumps({"roles": ROLES, "config": {"game": {"act_every": 3}}}))
    (run / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in _frames()))
    reg = tmp_path / "r2"; reg.mkdir()
    (reg / "meta.json").write_text(json.dumps({"roles": [f"{c}.{r}" for r in ROLES for c in ("north", "south")],
                                               "region": True, "cities": ["north", "south"]}))
    (reg / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in region_frames()))
    srv = serve(tmp_path, port=0, background=True)
    try:
        port = srv.server_address[1]
        d = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs/r1/collab"))
        assert d["score"]["score"] == d["score"]["series"]["score"][-1] and d["threads"]
        assert d["score"]["series"]["responsiveness"][6] == [75.0, 3, 4]
        r = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs/r2/collab?region=1"))
        assert r["region"] and set(r["score"]["cities"]) == {"north", "south"} and r["score"]["deals"]["n"] == 2
        c = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs/r2/collab?city=north"))
        assert "series" in c["score"] and "region" not in c
    finally:
        srv.shutdown()
