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
