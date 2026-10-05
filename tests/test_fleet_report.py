"""The fleet report's collaboration score (fleet/report.py: collaboration, run_collaboration, the Markdown section)."""
import json

from civitas.bench.collab import collab
from civitas.fleet.report import _collab_lines, collaboration, run_collaboration

ROLES = ["mayor", "planning", "utilities", "transport", "police", "fire", "health", "education"]


def msg(month, sender, to, text):
    return {"month": month, "sender": sender, "to": to, "text": text}


def res(typ, ok=True, cost=0.0, **args):
    return {"action": {"type": typ, **args}, "ok": ok, "message": f"{typ} done", "cost": cost}


def turn(month, role, *results):
    return {"month": month, "role": role, "results": list(results)}


def frame(month, council=(), turns=()):
    return {"month": month, "council": list(council), "turns": list(turns)}


def _frames():
    """act_every 1 (a 2-month window): health's ask met, police's partly met, fire's acted on by planning's road (which
    fire then builds beside: a hand-off), education's unanswered; transport got $500 it never asked for; two more
    planning roads nobody built on."""
    return [
        frame(1, [msg(0, "health", "mayor", "I need $2,000 for a landfill"),
                  msg(0, "police", "mayor", "Please fund $2,500 for a station"),
                  msg(0, "fire", "planning", "Please free a lot near (10,10) for a fire station"),
                  msg(0, "education", "mayor", "need $1,000 for a school")],
              [turn(1, "mayor", res("allocate_budget", department="health", amount=2000),
                    res("allocate_budget", department="police", amount=1000),
                    res("allocate_budget", department="transport", amount=500)),
               turn(1, "planning", res("build_road", cost=50, x1=9, y1=10, x2=12, y2=10),
                    res("build_road", cost=50, x1=30, y1=30, x2=33, y2=30),
                    res("build_road", cost=50, x1=30, y1=40, x2=33, y2=40))]),
        frame(2, [], [turn(2, "fire", res("build_fire_station", cost=2500, x=11, y=11))]),
        frame(5),
    ]


def test_score_components_and_rows():
    co = collaboration(collab(_frames(), ROLES, act_every=1))
    cp = co["components"]
    assert co["requests"] == 4 and co["open"] == 0 and co["window"] == 2
    assert cp["responsiveness"] == {"score": 75.0, "n": 4, "answered": 3}            # met, partial, acted / 4
    assert cp["follow_through"]["score"] == 52.5                                       # (1 + 0.6 + 0.5 + 0) / 4
    assert cp["follow_through"]["by_status"] == {"met": 1, "partial": 1, "acted": 1, "replied": 0, "unanswered": 1}
    assert cp["handoffs"] == {"score": 33.3, "n": 3, "used": 1}                       # 1 of planning's 3 roads
    assert cp["money_aim"] == {"score": 85.7, "n": 3, "aimed": 3000.0, "given": 3500.0}   # transport's $500 unasked
    assert co["score"] == 61.6                                                         # (75 + 52.5 + 33.3 + 85.7) / 4
    r = co["roles"]
    assert r["mayor"] == {"asked": 0, "answered": 0, "asked_of": 3, "answered_for": 2, "responsiveness": 66.7,
                          "follow_through": None, "handoffs": 0}
    assert r["planning"]["responsiveness"] is None and r["planning"]["answered_for"] == 1   # 1 case: n/a
    assert r["planning"]["handoffs"] == 1 and r["fire"]["handoffs"] == 1
    assert r["health"]["follow_through"] is None and "utilities" not in r                # no requests, no hand-offs
    assert co["unanswered"] == [{"month": 0, "from": "education", "to": "mayor", "text": "need $1,000 for a school"}]
    md = "\n".join(_collab_lines(co))
    assert "Score **61.6** / 100" in md and "| hand-offs | 33.3 | 1 of 3 planning roads" in md
    assert "| mayor | 0 | 0 | 3 | 2 | n/a | n/a | 0 |" not in md and "| mayor | 0 | 0 | 3 | 2 | 66.7 | n/a | 0 |" in md
    assert '- month 0, education -> mayor: "need $1,000 for a school"' in md


def test_few_cases_are_na_and_open_requests_are_left_out():
    """Under 3 cases a component is n/a, all n/a makes the score n/a, and nothing divides by zero; a request whose
    window hasn't passed is open and counts nowhere."""
    frames = [frame(1, [msg(0, "health", "mayor", "I need $2,000 for a landfill"),
                        msg(0, "police", "mayor", "Please fund $2,500")],
                    [turn(1, "mayor", res("allocate_budget", department="health", amount=2000))]),
              frame(4, [msg(3, "fire", "mayor", "need $2,500 for a station")])]
    co = collaboration(collab(frames, ROLES, act_every=1))
    assert co["requests"] == 3 and co["open"] == 1                     # fire asked in month 3, last month 4 < 3 + 2
    assert all(v["score"] is None for v in co["components"].values()) and co["score"] is None
    assert co["components"]["responsiveness"]["n"] == 2 and co["components"]["handoffs"]["n"] == 0
    md = "\n".join(_collab_lines(co))
    assert "Score **n/a** / 100" in md and "| responsiveness | n/a | 1 of 2 requests" in md
    empty = collaboration(collab([], ROLES))
    assert empty["score"] is None and empty["roles"] == {} and empty["unanswered"] == []
    assert _collab_lines(None) == ["n/a (no frames.jsonl to read)."]


def test_requests_outside_the_game_do_not_count():
    """A message to a role not in the line-up (another city's mayor) or to oneself starts no scored request."""
    frames = [frame(1, [msg(0, "health", "south.mayor", "Please send $2,000"), msg(0, "health", "health", "need $5")]),
              frame(9)]
    assert collaboration(collab(frames, ROLES))["requests"] == 0


def test_run_collaboration_reads_a_run_folder(tmp_path):
    assert run_collaboration(None) is None and run_collaboration(tmp_path) is None     # no frames.jsonl
    (tmp_path / "meta.json").write_text(json.dumps({"roles": ROLES, "config": {"game": {"act_every": 1}}}))
    (tmp_path / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in _frames()))
    co = run_collaboration(tmp_path)
    assert co == collaboration(collab(_frames(), ROLES, act_every=1)) and co["score"] == 61.6
