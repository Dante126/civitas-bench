"""The fleet report's collaboration score (fleet/report.py: collaboration, run_collaboration, the Markdown section)."""
import json

from civitas.bench.collab import collab
from civitas.fleet.report import _collab_lines, collab_as_of, collaboration, run_collaboration

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
    assert cp["money_aim"] == {"score": 85.7, "n": 3, "aimed": 3000.0, "given": 3500.0,   # transport's $500 unasked
                               "basis": "messages", "needed": 0.0, "asked": 3000.0, "needs_missing": 0}        # no needs record
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


NEEDS = {"planning": ["growth", 300, 1], "utilities": ["nothing", 0, 0], "transport": ["idle", 0, 0],
         "police": ["first/supply", 2500, 1], "fire": ["another", 2500, 1], "health": ["another", 2000, 0],
         "education": ["nothing", 0, 0]}


def _need_frames(needs=NEEDS):
    """act_every 1. Education asks the mayor for $1,000 in month 0. Frame 1 carries the needs the mayor saw in month 1
    (FleetState.needs); its top-ups that month (in frame 2): police $2,000 then $1,000 for a first station short
    $2,500, planning $300 for its road, fire $2,500 for another station the money covers, health $2,000 for another
    clinic it doesn't, transport $500 with idle money and education the $1,000 it asked for with nothing to build.
    Month 3 has no record: police's $1,000 then counts only by a request (it made none)."""
    gifts = [("police", 2000), ("police", 1000), ("planning", 300), ("fire", 2500), ("health", 2000),
             ("transport", 500), ("education", 1000)]
    f1 = {**frame(1, [msg(0, "education", "mayor", "need $1,000 for a school")]),
          **({"needs": needs} if needs else {})}
    return [frame(0), f1,
            frame(2, [], [turn(1, "mayor", *(res("allocate_budget", department=d, amount=a) for d, a in gifts),
                               res("allocate_budget", department="police", amount=-400))]),     # a take-back: left out
            frame(3), frame(4, [], [turn(3, "mayor", res("allocate_budget", department="police", amount=1000))]),
            frame(9)]


def test_money_aim_reads_the_needs_record():
    """With a needs record a top-up is aimed when it went to a first building / supply, growth or another building the
    money covered, up to what the needs line lacked that month, or answered the department's request; else not."""
    c = collab(_need_frames(), ROLES, act_every=1)
    assert c["needs_from"] == 1 and collaboration(collab_as_of(c, 0))["components"]["money_aim"]["basis"] == "messages"
    police = [e for e in c["events"] if e["kind"] == "money" and "police" in (e["to"], e["from"]) and e["month"] == 1]
    assert police[0]["need"] == ["first/supply", 2500, 1] and "need" not in police[-1]       # the take-back
    ma = collaboration(c)["components"]["money_aim"]
    # police 2,000 + 500 of its 1,000 (short 2,500), planning 300, fire 2,500 (covered), education 1,000 (asked);
    # health's uncovered 2,000, transport's idle 500 and month 3's unrecorded 1,000 miss
    assert ma == {"score": round(100 * 6300 / 10300, 1), "n": 8, "aimed": 6300.0, "given": 10300.0, "basis": "needs",
                  "needed": 5300.0, "asked": 1000.0, "needs_missing": 4}   # frames 2, 3, 4 and 9 have none
    md = "\n".join(_collab_lines(collaboration(c)))
    assert ("| money aim | 61.2 | $6,300 of $10.3k in 8 top-ups went to a first building, supply, planning's road or "
            "another building the money covered ($5,300), or answered the department's request ($1,000); 4 months "
            "without a needs record |") in md


def test_money_aim_without_a_needs_record_is_message_based():
    """A run from before the record (no `needs` in its frames) keeps the message rule, and the report says so."""
    co = collaboration(collab(_need_frames(needs=None), ROLES, act_every=1))
    ma = co["components"]["money_aim"]
    assert collab(_need_frames(needs=None), ROLES, act_every=1)["needs_from"] is None
    assert ma == {"score": round(100 * 1000 / 10300, 1), "n": 8, "aimed": 1000.0, "given": 10300.0,
                  "basis": "messages", "needed": 0.0, "asked": 1000.0, "needs_missing": 0}
    assert "answered the department's request (message-based: this run has no needs record) |" in \
        "\n".join(_collab_lines(co))


def test_a_failing_needs_record_is_logged_once(monkeypatch, caplog):
    """Review: FleetState.needs swallowed every exception silently. It stays non-fatal (a frame must never end a run),
    logs the first failure of a run with its traceback and counts the rest; the frame goes without a record, which
    collab counts (needs_missing, test_money_aim_reads_the_needs_record) and the report notes."""
    import logging

    from civitas.core.env import CivitasEnv, GameConfig
    from civitas.fleet import tools as T
    from civitas.fleet.memory import FleetState
    env = CivitasEnv(GameConfig(seed=0, months=6))
    fleet = FleetState.of(env)
    assert fleet.needs(env)                                              # a record while it works
    monkeypatch.setattr(T, "funding_gaps", lambda *a: 1 / 0)
    with caplog.at_level(logging.WARNING, logger="civitas.fleet.memory"):
        assert fleet.needs(env) is None and fleet.needs(env) is None and "needs" not in env.frame()
    assert fleet.needs_failures == 3
    assert [r.getMessage()[:31] for r in caplog.records] == ["FleetState.needs failed at mont"]
    assert "ZeroDivisionError" in caplog.text


def test_money_aim_reads_a_region_citys_needs():
    """A region frame nests each city's needs under `cities`; collab(city=...) reads that city's."""
    frames = [{"month": 0, "region": True, "council": [], "turns": [],
               "cities": {"north": {"needs": NEEDS}, "south": {"needs": {"police": ["idle", 0, 0]}}}},
              {"month": 1, "region": True, "council": [], "turns": [],
               "cities": {c: {"turns": [turn(0, "mayor", res("allocate_budget", department="police", amount=2500))]}
                          for c in ("north", "south")}}]
    aim = {c: collaboration(collab(frames, ROLES, act_every=1, city=c, external=False))["components"]["money_aim"]
           for c in ("north", "south")}
    assert aim["north"]["needed"] == 2500.0 and aim["south"]["needed"] == 0.0
    assert aim["north"]["basis"] == aim["south"]["basis"] == "needs"


def test_a_region_frame_nests_each_fleet_citys_needs():
    """A region frame carries `needs` in the frame of each city with a fleet state (CivitasEnv.frame), not at the top."""
    from civitas.config import game_config, load_config, scoring_config
    from civitas.core.env import make_env
    from civitas.fleet.memory import FleetState
    cfg = load_config("configs/region.toml")
    game = game_config(cfg, 0)
    env = make_env(game, scoring_config(cfg, game))
    FleetState.of(env.envs["north"])
    f = env.frame()
    assert "needs" not in f and "needs" not in f["cities"]["south"]
    assert set(f["cities"]["north"]["needs"]) == set(ROLES[1:])


def test_run_collaboration_reads_a_run_folder(tmp_path):
    assert run_collaboration(None) is None and run_collaboration(tmp_path) is None     # no frames.jsonl
    (tmp_path / "meta.json").write_text(json.dumps({"roles": ROLES, "config": {"game": {"act_every": 1}}}))
    (tmp_path / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in _frames()))
    co = run_collaboration(tmp_path)
    assert co == collaboration(collab(_frames(), ROLES, act_every=1)) and co["score"] == 61.6
