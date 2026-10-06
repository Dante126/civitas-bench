"""The Compare view's backend (bench/compare.py, /api/compare): runs side by side from their folders alone."""
import json
import urllib.request

from test_collab import region_frames
from test_collab_score import ROLES, _frames as team_frames

from civitas.bench.compare import kind_of, run_compare

FLEET = {r: "fleet:trinity-large-thinking" for r in ROLES}


def _scored(frames, base=40.0):
    """frames with a city score and population per month (city = base + month, roles = city + 1)."""
    out = []
    for f in frames:
        m = f["month"]
        out.append({**f, "scores": {"city": base + m, **{r: base + m + 1 for r in ROLES}},
                    "metrics": {"population": 100 * m}})
    return out


def _write(d, meta, frames, summary=None, report=None):
    d.mkdir()
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
    if summary is not None:
        (d / "summary.json").write_text(json.dumps(summary))
    if report is not None:
        (d / "fleet_report.json").write_text(json.dumps(report))


def _single(tmp_path, name="fleet", warm=0, report=True):
    frames = _scored(team_frames())                   # months 0..12, act_every 3, requests to the mayor
    last = frames[-1]
    meta = {"name": name, "seed": 0, "months": 12, "roles": ROLES, "agents": FLEET, "signature": "fleet sig",
            "config": {"game": {"act_every": 3}}, **({"warmup_months": warm} if warm else {})}
    summ = {"run_id": name, "months_played": 12, "bankrupt": False, "final_scores": last["scores"],
            "avg_city_score": 46.5, "final_metrics": {"population": 1200}, "wall_time_s": 3.0}
    if warm:
        summ["warmup"] = {"months": warm, "handed_over": True,
                          "at_handover": {"month": warm, "city": 40.0 + warm, "population": 100 * warm},
                          "at_end": {"month": 12, "city": 52.0, "population": 1200}}
    rep = {"roles": {"mayor": {"turns": 4, "tool_calls": 30, "actions": 10, "failed": 2, "tool_errors": 1,
                               "repeats": 1, "refused": 3, "overrides": 1},
                     "police": {"turns": 4, "tool_calls": 20, "actions": 5, "failed": 1, "tool_errors": 0,
                                "repeats": 0, "refused": 2, "overrides": 0}}} if report else None
    _write(tmp_path / name, meta, frames, summ, rep)
    return tmp_path / name


def test_a_single_city_fleet_run(tmp_path):
    c = run_compare(_single(tmp_path))
    json.dumps(c)
    assert (c["kind"], c["model"], c["tags"], c["region"]) == ("LLM fleet", "trinity-large-thinking", [], False)
    assert (c["months"], c["warmup"], c["finished"], c["city"], c["avg_city"]) == (12, 0, True, 52.0, 46.5)
    assert c["roles"]["mayor"] == 53.0 and "city" not in c["roles"] and c["population"] == 1200
    assert c["calls"] == {"turns": 8, "tool_calls": 50, "actions": 15, "failed": 3, "tool_errors": 1, "repeats": 1,
                          "refused": 5, "overrides": 1}
    t = c["teamwork"]
    assert set(t["components"]) == {"responsiveness", "follow_through", "handoffs", "money_aim"}
    assert t["score"] == t["series"][-1] and len(t["series"]) == 13 and t["requests"] == 4
    s = c["series"]
    assert c["month"] == list(range(13)) and s["city"][0] == 40.0 and s["city"][-1] == 52.0
    assert s["population"][-1] == 1200 and s["teamwork"] == t["series"]


def test_a_warm_up_run_without_a_fleet_report(tmp_path):
    c = run_compare(_single(tmp_path, "late", warm=6, report=False))
    assert c["warmup"] == 6 and c["tags"] == ["late (warm-up 6)"] and c["calls"] is None
    assert c["at_handover"] == {"month": 6, "city": 46.0, "population": 600}
    # frames 7..12 are the months after the warm-up: city 47..52
    assert c["avg_after_warmup"] == 49.5 and c["avg_city"] == 46.5
    assert c["teamwork"] is not None                  # from frames, since the warm-up (run_collab's `since`)


def test_a_region_run_per_city_and_region(tmp_path):
    frames = []
    for f in region_frames():                         # months 1..5, deals and asks between north and south
        m = f["month"]
        cities = {c: {**f["cities"][c], "scores": {"city": 50.0 + m + k, **{r: 60.0 + k for r in ROLES}},
                      "metrics": {"population": 1000 * (k + 1) + m}} for k, c in enumerate(("north", "south"))}
        frames.append({**f, "cities": cities, "scores": {"city": 50.5 + m}, "metrics": {"population": 3000 + 2 * m}})
    meta = {"name": "reg", "seed": 1, "months": 5, "region": True, "cities": ["north", "south"],
            "roles": [f"{c}.{r}" for r in ROLES for c in ("north", "south")],
            "agents": {f"{c}.{r}": "scripted" for r in ROLES for c in ("north", "south")}}
    rep = {"cities": {"north": {"roles": {"mayor": {"turns": 2, "failed": 1, "refused": 2}}},
                      "south": {"roles": {"mayor": {"turns": 2, "failed": 0, "refused": 1}}}}}
    _write(tmp_path / "reg", meta, frames, report=rep)            # still running: no summary.json
    c = run_compare(tmp_path / "reg")
    json.dumps(c)
    assert (c["kind"], c["tags"], c["region"], c["finished"], c["months"]) == ("scripted", ["region"], True, False, 5)
    assert c["city"] == 55.5 and c["population"] == 3010 and c["roles_mean_of_cities"]
    assert c["roles"]["mayor"] == 60.5                         # the mean of north's 60 and south's 61
    assert c["calls"]["failed"] == 1 and c["calls"]["refused"] == 3 and c["calls"]["turns"] == 4
    n, s = c["by_city"]["north"], c["by_city"]["south"]
    assert (n["city"], s["city"], n["population"], s["population"]) == (55.0, 56.0, 1005, 2005)
    assert n["roles"]["mayor"] == 60.0 and "city" not in n["roles"]
    assert n["calls"]["refused"] == 2 and s["calls"]["refused"] == 1
    assert n["series"]["city"] == [51.0, 52.0, 53.0, 54.0, 55.0] and s["series"]["population"][-1] == 2005
    assert c["teamwork"]["mean_of_cities"] and set(c["teamwork"]["components"]) >= {"responsiveness"}
    assert len(n["series"]["teamwork"]) == 6                   # collab months 0..5
    assert c["avg_city"] == 53.5                               # frames 1..5: city 51.5..55.5


def test_kinds():
    assert kind_of({r: "fleet:scripted" for r in ROLES}) == "scripted fleet"
    assert kind_of({"a": "fleet:scripted", "b": "fleet:x"}) == "LLM fleet"
    assert kind_of({"a": "fleet:x", "b": "scripted"}) == "LLM fleet (mixed)"
    assert kind_of({"a": "llm:x", "b": "llm:y"}) == "LLM"
    assert kind_of({"a": "scripted"}) == "scripted" and kind_of({}) == "unknown"
    assert kind_of({"a": "scripted", "b": "noop"}) == "mixed"


def test_compare_endpoint(tmp_path):
    from civitas.server import serve
    _single(tmp_path, "a")
    _single(tmp_path, "b", warm=6, report=False)
    srv = serve(tmp_path, port=0, background=True)
    try:
        port = srv.server_address[1]
        d = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/compare?runs=a,b,nope,..,a"))
        assert [r["run_id"] for r in d["runs"]] == ["a", "b"] and d["missing"] == ["nope", ".."]
        assert d["runs"][0]["status"] == "finished" and d["runs"][1]["warmup"] == 6
        again = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/compare?runs=b"))
        assert again["runs"][0] == d["runs"][1]                    # cached until the run's files change
        empty = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/compare"))
        assert empty == {"runs": [], "missing": []}
    finally:
        srv.shutdown()
