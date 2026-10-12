"""The Runs tab's list (/api/runs): each run's short line-up, start time and the eval folder that lists it."""
import json
import urllib.request

from civitas.server import eval_of_runs, lineup, serve

FLEET = " ".join(f"{r}=fleet:zai-org/glm-5.3-flash" for r in
                 ("mayor", "planning", "utilities", "transport", "police", "fire", "health", "education"))


def test_lineup_collapses_identical_roles():
    assert lineup({"signature": FLEET + " warmup:96 [autonomous]"}) == "8 × fleet · glm-5.3-flash · warm-up 96"
    assert lineup({"signature": FLEET + " [supervised]"}) == "8 × fleet · glm-5.3-flash · supervised"
    mixed = ("mayor=scripted planning=fleet:zai-org/glm-5.3-flash utilities=scripted transport=scripted "
             "police=scripted fire=scripted health=scripted education=scripted")
    assert lineup({"signature": mixed}) == "7 × scripted + planning: fleet · glm-5.3-flash"
    two = mixed.replace("transport=scripted", "transport=fleet:zai-org/glm-5.3-flash").replace(
        "fire=scripted", "fire=llm:trinity-mini")
    assert lineup({"signature": two}) == "5 × scripted + planning, transport: fleet · glm-5.3-flash + fire: llm · trinity-mini"
    region = {"signature": "mayor=scripted planning=scripted", "region": True, "cities": ["north", "south"]}
    assert lineup(region) == "2 × scripted · 2 cities"
    assert lineup({"signature": "showcase"}) == "showcase"
    assert lineup({}) == ""


def test_api_runs_has_lineup_start_and_eval(tmp_path):
    for rid, started in (("20261001-100000_a_s0", 1790000000.0), ("20261002-100000_b_s0", None)):
        d = tmp_path / rid
        d.mkdir()
        meta = {"name": rid.split("_")[1], "seed": 0, "signature": FLEET + " warmup:96 [autonomous]"}
        if started:
            meta["started"] = started
        (d / "meta.json").write_text(json.dumps(meta))
    old, new = tmp_path / "eval_m_20261001-090000", tmp_path / "eval_m-again_20261003-090000"
    for ev, ids in ((old, ["20261001-100000_a_s0"]), (new, ["20261001-100000_a_s0", "gone"])):
        ev.mkdir()
        (ev / "eval.json").write_text(json.dumps({"name": ev.name, "runs": [{"run_id": i} for i in ids]}))
    assert eval_of_runs(tmp_path) == {"20261001-100000_a_s0": new.name, "gone": new.name}   # the newest eval
    srv = serve(tmp_path, port=0, background=True)
    try:
        runs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/api/runs"))
    finally:
        srv.shutdown()
    by = {r["run_id"]: r for r in runs}
    assert set(by) == {"20261001-100000_a_s0", "20261002-100000_b_s0"}             # eval folders aren't runs
    a, b = by["20261001-100000_a_s0"], by["20261002-100000_b_s0"]
    assert a["eval"] == new.name and b["eval"] is None
    assert a["lineup"] == "8 × fleet · glm-5.3-flash · warm-up 96" and a["signature"].startswith("mayor=fleet:")
    assert a["started"] == 1790000000.0 and b["started"] > 0                     # meta.json's mtime without one


def test_leaderboard_skips_summaries_without_a_signature(tmp_path):
    from civitas.bench.runner import leaderboard
    (tmp_path / "_showcase").mkdir()
    (tmp_path / "_showcase" / "summary.json").write_text(json.dumps({"final_scores": {"city": 0}}))
    assert leaderboard(tmp_path) == []
