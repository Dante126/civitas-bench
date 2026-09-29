"""The Problems tab's analysis (bench/problems.py) and its endpoint."""
import json

from civitas.bench.problems import run_problems, turn_problems


def call(tool, args, ok=True, **res):
    r = {"ok": ok, **res}
    if ok and "message" not in r:
        r["message"] = "done"
    return {"step": 0, "tool": tool, "args": json.dumps(args), "result": r}


def rec(month, role, calls, finish="end_turn"):
    return {"month": month, "role": role, "agent": "fleet:test", "results": [], "tool_trace": calls, "finish_reason": finish}


def test_a_failed_call_fixed_by_a_later_call_of_the_same_tool():
    ps = turn_problems(rec(3, "planning", [
        call("zone", {"zone": "R", "x1": 1}, ok=False, error="no empty land tiles in that area"),
        call("zone", {"zone": "R", "x1": 5})]))
    assert [(p["kind"], p["handled"], p["at"], p["cause"]) for p in ps] == [("failed action", "fixed", 2, "taken land")]


def test_repeated_escalated_and_dropped():
    ps = turn_problems(rec(3, "utilities", [
        call("build_power_plant", {"kind": "coal"}, ok=False, error="costs $4,000 but the utilities budget only has $0"),
        call("build_power_plant", {"kind": "coal"}, ok=False, error="costs $4,000 but the utilities budget only has $0"),
        call("build_water", {"kind": "pump"}, ok=False, error="(3,4) is not an empty land tile"),
        call("send_message", {"to": "mayor", "text": "need $4,000"}),
    ]))
    assert [p["handled"] for p in ps] == ["repeated", "escalated", "escalated"]
    assert ps[0]["cause"] == "money"
    ps = turn_problems(rec(3, "utilities", [call("build_water", {"kind": "pump"}, ok=False, error="(3,4) is not an empty land tile")]))
    assert ps[0]["handled"] == "dropped"


def test_guards_are_heeded_or_overridden_and_the_action_cap_is_no_problem():
    ps = turn_problems(rec(6, "mayor", [
        call("take_loan", {"amount": 5000}, ok=False, refused="loan", error="no first building is waiting"),
        call("allocate_budget", {"department": "fire", "amount": 2500}, ok=False, refused="reserve", error="that leaves the treasury at $303"),
        call("allocate_budget", {"department": "fire", "amount": 2500, "confirm": True}, override="reserve"),
        call("allocate_budget", {"department": "police", "amount": 1}, ok=False, error="action limit reached for this turn (8 successful actions); call end_turn"),
    ], finish="max_steps"))
    assert [(p["kind"], p["handled"]) for p in ps] == [("guard", "heeded"), ("guard", "overridden"), ("turn end", "ended")]
    assert ps[1]["at"] == 3 and ps[2]["cause"] == "step limit"


def test_run_rates_recurrence_and_hints(tmp_path):
    fail = lambda: call("zone", {"zone": "R"}, ok=False, error="no empty land tiles in that area")
    traj = tmp_path / "trajectories.jsonl"
    traj.write_text("".join(json.dumps(r) + "\n" for r in [
        rec(0, "planning", [fail()]), rec(3, "planning", [fail(), call("zone", {"zone": "C"})]), rec(3, "mayor", [])]))
    (tmp_path / "fleet_report.json").write_text(json.dumps({"hints": [{"role": "planning", "text": "zone beside roads"}]}))
    d = run_problems(traj, tmp_path / "fleet_report.json")
    assert [p["recurred"] for p in d["problems"]] == [True, False]
    pl = d["roles"]["planning"]
    assert (pl["problems"], pl["fixed"], pl["dropped"], pl["recurred"], pl["fixable"]) == (2, 1, 1, 1, 2)
    assert d["roles"]["mayor"]["problems"] == 0 and d["total"]["problems"] == 2
    assert d["hints"][0]["text"] == "zone beside roads"


def test_problems_endpoint(tmp_path):
    import threading, urllib.request
    from civitas.server import serve
    run = tmp_path / "r1"; run.mkdir()
    (run / "meta.json").write_text("{}"); (run / "frames.jsonl").write_text("")
    (run / "trajectories.jsonl").write_text(json.dumps(rec(0, "planning", [call("zone", {}, ok=False, error="no empty land tiles")])) + "\n")
    srv = serve(tmp_path, port=0, background=True)
    try:
        port = srv.server_address[1]
        d = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs/r1/problems"))
        assert d["total"]["problems"] == 1 and d["problems"][0]["handled"] == "dropped"
    finally:
        srv.shutdown()


def test_reasoning_splits_at_step_markers():
    from civitas.server import reasoning_steps, STEP_CHARS
    steps = reasoning_steps("[step 0] look at the plan\n[step 1] the loan was refused; use the treasury\n[step 3] " + "x" * (STEP_CHARS + 50))
    assert [(s["step"], s["text"][:12]) for s in steps] == [(0, "look at the "), (1, "the loan was"), (3, "xxxxxxxxxxxx")]
    assert steps[2]["chars"] == STEP_CHARS + 50 and len(steps[2]["text"]) == STEP_CHARS
    assert reasoning_steps("no markers here")[0] == {"step": 0, "text": "no markers here", "chars": 15}
    assert reasoning_steps("") == []
