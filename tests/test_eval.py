"""Model evals (`civitas eval`, bench/evals.py): suites, plans and their refusals, a tiny end-to-end eval on a fake
model in subprocesses, resume, report, compare, adopt and /api/evals. No test calls a real model."""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from civitas import cli
from civitas.bench import evals as E
from civitas.config import apply_cli_overrides, load_config
from civitas.fleet.config import fleet_agent_config

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs/eval_late.toml"
KEY_ENV = "CIVITAS_FAKE_EVAL_KEY"


def _fleet_roles(cfg: dict) -> dict:
    """role -> the model its fleet agent plays, for every fleet role of `cfg` (a region: per city)."""
    return {x["role"]: x["model"] for x in E.lineup(cfg) if x["kind"] == "fleet"}


# ------------------------------------------------------------------ suites
def test_the_late_suite_parses():
    s = E.load_suite(SUITE)
    assert s.name == "late" and s.workers == 5 and s.caffeinate and s.reference_model == "trinity-large-thinking"
    assert [(e.config, e.seeds) for e in s.entries] == [("configs/fleet_smoke_late.toml", [0, 1, 2]),
                                                       ("configs/fleet_region_late.toml", [0, 1])]
    assert all(e.path.is_file() and e.minutes and e.prompt_tokens and e.completion_tokens for e in s.entries)


def test_seeds_and_bad_suites(tmp_path):
    assert E.parse_seeds([0, 2]) == [0, 2] and E.parse_seeds(3) == [3] and E.parse_seeds("0,3-5") == [0, 3, 4, 5]
    for bad in ("a-b", True, [0, "1"]):
        with pytest.raises(E.EvalError):
            E.parse_seeds(bad)
    p = tmp_path / "s.toml"
    p.write_text('name = "x"\n[game]\nmonths = 3\n')
    with pytest.raises(E.EvalError, match="not a suite"):
        E.load_suite(p)
    p.write_text('[[runs]]\nconfig = "configs/nope.toml"\nseeds = [0]\n')
    with pytest.raises(E.EvalError, match="not found"):
        E.load_suite(p)
    p.write_text('[[runs]]\nconfig = "configs/fleet_smoke_late.toml"\n')
    with pytest.raises(E.EvalError, match="no seeds"):
        E.load_suite(p)
    p.write_text('[eval]\nworkers = 0\n[[runs]]\nconfig = "configs/fleet_smoke_late.toml"\nseeds = "0-1"\n')
    with pytest.raises(E.EvalError, match="workers"):
        E.load_suite(p)


def test_the_launcher_leaves_suites_out(tmp_path):
    from civitas.server import Launcher
    files = {c["file"] for c in Launcher(tmp_path, ROOT / "configs").configs()}
    assert "fleet_smoke_late.toml" in files and "eval_late.toml" not in files


# ------------------------------------------------------------------ plans, overrides, refusals
def test_the_model_reaches_every_fleet_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCEE_API_KEY", "sk-test")
    ev = E.plan(E.load_suite(SUITE), model="trinity-mini", runs_dir=tmp_path)
    assert len(ev["runs"]) == 5 and ev["model"] == "trinity-mini" and ev["provider"] == "arcee"
    assert ev["name"] == "trinity-mini" and Path(ev["dir"]).name.startswith("eval_trinity-mini_")
    for r in ev["runs"]:
        cfg = E.effective(Path(r["config_path"]), r["agent_args"], r["set_args"])
        models = _fleet_roles(cfg)
        assert len(models) == (16 if r["region"] else 8) and set(models.values()) == {"trinity-mini"}, models
        assert r["fleet_turns"] == (64 if r["region"] else 32)
        assert "--set" in E.command(ev, r) and "fleet.model=trinity-mini" in E.command(ev, r)
    assert E.check(ev) == []
    # a role that names its own model (and provider) gets the asked one too; the playbook stays
    cfg = load_config(ROOT / "configs/fleet_smoke_late.toml")
    cfg["agents"]["police"]["model"] = "other"
    cfg["agents"]["fire"].update(model="other", provider="openrouter")
    agents, sets = E.model_overrides(cfg, "m2", None)
    assert sets == ["fleet.model=m2"] and sorted(agents) == ["fire=fleet:openrouter/m2", "police=fleet:arcee/m2"]
    out = apply_cli_overrides(cfg, agents, sets)
    assert set(_fleet_roles(out).values()) == {"m2"}
    assert fleet_agent_config(out, "police").playbook == fleet_agent_config(cfg, "police").playbook != ""
    agents, sets = E.model_overrides(cfg, "m2", "ollama")
    assert sets == ["fleet.model=m2", "fleet.provider=ollama"] and "fire=fleet:ollama/m2" in agents


def test_refusals(monkeypatch, tmp_path):
    monkeypatch.delenv("ARCEE_API_KEY", raising=False)
    ev = E.plan(E.load_suite(SUITE), model="trinity-mini", runs_dir=tmp_path)
    errs = E.check(ev)
    assert len(errs) == 1 and "ARCEE_API_KEY" in errs[0], errs
    with pytest.raises(E.EvalError, match="ARCEE_API_KEY"):
        E.start(ev, verbose=False)
    assert not Path(ev["dir"]).exists()
    monkeypatch.setenv("ARCEE_API_KEY", "sk-test")
    # a fleet agent pinned to the scripted baseline while a model is asked for
    cfg = tmp_path / "pinned.toml"
    cfg.write_text((ROOT / "configs/fleet_smoke_late.toml").read_text().replace(
        '[agents.police]\nkind = "fleet"', '[agents.police]\nkind = "fleet"\nmodel = "scripted"'))
    suite = tmp_path / "s.toml"
    suite.write_text(f'[[runs]]\nconfig = "{cfg}"\nseeds = [0]\n')
    errs = E.check(E.plan(E.load_suite(suite), model="trinity-mini", runs_dir=tmp_path))
    assert len(errs) == 1 and "police=scripted" in errs[0] and "pinned" in errs[0], errs
    # no warm-up, nothing model-backed, an unknown provider
    suite.write_text('[[runs]]\nconfig = "configs/baseline.toml"\nseeds = [0]\n')
    errs = E.check(E.plan(E.load_suite(suite), runs_dir=tmp_path))
    assert any("no warm-up" in e for e in errs) and any("nothing to evaluate" in e for e in errs)
    suite.write_text('[[runs]]\nconfig = "configs/fleet_smoke_late.toml"\nseeds = [0]\n')
    errs = E.check(E.plan(E.load_suite(suite), model="x", provider="nowhere", runs_dir=tmp_path))
    assert any("unknown provider 'nowhere'" in e for e in errs), errs


def test_dry_run_runs_nothing(monkeypatch, tmp_path, capsys):
    def boom(*a, **k):
        raise AssertionError("a dry run started a process")
    monkeypatch.setenv("ARCEE_API_KEY", "sk-test")
    monkeypatch.setattr(E.subprocess, "Popen", boom)
    cli.main(["eval", "-s", str(SUITE), "--model", "trinity-mini", "--dry-run", "--runs-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "5 runs, model trinity-mini (provider arcee), 5 at a time" in out
    assert "224 model turns" in out and "this model's will differ" in out and "Nothing was run" in out
    assert out.count("_trinity-mini  ") == 5 and "--set fleet.model=trinity-mini" in out   # 5 table lines
    assert list(tmp_path.iterdir()) == []
    monkeypatch.delenv("ARCEE_API_KEY")
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)   # a checkout's ./.env would supply the key
    with pytest.raises(SystemExit) as e:
        cli.main(["eval", "-s", str(SUITE), "--model", "trinity-mini", "--dry-run", "--runs-dir", str(tmp_path)])
    assert e.value.code == 2 and "would refuse to start" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_estimates():
    assert E.makespan([10, 10, 10, 24, 24], 5) == 24 and E.makespan([10, 10, 10, 24, 24], 2) == 44
    assert E.makespan([], 3) == 0


# ------------------------------------------------------------------ end to end, on a fake model
def _end_turn(i):
    return {"id": f"c{i}", "type": "function", "function": {"name": "end_turn",
                                                            "arguments": json.dumps({"report": "all quiet"})}}


class FakeLLM(BaseHTTPRequestHandler):
    """Every fleet agent ends its turn at once (a refused end_turn gets another)."""
    calls = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLLM.calls += 1
        step = sum(m["role"] == "assistant" for m in body["messages"])
        msg = {"role": "assistant", "content": "", "tool_calls": [_end_turn(step)]}
        out = json.dumps({"choices": [{"message": msg, "finish_reason": "tool_calls"}],
                          "usage": {"prompt_tokens": 50, "completion_tokens": 10}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def _config(port: int, region: bool = False) -> str:
    roles = ["mayor", "planning", "utilities", "transport", "police", "fire", "health", "education"]
    return "\n".join([
        f'name = "{"eval_region" if region else "eval_city"}"', "[game]", "months = 8", "warmup_months = 6",
        "act_every = 3", *(['region = true', 'cities = ["north", "south"]'] if region else []),
        "[providers.fake]", f'base_url = "http://127.0.0.1:{port}/v1"', f'api_key_env = "{KEY_ENV}"',
        "stream = false", "max_retries = 0",
        "[fleet]", 'provider = "fake"', 'model = "fake"', "max_steps = 4",
        *[f'[agents.{r}]\nkind = "fleet"' for r in roles]]) + "\n"


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    """One eval of a cheap suite (a single city, seed 0 played twice, and a region, seed 0; 8 months of which 6
    warm-up), played in subprocesses against the fake model."""
    os.environ[KEY_ENV] = "sk-fake"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    base = tmp_path_factory.mktemp("eval")
    port = srv.server_address[1]
    (base / "city.toml").write_text(_config(port))
    (base / "region.toml").write_text(_config(port, region=True))
    (base / "suite.toml").write_text(
        'name = "tiny"\ndescription = "two late cities and a late region"\n[eval]\nworkers = 3\ncaffeinate = false\n'
        f'[eval.prices]\nfake = [1.0, 2.0]\n[[runs]]\nconfig = "{base / "city.toml"}"\nseeds = [0]\nrepeats = 2\nminutes = 0.1\n'
        f'prompt_tokens = 1000\ncompletion_tokens = 100\n[[runs]]\nconfig = "{base / "region.toml"}"\nseeds = [0]\n')
    runs = base / "runs"
    lines = []
    try:
        ev = E.start(E.plan(E.load_suite(base / "suite.toml"), model="fake", provider="fake", runs_dir=runs,
                            name="fake-v1"), out=lines.append)
        yield {"ev": ev, "runs": runs, "base": base, "lines": lines, "port": port}
    finally:
        srv.shutdown()
        os.environ.pop(KEY_ENV, None)


@pytest.mark.slow
def test_an_eval_end_to_end(evaluated):
    ev, runs = evaluated["ev"], evaluated["runs"]
    assert ev["status"] == "finished" and ev["model"] == "fake" and FakeLLM.calls > 0
    d = Path(ev["dir"])
    assert d.parent == runs and d.name.startswith("eval_fake-v1_")
    saved = json.loads((d / E.FILE).read_text())
    assert [r["status"] for r in saved["runs"]] == ["finished"] * 3 and saved["code"] is not None
    for r in saved["runs"]:
        rd = runs / r["run_id"]
        assert (rd / "summary.json").exists() and (rd / "yardstick.json").exists() and (rd / "fleet_report.json").exists()
        assert (runs / "_logs" / f"{r['run_id']}.log").exists() and r["returncode"] == 0
        meta = json.loads((rd / "meta.json").read_text())
        assert set(meta["agents"].values()) == {"fleet:fake"}
    assert [(r["seed"], r["repeat"]) for r in saved["runs"]] == [(0, 1), (0, 2), (0, 1)]
    assert "_eval_city_s0_r1_fake-v1" in saved["runs"][0]["run_id"] and "_s0_r2_" in saved["runs"][1]["run_id"]
    assert saved["runs"][2]["run_id"].endswith("_eval_region_s0_fake-v1")         # one repeat: no _r in the id
    a = saved["scorecard"]["aggregate"]
    assert a["runs"] == a["finished"] == a["scored"] == 3 and a["n_units"] == 2 and a["repeats"] == [1, 2]
    assert a["city_delta"]["n"] == 2 and a["city_delta"]["lo"] is not None and len(a["city_delta"]["values"]) == 3
    assert a["noise"] == {"sd": 0.0, "df": 1} and a["mde"]["1"] == 0.0      # the fake model plays every repeat alike
    assert set(a["units"]) == {"eval_city s0", "eval_region s0"} and a["units"]["eval_city s0"]["runs"] == 2
    assert a["tokens"]["turns"] == 8 + 8 + 16                          # one acting month (6) after the warm-up
    assert a["tokens"]["prompt"] >= 50 * 32 and a["usd"] > 0 and a["calls"]["turns"] == 32
    assert set(saved["scorecard"]["by_config"]) == {"eval_city", "eval_region"}
    reg = [c for c in saved["scorecard"]["runs"] if c["region"]][0]
    assert set(reg["by_city"]) == {"north", "south"} and reg["total"]["tokens"]["turns"] == 16
    assert all(reg["by_city"][c]["tokens"]["turns"] == 8 and reg["by_city"][c]["delta"] for c in ("north", "south"))
    for c in saved["scorecard"]["runs"]:
        t = c["total"]
        assert t["delta"]["city"] == pytest.approx(t["change"]["city"] - t["yard_change"]["city"], abs=1e-3)
        assert c["yardstick_matches"] is True and set(t["role_delta"]) >= {"mayor", "education"}
    card = (d / E.CARD).read_text()
    assert "# Eval fake-v1: fake on tiny" in card and "| city score vs scripted (mean, 95% CI) |" in card
    assert "| units (config + seed) x repeats | 2 x 1/2 |" in card and "| eval_city s0 r2 |" in card
    assert "**eval_region s0 (region)**" in card and "| · north |" in card and "cost at the suite's prices" in card
    assert any(x.startswith("[eval fake-v1] finished: eval_city s0 r1") for x in evaluated["lines"])


@pytest.mark.slow
def test_resume_reruns_what_did_not_finish(evaluated, capsys):
    ev, runs = evaluated["ev"], evaluated["runs"]
    first = ev["runs"][0]["run_id"]
    (runs / first / "summary.json").rename(runs / first / "summary.json.bak")      # as if it had died
    os.environ[KEY_ENV] = "sk-fake"
    ev2 = E.resume(ev["dir"], verbose=False)
    r = ev2["runs"][0]
    assert ev2["status"] == "finished" and r["run_id"] == first + "_a2" and r["attempts"][0]["run_id"] == first
    assert (runs / r["run_id"] / "summary.json").exists()
    assert [x["run_id"] for x in ev2["runs"][1:]] == [x["run_id"] for x in ev["runs"][1:]]   # finished: kept
    assert ev2["scorecard"]["aggregate"]["scored"] == 3
    card = (Path(ev["dir"]) / E.CARD).read_text()
    assert f"earlier attempts {first}" in card
    # an eval that still runs isn't resumed
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        saved = json.loads((Path(ev["dir"]) / E.FILE).read_text())
        saved.update(status="running", pid=sleeper.pid)
        (Path(ev["dir"]) / E.FILE).write_text(json.dumps(saved))
        with pytest.raises(E.EvalError, match="still running"):
            E.resume(ev["dir"], verbose=False)
    finally:
        sleeper.kill()
        sleeper.wait()
    saved.update(status="finished", pid=None)
    (Path(ev["dir"]) / E.FILE).write_text(json.dumps(saved))


@pytest.mark.slow
def test_report_adopt_and_compare(evaluated, capsys):
    ev, runs = evaluated["ev"], evaluated["runs"]
    (Path(ev["dir"]) / E.CARD).unlink()
    cli.main(["eval", "--report", ev["dir"]])
    assert "# Eval fake-v1" in capsys.readouterr().out and (Path(ev["dir"]) / E.CARD).exists()
    ids = [r["run_id"] for r in json.loads((Path(ev["dir"]) / E.FILE).read_text())["runs"]]
    cli.main(["eval", "-s", str(evaluated["base"] / "suite.toml"), "--adopt", *ids[:2], "--name", "again",
              "--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert "Adopted from runs" in out and "Suite runs not in this eval: eval_region s0." in out
    adopted = [d for d in runs.iterdir() if d.name.startswith("eval_again_")]
    assert len(adopted) == 1
    a2 = json.loads((adopted[0] / E.FILE).read_text())
    assert a2["adopted"] and a2["model"] == "fake" and [r["run_id"] for r in a2["runs"]] == ids[:2]
    assert [(r["repeat"], r["repeats"]) for r in a2["runs"]] == [(1, 2), (2, 2)]      # one unit, played twice
    assert all(r["code"] and r["code"]["inferred"] for r in a2["runs"])
    one = json.loads((Path(ev["dir"]) / E.FILE).read_text())["scorecard"]["by_config"]["eval_city"]
    assert a2["scorecard"]["aggregate"]["city_delta"] == one["city_delta"]
    with pytest.raises(SystemExit, match="given twice"):
        cli.main(["eval", "-s", str(evaluated["base"] / "suite.toml"), "--adopt", ids[0], ids[0], "--name", "x",
                  "--runs-dir", str(runs)])
    with pytest.raises(SystemExit, match="isn't in suite"):
        cli.main(["eval", "-s", str(SUITE), "--adopt", ids[0], "--name", "x", "--runs-dir", str(runs)])
    cli.main(["eval", "--compare", ev["dir"], str(adopted[0])])
    table = capsys.readouterr().out
    head = table.splitlines()[0]
    assert "fake-v1" in head and "again" in head and "city score vs scripted (mean, 95% CI)" in table
    assert "beats scripted" in table and "education vs scripted" in table
    assert "again vs fake-v1: only 1 shared unit, no verdict" in table
    assert "left out (only one eval has them): eval_region s0 (fake-v1)" in table


@pytest.mark.slow
def test_api_evals(evaluated):
    from civitas.server import serve
    srv = serve(evaluated["runs"], port=0, background=True)
    try:
        port = srv.server_address[1]
        evs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/evals"))
        mine = [e for e in evs if e["name"] == "fake-v1"][0]
        assert mine["model"] == "fake" and mine["suite"]["name"] == "tiny" and len(mine["runs"]) == 3
        assert all(r["status"] == "finished" for r in mine["runs"]) and mine["aggregate"]["scored"] == 3
        assert mine["dir"].startswith("eval_fake-v1_") and set(mine["by_config"]) == {"eval_city", "eval_region"}
        assert set(mine["units"]) == {"eval_city s0", "eval_region s0"} and mine["aggregate"]["n_units"] == 2
        runs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs"))
        assert runs and not any(r["run_id"].startswith("eval_") for r in runs)      # eval folders aren't runs
        board = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/leaderboard"))
        assert board
    finally:
        srv.shutdown()
