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


# ------------------------------------------------------------------ plug-in evals: prepare, progress, leaderboard
@pytest.fixture()
def fake_llm(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    monkeypatch.setenv(KEY_ENV, "sk-fake")
    yield srv.server_address[1]
    srv.shutdown()


def _env_stub(match=True, frozen=True):
    def check(s, **kw):
        return {"env": s.env, "frozen": frozen, "match": match if frozen else None,
                "diff": [] if match else ["briefings"], "fingerprint": "f" * 16,
                "frozen_fingerprint": ("f" if match else "e") * 16 if frozen else None,
                "probe": {"seconds": 1.5, "cached": False, "code": {"commit": "abc"}}}
    return check


def test_prepare_from_the_registry(tmp_path, fake_llm):
    from civitas.bench import models as M
    reg = tmp_path / "models.toml"
    reg.write_text(f'[providers.fake]\nbase_url = "http://127.0.0.1:{fake_llm}/v1"\napi_key_env = "{KEY_ENV}"\n'
                   f'stream = false\nmax_retries = 0\n')
    M.save_entry("fake-model", {"provider": "fake", "id": "served-fake", "temperature": 0.2, "max_tokens": 900}, reg)
    suite = tmp_path / "s.toml"
    suite.write_text(f'name = "tiny"\n[eval]\nenv = "tiny-v1"\nbaseline_model = "base"\n[[runs]]\n'
                     f'config = "{ROOT}/configs/fleet_smoke_late.toml"\nseeds = [0, 1]\n')
    runs = tmp_path / "runs"
    calls = FakeLLM.calls
    ev = E.prepare("fake-model", suite=suite, registry=reg, runs_dir=runs, env_check=_env_stub(), stamp="T")
    assert ev["errors"] == [] and ev["model"] == "served-fake" and ev["provider"] == "fake"
    assert ev["model_entry"] == {"key": "fake-model", "provider": "fake", "id": "served-fake", "temperature": 0.2,
                                 "max_tokens": 900}
    assert ev["preflight"]["ok"] and FakeLLM.calls == calls + 1 and E.model_key(ev) == "fake-model"
    assert ev["env"]["name"] == "tiny-v1" and ev["env"]["match"] and not ev["env"]["drift"]
    assert Path(ev["dir"]).name == "eval_fake-model_T" and not runs.exists()          # nothing written
    assert ev["suite"]["env"] == "tiny-v1" and ev["suite"]["baseline_model"] == "base" and ev["sets"] == []
    cmd = E.command(ev, ev["runs"][0])
    for s in ("fleet.model=served-fake", "fleet.provider=fake", "fleet.temperature=0.2", "fleet.max_tokens=900",
              f"providers.fake.base_url=http://127.0.0.1:{fake_llm}/v1", f"providers.fake.api_key_env={KEY_ENV}"):
        assert s in cmd, s
    cfg = E.effective(Path(ev["runs"][0]["config_path"]), ev["runs"][0]["agent_args"], ev["runs"][0]["set_args"])
    assert set(_fleet_roles(cfg).values()) == {"served-fake"} and fleet_agent_config(cfg, "police").temperature == 0.2
    text = "\n".join(E.prepare_text(ev))
    assert "fake-model from the registry" in text and "Preflight: ok" in text and "tiny-v1: matches" in text
    # the code changed since the version: refused (no preflight spent), or recorded as drifted
    ev = E.prepare("fake-model", suite=suite, registry=reg, runs_dir=runs, env_check=_env_stub(match=False))
    assert any("no longer matches environment tiny-v1 (changed: briefings)" in e for e in ev["errors"])
    assert ev["preflight"]["skipped"] and FakeLLM.calls == calls + 1
    ev = E.prepare("fake-model", suite=suite, registry=reg, runs_dir=runs, env_check=_env_stub(match=False),
                   allow_drift=True)
    assert ev["errors"] == [] and ev["env"]["drift"] and ev["preflight"]["ok"]
    assert "recorded as drifted" in "\n".join(E.prepare_text(ev))
    ev = E.prepare("fake-model", suite=suite, registry=reg, runs_dir=runs, env_check=_env_stub(frozen=False))
    assert any("isn't frozen" in e for e in ev["errors"])
    ev = E.prepare("fake-model", suite=suite, registry=reg, runs_dir=runs, env_check=_env_stub(), dry_run=True)
    assert ev["errors"] == [] and ev["preflight"]["skipped"] == "dry run: no model call"
    # a preflight that passed earlier (the viewer's cache, --preflight-result): recorded as this eval's, no request
    before = FakeLLM.calls
    passed = {"ok": True, "model": "fake-model", "seconds": 0.3, "steps": [{"name": "key", "ok": True, "detail": "k"}],
              "error": None}
    ev = E.prepare("fake-model", suite=suite, registry=reg, runs_dir=runs, env_check=_env_stub(), preflight=False,
                   preflight_result=passed)
    assert ev["errors"] == [] and FakeLLM.calls == before
    assert ev["preflight"]["ok"] and ev["preflight"]["cached"] and not ev["preflight"].get("skipped")
    assert "passed earlier (cached)" in "\n".join(E.prepare_text(ev))
    # a failed preflight refuses
    os.environ.pop(KEY_ENV)
    ev = E.prepare("fake-model", suite=suite, registry=reg, runs_dir=runs, env_check=_env_stub())
    assert any(KEY_ENV in e for e in ev["errors"])


def test_prepare_without_the_registry(tmp_path, fake_llm):
    (tmp_path / "city.toml").write_text(_config(fake_llm))
    suite = tmp_path / "s.toml"
    suite.write_text(f'[[runs]]\nconfig = "{tmp_path / "city.toml"}"\nseeds = [0]\n')
    ev = E.prepare("fake", provider="fake", suite=suite, registry=tmp_path / "none.toml", runs_dir=tmp_path / "r")
    assert ev["errors"] == [] and ev["model_entry"] is None and ev["preflight"]["ok"]
    assert ev["env"] == {"name": None, "fingerprint": None, "frozen_fingerprint": None, "match": None, "diff": [],
                         "frozen": None, "drift": False}
    assert "won't be on a leaderboard" in "\n".join(E.prepare_text(ev))


def test_dry_run_with_an_environment(tmp_path, monkeypatch, capsys):
    from civitas.agents import llm_client

    def boom(*a, **k):
        raise AssertionError("a dry run called a model")
    monkeypatch.setattr(llm_client.ChatClient, "chat", boom)
    monkeypatch.setenv("ARCEE_API_KEY", "sk-test")
    suite = tmp_path / "s.toml"
    suite.write_text(f'name = "tiny"\n[eval]\nenv = "tiny-v1"\n[[runs]]\n'
                     f'config = "{ROOT}/configs/fleet_smoke_late.toml"\nseeds = [0, 1]\n')
    with pytest.raises(SystemExit) as e:
        cli.main(["eval", "-s", str(suite), "--model", "trinity-mini", "--dry-run", "--runs-dir", str(tmp_path / "r"),
                  "--set", "game.months=10", "--set", "game.warmup_months=6"])
    out = capsys.readouterr().out
    assert e.value.code == 2 and "Preflight: skipped (dry run: no model call)." in out
    assert "Environment tiny-v1: not frozen yet" in out and "trinity-mini from the registry" in out
    assert "--set fleet.temperature=0.4" in out and "isn't frozen" in out
    assert not list((tmp_path / "r").glob("eval_*"))


def test_progress(tmp_path):
    runs = tmp_path / "runs"
    now = time.time()
    rs = []
    for i, st in enumerate(("finished", "running", "pending")):
        rid = f"run{i}"
        rs.append({"run_id": rid, "config": "c.toml", "name": "c", "seed": i, "months": 108, "warmup": 96,
                   "cities": [], "status": st, "pid": os.getpid() if st == "running" else None,
                   "started": now - (600 if st == "finished" else 300) if st != "pending" else None,
                   "ended": now - 10 if st == "finished" else None, "estimate": {"minutes": 10.0}})
        if st != "pending":
            (runs / rid).mkdir(parents=True)
            usage = {"prompt_tokens": 100, "completion_tokens": 10}
            (runs / rid / "trajectories.jsonl").write_text(
                json.dumps({"agent": "fleet:m", "role": "mayor", "usage": usage}) + "\n"
                + json.dumps({"agent": "fleet:scripted", "role": "mayor", "usage": usage, "warmup": True}) + "\n")
    (runs / "run0" / "summary.json").write_text("{}")
    (runs / "run1" / "live.jsonl").write_text("".join(json.dumps({"i": k, "month": m, "kind": "month"}) + "\n"
                                                      for k, m in enumerate((100, 102))))
    ev = {"name": "p", "dir": str(runs / "eval_p_T"), "runs_dir": str(runs), "status": "running", "pid": os.getpid(),
          "started": now - 600, "workers": 1, "runs": rs, "suite": {}}
    E.save(ev)
    p = E.progress(ev["dir"])
    assert (p["status"], p["total"], p["finished"], p["running"], p["pending"], p["failed"]) == \
        ("running", 3, 1, 1, 1, 0)
    assert p["tokens"] == 220 and [r["month"] for r in p["runs"]] == [108, 102, None]
    assert [r["status"] for r in p["runs"]] == ["finished", "running", "pending"]
    # 6 of 12 months in 300 s: ~300 s left on the one slot, then the pending run at the finished one's 590 s
    assert p["eta_s"] == pytest.approx(300 + 590, abs=5)
    rs[1]["pid"] = 999999
    E.save({**ev, "pid": 999999})
    p = E.progress(ev["dir"])
    assert p["status"] == "stopped" and p["runs"][1]["status"] == "stopped" and p["failed"] == 1 and p["eta_s"] is None


def _lb_card(seed, delta, final=60.0):
    return {"run_id": f"r_s{seed}", "name": "late", "config": "configs/late.toml", "seed": seed, "region": False,
            "repeat": 1, "repeats": 1, "wall_time_s": 60.0, "prompts": "p1", "code": {"commit": "abc"},
            "status": "finished",
            "total": {"final": final + delta, "delta": {"city": delta, "population": 100 * delta},
                      "change": {"city": delta + 1, "population": 0}, "yard_change": {"city": 1.0, "population": 0},
                      "at_handover": {"city": 60.0, "population": 5000}, "role_delta": {"mayor": delta / 2},
                      "roles": {}, "teamwork": 80.0, "calls": None,
                      "tokens": {"turns": 32, "prompt": 1000, "completion": 100}}}


def _lb_eval(runs, label, model, deltas, created, env="tiny-v1", env_rec=None, **extra):
    ev = {"version": 1, "name": label, "model": model, "dir": str(runs / f"eval_{label}"), "runs_dir": str(runs),
          "created": created, "status": "finished",
          "suite": {"name": "tiny", "file": "s.toml", "description": "", "env": env, "baseline_model": "base"},
          "runs": [], "model_entry": {"key": model, "provider": "arcee"},
          "env": None if env is None else {"name": env, "fingerprint": "f", "match": True, "diff": [], "drift": False,
                                           **(env_rec or {})}, **extra}
    cards = [_lb_card(s, d) for s, d in enumerate(deltas)]
    ev["scorecard"] = {"runs": cards, "aggregate": E._agg(cards, ev), "by_config": {}}
    E.save(ev)
    return ev


def test_leaderboard(tmp_path):
    runs = tmp_path / "runs"
    d = [0.5, -0.5, 1.0, 0.0, 0.2]
    _lb_eval(runs, "base1", "base", d, 1)
    _lb_eval(runs, "m2-old", "m2", [x - 3 for x in d], 2)
    _lb_eval(runs, "m2-new", "m2", [x + 1 + 0.1 * i for i, x in enumerate(d)], 3)
    _lb_eval(runs, "drift", "m3", [x + 5 for x in d], 4, env_rec={"drift": True, "match": False})
    _lb_eval(runs, "other", "m4", d, 5, env="civitas-v0")
    _lb_eval(runs, "noenv", "m5", d, 6, env=None)
    _lb_eval(runs, "live", "m6", d, 7, status="running", pid=999999)
    lb = E.leaderboard("tiny-v1", runs)
    assert lb["env"] == "tiny-v1" and lb["envs"] == ["civitas-v0", "tiny-v1"] and lb["baseline"] == "base"
    assert [r["model"] for r in lb["rows"]] == ["m2", "base"] and [r["rank"] for r in lb["rows"]] == [1, 2]
    top, base = lb["rows"]
    assert top["eval_dir"] == "eval_m2-new" and top["n_units"] == 5      # the folder name, as list_evals' "dir"
    listed = {e["dir"] for e in E.list_evals(runs)}                      # (the Evals tab joins verdicts on it)
    assert {r["eval_dir"] for r in lb["rows"]} <= listed and {s["eval_dir"] for s in lb["skipped"]} <= listed
    assert top["vs_yardstick"]["mean"] == pytest.approx(1.44) and top["vs_yardstick"]["lo"] is not None
    assert top["score"]["mean"] == pytest.approx(61.44) and top["departments"]["mayor"] == pytest.approx(0.72)
    assert top["tokens"] == 5 * 1100 and top["verdict"].startswith("m2-new vs base1: m2-new is BETTER")
    assert top["paired"]["wins"] == 5 and base["baseline"] and base["verdict"] is None
    why = {s["eval_dir"]: s["reason"] for s in lb["skipped"]}
    assert why["eval_m2-old"].startswith("superseded by a newer eval of m2") and why["eval_drift"].startswith("drifted")
    assert why["eval_other"] == "environment civitas-v0, not tiny-v1" and "no environment" in why["eval_noenv"]
    assert why["eval_live"] == "not finished (stopped)" and "eval_base1" not in why
    md = E.leaderboard_markdown(lb)
    assert "# Leaderboard: tiny-v1" in md and "| 1 | m2 | +1.4 (" in md and "| 2 | base |" in md
    assert "| baseline |" in md and "## Not ranked" in md and "`civitas-v0`" in md
    p = E.write_leaderboard(lb, runs)
    assert p == runs / "leaderboard_tiny-v1.md" and p.read_text() == md
    assert [r["model"] for r in E.leaderboard("civitas-v0", runs)["rows"]] == ["m4"]
    assert E.leaderboard("civitas-v9", runs)["rows"] == []


def test_leaderboard_rows_split_the_score_and_count_failures(tmp_path):
    """civitas-v2: each row splits vs scripted into the population term and the quality terms and carries failed /
    refused actions per turn beside Teamwork; an eval saved before the split gets it from its run cards, one saved
    before the call counts reads None."""
    runs = tmp_path / "runs"
    ev = _lb_eval(runs, "base1", "base", [0.5, 1.0, 0.0], 1)
    for c in ev["scorecard"]["runs"]:
        c["total"]["change"]["population"] = 600
        c["total"]["calls"] = {k: 0 for k in E.CALLS} | {"turns": 32, "failed": 8, "refused": 4}
    ev["scorecard"]["aggregate"] = E._agg(ev["scorecard"]["runs"], ev)
    E.save(ev)
    old = _lb_eval(runs, "m-old", "m", [1.5, 2.0, 1.0], 2)
    for k in ("pop_term_delta", "quality_delta", "failed_per_turn", "refused_per_turn"):
        del old["scorecard"]["aggregate"][k]
    E.save(old)
    rows = {r["model"]: r for r in E.leaderboard("tiny-v1", runs)["rows"]}
    pt = 50 * ((5600 / 30000) ** 0.5 - (5000 / 30000) ** 0.5)
    b, m = rows["base"], rows["m"]
    assert b["pop_term"]["mean"] == pytest.approx(pt, abs=1e-3)
    assert b["quality"]["mean"] == pytest.approx(0.5 - pt, abs=1e-3)
    assert b["failed_per_turn"] == 0.25 and b["refused_per_turn"] == 0.125 and b["teamwork"] == 80.0
    assert m["pop_term"]["mean"] == 0.0 and m["quality"]["mean"] == pytest.approx(1.5)     # from its cards
    assert m["pop_term"]["lo"] is not None and m["failed_per_turn"] is None
    md = E.leaderboard_markdown(E.leaderboard("tiny-v1", runs))
    assert ("| · population term | · quality terms | final city score | units | Teamwork | failed / turn "
            "| refused / turn |") in md
    assert f"| {pt:+.1f} | {0.5 - pt:+.1f} | 60.5 (" in md and "| 80.0 | 0.25 | 0.12 |" in md and "| 80.0 | n/a | n/a |" in md


def _turn(tools, steps=None, calls=3, prompt=600, completion=60, **extra):
    return {"agent": "fleet:m", "role": "planning", "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
            "finish_reasons": ["tool_calls"] * calls, "steps": steps if steps is not None else len(tools),
            "tool_trace": [{"step": i, "tool": t} for i, t in enumerate(tools)], **extra}


def _efficiency_run(d: Path, turns: list[dict], max_steps=4):
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"config": {"fleet": {"max_steps": max_steps}}}))
    warm = {"agent": "fleet:scripted", "role": "mayor", "warmup": True, "usage": {"prompt_tokens": 10 ** 6},
            "finish_reasons": ["x"] * 9, "tool_trace": [{"tool": "inspect"}] * 9}
    (d / "trajectories.jsonl").write_text("".join(json.dumps(t) + "\n" for t in [warm, *turns]))


def test_run_efficiency_counts_calls_tools_and_capped_turns(tmp_path):
    """The efficiency view: model calls are API replies (finish_reasons), tool calls split into game actions,
    look-ups (every other tool but the turn controls) and turn controls; a turn that used max_steps is capped; the
    scripted warm-up counts nothing."""
    d = tmp_path / "r"
    _efficiency_run(d, [_turn(["find_sites", "inspect", "build_road", "zone"], calls=4),     # 4 steps: the cap
                        _turn(["map_overview", "send_message", "update_memory", "end_turn"], steps=3, calls=2,
                              prompt=400, completion=40),
                        _turn(["offer_deal", "end_turn"], calls=1, prompt=200, completion=20, role="north.mayor")])
    e = E.run_efficiency(d)
    assert e == {"turns": 3, "model_calls": 7, "tool_calls": 10, "actions": 3, "lookups": 3, "capped": 1,
                 "prompt": 1200, "completion": 120, "budget_turns": 0, "budget_cuts": 0, "budget_used": 0}
    assert E.run_efficiency(tmp_path / "missing") is None
    # cached on the file: a second read doesn't parse it again
    E._EFF_CACHE[str(d / "trajectories.jsonl")][1]["turns"] = 99
    assert E.run_efficiency(d)["turns"] == 99
    del E._EFF_CACHE[str(d / "trajectories.jsonl")]
    agg = E.efficiency_of([{"run_id": "r", "total": {}}, {"run_id": "r", "total": {}}, {"run_id": "r"}], tmp_path)
    assert agg["runs"] == 2 and agg["turns"] == 6 and agg["tokens_per_turn"] == 440
    assert agg["model_calls_per_turn"] == pytest.approx(7 / 3, abs=0.01) and agg["lookups_per_turn"] == 1.0
    assert agg["capped_share"] == pytest.approx(1 / 3, abs=1e-3) and agg["context_per_call"] == round(1200 / 7)
    assert agg["tokens_per_run"] == 1320 and agg["budget_cut_rate"] is None and agg["budget_used_mean"] is None
    assert E.efficiency_of([{"run_id": "nope", "total": {}}], tmp_path) is None


def test_efficiency_counts_token_budget_cuts(tmp_path):
    """civitas-v3: the turns played with a token budget, the share it cut and their mean units (run_efficiency,
    efficiency_of, a run card's and a leaderboard row's own budget_cut_rate / budget_used_mean)."""
    d = tmp_path / "r"
    _efficiency_run(d, [_turn(["inspect"], turn_budget=100_000, budget_used=100_000, finish_reason="turn_budget"),
                        _turn(["end_turn"], turn_budget=100_000, budget_used=40_000, finish_reason="end_turn"),
                        _turn(["end_turn"], turn_budget=100_000, budget_used=30_000, finish_reason="end_turn"),
                        _turn(["end_turn"], turn_budget=100_000, budget_used=10_000, finish_reason="end_turn")])
    e = E.run_efficiency(d)
    assert (e["budget_turns"], e["budget_cuts"], e["budget_used"]) == (4, 1, 180_000)
    agg = E.efficiency_of([{"run_id": "r", "total": {}}], tmp_path)
    assert agg["budget_cut_rate"] == 0.25 and agg["budget_used_mean"] == 45_000
    assert E._budget_fields(agg) == {"budget_cut_rate": 0.25, "budget_used_mean": 45_000}
    rows = dict(E._budget_rows(E._budget_fields(agg)))          # the scorecard's summary rows (none without a budget)
    assert E._budget_rows({"budget_cut_rate": None}) == []
    assert rows["turns cut by the token budget"] == "25.0%"
    assert rows["token budget used per turn (prompt chars / 4 + completion)"] == "45k"


def test_leaderboard_rows_carry_efficiency(tmp_path):
    """Each row's efficiency comes from its runs' trajectories (an eval saved before it has none in its cards), with
    vs scripted per 10M tokens of one run."""
    runs = tmp_path / "runs"
    ev = _lb_eval(runs, "base1", "base", [1.0, 2.0], 1)
    for c in ev["scorecard"]["runs"]:
        _efficiency_run(runs / c["run_id"], [_turn(["find_sites", "build_road"], prompt=900_000, completion=100_000)])
    row = E.leaderboard("tiny-v1", runs)["rows"][0]
    eff = row["efficiency"]
    assert eff["turns"] == 2 and eff["tokens_per_turn"] == 1_000_000 and eff["tokens_per_run"] == 1_000_000
    assert eff["vs_per_10m"] == pytest.approx(1.5 / 0.1) and row["vs_yardstick"]["mean"] == pytest.approx(1.5)
    assert row["budget_cut_rate"] is None and row["budget_used_mean"] is None       # no token budget (v2)
    for c in ev["scorecard"]["runs"]:                                               # civitas-v3: one turn of two cut
        _efficiency_run(runs / c["run_id"], [_turn(["inspect"], turn_budget=10, budget_used=12,
                                                   finish_reason="turn_budget"),
                                             _turn(["end_turn"], turn_budget=10, budget_used=4)])
    row = E.leaderboard("tiny-v1", runs)["rows"][0]
    assert row["budget_cut_rate"] == row["efficiency"]["budget_cut_rate"] == 0.5 and row["budget_used_mean"] == 8
    assert E.leaderboard("tiny-v1", tmp_path / "empty")["rows"] == []


def _city_frames(d: Path, n: int):
    d.mkdir(parents=True, exist_ok=True)
    (d / "frames.jsonl").write_text("".join(json.dumps({"month": m, "tiles": "."}) + "\n" for m in range(n)))


def test_cities_pick_each_models_best_run_or_one_seed(tmp_path):
    """The Cities view: per ranked model its best run (highest vs scripted) or its run on a seed, with the final
    frame's index; on a seed, the hand-over city every model started from; cached until a file changes."""
    runs = tmp_path / "runs"
    a = _lb_eval(runs, "base1", "base", [0.5, 2.0, 1.0], 1)
    b = _lb_eval(runs, "m-1", "m", [3.0, 1.0], 2)
    for ev in (a, b):
        for c in ev["scorecard"]["runs"]:
            c["run_id"] = f"{ev['name']}_{c['run_id']}"
            c["warmup"] = 4
            _city_frames(runs / c["run_id"], 7)
        E.save(ev)
    out = E.cities("tiny-v1", None, runs)
    assert out["env"] == "tiny-v1" and out["seed"] is None and out["handover"] is None
    assert out["seeds"] == [0, 1, 2] and out["common_seeds"] == [0, 1]
    m, base = out["models"]
    assert (m["model"], m["rank"], m["run"]["seed"], m["run"]["vs"]) == ("m", 1, 0, 3.0)
    assert (base["model"], base["baseline"], base["run"]["seed"], base["run"]["run_id"]) == ("base", True, 1, "base1_r_s1")
    assert m["run"]["frame"] == 6 and m["run"]["handover_frame"] == 4 and m["run"]["city"] == 63.0
    assert [r["seed"] for r in base["runs"]] == [0, 1, 2]
    assert not out["yardstick"]["frames"] and "frames" in out["yardstick"]["reason"]
    one = E.cities("tiny-v1", 1, runs)
    assert [x["run"]["seed"] for x in one["models"]] == [1, 1]
    assert one["handover"] == {"run_id": "m-1_r_s1", "frame": 4, "month": 4, "city": 60.0, "population": 5000}
    two = E.cities("tiny-v1", 2, runs)
    assert two["models"][0]["run"] is None and two["models"][0]["missing"] == "no run on seed 2"
    assert two["models"][1]["run"]["seed"] == 2
    assert E.cities("tiny-v1", 1, runs) is one                           # cached
    _city_frames(runs / "m-1_r_s1", 9)                                   # the run's frames changed
    again = E.cities("tiny-v1", 1, runs)
    assert again is not one and again["models"][0]["run"]["frame"] == 8
    (runs / "base1_r_s2" / "frames.jsonl").unlink()
    assert E.cities("tiny-v1", 2, runs)["models"][1]["missing"] == "no frames.jsonl in its run folder"


def test_leaderboard_cli(tmp_path, capsys):
    runs = tmp_path / "runs"
    _lb_eval(runs, "base1", "base", [0.5, 1.0], 1)
    cli.main(["eval", "--leaderboard", "--env", "tiny-v1", "--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert "# Leaderboard: tiny-v1" in out and (runs / "leaderboard_tiny-v1.md").exists()


def test_sigterm_stops_the_runs(tmp_path):
    """Terminating the eval process (the viewer's Stop) stops its runs, which play in their own sessions."""
    runs = tmp_path / "runs"
    code = f"""
import sys
from pathlib import Path
from civitas.bench import evals as E
E.command = lambda ev, r: [sys.executable, "-c", "import time; time.sleep(120)"]
suite = Path({str(tmp_path / 's.toml')!r})
suite.write_text('[eval]\\ncaffeinate = false\\n[[runs]]\\nconfig = "{ROOT}/configs/fleet_smoke_late.toml"\\nseeds = [0, 1]\\n')
ev = E.plan(E.load_suite(suite), runs_dir={str(runs)!r}, name="stop", stamp="T")
try:
    E.execute(ev, verbose=False)
except KeyboardInterrupt:
    sys.exit(130)
"""
    proc = subprocess.Popen([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(ROOT)}, cwd=ROOT)
    f = runs / "eval_stop_T" / E.FILE
    pids = []
    try:
        for _ in range(200):
            time.sleep(0.1)
            try:
                saved = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            pids = [r.get("pid") for r in saved["runs"]]
            if all(pids):
                break
        assert all(pids) and all(E._alive(p) for p in pids)
        proc.terminate()
        assert proc.wait(timeout=30) == 130
        time.sleep(0.2)
        assert not any(E._alive(p) for p in pids)
        saved = json.loads(f.read_text())
        assert saved["status"] == "interrupted" and saved["stopped_by"] == "signal 15"
        assert [r["status"] for r in saved["runs"]] == ["interrupted", "interrupted"]
    finally:
        if proc.poll() is None:
            proc.kill()
        for p in pids:
            if p and E._alive(p):
                os.killpg(p, 9)


# ------------------------------------------------------------------ the environment on resume and during the eval
def _env_eval(tmp_path, monkeypatch, **env):
    """A saved, interrupted eval of fleet_smoke_late on environment tiny-v1 (no run started)."""
    from civitas.bench import envs as V
    monkeypatch.setenv("ARCEE_API_KEY", "sk-test")
    tmp_path.mkdir(parents=True, exist_ok=True)
    suite = tmp_path / "s.toml"
    suite.write_text(f'name = "tiny"\n[eval]\nenv = "tiny-v1"\ncaffeinate = false\n[[runs]]\n'
                     f'config = "{ROOT}/configs/fleet_smoke_late.toml"\nseeds = [0, 1]\n')
    ev = E.plan(E.load_suite(suite), model="trinity-mini", runs_dir=tmp_path / "runs", sets=["game.act_every=3"],
                name="envy", stamp="T", workers=1)
    ev["env"] = {"name": "tiny-v1", "fingerprint": "f" * 16, "match": True, "diff": [], "frozen": True,
                 "drift": False, "allow_drift": False, "code": V.code_info(), **env}
    ev["status"] = "interrupted"
    E.save(ev)
    return ev


def _recheck(match=True, env="tiny-v1", seen=None):
    from civitas.bench import envs as V

    def check(s, **kw):
        if seen is not None:
            seen.append(kw)
        if match is None:
            raise RuntimeError("probe blew up")
        return {"env": env, "frozen": True, "match": match, "diff": [] if match else ["model"],
                "fingerprint": ("f" if match else "e") * 16, "frozen_fingerprint": "f" * 16,
                "probe": {"seconds": 1.0, "cached": True, "code": V.code_info()}}
    return check


def test_resume_rechecks_the_environment(tmp_path, monkeypatch):
    from civitas.bench import envs as V
    ev = _env_eval(tmp_path, monkeypatch)
    started = []
    monkeypatch.setattr(E, "execute", lambda ev, **kw: started.append(ev) or ev)
    seen = []
    out = E.resume(ev["dir"], verbose=False, env_check=_recheck(seen=seen))
    assert len(started) == 1 and seen[0]["sets"] == ["game.act_every=3"] and seen[0]["agents"] == []
    assert Path(seen[0]["cache_dir"]) == tmp_path / "runs" / "_cache"
    r = out["env"]["resumes"][-1]
    assert r["ok"] and r["match"] and not out["env"]["drift"] and out["env"]["run_code"] == V.code_state()
    # the code changed since: refused, and the refused check is in eval.json
    for check, why in ((_recheck(match=False), "no longer matches environment tiny-v1 (changed: model)"),
                       (_recheck(env="other-v2"), "the suite now names environment 'other-v2'"),
                       (_recheck(match=None), "the environment probe failed (RuntimeError: probe blew up)")):
        with pytest.raises(E.EvalError) as err:
            E.resume(ev["dir"], verbose=False, env_check=check)
        assert str(err.value).startswith("can't resume: ") and why in str(err.value)
        saved = E.load(ev["dir"])
        assert saved["env"]["resumes"][-1]["ok"] is False and not saved["env"]["drift"]
    assert len(started) == 1 and len(E.load(ev["dir"])["env"]["resumes"]) == 3   # the first pass wasn't saved here
    # an eval started with --allow-drift resumes as drifted
    ev = _env_eval(tmp_path / "b", monkeypatch, allow_drift=True)
    lines = []
    out = E.resume(ev["dir"], verbose=True, out=lines.append, env_check=_recheck(match=False))
    assert out["env"]["drift"] and out["env"]["resumes"][-1]["drift"]
    assert "resumed on other code" in out["env"]["drift_reason"] and any("resuming as drifted" in x for x in lines)
    # an eval on no version: no check
    ev = _env_eval(tmp_path / "c", monkeypatch, name=None)
    out = E.resume(ev["dir"], verbose=False, env_check=_recheck(match=None))
    assert "resumes" not in out["env"]


def test_a_code_change_during_the_eval_marks_it_drifted(tmp_path, monkeypatch):
    from civitas.bench import envs as V
    monkeypatch.setattr(E, "command", lambda ev, r: [sys.executable, "-c", "pass"])
    monkeypatch.setattr(E, "finish", lambda ev: ev)
    monkeypatch.setattr(E, "POLL_S", 0.05)
    ev = _env_eval(tmp_path / "a", monkeypatch)            # workers 1: the second run starts after the first ends
    ev = E.execute(ev, verbose=False)
    assert not ev["env"]["drift"] and "code_changed" not in ev["env"]
    ev = _env_eval(tmp_path / "b", monkeypatch)
    now = iter([V.code_state(), "other:edited"])
    monkeypatch.setattr(V, "code_state", lambda root=V.ROOT: next(now))
    ev = E.execute(ev, verbose=False)
    e = E.load(ev["dir"])["env"]
    assert e["drift"] and e["drift_reason"] == "code changed during the eval"
    assert e["code_changed"]["before_run"] == ev["runs"][1]["run_id"] and e["code_changed"]["now"] == "other:edited"
    assert all(r["status"] != "pending" for r in ev["runs"])                     # the eval went on
