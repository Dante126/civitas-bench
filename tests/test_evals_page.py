"""The Evals page's server endpoints (server.py): the model registry, API keys, environment versions, starting and
stopping evals, the leaderboard. bench/models.py, bench/envs.py and evals.prepare / progress / leaderboard are
replaced by fakes with the plan's return shapes (docs/plans/2026-10-10-plug-in-model-evals-plan.md): no model is
called, no game is played."""
import json
import os
import signal
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from civitas import server
from civitas.bench import evals as E

ROOT = Path(__file__).resolve().parents[1]
SECRET = "sk-test-0123456789-secret"


@dataclass
class FakeEntry:
    key: str
    provider: str = "arcee"
    id: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra_body: dict = field(default_factory=dict)
    notes: str = ""

    @property
    def served_id(self):
        return self.id or self.key

    def to_dict(self):
        out = {"key": self.key, "provider": self.provider}
        for k in ("id", "temperature", "max_tokens", "notes"):
            if getattr(self, k) not in (None, ""):
                out[k] = getattr(self, k)
        if self.extra_body:
            out["extra_body"] = self.extra_body
        return out


def fake_models(calls):
    reg = {"trinity-large-thinking": FakeEntry("trinity-large-thinking", temperature=0.4, max_tokens=16000),
           "gpt-5": FakeEntry("gpt-5", provider="openai", notes="third party")}
    m = types.ModuleType("civitas.bench.models")
    m.MODEL_KEYS = ("provider", "id", "temperature", "max_tokens", "extra_body", "notes")
    m.load_registry = lambda path=None: dict(sorted(reg.items()))
    m.registry_providers = lambda path=None: {"together": {"base_url": "https://api.together.xyz/v1",
                                                           "api_key_env": "TOGETHER_API_KEY"}}

    def validate(key, fields, path=None):
        errs = []
        if not key or not key.replace("-", "").replace(".", "").replace("_", "").isalnum():
            errs.append(f"bad model key {key!r}")
        errs += [f"{k} is an environment key: the same for every model" for k in fields if k == "max_steps"]
        errs += [f"unknown key {k}" for k in fields if k not in m.MODEL_KEYS and k != "max_steps"]
        return errs
    m.validate = validate

    def save_entry(key, fields, path=None):
        reg[key] = FakeEntry(key, **fields)
        return reg[key]
    m.save_entry = save_entry
    m.delete_entry = lambda key, path=None: reg.pop(key, None) is not None
    m.key_status = lambda e: {"provider": e.provider, "env": f"{e.provider.upper()}_API_KEY",
                              "set": bool(os.environ.get(f"{e.provider.upper()}_API_KEY"))}

    def set_api_key(env_var, value, dotenv=None):
        calls.append(("set_api_key", env_var))
        if value == "boom":
            raise ValueError(f"could not write {value} to .env")
        os.environ[env_var] = value
    m.set_api_key = set_api_key

    def preflight(entry, path=None):
        calls.append(("preflight", entry.key))
        return {"ok": True, "model": entry.key, "served_id": entry.served_id, "provider": entry.provider,
                "seconds": 0.4, "steps": [{"name": "key", "ok": True, "detail": "ARCEE_API_KEY set"},
                                          {"name": "request", "ok": True, "detail": "200 in 0.4 s"},
                                          {"name": "tool_call", "ok": True, "detail": "called end_turn"}],
                "error": None}
    m.preflight = preflight
    return m


def fake_envs(calls, delay=0.3):
    m = types.ModuleType("civitas.bench.envs")
    probe = {"fingerprint": "0123456789abcdef", "parts": {"config": "a", "prompts": "b", "tools": "c", "seeds": {}},
             "code": {"commit": "x", "dirty": False, "diff_hash": "y"}, "seconds": delay}

    def check(suite, *, probe_result=None, cache_dir=None):
        calls.append(("check", suite))
        time.sleep(delay)
        return {"env": "civitas-v1", "frozen": True, "match": False, "diff": ["briefings", "results"],
                "fingerprint": probe["fingerprint"], "probe": probe}
    m.check = check

    def freeze(name, suite, *, tag=True, probe_result=None):
        calls.append(("freeze", name, probe_result is not None))
        if name == "civitas-v1":
            raise ValueError("civitas-v1 exists already")
        return {"name": name, "fingerprint": probe["fingerprint"], "created": time.time()}
    m.freeze = freeze
    m.list_envs = lambda: [{"name": "civitas-v1", "fingerprint": probe["fingerprint"], "created": 1.0,
                            "commit": "x", "suite": "configs/eval_singles.toml"}]
    return m


LB = {"env": "civitas-v1", "envs": ["civitas-v1"], "baseline": "trinity-large-thinking",
      "rows": [{"model": "trinity-large-thinking", "eval_dir": "eval_trinity-large-thinking_20261010-100000",
                "created": 1.0, "n_units": 8, "vs_yardstick": {"mean": 1.2, "lo": 0.4, "hi": 2.0},
                "score": {"mean": 70.1, "lo": 68.0, "hi": 72.2}, "departments": {"mayor": 0.5}, "tokens": 12_000_000,
                "cost": None, "verdict": None, "paired": None}],
      "skipped": [{"eval_dir": "eval_x_20261009-100000", "reason": "drifted environment"}]}


@pytest.fixture
def fakes(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "civitas.bench.models", fake_models(calls))
    monkeypatch.setitem(sys.modules, "civitas.bench.envs", fake_envs(calls))
    monkeypatch.setattr(E, "DEFAULT_SUITE", "configs/eval_singles.toml", raising=False)
    monkeypatch.setattr(E, "leaderboard", lambda env=None, runs_dir="runs": {**LB, "env": env or LB["env"]},
                        raising=False)

    def prepare(model, *, suite=None, name=None, allow_drift=False, workers=None, runs_dir="runs", preflight=True,
                preflight_result=None):
        calls.append(("prepare", model, suite, allow_drift))
        errors = [] if model != "bad" else ["preflight failed: 401 Unauthorized"]
        if preflight_result is not None:                     # as evals.prepare: recorded, no request
            pf = {**preflight_result, "cached": preflight_result.get("cached") or True}
        elif preflight and not errors:
            calls.append(("preflight", model))               # the paid request
            pf = {"ok": True, "model": model, "seconds": 0.5, "steps": [{"name": "tool_call", "ok": True,
                                                                         "detail": "called end_turn"}], "error": None}
        else:
            pf = {"ok": None, "skipped": "turned off (--no-preflight)" if not preflight else "refused anyway",
                  "model": model, "steps": [], "error": None}
        return {"name": f"{model}", "dir": f"{runs_dir}/eval_{model}_20261010-120000", "workers": 2,
                "suite": {"name": "singles", "reference_model": "trinity-large-thinking"},
                "prices": {model: [1.0, 4.0]}, "model": model,
                "runs": [{"config": "c", "seed": s, "estimate": {"minutes": 10.0, "prompt_tokens": 1_000_000,
                                                                 "completion_tokens": 100_000}} for s in range(4)],
                "env": {"name": "civitas-v1", "fingerprint": "0123456789abcdef", "match": True, "diff": [],
                        "drift": False},
                "preflight": pf, "model_entry": None, "errors": errors}
    monkeypatch.setattr(E, "prepare", prepare, raising=False)

    def progress(eval_dir):
        calls.append(("progress", Path(eval_dir).name))
        return {"status": "running", "total": 4, "finished": 1, "running": 2, "failed": 0, "pending": 1,
                "tokens": 3_000_000, "eta_s": 600.0,
                "runs": [{"run_id": "r0", "seed": 0, "status": "finished", "month": 107, "months": 108}]}
    monkeypatch.setattr(E, "progress", progress, raising=False)
    monkeypatch.setattr(server, "_CODE", {})
    monkeypatch.setattr(server, "_ENV_CODE", {})
    return calls


def _server(tmp_path, **kw):
    httpd = server.serve(tmp_path / "runs", port=0, background=True, configs_dir=str(ROOT / "configs"), **kw)
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _get(base, path):
    try:
        r = urllib.request.urlopen(base + path)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _post(base, path, body, headers=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        r = urllib.request.urlopen(req)
        return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _wait(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_models_list_add_edit_delete_and_preflight(tmp_path, fakes):
    httpd, base = _server(tmp_path)
    try:
        code, d = _get(base, "/api/models")
        assert code == 200 and d["can_edit"]
        keys = [m["key"] for m in d["models"]]
        assert keys == ["gpt-5", "trinity-large-thinking"]
        trinity = d["models"][1]
        assert trinity["served_id"] == "trinity-large-thinking" and "set" in trinity["key_status"]
        assert trinity["latest"]["eval_dir"].startswith("eval_trinity") and trinity["latest"]["score"] == 1.2
        assert d["models"][0]["latest"] is None
        names = {p["name"]: p for p in d["providers"]}
        assert names["arcee"]["builtin"] and not names["together"]["builtin"]
        assert names["together"]["env"] == "TOGETHER_API_KEY"

        code, body = _post(base, "/api/models", {"key": "qwen3", "fields": {"provider": "together", "id": "Qwen/Q3",
                                                                            "temperature": 0.6}})
        assert code == 200 and json.loads(body)["model"]["id"] == "Qwen/Q3"
        code, body = _post(base, "/api/models", {"key": "qwen3", "fields": {"max_steps": 99}})
        assert code == 400 and "environment key" in json.loads(body)["errors"][0]
        code, body = _post(base, "/api/models", {"key": "bad key!", "fields": {}})
        assert code == 400 and json.loads(body)["errors"]
        assert "qwen3" in [m["key"] for m in _get(base, "/api/models")[1]["models"]]

        code, body = _post(base, "/api/models/preflight", {"key": "qwen3"})
        pf = json.loads(body)
        assert code == 200 and pf["ok"] and [s["name"] for s in pf["steps"]] == ["key", "request", "tool_call"]
        assert _post(base, "/api/models/preflight", {"key": "nope"})[0] == 404

        assert _post(base, "/api/models/delete", {"key": "qwen3"})[0] == 200
        assert _post(base, "/api/models/delete", {"key": "qwen3"})[0] == 404
    finally:
        httpd.shutdown()


def test_new_run_models_come_from_the_registry(tmp_path, fakes):
    opts = server.Launcher(tmp_path, ROOT / "configs").options()
    assert "openai/gpt-5" in opts["models"] and "arcee/trinity-large-thinking" in opts["models"]
    models, keys = server.launch_models()
    assert keys["together"] == "TOGETHER_API_KEY" and keys["arcee"] == "ARCEE_API_KEY"


def test_new_run_models_fall_back_without_the_registry(monkeypatch):
    monkeypatch.setitem(sys.modules, "civitas.bench.models", None)
    assert server.launch_models() == (server.MODELS, server.KEYS)


def test_endpoints_say_when_a_track_is_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "civitas.bench.models", None)
    httpd, base = _server(tmp_path)
    try:
        code, d = _get(base, "/api/models")
        assert code == 501 and d["unavailable"]
    finally:
        httpd.shutdown()


def test_writes_need_the_launcher_and_this_computer(tmp_path, fakes, monkeypatch):
    posts = [("/api/models", {"key": "x", "fields": {}}), ("/api/models/delete", {"key": "gpt-5"}),
             ("/api/models/preflight", {"key": "gpt-5"}), ("/api/keys", {"env": "ARCEE_API_KEY", "value": SECRET}),
             ("/api/env/freeze", {"name": "civitas-v2"}), ("/api/evals/plan", {"model": "gpt-5"}),
             ("/api/evals/start", {"model": "gpt-5"}), ("/api/evals/stop", {"dir": "eval_x"})]
    httpd, base = _server(tmp_path, allow_launch=False)
    try:
        code, d = _get(base, "/api/models")
        assert code == 200 and d["can_edit"] is False          # browsing works read-only
        assert _get(base, "/api/evals/leaderboard")[0] == 200
        for path, body in posts:
            code, text = _post(base, path, body)
            assert code == 403, (path, text)
            assert SECRET not in text
    finally:
        httpd.shutdown()
    assert not [c for c in fakes if c[0] in ("set_api_key", "preflight", "freeze", "prepare")]

    httpd, base = _server(tmp_path)
    try:
        code, text = _post(base, "/api/keys", {"env": "ARCEE_API_KEY", "value": SECRET},
                           headers={"Origin": "http://evil.example"})
        assert code == 403 and "another site" in text
        # DNS rebinding: a page on rebind.example resolving to 127.0.0.1 sends its own name as Host and Origin
        port = base.rsplit(":", 1)[1]
        rebind = {"Host": f"rebind.example:{port}", "Origin": f"http://rebind.example:{port}"}
        for path, body in [*posts, ("/api/launch", {"map": "city"}), ("/api/stop", {"run_id": "x"})]:
            code, text = _post(base, path, body, headers=rebind)
            assert code == 403, (path, text)
        assert _post(base, "/api/keys", {"env": "ARCEE_API_KEY", "value": SECRET},
                     headers={"Origin": "null"})[0] == 403
        monkeypatch.setattr(server, "is_local", lambda host: False)
        assert _get(base, "/api/models")[1]["can_edit"] is False
        for path, body in posts:
            assert _post(base, path, body)[0] == 403, path
    finally:
        httpd.shutdown()
    assert not [c for c in fakes if c[0] in ("set_api_key", "preflight", "freeze", "prepare")]


def test_api_keys_are_written_but_never_sent_back(tmp_path, fakes, monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    httpd, base = _server(tmp_path)
    try:
        code, text = _post(base, "/api/keys", {"env": "TOGETHER_API_KEY", "value": SECRET},
                           headers={"Origin": base})
        assert code == 200 and SECRET not in text and json.loads(text) == {"ok": True, "env": "TOGETHER_API_KEY",
                                                                           "set": True}
        assert ("set_api_key", "TOGETHER_API_KEY") in fakes
        for path in ("/api/models", "/api/launch"):
            assert SECRET not in urllib.request.urlopen(base + path).read().decode()
        together = {p["name"]: p for p in _get(base, "/api/models")[1]["providers"]}["together"]
        assert together["set"] is True and set(together) == {"name", "env", "set", "builtin", "base_url"}
        code, text = _post(base, "/api/keys", {"env": "PATH", "value": SECRET})
        assert code == 400 and SECRET not in text
        code, text = _post(base, "/api/keys", {"env": "TOGETHER_API_KEY", "value": "boom"})
        assert code == 400 and "boom" not in text
        assert _post(base, "/api/keys", {"env": "TOGETHER_API_KEY", "value": "a\nb"})[0] == 400
    finally:
        httpd.shutdown()
        os.environ.pop("TOGETHER_API_KEY", None)


def test_env_check_computes_in_the_background_then_answers_from_its_cache(tmp_path, fakes):
    httpd, base = _server(tmp_path)
    try:
        code, d = _get(base, "/api/env")
        assert code == 200 and d["computing"]
        assert _wait(lambda: "computing" not in _get(base, "/api/env")[1])
        d = _get(base, "/api/env")[1]
        assert d["env"] == "civitas-v1" and d["match"] is False and d["diff"] == ["briefings", "results"]
        assert "probe" not in d and d["suite"] == "eval_singles.toml"
        assert [c for c in fakes if c[0] == "check"] == [("check", str(ROOT / "configs" / "eval_singles.toml"))]
        _get(base, "/api/env")
        assert len([c for c in fakes if c[0] == "check"]) == 1            # cached for this code state
        assert _get(base, "/api/env?refresh=1")[1]["computing"]
        assert _wait(lambda: "computing" not in _get(base, "/api/env")[1])
        assert len([c for c in fakes if c[0] == "check"]) == 2

        code, text = _post(base, "/api/env/freeze", {"name": "civitas-v2"})
        assert code == 200 and json.loads(text)["env"]["name"] == "civitas-v2"
        assert ("freeze", "civitas-v2", True) in fakes                    # reused the probe it had
        code, text = _post(base, "/api/env/freeze", {"name": "civitas-v1"})
        assert code == 400 and "exists" in text
        assert _post(base, "/api/env/freeze", {"name": "../x"})[0] == 400
        assert _get(base, "/api/env")[1]["computing"]                     # a freeze checks again
        assert _get(base, "/api/envs")[1][0]["name"] == "civitas-v1"
    finally:
        httpd.shutdown()


def test_leaderboard_scorecard_and_suites(tmp_path, fakes):
    httpd, base = _server(tmp_path)
    try:
        code, d = _get(base, "/api/evals/leaderboard?env=civitas-v1")
        assert code == 200 and d["rows"][0]["vs_yardstick"]["lo"] == 0.4 and d["skipped"][0]["reason"]
        ev = tmp_path / "runs" / "eval_m_20261010-100000"
        ev.mkdir(parents=True)
        (ev / "scorecard.md").write_text("# Eval m\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
        code, d = _get(base, "/api/evals/scorecard?dir=eval_m_20261010-100000")
        assert code == 200 and d["markdown"].startswith("# Eval m")
        for bad in ("..", "eval_m_20261010-100000/../..", "runs", "eval_missing"):
            assert _get(base, f"/api/evals/scorecard?dir={bad}")[0] == 404
        code, d = _get(base, "/api/evals/suites")
        files = {s["file"]: s for s in d}
        assert files["eval_singles.toml"]["default"] and files["eval_singles.toml"]["units"] == 8
        assert "baseline.toml" not in files
    finally:
        httpd.shutdown()


def test_plan_reports_the_estimate_and_refusals(tmp_path, fakes):
    httpd, base = _server(tmp_path)
    try:
        code, text = _post(base, "/api/evals/plan", {"model": "gpt-5", "suite": "eval_singles.toml"})
        p = json.loads(text)
        assert code == 200 and p["runs"] == 4 and p["errors"] == [] and p["env"]["match"]
        assert p["estimate"]["prompt_tokens"] == 4_000_000 and p["estimate"]["minutes"] == 20.0
        assert p["estimate"]["cost"] == pytest.approx(4.0 + 1.6)
        assert ("prepare", "gpt-5", str(ROOT / "configs" / "eval_singles.toml"), False) in fakes
        assert _post(base, "/api/evals/plan", {"model": "gpt-5", "suite": "../pyproject.toml"})[0] == 400
        code, text = _post(base, "/api/evals/start", {"model": "bad"})
        assert code == 400 and json.loads(text)["errors"] == ["preflight failed: 401 Unauthorized"]
        assert not httpd.jobs.jobs                                         # nothing was launched
    finally:
        httpd.shutdown()


CHILD = r"""
import os, subprocess, sys, time, json
runs, pids = sys.argv[1], sys.argv[2]
time.sleep(0.3)
d = os.path.join(runs, "eval_gpt-5_20261010-120001")
os.makedirs(d)
open(os.path.join(d, "eval.json"), "w").write(json.dumps({"name": "gpt-5", "runs": [], "status": "running",
                                                          "pid": os.getpid()}))
p = subprocess.Popen(["sleep", "60"])
open(pids, "w").write(f"{os.getpid()} {p.pid}")
time.sleep(60)
"""


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:                                    # a zombie child of this test process counts as gone
        return os.waitpid(pid, os.WNOHANG) == (0, 0)
    except ChildProcessError:
        return True


def test_start_launches_the_eval_and_stop_ends_its_process_group(tmp_path, fakes, monkeypatch):
    pids = tmp_path / "pids"
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    seen = []

    def command(self, model, suite, allow_drift, name, stamp=None, preflight=None):
        seen.append((model, suite, allow_drift, name, self.command.__wrapped__(self, model, suite, allow_drift, name)))
        return [sys.executable, str(script), str(self.runs_dir), str(pids)]
    command.__wrapped__ = server.EvalJobs.command
    monkeypatch.setattr(server.EvalJobs, "command", command)
    httpd, base = _server(tmp_path)
    try:
        code, text = _post(base, "/api/evals/start", {"model": "gpt-5", "allow_drift": True})
        r = json.loads(text)
        assert code == 200 and r["ok"] and r["job"] == "gpt-5", text
        real = seen[0][4]
        assert real[1:6] == ["-m", "civitas.cli", "eval", "--model", "gpt-5"] and "--allow-drift" in real
        assert real[real.index("--name") + 1] == "gpt-5" and "-s" not in real
        assert "--stamp" not in real                            # the fake prepare() plans no stamp
        stamped = command.__wrapped__(server.EvalJobs(tmp_path, tmp_path), "gpt-5", None, False, "gpt-5",
                                          "20261010-120000")
        assert stamped[-3:] == ["--stamp", "20261010-120000", "--no-preflight"]  # prepare()'s folder, one preflight
        passed = {"ok": True, "steps": [], "seconds": 0.5}
        handed = command.__wrapped__(server.EvalJobs(tmp_path, tmp_path), "gpt-5", None, False, "gpt-5",
                                     "20261010-120000", preflight=passed)
        assert handed[-2] == "--preflight-result" and json.loads(handed[-1]) == passed    # recorded by the eval
        assert "--preflight-result" not in command.__wrapped__(server.EvalJobs(tmp_path, tmp_path), "gpt-5", None,
                                                               False, "gpt-5", "20261010-120000",
                                                               preflight={"ok": None, "skipped": "x"})
        assert _wait(lambda: _get(base, "/api/evals/progress?dir=gpt-5")[1].get("eval_dir"))
        code, d = _get(base, "/api/evals/progress?dir=gpt-5")
        assert d["eval_dir"] == "eval_gpt-5_20261010-120001" and d["total"] == 4 and d["running"] is True
        assert _get(base, "/api/evals/progress?dir=eval_gpt-5_20261010-120001")[1]["finished"] == 1
        assert _get(base, "/api/evals/jobs")[1][0]["eval_dir"] == "eval_gpt-5_20261010-120001"
        assert _post(base, "/api/evals/start", {"model": "gpt-5"})[0] == 400       # one at a time per name
        assert _wait(lambda: pids.exists() and len(pids.read_text().split()) == 2)
        child, grandchild = map(int, pids.read_text().split())
        assert _alive(child) and _alive(grandchild)
        assert _post(base, "/api/evals/stop", {"dir": "eval_gpt-5_20261010-120001"})[0] == 200
        assert _wait(lambda: not _alive(grandchild) and not _alive(child), 15)
        assert _wait(lambda: not _get(base, "/api/evals/jobs")[1][0]["running"])
        assert _post(base, "/api/evals/stop", {"dir": "gpt-5"})[0] == 404           # already stopped
        assert _post(base, "/api/evals/stop", {"dir": "../x"})[0] == 404
    finally:
        httpd.shutdown()
        if pids.exists():
            for pid in map(int, pids.read_text().split()):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass


FAILING_CHILD = r"""
import sys, time
time.sleep(1.0)
print("[eval gpt-5] checking the environment")
print("civitas eval: refusing to start the eval:")
print("  - the code no longer matches environment civitas-v1 (changed: briefings)")
sys.exit(1)
"""


def test_an_eval_that_ends_before_its_eval_json_failed_to_start(tmp_path, fakes, monkeypatch):
    """A stamped eval's folder is named before the child creates it: progress says "starting" (not a 500) and, once
    the child exits without writing eval.json (its own check refused), "failed" with the log's tail."""
    script = tmp_path / "fail.py"
    script.write_text(FAILING_CHILD)
    plain = E.prepare

    def stamped(model, **kw):               # as the real prepare(): a stamp and the folder it names
        return {**plain(model, **kw), "stamp": "20261010-130000",
                "dir": f"{kw['runs_dir']}/eval_{model}_20261010-130000"}
    monkeypatch.setattr(E, "prepare", stamped)
    monkeypatch.setattr(server.EvalJobs, "command", lambda self, *a, **k: [sys.executable, str(script)])
    folder = "eval_gpt-5_20261010-130000"
    httpd, base = _server(tmp_path)
    try:
        code, text = _post(base, "/api/evals/start", {"model": "gpt-5"})
        assert code == 200 and json.loads(text)["eval_dir"] == folder, text     # known before the folder exists
        for ref in ("gpt-5", folder):
            code, d = _get(base, f"/api/evals/progress?dir={ref}")
            assert code == 200 and d["status"] == "starting" and d["eval_json"] is False and d["running"], d
        assert _wait(lambda: not _get(base, "/api/evals/jobs")[1][0]["running"])
        job = _get(base, "/api/evals/jobs")[1][0]
        assert job["eval_dir"] == folder and job["eval_json"] is False and job["returncode"] == 1
        assert "refusing to start" in job["error"] and "no longer matches" in job["error"]
        assert "no longer matches" in job["log_tail"]
        for ref in ("gpt-5", folder):
            code, d = _get(base, f"/api/evals/progress?dir={ref}")
            assert code == 200 and d["status"] == "failed" and "no longer matches" in d["error"], d
        assert not [c for c in fakes if c[0] == "progress"]                    # never read a missing eval.json
    finally:
        httpd.shutdown()


def test_a_passing_preflight_is_reused_for_ten_minutes(tmp_path, fakes, monkeypatch):
    handed = []

    def command(self, model, suite, allow_drift, name, stamp=None, preflight=None):
        handed.append(preflight)
        return [sys.executable, "-c", "pass"]
    monkeypatch.setattr(server.EvalJobs, "command", command)
    paid = lambda: len([c for c in fakes if c[0] == "preflight"])
    plan = lambda body: json.loads(_post(base, "/api/evals/plan", body)[1])
    httpd, base = _server(tmp_path)
    try:
        p = plan({"model": "gpt-5"})
        assert paid() == 1 and p["preflight"]["ok"] and not p["preflight"].get("cached")
        # the dialog again, another suite, the drift toggle: no new request
        for body in ({"model": "gpt-5"}, {"model": "gpt-5", "allow_drift": True},
                     {"model": "gpt-5", "suite": "eval_singles.toml"}):
            p = plan(body)
            assert p["preflight"]["ok"] and p["preflight"]["cached"] and p["preflight"]["checked"], p
        assert paid() == 1
        code, text = _post(base, "/api/evals/start", {"model": "gpt-5"})
        assert code == 200 and paid() == 1, text
        assert handed[-1]["ok"] and handed[-1]["cached"] and not handed[-1].get("skipped")   # the eval records it
        # Check again: a fresh request, which is then the cached one
        p = plan({"model": "gpt-5", "fresh_preflight": True})
        assert paid() == 2 and not p["preflight"].get("cached")
        assert plan({"model": "gpt-5"})["preflight"]["cached"] and paid() == 2
        # Test: always a request; plan then reuses it
        assert json.loads(_post(base, "/api/models/preflight", {"key": "trinity-large-thinking"})[1])["ok"]
        assert json.loads(_post(base, "/api/models/preflight", {"key": "trinity-large-thinking"})[1])["ok"]
        assert paid() == 4
        assert plan({"model": "trinity-large-thinking"})["preflight"]["cached"] and paid() == 4
        # an edited registry entry asks again
        assert _post(base, "/api/models", {"key": "gpt-5", "fields": {"provider": "openai", "notes": "edited"}})[0] \
            == 200
        assert not plan({"model": "gpt-5"})["preflight"].get("cached") and paid() == 5
        # ten minutes later
        httpd.jobs.preflights.ttl = 0
        assert not plan({"model": "gpt-5"})["preflight"].get("cached") and paid() == 6
        # a refused plan (no preflight) keeps nothing
        assert plan({"model": "bad"})["errors"] and paid() == 6
        assert httpd.jobs.preflights.get("bad") is None
    finally:
        httpd.shutdown()


def _write_eval(runs: Path, name: str, pid, runs_list=()):
    d = runs / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval.json").write_text(json.dumps({"name": name, "runs": list(runs_list), "status": "running", "pid": pid}))
    return d


def test_stop_signals_only_live_civitas_processes(tmp_path, fakes):
    runs = tmp_path / "runs"
    sleeper = subprocess.Popen(["sleep", "30"])
    evalish = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "civitas.cli", "eval"])
    gone = subprocess.Popen(["true"])
    gone.wait()
    httpd, base = _server(tmp_path)
    try:
        _write_eval(runs, "eval_other_1", sleeper.pid)              # a stale pid now held by something else
        code, text = _post(base, "/api/evals/stop", {"dir": "eval_other_1"})
        assert code == 409 and "isn't a `civitas eval` process" in json.loads(text)["error"], text
        time.sleep(0.3)
        assert sleeper.poll() is None
        _write_eval(runs, "eval_other_2", gone.pid)
        code, text = _post(base, "/api/evals/stop", {"dir": "eval_other_2"})
        assert code == 409 and "isn't running" in json.loads(text)["error"]
        _write_eval(runs, "eval_other_3", evalish.pid)
        assert _wait(lambda: server.civitas_process(evalish.pid, "eval") is None)
        code, text = _post(base, "/api/evals/stop", {"dir": "eval_other_3"})
        assert code == 200, text
        assert evalish.wait(10) != 0                                 # SIGINT
    finally:
        httpd.shutdown()
        for p in (sleeper, evalish):
            p.kill()
            p.wait()


def test_stop_runs_leaves_pids_that_are_not_civitas_runs(tmp_path):
    runs = tmp_path / "runs"
    other = subprocess.Popen(["sleep", "30"], start_new_session=True)
    run = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "civitas.cli", "run"],
                           start_new_session=True)
    _write_eval(runs, "eval_n_1", 0, [{"run_id": "r0", "status": "running", "pid": other.pid},
                                      {"run_id": "r1", "status": "running", "pid": run.pid},
                                      {"run_id": "r2", "status": "finished", "pid": 1}])
    jobs = server.EvalJobs(runs, tmp_path)
    jobs.jobs["n"] = {"proc": other, "dir": "eval_n_1", "before": set(), "model": "m", "stopping": True,
                      "started": time.time(), "log": tmp_path / "n.log"}
    try:
        assert _wait(lambda: server.civitas_process(run.pid, "run") is None)
        skipped = jobs._stop_runs("n")
        assert run.wait(10) == -signal.SIGTERM
        assert other.poll() is None and len(skipped) == 1 and skipped[0].startswith("r0: ")
        assert "civitas run" in skipped[0]
        assert jobs.state("n", jobs.jobs["n"])["stop_skipped"] == skipped
    finally:
        for p in (other, run):
            p.kill()
            p.wait()


def _get_h(base, path, headers):
    try:
        r = urllib.request.urlopen(urllib.request.Request(base + path, headers=headers))
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_env_check_never_probes_read_only_and_refresh_needs_this_computer(tmp_path, fakes, monkeypatch):
    checks = lambda: [c for c in fakes if c[0] == "check"]
    httpd, base = _server(tmp_path, allow_launch=False)
    try:
        code, d = _get(base, "/api/env")
        assert code == 200 and d["unavailable_readonly"] and "computing" not in d
        assert _get(base, "/api/env?refresh=1")[0] == 403
        time.sleep(0.2)
        assert not checks()
    finally:
        httpd.shutdown()
    httpd, base = _server(tmp_path)
    try:
        port = base.rsplit(":", 1)[1]
        assert _get_h(base, "/api/env?refresh=1", {"Host": f"rebind.example:{port}"})[0] == 403   # DNS rebinding
        assert _get_h(base, "/api/env?refresh=1", {"Origin": "http://evil.example"})[0] == 403
        monkeypatch.setattr(server, "is_local", lambda host: False)
        assert _get(base, "/api/env?refresh=1")[0] == 403
        time.sleep(0.2)
        assert not checks()
    finally:
        httpd.shutdown()


def test_env_check_keys_on_envs_code_state(fakes, monkeypatch):
    assert server.env_code_state() == server.code_state()          # envs without code_state(): the server's own
    monkeypatch.setattr(sys.modules["civitas.bench.envs"], "code_state", lambda root=None: "head:untracked",
                        raising=False)
    monkeypatch.setattr(server, "_ENV_CODE", {})
    assert server.env_code_state() == "head:untracked"


def test_cli_preflight_result():
    from civitas import cli
    assert cli._preflight_result(None) is None
    assert cli._preflight_result('{"ok": true, "steps": []}') == {"ok": True, "steps": []}
    for bad in ("nope", '{"ok": false}', "[1]"):
        with pytest.raises(SystemExit):
            cli._preflight_result(bad)
