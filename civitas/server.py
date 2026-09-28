"""HTTP server: 3D web interface + replay API + run launcher + live API (stdlib only).

Endpoints
  GET  /api/runs                     list runs with status (running / finished / stopped / incomplete)
  GET  /api/runs/<id>/meta|summary   run metadata / final summary
  GET  /api/runs/<id>/frames?since=N replay frames from index N
  GET  /api/runs/<id>/turns?month=M  every agent turn of month M: prompt / briefing, reasoning, reply (+ a
                                     fleet turn's tool-call summary under "trace")
  GET  /api/runs/<id>/trace?month=M&role=R   one fleet turn's tool calls in order, each with its outcome
  GET  /api/leaderboard              finished runs grouped by agent line-up
  GET  /api/configs                  configs/*.toml available to launch
  GET  /api/processes                runs started from the browser
  POST /api/launch                   {config, seed, months?, act_every?, agents: {role: "kind[:provider/model]"}}
  POST /api/stop                     {run_id}
  GET  /api/live/turn?role=R         remote agents (only with `civitas run --serve`)
  POST /api/live/act
"""
from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

VIEWER_DIR = Path(__file__).parent / "viewer"
STALE_AFTER_S = 45 * 60      # a run with no new frame for this long is "incomplete"
# kind[:provider/model], as `--agent role=...` takes it; fleet without a model keeps the config's
AGENT_SPEC = re.compile(r"^(scripted|noop|random|(llm|fleet)(:[A-Za-z0-9._/-]+)?)$")
from .core.roles import ROLES as ROLE_SPECS, TURN_ORDER as ROLES  # noqa: E402

ACTIONS = {a for spec in ROLE_SPECS.values() for a in spec["actions"]}   # fleet tools that run a game action
# how the runner writes a trajectory record's first keys; a line that doesn't start like this is parsed whole
TURN_HEAD = re.compile(rb'\{"run_id": "(?:[^"\\]|\\.)*", "seed": [^,]*, "month": (-?\d+), "role": "((?:[^"\\]|\\.)*)"')


def _turn_key(line: bytes) -> tuple[int, str] | None:
    """(month, role) of one trajectories.jsonl line, from its head when it has the runner's key order."""
    try:
        m = TURN_HEAD.match(line)
        if m:
            return int(m.group(1)), json.loads(b'"' + m.group(2) + b'"')
        rec = json.loads(line)
        return int(rec["month"]), str(rec["role"])
    except (ValueError, KeyError, TypeError):
        return None


class TurnIndex:
    """Where each record of a run's trajectories.jsonl starts, by month and by (month, role), so a request reads
    only the lines it returns. Built on a run's first request and extended as a live run appends (complete lines
    only); a file that shrank or was replaced is indexed again, as is one whose record doesn't match its entry."""

    def __init__(self):
        self.lock = threading.Lock()
        self.files: dict[str, dict] = {}

    def _update(self, path: Path, fresh: bool) -> dict:
        st = path.stat()
        ix = self.files.get(str(path))
        if fresh or ix is None or ix["ino"] != st.st_ino or st.st_size < ix["end"]:
            ix = self.files[str(path)] = {"ino": st.st_ino, "end": 0, "month": {}, "turn": {}}
        if st.st_size > ix["end"]:
            with open(path, "rb") as fh:
                fh.seek(ix["end"])
                pos = ix["end"]
                for line in fh:
                    if not line.endswith(b"\n"):     # still being written
                        break
                    key = _turn_key(line)
                    if key is not None:
                        ix["month"].setdefault(key[0], []).append(pos)
                        ix["turn"].setdefault(key, pos)
                    pos += len(line)
                ix["end"] = pos
        return ix

    def records(self, path: Path, month: int, role: str | None = None) -> list[dict]:
        """Month `month`'s records in file order, or just `role`'s turn (a list of at most one)."""
        if not path.exists():
            return []
        for fresh in (False, True):
            with self.lock:
                ix = self._update(path, fresh)
                offs = list(ix["month"].get(month, [])) if role is None else \
                    [ix["turn"][(month, role)]] if (month, role) in ix["turn"] else []
            out = []
            with open(path, "rb") as fh:
                for off in offs:
                    fh.seek(off)
                    try:
                        out.append(json.loads(fh.readline()))
                    except ValueError:
                        out.append(None)
            if all(isinstance(r, dict) and r.get("month") == month and role in (None, r.get("role")) for r in out):
                return out
        return [r for r in out if isinstance(r, dict) and r.get("month") == month and role in (None, r.get("role"))]


def trace_call(n: int, call: dict) -> dict:
    """One tool_trace entry for the viewer: parsed args, whether it is a game action and ran, and its outcome:
    ok, failed (a failed action or a tool error), refused (a repeated failed call or a guard) or repeat (a look-up
    answered with an earlier result). `n` counts from 1, as the tools' "same as call N" notes do."""
    res = call.get("result") if isinstance(call.get("result"), dict) else {}
    args = call.get("args")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            pass                                    # the raw text the model sent
    tool = str(call.get("tool") or "")
    outcome = ("refused" if "refused" in res else "repeat" if "repeat" in res else
               "failed" if res.get("ok") is False else "ok")
    text = str(res.get("error") or res.get("message") or "")
    if outcome == "repeat":
        text = str(res["repeat"]) + (f": {text}" if text else "")
    out = {"n": n, "step": call.get("step"), "tool": tool, "args": args, "action": tool in ACTIONS,
           "ran": tool in ACTIONS and "message" in res and "refused" not in res, "outcome": outcome, "text": text,
           "finish_reason": call.get("finish_reason")}
    out.update({k: res[k] for k in ("cost", "budget_left", "refused") if k in res})
    return out


def trace_summary(rec: dict, calls: list[dict] | None = None) -> dict:
    """Counts for a fleet turn's header: calls, actions that ran (and failed), tool errors, refusals, repeats,
    money spent, model replies and how the turn ended."""
    if calls is None:
        calls = [trace_call(i + 1, c) for i, c in enumerate(rec.get("tool_trace") or [])]
    ran = [c for c in calls if c["ran"]]
    count = lambda outcome, cs=calls: sum(c["outcome"] == outcome for c in cs)
    return {"calls": len(calls), "actions": len(ran), "failed_actions": count("failed", ran),
            "errors": sum(c["outcome"] == "failed" and not c["ran"] for c in calls),
            "refused": count("refused"), "repeats": count("repeat"),
            "spent": round(sum(float(c.get("cost") or 0) for c in ran if c["outcome"] == "ok"), 2),
            "model_calls": len(rec.get("finish_reasons") or []), "finish_reason": rec.get("finish_reason"),
            "ended": rec.get("ended"), "length_stops": rec.get("length_stops") or 0,
            "recovered_calls": rec.get("recovered_calls") or 0}


class Launcher:
    """Starts `civitas run` subprocesses on behalf of the browser."""

    def __init__(self, runs_dir: Path, configs_dir: Path):
        self.runs_dir = runs_dir
        self.configs_dir = configs_dir
        self.procs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def configs(self) -> list[dict]:
        from .config import load_config
        from .fleet.config import fleet_agent_config
        out = []
        if not self.configs_dir.is_dir():
            return out
        for p in sorted(self.configs_dir.glob("*.toml")):
            try:
                cfg = load_config(p)
            except Exception as e:  # pragma: no cover
                out.append({"file": p.name, "error": str(e)})
                continue
            agents = {}
            for r in ROLES:
                a = cfg["agents"].get(r, {"kind": "scripted"})
                agents[r] = {"kind": a.get("kind", "scripted"), "model": a.get("model"),
                             "provider": a.get("provider", "arcee")}
                # what a fleet agent in this role runs: [fleet] + [agents.<role>] for a fleet role; a role
                # switched to fleet in the launcher drops its old spec (like `--agent role=fleet`), so [fleet] only
                src = cfg if agents[r]["kind"] == "fleet" else {"fleet": cfg.get("fleet") or {}}
                try:
                    fc = fleet_agent_config(src, r)
                    agents[r]["fleet"] = {"model": fc.model, "provider": fc.provider, "max_steps": fc.max_steps}
                except ValueError:
                    pass
                if agents[r]["kind"] == "fleet" and "fleet" in agents[r]:
                    agents[r].update(model=agents[r]["fleet"]["model"], provider=agents[r]["fleet"]["provider"])
            doc = ""
            try:
                first = [ln[1:].strip() for ln in p.read_text().splitlines() if ln.startswith("#")]
                doc = " ".join(first[:2])
            except OSError:
                pass
            out.append({"file": p.name, "name": cfg["name"], "doc": doc, "agents": agents,
                        "months": cfg["game"].get("months", 120), "act_every": cfg["game"].get("act_every", 1),
                        "remote": any(a["kind"] == "remote" for a in agents.values()),
                        "region": bool(cfg["game"].get("region"))})
        return out

    def launch(self, body: dict) -> dict:
        cfg_file = str(body.get("config", "baseline.toml"))
        path = (self.configs_dir / cfg_file).resolve()
        if path.parent != self.configs_dir.resolve() or not path.is_file():
            raise ValueError(f"unknown config {cfg_file!r}")
        seed = int(body.get("seed", 0))
        name = re.sub(r"[^A-Za-z0-9_-]", "", path.stem)[:40]
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{name}_s{seed}_ui"
        cmd = [sys.executable, "-m", "civitas.cli", "run", "-c", str(path), "--seed", str(seed),
               "--runs-dir", str(self.runs_dir), "--run-id", run_id]
        for role, spec in (body.get("agents") or {}).items():
            if role not in ROLES or not spec:
                continue
            spec = str(spec)
            if not AGENT_SPEC.match(spec):
                raise ValueError(f"bad agent spec for {role}: {spec!r}")
            cmd += ["--agent", f"{role}={spec}"]
        for key, conv in (("months", int), ("act_every", int)):
            if body.get(key) not in (None, ""):
                cmd += ["--set", f"game.{key}={conv(body[key])}"]
        logs = self.runs_dir / "_logs"
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{run_id}.log"
        log = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=os.getcwd(),
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        with self.lock:
            self.procs[run_id] = {"proc": proc, "log": log_path, "started": time.time(), "cmd": cmd[3:]}
        return {"run_id": run_id, "pid": proc.pid}

    def stop(self, run_id: str) -> bool:
        with self.lock:
            p = self.procs.get(run_id)
        if not p or p["proc"].poll() is not None:
            return False
        p["proc"].terminate()
        d = self.runs_dir / run_id
        if d.is_dir():
            (d / "STOPPED").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        return True

    def status(self) -> list[dict]:
        out = []
        with self.lock:
            items = list(self.procs.items())
        for rid, p in items:
            rc = p["proc"].poll()
            tail = ""
            try:
                tail = p["log"].read_text()[-1500:]
            except OSError:
                pass
            out.append({"run_id": rid, "pid": p["proc"].pid, "running": rc is None, "returncode": rc,
                        "started": p["started"], "log_tail": tail})
        return sorted(out, key=lambda r: -r["started"])

    def is_running(self, run_id: str) -> bool | None:
        with self.lock:
            p = self.procs.get(run_id)
        return None if p is None else p["proc"].poll() is None

    def failed(self, run_id: str) -> bool:
        with self.lock:
            p = self.procs.get(run_id)
        return bool(p and p["proc"].poll() not in (None, 0))


def make_handler(runs_dir: Path, hub=None, launcher: Launcher | None = None):
    index = TurnIndex()

    class Handler(SimpleHTTPRequestHandler):
        extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript",
                          ".mjs": "text/javascript", ".css": "text/css"}

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(VIEWER_DIR), **kw)

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def end_headers(self):
            if not self.path.startswith("/api/"):
                self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def _json(self, obj, code=200):
            body = json.dumps(obj, separators=(",", ":")).encode()
            gz = len(body) > 20000 and "gzip" in (self.headers.get("Accept-Encoding") or "")
            if gz:
                body = gzip.compress(body, compresslevel=5)
            self.send_response(code)
            if gz:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(n) or b"{}")

        def _run_dir(self, run_id: str) -> Path | None:
            d = (runs_dir / run_id).resolve()
            return d if d.parent == runs_dir.resolve() and d.is_dir() else None

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            parts = [p for p in url.path.split("/") if p]
            if not parts or parts[0] != "api":
                return super().do_GET()
            try:
                if parts[1:] == ["runs"]:
                    return self._json(self._list_runs())
                if len(parts) == 4 and parts[1] == "runs":
                    d = self._run_dir(parts[2])
                    if d is None:
                        return self._json({"error": "no such run"}, 404)
                    if parts[3] == "meta":
                        return self._json(json.loads((d / "meta.json").read_text()))
                    if parts[3] == "summary":
                        f = d / "summary.json"
                        return self._json(json.loads(f.read_text()) if f.exists() else None)
                    if parts[3] == "turns":         # full LLM traces for one month
                        month = int(q.get("month", ["0"])[0])
                        out = []
                        for t in index.records(d / "trajectories.jsonl", month):
                            prompt = t.get("prompt") or []
                            fleet = str(t.get("agent", "")).startswith("fleet:")   # briefing = 1st user msg
                            user = next((m["content"] for m in (prompt if fleet else reversed(prompt))
                                         if m.get("role") == "user"), t.get("observation_text", ""))
                            out.append({"role": t["role"], "agent": t.get("agent"), "reward": t.get("reward"),
                                        "response": (t.get("response") or "")[:20000],
                                        "reasoning": (t.get("reasoning") or "")[:40000],
                                        "observation": (user or "")[:30000], "usage": t.get("usage"),
                                        "latency_s": t.get("latency_s"), "finish_reason": t.get("finish_reason"),
                                        "notes": t.get("notes"),
                                        **({"trace": trace_summary(t)} if "tool_trace" in t else {})})
                        return self._json(out)
                    if parts[3] == "trace":         # one fleet turn's tool calls
                        try:
                            month, role = int(q["month"][0]), q["role"][0]
                        except (KeyError, ValueError):
                            return self._json({"error": "trace needs ?month=<int>&role=<role>"}, 400)
                        recs = index.records(d / "trajectories.jsonl", month, role)
                        if not recs:
                            return self._json({"error": f"no turn for {role} in month {month}"}, 404)
                        t = recs[0]
                        if "tool_trace" not in t:
                            return self._json({"error": f"{role}'s month-{month} turn has no tool trace "
                                                        f"(agent {t.get('agent')})"}, 404)
                        calls = [trace_call(i + 1, c) for i, c in enumerate(t.get("tool_trace") or [])]
                        return self._json({"run_id": d.name, "month": month, "role": role, "agent": t.get("agent"),
                                           "summary": trace_summary(t, calls), "calls": calls})
                    if parts[3] == "frames":
                        since = int(q.get("since", ["0"])[0])
                        frames = []
                        with open(d / "frames.jsonl") as fh:
                            for i, line in enumerate(fh):
                                if i >= since and line.endswith("\n"):
                                    frames.append(json.loads(line))
                        return self._json({"since": since, "frames": frames, "status": self._status(d),
                                           "finished": (d / "summary.json").exists()})
                if parts[1:] == ["leaderboard"]:
                    from .bench.runner import leaderboard
                    return self._json(leaderboard(runs_dir))
                if parts[1:] == ["configs"]:
                    return self._json(launcher.configs() if launcher else [])
                if parts[1:] == ["processes"]:
                    return self._json(launcher.status() if launcher else [])
                if parts[1] == "live":
                    if hub is None:
                        return self._json({"error": "no live game in this server (start one with `civitas run --serve`)"}, 404)
                    if parts[2:] == ["turn"]:
                        return self._json(hub.get_turn(q.get("role", [""])[0]))
                    if parts[2:] == ["state"]:
                        return self._json({"run_id": hub.run_id, "frame": hub.latest_frame,
                                           "waiting_for": list(hub.pending)})
                return self._json({"error": "not found"}, 404)
            except Exception as e:  # pragma: no cover
                return self._json({"error": str(e)}, 500)

        def do_POST(self):
            path = urlparse(self.path).path.rstrip("/")
            try:
                body = self._body()
            except json.JSONDecodeError:
                return self._json({"ok": False, "error": "body must be JSON"}, 400)
            try:
                if path == "/api/launch":
                    if launcher is None:
                        return self._json({"ok": False, "error": "launching is disabled"}, 403)
                    return self._json({"ok": True, **launcher.launch(body)})
                if path == "/api/stop":
                    ok = bool(launcher and launcher.stop(str(body.get("run_id", ""))))
                    return self._json({"ok": ok}, 200 if ok else 404)
                if path == "/api/live/act":
                    if hub is None:
                        return self._json({"error": "no live game"}, 404)
                    ok, msg = hub.submit(str(body.get("role", "")), str(body.get("turn_id", "")),
                                         body.get("decision") or {})
                    return self._json({"ok": ok, "message": msg}, 200 if ok else 409)
            except (ValueError, TypeError) as e:
                return self._json({"ok": False, "error": str(e)}, 400)
            return self._json({"error": "not found"}, 404)

        def _status(self, d: Path) -> str:
            if (d / "summary.json").exists():
                return "finished"
            if (d / "STOPPED").exists():
                return "stopped"
            if launcher:
                running = launcher.is_running(d.name)
                if running is not None:
                    if running:
                        return "running"
                    return "failed" if launcher.failed(d.name) else "incomplete"
            if hub is not None and hub.run_id == d.name:
                return "running"
            f = d / "frames.jsonl"
            if f.exists() and time.time() - f.stat().st_mtime < STALE_AFTER_S:
                return "running"
            return "incomplete"

        def _list_runs(self):
            out = []
            if not runs_dir.exists():
                return out
            for d in sorted(runs_dir.iterdir(), reverse=True):
                if not (d / "meta.json").exists():
                    continue
                try:
                    meta = json.loads((d / "meta.json").read_text())
                except json.JSONDecodeError:
                    continue
                s = d / "summary.json"
                summ = json.loads(s.read_text()) if s.exists() else None
                out.append({"run_id": d.name, "name": meta.get("name"), "seed": meta.get("seed"),
                            "signature": meta.get("signature"), "agents": meta.get("agents"),
                            "status": self._status(d), "live": summ is None,
                            "scores": summ["final_scores"] if summ else None,
                            "population": summ["final_metrics"]["population"] if summ else None})
            return out

    return Handler


def serve(runs_dir: str | Path = "runs", port: int = 8765, hub=None, background: bool = False,
          host: str = "127.0.0.1", configs_dir: str | Path = "configs", allow_launch: bool = True):
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    launcher = Launcher(runs_dir.resolve(), Path(configs_dir).resolve()) if allow_launch else None
    httpd = ThreadingHTTPServer((host, port), make_handler(runs_dir, hub, launcher))
    httpd.launcher = launcher
    print(f"Civitas 3D: http://{host}:{httpd.server_address[1]}/   (runs from {runs_dir.resolve()})")
    if background:
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        return httpd
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if launcher:
            for r in launcher.status():
                if r["running"]:
                    launcher.stop(r["run_id"])
    return httpd
