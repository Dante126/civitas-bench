"""HTTP server: 3D web interface + replay API + run launcher + live API (stdlib only).

Endpoints
  GET  /api/runs                     list runs with status (running / finished / stopped / incomplete)
  GET  /api/runs/<id>/meta|summary   run metadata / final summary
  GET  /api/runs/<id>/frames?since=N replay frames from index N
  GET  /api/runs/<id>/live?offset=B  the live feed (bench/live.py) from byte B (-1: its tail): {offset, events,
                                     status, held}; each agent's turns, model calls and tool calls as they happen
  GET  /api/runs/<id>/liveframe      the city mid-month after the latest builds (live_frame.json), else null
  GET  /api/runs/<id>/turns?month=M  every agent turn of month M: prompt / briefing, reasoning, reply (+ a
                                     fleet turn's tool-call summary under "trace"; "warmup": true on a warm-up turn)
  GET  /api/runs/<id>/trace?month=M&role=R   one fleet turn's tool calls in order, each with its outcome
  GET  /api/runs/<id>/problems   every failed / refused call and bad turn end, how the agent handled it, per-role rates
  GET  /api/runs/<id>/collab?city=C   requests between agents, the money and builds that answered them (bench/collab.py),
                                      with the collaboration score per month as `score` (fleet/report.py collab_score)
  GET  /api/runs/<id>/collab?region=1 a region run's cities: mayors' deals and messages between cities
                                      (`score`: each city's, asks and deals answered between them)
  GET  /api/runs/<id>/yardstick      a warm-up run's scripted yardstick (yardstick.json, bench/yardstick.py), else
                                     {"missing": true, ...}: a finished run's is computed in the background
                                     (`computing`), a `civitas yardstick` process; poll again
  GET  /api/compare?runs=a,b,c       runs side by side (bench/compare.py), each warm-up run's yardstick included
  GET  /api/evals                    model evals (civitas eval, bench/evals.py): each eval folder's model, suite, runs
                                     and the aggregate of its scorecard
  The Evals page (bench/models.py registry, bench/envs.py versions, bench/evals.py; 501 while a module is missing).
  Every POST here needs the launcher and a request from this computer (no other site's Origin):
  GET  /api/models                   registry entries + key_status + latest {eval_dir, score} on the current version,
                                     providers [{name, env, set, builtin}], can_edit
  POST /api/models | /api/models/delete | /api/models/preflight   {key, fields} save (400 {errors}) | {key} | {key}
  POST /api/keys                     {env, value}: writes NAME_API_KEY to ./.env; the value is never sent back
  GET  /api/env[?refresh=1]          envs.check() of the default suite, probed in a background thread per code state:
                                     {"computing": true} until done
  GET  /api/envs                     frozen versions;  POST /api/env/freeze {name}
  GET  /api/evals/suites             eval suites in configs/
  POST /api/evals/plan               {model, suite?, allow_drift?}: prepare() without starting (runs, estimates, env,
                                     preflight, errors)
  POST /api/evals/start              same body: prepare(), then a background `civitas eval` in its own process group;
                                     {eval_dir (null until its folder appears), job}; 400 {errors}
  GET  /api/evals/progress?dir=D     progress() of an eval folder, or of a job by name while it starts
  GET  /api/evals/jobs               evals started from this server;  POST /api/evals/stop {dir}: SIGINT to its group
  GET  /api/evals/leaderboard?env=   leaderboard();  GET /api/evals/scorecard?dir=D  {markdown}
  GET  /api/version                  a stamp of the viewer's files: an open page offers a reload when it changes
  GET  /api/leaderboard              finished runs grouped by agent line-up
  GET  /api/configs                  configs/*.toml available to launch
  GET  /api/launch                   what the New run dialog offers: maps, models, defaults, which API keys are set
  GET  /api/processes                runs started from the browser
  POST /api/launch                   {map: city|two|three, seed, months, act_every, agents: {role: spec}}: every role's
                                     "kind[:provider/model]" over configs/fleet_trinity.toml's settings; or the older
                                     {config, seed, months?, act_every?, agents?: {role: spec}} over a config file
  POST /api/pause                    {run_id}: hold a running game before its next turn (bench/control.py)
  POST /api/resume                   {run_id}: resume a held game, or continue a "paused" supervised run with its
                                     playbooks.toml
  POST /api/stop                     {run_id}: end a running or held game (STOPPED, no summary)
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
from .bench import control, live  # noqa: E402

# the New run dialog (POST /api/launch with `map`): a map, a length and a team over one config's settings
LAUNCH_BASE = "fleet_trinity.toml"          # [fleet] playbooks and limits, [providers], max actions, scoring
MAPS = {"city": [], "two": ["north", "south"], "three": ["north", "south", "east"]}   # [] = one city
MODELS = ["arcee/trinity-large-thinking", "arcee/trinity-large-v1", "arcee/trinity-mini"]   # without a registry
KEYS = {"arcee": "ARCEE_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
ROOT = Path(__file__).resolve().parents[1]          # the checkout this package runs from

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


STEP = re.compile(r"\[step (\d+)\]\s*")
STEP_CHARS = 8000


def reasoning_steps(text: str) -> list[dict]:
    """A fleet turn's reasoning split at its "[step N]" markers (the agent loop writes one per model reply), each cut
    at STEP_CHARS: the viewer shows each step above the tool calls it made. Text before any marker is step 0."""
    parts, out = STEP.split(text), []
    if parts and parts[0].strip():
        out.append({"step": 0, "text": parts[0].strip()[:STEP_CHARS], "chars": len(parts[0].strip())})
    for n, body in zip(parts[1::2], parts[2::2]):
        body = body.strip()
        if body:
            out.append({"step": int(n), "text": body[:STEP_CHARS], "chars": len(body)})
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


_PROBLEMS: dict[str, tuple] = {}   # trajectories path -> ((path, size), run_problems result): recomputed as a live run grows
_COLLAB: dict[tuple, tuple] = {}    # (frames path, city) -> ((path, size, city), run_collab result): likewise


def collab_data(d: Path, city: str | None) -> dict:
    """run_collab of a run folder (city "*": the region view) with its collaboration score as `score`
    (fleet/report.py: collab_score, region_score from each city's), cached in _COLLAB until frames.jsonl grows."""
    from .bench.collab import run_collab
    from .fleet.report import collab_score, region_score
    fr = d / "frames.jsonl"
    key = (str(fr), fr.stat().st_size if fr.exists() else 0, city)
    hit = _COLLAB.get((str(fr), city))
    if hit is not None and hit[0] == key:
        return hit[1]
    c = run_collab(d, city)
    try:
        c["score"] = (region_score(c, {x: collab_data(d, x).get("score") or {} for x in c["cities"]}) if city == "*"
                      else collab_score(c))
    except Exception as e:          # the score is extra: never lose the Team tab over it
        c["score"] = {"error": f"{type(e).__name__}: {e}"}
    _COLLAB[(str(fr), city)] = (key, c)
    return c


_COMPARE: dict[str, tuple] = {}    # run folder -> ((size, mtime) of its files, run_compare result): a live run recomputes
COMPARE_MAX = 12                    # runs per /api/compare request


def compare_data(d: Path) -> dict:
    """run_compare of a run folder (the Compare view), Teamwork from collab_data, cached in _COMPARE until frames.jsonl,
    summary.json or fleet_report.json change."""
    from .bench.compare import run_compare

    def stamp(f: Path):
        try:
            st = f.stat()
            return st.st_size, st.st_mtime_ns
        except OSError:
            return None
    key = tuple(stamp(d / f) for f in ("frames.jsonl", "summary.json", "fleet_report.json", "meta.json",
                                        "yardstick.json"))
    hit = _COMPARE.get(str(d))
    if hit is not None and hit[0] == key:
        return hit[1]
    c = run_compare(d, collab_data)
    _COMPARE[str(d)] = (key, c)
    return c


_YARDSTICKS: dict[str, dict] = {}  # run folder -> {"proc", "log", "started"}: yardsticks computing in the background
_YARD_LOCK = threading.Lock()
YARDSTICK_JOBS = 2                  # yardsticks computed at once (each a `civitas yardstick` process, seconds)


def yardstick_data(d: Path, compute: bool = True) -> dict:
    """A run's scripted yardstick (bench/yardstick.py): yardstick.json when there is one, else `{"missing": true}`
    with why, `computing` while a background `civitas yardstick` process (started here when `compute`, for a
    finished warm-up run that handed over) works on it, or the `error` it ended with. Never blocks on the game."""
    from .bench import yardstick as Y
    ys = Y.read(d)
    if ys is not None:
        return ys
    try:
        meta = json.loads((d / "meta.json").read_text())
    except (OSError, ValueError):
        return {"missing": True, "reason": "no meta.json"}
    if not int(meta.get("warmup_months") or 0):
        return {"missing": True, "reason": "not a warm-up run"}
    try:
        summ = json.loads((d / "summary.json").read_text())
    except (OSError, ValueError):
        return {"missing": True, "reason": "the run hasn't finished (it writes its yardstick at the end)"}
    if not (summ.get("warmup") or {}).get("handed_over"):
        return {"missing": True, "reason": "the game ended during the warm-up"}
    with _YARD_LOCK:
        job = _YARDSTICKS.get(str(d))
        if job is not None:
            rc = job["proc"].poll()
            if rc is None:
                return {"missing": True, "computing": True, "started": job["started"]}
            if rc != 0:
                try:
                    tail = job["log"].read_text().strip().splitlines()[-1:]
                except OSError:
                    tail = []
                return {"missing": True, "error": tail[0] if tail else f"exit code {rc}"}
            _YARDSTICKS.pop(str(d))                 # done, but no file: try again below
        if not compute:
            return {"missing": True}
        if sum(j["proc"].poll() is None for j in _YARDSTICKS.values()) >= YARDSTICK_JOBS:
            return {"missing": True, "queued": True}
        logs = d.parent / "_logs"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{d.name}.yardstick.log"
        with open(log, "w") as fh:                  # the child keeps its own copy of the handle
            proc = subprocess.Popen([sys.executable, "-m", "civitas.cli", "yardstick", d.name, "--runs-dir",
                                     str(d.parent), "--quiet"], stdout=fh, stderr=subprocess.STDOUT,
                                    cwd=os.getcwd(), env={**os.environ, "PYTHONUNBUFFERED": "1"})
        _YARDSTICKS[str(d)] = {"proc": proc, "log": log, "started": time.time()}
    return {"missing": True, "computing": True, "started": _YARDSTICKS[str(d)]["started"]}


def viewer_version() -> str:
    """A stamp of the viewer's files (newest change and count)."""
    files = [p.stat().st_mtime_ns for p in VIEWER_DIR.rglob("*") if p.is_file()]
    return f"{max(files, default=0)}-{len(files)}"


def _has_key(env_var: str) -> bool:
    return bool(os.environ.get(env_var))            # the cli loads ./.env before `civitas serve` starts


def run_folder(runs_dir: Path, run_id: str) -> Path | None:
    """The run's folder, only when it sits directly in `runs_dir` once resolved (no "..", ".", "a/b", links out)."""
    d = (runs_dir / run_id).resolve()
    return d if d.parent == runs_dir.resolve() and d.is_dir() else None


class Launcher:
    """Starts `civitas run` subprocesses on behalf of the browser."""

    def __init__(self, runs_dir: Path, configs_dir: Path):
        self.runs_dir = runs_dir
        self.configs_dir = configs_dir
        self.procs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def configs(self) -> list[dict]:
        from .bench.evals import is_suite
        try:
            from .bench.models import is_registry
        except ImportError:                         # pragma: no cover
            def is_registry(cfg): return False
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
            if is_suite(cfg) or is_registry(cfg):  # an eval suite (civitas eval -s) or configs/models.toml
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

    def options(self) -> dict:
        """What the New run dialog offers. `keys`: whether each provider's API key is set for the runs it starts."""
        from .config import load_config
        from .fleet.config import fleet_agent_config
        path = self.configs_dir / LAUNCH_BASE
        cfg = load_config(path) if path.is_file() else {"game": {}, "fleet": {}}
        listed, keys = launch_models()
        try:
            fc = fleet_agent_config(cfg, "mayor")
            fleet = {"model": f"{fc.provider}/{fc.model}", "max_steps": fc.max_steps}
        except ValueError:
            fleet = {"model": listed[0], "max_steps": 20}
        models = listed if fleet["model"] in listed else [fleet["model"], *listed]
        return {"base": LAUNCH_BASE if path.is_file() else None, "roles": ROLES, "maps": MAPS, "models": models,
                "fleet": fleet, "months": 24, "act_every": cfg["game"].get("act_every", 3),
                "keys": {p: _has_key(k) for p, k in keys.items()}}

    def compose(self, body: dict) -> tuple[str, list[str]]:
        """The New run dialog's run: `civitas run` on LAUNCH_BASE with the map, length and every role's agent."""
        path = (self.configs_dir / LAUNCH_BASE).resolve()
        if not path.is_file():
            raise ValueError(f"no {LAUNCH_BASE} in {self.configs_dir}")
        cities = MAPS.get(str(body.get("map")))
        if cities is None:
            raise ValueError(f"unknown map {body.get('map')!r} ({' | '.join(MAPS)})")
        seed, months, every = int(body.get("seed", 0)), int(body.get("months", 24)), int(body.get("act_every", 3))
        if not (1 <= months <= 600 and 1 <= every <= 12 and seed >= 0):
            raise ValueError("months must be 1-600, act_every 1-12, seed 0 or more")
        agents = body.get("agents") or {}
        specs = {r: str(agents.get(r) or "") for r in ROLES}
        bad = [f"{r}={v!r}" for r, v in specs.items() if not AGENT_SPEC.match(v)]
        if bad:
            raise ValueError(f"every role needs an agent (scripted | fleet[:provider/model] | llm[:provider/model]): "
                             f"{', '.join(bad)}")
        kinds = {v for v in specs.values()}
        team = "scripted" if kinds == {"scripted"} else \
            (next(iter(kinds)).partition(":")[2].rpartition("/")[2] or "fleet") if len(kinds) == 1 else "mixed"
        name = re.sub(r"[^A-Za-z0-9_.-]", "", f"{'city' if not cities else f'region{len(cities)}'}_{team}")[:40]
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{name}_s{seed}_ui"
        cmd = [sys.executable, "-m", "civitas.cli", "run", "-c", str(path), "--seed", str(seed),
               "--runs-dir", str(self.runs_dir), "--run-id", run_id, "--set", f"name={name}",
               "--set", f"game.months={months}", "--set", f"game.act_every={every}"]
        if cities:
            cmd += ["--set", "game.region=true", "--set", f"game.cities={json.dumps(cities, separators=(',', ':'))}"]
        for r, v in specs.items():
            cmd += ["--agent", f"{r}={v}"]
        return run_id, cmd

    def launch(self, body: dict) -> dict:
        if "map" in body:
            run_id, cmd = self.compose(body)
            return self._start(run_id, cmd)
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
        return self._start(run_id, cmd)

    def _start(self, run_id: str, cmd: list[str]) -> dict:
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
        """End a game: one this server started at once, any other unfinished one at its next turn (STOP)."""
        with self.lock:
            p = self.procs.get(run_id)
        d = run_folder(self.runs_dir, run_id)
        if p and p["proc"].poll() is None:
            p["proc"].terminate()
            if d is not None:
                control.mark_stopped(d)
            return True
        if d is None or (d / "summary.json").exists() or (d / control.STOPPED).exists() or (d / "PAUSED").exists():
            return False
        control.stop(d)
        return True

    def pause(self, run_id: str) -> bool:
        """Hold an unfinished game before its next turn."""
        d = run_folder(self.runs_dir, run_id)
        if d is None or any((d / m).exists() for m in ("summary.json", control.STOPPED, "PAUSED")):
            return False
        control.pause(d)
        return True

    def resume(self, run_id: str) -> dict:
        """Release a held game (bench/control.py), or `civitas resume <run_id>`: the next session of a paused
        supervised run, with the playbooks in its playbooks.toml."""
        d = run_folder(self.runs_dir, run_id)
        if d is not None and control.release(d):
            return {"run_id": d.name, "held": False}
        if d is None or not (d / "PAUSED").exists():
            raise ValueError(f"no paused run {run_id!r}")
        run_id = d.name
        if self.is_running(run_id):
            raise ValueError(f"run {run_id!r} is already running")
        cmd = [sys.executable, "-m", "civitas.cli", "resume", run_id, "--runs-dir", str(self.runs_dir)]
        logs = self.runs_dir / "_logs"
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{run_id}.log"
        log = open(log_path, "a")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=os.getcwd(),
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        with self.lock:
            self.procs[run_id] = {"proc": proc, "log": log_path, "started": time.time(), "cmd": cmd[3:]}
        return {"run_id": run_id, "pid": proc.pid}

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


# ---------------------------------------------------------------------------------------------------- the Evals page
# the model registry (bench/models.py), environment versions (bench/envs.py) and evals (bench/evals.py) from the
# viewer: GET reads work without the launcher; every write needs it and a request from this computer (`writable`)
ENV_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
KEY_VAR = re.compile(r"^[A-Z][A-Z0-9_]*_API_KEY$")
LOCAL = {"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"}
LAUNCHER_POSTS = ("/api/launch", "/api/stop", "/api/pause", "/api/resume")   # a browser on this computer: cross_site
CODE_STATE_S = 5.0          # how long a code state (commit + diff) is trusted before git is asked again
START_WAIT_S = 3.0          # how long /api/evals/start waits for the eval's folder before answering with its job
STOP_GRACE_S = 20.0         # after Stop's SIGINT, how long the eval may take to stop its runs before SIGTERM
EVALS_POSTS = {"/api/models", "/api/models/delete", "/api/models/preflight", "/api/keys", "/api/env/freeze",
               "/api/evals/plan", "/api/evals/start", "/api/evals/stop"}


class Unavailable(RuntimeError):
    """A track of the Evals page that isn't installed (bench/models.py, bench/envs.py or an evals.py function)."""


def track(name: str):
    """civitas.bench.<name> (models / envs / evals), Unavailable when the module isn't there."""
    import importlib
    full = f"{__package__}.bench.{name}"
    try:
        return importlib.import_module(full)
    except ModuleNotFoundError as e:
        if e.name == full:
            raise Unavailable(f"bench/{name}.py is not in this checkout") from None
        raise


def evals_fn(name: str):
    fn = getattr(track("evals"), name, None)
    if fn is None:
        raise Unavailable(f"bench/evals.py has no {name}() in this checkout")
    return fn


def is_local(host: str) -> bool:
    return host in LOCAL


LOOPBACK_NAMES = {"127.0.0.1", "::1", "localhost"}


def loopback_name(netloc: str | None) -> bool:
    """A Host header / Origin netloc naming this computer by a loopback name (any port). A DNS-rebinding page
    sends its own domain in both Host and Origin, so matching the two isn't enough."""
    host = (urlparse("//" + (netloc or "")).hostname or "").rstrip(".").lower()
    return host in LOOPBACK_NAMES


def cross_site(headers) -> str | None:
    """Why a browser request is refused as another site's (None: it isn't): Host must be a loopback name and an
    Origin, when sent, a loopback origin."""
    if not loopback_name(headers.get("Host")):
        return "refused: open the viewer as http://127.0.0.1 or http://localhost to change anything"
    origin = headers.get("Origin")
    if origin and (origin == "null" or not loopback_name(urlparse(origin).netloc)):
        return "refused: the request came from another site"
    return None


def default_suite(configs_dir: Path) -> Path:
    """The suite an eval runs without -s: evals.DEFAULT_SUITE, looked up in the server's configs folder."""
    name = getattr(track("evals"), "DEFAULT_SUITE", "configs/eval_singles.toml")
    return configs_dir / Path(name).name


def suite_path(configs_dir: Path, suite) -> Path | None:
    """A suite file named by the browser: only a file directly in the configs folder (None: the default)."""
    if suite in (None, ""):
        return None
    p = (configs_dir / Path(str(suite)).name).resolve()
    if p.parent != configs_dir.resolve() or not p.is_file():
        raise ValueError(f"unknown suite {suite!r}")
    return p


def launch_models() -> tuple[list[str], dict[str, str]]:
    """The New run dialog's models ("provider/served id") and provider -> API key variable, from the registry
    (configs/models.toml) and the providers; MODELS / KEYS while bench/models.py isn't there or lists nothing."""
    try:
        M = track("models")
        reg = M.load_registry()
        models = [f"{e.provider}/{e.served_id}" for e in reg.values()]
        models = [m for m in dict.fromkeys(models) if AGENT_SPEC.match(f"fleet:{m}")]
        keys = {p["name"]: p["env"] for p in providers(M) if p["env"]}
    except Unavailable:
        return MODELS, KEYS
    except Exception:           # a broken registry file must not take the New run dialog down
        return MODELS, KEYS
    return models or MODELS, {**KEYS, **keys}


def providers(M) -> list[dict]:
    """Built-in providers (config.DEFAULT_PROVIDERS) and the registry's [providers.X]: name, key variable, set."""
    from .config import DEFAULT_PROVIDERS
    reg = M.registry_providers() or {}
    out = []
    for name in dict.fromkeys([*DEFAULT_PROVIDERS, *reg]):
        spec = {**DEFAULT_PROVIDERS.get(name, {}), **(reg.get(name) or {})}
        env = str(spec.get("api_key_env") or "")
        out.append({"name": name, "env": env, "set": bool(env) and _has_key(env), "builtin": name in DEFAULT_PROVIDERS,
                    "base_url": spec.get("base_url")})
    return out


_CODE: dict = {}


def code_state(root: Path = ROOT) -> str:
    """The code an environment probe ran on: HEAD and a hash of the diff against it (trusted for CODE_STATE_S)."""
    import hashlib
    now = time.time()
    if _CODE.get("at", 0) > now - CODE_STATE_S and _CODE.get("root") == str(root):
        return _CODE["state"]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=root, capture_output=True, timeout=20)
        state = f"{head.stdout.strip() or 'nogit'}:{hashlib.sha1(diff.stdout).hexdigest()[:12]}"
    except (OSError, subprocess.SubprocessError):
        state = "nogit"
    _CODE.update(at=now, root=str(root), state=state)
    return state


class EnvCheck:
    """envs.check() of the default suite, probed in a background thread (the scripted fleet replays every seed, ~10-20
    s) and kept per code state: `get()` answers {"computing": true} until it is done; refresh probes again."""

    def __init__(self, configs_dir: Path, runs_dir: Path | None = None):
        self.configs_dir = configs_dir
        self.cache_dir = None if runs_dir is None else Path(runs_dir) / "_cache"   # shared with prepare()'s probes
        self.lock = threading.Lock()
        self.key = None
        self.result: dict | None = None
        self.thread: threading.Thread | None = None
        self.started = None

    def suite(self) -> Path:
        return default_suite(self.configs_dir)

    def get(self, refresh: bool = False) -> dict:
        envs = track("envs")
        key = code_state()
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return {"computing": True, "started": self.started}
            if not refresh and self.key == key and self.result is not None:
                return self.result
            self.started = time.time()
            self.thread = threading.Thread(target=self._run, args=(envs, key), daemon=True)
            self.thread.start()
            return {"computing": True, "started": self.started}

    def _run(self, envs, key: str) -> None:
        suite = self.suite()
        try:
            res = envs.check(str(suite), **({"cache_dir": self.cache_dir} if self.cache_dir else {}))
            out = {k: v for k, v in res.items() if k != "probe"}
            probe = res.get("probe") or {}
            out.update(suite=suite.name, code=key, checked=time.time(), seconds=probe.get("seconds"))
            self._probe = probe
        except Exception as e:
            out = {"error": f"{type(e).__name__}: {e}", "suite": suite.name, "code": key, "checked": time.time()}
            self._probe = None
        with self.lock:
            self.key, self.result = key, out

    def probe(self) -> dict | None:
        """The last probe, when it was of the code as it is now (freeze reuses it instead of probing again)."""
        with self.lock:
            fresh = self.key == code_state() and self.result is not None and "error" not in self.result
        return getattr(self, "_probe", None) if fresh else None

    def forget(self) -> None:
        with self.lock:
            self.key = self.result = None


class EvalJobs:
    """`civitas eval` processes started from the Evals page, each in its own process group (its runs are stopped
    with it). The eval's folder is stamped by the eval process when it plans, so a job is known by its name until
    a new runs/eval_<name>_* folder appears (`resolve`)."""

    def __init__(self, runs_dir: Path, configs_dir: Path):
        self.runs_dir = runs_dir
        self.configs_dir = configs_dir
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def _folders(self, name: str) -> set[str]:
        return {p.name for p in self.runs_dir.glob(f"eval_{name}_*") if p.is_dir()}

    def command(self, model: str, suite: Path | None, allow_drift: bool, name: str, stamp: str | None = None
                ) -> list[str]:
        """The eval process: prepare()'s name and stamp, so it writes the folder prepare() planned, and no second
        preflight (prepare() ran it; the probe is cached)."""
        cmd = [sys.executable, "-m", "civitas.cli", "eval", "--model", model, "--name", name,
               "--runs-dir", str(self.runs_dir)]
        if stamp:
            cmd += ["--stamp", stamp, "--no-preflight"]
        if suite is not None:
            cmd += ["-s", str(suite)]
        if allow_drift:
            cmd.append("--allow-drift")
        return cmd

    def prepare(self, body: dict) -> tuple[dict, Path | None]:
        model = str(body.get("model") or "").strip()
        if not model or len(model) > 200:
            raise ValueError("name a model")
        suite = suite_path(self.configs_dir, body.get("suite"))
        ev = evals_fn("prepare")(model, suite=None if suite is None else str(suite),
                                 allow_drift=bool(body.get("allow_drift")), runs_dir=str(self.runs_dir))
        return ev, suite

    def start(self, body: dict) -> dict:
        """prepare(); with no errors, the eval process. {"errors"} when it may not start."""
        import shutil
        ev, suite = self.prepare(body)
        if ev.get("errors"):
            return {"errors": list(ev["errors"])}
        model = str(body.get("model")).strip()
        name = str(ev.get("name") or model)
        with self.lock:
            old = self.jobs.get(name)
            if old is not None and old["proc"].poll() is None:
                raise ValueError(f"an eval named {name} is already running from this page")
            before = self._folders(name)
            cmd = self.command(model, suite, bool(body.get("allow_drift")), name, ev.get("stamp"))
            if sys.platform == "darwin" and shutil.which("caffeinate") and cmd[:1] != ["caffeinate"]:
                cmd = ["caffeinate", "-dims", *cmd]          # the Mac sleeps during long evals otherwise
            logs = self.runs_dir / "_logs"
            logs.mkdir(parents=True, exist_ok=True)
            log = logs / f"eval_{name}_{time.strftime('%Y%m%d-%H%M%S')}.log"
            path = os.environ.get("PYTHONPATH", "")
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT) + (os.pathsep + path if path else "")}
            with open(log, "w") as fh:                       # the child keeps its own handle
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=os.getcwd(), env=env,
                                        start_new_session=True)
            self.jobs[name] = {"proc": proc, "log": log, "started": time.time(), "before": before,
                               "model": model, "stopping": False,
                               "dir": Path(ev["dir"]).name if ev.get("stamp") and ev.get("dir") else None}
        end = time.time() + START_WAIT_S
        while time.time() < end and self.resolve(name) is None and proc.poll() is None:
            time.sleep(0.1)
        return {"eval_dir": self.resolve(name), "job": name, "pid": proc.pid}

    def resolve(self, name: str) -> str | None:
        """The folder of job `name`: the eval_<name>_* folder that appeared after it started."""
        with self.lock:
            job = self.jobs.get(name)
            if job is None:
                return None
            if job["dir"] is None:
                new = sorted(self._folders(name) - job["before"])
                if new:
                    job["dir"] = new[-1]
            return job["dir"]

    def job_of(self, ref: str) -> tuple[str, dict] | None:
        with self.lock:
            if ref in self.jobs:
                return ref, self.jobs[ref]
            for name, job in self.jobs.items():
                if job["dir"] == ref:
                    return name, job
        return None

    def state(self, name: str, job: dict) -> dict:
        rc = job["proc"].poll()
        out = {"job": name, "model": job["model"], "eval_dir": self.resolve(name), "started": job["started"],
               "running": rc is None, "returncode": rc, "stopping": job["stopping"]}
        if rc not in (None, 0):
            try:
                lines = [x for x in job["log"].read_text().strip().splitlines() if x.strip()]
            except OSError:
                lines = []
            out["error"] = lines[-1][:400] if lines else f"exit code {rc}"
        return out

    def status(self) -> list[dict]:
        with self.lock:
            items = list(self.jobs.items())
        return sorted((self.state(n, j) for n, j in items), key=lambda r: -r["started"])

    def stop(self, ref: str) -> bool:
        """SIGINT to the eval's process group (the eval stops its runs and records them interrupted), SIGTERM then
        SIGKILL if it hasn't ended within STOP_GRACE_S; an eval this server didn't start: SIGINT to its pid."""
        import signal
        hit = self.job_of(ref)
        if hit is None:
            d = (self.runs_dir / ref).resolve()
            if d.parent != self.runs_dir.resolve() or not ref.startswith("eval_"):
                return False
            try:
                ev = json.loads((d / "eval.json").read_text())
                pid = int(ev.get("pid") or 0)
            except (OSError, ValueError, TypeError):
                return False
            if ev.get("status") != "running" or pid <= 0:
                return False
            try:
                os.kill(pid, signal.SIGINT)
            except OSError:
                return False
            return True
        name, job = hit
        proc = job["proc"]
        if proc.poll() is not None:
            return False
        job["stopping"] = True
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except OSError:
            proc.send_signal(signal.SIGINT)

        def escalate():
            for sig, wait in ((signal.SIGTERM, STOP_GRACE_S), (signal.SIGKILL, 5.0)):
                try:
                    proc.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, sig)
                    except OSError:
                        pass
            self._stop_runs(name)
        threading.Thread(target=escalate, daemon=True).start()
        return True

    def _stop_runs(self, name: str) -> None:
        """Runs the eval left running (each in its own session) once it has ended: SIGTERM to their groups."""
        import signal
        d = self.resolve(name)
        if d is None:
            return
        try:
            ev = json.loads((self.runs_dir / d / "eval.json").read_text())
        except (OSError, ValueError):
            return
        for r in ev.get("runs") or []:
            if r.get("status") == "running" and r.get("pid"):
                try:
                    os.killpg(int(r["pid"]), signal.SIGTERM)
                except (OSError, ValueError):
                    pass


def eval_folder(runs_dir: Path, name: str) -> Path | None:
    """An eval's folder, only an eval_* folder directly in `runs_dir`."""
    if not str(name).startswith("eval_"):
        return None
    return run_folder(runs_dir, str(name))


def suites(configs_dir: Path) -> list[dict]:
    """The eval suites in the configs folder (files with [[runs]]): file, name, description, units and runs."""
    E = track("evals")
    out = []
    default = default_suite(configs_dir).name
    for p in sorted(configs_dir.glob("*.toml")):
        try:
            if "[[runs]]" not in p.read_text():
                continue
            s = E.load_suite(p)
        except (OSError, ValueError) as e:
            if "is not a suite" not in str(e):
                out.append({"file": p.name, "error": str(e)})
            continue
        units = sum(len(e.seeds) for e in s.entries)
        out.append({"file": p.name, "name": s.name, "description": s.description, "units": units,
                    "runs": sum(len(e.seeds) * (e.repeats or s.repeats) for e in s.entries), "workers": s.workers,
                    "env": getattr(s, "env", None), "baseline_model": getattr(s, "baseline_model", None),
                    "default": p.name == default})
    return out


def plan_summary(ev: dict) -> dict:
    """What the Evaluate dialog shows from prepare(): runs, the estimated tokens / time / cost, the environment, the
    preflight, the errors (empty: it may start)."""
    E = track("evals")
    runs = ev.get("runs") or []
    est = [r.get("estimate") or {} for r in runs]
    total = lambda k: None if any(x.get(k) is None for x in est) or not est else sum(x[k] for x in est)
    prompt, completion = total("prompt_tokens"), total("completion_tokens")
    mins = [x.get("minutes") for x in est]
    minutes = E.makespan(mins, int(ev.get("workers") or 1)) if mins and None not in mins else None
    cost = None
    if prompt is not None and completion is not None and hasattr(E, "_usd"):
        try:
            cost = E._usd(ev, prompt, completion)
        except Exception:
            cost = None
    entry = ev.get("model_entry")
    if entry is not None and hasattr(entry, "to_dict"):
        entry = entry.to_dict()
    return {"name": ev.get("name"), "dir": Path(str(ev.get("dir") or "")).name or None, "model": ev.get("model"),
            "provider": ev.get("provider"), "suite": ev.get("suite"), "runs": len(runs), "workers": ev.get("workers"),
            "units": len({(r.get("config"), r.get("seed")) for r in runs}),
            "estimate": {"prompt_tokens": prompt, "completion_tokens": completion, "minutes": minutes, "cost": cost,
                         "reference_model": (ev.get("suite") or {}).get("reference_model")},
            "env": ev.get("env"), "preflight": ev.get("preflight"), "model_entry": entry,
            "errors": list(ev.get("errors") or [])}


def make_handler(runs_dir: Path, hub=None, launcher: Launcher | None = None, configs_dir: str | Path = "configs",
                 envcheck: EnvCheck | None = None, jobs: EvalJobs | None = None):
    index = TurnIndex()
    configs_dir = launcher.configs_dir if launcher is not None else Path(configs_dir).resolve()
    envcheck = envcheck or EnvCheck(configs_dir, runs_dir)
    jobs = jobs or (EvalJobs(runs_dir.resolve(), configs_dir) if launcher is not None else None)

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
            return run_folder(runs_dir, run_id)

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
                                        **({"warmup": True} if t.get("warmup") else {}),   # the scripted warm-up's
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
                                           "summary": trace_summary(t, calls), "calls": calls,
                                           "steps": reasoning_steps(t.get("reasoning") or "")})
                    if parts[3] == "problems":      # every problem in the run and how the agents handled it
                        from .bench.problems import run_problems
                        traj = d / "trajectories.jsonl"
                        key = (str(traj), traj.stat().st_size if traj.exists() else 0)
                        if _PROBLEMS.get(str(traj), (None,))[0] != key:
                            _PROBLEMS[str(traj)] = (key, run_problems(traj, d / "fleet_report.json") if traj.exists()
                                                    else {"problems": [], "roles": {}, "total": {}, "hints": []})
                        return self._json(_PROBLEMS[str(traj)][1])
                    if parts[3] == "collab":        # who asked whom for what, and what came of it (Team tab)
                        city = (q.get("city") or [None])[0]
                        if (q.get("region") or ["0"])[0] not in ("", "0", "false"):
                            city = "*"              # the region view: the cities, their deals and messages
                        return self._json(collab_data(d, city))
                    if parts[3] == "yardstick":     # the scripted agents from the same hand-over (a warm-up run)
                        return self._json(yardstick_data(d, compute=launcher is not None))
                    if parts[3] == "live":          # what the agents are doing now (bench/live.py)
                        off = int(q.get("offset", ["-1"])[0])
                        return self._json({**live.read_events(d, off), "status": self._status(d),
                                           "held": control.held(d)})
                    if parts[3] == "liveframe":     # the city mid-month, after the latest builds
                        f = d / live.FRAME
                        try:
                            return self._json(json.loads(f.read_text()))
                        except (OSError, ValueError):
                            return self._json(None)
                    if parts[3] == "frames":
                        since = int(q.get("since", ["0"])[0])
                        frames = []
                        with open(d / "frames.jsonl") as fh:
                            for i, line in enumerate(fh):
                                if i >= since and line.endswith("\n"):
                                    frames.append(json.loads(line))
                        return self._json({"since": since, "frames": frames, "status": self._status(d),
                                           "finished": (d / "summary.json").exists(),
                                           "held": control.held(d), "control": launcher is not None})
                if parts[1:] == ["compare"]:        # runs side by side (Compare view): ?runs=a,b,c
                    ids = [x for x in ",".join(q.get("runs", [])).split(",") if x][:COMPARE_MAX]
                    out, missing = [], []
                    for rid in dict.fromkeys(ids):
                        d = self._run_dir(rid)
                        if d is None or not (d / "meta.json").exists():
                            missing.append(rid)
                            continue
                        out.append({**compare_data(d), "status": self._status(d)})
                    return self._json({"runs": out, "missing": missing})
                if parts[1:] == ["evals"]:          # model evals (civitas eval): eval_*/eval.json, newest first
                    from .bench.evals import list_evals
                    return self._json(list_evals(runs_dir))
                if parts[1] in ("models", "env", "envs", "evals"):     # the Evals page
                    return self._evals_page_get(parts[1:], q)
                if parts[1:] == ["leaderboard"]:
                    from .bench.runner import leaderboard
                    return self._json(leaderboard(runs_dir))
                if parts[1:] == ["version"]:       # changes when a viewer file does: open pages offer a reload
                    return self._json({"viewer": viewer_version()})
                if parts[1:] == ["configs"]:
                    return self._json(launcher.configs() if launcher else [])
                if parts[1:] == ["launch"]:
                    if launcher is None:
                        return self._json({"error": "launching is disabled"}, 403)
                    return self._json(launcher.options())
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
            except Unavailable as e:
                return self._json({"error": str(e), "unavailable": True}, 501)
            except Exception as e:  # pragma: no cover
                return self._json({"error": str(e)}, 500)

        def do_POST(self):
            path = urlparse(self.path).path.rstrip("/")
            try:
                body = self._body()
            except json.JSONDecodeError:
                return self._json({"ok": False, "error": "body must be JSON"}, 400)
            try:
                if path in LAUNCHER_POSTS and is_local(self.client_address[0]):
                    refusal = cross_site(self.headers)
                    if refusal:
                        return self._json({"ok": False, "error": refusal}, 403)
                if path == "/api/launch":
                    if launcher is None:
                        return self._json({"ok": False, "error": "launching is disabled"}, 403)
                    return self._json({"ok": True, **launcher.launch(body)})
                if path in ("/api/stop", "/api/pause"):
                    act = launcher and (launcher.stop if path == "/api/stop" else launcher.pause)
                    ok = bool(act and act(str(body.get("run_id", ""))))
                    return self._json({"ok": ok}, 200 if ok else 404)
                if path == "/api/resume":
                    if launcher is None:
                        return self._json({"ok": False, "error": "launching is disabled"}, 403)
                    return self._json({"ok": True, **launcher.resume(str(body.get("run_id", "")))})
                if path == "/api/live/act":
                    if hub is None:
                        return self._json({"error": "no live game"}, 404)
                    ok, msg = hub.submit(str(body.get("role", "")), str(body.get("turn_id", "")),
                                         body.get("decision") or {})
                    return self._json({"ok": ok, "message": msg}, 200 if ok else 409)
                if path in EVALS_POSTS:
                    if not isinstance(body, dict):
                        return self._json({"ok": False, "error": "body must be a JSON object"}, 400)
                    refusal = self._write_refusal()
                    if refusal:
                        return self._json({"ok": False, "error": refusal}, 403)
                    return self._evals_page_post(path, body)
            except Unavailable as e:
                return self._json({"ok": False, "error": str(e), "unavailable": True}, 501)
            except (ValueError, TypeError) as e:
                return self._json({"ok": False, "error": str(e)}, 400)
            except Exception as e:  # pragma: no cover
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
            return self._json({"error": "not found"}, 404)

        # ------------------------------------------------------------------------------------ the Evals page
        def _write_refusal(self) -> str | None:
            """Why this request may not write (None: it may): the launcher must be on, the client this computer,
            Host a loopback name and a browser's Origin, when it sends one, a loopback origin (cross_site)."""
            if launcher is None:
                return "editing is disabled: start `civitas serve` with launching allowed"
            if not is_local(self.client_address[0]):
                return "editing is allowed from this computer only"
            return cross_site(self.headers)

        def _latest(self, env_name: str | None) -> tuple[dict, str | None]:
            """model -> its leaderboard row on `env_name` as {eval_dir, score, interval, city, env}."""
            try:
                lb = evals_fn("leaderboard")(env_name, runs_dir=str(runs_dir))
            except Unavailable:
                return {}, env_name
            except Exception:
                return {}, env_name
            out = {}
            for row in lb.get("rows") or []:
                vs = row.get("vs_yardstick") or {}
                out.setdefault(str(row.get("model")), {"eval_dir": row.get("eval_dir"), "score": vs.get("mean"),
                                                      "interval": vs, "city": row.get("score"),
                                                      "env": lb.get("env"), "verdict": row.get("verdict")})
            return out, lb.get("env")

        def _evals_page_get(self, parts: list[str], q: dict):
            arg = lambda k, d=None: (q.get(k) or [d])[0]
            if parts == ["models"]:
                M = track("models")
                cur = envcheck.result or {}
                latest, env_name = self._latest(cur.get("env"))
                rows = []
                for e in M.load_registry().values():
                    rows.append({**e.to_dict(), "key": e.key, "served_id": e.served_id, "key_status": M.key_status(e),
                                 "latest": latest.get(e.key) or latest.get(e.served_id)})
                return self._json({"models": rows, "providers": providers(M), "env": env_name,
                                   "can_edit": launcher is not None and is_local(self.client_address[0])})
            if parts == ["env"]:
                refresh = arg("refresh", "0") not in ("", "0", "false")
                if refresh and launcher is None:
                    refresh = False                 # a probe costs the server's CPU: only with launching on
                return self._json(envcheck.get(refresh=refresh))
            if parts == ["envs"]:
                return self._json(track("envs").list_envs())
            if parts == ["evals", "leaderboard"]:
                return self._json(evals_fn("leaderboard")(arg("env") or None, runs_dir=str(runs_dir)))
            if parts == ["evals", "suites"]:
                return self._json(suites(configs_dir))
            if parts == ["evals", "jobs"]:
                return self._json(jobs.status() if jobs else [])
            if parts == ["evals", "progress"]:
                ref = str(arg("dir", ""))
                hit = jobs.job_of(ref) if jobs else None
                if hit is not None:                 # an eval started from this page: by its name until it has a folder
                    st = jobs.state(*hit)
                    if st["eval_dir"] is None:
                        return self._json({**st, "status": "starting" if st["running"] else
                                           ("failed" if st.get("error") else "stopped"),
                                           "total": None, "finished": 0, "tokens": 0, "eta_s": None, "runs": []})
                    d = eval_folder(runs_dir, st["eval_dir"])
                    return self._json({**evals_fn("progress")(str(d)), **st})
                d = eval_folder(runs_dir, ref)
                if d is None or not (d / "eval.json").exists():
                    return self._json({"error": f"no eval {ref!r}"}, 404)
                return self._json({**evals_fn("progress")(str(d)), "eval_dir": d.name})
            if parts == ["evals", "scorecard"]:
                d = eval_folder(runs_dir, str(arg("dir", "")))
                f = d / "scorecard.md" if d is not None else None
                if f is None or not f.is_file():
                    return self._json({"error": "no scorecard yet"}, 404)
                return self._json({"eval_dir": d.name, "markdown": f.read_text()})
            return self._json({"error": "not found"}, 404)

        def _evals_page_post(self, path: str, body: dict):
            if path == "/api/models":
                M = track("models")
                key, fields = str(body.get("key") or "").strip(), body.get("fields") or {}
                if not isinstance(fields, dict):
                    return self._json({"ok": False, "errors": ["fields must be an object"]}, 400)
                errors = M.validate(key, fields)
                if errors:
                    return self._json({"ok": False, "errors": list(errors)}, 400)
                try:
                    e = M.save_entry(key, fields)
                except ValueError as err:
                    return self._json({"ok": False, "errors": [str(err)]}, 400)
                return self._json({"ok": True, "model": {**e.to_dict(), "key": e.key}})
            if path == "/api/models/delete":
                ok = bool(track("models").delete_entry(str(body.get("key") or "")))
                return self._json({"ok": ok}, 200 if ok else 404)
            if path == "/api/models/preflight":
                M = track("models")
                e = M.load_registry().get(str(body.get("key") or ""))
                if e is None:
                    return self._json({"ok": False, "error": f"no model {body.get('key')!r} in the registry"}, 404)
                return self._json(M.preflight(e))
            if path == "/api/keys":                 # the value is written to ./.env and never sent back
                env, value = str(body.get("env") or ""), body.get("value")
                if not KEY_VAR.match(env):
                    return self._json({"ok": False, "error": "the variable must look like NAME_API_KEY"}, 400)
                if not isinstance(value, str) or not value.strip() or "\n" in value or len(value) > 4096:
                    return self._json({"ok": False, "error": "paste the key: one line of text"}, 400)
                try:
                    track("models").set_api_key(env, value.strip())
                except ValueError as err:
                    msg = str(err).replace(value.strip(), "***") if value.strip() else str(err)
                    return self._json({"ok": False, "error": msg}, 400)
                return self._json({"ok": True, "env": env, "set": _has_key(env)})
            if path == "/api/env/freeze":
                name = str(body.get("name") or "")
                if not ENV_NAME.match(name):
                    return self._json({"ok": False, "error": "a version name: letters, digits, . _ - (e.g. "
                                                             "civitas-v2)"}, 400)
                env = track("envs").freeze(name, str(envcheck.suite()), probe_result=envcheck.probe())
                envcheck.forget()
                return self._json({"ok": True, "env": env})
            if path == "/api/evals/plan":           # prepare() without starting: the Evaluate dialog's numbers
                ev, _ = jobs.prepare(body)
                return self._json(plan_summary(ev))
            if path == "/api/evals/start":
                out = jobs.start(body)
                if out.get("errors"):
                    return self._json({"ok": False, **out}, 400)
                return self._json({"ok": True, **out})
            if path == "/api/evals/stop":
                ok = jobs.stop(str(body.get("dir") or ""))
                return self._json({"ok": ok}, 200 if ok else 404)
            return self._json({"error": "not found"}, 404)

        def _status(self, d: Path) -> str:
            if (d / "summary.json").exists():
                return "finished"
            if (d / "PAUSED").exists():            # a supervised run between sessions (civitas resume)
                return "paused"
            if (d / "STOPPED").exists():
                return "stopped"
            if control.held(d) is not None:         # waiting on the viewer's Resume (fresh heartbeat)
                return "held"
            if control.pausing(d) and self._alive(d):
                return "pausing"
            return self._running_status(d)

        def _alive(self, d: Path) -> bool:
            return self._running_status(d) == "running"

        def _running_status(self, d: Path) -> str:
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
    envcheck = EnvCheck(Path(configs_dir).resolve(), runs_dir.resolve())
    jobs = EvalJobs(runs_dir.resolve(), Path(configs_dir).resolve()) if allow_launch else None
    httpd = ThreadingHTTPServer((host, port), make_handler(runs_dir, hub, launcher, configs_dir, envcheck, jobs))
    httpd.launcher, httpd.envcheck, httpd.jobs = launcher, envcheck, jobs     # evals keep running when it stops
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
