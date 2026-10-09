"""Model evaluations (`civitas eval`): a suite of late-start runs played by one model, scored against the scripted
yardstick of each run, in one scorecard.

A suite file (configs/eval_late.toml) names its runs (`[[runs]]`: a config and its seeds, with the reference wall
time and tokens per run that the dry run's estimates use) and `[eval]` defaults (workers, caffeinate). Every config
must have a warm-up (`[game].warmup_months`): the configured agents take over a grown city and the scorecard compares
what their months added with what the scripted agents added from the same hand-over (bench/yardstick.py).

`plan()` turns a suite plus the command line (`--model`, `--provider`, `--agent`, `--set`) into one `civitas run`
command per config and seed, and `check()` refuses a plan that would not test the model (a fleet agent left on
another model, e.g. "scripted"; no model-backed agent at all; a remote agent) or could not reach it (unknown
provider, its API key env var unset). `execute()` runs the commands as parallel subprocesses (`[eval].workers`, under
`caffeinate -dims` on macOS), each a normal run folder in runs/ with its log in runs/_logs/, and keeps
`runs/eval_<name>_<stamp>/eval.json` up to date; `resume()` reruns the runs that didn't finish (a new run id per
attempt, the old folder left as it is). `adopt()` makes an eval of runs that already exist.

`scorecard()` reads the run folders (summary.json, yardstick.json, fleet_report.json, trajectories.jsonl,
frames.jsonl through fleet/report.py: run_collaboration) into per-run records (a region: per city and the region
total) and aggregates; `markdown()` writes scorecard.md and `compare()` prints several evals side by side.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import stats as S

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

VERSION = 1
FILE = "eval.json"
CARD = "scorecard.md"
PREFIX = "eval_"
WORKERS = 5                  # runs at once: they wait on the API, not the CPU
POLL_S = 1.0
ROOT = Path(__file__).resolve().parents[2]          # the checkout this package runs from (its git commit)
MODEL_KINDS = ("fleet", "llm")


class EvalError(ValueError):
    """A suite, plan or eval folder the eval refuses, with what to do about it."""


# ---------------------------------------------------------------------------------------------------- suite
@dataclass
class Entry:
    config: str                         # as written in the suite
    path: Path                          # resolved
    seeds: list[int]
    minutes: float | None = None        # reference wall time per run
    prompt_tokens: int | None = None    # reference tokens per run
    completion_tokens: int | None = None
    repeats: int | None = None          # plays of each seed (None: the suite's [eval].repeats)


@dataclass
class Suite:
    file: str
    name: str
    description: str
    entries: list[Entry]
    workers: int = WORKERS
    caffeinate: bool = True
    reference_model: str | None = None
    prices: dict = field(default_factory=dict)      # model -> [USD per million prompt tokens, per million completion]
    repeats: int = 1                    # plays of each config + seed (a unit): [eval].repeats


def _repeats(v, where: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise EvalError(f"{where}: repeats must be a whole number of at least 1, got {v!r}")
    return v


def parse_seeds(v) -> list[int]:
    """[0, 1, 2], 2, "0-2" or "0,3-4"."""
    if isinstance(v, bool):
        raise EvalError(f"seeds must be numbers, got {v!r}")
    if isinstance(v, int):
        return [v]
    if isinstance(v, list):
        if not all(isinstance(s, int) and not isinstance(s, bool) for s in v):
            raise EvalError(f"seeds must be whole numbers, got {v!r}")
        return list(v)
    out = []
    for part in str(v).split(","):
        part = part.strip()
        try:
            if "-" in part:
                a, b = part.split("-")
                out.extend(range(int(a), int(b) + 1))
            elif part:
                out.append(int(part))
        except ValueError:
            raise EvalError(f"bad seeds {v!r}: use a list, a number or a range such as \"0-2\"") from None
    return out


def _resolve(config: str, suite_file: Path) -> Path:
    for base in (Path.cwd(), suite_file.parent, suite_file.parent.parent):
        p = (base / config) if not Path(config).is_absolute() else Path(config)
        if p.is_file():
            return p.resolve()
    raise EvalError(f"{suite_file}: config {config!r} not found (paths are relative to the project folder)")


def is_suite(data: dict) -> bool:
    """A TOML file read as a suite: it has `[[runs]]` entries with a `config` (the launcher leaves these out)."""
    runs = data.get("runs")
    return isinstance(runs, list) and bool(runs) and all(isinstance(r, dict) and "config" in r for r in runs)


def load_suite(path: str | Path) -> Suite:
    p = Path(path)
    try:
        data = tomllib.loads(p.read_text())
    except OSError as e:
        raise EvalError(f"no suite at {p}: {e.strerror or e}") from None
    except tomllib.TOMLDecodeError as e:
        raise EvalError(f"{p}: {e}") from None
    if not is_suite(data):
        raise EvalError(f"{p} is not a suite: it needs [[runs]] entries, each with a config and its seeds")
    ev = data.get("eval") or {}
    entries = []
    for i, r in enumerate(data["runs"]):
        if "seeds" not in r:
            raise EvalError(f"{p}: [[runs]] #{i + 1} ({r['config']}) has no seeds")
        seeds = parse_seeds(r["seeds"])
        if not seeds:
            raise EvalError(f"{p}: [[runs]] #{i + 1} ({r['config']}) has no seeds")
        num = lambda k, t: None if r.get(k) is None else t(r[k])
        entries.append(Entry(config=str(r["config"]), path=_resolve(str(r["config"]), p), seeds=seeds,
                             minutes=num("minutes", float), prompt_tokens=num("prompt_tokens", int),
                             completion_tokens=num("completion_tokens", int),
                             repeats=None if r.get("repeats") is None
                             else _repeats(r["repeats"], f"{p}: [[runs]] #{i + 1}")))
    workers = ev.get("workers", WORKERS)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise EvalError(f"{p}: [eval].workers must be a whole number of at least 1, got {workers!r}")
    prices = {str(k): [float(x) for x in v] for k, v in (ev.get("prices") or {}).items()
              if isinstance(v, list) and len(v) == 2}
    return Suite(file=str(path), name=str(data.get("name") or p.stem), description=str(data.get("description") or ""),
                 entries=entries, workers=workers, caffeinate=bool(ev.get("caffeinate", True)),
                 reference_model=ev.get("reference_model"), prices=prices,
                 repeats=_repeats(ev.get("repeats", 1), f"{p}: [eval]"))


# ---------------------------------------------------------------------------------------------------- plan
def _label(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-.")[:48] or "eval"


def lineup(cfg: dict) -> list[dict]:
    """Every role of `cfg` (a region: "north.mayor" ...) with its kind, and for a model-backed agent its provider and
    model: {role, kind, provider, model}."""
    from ..config import agent_spec, game_config
    from ..fleet.config import fleet_agent_config
    g = game_config(cfg)
    roles = [f"{c}.{r}" for r in g.roles for c in g.cities] if g.region else list(g.roles)
    out = []
    for role in roles:
        spec = agent_spec(cfg, role)
        kind = spec.get("kind", "scripted")
        row = {"role": role, "kind": kind, "provider": None, "model": None}
        if kind == "fleet":
            fc = fleet_agent_config(cfg, role)
            row.update(provider=fc.provider, model=fc.model)
        elif kind == "llm":
            row.update(provider=spec.get("provider", "arcee"), model=spec.get("model"))
        out.append(row)
    return out


def model_backed(rows: list[dict]) -> list[dict]:
    from ..fleet.scripted import MODEL as SCRIPTED
    return [r for r in rows if r["kind"] == "llm" or (r["kind"] == "fleet" and r["model"] != SCRIPTED)]


def model_overrides(cfg: dict, model: str | None, provider: str | None) -> tuple[list[str], list[str]]:
    """(--agent, --set) arguments that put every fleet agent (and llm agent) of `cfg` on `model` / `provider`:
    `[fleet].model` / `.provider`, and each `[agents.<role>]` of kind fleet or llm that sets its own model or provider
    (not one pinned to "scripted", which check() then refuses)."""
    from ..fleet.scripted import MODEL as SCRIPTED
    if not model and not provider:
        return [], []
    sets = ([f"fleet.model={model}"] if model else []) + ([f"fleet.provider={provider}"] if provider else [])
    fleet = cfg.get("fleet") or {}
    agents = []
    for key, spec in (cfg.get("agents") or {}).items():
        if not isinstance(spec, dict) or spec.get("kind") not in MODEL_KINDS:
            continue
        kind = spec["kind"]
        if kind == "fleet" and "model" not in spec and "provider" not in spec:
            continue                                     # it plays on [fleet]'s model
        if kind == "fleet" and spec.get("model") == SCRIPTED:
            continue
        prov = provider or spec.get("provider") or (fleet.get("provider") if kind == "fleet" else None) or "arcee"
        mod = model or spec.get("model") or (fleet.get("model") if kind == "fleet" else None)
        if mod:
            agents.append(f"{key}={kind}:{prov}/{mod}")
    return agents, sets


def fleet_turns(cfg: dict) -> int:
    """Turns of model-backed agents after the warm-up: acting months x such roles."""
    from ..config import game_config
    g = game_config(cfg)
    k = max(1, g.act_every)
    months = sum(1 for m in range(g.warmup_months or 0, g.months) if m % k == 0)
    return months * len(model_backed(lineup(cfg)))


def effective(entry_path: Path, agents: list[str], sets: list[str]) -> dict:
    from ..config import apply_cli_overrides, load_config
    return apply_cli_overrides(load_config(entry_path), agents, sets)


def plan(suite: Suite, *, model: str | None = None, provider: str | None = None, agents: list[str] = (),
         sets: list[str] = (), name: str | None = None, runs_dir: str | Path = "runs", workers: int | None = None,
         stamp: str | None = None, repeats: int | None = None) -> dict:
    """The eval record before it runs: one run per config, seed and repeat with its run id and `civitas run` command,
    the model, the folder (`dir`), estimates from the suite's reference numbers. `repeats` (--repeats) plays every unit
    (config + seed) that many times over the suite's own repeats; repeats of a unit share its hand-over, and their run
    ids end in `_r<i>` before the label when a unit has more than one. `check(ev)` says what would refuse it."""
    from ..config import apply_cli_overrides, load_config
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    runs_dir = Path(runs_dir)
    runs, models = [], set()
    for e in suite.entries:
        base = load_config(e.path)
        user = apply_cli_overrides(base, list(agents), list(sets))
        m_agents, m_sets = model_overrides(user, model, provider)
        a_args, s_args = [*agents, *m_agents], [*sets, *m_sets]
        cfg = apply_cli_overrides(base, a_args, s_args)
        rows = model_backed(lineup(cfg))
        models |= {(r["provider"], r["model"]) for r in rows}
        turns, ref_turns = fleet_turns(cfg), fleet_turns(base)
        scale = turns / ref_turns if ref_turns else 1.0
        from ..config import game_config
        g = game_config(cfg)
        k = repeats if repeats is not None else (e.repeats or suite.repeats)
        for seed, rep in [(s, i) for s in e.seeds for i in range(1, _repeats(k, "--repeats") + 1)]:
            runs.append({"config": e.config, "config_path": str(e.path), "name": cfg["name"], "seed": seed,
                         "repeat": rep, "repeats": k, "region": bool(g.region), "cities": list(g.cities) if g.region else [],
                         "warmup": g.warmup_months, "months": g.months, "fleet_turns": turns,
                         "agent_args": a_args, "set_args": s_args,
                         "estimate": {"minutes": None if e.minutes is None else round(e.minutes * scale, 1),
                                      "prompt_tokens": None if e.prompt_tokens is None
                                      else int(e.prompt_tokens * scale),
                                      "completion_tokens": None if e.completion_tokens is None
                                      else int(e.completion_tokens * scale)},
                         "status": "pending", "attempts": []})
    eff_model = model or (next(iter(models))[1] if len(models) == 1 else ("mixed" if models else None))
    eff_prov = provider or (next(iter(models))[0] if len({p for p, _ in models}) == 1 else None)
    label = _label(name or eff_model or suite.name)
    for r in runs:
        r["run_id"] = f"{stamp}_{r['name']}_s{r['seed']}" + (f"_r{r['repeat']}" if r["repeats"] > 1 else "") + f"_{label}"
    ids = [r["run_id"] for r in runs]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    ev = {"version": VERSION, "name": label, "dir": str(runs_dir / f"{PREFIX}{label}_{stamp}"),
          "runs_dir": str(runs_dir), "created": time.time(), "stamp": stamp,
          "suite": {"file": suite.file, "name": suite.name, "description": suite.description,
                    "reference_model": suite.reference_model,
                    "entries": [{"config": e.config, "seeds": e.seeds} for e in suite.entries]},
          "repeats": sorted({r["repeats"] for r in runs}),
          "model": eff_model, "provider": eff_prov, "agents": list(agents), "sets": list(sets),
          "workers": int(workers or suite.workers), "caffeinate": suite.caffeinate, "prices": suite.prices,
          "status": "planned", "runs": runs}
    if dup:
        ev["duplicate_runs"] = dup
    return ev


def check(ev: dict) -> list[str]:
    """Why the eval must not start (empty: it may). Every run's effective config: a warm-up (the scorecard compares
    with the scripted yardstick from its hand-over), at least one model-backed agent, no remote agent, each fleet
    agent on the asked model (none left on "scripted"), each provider known and its API key env var set."""
    from ..config import provider_config, warmup_months
    from ..fleet.scripted import MODEL as SCRIPTED
    errs, keys = [], {}
    if ev.get("duplicate_runs"):
        errs.append(f"the suite lists a config and seed twice: {', '.join(ev['duplicate_runs'])}")
    want = ev.get("model") if ev.get("model") != "mixed" else None
    asked = bool(want) and any(s.startswith("fleet.model=") for r in ev["runs"] for s in r["set_args"])
    for r in ev["runs"]:
        where = f"{r['config']} seed {r['seed']}"
        try:
            cfg = effective(Path(r["config_path"]), r["agent_args"], r["set_args"])
            if not warmup_months(cfg):
                errs.append(f"{where}: no warm-up ([game].warmup_months): an eval scores the months after a "
                            f"scripted warm-up against the scripted yardstick from the same hand-over")
        except (ValueError, OSError) as e:
            errs.append(f"{where}: {e}")
            continue
        rows = lineup(cfg)
        if any(x["kind"] == "remote" for x in rows):
            errs.append(f"{where}: remote agents can't be evaluated here (they need a live process)")
        backed = model_backed(rows)
        if not backed:
            errs.append(f"{where}: no agent calls a model (all scripted): nothing to evaluate")
        if asked:
            off = [f"{x['role']}={x['model']}" for x in rows if x["kind"] == "fleet" and x["model"] != want]
            if off:
                scripted = [o for o in off if o.endswith("=" + SCRIPTED)]
                errs.append(f"{where}: --model {want} but these fleet agents would play on another model: "
                            f"{', '.join(off)}" + (" (pinned to the scripted baseline in [agents.<role>]; drop "
                                                   "their model key or evaluate a config without them)"
                                                   if scripted else ""))
        for x in backed:
            keys.setdefault(x["provider"], (cfg, where))
    for prov, (cfg, where) in keys.items():
        try:
            pc = provider_config(cfg, prov)
        except ValueError as e:
            errs.append(f"{where}: {e}")
            continue
        if pc.api_key_env and not os.environ.get(pc.api_key_env):
            errs.append(f"provider {prov!r} needs the API key in ${pc.api_key_env}, which isn't set "
                        f"(export it or put {pc.api_key_env}=... in ./.env)")
    return list(dict.fromkeys(errs))


def command(ev: dict, r: dict) -> list[str]:
    """The run's `civitas run` command (without caffeinate)."""
    cmd = [sys.executable, "-m", "civitas.cli", "run", "-c", r["config_path"], "--seed", str(r["seed"]),
           "--runs-dir", str(Path(ev["runs_dir"]).resolve()), "--run-id", r["run_id"]]
    for a in r["agent_args"]:
        cmd += ["--agent", a]
    for s in r["set_args"]:
        cmd += ["--set", s]
    return cmd


def makespan(minutes: list[float], workers: int) -> float:
    """Wall time of jobs this long on `workers` slots, longest first (as the eval starts them in suite order, close)."""
    slots = [0.0] * max(1, workers)
    for m in sorted(minutes, reverse=True):
        i = slots.index(min(slots))
        slots[i] += m
    return max(slots) if minutes else 0.0


def _mins(m: float | None) -> str:
    if m is None:
        return "?"
    return f"{m / 60:.1f} h" if m >= 90 else f"{m:.0f} min"


def _tok(n: float | None) -> str:
    if n is None:
        return "?"
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e3 else f"{n:.0f}"


def _usd(ev: dict, prompt: float, completion: float) -> float | None:
    p = (ev.get("prices") or {}).get(ev.get("model") or "")
    return None if not p else prompt / 1e6 * p[0] + completion / 1e6 * p[1]


def dry_run_text(ev: dict, errors: list[str]) -> str:
    """The plan as `civitas eval --dry-run` prints it: runs, model, estimates, the first command, refusals."""
    runs = ev["runs"]
    units = len({(r["name"], r["seed"]) for r in runs})
    ks = sorted({r.get("repeats") or 1 for r in runs})
    reps = f" ({units} units x {ks[0]} repeats)" if ks != [1] and len(ks) == 1 else \
        f" ({units} units, {'/'.join(map(str, ks))} repeats)" if ks != [1] else ""
    L = [f"Eval plan '{ev['name']}': suite {ev['suite']['name']} ({ev['suite']['file']}), {len(runs)} runs{reps}, "
         f"model {ev['model'] or '?'}" + (f" (provider {ev['provider']})" if ev.get("provider") else "")
         + f", {ev['workers']} at a time" + (" under caffeinate -dims" if _caffeinate(ev) else ""), ""]
    w = max(len(r["run_id"]) for r in runs) if runs else 10
    L.append(f"  {'run id':<{w}}  {'config':<34} {'seed':>4}  {'turns':>5}  {'tokens (est.)':>13}  {'time (est.)':>11}")
    for r in runs:
        e = r["estimate"]
        tok = None if e["prompt_tokens"] is None else e["prompt_tokens"] + (e["completion_tokens"] or 0)
        L.append(f"  {r['run_id']:<{w}}  {r['config']:<34} {r['seed']:>4}  {r['fleet_turns']:>5}  {_tok(tok):>13}  "
                 f"{_mins(e['minutes']):>11}")
    est = [r["estimate"] for r in runs]
    p = sum(e["prompt_tokens"] or 0 for e in est)
    c = sum(e["completion_tokens"] or 0 for e in est)
    known = [e["minutes"] for e in est if e["minutes"] is not None]
    L += ["", f"Total: {sum(r['fleet_turns'] for r in runs)} model turns, ~{_tok(p + c)} tokens "
              f"(~{_tok(p)} prompt, ~{_tok(c)} completion), ~{_mins(makespan(known, ev['workers']))} with "
              f"{ev['workers']} workers ({_mins(sum(known))} of runs in all)."]
    usd = _usd(ev, p, c)
    if usd is not None:
        L.append(f"Cost at the suite's prices for {ev['model']}: ~${usd:,.2f}.")
    ref = ev["suite"].get("reference_model")
    if ref:
        L.append(f"Estimates are the suite's reference numbers ({ref}"
                 + (", this model's will differ" if ev.get("model") != ref else "") + ").")
    L += ["", f"Output: {ev['dir']}/ (eval.json, scorecard.md); runs in {ev['runs_dir']}/, logs in "
              f"{ev['runs_dir']}/_logs/", "", "First command:", "  " + " ".join(_quote(x) for x in command(ev, runs[0]))
          if runs else "  (no runs)"]
    if errors:
        L += ["", "The eval would refuse to start:"] + [f"  - {e}" for e in errors]
    else:
        L += ["", "Nothing was run (dry run). Drop --dry-run to start."]
    return "\n".join(L)


def _quote(s: str) -> str:
    return s if re.fullmatch(r"[A-Za-z0-9_./=:@%+-]+", s) else "'" + s.replace("'", "'\\''") + "'"


# ---------------------------------------------------------------------------------------------------- execute
def _caffeinate(ev: dict) -> bool:
    return bool(ev.get("caffeinate")) and sys.platform == "darwin" and shutil.which("caffeinate") is not None


def save(ev: dict) -> None:
    d = Path(ev["dir"])
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (FILE + ".tmp")
    tmp.write_text(json.dumps(ev, indent=2))
    os.replace(tmp, d / FILE)


def load(eval_dir: str | Path) -> dict:
    d = Path(eval_dir)
    try:
        ev = json.loads((d / FILE).read_text())
    except (OSError, ValueError) as e:
        raise EvalError(f"no eval at {d} ({FILE}: {e})") from None
    ev["dir"] = str(d)                          # the folder may have moved since
    if not Path(ev.get("runs_dir") or "").is_dir():
        ev["runs_dir"] = str(d.parent)
    return ev


def _env() -> dict:
    """The child's environment: this checkout's package first on the path (a worktree runs its own code)."""
    path = os.environ.get("PYTHONPATH", "")
    return {**os.environ, "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(ROOT) + (os.pathsep + path if path else "")}


def _alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def run_status(runs_dir: Path, r: dict) -> str:
    """finished (summary.json), paused, or what eval.json says (pending / running / failed / interrupted)."""
    d = runs_dir / r["run_id"]
    if (d / "summary.json").exists():
        return "finished"
    if (d / "PAUSED").exists():
        return "paused"
    st = r.get("status") or "pending"
    return "incomplete" if st == "finished" else st     # eval.json said finished, the run folder doesn't


def code_version(root: Path = ROOT) -> dict | None:
    """{commit, dirty} of the checkout this package runs from, None outside git."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=root, capture_output=True, text=True,
                              timeout=10)
        if head.returncode:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, timeout=10)
        return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return None


def commit_before(ts: float | None, root: Path = ROOT) -> dict | None:
    """The last commit in this checkout's history made before `ts` (a run's start): {commit, date, subject,
    inferred: True}. A guess at the code a run that recorded none played: uncommitted changes don't show."""
    if not ts:
        return None
    try:
        out = subprocess.run(["git", "log", "-1", f"--before=@{int(ts)}", "--format=%h%x09%cI%x09%s"], cwd=root,
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    parts = out.stdout.strip().split("\t", 2)
    if out.returncode or len(parts) < 3:
        return None
    return {"commit": parts[0], "date": parts[1], "subject": parts[2], "inferred": True}


def unit_key(r: dict) -> str:
    """A run's unit (config name + seed): what repeats share and what a paired comparison matches."""
    return f"{r.get('name') or r.get('config')} s{r.get('seed')}"


def run_label(r: dict) -> str:
    """The unit, and the repeat when the unit has several: "fleet_smoke_late s1 r2"."""
    return unit_key(r) + (f" r{r.get('repeat') or 1}" if (r.get("repeats") or 1) > 1 else "")


def _start(ev: dict, r: dict, runs_dir: Path) -> tuple[subprocess.Popen, Path]:
    logs = runs_dir / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{r['run_id']}.log"
    cmd = command(ev, r)
    if _caffeinate(ev):
        cmd = ["caffeinate", "-dims", *cmd]
    with open(log, "w") as fh:                      # the child keeps its own handle
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=os.getcwd(), env=_env(),
                                start_new_session=True)
    r.update(status="running", pid=proc.pid, log=str(log), started=time.time(), ended=None, returncode=None,
             cmd=cmd)
    return proc, log


def _stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()


def _tail(log: str | None) -> str:
    try:
        lines = [x for x in Path(log).read_text().strip().splitlines() if x.strip()]
    except (OSError, TypeError):
        return ""
    return lines[-1][:300] if lines else ""


def execute(ev: dict, *, verbose: bool = True, out=print) -> dict:
    """Run every run of `ev` whose status isn't finished, `ev["workers"]` at a time, saving eval.json as they start
    and end; then the scorecard (eval.json `scorecard`, scorecard.md). Ctrl+C stops the runs (status
    "interrupted"; `civitas eval --resume <dir>` reruns them)."""
    runs_dir = Path(ev["runs_dir"])
    todo = [r for r in ev["runs"] if run_status(runs_dir, r) != "finished"]
    for r in ev["runs"]:
        if r not in todo:
            r["status"] = "finished"
    ev.update(status="running", pid=os.getpid(), started=ev.get("started") or time.time(), ended=None)
    ev.setdefault("code", code_version())
    save(ev)
    running: dict[str, tuple[subprocess.Popen, dict]] = {}
    queue = list(todo)
    t0 = time.time()
    try:
        while queue or running:
            for rid, (proc, r) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                del running[rid]
                r.update(returncode=rc, ended=time.time(), pid=None)
                st = run_status(runs_dir, {**r, "status": "failed"})
                r["status"] = st
                if st != "finished":
                    r["error"] = _tail(r.get("log")) or f"exit code {rc}"
                save(ev)
                if verbose:
                    took = (r["ended"] - r["started"]) / 60
                    line = f"{st}: {run_label(r)} after {took:.1f} min"
                    if st == "finished":
                        c = run_card(runs_dir / r["run_id"])
                        t = c.get("total") or {}
                        if t.get("delta") is not None:
                            line += (f": city {t['change']['city']:+.1f} vs scripted {t['yard_change']['city']:+.1f} "
                                     f"({t['delta']['city']:+.1f})")
                    else:
                        line += f" ({r['error']})"
                    out(f"[eval {ev['name']}] {line}")
            while queue and len(running) < max(1, int(ev["workers"])):
                r = queue.pop(0)
                proc, log = _start(ev, r, runs_dir)
                running[r["run_id"]] = (proc, r)
                save(ev)
                if verbose:
                    out(f"[eval {ev['name']}] started {run_label(r)} as {r['run_id']} (log {log})")
            if running:
                time.sleep(POLL_S)
    except KeyboardInterrupt:
        for proc, r in running.values():
            _stop(proc)
            r.update(status="interrupted", ended=time.time(), pid=None)
        ev.update(status="interrupted", pid=None, ended=time.time())
        save(ev)
        out(f"[eval {ev['name']}] interrupted; rerun what didn't finish with: civitas eval --resume {ev['dir']}")
        raise
    ev["wall_time_s"] = round((ev.get("wall_time_s") or 0) + time.time() - t0, 1)
    finish(ev)
    return ev


def finish(ev: dict) -> dict:
    """Status from the runs, the scorecard into eval.json and scorecard.md."""
    runs_dir = Path(ev["runs_dir"])
    for r in ev["runs"]:
        r["status"] = run_status(runs_dir, r)
    done = sum(r["status"] == "finished" for r in ev["runs"])
    ev.update(status="finished" if done == len(ev["runs"]) else "partial", pid=None, ended=time.time())
    ev["scorecard"] = scorecard(ev)
    save(ev)
    (Path(ev["dir"]) / CARD).write_text(markdown(ev))
    return ev


def resume(eval_dir: str | Path, *, workers: int | None = None, verbose: bool = True, out=print) -> dict:
    """Rerun an eval's runs that didn't finish: each under a new run id (`<id>_a2`, ...; the old folder stays), after
    the same checks as a new eval. EvalError while it still runs."""
    ev = load(eval_dir)
    live = [r["run_id"] for r in ev["runs"] if r.get("status") == "running" and _alive(r.get("pid"))]
    if (ev.get("status") == "running" and _alive(ev.get("pid")) and ev.get("pid") != os.getpid()) or live:
        raise EvalError(f"eval {ev['name']} is still running (pid {ev.get('pid')}"
                        + (f"; runs {', '.join(live)}" if live else "") + ")")
    errs = check(ev)
    if errs:
        raise EvalError("can't resume:\n  - " + "\n  - ".join(errs))
    runs_dir = Path(ev["runs_dir"])
    for r in ev["runs"]:
        st = run_status(runs_dir, r)
        if st == "finished":
            r["status"] = st
            continue
        if (runs_dir / r["run_id"]).exists():        # a failed or interrupted attempt: keep it, start a new one
            r["attempts"] = [*r.get("attempts", []), {"run_id": r["run_id"], "status": st, "error": r.get("error")}]
            base = r.get("base_id") or r["run_id"]
            r["base_id"] = base
            r["run_id"] = f"{base}_a{len(r['attempts']) + 1}"
        r.update(status="pending", error=None)
    if workers:
        ev["workers"] = workers
    ev["resumed"] = [*ev.get("resumed", []), time.time()]
    return execute(ev, verbose=verbose, out=out)


def start(ev: dict, *, verbose: bool = True, out=print) -> dict:
    errs = check(ev)
    if errs:
        raise EvalError("refusing to start the eval:\n  - " + "\n  - ".join(errs))
    if Path(ev["dir"]).exists():
        raise EvalError(f"{ev['dir']} exists already")
    ev["code"] = code_version()
    save(ev)
    if verbose:
        out(f"[eval {ev['name']}] {len(ev['runs'])} runs, {ev['workers']} at a time; folder {ev['dir']}")
    return execute(ev, verbose=verbose, out=out)


# ---------------------------------------------------------------------------------------------------- adopt
def adopt(run_ids: list[str], suite: Suite, *, name: str, runs_dir: str | Path = "runs") -> dict:
    """An eval of runs that exist already (no run is played): each matched to the suite's entry by its config name
    and seed, its code version inferred from git (the last commit before it started); then the scorecard. Several
    runs of one config + seed are repeats of that unit, numbered in the order given."""
    from ..config import load_config
    from .compare import model_of
    runs_dir = Path(runs_dir)
    by_name = {}
    for e in suite.entries:
        by_name[load_config(e.path)["name"]] = e
    runs, models, provs = [], set(), set()
    if len(set(run_ids)) < len(run_ids):
        raise EvalError("a run is given twice")
    for rid in run_ids:
        d = runs_dir / rid
        try:
            meta = json.loads((d / "meta.json").read_text())
        except (OSError, ValueError):
            raise EvalError(f"no run {rid!r} in {runs_dir}") from None
        cfg = meta.get("config") or {}
        e = by_name.get(meta.get("name"))
        seed = meta.get("seed")
        if e is None or seed not in e.seeds:
            raise EvalError(f"run {rid} ({meta.get('name')} seed {seed}) isn't in suite {suite.name}: its runs are "
                            + ", ".join(f"{n} seeds {x.seeds}" for n, x in by_name.items()))
        m = model_of(meta.get("agents") or {})
        models.add(m)
        provs.add((cfg.get("fleet") or {}).get("provider"))
        g = cfg.get("game") or {}
        runs.append({"run_id": rid, "config": e.config, "config_path": str(e.path), "name": meta.get("name"),
                     "seed": seed, "region": bool(meta.get("region")), "cities": meta.get("cities") or [],
                     "warmup": int(meta.get("warmup_months") or 0), "months": meta.get("months"),
                     "fleet_turns": None, "agent_args": [], "set_args": [], "estimate": {},
                     "status": run_status(runs_dir, {"run_id": rid, "status": "failed"}), "attempts": [],
                     "adopted": True, "started": meta.get("started"),
                     "code": commit_before(meta.get("started")), "act_every": g.get("act_every")})
    keys = [(r["name"], r["seed"]) for r in runs]
    for r in runs:
        mine = [x for x in runs if (x["name"], x["seed"]) == (r["name"], r["seed"])]
        r.update(repeat=mine.index(r) + 1, repeats=len(mine))
    missing = [f"{n} s{s}" for n, e in by_name.items() for s in e.seeds if (n, s) not in keys]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    label = _label(name)
    ev = {"version": VERSION, "name": label, "dir": str(runs_dir / f"{PREFIX}{label}_{stamp}"),
          "runs_dir": str(runs_dir), "created": time.time(), "stamp": stamp, "adopted": True,
          "suite": {"file": suite.file, "name": suite.name, "description": suite.description,
                    "reference_model": suite.reference_model,
                    "entries": [{"config": e.config, "seeds": e.seeds} for e in suite.entries]},
          "model": next(iter(models)) if len(models) == 1 else "mixed",
          "provider": next(iter(provs)) if len(provs) == 1 else None, "agents": [], "sets": [],
          "workers": None, "caffeinate": None, "prices": suite.prices, "status": "adopted", "runs": runs,
          "missing": missing}
    if Path(ev["dir"]).exists():
        raise EvalError(f"{ev['dir']} exists already")
    return finish(ev)


# ---------------------------------------------------------------------------------------------------- scorecard
def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _r(v, n=3):
    return None if v is None else round(float(v), n)


def _tokens(d: Path, cities: list[str]) -> dict:
    """Model turns and tokens after the warm-up from trajectories.jsonl: {"": totals, city: totals}, each
    {turns, prompt, completion}; a model turn is a fleet agent's (not the scripted fleet's) or an llm agent's."""
    out = {c: {"turns": 0, "prompt": 0, "completion": 0} for c in ["", *cities]}
    p = d / "trajectories.jsonl"
    if not p.exists():
        return out
    with open(p) as fh:
        for line in fh:
            if '"warmup": true' in line or not line.endswith("\n"):
                continue
            try:
                t = json.loads(line)
            except ValueError:
                continue
            agent = str(t.get("agent") or "")
            if t.get("warmup") or agent == "fleet:scripted" or not agent.startswith(("fleet:", "llm:")):
                continue
            u = t.get("usage") or {}
            city = str(t.get("role") or "").split(".")[0] if "." in str(t.get("role") or "") else None
            for k in ("", city) if city in out else ("",):
                out[k]["turns"] += 1
                out[k]["prompt"] += int(u.get("prompt_tokens") or 0)
                out[k]["completion"] += int(u.get("completion_tokens") or 0)
    return out


CALLS = ("turns", "tool_calls", "actions", "failed", "refused", "tool_errors", "repeats", "api_errors")


def _calls(roles: dict | None) -> dict | None:
    if not roles:
        return None
    return {k: sum(int((r or {}).get(k) or 0) for r in roles.values()) for k in CALLS}


def _teamwork(d: Path, city: str | None, fallback: dict | None) -> float | None:
    from ..fleet.report import run_collaboration
    try:
        c = run_collaboration(d, city)
    except Exception:                               # the score is extra: fall back to the report's
        c = fallback
    return _r((c or {}).get("score"), 1)


def _handover_by_city(d: Path, warm: int, cities: list[str]) -> dict:
    """Each city's score and population in the frame of the hand-over month (frames.jsonl)."""
    out = {}
    p = d / "frames.jsonl"
    if not p.exists():
        return out
    key = f'"month":{warm},'
    with open(p) as fh:
        for line in fh:
            if key not in line[:40]:
                continue
            try:
                f = json.loads(line)
            except ValueError:
                continue
            if f.get("month") != warm:
                continue
            for c in cities:
                cf = (f.get("cities") or {}).get(c) or {}
                out[c] = {"city": (cf.get("scores") or {}).get("city"),
                          "population": (cf.get("metrics") or {}).get("population")}
            break
    return out


def _unit(final: float | None, pop: int | None, hand: dict | None, yard: dict | None, roles: dict,
          yard_roles: dict, teamwork, calls, tok) -> dict:
    """One scorecard line: the run's change since the hand-over and the yardstick's, their difference (delta), the
    departments' final scores and their difference from the yardstick's, Teamwork, calls and tokens."""
    change = None
    if hand and final is not None and hand.get("city") is not None:
        change = {"city": _r(final - hand["city"]), "population": None if pop is None or hand.get("population") is None
                  else pop - hand["population"]}
    ych = (yard or {}).get("change")
    delta = None
    if change and ych:
        delta = {"city": _r(change["city"] - ych["city"]),
                 "population": None if change["population"] is None else change["population"] - ych["population"]}
    rd = {k: _r(v - yard_roles[k]) for k, v in roles.items() if v is not None and yard_roles.get(k) is not None}
    return {"final": _r(final), "population": pop, "at_handover": hand, "change": change, "yard_change": ych,
            "yard_final": (yard or {}).get("at_end", {}).get("city") if yard else None, "delta": delta,
            "roles": {k: _r(v) for k, v in roles.items()}, "yard_roles": {k: _r(v) for k, v in yard_roles.items()},
            "role_delta": rd, "teamwork": teamwork, "calls": calls, "tokens": tok}


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def prompts_hash(meta: dict) -> str | None:
    p = meta.get("fleet_prompts")
    return hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:12] if p else None


def run_card(d: str | Path, *, compute_yardstick: bool = True) -> dict:
    """One run folder's scorecard record: status, model, versions, wall time, `total` (the run; a region's region
    values: its score and population, the departments' mean over its cities, Teamwork's mean, calls and tokens
    summed) and for a region `cities` (one _unit each). A finished warm-up run without yardstick.json gets it
    computed (scripted agents, seconds, no model calls)."""
    from . import yardstick as Y
    from .compare import model_of
    d = Path(d)
    meta = _read(d / "meta.json") or {}
    summ = _read(d / "summary.json")
    out = {"run_id": d.name, "name": meta.get("name"), "seed": meta.get("seed"), "region": bool(meta.get("region")),
           "cities": list(meta.get("cities") or []), "model": model_of(meta.get("agents") or {}),
           "fleet_version": meta.get("fleet_version"), "prompts": prompts_hash(meta), "started": meta.get("started"),
           "finished": summ is not None, "wall_time_s": (summ or {}).get("wall_time_s"),
           "bankrupt": (summ or {}).get("bankrupt"), "warmup": int(meta.get("warmup_months") or 0)}
    if not meta:
        out["error"] = "no meta.json"
        return out
    if summ is None:
        return out
    ys = Y.read(d)
    if ys is None and compute_yardstick and out["warmup"] and (summ.get("warmup") or {}).get("handed_over"):
        try:
            ys = Y.yardstick(d)
        except Exception as e:                      # the scorecard shows the run without its yardstick
            out["yardstick_error"] = f"{type(e).__name__}: {e}"
    out["yardstick_matches"] = (ys or {}).get("matches_run")
    w = summ.get("warmup") or {}
    hand = w.get("at_handover")
    hand = {"city": hand.get("city"), "population": hand.get("population")} if hand else None
    scores, fm = summ.get("final_scores") or {}, summ.get("final_metrics") or {}
    yscores = (ys or {}).get("final_scores") or {}
    rep = _read(d / "fleet_report.json") or {}
    cities = out["cities"] if out["region"] else []
    tok = _tokens(d, cities)
    out["fleet_turns"] = tok[""]["turns"]
    if not out["region"]:
        roles = {k: v for k, v in scores.items() if k != "city"}
        yroles = {k: v for k, v in yscores.items() if k != "city"}
        out["total"] = _unit(scores.get("city"), fm.get("population"), hand, ys, roles, yroles,
                             _teamwork(d, None, rep.get("collaboration")), _calls(rep.get("roles")), tok[""])
        return out
    units = {}
    yby = (ys or {}).get("by_city") or {}
    need = [c for c in cities if not ((yby.get(c) or {}).get("at_handover") and ys.get("matches_run"))]
    framed = _handover_by_city(d, out["warmup"], need) if need else {}
    for c in cities:
        pre = c + "."
        roles = {k[len(pre):]: v for k, v in scores.items() if k.startswith(pre) and k != pre + "city"}
        yroles = {k[len(pre):]: v for k, v in yscores.items() if k.startswith(pre) and k != pre + "city"}
        yh = (yby.get(c) or {}).get("at_handover")
        chand = framed.get(c) or ({"city": yh.get("city"), "population": yh.get("population")} if yh else None)
        crep = (rep.get("cities") or {}).get(c) or {}
        units[c] = _unit(scores.get(pre + "city"), ((fm.get("by_city") or {}).get(c) or {}).get("population"), chand,
                         yby.get(c), roles, yroles, _teamwork(d, c, crep.get("collaboration")),
                         _calls(crep.get("roles")), tok[c])
    names = list(dict.fromkeys(r for u in units.values() for r in u["roles"]))
    mean_roles = {r: _mean([u["roles"].get(r) for u in units.values()]) for r in names}
    mean_yroles = {r: _mean([u["yard_roles"].get(r) for u in units.values()]) for r in names}
    calls = [u["calls"] for u in units.values() if u["calls"]]
    out["total"] = _unit(scores.get("city"), fm.get("population"), hand, ys, mean_roles, mean_yroles,
                         _r(_mean([u["teamwork"] for u in units.values()]), 1),
                         {k: sum(x[k] for x in calls) for k in CALLS} if calls else None, tok[""])
    out["cities"] = cities
    out["by_city"] = units
    return out


def unit_values(cards: list[dict]) -> dict:
    """The scored runs grouped by unit (config + seed), in order: {unit: {runs, city, population, teamwork, roles,
    handover, values}}, each the mean over the unit's repeats of the run's value vs scripted (a region by its region
    values); `values` the city deltas of its runs, `handover` the city score at the hand-over (the same for every
    repeat while the warm-up code is the same)."""
    groups: dict[str, list] = {}
    for c in cards:
        u = c.get("total")
        if u and u.get("delta"):
            groups.setdefault(unit_key(c), []).append(u)
    out = {}
    for k, us in groups.items():
        roles = list(dict.fromkeys(r for u in us for r in u["role_delta"]))
        out[k] = {"runs": len(us), "city": _r(_mean([u["delta"]["city"] for u in us])),
                  "population": _r(_mean([u["delta"]["population"] for u in us]), 1),
                  "teamwork": _r(_mean([u.get("teamwork") for u in us]), 1),
                  "roles": {r: _r(_mean([u["role_delta"].get(r) for u in us])) for r in roles},
                  "handover": (us[0].get("at_handover") or {}).get("city"),
                  "values": [u["delta"]["city"] for u in us]}
    return out


def _ci(values, d=3) -> dict:
    """stats.mean_ci rounded: {n, mean, std, lo, hi} (all None without values)."""
    c = S.mean_ci(values) or {"n": 0, "mean": None, "std": None, "lo": None, "hi": None}
    return {k: v if k == "n" else _r(v, d) for k, v in c.items()}


def noise_of(units: dict) -> dict | None:
    """The replay noise of the city score vs scripted: the pooled standard deviation within units ({sd, df}), None
    while no unit has two runs."""
    n = S.pooled_sd([u["values"] for u in units.values()])
    return None if n is None else {"sd": _r(n["sd"]), "df": n["df"]}


def _mde(noise: dict | None, n_units: int) -> dict | None:
    if not noise or n_units < 2:
        return None
    return {str(k): _r(S.mde(noise["sd"], n_units, k), 2) for k in S.MDE_REPEATS}


def _agg(cards: list[dict], ev: dict) -> dict:
    """Aggregates over finished runs. Scores vs scripted are over units (config + seed, a region once by its region
    values), each the mean of its repeats: mean, std of the units (the seed spread) and a 95% t-interval (lo, hi);
    `values` lists every run's. `noise`: the replay noise (K >= 2), `mde`: the smallest difference a paired comparison
    with another eval on these units can see, by repeats."""
    units = [c["total"] for c in cards if c.get("total")]
    scored = [u for u in units if u.get("delta")]
    cd = [u["delta"]["city"] for u in scored]
    uv = unit_values(cards)
    U = list(uv.values())
    roles = list(dict.fromkeys(r for u in units for r in u["role_delta"]))
    calls = [u["calls"] for u in units if u.get("calls")]
    ct = {k: sum(x[k] for x in calls) for k in CALLS} if calls else None
    tk = {k: sum((u.get("tokens") or {}).get(k) or 0 for u in units) for k in ("turns", "prompt", "completion")}
    per = lambda k: _r(ct[k] / ct["turns"], 3) if ct and ct["turns"] else None
    wall = [c.get("wall_time_s") for c in cards if c.get("wall_time_s") is not None]
    roles_ci = {r: _ci([u["roles"].get(r) for u in U]) for r in roles}
    team = _ci([u["teamwork"] for u in U], 1) if U else _ci([u.get("teamwork") for u in units], 1)
    noise = noise_of(uv)
    out = {"runs": len(cards), "finished": len(units), "scored": len(scored), "n_units": len(uv),
           "repeats": sorted({u["runs"] for u in U}),
           "city_delta": {**_ci([u["city"] for u in U]), "values": cd},
           "population_delta": _ci([u["population"] for u in U], 1),
           "city_change": _r(_mean([u["change"]["city"] for u in units if u.get("change")])),
           "yard_change": _r(_mean([u["yard_change"]["city"] for u in scored])),
           "beats_scripted": sum(u["city"] > 0 for u in U),
           "roles": {r: c["mean"] for r, c in roles_ci.items()}, "roles_ci": roles_ci,
           "teamwork": team["mean"], "teamwork_ci": team,
           "noise": noise, "mde": _mde(noise, len(uv)), "units": uv,
           "calls": ct, "failed_per_turn": per("failed"), "refused_per_turn": per("refused"),
           "tool_errors_per_turn": per("tool_errors"),
           "tokens": {**tk, "total": tk["prompt"] + tk["completion"],
                      "per_turn": round((tk["prompt"] + tk["completion"]) / tk["turns"]) if tk["turns"] else None},
           "wall_time_s": {"sum": _r(sum(wall), 1), "max": _r(max(wall), 1) if wall else None},
           "bankrupt": sum(bool(c.get("bankrupt")) for c in cards if c.get("total"))}
    usd = _usd(ev, tk["prompt"], tk["completion"])
    if usd is not None:
        out["usd"] = round(usd, 4)
    return out


def scorecard(ev: dict, *, compute_yardstick: bool = True) -> dict:
    """{runs: [run_card per run of the eval], aggregate: over all, by_config: {name: aggregate}}."""
    runs_dir = Path(ev["runs_dir"])
    cards = []
    for r in ev["runs"]:
        c = run_card(runs_dir / r["run_id"], compute_yardstick=compute_yardstick)
        c.update(config=r["config"], status=run_status(runs_dir, r), code=r.get("code") or ev.get("code"),
                 repeat=r.get("repeat") or 1, repeats=r.get("repeats") or 1)
        if r.get("attempts"):
            c["attempts"] = [a["run_id"] for a in r["attempts"]]
        cards.append(c)
    names = list(dict.fromkeys(c["name"] or c["config"] for c in cards))
    return {"created": time.time(), "runs": cards, "aggregate": _agg(cards, ev),
            "by_config": {n: _agg([c for c in cards if (c["name"] or c["config"]) == n], ev) for n in names}}


# ---------------------------------------------------------------------------------------------------- text
def _f(v, d=1, sign=False):
    if v is None:
        return "n/a"
    return f"{v:+,.{d}f}" if sign else f"{v:,.{d}f}"


def plus_minus(a: dict, d=1) -> str:
    if a.get("mean") is None:
        return "n/a"
    return _f(a["mean"], d, True) + (f" ± {a['std']:,.{d}f}" if a.get("std") is not None else "")


def ci_text(c: dict | None, d=1) -> str:
    """"+1.4 (-0.5 to +3.3)": a mean and its 95% interval; the mean alone from one unit, "n/a" without."""
    if not c or c.get("mean") is None:
        return "n/a"
    if c.get("lo") is None:
        return _f(c["mean"], d, True)
    return f"{_f(c['mean'], d, True)} ({_f(c['lo'], d, True)} to {_f(c['hi'], d, True)})"


def mde_text(a: dict) -> str:
    """The detectable-difference line: "~2.1 (1 repeat), ~1.3 (3), ..." or why it is unknown."""
    m = a.get("mde")
    if not m:
        return "n/a: needs --repeats 2+" if not a.get("noise") else "n/a: needs 2+ units"
    return ", ".join(f"~{v:.1f} ({k} repeat{'s' if k != '1' else ''})" for k, v in m.items())


def _dur(s) -> str:
    if s is None:
        return "n/a"
    return f"{s / 3600:.1f} h" if s >= 5400 else f"{s / 60:.1f} min" if s >= 60 else f"{s:.0f} s"


def _summary_rows(a: dict) -> list[tuple[str, str]]:
    t = a["tokens"]
    reps = a.get("repeats") or []
    n_units = a.get("n_units", a["scored"])
    noise = a.get("noise")
    rows = [("runs scored / finished / planned", f"{a['scored']} / {a['finished']} / {a['runs']}"),
            ("units (config + seed) x repeats",
             f"{n_units} x {'/'.join(map(str, reps)) or 'n/a'}"),
            ("city score vs scripted (mean, 95% CI)", ci_text(a["city_delta"])),
            ("seed spread (std of the units)", _f(a["city_delta"].get("std"))),
            ("replay noise (std on the same seed)", _f(noise["sd"]) if noise else "n/a: needs --repeats 2+"),
            ("detectable difference vs another eval", mde_text(a)),
            ("beats scripted", f"{a['beats_scripted']} of {n_units} units"),
            ("city score change since hand-over (mean)", _f(a["city_change"], 1, True)),
            ("scripted change from the same start (mean)", _f(a["yard_change"], 1, True)),
            ("population vs scripted (mean, 95% CI)", ci_text(a["population_delta"], 0)),
            ("Teamwork (mean)", _f(a["teamwork"])),
            ("failed actions per turn", _f(a["failed_per_turn"], 2)),
            ("refusals per turn", _f(a["refused_per_turn"], 2)),
            ("model turns", f"{t['turns']:,}"),
            ("tokens per model turn", _tok(t["per_turn"])),
            ("tokens: prompt / completion", f"{_tok(t['prompt'])} / {_tok(t['completion'])}"),
            ("run time (sum / longest)", f"{_dur(a['wall_time_s']['sum'])} / {_dur(a['wall_time_s']['max'])}")]
    if a.get("usd") is not None:
        rows.append(("cost at the suite's prices", f"${a['usd']:,.2f}"))
    if a.get("bankrupt"):
        rows.append(("bankrupt runs", str(a["bankrupt"])))
    return rows


def _line(c: dict, u: dict, label: str) -> str:
    ch, ych, dl = u.get("change") or {}, u.get("yard_change") or {}, u.get("delta") or {}
    calls, tk = u.get("calls") or {}, u.get("tokens") or {}
    turns = calls.get("turns") or 0
    per = lambda k: f"{calls[k] / turns:.2f}" if turns else "n/a"
    tok = (tk.get("prompt") or 0) + (tk.get("completion") or 0)
    hand = u.get("at_handover") or {}
    return (f"| {label} | {_f(hand.get('city'))} | {_f(ch.get('city'), 1, True)} | {_f(ych.get('city'), 1, True)} "
            f"| **{_f(dl.get('city'), 1, True)}** | {_f(ch.get('population'), 0, True)} "
            f"| {_f(ych.get('population'), 0, True)} | {_f(dl.get('population'), 0, True)} | {_f(u.get('teamwork'))} "
            f"| {per('failed')} | {per('refused')} | {tk.get('turns') or 0} | {_tok(tok) if tok else 'n/a'} "
            f"| {_dur(c.get('wall_time_s')) if label.startswith('**') or not c.get('region') else ''} |")


def markdown(ev: dict) -> str:
    """scorecard.md: summary, departments, per-run lines (a region: the region, then each city), versions."""
    sc = ev.get("scorecard") or scorecard(ev)
    a, cards = sc["aggregate"], sc["runs"]
    groups = sc.get("by_config") or {}
    L = [f"# Eval {ev['name']}: {ev.get('model') or '?'} on {ev['suite']['name']}", ""]
    desc = (ev["suite"].get("description") or "").strip()
    head = (f"Suite `{ev['suite']['name']}` ({ev['suite']['file']})"
            + (f": {desc}" + ("" if desc.endswith((".", "!", "?")) else ".") if desc else ".")
            + f" Model `{ev.get('model') or '?'}`" + (f" (provider {ev['provider']})" if ev.get("provider") else "")
            + f". {a['finished']} of {a['runs']} runs finished.")
    L += [head, ""]
    if ev.get("adopted"):
        L += ["Adopted from runs played before this eval existed (`civitas eval --adopt`): nothing was rerun.", ""]
    if ev.get("missing"):
        L += [f"Suite runs not in this eval: {', '.join(ev['missing'])}.", ""]
    if ev.get("sets") or ev.get("agents"):
        L += ["Overrides: " + " ".join([*(f"--agent {x}" for x in ev.get("agents") or []),
                                        *(f"--set {x}" for x in ev.get("sets") or [])]), ""]
    L += ["Each run is a late start: scripted agents play the warm-up, then the model's fleet takes over the grown "
          "city. **vs scripted** is the run's change since the hand-over minus the change the scripted agents make "
          "from the same hand-over (the run's yardstick.json): above 0, the model added more than the scripted "
          "baseline would have. Department scores are final scores minus the yardstick's. A region run counts once, "
          "by its region values (the region score, departments and Teamwork averaged over its cities).", "",
          "A **unit** is a config and seed; its value is the mean over its repeats (runs of the same hand-over). "
          "Intervals are 95% t-intervals over the units: an interval above 0 means the model beats scripted on this "
          "suite beyond seed luck. **Replay noise** is how much one unit varies between runs of the same model; the "
          "**detectable difference** is the smallest gap between two evals on these units that a paired comparison "
          "(`civitas eval --compare`) finds 80% of the time, from the replay noise alone (a lower bound).", ""]
    cols = list(groups) if len(groups) > 1 else []
    L += ["## Summary", "", "| | all runs" + "".join(f" | {g}" for g in cols) + " |",
          "|---|---:" + "|---:" * len(cols) + "|"]
    per_group = [dict(_summary_rows(groups[g])) for g in cols]
    for k, v in _summary_rows(a):
        L.append(f"| {k} | {v}" + "".join(f" | {pg.get(k, '')}" for pg in per_group) + " |")
    roles = list(a["roles"])
    if roles:
        L += ["", "## Departments: final score vs scripted (mean, 95% CI)", "",
              "| department | all runs" + "".join(f" | {g}" for g in cols) + " |",
              "|---|---:" + "|---:" * len(cols) + "|"]
        rc = lambda x, r: ci_text((x.get("roles_ci") or {}).get(r)) if x.get("roles_ci") else \
            _f(x["roles"].get(r), 1, True)
        for r in roles:
            L.append(f"| {r} | {rc(a, r)}" + "".join(f" | {rc(groups[g], r)}" for g in cols) + " |")
    L += ["", "## Runs", "",
          "| run | city at hand-over | city change | scripted | vs scripted | population change | scripted "
          "| vs scripted | Teamwork | failed / turn | refused / turn | model turns | tokens | run time |",
          "|---|" + "---:|" * 13]
    for c in cards:
        name = run_label(c)
        if not c.get("total"):
            L.append(f"| {name} (`{c['run_id']}`) | {c.get('status')}" + " |" * 13)
            continue
        if c["region"]:
            L.append(_line(c, c["total"], f"**{name} (region)**"))
            for city in c["cities"]:
                L.append(_line(c, c["by_city"][city], f"· {city}"))
        else:
            L.append(_line(c, c["total"], name))
    if roles:
        L += ["", "## Departments per run", "", "Final score (vs scripted).", "",
              "| run | " + " | ".join(roles) + " |", "|---|" + "---:|" * len(roles)]
        for c in cards:
            units = ([(run_label(c), c["total"])] if not c["region"] else
                     [(f"{run_label(c)} · {x}", c["by_city"][x]) for x in c["cities"]]) if c.get("total") else []
            for label, u in units:
                L.append(f"| {label} | " + " | ".join(
                    f"{_f(u['roles'].get(r))} ({_f(u['role_delta'].get(r), 1, True)})" for r in roles) + " |")
    L += ["", "## Versions", "", "| run | run id | status | fleet_version | prompts | code |", "|---|---|---|---|---|---|"]
    for c in cards:
        code = c.get("code") or {}
        ctext = ("n/a" if not code else f"{code['commit']}" + (" (dirty)" if code.get("dirty") else "")
                 + (f", inferred: last commit before the run ({code.get('date', '')[:16]})" if code.get("inferred")
                    else ""))
        extra = (f"; earlier attempts {', '.join(c['attempts'])}" if c.get("attempts") else "") + \
            ("; WARNING: the yardstick's hand-over differs from the run's" if c.get("yardstick_matches") is False
             else "") + (f"; yardstick failed: {c['yardstick_error']}" if c.get("yardstick_error") else "")
        L.append(f"| {run_label(c)} | `{c['run_id']}` | {c.get('status')}{extra} | {c.get('fleet_version') or 'n/a'} "
                 f"| {c.get('prompts') or 'n/a'} | {ctext} |")
    notes = _version_notes(cards, ev)
    if notes:
        L += [""] + notes
    L += ["", f"`fleet_version` hashes the config's coaching and limits ([fleet] and the fleet agents' specs); "
              f"`prompts` hashes the system prompts the run logged (meta.json fleet_prompts), which also change with "
              f"the code (briefing rules, region rules). Rebuild this card with `civitas eval --report {ev['dir']}`."]
    return "\n".join(L) + "\n"


def _version_notes(cards: list[dict], ev: dict) -> list[str]:
    fv = {c.get("fleet_version") for c in cards if c.get("total")}
    code = {(c.get("code") or {}).get("commit") for c in cards if c.get("total")}
    by_name: dict[str, set] = {}
    for c in cards:
        if c.get("total"):
            by_name.setdefault(c["name"], set()).add(c.get("prompts"))
    out = []
    if len(code - {None}) > 1 or len(fv - {None}) > 1 or any(len(v) > 1 for v in by_name.values()):
        what = []
        if len(fv - {None}) > 1:
            what.append(f"{len(fv - {None})} fleet_versions")
        if any(len(v) > 1 for v in by_name.values()):
            what.append("different system prompts within a config")
        if len(code - {None}) > 1:
            what.append(f"{len(code - {None})} code commits"
                        + (" (inferred)" if any((c.get("code") or {}).get("inferred") for c in cards) else ""))
        out.append(f"**Note:** these runs came from slightly different code versions ({', '.join(what)}); their "
                   f"numbers compare less exactly than one eval's runs from one commit.")
    return out


def aggregate_of(ev: dict) -> dict:
    """The eval's scorecard aggregate; one saved before units and intervals is recomputed from its run cards."""
    a = (ev.get("scorecard") or {}).get("aggregate") or {}
    return _agg(_cards(ev), ev) if a and "n_units" not in a and _cards(ev) else a


def compare_rows(evs: list[dict]) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """(column heads, rows) of several evals' aggregates side by side."""
    heads = [f"{e['name']}" for e in evs]
    aggs = [aggregate_of(e) for e in evs]
    roles = list(dict.fromkeys(r for a in aggs for r in (a.get("roles") or {})))
    rows = [("model", [str(e.get("model") or "?") for e in evs]),
            ("suite", [e["suite"]["name"] for e in evs])]
    keys = None
    for a in aggs:
        if a:
            keys = [k for k, _ in _summary_rows(a)]
            break
    table = [dict(_summary_rows(a)) if a else {} for a in aggs]
    for k in keys or []:
        rows.append((k, [t.get(k, "n/a") for t in table]))
    for r in roles:
        rows.append((f"{r} vs scripted", [ci_text((a.get("roles_ci") or {}).get(r)) if a.get("roles_ci")
                                          else _f((a.get("roles") or {}).get(r), 1, True) for a in aggs]))
    return heads, rows


def _cards(ev: dict) -> list[dict]:
    return (ev.get("scorecard") or {}).get("runs") or []


def _versions(ev: dict, what: str) -> set:
    cards = [c for c in _cards(ev) if c.get("total")]
    if what == "code":
        return {(c.get("code") or {}).get("commit") for c in cards} - {None}
    return {c.get("prompts") for c in cards} - {None}


def paired(base: dict, other: dict) -> dict:
    """`other` against `base` on the units both scored (config + seed): per unit other's value vs scripted minus
    base's (the shared hand-over's difficulty cancels), a stats.verdict over those differences for the city score,
    population and each department, the units it won, the detectable difference at these repeats (the replay noise
    pooled over both evals), and what makes the two less comparable: units only one side has, a unit whose hand-over
    differs (other warm-up code), different code commits or system prompts."""
    ua, ub = unit_values(_cards(base)), unit_values(_cards(other))
    shared = [k for k in ua if k in ub]
    diff = lambda f: [None if f(ub[k]) is None or f(ua[k]) is None else f(ub[k]) - f(ua[k]) for k in shared]
    rnd = lambda v, d=3: None if v is None else {k: (x if k in ("n", "call") else _r(x, d)) for k, x in v.items()}
    roles = [r for r in dict.fromkeys(r for k in shared for r in ua[k]["roles"]) if any(r in ub[k]["roles"] for k in shared)]
    city = diff(lambda u: u["city"])
    noise = S.pooled_sd([u["values"] for u in [*ua.values(), *ub.values()]])
    ka = min((ua[k]["runs"] for k in shared), default=1)
    kb = min((ub[k]["runs"] for k in shared), default=1)
    mde = S.mde(noise["sd"], len(shared), 2 / (1 / ka + 1 / kb)) if noise and len(shared) >= 2 else None
    return {"base": base["name"], "other": other["name"], "units": shared,
            "only_base": [k for k in ua if k not in ub], "only_other": [k for k in ub if k not in ua],
            "city": rnd(S.verdict(city)), "population": rnd(S.verdict(diff(lambda u: u["population"])), 1),
            "teamwork": rnd(S.verdict(diff(lambda u: u["teamwork"])), 1),
            "roles": {r: rnd(S.verdict(diff(lambda u: u["roles"].get(r)))) for r in roles},
            "wins": sum(d > 0 for d in city if d is not None), "repeats": [ka, kb],
            "noise": None if not noise else _r(noise["sd"]), "mde": _r(mde, 2),
            "handover_differs": [k for k in shared if ua[k]["handover"] is not None and ub[k]["handover"] is not None
                                 and abs(ua[k]["handover"] - ub[k]["handover"]) > 0.05],
            "code_differs": _versions(base, "code") != _versions(other, "code"),
            "prompts_differ": _versions(base, "prompts") != _versions(other, "prompts")}


def verdict_text(p: dict) -> list[str]:
    """A paired comparison in words: the verdict, the departments, the caveats."""
    a, b, v, n = p["base"], p["other"], p["city"], len(p["units"])
    ci = lambda x, d=1: f"{_f(x['mean'], d, True)} (95% CI {_f(x['lo'], d, True)} to {_f(x['hi'], d, True)})"
    if not v or v["n"] < 2:
        L = [f"{b} vs {a}: only {n} shared unit{'s' if n != 1 else ''}, no verdict"
             + (f" ({_f(v['mean'], 1, True)} on it)" if v else "") + "."]
    elif v["call"] == "unclear":
        L = [f"{b} vs {a}: no clear difference: {ci(v)} on {n} shared units, {b} ahead on {p['wins']}."
             + (f" Differences under ~{p['mde']:.1f} can't be seen on this suite at these repeats." if p["mde"]
                else " Run both with --repeats 2 or more to learn how small a difference this suite can see.")]
    else:
        L = [f"{b} vs {a}: {b} is {v['call'].upper()}: {ci(v)} on {n} shared units, {b} ahead on {p['wins']}."]
    if v and v["n"] >= 2:
        pop = p["population"]
        if pop:
            L.append(f"- population: {ci(pop, 0)}, {pop['call']}")
        if p["teamwork"]:
            L.append(f"- Teamwork: {ci(p['teamwork'])}, {p['teamwork']['call']}")
        for r, x in p["roles"].items():
            if x and x["n"] >= 2:
                L.append(f"- {r}: {ci(x)}, {x['call']}")
    if p["handover_differs"]:
        L.append(f"- WARNING: the hand-over differs on {', '.join(p['handover_differs'])}: the warm-up changed between "
                 f"the evals, so they didn't start from the same cities and don't compare")
    if p["only_base"] or p["only_other"]:
        L.append("- left out (only one eval has them): " + ", ".join(
            [*(f"{k} ({a})" for k in p["only_base"]), *(f"{k} ({b})" for k in p["only_other"])]))
    if p["code_differs"] or p["prompts_differ"]:
        L.append("- note: different " + " and ".join(x for x, on in (("code commits", p["code_differs"]),
                                                                    ("system prompts", p["prompts_differ"])) if on)
                 + ": fine when comparing models, but the difference is not the model's alone")
    return L


def compare(evs: list[dict]) -> str:
    """Several evals' aggregates side by side, then each against the first, paired by unit."""
    heads, rows = compare_rows(evs)
    w0 = max(len(k) for k, _ in rows)
    ws = [max(len(h), *(len(v[i]) for _, v in rows)) for i, h in enumerate(heads)]
    out = [" " * w0 + "  " + "  ".join(h.rjust(w) for h, w in zip(heads, ws))]
    out.append("-" * len(out[0]))
    for k, vals in rows:
        out.append(k.ljust(w0) + "  " + "  ".join(v.rjust(w) for v, w in zip(vals, ws)))
    for ev in evs[1:]:
        out += ["", *verdict_text(paired(evs[0], ev))]
    return "\n".join(out)


def list_evals(runs_dir: str | Path) -> list[dict]:
    """The eval folders in `runs_dir` (eval_*/eval.json), newest first, each with its runs' current status and the
    aggregate of its last scorecard (the /api/evals list)."""
    out = []
    rd = Path(runs_dir)
    if not rd.is_dir():
        return out
    for p in sorted(rd.glob(f"{PREFIX}*/{FILE}"), reverse=True):
        ev = _read(p)
        if not isinstance(ev, dict) or "runs" not in ev:
            continue
        sc = ev.get("scorecard") or {}
        if sc.get("aggregate"):
            sc = {**sc, "aggregate": aggregate_of(ev)}
        runs = [{"run_id": r["run_id"], "config": r.get("config"), "name": r.get("name"), "seed": r.get("seed"),
                 "region": r.get("region"), "status": run_status(rd, r) if r.get("run_id") else r.get("status")}
                for r in ev["runs"]]
        st = ev.get("status")
        if st == "running" and not _alive(ev.get("pid")):
            st = "stopped"
        out.append({"dir": p.parent.name, "name": ev.get("name"), "model": ev.get("model"),
                    "provider": ev.get("provider"), "suite": {k: (ev.get("suite") or {}).get(k)
                                                              for k in ("name", "file", "description")},
                    "created": ev.get("created"), "status": st, "adopted": bool(ev.get("adopted")),
                    "runs": runs, "aggregate": sc.get("aggregate"), "by_config": sc.get("by_config"),
                    "units": unit_values(sc.get("runs") or [])})
    return out
