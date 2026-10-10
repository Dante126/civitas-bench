"""Environment versions: a frozen, named fingerprint of what a model is evaluated on (`configs/envs/<name>.json`, a
git tag `env/<name>`), so evals of different code are never ranked together.

The fingerprint is behavioural. `probe(suite)` replays every seed of the suite with the yardstick's line-up (the
scripted fleet in a single city, plain scripted agents in a region: bench/yardstick.py:play, with its safety checks,
no model) and hashes, per part:
- `config`: the suite's entries (config, seeds) and each entry's effective config with the model keys removed
  (`[fleet]` / `[agents.<role>]` model, provider, temperature, max_tokens, extra_body, and `[providers]`);
- `prompts` / `tools`: each fleet role's system prompt and tool schemas, as a model-backed fleet agent gets them;
- per seed (`"<config>:<seed>"`): `briefings` and `results`, the briefing text and tool results (the observation
  text and action results for plain agents) of every turn from the hand-over on, and `metrics`, every month's
  metrics (floats rounded to 6 places) and the scores after the hand-over.
A refactor or a comment leaves it alone; a sim, briefing, tool or config change moves the part it touches. Seeds run
in a process pool, one process per seed (balance overrides are global state); a probe is cached per code state
(commit, a hash of `git diff HEAD` and of untracked source files, the suite and its configs) in
runs/_cache/probe_<key>.json.

`freeze(name, suite)` writes the probe as configs/envs/<name>.json and tags HEAD `env/<name>` when the tree is
clean; `check(suite)` probes and compares with the suite's `[eval].env`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENVS_DIR = ROOT / "configs/envs"
CACHE_DIR = Path("runs") / "_cache"
VERSION = 1                          # of the probe: bump when what it hashes changes
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
MODEL_CFG_KEYS = ("model", "provider", "temperature", "max_tokens", "extra_body")
SEED_PARTS = ("briefings", "results", "metrics")
PARTS = ("config", "prompts", "tools", "seeds", *SEED_PARTS)
DIGITS = 6


class EnvError(ValueError):
    """A version name, file or freeze the environment registry refuses."""


# ---------------------------------------------------------------------------------------------------- hashing
def _canon(o):
    """JSON-ready copy: numpy scalars to Python, floats rounded to DIGITS places, tuples and sets to lists."""
    if hasattr(o, "item") and not isinstance(o, (dict, list, tuple, str)):
        try:
            o = o.item()
        except (TypeError, ValueError):
            o = o.tolist()
    if isinstance(o, bool) or o is None or isinstance(o, (int, str)):
        return o
    if isinstance(o, float):
        r = round(o, DIGITS)
        return 0.0 if r == 0 else r
    if isinstance(o, dict):
        return {str(k): _canon(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_canon(x) for x in o]
    if isinstance(o, (set, frozenset)):
        return sorted(_canon(x) for x in o)
    if hasattr(o, "tolist"):
        return _canon(o.tolist())
    return str(o)


def canonical(o) -> str:
    return json.dumps(_canon(o), sort_keys=True, separators=(",", ":"))


def _h(o) -> str:
    data = o if isinstance(o, str) else canonical(o)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------- config
def _suite(suite):
    from .evals import Suite, load_suite
    return suite if isinstance(suite, Suite) else load_suite(suite)


def strip_model(cfg: dict) -> dict:
    """`cfg` without the model keys: [providers], and model / provider / temperature / max_tokens / extra_body in
    [fleet] and in each [agents.<role>] (a role pinned to the scripted fleet keeps its model: that is the line-up)."""
    out = copy.deepcopy(cfg)
    out.pop("providers", None)
    fleet = out.get("fleet")
    if isinstance(fleet, dict):
        for k in MODEL_CFG_KEYS:
            fleet.pop(k, None)
    for spec in (out.get("agents") or {}).values():
        if isinstance(spec, dict) and spec.get("model") != "scripted":
            for k in MODEL_CFG_KEYS:
                spec.pop(k, None)
    return out


def entry_config(entry, agents=(), sets=()) -> dict:
    from ..config import apply_cli_overrides, load_config
    return strip_model(apply_cli_overrides(load_config(entry.path), list(agents), list(sets)))


# ---------------------------------------------------------------------------------------------------- one seed
def _fleet_prompts(cfg: dict, seed: int) -> tuple[dict, dict]:
    """(role -> system prompt, role -> tool schemas) of the fleet roles of `cfg`, as a model-backed fleet agent gets
    them when bound at month 0 (fleet/agent.py:ToolAgent.bind and act). The agents are built and bound only to read
    those: never asked to act, so no model is called."""
    from ..agents.llm_client import ProviderConfig
    from ..config import agent_spec, game_config, scoring_config
    from ..core.env import make_env
    from ..fleet.agent import ToolAgent
    from ..fleet.config import fleet_agent_config
    from ..fleet.tools import ToolBox
    game = game_config(cfg, seed)
    env = make_env(game, scoring_config(cfg, game))
    prompts, tools = {}, {}
    for role in env.roles:
        if agent_spec(cfg, role).get("kind") != "fleet":
            continue
        fc = fleet_agent_config(cfg, role)
        if fc.model == "scripted":
            continue
        base, city = role.split(".", 1)[-1], (role.split(".", 1)[0] if "." in role else None)
        a = ToolAgent(base, cfg=fc, provider=ProviderConfig(api_key_env=""), city=city)
        a.bind(env)
        prompts[role] = a.system
        tools[role] = ToolBox(a.env, base, branch_dezone=True, max_steps=fc.max_steps, next_line=True).schemas()
    return prompts, tools


def probe_seed(cfg: dict, seed: int) -> dict:
    """One seed of the probe (in its own process from `probe`): the yardstick game of `cfg` (model keys already
    removed) with hashes of what the agents see after the hand-over and of every month's metrics."""
    from ..config import warmup_months
    from .yardstick import play
    t0 = time.time()
    n = warmup_months(cfg)
    prompts, tools = _fleet_prompts(cfg, seed)
    hb, hr = hashlib.sha256(), hashlib.sha256()
    turns = [0]

    def hook(env, role, obs, decision, results):
        month = env.world.month
        if month < n:
            return
        turns[0] += 1
        head = f"{month}|{role}|"
        msgs = (getattr(decision, "meta", None) or {}).get("messages")
        if msgs:                                       # a fleet turn: its briefing, then each call and its result
            brief = next((m.get("content") or "" for m in msgs if m.get("role") == "user"), "")
            rest = [m for m in msgs if m.get("role") in ("assistant", "tool")]
            hb.update((head + brief + "\n").encode())
            hr.update((head + canonical(rest) + "\n").encode())
        else:                                          # a plain agent: the observation text and the results
            hb.update((head + str((obs or {}).get("text") or "") + "\n").encode())
            hr.update((head + canonical(results) + "\n").encode())

    ys = play(cfg, seed, hook=hook, keep_env=True)
    env = ys.pop("env")
    envs = getattr(env, "envs", None) or {"": env}
    metrics = {c: [_canon(m) for m in e.world.history] for c, e in envs.items()}
    scores = {k: ys.get(k) for k in ("final_scores", "series", "by_city", "at_handover", "at_end", "bankrupt")}
    return {"briefings": hb.hexdigest()[:16], "results": hr.hexdigest()[:16],
            "metrics": _h({"history": metrics, "scores": scores}), "prompts": _h(prompts), "tools": _h(tools),
            "turns": turns[0], "months": len(next(iter(envs.values())).world.history),
            "seconds": round(time.time() - t0, 2)}


def _task(args):
    return probe_seed(*args)


# ---------------------------------------------------------------------------------------------------- code state
def _git(*args, root: Path = ROOT, text: bool = True):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=text, timeout=30)


def code_state(root: Path = ROOT) -> dict:
    """{commit, dirty, diff_hash} of the checkout: `git diff HEAD` of tracked files plus the untracked source and
    config files (a new module counts); commit None outside git."""
    try:
        head = _git("rev-parse", "--short=12", "HEAD", root=root)
        if head.returncode:
            return {"commit": None, "dirty": None, "diff_hash": None}
        diff = _git("diff", "HEAD", "--binary", root=root, text=False).stdout
        others = _git("ls-files", "--others", "--exclude-standard", "--", "civitas", "configs", "pyproject.toml",
                      root=root).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "diff_hash": None}
    h = hashlib.sha256(diff)
    for f in sorted(others):
        try:
            h.update(f.encode() + b"\0" + (root / f).read_bytes())
        except OSError:
            continue
    return {"commit": head.stdout.strip(), "dirty": bool(diff.strip()), "diff_hash": h.hexdigest()[:16]}


def _cache_key(suite, code: dict, agents, sets) -> str | None:
    if not code.get("commit"):
        return None
    files = {}
    for p in [Path(suite.file), *(e.path for e in suite.entries)]:
        try:
            files[str(p)] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        except OSError:
            files[str(p)] = None
    return _h({"v": VERSION, "code": code, "files": files, "agents": list(agents), "sets": list(sets),
               "py": sys.version_info[:2]})


# ---------------------------------------------------------------------------------------------------- probe
def probe(suite, *, workers: int = 8, cache: bool = True, agents=(), sets=(), cache_dir=None) -> dict:
    """The suite's fingerprint (see the module doc): {"fingerprint", "parts": {"config", "prompts", "tools",
    "seeds": {"<config>:<seed>": {"briefings", "results", "metrics"}}}, "code", "seconds", "cached", "suite",
    "seeds": per-seed details}. `agents` / `sets` are the eval's own overrides (model keys drop out). `workers` = 0
    plays the seeds in this process (tests only: a balance override would stay in it)."""
    s = _suite(suite)
    t0 = time.time()
    code = code_state()
    key = _cache_key(s, code, agents, sets) if cache else None
    cdir = Path(cache_dir) if cache_dir else CACHE_DIR
    cpath = cdir / f"probe_{key}.json" if key else None
    if cpath and cpath.exists():
        try:
            out = json.loads(cpath.read_text())
            if out.get("fingerprint"):
                return {**out, "cached": True}
        except (OSError, ValueError):
            pass
    cfgs = {e.config: entry_config(e, agents, sets) for e in s.entries}
    tasks = [(f"{e.config}:{seed}", cfgs[e.config], seed) for e in s.entries for seed in e.seeds]
    if workers and workers > 0 and len(tasks) > 0:
        kw = {"max_tasks_per_child": 1} if sys.version_info >= (3, 11) else {}
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=mp.get_context("spawn"),
                                 **kw) as ex:
            got = list(ex.map(_task, [(c, seed) for _, c, seed in tasks]))
    else:
        got = [probe_seed(c, seed) for _, c, seed in tasks]
    seeds = {k: r for (k, _, _), r in zip(tasks, got)}
    parts = {"config": _h({"entries": [{"config": e.config, "seeds": e.seeds} for e in s.entries],
                           "configs": cfgs}),
             "prompts": _h({k: r["prompts"] for k, r in seeds.items()}),
             "tools": _h({k: r["tools"] for k, r in seeds.items()}),
             "seeds": {k: {p: r[p] for p in SEED_PARTS} for k, r in seeds.items()}}
    out = {"version": VERSION, "fingerprint": _h(parts), "parts": parts, "code": code,
           "seconds": round(time.time() - t0, 2), "cached": False, "suite": s.file, "suite_name": s.name,
           "agents": list(agents), "sets": list(sets), "created": time.time(),
           "seeds": {k: {x: r[x] for x in ("turns", "months", "seconds")} for k, r in seeds.items()}}
    if cpath:
        try:
            cdir.mkdir(parents=True, exist_ok=True)
            tmp = cpath.with_suffix(".tmp")
            tmp.write_text(json.dumps(out, indent=1))
            os.replace(tmp, cpath)
        except OSError:
            pass
    return out


# ---------------------------------------------------------------------------------------------------- versions
def _check_name(name: str) -> None:
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise EnvError(f"bad environment name {name!r}: lower-case letters, digits, '.', '_' and '-', starting with "
                       f"a letter or digit (e.g. civitas-v2)")


def load_env(name: str) -> dict | None:
    """The frozen version `name` (configs/envs/<name>.json), None when there is none."""
    if not isinstance(name, str) or not NAME_RE.match(name):
        return None
    try:
        return json.loads((ENVS_DIR / f"{name}.json").read_text())
    except (OSError, ValueError):
        return None


def list_envs() -> list[dict]:
    """[{"name", "fingerprint", "created", "commit", "suite"}], newest first."""
    out = []
    for p in ENVS_DIR.glob("*.json") if ENVS_DIR.is_dir() else []:
        try:
            e = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        out.append({"name": e.get("name") or p.stem, "fingerprint": e.get("fingerprint"),
                    "created": e.get("created"), "commit": e.get("commit"), "suite": e.get("suite"),
                    "tag": e.get("tag"), "untagged": e.get("untagged")})
    return sorted(out, key=lambda e: e.get("created") or 0, reverse=True)


def _tag(name: str, probed: dict) -> dict:
    """{"tag": "env/<name>"} once tagged, else {"untagged": why}. Never raises."""
    from .evals import code_version
    tag = f"env/{name}"
    try:
        cv = code_version()
        if cv is None:
            return {"untagged": "not a git checkout"}
        if cv["dirty"]:
            return {"untagged": "the tree has uncommitted changes to tracked files: commit, then tag HEAD "
                                f"`git tag {tag}` by hand"}
        if probed.get("code", {}).get("commit") and probed["code"]["commit"] != cv["commit"]:
            return {"untagged": f"the probe ran on {probed['code']['commit']}, HEAD is {cv['commit']}"}
        if probed.get("code", {}).get("dirty"):
            return {"untagged": "the probe ran on uncommitted changes"}
        if _git("rev-parse", "-q", "--verify", f"refs/tags/{tag}").returncode == 0:
            return {"untagged": f"tag {tag} exists already"}
        r = _git("tag", tag)
        if r.returncode:
            return {"untagged": f"git tag failed: {(r.stderr or r.stdout).strip()[:200]}"}
        return {"tag": tag}
    except (OSError, subprocess.SubprocessError) as e:
        return {"untagged": f"git: {e}"}


def freeze(name: str, suite, *, tag: bool = True, probe_result=None) -> dict:
    """Freeze the suite's probe as version `name`: configs/envs/<name>.json (an existing name is refused) and, with
    `tag` on a clean tree, the git tag env/<name> on HEAD (else the record says why it is untagged)."""
    _check_name(name)
    path = ENVS_DIR / f"{name}.json"
    if path.exists():
        raise EnvError(f"environment {name} exists already ({path}): freeze a new name, e.g. a higher version")
    s = _suite(suite)
    pr = probe_result or probe(s)
    rec = {"name": name, "fingerprint": pr["fingerprint"], "created": time.time(),
           "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": (pr.get("code") or {}).get("commit"),
           "dirty": (pr.get("code") or {}).get("dirty"), "suite": s.file, "suite_name": s.name,
           "probe_version": pr.get("version", VERSION), "probe_seconds": pr.get("seconds"),
           "agents": pr.get("agents") or [], "sets": pr.get("sets") or [], "parts": pr["parts"]}
    rec.update(_tag(name, pr) if tag else {"untagged": "not asked (--no-tag)"})
    ENVS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=2) + "\n")
    os.replace(tmp, path)
    return rec


def diff(frozen: dict, probed: dict) -> list[str]:
    """The parts that differ (in PARTS order), e.g. ["briefings", "metrics"]; "seeds" when the seed lists differ."""
    fp, pp = (frozen or {}).get("parts") or {}, (probed or {}).get("parts") or {}
    if frozen and probed and frozen.get("fingerprint") == probed.get("fingerprint"):
        return []
    out = [k for k in ("config", "prompts", "tools") if fp.get(k) != pp.get(k)]
    fs, ps = fp.get("seeds") or {}, pp.get("seeds") or {}
    if set(fs) != set(ps):
        out.append("seeds")
    shared = [k for k in fs if k in ps]
    out += [p for p in SEED_PARTS if any((fs[k] or {}).get(p) != (ps[k] or {}).get(p) for k in shared)]
    return out


def check(suite, *, probe_result=None, agents=(), sets=(), workers: int = 8, cache: bool = True,
          cache_dir=None) -> dict:
    """The suite's code against its version (`[eval].env`): {"env": name | None, "frozen": bool, "match": bool | None
    (None without a frozen version), "diff": parts, "fingerprint", "frozen_fingerprint", "probe"}."""
    s = _suite(suite)
    pr = probe_result or probe(s, workers=workers, cache=cache, agents=agents, sets=sets, cache_dir=cache_dir)
    frozen = load_env(s.env) if s.env else None
    return {"env": s.env, "frozen": frozen is not None,
            "match": None if frozen is None else frozen.get("fingerprint") == pr["fingerprint"],
            "diff": diff(frozen, pr) if frozen else [], "fingerprint": pr["fingerprint"],
            "frozen_fingerprint": (frozen or {}).get("fingerprint"), "probe": pr}


def check_text(c: dict) -> str:
    """One line: "environment civitas-v1: matches (fingerprint ...)" or what changed."""
    fp = c.get("fingerprint")
    if not c.get("env"):
        return f"environment: the suite names no version ([eval].env); fingerprint {fp}"
    if not c.get("frozen"):
        return f"environment {c['env']}: not frozen yet (civitas env freeze {c['env']}); fingerprint {fp}"
    if c.get("match"):
        return f"environment {c['env']}: matches (fingerprint {fp})"
    return (f"environment {c['env']}: CHANGED ({', '.join(c.get('diff') or ['?'])}): fingerprint {fp}, frozen "
            f"{c.get('frozen_fingerprint')}")
