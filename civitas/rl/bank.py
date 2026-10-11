"""The RL task bank: one department's turn from a saved city (`civitas rl bank build`).

A task is (config, seed, month, role): the city the scripted fleet built by `month` of seed `seed` under the config's
game (bench/runner.py's warm-up line-up, `config.warmup_config`: no model, deterministic), the roles before `role` in
turn order having played that month, and `role` to move. A model plays that one turn through the fleet tools
(episode.py) and is scored against the scripted fleet's own turn H months on (reward.py).

`build()` plays each (config, seed) once from month 0 with the scripted fleet, in a fresh process per job (balance
overrides are process-global: bench/envs.py:_isolated_map), and at each chosen month M (an acting month), before
any turn, writes a snapshot of the env and the scripted agents. Then, as the same game goes on, it notes per role
what that role's task needs: a fingerprint of the city at its turn (`state_fp`, checked on every load so a bank made
by other code is refused), the hand-over metrics, whether the role could do anything (`actionable`: the mayor
always; a department when its funds pay for a ready site or a tier upgrade, or the scripted turn acted or asked the
mayor for money; the others stay in the index, left out by `TaskBank.tasks` unless asked for), what the scripted
turn did, and the scripted game's own scores for the H_max months after M: the baseline branch, cached so a
reward needs one replay (the model's). With `noop` (the default) it also plays the noop branch of each task (end_turn
at once) for the same months: the bank's headroom figures.

Layout under runs/_tasks/<name>/: `bank.json` (the manifest: configs as played, seeds, months, roles, horizon,
code state, composition), `index.jsonl` (one task per line), `snapshots/<config>_s<seed>_m<month>.pkl.gz`.

Seeds: the evals' seeds are never used. Seeds below 100 are reserved for evals (v1 played 0-7, v2 / the open track
play 0, 4, 5, 11, 12, 16, 18, 20, the hand-over scan looked at 0-23), and so is any seed an eval suite in configs/
names; the default training seeds are 100-299 and the held-out test seeds 300-339 (`split` on each task; seeds from
340 are "extra").

Snapshots are gzipped pickles (like bench/session.py's checkpoints, agents copied without their chat client by
`session.strip_agent`). They are not HMAC-signed: a bank is meant to be built once and copied to training machines
(Modal containers), where a per-installation key would refuse it. Instead the index holds each snapshot's SHA-256,
checked before unpickling, which catches a damaged or swapped file but not a forged index: a bank is code, so load
only banks you built or got from someone you trust (unpickling runs whatever the file says).
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import pickle
import sys
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import build_agents, game_config, load_config, scoring_config, warmup_config
from ..core.env import make_env
from ..core.roles import TURN_ORDER

ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = Path("runs") / "_tasks"
VERSION = 1
MAGIC = b"CIVITAS-TASK-SNAPSHOT 1\n"
INDEX = "index.jsonl"
MANIFEST = "bank.json"
SNAPSHOTS = "snapshots"
RESERVED_BELOW = 100                                   # seeds 0-99 belong to evals
EVAL_SEEDS = frozenset({0, 4, 5, 11, 12, 16, 18, 20, *range(8)})    # v2 / open track, and v1's 0-7
TRAIN_SEEDS = range(100, 300)
TEST_SEEDS = range(300, 340)
DEFAULT_MONTHS = (24, 48, 72, 96)
HORIZON_MAX = 12                                       # months of scores cached after a task's month
CACHE_STATES = 32                                      # prepared task states kept per process (episode.py)


class BankError(ValueError):
    """A bank, seed list or task the RL tools refuse, with what to do about it."""


# ------------------------------------------------------------------------------------------------------- seeds
def eval_suite_seeds(root: Path = ROOT) -> set[int]:
    """Every seed an eval suite in configs/ (a TOML with [[runs]]) names."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore
    from ..bench.evals import parse_seeds
    out: set[int] = set()
    for p in sorted((root / "configs").glob("*.toml")):
        try:
            data = tomllib.loads(p.read_text())
        except (OSError, ValueError):
            continue
        for run in data.get("runs") or []:
            if isinstance(run, dict) and "seeds" in run:
                try:
                    out.update(parse_seeds(run["seeds"]))
                except ValueError:
                    continue
    return out


def check_seeds(seeds) -> list[int]:
    """The seeds, refused (BankError) when any is an eval's: below RESERVED_BELOW or named by an eval suite."""
    seeds = sorted(set(int(s) for s in seeds))
    if not seeds:
        raise BankError("no seeds given")
    bad = sorted(s for s in seeds if s < RESERVED_BELOW or s in EVAL_SEEDS or s in eval_suite_seeds())
    if bad:
        raise BankError(f"seeds {bad[:12]}{' ...' if len(bad) > 12 else ''} are reserved for evals (every seed below "
                        f"{RESERVED_BELOW} and any an eval suite names); train on {TRAIN_SEEDS.start}-"
                        f"{TRAIN_SEEDS.stop - 1}, test on {TEST_SEEDS.start}-{TEST_SEEDS.stop - 1}")
    return seeds


def split_of(seed: int) -> str:
    return "train" if seed in TRAIN_SEEDS else "test" if seed in TEST_SEEDS else "extra"


def parse_months(v) -> list[int]:
    if isinstance(v, str):
        v = [int(x) for x in v.replace(" ", "").split(",") if x]
    return sorted(set(int(x) for x in v))


def parse_roles(v) -> list[str]:
    if v in (None, "", "all"):
        return list(TURN_ORDER)
    roles = [r.strip() for r in (v.split(",") if isinstance(v, str) else v) if r.strip()]
    bad = [r for r in roles if r not in TURN_ORDER]
    if bad:
        raise BankError(f"unknown role(s) {', '.join(bad)}; roles: {', '.join(TURN_ORDER)}")
    return [r for r in TURN_ORDER if r in roles]


# --------------------------------------------------------------------------------------------------- snapshots
def dump_state(env, agents: dict, level: int = 6) -> bytes:
    """The env and the agents in one gzipped pickle (agents without chat client or hub, as checkpoints keep them)."""
    from ..bench.session import strip_agent
    state = {"env": env, "agents": {r: strip_agent(a) for r, a in agents.items()}}
    return MAGIC + gzip.compress(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=level)


def load_state(data: bytes, sha256: str | None = None) -> tuple:
    """(env, agents) from `dump_state`'s bytes, after checking their SHA-256 against the index's (when given) and
    re-applying the env's balance overrides (an unpickled env never ran catalog.apply_overrides). Unpickling runs
    what the file says: only bank files whose digest the index lists get here (see the module doc)."""
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise BankError("a snapshot's SHA-256 doesn't match the bank index: the file was changed or damaged")
    if not data.startswith(MAGIC):
        raise BankError("not a civitas task snapshot (bad header)")
    state = pickle.loads(gzip.decompress(data[len(MAGIC):]))       # digest-checked bank file (module doc)
    env, agents = state["env"], state["agents"]
    _balance(env)
    return env, agents


_BALANCE: list[str] = []


def _balance(env) -> None:
    """Re-apply the env's balance overrides; refuse a second, different set in one process (overrides are global
    catalog state and can't be undone: one process per balance)."""
    from ..bench.session import apply_balance
    key = json.dumps(env.cfg.balance or {}, sort_keys=True, default=str)
    if _BALANCE and _BALANCE[0] != key:
        raise BankError("tasks with different balance overrides in one process: play them in separate processes")
    if not _BALANCE:
        _BALANCE.append(key)
    apply_balance(env)


def state_fp(env) -> str:
    """A fingerprint of the city at a task's turn: month, money, the map grids, last month's metrics, the council and
    the fleet log. Recorded when the bank is built and checked on every load, so a bank made by other sim code (or a
    snapshot that doesn't unpickle to the same city) is refused rather than scored against stale baselines."""
    from ..bench.envs import canonical
    w = env.world
    h = hashlib.sha256()
    h.update(canonical({"month": w.month, "treasury": float(w.treasury), "debt": float(w.debt),
                        "funds": {k: float(v) for k, v in sorted(w.dept_funds.items())},
                        "metrics": w.history[-1] if w.history else None,
                        "council": len(env.council.messages),
                        "log": len(env.fleet.log) if env.fleet is not None else 0}).encode())
    for name in ("struct", "level", "road_class", "density", "terrain", "zone"):
        a = getattr(w, name, None)
        if isinstance(a, np.ndarray):
            h.update(name.encode() + np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def advance(env, agents: dict, role: str) -> None:
    """At the start of an acting month: the roles before `role` in turn order play their turn (the scripted line-up),
    as the runner plays them."""
    for r in env.roles[:env.roles.index(role)]:
        if env.acts_this_month(r):
            env.apply(r, agents[r].act(env.observe(r)))


def settle(env, role: str) -> None:
    """What `runner.settle_fleet` does at a hand-over for a model agent taking over from the scripted fleet: the
    role's fleet memory starts empty (the scripted fleet writes its failures there as lessons and its requests as
    open tasks). It changes nothing the scripted policies read, so the baseline branch is the scripted game itself."""
    from ..fleet.memory import FleetState, MEMORY_FIELDS
    FleetState.of(env).memory[role] = {k: [] for k in MEMORY_FIELDS}


def _upgrade_costs(o) -> list[float]:
    if isinstance(o, dict):
        out = []
        for k, v in o.items():
            if k == "upgrades":
                items = [x for lst in v.values() for x in lst] if isinstance(v, dict) else list(v or [])
                out += [float(x["cost"]) for x in items if isinstance(x, dict) and "cost" in x]
            elif isinstance(v, dict):
                out += _upgrade_costs(v)
        return out
    return []


def affordable(env, role: str) -> list[str]:
    """What `role`'s department funds pay for now: the purposes of its ready sites (the briefing's Ready sites) and
    "upgrade" for a tier upgrade its observation offers. The mayor's money tools are always open: []."""
    from ..fleet.tools import ready_prices, role_obs, short_of, site_lists
    if role == "mayor":
        return []
    w = env.world
    obs = role_obs(env, role)
    funds = float(w.dept_funds.get(role, 0.0))
    out = sorted(p for p, pr in ready_prices(w, role, site_lists(w, role, obs)).items()
                 if not short_of(funds, pr.cost))
    if any(not short_of(funds, c) for c in _upgrade_costs(obs)):
        out.append("upgrade")
    return out


# ------------------------------------------------------------------------------------------------------- build
def _style(cfg: dict, role: str) -> str:
    from ..fleet.config import fleet_agent_config
    return fleet_agent_config(cfg, role).briefing


def _check_config(cfg: dict) -> dict:
    """The scripted line-up of `cfg` (refusing a region, simultaneous turns, or a role that isn't a fleet agent: a
    task's model plays through the fleet tools); ModelAgentError if it could call a model."""
    from ..bench.yardstick import yardstick_config
    from ..config import agent_spec
    g = game_config(cfg)
    if g.region:
        raise BankError("RL tasks are single-city: the config sets game.region")
    if g.simultaneous:
        raise BankError("RL tasks need turns in order: the config sets game.simultaneous")
    kinds = {r: agent_spec(cfg, r).get("kind", "scripted") for r in g.roles}
    bad = [r for r, k in kinds.items() if k != "fleet"]
    if bad:
        raise BankError(f"every role must be a fleet agent (the task's model plays through the fleet tools); "
                        f"{', '.join(bad)} {'is' if len(bad) == 1 else 'are'} not")
    plain = dict(cfg)
    plain["game"] = {**cfg.get("game", {}), "warmup_months": 0}
    return yardstick_config(plain)


def _series(scores_by_month: dict, month: int, hmax: int, role: str, pops: dict) -> dict:
    """The scores of the hmax months after `month` (a game that ended earlier repeats its last month)."""
    out = {"months": [], "city": [], "role": [], "population": []}
    last = None
    for h in range(1, hmax + 1):
        m = month + h
        if m in scores_by_month:
            last = m
        if last is None:
            break
        out["months"].append(m)
        out["city"].append(round(float(scores_by_month[last]["city"]), 6))
        out["role"].append(round(float(scores_by_month[last][role]), 6))
        out["population"].append(int(pops[last]))
    return out


def _build_job(job: dict) -> dict:
    """One (config, seed) in its own process: the scripted game from month 0, a snapshot at each task month, and the
    task rows (with the noop branch when asked)."""
    from ..bench.yardstick import check_agents
    t0 = time.time()
    cfg, name, seed = job["cfg"], job["config"], job["seed"]
    months, roles, hmax, bank_dir = job["months"], job["roles"], job["horizon_max"], Path(job["bank_dir"])
    wcfg = _check_config(cfg)
    game = game_config(wcfg, seed)
    env = make_env(game, scoring_config(wcfg, game))
    agents = build_agents(wcfg, env.roles, game, env.scoring)
    check_agents(agents)                                   # scripted fleet only: never a model
    for a in agents.values():
        a.reset(seed)
        a.bind(env)
    end = min(game.months, max(months) + hmax)
    scores, pops, rows, snaps = {}, {}, [], {}
    while not env.done and env.world.month < end:
        m = env.world.month
        acting = [r for r in env.roles if env.acts_this_month(r)]
        if m in months:
            blob = dump_state(env, agents)
            rel = f"{SNAPSHOTS}/{name}_s{seed}_m{m}.pkl.gz"
            (bank_dir / rel).write_bytes(blob)
            snaps[m] = (rel, hashlib.sha256(blob).hexdigest(), blob)
        for r in acting:
            row = None
            if m in months and r in roles:
                last = env.world.history[-1] if env.world.history else {}
                aff = affordable(env, r)
                row = {"config": name, "seed": seed, "month": m, "role": r, "style": _style(cfg, r),
                       "split": split_of(seed), "snapshot": snaps[m][0], "snapshot_sha256": snaps[m][1],
                       "state_fp": state_fp(env),
                       "handover": {"population": int(last.get("population", 0)),
                                    "city": round(float(env.last_scores.get("city", 0.0)), 3),
                                    "role_score": round(float(env.last_scores.get(r, 0.0)), 3),
                                    "treasury": round(float(env.world.treasury), 2),
                                    "funds": round(float(env.world.dept_funds.get(r, 0.0)), 2) if r != "mayor"
                                    else None,
                                    "happiness": round(float(last.get("happiness", 0.0)), 4)},
                       "affordable": aff}
            d = agents[r].act(env.observe(r))
            res = env.apply(r, d)
            if row is not None:
                ok = sum(1 for x in res if x.get("ok"))
                row["scripted"] = {"actions": ok, "failed": len(res) - ok, "messages": len(d.messages),
                                   "calls": len((d.meta or {}).get("tool_trace") or [])}
                asks = sum(1 for x in d.messages if x.get("to") == "mayor")
                row["actionable"] = bool(r == "mayor" or ok or aff or asks)
                row["why"] = ("the mayor's money tools" if r == "mayor" else
                              f"the scripted turn acted ({ok})" if ok else
                              f"funds pay for {', '.join(aff)}" if aff else
                              "the scripted turn asked the mayor for money" if asks else
                              "no affordable site or upgrade, and the scripted turn neither acted nor asked")
                rows.append(row)
        info = env.end_month()
        scores[env.world.month] = info["scores"]
        pops[env.world.month] = info["metrics"]["population"]
    for row in rows:
        h = min(hmax, game.months - row["month"])
        row["horizon_max"] = h
        row["baseline"] = _series(scores, row["month"], h, row["role"], pops)
    if job.get("noop"):
        from .episode import TaskEpisode
        from .policies import play
        for row in rows:
            ep = TaskEpisode.from_blob(snaps[row["month"]][2], row, cfg)
            ep.reset()
            play(ep, "noop")
            row["noop"] = {k: v for k, v in ep.finish(horizon=row["horizon_max"], series_only=True).items()
                           if k in ("city", "role", "population")}
    return {"rows": rows, "seconds": round(time.time() - t0, 2), "config": name, "seed": seed}


def _task_id(row: dict) -> str:
    return f"{row['config']}/s{row['seed']}/m{row['month']}/{row['role']}"


def build(name: str, configs: list, seeds, months=DEFAULT_MONTHS, roles="all", *, horizon_max: int = HORIZON_MAX,
          noop: bool = True, workers: int = 4, root: Path | str = TASKS_DIR, force: bool = False,
          verbose: bool = True) -> dict:
    """Build the bank `name` (see the module doc) from `configs` (paths, or config dicts as load_config returns
    them); returns its manifest."""
    from ..bench.envs import _isolated_map, code_info
    if not name or "/" in name or name.startswith("."):
        raise BankError(f"bad bank name {name!r}")
    seeds, months, roles = check_seeds(seeds), parse_months(months), parse_roles(roles)
    d = Path(root) / name
    if (d / MANIFEST).exists() and not force:
        raise BankError(f"bank {name!r} exists at {d}: pass --force to rebuild it")
    cfgs = {}
    for path in configs:
        cfg = copy.deepcopy(path) if isinstance(path, dict) else load_config(path)
        path = cfg.get("name") if isinstance(path, dict) else path
        cname = cfg["name"]
        if cname in cfgs:
            raise BankError(f"two configs named {cname!r}")
        _check_config(cfg)
        g = game_config(cfg)
        bad = [m for m in months if m % max(1, g.act_every) or m >= g.months or m < 0]
        if bad:
            raise BankError(f"{cname}: months {bad} aren't acting months of the game (multiples of act_every "
                            f"{g.act_every} below {g.months})")
        cfgs[cname] = {"path": str(path), "cfg": cfg}
    (d / SNAPSHOTS).mkdir(parents=True, exist_ok=True)
    jobs = [{"cfg": c["cfg"], "config": cname, "seed": s, "months": months, "roles": roles,
             "horizon_max": int(horizon_max), "bank_dir": str(d), "noop": noop}
            for cname, c in cfgs.items() for s in seeds]
    t0 = time.time()
    if verbose:
        print(f"bank {name}: {len(jobs)} games ({', '.join(cfgs)} x {len(seeds)} seeds), months {months}, "
              f"{len(roles)} roles, {workers} workers ...", flush=True)
    results = _isolated_map(_build_job, jobs, workers) if workers > 1 else [_build_job(j) for j in jobs]
    order = {r: i for i, r in enumerate(TURN_ORDER)}
    rows = sorted((r for res in results for r in res["rows"]),
                  key=lambda r: (r["config"], r["seed"], r["month"], order[r["role"]]))
    for r in rows:
        r["task_id"] = _task_id(r)
    with open(d / INDEX, "w") as f:
        for r in rows:
            f.write(json.dumps({"task_id": r.pop("task_id"), **r}, separators=(",", ":")) + "\n")
    manifest = {"name": name, "version": VERSION, "created": time.time(), "seeds": seeds, "months": months,
                "roles": roles, "horizon_max": int(horizon_max), "noop": noop,
                "configs": {k: {"path": v["path"], "cfg": v["cfg"]} for k, v in cfgs.items()},
                "code": code_info(ROOT), "build_seconds": round(time.time() - t0, 1),
                "game_seconds": {f"{r['config']}:{r['seed']}": r["seconds"] for r in results}}
    manifest["composition"] = composition(load_rows(d / INDEX))
    (d / MANIFEST).write_text(json.dumps(manifest, indent=2, default=str))
    if verbose:
        print(composition_text(manifest["composition"]))
        print(f"bank {name}: {len(rows)} tasks in {time.time() - t0:.0f} s -> {d}")
    return manifest


# --------------------------------------------------------------------------------------------------- composition
def composition(rows: list[dict]) -> dict:
    """Counts by split, style, month, role and actionable; the noop branch's headroom (baseline - noop city score at
    6 months) where computed."""
    def count(key):
        return dict(sorted(Counter(key(r) for r in rows).items(), key=lambda kv: str(kv[0])))
    out = {"tasks": len(rows), "actionable": sum(1 for r in rows if r.get("actionable")),
           "by_split": count(lambda r: r["split"]), "by_style": count(lambda r: r["style"]),
           "by_month": count(lambda r: r["month"]), "by_role": count(lambda r: r["role"]),
           "actionable_by_role": {role: sum(1 for r in rows if r["role"] == role and r.get("actionable"))
                                  for role in TURN_ORDER if any(r["role"] == role for r in rows)},
           "scripted_idle": sum(1 for r in rows if not (r.get("scripted") or {}).get("actions"))}
    heads = [_headroom(r, 6) for r in rows if r.get("noop")]
    heads = [h for h in heads if h is not None]
    if heads:
        out["noop_headroom_6"] = {"n": len(heads), "mean": round(float(np.mean(heads)), 4),
                                  "zero": sum(1 for h in heads if abs(h) < 1e-9),
                                  "positive": sum(1 for h in heads if h > 1e-9),
                                  "negative": sum(1 for h in heads if h < -1e-9)}
    return out


def _headroom(row: dict, h: int) -> float | None:
    b, n = row.get("baseline") or {}, row.get("noop") or {}
    if not b.get("city") or not n.get("city"):
        return None
    i = min(h, len(b["city"]), len(n["city"])) - 1
    return b["city"][i] - n["city"][i]


def composition_text(c: dict) -> str:
    fmt = lambda d: ", ".join(f"{k} {v}" for k, v in d.items())
    L = [f"tasks {c['tasks']} ({c['actionable']} actionable; scripted turn idle in {c['scripted_idle']})",
         f"  split: {fmt(c['by_split'])} | style: {fmt(c['by_style'])}",
         f"  month: {fmt(c['by_month'])}",
         f"  role: {fmt(c['by_role'])}",
         f"  actionable by role: {fmt(c['actionable_by_role'])}"]
    h = c.get("noop_headroom_6")
    if h:
        L.append(f"  noop headroom at 6 months (scripted - noop city score): mean {h['mean']:+.3f} over {h['n']}; "
                 f"{h['positive']} scripted better, {h['zero']} equal, {h['negative']} noop better")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------------- banks
def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


@dataclass
class Task:
    """One row of a bank's index (see build) and the bank it belongs to."""
    row: dict
    bank: "TaskBank" = field(repr=False)

    def __getattr__(self, k):
        try:
            return self.__dict__["row"][k]
        except KeyError:
            raise AttributeError(k) from None

    @property
    def cfg(self) -> dict:
        return self.bank.config(self.row["config"])

    def blob(self) -> bytes:
        return self.bank.snapshot(self.row)


class TaskBank:
    """A built bank (`build`), by name under `root` or by its folder."""

    def __init__(self, name_or_path: str | Path, root: Path | str = TASKS_DIR):
        p = Path(name_or_path)
        d = p if (p / MANIFEST).exists() else Path(root) / str(name_or_path)
        if not (d / MANIFEST).exists():
            raise BankError(f"no task bank at {d} (build one with `civitas rl bank build --name ...`)")
        self.dir = d
        self.manifest = json.loads((d / MANIFEST).read_text())
        if self.manifest.get("version") != VERSION:
            raise BankError(f"bank {d.name}: version {self.manifest.get('version')}, this code reads {VERSION}")
        self.name = self.manifest["name"]
        self.rows = load_rows(d / INDEX)
        self._by_id = {r["task_id"]: r for r in self.rows}
        self._snaps: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()

    def config(self, name: str) -> dict:
        return copy.deepcopy(self.manifest["configs"][name]["cfg"])

    def snapshot(self, row: dict) -> bytes:
        rel = row["snapshot"]
        with self._lock:
            data = self._snaps.get(rel)
            if data is not None:
                self._snaps.move_to_end(rel)
                return data
        data = (self.dir / rel).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["snapshot_sha256"]:
            raise BankError(f"{self.dir / rel}: SHA-256 doesn't match the index (changed or damaged)")
        with self._lock:
            self._snaps[rel] = data
            while len(self._snaps) > 8:
                self._snaps.popitem(last=False)
        return data

    def get(self, task_id: str) -> Task:
        if task_id not in self._by_id:
            raise BankError(f"bank {self.name} has no task {task_id!r}")
        return Task(self._by_id[task_id], self)

    def tasks(self, split: str | None = None, *, actionable: bool | None = True, roles=None, styles=None,
              months=None, limit: int | None = None, sample_seed: int | None = None) -> list[Task]:
        """Tasks filtered by split, actionable (None: both), roles, styles and months, in index order; `limit` keeps
        that many, spread evenly over the list (or a seeded random sample with `sample_seed`)."""
        rows = [r for r in self.rows
                if (split in (None, "all") or r["split"] == split)
                and (actionable is None or bool(r.get("actionable")) == actionable)
                and (not roles or r["role"] in roles) and (not styles or r["style"] in styles)
                and (not months or r["month"] in months)]
        if limit is not None and limit < len(rows):
            if sample_seed is not None:
                rng = np.random.default_rng(sample_seed)
                idx = sorted(rng.choice(len(rows), size=limit, replace=False).tolist())
            else:
                idx = sorted({int(i * len(rows) / limit) for i in range(limit)})
            rows = [rows[i] for i in idx]
        return [Task(r, self) for r in rows]

    def info(self) -> str:
        m = self.manifest
        cfgs = ", ".join(f"{k} ({_style_of(v['cfg'])})" for k, v in m["configs"].items())
        return (f"bank {self.name} at {self.dir}: configs {cfgs}; seeds {m['seeds'][0]}-{m['seeds'][-1]} "
                f"({len(m['seeds'])}); months {m['months']}; horizon up to {m['horizon_max']}; built at "
                f"{(m.get('code') or {}).get('commit', '?')[:10]}{' (dirty)' if (m.get('code') or {}).get('dirty') else ''}\n"
                + composition_text(composition(self.rows)))


def _style_of(cfg: dict) -> str:
    return (cfg.get("fleet") or {}).get("briefing", "guided")


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)
