"""Scripted yardsticks for warm-up runs: the same game from the same hand-over month, played by scripted agents.

A warm-up run (`[game].warmup_months = N`, bench/runner.py) hands a grown city to the configured agents at month N.
What their months added (`summary["warmup"]["change"]`) means little alone: a city grows by itself. The yardstick is
the scripted agents' answer from the same start: the run's config with every costly agent made scripted for the
whole game (`config.warmup_config`, the warm-up's own line-up: the scripted fleet in a single city, plain scripted
agents in a region), the run's seed, `[game]` and `[scoring]`. Months 0..N-1 are then the run's warm-up exactly
(the city at hand-over matches the run's, which `yardstick()` checks as `matches_run`), and only the agents after N
differ.

Safety: a yardstick never calls a model. `yardstick_config` refuses a line-up with an `llm` / `remote` agent or a
fleet agent whose model isn't "scripted", and `check_agents` checks the built agents' exact classes (and that none
holds a model client) before a single month is played (`ModelAgentError`).

`play(cfg, seed)` plays it with no logs (like `runner.warm_up`); `yardstick(run_dir)` reads a run's meta.json,
plays, compares with its summary and writes `yardstick.json`. The record has `summary["warmup"]`'s shape (the city
score and population at hand-over and at the end, the change, the average city score after the hand-over), the
departments' final scores and per frame month from N on the city score and population (a region: per city too).

The do-nothing reference (`noop`): the same window from the same hand-over with the run's model roles idle after it
(`idle_roles`: the roles the warm-up line-up stands in for, i.e. every fleet / llm / remote agent; all roles when there
is none), the other roles as in the yardstick line-up. An idle role is a `NoopAgent`: each acting turn no action and no
message, which is what the RL noop policy (rl/policies.py: end_turn at once) amounts to in the game. A fleet line-up
is idle as a whole: no budget top-ups, no roads, no zoning, no buildings, the treasury and taxes left as handed over.
`play(..., idle=roles)` plays it; its record has the scripted part's shape and goes into yardstick.json under `noop`
(`noop_part`), with its own `matches_run` and `matches_yardstick` (the hand-over must be the run's and the scripted
part's; `noop_ok` is the part only when neither is False). `yardstick(run_dir, scripted=False)` adds the noop part to
an existing yardstick.json (`civitas yardstick <run> --noop`), `backfill()` does so for many runs in worker processes
(`civitas yardstick --noop --evals`).
"""
from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..agents.base import NoopAgent, RandomAgent
from ..agents.scripted import SCRIPTED
from ..config import agent_spec, build_agents, game_config, scoring_config, warmup_config, warmup_months
from ..core.env import make_env
from ..fleet.config import fleet_agent_config
from ..fleet.scripted import MODEL as SCRIPTED_FLEET, ScriptedFleetAgent

FILE = "yardstick.json"
VERSION = 1
NOOP_VERSION = 1
NOOP_KEYS = ("handed_over", "bankrupt", "idle", "agents", "at_handover", "at_end", "change", "avg_city_score",
             "final_scores", "series", "by_city", "wall_time_s")
MODEL_KINDS = ("llm", "remote")                 # kinds that call a model or a live process


class ModelAgentError(RuntimeError):
    """A yardstick line-up holds an agent that could call a model: refused before playing."""


def safe_types() -> tuple[type, ...]:
    """The agent classes a yardstick may play with: the scripted baselines, the scripted fleet, noop and random."""
    return (ScriptedFleetAgent, NoopAgent, RandomAgent, *SCRIPTED.values())


def _roles(cfg: dict) -> list[str]:
    g = game_config(cfg)
    roles = list(g.roles)
    return [f"{c}.{r}" for r in roles for c in g.cities] if g.region else roles


def model_roles(cfg: dict) -> list[str]:
    """The roles `cfg` gives an agent that calls a model: llm, remote, or a fleet agent not on the scripted model."""
    bad = []
    for role in _roles(cfg):
        spec = agent_spec(cfg, role)
        kind = spec.get("kind", "scripted")
        if kind in MODEL_KINDS or (kind == "fleet" and fleet_agent_config(cfg, role).model != SCRIPTED_FLEET):
            bad.append(f"{role}={kind}")
    return bad


def yardstick_config(cfg: dict) -> dict:
    """`cfg` played by the warm-up line-up for the whole game (`config.warmup_config`: no `warmup_months`); refuses
    (ModelAgentError) a line-up that would still hold a model-backed agent."""
    out = warmup_config(cfg)
    bad = model_roles(out)
    if bad:
        raise ModelAgentError(f"the yardstick line-up would call a model ({', '.join(bad)}); refusing to play it")
    return out


def idle_roles(cfg: dict) -> list[str]:
    """The roles a noop reference leaves idle after the hand-over: those the warm-up line-up plays in place of the
    configured agent (fleet on a model, llm, remote: `warmup_config`'s changes, as runner.lineups finds them); every
    role when there are none (an all-scripted run: the noop floor of the whole line-up)."""
    w = warmup_config(cfg)
    roles = _roles(cfg)
    return [r for r in roles if agent_spec(w, r) != agent_spec(cfg, r)] or roles


def check_agents(agents: dict) -> None:
    """Raise ModelAgentError unless every built agent is exactly one of `safe_types()` and holds no model client
    (a scripted fleet agent on the scripted model). Called before the yardstick plays a month."""
    safe = safe_types()
    bad = []
    for role, a in agents.items():
        ok = type(a) in safe and getattr(a, "client", None) is None and a.kind not in MODEL_KINDS
        if ok and a.kind == "fleet":
            ok = isinstance(a, ScriptedFleetAgent) and getattr(a.cfg, "model", None) == SCRIPTED_FLEET
        if not ok:
            bad.append(f"{role}: {type(a).__name__} ({a.describe()})")
    if bad:
        raise ModelAgentError("the yardstick line-up holds agents that could call a model: " + "; ".join(bad))


def _pop(env, city: str | None = None) -> int:
    envs = getattr(env, "envs", None) or {"": env}
    if city is not None:
        envs = {city: envs[city]}
    return sum(e.world.history[-1]["population"] if e.world.history else 0 for e in envs.values())


def _mark(env, city: str | None, scores: dict) -> dict:
    key = f"{city}.city" if city else "city"
    return {"city": scores.get(key), "population": _pop(env, city)}


def play(cfg: dict, seed: int, *, verbose: bool = False, hook=None, keep_env: bool = False,
         on_handover=None, idle: list[str] | None = None) -> dict:
    """Play `cfg` (a warm-up config) from month 0 with the yardstick line-up, no logs. Returns the yardstick record
    (see the module doc). ValueError without a warm-up; ModelAgentError for a line-up that could call a model.
    `hook(env, role, obs, decision, results)` runs after each turn is applied, only reading (bench/envs.py's probe);
    `on_handover(env, agents)` runs once at the hand-over, before that month's turns, only reading (the probe's model
    part plays on a deep copy); `keep_env` puts the finished env on the record as `env` (not JSON). `idle` (roles,
    e.g. `idle_roles(cfg)`): the noop reference: from the hand-over those roles are NoopAgents (the record's `idle`
    and `agents` say so; `on_handover` sees the idle line-up)."""
    n = warmup_months(cfg)
    if not n:
        raise ValueError("a yardstick needs a warm-up run ([game].warmup_months > 0)")
    t0 = time.time()
    ycfg = yardstick_config(cfg)
    game = game_config(ycfg, seed)
    scoring = scoring_config(ycfg, game)
    env = make_env(game, scoring)
    agents = build_agents(ycfg, env.roles, game, scoring)
    check_agents(agents)                                   # before any agent is bound or a month played
    for a in agents.values():
        a.reset(seed)
        a.bind(env)
    cities = list(getattr(env, "envs", None) or [])
    env.frame()                                            # as the runner does (a frame reads the fleet's needs)
    hand, city_scores = None, []
    series = {"month": [], "city": [], "population": []}
    by_city = {c: {"city": [], "population": []} for c in cities}

    def record(scores: dict) -> None:
        series["month"].append(env.world.month)
        series["city"].append(round(float(scores["city"]), 3))
        series["population"].append(_pop(env))
        for c in cities:
            v = scores.get(f"{c}.city")
            by_city[c]["city"].append(None if v is None else round(float(v), 3))
            by_city[c]["population"].append(_pop(env, c))

    info = None
    while not env.done:
        if hand is None and env.world.month >= n:          # the runner's hand-over point (_hand_over)
            hand = {"month": env.world.month, **_mark(env, None, env.last_scores), "scores": dict(env.last_scores)}
            hand_city = {c: _mark(env, c, env.last_scores) for c in cities}
            record(env.last_scores)
            if idle:                                       # the noop reference: these roles do nothing from here
                for r in idle:
                    agents[r] = NoopAgent(r.split(".", 1)[-1])
                    agents[r].reset(seed)
                    agents[r].bind(env)
                check_agents(agents)
            if on_handover is not None:
                on_handover(env, agents)
        acting = [r for r in env.roles if env.acts_this_month(r)]
        early = {r: env.observe(r) for r in acting} if game.simultaneous else {}
        for r in acting:
            obs = early.get(r) or env.observe(r)
            decision = agents[r].act(obs)
            results = env.apply(r, decision)
            if hook is not None:
                hook(env, r, obs, decision, results)
        info = env.end_month()
        city_scores.append(info["scores"]["city"])
        env.frame(turns=info["turns"], scores=info["scores"])
        if hand is not None:
            record(info["scores"])
        if verbose and env.world.month % 12 == 0:
            print(f"  yardstick month {env.world.month:>3}: pop {_pop(env):>6}  city {info['scores']['city']:.1f}")
    scores = info["scores"] if info else dict(env.last_scores)
    out = {"version": VERSION, "seed": seed, "months": n, "game_months": game.months,
           "signature": None, "agents": {r: agents[r].describe() for r in env.roles},
           "handed_over": hand is not None,
           "bankrupt": any(e.world.bankrupt for e in (getattr(env, "envs", None) or {"": env}).values()),
           "wall_time_s": None}
    if idle is not None:
        out["idle"] = list(idle)
    from .runner import agent_signature                  # (runner imports this module)
    out["signature"] = agent_signature(ycfg, env.roles)
    if hand is not None:
        final = {"month": env.world.month, **_mark(env, None, scores)}
        after = city_scores[n:]
        out.update(at_handover=hand, at_end=final,
                   change={"city": round(float(final["city"]) - (hand["city"] or 0.0), 3),
                           "population": final["population"] - hand["population"]},
                   avg_city_score=round(statistics.fmean(after), 3) if after else None,
                   final_scores={k: round(float(v), 3) for k, v in scores.items()}, series=series)
        if cities:
            out["by_city"] = {}
            for c in cities:
                h, e = hand_city[c], _mark(env, c, scores)
                out["by_city"][c] = {"at_handover": h, "at_end": e,
                                     "change": {"city": round(float(e["city"]) - (h["city"] or 0.0), 3),
                                                "population": e["population"] - h["population"]},
                                     "series": by_city[c]}
    out["wall_time_s"] = round(time.time() - t0, 2)
    if keep_env:
        out["env"] = env
    return out


def _same(a: dict | None, b: dict | None) -> bool:
    return bool(a and b) and a.get("month") == b.get("month") and a.get("population") == b.get("population") \
        and abs(float(a.get("city") or 0) - float(b.get("city") or 0)) < 1e-6


def compare_with_run(ys: dict, summary: dict | None) -> dict:
    """`ys` with `run` (the run's own change since the hand-over, from its summary) and `matches_run`: whether the
    city at hand-over is the run's (it must be: the same game up to N)."""
    w = (summary or {}).get("warmup") or {}
    if w.get("handed_over"):
        ys["run"] = {k: w.get(k) for k in ("at_handover", "at_end", "change", "avg_city_score")}
        ys["matches_run"] = _same(ys.get("at_handover"), w.get("at_handover"))
    return ys


def noop_part(rec: dict, ys: dict | None = None, summary: dict | None = None) -> dict:
    """The `noop` entry of yardstick.json from a `play(..., idle=...)` record: the scripted part's fields
    (NOOP_KEYS), `matches_yardstick` (its hand-over is the scripted part's, `ys`) and `matches_run` (the run's, from
    its summary) when there is something to compare with."""
    out = {"version": NOOP_VERSION, **{k: rec[k] for k in NOOP_KEYS if k in rec}}
    if ys and ys.get("at_handover"):
        out["matches_yardstick"] = _same(out.get("at_handover"), ys.get("at_handover"))
    w = (summary or {}).get("warmup") or {}
    if w.get("handed_over"):
        out["matches_run"] = _same(out.get("at_handover"), w.get("at_handover"))
    out["created"] = time.time()
    return out


def play_noop(cfg: dict, seed: int, ys: dict | None = None, summary: dict | None = None, *,
              verbose: bool = False) -> dict:
    """The noop reference of `cfg` (idle_roles idle from the hand-over) as yardstick.json's `noop` entry."""
    return noop_part(play(cfg, seed, verbose=verbose, idle=idle_roles(cfg)), ys, summary)


def noop_ok(ys: dict | None) -> dict | None:
    """The usable noop part of a yardstick record: present, handed over, and from the run's hand-over (neither match
    flag False); None otherwise (an old yardstick.json, or a noop part played by code that no longer plays the run's
    warm-up)."""
    n = (ys or {}).get("noop")
    if not n or not n.get("handed_over") or not n.get("change"):
        return None
    if n.get("matches_run") is False or n.get("matches_yardstick") is False:
        return None
    return n


def _meta(d: Path) -> dict:
    try:
        meta = json.loads((d / "meta.json").read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"no run at {d}: {e}") from None
    cfg = meta.get("config") or {}
    if not int(meta.get("warmup_months") or (cfg.get("game") or {}).get("warmup_months") or 0):
        raise ValueError(f"run {d.name} has no warm-up: a yardstick plays from a warm-up's hand-over")
    return meta


def yardstick(run_dir: str | Path, *, write: bool = True, verbose: bool = False, scripted: bool = True,
              noop: bool | str = True) -> dict:
    """The yardstick of the warm-up run in `run_dir` (its meta.json config and seed), compared with its summary when
    finished, written to `yardstick.json` there (`write`). `scripted` False keeps the scripted part of an existing
    yardstick.json (played when there is none); `noop` adds the do-nothing reference (`noop`: play_noop), which with
    `scripted` False is kept when present unless `noop` is "force". ValueError for a run without a warm-up."""
    d = Path(run_dir)
    meta = _meta(d)
    cfg, seed = meta.get("config") or {}, int(meta.get("seed") or 0)
    try:
        summary = json.loads((d / "summary.json").read_text())
    except (OSError, ValueError):
        summary = None
    ys = None if scripted else read(d)
    if not ys or not ys.get("version"):
        ys = play(cfg, seed, verbose=verbose)
        ys["run_id"] = meta.get("run_id") or d.name
        compare_with_run(ys, summary)
        ys["created"] = time.time()
    if noop and (scripted or noop == "force" or not ys.get("noop")):
        ys["noop"] = play_noop(cfg, seed, ys, summary, verbose=verbose)
    if write:
        (d / FILE).write_text(json.dumps(ys, indent=2))
    return ys


def eval_run_dirs(runs_dir: str | Path, pattern: str = "eval_*") -> list[Path]:
    """The run folders of the evals in `runs_dir` whose folder matches `pattern` (each eval.json's run ids, not
    earlier attempts), in order, each once; run ids without a folder are skipped."""
    rd, out = Path(runs_dir), []
    for p in sorted(rd.glob(f"{pattern}/eval.json")):
        try:
            ev = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        for r in ev.get("runs") or []:
            d = rd / str(r.get("run_id") or "")
            if r.get("run_id") and (d / "meta.json").exists() and d not in out:
                out.append(d)
    return out


def _backfill_one(args) -> dict:
    d, force = args
    t0 = time.time()
    try:
        ys = yardstick(d, scripted=False, noop="force" if force else True)
        n = ys.get("noop") or {}
        return {"run": Path(d).name, "ok": True, "noop": (n.get("change") or {}).get("city"),
                "scripted": (ys.get("change") or {}).get("city"), "matches": noop_ok(ys) is not None,
                "s": round(time.time() - t0, 1)}
    except Exception as e:                                # one run's failure doesn't stop the others
        return {"run": Path(d).name, "ok": False, "error": f"{type(e).__name__}: {e}", "s": round(time.time() - t0, 1)}


def backfill(run_dirs, *, workers: int = 4, force: bool = False, out=print) -> list[dict]:
    """Add the noop part to each run's yardstick.json (yardstick(scripted=False): the scripted part kept, played only
    where missing; a run that has a noop part is skipped unless `force`), `workers` processes at a time (scripted and
    noop play only: never a model). Returns one result per run, printed as each finishes."""
    todo = [Path(d) for d in run_dirs]
    if not force:
        todo = [d for d in todo if not (read(d) or {}).get("noop")]
    res: list[dict] = []
    if not todo:
        return res
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(todo)))) as ex:
        for r in ex.map(_backfill_one, [(str(d), force) for d in todo]):
            res.append(r)
            if out is None:
                continue
            if not r["ok"] or r["noop"] is None:
                out(f"{r['run']}: {r.get('error') or 'no hand-over'} ({r['s']} s)")
            else:
                out(f"{r['run']}: noop {r['noop']:+.2f}, scripted {r['scripted']:+.2f}"
                    + ("" if r["matches"] else "  (hand-over differs from the run's: not used)") + f" ({r['s']} s)")
    return res


def read(run_dir: str | Path) -> dict | None:
    try:
        return json.loads((Path(run_dir) / FILE).read_text())
    except (OSError, ValueError):
        return None


def line(ys: dict) -> str:
    """'scripted from the same start: +1.4 / +868' (city score / residents since the hand-over)."""
    c = ys.get("change") or {}
    if not ys.get("handed_over") or not c:
        return "scripted from the same start: no hand-over (the game ended in the warm-up)"
    n = (noop_ok(ys) or {}).get("change")
    return (f"scripted from the same start: {c['city']:+.1f} / {c['population']:+,}"
            + (f"; doing nothing: {n['city']:+.1f} / {n['population']:+,}" if n else ""))


def comparison(ys: dict) -> str:
    """The run against its yardstick, as `civitas yardstick` prints it."""
    run = (ys.get("run") or {}).get("change")
    h, e = ys.get("at_handover") or {}, ys.get("at_end") or {}
    head = (f"{ys.get('run_id', 'run')}: months {ys['months']}-{(e.get('month') or ys['game_months']) - 1} "
            f"from city {float(h.get('city') or 0):.1f}, population {h.get('population', 0):,}")
    out = [head]
    if run:
        out.append(f"  the run's agents:    {run['city']:+.1f} city / {run['population']:+,} residents")
    c = ys.get("change")
    if c:
        out.append(f"  scripted yardstick:  {c['city']:+.1f} city / {c['population']:+,} residents"
                   f"  (city {float(e['city']):.1f}, population {e['population']:,}; {ys.get('wall_time_s')} s)")
    for city, b in (ys.get("by_city") or {}).items():
        out.append(f"    {city}: {b['change']['city']:+.1f} / {b['change']['population']:+,}")
    n = ys.get("noop") or {}
    if n.get("change"):
        ne = n.get("at_end") or {}
        out.append(f"  doing nothing:       {n['change']['city']:+.1f} city / {n['change']['population']:+,} residents"
                   f"  (city {float(ne['city']):.1f}, population {ne['population']:,}; {len(n.get('idle') or [])} "
                   f"roles idle; {n.get('wall_time_s')} s)")
        for city, b in (n.get("by_city") or {}).items():
            out.append(f"    {city}: {b['change']['city']:+.1f} / {b['change']['population']:+,}")
        if noop_ok(ys) is None:
            out.append("  WARNING: the noop reference's hand-over differs from the run's or the yardstick's: not used")
    if ys.get("matches_run") is False:
        out.append("  WARNING: the city at hand-over differs from the run's: the engine or config changed since it ran")
    return "\n".join(out)
