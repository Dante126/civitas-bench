"""Run episodes, log replays + trajectories, and aggregate benchmarks."""
from __future__ import annotations

import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..config import agent_spec, build_agents, game_config, scoring_config
from ..core.env import make_env


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


class RunLogger:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.frames = open(self.dir / "frames.jsonl", "a", buffering=1)
        self.traj = open(self.dir / "trajectories.jsonl", "a", buffering=1)

    def meta(self, meta: dict) -> None:
        (self.dir / "meta.json").write_text(json.dumps(_jsonable(meta), indent=2))

    def frame(self, frame: dict) -> None:
        self.frames.write(json.dumps(_jsonable(frame), separators=(",", ":")) + "\n")

    def turn(self, rec: dict) -> None:
        self.traj.write(json.dumps(_jsonable(rec)) + "\n")

    def summary(self, summary: dict) -> None:
        (self.dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))

    def close(self) -> None:
        self.frames.close()
        self.traj.close()


def agent_signature(cfg: dict, roles: list[str]) -> str:
    parts = []
    seen = []
    for r in roles:
        base = r.split(".", 1)[-1]
        if "." in r and base in seen and agent_spec(cfg, r) == agent_spec(cfg, f"north.{base}"):
            continue                      # region mode: list each role once when both cities use the same agent
        seen.append(base)
        a = agent_spec(cfg, r)
        k = a.get("kind", "scripted")
        r = base if "." in r and agent_spec(cfg, f"north.{base}") == agent_spec(cfg, f"south.{base}") else r
        parts.append(f"{r}={k}:{a['model']}" if k == "llm" else f"{r}={k}")
    return " ".join(parts)


def run_episode(cfg: dict, seed: int, runs_dir: str | Path = "runs", *, hub=None,
                verbose: bool = True, log_observations: bool = True, run_id: str | None = None) -> dict:
    game = game_config(cfg, seed)
    scoring = scoring_config(cfg, game)
    env = make_env(game, scoring)
    agents = build_agents(cfg, env.roles, game, scoring, hub=hub)
    for a in agents.values():
        a.reset(seed)

    run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg['name']}_s{seed}_{os.getpid() % 1000:03d}"
    log = RunLogger(Path(runs_dir) / run_id)
    sig = agent_signature(cfg, env.roles)
    log.meta({"run_id": run_id, "name": cfg["name"], "seed": seed, "width": game.width,
              "height": game.height, "months": game.months, "roles": env.roles,
              "agents": {r: agents[r].describe() for r in env.roles}, "signature": sig,
              "config": cfg, "started": time.time(), **env.meta()})
    if hub is not None:
        hub.run_id = run_id
    log.frame(env.frame())
    if verbose:
        print(f"[{run_id}] {sig}")

    stats = {r: {"turns": 0, "actions": 0, "invalid": 0, "messages": 0, "spent": 0.0,
                 "parse_errors": 0, "reward": 0.0} for r in env.roles}
    city_scores, t0 = [], time.time()
    while not env.done:
        pending = []
        acting = [r for r in env.roles if env.acts_this_month(r)]
        # simultaneous mode: everybody sees the same start-of-month state, then all decisions apply in turn order
        early = {r: env.observe(r) for r in acting} if game.simultaneous else {}
        for role in acting:
            obs = early.get(role) or env.observe(role)
            decision = agents[role].act(obs)
            results = env.apply(role, decision)
            st = stats[role]
            st["turns"] += 1
            st["actions"] += len(results)
            st["invalid"] += sum(1 for r in results if not r["ok"])
            st["messages"] += len(decision.messages)
            st["spent"] += sum(r.get("cost", 0) for r in results if r["ok"])
            st["parse_errors"] += 1 if decision.parse_error and decision.parse_error != "recovered_after_retry" else 0
            rec = {"run_id": run_id, "seed": seed, "month": obs["month"], "role": role,
                   "agent": agents[role].describe(), "thinking": decision.thinking,
                   "actions": decision.actions, "messages": decision.messages, "notes": decision.notes,
                   "results": results, "parse_error": decision.parse_error}
            if "messages" in decision.meta:
                rec["prompt"] = decision.meta["messages"]
                rec["response"] = decision.meta.get("response")
                rec["reasoning"] = decision.meta.get("reasoning")
                rec["finish_reason"] = decision.meta.get("finish_reason")
                rec["usage"] = decision.meta.get("usage")
                rec["latency_s"] = decision.meta.get("latency_s")
            elif log_observations:
                rec["observation_text"] = obs["text"]
            pending.append(rec)
        info = env.end_month()
        for rec in pending:
            rec["reward"] = info["rewards"][rec["role"]]
            rec["scores_after"] = info["scores"]
            log.turn(rec)
        for r in env.roles:
            stats[r]["reward"] += info["rewards"][r]
        city_scores.append(info["scores"]["city"])
        frame = env.frame(turns=info["turns"], scores=info["scores"])
        log.frame(frame)
        if hub is not None:
            hub.latest_frame = frame
        m = info["metrics"]
        if verbose and (m["month"] % 12 == 0 or env.done):
            print(f"  month {m['month']:>3}: pop {m['population']:>6}  treasury {m['treasury']:>10,.0f}  "
                  f"happy {m['happiness']:.2f}  scores " + " ".join(f"{k}={v:.1f}" for k, v in info["scores"].items()))

    final = env.final_metrics()
    if "cities" in env.meta():
        fm = final
        final = {"population": sum(m["population"] for m in fm.values()),
                 "happiness": round(sum(m["happiness"] for m in fm.values()) / len(fm), 4),
                 "treasury": sum(m["treasury"] for m in fm.values()), "debt": sum(m["debt"] for m in fm.values()),
                 "net_worth": sum(m["net_worth"] for m in fm.values()),
                 "unemployment": round(sum(m["unemployment"] for m in fm.values()) / len(fm), 4),
                 **{k: None for k in ("power_coverage", "water_coverage", "pollution_res", "land_value_dev")},
                 "developed_total": sum(m["developed_total"] for m in fm.values()),
                 "by_city": {c: {k: m[k] for k in ("population", "happiness", "treasury", "river_pollution")}
                             for c, m in fm.items()}}
    for r in env.roles:
        st = stats[r]
        st["invalid_rate"] = round(st["invalid"] / st["actions"], 4) if st["actions"] else 0.0
        st["spent"] = round(st["spent"], 2)
        st["reward"] = round(st["reward"], 4)
        if hasattr(agents[r], "usage"):
            st["llm_usage"] = agents[r].usage
    summary = {
        "run_id": run_id, "name": cfg["name"], "seed": seed, "signature": sig,
        "agents": {r: agents[r].describe() for r in env.roles},
        "months_played": env.world.month,
        "bankrupt": any(e.world.bankrupt for e in getattr(env, "envs", {"": env}).values()),
        "final_scores": info["scores"],
        "avg_city_score": round(statistics.fmean(city_scores), 3) if city_scores else 0.0,
        "final_metrics": {k: final.get(k) for k in ("population", "happiness", "treasury", "debt", "net_worth",
                                                   "unemployment", "power_coverage", "water_coverage",
                                                   "pollution_res", "land_value_dev", "developed_total", "by_city")
                          if k in final},
        "agent_stats": stats,
        "wall_time_s": round(time.time() - t0, 2),
    }
    log.summary(summary)
    log.close()
    if verbose:
        print(f"[{run_id}] done: city {info['scores']['city']:.1f}  " +
              " ".join(f"{r}={info['scores'][r]:.1f}" for r in env.roles) + f"  ({summary['wall_time_s']}s)")
    return summary


def _run_one(args):
    cfg, seed, runs_dir, verbose = args
    return run_episode(cfg, seed, runs_dir, verbose=verbose)


def run_benchmark(cfg: dict, seeds: list[int], runs_dir="runs", workers: int = 1, verbose=True) -> dict:
    jobs = [(cfg, s, runs_dir, verbose) for s in seeds]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_run_one, jobs))
    else:
        results = [_run_one(j) for j in jobs]
    agg = aggregate(results)
    out = Path(runs_dir) / f"bench_{cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"config": cfg["name"], "seeds": seeds, "aggregate": agg, "runs": results}, indent=2))
    print_table([agg])
    print(f"saved {out}")
    return agg


def _ms(xs):
    xs = list(xs)
    if not xs:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": round(statistics.fmean(xs), 2), "std": round(statistics.pstdev(xs), 2) if len(xs) > 1 else 0.0}


def aggregate(summaries: list[dict]) -> dict:
    s0 = summaries[0]
    roles = list(s0["agents"])
    return {
        "name": s0["name"], "signature": s0["signature"], "n": len(summaries),
        "city": _ms(s["final_scores"]["city"] for s in summaries),
        "avg_city": _ms(s["avg_city_score"] for s in summaries),
        "roles": {r: _ms(s["final_scores"][r] for s in summaries) for r in roles},
        "invalid_rate": {r: _ms(s["agent_stats"][r]["invalid_rate"] for s in summaries) for r in roles},
        "population": _ms(s["final_metrics"]["population"] for s in summaries),
        "bankruptcies": sum(1 for s in summaries if s["bankrupt"]),
    }


def leaderboard(runs_dir="runs") -> list[dict]:
    groups: dict[str, list] = {}
    for p in Path(runs_dir).glob("*/summary.json"):
        try:
            s = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        groups.setdefault(s["signature"], []).append(s)
    rows = [aggregate(v) for v in groups.values()]
    rows.sort(key=lambda r: -r["city"]["mean"])
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("no runs found")
        return
    roles = list(rows[0]["roles"])
    hdr = f"{'agents':<58} {'n':>3} {'city':>12} {'pop':>8} " + " ".join(f"{r:>12}" for r in roles) + "  bankrupt"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['signature'][:58]:<58} {r['n']:>3} {r['city']['mean']:>6.1f}±{r['city']['std']:<5.1f} "
              f"{r['population']['mean']:>8.0f} " +
              " ".join(f"{r['roles'][k]['mean']:>6.1f}±{r['roles'][k]['std']:<5.1f}" for k in roles) +
              f"  {r['bankruptcies']}")
