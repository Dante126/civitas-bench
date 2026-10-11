"""`civitas rl check`: does the reward mean something? Hand-made policies (policies.py, no model) on bank tasks.

For each task (actionable ones of a split, spread evenly over the bank) every policy plays one episode and its
branch runs to the longest horizon; the reward is read at each horizon (default 3, 6 and 12 months). The report:
mean reward per policy with 95% t-intervals, the ordering, the paired differences against noop, how often the
policies of one task disagree (GRPO needs differing rewards within a group), determinism (episodes replayed in fresh
processes must give identical branches), and the wall time of a reward. Jobs run one (config, seed) per fresh
process (bench/envs.py:_isolated_map), `workers` at a time.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from ..bench import stats as S
from . import bank as B
from . import reward as R
from .policies import POLICIES

ALL = ("baseline", *POLICIES)
HORIZONS = (3, 6, 12)
GROUP = ("noop", "scripted", "degraded", "random", "oracle")     # a GRPO-like group of different policies


def play_task(task, policy: str, horizon: int, cfg: R.RewardConfig | None = None) -> dict:
    """One episode of `policy` on `task`, its branch to `horizon` months: {"series", "counts", "seconds"}."""
    from .episode import TaskEpisode
    from .policies import play
    t0 = time.perf_counter()
    ep = TaskEpisode(task)
    if policy == "baseline":                       # the scripted fleet agent's own turn, as the bank's game played it
        ep.prepare()
        t1 = time.perf_counter()
        d = ep.agents[ep.role].act(ep.env.observe(ep.role))
        t2 = time.perf_counter()
        out = ep.finish(decision=d, horizon=horizon, reward_cfg=cfg)
    else:
        ep.prepare()
        t1 = time.perf_counter()
        ep.reset()
        play(ep, policy)
        t2 = time.perf_counter()
        out = ep.finish(horizon=horizon, reward_cfg=cfg)
    t3 = time.perf_counter()
    return {"series": out["series"], "counts": out["counts"],
            "seconds": {"prepare": round(t1 - t0, 4), "turn": round(t2 - t1, 4), "branch": out["series"]["seconds"],
                        "total": round(t3 - t0, 4)}}


def _job(job: dict) -> list[dict]:
    """Every (task, policy) of one job, in one process."""
    bank = B.TaskBank(job["bank"], root=job["root"])
    out = []
    for tid in job["tasks"]:
        task = bank.get(tid)
        for p in job["policies"]:
            rec = play_task(task, p, job["horizon"])
            out.append({"task_id": tid, "policy": p, **rec})
    return out


def _jobs(bank: B.TaskBank, tasks, policies, horizon: int, root, per_job: int | None = None) -> list[dict]:
    groups: dict[str, list[str]] = {}
    for t in tasks:
        groups.setdefault(f"{t.config}:{t.seed}", []).append(t.task_id)
    jobs = []
    for ids in groups.values():
        step = per_job or len(ids)
        for i in range(0, len(ids), step):
            jobs.append({"bank": str(bank.dir), "root": str(root), "tasks": ids[i:i + step],
                         "policies": list(policies), "horizon": horizon})
    return jobs


def run(bank_name: str, *, split: str = "train", n: int = 150, policies=ALL, horizons=HORIZONS, workers: int = 4,
        determinism: int = 12, root=B.TASKS_DIR, cfg: R.RewardConfig | None = None, verbose: bool = True) -> dict:
    """Play the policies on `n` tasks and summarize (see the module doc); writes check_<stamp>.json / .md in the
    bank's folder and returns the summary."""
    from ..bench.envs import _isolated_map
    cfg = cfg or R.RewardConfig()
    bank = B.TaskBank(bank_name, root=root)
    tasks = bank.tasks(split, actionable=True, limit=n)
    if not tasks:
        raise B.BankError(f"bank {bank.name} has no actionable {split} tasks")
    hmax = max(horizons)
    t0 = time.time()
    if verbose:
        print(f"check {bank.name}: {len(tasks)} {split} tasks x {len(policies)} policies, horizons {list(horizons)}, "
              f"{workers} workers ...", flush=True)
    recs = [r for res in _isolated_map(_job, _jobs(bank, tasks, policies, hmax, root), workers) for r in res]
    rows = {t.task_id: t.row for t in tasks}
    for r in recs:
        r["rewards"] = {h: R.compute(rows[r["task_id"]], r["series"], r["counts"],
                                     R.RewardConfig(**{**cfg.as_dict(), "horizon": h})) for h in horizons}
    det = _determinism(bank, tasks, recs, hmax, root, workers, determinism) if determinism else None
    summary = summarize(recs, rows, horizons, cfg, policies)
    summary.update(bank=bank.name, split=split, n_tasks=len(tasks), policies=list(policies),
                   horizons=list(horizons), reward=cfg.as_dict(), determinism=det,
                   seconds=round(time.time() - t0, 1))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (bank.dir / f"check_{stamp}.json").write_text(json.dumps({"summary": summary, "records": [
        {k: v for k, v in r.items() if k != "series"} | {"city": r["series"]["city"], "role": r["series"]["role"]}
        for r in recs]}, indent=1, default=str))
    md = markdown(summary)
    (bank.dir / f"check_{stamp}.md").write_text(md)
    if verbose:
        print(md)
        print(f"-> {bank.dir / f'check_{stamp}.md'}")
    return summary


def _determinism(bank, tasks, recs, hmax, root, workers, k) -> dict:
    """Replay `k` tasks' degraded and random episodes in fresh processes, one task per process (so a different
    process and a cold cache from the first pass): their branches must be identical."""
    from ..bench.envs import _isolated_map
    pols = [p for p in ("degraded", "random", "oracle") if any(r["policy"] == p for r in recs)][:2]
    pick = tasks[:: max(1, len(tasks) // k)][:k]
    first = {(r["task_id"], r["policy"]): r for r in recs}
    again = [r for res in _isolated_map(_job, _jobs(bank, pick, pols, hmax, root, per_job=1), workers) for r in res]
    same = [first[(r["task_id"], r["policy"])]["series"]["city"] == r["series"]["city"]
            and first[(r["task_id"], r["policy"])]["series"]["role"] == r["series"]["role"]
            and first[(r["task_id"], r["policy"])]["counts"] == r["counts"] for r in again]
    return {"replayed": len(again), "identical": sum(same), "policies": pols,
            "differ": [r["task_id"] + ":" + r["policy"] for r, s in zip(again, same) if not s]}


def _q(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def summarize(recs, rows, horizons, cfg, policies) -> dict:
    by = {}
    for r in recs:
        by.setdefault(r["policy"], {})[r["task_id"]] = r
    tids = sorted(rows)
    out = {"by_policy": {}, "vs_noop": {}, "signal": {}, "within_task": {}, "by_role": {}, "timing": {}}
    for p in policies:
        recs_p = by.get(p, {})
        out["by_policy"][p] = {}
        for h in horizons:
            vals = [recs_p[t]["rewards"][h] for t in tids if t in recs_p]
            ci = S.mean_ci([v["reward"] for v in vals])
            out["by_policy"][p][h] = {
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in (ci or {}).items()},
                "city": round(statistics.fmean(v["city"] for v in vals), 4) if vals else None,
                "role": round(statistics.fmean(v["role"] for v in vals), 4) if vals else None,
                "penalty": round(statistics.fmean(v["penalty"] for v in vals), 4) if vals else None,
                "positive": sum(v["reward"] > 1e-9 for v in vals), "zero": sum(abs(v["reward"]) <= 1e-9 for v in vals),
                "negative": sum(v["reward"] < -1e-9 for v in vals)}
        tot = [recs_p[t]["seconds"] for t in tids if t in recs_p]
        if tot:
            out["timing"][p] = {}
            for h in horizons:
                wall = [s["prepare"] + s["turn"] + s["branch"][min(h, len(s["branch"])) - 1] for s in tot]
                out["timing"][p][h] = {"mean": round(statistics.fmean(wall), 3), "median": round(_q(wall, 0.5), 3),
                                       "p90": round(_q(wall, 0.9), 3), "max": round(max(wall), 3)}
            out["timing"][p]["prepare"] = round(statistics.fmean(s["prepare"] for s in tot), 4)
            out["timing"][p]["turn"] = round(statistics.fmean(s["turn"] for s in tot), 4)
    noop = by.get("noop", {})
    for p in policies:
        if p == "noop" or not noop:
            continue
        out["vs_noop"][p], out["signal"][p] = {}, {}
        for h in horizons:
            d = [by[p][t]["rewards"][h]["reward"] - noop[t]["rewards"][h]["reward"] for t in tids
                 if t in by.get(p, {}) and t in noop]
            v = S.verdict(d)
            out["vs_noop"][p][h] = ({k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()}
                                    | {"wins": sum(x > 1e-9 for x in d), "ties": sum(abs(x) <= 1e-9 for x in d),
                                       "losses": sum(x < -1e-9 for x in d)}) if v else None
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            out["signal"][p][h] = round(statistics.fmean(d) / sd, 3) if sd > 0 else None   # paired effect size
    group = [p for p in GROUP if p in by]
    calm = [p for p in group if p != "random"]          # random always differs (its penalties): the group without it
    for h in horizons:
        differ, differ_calm, spreads, distinct = 0, 0, [], []
        for t in tids:
            vals = [by[p][t]["rewards"][h]["reward"] for p in group if t in by[p]]
            if len(vals) < 2:
                continue
            k = len({round(v, 6) for v in vals})
            distinct.append(k)
            differ += k > 1
            differ_calm += len({round(by[p][t]["rewards"][h]["reward"], 6) for p in calm if t in by[p]}) > 1
            spreads.append(statistics.pstdev(vals))
        out["within_task"][h] = {"tasks": len(spreads), "policies": group, "differ": differ,
                                 "share_differ": round(differ / len(spreads), 3) if spreads else None,
                                 "differ_without_random": differ_calm,
                                 "mean_distinct": round(statistics.fmean(distinct), 2) if distinct else None,
                                 "mean_std": round(statistics.fmean(spreads), 4) if spreads else None}
    h0 = cfg.horizon if cfg.horizon in horizons else horizons[0]
    roles = sorted({rows[t]["role"] for t in tids})
    for p in policies:
        out["by_role"][p] = {}
        for role in roles:
            vals = [by[p][t]["rewards"][h0]["reward"] for t in tids if t in by.get(p, {}) and rows[t]["role"] == role]
            out["by_role"][p][role] = round(statistics.fmean(vals), 4) if vals else None
    order = sorted((p for p in policies if out["by_policy"][p].get(h0, {}).get("mean") is not None),
                   key=lambda p: -out["by_policy"][p][h0]["mean"])
    out["ordering"] = {h: sorted((p for p in policies if out["by_policy"][p].get(h, {}).get("mean") is not None),
                                 key=lambda p, h=h: -out["by_policy"][p][h]["mean"]) for h in horizons}
    out["order_default"] = order
    return out


def _f(v, d=3):
    return "-" if v is None else f"{v:+.{d}f}"


def markdown(s: dict) -> str:
    H = s["horizons"]
    L = [f"# RL check: bank {s['bank']}, {s['n_tasks']} {s['split']} tasks", "",
         f"Reward: {json.dumps(s['reward'])}; {s['seconds']} s.", "",
         "## Mean reward per policy (95% CI)", "",
         "| policy | " + " | ".join(f"H={h}" for h in H) + f" | city H={H[1] if len(H) > 1 else H[0]} | "
         f"role | penalty | +/0/- |",
         "|---|" + "---|" * (len(H) + 4)]
    hm = H[1] if len(H) > 1 else H[0]
    for p, d in s["by_policy"].items():
        cells = []
        for h in H:
            c = d.get(h) or {}
            cells.append(f"{_f(c.get('mean'))} [{_f(c.get('lo'))}, {_f(c.get('hi'))}]" if c.get("lo") is not None
                         else _f(c.get("mean")))
        m = d.get(hm) or {}
        L.append(f"| {p} | " + " | ".join(cells) + f" | {_f(m.get('city'))} | {_f(m.get('role'))} | "
                 f"{_f(m.get('penalty'))} | {m.get('positive')}/{m.get('zero')}/{m.get('negative')} |")
    L += ["", "Ordering by mean: " + "; ".join(f"H={h}: " + " > ".join(o) for h, o in s["ordering"].items()), "",
          "## Paired difference against noop (policy - noop)", "",
          "| policy | " + " | ".join(f"H={h} mean [CI] (wins/ties/losses), effect size" for h in H) + " |",
          "|---|" + "---|" * len(H)]
    for p, d in s["vs_noop"].items():
        cells = []
        for h in H:
            v = d.get(h)
            cells.append("-" if not v else f"{_f(v['mean'])} [{_f(v.get('lo'))}, {_f(v.get('hi'))}] "
                         f"({v['wins']}/{v['ties']}/{v['losses']}), d={s['signal'][p].get(h)}")
        L.append(f"| {p} | " + " | ".join(cells) + " |")
    L += ["", "## Within-task variance (the group " + ", ".join(next(iter(s["within_task"].values()))["policies"])
          + ")", "", "| H | tasks | rewards differ | ... without random | distinct values (mean) | std within task "
          "(mean) |", "|---|---|---|---|---|---|"]
    for h, w in s["within_task"].items():
        L.append(f"| {h} | {w['tasks']} | {w['differ']} ({w['share_differ']}) | {w.get('differ_without_random')} | "
                 f"{w['mean_distinct']} | {w['mean_std']} |")
    L += ["", f"## By role (H={s['reward']['horizon']})", "",
          "| policy | " + " | ".join(next(iter(s["by_role"].values()))) + " |",
          "|---|" + "---|" * len(next(iter(s["by_role"].values())))]
    for p, d in s["by_role"].items():
        L.append(f"| {p} | " + " | ".join(_f(v) for v in d.values()) + " |")
    det = s.get("determinism")
    if det:
        L += ["", f"## Determinism", "", f"{det['identical']}/{det['replayed']} episodes ({', '.join(det['policies'])})"
              " replayed in fresh processes gave identical branches" + (f"; differ: {det['differ']}" if det['differ']
                                                                          else ".")]
    L += ["", "## Wall time per reward (prepare + turn + branch, seconds)", "",
          "| policy | " + " | ".join(f"H={h} mean / median / p90" for h in H) + " | prepare | turn |",
          "|---|" + "---|" * (len(H) + 2)]
    for p, d in s["timing"].items():
        L.append(f"| {p} | " + " | ".join(f"{d[h]['mean']:.2f} / {d[h]['median']:.2f} / {d[h]['p90']:.2f}"
                                          for h in H) + f" | {d['prepare']:.3f} | {d['turn']:.3f} |")
    return "\n".join(L) + "\n"
