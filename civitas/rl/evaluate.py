"""`civitas rl eval`: a model on an OpenAI-compatible endpoint (vLLM on Modal, a local server, a provider) over a bank
split, each task played by the fleet's own ToolAgent loop (fleet/agent.py, one ChatClient) limited to that one turn,
and scored by the task's reward (reward.py).

Per task and sample: the task's state is prepared (episode.py), a ToolAgent with the task config's fleet settings
(playbooks, max_steps, max_tokens, turn_tokens, briefing style) but the given model and endpoint is bound to it and
plays the turn (its nudges, recovered text calls, step and token limits), then the branch runs H months against the
cached baseline. Written to runs/_rl_evals/<name>_<stamp>/: results.jsonl (one line per task and sample: reward and
components, counts, tokens, steps, finish reason, errors), transcripts.jsonl (the messages a ToolAgent sent and got,
unless `transcripts=False`), summary.json and summary.md (mean reward with a 95% t-interval over tasks, by role and
style, the noop reference from the bank, failed / refused / invalid calls per task, tokens).
"""
from __future__ import annotations

import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..bench import stats as S
from . import bank as B
from . import reward as R

OUT_DIR = Path("runs") / "_rl_evals"


def play_one(task, *, base_url: str, model: str, api_key_env: str = "", stream: bool = True, timeout: float = 300.0,
             max_retries: int = 2, temperature=None, max_tokens: int | None = None, extra_body: dict | None = None,
             reward_cfg: R.RewardConfig | None = None) -> dict:
    """One task played by a ToolAgent on the endpoint; the result row (see the module doc) and the transcript."""
    from ..agents.llm_client import ProviderConfig
    from ..fleet.agent import ToolAgent
    from .episode import TaskEpisode
    t0 = time.perf_counter()
    ep = TaskEpisode(task)
    env = ep.prepare()
    fc = ep.fleet_cfg
    fc.model, fc.provider = model, "rl-eval"
    if temperature is not None:
        fc.temperature = temperature
    if max_tokens:
        fc.max_tokens = int(max_tokens)
    if extra_body is not None:
        fc.extra_body = dict(extra_body)
    prov = ProviderConfig(base_url=base_url, api_key_env=api_key_env or "", stream=stream, timeout=timeout,
                          max_retries=max_retries)
    agent = ToolAgent(task.role, cfg=fc, provider=prov)
    agent.bind(env)
    d = agent.act(env.observe(task.role))
    t1 = time.perf_counter()
    r = ep.finish(decision=d, reward_cfg=reward_cfg)
    meta = d.meta or {}
    row = {"task_id": task.task_id, "role": task.role, "month": task.month, "seed": task.seed, "style": task.style,
           "reward": r["reward"], "city": r["city"], "role_score": r["role"],
           **{k: r[k] for k in ("penalty", "horizon", "population")},
           "counts": r["counts"], "usage": meta.get("usage"), "steps": meta.get("steps"), "ended": meta.get("ended"),
           "finish_reason": meta.get("finish_reason"), "length_stops": meta.get("length_stops"),
           "recovered_calls": meta.get("recovered_calls"), "error": d.parse_error,
           "tools": [t["tool"] for t in meta.get("tool_trace") or []],
           "latency_s": meta.get("latency_s"), "turn_s": round(t1 - t0, 2),
           "reward_s": round(time.perf_counter() - t1, 3)}
    transcript = {"task_id": task.task_id, "messages": meta.get("messages") or [], "system": agent.system,
                  "tool_trace": meta.get("tool_trace") or [], "report": d.thinking}
    return {"row": row, "transcript": transcript}


def noop_reward(row: dict, cfg: R.RewardConfig) -> float | None:
    """The noop turn's reward from the bank's noop branch (an end_turn: no failed or invalid calls)."""
    n = row.get("noop")
    if not n:
        return None
    zero = {"calls": 1, "actions": 0, "failed": 0, "refused": 0, "invalid": 0, "ended": True}
    return R.compute(row, n, zero, cfg)["reward"]


def run(bank_name: str, *, base_url: str, model: str, split: str = "test", n: int = 1, limit: int | None = None,
        roles=None, styles=None, api_key_env: str = "", stream: bool = True, timeout: float = 300.0,
        temperature=None, max_tokens: int | None = None, extra_body: dict | None = None, workers: int = 4,
        reward_cfg: R.RewardConfig | None = None, root=B.TASKS_DIR, out_dir=OUT_DIR, name: str | None = None,
        transcripts: bool = True, verbose: bool = True) -> dict:
    cfg = reward_cfg or R.RewardConfig()
    bank = B.TaskBank(bank_name, root=root)
    tasks = bank.tasks(split, actionable=True, roles=roles, styles=styles, limit=limit)
    if not tasks:
        raise B.BankError(f"bank {bank.name} has no actionable {split} tasks for that selection")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    d = Path(out_dir) / f"{name or model.replace('/', '_')}_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    jobs = [(t, i) for t in tasks for i in range(n)]
    lock = threading.Lock()
    rows: list[dict] = []
    res_f = open(d / "results.jsonl", "w", buffering=1)
    tr_f = open(d / "transcripts.jsonl", "w", buffering=1) if transcripts else None
    t0 = time.time()
    if verbose:
        print(f"rl eval {model} @ {base_url}: {len(tasks)} {split} tasks of bank {bank.name} x {n} -> {d}", flush=True)

    def one(job):
        task, i = job
        try:
            out = play_one(task, base_url=base_url, model=model, api_key_env=api_key_env, stream=stream,
                           timeout=timeout, temperature=temperature, max_tokens=max_tokens, extra_body=extra_body,
                           reward_cfg=cfg)
            row, tr = out["row"], out["transcript"]
        except Exception as e:                       # a task that fails is recorded, not fatal to the eval
            row, tr = {"task_id": task.task_id, "role": task.role, "month": task.month, "seed": task.seed,
                       "style": task.style, "reward": None, "error": f"{type(e).__name__}: {e}"}, None
        row["sample"] = i
        with lock:
            rows.append(row)
            res_f.write(json.dumps(row, default=str) + "\n")
            if tr_f and tr:
                tr_f.write(json.dumps({**tr, "sample": i}, default=str) + "\n")
            if verbose:
                r = row.get("reward")
                print(f"  [{len(rows)}/{len(jobs)}] {task.task_id} #{i}: "
                      + (f"reward {r:+.3f} (city {row['city']:+.3f}, role {row['role_score']:+.3f}, "
                         f"{row['counts']['actions']} actions, {row['counts']['failed']} failed)" if r is not None
                         else f"error {row['error']}"), flush=True)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(one, jobs))
    res_f.close()
    if tr_f:
        tr_f.close()
    summary = summarize(rows, {t.task_id: t.row for t in tasks}, cfg)
    summary.update(bank=bank.name, split=split, model=model, base_url=base_url, n=n, tasks=len(tasks),
                   reward_cfg=cfg.as_dict(), seconds=round(time.time() - t0, 1), dir=str(d))
    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    md = markdown(summary)
    (d / "summary.md").write_text(md)
    if verbose:
        print(md)
    return summary


def _ci(xs) -> dict | None:
    c = S.mean_ci(xs)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} if c else None


def summarize(rows: list[dict], task_rows: dict, cfg: R.RewardConfig) -> dict:
    ok = [r for r in rows if r.get("reward") is not None]
    per_task: dict[str, list[float]] = {}
    for r in ok:
        per_task.setdefault(r["task_id"], []).append(r["reward"])
    means = {t: statistics.fmean(v) for t, v in per_task.items()}
    noop = {t: noop_reward(task_rows[t], cfg) for t in means}
    noop = {t: v for t, v in noop.items() if v is not None}
    by = lambda key: {k: _ci([means[t] for t in means if task_rows[t][key] == k])
                      for k in sorted({task_rows[t][key] for t in means}, key=str)}
    tok = [r.get("usage") or {} for r in ok]
    return {"reward": _ci(means.values()), "city": _ci([r["city"] for r in ok]),
            "role_score": _ci([r["role_score"] for r in ok]),
            "noop_reference": _ci(noop.values()),
            "vs_noop": _ci([means[t] - noop[t] for t in noop]),
            "by_role": by("role"), "by_style": by("style"), "by_month": by("month"),
            "errors": sum(1 for r in rows if r.get("reward") is None or r.get("error")),
            "per_task": {k: round(statistics.fmean((r.get("counts") or {}).get(k, 0) for r in ok), 3) if ok else None
                         for k in ("actions", "failed", "refused", "invalid", "calls")},
            "ended_share": round(sum(1 for r in ok if r.get("ended")) / len(ok), 3) if ok else None,
            "tokens": {"prompt": sum(int(t.get("prompt_tokens") or 0) for t in tok),
                       "completion": sum(int(t.get("completion_tokens") or 0) for t in tok),
                       "per_task_completion": round(statistics.fmean(int(t.get("completion_tokens") or 0)
                                                                     for t in tok), 1) if tok else None},
            "within_task_std": round(statistics.fmean(statistics.pstdev(v) for v in per_task.values()), 4)
            if any(len(v) > 1 for v in per_task.values()) else None,
            "results": len(rows)}


def markdown(s: dict) -> str:
    f = lambda c: "-" if not c else (f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] (n={c['n']})"
                                     if c.get("lo") is not None else f"{c['mean']:+.3f} (n={c['n']})")
    L = [f"# RL eval: {s['model']} on bank {s['bank']} ({s['split']}, {s['tasks']} tasks x {s['n']})", "",
         f"- reward (mean over tasks, 95% CI): {f(s['reward'])}",
         f"- components: city {f(s['city'])}, role score {f(s['role_score'])}",
         f"- noop reference (the bank's noop turn on the same tasks): {f(s['noop_reference'])}; model - noop: "
         f"{f(s['vs_noop'])}",
         f"- per task: {s['per_task']}; ended with end_turn: {s['ended_share']}; errors: {s['errors']}",
         f"- tokens: {s['tokens']}; within-task std over samples: {s['within_task_std']}",
         f"- reward settings: {json.dumps(s['reward_cfg'])}; {s['seconds']} s", "", "| by | value | reward |",
         "|---|---|---|"]
    for key in ("by_role", "by_style", "by_month"):
        for k, c in s[key].items():
            L.append(f"| {key[3:]} | {k} | {f(c)} |")
    return "\n".join(L) + "\n"


def _per_task(d) -> dict:
    """{task_id: (mean reward over samples, role)} from an eval folder's results.jsonl."""
    by: dict[str, list] = {}
    for line in (Path(d) / "results.jsonl").read_text().splitlines():
        r = json.loads(line) if line.strip() else {}
        if r.get("reward") is not None:
            by.setdefault(r["task_id"], [[], r.get("role")])[0].append(float(r["reward"]))
    return {t: (statistics.fmean(v), role) for t, (v, role) in by.items()}


def compare(a, b) -> dict:
    """Paired comparison of two `civitas rl eval` folders on their common tasks (b - a per task: mean and 95%
    t-interval, overall and by role; wins / ties / losses)."""
    pa, pb = _per_task(a), _per_task(b)
    common = sorted(set(pa) & set(pb))
    if not common:
        raise B.BankError(f"{a} and {b} share no scored task")
    diff = {t: pb[t][0] - pa[t][0] for t in common}
    roles = sorted({pa[t][1] for t in common}, key=str)
    return {"a": str(a), "b": str(b), "tasks": len(common), "only_a": len(set(pa) - set(pb)),
            "only_b": len(set(pb) - set(pa)), "a_mean": _ci([pa[t][0] for t in common]),
            "b_mean": _ci([pb[t][0] for t in common]), "diff": _ci(diff.values()),
            "wins": sum(1 for v in diff.values() if v > 1e-9), "ties": sum(1 for v in diff.values() if abs(v) <= 1e-9),
            "losses": sum(1 for v in diff.values() if v < -1e-9),
            "by_role": {r: _ci([diff[t] for t in common if pa[t][1] == r]) for r in roles}}


def compare_text(c: dict) -> str:
    f = lambda x: "-" if not x else (f"{x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] (n={x['n']})"
                                     if x.get("lo") is not None else f"{x['mean']:+.3f} (n={x['n']})")
    L = [f"paired: {c['b']} - {c['a']} on {c['tasks']} common tasks ({c['only_a']} only in a, {c['only_b']} only in b)",
         f"  a {f(c['a_mean'])}", f"  b {f(c['b_mean'])}",
         f"  b - a {f(c['diff'])}; b better in {c['wins']}, equal in {c['ties']}, worse in {c['losses']}"]
    L += [f"  {r}: {f(v)}" for r, v in c["by_role"].items()]
    return "\n".join(L)
