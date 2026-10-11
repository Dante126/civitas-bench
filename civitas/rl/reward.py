"""The counterfactual reward of one task turn.

From the task's state (bank.py) two branches go on: the baseline, where the scripted fleet agent plays the role's
turn, and the model's, where the model played it (episode.py). In both the rest of that month and the next H months
are played by the scripted fleet, every role (the trained role too, after its one turn), exactly as the warm-up plays
them. The baseline branch is the scripted game itself (bank.settle changes nothing the scripted agents read), so the
bank caches its scores per task and a reward needs one replay: the model's branch.

    reward = (city_model - city_baseline) + w_role * (role_model - role_baseline)
             - p_failed * failed actions - p_refused * guard refusals - p_invalid * invalid calls
             - p_unended * (turn ended without end_turn)

with scores (0-100) read H months after the task's month (`compute`; components reported apart). The scripted
policy replayed through the baseline path scores exactly 0 (`civitas rl check`, policy "baseline"). Defaults
(`RewardConfig`): H = 6, w_role = 0.25, p_failed = 0.05, p_refused = 0.02, p_invalid = 0.02, p_unended = 0.1;
README "RL tasks" has the measurements behind them.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

VERSION = 1


@dataclass
class RewardConfig:
    horizon: int = 6             # months after the task's month the scores are read (6 = two turns at act_every 3)
    w_role: float = 0.25         # weight of the role's own department score difference
    p_failed: float = 0.05       # per action that ran and failed (in score points)
    p_refused: float = 0.02      # per guard refusal (disconnect, duplicate, take_back, reserve, loan, repeat, ...)
    p_invalid: float = 0.02      # per call with an unknown tool, bad arguments, or past the action limit
    p_unended: float = 0.1       # the turn ended without a successful end_turn (step limit, no reply)

    def as_dict(self) -> dict:
        return asdict(self)


def play_on(env, agents: dict, role: str, months: int) -> dict:
    """After `role`'s turn in the current (acting) month: the roles after it play theirs, then every role plays (the
    scripted agents in `agents`) until `months` months have ended. Per month after the task's: the city score, the
    role's score, the population and the seconds since the call. A game that ended (bankruptcy, recall) repeats its
    last month."""
    m0, t0 = env.world.month, time.perf_counter()
    out = {"months": [], "city": [], "role": [], "population": [], "seconds": []}
    after = env.roles[env.roles.index(role) + 1:]
    first = True
    while not env.done and env.world.month < m0 + months:
        for r in (after if first else env.roles):
            if env.acts_this_month(r):
                env.apply(r, agents[r].act(env.observe(r)))
        first = False
        info = env.end_month()
        out["months"].append(env.world.month)
        out["city"].append(round(float(info["scores"]["city"]), 6))
        out["role"].append(round(float(info["scores"][role]), 6))
        out["population"].append(int(info["metrics"]["population"]))
        out["seconds"].append(round(time.perf_counter() - t0, 4))
    while out["months"] and len(out["months"]) < months:
        for k in out:
            out[k].append(out[k][-1] + (1 if k == "months" else 0))
    return out


def counts(executed: list[dict], trace: list[dict], ended: bool) -> dict:
    """What a turn's calls did: successful and failed actions (`executed`: results of actions that ran), guard
    refusals and invalid calls (from the tool trace's slim results)."""
    ok = sum(1 for r in executed if r.get("ok"))
    failed = len(executed) - ok
    refused = sum(1 for t in trace if "refused" in (t.get("result") or {}))
    bad = sum(1 for t in trace if (t.get("result") or {}).get("ok") is False)
    return {"calls": len(trace), "actions": ok, "failed": failed, "refused": refused,
            "invalid": max(0, bad - failed - refused), "ended": bool(ended)}


def compute(row: dict, series: dict, n: dict, cfg: RewardConfig | None = None) -> dict:
    """The reward of a model branch (`series` from play_on, `n` from counts) against the task's cached baseline."""
    cfg = cfg or RewardConfig()
    base = row["baseline"]
    h = max(1, min(int(cfg.horizon), len(series["city"]), len(base["city"])))
    city = series["city"][h - 1] - base["city"][h - 1]
    role = series["role"][h - 1] - base["role"][h - 1]
    penalty = (cfg.p_failed * n["failed"] + cfg.p_refused * n["refused"] + cfg.p_invalid * n["invalid"]
               + (0.0 if n["ended"] else cfg.p_unended))
    return {"reward": round(city + cfg.w_role * role - penalty, 6), "city": round(city, 6), "role": round(role, 6),
            "penalty": round(penalty, 6), "horizon": h,
            "population": series["population"][h - 1] - base["population"][h - 1], **n, "version": VERSION}


def at_horizons(row: dict, series: dict, n: dict, horizons, cfg: RewardConfig | None = None) -> dict:
    """compute() at each horizon (the same branch read at several months)."""
    cfg = cfg or RewardConfig()
    return {h: compute(row, series, n, RewardConfig(**{**cfg.as_dict(), "horizon": h})) for h in horizons}
