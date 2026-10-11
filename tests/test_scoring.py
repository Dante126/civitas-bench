"""bench/scoring.py: scores as pure functions of the monthly metrics history."""
import copy

import pytest

from civitas.agents.scripted import SCRIPTED
from civitas.bench.scoring import ScoringConfig, _deployment, mayor_score
from civitas.core.env import CivitasEnv, GameConfig


@pytest.fixture(scope="module")
def history():
    env = CivitasEnv(GameConfig(seed=0, months=24))
    agents = {r: SCRIPTED[r](r) for r in env.roles}
    for _ in range(24):
        for r in env.roles:
            env.apply(r, agents[r].act(env.observe(r)))
        env.end_month()
    return env.world.history


def _with_funds(history, per_month):
    """The history with each of the last 12 months' department funds replaced by per_month[i] (all in planning)."""
    h = copy.deepcopy(history)
    for m, v in zip(h[-12:], per_month):
        m["dept_funds"] = {d: 0.0 for d in m["dept_funds"]} | {"planning": float(v)}
    return h


def test_deployment_averages_idle_money_over_the_window(history):
    """civitas-v2: the mayor's deployment term weighs the department money held idle over the whole scoring window,
    not the final month only (v1's scripted mayor held ~$30k all year; taking it back in the last month would have
    counted as never holding it)."""
    cfg = ScoringConfig()
    spent = sum(sum(m["dept_spent"].values()) for m in history[-12:])
    assert spent > 0
    held_all_year = _with_funds(history, [12000] * 12)
    dropped_at_the_end = _with_funds(history, [12000] * 11 + [0])
    never = _with_funds(history, [0] * 12)
    assert _deployment(held_all_year[-12:]) == pytest.approx(spent / (spent + 12000))
    assert _deployment(dropped_at_the_end[-12:]) == pytest.approx(spent / (spent + 11000))
    assert _deployment(never[-12:]) == 1.0
    s = [mayor_score(h, cfg) for h in (held_all_year, dropped_at_the_end, never)]
    assert s[0] < s[1] < s[2] and s[2] - s[1] > 1.0                 # the last month alone no longer clears it
    same_mean = _with_funds(history, [24000, 0] * 6)                 # only the average counts
    assert mayor_score(same_mean, cfg) == pytest.approx(s[0])


def test_deployment_reads_histories_without_department_records():
    assert _deployment([{"population": 1}]) == 0.0
    assert _deployment([{"dept_spent": {"fire": 100.0}, "dept_funds": {"fire": 100.0}}] * 3) == pytest.approx(0.75)
