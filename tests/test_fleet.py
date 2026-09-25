"""Agent fleets: tool-using manager + departments."""
from civitas.core.actions import apply_action
from civitas.core.decision import Decision
from civitas.core.env import CivitasEnv, GameConfig


def _env(**kw):
    return CivitasEnv(GameConfig(seed=0, months=12, **kw))


def test_env_apply_logs_already_executed_results():
    env = _env()
    env.world.treasury = 50000
    res = apply_action(env.world, "mayor", {"type": "allocate_budget", "department": "health", "amount": 1000}).to_dict()
    bad = apply_action(env.world, "mayor", {"type": "set_tax", "sector": "Q", "rate": 9}).to_dict()
    out = env.apply("mayor", Decision(executed=[res, bad]))
    assert out == [res, bad]
    assert env.invalid_counts["mayor"] == 1
    assert env.turns[-1]["results"] == [res, bad]
    assert env.world.dept_funds["health"] == 1000          # not applied a second time


def test_agents_get_bound_to_the_env():
    from civitas.agents.base import Agent
    seen = {}

    class Probe(Agent):
        def bind(self, env):
            seen["env"] = env
        def act(self, obs):
            return Decision()
    env = _env()
    Probe("mayor").bind(env)
    assert seen["env"] is env
