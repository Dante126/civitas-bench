"""Minimal RL-style loop on the parallel (PettingZoo-like) API.

    obs = env.reset(seed)
    obs, rewards, terminations, truncations, infos = env.step({role: Decision(...)})

Rewards per role = team_weight * delta(city score) + (1 - team_weight) *
delta(role score) - small penalty per invalid action. Set `team_weight`
in [game] to trade off cooperation vs. specialisation.

`env.tensor_observation()` gives a (14, H, W) grid + feature vector for
neural policies; `obs[role]["text"]` is the LLM prompt for text policies.
"""
from civitas.agents.scripted import SCRIPTED
from civitas.core.decision import Decision
from civitas.core.env import CivitasEnv, GameConfig


def main():
    env = CivitasEnv(GameConfig(seed=0, months=36, team_weight=0.5))
    policies = {r: SCRIPTED[r](r) for r in env.roles}   # swap in your learner here
    for p in policies.values():
        p.reset()
    obs = env.reset(seed=0)
    returns = {r: 0.0 for r in env.roles}
    done = False
    while not done:
        actions: dict[str, Decision] = {r: policies[r].act(obs[r]) for r in env.roles}
        obs, rew, term, trunc, info = env.step(actions)
        for r in env.roles:
            returns[r] += rew[r]
        done = any(term.values()) or any(trunc.values())
    t = env.tensor_observation()
    print("returns:", {r: round(v, 3) for r, v in returns.items()})
    print("final scores:", info[env.roles[0]]["scores"])
    print("tensor obs:", t["grid"].shape, t["vector"].shape)


if __name__ == "__main__":
    main()
