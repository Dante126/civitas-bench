"""Agent interface. Any agent maps an observation dict to a Decision."""
from __future__ import annotations

import random

from ..core.decision import Decision
from ..core.roles import ROLES


class Agent:
    kind = "base"

    def __init__(self, role: str, **kwargs):
        self.role = role
        self.options = kwargs

    def reset(self, seed: int = 0) -> None:
        pass

    def bind(self, env) -> None:
        """Called once by the runner before the episode (tool agents act on the env directly)."""

    def act(self, obs: dict) -> Decision:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> str:
        return self.kind


class NoopAgent(Agent):
    """Does nothing. Floor baseline."""
    kind = "noop"

    def act(self, obs: dict) -> Decision:
        return Decision(actions=[], thinking="noop baseline")


class RandomAgent(Agent):
    """Uniformly random (mostly invalid) actions. Sanity baseline."""
    kind = "random"

    def reset(self, seed: int = 0) -> None:
        self.rng = random.Random(seed * 7919 + hash(self.role) % 1000)

    def act(self, obs: dict) -> Decision:
        if not hasattr(self, "rng"):
            self.reset()
        r = self.rng
        W = H = 48
        if "map" in obs:
            lines = obs["map"].splitlines()[2:]
            H, W = len(lines), len(lines[0]) - 4
        acts = []
        for _ in range(r.randint(0, 3)):
            t = r.choice(ROLES[self.role]["actions"])
            x, y = r.randrange(W), r.randrange(H)
            a = {"type": t, "x": x, "y": y, "x1": x, "y1": y,
                 "x2": min(W - 1, x + r.randint(0, 6)), "y2": min(H - 1, y + r.randint(0, 2))}
            if t == "zone":
                a["zone"] = r.choice("RCIO")
                a["density"] = r.choice(["low", "high"])
            elif t == "build_power_plant":
                a["kind"] = r.choice(["coal", "gas", "solar", "wind"])
            elif t == "build_water":
                a["kind"] = r.choice(["pump", "tower"])
            elif t == "build_school":
                a["kind"] = r.choice(["elementary", "high", "university"])
            elif t == "set_tax":
                a.update(sector=r.choice("RCI"), rate=r.randint(5, 14))
            elif t == "allocate_budget":
                a.update(department=r.choice(["planning", "utilities", "transport", "police", "fire", "health",
                                              "education"]),
                         amount=r.randint(500, 5000))
            elif t in ("take_loan", "repay_loan"):
                a["amount"] = r.choice([5000, 10000])
            acts.append(a)
        return Decision(actions=acts, thinking="random baseline")
