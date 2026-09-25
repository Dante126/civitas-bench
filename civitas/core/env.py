"""CivitasEnv: the central hub every agent connects to.

It owns the World (single source of truth), the council board, each role's
memory (last results, notes), scores and rewards. Two ways to drive it:

* Sequential (used by the runner / benchmark): for each role in turn order
  call `observe(role)` then `apply(role, decision)`; then `end_month()`.
  Later roles see the effects of earlier roles within the same month.
* Parallel, PettingZoo-style: `reset()` then `step({role: decision})`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..bench.scoring import ROLE_SCORERS, ScoringConfig, all_scores
from ..sim import catalog as C
from ..sim.world import World
from .actions import apply_action
from .council import Council
from .decision import Decision
from .observation import build_observation, render_text
from .roles import ROLES, TURN_ORDER


@dataclass
class GameConfig:
    seed: int = 0
    width: int = 48
    height: int = 48
    months: int = 120
    start_funds: float = 30000.0
    river_width: int = 3
    act_every: int = 1
    max_actions_per_turn: int = 8
    roles: list = field(default_factory=lambda: list(TURN_ORDER))
    map_for_roles: list = field(default_factory=lambda: ["planning", "utilities", "transport", "police", "fire",
                                                          "health", "education"])
    site_hints: bool = True
    council_window: int = 3          # months of council history shown
    team_weight: float = 0.5         # reward = w * city delta + (1-w) * role delta
    invalid_action_penalty: float = 0.2
    balance: dict = field(default_factory=dict)
    events: object = "auto"          # "auto" (seeded schedule), "none", or a list of event dicts
    milestones: bool = True          # buildings unlock as the population grows
    simultaneous: bool = False       # all roles see the same state before anyone acts
    region: bool = False             # several cities, each with its own team (see core/region.py)
    cities: list = field(default_factory=lambda: ["north", "south"])   # region mode: which cities
    seasons: bool = True             # winter heating, summer cooling, snow, seasonal tourism


class CivitasEnv:
    def __init__(self, game: GameConfig | None = None, scoring: ScoringConfig | None = None, *,
                 council: Council | None = None, city: str | None = None, river_start: float | None = None):
        self.city = city                     # set in region mode: roles post as "<city>.<role>"
        self._shared_council = council
        self._river_start = river_start
        self.cfg = game or GameConfig()
        self.scoring = scoring or ScoringConfig(start_funds=self.cfg.start_funds)
        C.apply_overrides(self.cfg.balance)
        for r in self.cfg.roles:
            if r not in ROLES:
                raise ValueError(f"unknown role {r!r}; known: {list(ROLES)}")
        self.roles = [r for r in self.cfg.roles]   # turn order = order of cfg.roles
        self.reset(self.cfg.seed)

    # --------------------------------------------------------------- setup
    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.cfg.seed = seed
        c = self.cfg
        self.world = World(c.width, c.height, c.seed, c.start_funds, c.river_width, months=c.months,
                           events=c.events if c.events != "none" else None, milestones=c.milestones,
                           river_start=self._river_start, seasons=c.seasons)
        self.council = self._shared_council if self._shared_council is not None else Council()
        self.memory = {r: {"last_results": [], "notes": ""} for r in self.roles}
        self.turns: list[dict] = []           # turns taken in the month in progress
        self.last_scores = {k: 0.0 for k in ["city", *self.roles]}
        self.invalid_counts = {r: 0 for r in self.roles}
        self.done = False
        return {r: self.observe(r) for r in self.roles}

    @property
    def month(self) -> int:
        return self.world.month

    def acts_this_month(self, role: str) -> bool:
        return self.world.month % max(1, self.cfg.act_every) == 0

    # ------------------------------------------------------------- observe
    def _name(self, role: str) -> str:
        return f"{self.city}.{role}" if self.city else role

    def _local(self, m: dict) -> dict:
        """Region mode: show my own team's roles without the city prefix."""
        if not self.city:
            return m
        pre = self.city + "."
        return {**m, "sender": m["sender"][len(pre):] if m["sender"].startswith(pre) else m["sender"],
                "to": m["to"][len(pre):] if m["to"].startswith(pre) else m["to"]}

    def observe(self, role: str) -> dict:
        w = self.world
        msgs = [self._local(m) for m in self.council.visible_to(self._name(role),
                                                                 since_month=w.month - self.cfg.council_window)]
        obs = build_observation(
            w, role, council_msgs=msgs, memory=self.memory[role], scores=self.last_scores,
            months_total=self.cfg.months, include_map=role in self.cfg.map_for_roles,
            site_hints=self.cfg.site_hints)
        obs["max_actions"] = self.cfg.max_actions_per_turn
        obs["text"] = render_text(obs)
        return obs

    # --------------------------------------------------------------- act
    def apply(self, role: str, decision: Decision) -> list[dict]:
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        w = self.world
        results = list(decision.executed)
        actions = decision.actions[: self.cfg.max_actions_per_turn]
        for a in actions:
            results.append(apply_action(w, role, a).to_dict())
        if len(decision.actions) > len(actions):
            results.append({"action": {"type": "(truncated)"}, "ok": False, "cost": 0,
                            "message": f"only the first {self.cfg.max_actions_per_turn} actions were executed"})
        for msg in decision.messages:
            self.council.post(w.month, self._name(role), msg.get("text", ""), msg.get("to", "all"))
        if decision.notes is not None:
            self.memory[role]["notes"] = str(decision.notes)[:1500]
        self.memory[role]["last_results"] = results
        n_bad = sum(1 for r in results if not r["ok"])
        self.invalid_counts[role] += n_bad
        self.turns.append({
            "month": w.month, "role": role, "thinking": decision.thinking,
            "results": results, "messages": decision.messages,
            "invalid": n_bad, "parse_error": decision.parse_error, "meta": decision.meta,
        })
        return results

    # ------------------------------------------------------------ advance
    def end_month(self) -> dict:
        """Simulate the month, compute scores and per-role rewards."""
        self.world.tick()
        scores = all_scores(self.world.history, self.scoring, self.roles)
        d_city = (scores["city"] - self.last_scores["city"]) / 100.0
        rewards = {}
        tw = self.cfg.team_weight
        for r in self.roles:
            d_role = (scores[r] - self.last_scores[r]) / 100.0
            bad = sum(t["invalid"] for t in self.turns if t["role"] == r)
            rewards[r] = round(tw * d_city + (1 - tw) * d_role - self.cfg.invalid_action_penalty * 0.01 * bad, 6)
        self.last_scores = scores
        terminated = self.world.bankrupt or self.world.recalled
        truncated = self.world.month >= self.cfg.months
        self.done = terminated or truncated
        info = {"metrics": self.world.history[-1], "scores": scores, "rewards": rewards,
                "terminated": terminated, "truncated": truncated, "turns": self.turns}
        self.turns = []
        return info

    # --------------------------------------------------- parallel API (RL)
    def step(self, decisions: dict[str, Decision]):
        for r in self.roles:
            if r in decisions and self.acts_this_month(r):
                self.apply(r, decisions[r])
        info = self.end_month()
        obs = {r: self.observe(r) for r in self.roles}
        rew = info["rewards"]
        term = {r: info["terminated"] for r in self.roles}
        trunc = {r: info["truncated"] for r in self.roles}
        return obs, rew, term, trunc, {r: {"scores": info["scores"]} for r in self.roles}

    def tensor_observation(self) -> dict:
        """Numeric observation for neural RL policies: (channels, H, W) + vector."""
        w = self.world
        s = w.struct
        chans = [w.terrain == C.WATER, s == C.ROAD, s == C.RES, s == C.COM, s == C.IND, s == C.PARK,
                 np.isin(s, list(C.POWER_PLANTS.values())), np.isin(s, list(C.WATER_PLANTS.values())),
                 w.level / 3.0, w.pollution, w.land_value, w.powered, w.watered, w.access]
        grid = np.stack([np.asarray(c, dtype=np.float32) for c in chans])
        m = w.history[-1] if w.history else w.metrics()
        vec = np.array([
            m["population"] / 30000, w.treasury / 1e5, w.debt / 1e5, m["happiness"], m["unemployment"],
            *[w.demand[k] for k in "RCIO"], *[w.taxes[k] for k in "RCI"],
            *[w.dept_funds[d] / 1e5 for d in ("planning", "utilities")],
            (m["power_coverage"] or 0), (m["water_coverage"] or 0), w.month / self.cfg.months,
        ], dtype=np.float32)
        return {"grid": grid, "vector": vec}

    # -------------------------------------------------------------- replay
    def frame(self, turns: list[dict] | None = None, scores: dict | None = None) -> dict:
        w = self.world
        q = lambda a: "".join(str(int(v)) for v in np.clip(a * 9.999, 0, 9).astype(int).ravel())
        return {
            "month": w.month,
            "tiles": w.tiles_string(),
            "levels": "".join(str(int(v)) for v in w.frame_levels().ravel()),
            "variant": "".join(str(int(v)) for v in np.where(np.isin(w.struct, C.HOMES), w.wealth, np.where(w.struct == C.IND, w.ind_type, 0)).ravel()),
            "density": "".join(str(int(v)) for v in w.density.ravel()),
            "pollution": q(w.pollution),
            "land_value": q(w.land_value),
            "power": "".join("1" if v else "0" for v in w.powered.ravel()),
            "water": "".join("1" if v else "0" for v in w.watered.ravel()),
            "sewage": "".join("1" if v else "0" for v in w.sewered.ravel()),
            "layers": {k: q(g) for k, g in w.frame_layers().items()},
            "road_class": "".join(str(int(v)) for v in w.road_class.ravel()),
            "bus_lines": [{"id": l["id"], "stops": [list(p) for p in l["stops"]]} for l in w.bus_lines],
            "metro_lines": [{"id": l["id"], "stations": [list(p) for p in l["stations"]]} for l in w.metro_lines],
            "burning": list(w.burning),
            "metrics": w.history[-1] if w.history else w.metrics(),
            "scores": scores or self.last_scores,
            "events": list(w.last_events),
            "turns": [{k: v for k, v in t.items() if k != "meta"} for t in (turns or [])],
            "flooded": list(getattr(w, "flooded", [])),
            "season": w.season() if w.seasons_on else None, "wind": [round(v, 2) for v in w.wind_dir()],
            "multi": w.multi_buildings(),
            "districts": "".join(str(int(v)) for v in w.district_id.ravel()) if w.districts else "",
            "district_names": {str(d["id"]): n for n, d in w.districts.items()},
            "council": [m for m in self.council.posted_in(w.month - 1)
                        if not self.city or m["sender"].startswith(self.city + ".")] if w.month else [],
        }

    def meta(self) -> dict:
        """Static facts about the map for the viewer."""
        return {"frame_format": C.FRAME_FORMAT, "roles": self.roles,
                "resources": "".join(str(int(v)) for v in self.world.resource.ravel())}

    def config_dict(self) -> dict:
        return {"game": asdict(self.cfg), "scoring": asdict(self.scoring)}

    def final_metrics(self) -> dict:
        return self.world.history[-1] if self.world.history else self.world.metrics()


def make_env(game: GameConfig, scoring: ScoringConfig | None = None):
    """A single city, or two cities on one river when game.region is set."""
    if game.region:
        from .region import RegionEnv
        return RegionEnv(game, scoring)
    return CivitasEnv(game, scoring)
