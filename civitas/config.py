"""Load TOML run configs and build agents from them."""
from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from .agents.base import NoopAgent, RandomAgent
from .agents.llm import LLMAgent
from .agents.llm_client import ProviderConfig
from .agents.remote import RemoteAgent
from .agents.scripted import SCRIPTED
from .bench.scoring import ScoringConfig
from .core.env import GameConfig
from .fleet.agent import ToolAgent
from .fleet.config import fleet_agent_config

DEFAULT_PROVIDERS = {
    "arcee": {"base_url": "https://api.arcee.ai/api/v1", "api_key_env": "ARCEE_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key_env": ""},
}


def load_config(path: str | Path | None) -> dict:
    cfg: dict = {"name": "default", "game": {}, "scoring": {}, "providers": {}, "agents": {}}
    if path:
        with open(path, "rb") as f:
            loaded = tomllib.load(f)
        for k, v in loaded.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
        cfg.setdefault("name", Path(path).stem)
        if cfg["name"] == "default":
            cfg["name"] = Path(path).stem
    return cfg


def _coerce(v: str):
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def apply_cli_overrides(cfg: dict, agents: list[str] | None = None, sets: list[str] | None = None) -> dict:
    """--agent planning=llm:arcee/trinity-mini   --set game.months=24"""
    cfg = copy.deepcopy(cfg)
    for spec in agents or []:
        role, _, rhs = spec.partition("=")
        kind, _, model = rhs.partition(":")
        entry = {"kind": kind}
        if model:
            prov, _, m = model.partition("/")
            if m:
                entry.update(provider=prov, model=m)
            else:
                entry.update(provider="arcee", model=prov)
        base = cfg["agents"].get(role, {}) if kind == cfg["agents"].get(role, {}).get("kind") else {}
        cfg["agents"][role] = {**base, **entry}
    for s in sets or []:
        key, _, val = s.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
    return cfg


def game_config(cfg: dict, seed: int | None = None) -> GameConfig:
    fields = {f.name for f in dataclasses.fields(GameConfig)}
    g = GameConfig(**{k: v for k, v in cfg["game"].items() if k in fields})
    if seed is not None:
        g.seed = seed
    return g


def scoring_config(cfg: dict, game: GameConfig) -> ScoringConfig:
    fields = {f.name for f in dataclasses.fields(ScoringConfig)}
    s = ScoringConfig(**{k: v for k, v in cfg["scoring"].items() if k in fields})
    s.start_funds = game.start_funds
    return s


def provider_config(cfg: dict, name: str) -> ProviderConfig:
    spec = {**DEFAULT_PROVIDERS.get(name, {}), **cfg.get("providers", {}).get(name, {})}
    if not spec:
        raise ValueError(f"unknown provider {name!r}; define [providers.{name}] in the config")
    fields = {f.name for f in dataclasses.fields(ProviderConfig)}
    return ProviderConfig(**{k: v for k, v in spec.items() if k in fields})


def agent_spec(cfg: dict, role: str) -> dict:
    """Agent settings for a role; in region mode "north.planning" falls back to "planning"."""
    agents = cfg["agents"]
    key = role if role in agents else role.split(".", 1)[-1]
    spec = agents.get(key, {"kind": "scripted"})
    if not isinstance(spec, dict):
        raise ValueError(f"[agents.{key}] must be a table such as {{kind = \"scripted\"}}, got {spec!r}")
    return dict(spec)


def build_agents(cfg: dict, roles: list[str], game: GameConfig, scoring: ScoringConfig, hub=None) -> dict:
    agents = {}
    modes = set()
    for role in roles:
        spec = agent_spec(cfg, role)
        base = role.split(".", 1)[-1]
        city = role.split(".", 1)[0] if "." in role else None
        kind = spec.pop("kind", "scripted")
        if kind == "scripted":
            agents[role] = SCRIPTED[base](base)
        elif kind == "noop":
            agents[role] = NoopAgent(base)
        elif kind == "random":
            agents[role] = RandomAgent(base)
        elif kind == "remote":
            agents[role] = RemoteAgent(role, hub=hub, timeout=float(spec.get("timeout", 600)))
        elif kind == "llm":
            prov = provider_config(cfg, spec.pop("provider", "arcee"))
            if "model" not in spec:
                raise ValueError(f"agents.{role}: llm agents need a 'model'")
            agents[role] = LLMAgent(
                base, model=spec.pop("model"), provider=prov, city=city,
                temperature=float(spec.pop("temperature", 0.4)),
                max_tokens=int(spec.pop("max_tokens", 12000)),
                history_turns=int(spec.pop("history_turns", 0)),
                extra_body=spec.pop("extra_body", None),
                months=game.months, target_pop=scoring.target_population,
                max_actions=game.max_actions_per_turn)
        elif kind == "fleet":
            if city:
                raise ValueError("fleet agents support single-city games only")
            fc = fleet_agent_config(cfg, role)
            modes.add(fc.mode)
            agents[role] = ToolAgent(base, cfg=fc, provider=provider_config(cfg, fc.provider))
        else:
            raise ValueError(f"agents.{role}: unknown kind {kind!r} (scripted|llm|fleet|remote|noop|random)")
    if len(modes) > 1:
        raise ValueError(f"fleet agents must share one mode, got {', '.join(sorted(modes))}")
    return agents
