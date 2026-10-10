"""Load TOML run configs and build agents from them."""
from __future__ import annotations

import copy
import dataclasses
import json
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
from .fleet.config import FleetAgentConfig, fleet_agent_config
from .fleet.scripted import MODEL as SCRIPTED_FLEET, ScriptedFleetAgent

DEFAULT_PROVIDERS = {
    "arcee": {"base_url": "https://api.arcee.ai/api/v1", "api_key_env": "ARCEE_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key_env": ""},
    "openai": {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY",
               "max_tokens_field": "max_completion_tokens"},
    "anthropic": {"base_url": "https://api.anthropic.com/v1/", "api_key_env": "ANTHROPIC_API_KEY"},  # OpenAI-compatible
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
    if v[:1] in ("[", "{"):                     # --set 'game.cities=["north","south","east"]'
        try:
            return json.loads(v)
        except ValueError:
            pass
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


WARMUP_KINDS = ("fleet", "llm", "remote")      # kinds a warm-up hands to scripted agents (the costly or live ones)
SCRIPTED_STEPS = FleetAgentConfig().max_steps   # the scripted fleet's step limit in a warm-up (fleet_scripted.toml's)


def warmup_months(cfg: dict) -> int:
    """`[game].warmup_months`: the first N months a run plays with the warm-up line-up (`warmup_config`) before the
    configured agents take over the same game (bench/runner.py). 0 = none. ValueError unless 0 <= N < game.months."""
    v = (cfg.get("game") or {}).get("warmup_months", 0)
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError(f"game.warmup_months must be a whole number of months (0 = no warm-up), got {v!r}")
    months = game_config(cfg).months
    if v and v >= months:
        raise ValueError(f"game.warmup_months ({v}) must be less than game.months ({months}): the configured agents "
                         f"play the months after the warm-up (e.g. game.months={v + 12} for 12 of them)")
    return v


def warmup_config(cfg: dict) -> dict:
    """The warm-up line-up: `cfg` with every fleet, llm and remote agent made scripted, the rest as they are. A fleet
    agent in a single city becomes a scripted fleet agent (`model = "scripted"`, playbooks and mode kept), so the plan,
    memory and history log it leaves are the fleet tools' own; its step limit is the scripted fleet's own (`max_steps`
    20, configs/fleet_scripted.toml), whatever the configured model's, so a warm-up plays fleet_scripted's game up to
    the hand-over. In a region (where the scripted fleet doesn't play) and for llm / remote agents a plain scripted
    agent. No `warmup_months` (the warm-up has none of its own)."""
    out = copy.deepcopy(cfg)
    out["game"] = {k: v for k, v in (cfg.get("game") or {}).items() if k != "warmup_months"}
    region = bool(game_config(cfg).region)
    roles = list(game_config(cfg).roles)
    if region:
        roles = [f"{c}.{r}" for r in roles for c in game_config(cfg).cities]
    agents = dict(out.get("agents") or {})
    for role in roles:
        spec = agent_spec(cfg, role)
        kind = spec.get("kind", "scripted")
        if kind not in WARMUP_KINDS:
            continue
        if kind == "fleet" and fleet_agent_config(cfg, role).model == SCRIPTED_FLEET:
            continue                                   # already the scripted fleet
        agents[role] = ({**spec, "model": SCRIPTED_FLEET, "max_steps": SCRIPTED_STEPS}
                        if kind == "fleet" and not region else {"kind": "scripted"})
    out["agents"] = agents
    return out


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
            fc = fleet_agent_config(cfg, role)
            modes.add(fc.mode)
            if fc.model == SCRIPTED_FLEET:                # the scripted baselines through the fleet tools
                agents[role] = ScriptedFleetAgent(base, cfg=fc)
                continue
            agents[role] = ToolAgent(base, cfg=fc, provider=provider_config(cfg, fc.provider), city=city)
        else:
            raise ValueError(f"agents.{role}: unknown kind {kind!r} (scripted|llm|fleet|remote|noop|random)")
    if len(modes) > 1:
        raise ValueError(f"fleet agents must share one mode, got {', '.join(sorted(modes))}")
    return agents
