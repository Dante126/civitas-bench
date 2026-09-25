"""Fleet config: per-agent coaching (playbooks), model and limits.

A run config's `[fleet]` table holds fleet-wide settings (`playbook` is the fleet playbook) and every
`[agents.<role>]` with `kind = "fleet"` can override any of them (its `playbook` is the role's playbook).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields

MODES = ("autonomous", "supervised")


@dataclass
class FleetAgentConfig:
    mode: str = "autonomous"
    fleet_playbook: str = ""
    playbook: str = ""
    provider: str = "arcee"
    model: str = "trinity-large-thinking"
    max_steps: int = 20              # tool calls per turn
    max_tokens: int = 16000          # per model call
    turn_tokens: int = 80000         # per turn
    temperature: float = 0.4
    extra_body: dict = field(default_factory=dict)


def _role_spec(cfg: dict, role: str) -> dict:
    agents = cfg.get("agents") or {}
    return dict(agents.get(role) or agents.get(role.split(".", 1)[-1]) or {})


def fleet_agent_config(cfg: dict, role: str) -> FleetAgentConfig:
    """`[fleet]` merged with `[agents.<role>]`; unknown keys are dropped like everywhere else in the config."""
    fleet = dict(cfg.get("fleet") or {})
    merged = {"fleet_playbook": fleet.pop("playbook", ""), **fleet, **_role_spec(cfg, role)}
    known = {f.name: f for f in fields(FleetAgentConfig)}
    out = FleetAgentConfig(**{k: v for k, v in merged.items() if k in known})
    for k in ("max_steps", "max_tokens", "turn_tokens"):
        setattr(out, k, int(getattr(out, k)))
    out.temperature = float(out.temperature)
    out.extra_body = dict(out.extra_body or {})
    if out.mode not in MODES:
        raise ValueError(f"agents.{role}: fleet mode must be one of {', '.join(MODES)}, got {out.mode!r}")
    return out


def fleet_agents(cfg: dict) -> dict:
    return {r: a for r, a in (cfg.get("agents") or {}).items() if a.get("kind") == "fleet"}


def fleet_version(cfg: dict) -> str:
    """Short hash of the fleet's coaching and limits: `[fleet]` plus the fleet agents' specs."""
    obj = {"fleet": cfg.get("fleet") or {}, "agents": fleet_agents(cfg)}
    return hashlib.sha1(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
