"""Fleet config: per-agent coaching (playbooks), model and limits.

A run config's `[fleet]` table holds fleet-wide settings (`playbook` is the fleet playbook) and every
`[agents.<role>]` with `kind = "fleet"` can override any of them (its `playbook` is the role's playbook; its
`extra_body` replaces `[fleet].extra_body` as a whole; `temperature = "default"` sends no temperature, the
provider's own, which OpenAI's reasoning models require). `[fleet].session_months` is run-wide: a supervised run
pauses after that many months (see `bench/session.py`). `briefing` ("guided", the default, or "open") is the
briefing style (fmt.py): open briefings and tool results state facts and options, never an exact call or a chosen
action. `turn_budget` (tokens, prompt as characters / 4 + completion over a turn's model calls; 0 = off) is
civitas-v3's equal token budget per turn (fleet/budget.py); `budget_reserve` is the least it keeps for the next reply.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields

from ..agents.llm_client import temperature_of
from .fmt import STYLES

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
    turn_tokens: int = 80000         # completion tokens per turn
    turn_budget: int = 0             # prompt (characters / 4) + completion tokens per turn (0 = off: civitas-v2);
                                     # civitas-v3: 100000 (fleet/budget.py)
    budget_reserve: int = 4000       # the least completion the budget keeps for the next reply (fleet/budget.py:
                                     # max(this, the turn's largest completion))
    temperature: float | None = 0.4  # None (`temperature = "default"`): the request carries none
    extra_body: dict = field(default_factory=dict)
    briefing: str = "guided"         # "guided" names the build and the call; "open" only the facts (fmt.STYLES)


def _table(v, name: str) -> dict:
    if not isinstance(v, dict):
        raise ValueError(f"[{name}] must be a table of settings, got {type(v).__name__} {v!r}")
    return v


def _role_spec(cfg: dict, role: str) -> dict:
    agents = cfg.get("agents") or {}
    key = role if role in agents else role.split(".", 1)[-1]
    return dict(_table(agents[key], f"agents.{key}")) if key in agents else {}


def fleet_agent_config(cfg: dict, role: str) -> FleetAgentConfig:
    """`[fleet]` merged with `[agents.<role>]`; unknown keys are dropped like everywhere else in the config."""
    fleet = dict(_table(cfg.get("fleet") or {}, "fleet"))
    merged = {"fleet_playbook": fleet.pop("playbook", ""), **fleet, **_role_spec(cfg, role)}
    known = {f.name: f for f in fields(FleetAgentConfig)}
    out = FleetAgentConfig(**{k: v for k, v in merged.items() if k in known})
    for k in ("max_steps", "max_tokens", "turn_tokens"):
        setattr(out, k, int(getattr(out, k)))
        if getattr(out, k) < 1:
            raise ValueError(f"agents.{role}: fleet {k} must be at least 1, got {getattr(out, k)}")
    for k in ("turn_budget", "budget_reserve"):
        v = getattr(out, k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or v != int(v):
            raise ValueError(f"agents.{role}: fleet {k} must be a whole number of tokens (0 or more), got {v!r}")
        setattr(out, k, int(v))
    try:
        out.temperature = temperature_of(out.temperature)
    except ValueError as e:
        raise ValueError(f"agents.{role}: fleet {e}") from None
    out.extra_body = dict(out.extra_body or {})
    if out.mode not in MODES:
        raise ValueError(f"agents.{role}: fleet mode must be one of {', '.join(MODES)}, got {out.mode!r}")
    if out.briefing not in STYLES:
        raise ValueError(f"agents.{role}: fleet briefing must be one of {', '.join(STYLES)}, got {out.briefing!r}")
    return out


def fleet_agents(cfg: dict) -> dict:
    return {r: a for r, a in (cfg.get("agents") or {}).items() if _table(a, f"agents.{r}").get("kind") == "fleet"}


SCHEDULING = ("session_months",)     # [fleet] keys about when a run pauses, not how the fleet plays


def _sha(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


def fleet_version(cfg: dict) -> str:
    """Short hash of the fleet's coaching and limits: `[fleet]` plus the fleet agents' specs. Scheduling keys are
    left out, so a supervised run that pauses between sessions keeps the version of one that plays through."""
    fleet = {k: v for k, v in _table(cfg.get("fleet") or {}, "fleet").items() if k not in SCHEDULING}
    return _sha({"fleet": fleet, "agents": fleet_agents(cfg)})


def session_months(cfg: dict) -> int:
    """`[fleet].session_months`: months per session of a supervised run (0 = no pauses, the default)."""
    v = _table(cfg.get("fleet") or {}, "fleet").get("session_months", 0)
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError(f"fleet.session_months must be a whole number of months (0 = no pauses), got {v!r}")
    return v


def playbook_versions(cfg: dict, roles) -> dict:
    """Short hashes of the fleet playbook and each fleet role's own playbook (None when empty)."""
    h = lambda text: _sha(text) if str(text or "").strip() else None
    out = {"fleet": h(_table(cfg.get("fleet") or {}, "fleet").get("playbook", ""))}
    for r in roles:
        out[r] = h(fleet_agent_config(cfg, r).playbook)
    return out
