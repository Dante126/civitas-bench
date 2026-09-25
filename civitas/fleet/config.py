"""Fleet config: per-agent coaching (playbooks), model and limits."""
from __future__ import annotations

from dataclasses import dataclass, field


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
