"""Remote agents: any external process (any language) plays a role over HTTP.

The game loop blocks on `RemoteAgent.act` until the external agent fetches
its observation (GET /api/live/turn?role=...) and posts a decision
(POST /api/live/act). See examples/remote_agent.py.
"""
from __future__ import annotations

import threading
import uuid

from ..core.decision import Decision
from .base import Agent


class LiveHub:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pending: dict[str, dict] = {}
        self.latest_frame: dict | None = None
        self.run_id: str | None = None

    def request(self, role: str, obs: dict, timeout: float) -> Decision | None:
        turn = {"turn_id": uuid.uuid4().hex[:12], "observation": obs,
                "event": threading.Event(), "decision": None}
        with self.lock:
            self.pending[role] = turn
        got = turn["event"].wait(timeout)
        with self.lock:
            self.pending.pop(role, None)
        return turn["decision"] if got else None

    def get_turn(self, role: str) -> dict:
        with self.lock:
            t = self.pending.get(role)
            if not t:
                return {"waiting": False, "role": role}
            return {"waiting": True, "role": role, "turn_id": t["turn_id"], "observation": t["observation"]}

    def submit(self, role: str, turn_id: str, decision: dict) -> tuple[bool, str]:
        with self.lock:
            t = self.pending.get(role)
            if not t:
                return False, f"it is not {role}'s turn"
            if t["turn_id"] != turn_id:
                return False, "stale turn_id"
            t["decision"] = Decision.from_dict(decision)
            t["event"].set()
        return True, "accepted"


class RemoteAgent(Agent):
    kind = "remote"

    def __init__(self, role: str, hub: LiveHub | None = None, timeout: float = 600.0, **kw):
        super().__init__(role, **kw)
        if hub is None:
            raise ValueError("remote agents need the live server: run with --serve")
        self.hub = hub
        self.timeout = timeout

    def act(self, obs: dict) -> Decision:
        d = self.hub.request(self.role, obs, self.timeout)
        if d is None:
            return Decision(thinking="(remote agent timed out)", parse_error="timeout")
        return d
