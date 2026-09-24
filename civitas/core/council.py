"""The council board: a shared message board every agent can read and post to."""
from __future__ import annotations

from dataclasses import asdict, dataclass

MAX_LEN = 500


@dataclass
class Message:
    month: int
    sender: str
    to: str       # "all" or a role name
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


class Council:
    def __init__(self) -> None:
        self.messages: list[Message] = []

    def post(self, month: int, sender: str, text: str, to: str = "all") -> Message:
        msg = Message(month, sender, to or "all", str(text)[:MAX_LEN])
        self.messages.append(msg)
        return msg

    def visible_to(self, role: str, since_month: int) -> list[dict]:
        """Messages to everyone, to this role, from this role, or (region mode) to its whole team."""
        team = role.split(".")[0] if "." in role else None
        return [m.to_dict() for m in self.messages
                if m.month >= since_month and (m.to in ("all", role) or m.sender == role or (team and m.to == team))]

    def posted_in(self, month: int) -> list[dict]:
        return [m.to_dict() for m in self.messages if m.month == month]
