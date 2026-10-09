"""The live feed of a running game: what each agent does while its turn is still running, for the viewer's live mode.

`run_episode` attaches a `LiveLog` to the env (and to each city env of a region) and the game reports into it:
- `live.jsonl`: one event per line, `{"i", "t", "month", "role", "kind", ...}` with role the council name
  ("planning", "north.planning"). Kinds: `turn` (a role's turn starts: `agent`), `think` (a model call starts:
  `step`), `reply` (it returned: `step`, `text`, `latency_s`), `call` (a tool call or action: `tool`, `args`, `ok`,
  `action`, and `error` / `message` / `cost` / `refused`), `done` (the turn ended: `actions`, `spent`, `report`),
  `month` (the month ended: `frame`, the index of its frame in frames.jsonl), `frame` (live_frame.json was
  rewritten: `n`).
  The fleet's ToolBox reports its calls as they run (`fleet/tools.py:ToolBox.call`), the ToolAgent its model calls;
  for any other agent the runner reports the turn's actions after `env.apply`.
- `live_frame.json`: the city as it is now, mid-month (`env.frame(live=True)`, with `live` = its sequence number),
  rewritten after successful actions at most every `FRAME_EVERY_S` (and before every model call), so a build shows
  on the map as it happens. Replaced atomically; frames.jsonl keeps one frame per month as before.
The feed only reads the game: a run with it plays exactly as one without (tests/test_live.py)."""
from __future__ import annotations

import json
import os
import time
import weakref
from pathlib import Path

EVENTS, FRAME = "live.jsonl", "live_frame.json"
FRAME_EVERY_S = 0.5
TEXT_CHARS = 400          # a reply's text, a message, a report
ARG_CHARS = 200           # each string argument
NOTE_CHARS = 240          # a result's error / message

_SINKS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()     # env -> (LiveLog, city or None)


def _clip(v, n: int):
    return v if not isinstance(v, str) or len(v) <= n else v[:n - 1] + "…"


def _args(arguments) -> dict | str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            return _clip(arguments, ARG_CHARS)
    if not isinstance(arguments, dict):
        return _clip(str(arguments), ARG_CHARS)
    return {k: _clip(v, TEXT_CHARS if k in ("text", "message", "report") else ARG_CHARS)
            if isinstance(v, str) else v for k, v in arguments.items()}


class LiveLog:
    def __init__(self, run_dir: Path, env, frame_every: float | None = None):
        self.dir, self.env = Path(run_dir), env
        self.frame_every = FRAME_EVERY_S if frame_every is None else frame_every
        path = self.dir / EVENTS
        with open(path, "a+") as fh:                # a resumed run numbers on from its earlier events
            fh.seek(0)
            self.seq = sum(1 for _ in fh)
        self.out = open(path, "a", buffering=1)
        self.frames = 0
        self.dirty = False                   # an action succeeded since the last live frame
        self.last_frame = 0.0
        _SINKS[env] = (self, None)
        for city, e in (getattr(env, "envs", None) or {}).items():
            _SINKS[e] = (self, city)

    def emit(self, kind: str, role: str, **data) -> None:
        if self.out.closed:
            return
        self.seq += 1
        rec = {"i": self.seq, "t": round(time.time(), 2), "month": self.env.world.month if hasattr(self.env, "world")
               else self.env.month, "role": role, "kind": kind, **data}
        self.out.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n")
        if kind == "call" and data.get("action") and data.get("ok"):
            self.dirty = True
        if self.dirty and (kind == "think" or time.time() - self.last_frame >= self.frame_every):
            self.write_frame()               # before a model call (seconds of quiet follow), else throttled

    def write_frame(self) -> None:
        self.dirty, self.last_frame = False, time.time()
        self.frames += 1
        f = self.env.frame(live=True)
        f["live"] = self.frames
        tmp = self.dir / (FRAME + ".tmp")
        tmp.write_text(json.dumps(f, separators=(",", ":"), default=str))
        os.replace(tmp, self.dir / FRAME)
        self.emit("frame", "", n=self.frames)          # the viewer fetches live_frame.json on this

    def month(self, frame_index: int) -> None:
        """The month ended: its frame is in frames.jsonl, so the live frame is stale."""
        self.dirty = False
        (self.dir / FRAME).unlink(missing_ok=True)
        self.emit("month", "", frame=frame_index)

    def close(self) -> None:
        self.out.close()
        (self.dir / FRAME).unlink(missing_ok=True)
        for e in [k for k, (s, _) in list(_SINKS.items()) if s is self]:
            _SINKS.pop(e, None)


def sink(env) -> tuple[LiveLog, str | None] | None:
    try:
        return _SINKS.get(env)
    except TypeError:                        # an unhashable stand-in env (tests)
        return None


def emit(env, role: str, kind: str, **data) -> None:
    """Report one event of `role` (a base role in a city env: named "<city>.<role>") when the env has a feed."""
    s = sink(env)
    if s is None:
        return
    log, city = s
    log.emit(kind, f"{city}.{role}" if city and "." not in role else role, **data)


def _call_data(tool: str, arguments, res: dict, action: bool) -> dict:
    data = {"tool": tool, "args": _args(arguments), "ok": bool(res.get("ok", True)), "action": action}
    for k in ("error", "message"):
        if res.get(k):
            data[k] = _clip(str(res[k]), NOTE_CHARS)
    data.update({k: res[k] for k in ("cost", "refused", "repeat") if res.get(k)})
    return data


def call(env, role: str, tool: str, arguments, res: dict, action: bool) -> None:
    """A fleet tool call's event (ToolBox.call): its arguments and outcome, clipped."""
    if sink(env) is not None:
        emit(env, role, "call", **_call_data(tool, arguments, res, action))


def emit_result(log: LiveLog, role: str, tool: str, arguments, res: dict, action: bool = True) -> None:
    """An action another agent took (the runner reports it after `env.apply`)."""
    log.emit("call", role, **_call_data(tool, arguments, res, action))


TAIL_BYTES = 64_000


def read_events(d: Path, offset: int = -1) -> dict:
    """New events of a run's feed from byte `offset` (the `offset` a previous read returned), complete lines only;
    `offset` < 0: the last TAIL_BYTES of it. -> {"offset", "events"}."""
    path = Path(d) / EVENTS
    try:
        size = path.stat().st_size
    except OSError:
        return {"offset": 0, "events": []}
    start = max(0, size - TAIL_BYTES) if offset < 0 else min(offset, size)
    with open(path, "rb") as fh:
        fh.seek(start)
        data = fh.read(size - start)
    if offset < 0 and start > 0:                 # from the middle of a line: skip to the next one
        cut = data.find(b"\n")
        start, data = start + cut + 1, data[cut + 1:]
    end = data.rfind(b"\n") + 1                 # a line still being written waits for the next read
    events = []
    for line in data[:end].splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    return {"offset": start + end, "events": events}
