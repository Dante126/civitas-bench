"""Live control of a running game through marker files in its run folder: the viewer's Pause / Resume / Stop.

The server writes a request and the runner (`bench/runner.py:_play`) reads it before every turn and at every month's
start, so a pause lands between two agents' turns (a fleet turn in progress finishes first):
- HOLD: pause. The runner writes HELD (`{"month", "role", "since"}`, role = the turn it waits before) and touches
  it every `HEARTBEAT_S` while it waits; a HELD older than `STALE_S` belongs to a process that died.
- STOP: end the game at the next turn: STOPPED, no summary (the server stops its own launches at once instead).
Removing HOLD resumes. A held game plays on exactly as it would have; the wait doesn't count in its wall time."""
from __future__ import annotations

import json
import time
from pathlib import Path

HOLD, HELD, STOP, STOPPED = "HOLD", "HELD", "STOP", "STOPPED"
HEARTBEAT_S = 1.0
STALE_S = 15.0


def pause(d: Path) -> None:
    (d / HOLD).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def release(d: Path) -> bool:
    """Resume a held (or pausing) game; False when no pause was asked for."""
    had = (d / HOLD).exists()
    (d / HOLD).unlink(missing_ok=True)
    return had


def stop(d: Path) -> None:
    (d / STOP).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    (d / HOLD).unlink(missing_ok=True)


def mark_stopped(d: Path) -> None:
    (d / STOPPED).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    clear(d)


def clear(d: Path) -> None:
    for name in (HOLD, HELD, STOP):
        (d / name).unlink(missing_ok=True)


def held(d: Path) -> dict | None:
    """The HELD record of a game waiting now (fresh heartbeat), else None."""
    f = d / HELD
    try:
        if time.time() - f.stat().st_mtime > STALE_S:
            return None
        return json.loads(f.read_text() or "{}")
    except (OSError, ValueError):
        return None


def pausing(d: Path) -> bool:
    return (d / HOLD).exists()


def wait(d: Path, month: int, role: str | None) -> tuple[bool, float]:
    """The runner's check before a turn: wait while HOLD is there. Returns (stop now, seconds held)."""
    if (d / STOP).exists():
        return True, 0.0
    if not (d / HOLD).exists():
        return False, 0.0
    t0 = time.time()
    rec = json.dumps({"month": month, "role": role, "since": t0})
    while (d / HOLD).exists() and not (d / STOP).exists():
        (d / HELD).write_text(rec)                 # the heartbeat: the server reads its mtime
        time.sleep(HEARTBEAT_S)
    (d / HELD).unlink(missing_ok=True)
    return (d / STOP).exists(), time.time() - t0
