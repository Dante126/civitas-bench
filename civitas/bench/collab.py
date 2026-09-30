"""How the agents worked together in a run: the viewer's Team tab (/api/runs/<id>/collab).

Everything comes from frames.jsonl, which every run writes (scripted, LLM and fleet alike): the council messages and each
turn's action results. Three kinds of event: messages (requests and reports), money (the mayor's allocate_budget) and
builds (a department's successful build, road, zone or upgrade). A thread links a request to its response and outcome:
health asks the mayor for $2,000 for a landfill, the mayor funds it, health builds the landfill.

Matching, inside a window of 2 x act_every months after the request (turn order breaks ties within a month):
- a request is a message to one role that asks for something ($, need, request, please, can you, ...); messages to
  "all" are broadcasts;
- a money request (to the mayor, naming dollars or funds) is met by the mayor's top-ups to the sender reaching
  MET_SHARE of the amount (any top-up when no amount is named), partial below that;
- another request is acted on when the recipient acts within NEAR tiles of a tile it names, replied to when the
  recipient messages the sender back;
- else unanswered, or open while the window hasn't passed (a live run);
- the outcome is the requester's first build after the response, inside the window.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ASK = re.compile(r"\$\s?\d|\bneed|\brequest|\bplease\b|\bcould you\b|\bcan you\b|\bask|\brequir|\bfund|\bfree (up )?a",
                 re.I)
MONEY = re.compile(r"\$\s?\d|\bfund|\bbudget|\bmoney|\ballocat", re.I)
LAND = re.compile(r"\broad|\blot\b|\blots\b|\bzone|\bdezone|\briverbank|\bland\b|\btile", re.I)
AMOUNT = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k\b)?", re.I)
TILE = re.compile(r"\((\d{1,3})\s*,\s*(\d{1,3})\)")
BUILDS = ("build_", "create_", "upgrade_")      # outcomes: something new stands in the city
MET_SHARE = 0.8
NEAR = 3
TEXT_CHARS = 400


ASKED = re.compile(r"(need|request|ask|requir|additional|another|extra|more|top.?up|allocat|fund|send|give)\w*\W{1,12}"
                   r"(?:\w+\W+){0,4}?\$", re.I)


def _amount(text: str) -> float | None:
    """The dollars a message asks for: the first amount right after an ask word ("Requesting additional funds: $2,000"),
    else the largest amount ("Budget remaining: $61 ... $2,000 for roads" is a $2,000 ask); None for none or $0."""
    vals = [(m.start(), float(m[1].replace(",", "")) * (1000 if m[2] else 1)) for m in AMOUNT.finditer(text)]
    vals = [(i, v) for i, v in vals if v > 0]
    if not vals:
        return None
    for m in ASKED.finditer(text):
        at = next((v for i, v in vals if i == m.end() - 1), None)
        if at is not None:
            return at
    return max(v for _, v in vals)


def _tiles(text: str) -> list[list[int]]:
    return [[int(a), int(b)] for a, b in TILE.findall(text)]


def action_tile(a: dict) -> list[int] | None:
    """Where an action happened: its x,y, the middle of its x1..x2 / y1..y2, or a bus line's first stop."""
    try:
        if "x" in a and "y" in a:
            return [int(a["x"]), int(a["y"])]
        if "x1" in a and "y1" in a:
            return [(int(a["x1"]) + int(a.get("x2", a["x1"]))) // 2, (int(a["y1"]) + int(a.get("y2", a["y1"]))) // 2]
        if a.get("stops"):
            x, y = a["stops"][0]
            return [int(x), int(y)]
    except (TypeError, ValueError, KeyError):
        return None
    return None


def _what(a: dict) -> str:
    """"built clinic", "upgraded facility", "zoned R", "created bus line" from an action."""
    t = str(a.get("type", ""))
    if t == "zone":
        return f"zoned {a.get('zone', '')}".strip()
    for pre, verb in (("build_", "built "), ("create_", "created "), ("upgrade_", "upgraded ")):
        if t.startswith(pre):
            return verb + t[len(pre):].replace("_", " ")
    return t.replace("_", " ")


def _local(frame: dict, city: str | None) -> tuple[list, list]:
    """(council, turns) of a frame, for one city of a region run with the city prefix dropped."""
    if city is None or "cities" not in frame:
        return frame.get("council") or [], frame.get("turns") or []
    pre = city + "."
    strip = lambda r: r[len(pre):] if isinstance(r, str) and r.startswith(pre) else r
    council = [{**m, "sender": strip(m.get("sender")), "to": strip(m.get("to"))} for m in frame.get("council") or []
               if str(m.get("sender", "")).startswith(pre)]
    turns = [{**t, "role": strip(t.get("role"))} for t in (frame["cities"].get(city) or {}).get("turns") or []]
    return council, turns


def collab(frames: list[dict], roles: list[str], act_every: int = 1, city: str | None = None) -> dict:
    """The events, threads and per-role stats of a run's frames (see the module doc)."""
    order = {r: i for i, r in enumerate(roles)}
    events: list[dict] = []
    last = 0
    for f in frames:
        last = max(last, int(f.get("month", 0)))
        council, turns = _local(f, city)
        for m in council:
            to, text = str(m.get("to") or "all"), str(m.get("text") or "")
            events.append({"month": int(m.get("month", 0)), "kind": "message", "from": m.get("sender"), "to": to,
                           "text": text[:TEXT_CHARS], "amount": _amount(text), "tiles": _tiles(text),
                           "ask": to != "all" and bool(ASK.search(text))})
        for t in turns:
            role, month = t.get("role"), int(t.get("month", f.get("month", 0)))
            for r in t.get("results") or []:
                a = r.get("action") or {}
                if not r.get("ok"):
                    continue
                typ = str(a.get("type", ""))
                if typ == "allocate_budget":
                    try:
                        amt = float(a.get("amount", 0))
                    except (TypeError, ValueError):
                        continue
                    dept = str(a.get("department", ""))
                    if amt:
                        events.append({"month": month, "kind": "money", "from": role if amt > 0 else dept,
                                       "to": dept if amt > 0 else role, "amount": abs(amt), "text": r.get("message", ""),
                                       "tile": None, "take_back": amt < 0})
                    continue
                tile = action_tile(a)
                if typ.startswith(BUILDS) or typ in ("zone", "dezone", "bulldoze", "demolish"):
                    events.append({"month": month, "kind": "build" if typ.startswith(BUILDS) else "act",
                                   "from": role, "to": role, "text": _what(a), "tile": tile,
                                   "amount": float(r.get("cost") or 0) or None})
    key = lambda e: (e["month"], order.get(e["from"], len(order)), 0 if e["kind"] == "message" else 1)
    events.sort(key=key)
    for i, e in enumerate(events):
        e["id"], e["thread"] = i, None
    window = 2 * max(1, int(act_every))
    threads = [_thread(e, events, key, window, last) for e in events if e["kind"] == "message" and e["ask"]]
    for i, th in enumerate(threads):
        th["id"] = i
        for eid in (th.pop("_ids")):
            if events[eid]["thread"] is None:
                events[eid]["thread"] = i
    return {"roles": roles, "act_every": act_every, "last_month": last, "window": window,
            "events": [{k: v for k, v in e.items() if k != "tiles" or v} for e in events],
            "threads": threads, "stats": _stats(roles, events, threads)}


def _thread(req: dict, events: list[dict], key, window: int, last: int) -> dict:
    """One request's thread: its response and outcome (see the module doc)."""
    a, b, k0, end = req["from"], req["to"], key(req), req["month"] + window
    later = [e for e in events if key(e) > k0 and e["month"] <= end]
    money_ask = b == "mayor" and bool(MONEY.search(req["text"]))
    kind = "money" if money_ask else "land" if req.get("tiles") or LAND.search(req["text"]) else "other"
    ids, response, status = [req["id"]], None, None
    if money_ask:
        gifts = [e for e in later if e["kind"] == "money" and e["from"] == "mayor" and e["to"] == a]
        got = sum(e["amount"] for e in gifts)
        if gifts:
            want = req["amount"]
            status = "met" if not want or got >= MET_SHARE * want else "partial"
            response = {"month": gifts[0]["month"], "kind": "money", "amount": got, "text": gifts[0]["text"]}
            ids += [e["id"] for e in gifts]
    else:
        near = [e for e in later if e["from"] == b and e["kind"] in ("build", "act") and e["tile"] and any(
            abs(e["tile"][0] - x) <= NEAR and abs(e["tile"][1] - y) <= NEAR for x, y in req.get("tiles") or [])]
        reply = [e for e in later if e["kind"] == "message" and e["from"] == b and e["to"] == a]
        first = min(near[:1] + reply[:1], key=key, default=None)
        if first is not None:
            status = "acted" if first in near else "replied"
            response = {"month": first["month"], "kind": first["kind"], "amount": first.get("amount"),
                        "text": first["text"], "tile": first.get("tile")}
            ids.append(first["id"])
    outcome = None
    if response is not None:
        built = next((e for e in later + [e for e in events if end < e["month"] <= response["month"] + window]
                      if e["kind"] == "build" and e["from"] == a and e["month"] >= response["month"]), None)
        if built is not None:
            outcome = {"month": built["month"], "what": built["text"], "tile": built["tile"]}
            ids.append(built["id"])
    if status is None:
        status = "open" if last < end else "unanswered"
    return {"month": req["month"], "from": a, "to": b, "kind": kind, "amount": req["amount"], "text": req["text"],
            "status": status, "response": response, "outcome": outcome, "_ids": ids}


def _stats(roles: list[str], events: list[dict], threads: list[dict]) -> dict:
    """Per role: requests it sent / had answered, requests sent to it / it answered, money in and out, builds."""
    out = {r: {"asked": 0, "answered": 0, "asked_of": 0, "answered_by": 0, "money_in": 0.0, "money_out": 0.0,
               "builds": 0, "messages": 0} for r in roles}
    for th in threads:
        done = th["status"] in ("met", "partial", "acted", "replied")
        if th["from"] in out:
            out[th["from"]]["asked"] += 1
            out[th["from"]]["answered"] += done
        if th["to"] in out:
            out[th["to"]]["asked_of"] += 1
            out[th["to"]]["answered_by"] += done
    for e in events:
        if e["kind"] == "money":
            if e["to"] in out:
                out[e["to"]]["money_in"] += e["amount"]
            if e["from"] in out:
                out[e["from"]]["money_out"] += e["amount"]
        elif e["kind"] == "build" and e["from"] in out:
            out[e["from"]]["builds"] += 1
        elif e["kind"] == "message" and e["from"] in out:
            out[e["from"]]["messages"] += 1
    return out


def run_collab(run_dir: str | Path, city: str | None = None) -> dict:
    """collab() of a run folder: frames.jsonl, with roles and act_every from meta.json."""
    d = Path(run_dir)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    frames = []
    if (d / "frames.jsonl").exists():
        with open(d / "frames.jsonl") as fh:
            frames = [json.loads(line) for line in fh if line.endswith("\n")]
    roles = [r.split(".")[-1] for r in meta.get("roles") or []]
    roles = list(dict.fromkeys(roles)) or ["mayor", "planning", "utilities", "transport", "police", "fire", "health",
                                           "education"]
    act = int(((meta.get("config") or {}).get("game") or {}).get("act_every", 1) or 1)
    return collab(frames, roles, act, city)
