"""How the agents worked together in a run: the viewer's Team tab (/api/runs/<id>/collab).

Everything comes from frames.jsonl, which every run writes (scripted, LLM and fleet alike): the council messages and each
turn's action results. Three kinds of event: messages (requests and reports), money (the mayor's allocate_budget) and
builds (a department's successful build, road, zone or upgrade). A thread links a request to its response and outcome:
health asks the mayor for $2,000 for a landfill, the mayor funds it, health builds the landfill.

Matching, inside a window of 2 x act_every months after the request (turn order breaks ties within a month):
- a request is a message to one role that asks for something ($, need, request, please, can you, ...); messages to
  "all" are broadcasts;
- a money request (to the mayor, naming dollars or funds) is met by the mayor's top-ups to the sender reaching
  MET_SHARE of the amount (any top-up when no amount is named), partial below that, else replied to when the mayor
  messages the sender back (a no, with why or when, is an answer);
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
OPENERS = {"build_road": "road", "dezone": "freed lot", "bulldoze": "cleared land", "demolish": "cleared land"}
HANDOFF_NEAR = 2
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


def action_rect(a: dict) -> list[int] | None:
    """[x1, y1, x2, y2] an action covered (its rectangle or road line's box, or its one tile), else None."""
    try:
        if "x1" in a and "y1" in a:
            x1, y1, x2, y2 = int(a["x1"]), int(a["y1"]), int(a.get("x2", a["x1"])), int(a.get("y2", a["y1"]))
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        if "x" in a and "y" in a:
            return [int(a["x"]), int(a["y"]), int(a["x"]), int(a["y"])]
    except (TypeError, ValueError):
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
    """(council, turns) of a frame, for one city of a region run with the city prefix dropped. Messages to another
    city are left out (they are the city's `external` events, see region_collab); one to the whole own team is "all"."""
    if city is None or "cities" not in frame:
        return frame.get("council") or [], frame.get("turns") or []
    pre = city + "."
    strip = lambda r: "all" if r == city else r[len(pre):] if isinstance(r, str) and r.startswith(pre) else r
    council = [{**m, "sender": strip(m.get("sender")), "to": strip(m.get("to"))} for m in frame.get("council") or []
               if str(m.get("sender", "")).startswith(pre) and not _external(m, frame["cities"])]
    turns = [{**t, "role": strip(t.get("role"))} for t in (frame["cities"].get(city) or {}).get("turns") or []]
    return council, turns


def collab(frames: list[dict], roles: list[str], act_every: int = 1, city: str | None = None,
           external: bool = True) -> dict:
    """The events, threads and per-role stats of a run's frames (see the module doc). For one city of a region run
    (external=True) also that city's deals and its messages with other cities (city_external)."""
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
                           "ask": is_request(to, text)})
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
                                   "from": role, "to": role, "text": _what(a), "tile": tile, "rect": action_rect(a),
                                   "type": typ, "amount": float(r.get("cost") or 0) or None})
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
    out = {"roles": roles, "act_every": act_every, "last_month": last, "window": window,
           "events": [{k: v for k, v in e.items() if k != "tiles" or v} for e in events],
           "threads": threads, "handoffs": handoffs(events, key, window), "stats": _stats(roles, events, threads)}
    if city is not None and external and any("cities" in f for f in frames):
        out.update(city_external(frames, roles, act_every, city))
    return out


def _gap(rect: list[int], tile: list[int]) -> int:
    """Chebyshev distance from a tile to a rectangle (0 inside it)."""
    x1, y1, x2, y2 = rect
    return max(x1 - tile[0], tile[0] - x2, y1 - tile[1], tile[1] - y2, 0)


def handoffs(events: list[dict], key, window: int) -> list[dict]:
    """Land handed over: another department's build within HANDOFF_NEAR tiles of an earlier planning road, dezone or
    clearance (OPENERS) inside the window; the latest such opener for each build. The Team view draws them on the map."""
    opens = [e for e in events if e["from"] == "planning" and e.get("type") in OPENERS and e.get("rect")]
    out = []
    for b in events:
        if b["kind"] != "build" or b["from"] == "planning" or not b.get("tile"):
            continue
        kb = key(b)
        src = [e for e in opens if key(e) < kb and b["month"] - e["month"] <= window and _gap(e["rect"], b["tile"]) <= HANDOFF_NEAR]
        if not src:
            continue
        e = src[-1]
        x1, y1, x2, y2 = e["rect"]
        at = [min(max(b["tile"][0], x1), x2), min(max(b["tile"][1], y1), y2)]     # the opener's tile nearest the build
        out.append({"month": e["month"], "from": "planning", "what_from": OPENERS[e["type"]], "rect_from": e["rect"],
                    "tile_from": at, "month_to": b["month"], "to": b["from"], "what": b["text"], "tile_to": b["tile"],
                    "event_from": e["id"], "event_to": b["id"]})
    return out


def is_request(to: str, text: str) -> bool:
    """A council message that asks something of one role (ASK words); messages to "all" are broadcasts."""
    return str(to or "all") != "all" and bool(ASK.search(str(text or "")))


def money_ask(req: dict) -> bool:
    """A request to the mayor that names dollars or funds (MONEY)."""
    return req["to"] == "mayor" and bool(MONEY.search(req["text"]))


def answer(req: dict, later: list[dict]) -> tuple[str | None, list[dict]]:
    """How a request ({"from", "to", "text", "amount", "tiles"}) was answered by `later`, the events after it inside
    its window in turn order (messages {"kind": "message", "from", "to"}, top-ups {"kind": "money", "from", "to",
    "amount"}, actions {"kind": "build" / "act", "from", "tile"}): ("met" / "partial", the top-ups) for a money ask
    met by the mayor's top-ups to the sender, ("acted", [action]) for another request acted on within NEAR tiles of a
    tile it names, ("replied", [message]) for a message from the one asked back to the asker (the first of acted /
    replied for other requests), else (None, []). collab's threads and the mayor's Open requests block
    (fleet/tools.py:open_requests) both decide by it."""
    a, b = req["from"], req["to"]
    reply = [e for e in later if e["kind"] == "message" and e["from"] == b and e["to"] == a]
    if money_ask(req):
        gifts = [e for e in later if e["kind"] == "money" and e["from"] == "mayor" and e["to"] == a]
        if gifts:
            want = req.get("amount")
            return ("met" if not want or sum(e["amount"] for e in gifts) >= MET_SHARE * want else "partial"), gifts
        return ("replied", reply[:1]) if reply else (None, [])
    near = [e for e in later if e["from"] == b and e["kind"] in ("build", "act") and e.get("tile") and any(
        abs(e["tile"][0] - x) <= NEAR and abs(e["tile"][1] - y) <= NEAR for x, y in req.get("tiles") or [])]
    first = next((e for e in later if e in near[:1] + reply[:1]), None)
    if first is None:
        return None, []
    return ("acted" if first in near else "replied"), [first]


def _thread(req: dict, events: list[dict], key, window: int, last: int) -> dict:
    """One request's thread: its response and outcome (see the module doc)."""
    a, b, k0, end = req["from"], req["to"], key(req), req["month"] + window
    later = [e for e in events if key(e) > k0 and e["month"] <= end]
    kind = "money" if money_ask(req) else "land" if req.get("tiles") or LAND.search(req["text"]) else "other"
    ids, response = [req["id"]], None
    status, got = answer(req, later)
    if status in ("met", "partial"):
        response = {"month": got[0]["month"], "kind": "money", "amount": sum(e["amount"] for e in got),
                    "text": got[0]["text"]}
    elif status is not None:
        first = got[0]
        response = {"month": first["month"], "kind": first["kind"], "amount": first.get("amount"),
                    "text": first["text"], "tile": first.get("tile")}
    ids += [e["id"] for e in got]
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


# ================================================================== region runs: cities working together
# The cities as a network: the mayors' deals (offer_deal / accept_deal / cancel_deal; from each frame's `deals`
# snapshot and the deal actions in the turns) and the council messages from one city to another. region_collab() is
# the Team tab's Region view (/api/runs/<id>/collab?region=1); city_external() adds one city's share of it to
# collab(city=C) as `deals`, `external` (events) and `external_threads`.
#
# Months: a frame carries the turns of the month before it, and its deal snapshot is taken after that month's turns
# and deal payments, so a deal first seen offered / active / ended in frame M was offered / accepted / ended in month
# M - 1 (a deal action in the turns gives the month and who acted directly). A deal that ran out was paid in its last
# month. Threads between cities: every deal offer is one (accepted: met; cancelled before that by the partner:
# declined, by the proposer: withdrawn; else a reply from the partner city: replied), and so is every other message to
# another city that asks for something (the ASK words or a question mark): acted on when the other city's mayor offers
# or accepts a deal with the asker inside the window (even after a reply), else replied to when any of its roles
# messages the asker's city back.
DEAL_ACTIONS = ("offer_deal", "accept_deal", "cancel_deal")
DEAL_MSG = re.compile(r"^\s*DEAL #(\d+) proposed", re.I)
DEAL_ID = re.compile(r"#(\d+)")
TRADE = re.compile(r"\bdeal|\bsell|\bbuy|\bpower\b|\bwater\b|\bgoods\b|\btrade|\bsurplus|\bimport|\bexport|\bcapacit", re.I)
LINKING = re.compile(r"\bconnector|\bhighway|\brail|\bharbou?r|\bairport|\blink|\bcommut|\btouris", re.I)
DONE = ("met", "partial", "acted", "replied")
LATE = 99                                   # sort rank of a deal running out: after every turn of its month


def _city_of(name) -> str:
    return str(name).split(".", 1)[0]


def _external(m: dict, cities) -> bool:
    """A council message from one city to another city's team ("south") or one of its roles ("south.mayor")."""
    to = str(m.get("to") or "all")
    return to != "all" and _city_of(to) in cities and _city_of(to) != _city_of(m.get("sender"))


def _cities(frames: list[dict]) -> list[str]:
    return next((list(f["cities"]) for f in reversed(frames) if isinstance(f.get("cities"), dict)), [])


def _new_deal(i: int) -> dict:
    return {"id": i, "seller": None, "buyer": None, "resource": None, "amount": 0.0, "price": 0.0, "months": None,
            "proposer": None, "offered": None, "accepted": None, "ended": None, "end": None, "ended_by": None,
            "_first": None, "_last": None}


def deal_history(frames: list[dict], cities) -> list[dict]:
    """Every deal of a region run with its life: offered / accepted / ended months, how it ended (cancelled by a city,
    or expired), the months paid and what the buyer paid the seller in all (see the region block's doc)."""
    seen: dict[int, dict] = {}
    for f in frames:
        fm = max(0, int(f.get("month", 0)) - 1)
        for t in f.get("turns") or []:
            c = _city_of(t.get("role", ""))
            if c not in cities:
                continue
            month = int(t.get("month", fm))
            for r in t.get("results") or []:
                a = r.get("action") or {}
                typ = a.get("type")
                if typ not in DEAL_ACTIONS or not r.get("ok"):
                    continue
                try:
                    i = int(DEAL_ID.search(str(r.get("message", "")))[1]) if typ == "offer_deal" else int(a.get("deal"))
                except (TypeError, ValueError):
                    continue
                d = seen.setdefault(i, _new_deal(i))
                if typ == "offer_deal":
                    d["offered"], d["proposer"] = month, c
                    try:
                        d["months"] = int(a.get("months", 24))
                    except (TypeError, ValueError):
                        pass
                elif typ == "accept_deal":
                    d["accepted"] = month
                else:
                    d["ended"], d["end"], d["ended_by"] = month, "cancelled", c
        for s in f.get("deals") or []:
            try:
                d = seen.setdefault(int(s["id"]), _new_deal(int(s["id"])))
            except (KeyError, TypeError, ValueError):
                continue
            for k in ("seller", "buyer", "resource", "amount", "price", "proposer"):
                if s.get(k) is not None:
                    d[k] = s[k]
            st, left = s.get("status"), s.get("months_left")
            if d["_first"] is None:
                d["_first"] = (left, st)
            d["_last"] = (left, st)
            if d["offered"] is None:
                d["offered"] = fm
            if st == "active" and d["accepted"] is None:
                d["accepted"] = fm
            if st == "ended" and d["ended"] is None:
                d["ended"], d["end"] = fm, "expired" if (left or 0) <= 0 else "cancelled"
    out = []
    for i in sorted(seen):
        d = seen[i]
        first, lastsnap = d.pop("_first"), d.pop("_last")
        if d["seller"] is None or first is None:            # an action on a deal no frame shows
            continue
        if d["proposer"] is None:
            d["proposer"] = d["seller"]
        d["partner"] = d["buyer"] if d["proposer"] == d["seller"] else d["seller"]
        if d["months"] is None:
            d["months"] = int(first[0] or 0) + (first[1] != "offered")
        d["months_left"], d["status"] = int(lastsnap[0] or 0), lastsnap[1]
        d["months_paid"] = max(0, d["months"] - d["months_left"]) if d["accepted"] is not None else 0
        d["monthly"] = round(float(d["amount"]) * float(d["price"]), 2)
        d["paid"] = round(d["monthly"] * d["months_paid"], 2)
        d["text"] = (f"{d['seller']} sells {float(d['amount']):.0f} {d['resource']} to {d['buyer']} at "
                     f"${float(d['price']):g}/unit/month for {d['months']} months")
        out.append(d)
    return out


def _deal_events(deals: list[dict]) -> list[dict]:
    """offered / accepted / cancelled / expired events of each deal, mayor to mayor."""
    ev = []
    for d in deals:
        base = {"kind": "deal", "deal": d["id"], "resource": d["resource"], "amount": d["amount"], "price": d["price"],
                "seller": d["seller"], "buyer": d["buyer"]}
        two = lambda a, b: {"from": f"{a}.mayor", "to": f"{b}.mayor", "from_city": a, "to_city": b}
        ev.append({**base, **two(d["proposer"], d["partner"]), "month": d["offered"], "what": "offered",
                   "text": f"deal #{d['id']} offered: {d['text']}"})
        if d["accepted"] is not None:
            ev.append({**base, **two(d["partner"], d["proposer"]), "month": d["accepted"], "what": "accepted",
                       "text": f"deal #{d['id']} accepted: {d['text']}"})
        if d["end"] == "cancelled":
            by = d["ended_by"] or d["proposer"]
            ev.append({**base, **two(by, d["buyer"] if by == d["seller"] else d["seller"]), "month": d["ended"],
                       "what": "cancelled", "text": f"deal #{d['id']} cancelled by {by}"})
        elif d["end"] == "expired":
            ev.append({**base, **two(d["seller"], d["buyer"]), "month": d["ended"], "what": "expired",
                       "text": f"deal #{d['id']} ran out after {d['months_paid']} months, ${d['paid']:,.0f} paid"})
    return ev


def _cross_messages(frames: list[dict], cities) -> list[dict]:
    ev = []
    for f in frames:
        for m in f.get("council") or []:
            if not _external(m, cities):
                continue
            text, to = str(m.get("text") or ""), str(m.get("to"))
            dm = DEAL_MSG.match(text)
            ev.append({"month": int(m.get("month", 0)), "kind": "message", "from": m.get("sender"), "to": to,
                       "from_city": _city_of(m.get("sender")), "to_city": _city_of(to), "text": text[:TEXT_CHARS],
                       "amount": None if dm else _amount(text), "deal": int(dm[1]) if dm else None,
                       "ask": not dm and bool(ASK.search(text) or "?" in text)})
    return ev


def _region_parts(frames: list[dict], roles: list[str], act_every: int, cities: list[str]) -> dict:
    """The cross-city events (messages and deal events, in turn order), threads and deals of a region run."""
    rank = {r: i for i, r in enumerate(roles)}
    ci = {c: i for i, c in enumerate(cities)}
    deals = deal_history(frames, cities)
    events = _cross_messages(frames, cities) + _deal_events(deals)
    key = lambda e: (e["month"], LATE if e.get("what") == "expired" else rank.get(str(e["from"]).split(".")[-1], LATE),
                     ci.get(e["from_city"], len(ci)), 0 if e["kind"] == "message" else 1)
    events.sort(key=key)
    for i, e in enumerate(events):
        e["id"] = i
    last = max([int(f.get("month", 0)) for f in frames] or [0])
    window = 2 * max(1, int(act_every))
    threads = [_deal_thread(d, events, key, window, last) for d in deals]
    threads += [_cross_thread(e, events, deals, key, window, last) for e in events if e["kind"] == "message" and e["ask"]]
    threads.sort(key=lambda t: (t["month"], t.pop("_k")))
    for i, t in enumerate(threads):
        t["id"] = i
    return {"events": events, "threads": threads, "deals": deals}


def _deal_thread(d: dict, events: list[dict], key, window: int, last: int) -> dict:
    a, b, t0 = d["proposer"], d["partner"], d["offered"]
    offer = next(e for e in events if e["kind"] == "deal" and e["deal"] == d["id"] and e["what"] == "offered")
    response, status = None, None
    if d["accepted"] is not None:
        status, response = "met", {"month": d["accepted"], "kind": "deal", "amount": None, "deal": d["id"],
                                   "text": f"{b} accepted deal #{d['id']}"}
    elif d["end"] == "cancelled":
        by = d["ended_by"] or a
        status = "withdrawn" if by == a else "declined"
        response = {"month": d["ended"], "kind": "deal", "amount": None, "deal": d["id"],
                    "text": f"{by} cancelled deal #{d['id']} before it started"}
    else:
        reply = next((e for e in events if key(e) > key(offer) and e["month"] <= t0 + window and e["kind"] == "message"
                      and e["from_city"] == b and e["to_city"] == a and e["deal"] is None), None)
        if reply is not None:
            status, response = "replied", {"month": reply["month"], "kind": "message", "amount": None, "deal": None,
                                           "text": reply["text"]}
    if status is None:
        status = "open" if last < t0 + window else "unanswered"
    return {"month": t0, "from": f"{a}.mayor", "to": f"{b}.mayor", "from_city": a, "to_city": b, "kind": "deal",
            "deal": d["id"], "amount": None, "text": f"offers deal #{d['id']}: {d['text']}", "status": status,
            "response": response, "outcome": None, "_k": key(offer)}


def _cross_thread(req: dict, events: list[dict], deals: list[dict], key, window: int, last: int) -> dict:
    a, b, k0, end = req["from_city"], req["to_city"], key(req), req["month"] + window
    later = [e for e in events if key(e) > k0 and e["month"] <= end]
    reply = [e for e in later if e["kind"] == "message" and e["from_city"] == b and e["to_city"] == a and e["deal"] is None]
    acted = [e for e in later if e["kind"] == "deal" and e["what"] in ("offered", "accepted") and e["from_city"] == b
             and e["to_city"] == a]
    first = (acted or reply or [None])[0]               # a deal is the answer, even after a first reply
    response = status = outcome = None
    if first is not None:
        status = "acted" if first in acted else "replied"
        response = {"month": first["month"], "kind": first["kind"], "amount": None, "deal": first.get("deal"),
                    "text": first["text"]}
        done = next((d for d in deals if {d["seller"], d["buyer"]} == {a, b} and d["accepted"] is not None
                     and response["month"] <= d["accepted"] <= response["month"] + window), None)
        if done is not None:
            outcome = {"month": done["accepted"], "what": f"deal #{done['id']} active: {done['text']}", "deal": done["id"]}
    if status is None:
        status = "open" if last < end else "unanswered"
    kind = "trade" if acted or TRADE.search(req["text"]) else "link" if LINKING.search(req["text"]) else "other"
    return {"month": req["month"], "from": req["from"], "to": req["to"], "from_city": a, "to_city": b, "kind": kind,
            "deal": None, "amount": req["amount"], "text": req["text"], "status": status, "response": response,
            "outcome": outcome, "_k": k0}


def _region_stats(cities: list[str], roles: list[str], frames: list[dict], act_every: int, parts: dict) -> dict:
    """Per city: requests inside the city and how many were answered, builds, the mayor's money moved, messages to
    and from other cities, asks between cities and their answers, deals proposed / offered to it / accepted / active,
    and what it earned selling and paid buying through deals."""
    out = {}
    events, threads, deals = parts["events"], parts["threads"], parts["deals"]
    for c in cities:
        st = collab(frames, roles, act_every, c, external=False)["stats"]
        tot = lambda k: sum(v[k] for v in st.values())
        msgs = [e for e in events if e["kind"] == "message"]
        asks = [t for t in threads if t["kind"] != "deal"]
        out[c] = {"requests": tot("asked"), "answered": tot("answered"), "builds": tot("builds"),
                  "money_moved": tot("money_in"),
                  "sent": sum(e["from_city"] == c for e in msgs), "received": sum(e["to_city"] == c for e in msgs),
                  "asks_out": sum(t["from_city"] == c for t in asks),
                  "asks_out_answered": sum(t["from_city"] == c and t["status"] in DONE for t in asks),
                  "asks_in": sum(t["to_city"] == c for t in asks),
                  "asks_in_answered": sum(t["to_city"] == c and t["status"] in DONE for t in asks),
                  "deals_proposed": sum(d["proposer"] == c for d in deals),
                  "deals_offered": sum(d["partner"] == c for d in deals),
                  "deals_accepted": sum(d["partner"] == c and d["accepted"] is not None for d in deals),
                  "deals_active": sum(c in (d["seller"], d["buyer"]) and d["status"] == "active" for d in deals),
                  "sold": round(sum(d["paid"] for d in deals if d["seller"] == c), 2),
                  "bought": round(sum(d["paid"] for d in deals if d["buyer"] == c), 2)}
    return out


def region_collab(frames: list[dict], roles: list[str], act_every: int = 1, cities: list[str] | None = None) -> dict:
    """The Team tab's Region view: the cities, the events between them (messages; deals offered, accepted, cancelled,
    expired), their threads, every deal's life, and per-city stats (see the region block's doc)."""
    cities = list(cities or _cities(frames))
    parts = _region_parts(frames, roles, act_every, cities)
    return {"region": True, "cities": cities, "roles": roles, "act_every": act_every,
            "last_month": max([int(f.get("month", 0)) for f in frames] or [0]), "window": 2 * max(1, int(act_every)),
            **parts, "stats": _region_stats(cities, roles, frames, act_every, parts)}


def city_external(frames: list[dict], roles: list[str], act_every: int, city: str) -> dict:
    """One city's part of region_collab: its deals, its events with other cities and their threads."""
    parts = _region_parts(frames, roles, act_every, _cities(frames))
    mine = lambda x: city in (x.get("from_city"), x.get("to_city"))
    return {"deals": [d for d in parts["deals"] if city in (d["seller"], d["buyer"])],
            "external": [e for e in parts["events"] if mine(e)],
            "external_threads": [t for t in parts["threads"] if mine(t)]}
# ================================================================== end of region runs


def run_collab(run_dir: str | Path, city: str | None = None, region: bool = False) -> dict:
    """collab() of a run folder: frames.jsonl, with roles and act_every from meta.json; region_collab() with
    region=True (or city="*")."""
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
    if region or city == "*":
        return region_collab(frames, roles, act, meta.get("cities"))
    return collab(frames, roles, act, city)
