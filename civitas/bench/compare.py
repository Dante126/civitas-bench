"""Runs side by side: the viewer's Compare view (/api/compare?runs=a,b,c).

`run_compare(run_dir)` reads one run folder (meta.json, summary.json when finished, frames.jsonl, fleet_report.json
when there is one) into a compact record: what kind of run it is (agents, region, warm-up), how long it played, its
final and average city score, the departments' final scores, population, the Teamwork score with its components
(fleet/report.py: collab_score, region_score; as of every month too), the fleet agents' failed and refused calls, and
per month the city score, population and Teamwork; a warm-up run its scripted yardstick (`yardstick`, from
yardstick.json: bench/yardstick.py) when it has one. A region run carries the same record per city under `by_city`
and region values at the top: the region's city score and population, the departments' and Teamwork's mean over the
cities, the calls summed. Works on running, paused and finished runs alike (a run without summary.json reads its
last frame).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

COUNTS = ("turns", "tool_calls", "actions", "failed", "tool_errors", "repeats", "refused", "overrides")
COMPONENTS = ("responsiveness", "follow_through", "handoffs", "money_aim")


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def kind_of(agents: dict) -> str:
    """The run's line-up in a word or two: LLM fleet, scripted fleet, LLM, scripted, ..., or mixed."""
    specs = [str(s) for s in (agents or {}).values()]
    if not specs:
        return "unknown"
    fleet = [s for s in specs if s.startswith("fleet:")]
    if fleet and all(s == "fleet:scripted" for s in fleet) and len(fleet) == len(specs):
        return "scripted fleet"
    if any(s != "fleet:scripted" for s in fleet):
        return "LLM fleet" if len(fleet) == len(specs) else "LLM fleet (mixed)"
    heads = {s.split(":")[0] for s in specs}
    if heads == {"llm"}:
        return "LLM"
    if len(heads) == 1:
        return next(iter(heads))
    return "LLM (mixed)" if "llm" in heads else "mixed"


def model_of(agents: dict) -> str | None:
    """The model the line-up plays with when every non-scripted agent uses the same one, e.g. trinity-large-thinking."""
    models = {str(s).split(":", 1)[1] for s in (agents or {}).values() if ":" in str(s)} - {"scripted"}
    return next(iter(models)) if len(models) == 1 else None


def _frames(path: Path, cities: list[str]) -> dict:
    """Per month (frame) the city score and population, and per city of a region the same; complete lines only."""
    out = {"month": [], "city": [], "population": [], "roles": None,
           "by_city": {c: {"city": [], "population": [], "roles": None} for c in cities}}
    if not path.exists():
        return out
    with open(path) as fh:
        for line in fh:
            if not line.endswith("\n"):
                continue
            try:
                f = json.loads(line)
            except ValueError:
                continue
            sc, m = f.get("scores") or {}, f.get("metrics") or {}
            out["month"].append(f.get("month", len(out["month"])))
            out["city"].append(_r(sc.get("city")))
            out["population"].append(m.get("population"))
            out["roles"] = sc
            for c in cities:
                cf = (f.get("cities") or {}).get(c) or {}
                csc = cf.get("scores") or {}
                b = out["by_city"][c]
                b["city"].append(_r(csc.get("city")))
                b["population"].append((cf.get("metrics") or {}).get("population"))
                b["roles"] = csc
    return out


def _r(v, n=2):
    return None if v is None else round(float(v), n)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.fmean(vals), 3) if vals else None


def _avg(months: list, city: list, warmup: int) -> tuple[float | None, float | None]:
    """(average city score over the months played, over the months after the warm-up): frame m >= 1 is the state after
    month m - 1, as the runner's city_scores."""
    played = [(m, v) for m, v in zip(months, city) if m >= 1]
    return _mean(v for _, v in played), (_mean(v for m, v in played if m > warmup) if warmup else None)


def _counts(roles: dict | None) -> dict | None:
    if not roles:
        return None
    return {k: sum(int((r or {}).get(k) or 0) for r in roles.values()) for k in COUNTS}


def _teamwork(sc: dict | None) -> dict | None:
    """A collab_score dict (single city) as {score, components: {k: score}, n: {k: cases}, requests, series}."""
    if not sc or sc.get("error"):
        return None
    comps = sc.get("components") or {}
    return {"score": sc.get("score"), "requests": sc.get("requests"),
            "components": {k: (comps.get(k) or {}).get("score") for k in COMPONENTS},
            "n": {k: (comps.get(k) or {}).get("n") for k in COMPONENTS},
            "series": [_r(v, 1) for v in (sc.get("series") or {}).get("score", [])]}


def _region_teamwork(rs: dict | None) -> tuple[dict | None, dict]:
    """(region values, per city) from region_score: a city's score, components and series as collab_score's; the
    region's the mean over the cities, with the asks between cities and the deals."""
    if not rs or rs.get("error"):
        return None, {}
    per = {}
    for c, s in (rs.get("cities") or {}).items():
        per[c] = {"score": s.get("score"), "requests": s.get("requests"),
                  "components": {k: (s.get("components") or {}).get(k) for k in COMPONENTS}, "n": None,
                  "series": [_r(v, 1) for v in s.get("series") or []]}
    if not per:
        return None, per
    n = max(len(t["series"]) for t in per.values())
    series = [_r(_mean([t["series"][i] for t in per.values() if i < len(t["series"])]), 1) for i in range(n)]
    region = {"score": _r(_mean([t["score"] for t in per.values()]), 1),
              "requests": sum(t["requests"] or 0 for t in per.values()),
              "components": {k: _r(_mean([t["components"][k] for t in per.values()]), 1) for k in COMPONENTS},
              "n": None, "series": series, "mean_of_cities": True,
              "asks": (rs.get("asks") or {}).get("score"), "deals": (rs.get("deals") or {}).get("score")}
    return region, per


def _collab(d: Path, city: str | None) -> dict:
    """run_collab with its score, as the server's collab_data (uncached)."""
    from ..fleet.report import collab_score, region_score
    from .collab import run_collab
    c = run_collab(d, city)
    if city == "*":
        return {"score": region_score(c, {x: _collab(d, x)["score"] for x in c["cities"]})}
    return {"score": collab_score(c)}


def _yard(ys: dict | None, city: str | None = None) -> dict | None:
    """A warm-up run's scripted yardstick (yardstick.json, bench/yardstick.py) as a column shows it: the change since
    the hand-over, the average city score after it, the end, and per frame month from the hand-over the city score
    and population; `city` for a region's city."""
    if not ys or not ys.get("handed_over"):
        return None
    src = ((ys.get("by_city") or {}).get(city) or None) if city else ys
    if not src or not src.get("change"):
        return None
    s = src.get("series") or {}
    return {"change": src["change"], "at_handover": src.get("at_handover"), "at_end": src.get("at_end"),
            "avg_city_score": None if city else ys.get("avg_city_score"), "matches_run": ys.get("matches_run"),
            "series": {"month": (ys.get("series") or {}).get("month") or [], "city": s.get("city") or [],
                       "population": s.get("population") or []}}


def run_compare(d: str | Path, collab=None) -> dict:
    """One run folder as the Compare view reads it (see the module doc). `collab(d, city)` returns run_collab's dict
    with `score` (the server passes its cached collab_data); city None for a single city, "*" for a region."""
    d = Path(d)
    collab = collab or _collab
    meta = _read(d / "meta.json") or {}
    summ = _read(d / "summary.json")
    rep = _read(d / "fleet_report.json")
    region = bool(meta.get("region"))
    cities = list(meta.get("cities") or []) if region else []
    warm = int(meta.get("warmup_months") or 0)
    agents = meta.get("agents") or (summ or {}).get("agents") or {}
    fr = _frames(d / "frames.jsonl", cities)
    months = (summ or {}).get("months_played")
    if months is None:
        months = fr["month"][-1] if fr["month"] else 0
    tags = (["region"] if region else []) + ([f"late (warm-up {warm})"] if warm else [])
    out = {"run_id": d.name, "name": meta.get("name"), "seed": meta.get("seed"), "signature": meta.get("signature"),
           "kind": kind_of(agents), "model": model_of(agents), "tags": tags, "region": region, "cities": cities,
           "months": months, "months_configured": meta.get("months"), "warmup": warm,
           "finished": summ is not None, "bankrupt": (summ or {}).get("bankrupt"),
           "fleet_version": meta.get("fleet_version"), "wall_time_s": (summ or {}).get("wall_time_s"),
           "month": fr["month"]}
    scores = (summ or {}).get("final_scores") or fr["roles"] or {}
    fm = (summ or {}).get("final_metrics") or {}
    wsum = (summ or {}).get("warmup") or {}
    hand = wsum.get("at_handover")
    ys = _read(d / "yardstick.json") if warm else None

    def entry(city_series: list, pop_series: list, roles: dict, final_city, pop, avg, team, counts, at_hand,
              yard=None) -> dict:
        a_all, a_after = _avg(fr["month"], city_series, warm)
        return {"city": _r(final_city, 3), "avg_city": avg if avg is not None else a_all, "avg_after_warmup": a_after,
                "roles": {k: _r(v, 3) for k, v in roles.items()}, "population": pop,
                "at_handover": at_hand, "teamwork": team, "calls": counts, "yardstick": yard,
                "series": {"city": city_series, "population": pop_series,
                           "teamwork": (team or {}).get("series") or []}}

    try:
        c = collab(d, "*" if region else None)
        sc = c.get("score") if isinstance(c, dict) else None
    except Exception:                       # the Teamwork score is extra: a run without frames still compares
        sc = None
    if not region:
        roles = {k: v for k, v in scores.items() if k != "city"}
        hand_at = {"month": hand["month"], "city": hand.get("city"), "population": hand.get("population")} if hand else None
        out.update(entry(fr["city"], fr["population"], roles, scores.get("city"),
                         fm.get("population", fr["population"][-1] if fr["population"] else None),
                         (summ or {}).get("avg_city_score"), _teamwork(sc),
                         _counts((rep or {}).get("roles")), hand_at, _yard(ys)))
        return out
    team, team_by = _region_teamwork(sc)
    by_city = {}
    for cty in cities:
        b = fr["by_city"][cty]
        pre = cty + "."
        croles = ({k[len(pre):]: v for k, v in scores.items() if k.startswith(pre) and k != pre + "city"}
                  if summ else {k: v for k, v in (b["roles"] or {}).items() if k != "city"})
        cfinal = scores.get(pre + "city") if summ else (b["roles"] or {}).get("city")
        cpop = ((fm.get("by_city") or {}).get(cty) or {}).get("population")
        if cpop is None and b["population"]:
            cpop = b["population"][-1]
        crep = ((rep or {}).get("cities") or {}).get(cty) or {}
        i = fr["month"].index(warm) if hand and warm in fr["month"] else None    # the city at hand-over, its frame
        chand = {"month": warm, "city": b["city"][i], "population": b["population"][i]} if i is not None else None
        by_city[cty] = entry(b["city"], b["population"], croles, cfinal, cpop, None, team_by.get(cty),
                             _counts(crep.get("roles")), chand, _yard(ys, cty))
    role_names = list(dict.fromkeys(r for e in by_city.values() for r in e["roles"]))
    mean_roles = {r: _mean([e["roles"].get(r) for e in by_city.values()]) for r in role_names}
    calls = [e["calls"] for e in by_city.values() if e["calls"]]
    total = {k: sum(x[k] for x in calls) for k in COUNTS} if calls else None
    out.update(entry(fr["city"], fr["population"], mean_roles, scores.get("city"),
                     fm.get("population", fr["population"][-1] if fr["population"] else None),
                     (summ or {}).get("avg_city_score"), team, total,
                     {"month": hand["month"], "city": hand.get("city"), "population": hand.get("population")}
                     if hand else None, _yard(ys)))
    out["roles_mean_of_cities"] = True
    out["by_city"] = by_city
    return out
