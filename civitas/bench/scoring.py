"""Benchmark scores (0-100) for the city and for each role.

Every score is a function of the monthly metrics history, so the same code
gives (a) the final benchmark score, (b) a live "how am I doing" signal in
observations, and (c) dense RL rewards as score deltas between months.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ScoringConfig:
    target_population: int = 30000
    gate_population: int = 300
    start_funds: float = 30000.0
    window: int = 12


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _mean(xs, default: float = 0.0) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else default


def components(m: dict, cfg: ScoringConfig) -> dict:
    pop = m["population"]
    finance = 0.0 if m["bankrupt"] else _clip(0.5 + 0.5 * math.tanh(m["net_worth"] / cfg.start_funds))
    return {
        "pop_score": _clip((pop / cfg.target_population) ** 0.5),
        "gate": _clip(pop / cfg.gate_population),
        "happiness": m["happiness"],
        "finance": finance,
        "services": _mean([m["power_coverage"] or 0.0, m["water_coverage"] or 0.0, m.get("sewage_coverage", 0.0) or 0.0]),
        "environment": 1.0 - m["pollution_res"],
        "employment": 1.0 - m["unemployment"],
        "safety": _clip(1.0 - m.get("crime_res", 0.0) / 0.35),
        "health": m.get("health", 0.75),
        "education": _clip(0.5 * _enrolled(m) + m.get("education_index", 0.0)),
    }


def _enrolled(m: dict) -> float:
    e = m.get("enrollment") or {}
    c = m.get("cohorts") or {}
    kids, teens = c.get("children", 0), c.get("teens", 0)
    if kids + teens <= 0:
        return 0.0
    return (e.get("elementary", 0.0) * kids + e.get("high", 0.0) * teens) / (kids + teens)


def _cost_eff(m: dict, dept: str, ref_per_resident: float) -> float:
    """1.0 while a department costs no more than a reference $/resident/month."""
    upkeep = (m.get("dept_upkeep") or {}).get(dept, 0.0)
    if upkeep <= 0:
        return 0.0
    return _clip(ref_per_resident * max(1, m["population"]) / upkeep)


def population_term(population: float, cfg: ScoringConfig) -> float:
    """The city score's population half, 50 x pop_score. city_score = this + the quality half (50 x gate x quality),
    times the recall / bankruptcy penalties: a split of a score change into population and quality reads the
    population part here and leaves the rest (penalties included) to quality."""
    return 100 * 0.5 * _clip((max(0.0, population) / cfg.target_population) ** 0.5)


def city_score(history: list[dict], cfg: ScoringConfig) -> float:
    if not history:
        return 0.0
    c = components(history[-1], cfg)
    quality = (0.25 * c["happiness"] + 0.15 * c["finance"] + 0.15 * c["services"]
               + 0.1 * c["environment"] + 0.1 * c["employment"] + 0.1 * c["safety"]
               + 0.1 * c["health"] + 0.05 * c["education"])
    score = 100 * (0.5 * c["pop_score"] + 0.5 * c["gate"] * quality)
    if history[-1].get("recalled"):
        score *= 0.75
    return score * (0.5 if history[-1]["bankrupt"] else 1.0)


def _deployment(win: list[dict]) -> float:
    """What the departments spent over the window against that plus the money they held idle: the department funds
    averaged over the window's months (civitas-v2; v1 read the final month only, so money taken back in the last
    month counted as never held)."""
    spent = sum(sum((h.get("dept_spent") or {}).values()) for h in win)
    idle = _mean([sum((h.get("dept_funds") or {}).values()) for h in win])
    return spent / (spent + idle) if spent + idle > 0 else 0.0


def mayor_score(history: list[dict], cfg: ScoringConfig) -> float:
    if not history:
        return 0.0
    m = history[-1]
    if m["bankrupt"] or m.get("recalled"):
        return 0.0
    c = components(m, cfg)
    win = history[-cfg.window:]
    ref_income = cfg.target_population * 8 * 0.09
    revenue = _clip(math.log1p(max(0.0, m["income"])) / math.log1p(ref_income))
    solvency = sum(1 for h in win if h["treasury"] >= 0) / len(win)
    deployment = _deployment(win)
    approval = _mean([h.get("approval", 0.6) for h in win])
    return 100 * (0.3 * revenue + 0.7 * c["gate"] * (
        0.25 * c["finance"] + 0.2 * solvency + 0.2 * c["happiness"] + 0.15 * deployment + 0.2 * approval))


def planning_score(history: list[dict], cfg: ScoringConfig) -> float:
    if not history:
        return 0.0
    m = history[-1]
    c = components(m, cfg)
    win = history[-cfg.window:]
    util = m["developed_total"] / m["zoned_total"] if m["zoned_total"] else 0.0
    balance = 1.0 - _mean([sum(abs(v) for v in h["demand"].values()) / 3 for h in win])
    road_eff = _clip(m["developed_total"] / (1.5 * m["road_tiles"])) if m["road_tiles"] else 0.0
    return 100 * (0.4 * c["pop_score"] + 0.6 * c["gate"] * (
        0.2 * util + 0.2 * balance + 0.2 * m["land_value_dev"] + 0.2 * road_eff + 0.2 * c["employment"]))


def _margin(supply: float, demand: float) -> float | None:
    if demand <= 0:
        return None
    r = supply / demand
    if r < 1.1:
        return _clip((r - 0.7) / 0.4)
    if r > 1.6:
        return _clip(1.0 - (r - 1.6) / 2.0)
    return 1.0


def utilities_score(history: list[dict], cfg: ScoringConfig) -> float:
    if not history:
        return 0.0
    m = history[-1]
    c = components(m, cfg)
    win = history[-cfg.window:]
    pcov = _mean([h["power_coverage"] for h in win])
    wcov = _mean([h["water_coverage"] for h in win])
    margin = _mean([_margin(h["power_supply"], h["power_demand"]) for h in win]
                   + [_margin(h["water_supply"], h["water_demand"]) for h in win])
    scov = _mean([h.get("sewage_coverage") or 0.0 for h in win])
    low_pollution = 1.0 - m["plant_pollution_res"]
    wq = m["water_quality_res"]
    river = 1.0 - m.get("river_pollution", 0.0)
    total_power_cap = m["power_supply"]
    clean_share = m.get("power_clean_capacity", 0) / total_power_cap if total_power_cap else 0.0
    # cost efficiency vs. a cheap reference mix (coal for power, pumps for water) with 30% headroom
    ref = 1.3 * (0.32 * m["power_demand"] + 0.10 * m["water_demand"] + 0.12 * m.get("sewage_demand", 0.0))
    upkeep = m.get("utility_upkeep", 0.0)
    cost_eff = _clip(ref / upkeep) if upkeep > 0 else 0.0
    return 100 * c["gate"] * (0.2 * pcov + 0.2 * wcov + 0.15 * scov + 0.12 * margin + 0.1 * cost_eff
                              + 0.08 * clean_share + 0.05 * low_pollution + 0.05 * wq + 0.05 * river)


def transport_score(history: list[dict], cfg: ScoringConfig) -> float:
    if not history:
        return 0.0
    m = history[-1]
    c = components(m, cfg)
    win = history[-cfg.window:]
    commute = _mean([_clip(1.0 - (h.get("commute_minutes", 0.0) - 12.0) / 30.0) for h in win])
    flow = _mean([_clip(1.0 - 3.0 * h.get("congestion_index", 0.0)) for h in win])
    transit = _clip(_mean([h.get("transit_share", 0.0) for h in win]) / 0.12)
    up = (m.get("dept_upkeep") or {}).get("transport", 0.0)
    cost = _clip(0.06 * max(1, m["population"]) / up) if up > 0 else 1.0
    return 100 * c["gate"] * (0.35 * commute + 0.25 * flow + 0.2 * transit + 0.2 * cost)


def police_score(history: list[dict], cfg: ScoringConfig) -> float:
    if not history:
        return 0.0
    m = history[-1]
    c = components(m, cfg)
    win = history[-cfg.window:]
    safety = _mean([_clip(1.0 - h.get("crime_res", 0.0) / 0.35) for h in win])
    cov = _mean([h.get("police_coverage", 0.0) for h in win])
    jail = 1.0 if m.get("inmates", 0.0) <= max(1.0, m.get("jail_capacity", 0)) else 0.4
    return 100 * c["gate"] * (0.45 * safety + 0.25 * cov + 0.1 * jail + 0.2 * _cost_eff(m, "police", 0.03))


def fire_score(history: list[dict], cfg: ScoringConfig) -> float:
    if not history:
        return 0.0
    m = history[-1]
    c = components(m, cfg)
    win = history[-cfg.window:]
    lost = sum(h.get("fire_tiles_lost", 0) for h in win)
    losses = _clip(1.0 - 10.0 * lost / max(20, m["developed_total"]))
    cov = _mean([h.get("fire_coverage", 0.0) for h in win])
    return 100 * c["gate"] * (0.4 * losses + 0.4 * cov + 0.2 * _cost_eff(m, "fire", 0.03))


def health_score(history: list[dict], cfg: ScoringConfig) -> float:
    if not history:
        return 0.0
    m = history[-1]
    c = components(m, cfg)
    win = history[-cfg.window:]
    health = _mean([h.get("health", 0.0) for h in win])
    clean = _mean([1.0 - h.get("garbage_homes_pct", 0.0) for h in win])
    fill = m.get("landfill_fill_pct")
    room = 1.0 if fill is None else _clip((100 - fill) / 30)
    dc = 1.0 if m.get("death_backlog", 0.0) < 10 else _clip(1.0 - m["death_backlog"] / 200)
    return 100 * c["gate"] * (0.3 * health + 0.25 * clean + 0.1 * m.get("health_coverage", 0.0)
                              + 0.1 * room + 0.1 * dc + 0.15 * _cost_eff(m, "health", 0.05))


def education_score(history: list[dict], cfg: ScoringConfig) -> float:
    if not history:
        return 0.0
    m = history[-1]
    c = components(m, cfg)
    win = history[-cfg.window:]
    enrolled = _mean([_enrolled(h) for h in win])
    skilled = _mean([h.get("skilled_fill", 1.0) for h in win])
    return 100 * c["gate"] * (0.4 * enrolled + 0.2 * skilled + 0.2 * _clip(m.get("education_index", 0.0) / 0.5)
                              + 0.2 * _cost_eff(m, "education", 0.05))


ROLE_SCORERS = {"mayor": mayor_score, "planning": planning_score, "utilities": utilities_score,
                "transport": transport_score,
                "police": police_score, "fire": fire_score, "health": health_score, "education": education_score}


def all_scores(history: list[dict], cfg: ScoringConfig, roles) -> dict:
    out = {"city": round(city_score(history, cfg), 3)}
    for r in roles:
        out[r] = round(ROLE_SCORERS[r](history, cfg), 3)
    return out
