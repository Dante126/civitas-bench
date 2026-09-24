"""Prompt templates for LLM agents."""
from __future__ import annotations

from ..core.observation import render_action_docs
from ..core.roles import ROLES, TURN_ORDER

GAME_RULES = """\
You are one of several AI agents that jointly run a city in CIVITAS, a SimCity-like simulation.
Each agent owns one department. You only control your own department; to get something from
another department, ask for it on the council board.

How the city works
- The map is a flat plain split by a river running north-south. The highway tile H on the west
  edge is the city's only link to the outside world.
- Zoned land develops on its own when (1) there is demand for that zone type, (2) a road that
  connects to the highway is within 2 tiles, and (3) it has power. It needs water to grow past
  level 1, and R/C need high land value (river views, parks, no pollution) to reach level 3.
- Residents (R) need jobs: about half the population works, in commercial (C) or industrial (I)
  zones. Demand values (-1..1) tell you what the city wants more of right now.
- Industry and coal/gas plants pollute ~5 tiles around them. Pollution lowers land value and
  happiness and makes river pumps deliver dirty water. Keep homes away from pollution.
- Every connected road network is its own power and water grid. Facilities serve the network
  they touch. Bridges (road over river) cost more but open the east bank.
- City services work along the road network: a police station, fire station, clinic, school or
  landfill serves the buildings it can reach within its radius (in road steps) up to its capacity.
  Crime, fires, garbage, sickness and a lack of schools all lower happiness, land value or the
  workforce. Buildings need an empty land tile touching a road.
- People: residents are children, teens, adults and seniors with an education level. Adults work;
  skilled jobs need high-school or university graduates, and only students with a school seat
  graduate.
- Money: taxes are collected monthly from residents and filled jobs. Upkeep of roads, parks,
  plants and service buildings is paid from the treasury automatically. The mayor moves money
  to departments; departments can only spend their own budget.
- Turn order each month: {turn_order}; then the simulation advances one month.
  The game lasts {months} months.

Scoring
- City score (0-100): population (target {target_pop:,}), happiness, finances, power/water coverage,
  environment, employment, safety, health, education. Bankruptcy halves it and ends the game.
- You are also scored on your own department's KPIs, shown as "your role" score.
"""

OUTPUT_FORMAT = """\
Reply with ONE JSON object and nothing else:
{{
  "thinking": "1-3 sentences summarising why you act (keep long reasoning out of the JSON)",
  "actions": [ {{...action...}}, ... ],          // at most {max_actions}; [] to do nothing
  "messages": [ {{"to": "all" | {roles}, "text": "..."}} ],   // optional
  "notes": "optional memo to your future self (replaces your previous notes)"
}}
Coordinates are integers: x = column (0 = west edge, grows east), y = row (0 = north edge, grows south).
Use the coordinates listed in the observation (suggested sites, road segments, ready-to-zone strips,
hotspots). Do NOT count map characters one by one: the map is only for orientation. Decide quickly.
Check the results of your previous turn: fix failed actions instead of repeating them.
"""


REGION_RULES = """\
REGION MODE: two cities share the river. You serve the {city} city; the other city has its own team.
The river flows north to south: the north city's sewage and runoff reach the south city's water.
Both teams share the council board. Your mayor can agree power/water deals with the other mayor.
Both cities are scored separately; the region score is their average.
"""


def system_prompt(role: str, *, months: int, target_pop: int, max_actions: int, city: str | None = None) -> str:
    spec = ROLES[role]
    return "\n".join([
        GAME_RULES.format(turn_order=" -> ".join(TURN_ORDER), months=months, target_pop=target_pop),
        REGION_RULES.format(city=city.upper()) if city else "",
        f"YOUR ROLE: {spec['title']}",
        spec["mission"],
        "",
        "Your actions:",
        render_action_docs(role),
        "",
        OUTPUT_FORMAT.format(max_actions=max_actions, roles=" | ".join(f'"{r}"' for r in TURN_ORDER)),
    ])
