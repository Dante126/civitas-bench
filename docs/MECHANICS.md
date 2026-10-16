# City-builder mechanics roadmap: implementation status

This file tracks the city-builder mechanics roadmap that shaped Civitas
(originally written as a brief for an AI coding assistant) and what is implemented.
Legend: ✅ implemented · 🟡 simplified · ❌ not implemented.

Where a mechanic lives in the code is noted in brackets. All numbers are in
`civitas/sim/catalog.py` and can be overridden per config with `[game.balance]`.

## Ground rules (still apply to any change)

1. Constants in `sim/catalog.py`.
2. State and systems in `sim/` mixins, computed in `World.tick()`, exposed in `World._metrics()`.
3. Actions in `core/actions.py` (`HANDLERS`, `action_docs()`, spend from the right budget with `_spend()`, gate locked
   buildings with `_require()`).
4. Roles in `core/roles.py` (+ `TURN_ORDER`, `sim/world.py:DEPARTMENTS`).
5. Observations in `core/observation.py` (keep prompts around 3–4k tokens per role).
6. Scores in `bench/scoring.py` (0–100, gated on population).
7. A scripted baseline per role in `agents/scripted.py`, reading only the observation.
8. Viewer: tile character in `catalog.CHAR`, model in `viewer/scene.js`, `NAMES` / `MINI` in `viewer/app.js`,
   map layers via `World.frame_layers()` + `OVERLAYS` in `scene.js`.
9. Tests: the baseline must finish 120 months solvent on seeds 0–4 with a city score of 60–80, `noop` must score ~0,
   a scripted episode must stay fast (`tests/test_benchmark.py`, marked `slow`).
10. Determinism: randomness goes through `world.rng`; event schedules, natural resources and the wind map use their own
    seeded RNGs so they are identical for every agent line-up.

Note: the roadmap suggested `O` for the office zone, but `O` was the water tower. Water towers are now `V`
(`catalog.FRAME_FORMAT = 2`), and the mixed-use zone took `M` from the hospital, which is now `h`
(`FRAME_FORMAT = 3`); the viewer converts older replays. Building tiers added `u` (nuclear plant) and `z` (crematorium)
without a format change: other tiers reuse their line's character and the tier travels in the frame's `levels` string.

## Status by stage

| # | Stage | Status |
|---|---|---|
| 1 | Coverage framework: road-network BFS distance, capacity, funding, cached per road network [`services.py`, `grid.road_bfs`] | ✅ |
| 2 | Police + crime + jails, Fire + fire events, Health & Sanitation (garbage, landfills, incinerators, clinics, hospitals, cemeteries, sick workers), Education + age × education cohorts [`services.py`, `population.py`] | ✅ |
| 3 | Sewage (outlets, treatment plants) + river flowing north → south, pumps downstream get dirty water [`world.refresh`, `world._river_flow`] | ✅ |
| 4 | Needs-based happiness (14 needs), complaints feed, approval + recall election, funding levels 50–150%, 7 policies [`civics.py`] | ✅ |
| 5 | Aggregate traffic model (gravity commutes, shortest-path trees, BPR congestion), road classes, bus lines + fares, train stations, Transport role [`traffic.py`] | ✅ |
| 6 | Office zone, low/high density, levels 1–5, wealth classes, household budget (wages, rent, affordability, homelessness), specialised industry on natural resources [`economy.py`] | ✅ |
| 7 | Seeded events (flood with forecast, heatwave, recession, factory offer), levees + shelters, districts (policies, taxes, industry specialisation), milestones + unlocks, landmarks + tourism [`events.py`, `civics.py`, `progress.py`] | ✅ |
| 8 | Goods & freight, trade with the region, company profitability and closures, service imports, surplus power sales, plant ageing, wind map, facility upgrades, harbour, two-city region mode with neighbour deals [`economy.TradeMixin`, `core/region.py`] | ✅ |
| 9 | Mixed use, historic preservation, soil pollution, wind direction, seasons, metro, multi-tile buildings [`economy.py`, `progress.HeritageMixin`, `traffic.py`, `world.place_building`] | ✅ |
| 10 | Region of 2–6 cities: highway/rail connectors, boat and air links, inter-city trade, commuting and tourism, specialisations (mining, high tech, gambling, tourism, trade, education), casinos, airports [`core/region.py`] | ✅ |
| 11 | Building tiers: ladders per service, upgrades in place, crowding, downtown cover, nuclear plant, crematorium [`catalog.TIER_LINES`, `world.facility_spec`, `world.upgrade_building`, `world._crowd_factors`, `services.coverage`] | ✅ |

## Details and simplifications

**Zoning & development.** ✅ R/C/I/O zones, low/high density per tile, levels 1–5 with the roadmap's conditions (L3
sewage + garbage, L4 schools / skilled workers + land value, L5 land value + low crime + park), wealth classes, high-rise
ban, industry sub-types (agriculture, forestry, ore, oil, high-tech), ✅ mixed-use zones (M: flats above shops), ✅ historic
preservation (listed buildings keep their level, add land value and tourists).

**Demand.** ✅ Split residential demand (`R_low`, `R_high`), commercial demand from household spending and tourists,
industrial demand from population plus export demand (spare freight capacity), office demand from the educated workforce.
✅ Demand caps: residential/commercial capped without a train station or harbour above 6k residents and without enough
parks above 2k.

**Population.** ✅ Aggregate `cohorts[age, education]`, compressed ageing, births, deaths (worse when unhealthy), schooling
limited by seats in reach, migration fills housing, job matching by education, unemployment per education level.
🟡 Population size is still set by housing capacity; cohorts describe who lives there.

**Economy.** ✅ Taxes per sector with district modifiers, funding levels, policies, fares, trade fees, tourism, power
sales, service imports (can be stopped), loans, bankruptcy. ✅ Household budgets. ✅ Company margins, closures after six
months of losses. 🟡 Goods are two commodities (goods, raw) with fixed prices (lower in a recession).

**Land value.** ✅ Parks, landmarks, water view (worse when the river is dirty), centre, services, police/fire/health/school
coverage minus pollution, crime, garbage, noise, polluted river.

**Services.** ✅ Police (+ jails), fire (+ spread, call-out capacity), health (clinics, hospitals, sick workers), garbage
(landfills fill up, incinerators make power), deathcare, education (elementary, high school, university, libraries),
shelters. ❌ Communications/post, vehicles as a separate capacity (coverage capacity stands in).

**Utilities.** ✅ Power (coal, gas, solar, wind with a wind map, hydro dams, nuclear), water, sewage, treatment, ageing,
filters, expansions, overhauls, tiers and crowding (see Tiers), imports and surplus sales.

**Pollution & environment.** ✅ Air (industry by type, plants, traffic) pushed downwind by a monthly wind (prevailing
westerly), river water (sewage + runoff, downstream flow), noise, ✅ soil (slow build-up under dirty industry, landfills,
coal; hurts health, land value and farms; health can clean it), ✅ seasons (winter heating + snow on roads, summer
cooling + fire risk, seasonal tourism).

**Transport.** ✅ Streets / avenues / highways (no zone access next to highways), congestion, commute times feeding
happiness and demand, slower emergency services, bus lines (stops, radius, capacity, fares, free transit policy), train
stations (second outside connection), freight trucks, harbour at the river mouth, ✅ metro (stations + underground lines,
fast and jam-free), ✅ airports. ❌ Parking, one-way roads.

**Happiness & feedback.** ✅ 14 weighted needs (rich households weigh pollution, safety, noise, leisure more), complaints
top-5 in every observation and the viewer, approval and recall.

**Events & disasters.** ✅ Seeded schedule (`[game] events = "auto" | "none" | [...]`): river flood (3-month forecast,
levees protect, shelters reduce casualties, riverside facilities offline), heatwave, recession, factory relocation offer.
❌ Tornadoes, earthquakes, fire storms.

**Regions & progression.** ✅ Districts, milestones with rewards and unlocks (locked actions return a clear message; the
observation lists what is locked), landmarks, region mode with two teams, shared council, river coupling and neighbour
power/water deals. ✅ Multi-tile footprints (2×2 hospitals, universities, stadiums, casinos; 3×3 airports).
✅ Region collaboration: links (highway, rail, boat, air), inter-city trade in goods and raw materials, commuting, regional
tourism, casinos and gambling (low education, crime, tourist money) vs high tech (graduates), specialisation reports.

**Tiers.** ✅ Ladders (`catalog.TIER_LINES`, tier 1 = the existing `CATALOG` / `SERVICES` dict)
for power (coal, gas, wind, solar), water (pump, tower), sewage (outlet → treatment → advanced treatment), garbage
(landfill → recycling centre → waste complex), deathcare (cemetery → crematorium), health (clinic → hospital → medical
centre), police (station → HQ), jail (jail → prison), fire (station → HQ) and schools (large elementary / high school),
unlocked at 1k / 5k / 10k. New single-tier nuclear plant (10k, riverbank 2×2, uses cooling water). The tier lives in
`facility_state[anchor]["tier"]`; `world.facility_spec()` returns the active spec for everything that reads a placed
building (refresh, coverage, upkeep, pollution, garbage, deathcare, modules). Build a tier fresh (`"tier": n` or a
tier name as `kind`) or `upgrade_facility {"module": "tier"}` for the price difference × 1.2 (grows into free tiles,
may move the anchor, swaps outlet → treatment and clinic → hospital). Pressure to move up: crowding (pumps, outlets,
towers, turbines lose 15% per neighbour of the same group, floor 40%) and downtown cover (level 4–5 tiles get 60% of
tier-1 police / fire / health cover; coverage is computed per radius group and a tier-2+ source gives full cover).
Recycling centres and waste complexes never fill; crematoria have no plots. Overrides: `TIERS.<line>.<tier>.<field>`.
🟡 Tiers don't change a building's look in the ASCII map (only the viewer draws them bigger).

**Roles.** Mayor, Planning, Utilities, Transport, Police, Fire & Emergency, Health & Sanitation, Education. Turn order is
`game.roles`; `game.simultaneous = true` makes every role see the same start-of-month state.
Agents: `llm` (one JSON reply per turn), `scripted`, `remote`, RL, and `fleet` (`civitas/fleet/`): tool-calling agents
where the mayor manages a city plan and budgets and the departments carry it out. Each turn is a tool loop over look-ups,
the role's actions (run at once through `apply_action`) and memory tools, with a plan, per-department memory and a
history log between turns, coach playbooks in the system prompt and an end-of-run `fleet_report`. 🟡 Single-city only;
the supervised mode isn't scheduled yet (no pause / resume) and there is no scripted fleet.
