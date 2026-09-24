# Building tiers — design

Goal: Cities: Skylines-style service tiers. A water pump is fine for a 1,000-person town; a 10,000-person city needs
bigger, more efficient infrastructure. Today every category has one building that scales linearly forever (the
scripted seed-0 baseline ends at 14k people with 23 wind turbines, 8 pumps, 8 raw outlets, 14 police stations,
12 fire stations, 12 landfills, 13 elementary schools and no treatment plant).

## Decisions

- Pressure = **efficiency + crowding**: higher tiers give far more output per $ of upkeep and per tile; tier-1
  production buildings lose output when clustered; tier-1 services under-serve level 4–5 ("downtown") tiles.
  No hard caps.
- Tiers are reached **both** ways: upgrade an existing building in place, or build a higher tier fresh.
- Scope: power, water, sewage, garbage, deathcare, health, police, jail, fire, education.
- Tier stored per facility in `facility_state` (not in `world.level`, which some sim code reads without a zone
  mask). `env.frame()` writes it into the frame's per-tile `levels` string, so `FRAME_FORMAT` stays 3.
- Scoring formulas unchanged. The scripted agents are retuned (not the balance) to keep the 60–80 band.

## Core model

- `catalog.TIER_LINES = {line: [tier1_spec, tier2_spec, ...]}`. Each spec: `name`, `struct`, `tier`, `cost`,
  `upkeep`, output fields (`power`, `water`, `sewage`, `discharge`, `capacity`, `radius`, `seats`, `collect`, `fill`,
  `rate`, `plots`, `holding`, `responses`, `pollution`, …), `footprint`, `unlock` (a `UNLOCK_POP` key or `None`),
  `buildable` (can be built fresh).
- Tier-1 specs are the existing `CATALOG` / `SERVICES` entries (same dict objects), so old code paths and
  `CATALOG.*` / `SERVICES.*` overrides keep working. New overrides: `TIERS.<line>.<tier>.<field>`.
- `facility_state[(x, y)]["tier"]` (default 1); `world.facility_spec(x, y)` returns the active spec. Everything that
  reads a spec for a *placed* building (refresh, coverage, upkeep, pollution, garbage, deathcare, modules) goes
  through it. `coverage()` takes per-source capacity and radius.
- Build fresh: `build_* ... "tier": 2`. `kind` aliases map to (line, tier): `kind=treatment` → sewage tier 2,
  `build_hospital` → health tier 2.
- Upgrade in place: `upgrade_facility {x, y, module: "tier"}` costs `(next.cost − cur.cost) × 1.2`, keeps age and
  modules, grows the footprint (extra tiles must be free, else a clear error), swaps the struct when the tier's
  struct differs (outlet → treatment, clinic → hospital).
- `expansion` still stacks (+50 % of the tier's output); `overhaul` / module costs use the current tier's cost.

## Ladder

Milestones: Town 1k, City 5k, Metropolis 10k. Numbers are starting points for tuning.

| Line | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| pump (riverbank) | water pump: 250 water, $1,500, $25/mo | pumping station (Town): 700, $4,000, $45 | water works (Metro, 2×2): 2,000, $12,000, $100, intake quality ≥ 0.85 |
| tower (anywhere) | water tower: 120, $2,200, $30 | inland water plant (City): 600, $7,000, $70 | — |
| sewage (riverbank) | outlet: 250, discharge 1.0, $1,200, $15 | treatment plant (Town): 700, discharge 0.2, $5,000, $50 | advanced treatment (Metro, 2×2): 2,000, discharge 0.05, $14,000, $110 |
| coal | coal plant: 250, pollution 25 | advanced coal (City): 500, pollution 12, $9,000, $120 | — |
| gas | gas plant: 200, pollution 10 | combined-cycle (City): 450, pollution 6, $12,000, $90 | — |
| wind | wind turbine: 40 | advanced turbine (Town): 100, $5,000, $20 | — |
| solar | solar farm: 70 | solar plant (Metro, 2×2): 400, $16,000, $50 | — |
| nuclear (new kind, riverbank) | — | — | nuclear plant (Metro, 2×2): 2,000 clean, $40,000, $400, consumes 100 water |
| garbage | landfill: collect 450, fills 30,000 | recycling centre (Town): collect 900, never fills, pollution 1.5, $6,000, $50 | waste processing complex (Metro, 2×2): collect 3,000, +60 power, pollution 5, $20,000, $150 |
| deathcare | cemetery: 30/mo, 2,500 plots | crematorium (Town, new struct): 120/mo, no plots, $5,000, $35 | — |
| health | clinic: 2,500, r10 | hospital (City, 2×2): 9,000, r14 (existing) | medical centre (Metro, 2×2): 22,000, r20, $20,000, $220 |
| police | police station: 4,000, r14, 25 cells | police HQ (City): 14,000, r22, 100 cells, $9,000, $130 | — |
| jail | jail: 250 | prison (Metro, 2×2): 1,200, $14,000, $120 | — |
| fire | fire station: cap 260, 5 responses, r14 | fire HQ (City): cap 900, 15 responses, r20, $9,000, $130 | — |
| elementary | elementary school: 450 seats | large elementary (Town): 1,200, $6,000, $80 | — |
| high school | high school: 700 seats | large high school (City): 2,000, r26, $11,000, $130 | — |

Single-tier: incinerator, hydro dam, university, library, shelter, levee, transport buildings, landmarks.
New tile characters only for nuclear and crematorium; every other tier reuses its line's character.

## Pressure rules

**Crowding (production).** `crowd = max(0.4, 1 − 0.15 × n)`, `n` = other buildings of the same group within the
group radius (distance between footprints). Every tier counts as one neighbour.

| Group | Members | Radius |
|---|---|---|
| river intake | pump line | 4 |
| river outfall | sewage line | 4 |
| aquifer | tower line | 5 |
| wind wake | wind line | 2 (stacks with the wind map) |

Coal, gas, solar, nuclear: no crowding (pollution already punishes clustering).

**Downtown (services).** Tiles at level ≥ `DOWNTOWN_LEVEL = 4` get `T1_DOWNTOWN = 0.6` of tier-1 police, fire and
health coverage: `cov = max(0.6 × cov_all, cov_tier2plus)` on downtown tiles.

**Existing pressure with a new exit.** Landfills / cemeteries fill (recycling centres / crematoria don't); outlets
dirty the river pumps draw from (water works filter).

**Visibility.** Plant / facility rows in the observation show tier, crowding (`crowd 0.70 (3 intakes within 4)`) and
the next upgrade (`→ pumping station $3,000 (+450)`); service sections report `downtown tiles lacking tier-2 cover`;
suggested sites skip crowded spots; the locked list shows the next tiers. `World._metrics()` adds
`downtown_uncovered` and per-line tier counts. Budget ~+150 prompt tokens per role.

## Touch points

- `sim/catalog.py`: `TIER_LINES`, `CROWD_GROUPS`, `DOWNTOWN_LEVEL`, `T1_DOWNTOWN`, `TIER_UPGRADE_PREMIUM`,
  `NUCLEAR`, `CREMATORIUM`, tier-aware `upkeep_of`, footprint by (struct, tier), `UNLOCK_POP` keys, overrides.
- `sim/world.py`: `facility_spec`, `facility_tier`, `crowd_factor`, tier-aware `refresh()` (nuclear cooling water),
  `place_building` / `footprint_tiles` with tier, `next_tier` / `upgrade_building` (grows into any free block that
  contains the old footprint, moving the anchor if needed), `frame_levels()` / `multi_buildings()` for frames, metrics.
- `sim/services.py`: per-source capacity / radius, downtown rule, recycling / crematorium behaviour, jail tiers.
- `sim/economy.py`: upkeep, modules, overhaul by tier.
- `core/actions.py`: `tier` param, kind aliases, `upgrade_facility module="tier"`, `action_docs()`, nuclear.
- `core/observation.py`: tier / crowding / upgrade info, downtown gaps, locked tiers.
- `agents/scripted.py`: utilities upgrades crowded / best-value plants and builds tier 2/3 fresh; police / fire /
  health upgrade the station nearest downtown when `downtown_uncovered > 0`; health upgrades full landfills and
  cemeteries.
- Viewer: tier variants by level in `recipes.js` and `scene.js`, nuclear and crematorium models, `NAMES` / `MINI`
  per tier in `app.js`, 2×2 tier-3 buildings in `multi`; old replays (level 0) read as tier 1.
- Docs: README tiers table, actions, override keys, reference numbers; `docs/MECHANICS.md` row.

## Testing

- `tests/test_tiers.py`: unlock gating; fresh-build cost; in-place upgrade (cost, age / modules kept, footprint growth,
  blocked footprint, struct swap, top tier); crowding values (1 / 3 / 6 pumps, spread-out pumps, one water works);
  downtown 60 % → 100 % with a police HQ; recycling / crematorium never fill; nuclear water draw; backward compat
  (`kind=treatment`, `build_hospital`, old overrides, old replays); frames carry tier in `level` and 2×2 in `multi`;
  determinism.
- Benchmark change, called out: retune scripted agents until seeds 0–4 are solvent and in the 60–80 band (target ≈ 77);
  update reference numbers in `CLAUDE.md` / README; `noop` / `random` still 0; episode time still ~4 s.
- Prompt check with `civitas prompt` per role and a smoke run with trinity-large-thinking.

## Build order

catalog/specs → sim (refresh, coverage, upkeep) → actions → observation → scripted tuning + benchmark → viewer → docs.
`pytest -q -m "not slow"` green after each step.
