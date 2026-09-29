# Civitas Bench

A SimCity-style city simulation built for **multi-agent AI benchmarking**.
Each city department is played by a separate agent (an Arcee LLM, a scripted
baseline, a remote program, or an RL policy). All of them read from one central
city state, act only through validated actions, talk on a shared council board,
and are scored on the city as a whole *and* on their own department.

The map starts as a flat plain with a river flowing north to south. The highway
on the west edge connects the city to the rest of the region.

```
roles:    Mayor · Planning & Housing · Utilities (power, water, sewage) · Transport
          Police · Fire & Emergency · Health & Sanitation · Education
episode:  120 months (10 years), each agent acts once a month
cost:     scripted run ≈ 4–5 s · every role on an LLM ≈ 960 model calls
modes:    one city, or a region of 2-6 linked cities that trade and specialise
```

---

## 1. Setup (macOS)

You need Python 3.10 or newer. The Python that ships with macOS is usually 3.9, so
the simplest path is [uv](https://docs.astral.sh/uv/):

```bash
cd ~/Desktop/civitas-bench
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q                       # ~150 tests, ~25 s (pytest -q -m "not slow" skips the 5-episode benchmark test)
```

(With plain pip: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`.)

## 2. Open the control center

```bash
civitas serve            # then open http://127.0.0.1:8765
```

Everything happens in the browser from here. You can:

- **Watch** the city in the OpenTTD-style pixel view (the default) or in 3D (toggle with `X`
  or the `PX`/`3D` button). The pixel view is a 2:1 isometric pixel-art map: drag to
  scroll, zoom in steps with the wheel, rotate in quarter turns with `Q`/`E`. Buildings go
  up behind scaffolding, cars, buses, trains, boats and planes move around, cars queue at
  traffic lights on junctions, planes land, taxi to the terminal and take off (in region runs
  they fly to the other cities' airports), container ships berth under the harbour crane, and night lights the windows. Around the city a distant view
  fades into haze: fields, the river and highways running on, villages where roads leave the
  map and, in region runs, the neighbouring cities with the rivers flowing between them and
  their highway and rail links routed from connector to connector (`T` zooms out over all of it). Both views
  show the river (a little murkier where it is polluted), bridges,
  houses, row houses and villas that become apartments and towers, offices, shop-houses,
  factories, farms, sawmills, mines and oil fields, power plants (nuclear too), pumps, sewage works,
  police and fire stations, clinics, 2×2 hospitals and universities, schools, landfills,
  crematoria, higher-tier buildings drawn bigger or with annexes,
  bus and metro lines, casinos, airports, levees, landmarks, floods, historic plaques,
  snow in winter and orange trees in autumn. Smoke drifts with the wind.
- **Replay or follow live.** Use the timeline to play, step or scrub. Marks on the
  timeline show council messages (purple), failed actions (amber), money trouble (red),
  disasters and offers (blue) and milestones (gold). A run in progress is followed
  automatically.
- **See why things happen.** The Agents tab shows each month's decisions with ✓/✗
  results. Click an action to fly the camera to where it happened. For LLM agents you
  can expand the model's full reasoning, its raw reply and the exact prompt it saw. A fleet
  turn also has a **Tool calls** panel: every call in order with its arguments, outcome (ok,
  failed, refused or repeat), cost and result, one line each (click one to expand it), under
  a header counting calls, actions, failures and how the turn ended. The
  Events tab starts with what residents complain about this month.
- **Inspect** any tile by clicking it: stats, services, wealth or industry type, its
  building level over time, and which agent zoned or built it.
- **Switch map layers** (press `V`, or `1`–`9`): pollution, land value, happiness, noise,
  power, water, sewage, river pollution, traffic, bus coverage, police/fire/clinic/school
  cover, crime, fire risk, health, garbage, soil pollution, historic buildings, business
  profit, affordability, natural resources, flood risk and districts.
- **Launch runs** with **＋ New run**: pick a config, seed, length and each role's agent
  (scripted, LLM with any model, fleet, noop or random); each role starts at the config's own
  agent and only the roles you change are overridden. Stop runs from the Runs tab, which also
  holds the leaderboard.
- **Browse the Charts tab**: population, scores, money, happiness and approval, demand,
  jobs, pollution, commute and traffic, trade, service coverage, crime and health,
  education.
- **Region runs** get a city switch in the top bar and a **Region** tab: a map of the
  cities with their links and the goods, commuters and tourists flowing between them,
  plus each city's score and specialisations.

Press `?` in the app for all keyboard shortcuts. `civitas run ... --serve` also opens the
interface while a run from the terminal is playing. The old 2D viewer is still available
at `/classic.html`.

## 3. Plug in Arcee models (trinity-large-thinking)

Put your key in a `.env` file in the project folder. It's gitignored and `civitas` loads
it automatically. `export ARCEE_API_KEY=...` works too.

```bash
echo 'ARCEE_API_KEY=rcai-...' > .env
civitas check-llm                            # one test call to trinity-large-thinking
civitas run -c configs/smoke.toml --serve    # 2 years, 3 LLM roles acting every 3 months (~24 calls)
civitas run -c configs/arcee.toml --serve    # 10 years, mayor/planning/utilities on the LLM, the rest scripted
civitas run -c configs/arcee_full.toml       # 10 years, all 8 roles on the LLM (~960 calls)
civitas run -c configs/fleet_smoke.toml      # 1 year, all 8 roles as a tool-calling fleet (see Agent fleets)
civitas run -c configs/fleet_region_smoke.toml   # 1 year, two cities: the fleet in north, scripted south
```

The Arcee API is OpenAI-compatible (`https://api.arcee.ai/api/v1`). Notes on the
thinking model:

- **Token budget.** The Arcee configs give each call `max_tokens = 12000` so there's room
  for the reasoning *and* the JSON answer. If a reply is cut off, the agent retries once,
  hands the model the end of its own reasoning and asks for the JSON only; as a last resort
  it recovers the action objects from the end of the reasoning. Prompts are about 3–4.5k
  tokens per role (a system prompt with the rules and action docs, then the observation).
- **Where the answer comes from.** The agent reads plain JSON from the reply rather than
  using tool calling. If the final answer has no JSON, the agent checks the reasoning text.
- **What gets logged.** The reasoning is saved in `trajectories.jsonl` along with the
  answer, which is useful for studying or training on how the model plans.
- **Speed.** Calls are slow, so a full game with several LLM roles can take hours. Start
  with `smoke.toml`, or set `game.act_every` to 2–3.
- **Stalls.** A call attempt is retried after `timeout` s (300) without data or `deadline` s (1,800) in all, both timed on the wall clock: after a laptop sleep the attempt ends on waking and its `[llm] ... retrying` log line says `system slept ~N s`.

Any role can be switched from the CLI, and any value overridden:

```bash
civitas run -c configs/arcee.toml --agent mayor=scripted --agent police=llm:arcee/trinity-mini --set game.act_every=3
```

Other OpenAI-compatible endpoints (OpenRouter, vLLM, Ollama and so on) work the same way.
Add a `[providers.name]` block with `base_url` and `api_key_env`, then use `llm:name/model`.

## 4. Benchmark

```bash
# the same line-up across 5 seeds (maps and disaster schedules differ by seed)
civitas bench -c configs/arcee.toml --seeds 0-4 --workers 3

# role-isolated eval: only Planning is an LLM, the other roles are fixed baselines,
# so any score difference comes from the Planning agent
civitas bench -c configs/eval_planning.toml --seeds 0-4
civitas bench -c configs/eval_planning.toml --seeds 0-4 --agent planning=llm:arcee/trinity-mini   # compare models

# the fleet reference: the scripted baselines played through the fleet tools on the fleet's settings (free)
civitas bench -c configs/fleet_scripted.toml --seeds 0-4 --workers 5

civitas leaderboard            # every finished run, grouped by agent line-up
```

Baseline reference (all scripted, seeds 0–4): **city 77.5 ± 1.5**, mayor 88.1, planning
68.4, utilities 89.4, transport 75.9, police 74.9, fire 85.7, health 86.6, education 83.2,
population ≈ 14.5k, no bankruptcies, ~4–5 s per game (seeds 5–9: city 75.9 ± 0.8). `noop` and
`random` line-ups score 0. Building tiers changed these numbers. Just before tiers were
added, the same baseline scored city 78.1 ± 1.8 with population ≈ 14.7k.
Acting every 3 months, as the fleet configs do, the same agents reach city 67.0 ± 2.8 and
population ≈ 7.3k; `configs/fleet_scripted.toml` plays them through the fleet tools and
scores exactly that (see [Agent fleets](#agent-fleets), "Scripted fleet").
The scripted agents leave obvious room: they never use districts, mixed use, historic
listings or soil clean-up, rarely use policies or funding levels, build services by simple
coverage rules, hoard money late in the game and never make neighbour deals.

Useful switches in `[game]` (or `--set game.X=...`):

| setting | effect |
|---|---|
| `events = "auto"` / `"none"` | seeded floods, heatwave, recession and a factory offer (same on every line-up for a seed) |
| `milestones = true` | buildings unlock at 1k / 5k / 10k residents |
| `simultaneous = true` | every role sees the same start-of-month state (harder: no reacting to earlier roles) |
| `roles = [...]` | which roles play, in turn order |
| `seasons = true` | winter heating and snow, summer cooling and fires, seasonal tourism |
| `region = true`, `cities = [...]` | 2–6 cities, each with its own team (see below) |
| `team_weight` | reward = w·Δcity + (1−w)·Δrole − penalty·invalid actions |
| `balance` | override any constant, e.g. `"CATALOG.coal.cost" = 3000`, `"SERVICES.clinic.radius" = 9` or `"TIERS.pump.2.water" = 800` |

---

## How it works

```mermaid
flowchart LR
  subgraph Agents
    M[Mayor] --- P[Planning] --- U[Utilities] --- T[Transport]
    PO[Police] --- F[Fire] --- H[Health] --- E[Education]
  end
  subgraph Hub["CivitasEnv (central city state)"]
    O[role-scoped observation<br/>JSON + text prompt]
    A[action validator<br/>budgets, permissions, unlocks]
    B[(council board)]
    W[(World: map, networks, services,<br/>people, traffic, economy, events)]
    S[scores & rewards]
  end
  Agents -- observe --> O
  Agents -- actions + messages --> A
  A --> W
  A --> B
  W --> O
  B --> O
  W -- monthly tick --> S
  S --> O
  S --> L[runs/*: frames, trajectories, summary]
  L --> V[web viewer / leaderboard]
```

**The central state.** `CivitasEnv` is the hub. It owns the `World` (the single source
of truth), the council board, each role's memory (the results of its last turn and its
notes to self) and the scores. Agents never touch the world directly. Each month every
role observes and acts in turn order (later roles see what earlier roles just did), then
`end_month()` simulates the month and recomputes the scores. Each role's reward is the
change in score.

Every observation is available as structured JSON (for RL policies and remote agents)
and as a rendered text prompt (for LLMs). Everyone sees the same city summary, the top
complaints, the finances, events, forecasts and council messages; each role also gets a
detailed section on its own system (its buildings with load vs capacity, problem hotspots
with coordinates, suggested sites) and, except the mayor, the ASCII map. Other systems
only appear as summaries, so agents have to talk to each other.

**Five ways to connect an agent.** All use the same actions:

| kind | what it is |
|---|---|
| `llm` | Any OpenAI-compatible model (Arcee by default). It gets a system prompt with the rules and action docs, plus the observation text, and replies with JSON. |
| `fleet` | A tool-calling model: a short briefing, then look-up and action tools in a loop, with a city plan and memory between turns. See [Agent fleets](#agent-fleets). |
| `scripted` | Rule-based baselines. They read only the observation, so they are fair references and a source of demonstration data. |
| `remote` | Any external process, in any language, over HTTP: `GET /api/live/turn?role=planning`, then `POST /api/live/act`. See `examples/remote_agent.py` and `configs/remote.toml`. |
| RL | `env.reset()` / `env.step({role: Decision})` (PettingZoo-style), with `env.tensor_observation()` returning a 14×H×W grid and a feature vector. See `examples/rl_loop.py`. |

### The simulation

- **Zones and levels.** Residential (R), commercial (C), industrial (I) and office (O)
  zones develop by themselves when there is demand. Each level needs more of the city:
  L1 road access + power, L2 water, L3 sewage + garbage pickup + land value 40%, L4 a
  school in reach (homes) or skilled workers (jobs) + land value 58%, L5 land value 75% +
  low crime + a park nearby. Low-density zones stop at L2; high density goes to L5 (up to
  240 residents or 140 office jobs per tile). Industry stops at L3 unless it is high-tech.
- **People.** Residents are tracked as an age × education vector (children, teens, adults,
  seniors × none, elementary, high school, university). Migrants fill new housing, people
  age and die, and students only graduate when they have a school seat. Jobs need education:
  offices and high-tech want graduates, over-educated workers take lower jobs.
- **Wealth and households.** Homes are low, middle or high income depending on land value
  when they are built. Wages minus rent is what households spend in shops; rich households
  pay more tax, care more about pollution, crime and noise, and leave first when unhappy.
- **Industry types.** Industry zoned on natural resources becomes farms (fertile soil),
  sawmills (forest), mines (ore) or oil fields; resources run out. Educated cities attract
  high-tech industry.
- **Demand** depends on jobs vs. workers, taxes, happiness, commute times, household
  spending and export demand. Big cities need a train station or harbour, and enough parks,
  or residential and commercial demand is capped.
- **Utilities.** Every connected road network is its own grid for power, water and sewage.
  Facilities serve the network they touch, nearest buildings first. Pumps and sewage
  facilities sit on the riverbank, and clustered pumps, outlets, towers and turbines lose
  output (see [Building tiers](#building-tiers)). Plants lose output after five years unless overhauled;
  wind output follows a wind map. Shortfalls are covered by costly imports from the region
  (the mayor can stop them) and surplus power is sold.
- **The river** flows north to south. Sewage outlets and riverside industry pollute it
  downstream: pumps below an outlet deliver dirty water. Treatment plants remove 80% (advanced
  treatment 95%).
- **City services** (police, fire, clinics and hospitals, schools, garbage trucks,
  shelters) cover buildings they can reach along the road network within their radius, up
  to their capacity, scaled by the department's funding level. Crime grows with
  unemployment, density, low education and low land value; uncovered fires spread;
  uncollected garbage piles up; pollution, dirty water and garbage make people sick (sick
  workers stay home); deaths need cemetery space or a crematorium; landfills fill up
  (recycling centres don't).
- **Traffic.** Workers commute from home to the busiest job areas along shortest road
  paths (a gravity model); trucks carry goods to the highway, train stations and the
  harbour. Load over capacity means congestion: longer commutes, noise, exhaust, and slower
  police, fire and ambulances. Avenues and highways carry more; bus lines take trips off
  the road.
- **Trade and businesses.** Industry makes goods and raw materials; shops need goods.
  Surpluses are exported and shortfalls imported, limited by freight capacity. Every shop,
  factory and office has a profit margin (customers, taxes, shortages, traffic, crime);
  six months of losses and it closes.
- **Happiness** is a weighted sum of 14 needs (power, water, sewage, garbage, safety,
  health, education, commute, leisure, noise, pollution, tax, affordability, jobs). The
  needs that cost the most happiness become the **complaints feed**. **Approval** follows
  happiness and taxes; 12 months below 30% and the mayor loses the recall election.
- **Mixed use** (M zones) puts shops below flats; **historic listings** freeze a
  building's level and add land value and tourists; **soil pollution** builds up under dirty
  industry, landfills and coal plants, fades very slowly, and hurts health, land value and
  farms (the health department can clean it); **wind** changes every month around a
  westerly and pushes air pollution downwind; **seasons** raise power demand in winter
  (and slow the roads with snow), water and fire risk in summer, and move tourism.
- **Metro**: stations serve 5 tiles around them; lines between stations carry trips
  three times faster than driving, immune to jams. Hospitals, universities, stadiums
  and casinos cover 2×2 tiles, airports 3×3 (x, y is the top-left tile).
- **Casinos and airports**: a casino gives low-skill jobs and gamblers who pay a gambling
  tax, but raises crime around it; an airport adds freight, tourists and air links.
- **Events** come from the seed: a river flood (forecast 3 months ahead; levees protect,
  shelters save lives), a heatwave, a recession and a factory offer the mayor must accept
  or decline (jobs and a bonus vs. heavy pollution).
- **Milestones** at 1k / 5k / 10k / 20k residents pay a reward and unlock buildings (high
  density, offices, avenues, high schools... then hospitals, universities, train stations,
  hydro dams, stadiums, the harbour... then monuments). Landmarks raise land value and
  bring tourists.
- **Money.** Taxes (per sector, with district modifiers), fares, trade fees, tourism and
  power sales come in; upkeep of roads, plants and service buildings (times funding
  levels), policies, imports and loan interest go out. Departments can only spend what the
  mayor gives them. Twelve months in a row with a negative treasury means bankruptcy.

All balance numbers live in `civitas/sim/catalog.py`. You can override them per config
with `[game.balance]`.

### Building tiers

Most service and utility buildings belong to a ladder, like the ones in Cities: Skylines
(`catalog.TIER_LINES`). Higher tiers unlock at the Town (1k), City (5k) and Metropolis (10k)
milestones. They are bigger buildings with much more output per dollar of upkeep, so a
growing city replaces a cluster of small buildings with a few large ones. Each cell below
shows the milestone, the main output, the build cost and the upkeep per month.

| line | tier 1 | tier 2 | tier 3 |
|---|---|---|---|
| pump (riverbank) | water pump: 250 water, $1,500, $25 | pumping station (1k): 650, $4,000, $45 | water works (10k, 2×2): 2,000, $12,000, $100, filters dirty river water |
| tower | water tower: 120 water, $2,200, $30 | inland water plant (5k): 600, $7,000, $70 | |
| outlet (riverbank) | outlet: 250 sewage, raw, $1,200, $15 | treatment plant (1k): 700, removes 80%, $5,000, $50 | advanced treatment (10k, 2×2): 2,000, removes 95%, $14,000, $110 |
| coal | coal plant: 250 power, pollution 25, $4,000, $80 | advanced coal (5k): 500, pollution 12, $9,000, $120 | |
| gas | gas plant: 200 power, pollution 10, $6,500, $60 | combined cycle (5k): 450, pollution 6, $12,000, $90 | |
| wind | wind turbine: 40 power, $2,500, $12 | advanced turbine (1k): 100, $5,000, $20 | |
| solar | solar farm: 70 power, $5,000, $15 | solar plant (10k, 2×2): 400, $16,000, $50 | |
| landfill | landfill: collects 450/mo, fills at 30,000, $2,000, $30 | recycling centre (1k): 900, never fills, $6,000, $50 | waste complex (10k, 2×2): 3,000, never fills, +60 power, $20,000, $150 |
| cemetery | cemetery: 30/mo, 2,500 plots, $1,500, $12 | crematorium (1k): 120/mo, no plots, $5,000, $35 | |
| clinic | clinic: capacity 2,500, radius 10, $2,000, $40 | hospital (5k, 2×2): 9,000, r14, $8,000, $120 | medical centre (10k, 2×2): 22,000, r20, $20,000, $220 |
| police station | station: 4,000, r14, 25 cells, $2,500, $55 | police HQ (5k): 14,000, r22, 100 cells, $9,000, $130 | |
| jail (1k) | jail: 250 inmates, $4,000, $45 | prison (10k, 2×2): 1,200, $14,000, $120 | |
| fire station | station: capacity 260, 5 call-outs, r14, $2,500, $55 | fire HQ (5k): 900, 15 call-outs, r20, $9,000, $130 | |
| elementary | school: 450 seats, $2,500, $45 | large elementary (1k): 1,200, $6,000, $80 | |
| high school (1k) | school: 700 seats, r22, $5,000, $75 | large high school (5k): 2,000, r26, $11,000, $130 | |

The **nuclear plant** (10k, map character `u`) is a new single-tier building. It is a 2×2
plant on the riverbank that makes 2,000 clean power for $40,000 and $400/mo, and it draws
100 water for cooling from its road network. The crematorium has its own character, `z`.
Every other tier keeps its line's character, and frames carry the tier in the per-tile
`levels` string. Hydro dams, incinerators, universities, libraries and shelters have a
single tier.

- **Build a tier directly.** Every facility build action (power, water, sewage and service
  buildings) takes `"tier": n` (counted within the line, 1 = base), or a tier name as `kind`:
  `{"type":"build_water","kind":"pumping_station","x":29,"y":24}`,
  `{"type":"build_police_station","kind":"police_hq","x":14,"y":25}`,
  `{"type":"build_cemetery","tier":2,...}`. Nuclear is `build_power_plant` with
  `kind: "nuclear"`. `kind: "treatment"` and `build_hospital` still work; they build tier 2
  of their lines.
- **Upgrade in place.** `{"type":"upgrade_facility","x":..,"y":..,"module":"tier"}` moves a
  building up one tier. It costs the price difference × 1.2 and keeps the building's age
  and modules. A 2×2 tier grows into free tiles around the building (the top-left tile may
  move), and the action fails with a clear message if none are free. Outlets become
  treatment plants and clinics become hospitals. An expansion adds 50% of the current
  tier's output.
- **Crowding.** Production buildings of the same group lose 15% output for each neighbour
  within the group's radius, down to a floor of 40%. A neighbour of any tier counts. The
  groups are pumps (4 tiles), outlets (4), water towers (5) and wind turbines (2, on top of
  the wind map). Coal, gas, solar and nuclear plants don't crowd. One water works beats
  eight pumps on one stretch of river.
- **Downtown.** Level 4–5 buildings get only 60% of the police, fire and health cover that
  tier-1 buildings give. A tier-2+ building in range gives them full cover.
- **What agents see.** The utilities section shows the crowding for each group, the
  observation offers the best upgrades with their real gains, service sections report
  downtown tiles without tier-2 cover, and suggested sites include riverbank spots with
  room for a 2×2 building. The windy, pump and sewage site lists put uncrowded spots first and mark
  crowded ones (`near 1 turbine`). `World._metrics()` records `tiers` (count per line and tier)
  and `downtown_uncovered`.
- **Overrides.** Use `"TIERS.<line>.<tier>.<field>"`, e.g.
  `"TIERS.pump.2.water" = 800` in `[game.balance]`, a nested `[game.balance.TIERS.pump.2]`
  table, or `--set game.balance.TIERS.pump.2.water=800`. Tier 1 is the same dict as the
  `CATALOG` / `SERVICES` entry, so `"CATALOG.pump.water"` still works. Tiers 2+ are
  copies, so a tier-1 override doesn't reach them. The crowding and downtown rules use
  `CROWD_STEP`, `CROWD_FLOOR`, `T1_DOWNTOWN` and `DOWNTOWN_LEVEL`.

### Roles and actions

| role | actions |
|---|---|
| mayor | `set_tax` (R/C/I %), `allocate_budget`, `set_funding` (department 50–150%), `enact_policy` / `repeal_policy` (city-wide or per district), `set_district_tax`, `accept_offer` / `decline_offer`, `set_service_imports`, `take_loan`, `repay_loan` |
| planning | `build_road`, `zone` (R/C/I/O/M, low or high density), `dezone`, `build_park`, `build_landmark` (plaza/stadium/monument), `build_casino`, `paint_district`, `specialize_district`, `preserve_historic` / `delist_historic`, `bulldoze`, `demolish` |
| utilities | `build_power_plant` (coal / gas / solar / wind / hydro / nuclear), `build_water` (pump / tower), `build_sewage` (outlet / treatment), `upgrade_facility` (tier / filter / expansion / overhaul), `demolish` |
| transport | `upgrade_road` (avenue / highway), `create_bus_line`, `remove_bus_line`, `set_fare`, `build_metro_station`, `create_metro_line`, `remove_metro_line`, `build_train_station`, `build_harbour`, `build_airport`, `build_connector` (region mode), `upgrade_facility`, `demolish` |
| police | `build_police_station`, `build_jail`, `upgrade_facility`, `demolish` |
| fire | `build_fire_station`, `build_levee`, `build_shelter`, `upgrade_facility`, `demolish` |
| health | `build_clinic`, `build_hospital`, `build_landfill`, `build_incinerator`, `build_cemetery` (tier 2: crematorium), `clean_soil`, `upgrade_facility`, `demolish` |
| education | `build_school` (elementary / high / university), `build_library`, `upgrade_facility`, `demolish` |

Policies: recycling, neighborhood watch, smoke detectors, education boost, industry
filters, high-rise ban, free transit. Every role can post `messages` to the council board
(to `all` or a role) and keep `notes` for its next turn. Invalid actions are rejected with
a helpful message and counted in the invalid-action rate. Every facility build action (power,
water, sewage and service buildings) also takes a `tier` (see [Building tiers](#building-tiers)).

### Scores (0–100)

| score | built from |
|---|---|
| **city** | ½ population (√ of pop / 30k) + ½ quality: happiness, finances, power/water/sewage coverage, environment, employment, safety, health, education. Quality counts only once the city has people; bankruptcy halves the total, a lost recall election cuts it by a quarter. |
| **mayor** | revenue, solvency, net worth, happiness, approval, and how much of the money given to departments actually got spent |
| **planning** | growth, share of zoned land that developed, demand balance, land value, road efficiency, employment |
| **utilities** | power, water and sewage coverage, spare capacity in the 10–60% range, cost efficiency, clean-energy share (solar, wind, nuclear), pollution near homes, water quality, river quality |
| **transport** | commute time, share of congested road, bus share of trips, cost per resident |
| **police** | crime at homes, coverage, jail space, cost per resident |
| **fire** | buildings lost to fire, coverage, cost per resident |
| **health** | health of residents, garbage at homes, clinic coverage, landfill room, deathcare, cost |
| **education** | enrolment, skilled jobs filled, share of adults with high school+, cost |

Every score is a function of the monthly history, so the same code produces the final
benchmark number, the live "your score" shown to agents, and dense RL rewards.

### Region mode: cities that trade and specialise

`configs/region.toml` (two cities) and `configs/region_trio.toml` (three) run a region:
`[game] region = true` and `cities = [...]` from north, south, east, west, southeast,
southwest. Every city has its own full team; roles are named `north.planning`,
`south.mayor` and so on. Agents configured for a base role (`[agents.planning]`) play it
in every city, `[agents."south.planning"]` sets one city.

- **The river** runs through the north-south column: upstream sewage and runoff arrive
  in the next city's water.
- **Links.** Transport departments build `build_connector` (highway or rail) on the map
  edge facing a neighbouring city, with a feeder road inland; the link opens when both
  cities have built their side. Any two cities with a harbour are linked by boat, any two
  with an airport by air. Each link type has a goods and a people capacity.
- **Commerce.** Every month, over those links, surplus goods and raw materials go to
  cities that are short (the seller collects part of the value; the buyer avoids costly
  imports and shortages), unemployed workers commute to open jobs, and residents visit
  their partners' casinos, landmarks and historic districts.
- **Specialisation** follows from each city's land and people and is reported to every
  agent: mining & farming (natural resources, pollutes the soil), high tech (needs
  graduates and raw materials), gambling (casinos: little education needed, crime, tourist
  money), tourism, trade (freight), education. Complementary cities gain the most from links.
- **Diplomacy.** All teams share one council board (address `"south"`, `"north.mayor"` or
  `"all"`) and mayors can agree power/water deals (`offer_deal`, `accept_deal`, `cancel_deal`).

Each city is scored on its own; the region score is their average. Fleet agents play region
roles too (see [Agent fleets](#agent-fleets), "Regions").

### Agent fleets

`kind = "fleet"` plays a role with a tool-calling agent (`civitas/fleet/`) instead of the
one-shot JSON `llm` agent. A fleet is built to measure how well a team of agents manages
tasks and money, and to let people coach it.

- **One manager, seven departments.** The mayor manages. It writes the city plan (goals, a
  budget target and a one-line priority per department, notes) and moves the money. The
  seven departments carry the plan out, and each one sees only its own slice. Turn order
  doesn't change, so the mayor acts first each month.
- **A turn is a tool loop.** The agent starts from a short briefing and then calls tools.
  The mayor's briefing covers the city, the finances, each department's funds, target
  (`funds $0 < target $2,500` when it still needs funding), what it needs when its funds
  are short of its cheapest ready building (below), what it spent over the last 3 months,
  its idle money, and its latest report.
  - **Needs line.** It names the cheapest ready building, and says whether it is the
    first:
    - `needs $2,500 for its first police station (5 sites)` while the city has none;
    - `needs $2,500 for another police station (coverage 74%)` once it has one. Clinics,
      landfills, schools and high schools show their coverage too, and bus lines the
      transit share.
    - A hospital counts as a clinic. For utilities, any building that makes the same
      supply counts, and the line shows how much of the supply is used: `another water
      tower (water 117% used)`. A utilities building is named only while the city lacks
      that supply or uses at least 90% of it. In smoke v14–v15 seed 1 the line named a
      water tower with water spare, and towers were built until water was 4.7–4.9×
      demand.
    - On a tie each building at that price is named (`another clinic (coverage 21%) or
      another landfill (coverage 83%)`). A first one is named alone (`its first landfill
      (4 sites)` once a clinic is built).
    - A per-tile site counts its best stretch whole: `needs $90 for 3 avenue tiles (5
      sites)`.
    - Planning's is its best `road_extension` site (a dead end's extension or a branch
      road), priced as `build_road` charges it plus any bulldozing: `needs $150 for a road
      into open land (opens 71 tiles)`. A road is no building, so it is neither a first nor
      another, and it never shows on the first-buildings, leave-free or blocked lines. In
      the live 120-month run on seed 0, at month 9, the mayor read planning as `idle
      $1,243`. It took that back and $1,000 more. Planning's $150 extension then failed
      twice on "the planning budget only has $0" while its line showed no need.
  - **First buildings.** When a department's needs line is for its first building and
    another's is for another one, a line under the block says `- first buildings still
    unfunded: education (elementary school, $2,500): fund these before second buildings`.
    Past 160 characters it gives the department names only. In smoke v16 seed 0 the
    treasury ran short; the mayor funded second police and fire stations and a second
    clinic first, and education never got its school (33.2; v17: 33.3 and 33.4).
  - **Idle money.** A department keeps max(2 × its 3-month spending, its dearest ready
    first building, $1,000); for utilities only buildings the needs line could name
    count. Funds that are $1,000 or more above that, excluding this month's top-ups, show
    as `idle $3,400`, and a summary line says to take the money back (a negative
    `allocate_budget`) before taking loans or giving new top-ups.
  - **Blocked buildings.** A first building with no lot for lack of land or roads counts
    toward the keep, because the department still needs that money. That covers:
    - no free roadside land;
    - no free roadside lot near the homes that need it, or no free roadside land at all,
      for a need-ranked station, clinic or school when at least 15% of what it serves is
      under 30% coverage (a well-covered department with no land isn't blocked);
    - no roadside lot away from homes, for a landfill or jail;
    - no free 2x2 roadside block;
    - for a pump or sewage outlet, no road to the river or the bank taken, but only while
      the city lacks one or its demand is at least 90% of supply. The 90% test also
      applies to a water tower (on water) and a power plant or wind turbine (on power).

    When a department has no ready site, its line says `; blocked: a police station`, and
    one line under the block says `- blocked for lack of land: ask planning for lots`.
    In smoke v13 seed 0, planning had zoned every roadside lot, and the mayor took the
    station, clinic and school money back from four departments as idle.
  - **Below target.** A department counts as below its plan target only if it isn't idle
    and would use a top-up: it spent more than half of the most it held over those 3
    months, or it never held its keep. In smoke v9–v12, planning spent about $1,000 a
    quarter and still read "funds $7,834 < target $9,000", so the mayor topped it up.
  - **Totals.** A line gives the total funding gap. It ends in `allocate_budget`, or in
    "hold" when the treasury is at or under the reserve. The plan comes last.

  A department's briefing covers its plan slice, its budget
  (when its funds buy none of its ready sites, a line names the cheapest, all of them on a
  tie, and tells it to ask the mayor and end the turn instead of calling build tools), its
  key metrics, a "since your last turn" digest (failed actions, events, and the newest
  council messages addressed to it or to `all`, quoted with sender and month), a Ready
  sites block (one line per `find_sites` purpose: the count and best site with the price
  of the cheapest unlocked building the sites are for (a bus line for bus stops; per tile
  for levees, `$300/tile`, and for congested roads, `$30/tile` for an avenue, with
  stretches that are avenues already listed last), and `; $2,500 - you have $1,000` when
  the funds are short; or why there is none, e.g. locked, under 1,500 residents for bus
  stops, no connected road reaches the river, the riverbank beside the roads is taken, no
  free roadside lot near the homes that need it (around a named tile), no free 2x2
  roadside block, or no roadside lot away from homes for a landfill, cemetery or jail; a site on one of
  the last two free riverbank lots is marked to keep for a pump or sewage outlet) and its
  memory. Every purpose outside planning is priced, and so is planning's `road_extension`
  (`opens 71 land tiles; $150 - you have $0`), though never on its budget line. A library
  (it only adds to schools), a
  shelter (it only matters in disasters), a cemetery (the landfill comes first) and a
  train station, harbour or metro station (bus lines come first) are priced on their
  lines, but the mayor's needs line never names one, and the budget line only when nothing
  else is ready (smoke v9's mayor funded fire $1,500 for "a shelter", and fire never
  reached the $2,500 station). Prices are what the build actions charge, and funds count
  as short exactly when the build action would refuse them. Both have a short Map block
  (size, highway entry, river columns by row band, the extent of the connected roads and
  their straight road lines, e.g. `y=24 x 0-47; x=19 y 19-29`, up to 8 with the longest
  first, any road tiles not connected to the highway, and the built-up area) and end with
  the next milestone. While high density is locked, planning's Ready sites block says so
  and tells it to zone at low density. When no dead end extends into open land, planning's
  `road_extension` line (and `find_sites road_extension`) offers branch roads instead:
  `- road_extension: none - no dead end; branch: dezone (23,25) (empty lot, free), then
  build_road (23,25)-(23,39), opens 75 land tiles; $150`.
  - A branch leaves a road tile of the connected network and runs straight into open land
    (up to 15 tiles, bridges allowed). It starts from a free lot beside the road, or else
    through up to 2 empty zoned lots to dezone. Failing those, it bulldozes the cheapest
    buildings (`bulldoze (17,23) (a level-1 home, $5)`).
  - Branches are ranked by land opened (as the dead-end extensions count it), then by
    fewer tiles to clear, then by distance to the town hall. `find_sites` lists up to 6,
    each with its cost, the road tile it leaves and what to run first.
  - In smoke v17 seed 0 the only road crossed the map with every tile beside it zoned or
    built. Planning read "no straight extension from a dead end" and spent three turns on
    roads into buildings and past the map edge.

  Two more lines there are about the other departments' buildings:
  - **`- leave free (funded): fire station (20,22), clinic (21,25): zone around them`**
    covers each department whose funds already pay for its cheapest ready first
    building (all of them on a tie; utilities' only while that supply is at least 90% used). It gives each building a site, in turn order,
    that no earlier building has claimed. Past 240 characters whole items are dropped
    (`(+2 more)`).
    - Zone strips that cover a claimed site are listed after the others. When one still
      comes first, it's marked `(leave (20,22) for the fire station)`.
    - A failed zone's "zone beside it" suggestions skip these sites.
  - **`- blocked for lack of land: utilities (sewage outlet: a road to the river), police
    (police station: near homes): extend a road into open land, or free a lot, for
    them`** gives each department's cheapest blocked first building, in turn order, and
    what frees it. Past 240 characters the tags are dropped (`away from homes` first),
    then the building names.
    - A landfill or jail is tagged `(landfill: away from homes)`, and a line under it says
      where to leave its lot: `- away from homes (health's landfill): leave a lot free by
      (12,9), the far end of road_extension (12,23)-(12,9), with no homes within 4 tiles`.
      It names the first extension or branch whose far end has under 3 homes within 4
      tiles; without one it says to extend a road into open land at the town edge. In
      smoke v15 seed 1 the landfill read "no roadside lot away from homes" from month 6,
      planning never left one, and garbage coverage stayed 0 all run.

  In smoke v14 seed 0, planning zoned the lots near homes after the mayor had funded fire,
  health and education, and fire held $2,500 for nine months without a site.

  Actions run immediately and return `ok`, a message,
  the cost and the money left, so a failed action can be fixed in the same turn. Bad arguments, unknown tools and
  refused actions come back as `{"ok": false, "error": ...}` for the model to correct. A
  failed `build_road`, `zone` or `dezone` also says how to fix it: a road blocked by empty zoned
  lots names them and the `dezone` rectangles to run first (free; they stay on the road's
  line, at most 2 named), and one blocked by a building says to route around it.
  - While no dead end extends, a road blocked by a building or run off the map also names
    the best branch road (`no dead end to extend: branch instead, e.g. dezone (12,23)
    (empty lot, free), then build_road (12,23)-(12,9) (find_sites road_extension)`).
  - "Already road" names the road line it runs along ("branch off it into free land") and
    points to `find_sites road_extension`. When the line runs along the role's own
    `build_road` of that turn, it names that call (`you built y=24 x 19-34 at call 2 this
    turn`).
  - A zone on land that isn't free says what the rectangle holds ("all 14 tiles are road",
    "6 zoned C (4 developed), 2 road"), and names the role's own zones of that turn that
    took those tiles (`you zoned them at call 17 this turn`). Where free roadside land lies
    beside it, the hint suggests up to 2 one-tile-wide strips there to zone instead. While
    the city lacks a pump or a sewage outlet, those strips skip free riverbank lots.
  - A failed dezone says what is there.
  - High density before its unlock gets `zone with density "low" now`.
  - A build on a tile that isn't free says what is there (`(29,23) now holds your water
    pump (call 5 this turn)` when the role's own build took it that turn) and names the next
    free site on the role's list for that building, or why the list is empty. In smoke
    v12–v16, 15 of 19 such builds went where the role had just built.

  These hints go to the model only; the history log and the turn's action results keep the
  game's message. From the 4th `inspect` in a turn, while `map_overview` hasn't been called,
  the result points to `map_overview`. After it, from the 3rd further `inspect`, the result
  says its map already shows the layout and to inspect only a tile about to be acted on.
- **Guards.** A `bulldoze` that would cut highway-connected road tiles off the highway
  (the road network relabelled without the removed tiles, as `World.refresh()` does) is
  refused before it runs (`refused: "disconnect"`: not executed, not charged as an invalid
  action, still a tool call), naming how many tiles and where; the agent can build a new link
  first or repeat with `confirm: true`. Transport's `demolish` of a train station, harbour or
  airport that is some roads' only link to the region (metro stations don't link) is refused
  the same way; the refusal suggests asking planning for a road to the highway or building
  another link first. The mayor's `allocate_budget` identical to one that already succeeded
  this turn (same department and amount; allocations add up) is refused too
  (`refused: "duplicate"`), naming what the department now holds; other amounts and taking
  money back pass. A `zone` that takes the last free riverbank lot beside the connected
  roads while the city still has no pump or water tower, or no sewage outlet / treatment
  plant, succeeds, with a warning naming what utilities still needs the bank for. So does
  a `zone` that leaves a department with funds for its cheapest ready first building no
  unclaimed site for it (`WARNING: that zoned the last site for fire's fire station near homes (fire holds
  $2,500 for it): dezone one, e.g. (20,22)`): only when claiming the sites afresh in turn
  order seats fewer of the funded buildings than before; the buildings named are those
  left out when each keeps its earlier site where it can. A `zone` that leaves no dead end to extend and
  no free lot beside the roads from which a straight road reaches open land warns as well
  (`WARNING: that zoned the last lot a new road could branch from; leave one free, e.g.
  dezone (12,23)`). After another warning on the same zone (the riverbank one, or the
  game's "no road access yet"), it adds `; it also zoned the last lot a new road could
  branch from` instead of a second warning.

| tools | |
|---|---|
| look-ups (no side effects) | `city_overview` (headline numbers and a 3-month trend), `map_overview` (the whole map as ASCII with rulers and a legend, downsampled on maps too big for one tool result, plus the facts of the briefing's Map block; when the map and the facts together don't fit, as from about month 66 of a long game, the road lines go first, the longest kept while they fit, then the facts give way to a pointer to the Map block), `inspect(x, y, radius)` (tiles within 4: zones and levels, buildings and tiers, roads, river, power / water / pollution), `find_sites(purpose, near_x, near_y)` (ranked sites for the role, e.g. `zone_R`, `road_extension`, `pump`, `clinic`, `bus_stops`, `levee`, `library` (the school list); sites on roads cut off from the highway are dropped, and the pump and sewage lists then top up from free riverbank lots beside the connected roads; the mayor's read-only version takes `<department>.<purpose>`, e.g. `utilities.sewage`, to check which sites a department has), `my_facilities` (the role's buildings with load, capacity, tier and the upgrades on offer; for the mayor, funding levels, policies and imports), `unlocks`, `get_plan`, `get_messages`, `history(kind, months)` |
| turn control | `send_message(to, text)`, `update_memory(commitments, open_tasks, lessons)`, `update_plan(goals, budgets, priorities, notes)` (mayor only; the result names any idle money to take back first, lists the departments below their budget target that would use a top-up (as in the briefing), with the gap and the treasury, and warns when the gaps exceed the treasury above the reserve, or says "hold" when there is nothing above it; the reserve rule in `fleet/tools.py:reserve`, $3,000 or 3 months of expenses, 6 in a deficit, mirrors the fleet configs' mayor playbook), `end_turn(report)` |
| actions | one tool per action of the role (see [Roles and actions](#roles-and-actions)). The JSON schemas come from `actions.action_docs()`, and the actions run through `apply_action`, so validation, costs and milestone locks are the same as for every other agent |

**Three memory layers.** Nothing else carries over between turns.

1. **City plan**, owned by the mayor. In `update_plan`, goals (up to 5) and notes replace the old ones, while budgets and priorities merge per department. Every version is kept.
2. **Department memory**, owned by each agent: `commitments`, `open_tasks` and `lessons`, up to 8 items of 200 characters each. It's shown in full in the agent's briefing.
3. **History log**, written by the game: every action with its result and cost, end-of-turn reports, city events and a monthly metrics snapshot. Agents query it with `history(kind = action | failure | event | report | metrics | all, months up to 12)`. A department sees its own entries and the city's; the mayor sees everything.

The memory lives on the env (`env.fleet`) for the length of the run. The final plan and
lessons go into the fleet report, and every edit is in the tool traces.

**Limits per turn** are set in `[fleet]` or per role:

- `max_steps` tool calls (default 20)
- `max_tokens` per model call (16,000)
- `turn_tokens` completion tokens per turn (80,000)
- the game's `max_actions_per_turn` successful actions

A model that answers without a tool call gets one nudge, and a second one ends the turn.
A reply cut off at `max_tokens` before any tool call gets its own nudge instead (keep the
reasoning short, pick one action and call its tool now), up to 2 a turn; after those it
counts as any other reply without a call. No nudge is sent once the turn has used
`turn_tokens`: the turn ends there as `turn_tokens`.
Tool calls a model leaves as text (`<tool_call>` blocks in its reasoning, which
trinity-large-thinking sometimes does) still run, and the trajectory counts them in
`recovered_calls`.
On a limit or an API error the turn ends, and completed actions stay. Every turn records
why it ended: `end_turn`, `max_steps`, `turn_tokens`, `no_tool_calls` or `api_error`.
Every tool result carries `steps_left` (except `end_turn`'s and the "not run" ones for calls
after `end_turn` or past the limit), plus a `wrap_up` note once 3 or fewer calls are left.
Within a turn, an identical repeat of a look-up (same tool, same arguments after type
coercion) returns the earlier result marked `repeat`, and an identical repeat of an action
that failed is refused without running (`refused: "repeat"`, not charged as an invalid
action), as is one of an invalid call (bad arguments or JSON, or an unknown tool; same
arguments as sent), with the earlier error, until the agent's next successful action. A
successful `update_plan` also refreshes look-ups, and a failed action refreshes `history`
(which lists it). Every kind of repeat still counts as a tool call. A guard refusal (above)
is not remembered as a failure: an identical repeat is checked, and refused, again.

**Coaching with playbooks.** The system prompt starts with a fixed part: how to play (start
from the briefing and look up only what it lacks), the role's mission and the game rules.
Then comes the coach's playbook: the fleet playbook (`[fleet].playbook`, shared by every role) and the role's own (`[agents.<role>].playbook`).
Coaches can steer strategy this way but can't change the rules or the tools.
The "hold" advice follows a reserve rule that lives in code (`fleet/tools.py:reserve`): a coach
who changes the mayor playbook's reserve rule must change it there too.
`civitas prompt -c configs/fleet_trinity.toml --role planning` prints a fleet agent's
system prompt, its first briefing and its tool names.

**Modes.** `mode = "autonomous"` fixes the instructions at month 0 and the fleet plays to the
end on its own; this is the standard eval track. `"supervised"` plays in sessions, and the
coach may revise the playbooks between them. The mode is recorded in the run (the signature
ends in `[autonomous]` or `[supervised]`) and shown in briefings. All fleet agents in a run
share one mode.

- **Pausing.** With `[fleet].session_months = 12` (default 0: no pauses), a supervised run
  stops every 12 months, before that month's turns, prints how to resume and leaves three
  files in `runs/<run_id>/`:
  - `checkpoint.ckpt`, the whole game as a gzipped pickle: the env with its world and RNG,
    the council, the scores history and the fleet's plan, memory and log; every agent's
    state; the runner's stats; and where the logs ended. With a fake model, fleet_smoke's
    fleet paused at month 12 left 49 KB (1.1 MB unzipped), and 81 KB with scripted
    departments growing the city. Chat clients aren't saved (they hold the API key) and are
    rebuilt on resume.
  - **Signed.** Unpickling runs code, so the file starts with an HMAC-SHA256 of the pickle,
    keyed by a secret made on the first pause: `~/.config/civitas/checkpoint.key`, mode
    0600 (`$XDG_CONFIG_HOME` moves it, and so does `CIVITAS_CHECKPOINT_KEY_DIR`, which the
    tests use). Resume checks it before unpickling. A checkpoint with no signature or a
    wrong one is refused, saying why: the file was changed, or it was made on another
    machine or account. So is one with no key to check against, or a key file other users
    can read. A checkpoint only resumes where it was made, so a shared run folder can be
    replayed but not resumed.
  - `playbooks.toml`, the run's current playbooks and `session_months`, ready to edit.
  - `PAUSED`, with the month, the session and the resume command.
- **Resuming.** `civitas resume <run_id>` merges `playbooks.toml` over the run's config, or
  the file given with `-c` instead (an edited copy of the run's config works). Tables merge
  key by key, and so does each `[agents.<role>]`. It then applies `--agent` / `--set`, builds
  the fleet agents again from the merged config, keeping their usage and state, and plays the
  next session with the new system prompts.
  - Playbooks and the other fleet settings may change: model, limits, `session_months`
    (`--set fleet.session_months=0` plays to the end).
  - A change to `[game]`, `[scoring]`, the line-up or the mode is refused, naming what differs.
- **Logs.** Frames and trajectories append; lines an interrupted session logged after the
  checkpoint are dropped first. Each trajectory record carries its `session`. `meta.json`
  gets a `sessions` list: months, status, `fleet_version`, a hash of each playbook, and the
  roles whose system prompt changed with the new prompts (`sessions[i].fleet_prompts`, which
  those roles' later turns point to; `fleet_prompts` keeps session 1's). The summary counts
  the sessions. A run that played more than one `fleet_version` lists them in
  `fleet_versions` and gets a `fleet_version` hashed from the sequence.
- **Determinism.** `session_months` is left out of `fleet_version`, and a run resumed with
  unchanged playbooks plays exactly like one that never paused: same frames, trajectories
  and scores (`tests/test_sessions.py` resumes in a fresh process).
- Autonomous runs and `civitas bench` ignore `session_months`. `civitas serve` lists a
  paused run as `paused`, and `POST /api/resume {"run_id": ...}` resumes it with its
  `playbooks.toml` (the viewer has no button for it yet).

```bash
civitas run -c coached.toml                 # [fleet] mode = "supervised", session_months = 12
$EDITOR runs/<run_id>/playbooks.toml        # after each pause
civitas resume <run_id>                     # or: civitas resume <run_id> -c edited.toml
```

**Config.**

```toml
[game]
act_every = 3                  # the city runs monthly; the fleet plans by the quarter

[fleet]                        # every fleet agent; any key can be overridden per role
mode = "autonomous"
session_months = 0             # "supervised" only: pause every N months (run-wide, see Modes)
provider = "arcee"
model = "trinity-large-thinking"
max_steps = 20
max_tokens = 16000
turn_tokens = 80000
temperature = 0.4
extra_body = {}                # extra request fields (a role's extra_body replaces this one)
playbook = """
- Keep a treasury reserve; utilities first, then services as the city grows ...
"""

[agents.mayor]
kind = "fleet"
playbook = """
- First turn: write the plan, then fund it ...
"""

[agents.health]
kind = "fleet"
model = "trinity-mini"         # a cheaper model for one department
max_steps = 12
```

Fleet and non-fleet roles mix freely, for example a fleet mayor over scripted departments.
`configs/fleet_trinity.toml` is the full fleet: all eight roles on trinity-large-thinking for
120 months, acting every 3 months (320 turns), with a default fleet playbook and one
playbook per department. The fleet playbook tells departments that their briefing already
has their plan slice, budget, new messages, map and ready sites, so they shouldn't re-read
them with `get_plan` / `get_messages` / `history`, and that when it says they can't pay for
any ready site they should send at most one request and end the turn.
`configs/fleet_smoke.toml` is the same fleet for 12 months.
`meta.json` records `fleet_version` and every fleet agent's system prompt once
(`fleet_prompts`); `summary.json` records `fleet_version` too. `fleet_version` is a hash of
`[fleet]` and the fleet agents' settings, so runs with the same coaching can be compared: the
leaderboard groups fleet runs by signature and `fleet_version` (shown in its own column). The
signature reads `role=fleet:<model> ... [mode]`.

**Regions.** Fleets play region games too: `[agents.<role>]` puts the fleet in every city
and `[agents."north.<role>"]` in one, so fleet and scripted cities, and fleet and non-fleet
roles, mix freely. A fleet agent binds to its own city's env (`fleet/region.py`), so each
city has its own plan, department memory, history log and briefings, and the north mayor
plans and funds north's departments only. The system prompt gains a Region section (the
other cities and where they lie, the council's addresses, the river, links and, for the
mayor, deals). The mayor's briefing gains a Region block: per other city its population,
specialisations, surplus and shortfall, unemployed and open jobs, links, what a connector
would cost transport and last month's goods, commuters and tourists between the two, then
the river's pollution and up to 4 deals with what to do about them. Transport's block has
each neighbour's best connector site priced as `build_connector` charges it, and boat and
air links; utilities' has the river and what the deals do to supply. Every role gets
`region_overview`, the mayor gets `offer_deal`, `accept_deal` and `cancel_deal` (run as
the region runs any mayor's deals, and counted in the action cap), and `send_message`
takes a role of the city (`"mayor"`), a team (`"south"`), another city's agent
(`"south.mayor"`) or `"all"`. The fleet report has the region scoreboard and one section
per city with fleet agents. `configs/fleet_region_smoke.toml` runs the `fleet_smoke.toml`
fleet in the north city against the scripted baseline in the south.

**Scripted fleet.** `model = "scripted"` makes a fleet agent play the role's scripted
baseline (`agents/scripted.py`) through the fleet tools instead of calling a model
(`fleet/scripted.py`). It costs nothing and runs a 120-month game in about 5 s (10 s acting
monthly). Each turn:

- it gets the briefing (logged as the turn's first message, like a model's);
- it reads what the policy needs with look-ups on top of `role_obs()`: `map_overview` for
  the departments that read the map, `get_messages` for the mayor's council requests;
- it runs the scripted decision as tool calls, arguments sent as JSON text. The mayor first
  writes its working balances into the plan with `update_plan` (goals, a budget and a
  priority per department). Each action is one call (a duplicate allocation the policy
  means is repeated with `confirm: true`), then come `send_message`, `update_memory` (open
  requests, one lesson per kind of failed action) and `end_turn` with a report.

`configs/fleet_scripted.toml` plays all eight roles with the game settings of
`fleet_trinity.toml` (120 months, `act_every = 3`). `--agent police=fleet:scripted` puts
one scripted fleet role into any line-up, for example an LLM fleet department with scripted
colleagues that talk to it through the same plan, messages and reports.

It plays exactly as the scripted baselines: on seeds 0–4, at both cadences, every turn
has the same actions, results, costs and council messages
(`tests/test_fleet_scripted.py` checks a 24-month game). The tool layer changes nothing
for these policies.

| seeds 0–4, 120 months | city | population | mayor | planning | utilities | transport | police | fire | health | education |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| monthly: `baseline.toml`, or `fleet_scripted.toml --set game.act_every=1` | 77.5 ± 1.5 | 14,466 | 88.1 | 68.4 | 89.4 | 75.9 | 74.9 | 85.7 | 86.6 | 83.2 |
| every 3 months: `fleet_scripted.toml`, or `baseline.toml --set game.act_every=3` | 67.0 ± 2.8 | 7,319 | 84.3 | 65.1 | 88.7 | 92.3 | 74.5 | 83.8 | 85.5 | 70.0 |

So compare a fleet with 67.0, not the monthly 77.5. The 10.5 points are the quarterly
cadence, since the scripted policies do about a month's work per turn:

- **Planning** lays about one road and up to three zone strips per type a turn. It built 18
  road segments and 160 zone strips instead of 46 and 311, and spent $9.6k instead of $31k.
- **The mayor** reads only the requests posted the month before its turn. At `act_every =
  3` that is none of the 146 that departments sent (monthly: 529 of 531).
- **Money:** the mayor moved $166k to departments instead of $435k. Police and education
  asked for their first building from month 6–9 and built it in months 33–48 and 39–63
  (monthly: 34–40 and 35–48, the baseline's own slow start).
- **Transport** scores higher (92.3) because it spends little in a smaller city.

What the runs showed about the tools:

- No guard fired. The policies never bulldoze or demolish, and never send the same
  allocation twice.
- There were no tool errors, no result was trimmed, and briefings stayed under 4,700
  characters for all 120 months.
- All 47 "zone beside it instead" suggestions after a failed zone worked when applied.
- Both funded-site warnings were right. A zone took utilities' only sewage-outlet lot, and
  sewage coverage then fell to 81–85% for months (the scripted planner doesn't read
  results).
- No briefing said "can't pay" on a turn the policy built something, and none said "no
  site" for a building the policy then built, except once: health's policy falls back to a
  clinic lot for a landfill when, as the briefing said, no lot away from homes is left.
- The coaching hints were noisy on a long game. The idle rule fired for 6 or 7 departments
  per run. That covered departments waiting for money they asked for every turn (police
  held $1,494–2,436 against a $2,500 station in months 6–33 of seed 0) and departments with
  nothing left to do (fire at 83–98% coverage). Planning's repeated failures (a plaza spot
  off the road, zones and parks on taken land) gave 3–4 repeated-failure hints per run.
  With the idle rule split by what the briefing showed and failures grouped by cause (below),
  seeds 0–4 give 27 hints instead of 46: 9 idle (30 before), 5 for the mayor (one per run,
  naming 5–6 departments short of their first building, education longest at 11–20 turns),
  3 for planning (utilities' sewage outlet lot, taken for 9–15 months), 9 repeated-failure
  (15) and 1 idle-money. The idle hints left:
  - transport holding $1,500–2,000 for a ready bus line (4 runs), or an avenue upgrade while
    it asked for $10k;
  - health or police holding $3,000 for a ready clinic, landfill or station while asking
    the mayor for $7,700–8,300 for an upgrade.

  The policies had reasons the briefing doesn't show (bus ridership, the upgrade).

**Reports.** Every run with fleet agents writes `fleet_report.json` and `fleet_report.md`
next to its summary. They contain:

- the scoreboard
- the final plan and the months it changed
- per-role stats: turns, tool calls, actions, failures, tool errors, repeated look-ups and refused calls (repeats and guard refusals), spending, tokens, turns ended on a limit, length stops and turns cut short by an API or briefing error
- the top failures, last reports and the lessons in memory
- coaching hints

A hint fires on any of these:

- 3 or more failures of one role with one cause. Taken land, no road access and money
  group their wordings ("no empty land tiles there", "(21,26) is not an empty land tile",
  "tile (35,23) is occupied by coal_plant"); any other wording is a cause of its own. The
  hint names each action's first wording and count, and the lesson to write down.
- a department holding more than 2× its plan budget for 3+ months
- 2+ consecutive department turns of one kind without a successful action, spanning 6+
  months. The kind comes from what the turn's briefing showed (its funds, Ready sites and
  Key metrics):
  - **idle**: its funds covered a ready site it needed. The department's hint names the
    funds, the building, the metrics, what it asked the mayor for and its failed actions.
  - **starved**: its funds were short of the cheapest such site. One hint for the mayor
    names every such department (the three longest runs in full), what it was short of and
    how often it asked.
  - **no land**: none was ready, but some was blocked for lack of land or roads. One hint
    for planning names the departments and the cheapest buildings.
  - **nothing needed**: no hint. That covers coverage at 80% or more (for a utility, also
    demand under 90% of supply), no congested road, "nothing needs it", locked, and
    supplementary buildings (library, shelter, cemetery, train station, harbour, metro).

  Planning goes by its priced `road_extension` line: idle when its funds covered the road,
  starved when short (`planning short of its $150 road into open land`). Without a road to
  build it is idle with $1 or more, since it can still zone.
  Without a briefing (older records), the funds come from the history log: under $1
  against a plan budget is starved, under $1 without one is nothing needed, and anything
  else is idle.
- a negative treasury
- a role ending more than 20% of its turns on a limit

`summary.json` gets a `fleet` entry with the report name and the number of hints. In
`trajectories.jsonl`, each fleet turn carries its full `tool_trace` (tool, arguments and a
slim result) along with the messages, reasoning, token usage and finish reasons. The viewer's
Agents tab shows the trace in each fleet turn's Tool calls panel, fetched when the panel is
opened from `GET /api/runs/<id>/trace?month=M&role=R` (the server keeps an index of where each
turn starts in the file).

**First smoke runs** (2026-09-25, `fleet_smoke.toml`, seed 0) took 9–11 minutes for 32
turns. Three runs (v2–v4) used revised playbooks on intermediate code, as the fixes from
each run landed. Turns averaged 12–16 tool calls and 48–66k prompt tokens, and 21–30% of
actions failed, mostly departments building without funds or on taken tiles. The fleet
reached 1.0k–1.7k residents and a city score of 40.6–43.5. A final run (v5) on the
committed code reached 2,244 residents and 43.1: 13.8 tool calls and 53k prompt / 4.7k
completion tokens per turn, 24% of actions failed, 7 of 32 turns ended on the step limit.
The scripted baseline on the same settings (`act_every = 3`) reaches 468 residents and
43.8 on seed 0 and 384 and 36.2 on seed 1; acting monthly, 916 and 45.4 (seed 1: 872 and
43.4).

**v6** (2026-09-26, after the briefing blocks, `map_overview`, repeats, the mayor's
`find_sites` and the playbooks above; two seeds run in parallel, 14.5–16 minutes each):
seed 0 reached 1,848 residents and a city score of 45.1, seed 1 2,756 and 46.6. Sewage was
built on month 0 in both (v5 never built it), the mayor funded the plan in its first turn,
and planning freed a riverbank lot when asked. Turns averaged 14.6–14.7 tool calls and
68–71k prompt / 5–6k completion tokens (up from 53k / 4.7k: agents call `map_overview`
most turns, the mayor calls `find_sites` about 5 times a turn and `inspect` use rose);
19–27% of actions failed and 6–7 of 32 turns ended on the step limit, about as in v5.
Twelve months is too short to compare with the 120-month benchmark, and single runs vary
a lot (the two v6 seeds differ by 900 residents).

**v7** (2026-09-26, after the wording changes: `map_overview` only for tile-level layout,
the mayor's `find_sites` limited to checking "no site" claims, "hold" allowing money to be
taken back, the health `cemetery` purpose and the away-from-homes site notes) went badly on
both seeds: 736 residents / 35.4 and 180 / 22.6. On seed 1 planning bulldozed the two road
tiles linking the city to the highway (after health asked for a landfill lot "at the town
edge"), cutting the city off: garbage coverage stayed at 0% and 1,268 residents fell to 180.
On seed 0 planning zoned both riverbanks on month 0, so utilities never built a pump or
outlet. The guards above (bulldoze refusal, riverbank warning, unconnected-roads line) came
from these runs. **v8** (2026-09-27, with the guards) was healthy on both seeds: 2,504
residents / 46.4 and 1,984 / 45.1, full water and sewage coverage, 11–13 tool calls and
53–68k prompt tokens per turn, 25–27% of actions failed, 2 of 64 turns on the step limit.
The guards did not fire in v8, so the recovery from v7 is mostly run-to-run variance; they
stay as a safety net against the v7 failures.

**v9** (2026-09-27, after prices in the briefings, the action-error hints, the
briefing-first playbook and the idle rule) reached 952 residents / 41.8 on seed 0 and 1,844 /
45.2 on seed 1.
- **Funding failures:** "costs $X but the budget only has $Y" fell from 10 and 4 to 1 and 2.
- **Tokens and calls:** prompt tokens per turn fell from 53–68k to 41–50k, and calls per turn from 11–13 to 7–9.
- **Idle turns:** their share of prompt tokens fell from 37–48% to 15–30%. Idle hints fell from 5 and 6 to 0 and 2.
- **Hints worked:** after each road or zone hint, planning ran the suggested dezone or zoned at low density on its next call.
- **Two new problems:**
  - Seed 0's planning lost its whole month-0 turn to two replies cut at `max_tokens` with no tool call.
  - On both seeds the mayor funded fire's cheapest ready site ("needs $1,500 for a shelter"), so fire never reached a station (seed 1: fire score 31.4).

**v10** (2026-09-27; shelter and cemetery are never named as the first build, and cut
replies get their own nudge) reached 2,076 / 47.5 and 1,028 / 45.6.
- **Lost turns:** none. Four turns were cut at the token limit, and all recovered.
- **Stations:** the mayor funded police and fire with $2,500 each, and both built a station (fire 69.8 and 78.6, up from 53.3 and 31.4).
- **Tokens and calls:** 41–57k prompt tokens and 7–10 calls per turn; idle turns took 35% and 18% of prompt tokens.
- **Failed actions: still 27–30%.** Funding failures stay low (3 and 3), but planning's road and zone attempts now make up most failures: roads along existing roads, roads through buildings, zoning taken land. Planning also ended 2–3 of its 4 turns on the step limit.
- **Other problems:**
  - On seed 1 the mayor gave health $5,000 while education, asking for $2,500 every turn, got nothing.
  - On seed 1 planning spent two turns routing a road through a wind turbine.

**v11** (2026-09-27, after the final review fixes: bus lines and avenue tiles priced;
train station, harbour and metro never named as the first build; no bus stops before 1,500
residents) reached 2,008 / 46.1 and 1,916 / 45.4. A first seed-0 attempt stalled on the
Arcee API and was rerun.
- **Failed actions:** 15.9% and 17.2%, the lowest so far. There were 2 and 1 funding failures, 44–52k prompt tokens and about 9 calls per turn, and no turn was cut at the token limit.
- **Funding:** at month 3 the mayor gave police, fire and health enough for a first building.
- **Transport:** its idle turns took 1–4 calls.
- **Remaining failures:** mostly planning. Roads along existing roads and zoning taken land. It ended 2–3 of 4 turns on the step limit; on seed 1 every planning action in months 6–9 failed.
- **Idle departments:** on seed 1 fire, health and education held funds for two turns without building.

**v12** (2026-09-28, after the planning fixes: road lines in the Map block, zone / dezone
hints with strips to zone beside a road, road-line names on "already road", the inspect
note, the high-density line and the planning playbook) reached 1,564 / 43.8 and 1,820 /
44.3, a little below v11 but within run-to-run variance.
- **Planning:** across the 8 turns, 25% of its actions failed (v9–v11: 35% over 24 turns). 2 of 8 turns ended on the step limit (v9–v11: 12 of 24), and it made 3.0 and 5.5 inspect calls per turn (5.4).
- **Hints followed:** it zoned the suggested strip after trying to zone its new road, and it ran the suggested dezone before rebuilding a road through lots it had just zoned.
- **Still failing:**
  - It rebuilt its own new road twice.
  - It zoned a second type over a rectangle it had just zoned, 4 times, even though the hint named what was there.
  - The inspect note fired once for planning (seed 1, month 6) and twice for utilities. On its other turns planning called `map_overview` before a 4th inspect and kept inspecting afterwards (up to 8 a turn), which the note doesn't cover.
- **Fleet-wide:** 16% and 26% of actions failed, with 44–49k prompt tokens and about 9 calls per turn. On seed 0 education only ever got $1,500, short of a $2,500 school (score 33.3), and at month 3 the mayor sent the same allocations two or three times each.

**v13** (2026-09-28, after the mayor's money lines: spending, keep and idle money, below
target only when a top-up would be used, the duplicate-allocation guard and a new mayor
playbook) reached 1,836 residents / 39.7 on seed 0 and 928 / 39.9 on seed 1.

Compared with v9–v12 (8 runs):

- **Idle money:** planning held $1,320 and $849 at month 6 (v9–v12: $7,300–10,700 in every
  run). The mayor took idle money back 6 and 3 times.
- **Loans:** $0 and $15,000. On seed 1 the mayor sent utilities $7,000 at month 3 against a
  $4,000 gap, and then borrowed.
- **Duplicate allocations:** none (v9–v12: 7).
- **First buildings:** seed 1 was fine, with health at month 0, police and fire at 3, and a
  school at 6 (education 61.1). Seed 0 was not:
  - By month 6, planning had zoned every roadside lot. Police, fire, health and education
    each held exactly their first building's price, and they read as idle above the $1,000
    floor.
  - The mayor took that money back. Police and fire built only at month 9, and health and
    education never did (education 33.3).
  - This led to "blocked is not idle".
- **Planning on seed 1:** it zoned the last free riverbank lot at month 0, then spent three
  turns bulldozing and dezoning its own roads to undo it. That kept the city small.

**v14** (2026-09-28, after "blocked is not idle": a first building with no lot for lack of
land counts toward the keep, the mayor's line says `blocked: ... - ask planning`, and
planning's Ready sites name the departments it has blocked) reached 1,956 / 45.4 and 912 /
42.2. The laptop slept through the runs' first 35 minutes, and both first calls were
retried, so the wall-clock times aren't comparable.

- **Money:** loans were $0 and $7,500, and there were no duplicate allocations.
- **Idle and take-backs:** only one idle line in the whole run, planning's $1,730 on seed
  1, which the mayor took back. No blocked department's money was taken.
- **First buildings:** on seed 1, police, fire, health and education all built at month 6
  (education 51.4).
- **Two new problems:**
  - **Riverbank false alarm.** Planning's blocked line named utilities' pump and sewage
    from month 0 to 9 on both seeds. It was right while no road reached the river, and
    while seed 1 had no outlet. Later, with a pump and an outlet built and room to spare
    (water 2–4× demand, sewage 1.4–2.5×), it still told planning to free a riverbank lot.
  - **Lots near homes.** On seed 0, the mayor funded fire, health and education at months
    3 and 6, when their sites were ready. Planning then zoned the lots near homes, so the
    services read "station / clinic / school: none - nothing needs it right now" with
    coverage at 0. Fire held $2,500 from month 3 to 12 without building (fire 37.0), and
    fire and health spent 13–20 calls a turn at months 3–6 looking for sites. Turns with
    no successful action took 46.5% of seed 0's prompt tokens (71k per turn on average).
- **Utilities over-building:** on seed 1, utilities built three $2,200 water towers,
  leaving water at 4.7× demand with coverage already full.

**v15** (2026-09-28, after the fixes: pump and outlet blocked only while needed, 2x2
blocks, lots near homes counted as blocked, the leave-free line, the zone warning, and
zone strips that spare funded sites) reached 1,508 / 43.0 and 1,424 / 41.1.

- **First buildings:** every service got its first building on time, the aim of this
  work. On seed 0, police, fire, health and education all built at month 3; on seed 1,
  all at month 6. Education scored 58.7 and 60.1 (v13–v14 seed 0: 33.3), and fire 57.1
  and 64.6 (v14 seed 0: 37.0).
- **Failed actions:** 20–22%.
- **Loans:** $12,000 and $0. On seed 0 the mayor followed the needs lines into second and
  third stations and clinics at 740–1,016 residents. The treasury fell from $20k to
  $3.3k by month 6, and the mayor took planning's $2,310 back, leaving planning at $0 for
  months 9–12.
- **What the review and the runs found in the new lines (fixed afterwards):**
  - Every leave-free line named a water tower, although water was never short. Utilities'
    near-homes list ranks lots by the homes around them, so planning was told to hold
    back its best residential lot.
  - Departments shared a best site: seed 1, month 6 named (34,25) for four buildings.
  - The line was cut at 240 characters mid-word on seed 0, month 9.
  - A 10% share measured on fractional coverage would flag well-covered mature cities.
- **Still open:**
  - On seed 1, health's landfill read "blocked … away from homes" at months 6 and 9, and
    planning never made a lot for it, so garbage coverage stayed at 0.
  - On seed 1, the mayor funded no service at month 3, despite the playbook.

**v16** (2026-09-28, after those fixes: utilities only while its supply is at least 90% used,
claimed sites, whole-item truncation, and need measured under 30% coverage with a 15%
share) reached 2,000 / 47.4 and 2,112 / 46.0. Its mean score of 46.7 is the best of any
smoke pair so far (v10: 46.6; v8 and v11: 45.8).

- **Failed actions:** 15.1% and 22.1%. There were 50–52k prompt tokens and about 9 calls
  per turn.
- **First buildings:** on seed 1, police, fire, health and education all built at month 3
  (education 64.5). On seed 0, police and fire built at month 3 and health's clinic at 6.
- **Leave-free line:** after month 0 it named a utilities building only while supply was
  short: seed 0's wind turbine at month 3 (power 70.8 of 70.8) and water tower at month 9
  (demand 197 against supply 169). Buildings no longer shared a site.
- **Zone warning:** it fired once, on seed 0 at month 9, when a zone took health's last
  clinic lot. Health built on the lot it named.
- **Still open: the mayor's needs lines.**
  - A second station reads just like a first building: `needs $2,500 for a police
    station (5 ready sites)`, whether the department has none or one already.
  - On seed 0 the treasury was short at months 6 and 9. The mayor funded second police and
    fire stations and a second clinic first, and education's allocations were refused
    ("treasury only has $500"). Education asked every quarter and never got its school
    (33.2).
  - On seed 1 the mayor borrowed $10,500 at month 6 (528 residents), for utilities'
    $7,000 and second police and fire stations.
  - Utilities' needs line still names a water tower when water isn't short (v14 and v15
    seed 1: towers built with water at 4.7–4.9× demand).

**v17** (2026-09-28, main at 760ea72: the planning hints naming its own earlier calls, the
inspect note after `map_overview`, the taken-tile hint and the `map_overview` road-line
fallback) reached 1,152 / 42.4 and 1,832 / 46.9. The mayor's needs line is unchanged, so
education again got no school on either seed (33.3 and 33.4).

- **Seed 1:** 15.6% of actions failed (the lowest on that seed) and planning failed 7 of
  31. Police, fire and health built at month 3.
- **Seed 0: planning boxed itself in.**
  - By month 3 the city was one road across the whole map (`y=24 x 0-47`), with zoning or
    buildings on every tile along both sides. That left no dead end to extend and no free
    tile to branch a new road from.
  - Police built at month 3. Fire, health and education held their money as blocked for the
    rest of the run.
  - Planning spent months 6–9 trying to build roads through buildings and past the map
    edge. It failed 15 of 29 actions, and turns with no successful action took 64% of the
    prompt tokens.
  - Nothing yet tells it to give up one lot to open a branch road.

### Output files (`runs/<run_id>/`)

- `meta.json`: config, agents, map size, natural resources
- `frames.jsonl`: one frame per month (tiles, levels, variants, road classes, overlay layers, metrics, scores, turns, council). The viewer replays these.
- `trajectories.jsonl`: one record per agent turn: the full prompt and raw response (LLM agents) or the observation text (scripted agents), actions, results, reward and scores
- `summary.json`: final and average scores, per-agent stats (invalid rate, spending, messages, LLM tokens, latency)
- `fleet_report.json` / `fleet_report.md` (runs with fleet agents): scoreboard, plan, per-role tool use and failures, coaching hints (see [Agent fleets](#agent-fleets)); in a region, one section per city with fleet agents
- `checkpoint.ckpt` / `playbooks.toml` / `PAUSED` (a supervised run between sessions): the game to resume, the playbooks to edit, the resume command (see Modes under [Agent fleets](#agent-fleets)). The checkpoint is a pickle signed with this installation's key (`~/.config/civitas/checkpoint.key`); resume refuses one whose signature is missing or doesn't match, so it only resumes on the machine and account that made it

`python examples/export_sft.py runs/ --role planning --min-reward 0.001` turns the good
turns into a chat-format dataset for fine-tuning or as the start of a GRPO / preference loop.
It skips fleet turns (multi-step tool-calling traces, which don't fit its one-shot format) and prints how many.

---

## Project layout

```
civitas/
  sim/        catalog.py (all balance numbers) · world.py (map, networks, tick, metrics) · grid.py (numpy helpers)
              population.py (cohorts, schooling, jobs) · services.py (coverage, crime, fire, garbage, health)
              civics.py (needs, complaints, approval, funding, policies, districts) · traffic.py (commutes, buses)
              economy.py (demand, levels, wealth, industry, households, trade, profits) · events.py
              progress.py (milestones, landmarks; heritage, soil, wind, seasons)
  core/       env.py (the hub) · region.py (2-6 linked cities) · observation.py · actions.py · council.py · roles.py · decision.py
  agents/     llm.py · llm_client.py · prompts.py · scripted.py · remote.py · base.py (noop, random)
  bench/      scoring.py · runner.py (episodes, logging, benchmark, leaderboard) · session.py (pause / resume)
  fleet/      agent.py (ToolAgent tool loop) · scripted.py (the scripted baselines through the tools) · tools.py · briefing.py
              · memory.py (plan, memory, history) · config.py · report.py · region.py · fmt.py
  viewer/     index.html · app.js · scene.js (three.js renderer) · life.js (3D vehicles)
              · pixel.js (OpenTTD-style 2D renderer) · recipes.js (its buildings, ground, vehicles)
              · sprites.js (pixel-art painter) · lifesim.js (vehicle movement shared by both views)
              · region.js (region map) · app.css · vendor/ (three.js, bundled; works offline)
  cli.py      civitas run | resume | bench | serve | leaderboard | prompt | check-llm
configs/      baseline · arcee · arcee_full · smoke · eval_planning · remote · region · region_trio · fleet_trinity · fleet_smoke
              · fleet_region_smoke · fleet_scripted
docs/         MECHANICS.md (the design roadmap and what is implemented)
examples/     remote_agent.py · rl_loop.py · export_sft.py
tests/
```

`civitas prompt -c configs/arcee.toml --role police` prints exactly what an LLM agent sees
on month 0. It's the fastest way to iterate on prompts.

## Adding a role

1. `core/roles.py`: add the role (title, mission, allowed actions, budget line) and put it in `TURN_ORDER`.
2. `sim/world.py`: add the department to `DEPARTMENTS` (and `catalog.FUNDED_DEPARTMENTS` if it has a funding level).
3. `sim/`: add the mechanic, usually in a mixin (see `services.py`); record what the score needs in `World._metrics()`.
4. `core/actions.py`: add handlers to `HANDLERS`, docs in `action_docs()`, and what it may demolish in `OWNED_STRUCTS`.
5. `core/observation.py`: add a role section (`ROLE_SECTIONS`) and its text rendering.
6. `bench/scoring.py`: add a KPI function to `ROLE_SCORERS`.
7. `agents/scripted.py`: add a baseline to `SCRIPTED`, and let the scripted mayor fund it.
8. Viewer: give new buildings a character in `catalog.CHAR`, a model in `scene.js` (`_buildSpecial`), a sprite recipe in `recipes.js` (`special`), and entries in `NAMES` / `MINI` in `app.js`; new map layers go in `World.frame_layers()` and `OVERLAYS` in `scene.js`.
9. Fleet: add the role's `find_sites` purposes to `fleet/tools.py:SITE_PURPOSES` (`{}` if it places no buildings; gate locked ones in `SITE_UNLOCKS`; price each in `SITE_BUILDS` (tier names, or `OFF_TIER_SPECS` entries for prices outside the tier lines, listed in `PER_TILE` when charged per tile); list a priced purpose that is never the role's first build in `SUPPLEMENTARY`), its briefing metrics to `fleet/briefing.py:KEY_METRICS`, and the role to `UNLOCK_ROLES` for the milestone unlocks it uses. (`fleet/memory.py:DEPARTMENTS` follows `TURN_ORDER`.) Give it an `[agents.<role>]` block with a playbook in both LLM fleet configs, a `kind = "fleet"` block in `fleet_scripted.toml`, and its plan priority line in `fleet/scripted.py:PRIORITIES` (and a map look-up in `MAP_ROLES` if its scripted baseline reads the map).

## Ideas for next steps

- Per-agent partial maps (only the role's own layer in detail) to shrink prompts further.
- An MCP server wrapping the live API (the fleet tools are a ready-made tool set).
- Council negotiation as structured objects (budget requests the mayor approves or denies).
- Region-level projects that need several cities to fund them (a shared university, a regional airport).
- Training: run GRPO on `trajectories.jsonl`, using per-turn rewards as advantages.
