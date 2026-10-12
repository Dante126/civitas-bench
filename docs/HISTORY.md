# Fleet development history

A lab notebook of the agent fleet's development, in date order: smoke runs v1-v23, the long (120-month)
runs and the first late-game smokes, with what each run showed and what changed after it. Model eval results
are in [EVALS.md](EVALS.md); the current design is in [FLEETS.md](FLEETS.md).

Contents:

- [Smoke runs v1-v19 (2026-09-25 to 09-28)](#smoke-runs-v1-v19-2026-09-25-to-09-28)
- [First 120-month runs (2026-09-28)](#first-120-month-runs-2026-09-28)
- [Smoke runs v20-v23 (2026-09-29 to 09-30)](#smoke-runs-v20-v23-2026-09-29-to-09-30)
- [120-month runs on b684672 (2026-09-30)](#120-month-runs-on-b684672-2026-09-30)
- [120-month region run (2026-10-05)](#120-month-region-run-2026-10-05)
- [Late-game smokes (2026-10-06/07)](#late-game-smokes-2026-10-0607)

## Smoke runs v1-v19 (2026-09-25 to 09-28)

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

**v18** (2026-09-28, main at f8e8c57, adding:
- the mayor's needs line saying first or another, with coverage;
- the first-buildings line;
- branch roads out of dead ends and the last-branch-lot warning;
- the landfill away-from-homes tag;
- coaching hints split by who should act;
- planning's road into open land priced for the mayor)

reached 1,412 / 44.9 and 2,120 / 45.9.

- **Failed actions:** 14.1% on seed 0, the lowest of any run, and 21.9% on seed 1. Loans were $3,000 and $4,000.
- **Seed 0:** police, fire, health and education all built by month 6 (education 58.1, up
  from 33.3 in v16 and v17). Each built on or next to the site the leave-free line named.
- **Needs lines worked:**
  - At month 6 on seed 1 the mayor read "first buildings still unfunded: education" and
    funded the school.
  - At month 9 on both seeds it read planning's "needs $50 / $150 for a road into open
    land" and funded it.
- **Still open:**
  - On seed 0 the mayor took back all $2,508 of planning's money when $1,508 was idle, then sent exactly
    the $50 road price, with nothing left for zoning.
  - On seed 1, education's only site was the lot police had claimed, so planning was never
    told it needed a second lot. At month 9 each zone warning named one example tile for
    several buildings, and planning dezoned only that tile. Education never built (33.4).

**v19** (2026-09-28, main at dd24a0d: v18 plus the take-back guard, zoning in planning's
needs line, "no lot of its own" for funded departments that share a lot, and one dezone
tile per building in the zone warning) reached 1,776 / 45.0 and 2,016 / 45.1.

- **First buildings on both seeds:** for the first time, police, fire, health and education
  all built by month 6 on both seeds. Education scored 55.6 and 54.5 (v16–v18: 33 on at least
  one seed each time).
- **Failed actions:** 26.7% and 17.1%. Loans were $3,315 and $0.
- **The new guards fired, but the mayor overrides them when money is tight:**
  - On seed 1 the take-back guard refused two take-backs, and planning's line named
    "education (elementary school): no lot of its own".
  - On seed 0 the guard refused a take-back of planning's $1,342, and the mayor repeated it
    with `confirm: true` to fund a fire station. At month 9 the treasury ran out on second
    police and fire stations before it reached planning's $315.

## First 120-month runs (2026-09-28)

**First 120-month run** (`fleet_trinity.toml`, seed 0, main at 760ea72, i.e. before the v18
and v19 fixes; 320 turns, 15.8M prompt tokens, 1.9 hours):
- **Result:** city 47.9 at the end (45.9 on average), 2,892 residents, no bankruptcy.
- **Against the scripted fleet** on the same 3-month cadence (68.8, about 7,300 residents),
  that is 21 points and 61% of the population short.
- **Money:** the mayor reached the $50,000 debt limit at month 48 and stayed there. 46
  allocations failed on the treasury or the debt limit.
- **Planning** had $0 for 23 attempted builds ("costs $150 but the planning budget only has
  $0"). The later fixes target this: planning's road price, the needs line and the
  take-back guard.
- **Actions:** 34.7% failed, and turns with no successful action took 46% of the prompt
  tokens.

**120-month runs on main at fd5112f** (after the v18 and v19 fixes; `fleet_trinity.toml`, 320 turns each,
16.3M and 18.9M prompt tokens, 2.1 and 2.2 hours, run in parallel):

| | seed 0 | seed 1 | first run (seed 0, 760ea72) |
|---|---|---|---|
| city score at month 120 (average over the run) | **73.5** (56.9) | **69.7** (54.7) | 47.9 (45.9) |
| population | **18,412** | **13,064** | 2,892 |
| debt at the end (months at the $50k cap) | $47.5k (0) | $22.0k (0) | $50k (78) |
| failed actions | 23.8% | 22.0% | 34.7% |
| turns with no successful action, share of tokens | 20% | 19% | 46% |

- **Growth:** both cities kept growing all game (seed 0: 3,960 residents at month 24, 8,140 at 72,
  14,308 at 96), where the first run stalled at about 2,000–2,900 from month 18.
- **Against the scripted fleet:** on the same 3-month cadence (`fleet_scripted.toml`, 67.9 and 66.6
  on these seeds, 6,472 and 6,104 residents), the LLM fleet scored higher on both seeds, with two
  to three times the population. Seed 0's 18,412 residents is more than the monthly scripted
  baseline's 14,092 (city 76.7).
- **Weak spots:** transport (56.3 and 64.1) and, on seed 1, fire (56.8).

## Smoke runs v20-v23 (2026-09-29 to 09-30)

**v20** (2026-09-29, main at c2947f0, adding the mayor money picture from
`docs/plans/2026-09-29-mayor-money.md`: a Finances split, a need-ordered Departments block with
upkeep, "to spend" / "to fund" lines, earned confirm, the loan guard and money result notes)
reached 1,280 / 47.0 and 504 / 42.8.

| Measure | v16–v19 (8 runs) | v20 (seed 0 / seed 1) |
|---|---|---|
| Education school by month 12 | 4 of 8 | 2 of 2 |
| Education score (mean) | 45.8 | 61.9 (66.8 / 56.9) |
| City score (mean) | 45.5 | 44.9 (47.0 / 42.8) |
| "treasury only has" failures | 2.5 per run | 2 per run |
| Loans | 8 ($25.4k) | 3 ($10.5k, all seed 0) |

- **Seed 0:** at month 3 the mayor funded exactly what the "to fund" line listed (every first building and planning's road), and all four services built that month. At month 6 the line read "$4,305 covers planning, police; the other $6,500 would go under the reserve", and the mayor funded second fire and health buildings anyway, leaving $303. At month 9 it borrowed $10.5k in a hold for second buildings. The loan guard didn't cover a turn that starts under $3,000. Since v20, a top-up for another building that leaves the treasury under the reserve, and a loan while no first building, supply or planning road waits, are refused once ([Guards](FLEETS.md#money-guards-and-money-results)).
- **Seed 1:** power coverage stayed at 0–18% until month 10, and the city ended at 504 residents.
  - At month 0 the "to fund" line named only utilities' cheapest wanted building (a $2,200 water tower). The mayor gave exactly that instead of the playbook's $7,000, and utilities built a pump.
  - At month 3, with $4,400, utilities spent all 20 steps on sewage sites (the best riverbank lot was marked for wind, pump and sewage) and built no power plant.

**v21** (2026-09-29, main at e783064: utilities' needs line covers every short supply, a "short now" line for utilities, and soft reserve and loan guards) reached 2,472 / 48.9 and 1,084 / 44.8.

| Measure | v16–v19 (8 runs) | v20 | v21 (seed 0 / seed 1) |
|---|---|---|---|
| City score (mean) | 45.5 | 44.9 | 46.9 (48.9 / 44.8) |
| Education score (mean) | 45.8 | 61.9 | 71.7 (74.6 / 68.8) |
| Loans | 8 ($25.4k) | 3 ($10.5k) | 1 ($3,000, for utilities) |
| "treasury only has" failures | 2.5 per run | 2 per run | 0 |
| Overrides | 7 | 3 | 0 |

- **Power:** seed 1 had full power from month 1 (v20: 0–18% until month 10).
- **Guards:** on seed 0 the reserve refusal led the mayor to take idle money back from planning and transport before funding a second police station. At month 9 it accepted two loan refusals and borrowed only $3,000, for utilities.

**v22** (2026-09-29, main at b461bc6, the first run with the v21 review fixes) reached 1,432 / 49.6 and 1,732 / 44.1. On seed 1, month 9, the loan guard refused a $4,500 loan while the treasury held $3,607 and a $5,000 first high school waited, which is a case the playbook allows. The mayor overrode it, then confirmed another landfill that left $1,107 of a $3,000 reserve. Both are fixed in a8a0ee8 (Guards, Earned confirm).

**v23** (2026-09-30, main at b99b235: the loan and reserve fixes, transport's first bus line before avenues, and planning's zone hints, one zone type per strip and branch roads that clear their own lots) reached 1,796 / 48.3 and 1,424 / 46.7.

| Measure | v21 | v22 | v23 (seed 0 / seed 1) |
|---|---|---|---|
| City score (mean) | 46.9 | 46.9 (49.6 / 44.1) | 47.5 (48.3 / 46.7) |
| Planning score (mean) | 42.1 | 43.8 | 45.5 (47.6 / 43.3) |
| Zones over budget (zoned less next) | – | 16 (0) | 8 (6) |
| Re-zones of its own strip / roads into zoned lots | – | 5 / 2 | 0 / 0 |
| Loans | 1 ($3,000) | 1 ($4,500, overridden) | 1 ($3,333) |
| Overrides | 0 | 4 | 1 |

- **Planning:** after a zone over budget it zoned the part that fits in 6 of 8 cases. In v22 it sent a message or moved on every time.
- **Transport:** neither city reached the 1,500 residents that open the bus stops before month 12 ended, so the bus-line changes need a long run.
- **Money:** one allocation failed on seed 1 because the money line rounds the treasury ($1,938) above what it held.

## 120-month runs on b684672 (2026-09-30)

**120-month runs on main at b684672** (2026-09-30: the v22/v23 fixes; 17.3M and 20.6M prompt tokens, run in
parallel; the Mac slept overnight, so wall times don't count):

| | seed 0 | seed 1 | v19 (fd5112f) |
|---|---|---|---|
| city score at month 120 (average over the run) | 64.3 (51.4) | **75.4** (60.5) | 73.5 / 69.7 |
| population | 10,848 | **16,156** | 18,412 / 13,064 |
| transport | **70.0** | 59.2 | 56.3 / 64.1 |
| debt at the end (months at the cap) | $49.0k (0) | $23.8k (0) | $47.5k / $22.0k |
| failed actions | 20.5% | 23.8% | 23.8% / 22.0% |

- **Transport:** both cities built a 7-stop bus line early (months 18 and 12; v19: never on seed 0, month 72 on
  seed 1), with transit at 15–21% by month 24. It then fell to 4–6% by month 48 and 1–4% by the end as the cities
  outgrew the line, and the full-buses rule offered no second line. Since 49fdafd another line goes where no stop
  reaches; since then it is offered while transit is under 12% and the line adds riders, named on the mayor's
  needs line before avenues, and transport's briefing has a Next line (Guards, "Transport's next step").
- **Seed 0:** planning asked the mayor for money 18 times (v19: 37), spent $7.8k (v19: $13.3k) and zoned 40% fewer
  tiles while residential demand stayed at its maximum. The zone hint and playbook line said "zone fewer tiles rather
  than asking"; they now say to zone the part that fits and ask for the rest.
- **Utilities:** power and water coverage stayed at 96–100% all game on both seeds; v19 dipped to 77–89% power.

## 120-month region run (2026-10-05)

**120-month region run on main at 615bff3** (2026-10-05, `configs/fleet_region_both_120.toml`: `fleet_region_smoke.toml`
with the fleet in both cities and `months = 120`; 640 fleet turns, 38.3M prompt tokens, 4.4 hours, seed 0). It is
the first long run with the mayor's Open requests block, departments asking only for what the mayor can fund, and
`send_message` to several departments at once.

| | north | south | region | scripted, same map and cadence |
|---|---|---|---|---|
| city score at month 120 | **79.3** | 73.6 | **76.4** (average over the run 61.6) | 67.3 (`region.toml`, `act_every = 3`) |
| population | 18,860 | 17,828 | **36,688** | 13,148 |
| teamwork (collaboration score) | 62.2 | 60.5 | | |

- **The mayors answered every request:** 78 of 82 and 74 of 75 requests to them got a top-up or a reply (100%
  responsiveness once open ones are left out), against 59% on v23. 98 of 492 `send_message` calls went to several
  departments, and none failed on a recipient name.
- **Departments' unfundable money asks:** 125 were held back once (`refused: "ask"`). Utilities' reasoning shows a
  conflict: its plan target ($3,000) against "no money ask to the mayor now: supply has room". By department:
  utilities 39, health 27, education 19, police 16, fire 14, transport 10; by reason: another building the mayor's
  money didn't cover 41 (in a hold 31 more), supply has room 28, blocked for lack of land 13, funds already covering
  the next build 12 (asks made right after building it). Every one came under a budget line saying not to ask; in 51
  the plan target was above the funds. 12 were sent again (6 topped up at the mayor's next turn). Of the 28 "supply
  has room" asks, 5 supplies reached 99%+ used before utilities' next turn, all 5 with demand plus the last quarter's
  growth at 90% of the supply. Since then a supply counts as wanted at 90% used now or within 3 months at its growth,
  a target above the funds of a department with nothing to buy reads `(unusable now)` on the mayor's side
  (`update_plan`'s `nothing_to_fund`), and the no-ask line and the hold-back call the target a ceiling, say when to ask
  and what to do this turn instead.
- **No deals and no messages between the cities** in 120 months, although both mayors had the deal tools and the
  Region block every turn. The Region view has still not shown real data. Why: each city imported power or water in
  about 20 of the 120 months (43 city-months, $5.1k of imports in all), and the other city had spare in 29 of those.
  But the City block's supply line counts imports, so it read "power 1,081 / 1,081" and the mayors concluded they
  weren't short. Nothing told either mayor that the other was short, what price would pay, or that no link is
  needed (the north mayor twice thought a connector was), and the 8-action cap went to allocations. Since then the Region block names
  each deal opportunity with its `offer_deal` call (see [Region mode](SIMULATION.md#region-mode-cities-that-trade-and-specialise)).
- **Weak spots:** south health 50.6, south police 54.8, transport 63–66 in both cities. Money aim (33–43) is low
  partly by construction: departments now ask only when the mayor can fund it, so many top-ups answer no message.
  Money aim now reads the needs the mayor saw (frames' `needs`); this run predates that record and keeps the
  message rule, and rebuilt offline from its mayors' to-fund lines it would read 92 / 89.

## Late-game smokes (2026-10-06/07)

First late smoke (2026-10-06, seed 0, `runs/20261006-152140_fleet_smoke_late_s0_861`, 32 fleet turns,
about 10 minutes): city 66.8 -> **69.1 (+2.3)**, population 6,036 -> **8,100 (+2,064)**, against the
scripted fleet's +1.1 / +268 over the same months. Transport read "another bus line ($1,500):
transit 11% now, about 22% with it", built it and an avenue at month 96 (transit 22% by month 99;
transport 92.1 -> 91.3), then ended its turns in 1 step once the Next line said nothing was left
("jammed roads are avenues already"). Teamwork 84.8 (money aim 100); 2 asks held back, 1 bus_line
refusal, and 3 failed actions in 32 turns.

Seed 1 (`runs/20261006-154117_fleet_smoke_late_s1_803`): city 61.3 -> 65.0 (+3.7), population
4,596 -> 6,472 (+1,876), against the scripted fleet's +6.2 / +1,532: more growth, a smaller score
gain. Teamwork only 44.1: departments' land requests to planning (riverbank sites, a jail, landfill
and cemetery sites away from homes) went unanswered. The region late smoke
(`fleet_region_late.toml`, seed 0, `runs/20261006-154117_fleet_region_late_s0_804`, 64 fleet turns,
29 minutes): city 66.5 -> 70.6 (+4.1), population 12,084 -> 17,984 (+5,900); teamwork north 91.7,
south 55.7. It made the first deal offer by an LLM mayor: south, short of water, was shown the
right call (`"sell": false`, buy 20 from north) but left `sell` out, the default made it an offer
to sell, and north rightly let it expire. `offer_deal` now requires `sell` (64d9ef2).

After planning's Next line, parks and Open requests block (233c529), the same two late smokes
(`runs/20261006-174455_fleet_smoke_late_s0_940` and `..._s1_941`): seed 0 city +4.2 (was +2.3),
population +3,524 (was +2,064), planning 62.9 -> 68.8; seed 1 city +5.6 (was +3.7), population +2,972
(was +1,876), teamwork 73.7 (was 44.1). Planning built 4 and 2 park blocks (the long runs: 1 park tile
in 244 planning turns). Some of the Next line's calls were sent again after they had succeeded earlier in
the turn (the model restarted from the stale briefing); every successful planning or transport action now
ends with the refreshed Next line (2495e4c). With that (`runs/20261006-215824_fleet_smoke_late_s0_868` and
`..._s1_869`): seed 0 +3.7 / +2,868, teamwork 99.4, 5 failed actions in 32 turns, planning in 47 steps
(was 71); seed 1 +6.7 / +2,324, past the scripted fleet's +6.2 on score too.

The region late smoke after the `sell` fix (`runs/20261006-222703_fleet_region_late_s0_550`): city
66.5 -> 70.7 (+4.2), population 12,084 -> 18,188 (+6,104), teamwork north 82.5 and south 93.0 (was
55.7). The first deals between LLM mayors: north sold south 20 water ($0.43, accepted the same turn,
ran its term), and north bought 40 power from south ($0.93, accepted the same turn). South cancelled
that one at month 105 although it earned $37/month with 243 power spare; an active deal's line now
says "keep it: it runs by itself, nothing to call" unless deal_check says to end it.

Two more seeds (2026-10-07; scripted yardsticks with every role scripted on the same start, months
96-107): single-city seed 2 (`runs/20261007-071421_fleet_smoke_late_s2_578`) +0.6 city / +1,524
residents against the scripted +1.4 / +868; region seed 1 (`..._fleet_region_late_s1_579`) +4.8 /
+5,388 against +5.1 / +3,520 (region seed 0's yardstick: +0.7 / +1,360 against the fleet's +4.2 /
+6,104). The fleet out-grows the scripted agents on every seed, but on seed 2 its services fell
behind: fire coverage 94% -> 78%, police 94 -> 79, health 83 -> 67, schools 93 -> 78, while the
treasury held $35-39k above the reserve. Each needs line asked for one building a quarter, and the
mayor funded exactly that. Replaying it showed the count mattered less than the site: the Ready line's "best"
station site, ranked by straight-line radius, sat on another road (a fire station serving one
building), and later no ready site reached the uncovered homes. With the keep-up needs lines
(6885467: the site that adds the most real coverage, the trend, several buildings when the money
allows, and what to do when no site helps) seed 2 reran at +2.3 city / +1,696 residents
(`runs/20261007-104643_fleet_smoke_late_s2_631`; was +0.6, scripted +1.4): fire 85.8 -> 84.2 (was
-> 70.1), police 74.5, health 89.3, education 83.1, and fire took the tier-2 hint (upgrade_facility).
On city score the fleet now beats the scripted agents' months 96-107 on all three single-city
seeds (+3.7 / +6.7 / +2.3 against +1.1 / +6.2 / +1.4).
