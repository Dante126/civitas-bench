# Mayor funding (after smoke runs v9–v12)

Branch `mayor-funding`. These are fleet-only changes (`civitas/fleet/`, `configs/fleet_*.toml`, `tests/test_fleet.py`,
docs). Don't touch `core/observation.py`, `core/actions.py` or sim code. The scripted benchmark must not move:
`civitas run -c configs/baseline.toml --seed 0` gives city 76.7, population 14,092.
Checkpoint after each task: `pytest -q -m "not slow"`. The full suite had 345 passing tests at the start.

## Evidence (`runs/fleet_smoke_s{0,1}_v{9..12}`, 8 runs of 12 months)

- **Planning sits on money.** It held $7,000–10,700 unspent all game in 7 of 8 runs; the one exception is v11
  seed 0, where the mayor took $4,600 back. Planning spends about $1,000 a quarter, since roads cost $10 a tile
  and zones $5–8. The mayor playbook's start ("planning $9,000") locks away a third of the treasury.
- **The briefing invites top-ups.** The mayor's Departments block shows "planning: funds $7,834 < target $9,000", and
  its summary says "N below target … allocate_budget". So the mayor tops up a department that doesn't spend: in
  v12 seed 0, planning went to $10,721. Nothing in the briefing shows what a department spent.
- **The treasury runs short.** The mayor took loans in 6 of 8 runs ($700–6,000 each, often two). Education
  never built a school in 5 of 8 runs (score 33.3): it got nothing, or $1,500 for a $2,500 school. Police and
  fire stations came a quarter late in several runs.
- **Duplicate allocations.** The same allocation was repeated within one turn in 2 of 8 runs. In v10 seed 0,
  utilities +$5,500 twice and health +$2,000 twice. In v12 seed 0 month 3, the batch of utilities +400,
  planning +1,112 and health +2,000 was sent three times. Each allocate_budget adds to what the department
  already holds, and the successful action clears the repeat cache, so nothing stops it.
- **Briefing size.** The mayor's briefing is at 7,409 of the 7,450 characters allowed by
  `test_briefings_stay_bounded_at_the_limits`.

## Tasks

1. **Spending and idle money in the mayor's Departments block.**
   - Each department line shows what it spent over the last 3 months (the history's monthly `dept_spent`),
     e.g. `- planning: funds $7,834, spent $1,166 in 3 months, target $9,000`.
   - **Idle money.** A department keeps max(2 × its 3-month spending, the price of its cheapest ready site
     (`site_price` / `cheapest`, the same figure as the needs line), $1,000). Anything above that, if it is
     $1,000 or more, is idle. The line gets `; idle: $X more than it uses - take it back (negative
     allocate_budget)`. Every department has had a turn since the mayor last moved money, because the mayor acts
     first each quarter. So money it holds at the mayor's briefing was there for at least one of its turns.
   - `funding_gaps` leaves idle departments out, so they don't count as "below target". The summary line names the
     idle total when there is any, e.g. "$6,500 idle in planning, transport: take it back before loans or new
     top-ups". The `update_plan` result uses `funding_gaps` and follows automatically.
   - To make room, the mayor's Map block drops the road lines (the mayor has no road actions). Every department
     keeps them.
   - Tests:
     - a planning with $7,834 that spent $1,166 is idle by $5,502 and is not a gap;
     - a fire saving $1,500 toward a $2,500 station is not idle;
     - a transport holding $2,000 with no ready priced site is idle by $1,000;
     - the spending text;
     - the summary line;
     - the mayor's Map block has no road lines and departments' do;
     - the size bound holds.
2. **Duplicate allocation guard.** An `allocate_budget` identical to one that already succeeded this turn (same
   department, same amount) is refused before it runs. It works like the bulldoze guard (`refused`, not
   executed, not logged as a failure) unless it passes `confirm: true`. The message: "you already moved $400 to
   utilities this turn (it now has $X); allocations add up - repeat with confirm: true only to add it again".
   Different amounts, and negative amounts (taking money back), are not refused.
   - Tests: the v12 batch replayed; confirm; different amounts; the refusal isn't logged or cached.
3. **Mayor playbook** (both configs, identical).
   - Start: planning $3,000 (roads $10 a tile, zones $5–8: a quarter's work), utilities $7,000.
   - From month 3: $2,500 each to police, fire and health, and to education once its needs line shows school
     sites.
   - Each turn, take idle money back (the briefing says how much) before allocating new money or taking a loan.
     Allocations add up, so allocate once per department per turn.
   - Keep the reserve and loan rules.
   - Trim so the playbook stays about the same length.
4. **Docs and smoke run v13.** Update the README "Agent fleets" section and the CLAUDE.md fleet paragraph. Run live v13
   on seeds 0 and 1 and compare with v9–v12 on these measures, then write the results into the README smoke paragraph:
   - money idle in departments per quarter;
   - loans;
   - duplicate allocations;
   - when police, fire, health and education got their first building, and education's score;
   - population and score.
