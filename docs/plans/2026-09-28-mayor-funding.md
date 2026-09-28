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
5. **After smoke v13: blocked is not idle.**
   - v13 seed 0, month 6: planning had zoned every roadside lot. Police, fire, health and education each held exactly
     their first building's price with nothing spent. With no ready sites the keep fell to the $1,000 floor, so
     each was flagged "idle $1,500 / $1,000" and the mayor took the money back. Utilities at month 9: "idle
     $6,000" while no plant site was free.
   - Fix:
     - Money for an unlocked primary first building whose list is empty for lack of land or roads ("no free
       roadside land", "no roadside lot away from homes", riverbank taken, "no road reaches the river") still
       counts toward the keep, at that building's price.
     - "Nothing needs it", locked, and the SITE_NOTES reasons don't count.
     - The mayor's line says "; blocked: no free lot for a <building> - ask planning" when a department's first
       buildings are all blocked, so the mayor pushes planning rather than taking money back.
   - Then run smoke v14.
6. **Review fixes for Task 5 (after smoke v14).**
   - **Riverbank need (Important).** `_why_empty` flags pump and sewage as blocked whenever the bank lots beside
     roads are taken, without checking whether they're needed. Live v14 seed 1, months 6 and 9: pump and outlet
     already built, water 370-490 against demand of 99-116, sewage 250 against 99-116. Yet planning read "blocked for
     lack of land: utilities (pump, sewage) ... dezone a roadside lot for them" (the playbook says to free a bank
     lot when asked). Fix: in `blocked_firsts`, count pump / sewage only when `_bank_needed(w)` names them or their
     demand is at least 0.9 × supply. Test from a grown scripted state with the bank taken: `land_blocked == {}`.
   - **2x2 lists.** University (and hospital) lists go empty for lack of a free 2x2 roadside block but read "nothing
     needs it right now", so the university keep ($12,000) is flagged idle past 5,000 residents. Fix: `_why_empty`
     checks for a free 2x2 roadside block (`w.site_candidates(..., size=2)`) and returns "no free 2x2 roadside
     block: ask planning for one", blocked.
   - **Planning's line.**
     - Name only each department's cheapest blocked first building(s), as the mayor's line does, so the full form
       fits past 1,000 residents.
     - Riverbank purposes say what fixes them (a road to the river, a free bank lot), not just "dezone a roadside
       lot".
   - **Size.** The mayor's size test's blocked departments hold realistic 4-digit funds. If under ~20 characters
       of headroom remain, say "ask planning" once in the summary line instead of on each department line.
   - **Tests and tidy-up.**
     - Assert the blocked flag for "riverbank all taken" and "no free 2x2 riverbank block".
     - Drop the redundant assert.
     - Name the jail in the `dearest == 4000` test.
     - Compute `min(...)` once in `first_build`.
   - Then run smoke v15.
7. **After smoke v14: lots near homes (planning zones them before the services can build).**
   - **Evidence.** Live v14 seed 0.
     - At months 3 and 6, the mayor's briefing read "fire: needs $2,500 for a fire station (4 ready sites)" and
       the mayor funded fire, health and education.
     - Planning then zoned the roadside lots near homes (it acts between the mayor and the services).
     - Fire, health and education read "station / clinic / school: none - nothing needs it right now", with
       coverage 0.
     - Fire held $2,500 from month 3 to 12 and never built (score 37). Health's clinic and education's school never
       came (33.3).
     - Across v9-v14 (12 runs), 16 turns showed "nothing needs it" on a primary list with coverage under 50%: 8 in
       v14 s0, 4 in v9 s0.
     - The need-ranked lists (police and fire stations within radius 6, clinic 5, school 6; observation.py ~303-433)
       only hold free roadside tiles with uncovered need within that radius. So an empty list with free land left
       elsewhere is a land problem, not a lack of need.
   - **Classify it.** In `_why_empty`'s general path, for the need-ranked first-building lists: when there are
     uncovered homes (people the building would serve) but no free roadside tile within the list's radius of them,
     return "no free roadside lot near the homes that need it (around (x,y)): ask planning for one", blocked
     True. Keep "nothing needs it" for real zero need (everyone covered; for schools, no children).
   - **Prevent it.**
     - Planning's Ready sites block names what funded departments are about to build, one short line: e.g. "-
       leave free (funded): fire station (12,23), clinic (20,27): zone around them". Show it only when a
       department's funds cover its cheapest ready primary first building. Reuse Task 5's per-department lists.
     - A `zone` that takes the last ready site(s) of such a building warns, like the riverbank guard: "WARNING: that
       zoned fire's last fire station site near homes (fire holds $2,500 for it): dezone one, e.g. (12,23)".
   - **Tests** for each.
   - Then run smoke v15.
8. **Review fixes for Tasks 6-7 (after smoke v15).**
   - **v15.**
     - s0: 1,508 / 43.0; s1: 1,424 / 41.1.
     - All four services got their first building at month 3 (s0) and 6 (s1). v14 s0: fire, clinic and school never.
     - Education 58.7 / 60.1; v13-v14 s0: 33.3.
   - **Utilities on the leave-free line (Important).** All 8 of v15's leave-free lines named a water tower with water
     never short. utilities' near_homes list ranks lots by homes around them, so planning is told to hold its best
     residential lot. Fix: in `funded_firsts` and `blocked_firsts`, count utilities' non-riverbank purposes only
     while wanted, the way `_bank_wanted` does for the bank: water tower on water, coal / gas / wind on power.
   - **Shared sites (Important).** v15 s1 month 6 named (34,25) for four buildings; only police could build there.
     Fix: `funded_spots` goes through departments in turn order and gives each building its first site not claimed
     by an earlier one. `spare_funded`, `_zone_beside` and `_funded_warning` use the claimed sites: warn when a zone
     leaves a funded department no unclaimed site.
   - **Truncation.** v15 s0 month 9 cut the leave-free line at 240 characters in mid-word ("...: zone arou").
     Drop whole items, never the tail; e.g. "(+2 more)".
   - **Mature cities (Important).** A 10% share of Σserved × (1 − coverage) flags well-covered cities: scripted month
     120 flags fire (11.5%, score 87.4), health and police. Fix: measure the share where coverage < 0.3 (the
     observation's `uncovered_buildings` test). Apply the same need gate to NEAR_NEED purposes on the "no free
     roadside land" path.
   - **Minor.**
     - Drop the shelter from NEAR_NEED.
     - 2x2 funded spots use `C.FOOTPRINT`.
     - The zone warning code falls back to no warning on an exception.
     - Tests: utilities on / off the leave-free line; riverbank plus funded gives one warning; `_funded` resets on
       a successful action and survives a failed one.
   - Then run smoke v16.
