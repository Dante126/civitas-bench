# Fleet failures and idle turns (after smoke run v8)

Branch `fleet-failures`. Fleet-only changes (`civitas/fleet/`, `configs/fleet_*.toml`, `tests/test_fleet.py`,
docs): don't touch `core/observation.py`, `core/actions.py` or sim code. The scripted benchmark must not move:
`civitas run -c configs/baseline.toml --seed 0` gives city 76.7, population 14,092 before and after.
Checkpoint after each task: `pytest -q -m "not slow"` (276 passed at the start).

## Evidence (smoke runs `runs/fleet_smoke_s{0,1}_v8`, 12 months)

- 17 of 63 actions failed on seed 0 and 11 of 44 on seed 1. The largest group is "costs $X but the <dept>
  budget only has $Y": 12 on seed 0 and 4 on seed 1. Departments try to build at month 0 with $0, and later with
  less than the build cost. The mayor gave police and fire $1,000 each when a station costs $2,500. Costs appear
  only in the tool descriptions, never next to the sites the departments act on.
- Planning's road failures: 4 on seed 0 and 4 on seed 1, all "tile (x,y) is occupied by residential /
  commercial / industrial". On seed 0 month 0 planning zoned row 23 and then tried three roads north through the
  lots it had just zoned, which were still empty (dezoning is free). There were also 2 "every tile on that line
  is already road" and 1 "high density unlocks at 1,000".
- Idle turns (no successful action) were 17 of 32 on seed 0 and 20 of 32 on seed 1. They used 37% and 48% of all
  prompt tokens (0.80M and 0.82M), mostly by re-reading what the briefing already holds (get_plan, get_messages,
  my_facilities, unlocks, history), then inspect scans. Transport never acts, as its playbook says (nothing to
  do before ~1,500 residents), but the report's idle hint still flags it.

## Tasks

1. **Build costs and affordability.** Add `build_cost(w, role, purpose) -> int | None` in `fleet/tools.py`. It
   gives the cheapest unlocked building that a find_sites purpose's site is for, and must match what the build
   handler charges. Names go through `C.TIER_BY_NAME` / `C.TIER_LINES`: utilities coal_or_gas → coal/gas, wind →
   wind, pump → pump, sewage → outlet, near_homes → tower/solar/wind, riverbank_2x2 → water_works /
   advanced_treatment / nuclear; police station / jail; fire station / shelter; health clinic, hospital,
   landfill, cemetery; education school (elementary_school), high_school, university. Transport train_station /
   harbour / metro_station use `C.TRAIN_STATION` / `C.HARBOUR_SPEC` / `C.METRO_STATION`. It returns `None` for
   planning, levee (priced per tile), bus_stops and congested_roads.
   - Ready sites: each line with sites and a cost ends with `; $2,500`, or `; $2,500 - you have $1,000` when
     the funds are short.
   - The department's "Your budget" block: when funds are below the cheapest cost among its purposes that have
     ready sites, add one line: "You can't pay for any ready site (cheapest: <purpose> $X). Don't call build
     tools this turn: ask the mayor for the amount with a reason (send_message) unless you already did, then
     end_turn."
   - The mayor's Departments block: a department whose funds are below its cheapest ready build gets
     `; needs $X for a <purpose> (N ready sites)` on its line.
   - Tests: for every purpose with a cost, set the department's funds to cost - 1 on a grown city, run the build
     action on the listed site through `apply_action`, and check that the charged amount in the error message
     equals `build_cost`. Also cover the Ready sites suffix, the budget line, and the mayor line.
2. **Clearer action errors.** Add a fleet-side suffix in `ToolBox._act` on failed actions only. The core
   message stays first; this is the same mechanism as `_bank_warning`.
   - `build_road` "occupied by": list the empty zoned lots on the line (zone struct, `level == 0`) and the
     buildings. When lots block it, say "dezone them first (free): dezone x1=..,y1=..,x2=..,y2=.." (the lots'
     bounding box on the line when every tile in that box is zoned and empty, else the first lot); for buildings,
     say "route around (the Map block has the road extent)". Walk the line with `actions._line`.
   - `build_road` "already road": give the connected-road extent (`_roads_text`) and "start on the road and run
     into free land (find_sites road_extension)".
   - A `zone` locked on high density: "zone with density \"low\" now".
   - Tests use `apply_action`-backed ToolBox calls on a small grown city.
3. **Idle turns.**
   - Fleet playbook in both configs, kept identical: "Your briefing already has your plan slice, budget, new
     messages, map and ready sites: don't re-read them with get_plan / get_messages / history. When it says you
     can't pay for anything and nothing needs an answer, send at most one request and end_turn."
   - `report._idle`: a department turn doesn't count as idle when the department had no plan budget (none or 0)
     for that month and its funds were under $1 all month (history `dept_funds`; check the history-index
     convention used by `target()` in `_hints`). Such a turn splits idle runs. Test: transport with $0 and no
     target gets no idle hint; a funded department that doesn't act still does.
4. **Small fixes.**
   - `_bank_warning`: count a water tower (`C.TOWER`) as water, so a tower-supplied city isn't told it needs a
     pump. The outlet side stays as it is.
   - Disconnect refusal: "1 road tile" singular.
   - Identical repeats of calls with invalid arguments (a ToolError from `_resolve`): key them on the tool
     name plus the parsed arguments (or the raw string when the JSON is bad) and return `"refused": "repeat"`
     with the earlier error, as failed actions already do. Clear the cache with the same rules (a successful
     action).
   - Tests for each.
5. **Gaps.**
   - Guard `demolish` of a train station, harbour or airport that would cut highway-connected roads off the
     region. `cut_off` gains an option to drop those links; the refusal and `confirm: true` work as for bulldoze
     (add `demolish` to the confirm-param tools only for roles that can demolish those buildings, i.e.
     transport).
   - An education `library` purpose: reuse the school list (like cemetery reuses landfill), locked until
     `library`, with a cost of $1,500.
   - Pump and sewage lists: the observation caps riverbank candidates at 6 before the fleet drops sites off the
     highway-connected network. When the filter drops any, top up from the free connected bank lots
     (`_riverbank_by_roads(w)[0] & _empty_land(w)`) not already listed, up to MAX_SITES. Pumps go north first,
     sewage south first, each with `pollution` computed as the observation does (`max(w.pollution[y, x],
     w._river_near(x, y))`).
   - Tests for each.
6. **Docs and smoke run v9.** Update the README "Agent fleets" section and the CLAUDE.md fleet paragraph where
   behaviour changed. Run live v9 on seeds 0 and 1 (trinity-large-thinking) and compare with v8: failed-action
   rate, idle-turn prompt tokens, population and score. Write the results into the README smoke paragraph.
