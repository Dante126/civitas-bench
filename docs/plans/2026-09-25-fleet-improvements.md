# Fleet improvements (after the first smoke runs)

Branch `fleet-improvements`. Fleet-only changes: the scripted benchmark and legacy agents stay unchanged.
Checkpoint after each task: `pytest -q -m "not slow"` (234 passed at the start).

## Evidence (smoke run `runs/fleet_smoke_s0_v5`, 12 months, seed 0)

- Sewage never got built. Planning bridged the river at y=24 on month 0 and then zoned the riverbank
  tiles beside that road, so no bank site was left. `find_sites(sewage)` still said "a road has to reach the
  river first", so utilities and planning chased a road that already existed.
- Planning used 7–12 of its 20 tool calls per turn re-running `find_sites` for every zone type, ended all 4
  turns on the step limit, and tried to build a road to x=55 on a 48-wide map.
- Seven council messages asked planning for road access / free lots; its briefing only showed a count, it
  read them in 2 of 4 turns and never acted on them.
- The mayor called `find_sites` 6 times (the shared fleet playbook recommends it; the mayor has no such tool).
- The mayor wrote plan budgets that departments then saw as targets with $0 in funds.

## Tasks

1. **Requests in the briefing.** The "since your last turn" digest lists the newest council messages from
   other roles addressed to this role or `all` (up to 5, ~200 chars each, sender + month), newest last, instead
   of only a count; keep the "(N more: get_messages)" pointer when there are more.
2. **Map orientation.** A short "Map" block in every briefing (map size, highway entry, river columns by row
   band, extent of the highway-connected roads, built-up box) and a `map_overview` look-up returning a compact
   ASCII map (≤ 3,800 chars, legend) for all roles.
3. **Ready sites in department briefings.** One line per `find_sites` purpose: count and best site, or the short
   reason when empty (locked / nothing needs it / no free land / riverbank taken). The riverbank note must tell
   apart "no road reaches the river" from "the bank tiles next to roads are taken - ask planning to free or
   leave a riverbank lot".
4. **Step budgeting and repeats.** Every tool result carries `steps_left`; with ≤ 3 left add "wrap up: call
   end_turn". An identical repeat of a look-up within a turn returns its result with "same as call N"; an
   identical repeat of an action that failed this turn is refused with the earlier error (not charged as an
   invalid action).
5. **Mayor fixes.** The mayor gets a read-only `find_sites` over every department's purposes
   (`<department>.<purpose>`), so it can check a department's claims. `update_plan`'s result lists departments
   whose funds are below their new target (with the gap) and reminds the mayor to `allocate_budget`.
6. **Playbooks, smoke run, docs.** Planning: keep roads inside the map, leave a riverbank lot beside the road
   for utilities, answer requests in the briefing before zoning. Fleet playbook: `find_sites` wording that fits
   every role. Live smoke run (v6, seed 0) compared with v5; README/CLAUDE.md updates.
