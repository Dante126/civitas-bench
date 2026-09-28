# Planning fixes (after smoke runs v9–v11)

Branch `planning-fixes`. These are fleet-only changes (`civitas/fleet/`, `configs/fleet_*.toml`, `tests/test_fleet.py`,
docs). Don't touch `core/observation.py`, `core/actions.py` or sim code. The scripted benchmark must not move:
`civitas run -c configs/baseline.toml --seed 0` gives city 76.7, population 14,092.
Checkpoint after each task: `pytest -q -m "not slow"`. The full suite had 332 passing tests at the start.

## Evidence (planning in `runs/fleet_smoke_s{0,1}_v{9,10,11}`, 24 turns)

- Planning makes most of the fleet's failed actions:
  - 13 × "every tile on that line is already road", e.g. rebuilding (0,24)-(47,24), the main road;
  - 11 × zone "no empty land tiles in that area", often zoning the road it just built: build_road (35,23)-(35,8),
    then zone R x=35 y 9-22;
  - 21 × build_road "occupied by …": side roads through lots it had zoned, or through buildings;
  - 6 × high density before 1,000 residents, repeated across turns.
- 12 of its 24 turns end on the 20-call step limit. It makes 5.4 inspect calls per turn (every other role 0.4–1.2),
  and each one returns a JSON record per tile.
- The Map block only gives the road network's bounding box, e.g. "highway-connected roads: x 0-25, y 19-29, 58
  tiles". The network is actually three straight lines, y=19, y=24 and x=19. Planning can't see the lines without
  inspect or `map_overview`, which is about 1.2k tokens and which the rules say to call at most once a turn.

## Tasks

1. **Road lines in the Map block.** `map_facts` / `map_lines` add a "road lines" line listing the maximal straight
   runs of highway-connected road, horizontal and vertical, of at least 3 tiles. List the longest first, up to 8,
   then "+N more". Example: `- road lines: y=24 x 0-47; x=19 y 19-29; y=19 x 3-25`. Every role's Map block and
   `map_overview`'s summary share it.
   - Tests: a hand-built road network gives the expected lines; a crossing tile appears in both lines; runs shorter
     than 3 and roads off the connected network are left out; the cap and "+N more" work.
2. **Zone and dezone hints.** `ToolBox._explain` gains two cases, on failure only, keeping the core message first:
   - zone "no empty land tiles in that area": say what the rectangle holds, e.g. "all 14 tiles are road" or
     "10 zoned R, 4 road". When empty roadside land lies right beside the rectangle (the lines just outside each
     side, over the rectangle's span), suggest up to 2 zone rectangles trimmed to that empty land: "zone beside it
     instead: x1=34,y1=9,x2=34,y2=22 or x1=36,...". Only land touching a highway-connected road counts.
   - dezone "no zoned tiles in that area": say what the rectangle holds.
   - Tests replay the v9 case (build_road (35,23)-(35,8), then zone x=35 y 9-22, then the suggested rectangle
     succeeds) and cover the counts, a rectangle with no free land beside it, and dezone.
3. **Fewer inspect calls, fewer locked zone attempts.**
   - inspect: from the 4th inspect in a turn, if `map_overview` hasn't been called this turn, the result gets a
     note: "inspect #N this turn: map_overview shows the whole map in one call". The count resets each turn (one
     ToolBox per turn). This applies to every role.
   - Planning's Ready sites block gets one line while `high_density` is locked: "- high density: locked until
     1,000 residents (now N): zone with density \"low\"".
   - Planning playbook (both configs, identical): build a road first, then zone the tiles beside it, never the road
     itself. Branch new roads off the road lines in your Map block into free land; lay side streets before zoning
     beside them, because a road can't cross zoned lots or buildings. When you need the layout, call map_overview
     once instead of several inspects.
   - Tests for the note (none on the 3rd call, one on the 4th, none after a map_overview) and the Ready line.
4. **Docs and smoke run v12.** Update the README "Agent fleets" section and the CLAUDE.md fleet paragraph. Run live v12
   on seeds 0 and 1 and compare planning with v9–v11: failed actions by kind, step-limit turns, inspect calls per
   turn, then population and score. Write the results into the README smoke paragraph.
