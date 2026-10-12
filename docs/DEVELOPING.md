# Developing

Where the code lives, how to add a role or building, and open ideas. `CLAUDE.md` at the repo root has the
detailed architecture notes.

Contents:

- [Project layout](#project-layout)
- [Adding a role](#adding-a-role)
- [Ideas for next steps](#ideas-for-next-steps)

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
              · evals.py (civitas eval) · envs.py (environment versions) · models.py (registry) · stats.py
              · yardstick.py · collab.py (collaboration score) · compare.py · problems.py · live.py · control.py
  fleet/      agent.py (ToolAgent tool loop) · scripted.py (the scripted baselines through the tools) · tools.py · briefing.py
              · memory.py (plan, memory, history) · config.py · report.py · region.py · fmt.py
  rl/         bank.py (task bank) · episode.py (TaskEpisode) · reward.py · policies.py · check.py · evaluate.py
              · trl_env.py (TRL environment) · train.py · sft.py · modal_app.py (Modal pipeline)
  viewer/     index.html · app.js · scene.js (three.js renderer) · life.js (3D vehicles)
              · pixel.js (OpenTTD-style 2D renderer) · recipes.js (its buildings, ground, vehicles)
              · sprites.js (pixel-art painter) · lifesim.js (vehicle movement shared by both views)
              · region.js (region map) · app.css · vendor/ (three.js, bundled; works offline)
  server.py   civitas serve: the viewer, replay API, run launcher and live API
  cli.py      civitas run | resume | bench | eval | env | models | yardstick | serve | leaderboard | prompt
              | check-llm | rl
configs/      baseline · arcee · arcee_full · smoke · eval_planning · remote · region · region_trio · fleet_trinity · fleet_smoke
              · fleet_region_smoke · fleet_region_both_120 · fleet_scripted · fleet_smoke_late · fleet_region_late
              · fleet_*_v3 / fleet_open_late* (eval games) · eval_*.toml (eval suites) · models.toml (model registry)
              · envs/ (frozen environment fingerprints) · rl_*.toml (RL training)
docs/         SETUP · VIEWER · SIMULATION · FLEETS · EVALS · RL · HISTORY · DEVELOPING · MECHANICS (the design roadmap)
              · plans/ (design notes) · images/
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
