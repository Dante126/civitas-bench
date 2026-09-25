# Agent fleets — design

Goal: rework how agents play so Civitas measures how well an agent fleet manages tasks and resources, and so
humans can coach fleets. Server / company deployment (daily ticks, accounts, web UI) is out of scope here and
designed later; this doc covers the agents.

## Why

The current `LLMAgent` gets one 3–5k-token observation dump per turn and answers with one JSON block; actions
are applied blind, memory is a single rewritable note, and eight peer agents coordinate through a message board.
A 36-month all-trinity run (2026-09-25) scored 47 vs ~77 for the scripted baseline: 19% of actions failed
(locked options, overspending, occupied tiles), 4 turns hit the token cap, 2 were unparseable, and the mayor
starved departments of money. These are structural problems, not model problems.

## Decisions

- **Fleet = 1 manager + 7 departments.** The manager (mayor) sets goals, budgets and priorities in a city plan;
  planning, utilities, transport, police, fire, health and education execute. Each agent has its own model,
  playbook and limits.
- **Tool-use loop per turn.** Agents start from a short briefing and call look-up and action tools; actions run
  immediately and return results, so mistakes are fixed within the turn. Native OpenAI-style tool calling
  (verified on trinity-large-thinking via the Arcee API, <1 s per step).
- **Three memory layers:** a shared city plan (manager-owned), per-department working memory (agent-owned), and
  an automatic history log written by the game. No chat history between turns.
- **Coaching surface:** fleet playbook + department playbooks, model per agent, per-agent limits. Versioned.
- **Two modes:** *supervised* (coach edits between days) and *autonomous* (instructions fixed at month 0, the
  fleet plays to the end on its own; can run at full speed — the standardized eval track). Ranked separately.
- The simulation, actions, scoring, scripted baselines, legacy `LLMAgent` and remote API stay unchanged.

## Turn cycle (per game month)

1. **Manager phase.** Briefing: headline metrics and trends, finances, events, last department reports, fleet
   playbook. Tool loop with look-up tools and manager actions (`set_tax`, `allocate_budget`, `set_funding`,
   `enact_policy`, `repeal_policy`, `take_loan`, `repay_loan`, offers, districts). Updates the city plan with
   `update_plan`. Ends with `end_turn`.
2. **Department phase**, in the existing turn order. Briefing: its slice of the plan (budget, priority, goals),
   budget, key metrics, "since your last turn" digest, messages, department memory, playbook. Tool loop with
   look-up tools and its existing actions. Ends with `end_turn(report)`; reports feed the manager's next briefing.

**Limits:** max tool calls per turn (default 20) and a token budget; on limit the turn ends, completed actions
stay. A crashed turn is logged and becomes a no-op.

## Tools

**Look-up (all agents, no side effects):**
- `city_overview()` — headline metrics, trends, finances.
- `inspect(x, y, radius)` — small tile window: building, tier, level, coverage, pollution (replaces the ASCII map).
- `find_sites(purpose, near?)` — ranked candidate sites (e.g. `clinic`, `pump`, `2x2 riverbank`), crowding-aware.
- `my_facilities()` — the department's buildings with load, capacity, tier, available upgrades.
- `get_plan()`, `get_messages()`, `history(kind, months)` — memory layers (structured query: failures, actions,
  events, reports, metrics).
- `unlocks()` — locked options and when they unlock.

**Actions:** existing actions exposed as functions with JSON schemas, executed through `apply_action` (same
validation, costs, milestone gates). Result: `ok`, message, cost, remaining budget. Malformed arguments return an
error the model can correct.

**Turn control:** `send_message(to, text)`, `update_memory(...)`, `update_plan(...)` (manager), `end_turn(report)`.

## Memory

1. **City plan** — `goals` (2–5), `budgets` ($ per department for the quarter), `priorities` (one line per
   department), `notes`. Manager writes via `update_plan`; departments see their slice. All versions kept.
2. **Department memory** — `commitments`, `open_tasks`, `lessons`; size-capped; edited via `update_memory`; shown
   in full in the briefing. In autonomous mode this is how strategy adapts.
3. **Automatic history** — every action with result/cost, monthly metric snapshots, events, department reports.
   Queried via `history(kind, months)`; briefings include a "since your last turn" digest.

All three live in the game state (survive between days, appear in replays).

## Coaching and reports

**Fleet config** (one TOML/JSON file per fleet; web UI edits it later): fleet playbook, department playbooks,
per-agent model / step limit / token budget, mode. Every change is saved as a timestamped version; runs record
the active version. Playbooks go into the system prompt below a fixed "how to play" section, so coaches steer
strategy but cannot change rules or tools.

**Daily / end-of-run report** (JSON + Markdown): scoreboard and delta, key metrics and trends, the plan and its
changes, per department (actions, failures, tool calls, tokens, reports), events, and coaching hints derived
from history (e.g. repeated failures, unused budgets).

**Per-turn measurements:** tool calls, tokens, latency, failed actions, lessons written — for side boards
(efficiency, best department) and autonomous-mode evals.

## Code structure

New package `civitas/fleet/`:
- `config.py` — fleet config, modes, versioning.
- `tools.py` — tool schemas and implementations over `World` (actions via `apply_action`).
- `memory.py` — city plan, department memory, history log (stored with the world state).
- `briefing.py` — briefings and digests (reusing observation code).
- `agent.py` — `ToolAgent`: generic tool loop on the streaming chat client (native tool calls, limits,
  argument validation, tool errors).
- `manager.py`, `department.py` — role prompts and tool sets.
- `report.py` — daily / end-of-run report.

The runner gains a fleet mode (manager phase, then department phase per month; every tool call logged to
`trajectories.jsonl`). `civitas run` / `civitas bench` accept fleet configs (e.g. `configs/fleet_trinity.toml`),
so autonomous fleets are benchmarked like any line-up. Supervised mode is approximated for now by stopping after
N months and resuming with an edited config; the daily server comes later.

## Testing

- Unit tests per tool (look-ups and actions, error paths).
- Fake OpenAI-compatible server replaying scripted tool calls: the full manager → departments loop in CI with
  no API key.
- Deferred: a scripted fleet playing through the tools. The scripted baselines stay as they are (benchmark
  unchanged); tool coverage is proven by unit tests and an end-to-end fleet run against a fake tool-calling server.
- A short trinity smoke run (autonomous, e.g. 12–24 months) checking tool-call validity, failures, tokens.

## Out of scope (later)

Server, daily scheduler, accounts / Google sign-in, web UI for coaching, shared-region leagues, hard guardrails
enforced by the engine, player-defined fleet shapes.
