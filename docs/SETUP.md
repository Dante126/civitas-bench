# Setup and running

How to install Civitas Bench, plug in a model provider, run single games and benchmarks from the command
line, and what a run writes to disk. The browser control center has its own page: [VIEWER.md](VIEWER.md).

Contents:

- [Install (macOS)](#install-macos)
- [Plug in Arcee models (trinity-large-thinking)](#plug-in-arcee-models-trinity-large-thinking)
- [Benchmark](#benchmark)
- [Output files (`runs/<run_id>/`)](#output-files-runsrun_id)

## Install (macOS)

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

## Plug in Arcee models (trinity-large-thinking)

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

## Benchmark

```bash
# the same line-up across 5 seeds (maps and disaster schedules differ by seed)
civitas bench -c configs/arcee.toml --seeds 0-4 --workers 3

# role-isolated eval: only Planning is an LLM, the other roles are fixed baselines,
# so any score difference comes from the Planning agent
civitas bench -c configs/eval_planning.toml --seeds 0-4
civitas bench -c configs/eval_planning.toml --seeds 0-4 --agent planning=llm:arcee/trinity-mini   # compare models

# the fleet reference: the scripted baselines played through the fleet tools on the fleet's settings (free)
civitas bench -c configs/fleet_scripted.toml --seeds 0-4 --workers 5

# a late-game smoke: scripted agents play months 0-95, the configured fleet months 96-107 (see FLEETS.md, "Late-game smokes")
civitas run -c configs/fleet_smoke_late.toml
civitas run -c configs/fleet_smoke.toml --set game.months=72 --warmup 60      # the same, from any config
civitas prompt -c configs/fleet_smoke_late.toml --role transport               # its first briefing after the warm-up
civitas yardstick <run_id>     # a warm-up run's scripted yardstick (written by `run` at the end; no model calls)

civitas leaderboard            # every finished run, grouped by agent line-up
```

Baseline reference (all scripted, seeds 0–4, civitas-v2): **city 76.6 ± 1.6**, mayor 89.6,
planning 67.3, utilities 90.6, transport 73.6, police 74.8, fire 87.4, health 86.1, education
83.8, population ≈ 13.6k, no bankruptcies, ~4–5 s per game (seeds 5–9: city 77.5 ± 1.2). `noop`
and `random` line-ups score 0. Acting every 3 months, as the fleet configs do, the same agents
reach city 69.2 ± 1.3 and population ≈ 7.4k; `configs/fleet_scripted.toml` plays them through
the fleet tools and scores exactly that (see [Scripted fleet](FLEETS.md#scripted-fleet)).

**civitas-v2 benchmark change** (the scripted mayor and the mayor's score; the sim is unchanged):

- The mayor's *deployment* term (spending against the money departments hold idle) averages the
  department funds over the scoring window instead of reading the final month only
  (`scoring.py:_deployment`), so money taken back in the last month no longer counts as never held.
- The scripted mayor reclaims idle department money (`agents/scripted.py:ScriptedMayor`): a service
  keeps its working balance until its first building, then twice what it spent over the last 6 months
  ($1,000 to its usual balance) and asks for a build it can't pay; an ask a department was fully
  funded for and sat on (scripted planning held $20k for a casino through the last years of seed 0)
  goes unfunded while it repeats it; and money well above a department's working balance and twice its
  recent spending is taken back before any loan or top-up. Planning keeps its
  $7,000 working balance: the scripted planner sizes its roads, zones, parks and plazas by its budget,
  and holding it to twice its spending cost 7 city points.
- Take-backs have hysteresis (`ScriptedMayor.HOLD_TURNS = 2`): a department keeps the largest working
  balance of this and its last two turns, and one that asked for money in them keeps it unless it sits on
  the ask. Without it the mayor flip-flopped: utilities' balance swings $7,000 ↔ $3,000 as supply's margin
  crosses 1.3x, so it took $4,000 back and returned it the next quarter, and a service lost $2,000 as its
  last build left the spending window, then got it back for its next station ask. Over the 96-month
  warm-up of `fleet_smoke_late` (seeds 0–23), take-backs refunded the next turn fell from 54 to 9 (0–6 per
  seed to 0–2) and take-backs from 170 to 82. Idle department money over the last 12 months went from
  ~$34k (v1) to ~$23k (monthly) and ~$30k to ~$21k (every 3 months); ~$19k / ~$17k before the
  hysteresis, which keeps money a turn or two longer.
- Reference numbers before (civitas-v1), with the first take-backs, and after (with hysteresis):

| scripted, 120 months | v1 city | v1 population | first take-backs | population | v2 city | v2 population |
|---|---:|---:|---:|---:|---:|---:|
| `baseline.toml` seeds 0–4 | 77.5 ± 1.5 | 14,466 | 76.8 ± 2.0 | 13,786 | 76.6 ± 1.6 | 13,598 |
| `baseline.toml` seeds 5–9 | 75.9 ± 0.8 | | 76.0 ± 2.1 | 13,172 | 77.5 ± 1.2 | 14,122 |
| `fleet_scripted.toml` seeds 0–4 | 68.8 ± 2.0 | 7,341 | 69.8 ± 1.9 | 7,786 | 69.2 ± 1.3 | 7,373 |
| `region.toml` seeds 0–2 | 77.2 ± 1.3 | 29,159 | 76.9 ± 1.8 | 28,479 | 77.0 ± 1.0 | 28,604 |
| `region_trio.toml` seeds 0–2 | 77.1 ± 1.0 | 42,004 | 75.8 ± 0.5 | 39,444 | 76.2 ± 1.1 | 40,095 |

Building tiers changed these numbers too. Just before tiers were added, the same baseline scored city
78.1 ± 1.8 with population ≈ 14.7k.
The scripted agents leave obvious room: they never use districts, mixed use, historic
listings or soil clean-up, rarely use policies or funding levels and build services by simple
coverage rules. In a region the scripted mayors make the
neighbour deals their observation names (see [Region mode](SIMULATION.md#region-mode-cities-that-trade-and-specialise)).

Useful switches in `[game]` (or `--set game.X=...`):

| setting | effect |
|---|---|
| `events = "auto"` / `"none"` | seeded floods, heatwave, recession and a factory offer (same on every line-up for a seed) |
| `milestones = true` | buildings unlock at 1k / 5k / 10k residents |
| `simultaneous = true` | every role sees the same start-of-month state (harder: no reacting to earlier roles) |
| `roles = [...]` | which roles play, in turn order |
| `seasons = true` | winter heating and snow, summer cooling and fires, seasonal tourism |
| `region = true`, `cities = [...]` | 2–6 cities, each with its own team (see below) |
| `warmup_months = N` | scripted agents play months 0 to N−1, the configured agents the rest (`--warmup N`; see [Late-game smokes](FLEETS.md#late-game-smokes-and-the-yardstick)) |
| `team_weight` | reward = w·Δcity + (1−w)·Δrole − penalty·invalid actions |
| `balance` | override any constant, e.g. `"CATALOG.coal.cost" = 3000`, `"SERVICES.clinic.radius" = 9` or `"TIERS.pump.2.water" = 800` |

## Output files (`runs/<run_id>/`)

(`civitas eval` also writes `runs/eval_<name>_<time>/` with `eval.json` and `scorecard.md`; its runs are ordinary run
folders beside it.)

- `meta.json`: config, agents, map size, natural resources; after a warm-up also `warmup_months` and the warm-up line-up (`warmup`)
- `frames.jsonl`: one frame per month (tiles, levels, variants, road classes, overlay layers, metrics, scores, turns, council; with fleet agents also `needs` in the months they act, read by the collaboration score's money aim). The viewer replays these.
- `trajectories.jsonl`: one record per agent turn: the full prompt and raw response (LLM agents) or the observation text (scripted agents), actions, results, reward and scores; a warm-up's turns carry `"warmup": true`
- `summary.json`: final and average scores, per-agent stats (invalid rate, spending, messages, LLM tokens, latency)
- `fleet_report.json` / `fleet_report.md` (runs with fleet agents): scoreboard, plan, per-role tool use and failures, collaboration score, coaching hints (see [Reports](FLEETS.md#reports-and-the-collaboration-score)); in a region, one section per city with fleet agents
- `checkpoint.ckpt` / `playbooks.toml` / `PAUSED` (a supervised run between sessions): the game to resume, the playbooks to edit, the resume command (see [Modes](FLEETS.md#modes-and-supervised-sessions)). The checkpoint is a pickle signed with this installation's key (`~/.config/civitas/checkpoint.key`); resume refuses one whose signature is missing or doesn't match, so it only resumes on the machine and account that made it

`python examples/export_sft.py runs/ --role planning --min-reward 0.001` turns the good
turns into a chat-format dataset for fine-tuning or as the start of a GRPO / preference loop.
It skips fleet turns (multi-step tool-calling traces, which don't fit its one-shot format) and prints how many.
