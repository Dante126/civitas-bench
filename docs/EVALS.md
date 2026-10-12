# Evaluating models

`civitas eval` plays a model's fleet from a scripted late-game hand-over and scores it against the scripted
agents from the same city. This page covers suites, the model registry, frozen environment versions
(civitas-v1, v2, v3), the statistics behind the leaderboard and the viewer's Evals page. The default suite is
`configs/eval_v3.toml` (civitas-v3, guided); `configs/eval_v3_open.toml` is the open track.

Contents:

- [Running an eval](#running-an-eval)
- [Registry, environment versions and the leaderboard](#registry-environment-versions-and-the-leaderboard)
- [Units, repeats and intervals](#units-repeats-and-intervals)
- [The scorecard and cost](#the-scorecard-and-cost)
- [Comparing models](#comparing-models)
- [First results and calibration (civitas-v1)](#first-results-and-calibration-civitas-v1)
- [civitas-v2: new seeds, guided and open tracks](#civitas-v2-new-seeds-guided-and-open-tracks)
- [civitas-v3: an equal token budget per turn](#civitas-v3-an-equal-token-budget-per-turn)

## Running an eval

`civitas eval` turns the late-game smokes into one repeatable model evaluation with one scorecard. A **suite**
(`configs/eval_late.toml`) lists the runs: `fleet_smoke_late.toml` seeds 0-2 (single city, 32 model turns each) and
`fleet_region_late.toml` seeds 0-1 (two cities, 64 each). In every run scripted agents play months 0-95, then the
model's fleet plays months 96-107, and the run is scored against its scripted yardstick: the same game from the same
hand-over played by scripted agents (`yardstick.json`, see [Late-game smokes](FLEETS.md#late-game-smokes-and-the-yardstick)).

```bash
civitas eval -s configs/eval_late.toml --model trinity-mini --dry-run   # the plan, its tokens and time; runs nothing
civitas eval -s configs/eval_late.toml --model trinity-mini             # 5 runs at once, ~25 min
civitas eval -s configs/eval_late.toml --model M --provider openrouter --name m-v1 --workers 3
civitas eval -s configs/eval_late.toml --model trinity-mini --repeats 3    # each seed 3 times: measures the noise
civitas eval --resume runs/eval_trinity-mini_<time>                     # rerun the runs that failed or didn't finish
civitas eval --report runs/eval_trinity-mini_<time>                     # rebuild the scorecard from the run folders
civitas eval --compare runs/eval_trinity-large-v1_<time> runs/eval_trinity-mini_<time>   # models side by side
civitas eval -s configs/eval_late.toml --adopt <run_id> ... --name x    # an eval of runs that exist already
```

- `--model` / `--provider` set `[fleet].model` / `.provider` and the model of every `[agents.<role>]` that names its
  own (fleet or llm). `--agent` and `--set` reach every run, as for `run`.
- The eval **refuses to start** (and `--dry-run` says why, exit code 2) when a fleet agent would play another model
  than the one asked for (e.g. a role pinned to `model = "scripted"`), when no agent calls a model, when a config has
  no warm-up, or when a provider is unknown or its API key env var (`ARCEE_API_KEY` for arcee, from `./.env`) is
  unset.
- Each run is a `civitas run` subprocess (`[eval].workers` at a time, default 5: the runs wait on the API, not the
  CPU), under `caffeinate -dims` on macOS, its log in `runs/_logs/<run_id>.log`. Every run is a normal run folder in
  `runs/`, so the viewer, Compare and the yardstick work as usual. Ctrl+C stops the runs; `--resume` reruns each run
  that didn't finish under a new id (`<id>_a2`), keeping the old folder.
- The eval folder `runs/eval_<name>_<time>/` (name: `--name`, else the model) holds `eval.json` (suite, model,
  provider, overrides, the code commit, each run's id, command, status and attempts, and the scorecard data) and
  `scorecard.md`.

## Registry, environment versions and the leaderboard

**Plug-in evals: the registry, environment versions, the leaderboard.** The everyday form is just a model key:

```bash
civitas models                                   # the registry (configs/models.toml) and whether each API key is set
civitas eval --model trinity-mini --dry-run      # the default suite (configs/eval_v3.toml): plan + env check
civitas eval --model trinity-mini                # env check, preflight, the 8 runs, the scorecard
civitas eval --leaderboard [--env civitas-v1]    # ranked evals of one version; writes runs/leaderboard_<env>.md
civitas env check | civitas env list | civitas env freeze civitas-v2   # environment versions
```

- **Registry** (`configs/models.toml`, `civitas/bench/models.py`): one `[models.<key>]` per model with only model
  keys (`provider`, `id` when the served id differs from the key, `temperature`, `max_tokens`, `extra_body`, `notes`).
  The compute budget (`max_steps`, `turn_tokens`, `turn_budget`, `act_every`, `max_actions_per_turn`, playbooks,
  `mode`) is the
  environment's, equal for every model, and refused in an entry. Built-in providers: arcee, openrouter, ollama,
  openai (sends `max_completion_tokens`) and anthropic (its OpenAI-compatible endpoint); any other gets a
  `[providers.<name>]` table there (`base_url`, `api_key_env`), passed to each run as `--set providers.<name>.*`.
  Keys live in `./.env`. A model not in the registry still works with `-s` and `--provider` as before.
- **Environment versions** (`civitas/bench/envs.py`): a suite's `[eval] env = "civitas-v1"` names the frozen
  fingerprint in `configs/envs/civitas-v1.json` (git tag `env/civitas-v1`). The fingerprint is behavioural: a probe
  replays every suite seed with the scripted fleet (the yardstick's line-up, no model, ~12 s for 8 seeds in parallel,
  cached per code state in `runs/_cache/`) and hashes the config without model keys, the fleet's system prompts and
  tool schemas, and per seed the briefings and tool results after the hand-over, every month's metrics, and a
  `model` part: what only a model-backed agent meets (the tool loop's nudges, the end_turn and money-ask refusals,
  refreshed Next lines). For it, on a copy of the city at the hand-over, every fleet role plays one real tool-loop
  turn against a canned stand-in model (no network): a reply without a tool call, one cut at the token limit, the
  scripted department's actions one call at a time, end_turn. A refactor leaves the fingerprint alone; a sim,
  briefing, tool or config change moves the part it touches (`civitas env check` names it). An eval on code that no
  longer matches is refused unless `--allow-drift` (then recorded as drifted and never ranked); `--resume` checks
  again (refused on a mismatch unless the eval allowed drift), and code that changes while an eval runs marks it
  drifted. Bumping is deliberate: freeze a new version and re-evaluate the reference models.
- **Preflight**: before any run, one real request with two fleet tool schemas must come back as a tool call (a wrong
  key, id or a model without tool calls fails here, for a few hundred tokens). `--dry-run` skips it and runs the env
  check; `eval.json` records `env` (name, fingerprint, match, drift), `model_entry` and `preflight`.
- **Leaderboard**: per version, the latest finished, non-drifted eval of each model, ranked by its score vs the
  scripted yardstick (95% interval) and its population / quality split (rows' `pop_term` / `quality`; an eval saved
  before the split gets it from its run cards), with the final city score, Teamwork, failed actions and refusals per
  model turn (`failed_per_turn` / `refused_per_turn`), departments, tokens, cost and the paired verdict
  against the suite's `baseline_model` (trinity-large-thinking). Evals of other versions, drifted, unfinished or
  superseded are listed as not ranked, with the reason. The viewer's Evals page (Models, Evals, Leaderboard) drives
  all of this from `civitas serve`. Stopping an eval (SIGTERM / SIGHUP to `civitas eval`, or Ctrl+C) stops its runs.

## Units, repeats and intervals

**Units, repeats and intervals.** A **unit** is a config and seed. Its warm-up is scripted and deterministic, so
every run of a unit starts from the same city, and the model's temperature makes the runs differ after it.
`repeats = K` on a `[[runs]]` entry, `[eval] repeats` or `--repeats K` plays each unit K times (run ids end in
`_r<i>` when K > 1; `--dry-run` scales runs, tokens and time). The scorecard averages a unit's repeats and computes
everything over the units (`civitas/bench/stats.py`, no dependencies): the mean vs scripted with a **95% t-interval**
(an interval above 0: the model beats scripted beyond seed luck), the **seed spread** (std of the units), the
**replay noise** (pooled std within units, K >= 2) and the **detectable difference**: the smallest gap between two
evals on these units that a paired comparison finds 80% of the time, for 1, 2, 3 and 5 repeats, from the replay
noise alone (a lower bound: two models also differ seed by seed). Departments, population and Teamwork get intervals
too. `--adopt` takes several runs of one unit as its repeats.

## The scorecard and cost

**The scorecard.** Per run: the city score at hand-over, its change by the end, the yardstick's change from the same
hand-over and their difference, **vs scripted** (above 0: the model added more than the scripted baseline would
have), split into the **population term** (half the city score is 50 x sqrt(population / target): the run's gain in
it minus the yardstick's from the same hand-over, `evals.card_split` with `scoring.population_term`) and the
**quality terms** (the rest: happiness, finances, services, environment, jobs, safety, health, education, and the
penalties), so a model that only grew the city faster shows as population; the same for population; Teamwork
(`fleet/report.py:run_collaboration`); failed actions and refusals per turn
(`fleet_report.json`); model turns and tokens (prompt + completion, from `trajectories.jsonl`); wall time. A region
run has a line for the region and one per city. Then, over the runs (a region counts once, by its region score):
city score vs scripted as mean ± std with its population / quality split, "beats scripted on N of M", population
vs scripted, each department's final
score minus the yardstick's, Teamwork, failures per turn, tokens per model turn and in all, run time; for all runs
and per config. A Versions table gives each run's `fleet_version`, a hash of its logged system prompts and the code
commit (adopted runs: the last commit before the run started, inferred from git).

**Cost.** The suite keeps reference numbers per run (trinity-large-thinking): a single city ~1.4M prompt + 0.15M
completion tokens and ~9.5 min, a region ~3.2M + 0.35M and ~23.5 min; the whole suite ~11.6M tokens (10.5M prompt,
1.1M completion) and ~24 min with 5 workers (75 min of runs). `--dry-run` scales them by the model turns and, with
`[eval.prices]` (USD per million prompt / completion tokens by model) in the suite, prints a price.

## Comparing models

**Comparing models.** `--compare A B ...` prints the evals' aggregates side by side, then each eval against the
first, **paired by unit**: per shared config + seed the difference of the two evals' scores vs scripted (the hand-over's
difficulty cancels), a t-interval over those differences and a verdict, `B is BETTER: +1.2 (95% CI +0.3 to +2.1)` or
`no clear difference: ...; differences under ~1.3 can't be seen on this suite at these repeats`, with a line per
department, population and Teamwork. It warns when a shared unit's hand-over differs (the warm-up changed: not
comparable), names units only one eval has, and notes different code commits or system prompts. An eval saved
before units gets them recomputed from its stored run cards. In the viewer, ⋯ → Model evals (or the
Compare view's Evals switch) shows one column per eval with the same rows, best value per row in green; "Load eval…"
in the Compare view adds an eval's runs, and when they don't fit beside the picked runs (8 at most) it offers to
compare the evals by their aggregate or to replace the picked runs. The Evals view shows intervals as
`+1.4 [−0.5, +3.3]`, units × repeats, the replay noise, the detectable difference and, with two evals or more, a
"vs <first eval>" row: the paired city verdict, green better, red worse, grey unclear (`compare.js:pairedCity`).
Backend: `GET /api/evals` (each eval's `units`).

## First results and calibration (civitas-v1)

The first eval, `trinity-large-v1` (trinity-large-thinking, adopted from the late smokes of 2026-10-06/07): city
score vs scripted **+1.4 ± 1.5** (95% CI −0.5 to +3.3: not yet clear of 0 on 5 units), beats scripted on 4 of 5 (single cities +1.3 ± 1.1, 3 of 3; regions +1.6 ± 2.6,
1 of 2), population +2,166 vs scripted (CI +145 to +4,188), mayor +3.6 (+1.6 to +5.6), planning +1.6 (+0.2 to +2.9),
police -3.7, fire -2.4, education -4.5 (−7.9 to −1.0),
Teamwork 84.4, 0.24 failed actions and 0.23 refusals per turn, 52k tokens per model turn, 11.6M tokens in all. Its
runs came from four commits of the same day (the fleet configs' `fleet_version` is the same in all).

**Calibration** (`trinity-large-cal`, 2026-10-09: `eval_late` with `--repeats 3`, 15 runs, 36.5M tokens, ~70 min):
the **replay noise is 0.4 points** (std of one model on one seed), against a **seed spread of 2.5**. Repeats of a
seed agree closely (s0: +4.4 / +4.2 / +3.4; s1: −0.5 / −0.8 / −1.5), while seeds differ a lot (s0 about +4, s1 about
−0.9). City score vs scripted +1.6 (95% CI −1.6 to +4.8): beating scripted is still not clear of seed luck on 5
units. Departments that are: mayor +3.6 (+1.9 to +5.2), utilities −0.9 (−1.7 to −0.2), education −4.8 (−9.6 to
−0.0); police −4.6 (−5.5 to −3.8) in the single cities. Paired against `trinity-large-v1` (same model, older code):
no clear difference, +0.2 (−1.4 to +1.7). The paired differences vary more than replay noise explains (std about 1.2
against about 0.5), so code or model changes move seeds differently: the detectable-difference line (~1.0 at 1 repeat
here) is a lower bound, and **more seeds help more than repeats**. Hence `configs/eval_singles.toml`, the checkpoint
eval: `fleet_smoke_late` seeds 0-7, one run each, ~12M tokens (as `eval_late`), ~12 min with 8 workers; seeds 0-2
pair with `eval_late` evals. Region runs cost 2.5x a single city per unit and two region seeds give intervals too wide
to use, so `eval_late` is for milestones. Seeds 0-9 all reach a working city at the hand-over (scripted fleet at
month 96: 66.7 ± 2.4, about 6,100 residents, none bankrupt).

## civitas-v2: new seeds, guided and open tracks

**civitas-v2's suite** (`configs/eval_v2.toml`; `eval_singles.toml` stays civitas-v1's frozen suite): the same late
starts on seeds 0, 4, 5, 11, 12, 16, 18 and 20. v1's seeds 1 and 6 handed over in a recession (industrial jobs from
~1,300 to 0 in months 84-92, unemployment ~45%), which the scripted yardstick recovers from by itself, so "vs
scripted" there measured the recovery. With v2's scripted mayor the warm-up of seeds 0-23 was scanned for shocks in
months 84-96: a recession under way or ended within 6 months, a flood, industrial jobs falling over 30%,
unemployment over 10%, population falling over 5%. 15 seeds had one (14 a recession; seed 2 a flood at month 85
that cost 45% of its population; seeds 13 and 15 a recession starting at months 91 and 94). Of the 9 clean ones,
seed 3 (the largest hand-over city) is left out: hand-over population 5,897 on average (all 24 seeds 5,756, seeds
0-7 5,824), city score 66.1 ± 2.2. The scan was repeated after the scripted mayor's take-back hysteresis changed
every hand-over city (before it: 6,246, 5,890, 6,302, 67.2 ± 2.3) and found the same 9 clean seeds. None of the
eight has a train station or harbour at the hand-over; seeds 0, 5, 18 and 20 are past 6,000 residents, so their R
and C demand is capped at +0.25 (seed 20's R also for parks).

**Guided and open tracks.** v1's evals measured briefing-following: 88-92% of department actions were an exact
briefing call or a briefing site, the mayor funded the to-fund line to the dollar ~92% of the time, and four models
scored within 0.8 points. `[fleet] briefing` picks the briefing style (per role like every fleet key):

- `"guided"` (the default; `eval_v2.toml`, env `civitas-v2`): briefings and tool results name the build, its exact
  call and the money to ask for (Next lines, "This turn: ...", the mayor's needs and to-fund lines, fixes after a
  failure).
- `"open"` (`configs/eval_open.toml` on `configs/fleet_open_late.toml`, env `civitas-v2-open`; same seeds, game,
  tools, guards and yardstick): the same facts, never a chosen action. Kept: the city, key metrics (coverage,
  enrolment, supply and demand, transit share, downtown cover, demand caps), funds and plan target, every Ready sites
  line with its price, estimates as data ("coverage estimate: ... ~97% with it", "transit estimate"), upgrade offers,
  requests, the map, Finances with the reserve and its rule, and per department its funds, spending, what a take-back
  must leave it and its priced options, each tagged as the reserve guard counts it ("the city has none" for a first
  building, "supply" for a short utility). Dropped: exact call text, Next lines (and their refresh in tool results),
  "This turn", the to-spend / to-fund lines and request states, "ask the mayor for $X" / "no money ask" lines, the
  repay_loan suggestion, fixes in failed-action results (the reason stays: "(24,23) holds a town hall", "costs $7,800
  but the police budget only has $6,632"), the "do X instead" part of guard refusals, and what the money buys in the
  mayor's money results (a top-up's result gives the treasury against the reserve; the take-back, reserve and loan
  guards state their rule, never a named build or an amount "needed"). No money ask to the mayor is held back: the
  mayor decides. The playbooks are rewritten without them. The scripted fleet renders in the configured style too
  (so the environment probe hashes open text) and plays the same in either. `civitas prompt -c configs/fleet_open_late.toml --role mayor` shows an open briefing.

## civitas-v3: an equal token budget per turn

**civitas-v3: an equal token budget per turn** (`configs/eval_v3.toml` on `fleet_smoke_late_v3.toml`, env
`civitas-v3`; `configs/eval_v3_open.toml` on `fleet_open_late_v3.toml`, env `civitas-v3-open`): civitas-v2 plus
`[fleet] turn_budget = 100000` tokens a turn, summed over the turn's model calls (`civitas/fleet/budget.py`). Same
seeds, game, tools, guards, briefings and yardstick; the scripted warm-up has no budget. *Why:* on v2 every model had
the same caps (20 tool calls, 80k completion tokens a turn, 16k a call), but token use differed 3x for the same
~2.1-2.5 actions a turn. Every reply re-sends the whole conversation (~6-7.6k tokens of context for every model), so
prompt tokens grow with the number of replies: on the open track trinity-large-thinking made 11.6 model calls a
department turn (one tool call per reply, 8.4 look-ups, the 20-call cap in 23% of turns), 92k tokens a turn; kimi-k3
4.3 calls, 1.6 look-ups, 34k; glm-5.3-flash 51k, deepseek-v4-flash 54k. Efficiency didn't enter the score.

*Units (tokenizer-neutral):* a call's prompt counts its request's characters / 4 (the tool schemas and every message
as sent), never the provider's prompt tokens, so the same conversation costs the same for every model; its
completion counts the provider's completion tokens (what the model chose to write and think), hardened to the largest
of `completion_tokens`, `completion_tokens_details.reasoning_tokens` and `total_tokens - prompt_tokens`, and the
reply's characters / 4 when the provider reports nothing. Trajectories log the provider's usage apart (`usage`, and
per call `call_usage`: the cost) and the budget's units (`budget_used`; per call under `call_usage[i].budget`).

*How it works:* the system prompt states the rule, the briefing header has a "Token budget: 100,000 tokens this turn"
line, and every tool result ends with "budget left: ~N tokens (~K more replies at the current size)" (K = what's left
over the next reply's cost; the same text on both tracks and for every model). The first call always runs; before
each further call its cost is the request's units plus max(`budget_reserve` = 4,000, the largest completion this
turn), and when that would pass the budget the call isn't made: the turn ends (`finish_reason` "turn_budget",
completed actions stay; the report is the one a refused end_turn carried, else "(no report: the turn's token budget
ran out before end_turn ...)"). The largest completion rather than the mean: the cutoff should admit a reply only when
it fits, and the longest reply a model has written this turn bounds its next one best; with a mean (or a fixed 4k) a
model writing 12k-token replies would overshoot by up to 8k on its last reply, so verbosity bought budget. Now a
verbose model is cut a reply earlier and a terse one keeps the 4k floor; what is left over is logged as `budget_over`
(units past the budget: a reply longer than any before it, bounded by `max_tokens`). Once the next reply is about the
last that fits, the result says "room for about one more reply: call end_turn with your report now", so a model that
heeds it always ends with end_turn; then the mayor's and planning's once-a-turn end_turn reminder of open requests
waits (`ToolBox.budget_low`, as it does with under 2 tool calls left), so heeding the note never costs the report.
max_steps 20 stays a hard cap. Trajectories log `turn_budget`, `budget_used`, `budget_cut` and `budget_over`;
fleet_report.md shows budget cuts apart from step-limit ends (and a Token budget section: cuts, units per turn,
overshoot); eval run cards, aggregates and leaderboard rows carry `budget_cut_rate` and `budget_used_mean` (the
scorecard's summary rows, the Leaderboard's Efficiency column "cut by budget"). Budget 0 (the default) is v2 exactly:
the civitas-v2 and civitas-v2-open fingerprints are unchanged.

*How 100k was chosen:* every v2 department turn (eval_*-v2 / eval_*-open of 2026-10-10, 256 turns per model and track)
replayed in these units: each call's prompt from the logged conversation (meta.json's system prompt plus v3's budget
rule and playbook line, the tool schemas, every message, the briefing line and a "budget left" line per tool result) /
4, and the turn's provider completion tokens split over its calls by each reply's characters (reasoning included).
Units per turn, the share over 100k, and the share the cutoff would cut:

| model | track | p50 | p90 | p95 | >100k | cut at 90k | cut at 100k | cut at 100k (even split) |
|---|---|---|---|---|---|---|---|---|
| kimi-k3 | guided | 19.7k | 46.0k | 53.2k | 0.4% | 0.8% | 0.4% | 0.4% |
| deepseek-v4-flash | guided | 19.0k | 39.6k | 49.6k | 0.0% | 1.2% | 0.8% | 0.0% |
| glm-5.3-flash | guided | 25.8k | 63.0k | 73.1k | 1.2% | 3.9% | 2.3% | 1.2% |
| trinity-large-thinking | guided | 32.9k | 105.0k | 122.9k | 11.7% | 15.6% | 12.9% | 12.9% |
| kimi-k3 | open | 26.3k | 60.2k | 84.8k | 3.1% | 7.0% | 4.7% | 3.5% |
| glm-5.3-flash | open | 39.3k | 93.8k | 111.5k | 8.2% | 18.4% | 12.5% | 8.2% |
| deepseek-v4-flash | open | 36.0k | 95.7k | 112.0k | 7.4% | 16.4% | 11.7% | 9.4% |
| trinity-large-thinking | open | 77.4k | 172.8k | 186.2k | 36.3% | 45.7% | 39.5% | 39.1% |

100k is the cleanest budget the most efficient model (kimi-k3) is cut at in under ~5% of its open-track turns (4.7%,
11 of its 12 cuts planning's; 95k: 5.5%, 90k: 7.0%: the largest-completion reserve cuts earlier than the earlier
calibration's fixed 4k, which put the line at 90k in provider tokens). It still binds the wasteful pattern: 40% of
trinity-large-thinking's open turns would be cut (losing 5.4 tool calls each on average; 18% of its turns would lose a
successful action), 13% of its guided ones, 12-13% of deepseek's and glm's open turns. Unchanged models would use 76%
(trinity, open) to 99.5% (kimi, guided) of their v2 units; told the budget, they can do better. RL tasks match it with
`TaskEpisode(task, turn_budget=100000)` and each reply's `usage`; `replay(messages, usage=trajectory["call_usage"])`
reproduces a logged v3 turn's cutoff exactly (rl/episode.py).
