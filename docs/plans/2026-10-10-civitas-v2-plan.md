# civitas-v2: plan

Why: the civitas-v1 leaderboard (4 models on eval_singles, 2026-10-10) put every model within +1.2..+2.0 of the
scripted yardstick. Seed explained 68% of the variance, model 7%. Department actions were an exact briefing call or a
briefing site 88-92% of the time, and the mayor funded the to-fund line to the dollar ~92% of the time: v1 measures
briefing-following. The analysis also found environment bugs that every model inherits. v2 fixes the bugs (guided
track) and adds an open track whose briefings state facts but no calls, on the same game.

## Track A: guided fleet fixes (fleet layer only: fleet/tools.py, fleet/briefing.py, fleet/planning.py)

1. Education: the needs / This turn line picks the school type by the enrolment gap per level (elementary, high
   school, university), so a high school is named when high-school enrolment trails. A university is not a
   first-building need while it doesn't count in enrolment; it comes after both school levels are covered (graduates
   for offices). v1: "build your first university ($12k)" on 7 of 8 seeds at month 96, "another elementary school"
   after, never a high school at 78-85% high-school enrolment.
2. Police, fire, health: offer tier upgrades (upgrade_facility, module "tier") in the needs line, Ready sites and
   keep_up estimates when the tier is unlocked and downtown tiles are uncovered (tier 1 covers T1_DOWNTOWN there);
   show the downtown uncovered count. v1: 23-26 tier-1 police stations per model, almost no upgrades; the scripted
   fleet upgraded 6 times.
3. Expensive builds: transport's $8-10k station / harbour named on its Next line was never funded while the mayor had
   $17-27k above the reserve. Find why it is missing from the mayor's to-fund line and fix it.

## Track B: scoring, yardstick, suite, reports (bench/scoring.py, agents/scripted.py, bench/evals.py, configs)

1. Mayor deployment term: idle department money averaged over the scoring window, not the final month only.
2. ScriptedMayor reclaims idle department money (v1: the scripted mayor ended with $31k idle, which made the
   models' mayor +4.7 an artifact).
3. Hand-over seeds: choose 8 eval seeds whose months 84-96 have no shock (recession, disaster) in the scripted
   warm-up; v1 seeds 1 and 6 handed over mid-recession.
4. Scorecard and leaderboard: split the city score change into population and quality terms (vs scripted), and show
   failed / refused actions per turn next to Teamwork.
5. Re-measure the reference numbers (baseline seeds 0-4, fleet_scripted) and update README / CLAUDE.md: a benchmark
   change by design.

## Track C: open track (after A merges)

`[fleet] briefing = "open"`: the same game, tools and guards, but briefings, Next lines and tool-result hints state
facts (coverage, enrolment, supply, sites, prices, money) and never an exact call, a "This turn" instruction or a
to-fund prescription. configs/fleet_open_late.toml + configs/eval_open.toml.

## Freeze

civitas-v2 (eval_singles, guided) and civitas-v2-open (eval_open), each with its baseline_model. v1 evals stay on
the civitas-v1 leaderboard. Re-running the reference models costs model tokens: ask first.

Not in v2: a stronger scripted planner (it plays the warm-up, so it would change every hand-over city and every
reference number at once; the scorecard's population / quality split shows its effect instead).
