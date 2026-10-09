# Eval statistics: repeats, confidence intervals, paired comparison (2026-10-09)

## Why

`civitas eval` is used internally to compare model checkpoints. The late suite has 5 units (config + seed), one run
each, and reports "vs scripted" as mean ± std (+1.4 ± 1.5 for trinity-large-v1). Nobody knows how much the same model
varies on the same seed, so nobody knows whether a 1-point difference between two checkpoints is real. This change
measures that noise and turns comparisons into verdicts with confidence intervals. Broader suites (a quick tier, a
bank of short tasks) wait for the noise numbers.

Facts that shape it: the warm-up is scripted and deterministic, so one config + seed always reaches the same hand-over
and the same yardstick; the fleet plays at temperature 0.4, so repeats differ, and so does the game after the
hand-over.

## 1. Repeats and the suite format

- `[[runs]] repeats = K` per entry (default 1), `[eval] repeats` for the suite, `civitas eval --repeats K` over both.
- Repeat ids: `<stamp>_<name>_s<seed>_r<i>_<label>` when K > 1; K = 1 keeps today's ids, so existing evals are valid.
  Each repeat is a normal run folder with its own yardstick.json (scripted, seconds).
- `--dry-run` multiplies runs, tokens and time by K. `--resume` treats a repeat like any run. `--adopt` reads the
  repeat from an `_r<i>` in the run id, else 1.
- A quick tier later is another suite file; nothing else in the format.

## 2. Statistics in one eval (`civitas/bench/stats.py`, no new dependencies)

- A **unit** is a config + seed. Its value is the mean over its repeats. A region counts once, by its region values.
  All statistics run over unit values, so repeats never count as extra seeds.
- **Headline:** mean with a 95% t-interval over the unit values (t critical values from a small table; at n = 5 a
  bootstrap is too narrow). "Beats scripted" is real when the interval is above 0. Population, Teamwork and each
  department's delta get the same mean + CI.
- **Replay noise** (K >= 2): pooled within-unit standard deviation. **Seed spread:** standard deviation of the unit
  values.
- **Detectable difference** for a paired comparison of two evals on this suite:
  `(t(0.975, n-1) + t(0.80, n-1)) * sqrt(2 * noise^2 / K / n)` for K in 1, 2, 3, 5. A lower bound: two different
  models also disagree seed by seed, which one model can't measure. Without repeats: "run with --repeats 2+".
- `_agg` keeps its existing keys (`city_delta.mean` / `std` / `values` ...) and adds `ci`, `units`, `noise`, `mde`, so
  the viewer and older eval.json files keep working.

## 3. Comparing evals

- `civitas eval --compare A B [C ...]`: the first eval is the baseline. Units are paired by config name + seed; units
  only one side ran are listed and left out.
- Per unit: B's unit value minus A's. The verdict is a t-interval over those differences: "B is better: +1.2 (95% CI
  +0.3 to +2.1) on 5 shared units", or "no clear difference: -0.4 (-1.9 to +1.1); differences under ~1.3 can't be
  seen on this suite". A verdict line per department follows.
- Warnings: a shared unit whose hand-over city score differs (the warm-up changed: not comparable); different code
  commits or prompt hashes (fine for models, worth knowing).
- Viewer (⋯ → Model evals): values with their CI, a "vs <first eval>" row with the verdict colour (green better, red
  worse, grey unclear). `/api/evals` passes the new fields; old eval.json files render as before.

## Testing

No paid runs. `stats.py` against textbook values; the repeat plan (ids, dry-run totals, `--adopt` `_r<i>`); the
scorecard and `--compare` on synthetic run folders as `tests/test_eval.py` builds them; the trinity-large-v1 eval
still renders (K = 1: no noise lines). Full suite once at the end.

## Calibration (after merge, only on the user's go)

`civitas eval -s configs/eval_late.toml --model trinity-large-thinking --repeats 3 --name trinity-large-cal`: 15
runs, ~35M tokens, ~75 min with 5 workers. Its replay noise fills the detectable-difference table and decides
whether a quick tier is worth building.
