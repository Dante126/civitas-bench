"""Small-sample statistics for model evals (bench/evals.py): t-intervals over units, replay noise, the smallest
difference a paired comparison can detect, and a paired verdict. No dependencies beyond the standard library.

A unit is one config + seed of a suite; its value is the mean over its repeats. Intervals are t-intervals over the
unit values (with 3-10 units a bootstrap gives intervals that are too narrow). `viewer/compare.js` copies T975 for
the Evals view's paired row, so keep the two in step.
"""
from __future__ import annotations

import math
import statistics

# Student's t quantiles by degrees of freedom: two-sided 95% (0.975) and one-sided 80% power (0.80). A df between two
# keys takes the smaller key's value (slightly wider: conservative); above the last key the normal quantile.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
T80 = {1: 1.376, 2: 1.061, 3: 0.978, 4: 0.941, 5: 0.920, 6: 0.906, 7: 0.896, 8: 0.889, 9: 0.883, 10: 0.879,
       12: 0.873, 15: 0.866, 20: 0.860, 25: 0.856, 30: 0.854, 40: 0.851, 60: 0.848, 120: 0.845}
Z = {0.975: 1.960, 0.80: 0.842}
MDE_REPEATS = (1, 2, 3, 5)          # the repeats the detectable-difference line is given for


def t_quantile(p: float, df: int) -> float:
    """The t quantile for p in (0.975, 0.80) at `df` >= 1 degrees of freedom."""
    table = {0.975: T975, 0.80: T80}[p]
    if df < 1:
        raise ValueError(f"t quantile needs df >= 1, got {df}")
    if df > max(table):
        return Z[p]
    return table[max(k for k in table if k <= df)]


def mean_ci(values) -> dict | None:
    """{n, mean, std, lo, hi} of the values (None dropped): a 95% t-interval for the mean; std, lo and hi are None
    with fewer than 2 values. None with no values."""
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return {"n": 1, "mean": m, "std": None, "lo": None, "hi": None}
    sd = statistics.stdev(xs)
    half = t_quantile(0.975, len(xs) - 1) * sd / math.sqrt(len(xs))
    return {"n": len(xs), "mean": m, "std": sd, "lo": m - half, "hi": m + half}


def pooled_sd(groups) -> dict | None:
    """The pooled within-group standard deviation of several groups of repeats ({sd, df}): how much one unit varies
    between replays. Groups of one add nothing; None when no group has two values."""
    ss, df = 0.0, 0
    for g in groups:
        xs = [float(v) for v in g if v is not None]
        if len(xs) >= 2:
            m = statistics.fmean(xs)
            ss += sum((x - m) ** 2 for x in xs)
            df += len(xs) - 1
    return {"sd": math.sqrt(ss / df), "df": df} if df else None


def mde(noise: float, n_units: int, repeats: int) -> float | None:
    """The smallest true difference a paired comparison of two evals on `n_units` units, each played `repeats` times
    by both, finds 80% of the time at the 5% level, from the replay noise alone (a lower bound: two models also differ
    seed by seed). None under 2 units."""
    if n_units < 2 or repeats < 1:
        return None
    df = n_units - 1
    return (t_quantile(0.975, df) + t_quantile(0.80, df)) * math.sqrt(2 * noise ** 2 / repeats / n_units)


def ratio_ci(nums, dens) -> dict | None:
    """The ratio of means sum(nums) / sum(dens) over paired values (a pair with a None dropped) with a 95% interval
    from the ratio estimator's linearization: residuals e = num - R x den, se = sd(e) / (sqrt(n) x mean(den)), a
    t-interval on n - 1 df. A unit with a small denominator weighs little here, unlike in a mean of per-unit ratios.
    {n, mean, lo, hi} (lo / hi None under 2 pairs); None without pairs or with a zero sum of denominators."""
    pairs = [(float(a), float(b)) for a, b in zip(nums, dens) if a is not None and b is not None]
    if not pairs or sum(b for _, b in pairs) == 0:
        return None
    n = len(pairs)
    r = sum(a for a, _ in pairs) / sum(b for _, b in pairs)
    if n < 2:
        return {"n": 1, "mean": r, "lo": None, "hi": None}
    se = statistics.stdev([a - r * b for a, b in pairs]) / (math.sqrt(n) * abs(statistics.fmean(b for _, b in pairs)))
    half = t_quantile(0.975, n - 1) * se
    return {"n": n, "mean": r, "lo": r - half, "hi": r + half}


def verdict(diffs) -> dict | None:
    """A paired comparison from per-unit differences (B - A): mean_ci plus `call`: "better" when the interval is
    above 0, "worse" below 0, "unclear" when it holds 0 or there are fewer than 2 units."""
    ci = mean_ci(diffs)
    if ci is None:
        return None
    call = "unclear" if ci["lo"] is None or ci["lo"] <= 0 <= ci["hi"] else ("better" if ci["lo"] > 0 else "worse")
    return {**ci, "call": call}
