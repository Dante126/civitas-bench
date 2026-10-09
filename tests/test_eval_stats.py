"""Eval statistics (bench/stats.py, bench/evals.py): t-intervals, replay noise, the detectable difference, repeats
in a plan, units in a scorecard and the paired comparison of two evals, on synthetic run cards. No run is played."""
from pathlib import Path

import pytest

from civitas.bench import evals as E
from civitas.bench import stats as S

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs/eval_late.toml"


# ------------------------------------------------------------------ stats
def test_t_quantiles_and_intervals():
    assert S.t_quantile(0.975, 4) == 2.776 and S.t_quantile(0.80, 4) == 0.941
    assert S.t_quantile(0.975, 11) == 2.228 and S.t_quantile(0.975, 500) == 1.96       # 11 takes 10's: wider
    with pytest.raises(ValueError):
        S.t_quantile(0.975, 0)
    c = S.mean_ci([1, 2, 3])                     # sd 1, half-width 4.303 / sqrt(3)
    assert c["n"] == 3 and c["mean"] == 2 and c["std"] == 1
    assert c["lo"] == pytest.approx(2 - 2.4843, abs=1e-3) and c["hi"] == pytest.approx(2 + 2.4843, abs=1e-3)
    assert S.mean_ci([5, None]) == {"n": 1, "mean": 5, "std": None, "lo": None, "hi": None}
    assert S.mean_ci([None]) is None


def test_noise_detectable_difference_and_verdict():
    p = S.pooled_sd([[1, 3], [10, 10, 13], [7]])            # ss 2 + 6, df 1 + 2; a group of one adds nothing
    assert p["df"] == 3 and p["sd"] == pytest.approx((8 / 3) ** 0.5)
    assert S.pooled_sd([[1], [2]]) is None
    m1, m3 = S.mde(1.0, 5, 1), S.mde(1.0, 5, 3)             # (2.776 + 0.941) * sqrt(2 / 5)
    assert m1 == pytest.approx(3.717 * (2 / 5) ** 0.5, abs=1e-3) and m3 == pytest.approx(m1 / 3 ** 0.5)
    assert S.mde(1.0, 1, 1) is None
    assert S.verdict([1.0, 1.2, 0.9])["call"] == "better" and S.verdict([-1, -1.1, -0.8])["call"] == "worse"
    assert S.verdict([1, -1, 0.5])["call"] == "unclear" and S.verdict([3])["call"] == "unclear"
    assert S.verdict([]) is None


# ------------------------------------------------------------------ repeats in a plan
def test_repeats_in_the_plan(tmp_path):
    suite = E.load_suite(SUITE)
    one = E.plan(suite, model="m", runs_dir=tmp_path, stamp="T")
    assert len(one["runs"]) == 5 and one["runs"][0]["run_id"] == "T_fleet_smoke_late_s0_m"     # ids as before
    three = E.plan(suite, model="m", runs_dir=tmp_path, stamp="T", repeats=3)
    ids = [r["run_id"] for r in three["runs"]]
    assert len(ids) == 15 and len(set(ids)) == 15 and ids[:3] == [f"T_fleet_smoke_late_s0_r{i}_m" for i in (1, 2, 3)]
    assert three["repeats"] == [3] and [r["repeat"] for r in three["runs"][:4]] == [1, 2, 3, 1]
    text = E.dry_run_text(three, [])
    assert "15 runs (5 units x 3 repeats)" in text
    tok = lambda ev: sum(r["estimate"]["prompt_tokens"] for r in ev["runs"])
    assert tok(three) == 3 * tok(one)
    p = tmp_path / "s.toml"
    p.write_text(f'[eval]\nrepeats = 2\n[[runs]]\nconfig = "{ROOT}/configs/fleet_smoke_late.toml"\nseeds = [0, 1]\n'
                 f'[[runs]]\nconfig = "{ROOT}/configs/fleet_region_late.toml"\nseeds = [0]\nrepeats = 1\n')
    ev = E.plan(E.load_suite(p), runs_dir=tmp_path, stamp="T")
    assert [(r["seed"], r["repeat"]) for r in ev["runs"]] == [(0, 1), (0, 2), (1, 1), (1, 2), (0, 1)]
    assert "(3 units, 1/2 repeats)" in E.dry_run_text(ev, [])
    for bad in ("0", "true", '"2"'):
        p.write_text(f'[eval]\nrepeats = {bad}\n[[runs]]\nconfig = "{ROOT}/configs/fleet_smoke_late.toml"\nseeds = [0]\n')
        with pytest.raises(E.EvalError, match="repeats"):
            E.load_suite(p)


# ------------------------------------------------------------------ units, intervals and the paired comparison
def _card(name, seed, delta, repeat=1, repeats=1, roles=None, handover=60.0, commit="abc"):
    roles = roles or {"mayor": delta / 2, "fire": -delta}
    return {"run_id": f"{name}_s{seed}_r{repeat}", "name": name, "config": f"configs/{name}.toml", "seed": seed,
            "region": False, "repeat": repeat, "repeats": repeats, "wall_time_s": 60.0, "prompts": "p1",
            "code": {"commit": commit}, "status": "finished",
            "total": {"delta": {"city": delta, "population": 100 * delta}, "change": {"city": delta + 1, "population": 0},
                      "yard_change": {"city": 1.0, "population": 0}, "at_handover": {"city": handover, "population": 5000},
                      "role_delta": roles, "roles": {}, "teamwork": 80.0, "calls": None,
                      "tokens": {"turns": 32, "prompt": 1000, "completion": 100}}}


def _ev(name, cards):
    ev = {"name": name, "model": name, "suite": {"name": "late", "file": "x.toml", "description": ""}, "dir": "x",
          "runs": []}
    ev["scorecard"] = {"runs": cards, "aggregate": E._agg(cards, ev), "by_config": {}}
    return ev


def test_units_average_their_repeats():
    cards = [_card("late", 0, 1.0, 1, 2), _card("late", 0, 3.0, 2, 2), _card("late", 1, 0.0, 1, 2),
             _card("late", 1, 1.0, 2, 2), _card("late", 2, -1.0)]
    a = E._agg(cards, {})
    assert a["n_units"] == 3 and a["repeats"] == [1, 2] and a["scored"] == 5
    assert [u["city"] for u in a["units"].values()] == [2.0, 0.5, -1.0]
    assert a["city_delta"]["mean"] == pytest.approx(0.5) and a["city_delta"]["values"] == [1.0, 3.0, 0.0, 1.0, -1.0]
    assert a["beats_scripted"] == 2                              # units above 0, not runs
    assert a["noise"]["df"] == 2 and a["noise"]["sd"] == pytest.approx(((2 + 0.5) / 2) ** 0.5, abs=1e-3)
    assert set(a["mde"]) == {"1", "2", "3", "5"} and a["mde"]["1"] > a["mde"]["5"]
    assert a["roles_ci"]["mayor"]["mean"] == pytest.approx(0.25) and a["roles"]["mayor"] == pytest.approx(0.25)
    ev = _ev("m", cards)
    rows = dict(E._summary_rows(ev["scorecard"]["aggregate"]))
    assert rows["units (config + seed) x repeats"] == "3 x 1/2" and rows["beats scripted"] == "2 of 3 units"
    assert rows["city score vs scripted (mean, 95% CI)"].startswith("+0.5 (") and "~" in rows[
        "detectable difference vs another eval"]


def test_paired_verdicts():
    base = _ev("a", [_card("late", s, d) for s, d in enumerate([1.0, 0.0, 2.0, -1.0, 0.5])])
    better = _ev("b", [_card("late", s, d + 1.0 + 0.1 * s) for s, d in enumerate([1.0, 0.0, 2.0, -1.0, 0.5])])
    p = E.paired(base, better)
    assert p["city"]["call"] == "better" and p["wins"] == 5 and p["city"]["mean"] == pytest.approx(1.2)
    assert p["roles"]["fire"]["call"] == "worse" and not p["handover_differs"] and not p["code_differs"]
    text = "\n".join(E.verdict_text(p))
    assert "b vs a: b is BETTER: +1.2 (95% CI" in text and "- fire:" in text and "worse" in text
    noisy = _ev("c", [_card("late", s, d + x) for s, (d, x) in
                      enumerate(zip([1.0, 0.0, 2.0, -1.0, 0.5], [2, -2, 1, -1, 0.5]))])
    q = E.paired(base, noisy)
    assert q["city"]["call"] == "unclear" and q["mde"] is None       # one repeat each: no replay noise
    assert "no clear difference" in "\n".join(E.verdict_text(q)) and "--repeats 2" in "\n".join(E.verdict_text(q))
    # repeats on both sides give the detectable difference; a changed warm-up and a missing unit are named
    rep = lambda name, shift, hand=60.0: _ev(name, [_card("late", s, d + shift * s + j * 0.4, j + 1, 2, handover=hand,
                                                          commit=name) for s, d in enumerate([1.0, 0.0, 2.0])
                                                    for j in range(2)])
    r = E.paired(rep("a", 0), rep("b", -0.3, hand=61.0))
    assert r["mde"] is not None and r["noise"] > 0 and r["handover_differs"] == ["late s0", "late s1", "late s2"]
    t = "\n".join(E.verdict_text(r))
    assert "can't be seen on this suite" in t and "WARNING: the hand-over differs" in t and "code commits" in t
    s = E.paired(base, _ev("d", [_card("late", 0, 1.0), _card("other", 0, 1.0)]))
    assert s["units"] == ["late s0"] and s["only_other"] == ["other s0"] and len(s["only_base"]) == 4
    assert "only 1 shared unit, no verdict" in E.verdict_text(s)[0]
    out = E.compare([base, better, noisy])
    assert "b vs a: b is BETTER" in out and "c vs a: no clear difference" in out


def test_an_old_eval_gets_units_and_intervals():
    """An eval.json saved before units: --compare and /api/evals recompute its aggregate from the run cards."""
    cards = [_card("late", s, d) for s, d in enumerate([1.0, 0.0, 2.0])]
    for c in cards:
        del c["repeat"], c["repeats"]
    ev = _ev("old", cards)
    old = {k: v for k, v in ev["scorecard"]["aggregate"].items()
           if k not in ("n_units", "units", "noise", "mde", "roles_ci", "teamwork_ci", "repeats")}
    old["city_delta"] = {"mean": 1.0, "std": 1.0, "values": [1.0, 0.0, 2.0]}
    ev["scorecard"]["aggregate"] = old
    a = E.aggregate_of(ev)
    assert a["n_units"] == 3 and a["city_delta"]["lo"] is not None
    assert "+1.0 (" in dict(E.compare_rows([ev])[1])["city score vs scripted (mean, 95% CI)"][0]
