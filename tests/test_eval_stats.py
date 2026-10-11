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


# ------------------------------------------------------------------ population / quality split (civitas-v2)
def _split_card(seed, delta, pop_change, yard_pop_change, hand_pop=5000, target=None):
    c = _card("late", seed, delta)
    c["total"]["change"]["population"] = pop_change
    c["total"]["yard_change"]["population"] = yard_pop_change
    c["total"]["at_handover"]["population"] = hand_pop
    if target is not None:
        c["target_population"] = target
    return c


def test_vs_scripted_splits_into_population_and_quality():
    """The city score's population half is 50 x sqrt(population / target): the split of a run's city score vs
    scripted takes the run's gain in it less the yardstick's from the same hand-over; quality is the rest. Read from
    fields every card has, so an eval saved before the split gets one (target 30,000 unless the card says)."""
    term = lambda p, t=30000: 50 * (p / t) ** 0.5
    pop = (term(6000) - term(5000)) - (term(5200) - term(5000))
    c = _split_card(0, 2.0, 1000, 200)
    assert E.card_split(c) == {"population": pytest.approx(pop, abs=1e-3), "quality": pytest.approx(2.0 - pop, abs=1e-3)}
    assert E.card_split(_split_card(0, 2.0, 1000, 200, target=10000))["population"] == pytest.approx(
        term(6000, 1e4) - term(5200, 1e4), abs=1e-3)
    no_pop = _card("late", 0, 1.0)
    no_pop["total"]["at_handover"]["population"] = None
    assert E.card_split(no_pop) is None
    region = {"region": True, "cities": ["north", "south"], "total": {},
              "by_city": {"north": _split_card(0, 2.0, 1000, 200)["total"], "south": _split_card(0, -1.0, 0, 0)["total"]}}
    r = E.card_split(region)                                         # the region score is its cities' mean
    assert r["population"] == pytest.approx(pop / 2, abs=1e-3)
    assert r["quality"] == pytest.approx((2.0 - pop - 1.0) / 2, abs=1e-3)
    cards = [_split_card(s, d, pc, 0) for s, (d, pc) in enumerate([(2.0, 1000), (0.5, 0), (-1.0, -500)])]
    a = E._agg(cards, {})
    pt = [term(6000) - term(5000), 0.0, term(4500) - term(5000)]
    assert a["pop_term_delta"]["mean"] == pytest.approx(sum(pt) / 3, abs=1e-3)
    assert a["quality_delta"]["mean"] == pytest.approx((1.5 - sum(pt)) / 3, abs=1e-3)
    assert [u["pop_term"] for u in a["units"].values()] == pytest.approx(pt, abs=1e-3)
    rows = dict(E._summary_rows(a))
    assert rows["· from the population term"].startswith(f"{sum(pt) / 3:+.1f} (")
    md = E.markdown({**_ev("m", cards), "suite": {"name": "late", "file": "x.toml"}})
    assert "| · population term | · quality terms |" in md
    assert f"| **+2.0** | n/a | n/a | {pt[0]:+.1f} | {2.0 - pt[0]:+.1f} |" in md       # no noop reference: n/a


# ------------------------------------------------------------------ the noop reference and the normalized score
def _noop_card(seed, change, yard, noop, repeat=1, repeats=1):
    c = _card("late", seed, change - yard, repeat=repeat, repeats=repeats)
    c["total"].update(change={"city": change, "population": 0}, yard_change={"city": yard, "population": 0})
    E._add_noop(c["total"], None if noop is None else {"change": {"city": noop, "population": 10}})
    return c


def test_ratio_of_means():
    r = S.ratio_ci([3.0, 1.0, 2.0], [1.0, 1.0, 2.0])               # R = 6 / 4; residuals 1.5, -0.5, -1
    sd = S.mean_ci([1.5, -0.5, -1.0])["std"]
    assert r["n"] == 3 and r["mean"] == 1.5
    assert r["hi"] - r["mean"] == pytest.approx(4.303 * sd / (3 ** 0.5 * 4 / 3))
    assert S.ratio_ci([2.0, None], [1.0, 5.0]) == {"n": 1, "mean": 2.0, "lo": None, "hi": None}
    assert S.ratio_ci([1.0], [0.0]) is None and S.ratio_ci([], []) is None


def test_normalized_score_and_its_guard():
    assert E.normalized(3.0, 2.0, 1.0) == (2.0, False)            # twice what scripted adds over doing nothing
    assert E.normalized(1.0, 2.0, 1.0) == (0.0, False) and E.normalized(0.0, 2.0, 1.0) == (-1.0, False)
    assert E.normalized(3.0, 1.4, 1.0) == (None, True)             # scripted within 0.5 of noop: no ratio
    assert E.normalized(3.0, 0.5, 1.0) == (pytest.approx(-4.0), False)   # noop beats scripted by 0.5: a ratio
    assert E.normalized(3.0, 1.4, 1.0, min_gap=0.25) == (pytest.approx(5.0), False)
    assert E.normalized(None, 2.0, 1.0) == (None, False) and E.normalized(3.0, 2.0, None) == (None, False)
    u = _noop_card(0, 3.0, 2.0, 1.0)["total"]
    assert u["noop_change"] == {"city": 1.0, "population": 10} and u["normalized"] == 2.0 and not u["norm_tiny"]
    old = _card("late", 0, 1.0)["total"]                           # a card from before the noop reference
    assert "noop_change" not in old and E._add_noop(dict(old), None)["normalized"] is None


def test_units_and_the_aggregate_normalized():
    """A unit's ratio is of its means (repeats averaged first); tiny units are counted and left out of the per-unit
    summary; the pooled ratio of means and the points over noop take every unit with a noop reference."""
    cards = [_noop_card(0, 3.0, 2.0, 1.0, 1, 2), _noop_card(0, 5.0, 2.0, 1.0, 2, 2),   # mean change 4: ratio 3
             _noop_card(1, 2.0, 2.0, 0.0),                                             # ratio 1
             _noop_card(2, 4.0, 1.2, 1.0),                                             # tiny gap 0.2
             _noop_card(3, 1.5, 1.0, 0.0)]                                             # ratio 1.5
    uv = E.unit_values(cards)
    assert uv["late s0"]["normalized"] == 3.0 and uv["late s0"]["noop_change"] == 1.0 and uv["late s0"]["change"] == 4.0
    assert uv["late s2"]["normalized"] is None and uv["late s2"]["norm_tiny"]
    a = E._agg(cards, {})
    n = a["normalized"]
    assert n["n"] == 3 and n["tiny"] == 1 and n["missing"] == 0 and n["mean"] == pytest.approx(5.5 / 3, abs=1e-3)
    assert n["median"] == 1.5 and n["lo"] is not None
    nums, dens = [3.0, 2.0, 3.0, 1.5], [1.0, 2.0, 0.2, 1.0]
    assert n["pooled"]["mean"] == pytest.approx(sum(nums) / sum(dens), abs=1e-3) and n["pooled"]["n"] == 4
    assert n["vs_noop"]["mean"] == pytest.approx(sum(nums) / 4, abs=1e-3)
    assert a["noop_change"] == pytest.approx(0.6) and a["yard_change"] is not None
    rows = dict(E._summary_rows(a))
    assert rows["doing-nothing change from the same start (mean)"] == "+0.6"
    assert rows["· units without a ratio (scripted within 0.5 of noop)"] == "1 of 4"
    assert rows["normalized (run - noop) / (scripted - noop), pooled (95% CI)"].startswith(f"{9.5 / 4.2:+.2f} (")
    md = E.markdown({**_ev("m", cards), "suite": {"name": "late", "file": "x.toml"}})
    assert "| vs scripted | noop | normalized |" in md and "| +1.0 | tiny |" in md and "| +0.0 | +1.00 |" in md
    row = E._noop_row(a)
    assert row["noop_change"] == a["noop_change"] and row["normalized"]["tiny"] == 1
    assert row["normalized"]["pooled"]["mean"] == n["pooled"]["mean"] and row["vs_noop"]["mean"] == n["vs_noop"]["mean"]


def test_evals_without_the_noop_reference():
    """Cards and aggregates from before the noop reference: None / n/a everywhere, nothing else changes."""
    cards = [_card("late", s, d) for s, d in enumerate([1.0, 0.0, 2.0])]
    a = E._agg(cards, {})
    assert a["noop_change"] is None and a["normalized"]["n"] == 0 and a["normalized"]["pooled"] is None
    assert a["normalized"]["missing"] == 3 and a["city_delta"]["mean"] == 1.0
    rows = dict(E._summary_rows(a))
    assert rows["doing-nothing change from the same start (mean)"] == "n/a"
    assert rows["normalized (run - noop) / (scripted - noop), pooled (95% CI)"] == "n/a"
    saved = {k: v for k, v in a.items() if k not in ("noop_change", "normalized")}   # an aggregate saved before it
    rows = dict(E._summary_rows(saved))
    assert rows["points over doing nothing (mean, 95% CI)"] == "n/a"
    assert rows["· units without a ratio (scripted within 0.5 of noop)"] == "n/a"
    assert E._noop_row(saved) == {"noop_change": None, "scripted_change": saved["yard_change"], "vs_noop": None,
                                  "normalized": None}
    lb = {"env": "e", "rows": [{"model": "m", "eval_dir": "d", "rank": 1, "n_units": 3, "departments": {},
                                "vs_yardstick": {"mean": 1.0, "lo": 0.0, "hi": 2.0}, "score": {"mean": 60.0},
                                **E._noop_row(saved)}], "skipped": []}
    assert "Against doing nothing" not in E.leaderboard_markdown(lb)
    lb["rows"][0].update(E._noop_row(E._agg([_noop_card(s, 3.0, 2.0, 1.0) for s in range(3)], {})))
    md = E.leaderboard_markdown(lb)
    assert "## Against doing nothing" in md and "| m | +3.0 | +2.0 | +1.0 | +2.0 (+2.0 to +2.0) | +2.00 (" in md
