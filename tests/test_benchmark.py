"""Benchmark guard rails (slow: runs five full episodes). Skip with: pytest -m "not slow"."""
import time

import pytest

from civitas.bench.runner import run_episode
from civitas.config import load_config


@pytest.mark.slow
def test_scripted_baseline_is_solvent_in_range_and_fast(tmp_path):
    cfg = load_config("configs/baseline.toml")
    scores = []
    for seed in range(5):
        t = time.time()
        s = run_episode(cfg, seed, tmp_path, verbose=False, log_observations=False)
        assert time.time() - t < 8.0, "a full scripted episode should take a few seconds"
        assert s["months_played"] == 120 and not s["bankrupt"]
        scores.append(s["final_scores"]["city"])
    mean = sum(scores) / len(scores)
    assert 60.0 <= mean <= 80.0, scores
