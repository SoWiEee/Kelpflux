"""Unit tests for eval/scripts/tail_metrics.py (heavy-tail live A/B stats)."""
import numpy as np
import pytest

from eval.scripts.tail_metrics import cvar, paired_delta, summarize


def test_cvar_is_mean_of_worst_tail():
    # worst 25% of 0..99 = 75..99, mean = 87
    vals = list(range(100))
    assert cvar(vals, beta=0.25, tail="high") == pytest.approx(87.0)


def test_cvar_ge_mean_for_high_tail():
    rng = np.random.default_rng(0)
    vals = rng.lognormal(0, 1, 5000)
    assert cvar(vals, beta=0.25) > vals.mean()  # tail average exceeds the mean


def test_cvar_empty_is_nan():
    assert np.isnan(cvar([]))


def test_summarize_panel_keys_and_order():
    jcts = [10, 20, 30, 40, 100]
    s = summarize(jcts, true_runtimes=[5, 10, 10, 20, 10], beta=0.4)
    assert s["n"] == 5
    assert s["mean"] == pytest.approx(40.0)
    assert s["max"] == 100.0
    assert s["p99"] >= s["p95"] >= s["p50"]  # monotone tail
    assert s["cvar"] >= s["mean"]            # worst-tail average >= mean
    assert "slowdown_p99" in s and "slowdown_mean" in s


def test_summarize_without_runtimes_omits_slowdown():
    s = summarize([1, 2, 3])
    assert "slowdown_p99" not in s


def test_paired_delta_positive_when_model_faster():
    score = [100.0] * 50
    model = [80.0] * 50              # model uniformly 20% faster
    d = paired_delta(score, model, n_boot=500)
    assert d["djct_pct"] == pytest.approx(20.0)
    assert d["dp99_pct"] == pytest.approx(20.0)
    assert d["dcvar_pct"] == pytest.approx(20.0)
    assert d["mean_diff"] == pytest.approx(20.0)   # >0 = model lower JCT
    assert d["ttest_p"] < 1e-6                      # clearly significant
    lo, hi = d["djct_pct_ci"]
    assert lo <= 20.0 <= hi


def test_paired_delta_tail_specific_improvement():
    # model caps the tail at p75 → worst-25% (CVaR) collapses to p75 while the
    # body is untouched, so the tail gain (ΔCVaR%) exceeds the mean gain (ΔJCT%).
    rng = np.random.default_rng(1)
    score = rng.uniform(10, 20, 200).tolist()
    p75 = float(np.percentile(score, 75))
    model = np.minimum(score, p75).tolist()
    d = paired_delta(score, model, beta=0.25, n_boot=500)
    assert d["dcvar_pct"] > d["djct_pct"]           # tail gain exceeds mean gain
    assert d["dp99_pct"] > 0


def test_paired_delta_shape_mismatch_raises():
    with pytest.raises(ValueError):
        paired_delta([1, 2, 3], [1, 2])
