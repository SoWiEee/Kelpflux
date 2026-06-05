"""Tail metrics for risk-sensitive scheduling: p99 JCT and tail slowdown.

These let CVaR vs mean be compared on the metric the distortion actually targets.
"""
from __future__ import annotations

from sim.metrics import MetricCollector


def _collector_with_jcts(n: int = 100) -> MetricCollector:
    """Jobs with submit=0, end=i+1 so JCT = i+1 in {1..n}, fixed runtime=60."""
    mc = MetricCollector()
    for i in range(n):
        jid = f"j{i}"
        mc.record_submit(jid, user="u", gpu_count=1, mps_req=25,
                         submit_ts=0.0, runtime=60.0)
        mc.record_start(jid, 0.0)
        mc.record_end(jid, float(i + 1))
    return mc


def test_summary_includes_tail_keys():
    s = _collector_with_jcts().summary()
    for key in ("jct_p99", "slowdown_p95", "slowdown_p99"):
        assert key in s, f"missing {key}"


def test_jct_p99_is_near_top():
    s = _collector_with_jcts(100).summary()
    # JCTs are 1..100; p99 should sit near 99
    assert 98.0 <= s["jct_p99"] <= 100.0


def test_tail_slowdown_exceeds_p90():
    s = _collector_with_jcts(100).summary()
    assert s["slowdown_p99"] >= s["slowdown_p90"] >= s["slowdown_mean"]
