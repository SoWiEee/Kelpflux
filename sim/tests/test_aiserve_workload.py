"""AI-serving workload (inference + training, moderate load) and SLO metrics.

These lock in the property that motivates the workload: on a small (2×1) MPS
cluster, a dual-class moderate-load trace + an SLO metric *discriminates*
schedulers (FCFS lets latency-SLO inference queue behind long training jobs →
high SLO-violation rate), whereas the old degenerate/saturated trace flattened
them. See docs/eval-writeup §4.5 / the workload-design discussion.
"""
from __future__ import annotations

from sim.loader import generate_ai_serving
from sim.runner import run


def test_generator_emits_two_classes_with_slo_on_inference():
    jobs = generate_ai_serving(300, seed=1, n_gpus=2, load=0.7)
    classes = {j.job_class for j in jobs}
    assert classes == {"inference", "training"}
    inf = [j for j in jobs if j.job_class == "inference"]
    tr = [j for j in jobs if j.job_class == "training"]
    # inference carries an SLO; training is best-effort
    assert all(j.slo_s > 0 for j in inf)
    assert all(j.slo_s == 0 for j in tr)
    # inference is short + MPS-fractional; training is longer
    assert all(j.mps_req in (25, 50) and j.gpu_count == 1 for j in inf)
    assert max(j.gpu_count for j in tr) == 2  # some 2-GPU gang jobs exist


def test_moderate_load_not_saturated():
    jobs = generate_ai_serving(400, seed=2, n_gpus=2, load=0.7)
    m, _ = run(jobs, n_nodes=2, gpus_per_node=1, scheduler_name="multifactor", mps_per_gpu=100)
    s = m.summary()
    # offered load ~0.7 → utilization should be meaningfully below saturation
    assert 0.3 < s["utilization"] < 0.95


def test_slo_metric_present_and_counts_violations():
    jobs = generate_ai_serving(300, seed=3, n_gpus=2, load=0.8)
    m, _ = run(jobs, n_nodes=2, gpus_per_node=1, scheduler_name="fcfs", mps_per_gpu=100)
    s = m.summary()
    assert s["slo_n"] > 0
    assert 0.0 <= s["slo_violation_rate"] <= 1.0
    assert s["slo_violations"] <= s["slo_n"]
    assert "inference" in s["by_class"] and "training" in s["by_class"]


def test_fcfs_violates_slo_more_than_multifactor():
    """The discrimination property: FCFS (no reorder) lets inference queue behind
    long training jobs, so it misses SLOs far more than a size-aware policy."""
    jobs = generate_ai_serving(400, seed=42, n_gpus=2, load=0.7)
    fcfs = run(jobs, n_nodes=2, gpus_per_node=1, scheduler_name="fcfs", mps_per_gpu=100)[0].summary()
    mf = run(jobs, n_nodes=2, gpus_per_node=1, scheduler_name="multifactor", mps_per_gpu=100)[0].summary()
    assert fcfs["slo_violation_rate"] > mf["slo_violation_rate"] + 0.10
    # and inference latency is much worse under FCFS
    assert fcfs["by_class"]["inference"]["jct_mean"] > mf["by_class"]["inference"]["jct_mean"]
