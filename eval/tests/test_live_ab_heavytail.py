"""Unit tests for eval/scripts/live_ab_heavytail.py (workload generator)."""
import numpy as np
import pytest

pytest.importorskip("gymnasium")  # generate_by_family pulls sim deps

from eval.scripts.live_ab_heavytail import (  # noqa: E402
    LIVE_GPU_MPS,
    _job_noise,
    gen_workload,
    peak_concurrent_mps,
    sbatch_cmd,
)


def test_sigma_zero_reported_equals_true():
    jobs = gen_workload("philly", 60, seed=1, sigma=0.0)
    assert jobs
    for j in jobs:
        assert j.reported_runtime_s == pytest.approx(j.true_runtime_s)


def test_job_noise_sigma_zero_is_exactly_one_no_draw():
    assert _job_noise("anything", 0.0) == 1.0


def test_job_noise_is_common_random_per_job():
    # same (seed, job_id) → identical multiplier across calls (paired arms)
    a = _job_noise("job-7", 1.0, seed=42)
    b = _job_noise("job-7", 1.0, seed=42)
    assert a == b
    assert _job_noise("job-8", 1.0, seed=42) != a  # different job → different draw


def test_job_noise_mean_preserving():
    # E[exp(σZ−σ²/2)] = 1; mean over many independent per-job draws ≈ 1.
    mults = np.array([_job_noise(f"j{i}", 1.0, seed=0) for i in range(5000)])
    assert mults.mean() == pytest.approx(1.0, abs=0.1)


def test_sigma_injects_variance_and_keeps_heavy_tail():
    jobs = gen_workload("philly", 400, seed=3, sigma=1.0)
    ratio = np.array([j.reported_runtime_s / j.true_runtime_s for j in jobs])
    assert ratio.std() > 0.1                       # genuine estimate spread
    trues = np.array([j.true_runtime_s for j in jobs])
    assert np.percentile(trues, 99) / np.percentile(trues, 50) > 3.0  # tail preserved


def test_workload_is_oversubscribed():
    jobs = gen_workload("philly", 200, seed=4, sigma=1.0, mps_oversub=4.0)
    peak = peak_concurrent_mps(jobs)
    assert peak > LIVE_GPU_MPS                      # high contention by construction


def test_time_compression_respects_target_max():
    jobs = gen_workload("ali", 150, seed=5, sigma=0.0, target_max_s=120.0)
    trues = [j.true_runtime_s for j in jobs]
    assert max(trues) == pytest.approx(120.0, rel=0.01)
    assert min(trues) >= 2.0                        # floor clamp


def test_only_single_gpu_jobs():
    jobs = gen_workload("philly", 120, seed=6, sigma=1.0)
    assert all(j.gpu_count <= 1 for j in jobs)
    assert all(1 <= j.mps_req <= LIVE_GPU_MPS for j in jobs)


def test_rejects_non_heavytail_family():
    with pytest.raises(ValueError):
        gen_workload("burst", 50, seed=1)


def test_sbatch_cmd_uses_reported_for_time_true_for_sleep():
    jobs = gen_workload("philly", 30, seed=7, sigma=1.0)
    j = jobs[0]
    cmd = sbatch_cmd(j, "RDSAC", 2)
    joined = " ".join(cmd)
    assert f"--gres=mps:{j.mps_req}" in joined
    assert f"sleep {int(round(j.true_runtime_s))}" in joined  # actual = true
    # --time is derived from the noisy *reported* estimate (minutes, ceil)
    expected_time = max(1, int(np.ceil(j.reported_runtime_s / 60.0)))
    assert f"--time={expected_time}" in joined
    assert "htab_RDSAC_2_" in joined


def test_paired_stream_identical_across_seeds_call():
    # same params → identical stream (deterministic, for CRN paired arms)
    a = gen_workload("philly", 80, seed=9, sigma=1.0)
    b = gen_workload("philly", 80, seed=9, sigma=1.0)
    assert [j.reported_runtime_s for j in a] == [j.reported_runtime_s for j in b]
    assert [j.mps_req for j in a] == [j.mps_req for j in b]
