"""Unit tests for eval/scripts/live_ab_heavytail.py (workload generator)."""
import numpy as np
import pytest

pytest.importorskip("gymnasium")  # generate_by_family pulls sim deps

from eval.scripts.live_ab_heavytail import (  # noqa: E402
    LIVE_GPU_MPS,
    LiveJob,
    _job_noise,
    gen_workload,
    job_name,
    join_records,
    parse_sacct_jct,
    peak_concurrent_mps,
    sbatch_cmd,
)

_SACCT_SAMPLE = """JobID|JobName|State|Submit|Start|End|ElapsedRaw
131|htab_score_1_jobA|COMPLETED|2026-06-14T10:00:00|2026-06-14T10:00:05|2026-06-14T10:01:05|60
132|htab_RDSAC-cvar_1_jobB|COMPLETED|2026-06-14T10:00:00|2026-06-14T10:00:10|2026-06-14T10:02:00|110
133|htab_SAC_0_jobC|FAILED|2026-06-14T10:00:00|2026-06-14T10:00:00|2026-06-14T10:00:01|1
999|unrelated-job|COMPLETED|2026-06-14T10:00:00|2026-06-14T10:00:00|2026-06-14T10:00:30|30
"""


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


def test_parse_sacct_jct_computes_jct_and_filters():
    parsed = parse_sacct_jct(_SACCT_SAMPLE)
    assert "unrelated-job" not in parsed            # only htab_* kept
    assert set(parsed) == {"htab_score_1_jobA", "htab_RDSAC-cvar_1_jobB", "htab_SAC_0_jobC"}
    assert parsed["htab_score_1_jobA"]["jct"] == 65.0   # End−Submit
    assert parsed["htab_score_1_jobA"]["wait"] == 5.0   # Start−Submit
    assert parsed["htab_RDSAC-cvar_1_jobB"]["jct"] == 120.0


def test_parse_sacct_jct_empty():
    assert parse_sacct_jct("") == {}


def _lj(job_id, true=50.0):
    return LiveJob(job_id, 0.0, true, true, 20)


def test_join_records_matches_completed_only():
    parsed = parse_sacct_jct(_SACCT_SAMPLE)
    jobs = [_lj("jobA"), _lj("jobMissing")]
    recs = join_records(jobs, parsed, "score", 1)
    assert len(recs) == 1                            # jobMissing absent → dropped
    assert recs[0]["job_id"] == "jobA"
    assert recs[0]["jct"] == 65.0
    assert recs[0]["true_runtime_s"] == 50.0


def test_join_records_drops_failed():
    parsed = parse_sacct_jct(_SACCT_SAMPLE)
    recs = join_records([_lj("jobC")], parsed, "SAC", 0)   # jobC FAILED
    assert recs == []


def test_job_name_roundtrips_into_join():
    name = job_name("RDSAC-cvar", 1, "jobB")
    assert name == "htab_RDSAC-cvar_1_jobB"
    parsed = parse_sacct_jct(_SACCT_SAMPLE)
    assert name in parsed
