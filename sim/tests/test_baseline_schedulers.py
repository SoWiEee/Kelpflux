"""Tests for the cited-SOTA baseline schedulers (review #4).

Kueue-style fair-share (max-min across users) and Volcano-style binpack
(largest-demand-first) orderings, plus registry/runner wiring.
"""
from __future__ import annotations

from sim.loader import Job
from sim.scheduler import make, REGISTRY
from sim.scheduler.kueue_fairshare import KueueFairShareScheduler
from sim.scheduler.volcano_binpack import VolcanoBinpackScheduler


def _job(job_id, user, gpu_count, submit_ts, mps_req=100):
    return Job(job_id=job_id, user=user, gpu_count=gpu_count, gpu_type="rtx4070",
               submit_ts=submit_ts, runtime=100.0, mem_req=8, mps_req=mps_req)


def test_registry_exposes_both_baselines():
    assert isinstance(make("kueue-fairshare"), KueueFairShareScheduler)
    assert isinstance(make("volcano-binpack"), VolcanoBinpackScheduler)
    assert {"kueue-fairshare", "volcano-binpack"} <= set(REGISTRY)


def test_kueue_fairshare_interleaves_users():
    # user A submits a burst (0,1,2); user B submits one job at t=0.5.
    # Fair-share must not let A monopolize: B's job should come after A's
    # first but before A's second (round-robin on per-user queue position).
    jobs = [
        _job("a0", "alice", 1, 0.0),
        _job("a1", "alice", 1, 1.0),
        _job("a2", "alice", 1, 2.0),
        _job("b0", "bob", 1, 0.5),
    ]
    order = [j.job_id for j in KueueFairShareScheduler().order(jobs, None, now=3.0)]
    assert order == ["a0", "b0", "a1", "a2"]


def test_kueue_is_a_permutation():
    jobs = [_job(f"j{i}", f"u{i % 3}", 1, float(i)) for i in range(9)]
    out = KueueFairShareScheduler().order(jobs, None, now=9.0)
    assert sorted(j.job_id for j in out) == sorted(j.job_id for j in jobs)


def test_volcano_binpack_largest_first():
    jobs = [
        _job("small", "u", 1, 0.0, mps_req=25),
        _job("big", "u", 4, 5.0),
        _job("mid", "u", 2, 1.0),
    ]
    order = [j.job_id for j in VolcanoBinpackScheduler().order(jobs, None, now=9.0)]
    assert order == ["big", "mid", "small"]


def test_volcano_tiebreaks_by_mps_then_submit():
    jobs = [
        _job("late_full", "u", 1, 9.0, mps_req=100),
        _job("early_frac", "u", 1, 0.0, mps_req=25),
    ]
    # same gpu_count → larger mps_req first (bigger demand), not submit order
    order = [j.job_id for j in VolcanoBinpackScheduler().order(jobs, None, now=9.0)]
    assert order == ["late_full", "early_frac"]


def test_both_expose_name_and_backfill():
    for sched in (KueueFairShareScheduler(), VolcanoBinpackScheduler()):
        assert isinstance(sched.name, str) and sched.name
        assert sched.backfill is True
