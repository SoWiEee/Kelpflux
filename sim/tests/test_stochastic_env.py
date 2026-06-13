"""Tests for opt-in stochastic execution in KubefluxSchedEnv.

The stochasticity (lognormal runtime noise + MPS co-residency interference)
is what gives the return distribution genuine spread so a distributional /
risk-sensitive critic (RDSAC) has something to model. It MUST be opt-in:
with runtime_sigma=0 and interference=0 the env is byte-identical to the
legacy deterministic env, so the existing 30-seed report and test suite stay
reproducible.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from sim.gym_env import KubefluxSchedEnv
from sim.loader import Job, MPS_PER_GPU


def _fixed_jobs() -> list[Job]:
    """A small deterministic batch of fractional-MPS single-GPU jobs.

    All fractional so they can co-reside on the single 1×1 GPU (exercises the
    interference path) and all submitted at t=0 so ordering is stable.
    """
    return [
        Job(job_id=f"j{i}", user="u", gpu_count=1, gpu_type="rtx4070",
            submit_ts=0.0, runtime=1000.0, mem_req=0.0, mps_req=25)
        for i in range(6)
    ]


def _run_episode(env: KubefluxSchedEnv, seed: int) -> list[float]:
    """Greedy first-legal-action rollout; returns per-job JCTs."""
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        mask = env.action_mask()
        legal = np.flatnonzero(mask)
        # prefer a real placement (lowest index) over no-op (last index)
        act = int(legal[0])
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    return env.episode_jcts()


def _make_env(**kwargs) -> KubefluxSchedEnv:
    return KubefluxSchedEnv(
        _fixed_jobs, n_nodes=1, gpus_per_node=1, mps_per_gpu=MPS_PER_GPU,
        max_steps=10_000, reward_mode="jct_aligned", **kwargs,
    )


def test_defaults_are_deterministic_and_unchanged():
    """σ=0, interference=0 → identical JCTs across runs and across seeds.

    This is the backward-compatibility guarantee: the realized runtime equals
    the nominal runtime exactly, so legacy eval/tests are unaffected.
    """
    base = _run_episode(_make_env(), seed=42)
    again = _run_episode(_make_env(), seed=999)  # different seed, no RNG draw
    assert base == again
    assert len(base) == 6


def test_sigma_is_reproducible_per_seed():
    """Same seed + same σ → identical realized JCTs."""
    a = _run_episode(_make_env(runtime_sigma=0.8), seed=7)
    b = _run_episode(_make_env(runtime_sigma=0.8), seed=7)
    assert a == b


def test_sigma_increases_variance_mean_preserved():
    """σ>0 widens the JCT distribution; mean stays ~unchanged (mean-preserving)."""
    det = np.array([
        _run_episode(_make_env(), seed=s) for s in range(40)
    ]).ravel()
    noisy = np.array([
        _run_episode(_make_env(runtime_sigma=0.7), seed=s) for s in range(40)
    ]).ravel()

    # The deterministic baseline already has queueing spread (6 jobs / 4 slots
    # → JCTs split 1000/2000). Noise must add variance ON TOP of that and turn
    # the 2 discrete JCT values into a genuine continuum.
    assert noisy.std() > det.std() * 1.3
    assert len(np.unique(det)) <= 3 < len(np.unique(noisy))
    # mean-preserving lognormal: realized mean within ~15% of nominal mean
    assert abs(noisy.mean() - det.mean()) / det.mean() < 0.15


def test_interference_inflates_colocated_runtime():
    """With interference>0, jobs sharing the GPU finish later than nominal.

    Six 25-MPS jobs share one 100-slot GPU (4 fit at once). Co-residency must
    push realized JCTs above the deterministic baseline.
    """
    det = _run_episode(_make_env(), seed=1)
    interf = _run_episode(_make_env(interference=0.5), seed=1)
    assert sum(interf) > sum(det)
