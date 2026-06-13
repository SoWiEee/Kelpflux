"""Tests for the co-location action mode (B).

With colocation_actions=True every (job, placement) splits into PACK / ISOLATE:
  PACK    — accept MPS sharing (legal whenever the job fits).
  ISOLATE — require an idle GPU (legal only when the target GPU is empty).

Default (colocation off) must stay byte-identical to the legacy 17-action env.
"""
from __future__ import annotations

import numpy as np

from sim.gym_env import (
    KubefluxSchedEnv, env_dims, MODE_PACK, MODE_ISOLATE, TOP_K,
)
from sim.loader import Job, MPS_PER_GPU


def _two_half_gpu_jobs() -> list[Job]:
    """Two 50-MPS single-GPU jobs at t=0 — both fit on one idle GPU together."""
    return [
        Job(job_id=f"h{i}", user="u", gpu_count=1, gpu_type="rtx4070",
            submit_ts=0.0, runtime=1000.0, mem_req=0.0, mps_req=50)
        for i in range(2)
    ]


def _env(colocation: bool) -> KubefluxSchedEnv:
    return KubefluxSchedEnv(
        _two_half_gpu_jobs, n_nodes=1, gpus_per_node=1, mps_per_gpu=MPS_PER_GPU,
        max_steps=10_000, reward_mode="jct_aligned", colocation_actions=colocation,
    )


def test_env_dims_doubles_with_colocation():
    base_obs, base_act = env_dims(1, 1, colocation=False)
    co_obs, co_act = env_dims(1, 1, colocation=True)
    assert base_obs == co_obs                      # obs layout unchanged
    assert base_act == TOP_K * 1 * 1 + 1           # 17
    assert co_act == TOP_K * 1 * 1 * 2 + 1         # 33


def test_default_action_space_unchanged():
    env = _env(colocation=False)
    assert env._n_modes == 1
    assert env.action_space.n == 17
    # legacy decode shape (with the new trailing mode always 0)
    assert env._decode(0) == (0, 0, 0, 0)


def test_encode_decode_roundtrip_with_modes():
    env = _env(colocation=True)
    assert env.action_space.n == 33
    for job_i in range(TOP_K):
        for mode in (MODE_PACK, MODE_ISOLATE):
            a = env._encode(job_i, 0, mode)
            ji, nj, gk, m = env._decode(a)
            assert (ji, nj, gk, m) == (job_i, 0, 0, mode)


def test_isolate_masked_once_gpu_is_occupied():
    env = _env(colocation=True)
    env.reset(seed=0)
    # Idle GPU: both PACK and ISOLATE legal for the head job.
    m0 = env.action_mask()
    assert m0[env._encode(0, 0, MODE_PACK)]
    assert m0[env._encode(0, 0, MODE_ISOLATE)]

    # Place the head job via PACK → GPU now half-full (occupied).
    env.step(env._encode(0, 0, MODE_PACK))
    m1 = env.action_mask()
    # The remaining 50-MPS job still PACKs (fits the 50 free slots) ...
    assert m1[env._encode(0, 0, MODE_PACK)]
    # ... but ISOLATE is now illegal — the GPU is not idle.
    assert not m1[env._encode(0, 0, MODE_ISOLATE)]
    assert m1[env._no_op]


def test_colocation_episode_completes():
    """A greedy PACK-first rollout still drains the queue."""
    env = _env(colocation=True)
    obs, _ = env.reset(seed=1)
    done = False
    info = {}
    while not done:
        mask = env.action_mask()
        act = int(np.flatnonzero(mask)[0])
        obs, _, term, trunc, info = env.step(act)
        done = term or trunc
    assert info["completed"] == 2
