"""UXP-RL reward-function tests (Lin et al. 2025, IEEE TNSM §IV-B.3).

Verifies the user-experience-balanced reward:
1. inference tasks earn strictly more than non-inference at equal turnaround (c2 > c1)
2. reward is inversely proportional to turnaround (faster → higher)
3. inference past the Δ^I deadline is damped by 1/(T_T − Δ^I + 1)
4. an inference task with no deadline (slo_s ≤ 0) incurs no penalty
5. the env wires reward_mode="uxprl" and emits ONLY this positive per-task term
   (no negative placement/shaping/final-charge terms leak in)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from sim.gym_env import KubefluxSchedEnv, uxprl_task_reward  # noqa: E402
from sim.loader import Job  # noqa: E402


def test_inference_outweighs_batch_at_equal_turnaround():
    r_inf = uxprl_task_reward(10.0, "inference", slo_s=100.0, c1=1.0, c2=2.0)
    r_batch = uxprl_task_reward(10.0, "batch", slo_s=0.0, c1=1.0, c2=2.0)
    assert r_inf > r_batch
    assert np.isclose(r_inf / r_batch, 2.0)  # c2/c1


def test_reward_decreases_with_turnaround():
    fast = uxprl_task_reward(5.0, "batch", 0.0)
    slow = uxprl_task_reward(50.0, "batch", 0.0)
    assert fast > slow
    assert np.isclose(fast, 1.0 / 5.0)


def test_inference_deadline_penalty_damps_reward():
    within = uxprl_task_reward(9.0, "inference", slo_s=10.0, c2=2.0)   # T_T ≤ Δ^I
    over = uxprl_task_reward(11.0, "inference", slo_s=10.0, c2=2.0)    # T_T > Δ^I
    # Base c2/T_T for the over case would be 2/11 ≈ 0.1818; the 1/(11-10+1)=0.5
    # damping halves it → 0.0909, far below the within-deadline reward.
    assert over < within
    assert np.isclose(over, (2.0 / 11.0) * (1.0 / (11.0 - 10.0 + 1.0)))


def test_inference_without_deadline_is_undamped():
    r = uxprl_task_reward(1000.0, "inference", slo_s=0.0, c2=2.0)
    assert np.isclose(r, 2.0 / 1000.0)  # Δ^I = ∞ → no penalty branch


def _single_job_factory():
    def _f():
        return [Job(job_id="j0", user="u", gpu_count=1, gpu_type="rtx4070",
                    submit_ts=0.0, runtime=8.0, mem_req=1.0, mps_req=4,
                    job_class="inference", slo_s=100.0)]
    return _f


def test_env_uxprl_reward_is_positive_only():
    env = KubefluxSchedEnv(_single_job_factory(), n_nodes=1, gpus_per_node=1,
                           reward_mode="uxprl", reward_scale=1.0,
                           uxprl_c1=1.0, uxprl_c2=2.0, max_steps=500)
    obs, _ = env.reset(seed=0)
    total = 0.0
    got_completion = False
    for _ in range(500):
        mask = env.action_mask()
        legal = np.flatnonzero(mask)
        # Prefer a real placement (index 0 = job0 on the only node) when legal.
        act = 0 if (0 < len(mask) and mask[0]) else int(legal[0])
        obs, rew, term, trunc, info = env.step(act)
        total += rew
        assert rew >= 0.0, "UXP-RL reward must never be negative"
        if info.get("completed", 0) >= 1:
            got_completion = True
        if term or trunc:
            break
    assert got_completion, "job never completed"
    assert total > 0.0, "a completed inference task must yield positive reward"
