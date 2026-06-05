"""Env exposes per-job JCTs so risk-sensitive tail metrics are measurable."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from sim.gym_env import KubefluxSchedEnv
from sim.loader import generate_by_family


def _factory(seed=42):
    def _build():
        return [j for j in generate_by_family("philly", n_jobs=30, seed=seed)
                if j.gpu_count <= 1]
    return _build


def test_episode_jcts_collected_for_completed_jobs():
    env = KubefluxSchedEnv(_factory(), n_nodes=1, gpus_per_node=1, max_steps=30 * 200)
    obs, _ = env.reset(seed=42)
    done = False
    info = {}
    while not done:
        mask = env.action_mask()
        # legal non-no-op action when available, else no-op
        legal = np.flatnonzero(mask)
        act = int(legal[0])
        obs, _, term, trunc, info = env.step(act)
        done = term or trunc
    jcts = env.episode_jcts()
    assert len(jcts) == info["completed"]
    assert all(j >= 0 for j in jcts)
    # mean of per-job JCTs matches the running avg_jct the env reports
    if jcts:
        assert abs(float(np.mean(jcts)) - info["avg_jct"]) < 1e-6
    env.close()
