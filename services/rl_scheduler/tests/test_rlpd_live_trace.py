from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")

from services.rl_scheduler.rlpd_finetune import load_live_trace_rollouts
from sim.gym_env import env_dims


def test_load_live_trace_rollouts_replays_normalized_trace(tmp_path: Path):
    trace = [
        {
            "job_id": "live-1",
            "user": "u",
            "gpu_count": 1,
            "gpu_type": "rtx4070",
            "submit_ts": 0.0,
            "runtime": 5.0,
            "mem_req": 0.0,
            "mps_req": 25,
            "latency_class": "gpu_warm",
        },
        {
            "job_id": "live-2",
            "user": "u",
            "gpu_count": 1,
            "gpu_type": "rtx4070",
            "submit_ts": 1.0,
            "runtime": 4.0,
            "mem_req": 0.0,
            "mps_req": 25,
            "latency_class": "gpu_warm",
        },
    ]
    path = tmp_path / "live-trace.json"
    path.write_text(json.dumps(trace))
    obs_dim, n_actions = env_dims(n_nodes=1, gpus_per_node=1)

    buf = load_live_trace_rollouts(
        [str(path)],
        obs_dim=obs_dim,
        n_actions=n_actions,
        capacity=32,
        n_nodes=1,
        gpus_per_node=1,
        n_jobs=10,
    )

    assert len(buf) > 0
    assert buf.obs.shape == (32, obs_dim)
    assert buf.masks.shape == (32, n_actions)
    assert buf.masks[:len(buf)].any(axis=1).all()
