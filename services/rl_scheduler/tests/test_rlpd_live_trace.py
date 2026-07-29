from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")

from services.rl_scheduler.rlpd_finetune import load_live_trace_rollouts, RLPDAgent
from services.rl_scheduler.dsac import DSACAgent
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


def test_rlpd_save_is_serve_loadable_and_actor_identical(tmp_path: Path):
    """A faithful RLPD checkpoint must load via DSACAgent.load (the path serve.py
    uses) and the served actor must be the RLPD-trained actor, not a random one."""
    import numpy as np
    import torch

    obs_dim, n_actions = env_dims(n_nodes=2, gpus_per_node=1)  # 2x1 → (168, 33)
    agent = RLPDAgent(obs_dim, n_actions, device="cpu")

    ckpt = tmp_path / "rlpd.pt"
    agent.save(ckpt)

    # Servable format: loads through the exact call serve.py makes.
    loaded = DSACAgent.load(str(ckpt))
    assert loaded.obs_dim == obs_dim
    assert loaded.n_actions == n_actions
    assert loaded.use_iqn is False
    assert (Path(str(ckpt) + ".rlpd")).exists()  # ensemble provenance sidecar

    # The served policy IS the RLPD policy: identical action distribution.
    rng = np.random.default_rng(0)
    obs = torch.as_tensor(rng.standard_normal((4, obs_dim)), dtype=torch.float32)
    mask = torch.ones((4, n_actions), dtype=torch.bool)
    with torch.no_grad():
        p_rlpd, _ = agent.actor.policy(obs, mask)
        p_serve, _ = loaded.actor.policy(obs, mask)
    assert torch.allclose(p_rlpd, p_serve, atol=1e-6)
