from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
from services.rl_scheduler.dsac import DSACAgent


def test_distributional_cvar_agent_updates_and_selects_feasible_action():
    obs_dim = 5
    n_actions = 4
    agent = DSACAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        risk_mode="cvar",
        risk_beta=0.25,
        device="cpu",
    )

    batch = {
        "obs": np.random.default_rng(1).normal(size=(8, obs_dim)).astype(np.float32),
        "acts": np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64),
        "rews": np.linspace(-1.0, 1.0, 8, dtype=np.float32),
        "next_obs": np.random.default_rng(2).normal(size=(8, obs_dim)).astype(np.float32),
        "dones": np.zeros((8,), dtype=np.float32),
        "masks": np.ones((8, n_actions), dtype=np.bool_),
        "next_masks": np.ones((8, n_actions), dtype=np.bool_),
        "gammas": np.full((8,), 0.99, dtype=np.float32),
    }

    losses = agent.update(batch)

    assert np.isfinite(losses["loss_critic"])
    assert losses["td_errors"].shape == (8,)

    mask = np.array([False, True, False, True], dtype=np.bool_)
    action = agent.select_action(np.zeros((obs_dim,), dtype=np.float32), mask, greedy=True)

    assert mask[action]
