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


def _extreme_batch(obs_dim, n_actions, reward=100.0):
    rng = np.random.default_rng(3)
    return {
        "obs": rng.normal(size=(8, obs_dim)).astype(np.float32),
        "acts": np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64),
        "rews": np.full((8,), reward, dtype=np.float32),
        "next_obs": rng.normal(size=(8, obs_dim)).astype(np.float32),
        "dones": np.zeros((8,), dtype=np.float32),
        "masks": np.ones((8, n_actions), dtype=np.bool_),
        "next_masks": np.ones((8, n_actions), dtype=np.bool_),
        "gammas": np.full((8,), 0.99, dtype=np.float32),
    }


def _seeded_agent(value_clip, obs_dim, n_actions):
    import torch
    torch.manual_seed(0)
    return DSACAgent(obs_dim=obs_dim, n_actions=n_actions, risk_mode="mean",
                     value_clip=value_clip, device="cpu")


def test_value_clip_bounds_extreme_target_and_shrinks_critic_loss():
    """Duan-style target return-clip: an extreme bootstrap target is bounded to
    ±b of the current value, so the critic loss is far smaller than unclipped."""
    import torch
    obs_dim, n_actions = 5, 4
    batch = _extreme_batch(obs_dim, n_actions, reward=100.0)

    a_noclip = _seeded_agent(0.0, obs_dim, n_actions)
    a_clip = _seeded_agent(1.0, obs_dim, n_actions)
    assert a_clip.value_clip == 1.0

    torch.manual_seed(123)
    l_noclip = a_noclip.update({k: v.copy() for k, v in batch.items()})
    torch.manual_seed(123)
    l_clip = a_clip.update({k: v.copy() for k, v in batch.items()})

    assert np.isfinite(l_clip["loss_critic"]) and np.isfinite(l_noclip["loss_critic"])
    # reward=100 with b=1 → clipped target O(1) vs unclipped O(100): >>2x smaller.
    assert l_clip["loss_critic"] < 0.5 * l_noclip["loss_critic"]


def test_value_clip_round_trips_through_checkpoint(tmp_path):
    agent = DSACAgent(obs_dim=5, n_actions=4, risk_mode="cvar", value_clip=8.0,
                      device="cpu")
    p = tmp_path / "clip.pt"
    agent.save(p)
    reloaded = DSACAgent.load(p)
    assert reloaded.value_clip == 8.0
