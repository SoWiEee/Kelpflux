"""Tests for the vanilla (non-distributional) discrete SAC path.

`DSACAgent(use_iqn=False)` swaps the dual-head IQN distributional critic for a
scalar twin-Q critic (Christodoulou 2019 discrete SAC): single Q head per
action, MSE soft-Bellman target with entropy folded in, twin-min over Q1/Q2.
Everything else (categorical actor, alpha auto-tune, PER, n-step) is shared with
RDSAC, so this is a clean distributional-vs-scalar ablation switchable by a flag.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from services.rl_scheduler.dsac import DSACAgent


def _batch(rng, n=16, obs_dim=6, n_actions=5):
    masks = np.ones((n, n_actions), dtype=np.bool_)
    return {
        "obs": rng.normal(size=(n, obs_dim)).astype(np.float32),
        "acts": rng.integers(0, n_actions, size=n).astype(np.int64),
        "rews": rng.normal(size=n).astype(np.float32),
        "next_obs": rng.normal(size=(n, obs_dim)).astype(np.float32),
        "dones": np.zeros(n, dtype=np.float32),
        "masks": masks,
        "next_masks": masks.copy(),
        "gammas": np.full(n, 0.99, dtype=np.float32),
    }


def _sac(obs_dim=6, n_actions=5, **kw):
    kw.setdefault("hidden", (32, 32))
    return DSACAgent(obs_dim=obs_dim, n_actions=n_actions, device="cpu",
                     use_iqn=False, **kw)


def test_sac_flag_recorded():
    agent = _sac()
    assert agent.use_iqn is False


def test_sac_critic_is_scalar_q():
    agent = _sac(obs_dim=6, n_actions=5)
    obs = torch.zeros(4, 6)
    q = agent.q1.q_values(obs)
    assert q.shape == (4, 5)
    # the scalar critic must NOT expose the IQN quantile interface
    assert not hasattr(agent.q1, "quantile_q")


def test_sac_update_returns_finite_losses():
    agent = _sac()
    rng = np.random.default_rng(0)
    out = agent.update(_batch(rng))
    for key in ("loss_critic", "loss_actor", "loss_alpha", "alpha", "entropy"):
        assert key in out and np.isfinite(out[key]), f"{key}={out.get(key)}"
    assert out["alpha"] > 0


def test_sac_update_returns_td_errors_for_per():
    agent = _sac()
    rng = np.random.default_rng(1)
    out = agent.update(_batch(rng, n=12))
    assert isinstance(out["td_errors"], np.ndarray)
    assert out["td_errors"].shape == (12,)
    assert np.all(out["td_errors"] >= 0)


def test_sac_select_action_never_picks_masked():
    agent = _sac()
    rng = np.random.default_rng(2)
    mask = np.array([False, True, False, True, True], dtype=np.bool_)
    for _ in range(200):
        obs = rng.normal(size=6).astype(np.float32)
        a = agent.select_action(obs, mask)
        assert mask[a]


def test_sac_learns_to_prefer_rewarded_action():
    """A scalar critic must raise Q for an action that always pays off."""
    agent = _sac(obs_dim=4, n_actions=3, hidden=(64, 64))
    rng = np.random.default_rng(7)
    obs = np.zeros((32, 4), dtype=np.float32)
    masks = np.ones((32, 3), dtype=np.bool_)
    for _ in range(300):
        acts = rng.integers(0, 3, size=32)
        rews = np.where(acts == 1, 1.0, -1.0).astype(np.float32)
        agent.update({
            "obs": obs, "acts": acts.astype(np.int64), "rews": rews,
            "next_obs": obs, "dones": np.ones(32, dtype=np.float32),
            "masks": masks, "next_masks": masks,
            "gammas": np.full(32, 0.99, dtype=np.float32),
        })
    q = agent.action_values(torch.zeros(1, 4)).squeeze(0)
    assert int(q.argmax()) == 1, f"Q={q.tolist()}"


def test_sac_save_load_round_trips_flag_and_action(tmp_path):
    agent = _sac()
    rng = np.random.default_rng(5)
    for _ in range(5):
        agent.update(_batch(rng))
    obs = rng.normal(size=6).astype(np.float32)
    mask = np.ones(5, dtype=np.bool_)
    before = agent.select_action(obs, mask, greedy=True)
    path = tmp_path / "sac.pt"
    agent.save(path)
    restored = DSACAgent.load(path, device="cpu")
    assert restored.use_iqn is False
    assert restored.select_action(obs, mask, greedy=True) == before


def test_iqn_default_unchanged():
    """RDSAC remains the default (use_iqn=True) so existing checkpoints load."""
    agent = DSACAgent(obs_dim=6, n_actions=5, device="cpu", hidden=(32, 32))
    assert agent.use_iqn is True
    assert hasattr(agent.q1, "quantile_q")


def test_sac_fixed_alpha_stays_constant():
    """With fixed_alpha, α must not drift and α-loss must be reported as 0."""
    agent = _sac(fixed_alpha=True, init_alpha=0.05)
    assert agent.opt_alpha is None
    assert not agent.log_alpha.requires_grad
    rng = np.random.default_rng(3)
    for _ in range(20):
        out = agent.update(_batch(rng))
        assert out["loss_alpha"] == 0.0
    assert abs(out["alpha"] - 0.05) < 1e-6


def test_sac_critic_target_uses_online_actor():
    """Faithful discrete SAC bootstraps the soft target from the ONLINE policy,
    so a stale actor_target must not influence the scalar critic update."""
    agent = _sac()
    # Desync the target actor: zero its weights so it differs from the online net.
    with torch.no_grad():
        for p in agent.actor_target.parameters():
            p.zero_()
    rng = np.random.default_rng(11)
    batch = _batch(rng)
    snapshot = {k: v.clone() for k, v in agent.actor_target.state_dict().items()}
    out = agent.update(batch)  # must run off the online actor, not the zeroed target
    assert np.isfinite(out["loss_critic"])
    # The scalar path never reads actor_target, so it stays exactly as set.
    for k, v in agent.actor_target.state_dict().items():
        assert torch.equal(v, snapshot[k])
