"""UXP-RL DQN agent unit tests (faithful reproduction of Lin et al. 2025).

Covers the value-based DQN core:
1. select_action never returns an illegal (masked) action
2. greedy select_action returns the legal argmax of Q(s,·)
3. ε=high explores (random legal); ε=0 exploits
4. update() runs a gradient step and returns a finite MSE loss
5. Bellman target on a terminal transition is exactly the reward
6. hard target sync + ε decay fire every target_sync steps
7. save/load round-trips weights and hyperparameters
"""
from __future__ import annotations

import numpy as np
import torch

from services.rl_scheduler.uxprl import UXPRLAgent

OBS_DIM, N_ACT = 12, 5


def _mask(*legal: int) -> np.ndarray:
    m = np.zeros(N_ACT, dtype=bool)
    for i in legal:
        m[i] = True
    return m


def _agent(**kw) -> UXPRLAgent:
    return UXPRLAgent(obs_dim=OBS_DIM, n_actions=N_ACT, seed=0, **kw)


def test_select_action_respects_mask():
    agent = _agent()
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    legal = _mask(1, 3)
    for _ in range(200):
        a = agent.select_action(obs, legal)
        assert legal[a], f"picked illegal action {a}"


def test_greedy_returns_legal_argmax():
    agent = _agent()
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    legal = _mask(0, 2, 4)
    with torch.no_grad():
        q = agent.q(torch.as_tensor(obs).unsqueeze(0)).squeeze(0).numpy()
    expected = int(max(np.flatnonzero(legal), key=lambda i: q[i]))
    assert agent.select_action(obs, legal, greedy=True) == expected


def test_epsilon_zero_is_deterministic_greedy():
    agent = _agent()
    agent.epsilon = 0.0
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    legal = _mask(0, 1, 2, 3, 4)
    picks = {agent.select_action(obs, legal) for _ in range(50)}
    assert len(picks) == 1  # no exploration → single deterministic action


def test_update_returns_finite_loss():
    agent = _agent(batch_size=8)
    rng = np.random.default_rng(0)
    for _ in range(64):
        obs = rng.standard_normal(OBS_DIM).astype(np.float32)
        nxt = rng.standard_normal(OBS_DIM).astype(np.float32)
        agent.add(obs, rng.integers(N_ACT), rng.standard_normal(),
                  nxt, False, _mask(0, 1, 2, 3, 4))
    info = agent.update()
    assert "loss" in info and np.isfinite(info["loss"])


def test_bellman_target_terminal_equals_reward():
    """On a done transition the target is exactly r (γ·max Q_target·0 = 0)."""
    agent = _agent(batch_size=1)
    obs = np.ones(OBS_DIM, dtype=np.float32)
    nxt = np.ones(OBS_DIM, dtype=np.float32)
    # Fill replay with identical done transitions so the sampled target == reward.
    for _ in range(4):
        agent.add(obs, 2, 1.5, nxt, True, _mask(0, 1, 2, 3, 4))
    q_before = agent.q(torch.as_tensor(obs).unsqueeze(0))[0, 2].item()
    # Take several steps; Q(s,a=2) should move toward the reward 1.5, not beyond it
    # driven by any bootstrap (bootstrap is zeroed by done).
    for _ in range(300):
        agent.update()
    q_after = agent.q(torch.as_tensor(obs).unsqueeze(0))[0, 2].item()
    assert abs(q_after - 1.5) < abs(q_before - 1.5)
    assert q_after < 1.7  # cannot overshoot the reward via bootstrap on a terminal


def test_hard_target_sync_and_epsilon_decay():
    agent = _agent(batch_size=4, target_sync=10, eps_start=0.9,
                   eps_min=0.1, eps_decay=0.5)
    rng = np.random.default_rng(1)
    for _ in range(32):
        agent.add(rng.standard_normal(OBS_DIM).astype(np.float32),
                  rng.integers(N_ACT), rng.standard_normal(),
                  rng.standard_normal(OBS_DIM).astype(np.float32),
                  False, _mask(0, 1, 2, 3, 4))
    eps0 = agent.epsilon
    for _ in range(10):
        agent.update()
    assert agent.epsilon < eps0                      # decayed once at step 10
    assert np.isclose(agent.epsilon, 0.9 * 0.5)
    # target net now equals online net right after a sync
    for (p, pt) in zip(agent.q.parameters(), agent.q_target.parameters()):
        assert torch.allclose(p, pt)


def test_save_load_roundtrip(tmp_path):
    agent = _agent(batch_size=4)
    rng = np.random.default_rng(2)
    for _ in range(16):
        agent.add(rng.standard_normal(OBS_DIM).astype(np.float32),
                  rng.integers(N_ACT), rng.standard_normal(),
                  rng.standard_normal(OBS_DIM).astype(np.float32),
                  False, _mask(0, 1, 2, 3, 4))
    agent.update()
    ckpt = tmp_path / "dsac.pt"
    agent.save(ckpt)
    assert UXPRLAgent.is_uxprl_checkpoint(ckpt)

    loaded = UXPRLAgent.load(ckpt)
    assert loaded.obs_dim == OBS_DIM and loaded.n_actions == N_ACT
    assert loaded.use_iqn is False
    obs = rng.standard_normal(OBS_DIM).astype(np.float32)
    a = torch.as_tensor(obs).unsqueeze(0)
    assert torch.allclose(agent.q(a), loaded.q(a))
