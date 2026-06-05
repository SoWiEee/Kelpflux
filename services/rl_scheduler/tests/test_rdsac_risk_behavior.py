"""Behavioural test: CVaR distortion makes the policy risk-averse.

Single-state, single-step bandit with two actions:
  action 0 — high mean, high variance reward  N(1.0, 1.5)  (risky)
  action 1 — low mean,  low variance  reward  N(0.5, 0.05) (safe)

Mean (risk-neutral) value prefers action 0 (1.0 > 0.5). CVaR(0.25) prefers the
safe action 1 (its bad tail is far less severe). A correct RDSAC implementation
must shift policy mass toward the safe action under CVaR relative to mean.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from services.rl_scheduler.dsac import DSACAgent

OBS_DIM = 4
N_ACTIONS = 2
# Non-zero constant context. The IQN critic combines state and quantile
# embeddings multiplicatively (s ⊙ φ(τ), Dabney et al. 2018); an all-zero obs
# would zero out s and collapse every quantile to a constant. Real scheduler
# observations are never all-zero (gpu_type one-hot always carries a 1.0).
CTX = np.array([1.0, -0.5, 0.3, 0.8], dtype=np.float32)


def _bandit_batch(rng, n=128):
    acts = rng.integers(0, N_ACTIONS, size=n)
    rews = np.where(
        acts == 0,
        rng.normal(1.0, 1.5, size=n),    # risky
        rng.normal(0.5, 0.05, size=n),   # safe
    ).astype(np.float32)
    obs = np.tile(CTX, (n, 1))
    return {
        "obs": obs,
        "acts": acts.astype(np.int64),
        "rews": rews,
        "next_obs": obs.copy(),
        "dones": np.ones(n, dtype=np.float32),   # 1-step bandit
        "masks": np.ones((n, N_ACTIONS), dtype=np.bool_),
        "next_masks": np.ones((n, N_ACTIONS), dtype=np.bool_),
        "gammas": np.full(n, 0.99, dtype=np.float32),
    }


def _train(risk_mode, n_updates=600, seed=0):
    rng = np.random.default_rng(seed)
    agent = DSACAgent(
        obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=(64, 64),
        risk_mode=risk_mode, risk_beta=0.25,
        fixed_alpha=True, init_alpha=0.02, device="cpu",
    )
    for _ in range(n_updates):
        agent.update(_bandit_batch(rng))
    import torch
    probs, _ = agent.actor.policy(
        torch.as_tensor(CTX).unsqueeze(0),
        torch.ones(1, N_ACTIONS, dtype=bool),
    )
    return float(probs[0, 1].item())   # P(safe action)


def test_cvar_is_more_risk_averse_than_mean():
    p_safe_mean = _train("mean", seed=1)
    p_safe_cvar = _train("cvar", seed=1)
    # CVaR must put more mass on the safe action than the risk-neutral policy
    assert p_safe_cvar > p_safe_mean + 0.1, (
        f"CVaR P(safe)={p_safe_cvar:.3f} not meaningfully above "
        f"mean P(safe)={p_safe_mean:.3f}"
    )
    # And CVaR should actively prefer the safe action
    assert p_safe_cvar > 0.5, f"CVaR did not prefer safe action: {p_safe_cvar:.3f}"
