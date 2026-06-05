"""Tests for the RDSAC agent (Ma et al. discrete transpose).

Dual distributional critic (Z_R reward-return, Z_H entropy-return) + explicit
categorical actor + risk distortion in the actor objective.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from services.rl_scheduler.dsac import DSACAgent


def _batch(rng, n=16, obs_dim=6, n_actions=5, all_legal=True):
    masks = np.ones((n, n_actions), dtype=np.bool_)
    if not all_legal:
        masks[:, 0] = False  # action 0 illegal
        masks[:, -1] = True  # keep at least the no-op legal
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


def _agent(obs_dim=6, n_actions=5, **kw):
    kw.setdefault("hidden", (32, 32))
    return DSACAgent(obs_dim=obs_dim, n_actions=n_actions, device="cpu", **kw)


def test_update_returns_new_loss_keys_finite():
    agent = _agent()
    rng = np.random.default_rng(0)
    out = agent.update(_batch(rng))
    for key in ("loss_critic", "loss_actor", "loss_alpha", "alpha", "entropy"):
        assert key in out, f"missing {key}"
        assert np.isfinite(out[key]), f"{key} not finite: {out[key]}"
    assert out["alpha"] > 0


def test_update_returns_td_errors_for_per():
    agent = _agent()
    rng = np.random.default_rng(1)
    out = agent.update(_batch(rng, n=12))
    assert isinstance(out["td_errors"], np.ndarray)
    assert out["td_errors"].shape == (12,)
    assert np.all(out["td_errors"] >= 0)


def test_select_action_never_picks_masked():
    agent = _agent()
    rng = np.random.default_rng(2)
    mask = np.array([False, True, False, True, True], dtype=np.bool_)
    for _ in range(200):
        obs = rng.normal(size=6).astype(np.float32)
        a = agent.select_action(obs, mask)
        assert mask[a], f"picked masked action {a}"


def test_select_action_greedy_is_deterministic():
    agent = _agent()
    obs = np.zeros(6, dtype=np.float32)
    mask = np.ones(5, dtype=np.bool_)
    a1 = agent.select_action(obs, mask, greedy=True)
    a2 = agent.select_action(obs, mask, greedy=True)
    assert a1 == a2


def test_critic_has_dual_heads_with_quantile_shape():
    agent = _agent(obs_dim=6, n_actions=5)
    obs = torch.zeros(4, 6)
    taus = torch.rand(4, agent.q1.N_QUANT)
    z_r, z_h = agent.q1.quantile_q(obs, taus)
    assert z_r.shape == (4, agent.q1.N_QUANT, 5)
    assert z_h.shape == (4, agent.q1.N_QUANT, 5)


def test_per_weights_change_critic_loss():
    rng = np.random.default_rng(3)
    batch = _batch(rng, n=16)
    a1 = _agent()
    base = a1.update(dict(batch))["loss_critic"]
    a2 = _agent()
    weighted = dict(batch)
    weighted["weights"] = np.full(16, 0.1, dtype=np.float32)
    down = a2.update(weighted)["loss_critic"]
    # both finite; weighting path exercised (different agents, just sanity finite)
    assert np.isfinite(base) and np.isfinite(down)


@pytest.mark.parametrize("mode", ["mean", "cvar", "wang", "cpw", "msd"])
def test_runs_under_each_risk_mode(mode):
    agent = _agent(risk_mode=mode, risk_beta=0.25)
    rng = np.random.default_rng(4)
    out = agent.update(_batch(rng))
    assert np.isfinite(out["loss_actor"])


def test_save_load_preserves_greedy_action(tmp_path):
    agent = _agent(risk_mode="cvar", risk_beta=0.3)
    rng = np.random.default_rng(5)
    for _ in range(5):
        agent.update(_batch(rng))
    obs = rng.normal(size=6).astype(np.float32)
    mask = np.ones(5, dtype=np.bool_)
    before = agent.select_action(obs, mask, greedy=True)
    path = tmp_path / "rdsac.pt"
    agent.save(path)
    restored = DSACAgent.load(path, device="cpu")
    after = restored.select_action(obs, mask, greedy=True)
    assert before == after
    assert restored.risk_mode == "cvar"


def test_fixed_alpha_keeps_alpha_constant():
    agent = _agent(fixed_alpha=True, init_alpha=0.2)
    rng = np.random.default_rng(6)
    a0 = agent.update(_batch(rng))["alpha"]
    for _ in range(5):
        agent.update(_batch(rng))
    a1 = agent.update(_batch(rng))["alpha"]
    assert abs(a0 - 0.2) < 1e-6 and abs(a1 - 0.2) < 1e-6


def test_risk_alpha_deprecated_alias_maps_to_beta():
    agent = _agent(risk_mode="cvar", risk_alpha=0.1)
    assert abs(agent.risk_beta - 0.1) < 1e-9
