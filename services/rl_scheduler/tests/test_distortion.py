"""Tests for risk distortion estimators (Ma et al. DSAC §4.2).

Distorted expectation over a sampled quantile representation:
    rho[Z] = E_distorted[Z], estimated from quantile values Z_{tau_i} and the
    sampling fractions tau_i. Reduces over the last axis.
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")
import torch

from services.rl_scheduler.distortion import distorted_values, RISK_MODES


def _sorted_quantiles(n: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Midpoint quantile fractions and an increasing return F^{-1}(tau)=tau."""
    taus = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    quantiles = taus.clone()  # inverse-CDF of Uniform(0,1): F^{-1}(tau)=tau
    return quantiles, taus


def test_mean_mode_is_plain_mean():
    q, taus = _sorted_quantiles()
    out = distorted_values(q, taus, mode="mean")
    assert torch.allclose(out, q.mean(), atol=1e-6)


def test_cvar_beta_one_equals_mean():
    q, taus = _sorted_quantiles()
    cvar = distorted_values(q, taus, mode="cvar", beta=1.0)
    assert torch.allclose(cvar, q.mean(), atol=1e-6)


def test_cvar_averages_lower_tail():
    q, taus = _sorted_quantiles()
    # CVaR(0.25) of Uniform(0,1) return = mean of lowest 25% = ~0.125
    cvar = distorted_values(q, taus, mode="cvar", beta=0.25)
    assert torch.allclose(cvar, torch.tensor(0.125, dtype=torch.float64), atol=2e-3)


def test_cvar_monotone_in_beta_for_increasing_return():
    q, taus = _sorted_quantiles()
    vals = [distorted_values(q, taus, mode="cvar", beta=b).item()
            for b in (0.1, 0.25, 0.5, 1.0)]
    assert vals == sorted(vals)  # smaller beta = more pessimistic = smaller


def test_wang_beta_zero_equals_mean():
    q, taus = _sorted_quantiles()
    out = distorted_values(q, taus, mode="wang", beta=0.0)
    assert torch.allclose(out, q.mean(), atol=1e-3)


def test_wang_positive_beta_is_risk_averse():
    q, taus = _sorted_quantiles()
    out = distorted_values(q, taus, mode="wang", beta=0.75)
    assert out.item() < q.mean().item()


def test_cpw_endpoints_preserved():
    # CPW distortion g(tau) must satisfy g(0)=0, g(1)=1.
    from services.rl_scheduler.distortion import _cpw_g
    taus = torch.tensor([0.0, 1.0], dtype=torch.float64)
    g = _cpw_g(taus, beta=0.71)
    assert torch.allclose(g, torch.tensor([0.0, 1.0], dtype=torch.float64), atol=1e-6)


def test_msd_beta_zero_equals_mean():
    q, taus = _sorted_quantiles()
    out = distorted_values(q, taus, mode="msd", beta=0.0)
    assert torch.allclose(out, q.mean(), atol=1e-6)


def test_msd_penalizes_downside():
    q, taus = _sorted_quantiles()
    out = distorted_values(q, taus, mode="msd", beta=1.0)
    assert out.item() < q.mean().item()


def test_batched_reduces_last_axis():
    q, taus = _sorted_quantiles(128)
    batch_q = q.expand(4, 3, 128).contiguous()       # (B, A, N)
    batch_taus = taus.expand(4, 3, 128).contiguous()
    out = distorted_values(batch_q, batch_taus, mode="cvar", beta=0.5)
    assert out.shape == (4, 3)


def test_all_modes_registered_and_finite():
    q, taus = _sorted_quantiles()
    for mode in RISK_MODES:
        out = distorted_values(q, taus, mode=mode, beta=0.5)
        assert torch.isfinite(out).all(), mode


def test_unknown_mode_raises():
    q, taus = _sorted_quantiles()
    with pytest.raises(ValueError):
        distorted_values(q, taus, mode="nope", beta=0.5)
