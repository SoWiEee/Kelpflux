"""Risk distortion estimators for distributional RL (Ma et al. DSAC, arXiv:2004.14547 §4.2).

Given a sampled quantile representation of a return distribution Z — quantile
values ``Z_{tau_i}`` at sampling fractions ``tau_i`` — these functions estimate a
risk-distorted value ``rho[Z]``. All estimators reduce over the **last axis**
(the quantile axis), so an input ``(B, A, N)`` with ``taus`` ``(B, A, N)`` yields
``(B, A)``.

Distortion modes
----------------
``mean`` : risk-neutral expectation E[Z].
``cvar`` : Conditional Value at Risk. g(tau)=min{tau/beta, 1}; the distorted
           expectation is the average of the lower ``beta`` tail.
``wang`` : Wang transform. g(tau)=Phi(Phi^{-1}(tau)+beta); beta>0 risk-averse.
``cpw``  : Cumulative Probability Weighting. g(tau)=tau^beta/(tau^beta+(1-tau)^beta)^{1/beta}.
``msd``  : Mean-Semideviation. rho=E[Z] - beta*sqrt(E[(Z-E[Z])_-^2]). Not a
           distortion measure; penalises downside deviation only.

For ``cvar``/``wang``/``cpw`` we use the derivative-weighting estimator
(Ma et al. §4.2, second form): ``rho_hat = sum_i (tau_{i+1}-tau_i) g'(tau_i) Z_{tau_i}``.
With randomly sampled tau this is a normalised weighted mean with weights
``g'(tau_i)``.
"""
from __future__ import annotations

import math

import torch

RISK_MODES: tuple[str, ...] = ("mean", "cvar", "wang", "cpw", "msd")

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_EPS = 1e-6


def _normal_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) / _SQRT_2PI


def _cpw_g(taus: torch.Tensor, beta: float) -> torch.Tensor:
    """CPW distortion function g(tau)=tau^b/(tau^b+(1-tau)^b)^{1/b}. g(0)=0, g(1)=1."""
    t = taus.clamp(0.0, 1.0)
    tb = t.pow(beta)
    ob = (1.0 - t).pow(beta)
    denom = (tb + ob).pow(1.0 / beta)
    return tb / denom.clamp(min=_EPS)


def _weights(taus: torch.Tensor, mode: str, beta: float) -> torch.Tensor:
    """Non-negative distortion weights g'(tau) for the weighting estimator."""
    if mode == "cvar":
        # g'(tau) = 1/beta for tau <= beta, else 0  (constant cancels on normalise)
        return (taus <= beta).to(taus.dtype)
    if mode == "wang":
        if beta == 0.0:
            return torch.ones_like(taus)
        # g(tau)=Phi(Phi^{-1}(tau)+beta) => g'(tau)=phi(Phi^{-1}(tau)+beta)/phi(Phi^{-1}(tau))
        z = torch.special.ndtri(taus.clamp(_EPS, 1.0 - _EPS))
        return _normal_pdf(z + beta) / _normal_pdf(z).clamp(min=_EPS)
    if mode == "cpw":
        with torch.enable_grad():
            t = taus.detach().clamp(_EPS, 1.0 - _EPS).requires_grad_(True)
            g = _cpw_g(t, beta).sum()
            (grad,) = torch.autograd.grad(g, t)
        return grad.clamp(min=0.0).to(taus.dtype)
    raise ValueError(f"no weights for mode={mode!r}")


def distorted_values(
    quantiles: torch.Tensor,
    taus: torch.Tensor,
    mode: str = "mean",
    beta: float = 0.25,
) -> torch.Tensor:
    """Risk-distorted value rho[Z], reducing over the last (quantile) axis.

    Parameters
    ----------
    quantiles : (..., N) quantile values Z_{tau_i}.
    taus      : (..., N) sampling fractions in (0, 1), broadcastable to quantiles.
    mode      : one of RISK_MODES.
    beta      : risk parameter (CVaR tail mass, Wang/CPW shape, MSD weight).
    """
    if mode not in RISK_MODES:
        raise ValueError(f"unknown risk mode {mode!r}; expected one of {RISK_MODES}")
    taus = taus.to(quantiles.dtype)

    if mode == "mean":
        return quantiles.mean(dim=-1)

    if mode == "msd":
        mean = quantiles.mean(dim=-1, keepdim=True)
        downside = (quantiles - mean).clamp(max=0.0)
        semivar = (downside * downside).mean(dim=-1)
        return mean.squeeze(-1) - beta * torch.sqrt(semivar.clamp(min=0.0))

    w = _weights(taus, mode, beta)
    total = w.sum(dim=-1).clamp(min=_EPS)
    return (w * quantiles).sum(dim=-1) / total
