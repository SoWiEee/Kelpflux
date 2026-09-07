"""Mean and CVaR estimators for a sampled return distribution."""
from __future__ import annotations

import torch

RISK_MODES: tuple[str, ...] = ("mean", "cvar")
_EPS = 1e-6


def distorted_values(
    quantiles: torch.Tensor,
    taus: torch.Tensor,
    mode: str = "mean",
    beta: float = 0.25,
) -> torch.Tensor:
    """Reduce quantile values over the last axis using mean or lower-tail CVaR."""
    if mode not in RISK_MODES:
        raise ValueError(f"unknown risk mode {mode!r}; expected one of {RISK_MODES}")
    if mode == "mean":
        return quantiles.mean(dim=-1)

    weights = (taus.to(quantiles.dtype) <= beta).to(quantiles.dtype)
    total = weights.sum(dim=-1).clamp(min=_EPS)
    return (weights * quantiles).sum(dim=-1) / total
