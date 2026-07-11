"""Tail-aware metrics + paired statistics for the heavy-tail live A/B (§4.4).

Why this module exists: score / SAC / RDSAC-cvar optimise *different* objectives.
score and vanilla SAC are mean-oriented; RDSAC-cvar is tail-oriented. So a single
mean-JCT number hides the comparison — at 1×1 the three tie on the mean (see
eval-writeup §4.1–4.3) and any RDSAC edge lives in the tail. The **discriminating
statistic between SAC and RDSAC is therefore the tail** (p99 JCT and CVaR), not the
mean. This module computes the metric panel from the spec (docs/live-ab-heavytail-spec.md
§4) plus paired deltas vs the score arm with a paired t-test and a bootstrap CI.

All functions are pure (no cluster / no I/O) so they unit-test without a live run.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - scipy is a hard dep in .venv-m11
    _scipy_stats = None


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")


def cvar(values: Sequence[float], beta: float = 0.25, tail: str = "high") -> float:
    """Conditional Value-at-Risk = mean of the worst ``beta`` fraction.

    For JCT, *worst = highest*, so default ``tail="high"``. This is exactly the
    quantity RDSAC-cvar's distortion optimises (mean of the worst tail mass), which
    is why it is the primary discriminator between RDSAC and SAC.
    """
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(beta * arr.size)))
    return float(arr[-k:].mean() if tail == "high" else arr[:k].mean())


def summarize(
    jcts: Sequence[float],
    true_runtimes: Optional[Sequence[float]] = None,
    *,
    beta: float = 0.25,
    slo_factor: float = 4.0,
) -> dict:
    """Metric panel for one (arm, round). mean/tail JCT + slowdown + CVaR + SLO.

    ``slo_viol`` = fraction of requests whose JCT exceeds its deadline
    ``slo_s = true_runtime × slo_factor`` (i.e. slowdown > slo_factor). This is the
    serving SLO-violation rate; only populated when ``true_runtimes`` is given."""
    j = np.asarray(jcts, dtype=float)
    out: dict = {
        "n": int(j.size),
        "mean": float(j.mean()) if j.size else float("nan"),
        "p50": _pct(j, 50),
        "p95": _pct(j, 95),
        "p99": _pct(j, 99),
        "max": float(j.max()) if j.size else float("nan"),
        "cvar": cvar(j, beta),
    }
    if true_runtimes is not None and j.size:
        t = np.maximum(np.asarray(true_runtimes, dtype=float), 1.0)
        slow = j / t
        out["slowdown_mean"] = float(slow.mean())
        out["slowdown_p99"] = _pct(slow, 99)
        out["slo_viol"] = float((slow > slo_factor).mean())
    return out


def _impr_pct(score_stat: float, model_stat: float) -> float:
    """(score - model)/score * 100. Positive = model better (lower JCT)."""
    if not np.isfinite(score_stat) or score_stat == 0:
        return float("nan")
    return (score_stat - model_stat) / score_stat * 100.0


def paired_delta(
    score: Sequence[float],
    model: Sequence[float],
    *,
    beta: float = 0.25,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired comparison of a model arm vs the score arm on the SAME job stream.

    Inputs must be aligned per job (same index = same (round, job_id) under CRN).
    Returns the headline deltas (ΔJCT%, Δp99%, ΔCVaR%), a paired t-test on per-job
    JCT, and a bootstrap 95% CI on mean ΔJCT%.
    """
    s = np.asarray(score, dtype=float)
    m = np.asarray(model, dtype=float)
    if s.shape != m.shape:
        raise ValueError(f"score/model must be aligned per job; got {s.shape} vs {m.shape}")

    res: dict = {
        "n": int(s.size),
        "djct_pct": _impr_pct(float(s.mean()) if s.size else float("nan"),
                              float(m.mean()) if m.size else float("nan")),
        "dp99_pct": _impr_pct(_pct(s, 99), _pct(m, 99)),
        "dcvar_pct": _impr_pct(cvar(s, beta), cvar(m, beta)),
    }

    if s.size > 1 and _scipy_stats is not None:
        t, p = _scipy_stats.ttest_rel(s, m)
        res["ttest_p"] = float(p)
        res["mean_diff"] = float((s - m).mean())  # >0 = model lower JCT

    if s.size > 0:
        rng = np.random.default_rng(seed)
        boots = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, s.size, s.size)
            ss = s[idx].mean()
            mm = m[idx].mean()
            boots[i] = (ss - mm) / ss * 100.0 if ss else float("nan")
        res["djct_pct_ci"] = [_pct(boots, 2.5), _pct(boots, 97.5)]

    return res
