"""Measure the runtime predictor's real log-residual σ (eval Item 0 / R1).

The sim's stochastic-execution noise (sim/gym_env.py) injects a mean-preserving
lognormal multiplier on the realized runtime: actual = predicted · exp(σZ − σ²/2),
so log(actual/predicted) ~ Normal(−σ²/2, σ²) and **σ is exactly the std of the
predictor's log-runtime residual**. To stop the σ-sweep from being a tautology
("inject noise → noise-robust method wins"), σ must come from the predictor's
*real* error, not a hand-picked value.

This trains the production LightGBM predictor (same features, same time-honest
80/20 split, same hyper-params as services/runtime_predictor/train.py) on a real
trace and reports the held-out residual std = the σ to feed the sim. It also
reports the residual shape (Gaussian vs heavy-tailed) so we know whether a
lognormal model under-states the tail.

Usage::
    PYTHONPATH=. .venv-m11/bin/python eval/scripts/measure_predictor_sigma.py \
        --trace sim/data/philly_subsample.json
"""
from __future__ import annotations

import argparse
import json
import sys

import lightgbm as lgb
import numpy as np

from services.runtime_predictor.features import FEATURE_COLS, build_training_frame
from services.runtime_predictor.train import _split_time_honest


def _fit_predict_residuals(trace_path: str, *, holdout_frac: float, seed: int):
    """Return the held-out log-residual r = log1p(actual) − pred (production config)."""
    with open(trace_path) as fh:
        jobs = json.load(fh)
    jobs = sorted(jobs, key=lambda j: j["submit_ts"])
    df = build_training_frame(jobs)
    train_df, test_df, _ = _split_time_honest(df, jobs, holdout_frac)

    model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=200, num_leaves=31,
        learning_rate=0.05, random_state=seed, n_jobs=-1, verbose=-1,
    )
    model.fit(train_df[FEATURE_COLS], train_df["log_runtime"])
    pred = model.predict(test_df[FEATURE_COLS])
    # residual in log space; sign chosen so + = actual ran LONGER than predicted
    return np.asarray(test_df["log_runtime"]) - np.asarray(pred)


def _bootstrap_std_ci(r: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    stds = [r[rng.integers(0, len(r), len(r))].std(ddof=1) for _ in range(n_boot)]
    return float(np.percentile(stds, 2.5)), float(np.percentile(stds, 97.5))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", default="sim/data/philly_subsample.json")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    r = _fit_predict_residuals(args.trace, holdout_frac=args.holdout_frac, seed=args.seed)
    std = float(r.std(ddof=1))
    mad_sigma = float(1.4826 * np.median(np.abs(r - np.median(r))))  # robust σ
    lo, hi = _bootstrap_std_ci(r)
    # excess kurtosis (0 = Gaussian; >0 = heavy-tailed)
    z = (r - r.mean()) / r.std()
    excess_kurt = float((z ** 4).mean() - 3.0)

    print(f"\n  trace                : {args.trace}  (n_test={len(r)})")
    print(f"  residual mean (bias) : {r.mean():+.3f}  (log space)")
    print(f"  residual std  = σ    : {std:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  robust σ (MAD-based) : {mad_sigma:.3f}")
    print(f"  excess kurtosis      : {excess_kurt:+.2f}  ({'heavy-tailed' if excess_kurt > 1 else 'near-Gaussian'})")
    print("  residual |quantiles| :", {
        q: round(float(np.percentile(np.abs(r), q)), 2) for q in (50, 90, 95, 99)
    })
    print(f"\n  → recommended sim σ  : {std:.2f}  (bulk); MAD σ {mad_sigma:.2f} if "
          f"down-weighting tail outliers")
    # what fold-change does this σ imply at ±1σ?
    print(f"  → at σ={std:.2f}, a +1σ job runs ×{np.exp(std):.2f}, −1σ ×{np.exp(-std):.2f} "
          f"vs predicted\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
