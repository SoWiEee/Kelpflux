"""Aggregate P0 review sweeps (run_review_p0.sh) into paper-ready tables.

#3 CVaR multi-seed ablation: mean±std of ΔJCT% vs score across train seeds,
   for SAC / RDSAC-mean / RDSAC-cvar at σ=1.0 (breaks the single-seed caveat).
#1 Scale crossover: ΔJCT% (RDSAC-cvar & SAC vs score) across 1×1/2×1/2×2 —
   does the learned edge grow with cluster scale?

ΔJCT% sign: positive = learned policy FASTER than score (lower JCT) = better.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RUNS = Path("runs")
CVAR_SEEDS = [42, 43, 44]
SCALES = [("1x1", "1×1"), ("2x1", "2×1"), ("2x2", "2×2")]
FAMS = ["philly", "ali"]


def _load(d: Path) -> list[dict]:
    f = d / "sweep.json"
    return json.loads(f.read_text()) if f.exists() else []


def _cvar_table() -> list[str]:
    # rows[(family, model)] = list of (delta_pct, p99_h) across seeds
    rows: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for s in CVAR_SEEDS:
        for r in _load(RUNS / f"review_cvar_s{s}"):
            rows.setdefault((r["family"], r["model"]), []).append(
                (r.get("delta_pct", float("nan")), r.get("p99_h", float("nan")))
            )
    out = [
        "## #3  CVaR multi-seed ablation (σ=1.0, fixed-α=0.05, "
        f"train seeds {CVAR_SEEDS})",
        "",
        "ΔJCT% vs score (mean±std across train seeds; + = beats score). "
        "p99 = per-job tail JCT (h).",
        "",
        "| family | model | ΔJCT% (mean±std) | p99 (h, mean) | n seeds |",
        "|---|---|---:|---:|---:|",
    ]
    for fam in FAMS:
        for model in ["sac", "rdsac-mean", "rdsac-cvar"]:
            vals = rows.get((fam, model))
            if not vals:
                continue
            d = np.array([v[0] for v in vals], float)
            p = np.array([v[1] for v in vals], float)
            d = d[np.isfinite(d)]
            out.append(
                f"| {fam} | {model} | "
                f"{np.mean(d):+.1f}±{np.std(d, ddof=1) if d.size > 1 else 0:.1f} | "
                f"{np.nanmean(p):.2f} | {d.size} |"
            )
    out += [
        "",
        "**Read:** if SAC→RDSAC-mean is a large positive jump, the gain is the "
        "*distributional critic*; RDSAC-mean→RDSAC-cvar isolates the *risk "
        "distortion* (tail-specific). Multi-seed std shows whether the mean-vs-"
        "cvar ordering is real or single-seed noise.",
        "",
    ]
    return out


def _scale_table() -> list[str]:
    out = [
        "## #1  Scale crossover (σ=1.0, fixed-α=0.05, train seed 42)",
        "",
        "ΔJCT% vs score (+ = learned beats score). Does the edge grow with scale?",
        "",
        "| scale | family | SAC ΔJCT% | RDSAC-cvar ΔJCT% |",
        "|---|---|---:|---:|",
    ]
    for key, label in SCALES:
        recs = _load(RUNS / f"review_scale_{key}")
        by = {(r["family"], r["model"]): r.get("delta_pct", float("nan")) for r in recs}
        for fam in FAMS:
            sac = by.get((fam, "sac"), float("nan"))
            cvar = by.get((fam, "rdsac-cvar"), float("nan"))
            out.append(f"| {label} | {fam} | {sac:+.1f} | {cvar:+.1f} |")
    out += [
        "",
        "**Read:** the central paper claim ('value requires scale') predicts "
        "RDSAC-cvar ΔJCT% should trend upward (less negative → positive) from "
        "1×1 → 2×2. A flat/negative trend would *falsify* the claim and must be "
        "reported honestly.",
        "",
    ]
    return out


def main(argv) -> int:
    stamp = argv[1] if len(argv) > 1 else "manual"
    lines = [f"# P0 review results — {stamp}", ""] + _cvar_table() + _scale_table()
    dst = RUNS / f"review_p0_{stamp}_TABLES.md"
    dst.write_text("\n".join(lines))
    print(f"[agg] wrote {dst}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
