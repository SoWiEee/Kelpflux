"""Combine per-arm baseline live-A/B runs into one paired multi-arm report.

Each baseline arm is run separately via ``run_heavytail_ab`` (one Slurm-native
policy at a time, since switching policy needs a controller restart), so every
arm's ``records.json`` labels its jobs ``arm="score"``. This relabels each run
with its real arm name, merges them, and reuses the harness's pure
``build_report`` / ``render_summary`` so the output is a single panel + paired
ΔJCT%/Δp99%/ΔCVaR% vs the score arm — identical in shape to §4.3.1's table.

CRN pairing: all arms used the same ``--seed`` so ``gen_workload`` produced the
same job_ids; ``build_report`` pairs on (round, job_id) across arms.

Usage:
    python -m eval.scripts.combine_baseline \
        score=runs/baseline_score_xxx multifactor=runs/baseline_multifactor_xxx \
        fcfs=runs/baseline_fcfs_xxx packing=runs/baseline_packing_xxx \
        --out runs/baseline_combined
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.scripts.run_heavytail_ab import build_report, render_summary


def load_arm_records(run_dir: str, arm: str) -> list[dict]:
    recs = json.loads((Path(run_dir) / "records.json").read_text())
    for r in recs:
        r["arm"] = arm                       # relabel from the harness's "score"
    return recs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Combine per-arm baseline runs into one paired report")
    p.add_argument("arms", nargs="+", help="arm=run_dir pairs, e.g. score=runs/baseline_score_x")
    p.add_argument("--family", default="philly")
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--out", default="runs/baseline_combined")
    args = p.parse_args(argv)

    records_by_arm: dict[str, list[dict]] = {}
    for spec in args.arms:
        if "=" not in spec:
            print(f"error: expected arm=run_dir, got {spec!r}", file=sys.stderr)
            return 2
        arm, run_dir = spec.split("=", 1)
        records_by_arm[arm] = load_arm_records(run_dir, arm)
        print(f"[combine] {arm}: {len(records_by_arm[arm])} records from {run_dir}")

    if "score" not in records_by_arm:
        print("warning: no 'score' arm → no paired deltas (panels only)", file=sys.stderr)

    report = build_report(records_by_arm, sigma=args.sigma, family=args.family, beta=args.beta)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reports.json").write_text(json.dumps([report], indent=2))
    summary = render_summary([report])
    (out / "SUMMARY.md").write_text(summary)
    print(summary)
    print(f"[combine] wrote {out}/SUMMARY.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
