"""Aggregate several baseline passes (different seeds AND different arm orders)
into per-seed tables + a cross-seed mean±std table, mirroring eval §4.4.

Each pass ran the 4 arms (score / multifactor / packing / fcfs) in a DIFFERENT
order so that run-order drift does not line up with any arm across passes. This
script reports, per seed: the tail panel + paired ΔJCT%/Δp99%/ΔCVaR% vs score;
then aggregates each arm's (mean, p95, p99, CVaR, ΔJCT%) across seeds as mean±std.

It also prints each arm's run-POSITION per pass, so the drift diagnostic is
explicit: if an arm wins regardless of whether it ran 1st or 4th, the win is not
a run-order artifact.

Usage:
    python -m eval.scripts.aggregate_baseline \
        42:runs/baseline_sweep_A:score,multifactor,packing,fcfs \
        43:runs/baseline_passB:fcfs,packing,multifactor,score \
        44:runs/baseline_passC:packing,score,fcfs,multifactor \
        --out runs/baseline_confirm
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

from eval.scripts.run_heavytail_ab import build_report

ARMS = ("score", "multifactor", "packing", "fcfs")
METRICS = ("mean", "p95", "p99", "cvar")


def _load(dir_: str, arm: str) -> list[dict]:
    recs = json.loads((Path(dir_) / arm / "records.json").read_text())
    for r in recs:
        r["arm"] = arm
    return recs


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("passes", nargs="+",
                   help="seed:dir:arm1,arm2,arm3,arm4 (order = run order in that pass)")
    p.add_argument("--family", default="philly")
    p.add_argument("--out", default="runs/baseline_confirm")
    args = p.parse_args(argv)

    per_seed: dict[int, dict] = {}
    positions: dict[str, dict[int, int]] = {a: {} for a in ARMS}
    for spec in args.passes:
        seed_s, dir_, order_s = spec.split(":", 2)
        seed = int(seed_s)
        order = order_s.split(",")
        for pos, arm in enumerate(order, 1):
            positions[arm][seed] = pos
        recs_by_arm = {a: _load(dir_, a) for a in ARMS}
        rep = build_report(recs_by_arm, sigma=1.0, family=args.family)
        per_seed[seed] = rep

    lines: list[str] = ["# Baseline confirm — drift-counterbalanced, multi-seed", ""]

    # run-position matrix (drift diagnostic)
    seeds = sorted(per_seed)
    lines.append("## Run position per pass (drift diagnostic)")
    lines.append("")
    lines.append("| arm | " + " | ".join(f"seed {s}" for s in seeds) + " |")
    lines.append("|---|" + "|".join("--:" for _ in seeds) + "|")
    for a in ARMS:
        lines.append(f"| {a} | " + " | ".join(str(positions[a].get(s, "-")) for s in seeds) + " |")
    lines.append("")

    # per-seed panels + paired deltas
    for s in seeds:
        rep = per_seed[s]
        lines.append(f"## seed {s}")
        lines.append("")
        lines.append("| arm | mean | p95 | p99 | CVaR | ΔJCT% vs score | p |")
        lines.append("|---|--:|--:|--:|--:|--:|--:|")
        for a in ARMS:
            pan = rep["panels"][a]
            d = rep["paired_vs_score"].get(a)
            dj = f"{d['djct_pct']:+.1f}" if d else "—"
            pp = f"{d.get('ttest_p', float('nan')):.2g}" if d else "—"
            lines.append(f"| {a} | {pan['mean']:.1f} | {pan['p95']:.1f} | {pan['p99']:.1f} "
                         f"| {pan['cvar']:.1f} | {dj} | {pp} |")
        lines.append("")

    # cross-seed aggregate (mean ± std)
    def agg(vals):
        return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)

    lines.append("## Cross-seed aggregate (mean ± std)")
    lines.append("")
    lines.append("| arm | mean JCT | p95 | p99 | CVaR | ΔJCT% vs score |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    for a in ARMS:
        cells = []
        for m in METRICS:
            mu, sd = agg([per_seed[s]["panels"][a][m] for s in seeds])
            cells.append(f"{mu:.1f}±{sd:.1f}")
        if a == "score":
            dj = "—"
        else:
            djs = [per_seed[s]["paired_vs_score"][a]["djct_pct"] for s in seeds
                   if a in per_seed[s]["paired_vs_score"]]
            mu, sd = agg(djs)
            dj = f"{mu:+.1f}±{sd:.1f}"
        lines.append(f"| {a} | " + " | ".join(cells) + f" | {dj} |")
    lines.append("")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    (out / "SUMMARY.md").write_text(text)
    print(text)
    print(f"[aggregate] wrote {out}/SUMMARY.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
