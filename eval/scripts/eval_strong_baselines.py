"""Review #4 — strengthen baselines: does the "flat strategy space" finding hold
against cited-SOTA analogs, not just home-grown heuristics?

Runs every heuristic — including the Kueue-style fair-share and Volcano-style
binpack orderings — through the discrete-event sim on the SLO-sensitive aiserve
workload, over N seeds, and reports mean JCT / SLO-violation / utilization.
Pure CPU; no training, no GPU.

    PYTHONPATH=. .venv-m11/bin/python eval/scripts/eval_strong_baselines.py \
        --seeds 8 --n-jobs 60 --n-nodes 2 --gpus-per-node 2
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from sim.loader import generate_by_family
from sim.runner import run

ARMS = ["fcfs", "multifactor", "score", "kueue-fairshare", "volcano-binpack"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return st.fmean(xs) if xs else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--n-jobs", type=int, default=60)
    p.add_argument("--n-nodes", type=int, default=2)
    p.add_argument("--gpus-per-node", type=int, default=2)
    p.add_argument("--family", default="aiserve")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    seeds = list(range(42, 42 + args.seeds))
    rows = []
    for arm in ARMS:
        jct, slo, util = [], [], []
        for s in seeds:
            jobs = generate_by_family(args.family, n_jobs=args.n_jobs, seed=s)
            total = args.n_nodes * args.gpus_per_node
            jobs = [j for j in jobs if j.gpu_count <= total]
            metrics, _ = run(jobs, n_nodes=args.n_nodes,
                             gpus_per_node=args.gpus_per_node, scheduler_name=arm)
            m = metrics.summary()
            jct.append(m.get("jct_mean"))
            slo.append(m.get("slo_violation_rate"))
            util.append(m.get("utilization"))
        rows.append((arm, _mean(jct), _mean(slo) * 100, _mean(util)))

    header = f"{'scheduler':18}{'JCT_mean(s)':>13}{'SLO_viol(%)':>13}{'util':>8}"
    lines = [f"# Strong-baseline sim comparison ({args.family}, {args.seeds} seeds, "
             f"{args.n_nodes}×{args.gpus_per_node})", "", "```", header]
    print(header)
    for arm, j, sl, u in rows:
        line = f"{arm:18}{j:>13.1f}{sl:>13.1f}{u:>8.2f}"
        print(line)
        lines.append(line)
    lines.append("```")
    lines += ["",
              "**Read:** if the Kueue/Volcano analogs land in the same band as the "
              "home-grown heuristics, the 'strategy space is flat / statistically "
              "equivalent at this scale' claim survives contact with cited-SOTA "
              "orderings, closing the 'only compared to your own heuristics' gap."]

    out = Path(args.out) if args.out else Path(
        f"runs/strong_baselines_{args.family}_{args.n_nodes}x{args.gpus_per_node}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"\n[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
