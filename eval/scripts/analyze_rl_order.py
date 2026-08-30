#!/usr/bin/env python3
"""What did the RL scheduler LEARN? — dispatch-order analysis (paper §5.8 companion).

The §5.8 result is behavioural (learned ordering beats Backfill on mean+tail) but a
black box. This script opens it: it re-runs the exact arrival-aware precompute drain
(``scontrol_ab.precompute_schedule`` → the same rolling top-16 /act interface the live
policy uses) for each learned checkpoint, reads off the dispatch RANK it imposes on
every job, and compares that ordering to two reference policies on the identical job
stream:

  * FCFS  — rank by arrival time (what Slurm's submit order would give)
  * SJF   — rank by *true* runtime, shortest first (the greedy mean-optimiser; this is
            also, structurally, what Backfill's short-job-first reorder approximates —
            and what starves the long jobs into the tail, §5.8)

Two questions, both answered per (arm, oversub) over the 10 workload seeds:

  1. Is the learned order SJF-like?  → Spearman ρ(rank, runtime) and ρ(rank, arrival).
     Positive ρ_runtime = short-first; ρ_arrival≈0 = it ignores submit order (not FCFS).
  2. Does it PROTECT the tail (vs pure SJF)?  → for the longest-20% jobs, compare their
     mean rank-percentile under RL vs under pure SJF. RANK-PCT_RL < RANK-PCT_SJF means
     the long jobs run EARLIER than greedy SJF would place them → bounded delay = the
     tail protection that lets RL beat Backfill's P99 while keeping SJF-like mean.

Needs a local 168-d serve on :8003 (any ckpt; we /reload per arm). No cluster/GPU —
pure policy inference + in-memory drain. Output: runs/rl_order_analysis/.

  PYTHONPATH=. .venv-m11/bin/python eval/scripts/analyze_rl_order.py
  OVERSUBS="6" SEEDS="42 43" ARMS="rdsac_cvar" python eval/scripts/analyze_rl_order.py
"""
from __future__ import annotations
import os
import numpy as np
from scipy.stats import spearmanr

from eval.scripts.scontrol_ab import gen_jobs, precompute_schedule, reload_serve, SERVE, NODES

CK = os.environ.get("CK", "runs/ckpts_aimix16_fair")
ARMS = os.environ.get("ARMS", "sac rdsac_mean rdsac_cvar rlpd_cvar").split()
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42 43 44 45 46 47 48 49 50 51").split()]
OVERSUBS = [float(x) for x in os.environ.get("OVERSUBS", "2 6").split()]
N_JOBS = int(os.environ.get("N_JOBS", "150"))
TARGET_MAX = float(os.environ.get("TARGET_MAX", "20"))
OUT = os.environ.get("OUT", "runs/rl_order_analysis")
LONG_FRAC = 0.20   # "long jobs" = the longest 20% by true runtime


def rank_pct(order_rank: dict, jids) -> dict:
    """Map {jid: integer dispatch rank} → {jid: rank percentile in [0,1]} (0 = first)."""
    n = len(jids)
    return {jid: order_rank[jid] / max(1, n - 1) for jid in jids}


def analyze_one(arm: str, seed: int, oversub: float):
    ck = f"{CK}/{arm}_s{seed}.pt"
    if not os.path.exists(ck):
        return None
    reload_serve(ck)
    jobs = gen_jobs(N_JOBS, seed, TARGET_MAX, arrival_mode="poisson", oversub=oversub)
    jids = [j["jid"] for j in jobs]
    rt = {j["jid"]: float(j["rt"]) for j in jobs}
    arr = {j["jid"]: float(j["arrival"]) for j in jobs}
    cls = {j["jid"]: j["cls"] for j in jobs}

    rl_rank, node_of = precompute_schedule(jobs)
    # reference orderings on the SAME stream
    sjf_rank = {jid: r for r, jid in enumerate(sorted(jids, key=lambda j: rt[j]))}
    # per-job vectors aligned by jids
    rl = np.array([rl_rank[j] for j in jids], float)
    rtv = np.array([rt[j] for j in jids], float)
    arrv = np.array([arr[j] for j in jids], float)

    rho_rt = spearmanr(rl, rtv).correlation
    rho_arr = spearmanr(rl, arrv).correlation

    # tail protection: longest-20% jobs' mean rank-pct under RL vs pure SJF
    n_long = max(1, int(round(LONG_FRAC * len(jids))))
    long_jids = sorted(jids, key=lambda j: rt[j], reverse=True)[:n_long]
    rl_pct = rank_pct(rl_rank, jids)
    sjf_pct = rank_pct(sjf_rank, jids)
    rl_long = np.mean([rl_pct[j] for j in long_jids])
    sjf_long = np.mean([sjf_pct[j] for j in long_jids])

    # node routing: fraction of long jobs RL sends to the fast card (index 0 = 4070)
    fast = NODES[0]
    long_on_fast = np.mean([1.0 if node_of.get(j) == fast else 0.0 for j in long_jids])

    # per-class mean rank-pct
    classes = sorted(set(cls.values()))
    cls_pct = {c: float(np.mean([rl_pct[j] for j in jids if cls[j] == c])) for c in classes}

    # per-job (runtime percentile, RL rank percentile) for the scatter figure
    rt_order = {jid: r for r, jid in enumerate(sorted(jids, key=lambda j: rt[j]))}
    rtpct = {j: rt_order[j] / max(1, len(jids) - 1) for j in jids}
    job_xy = [(rtpct[j], rl_pct[j]) for j in jids]

    return dict(rho_rt=rho_rt, rho_arr=rho_arr, rl_long=rl_long, sjf_long=sjf_long,
                long_on_fast=long_on_fast, cls_pct=cls_pct, classes=classes, job_xy=job_xy)


def make_figure(pts_by_ov, out_path):
    """Scatter: runtime-percentile (x) vs RL dispatch rank-percentile (y), pooled over
    seeds for rdsac_cvar. The pure-SJF policy is the y=x diagonal; RL's shallow slope +
    the way the rightmost (longest) jobs sit BELOW the diagonal = SJF-tilt-with-tail-cap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[fig] matplotlib unavailable ({e}); skipping figure")
        return
    fig, axes = plt.subplots(1, len(pts_by_ov), figsize=(5.2 * len(pts_by_ov), 4.6), squeeze=False)
    for ax, (ov, pts) in zip(axes[0], sorted(pts_by_ov.items())):
        x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1.2, label="pure SJF (y=x)")
        ax.scatter(x, y, s=9, alpha=0.30, color="#2a7", edgecolors="none")
        # binned mean trend
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(x, bins) - 1
        bx = [(bins[i] + bins[i + 1]) / 2 for i in range(10)]
        by = [y[idx == i].mean() if np.any(idx == i) else np.nan for i in range(10)]
        ax.plot(bx, by, "-o", color="#134", lw=2, ms=4, label="RL mean trend")
        rho = spearmanr(x, y).correlation
        ax.set_title(f"oversub={ov:g}  (ρ={rho:+.2f})")
        ax.set_xlabel("job runtime percentile (0=shortest)")
        ax.set_ylabel("RL dispatch rank percentile (0=first)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("What RL learned: FCFS-flat at shallow load → SJF-tilt with a long-job cap at deep load "
                 "(rdsac_cvar, pooled seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"[fig] {out_path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = ["# RL dispatch-order analysis — what did the scheduler learn?\n",
             f"serve={SERVE}  CK={CK}  n_jobs={N_JOBS}  seeds={SEEDS}\n",
             "ρ_runtime>0 → short-job-first (SJF-like); ρ_arrival≈0 → not FCFS.",
             "long20%: mean rank-pct of the longest 20% jobs (0=runs first, 1=runs last).",
             "**RL<SJF on long20% = tail protection** (long jobs run earlier than greedy SJF).\n"]
    rows = []
    fig_pts = {}   # oversub -> pooled (rt_pct, rank_pct) for rdsac_cvar
    for ov in OVERSUBS:
        lines.append(f"\n## oversub={ov:g}\n")
        lines.append("| arm | ρ(rank,runtime) | ρ(rank,arrival) | long20% RL | long20% SJF | Δ(RL−SJF) | long→fast |")
        lines.append("|---|--:|--:|--:|--:|--:|--:|")
        for arm in ARMS:
            acc = [analyze_one(arm, s, ov) for s in SEEDS]
            acc = [a for a in acc if a]
            if not acc:
                continue
            if arm == "rdsac_cvar":
                fig_pts[ov] = [xy for a in acc for xy in a["job_xy"]]
            def m(k): return np.mean([a[k] for a in acc])
            def sd(k): return np.std([a[k] for a in acc])
            d_long = m("rl_long") - m("sjf_long")
            lines.append(
                f"| {arm} | {m('rho_rt'):+.2f}±{sd('rho_rt'):.2f} | "
                f"{m('rho_arr'):+.2f}±{sd('rho_arr'):.2f} | {m('rl_long'):.2f} | "
                f"{m('sjf_long'):.2f} | {d_long:+.2f} | {m('long_on_fast'):.0%} |")
            rows.append((ov, arm, m('rho_rt'), m('rho_arr'), m('rl_long'), m('sjf_long'), d_long, m('long_on_fast')))
            # per-class (averaged over seeds)
            classes = acc[0]["classes"]
            cls_means = {c: np.mean([a["cls_pct"][c] for a in acc]) for c in classes}
            lines.append("")
            lines.append("  per-class mean rank-pct: " +
                         ", ".join(f"{c}={cls_means[c]:.2f}" for c in classes))
    md = "\n".join(lines) + "\n"
    open(os.path.join(OUT, "SUMMARY.md"), "w").write(md)
    # CSV
    import csv
    with open(os.path.join(OUT, "order_stats.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["oversub", "arm", "rho_runtime", "rho_arrival", "long20_rl", "long20_sjf", "delta_rl_minus_sjf", "long_on_fast"])
        w.writerows(rows)
    if fig_pts:
        make_figure(fig_pts, os.path.join(OUT, "rl_order_scatter.png"))
    print(md)
    print(f"[out] {OUT}/SUMMARY.md  +  order_stats.csv")


if __name__ == "__main__":
    main()
