#!/usr/bin/env python3
"""B-vs-static REALIZED dispatch-order mechanism analysis (paper §5.8 「待補」 fill-in).

The §5.8 mechanism paragraph originally read the ordering off a STATIC precompute
(`analyze_rl_order.py` + `precompute_schedule`), which explains the *static* probe but
not why **Option B wins at shallow load while static ties**. This script instead reads
the ACTUAL realized dispatch order from real runs — per-job records
(jid, cls, rt, arrival, submit, start, end) written by `scontrol_ab --order-log` — ranks
each seed's jobs by their real Start time, and compares the realized order to the job
stream for BOTH actuations:

  * ρ(rank, runtime)  — >0 = short-first (SJF-like); the greedy pattern that starves long jobs
  * ρ(rank, arrival)  — ≈1 = FCFS-like (submit order preserved)
  * long-20% rank-pct — mean dispatch-rank-percentile of the longest-20% jobs (0=first,1=last);
                        LOWER = long jobs run earlier = tail protection (pure SJF ≈ 0.9)

The hypothesis being tested: at ov2, STATIC's realized order is more SJF-like / starves the
long tail (Slurm backfill fills idle gaps short-first, overriding the frozen priorities),
whereas B's continuous re-assertion yields a fairer, tail-protected realized order — locating
the ov2 win in ACTUATION continuity rather than the policy's nominal ordering.

Reads a dir laid out as <root>/ov<N>/<label>_s<seed>.jsonl where label ends in "_reorder"
for Option B and is the bare arm name for static (priority). No cluster/GPU.

  PYTHONPATH=. .venv-m11/bin/python -m eval.scripts.analyze_order_realized runs/order_mech_<stamp>
"""
from __future__ import annotations
import glob, json, os, re, sys
import numpy as np
from scipy.stats import spearmanr

LONG_FRAC = 0.20


def load_seed(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if len(recs) < 5:
        return None
    recs.sort(key=lambda r: r["start"])               # realized dispatch order
    n = len(recs)
    rank = {id(r): i for i, r in enumerate(recs)}      # 0 = dispatched first
    rt = np.array([r["rt"] for r in recs], float)
    arr = np.array([r["arrival"] for r in recs], float)
    rk = np.array([rank[id(r)] for r in recs], float)
    rho_rt = spearmanr(rk, rt).correlation
    rho_arr = spearmanr(rk, arr).correlation
    # long-20% by true runtime → mean rank percentile (0=first .. 1=last)
    order_by_rt = sorted(range(n), key=lambda i: rt[i], reverse=True)
    n_long = max(1, int(round(LONG_FRAC * n)))
    long_idx = set(order_by_rt[:n_long])
    rank_pct = rk / max(1, n - 1)
    long_pct = float(np.mean([rank_pct[i] for i in range(n) if i in long_idx]))
    return dict(rho_rt=rho_rt, rho_arr=rho_arr, long_pct=long_pct, n=n)


def actuation_of(label):
    return "B (reorder)" if label.endswith("_reorder") else "static (priority)"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob("runs/order_mech_*"), key=os.path.getmtime)
    ovdirs = sorted(glob.glob(os.path.join(root, "ov*")),
                    key=lambda p: float(re.search(r"ov([0-9.]+)", os.path.basename(p)).group(1)))
    lines = ["# Realized dispatch-order mechanism — Option B vs static (§5.8 待補 fill-in)\n",
             f"root={root}",
             "ρ_runtime>0 = short-first (SJF-like, starves long tail); ρ_arrival≈1 = FCFS-like.",
             "long20% = mean dispatch rank-pct of longest-20% jobs (0=first,1=last); "
             "LOWER = tail protection. Pure SJF ≈ 0.90.\n"]
    pure_sjf_long = 1.0 - LONG_FRAC / 2  # ≈0.90 reference for the longest 20%
    for ovd in ovdirs:
        ov = re.search(r"ov([0-9.]+)", os.path.basename(ovd)).group(1)
        lines.append(f"\n## oversub={ov}\n")
        lines.append("| 致動 | ρ(rank,runtime) | ρ(rank,arrival) | long20% rank-pct | n seeds |")
        lines.append("|---|--:|--:|--:|--:|")
        # group seed files by actuation label
        by_act = {}
        for f in glob.glob(os.path.join(ovd, "*.jsonl")):
            label = re.sub(r"_s\d+$", "", os.path.basename(f)[:-6])  # strip _sNN.jsonl
            base = re.sub(r"_(reorder|online|priority)$", "", label)
            if base in ("backfill", "fcfs"):
                continue  # baselines have no learned actuation → not part of B-vs-static order comparison
            by_act.setdefault(actuation_of(label), []).append(f)
        for act in ["static (priority)", "B (reorder)"]:
            files = by_act.get(act, [])
            accs = [load_seed(f) for f in sorted(files)]
            accs = [a for a in accs if a]
            if not accs:
                continue
            def m(k): return np.mean([a[k] for a in accs])
            def sd(k): return np.std([a[k] for a in accs])
            lines.append(f"| {act} | {m('rho_rt'):+.2f} ± {sd('rho_rt'):.2f} | "
                         f"{m('rho_arr'):+.2f} ± {sd('rho_arr'):.2f} | "
                         f"{m('long_pct'):.2f} ± {sd('long_pct'):.2f} | {len(accs)} |")
        lines.append(f"\n_(pure-SJF long20% reference ≈ {pure_sjf_long:.2f})_")
    md = "\n".join(lines) + "\n"
    outp = os.path.join(root, "ORDER_MECH_SUMMARY.md")
    open(outp, "w").write(md)
    print(md)
    print(f"[out] {outp}")


if __name__ == "__main__":
    main()
