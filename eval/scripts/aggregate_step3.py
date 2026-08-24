#!/usr/bin/env python3
"""Aggregate the Step 3 payoff eval (scontrol held-job + aging vs backfill).

Reads runs/step3_<stamp>/{arm}_s{seed}.json ({arm,seed,jct[],wait[]}) and, for
each learned arm, reports per-seed PAIRED deltas vs the backfill baseline on the
SAME seed (mean JCT, P95, P99 — the tail is the whole point). Paired because the
job stream is seed-locked, so seed-to-seed variance cancels.

  python -m eval.scripts.aggregate_step3 runs/step3_20260824-XXXXXX
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

ORDER = ["fcfs", "backfill", "sac", "rdsac_mean", "rdsac_cvar", "rlpd_cvar"]


def load(d):
    out = {}  # arm -> seed -> {mean,p95,p99,n}
    for f in glob.glob(os.path.join(d, "*.json")):
        try:
            o = json.load(open(f))
        except Exception:
            continue
        jct = np.array(o.get("jct", []), dtype=float)
        if not len(jct):
            continue
        arm, seed = o["arm"], int(o["seed"])
        # a scontrol run's checkpoint arm name comes from the filename, not "arm"
        # (worker only knows scontrol/backfill); recover it from the file stem.
        stem = os.path.basename(f).rsplit("_s", 1)[0]
        arm = stem if stem in ORDER else arm
        out.setdefault(arm, {})[seed] = {
            "mean": float(jct.mean()), "p95": float(np.percentile(jct, 95)),
            "p99": float(np.percentile(jct, 99)), "max": float(jct.max()), "n": len(jct)}
    return out


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return (m - 1.96 * se, m + 1.96 * se)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob("runs/step3*"), key=os.path.getmtime)
    data = load(d)
    if "backfill" not in data:
        print("no backfill baseline found in", d); return 1
    base = data["backfill"]
    print(f"\n=== Step 3: scontrol held-job + aging vs backfill — {d} ===")
    print(f"backfill seeds: {sorted(base)} (n_jobs completed avg "
          f"{np.mean([v['n'] for v in base.values()]):.0f})\n")
    hdr = f"{'arm':<12} {'seeds':>5} {'ΔmeanJCT%':>18} {'ΔP95%':>9} {'ΔP99%':>9} {'P99<bf':>7}"
    print(hdr); print("-" * len(hdr))
    for arm in ORDER:
        if arm == "backfill" or arm not in data:
            continue
        arm_d = data[arm]
        seeds = sorted(set(arm_d) & set(base))
        if not seeds:
            continue
        dmean, dp95, dp99, win = [], [], [], 0
        for s in seeds:
            b, a = base[s], arm_d[s]
            dmean.append((a["mean"] - b["mean"]) / b["mean"] * 100)
            dp95.append((a["p95"] - b["p95"]) / b["p95"] * 100)
            dp99.append((a["p99"] - b["p99"]) / b["p99"] * 100)
            win += a["p99"] < b["p99"]
        lo, hi = ci95(dmean)
        print(f"{arm:<12} {len(seeds):>5} "
              f"{np.mean(dmean):>7.1f} [{lo:>6.1f},{hi:>6.1f}] "
              f"{np.mean(dp95):>8.1f} {np.mean(dp99):>8.1f} {win:>3}/{len(seeds)}")
    print("\n(neg = better than backfill; ΔP99 is the tail; P99<bf = seeds whose "
          "tail beat backfill)")
    # paired Wilcoxon on ΔmeanJCT if scipy available
    try:
        from scipy.stats import wilcoxon
        print("\nWilcoxon signed-rank (paired, ΔmeanJCT vs 0):")
        for arm in ORDER:
            if arm == "backfill" or arm not in data:
                continue
            seeds = sorted(set(data[arm]) & set(base))
            if len(seeds) < 6:
                continue
            dd = [(data[arm][s]["mean"] - base[s]["mean"]) / base[s]["mean"] * 100 for s in seeds]
            try:
                st, p = wilcoxon(dd)
                print(f"  {arm:<12} n={len(seeds)}  W={st:.0f}  p={p:.3f}")
            except Exception as e:
                print(f"  {arm:<12} wilcoxon err {e}")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
