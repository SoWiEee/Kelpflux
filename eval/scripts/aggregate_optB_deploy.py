#!/usr/bin/env python3
"""Aggregate the Option B DEPLOYMENT campaign into the paper's table-6-B column format.

Reads runs/step3prio_<stamp>/ov<N>/{arm}_s{seed}.json (arm stems carry the actuation
suffix, e.g. sac_reorder, rdsac_cvar_reorder — stripped here to the canonical arm) and,
per oversub level, prints the 6-arm table exactly as §5.8 表 6-B:

  策略 | 平均JCT±std | P50±std | P95±std | P99±std | ΔmeanJCT% [95% CI] | Wilcoxon p | P99<bf

All per-arm cells are the UNWEIGHTED mean ± std ACROSS seeds of that seed's own metric
(mean/P50/P95/P99 of its 150 jobs). ΔmeanJCT% is the seed-level PAIRED delta vs Backfill
(mean ± 1.96·SE, matching aggregate_step3.ci95); Wilcoxon is paired signed-rank on those
deltas; P99<bf counts seeds whose arm P99 beat Backfill's.

  python -m eval.scripts.aggregate_optB_deploy runs/step3prio_optB_ov24_6arm_<stamp> [more dirs...]
"""
from __future__ import annotations
import glob, json, os, re, sys
import numpy as np

# canonical arm order for the table; actuation suffix (_reorder/_online/_priority) stripped
SUFFIX = re.compile(r"_(reorder|online|priority)$")
LABEL = {"fcfs": "FCFS", "backfill": "Backfill", "sac": "SAC",
         "rdsac_mean": "RDSAC-mean", "rdsac_cvar": "RDSAC-cvar", "rlpd_cvar": "RLPD"}
ORDER = ["fcfs", "backfill", "sac", "rdsac_mean", "rdsac_cvar", "rlpd_cvar"]


def canon(stem):
    return SUFFIX.sub("", stem)


def load_dir(d):
    out = {}  # arm -> seed -> {mean,p50,p95,p99,n}
    for f in glob.glob(os.path.join(d, "*.json")):
        try:
            o = json.load(open(f))
        except Exception:
            continue
        jct = np.array(o.get("jct", []), dtype=float)
        if not len(jct):
            continue
        arm = canon(os.path.basename(f).rsplit("_s", 1)[0])
        out.setdefault(arm, {})[int(o["seed"])] = dict(
            mean=float(jct.mean()), p50=float(np.percentile(jct, 50)),
            p95=float(np.percentile(jct, 95)), p99=float(np.percentile(jct, 99)),
            n=len(jct))
    return out


def ms(vals):
    a = np.asarray(vals, float)
    return f"{a.mean():.1f} ± {a.std():.1f}"


def delta_ci(arm_d, base):
    seeds = sorted(set(arm_d) & set(base))
    d = np.array([(arm_d[s]["mean"] - base[s]["mean"]) / base[s]["mean"] * 100 for s in seeds])
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    lo, hi = m - 1.96 * se, m + 1.96 * se
    p = float("nan")
    try:
        from scipy.stats import wilcoxon
        if len(d) >= 6:
            _, p = wilcoxon(d)
    except Exception:
        pass
    wins = sum(arm_d[s]["p99"] < base[s]["p99"] for s in seeds)
    return m, lo, hi, p, wins, len(seeds)


def table(d):
    data = load_dir(d)
    base = data.get("backfill")
    ov = re.search(r"ov([0-9.]+)", os.path.basename(os.path.normpath(d)))
    print(f"\n### {d}  (oversub={ov.group(1) if ov else '?'})")
    if not base:
        print("  no backfill baseline"); return
    print(f"| 排程策略 | 平均 JCT (s) | P50 (s) | P95 (s) | P99 (s) | ΔmeanJCT% [95% CI] | Wilcoxon *p* | P99<bf |")
    print(f"|---|--:|--:|--:|--:|--:|--:|--:|")
    for arm in ORDER:
        if arm not in data:
            continue
        ad = data[arm]
        seeds = sorted(ad)
        mean = ms([ad[s]["mean"] for s in seeds])
        p50 = ms([ad[s]["p50"] for s in seeds])
        p95 = ms([ad[s]["p95"] for s in seeds])
        p99 = ms([ad[s]["p99"] for s in seeds])
        if arm == "backfill":
            print(f"| {LABEL[arm]} | {mean} | {p50} | {p95} | {p99} | — | — | — |")
        else:
            m, lo, hi, p, wins, n = delta_ci(ad, base)
            sgn = "+" if m >= 0 else ""
            slo = ("+" if lo >= 0 else "") + f"{lo:.1f}"
            shi = ("+" if hi >= 0 else "") + f"{hi:.1f}"
            pstr = f"{p:.3f}" if p == p else "n/a"
            print(f"| {LABEL[arm]} | {mean} | {p50} | {p95} | {p99} | "
                  f"{sgn}{m:.1f} [{slo}, {shi}] | {pstr} | {wins}/{n} |")


def main():
    dirs = sys.argv[1:] or [max(glob.glob("runs/step3prio_*"), key=os.path.getmtime)]
    for root in dirs:
        for ovd in sorted(glob.glob(os.path.join(root, "ov*")),
                          key=lambda p: float(re.search(r"ov([0-9.]+)", os.path.basename(os.path.normpath(p))).group(1))):
            table(ovd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
