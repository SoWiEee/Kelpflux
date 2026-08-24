#!/usr/bin/env python3
"""Aggregate the Step 3 poisson LOAD SWEEP: runs/step3prio_<stamp>/ov<N>/*.json.

For each oversub level (queue depth), computes each arm's per-seed PAIRED ΔmeanJCT%
vs the Backfill baseline, and prints an arm × oversub table so the ordering headroom
EMERGING with load is visible (live confirmation of the §5.6 ceiling analysis: at a
shallow queue RL ordering can't help; as the standing backlog deepens the learned
policies pull ahead of Slurm's own backfill ordering).

  python -m eval.scripts.aggregate_step3_sweep runs/step3prio_<stamp>
"""
from __future__ import annotations
import glob, json, os, re, sys
import numpy as np

ARMS = ["fcfs", "sac", "rdsac_mean", "rdsac_cvar", "rlpd_cvar"]


def load_dir(d):
    out = {}  # arm -> seed -> mean
    for f in glob.glob(os.path.join(d, "*.json")):
        try:
            o = json.load(open(f))
        except Exception:
            continue
        jct = np.array(o.get("jct", []), dtype=float)
        if not len(jct):
            continue
        stem = os.path.basename(f).rsplit("_s", 1)[0]
        out.setdefault(stem, {})[int(o["seed"])] = float(jct.mean())
    return out


def dmean(arm_d, base):
    seeds = sorted(set(arm_d) & set(base))
    if not seeds:
        return None
    d = [(arm_d[s] - base[s]) / base[s] * 100 for s in seeds]
    return np.mean(d), np.std(d), len(seeds)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob("runs/step3prio_*"), key=os.path.getmtime)
    ovdirs = sorted(glob.glob(os.path.join(root, "ov*")),
                    key=lambda p: float(re.search(r"ov([0-9.]+)", p).group(1)))
    if not ovdirs:
        print("no ov* subdirs in", root, "— not a sweep run"); return 1
    levels = [(float(re.search(r"ov([0-9.]+)", p).group(1)), p) for p in ovdirs]
    print(f"\n=== Step 3 poisson LOAD SWEEP — {root} ===")
    print("ΔmeanJCT% vs Backfill (neg = learned beats backfill); higher oversub = deeper queue\n")
    hdr = f"{'arm':<12}" + "".join(f"{'ov'+str(ov):>16}" for ov, _ in levels)
    print(hdr); print("-" * len(hdr))
    # backfill absolute per level (context)
    row = f"{'Backfill(s)':<12}"
    for ov, d in levels:
        data = load_dir(d); base = data.get("backfill", {})
        row += f"{(np.mean(list(base.values())) if base else float('nan')):>16.0f}"
    print(row); print("-" * len(hdr))
    for arm in ARMS:
        row = f"{arm:<12}"
        for ov, d in levels:
            data = load_dir(d); base = data.get("backfill", {})
            r = dmean(data.get(arm, {}), base) if base else None
            row += (f"{r[0]:>+9.1f}±{r[1]:>4.1f}" if r else f"{'—':>16}")
        print(row)
    print("\n(reads runs/step3prio_*/ov<N>/*.json; per-cell = mean±std of paired ΔmeanJCT% over seeds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
