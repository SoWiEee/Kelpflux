#!/usr/bin/env python3
"""Ceiling / headroom analysis — how much room is there to beat `score` at all?

The paper's negative result is "learned placement does not robustly beat the
score heuristic." The strongest explanation is that the *ceiling* is low: even
the best achievable schedule barely beats score, so no method — RL or otherwise —
has room to win at this scale.

The runner exposes one scheduling lever: dispatch **ordering** (placement is
fixed by `cluster.try_allocate`, identical for every scheduler). A schedule is
therefore a permutation of job priorities, and we bound the best achievable mean
JCT over that class:

  * LOAD SWEEP (main result): on the target 2x1 cluster, vary the job count to
    sweep load, and for each instance find the best ordering by random-restart +
    swap local search (seeded from clairvoyant SJF, a strong start). `headroom% =
    (score - best) / best` is what smarter ordering buys over score.
  * EXACT CALIBRATION: on small *contended* instances (1x1, burst arrivals, n<=7)
    enumerate ALL permutations → the exact optimum over work-conserving
    priority-list schedules, and confirm the search bound matches it.

Small headroom, even as load rises, ⇒ score already sits near the ceiling ⇒ the
negative result is structural. Placement — the other lever — is shown
(near-)inert by the joint-vs-decoupled ablation, so the two bound total headroom.

Usage:
    PYTHONPATH=. .venv-m11/bin/python eval/scripts/headroom_analysis.py \
        --loads 40 60 80 100 --seeds 0 1 2 3 4 5 6 7 8 9 \
        --out-dir runs/headroom_$(date +%Y%m%d-%H%M%S)
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
from scipy import stats

from sim.loader import generate_by_family
from sim.runner import run
from sim.scheduler.fixed_priority import FixedPriorityScheduler


def _jct_order(jobs, order, *, n_nodes, gpus_per_node) -> float:
    priority = {jid: rank for rank, jid in enumerate(order)}
    metrics, _ = run(jobs, n_nodes=n_nodes, gpus_per_node=gpus_per_node,
                     scheduler_obj=FixedPriorityScheduler(priority))
    return metrics.summary()["jct_mean"]


def _jct_named(jobs, name, *, n_nodes, gpus_per_node) -> float:
    metrics, _ = run(jobs, n_nodes=n_nodes, gpus_per_node=gpus_per_node,
                     scheduler_name=name)
    return metrics.summary()["jct_mean"]


def exact_best(jobs, *, n_nodes, gpus_per_node) -> float:
    """Exact optimum over all priority-list permutations (small n only)."""
    ids = [j.job_id for j in jobs]
    best = float("inf")
    for perm in itertools.permutations(ids):
        best = min(best, _jct_order(jobs, list(perm), n_nodes=n_nodes,
                                    gpus_per_node=gpus_per_node))
    return best


def search_best(jobs, *, n_nodes, gpus_per_node, iters, restarts, seed) -> float:
    """Random-restart + swap local search over orderings, seeded from clairvoyant
    SJF (shortest runtime first). Returns the lowest mean JCT found."""
    rng = random.Random(seed)
    ids = [j.job_id for j in jobs]
    n = len(ids)
    sjf = [j.job_id for j in sorted(jobs, key=lambda x: x.runtime)]
    kw = dict(n_nodes=n_nodes, gpus_per_node=gpus_per_node)

    global_best = float("inf")
    for r in range(max(1, restarts)):
        order = sjf[:] if r == 0 else rng.sample(ids, n)
        cur = _jct_order(jobs, order, **kw)
        for _ in range(iters // max(1, restarts)):
            i, k = rng.sample(range(n), 2)
            order[i], order[k] = order[k], order[i]
            val = _jct_order(jobs, order, **kw)
            if val <= cur:
                cur = val
            else:
                order[i], order[k] = order[k], order[i]  # revert
        global_best = min(global_best, cur)
    return global_best


def _agg(gaps):
    arr = np.asarray(gaps, float)
    n = len(arr)
    ci = stats.sem(arr) * stats.t.ppf(0.975, n - 1) if n >= 2 else 0.0
    return {"n": n, "mean_headroom_pct": float(np.mean(arr)),
            "ci95": float(ci), "max_headroom_pct": float(np.max(arr))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-nodes", type=int, default=2)
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument("--families", nargs="+", default=["philly", "burst"],
                   choices=["philly", "ali", "burst"])
    p.add_argument("--loads", type=int, nargs="+", default=[40, 60, 80, 100],
                   help="job counts to sweep load on the 2x1 cluster")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--search-iters", type=int, default=2500)
    p.add_argument("--search-restarts", type=int, default=5)
    # exact calibration tier (small, contended)
    p.add_argument("--calib-nodes", type=int, default=1)
    p.add_argument("--calib-gpus", type=int, default=1)
    p.add_argument("--calib-family", default="burst")
    p.add_argument("--calib-n", type=int, default=7)
    p.add_argument("--calib-seeds", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--out-dir", default=f"runs/headroom_{time.strftime('%Y%m%d-%H%M%S')}")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    rows = []
    t0 = time.time()

    # ── Load sweep (main result) ─────────────────────────────────────────────
    total_gpus = args.n_nodes * args.gpus_per_node
    kw = dict(n_nodes=args.n_nodes, gpus_per_node=args.gpus_per_node)
    print("=== Load sweep (2x1): headroom of best ordering over score ===", flush=True)
    for load in args.loads:
        for family in args.families:
            for seed in args.seeds:
                jobs = [j for j in generate_by_family(family, n_jobs=load, seed=seed)
                        if j.gpu_count <= total_gpus]
                score = _jct_named(jobs, "score", **kw)
                fcfs = _jct_named(jobs, "fcfs", **kw)
                best = search_best(jobs, **kw, iters=args.search_iters,
                                   restarts=args.search_restarts, seed=seed)
                gap = (score - best) / best * 100.0 if best > 0 else float("nan")
                rec = {"tier": "load", "load": load, "kept": len(jobs),
                       "family": family, "seed": seed, "score_jct": score,
                       "fcfs_jct": fcfs, "best_jct": best, "headroom_pct": gap}
                rows.append(rec)
                print(f"  load={load:3d} {family:6s} s{seed} kept={len(jobs):3d} "
                      f"score={score:8.1f} best={best:8.1f} headroom={gap:+5.1f}%",
                      flush=True)

    # ── Exact calibration (small, contended) ─────────────────────────────────
    ckw = dict(n_nodes=args.calib_nodes, gpus_per_node=args.calib_gpus)
    ctg = args.calib_nodes * args.calib_gpus
    print(f"\n=== Exact calibration ({args.calib_nodes}x{args.calib_gpus}, "
          f"{args.calib_family}, n<={args.calib_n}): exact vs search ===", flush=True)
    for seed in args.calib_seeds:
        jobs = [j for j in generate_by_family(args.calib_family, n_jobs=args.calib_n,
                                              seed=seed) if j.gpu_count <= ctg]
        score = _jct_named(jobs, "score", **ckw)
        ex = exact_best(jobs, **ckw)
        se = search_best(jobs, **ckw, iters=800, restarts=3, seed=seed)
        gap = (score - ex) / ex * 100.0 if ex > 0 else float("nan")
        match = abs(se - ex) < 1e-6
        rec = {"tier": "calib", "kept": len(jobs), "seed": seed, "score_jct": score,
               "exact_jct": ex, "search_jct": se, "headroom_pct": gap,
               "search_matches_exact": match}
        rows.append(rec)
        print(f"  s{seed} kept={len(jobs)} score={score:8.1f} exact={ex:8.1f} "
              f"search={se:8.1f} headroom={gap:+5.1f}% search==exact:{match}", flush=True)

    (out / "results.json").write_text(json.dumps(rows, indent=2))

    # ── Aggregate ─────────────────────────────────────────────────────────────
    summary = {"by_load": {}, "by_family": {}, "load_pooled": {}, "calib": {}}
    for load in args.loads:
        g = [r["headroom_pct"] for r in rows if r["tier"] == "load" and r["load"] == load]
        summary["by_load"][str(load)] = _agg(g)
    for family in args.families:
        g = [r["headroom_pct"] for r in rows if r["tier"] == "load" and r["family"] == family]
        if g:
            summary["by_family"][family] = _agg(g)
    summary["load_pooled"] = _agg([r["headroom_pct"] for r in rows if r["tier"] == "load"])
    calib = [r for r in rows if r["tier"] == "calib"]
    summary["calib"] = _agg([r["headroom_pct"] for r in calib])
    summary["calib"]["search_matches_exact_frac"] = float(
        np.mean([r["search_matches_exact"] for r in calib]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Headroom over `score` from optimal/near-optimal ordering ===")
    print(f"{'load (n_jobs)':14s}  {'n':>3s}  {'mean headroom':>14s}  {'95% CI':>9s}  {'max':>7s}")
    print("-" * 56)
    for load in args.loads:
        a = summary["by_load"][str(load)]
        print(f"{load:<14d}  {a['n']:3d}  {a['mean_headroom_pct']:+12.1f}%  "
              f"±{a['ci95']:6.1f}%  {a['max_headroom_pct']:+6.1f}%")
    lp = summary["load_pooled"]
    print(f"{'pooled':14s}  {lp['n']:3d}  {lp['mean_headroom_pct']:+12.1f}%  "
          f"±{lp['ci95']:6.1f}%  {lp['max_headroom_pct']:+6.1f}%")
    print("\nby family (pooled over loads):")
    for family, a in summary["by_family"].items():
        print(f"  {family:8s} n={a['n']:3d}  mean={a['mean_headroom_pct']:+5.1f}% "
              f"±{a['ci95']:.1f}%  max={a['max_headroom_pct']:+.1f}%")
    ca = summary["calib"]
    print(f"\nExact calibration ({args.calib_nodes}x{args.calib_gpus}, contended): "
          f"headroom {ca['mean_headroom_pct']:+.1f}% ±{ca['ci95']:.1f}%, "
          f"search matched exact in {ca['search_matches_exact_frac']*100:.0f}% of instances")
    print(f"headroom% = (score - best_ordering)/best_ordering. Small & flat across "
          f"load ⇒ score near the ceiling ⇒ little room for ANY method.")
    print(f"Wrote {out}/summary.json  ({(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
