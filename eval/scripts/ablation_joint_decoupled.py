#!/usr/bin/env python3
"""Joint-vs-decoupled placement ablation (docs/review.md P0).

The DSAC action encodes two independent decisions — *which job* to run next and
*which GPU* to place it on (MPS co-location mode is a separate, off-by-default
axis). This script isolates each by restricting the action mask during both
training and evaluation, then asks how much of the learned policy's JCT comes
from each axis versus their joint use:

  joint           agent chooses job AND placement  (full action space)
  placement_only  job frozen to the score scheduler's top pick; agent picks GPU
  job_only        placement frozen to first-fit; agent picks which job

All three train the SAME DSACAgent shape (obs_dim / n_actions identical) — only
action_mask() differs — so the comparison is apples-to-apples. Meaningful only
at >=2 placements, so the default cluster is 2x1 with the measured heterogeneous
cards (RTX 4070 + RTX 3080) and node-2's tight host RAM, which is the real
placement lever on this cluster (see sim/gym_env.py SPEED_MATRIX).

Honest-null expectation: if the three arms land within a few % of each other,
that IS the finding — the joint decision adds little at this scale, consistent
with the paper's negative result. The ablation is reported either way.

Usage (standard run, ~12h on one RTX 4070):
    PYTHONPATH=. .venv-m11/bin/python eval/scripts/ablation_joint_decoupled.py \
        --device cuda --total-steps 50000 \
        --train-seeds 42 43 44 45 46 \
        --out-dir runs/ablation_$(date +%Y%m%d-%H%M%S)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from sim.gym_env import KubefluxSchedEnv
from sim.loader import generate_by_family
from services.rl_scheduler.dsac import DSACAgent
from services.rl_scheduler.sim_train import sim_train

ARMS = ["joint", "placement_only", "job_only"]
_ARM_TO_MODE = {"joint": None, "placement_only": "placement_only", "job_only": "job_only"}


def eval_agent(
    agent: DSACAgent,
    *,
    arm: str,
    n_nodes: int,
    gpus_per_node: int,
    node_gpu_types: list,
    node_ram_gb: list,
    families: list[str],
    eval_seeds: list[int],
    n_jobs: int,
) -> dict[str, float]:
    """Greedy rollout of one agent under its arm restriction. Returns per-family
    mean JCT plus the pooled mean across all (family, seed) runs."""
    total_gpus = n_nodes * gpus_per_node
    per_family: dict[str, float] = {}
    pooled: list[float] = []
    for family in families:
        fam_jcts: list[float] = []
        for seed in eval_seeds:
            def _factory(_s=seed, _tg=total_gpus, _f=family):
                jobs = generate_by_family(_f, n_jobs=n_jobs, seed=_s)
                return [j for j in jobs if j.gpu_count <= _tg]

            env = KubefluxSchedEnv(
                _factory,
                n_nodes=n_nodes, gpus_per_node=gpus_per_node,
                max_steps=n_jobs * 200, reward_mode="jct_aligned",
                node_gpu_types=node_gpu_types, node_ram_gb=node_ram_gb,
                ablation_mode=_ARM_TO_MODE[arm],
            )
            obs, _ = env.reset(seed=seed)
            done, info = False, {}
            while not done:
                mask = env.action_mask()
                act = agent.select_action(obs, mask, greedy=True)
                obs, _, term, trunc, info = env.step(act)
                done = term or trunc
            jct = float(info.get("avg_jct", float("nan")))
            fam_jcts.append(jct)
            pooled.append(jct)
            env.close()
        per_family[family] = float(np.mean(fam_jcts))
    return {"per_family": per_family, "pooled_mean": float(np.mean(pooled))}


def paired_summary(joint: list[float], other: list[float]) -> dict:
    """Paired delta of ``other`` vs ``joint`` (per training seed). Positive delta%
    = arm has HIGHER JCT = worse than joint."""
    j = np.asarray(joint, dtype=float)
    o = np.asarray(other, dtype=float)
    delta_pct = (o - j) / j * 100.0
    n = len(delta_pct)
    mean_d = float(np.mean(delta_pct))
    if n >= 2 and np.std(delta_pct) > 0:
        sem = stats.sem(delta_pct)
        ci = sem * stats.t.ppf(0.975, n - 1)
        t_stat, p = stats.ttest_rel(o, j)
        p = float(p)
    else:
        ci, p = 0.0, float("nan")
    return {
        "n": n,
        "delta_pct_mean": mean_d,
        "delta_pct_ci95": float(ci),
        "paired_t_p": p,
        "per_seed_delta_pct": [float(x) for x in delta_pct],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-nodes", type=int, default=2)
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument("--node-gpu-types", nargs="+", default=["rtx4070", "rtx3080"])
    p.add_argument("--node-ram-gb", type=float, nargs="+", default=[32.0, 5.0],
                   help="usable host RAM per node; node-2 (3080) is the tight one")
    p.add_argument("--total-steps", type=int, default=50_000)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--n-jobs", type=int, default=50)
    p.add_argument("--train-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--families", nargs="+", default=["philly", "ali"],
                   choices=["philly", "ali", "burst"])
    p.add_argument("--train-trace", nargs="+", default=["philly", "ali"])
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default=f"runs/ablation_{time.strftime('%Y%m%d-%H%M%S')}")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    results_fh = open(out / "results.jsonl", "w")

    # (arm, train_seed) -> pooled mean JCT
    pooled: dict[str, dict[int, float]] = {a: {} for a in args.arms}
    per_family: dict[str, dict[int, dict]] = {a: {} for a in args.arms}

    t0 = time.time()
    n_runs = len(args.arms) * len(args.train_seeds)
    done_runs = 0
    for arm in args.arms:
        for seed in args.train_seeds:
            r0 = time.time()
            agent = sim_train(
                n_nodes=args.n_nodes, gpus_per_node=args.gpus_per_node,
                node_gpu_types=args.node_gpu_types, node_ram_gb=args.node_ram_gb,
                trace_family=args.train_trace, n_jobs=args.n_jobs,
                total_steps=args.total_steps, warmup_steps=args.warmup_steps,
                seed=seed, device=args.device, reward_mode="jct_aligned",
                ablation_mode=_ARM_TO_MODE[arm],
                out_dir=out / f"train_{arm}_s{seed}",
            )
            ckpt = out / f"train_{arm}_s{seed}" / "dsac.pt"
            agent.save(ckpt)
            ev = eval_agent(
                agent, arm=arm,
                n_nodes=args.n_nodes, gpus_per_node=args.gpus_per_node,
                node_gpu_types=args.node_gpu_types, node_ram_gb=args.node_ram_gb,
                families=args.families, eval_seeds=args.eval_seeds, n_jobs=args.n_jobs,
            )
            pooled[arm][seed] = ev["pooled_mean"]
            per_family[arm][seed] = ev["per_family"]
            done_runs += 1
            rec = {"arm": arm, "train_seed": seed,
                   "pooled_mean_jct": ev["pooled_mean"],
                   "per_family": ev["per_family"],
                   "run_seconds": round(time.time() - r0, 1)}
            results_fh.write(json.dumps(rec) + "\n")
            results_fh.flush()
            eta = (time.time() - t0) / done_runs * (n_runs - done_runs)
            print(f"[{done_runs}/{n_runs}] arm={arm} seed={seed} "
                  f"pooled_JCT={ev['pooled_mean']:.1f}s "
                  f"({rec['run_seconds']:.0f}s, ETA {eta/60:.0f}m)", flush=True)

    results_fh.close()

    # ── Aggregate + paired stats ────────────────────────────────────────────
    summary = {"arms": {}, "paired_vs_joint": {}}
    for arm in args.arms:
        vals = [pooled[arm][s] for s in args.train_seeds if s in pooled[arm]]
        arr = np.asarray(vals, dtype=float)
        n = len(arr)
        ci = (stats.sem(arr) * stats.t.ppf(0.975, n - 1)) if n >= 2 else 0.0
        summary["arms"][arm] = {
            "mean_jct_s": float(np.mean(arr)), "ci95_s": float(ci), "n": n,
            "per_seed": {str(s): pooled[arm][s] for s in args.train_seeds if s in pooled[arm]},
        }
    if "joint" in args.arms:
        jvals = [pooled["joint"][s] for s in args.train_seeds]
        for arm in args.arms:
            if arm == "joint":
                continue
            ovals = [pooled[arm][s] for s in args.train_seeds]
            summary["paired_vs_joint"][arm] = paired_summary(jvals, ovals)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ── Print table ──────────────────────────────────────────────────────────
    print("\n=== Joint-vs-Decoupled ablation (2x1, pooled JCT over families×seeds) ===")
    print(f"{'arm':16s}  {'mean JCT':>10s}  {'95% CI':>9s}  {'Δ vs joint':>12s}  {'paired p':>9s}")
    print("-" * 66)
    for arm in args.arms:
        a = summary["arms"][arm]
        if arm == "joint":
            print(f"{arm:16s}  {a['mean_jct_s']:9.1f}s  ±{a['ci95_s']:7.1f}s  "
                  f"{'—':>12s}  {'—':>9s}")
        else:
            pv = summary["paired_vs_joint"][arm]
            print(f"{arm:16s}  {a['mean_jct_s']:9.1f}s  ±{a['ci95_s']:7.1f}s  "
                  f"{pv['delta_pct_mean']:+7.1f}±{pv['delta_pct_ci95']:.1f}%  "
                  f"{pv['paired_t_p']:9.3f}")
    print(f"\nΔ>0 means the decoupled arm has HIGHER JCT (worse) than joint.")
    print(f"Wrote {out}/summary.json  ({(time.time()-t0)/60:.0f} min total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
