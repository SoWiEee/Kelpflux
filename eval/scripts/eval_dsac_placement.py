"""Step 4 eval: DSAC placement-aware policy vs score baseline (paired CI).

Trains DSAC in sim (or loads a checkpoint), then runs paired evaluation
over multiple seeds and trace families. Reports mean JCT, Δ%, 95% CI,
and significance. Saves results to CSV.

Usage::
    # Train + eval (30k steps, 3 families, 5 seeds each):
    PYTHONPATH=. .venv-m11/bin/python eval/scripts/eval_dsac_placement.py \\
        --n-nodes 1 --gpus-per-node 1 --total-steps 30000

    # Eval only (pre-trained checkpoint):
    PYTHONPATH=. .venv-m11/bin/python eval/scripts/eval_dsac_placement.py \\
        --ckpt runs/dsac_sim/dsac.pt --no-train
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from scipy import stats as _scipy_stats
    def _ttest_ci(diffs):
        t, p = _scipy_stats.ttest_1samp(diffs, 0.0)
        ci = _scipy_stats.t.interval(
            0.95, df=len(diffs)-1,
            loc=np.mean(diffs), scale=_scipy_stats.sem(diffs),
        )
        return p, ci
except ImportError:
    def _ttest_ci(diffs):
        n = len(diffs)
        m = np.mean(diffs)
        se = np.std(diffs, ddof=1) / np.sqrt(n)
        # crude normal approximation (no scipy)
        p = float("nan")
        ci = (m - 1.96 * se, m + 1.96 * se)
        return p, ci


from sim.gym_env import KubefluxSchedEnv, env_dims
from sim.loader import generate_by_family
from sim.runner import run as sim_run
from services.rl_scheduler.dsac import DSACAgent
from services.rl_scheduler.distortion import RISK_MODES
from services.rl_scheduler.sim_train import sim_train


def eval_dsac_jct(
    agent: DSACAgent,
    *,
    n_nodes: int,
    gpus_per_node: int,
    trace_family: str,
    n_jobs: int,
    seeds: list[int],
    greedy: bool = True,
    per_job_out: list | None = None,
) -> list[float]:
    """Run agent over seeds, return list of avg_jct (seconds).

    If ``per_job_out`` is given, per-job JCTs across all seeds are appended to it
    (used for risk-sensitive tail metrics: p95/p99 JCT).
    """
    total_gpus = n_nodes * gpus_per_node
    jcts = []
    for seed in seeds:
        def _factory(_s=seed, _tg=total_gpus):
            jobs = generate_by_family(trace_family, n_jobs=n_jobs, seed=_s)
            return [j for j in jobs if j.gpu_count <= _tg]

        env = KubefluxSchedEnv(
            _factory, n_nodes=n_nodes, gpus_per_node=gpus_per_node,
            max_steps=n_jobs * 200, reward_mode="jct_aligned",
        )
        obs, _ = env.reset(seed=seed)
        done = False
        info = {}
        while not done:
            mask = env.action_mask()
            act  = agent.select_action(obs, mask, greedy=greedy)
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
        if per_job_out is not None:
            per_job_out.extend(env.episode_jcts())
        env.close()
        jcts.append(info.get("avg_jct", float("nan")))
    return jcts


def eval_score_jct(
    *,
    n_nodes: int,
    gpus_per_node: int,
    trace_family: str,
    n_jobs: int,
    seeds: list[int],
) -> list[float]:
    total_gpus = n_nodes * gpus_per_node
    jcts = []
    for seed in seeds:
        jobs = generate_by_family(trace_family, n_jobs=n_jobs, seed=seed)
        jobs = [j for j in jobs if j.gpu_count <= total_gpus]
        metrics, _ = sim_run(
            jobs, n_nodes=n_nodes, gpus_per_node=gpus_per_node,
            scheduler_name="score",
        )
        jcts.append(metrics.summary()["jct_mean"])
    return jcts


def print_table(rows):
    header = f"{'Family':8s}  {'DSAC':>8s}  {'Score':>8s}  {'Δ':>7s}  {'CI95_lo':>8s}  {'CI95_hi':>8s}  {'p':>6s}  Sig"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        sig = "✓" if r["significant"] else " "
        print(
            f"{r['family']:8s}  {r['dsac_jct_mean_h']:8.3f}h  "
            f"{r['score_jct_mean_h']:8.3f}h  "
            f"{r['delta_pct']:+7.1f}%  "
            f"{r['ci95_lo_pct']:+8.1f}%  "
            f"{r['ci95_hi_pct']:+8.1f}%  "
            f"{r['p_value']:6.3f}  {sig}"
        )


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-nodes",        type=int, default=1)
    p.add_argument("--gpus-per-node",  type=int, default=1)
    p.add_argument("--total-steps",    type=int, default=500_000,
                   help="sim training steps (ignored with --no-train)")
    p.add_argument("--warmup-steps",   type=int, default=2_000,
                   help="random/score warmup steps before gradient updates")
    p.add_argument("--utd-ratio",      type=int, default=4,
                   help="gradient updates per environment step")
    p.add_argument("--batch-size",     type=int, default=256,
                   help="training batch size")
    p.add_argument("--n-jobs",         type=int, default=50)
    p.add_argument("--seeds",          type=int, nargs="+",
                   default=[42, 43, 44, 45, 46])
    p.add_argument("--trace-families", nargs="+",
                   default=["philly", "burst", "ali"])
    p.add_argument("--train-trace",    default=["philly", "burst", "ali"],
                   nargs="+", choices=["philly", "burst", "ali"],
                   help="trace(s) for training; multiple = mixed (default: all three)")
    p.add_argument("--out-dir",
                   default=f"runs/eval_dsac_{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--ckpt",           default=None,
                   help="path to pre-trained dsac.pt")
    p.add_argument("--no-train",       action="store_true",
                   help="skip training (requires --ckpt)")
    p.add_argument("--greedy",         action="store_true", default=True)
    p.add_argument("--device",         default="cpu",
                   help="torch device for DSAC: 'cpu' or 'cuda'")
    p.add_argument("--no-iqn",               action="store_true",
                   help="vanilla scalar-critic SAC instead of IQN distributional "
                        "critic (no risk distortion)")
    p.add_argument("--risk-mode",            choices=list(RISK_MODES),
                   default="mean",
                   help="risk distortion in the RDSAC actor objective")
    p.add_argument("--risk-beta",            type=float, default=0.25,
                   help="risk parameter (CVaR tail mass, Wang/CPW shape, MSD weight)")
    p.add_argument("--fixed-alpha",          action="store_true",
                   help="pin the entropy temperature α (disables auto-tuning)")
    p.add_argument("--init-alpha",           type=float, default=0.1,
                   help="initial α; with --fixed-alpha this is the constant value")
    p.add_argument("--target-entropy-ratio", type=float, default=0.1,
                   help="auto-α target entropy as a fraction of log(n_actions)")
    p.add_argument("--reward-scale",         type=float, default=20_000.0,
                   help="training reward divisor on -JCT (default 20000)")
    p.add_argument("--no-potential-shaping", action="store_true",
                   help="disable per-step potential shaping")
    p.add_argument("--no-per",               action="store_true",
                   help="disable Prioritized Experience Replay")
    p.add_argument("--curriculum",           action="store_true",
                   help="ramp n_jobs 10→30→50 during training")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Train or load ───────────────────────────────────────────────────
    if args.no_train:
        if not args.ckpt:
            print("error: --no-train requires --ckpt", file=sys.stderr)
            return 2
        print(f"[eval] loading checkpoint: {args.ckpt}")
        agent = DSACAgent.load(
            args.ckpt, risk_mode=args.risk_mode, risk_beta=args.risk_beta
        )
    elif args.ckpt:
        print(f"[eval] loading checkpoint: {args.ckpt}")
        agent = DSACAgent.load(
            args.ckpt, risk_mode=args.risk_mode, risk_beta=args.risk_beta
        )
    else:
        trains = args.train_trace if len(args.train_trace) > 1 else args.train_trace[0]
        use_iqn = not args.no_iqn
        arch_parts = [f"IQN-{args.risk_mode}:{args.risk_beta}" if use_iqn else "SAC"]
        arch_parts.append("MLP")
        arch_parts.append(f"fixedα={args.init_alpha}" if args.fixed_alpha
                          else f"autoα(te={args.target_entropy_ratio})")
        if not args.no_per:               arch_parts.append("PER")
        if not args.no_potential_shaping: arch_parts.append("Shaping")
        if args.curriculum:               arch_parts.append("Curr")
        arch_name = "+".join(arch_parts)
        print(f"[eval] training DSAC({arch_name}) for {args.total_steps:,} steps "
              f"(traces={trains}) ...")
        agent = sim_train(
            n_nodes=args.n_nodes, gpus_per_node=args.gpus_per_node,
            trace_family=trains, n_jobs=args.n_jobs,
            total_steps=args.total_steps,
            warmup_steps=args.warmup_steps,
            utd_ratio=args.utd_ratio,
            batch_size=args.batch_size,
            out_dir=out_dir / "train",
            log_every=max(1000, args.total_steps // 10),
            device=args.device,
            use_iqn=use_iqn,
            fixed_alpha=args.fixed_alpha, init_alpha=args.init_alpha,
            target_entropy_ratio=args.target_entropy_ratio,
            risk_mode=args.risk_mode,
            risk_beta=args.risk_beta,
            reward_scale=args.reward_scale,
            potential_shaping=not args.no_potential_shaping,
            use_per=not args.no_per,
            curriculum=args.curriculum,
        )
        print()

    # ── Paired evaluation ───────────────────────────────────────────────
    rows = []
    for family in args.trace_families:
        print(f"[eval] evaluating {family} ({len(args.seeds)} seeds) ...", end=" ",
              flush=True)
        dsac_per_job: list = []
        dsac_jcts  = eval_dsac_jct(
            agent, n_nodes=args.n_nodes, gpus_per_node=args.gpus_per_node,
            trace_family=family, n_jobs=args.n_jobs,
            seeds=args.seeds, greedy=args.greedy, per_job_out=dsac_per_job,
        )
        score_jcts = eval_score_jct(
            n_nodes=args.n_nodes, gpus_per_node=args.gpus_per_node,
            trace_family=family, n_jobs=args.n_jobs, seeds=args.seeds,
        )
        diffs      = [s - d for s, d in zip(score_jcts, dsac_jcts)]
        score_mean = np.mean(score_jcts)
        p_val, (ci_lo, ci_hi) = _ttest_ci(diffs)
        pct    = np.mean(diffs) / score_mean * 100
        rows.append({
            "family":           family,
            "dsac_jct_mean_h":  float(np.mean(dsac_jcts)) / 3600,
            "score_jct_mean_h": float(score_mean) / 3600,
            "delta_pct":        float(pct),
            "ci95_lo_pct":      float(ci_lo / score_mean * 100),
            "ci95_hi_pct":      float(ci_hi / score_mean * 100),
            "p_value":          float(p_val),
            "significant":      bool(p_val < 0.05) if not np.isnan(p_val) else False,
            # Risk-sensitive tail metrics (per-job JCT across all seeds, hours)
            "dsac_jct_p95_h":   float(np.percentile(dsac_per_job, 95)) / 3600
                                if dsac_per_job else float("nan"),
            "dsac_jct_p99_h":   float(np.percentile(dsac_per_job, 99)) / 3600
                                if dsac_per_job else float("nan"),
            "dsac_jcts_h":      [j / 3600 for j in dsac_jcts],
            "score_jcts_h":     [j / 3600 for j in score_jcts],
        })
        p99h = rows[-1]["dsac_jct_p99_h"]
        print(f"Δ={pct:+.1f}%  p={p_val:.3f}  p99={p99h:.2f}h")

    print_table(rows)

    # ── Save CSV (without list columns) ────────────────────────────────
    csv_cols = ["family", "dsac_jct_mean_h", "score_jct_mean_h",
                "delta_pct", "ci95_lo_pct", "ci95_hi_pct",
                "p_value", "significant", "dsac_jct_p95_h", "dsac_jct_p99_h"]
    csv_path = out_dir / "eval_dsac_placement.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Save full JSON (includes per-seed arrays)
    json_path = out_dir / "eval_dsac_placement.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n[eval] results → {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
