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


# Live /decide fail-safe guardrail defaults (services/rl_scheduler/serve.py).
# The sim eval lets the DRL policy act on EVERY step (so JCTs stay comparable to
# eval-writeup §3); these thresholds are used only to *report* how often live
# would have abstained → fallen back to the score baseline.
VALUE_ABSTAIN_DEFAULT   = -1.0
ENTROPY_ABSTAIN_DEFAULT = 2.5


def _decision_stats(agent: DSACAgent, obs, mask) -> tuple[float, float]:
    """(value, entropy) for one decision — mirrors serve.py `_AgentHolder.select`.

    value   = Σ_a π(a|s)·Q(a)   (expected action value under the policy)
    entropy = −Σ_a π(a|s)·log π(a|s)
    """
    import torch

    with torch.no_grad():
        obs_t  = torch.as_tensor(obs,  dtype=torch.float32, device=agent.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool,    device=agent.device).unsqueeze(0)
        probs, log_probs = agent.actor.policy(obs_t, mask_t)
        q       = agent.action_values(obs_t)
        entropy = float(-(probs * log_probs).sum(dim=-1).item())
        value   = float((probs * q).sum(dim=-1).item())
    return value, entropy


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
    value_abstain: float = VALUE_ABSTAIN_DEFAULT,
    entropy_abstain: float = ENTROPY_ABSTAIN_DEFAULT,
    accounting_out: dict | None = None,
) -> list[float]:
    """Run agent over seeds, return list of avg_jct (seconds).

    If ``per_job_out`` is given, per-job JCTs across all seeds are appended to it
    (used for risk-sensitive tail metrics: p95/p99 JCT).

    If ``accounting_out`` (a dict of counters) is given, every decision is
    classified — **report-only, the action taken is unchanged** — into:
      ``drl_placement`` : real DRL placement command (action ≠ no-op, passes the
                          live guardrail);
      ``no_op``         : DRL chose to wait (advance the sim);
      ``would_fallback``: live would abstain here (value < ``value_abstain`` or
                          entropy > ``entropy_abstain``) → score baseline drives.
    The no-op check takes precedence over the abstain check, mirroring serve.py.
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
            if accounting_out is not None:
                value, entropy = _decision_stats(agent, obs, mask)
                accounting_out["total"] += 1
                if act == env._no_op:
                    accounting_out["no_op"] += 1
                elif value < value_abstain or entropy > entropy_abstain:
                    accounting_out["would_fallback"] += 1
                else:
                    accounting_out["drl_placement"] += 1
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


def _print_decision_split(rows):
    """Report-only: how the DRL run's decisions split into real commands vs the
    fallback the live serve guardrail would have taken (→ score baseline)."""
    header = (f"{'Family':8s}  {'Decisions':>9s}  {'DRL-cmd':>8s}  "
              f"{'no-op':>7s}  {'→score':>8s}")
    print("\nDecision provenance (report-only; DRL acted on every step here):")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['family']:8s}  {r['decisions']:9d}  "
              f"{r['drl_placement_pct']:7.1f}%  {r['no_op_pct']:6.1f}%  "
              f"{r['score_fallback_pct']:7.1f}%")
    print("  DRL-cmd = genuine DSAC placement · →score = live would abstain "
          "(value<VALUE_ABSTAIN or entropy>ENTROPY_ABSTAIN) → score baseline")


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
                   default=["philly", "ali"], choices=["philly", "ali"])
    p.add_argument("--train-trace",    default=["philly", "ali"],
                   nargs="+", choices=["philly", "ali"],
                   help="trace(s) for training; multiple = mixed (default: both)")
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
    p.add_argument("--num-envs",             type=int, default=1,
                   help="parallel rollout envs during training (>1 = vectorized "
                        "path; score-warmup falls back to random-legal there)")
    # Report-only DRL-vs-score-fallback accounting (live /decide guardrail).
    p.add_argument("--value-abstain",   type=float, default=VALUE_ABSTAIN_DEFAULT,
                   help="report a decision as score-fallback when policy value < this "
                        "(live serve VALUE_ABSTAIN; default -1.0)")
    p.add_argument("--entropy-abstain", type=float, default=ENTROPY_ABSTAIN_DEFAULT,
                   help="report a decision as score-fallback when policy entropy > this "
                        "(live serve ENTROPY_ABSTAIN; default 2.5)")
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
            num_envs=args.num_envs,
        )
        print()

    # ── Paired evaluation ───────────────────────────────────────────────
    rows = []
    for family in args.trace_families:
        print(f"[eval] evaluating {family} ({len(args.seeds)} seeds) ...", end=" ",
              flush=True)
        dsac_per_job: list = []
        acct = {"total": 0, "drl_placement": 0, "no_op": 0, "would_fallback": 0}
        dsac_jcts  = eval_dsac_jct(
            agent, n_nodes=args.n_nodes, gpus_per_node=args.gpus_per_node,
            trace_family=family, n_jobs=args.n_jobs,
            seeds=args.seeds, greedy=args.greedy, per_job_out=dsac_per_job,
            value_abstain=args.value_abstain, entropy_abstain=args.entropy_abstain,
            accounting_out=acct,
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
            # Report-only decision provenance (DRL command vs live score-fallback)
            "decisions":          int(acct["total"]),
            "drl_placement_pct":  100.0 * acct["drl_placement"] / max(1, acct["total"]),
            "no_op_pct":          100.0 * acct["no_op"]         / max(1, acct["total"]),
            "score_fallback_pct": 100.0 * acct["would_fallback"]/ max(1, acct["total"]),
        })
        p99h = rows[-1]["dsac_jct_p99_h"]
        print(f"Δ={pct:+.1f}%  p={p_val:.3f}  p99={p99h:.2f}h  "
              f"[DRL-cmd {rows[-1]['drl_placement_pct']:.0f}% / "
              f"no-op {rows[-1]['no_op_pct']:.0f}% / "
              f"→score {rows[-1]['score_fallback_pct']:.0f}%]")

    print_table(rows)
    _print_decision_split(rows)

    # ── Save CSV (without list columns) ────────────────────────────────
    csv_cols = ["family", "dsac_jct_mean_h", "score_jct_mean_h",
                "delta_pct", "ci95_lo_pct", "ci95_hi_pct",
                "p_value", "significant", "dsac_jct_p95_h", "dsac_jct_p99_h",
                "decisions", "drl_placement_pct", "no_op_pct", "score_fallback_pct"]
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
