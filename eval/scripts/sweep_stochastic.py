"""σ-sweep: does stochasticity revive the score / SAC / RDSAC separation?

Background. At 1×1 with a *deterministic* oracle runtime, the transition is
deterministic given (s, a), so the return distribution Z_R collapses toward a
point mass and CVaR ≈ mean — RDSAC's risk machinery is structurally idle and
ties SAC. This harness injects mean-preserving lognormal runtime noise (σ) and
optional MPS co-residency interference, then measures whether the three
schedulers separate, and specifically whether RDSAC's tail (p95/p99) advantage
over vanilla SAC grows with σ.

Design.
  - For each σ we train SAC (scalar critic) and RDSAC (IQN + risk) at the SAME
    σ (matched train/eval), so each has seen the uncertainty it must handle.
  - All three (score / SAC / RDSAC) are evaluated through the SAME stochastic
    gym env. The idiosyncratic per-job noise is common-random (keyed on the
    eval seed), so the comparison is paired: the same job gets the same noise
    multiplier under every policy; only policy-dependent interference differs.
  - Metric: mean ΔJCT% vs score (paired) + per-job p95/p99 JCT.

This is a research harness — it deliberately does NOT touch the production
eval pipeline (eval_dsac_placement.py) so the report's deterministic 30-seed
numbers stay reproducible.

Usage (pilot)::
    PYTHONPATH=. .venv-m11/bin/python eval/scripts/sweep_stochastic.py \
        --sigmas 0.0 1.0 --total-steps 40000 --n-jobs 30 \
        --seeds 42 43 44 --trace-families philly burst \
        --device cuda --out-dir runs/stoch_sweep_$(date +%Y%m%d-%H%M%S)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from sim.gym_env import KubefluxSchedEnv, env_dims, MODE_PACK
from sim.loader import generate_by_family
from sim.scheduler.score import ScoreScheduler
from services.rl_scheduler.dsac import DSACAgent
from services.rl_scheduler.sim_train import sim_train


# ── Policies driven through the gym env ─────────────────────────────────────

def _score_action(env: KubefluxSchedEnv, score_sched: ScoreScheduler) -> int:
    """Highest-score job with a legal placement; no-op if none placeable."""
    st = env._state
    if st is None or not st.pending:
        return env._no_op
    mask = env.action_mask()
    top = env._top_k_jobs()
    ordered = score_sched.order(st.pending, st.cluster, now=st.now)
    for job in ordered:
        idx = next((i for i, j in enumerate(top) if j.job_id == job.job_id), None)
        if idx is None:
            continue
        # The heuristic is throughput-greedy: it always PACKs (accepts sharing).
        for nj in range(env.n_nodes):
            for gk in range(env.gpus_per_node):
                a = env._encode(idx, nj * env.gpus_per_node + gk, MODE_PACK)
                if a < len(mask) and mask[a]:
                    return int(a)
    return env._no_op


def _rollout(env: KubefluxSchedEnv, policy: Callable[[], int], seed: int):
    """Run one episode; return (avg_jct_seconds, per_job_jcts)."""
    env.reset(seed=seed)
    done = False
    info: dict = {}
    while not done:
        act = policy()
        _, _, term, trunc, info = env.step(act)
        done = term or trunc
    return info.get("avg_jct", float("nan")), env.episode_jcts()


def _eval_policy(
    *, make_policy, family: str, sigma: float, interference: float,
    n_jobs: int, seeds: list[int], n_nodes: int, gpus_per_node: int,
    colocation: bool = False,
) -> tuple[list[float], list[float]]:
    """Return (per-seed avg JCTs, pooled per-job JCTs) for one policy/family."""
    total_gpus = n_nodes * gpus_per_node
    avg_jcts: list[float] = []
    per_job: list[float] = []
    for seed in seeds:
        def _factory(_s=seed):
            jobs = generate_by_family(family, n_jobs=n_jobs, seed=_s)
            return [j for j in jobs if j.gpu_count <= total_gpus]

        env = KubefluxSchedEnv(
            _factory, n_nodes=n_nodes, gpus_per_node=gpus_per_node,
            max_steps=n_jobs * 200, reward_mode="jct_aligned",
            runtime_sigma=sigma, interference=interference,
            colocation_actions=colocation,
        )
        policy = make_policy(env)
        avg, jcts = _rollout(env, policy, seed)
        avg_jcts.append(avg)
        per_job.extend(jcts)
        env.close()
    return avg_jcts, per_job


def _train(*, use_iqn, risk_mode, risk_beta, sigma, interference, args) -> DSACAgent:
    return sim_train(
        n_nodes=args.n_nodes, gpus_per_node=args.gpus_per_node,
        trace_family=args.trace_families, n_jobs=args.n_jobs,
        total_steps=args.total_steps, warmup_steps=args.warmup_steps,
        utd_ratio=args.utd_ratio, batch_size=args.batch_size,
        device=args.device, out_dir=None, log_every=max(5000, args.total_steps // 5),
        use_iqn=use_iqn,
        risk_mode=risk_mode, risk_beta=risk_beta,
        curriculum=args.curriculum,
        runtime_sigma=sigma, interference=interference,
        fixed_alpha=args.fixed_alpha, init_alpha=args.init_alpha,
        colocation=args.colocation,
        seed=args.train_seed,
        balance_coef=args.balance_coef,
        normalize_reward=args.normalize_reward,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    p.add_argument("--interference", type=float, default=0.0)
    p.add_argument("--n-nodes", type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=60_000)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--utd-ratio", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-jobs", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--train-seed", type=int, default=42,
                   help="training RNG seed (the multi-seed knob: run the sweep "
                        "several times with different --train-seed to get "
                        "mean±std per (σ, arm) and beat the single-seed caveat)")
    p.add_argument("--balance-coef", type=float, default=0.0,
                   help="P1: potential-based node-balance shaping coefficient "
                        "(0 = off; ~5 penalizes crowding one card)")
    p.add_argument("--normalize-reward", action="store_true",
                   help="P2: running-std reward normalization (PopArt-lite)")
    p.add_argument("--trace-families", nargs="+", default=["philly", "burst", "ali"])
    p.add_argument("--risk-mode", default="cvar")
    p.add_argument("--risk-modes", nargs="+", default=None,
                   help="train one RDSAC per mode (e.g. mean cvar) for the "
                        "3-way distributional-vs-risk split; overrides --risk-mode")
    p.add_argument("--risk-beta", type=float, default=0.25)
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--fixed-alpha", action="store_true",
                   help="pin the entropy temperature α for both models "
                        "(control for the auto-α artifact; see eval §4.3.1)")
    p.add_argument("--init-alpha", type=float, default=0.05,
                   help="the constant α value when --fixed-alpha is set")
    p.add_argument("--colocation", action="store_true",
                   help="enable PACK/ISOLATE co-location action mode (B); "
                        "pair with --interference > 0 to make the choice matter")
    p.add_argument("--no-sac", action="store_true",
                   help="train/eval RDSAC only (skip the SAC arm) — halves "
                        "GPU cost for the focused colocation comparison")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=f"runs/stoch_sweep_{time.strftime('%Y%m%d-%H%M%S')}")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_sched = ScoreScheduler()
    results = []

    for sigma in args.sigmas:
        print(f"\n=== σ={sigma}  interference={args.interference} ===", flush=True)
        t0 = time.time()
        # Arms: SAC (scalar) + one RDSAC per risk mode. risk-modes=[mean,cvar]
        # gives the 3-way split that separates "distributional critic" (mean)
        # from "+ risk distortion" (cvar) — eval R1/M1.
        risk_modes = args.risk_modes if args.risk_modes else [args.risk_mode]
        agents: dict = {}
        if not args.no_sac:
            print("[train] SAC ...", flush=True)
            agents["sac"] = _train(use_iqn=False, risk_mode="mean",
                                   risk_beta=args.risk_beta, sigma=sigma,
                                   interference=args.interference, args=args)
        for m in risk_modes:
            print(f"[train] RDSAC-{m} ...", flush=True)
            agents[f"rdsac-{m}"] = _train(use_iqn=True, risk_mode=m,
                                          risk_beta=args.risk_beta, sigma=sigma,
                                          interference=args.interference, args=args)
        print(f"[train] done in {time.time()-t0:.0f}s; evaluating ...", flush=True)

        # Persist every arm immediately — a degenerate arm or a crash in the
        # eval/metric stage must not throw away hours of training.
        for name, ag in agents.items():
            try:
                ag.save(out_dir / f"ckpt_{name}_sigma{sigma}.pt")
            except Exception as e:  # noqa: BLE001
                print(f"[warn] could not save {name}: {e}", flush=True)

        def _mk(agent):  # bind agent so the closure doesn't capture the loop var
            return lambda env: (lambda: agent.select_action(
                env._build_obs(), env.action_mask(), greedy=True))
        policies = {"score": lambda env: (lambda: _score_action(env, score_sched))}
        for name, ag in agents.items():
            policies[name] = _mk(ag)

        for family in args.trace_families:
            ev = {}
            for name, make in policies.items():
                avg, per_job = _eval_policy(
                    make_policy=make, family=family, sigma=sigma,
                    interference=args.interference, n_jobs=args.n_jobs,
                    seeds=args.seeds, n_nodes=args.n_nodes,
                    gpus_per_node=args.gpus_per_node,
                    colocation=args.colocation,
                )
                ev[name] = {"avg": np.array(avg), "per_job": np.array(per_job)}

            score_avg = ev["score"]["avg"]
            for name in agents:
                m = ev[name]
                pj = m["per_job"]
                # An arm that collapses to no-op completes 0 jobs → nan avg_jct
                # and empty per_job. Record it (completed_frac, nan metrics)
                # instead of crashing the whole sweep.
                both = np.isfinite(score_avg) & np.isfinite(m["avg"])
                delta = (float(((score_avg[both] - m["avg"][both]) / score_avg[both]).mean() * 100.0)
                         if both.any() else float("nan"))
                row = {
                    "sigma": sigma, "family": family, "model": name,
                    "mean_jct_h": (float(np.nanmean(m["avg"])) / 3600
                                   if np.isfinite(m["avg"]).any() else float("nan")),
                    "score_jct_h": float(np.nanmean(score_avg)) / 3600,
                    "delta_pct": delta,
                    "p95_h": float(np.percentile(pj, 95)) / 3600 if pj.size else float("nan"),
                    "p99_h": float(np.percentile(pj, 99)) / 3600 if pj.size else float("nan"),
                    "completed_frac": float(np.isfinite(m["avg"]).mean()),
                }
                results.append(row)
                print(f"  σ={sigma} {family:6s} {name:10s}  "
                      f"Δ={row['delta_pct']:+7.1f}%  "
                      f"p95={row['p95_h']:.2f}h p99={row['p99_h']:.2f}h "
                      f"done={row['completed_frac']:.0%}", flush=True)
            # Incremental save: a late crash keeps everything computed so far.
            (out_dir / "sweep.json").write_text(json.dumps(results, indent=2))

    (out_dir / "sweep.json").write_text(json.dumps(results, indent=2))
    _write_summary(out_dir / "SUMMARY.md", results, args)
    print(f"\n[sweep] results → {out_dir}/sweep.json", flush=True)
    return 0


def _write_summary(path: Path, results: list[dict], args) -> None:
    lines = [
        "# Stochastic σ-sweep — score / SAC / RDSAC separation\n",
        f"- traces: {args.trace_families}  seeds: {args.seeds}  "
        f"n_jobs: {args.n_jobs}  steps: {args.total_steps:,}  "
        f"interference: {args.interference}  risk: {args.risk_mode}:{args.risk_beta}  "
        f"α: {'fixed=' + str(args.init_alpha) if args.fixed_alpha else 'auto'}\n",
        "\nΔJCT% vs score (paired, negative = slower than score); "
        "p95/p99 = per-job tail JCT (hours).\n",
        "\n| σ | family | model | ΔJCT% | p95 (h) | p99 (h) | done% |",
        "|---|--------|-------|------:|--------:|--------:|------:|",
    ]
    for r in results:
        lines.append(
            f"| {r['sigma']} | {r['family']} | {r['model']} | "
            f"{r['delta_pct']:+.1f} | {r['p95_h']:.2f} | {r['p99_h']:.2f} | "
            f"{r.get('completed_frac', 1.0):.0%} |"
        )
    lines.append(
        "\n**Read:** if the hypothesis holds, RDSAC's p95/p99 advantage over SAC "
        "(and its ΔJCT%) should widen as σ grows; at σ=0 the two should tie "
        "(deterministic transition → CVaR ≈ mean).\n"
    )
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
