"""Evaluate a DRL checkpoint vs heuristics (fcfs/score/multifactor) on the
AI-serving workload, on the metrics that matter for intelligent scheduling:
SLO-violation rate, inference latency, training throughput, utilization.

All arms run the SAME held-out aiserve traces (per eval seed = CRN), so the
comparison is paired. Heuristics run through ``sim.runner``; the DRL policy runs
greedily through the gym env on the identical jobs, and its per-job records are
summarised with the same SLO/per-class formula.

Usage:
    python -m eval.scripts.eval_aiserve --ckpt runs/aiserve_drl_s42/dsac.pt \
        --eval-seeds 100 101 102 --n-jobs 50 --load 0.7
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from sim.loader import generate_ai_serving
from sim.metrics import _pct
from sim.runner import run as run_sim


KEYS = ("jct_mean", "inf_jct", "inf_p99", "train_jct", "slo_viol", "util", "makespan")


def _summarize_records(records: list[dict], util_samples: list[tuple]) -> dict:
    """Per-job records (+ (t,util) samples) → the same panel as the heuristics."""
    jcts = [r["jct"] for r in records]
    inf = [r["jct"] for r in records if r["job_class"] == "inference"]
    tr = [r["jct"] for r in records if r["job_class"] == "training"]
    slo = [r for r in records if r["slo_s"] > 0]
    viol = [r for r in slo if r["jct"] > r["slo_s"]]
    # time-weighted utilization
    util = 0.0
    if len(util_samples) >= 2:
        wsum = tot = 0.0
        for (t0, f0), (t1, _f1) in zip(util_samples, util_samples[1:]):
            dt = t1 - t0
            if dt > 0:
                wsum += f0 * dt
                tot += dt
        util = wsum / tot if tot else 0.0
    makespan = (util_samples[-1][0] - util_samples[0][0]) if util_samples else 0.0
    return {
        "jct_mean": st.fmean(jcts) if jcts else 0.0,
        "inf_jct": st.fmean(inf) if inf else 0.0,
        "inf_p99": _pct(inf, 99) if inf else 0.0,
        "train_jct": st.fmean(tr) if tr else 0.0,
        "slo_viol": (len(viol) / len(slo)) if slo else 0.0,
        "util": util,
        "makespan": makespan,
    }


def eval_heuristic(jobs, sched: str, n_nodes: int, gpus_per_node: int, mps_per_gpu: int) -> dict:
    m, _ = run_sim(jobs, n_nodes=n_nodes, gpus_per_node=gpus_per_node,
                   scheduler_name=sched, mps_per_gpu=mps_per_gpu)
    s = m.summary()
    bc = s.get("by_class", {})
    return {
        "jct_mean": s["jct_mean"],
        "inf_jct": bc.get("inference", {}).get("jct_mean", 0.0),
        "inf_p99": bc.get("inference", {}).get("jct_p99", 0.0),
        "train_jct": bc.get("training", {}).get("jct_mean", 0.0),
        "slo_viol": s.get("slo_violation_rate", 0.0),
        "util": s.get("utilization", 0.0),
        "makespan": s.get("makespan", 0.0),
    }


def eval_drl(ckpt: str, jobs, n_nodes: int, gpus_per_node: int, mps_per_gpu: int,
             eval_seed: int) -> dict:
    from services.rl_scheduler.dsac import DSACAgent
    from sim.gym_env import KubefluxSchedEnv
    agent = DSACAgent.load(ckpt)  # map_location cpu
    env = KubefluxSchedEnv(lambda: list(jobs), n_nodes=n_nodes,
                           gpus_per_node=gpus_per_node, mps_per_gpu=mps_per_gpu)
    obs, _ = env.reset(seed=eval_seed)
    util_samples = [(env._state.now, env.cluster_utilization())]
    for _ in range(env.max_steps):
        mask = env.action_mask()
        a = agent.select_action(obs, mask, greedy=True)
        obs, _r, term, trunc, _info = env.step(a)
        util_samples.append((env._state.now, env.cluster_utilization()))
        if term or trunc:
            break
    return _summarize_records(env.episode_records(), util_samples)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Eval DRL vs heuristics on aiserve")
    p.add_argument("--ckpt", default=None, help="DRL checkpoint (omit to skip DRL arm)")
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[100, 101, 102])
    p.add_argument("--n-jobs", type=int, default=50)
    p.add_argument("--load", type=float, default=0.7)
    p.add_argument("--n-nodes", type=int, default=2)
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument("--mps-per-gpu", type=int, default=100)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    arms = ["fcfs", "multifactor", "score"] + (["DRL"] if args.ckpt else [])
    per_seed: dict[int, dict] = {}
    for s in args.eval_seeds:
        jobs = generate_ai_serving(args.n_jobs, seed=s, n_gpus=args.n_nodes * args.gpus_per_node,
                                   load=args.load)
        row = {}
        for arm in arms:
            if arm == "DRL":
                row[arm] = eval_drl(args.ckpt, jobs, args.n_nodes, args.gpus_per_node,
                                    args.mps_per_gpu, s)
            else:
                row[arm] = eval_heuristic(jobs, arm, args.n_nodes, args.gpus_per_node,
                                          args.mps_per_gpu)
        per_seed[s] = row

    def agg(arm, k):
        vals = [per_seed[s][arm][k] for s in args.eval_seeds]
        return st.fmean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)

    lines = ["# aiserve eval — DRL vs heuristics (mean±std over seeds %s)" % args.eval_seeds, "",
             "| arm | jct_mean | inf_jct | inf_p99 | train_jct | SLO_viol% | util |",
             "|---|--:|--:|--:|--:|--:|--:|"]
    for arm in arms:
        c = {k: agg(arm, k) for k in KEYS}
        lines.append(f"| {arm} | {c['jct_mean'][0]:.0f}±{c['jct_mean'][1]:.0f} | "
                     f"{c['inf_jct'][0]:.0f}±{c['inf_jct'][1]:.0f} | {c['inf_p99'][0]:.0f} | "
                     f"{c['train_jct'][0]:.0f} | {c['slo_viol'][0]*100:.1f}±{c['slo_viol'][1]*100:.1f} | "
                     f"{c['util'][0]:.2f} |")
    out = "\n".join(lines)
    print(out)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out + "\n")
        (Path(args.out).with_suffix(".json")).write_text(json.dumps(per_seed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
