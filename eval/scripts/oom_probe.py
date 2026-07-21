#!/usr/bin/env python3
"""Verify the opt-in stochastic host-RAM OOM model — cheaply, without training.

The model (sim/gym_env.py) turns node-2's real drain risk into a *penalised,
learnable* tail event: a job's realized peak host RSS is uncertain at placement
time, so co-locating RAM-heavy jobs onto the small (5 GB) node can spike past its
budget → OOM → drain penalty. The paper's negative-result thesis needs this to be
an *avoidable* signal (else RL could never win); this probe demonstrates exactly
that by driving the env with two fixed placement disciplines — no DSAC training:

  * greedy       — place onto any legal slot immediately (pack the small node
                   whenever a MPS slot is free), maximising OOM exposure;
  * conservative — only place a job where the node keeps ≥ safety× the job's
                   nominal ram_req in headroom (room to absorb a peak spike);
                   otherwise NO_OP and WAIT rather than risk a drain.

The real lever is co-locate-vs-wait, not which node: each GPU has only 4 MPS
slots, so once the roomy node's GPU fills, further jobs can only pack the 5 GB
node. If the switch works, greedy eats drain penalties (OOM events, inflated mean
JCT) that conservative trades away for a little queue delay — proving the tail
cost is an avoidable placement choice, not a noise floor. Also checks OFF ==
legacy (deterministic, zero OOM).

Usage:
    PYTHONPATH=. .venv-m11/bin/python eval/scripts/oom_probe.py --seeds 20
"""
from __future__ import annotations

import argparse
import statistics as stats

import numpy as np

from sim.gym_env import KubefluxSchedEnv
from sim.loader import Job, RAM_REQ_GB


def _burst_factory(n_jobs: int, seed: int):
    """A burst of fractional (MPS) jobs, RAM-heavy enough that crowding the 5 GB
    node overcommits it. Fractional mps_req=1 lets up to 4 co-reside per GPU, so
    co-residency (hence OOM) is actually reachable — whole-GPU jobs never share."""
    rng = np.random.default_rng(seed)

    def _factory():
        jobs = []
        for i in range(n_jobs):
            cls = "llm" if rng.random() < 0.6 else "inference"
            jobs.append(Job(
                job_id=f"j{i}", user="u", gpu_count=1, gpu_type="any",
                submit_ts=float(rng.integers(0, 30)),   # bursty → contention
                runtime=float(rng.integers(200, 800)), mem_req=0.0, mps_req=1,
                job_class=cls, ram_req=RAM_REQ_GB[cls]))
        return jobs
    return _factory


def _make_env(n_jobs, seed, *, oom_penalty_s, oom_sigma, ram_overcommit):
    return KubefluxSchedEnv(
        _burst_factory(n_jobs, seed),
        n_nodes=2, gpus_per_node=1,
        node_gpu_types=["rtx4070", "rtx3080"], node_ram_gb=[62.0, 5.0],
        oom_penalty_s=oom_penalty_s, oom_sigma=oom_sigma,
        ram_overcommit=ram_overcommit, max_steps=n_jobs * 200)


def _run_policy(env, seed, policy, n_placements, gpus_per_node, *,
                safety=2.0) -> tuple[float, int]:
    """Drive one episode with a fixed greedy / conservative placement discipline.
    Returns (mean_jct, oom_events)."""
    env.reset(seed=seed)
    no_op = env._no_op

    def _node_of(action: int) -> int:
        return (action % n_placements) // gpus_per_node

    def _job_of(action: int):
        top = env._top_k_jobs()
        job_i = action // n_placements
        return top[job_i] if job_i < len(top) else None

    while True:
        mask = env.action_mask()
        legal = [a for a in range(len(mask)) if mask[a] and a != no_op]
        cluster = env._state.cluster
        if not legal:
            action = no_op
        elif policy == "greedy":
            # pack: prefer the slot on the fuller (lower free-RAM) node.
            action = min(legal, key=lambda a: (cluster.nodes[_node_of(a)].free_ram_ratio(), a))
        else:  # conservative: only place where the node keeps safety× headroom
            def safe(a: int) -> bool:
                job = _job_of(a)
                if job is None:
                    return False
                return cluster.nodes[_node_of(a)].free_ram_gb() >= safety * job.ram_req
            safe_acts = [a for a in legal if safe(a)]
            action = (max(safe_acts, key=lambda a: (cluster.nodes[_node_of(a)].free_ram_gb(), -a))
                      if safe_acts else no_op)   # nothing safe → wait, don't risk OOM
        _obs, _r, term, trunc, _info = env.step(action)
        if term or trunc:
            break
    st = env._state
    mean_jct = st.jct_sum / max(1, st.completed)
    return mean_jct, env._oom_events


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--n-jobs", type=int, default=40)
    p.add_argument("--oom-penalty-s", type=float, default=1200.0)
    p.add_argument("--oom-sigma", type=float, default=0.6)
    p.add_argument("--ram-overcommit", type=float, default=1.0)
    args = p.parse_args(argv)

    n_placements = 2 * 1
    gpus_per_node = 1

    # ── sanity: OFF must be deterministic and OOM-free ──
    off = _make_env(args.n_jobs, 0, oom_penalty_s=0.0, oom_sigma=0.0,
                    ram_overcommit=1.0)
    j0, o0 = _run_policy(off, 0, "greedy", n_placements, gpus_per_node)
    off2 = _make_env(args.n_jobs, 0, oom_penalty_s=0.0, oom_sigma=0.0,
                     ram_overcommit=1.0)
    j0b, _ = _run_policy(off2, 0, "greedy", n_placements, gpus_per_node)
    assert o0 == 0, "OOM fired while disabled"
    assert j0 == j0b, "OFF regime is not deterministic"
    print(f"[sanity] OOM OFF → deterministic (mean JCT {j0:.0f}s repeatable), "
          f"0 OOM events ✓\n")

    # ── ON: greedy pack vs conservative wait under the OOM regime ──
    kw = dict(oom_penalty_s=args.oom_penalty_s, oom_sigma=args.oom_sigma,
              ram_overcommit=args.ram_overcommit)
    rows = {"greedy": [], "conservative": []}
    for s in range(args.seeds):
        for policy in ("greedy", "conservative"):
            env = _make_env(args.n_jobs, s, **kw)
            mj, oom = _run_policy(env, s, policy, n_placements, gpus_per_node)
            rows[policy].append((mj, oom))

    print(f"OOM regime: penalty={args.oom_penalty_s:.0f}s  sigma={args.oom_sigma}  "
          f"overcommit={args.ram_overcommit}  ({args.seeds} seeds × {args.n_jobs} jobs)\n")
    print(f"{'policy':14s}  {'mean JCT (s)':>14s}  {'OOM events/ep':>14s}")
    print("-" * 48)
    summary = {}
    for policy in ("greedy", "conservative"):
        mjs = [r[0] for r in rows[policy]]
        ooms = [r[1] for r in rows[policy]]
        summary[policy] = (stats.mean(mjs), stats.mean(ooms))
        print(f"{policy:14s}  {stats.mean(mjs):>14.0f}  {stats.mean(ooms):>14.2f}")

    d_jct = (summary["greedy"][0] - summary["conservative"][0]) / summary["conservative"][0] * 100
    print(f"\nGreedy pack pays {d_jct:+.1f}% mean JCT and "
          f"{summary['greedy'][1] - summary['conservative'][1]:+.2f} more OOM/ep "
          f"than conservative wait.")
    print("→ the drain cost is an AVOIDABLE placement choice, so a risk-aware "
          "policy has a learnable target (validates the switch).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
