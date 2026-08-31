#!/usr/bin/env python3
"""Option B — periodic re-prioritization daemon (the DEPLOYABLE production form).

The §5.8 static arm precomputes the WHOLE dispatch order once (needs the full job set)
and freezes it into fixed priorities — an evaluation construct, not deployable against an
open-ended production stream. Option C (held-job online select+place) is training-faithful
but operationally fragile (a held queue means an RL outage STALLS every job) and pays an
actuation-latency tax that made it lose to static.

Option B keeps the best of both and is what actually ships:
  * jobs are submitted UNHELD and flow through Slurm normally (NON-BLOCKING, FAIL-SAFE —
    if this daemon dies, jobs still run under Slurm's own priority; nothing is stuck);
  * every ``interval`` seconds the daemon reads the CURRENT pending queue + node state,
    ranks the pending jobs by the RL policy (rolling top-16, same interface as §5.8), and
    writes those ranks as administrator ``Priority`` via ``direct_set_prio`` (sticks on
    unheld jobs — validated in §5.8);
  * Slurm's own in-process backfill then actuates by that priority at NATIVE speed and
    places freely (placement left to cons_tres — §5.2 showed RL placement adds no robust
    benefit and -w pinning serializes under poisson).

So B reacts to LIVE completions/arrivals each cycle (the online reaction C wanted) but
WITHOUT the held-job latency confound, because it only nudges priorities on already-queued
jobs and lets Slurm actuate. Ordering-only, non-blocking, fail-safe.

``rank_pending`` and ``reorder_step`` are pure (I/O injected) so they unit-test offline.
Eval mode (``--eval``) submits a poisson job stream unheld and runs the loop against real
squeue/scontrol, writing the same {arm,seed,jct,wait} json as scontrol_ab for aggregation.
"""
from __future__ import annotations
import argparse, os, threading, time
import numpy as np

from services.rl_scheduler.placement_controller import (
    SlurmJob, SlurmNode, build_act_payload, post_act)

RELEASE_BASE, RELEASE_SPACING = 5_000_000, 10_000   # rank 0 → highest Slurm Priority


def rank_pending(pending, free, *, mps_per_gpu, nodes, serve, post=post_act):
    """Rank the CURRENT pending jobs by RL preference (a drain over the present snapshot;
    no arrival simulation — every pending job is already here). Returns an ordered list of
    job ids, best-dispatched-first. Repeatedly asks the policy for its top (job, node) over
    the rolling top-16 window, commits it against a LOCAL free-MPS tally, and continues;
    RL abstain / no-fit falls back to oldest-waiting first-fit so the ranking is total.

    ``pending``: list of (jid, mps, submit_ts). ``free``: {node: free_mps}. Pure except for
    the injected ``post`` (policy /act)."""
    remaining = {jid: (mps, ts) for jid, mps, ts in pending}
    localfree = dict(free)
    order = []
    while remaining:
        sjobs = [SlurmJob(job_id=jid, name="x", state="PENDING", reason="JobHeld",
                          mps_req=mps, gpu_count=1, gpu_type="rtx4070",
                          runtime=0.0, submit_ts=ts)
                 for jid, (mps, ts) in remaining.items()]
        fnodes = [SlurmNode(name=n, free_mps=int(localfree[n]), running_jobs=0,
                            gpu_type="rtx4070" if "4070" in n else "rtx3080") for n in nodes]
        try:
            act = post(build_act_payload(sjobs, fnodes, mps_per_gpu=mps_per_gpu), scheduler_url=serve)
        except Exception:
            act = {}
        sel = act.get("selected_job_id"); node_j = act.get("node_j")
        pick = node = None
        if sel in remaining and node_j is not None and 0 <= int(node_j) < len(nodes):
            cand = nodes[int(node_j)]
            if localfree[cand] >= remaining[sel][0]:
                pick, node = sel, cand
        if pick is None:   # abstain / no-fit → oldest-waiting first-fit
            for c in sorted(remaining, key=lambda j: remaining[j][1]):
                for n in nodes:
                    if localfree[n] >= remaining[c][0]:
                        pick, node = c, n; break
                if pick:
                    break
        if pick is None:   # nothing fits current free capacity → rank the rest by age
            order.extend(sorted(remaining, key=lambda j: remaining[j][1]))
            break
        order.append(pick); localfree[node] -= remaining[pick][0]; remaining.pop(pick)
    return order


def reorder_step(get_state, set_priorities, *, mps_per_gpu, nodes, serve, post=post_act):
    """One re-prioritization cycle: read pending+free, rank, write priorities. Returns the
    order applied (or [] if the queue is empty). I/O injected for offline testing."""
    pending, free = get_state()
    if not pending:
        return []
    order = rank_pending(pending, free, mps_per_gpu=mps_per_gpu, nodes=nodes, serve=serve, post=post)
    set_priorities({jid: RELEASE_BASE - i * RELEASE_SPACING for i, jid in enumerate(order)})
    return order


def reorder_loop(get_state, set_priorities, is_done, *, interval, mps_per_gpu, nodes,
                 serve, post=post_act, on_error=None):
    """Run reorder_step every ``interval`` s until ``is_done()``. Never raises out of the
    loop (fail-safe): a cycle error is logged and the next cycle retries — jobs keep running
    under whatever priorities Slurm last had. Returns the number of cycles run."""
    cycles = 0
    while not is_done():
        try:
            reorder_step(get_state, set_priorities, mps_per_gpu=mps_per_gpu, nodes=nodes,
                         serve=serve, post=post)
        except Exception as e:            # fail-safe: never stall the queue
            if on_error:
                on_error(e)
        cycles += 1
        time.sleep(interval)
    return cycles


# NOTE: the live EVAL of Option B lives in eval/scripts/scontrol_ab.py::run_reorder_arm
# (--arm reorder), which submits a job stream UNHELD and drives reorder_loop against real
# squeue/scontrol. A PRODUCTION daemon would instead wire get_state/set_priorities to
# slurmrestd (JWT, like snapshot_agent) and run reorder_loop as a long-lived k8s Deployment;
# because jobs stay unheld it is fail-safe (loop down → Slurm's own priority still runs them).
