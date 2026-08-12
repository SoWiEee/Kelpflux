"""Live RLPD transition collector — passive behaviour logging on real Slurm.

Polls squeue every POLL_INTERVAL seconds and records, for each job, the
(obs, act, rew, next_obs) transition that the PRODUCTION scheduler (Slurm
multifactor / score) actually realised:

  1. snapshot the decision-state (obs, mask) while a job is PENDING;
  2. when it starts RUNNING, encode which node Slurm placed it on as ``act``;
  3. when it finishes, charge the realised reward (−JCT) and append the
     transition to a JSONL file for later RLPD fine-tuning.

This is deliberately PASSIVE: it never issues srun and never overrides Slurm's
placement. The behaviour policy is whatever the production scheduler did, which
yields clean off-policy (s, a, r, s') data that RLPD (Ball et al. 2023) can
learn from without perturbing the live cluster.

  NOTE: this is NOT the production placement path. Slurm-safe hard placement
  (hold/release + ``required_nodes`` over slurmrestd) lives in
  ``placement_controller.py`` (deployed as rl-placement-controller). The old
  srun-based active-placement path was removed on purpose — srun bypasses Slurm
  priority/backfill/GRES-MPS accounting and the operator lifecycle guard. See
  docs/scheduler.md (§ "live 蒐集").

Usage::
    .venv-m11/bin/python -m services.rl_scheduler.live_daemon \\
        --node-name slurm-worker-gpu-rtx4070 slurm-worker-gpu-rtx3080 \\
        --log-dir live_logs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sim.gym_env import (
    GPU_FEAT_DIM, GPU_TYPES, GLOBAL_FEAT_DIM, JOB_FEAT_DIM,
    TOPO_FEAT_DIM, TOP_K, env_dims,
)
from sim.loader import MPS_PER_GPU
from services.rl_scheduler.rlpd_finetune import ReplayBuffer, Transition


# ── Cluster state from Slurm ──────────────────────────────────────────────

@dataclass
class LiveJob:
    job_id:    str
    mps_req:   int
    gpu_count: int
    gpu_type:  str
    runtime:   float   # predicted (from predictor API) or 0
    submit_ts: float
    state:     str     # PENDING / RUNNING / COMPLETED / etc.
    nodelist:  str     # empty if pending


@dataclass
class LiveGpu:
    node_name:    str
    gpu_index:    int
    free_mps:     int
    running_jobs: int = 0


@dataclass
class LiveCluster:
    nodes: Dict[str, List[LiveGpu]] = field(default_factory=dict)
    mps_per_gpu: int = MPS_PER_GPU


# Optional exec prefix so the daemon can run OFF-cluster (e.g. on the dev host
# where torch lives) and reach slurm via kubectl, e.g.
#   SLURM_EXEC_PREFIX="kubectl exec -n slurm slurm-controller-0 --"
# Empty (default) = run in-cluster, unchanged behaviour.
import shlex as _shlex  # noqa: E402
_EXEC_PREFIX = _shlex.split(os.environ.get("SLURM_EXEC_PREFIX", ""))


def _run(cmd: List[str], timeout: int = 20) -> str:
    try:
        r = subprocess.run(_EXEC_PREFIX + cmd, capture_output=True, text=True,
                           timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def _parse_squeue(raw: str) -> List[LiveJob]:
    """Parse squeue --json output into LiveJob list."""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    jobs = []
    for j in data.get("jobs", []):
        state = j.get("job_state", ["UNKNOWN"])
        if isinstance(state, list):
            state = state[0]
        # Extract MPS from GRES string, e.g. "gpu:mps:4"
        tres = j.get("tres_req_str", "") or j.get("gres", "") or ""
        mps_req = 0
        for part in str(tres).split(","):
            if "mps" in part.lower():
                nums = [int(x) for x in part.split(":") if x.isdigit()]
                if nums:
                    mps_req = nums[-1]
        gpu_count = j.get("gpus_total", 1) or 1
        jobs.append(LiveJob(
            job_id=str(j.get("job_id", "")),
            mps_req=mps_req or MPS_PER_GPU,  # default = full GPU
            gpu_count=int(gpu_count),
            gpu_type="rtx4070",
            runtime=float(j.get("time_limit", {}).get("number", 0) * 60
                          if isinstance(j.get("time_limit"), dict) else 0),
            submit_ts=float(j.get("submit_time", {}).get("number", time.time())
                            if isinstance(j.get("submit_time"), dict) else time.time()),
            state=state,
            nodelist=str(j.get("nodes", "") or ""),
        ))
    return jobs


def _parse_scontrol_node(raw: str, node_name: str,
                          mps_per_gpu: int, n_gpus: int) -> List[LiveGpu]:
    """Parse scontrol show node output to extract free MPS per GPU."""
    gpus = []
    # Try to find AllocTRES/CfgTRES for MPS slots
    free_mps = mps_per_gpu  # default = fully free

    for line in raw.split("\n"):
        if "AllocTRES" in line:
            for token in line.split():
                if "mps" in token.lower() and "=" in token:
                    try:
                        used = int(token.split("mps")[-1].lstrip(":="))
                        free_mps = max(0, mps_per_gpu - used)
                    except ValueError:
                        pass

    for gi in range(n_gpus):
        gpus.append(LiveGpu(
            node_name=node_name, gpu_index=gi,
            free_mps=free_mps,  # simplified: same free for all GPUs on node
        ))
    return gpus


def query_cluster(node_names: List[str], n_gpus: int,
                  mps_per_gpu: int) -> LiveCluster:
    """Query Slurm for current GPU state on each node."""
    cluster = LiveCluster(mps_per_gpu=mps_per_gpu)
    for node in node_names:
        raw  = _run(["scontrol", "show", "node", node])
        gpus = _parse_scontrol_node(raw, node, mps_per_gpu, n_gpus)
        cluster.nodes[node] = gpus
    return cluster


# ── Observation builder (mirrors gym_env.py) ──────────────────────────────

def _job_feat(j: LiveJob, now: float, mps_per_gpu: int) -> np.ndarray:
    gpu_oh = [1.0 if j.gpu_type == t else 0.0 for t in GPU_TYPES]
    wait   = max(0.0, now - j.submit_ts)
    return np.array([
        j.mps_req / mps_per_gpu,
        float(j.gpu_count),
        *gpu_oh,
        math.log1p(j.runtime),
        math.log1p(wait),
        math.log1p(wait),
        0.0, 0.0,
    ], dtype=np.float32)


def _gpu_feat_live(g: LiveGpu, mps_per_gpu: int) -> np.ndarray:
    free_ratio = g.free_mps / mps_per_gpu if mps_per_gpu > 0 else 0.0
    return np.array([
        free_ratio, free_ratio, float(g.running_jobs),
        1.0, 0.0, 0.0,   # rtx4070 one-hot
    ], dtype=np.float32)


def _topo_feat(pending: List[LiveJob]) -> np.ndarray:
    ddp_ratio = sum(1 for j in pending if j.gpu_count > 1) / max(1, len(pending))
    return np.array([1.0, 1.0, ddp_ratio, 0.0], dtype=np.float32)


def _global_feat(pending: List[LiveJob], cluster: LiveCluster, now: float) -> np.ndarray:
    queue_len = len(pending)
    if len(pending) >= 2:
        rts   = sorted(j.runtime for j in pending)
        n     = len(rts)
        p50   = rts[int(n * 0.50)]
        p90   = rts[min(int(n * 0.90), n - 1)]
        spread = (p90 / p50) if p50 > 0 else 1.0
    else:
        spread = 1.0
    tod = (now % 86400) / 86400.0
    return np.array([
        math.log1p(queue_len), spread, 0.0,
        math.sin(2 * math.pi * tod),
        math.cos(2 * math.pi * tod),
        0.0,
    ], dtype=np.float32)


def build_obs_and_mask(
    pending: List[LiveJob],
    cluster: LiveCluster,
    node_names: List[str],
    n_gpus: int,
    now: float,
) -> tuple[np.ndarray, np.ndarray, List[Optional[str]]]:
    n_nodes      = len(node_names)
    mps_per_gpu  = cluster.mps_per_gpu
    n_placements = n_nodes * n_gpus
    n_actions    = TOP_K * n_placements + 1
    no_op        = n_actions - 1

    top = sorted(pending, key=lambda j: j.submit_ts)[:TOP_K]

    job_feats: List[np.ndarray] = []
    top_ids:   List[Optional[str]] = []
    for i in range(TOP_K):
        if i < len(top):
            job_feats.append(_job_feat(top[i], now, mps_per_gpu))
            top_ids.append(top[i].job_id)
        else:
            job_feats.append(np.zeros(JOB_FEAT_DIM, dtype=np.float32))
            top_ids.append(None)

    gpu_feats: List[np.ndarray] = []
    for node in node_names:
        gpus = cluster.nodes.get(node, [])
        for gi in range(n_gpus):
            if gi < len(gpus):
                gpu_feats.append(_gpu_feat_live(gpus[gi], mps_per_gpu))
            else:
                gpu_feats.append(np.zeros(GPU_FEAT_DIM, dtype=np.float32))

    topo = _topo_feat(top)
    glob = _global_feat(top, cluster, now)

    obs = np.concatenate([*job_feats, *gpu_feats, topo, glob]).astype(np.float32)

    mask = np.zeros(n_actions, dtype=bool)
    for i, j in enumerate(top):
        for nj, node in enumerate(node_names):
            gpus = cluster.nodes.get(node, [])
            for gk in range(n_gpus):
                if gk < len(gpus) and gpus[gk].free_mps >= j.mps_req:
                    a = i * n_placements + nj * n_gpus + gk
                    mask[a] = True
    mask[no_op] = True

    return obs, mask, top_ids


# ── Daemon loop ───────────────────────────────────────────────────────────

def run_daemon(
    *,
    node_names: List[str],
    log_dir: Path,
    n_gpus: int = 1,
    mps_per_gpu: int = MPS_PER_GPU,
    poll_interval: float = 30.0,
    buf_capacity: int = 10_000,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    obs_dim, n_actions = env_dims(len(node_names), n_gpus)
    live_buf = ReplayBuffer(capacity=buf_capacity, obs_dim=obs_dim, n_actions=n_actions)
    log_path = log_dir / f"transitions_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    decisions_made = 0

    print(f"[daemon] starting — nodes={node_names}  poll={poll_interval}s "
          f"(passive: logs transitions, never issues srun)")
    print(f"[daemon] log → {log_path}")

    # Behaviour-policy observation (valid offline data, shadow-safe). We log the
    # placement that ACTUALLY happened — which node Slurm/score ran each job on —
    # paired with its realised reward, NOT a counterfactual RL action whose reward
    # would not match what ran. RLPD/offline RL learns from any behaviour policy,
    # so this yields clean (s, a, r, s') without executing anything on the cluster.
    n_placements = len(node_names) * n_gpus
    pending_obs: Dict[str, tuple] = {}   # job_id → (obs, mask, top_ids, obs_ts)
    in_flight:   Dict[str, tuple] = {}    # job_id → (obs, act, mask, obs_ts)

    def _node_j_of(nodelist: str) -> Optional[int]:
        for k, nm in enumerate(node_names):
            if nm and nm in (nodelist or ""):
                return k
        return None

    with open(log_path, "w") as log_fh:
        while True:
            now      = time.time()
            all_jobs = _parse_squeue(_run(["squeue", "--json"]))
            cluster  = query_cluster(node_names, n_gpus, mps_per_gpu)
            pending  = [j for j in all_jobs if "PENDING" in str(j.state).upper()]
            running  = [j for j in all_jobs if "RUNNING" in str(j.state).upper()]
            live_ids = {j.job_id for j in all_jobs}

            # (1) Snapshot the decision-state for each currently-pending job (the
            #     state a scheduler faced just before the job was placed).
            if pending:
                obs, mask, top_ids = build_obs_and_mask(
                    pending, cluster, node_names, n_gpus, now)
                for jid in top_ids:
                    if jid:
                        pending_obs[jid] = (obs.copy(), mask.copy(),
                                            list(top_ids), now)

            # (2) A job that STARTED running = the action the behaviour policy took
            #     (which job, onto which node). Encode it against its decision-obs.
            for rj in running:
                jid = rj.job_id
                if jid in pending_obs and jid not in in_flight:
                    o, m, tids, ts = pending_obs.pop(jid)
                    node_j = _node_j_of(rj.nodelist)
                    job_i  = tids.index(jid) if jid in tids else -1
                    if node_j is None or job_i < 0:
                        continue
                    act = job_i * n_placements + node_j * n_gpus + 0
                    in_flight[jid] = (o, int(act), m, ts)

            # (3) A job that FINISHED = realised reward (−JCT) → log the transition.
            if pending:
                next_obs, next_mask, _ = build_obs_and_mask(
                    pending, cluster, node_names, n_gpus, now)
            else:
                next_obs  = np.zeros(obs_dim, dtype=np.float32)
                next_mask = np.zeros(n_actions, dtype=bool); next_mask[-1] = True
            for jid in list(in_flight):
                if jid not in live_ids:
                    o, act, m, ts = in_flight.pop(jid)
                    jct = max(1.0, now - ts)
                    rew = -jct / 1000.0
                    live_buf.add(Transition(
                        obs=o, act=int(act), rew=float(rew),
                        next_obs=next_obs, done=False,
                        mask=m, next_mask=next_mask))
                    log_fh.write(json.dumps({
                        "obs": o.tolist(), "act": int(act), "rew": float(rew),
                        "next_obs": next_obs.tolist(), "done": False,
                        "mask": m.tolist(), "next_mask": next_mask.tolist(),
                        "jct_s": jct,
                    }) + "\n")
                    log_fh.flush()
                    decisions_made += 1

            print(f"[daemon] {time.strftime('%H:%M:%S')}  pending={len(pending)} "
                  f"running={len(running)}  in_flight={len(in_flight)}  "
                  f"logged={decisions_made}  buf={len(live_buf)}")
            time.sleep(poll_interval)


# ── Entry ─────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--node-name",     nargs="+", required=True,
                   help="Slurm node name(s) to observe")
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument("--mps-per-gpu",   type=int, default=MPS_PER_GPU)
    p.add_argument("--poll-interval", type=float, default=30.0)
    p.add_argument("--log-dir",       default="shadow_logs")
    p.add_argument("--buf-capacity",  type=int, default=10_000)
    args = p.parse_args(argv)

    run_daemon(
        node_names=args.node_name,
        log_dir=Path(args.log_dir),
        n_gpus=args.gpus_per_node,
        mps_per_gpu=args.mps_per_gpu,
        poll_interval=args.poll_interval,
        buf_capacity=args.buf_capacity,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
