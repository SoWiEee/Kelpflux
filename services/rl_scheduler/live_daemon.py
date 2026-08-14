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
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sim.gym_env import (
    JOB_FEAT_DIM, TOP_K, env_dims,
    _job_feat, _gpu_feat, _topo_feat, _global_feat,
)
from sim.loader import Job as SimJob, MPS_PER_GPU, RAM_REQ_GB
from sim.cluster import Cluster as SimCluster, Allocation
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
    job_class: str = "batch"   # from sbatch --comment; drives ram_req/slo/is_inference
    slo_s:     float = 0.0     # latency deadline (inference class), from --comment
    ram_req:   float = 0.0     # host-RAM footprint (GB), from --comment


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


_SACCT_FMT = "%Y-%m-%dT%H:%M:%S"
_DONE_STATES = ("COMPLETED", "FAILED", "TIMEOUT", "CANCEL", "OUT_OF")


def _sacct_jct(job_id: str) -> Optional[float]:
    """True realized JCT = End − Submit (seconds), read from sacct.

    ``squeue --json`` retains completed jobs for ``MinJobAge`` (300s here), so
    ``now − pending_ts`` overstates JCT by up to that window — a systematic reward
    bias that defeats the point of a faithful online-log. sacct carries the real
    End, and the End−Submit delta is timezone-independent. Returns None if the job
    is not yet terminal / sacct is unavailable so the caller can fall back."""
    raw = _run(["sacct", "-X", "-P", "-n", "-j", str(job_id),
                "-o", "Submit,End,State"])
    for line in raw.split("\n"):
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        submit, end, state = parts[0], parts[1], parts[2].upper()
        if not any(s in state for s in _DONE_STATES):
            continue
        try:
            t0 = datetime.strptime(submit, _SACCT_FMT)
            t1 = datetime.strptime(end, _SACCT_FMT)
        except (ValueError, TypeError):
            return None
        return max(1.0, (t1 - t0).total_seconds())
    return None


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
        # sbatch --comment carries the training-truth fields Slurm state can't
        # otherwise expose — job_class, gpu_type, slo_s, ram_req, reported runtime.
        # Without them the obs (ram/slo/is_inference/free_ram one-hot) can't match
        # the sim the policy trained on. See live_ab_heavytail.py::sbatch_cmd.
        meta = {}
        craw = j.get("comment", "") or ""
        if craw:
            try:
                meta = json.loads(craw)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        cls = str(meta.get("cls", "batch"))
        tl = j.get("time_limit", {})
        tl_s = (tl.get("number", 0) * 60) if isinstance(tl, dict) else 0
        st_ = j.get("submit_time", {})
        submit_ts = float(st_.get("number", time.time())) if isinstance(st_, dict) else time.time()
        jobs.append(LiveJob(
            job_id=str(j.get("job_id", "")),
            mps_req=int(meta.get("mps", 0) or mps_req or MPS_PER_GPU),
            gpu_count=int(gpu_count),
            gpu_type=str(meta.get("gtype", "rtx4070")),
            runtime=float(meta.get("reported", tl_s)),
            submit_ts=submit_ts,
            state=state,
            nodelist=str(j.get("nodes", "") or ""),
            job_class=cls,
            slo_s=float(meta.get("slo", 0.0) or 0.0),
            ram_req=float(meta.get("ram", RAM_REQ_GB.get(cls, 1.0))),
        ))
    return jobs


# ── Observation builder: map live Slurm state → sim objects, then reuse the
#    canonical gym_env feature extractors. Single source of truth = no drift.
#
#    (The previous hand-rolled _job_feat/_gpu_feat_live/_topo_feat/_global_feat
#    were frozen at the pre-host-RAM obs layout — 6-d GPU feats, hardcoded 4070
#    one-hot, placeholder topo — so they emitted a 166-d obs that silently
#    diverged from the 168-d policy. Reusing gym_env's extractors makes the two
#    provably identical; see sim/tests/test_live_daemon_obs_parity.py.) ────────

# Per-card usable host RAM (GB) — MUST match training's `--hetero-cluster
# --node-ram-gb 62,5` so live free_ram_ratio lands on the scale the policy
# trained on (node-1/4070 ≈ 62GB, node-2/3080 ≈ 5GB after system + slurmd).
NODE_RAM_GB = {"rtx4070": 62.0, "rtx3080": 5.0}
NODE_RAM_GB_DEFAULT = 62.0


def _node_gpu_type(node_name: str) -> str:
    """Card identity from the Slurm node name (drives the gpu-type one-hot)."""
    n = (node_name or "").lower()
    if "3080" in n:
        return "rtx3080"
    return "rtx4070"


def _live_to_sim_job(j: LiveJob) -> SimJob:
    """Rehydrate a sim Job from a LiveJob so the canonical extractors apply."""
    return SimJob(
        job_id=j.job_id, user="live", gpu_count=int(j.gpu_count),
        gpu_type=j.gpu_type, submit_ts=float(j.submit_ts),
        runtime=float(j.runtime), mem_req=0.0, mps_req=int(j.mps_req),
        job_class=j.job_class, slo_s=float(j.slo_s), ram_req=float(j.ram_req),
    )


def _reconstruct_sim_cluster(
    running: List[LiveJob], node_names: List[str], n_gpus: int, mps_per_gpu: int
) -> SimCluster:
    """Build a sim Cluster whose per-GPU free_mps, per-node used_ram, and active
    allocations mirror the live running set — the same state sim tracks, so the
    gym extractors produce a training-consistent obs. Assumes gpus_per_node==1
    (the 2×1 deployment invariant): a running job occupies GPU 0 on each node it
    landed on (gang jobs span both nodes)."""
    node_types = [_node_gpu_type(nm) for nm in node_names]
    node_ram = [NODE_RAM_GB.get(t, NODE_RAM_GB_DEFAULT) for t in node_types]
    cl = SimCluster(
        n_nodes=len(node_names), gpus_per_node=n_gpus, mps_per_gpu=mps_per_gpu,
        node_gpu_types=node_types, node_ram_gb=node_ram,
    )
    for rj in running:
        hit = [k for k, nm in enumerate(node_names) if nm and nm in (rj.nodelist or "")]
        if not hit:
            continue
        for ni in hit:
            for gi in range(n_gpus):
                g = cl.nodes[ni].gpus[gi]
                g.free_mps = max(0, g.free_mps - int(rj.mps_req))
        cl.active[rj.job_id] = [
            Allocation(job_id=rj.job_id, node_id=ni,
                       gpu_indices=list(range(n_gpus)), mps_per_gpu=int(rj.mps_req))
            for ni in hit
        ]
        for ni in set(hit):
            cl.nodes[ni].used_ram_gb += float(rj.ram_req)
        cl.active_ram[rj.job_id] = float(rj.ram_req)
    return cl


def build_obs_and_mask(
    pending: List[LiveJob],
    running: List[LiveJob],
    node_names: List[str],
    n_gpus: int,
    mps_per_gpu: int,
    now: float,
) -> tuple[np.ndarray, np.ndarray, List[Optional[str]]]:
    """Canonical 168-d obs for the live cluster: rehydrate sim Job/Cluster from
    the live pending+running sets, then run the SAME feature extractors gym_env
    uses in ``_build_obs`` (job feats top-k, per-GPU feats, full-pending topo +
    global). Guarantees the logged obs matches what the policy trained on."""
    n_nodes      = len(node_names)
    n_placements = n_nodes * n_gpus
    n_actions    = TOP_K * n_placements + 1
    no_op        = n_actions - 1

    cl = _reconstruct_sim_cluster(running, node_names, n_gpus, mps_per_gpu)
    sim_pending = [_live_to_sim_job(j) for j in pending]
    top = sorted(sim_pending, key=lambda j: j.submit_ts)[:TOP_K]

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
    for ni in range(n_nodes):
        for gi in range(n_gpus):
            gpu_feats.append(_gpu_feat(cl, ni, gi))

    topo = _topo_feat(sim_pending, cl)
    glob = _global_feat(sim_pending, cl, now)

    obs = np.concatenate([*job_feats, *gpu_feats, topo, glob]).astype(np.float32)

    mask = np.zeros(n_actions, dtype=bool)
    for i, jb in enumerate(top):
        for nj in range(n_nodes):
            for gk in range(n_gpus):
                if cl.nodes[nj].gpus[gk].free_mps >= jb.mps_req:
                    mask[i * n_placements + nj * n_gpus + gk] = True
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
            pending  = [j for j in all_jobs if "PENDING" in str(j.state).upper()]
            running  = [j for j in all_jobs if "RUNNING" in str(j.state).upper()]
            live_ids = {j.job_id for j in all_jobs}

            # (1) Snapshot the decision-state for each currently-pending job (the
            #     state a scheduler faced just before the job was placed).
            if pending:
                obs, mask, top_ids = build_obs_and_mask(
                    pending, running, node_names, n_gpus, mps_per_gpu, now)
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
                    pending, running, node_names, n_gpus, mps_per_gpu, now)
            else:
                next_obs  = np.zeros(obs_dim, dtype=np.float32)
                next_mask = np.zeros(n_actions, dtype=bool); next_mask[-1] = True
            for jid in list(in_flight):
                if jid not in live_ids:
                    o, act, m, ts = in_flight.pop(jid)
                    # True JCT from sacct End−Submit (not now−ts, which MinJobAge
                    # squeue retention inflates by up to 300s). Fall back to the
                    # wall-clock estimate only if sacct can't answer.
                    jct = _sacct_jct(jid)
                    if jct is None:
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
