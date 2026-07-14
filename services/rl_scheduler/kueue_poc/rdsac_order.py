#!/usr/bin/env python3
"""Turn the RDSAC placement policy into a queue *ordering* via /act rollout.

The RDSAC policy served at ``/act`` answers "given the pending jobs and the
cluster state, which job should be placed next, and where?" (returns
``selected_job_id`` + ``node_j``/``gpu_k``, or a no-op). We convert that
one-step primitive into a full admission ordering by rolling it out on the
pending batch:

  1. ask /act on the remaining pending jobs
  2. the selected job becomes the next in the order; debit that GPU's free MPS
  3. remove it and repeat until the policy abstains (no-op) or the queue empties

On a no-op / abstain the remaining jobs fall back to FIFO (submit order) — the
same fail-safe as the Slurm path. Rank 0 = highest admission priority.

This is a faithful use of the trained policy: it is literally the sequence of
placement decisions the policy would make on the current queue, read out as an
order. Reused by both the live Kueue controller and the offline A/B harness.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Sequence

# 2×1 topology the served checkpoint was trained on (obs_dim=166, n_actions=33).
N_NODES = 2
GPUS_PER_NODE = 1
MPS_PER_GPU = 4
GPU_TYPES = ["rtx4070", "rtx3080"]  # node 0 = fast, node 1 = slow


def _post(url: str, path: str, payload: dict, timeout: float = 3.0) -> dict:
    req = urllib.request.Request(
        f"{url.rstrip('/')}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _act_request(now: float, jobs: Sequence[dict], free: list[list[int]]) -> dict:
    return {
        "now": now,
        "pending_jobs": [
            {
                "job_id": j["job_id"],
                "mps_req": int(j.get("mps_req", 1)),
                "gpu_count": int(j.get("gpu_count", 1)),
                "gpu_type": j.get("gpu_type", "rtx4070"),
                "runtime": float(j.get("runtime", 60.0)),
                "submit_ts": float(j.get("submit_ts", now)),
                "can_fit": True,
            }
            for j in jobs
        ],
        "nodes": [
            {"gpus": [{"free_mps": free[n][g], "running_jobs": 0, "gpu_type": GPU_TYPES[n]}
                      for g in range(GPUS_PER_NODE)]}
            for n in range(N_NODES)
        ],
        "n_nodes": N_NODES,
        "gpus_per_node": GPUS_PER_NODE,
        "mps_per_gpu": MPS_PER_GPU,
    }


def rank_via_act(serve_url: str, jobs: Sequence[dict]) -> list[str]:
    """Return job_ids ordered by RDSAC admission preference (rank 0 = first)."""
    remaining = list(jobs)
    if not remaining:
        return []
    now = max((float(j.get("submit_ts", 0)) for j in remaining), default=0.0)
    free = [[MPS_PER_GPU for _ in range(GPUS_PER_NODE)] for _ in range(N_NODES)]
    order: list[str] = []
    for _ in range(len(jobs)):
        resp = _post(serve_url, "/act", _act_request(now, remaining, free))
        sel = resp.get("selected_job_id")
        if sel is None:  # no-op / abstain → remaining fall back to FIFO
            order.extend(j["job_id"] for j in remaining)
            break
        order.append(sel)
        picked = next((j for j in remaining if j["job_id"] == sel), None)
        node_j, gpu_k = resp.get("node_j"), resp.get("gpu_k")
        if picked and node_j is not None and gpu_k is not None:
            free[node_j][gpu_k] = max(0, free[node_j][gpu_k] - int(picked.get("mps_req", 1)))
        remaining = [j for j in remaining if j["job_id"] != sel]
    return order


if __name__ == "__main__":  # tiny self-test against a running serve
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8003"
    demo = [
        {"job_id": "a", "mps_req": 1, "runtime": 50, "gpu_type": "rtx4070", "submit_ts": 0},
        {"job_id": "b", "mps_req": 2, "runtime": 10, "gpu_type": "rtx4070", "submit_ts": 1},
        {"job_id": "c", "mps_req": 1, "runtime": 40, "gpu_type": "rtx3080", "submit_ts": 2},
        {"job_id": "d", "mps_req": 2, "runtime": 20, "gpu_type": "rtx4070", "submit_ts": 3},
    ]
    print("RDSAC order:", rank_via_act(url, demo))
