#!/usr/bin/env python3
"""RDSAC-driven Kueue admission-ordering controller (PoC).

Kueue admits Workloads from a ClusterQueue in (priority, creationTimestamp) order
— admission order is NOT a pluggable policy, but the Workload *priority value is
mutable* (kueue.sigs.k8s.io/docs/concepts/workload_priority_class). So this
controller drives admission order by watching pending Workloads and patching each
one's ``spec.priority``. This is the K8s-native analogue of the Slurm
``job_submit.lua`` priority path (§3.3 / §4.3.3 of the paper): there RDSAC nudges
Slurm's multifactor priority; here it sets Kueue's Workload priority.

Scorer is pluggable:
  --scorer sjf    shortest declared-runtime first (heuristic stub, no RL). Proves
                  the control loop end-to-end without the RL serve dependency.
  --scorer serve  POST job features to the RDSAC ``/decide`` endpoint. The Job ->
                  observation mapping is the next increment; on ANY error this
                  path falls back to the sjf score, preserving fail-safe.

Fail-safe: any scorer/patch error leaves the Workload's priority untouched, so
Kueue keeps its default FIFO order and nothing ever blocks admission.

Poll-based (no informer / k8s client dependency): shells out to kubectl, so it
runs anywhere a kubeconfig is present.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from rdsac_order import rank_via_act

# Annotations the Job generator stamps on each Job; the scorers read them. The
# sjf scorer needs only runtime; the serve (RDSAC /act) scorer uses the full
# GPU/MPS feature set the policy was trained on.
RUNTIME_ANNOTATION = "poc.kelpflux/runtime-s"
MPS_ANNOTATION = "poc.kelpflux/mps-req"
GPUTYPE_ANNOTATION = "poc.kelpflux/gpu-type"
# Marks Workloads this controller has already prioritised, so we patch once and
# don't fight Kueue on every poll.
DONE_ANNOTATION = "poc.kelpflux/prioritised"
PRIORITY_CEILING = 100_000  # sjf maps shorter runtime -> higher priority under this


def kubectl_json(args: list[str]) -> dict:
    out = subprocess.run(
        ["kubectl", *args, "-o", "json"], capture_output=True, text=True, timeout=30
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    return json.loads(out.stdout)


def get_job_features(namespace: str, job_name: str) -> dict:
    """Read the declared GPU/MPS features off the owning Job's annotations."""
    try:
        job = kubectl_json(["get", "job", job_name, "-n", namespace])
    except Exception:
        return {}
    ann = job.get("metadata", {}).get("annotations") or {}
    def _f(key, cast, default):
        try:
            return cast(ann[key])
        except (KeyError, ValueError, TypeError):
            return default
    return {
        "runtime": _f(RUNTIME_ANNOTATION, float, None),
        "mps_req": _f(MPS_ANNOTATION, int, 1),
        "gpu_type": ann.get(GPUTYPE_ANNOTATION, "rtx4070"),
    }


def score_sjf(runtime_s: float | None) -> int:
    """Shortest-job-first: shorter declared runtime => higher priority."""
    if runtime_s is None:
        return 1  # unknown runtime sinks to the bottom
    return max(1, int(PRIORITY_CEILING - runtime_s))


def priorities_from_order(order: list[str]) -> dict[str, int]:
    """Map a ranked job_id list (rank 0 = first) to descending priorities."""
    return {jid: PRIORITY_CEILING - rank for rank, jid in enumerate(order)}


def pending_workloads(namespace: str) -> list[dict]:
    """Workloads not yet admitted (no QuotaReserved/Admitted condition True)."""
    wls = kubectl_json(["get", "workloads", "-n", namespace]).get("items", [])
    out = []
    for wl in wls:
        conds = {c["type"]: c["status"] for c in wl.get("status", {}).get("conditions", [])}
        if conds.get("Admitted") == "True" or conds.get("QuotaReserved") == "True":
            continue
        if (wl["metadata"].get("annotations") or {}).get(DONE_ANNOTATION) == "1":
            continue
        out.append(wl)
    return out


def owning_job(wl: dict) -> str | None:
    for ref in wl["metadata"].get("ownerReferences", []):
        if ref.get("kind") == "Job":
            return ref["name"]
    return None


def patch_priority(namespace: str, wl_name: str, priority: int) -> None:
    body = json.dumps({"spec": {"priority": priority}})
    subprocess.run(
        ["kubectl", "patch", "workload", wl_name, "-n", namespace,
         "--type", "merge", "-p", body],
        capture_output=True, text=True, timeout=30, check=True,
    )
    # Mark as prioritised so we don't re-patch on the next poll.
    ann = json.dumps({"metadata": {"annotations": {DONE_ANNOTATION: "1"}}})
    subprocess.run(
        ["kubectl", "patch", "workload", wl_name, "-n", namespace,
         "--type", "merge", "-p", ann],
        capture_output=True, text=True, timeout=30,
    )


def reconcile_once(namespace: str, scorer: str, serve_url: str) -> int:
    pending = pending_workloads(namespace)
    if not pending:
        return 0
    # Gather each pending Workload with its owning Job's declared features.
    items = []
    for wl in pending:
        job = owning_job(wl)
        feats = get_job_features(namespace, job) if job else {}
        items.append({"wl": wl["metadata"]["name"], "job_id": job or wl["metadata"]["name"], **feats})

    if scorer == "serve":
        # Roll the RDSAC /act policy out over the pending batch to get an order,
        # then map rank -> priority. rank_via_act falls back to FIFO for any job
        # the policy abstains on, and on total serve failure we fall back to sjf.
        try:
            order = rank_via_act(serve_url, [
                {"job_id": it["job_id"], "mps_req": it.get("mps_req", 1),
                 "gpu_type": it.get("gpu_type", "rtx4070"),
                 "runtime": it.get("runtime") or 60.0, "submit_ts": 0}
                for it in items
            ])
            prio_by_job = priorities_from_order(order)
        except Exception as exc:  # fail-safe: whole serve path degrades to sjf
            print(f"[ctrl] serve path fell back to sjf ({exc})", file=sys.stderr)
            prio_by_job = {it["job_id"]: score_sjf(it.get("runtime")) for it in items}
    else:  # sjf
        prio_by_job = {it["job_id"]: score_sjf(it.get("runtime")) for it in items}

    n = 0
    for it in items:
        prio = prio_by_job.get(it["job_id"], 1)
        try:
            patch_priority(namespace, it["wl"], prio)
            print(f"[ctrl] {it['wl']} (job={it['job_id']}, mps={it.get('mps_req')}, "
                  f"runtime={it.get('runtime')}s) -> priority {prio}")
            n += 1
        except Exception as exc:  # fail-safe: leave default order, keep going
            print(f"[ctrl] skip {it['wl']}: {exc}", file=sys.stderr)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--namespace", default="kueue-poc")
    ap.add_argument("--scorer", choices=["sjf", "serve"], default="sjf")
    ap.add_argument("--serve-url", default="http://localhost:8003")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--once", action="store_true", help="single pass then exit")
    args = ap.parse_args()

    print(f"[ctrl] Kueue ordering controller: ns={args.namespace} scorer={args.scorer}")
    while True:
        try:
            reconcile_once(args.namespace, args.scorer, args.serve_url)
        except Exception as exc:  # never die on a transient kubectl error
            print(f"[ctrl] reconcile error (continuing): {exc}", file=sys.stderr)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
