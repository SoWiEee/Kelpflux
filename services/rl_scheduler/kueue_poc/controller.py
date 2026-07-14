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
import urllib.request

# The declared-runtime annotation the Job generator stamps on each Job; the sjf
# scorer reads it. Absent => treated as an unknown/large runtime (low priority).
RUNTIME_ANNOTATION = "poc.kelpflux/runtime-s"
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


def get_job_runtime(namespace: str, job_name: str) -> float | None:
    """Read the declared runtime (seconds) off the owning Job's annotation."""
    try:
        job = kubectl_json(["get", "job", job_name, "-n", namespace])
    except Exception:
        return None
    ann = (job.get("metadata", {}).get("annotations") or {}).get(RUNTIME_ANNOTATION)
    try:
        return float(ann) if ann is not None else None
    except ValueError:
        return None


def score_sjf(runtime_s: float | None) -> int:
    """Shortest-job-first: shorter declared runtime => higher priority."""
    if runtime_s is None:
        return 1  # unknown runtime sinks to the bottom
    return max(1, int(PRIORITY_CEILING - runtime_s))


def score_serve(serve_url: str, runtime_s: float | None, features: dict) -> int:
    """Best-effort RDSAC /decide call. Falls back to sjf on any error (fail-safe).

    NOTE: the faithful Job -> RDSAC observation mapping (166-dim obs, MPS/VRAM/
    wait-time/SLO-urgency features) is the next increment; this stub sends the
    minimal features it has and, crucially, never lets a serve fault block
    admission — it degrades to the heuristic score.
    """
    try:
        payload = json.dumps({"runtime_s": runtime_s, **features}).encode()
        req = urllib.request.Request(
            f"{serve_url.rstrip('/')}/decide",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            body = json.loads(resp.read())
        # Expect a scalar priority/score; adapt to the serve schema when wired.
        return int(body.get("priority", score_sjf(runtime_s)))
    except Exception as exc:  # fail-safe: never propagate
        print(f"[ctrl] serve scorer fell back to sjf ({exc})", file=sys.stderr)
        return score_sjf(runtime_s)


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
    n = 0
    for wl in pending_workloads(namespace):
        wl_name = wl["metadata"]["name"]
        job = owning_job(wl)
        runtime = get_job_runtime(namespace, job) if job else None
        if scorer == "serve":
            prio = score_serve(serve_url, runtime, {"job": job})
        else:
            prio = score_sjf(runtime)
        try:
            patch_priority(namespace, wl_name, prio)
            print(f"[ctrl] {wl_name} (job={job}, runtime={runtime}s) -> priority {prio}")
            n += 1
        except Exception as exc:  # fail-safe: leave default order, keep going
            print(f"[ctrl] skip {wl_name}: {exc}", file=sys.stderr)
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
