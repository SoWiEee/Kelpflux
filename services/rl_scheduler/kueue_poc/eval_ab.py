#!/usr/bin/env python3
"""Multi-method Kueue admission-ordering A/B on real hardware.

Compares three ordering policies over the SAME per-seed job stream, admitted by a
real Kueue ClusterQueue:
  fifo  : submission order (Kueue's native priority+timestamp default)
  sjf   : shortest declared runtime first (strong heuristic baseline)
  rdsac : the learned policy, ordered by rolling out /act over the pending batch
          (services/.../rdsac_order.py) — the K8s-native analogue of the Slurm path

Execution note (honest): the served RDSAC checkpoint holds each GPU in a
long-lived worker pod via an EXCLUSIVE DRA claim (verified: a second claim on the
same GPU is Unschedulable, "cannot allocate all claims"), so a native GPU pod
lane cannot co-reside with the Slurm workers. We therefore execute each job as a
**runtime-faithful proxy** (a pod that sleeps `runtime * --scale`) carrying the
real GPU/MPS features (mps_req, gpu_type, runtime) the policy reasons over. This
measures the *ordering quality* of each policy (its effect on wait time -> JCT)
on real pods + real Kueue admission + real wall-clock, while being upfront that
the compute is a proxy, not GPU co-location. Real-GPU execution needs a worker
evicted for an isolated window (see README increment 3).

Same rigor as the paper: per-seed paired deltas vs FIFO, seed-level aggregation.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time

from rdsac_order import rank_via_act

NS = "kueue-poc"
CQ = "poc-cq"
QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
LQ = "poc-lq"
CEIL = 100_000
QPATH = "/spec/resourceGroups/0/flavors/0/resources/0/nominalQuota"


def kc(args: list[str], check=True, timeout=60) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", *args], capture_output=True, text=True,
                          timeout=timeout, check=check)


def kc_json(args: list[str]) -> dict:
    return json.loads(kc([*args, "-o", "json"]).stdout)


def set_quota(n: int) -> None:
    kc(["patch", "clusterqueue", CQ, "--type=json",
        "-p", json.dumps([{"op": "replace", "path": QPATH, "value": str(n)}])])


def cleanup() -> None:
    kc(["delete", "jobs,workloads", "--all", "-n", NS, "--ignore-not-found"], check=False)
    time.sleep(3)


def gen_jobs(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    jobs = []
    for i in range(n):
        jobs.append({
            "job_id": f"j{i}",
            "idx": i,
            "runtime": round(rng.uniform(10, 60), 1),   # declared sim-seconds
            "mps_req": rng.choice([1, 2, 3, 4]),
            "gpu_type": rng.choice(["rtx4070", "rtx3080"]),
            "submit_ts": 0,
        })
    return jobs


def order_for(policy: str, jobs: list[dict], serve_url: str) -> list[str]:
    if policy == "fifo":
        return [j["job_id"] for j in sorted(jobs, key=lambda j: j["idx"])]
    if policy == "sjf":
        return [j["job_id"] for j in sorted(jobs, key=lambda j: j["runtime"])]
    if policy == "rdsac":
        return rank_via_act(serve_url, jobs)
    raise ValueError(policy)


def submit_job(name: str, j: dict, scale: float) -> None:
    sleep_s = max(1, round(j["runtime"] * scale))
    manifest = f"""
apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {NS}
  labels: {{ {QUEUE_LABEL}: {LQ}, poc-eval: "1" }}
  annotations:
    poc.kelpflux/runtime-s: "{j['runtime']}"
    poc.kelpflux/mps-req: "{j['mps_req']}"
    poc.kelpflux/gpu-type: "{j['gpu_type']}"
spec:
  backoffLimit: 0
  template:
    metadata: {{ labels: {{ poc-eval: "1" }} }}
    spec:
      restartPolicy: Never
      containers:
        - name: c
          image: busybox:1.36
          command: ["sh","-c","sleep {sleep_s}"]
          resources: {{ requests: {{ cpu: "1", memory: "32Mi" }}, limits: {{ cpu: "1", memory: "64Mi" }} }}
"""
    subprocess.run(["kubectl", "apply", "-f", "-"], input=manifest, text=True,
                   capture_output=True, check=True)


def workload_for_job(job_name: str) -> str | None:
    wls = kc_json(["get", "workloads", "-n", NS]).get("items", [])
    for wl in wls:
        for ref in wl["metadata"].get("ownerReferences", []):
            if ref.get("kind") == "Job" and ref["name"] == job_name:
                return wl["metadata"]["name"]
    return None


def patch_priority(wl_name: str, priority: int) -> None:
    kc(["patch", "workload", wl_name, "-n", NS, "--type", "merge",
        "-p", json.dumps({"spec": {"priority": priority}})])


def run_round(seed: int, policy: str, jobs: list[dict], order: list[str],
              scale: float, concurrency: int, timeout_s: int = 300) -> dict:
    cleanup()
    set_quota(0)                       # gate closed: nothing admits
    names = {j["job_id"]: f"s{seed}-{policy}-{j['job_id']}" for j in jobs}
    for j in jobs:
        submit_job(names[j["job_id"]], j, scale)
    time.sleep(3)                      # let Kueue create pending Workloads
    # assign priorities per the policy order (rank 0 = highest priority)
    for rank, jid in enumerate(order):
        wl = workload_for_job(names[jid])
        if wl:
            patch_priority(wl, CEIL - rank)
    set_quota(concurrency)             # gate open: admit in (priority, ts) order
    t0 = time.time()
    # poll completions
    done: dict[str, float] = {}
    deadline = t0 + timeout_s
    while len(done) < len(jobs) and time.time() < deadline:
        j = kc_json(["get", "jobs", "-n", NS])
        for item in j.get("items", []):
            nm = item["metadata"]["name"]
            jid = next((k for k, v in names.items() if v == nm), None)
            if jid and jid not in done and item.get("status", {}).get("succeeded", 0) >= 1:
                done[jid] = time.time() - t0     # JCT = admission-open -> completion
        time.sleep(1)
    jcts = {j["job_id"]: done.get(j["job_id"], float("nan")) for j in jobs}
    vals = [v for v in jcts.values() if v == v]  # drop nan
    return {
        "policy": policy, "seed": seed, "n_done": len(vals),
        "mean_jct": statistics.mean(vals) if vals else float("nan"),
        "p95_jct": (sorted(vals)[max(0, int(0.95 * len(vals)) - 1)] if vals else float("nan")),
        "makespan": max(vals) if vals else float("nan"),
        "order": order,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.12, help="actual sleep = runtime*scale")
    ap.add_argument("--policies", nargs="+", default=["fifo", "sjf", "rdsac"])
    ap.add_argument("--serve-url", default="http://localhost:8003")
    ap.add_argument("--out", default="/tmp/kueue_eval.json")
    args = ap.parse_args()

    results: list[dict] = []
    for seed in args.seeds:
        jobs = gen_jobs(seed, args.n_jobs)
        for pol in args.policies:
            order = order_for(pol, jobs, args.serve_url)
            r = run_round(seed, pol, jobs, order, args.scale, args.concurrency)
            results.append(r)
            print(f"[eval] seed={seed} {pol:6s} mean_jct={r['mean_jct']:6.1f}s "
                  f"p95={r['p95_jct']:6.1f}s makespan={r['makespan']:6.1f}s (done {r['n_done']}/{args.n_jobs})")
    cleanup(); set_quota(args.concurrency)

    # aggregate: per-policy mean-JCT across seeds + paired delta vs fifo
    print("\n=== 聚合(越低越好；ΔvsFIFO% + = 比 FIFO 快)===")
    print(f"{'policy':8s} {'mean_jct(s)':>14s} {'p95(s)':>10s} {'makespan(s)':>12s} {'ΔmeanJCT% vs fifo':>18s}")
    by_pol = {p: [r for r in results if r["policy"] == p] for p in args.policies}
    fifo_by_seed = {r["seed"]: r["mean_jct"] for r in by_pol.get("fifo", [])}
    agg = {}
    for pol in args.policies:
        means = [r["mean_jct"] for r in by_pol[pol]]
        p95s = [r["p95_jct"] for r in by_pol[pol]]
        mks = [r["makespan"] for r in by_pol[pol]]
        deltas = [100 * (fifo_by_seed[r["seed"]] - r["mean_jct"]) / fifo_by_seed[r["seed"]]
                  for r in by_pol[pol] if r["seed"] in fifo_by_seed and fifo_by_seed[r["seed"]]]
        dmean = statistics.mean(deltas) if deltas else 0.0
        dstd = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
        agg[pol] = {"mean_jct": statistics.mean(means), "p95": statistics.mean(p95s),
                    "makespan": statistics.mean(mks), "djct_pct": dmean, "djct_std": dstd}
        tag = "（基準）" if pol == "fifo" else f"{dmean:+.1f}±{dstd:.1f}"
        print(f"{pol:8s} {statistics.mean(means):14.1f} {statistics.mean(p95s):10.1f} "
              f"{statistics.mean(mks):12.1f} {tag:>18s}")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results, "agg": agg}, f, indent=2)
    print(f"\n[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
