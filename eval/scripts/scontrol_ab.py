#!/usr/bin/env python3
"""scontrol-actuated held-job placement A/B — does giving the RL job-SELECTION
(not just node) via scontrol pin+release reduce the wait tail vs the -w bind path?

Slurm 21.08.5's slurmrestd v0.0.37 disables ``required_nodes`` in job updates
(validated: "Operation not permitted"), but ``scontrol update job=X
ReqNodeList=Y`` + ``scontrol release`` works. This drives that path.

Two arms, SAME job stream (aimix, runtime scaled to <= --target-max, burst-submitted
so the scheduler is stressed), workload = ``sleep <rt>`` holding ``mps:<req>`` (the
tail is wait-dominated, so sleep+MPS reproduces the queueing without real CUDA):
  bind     : submit all immediately; RL picks node per job (-w); Slurm picks order.
  scontrol : submit all HELD; loop /act → RL picks (job, node) → scontrol pin+release.

Usage:
  python -m eval.scripts.scontrol_ab --arm scontrol --n-jobs 50 --seed 42
Compares JCT / wait tail from sacct.
"""
from __future__ import annotations
import argparse, subprocess, time, sys
import numpy as np
from sim.loader import generate_by_family
from services.rl_scheduler.placement_controller import (
    SlurmJob, SlurmNode, build_act_payload, post_act)

NS = "slurm"; CTL = "slurm-controller-0"
LOGIN = "slurm-login-7f8cfbc48-c875f"
NODES = ["slurm-worker-gpu-rtx4070-0", "slurm-worker-gpu-rtx3080-0"]
MPS_PER_GPU = 100
SERVE = "http://localhost:8003"


def _exec(pod, script, timeout=60):
    r = subprocess.run(["kubectl", "exec", "-n", NS, pod, "--", "bash", "-lc", script],
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def gen_jobs(n, seed, target_max):
    jobs = [j for j in generate_by_family("aimix", n_jobs=n, seed=seed) if j.gpu_count <= 2]
    mx = max(j.runtime for j in jobs)
    out = []
    for j in jobs:
        rt = max(2.0, round(j.runtime / mx * target_max))
        out.append({"jid": j.job_id, "cls": j.job_class, "mps": int(j.mps_req),
                    "rt": rt, "gtype": j.gpu_type, "submit": j.submit_ts})
    return out


def submit(job, held):
    h = "-H " if held else ""
    name = f"sc{job['jid'].split('-')[-1]}"
    cmd = (f"sbatch {h}-p gpu --gres=mps:{job['mps']} --time=5 -J {name} "
           f"--wrap 'sleep {job['rt']}' 2>&1 | grep -oE '[0-9]+'")
    sid = _exec(LOGIN, cmd)
    return sid.strip()


def squeue_snapshot():
    """Return (held_pending list of SlurmJob, node free_mps dict) from squeue."""
    out = _exec(CTL, "squeue -h -o '%i|%T|%r|%b|%N|%V|%j' 2>/dev/null")
    running_mps = {n: 0 for n in NODES}
    held = []
    for line in out.splitlines():
        p = line.split("|")
        if len(p) < 7:
            continue
        jid, state, reason, tres, node, subt, name = p
        mps = 0
        for tok in tres.replace("gres:", "").split(","):
            if "mps" in tok:
                try: mps = int(tok.split(":")[-1])
                except: pass
        if state == "RUNNING" and node in running_mps:
            running_mps[node] += mps
        elif state == "PENDING" and "JobHeld" in reason:
            held.append((jid, name, mps))
    nodes = [SlurmNode(name=n, free_mps=max(0, MPS_PER_GPU * 2 - running_mps[n]),
                       running_jobs=0, gpu_type="rtx4070" if "4070" in n else "rtx3080")
             for n in NODES]
    return held, nodes


def run_scontrol_arm(jobs):
    # submit all HELD; remember slurm-id → job features + a monotone submit_ts
    meta = {}
    for i, j in enumerate(jobs):
        sid = submit(j, held=True)
        if sid: meta[sid] = {**j, "order": i}
    print(f"[scontrol] submitted {len(meta)} held jobs", flush=True)
    # loop: /act over held → pin+release chosen job
    deadline = time.time() + 25 * 60
    while time.time() < deadline:
        held, nodes = squeue_snapshot()
        held = [(sid, nm, mps) for (sid, nm, mps) in held if sid in meta]
        if not held:
            break
        sjobs = [SlurmJob(job_id=sid, name=nm, state="PENDING", reason="JobHeld",
                          mps_req=meta[sid]["mps"], gpu_count=1, gpu_type=meta[sid]["gtype"],
                          runtime=meta[sid]["rt"], submit_ts=meta[sid]["order"])
                 for (sid, nm, mps) in held]
        payload = build_act_payload(sjobs, nodes, mps_per_gpu=MPS_PER_GPU)
        try:
            act = post_act(payload, scheduler_url=SERVE, timeout=10)
        except Exception as e:
            print("  /act err", e); time.sleep(2); continue
        sel = act.get("selected_job_id"); node_j = act.get("node_j")
        if sel is None or node_j is None or sel not in meta:
            # RL abstained / no fit → release the longest-waiting held job on its fit node
            sid = held[0][0]; node = NODES[0]
        else:
            sid = sel; node = NODES[int(node_j)] if int(node_j) < len(NODES) else NODES[0]
        _exec(CTL, f"scontrol update job={sid} ReqNodeList={node} 2>&1; scontrol release {sid} 2>&1")
        time.sleep(1.5)
    return list(meta.keys())


def run_bind_arm(jobs):
    # Precompute each job's RL node from the empty-cluster snapshot (so submission is
    # a fast burst — matching scontrol's all-held-at-once — instead of being spread
    # by a per-job /act round-trip, which would unfairly de-burst the bind arm).
    empty = [SlurmNode(name=n, free_mps=MPS_PER_GPU * 2, running_jobs=0,
                       gpu_type="rtx4070" if "4070" in n else "rtx3080") for n in NODES]
    plan = []
    for j in jobs:
        sjob = SlurmJob(job_id=j["jid"], name="x", state="PENDING", reason="",
                        mps_req=j["mps"], gpu_count=1, gpu_type=j["gtype"],
                        runtime=j["rt"], submit_ts=0.0)
        try:
            act = post_act(build_act_payload([sjob], empty, mps_per_gpu=MPS_PER_GPU),
                           scheduler_url=SERVE, timeout=10)
            nj = act.get("node_j"); node = NODES[int(nj)] if nj is not None and int(nj) < len(NODES) else NODES[0]
        except Exception:
            node = NODES[0]
        plan.append((j, node))
    # fast burst submit
    ids = []
    for j, node in plan:
        name = f"sc{j['jid'].split('-')[-1]}"
        sid = _exec(LOGIN, f"sbatch -p gpu -w {node} --gres=mps:{j['mps']} --time=5 -J {name} "
                           f"--wrap 'sleep {j['rt']}' 2>&1 | grep -oE '[0-9]+'").strip()
        if sid: ids.append(sid)
    print(f"[bind] submitted {len(ids)} jobs (burst)", flush=True)
    return ids


def wait_done(ids, timeout=1800):
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = _exec(CTL, "squeue -h -o '%i' 2>/dev/null")
        live = set(out.split()) & set(ids)
        if not live:
            return
        time.sleep(10)


def collect_jct(ids):
    idcsv = ",".join(ids)
    out = _exec(CTL, f"sacct -X -P -n -j {idcsv} -o JobID,Submit,Start,End,State 2>/dev/null")
    jcts, waits = [], []
    from datetime import datetime
    def dt(s):
        try: return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timestamp()
        except: return None
    for line in out.splitlines():
        p = line.split("|")
        if len(p) < 5 or "COMPLETED" not in p[4]:
            continue
        sub, sta, end = dt(p[1]), dt(p[2]), dt(p[3])
        if sub and end: jcts.append(end - sub)
        if sub and sta: waits.append(sta - sub)
    return np.array(jcts), np.array(waits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["bind", "scontrol"], required=True)
    ap.add_argument("--n-jobs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-max", type=float, default=20.0)
    a = ap.parse_args()
    jobs = gen_jobs(a.n_jobs, a.seed, a.target_max)
    print(f"[{a.arm}] {len(jobs)} aimix jobs seed={a.seed} rt<={a.target_max}s", flush=True)
    ids = run_scontrol_arm(jobs) if a.arm == "scontrol" else run_bind_arm(jobs)
    wait_done(ids)
    jct, wait = collect_jct(ids)
    if len(jct):
        print(f"\n=== {a.arm}  n={len(jct)}  "
              f"JCT p50={np.percentile(jct,50):.0f} p95={np.percentile(jct,95):.0f} "
              f"p99={np.percentile(jct,99):.0f} max={jct.max():.0f} | "
              f"wait p95={np.percentile(wait,95):.0f} max={wait.max():.0f}")
    else:
        print(f"=== {a.arm}: no completed jobs")


if __name__ == "__main__":
    sys.exit(main())
