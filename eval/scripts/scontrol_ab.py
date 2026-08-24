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
import argparse, json, subprocess, time, sys, urllib.request
import numpy as np
from sim.loader import generate_by_family
from services.rl_scheduler.placement_controller import (
    SlurmJob, SlurmNode, build_act_payload, post_act)
from eval.scripts.live_ab_heavytail import LiveJob, AiMixWorkloadSpec, LlmWorkloadSpec

NS = "slurm"; CTL = "slurm-controller-0"
LOGIN = "slurm-login-7f8cfbc48-c875f"
NODES = ["slurm-worker-gpu-rtx4070-0", "slurm-worker-gpu-rtx3080-0"]
MPS_PER_GPU = 100
SERVE = "http://localhost:8003"
REAL_WORKLOAD = None  # set by main() from --real-workload; None → sleep+MPS


def _to_livejob(j: dict) -> LiveJob:
    """Adapt a gen_jobs() dict to the LiveJob shape AiMixWorkloadSpec.wrap()/
    time_min() expect. true_runtime_s = reported_runtime_s = rt (no σ-noise here
    — the scontrol/priority arms don't model estimate error, matching §5.8)."""
    return LiveJob(job_id=j["jid"], arrival_offset_s=0.0,
                   true_runtime_s=float(j["rt"]), reported_runtime_s=float(j["rt"]),
                   mps_req=int(j["mps"]), job_class=j.get("cls", "batch"),
                   gpu_type=j.get("gtype", "rtx4070"))


def wrap_and_time(j: dict) -> tuple[str, int]:
    """(--wrap payload, --time minutes) for one job — real AiMix workload when
    REAL_WORKLOAD is set (module-level, from --real-workload), else sleep+MPS."""
    if REAL_WORKLOAD is None:
        return f"sleep {j['rt']}", 5
    lj = _to_livejob(j)
    return REAL_WORKLOAD.wrap(lj), REAL_WORKLOAD.time_min(lj)


def _exec(pod, script, timeout=60):
    r = subprocess.run(["kubectl", "exec", "-n", NS, pod, "--", "bash", "-lc", script],
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def reload_serve(ckpt_path, serve=SERVE, timeout=30):
    """Hot-swap the served checkpoint (serve.py /reload). Absolute path so the
    serve process — running from an arbitrary cwd — always resolves it."""
    import os
    ckpt = os.path.abspath(ckpt_path)
    data = json.dumps({"checkpoint": ckpt}).encode()
    req = urllib.request.Request(serve.rstrip("/") + "/reload", data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode() or "{}")
    print(f"[reload] {out.get('variant')} obs={out.get('obs_dim')} "
          f"act={out.get('n_actions')} ← {ckpt}", flush=True)
    return out


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
    wrap, tmin = wrap_and_time(job)
    cmd = (f"sbatch {h}-p gpu --gres=mps:{job['mps']} --time={tmin} -J {name} "
           f"--wrap '{wrap}' 2>&1 | grep -oE '[0-9]+'")
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
    # per-node capacity = MPS_PER_GPU (each node has 1 GPU × mps:100, CfgTRES
    # gres/mps=100, OverSubscribe=NO). NOT ×2 — that stale 2-GPU/node assumption
    # (a) made RL over-release so the overflow queued in Slurm PRIORITY order,
    # overriding RL's ordering, and (b) fed the model free_mps/mps_per_gpu ratios
    # up to 2.0 it never saw in training (mps:100 → max 1.0) — out-of-distribution.
    nodes = [SlurmNode(name=n, free_mps=max(0, MPS_PER_GPU - running_mps[n]),
                       running_jobs=0, gpu_type="rtx4070" if "4070" in n else "rtx3080")
             for n in NODES]
    return held, nodes


def run_scontrol_arm(jobs, deadline_min=25):
    # submit all HELD; remember slurm-id → job features + a monotone submit_ts
    meta = {}
    for i, j in enumerate(jobs):
        sid = submit(j, held=True)
        if sid: meta[sid] = {**j, "order": i}
    print(f"[scontrol] submitted {len(meta)} held jobs", flush=True)
    # BATCH-DRAIN loop (not one-release-per-1.5s — that serial throttle injected a
    # ~2-3s/job artificial wait that dominated short jobs and made RL look far worse
    # than backfill, which bursts all jobs so Slurm fills capacity at once). Each
    # cycle: snapshot free_mps once, then let RL repeatedly pick its top held job and
    # place it while capacity remains (RL owns ORDER + NODE), decrementing a LOCAL
    # free_mps tally; actuate the whole batch in ONE kubectl exec; then wait briefly
    # for running jobs to free MPS and re-snapshot. This mirrors backfill's fill-now
    # behaviour so the comparison isolates scheduling quality, not release cadence.
    deadline = time.time() + deadline_min * 60
    while time.time() < deadline:
        held, nodes = squeue_snapshot()
        held = [(sid, nm, mps) for (sid, nm, mps) in held if sid in meta]
        if not held:
            break
        free = {n.name: n.free_mps for n in nodes}
        remaining = {sid: (nm, mps) for (sid, nm, mps) in held}
        batch = []  # (sid, node)
        # fill available capacity: RL picks order+node; fall back to longest-wait/first-fit
        while remaining:
            sjobs = [SlurmJob(job_id=sid, name=nm, state="PENDING", reason="JobHeld",
                              mps_req=meta[sid]["mps"], gpu_count=1, gpu_type=meta[sid]["gtype"],
                              runtime=meta[sid]["rt"], submit_ts=meta[sid]["order"])
                     for sid, (nm, mps) in remaining.items()]
            fnodes = [SlurmNode(name=n.name, free_mps=int(free[n.name]), running_jobs=0,
                                gpu_type=n.gpu_type) for n in nodes]
            try:
                act = post_act(build_act_payload(sjobs, fnodes, mps_per_gpu=MPS_PER_GPU),
                               scheduler_url=SERVE, timeout=10)
            except Exception as e:
                print("  /act err", e); act = {}
            sel = act.get("selected_job_id"); node_j = act.get("node_j")
            if sel in remaining and node_j is not None and 0 <= int(node_j) < len(NODES):
                sid = sel; node = NODES[int(node_j)]
            else:
                sid = None  # RL abstained → fall to first-fit below
            need = meta[sid]["mps"] if sid else None
            # if RL's pick doesn't fit its node, or RL abstained, first-fit the
            # longest-waiting job onto any node with room (keeps capacity full).
            if sid is None or free.get(node, 0) < need:
                placed = False
                for cand in sorted(remaining, key=lambda s: meta[s]["order"]):
                    for n in NODES:
                        if free[n] >= meta[cand]["mps"]:
                            sid, node, need = cand, n, meta[cand]["mps"]; placed = True; break
                    if placed:
                        break
                if not placed:
                    break  # nothing fits any node right now → wait for capacity
            batch.append((sid, node)); free[node] -= need; remaining.pop(sid)
        if batch:
            cmd = "; ".join(f"scontrol update job={sid} ReqNodeList={node} 2>&1; scontrol release {sid} 2>&1"
                            for sid, node in batch)
            _exec(CTL, cmd, timeout=120)
        time.sleep(2)  # let running jobs free MPS before the next snapshot
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
        wrap, tmin = wrap_and_time(j)
        sid = _exec(LOGIN, f"sbatch -p gpu -w {node} --gres=mps:{j['mps']} --time={tmin} -J {name} "
                           f"--wrap '{wrap}' 2>&1 | grep -oE '[0-9]+'").strip()
        if sid: ids.append(sid)
    print(f"[bind] submitted {len(ids)} jobs (burst)", flush=True)
    return ids


def precompute_schedule(jobs, serve=SERVE):
    """Simulate a draining 2-node cluster with the SERVED policy to get RL's full
    dispatch order+node WITHOUT live latency: repeatedly /act over remaining jobs
    given the current free_mps, advancing sim time as running jobs finish. Returns
    {jid: (rank, node)} with rank 0 = dispatched first.

    This is the reactive→static conversion (cf. sim FixedPriorityScheduler): the
    policy is reactive, so we roll it forward against a deterministic drain to read
    off the order it WOULD impose, then hand that order to Slurm as fixed priorities
    (see run_priority_arm). Same /act semantics, capacity (100), and first-fit
    fallback as the live path — only the cluster is in-memory."""
    free = {n: MPS_PER_GPU for n in NODES}
    running = []           # (end_time, node, mps)
    t = 0.0
    # gen_jobs dicts have no "order"; assign a monotone submit index (the policy's
    # obs uses submit_ts for age, matching the live scontrol arm's meta[...] = order).
    pending = {j["jid"]: {**j, "order": i} for i, j in enumerate(jobs)}
    rank, node_of, counter = {}, {}, 0
    while pending:
        progressed = False
        while pending:
            sjobs = [SlurmJob(job_id=jid, name="x", state="PENDING", reason="JobHeld",
                              mps_req=j["mps"], gpu_count=1, gpu_type=j["gtype"],
                              runtime=j["rt"], submit_ts=j["order"])
                     for jid, j in pending.items()]
            fnodes = [SlurmNode(name=n, free_mps=int(free[n]), running_jobs=0,
                                gpu_type="rtx4070" if "4070" in n else "rtx3080") for n in NODES]
            try:
                act = post_act(build_act_payload(sjobs, fnodes, mps_per_gpu=MPS_PER_GPU),
                               scheduler_url=serve, timeout=10)
            except Exception:
                act = {}
            sel = act.get("selected_job_id"); node_j = act.get("node_j")
            node = need = None
            if sel in pending and node_j is not None and 0 <= int(node_j) < len(NODES):
                sel_node = NODES[int(node_j)]
                if free[sel_node] >= pending[sel]["mps"]:
                    jid, node, need = sel, sel_node, pending[sel]["mps"]
                else:
                    sel = None  # RL pick doesn't fit its node → first-fit
            else:
                sel = None
            if sel is None:  # RL abstained / no-fit → first-fit longest-waiting
                pick = None
                for cand in sorted(pending, key=lambda s: pending[s]["order"]):
                    for n in NODES:
                        if free[n] >= pending[cand]["mps"]:
                            pick = (cand, n, pending[cand]["mps"]); break
                    if pick:
                        break
                if not pick:
                    break  # nothing fits current free_mps → advance time
                jid, node, need = pick
            rank[jid] = counter; node_of[jid] = node; counter += 1
            free[node] -= need
            running.append((t + pending[jid]["rt"], node, need))
            pending.pop(jid); progressed = True
        if pending:
            if not running:  # deadlock guard: nothing running and nothing fits
                for cand in sorted(pending, key=lambda s: pending[s]["order"]):
                    rank[cand] = counter; node_of[cand] = NODES[0]; counter += 1
                break
            running.sort()
            end, node, mps = running.pop(0); t = end; free[node] += mps
    return rank, node_of


def run_priority_arm(jobs):
    """Fair in-process actuation of RL's schedule: precompute order+node, submit all
    HELD (nothing starts), then in ONE controller script RELEASE all + set each job's
    Priority via ``scontrol update`` (direct_set_prio — survives release, unlike a
    priority set while still held, which Slurm recomputes away). Slurm's own in-process
    backfill scheduler then actuates by that fixed priority at native speed — RL owns
    ORDER+NODE, Slurm owns TIMING. No Python poll loop, so the actuation latency that
    made every policy look identical in the hold/poll-release path is gone."""
    rank, node_of = precompute_schedule(jobs)
    sid_rank = {}
    for j in jobs:
        jid = j["jid"]; node = node_of.get(jid, NODES[0])
        name = f"sc{jid.split('-')[-1]}"
        wrap, tmin = wrap_and_time(j)
        sid = _exec(LOGIN, f"sbatch -H -p gpu -w {node} --gres=mps:{j['mps']} --time={tmin} -J {name} "
                           f"--wrap '{wrap}' 2>&1 | grep -oE '[0-9]+'").strip()
        if sid:
            sid_rank[sid] = rank.get(jid, len(jobs))
    # rank 0 → highest priority. SPACING (10000) >> the max age contribution under
    # fast-aging (PriorityWeightAge=1000 × age_norm≤1), so even if direct_set_prio
    # did NOT freeze aging, RL's order can't be reshuffled by wait time. BASE keeps
    # the lowest-ranked job (rank≤149) positive and distinct.
    BASE, SPACING = 5_000_000, 10_000
    ids = " ".join(sid_rank)
    setp = "; ".join(f"scontrol update jobid={sid} Priority={BASE - r * SPACING} 2>&1"
                     for sid, r in sid_rank.items())
    _exec(CTL, f"scontrol release {ids} 2>&1; {setp}", timeout=240)
    print(f"[priority] {len(sid_rank)} jobs released+prioritized (Slurm actuates in-process)",
          flush=True)
    return list(sid_rank)


def run_backfill_arm(jobs):
    """Baseline: submit every job UNHELD with no -w and no scontrol — Slurm's own
    scheduler picks both order AND node. Used for BOTH the Backfill control
    (cluster running SchedulerType=sched/backfill) and the FCFS control (cluster
    running SchedulerType=sched/builtin + PriorityType=priority/basic, no
    backfill-skip) — the submission is identical; only the orchestrator's cluster
    config differs between the two phases (see run_step3_prio.sh)."""
    ids = []
    for j in jobs:
        name = f"sc{j['jid'].split('-')[-1]}"
        wrap, tmin = wrap_and_time(j)
        sid = _exec(LOGIN, f"sbatch -p gpu --gres=mps:{j['mps']} --time={tmin} -J {name} "
                           f"--wrap '{wrap}' 2>&1 | grep -oE '[0-9]+'").strip()
        if sid: ids.append(sid)
    print(f"[backfill] submitted {len(ids)} jobs (Slurm schedules)", flush=True)
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
    ap.add_argument("--arm", choices=["bind", "scontrol", "backfill", "priority", "fcfs"], required=True)
    ap.add_argument("--n-jobs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-max", type=float, default=20.0)
    ap.add_argument("--deadline-min", type=float, default=25.0,
                    help="scontrol drain deadline (min); scale up with --n-jobs")
    ap.add_argument("--reload-ckpt", default="",
                    help="POST /reload this checkpoint into serve before the run "
                         "(the per-arm RL model for scontrol/bind); empty = no reload")
    ap.add_argument("--out-json", default="",
                    help="write {arm,seed,jct[],wait[]} here for aggregation")
    ap.add_argument("--real-workload", action="store_true",
                    help="run real AiMix GPU jobs (BERT/ResNet/Qwen/cuBLAS, §5.2/6b) "
                         "instead of sleep+MPS (§5.8's wait-dominated proxy)")
    ap.add_argument("--llm-model", default="/shared/models/qwen05b",
                    help="Qwen model path for --real-workload's llm class")
    a = ap.parse_args()
    if a.real_workload:
        global REAL_WORKLOAD
        REAL_WORKLOAD = AiMixWorkloadSpec(llm=LlmWorkloadSpec(model=a.llm_model))
    if a.reload_ckpt:
        reload_serve(a.reload_ckpt)
    jobs = gen_jobs(a.n_jobs, a.seed, a.target_max)
    print(f"[{a.arm}] {len(jobs)} aimix jobs seed={a.seed} rt<={a.target_max}s", flush=True)
    if a.arm == "scontrol":
        ids = run_scontrol_arm(jobs, deadline_min=a.deadline_min)
    elif a.arm == "priority":
        ids = run_priority_arm(jobs)
    elif a.arm in ("backfill", "fcfs"):
        ids = run_backfill_arm(jobs)
    else:
        ids = run_bind_arm(jobs)
    wait_done(ids)
    jct, wait = collect_jct(ids)
    if len(jct):
        print(f"\n=== {a.arm}  n={len(jct)}  "
              f"JCT p50={np.percentile(jct,50):.0f} p95={np.percentile(jct,95):.0f} "
              f"p99={np.percentile(jct,99):.0f} max={jct.max():.0f} | "
              f"wait p95={np.percentile(wait,95):.0f} max={wait.max():.0f}")
    else:
        print(f"=== {a.arm}: no completed jobs")
    if a.out_json:
        with open(a.out_json, "w") as fh:
            json.dump({"arm": a.arm, "seed": a.seed, "n_jobs": a.n_jobs,
                       "reload_ckpt": a.reload_ckpt,
                       "jct": jct.tolist(), "wait": wait.tolist()}, fh)
        print(f"[out] wrote {a.out_json} (n={len(jct)})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
