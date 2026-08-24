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
from eval.scripts.live_ab_heavytail import (
    LiveJob, AiMixWorkloadSpec, LlmWorkloadSpec, gen_workload)

NS = "slurm"; CTL = "slurm-controller-0"
LOGIN = "slurm-login-7f8cfbc48-c875f"
NODES = ["slurm-worker-gpu-rtx4070-0", "slurm-worker-gpu-rtx3080-0"]
MPS_PER_GPU = 100
SERVE = "http://localhost:8003"
REAL_WORKLOAD = None  # set by main() from --real-workload; None → sleep+MPS


def _to_livejob(j: dict) -> LiveJob:
    """Adapt a gen_jobs() dict to the LiveJob shape AiMixWorkloadSpec.wrap()/
    time_min() expect. true_runtime_s drives the real work; reported (σ-noisy from
    gen_workload) drives --time via time_min()."""
    return LiveJob(job_id=j["jid"], arrival_offset_s=float(j.get("arrival", 0.0)),
                   true_runtime_s=float(j["rt"]),
                   reported_runtime_s=float(j.get("reported", j["rt"])),
                   mps_req=int(j["mps"]), job_class=j.get("cls", "batch"),
                   gpu_type=j.get("gtype", "rtx4070"))


def wrap_and_time(j: dict) -> tuple[str, int]:
    """(--wrap payload, --time minutes) for one job — real AiMix workload when
    REAL_WORKLOAD is set (module-level, from --real-workload), else sleep+MPS."""
    if REAL_WORKLOAD is None:
        return f"sleep {int(round(j['rt']))}", 5
    lj = _to_livejob(j)
    return REAL_WORKLOAD.wrap(lj), REAL_WORKLOAD.time_min(lj)


def submit_stream(jobs, submit_one):
    """Submit jobs at their poisson ``arrival`` offsets (real-time paced) — the
    queue builds up over time exactly as in run_heavy150 (arrival_mode=poisson),
    instead of a single 150-job burst. submit_one(job) does the per-job sbatch
    (+priority for the priority arm). Burst streams (arrival≈0) submit near-instantly."""
    t0 = time.time()
    for j in sorted(jobs, key=lambda x: x.get("arrival", 0.0)):
        dt = j.get("arrival", 0.0) - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)
        submit_one(j)


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


def gen_jobs(n, seed, target_max, arrival_mode="poisson", oversub=2.0, sigma=1.0):
    """Same job stream as run_heavy150_aimix_5arm.sh: delegate to gen_workload
    (the canonical LiveJob generator) so arrival times, runtime compression, σ-noise
    and per-class MPS all match the prior campaign exactly. arrival_mode='poisson'
    gives spread-out arrivals (mean gap = mean(true)/oversub) — jobs arrive faster
    than served, so a queue BUILDS but is never the whole 150-job burst at once."""
    live = gen_workload("aimix", n_jobs=n, seed=seed, sigma=sigma,
                        target_max_s=target_max, mps_oversub=oversub,
                        arrival_mode=arrival_mode)
    out = []
    for j in live:
        out.append({"jid": j.job_id, "cls": j.job_class, "mps": int(j.mps_req),
                    "rt": j.true_runtime_s, "reported": j.reported_runtime_s,
                    "gtype": j.gpu_type, "arrival": j.arrival_offset_s,
                    "submit": j.arrival_offset_s})
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
    """Roll the SERVED policy forward against a deterministic, ARRIVAL-AWARE drain
    to read off the dispatch order+node it would impose, then hand that order to
    Slurm as fixed priorities (see run_priority_arm). Returns {jid: rank}, {jid: node}.

    Arrival-aware so it matches the real system under poisson arrival: a job only
    becomes eligible after its ``arrival`` offset, and the policy sees a ROLLING
    TOP-16 window of the currently-arrived-and-waiting queue (build_act_payload
    already truncates to TOP_K=16, sorted by submit_ts=arrival → oldest-waiting
    first) — exactly the top-16 select+place interface the live policy uses. The
    sim clock advances to the next event = min(next arrival, next completion).
    Reactive→static conversion (cf. sim FixedPriorityScheduler)."""
    all_jobs = {j["jid"]: dict(j) for j in jobs}
    arrivals = sorted(all_jobs.values(), key=lambda j: j.get("arrival", 0.0))
    ai = 0                              # index into arrivals not yet released
    free = {n: MPS_PER_GPU for n in NODES}
    running = []                       # (end_time, node, mps)
    t = 0.0
    pending = {}                       # jid -> job (arrived, not dispatched)
    rank, node_of, counter = {}, {}, 0

    def release_arrivals():
        nonlocal ai
        while ai < len(arrivals) and arrivals[ai].get("arrival", 0.0) <= t + 1e-9:
            pending[arrivals[ai]["jid"]] = arrivals[ai]; ai += 1

    release_arrivals()
    while pending or ai < len(arrivals) or running:
        progressed = False
        while pending:
            # submit_ts = arrival → build_act_payload's top-16 window = oldest-waiting 16.
            sjobs = [SlurmJob(job_id=jid, name="x", state="PENDING", reason="JobHeld",
                              mps_req=j["mps"], gpu_count=1, gpu_type=j["gtype"],
                              runtime=j["rt"], submit_ts=j.get("arrival", 0.0))
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
            if sel is None:  # RL abstained / no-fit → first-fit oldest-waiting
                pick = None
                for cand in sorted(pending, key=lambda s: pending[s].get("arrival", 0.0)):
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
        # advance the clock to the next event (arrival or completion)
        next_arr = arrivals[ai].get("arrival", 0.0) if ai < len(arrivals) else None
        next_end = min((e for e, _, _ in running), default=None)
        cands = [x for x in (next_arr, next_end) if x is not None]
        if not cands:
            break
        t = max(t, min(cands))
        # free any jobs that finished at/before t
        still = []
        for e, node, mps in running:
            if e <= t + 1e-9:
                free[node] += mps
            else:
                still.append((e, node, mps))
        running = still
        release_arrivals()
        if not progressed and not pending and ai >= len(arrivals) and not running:
            break
    # any never-dispatched (shouldn't happen) → append in arrival order
    for j in arrivals:
        if j["jid"] not in rank:
            rank[j["jid"]] = counter; node_of[j["jid"]] = NODES[0]; counter += 1
    return rank, node_of


RELEASE_BASE, RELEASE_SPACING = 5_000_000, 10_000  # rank 0 → highest Slurm Priority


def run_priority_arm(jobs):
    """Actuate RL's ORDERING natively: precompute the policy's dispatch order
    (arrival-aware, rolling top-16), then submit each job at its poisson arrival,
    UNHELD, and right after submit set its Slurm ``Priority`` from the precomputed
    rank via ``scontrol update`` (direct_set_prio — sticks on an unheld job; a
    priority set while held is recomputed away). Slurm's own in-process backfill
    scheduler then orders the queued jobs by that fixed priority AND places them
    freely — RL owns ORDER, Slurm owns PLACEMENT+TIMING.

    NB: we deliberately DO NOT ``-w`` pin the RL-chosen node. Pinning killed
    concurrency under poisson (a high-priority job pinned to a busy node blocks
    while the other node idles — Slurm can't backfill a pinned job elsewhere;
    under burst every node always had a ready pinned job so it didn't bite). Since
    RL PLACEMENT showed no benefit anyway (§5.2/table 6b), isolating RL ORDERING
    (priority only, Slurm places) is both the cleaner test of the §5.8 thesis and
    removes the serialization confound — placement is then identical to the
    Backfill/FCFS controls, so the comparison is purely about ordering."""
    rank, _node_of = precompute_schedule(jobs)
    sid_rank = {}

    def submit_one(j):
        jid = j["jid"]; r = rank.get(jid, len(jobs))
        name = f"sc{jid.split('-')[-1]}"
        wrap, tmin = wrap_and_time(j)
        sid = _exec(LOGIN, f"sbatch -p gpu --gres=mps:{j['mps']} --time={tmin} -J {name} "
                           f"--wrap '{wrap}' 2>&1 | grep -oE '[0-9]+'").strip()
        if sid:
            _exec(CTL, f"scontrol update jobid={sid} Priority={RELEASE_BASE - r * RELEASE_SPACING} 2>&1")
            sid_rank[sid] = r

    submit_stream(jobs, submit_one)
    print(f"[priority] {len(sid_rank)} jobs submitted+prioritized (order-only, Slurm places) "
          f"over {'poisson' if any(j.get('arrival',0)>3 for j in jobs) else 'burst'} arrivals",
          flush=True)
    return list(sid_rank)


def run_backfill_arm(jobs):
    """Baseline: submit every job UNHELD (no -w, no priority) at its poisson arrival
    time — Slurm's own scheduler picks both order AND node. Used for BOTH the Backfill
    control (SchedulerType=sched/backfill) and FCFS (sched/builtin + priority/basic);
    submission is identical, only the orchestrator's cluster config differs."""
    ids = []

    def submit_one(j):
        name = f"sc{j['jid'].split('-')[-1]}"
        wrap, tmin = wrap_and_time(j)
        sid = _exec(LOGIN, f"sbatch -p gpu --gres=mps:{j['mps']} --time={tmin} -J {name} "
                           f"--wrap '{wrap}' 2>&1 | grep -oE '[0-9]+'").strip()
        if sid: ids.append(sid)

    submit_stream(jobs, submit_one)
    print(f"[backfill] submitted {len(ids)} jobs over poisson arrivals (Slurm schedules)", flush=True)
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
    ap.add_argument("--arrival-mode", choices=["poisson", "burst"], default="poisson",
                    help="poisson (default; spread arrivals, matches run_heavy150) "
                         "or burst (all ~t0). poisson builds a queue over time.")
    ap.add_argument("--oversub", type=float, default=2.0,
                    help="MPS oversubscription factor → poisson mean gap = mean(rt)/oversub")
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="lognormal σ on the REPORTED runtime estimate (drives --time)")
    a = ap.parse_args()
    if a.real_workload:
        global REAL_WORKLOAD
        REAL_WORKLOAD = AiMixWorkloadSpec(llm=LlmWorkloadSpec(model=a.llm_model))
    if a.reload_ckpt:
        reload_serve(a.reload_ckpt)
    jobs = gen_jobs(a.n_jobs, a.seed, a.target_max, arrival_mode=a.arrival_mode,
                    oversub=a.oversub, sigma=a.sigma)
    span = max((j.get("arrival", 0.0) for j in jobs), default=0.0)
    print(f"[{a.arm}] {len(jobs)} aimix jobs seed={a.seed} rt<={a.target_max}s "
          f"arrival={a.arrival_mode} span={span:.0f}s", flush=True)
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
