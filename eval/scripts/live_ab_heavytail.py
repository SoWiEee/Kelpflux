"""Heavy-tail + high-contention live A/B workload generator (§4.4 / spec).

Builds a stream of real Slurm sleep jobs that (A) heavily contend for the single
GPU's MPS slots, (B) have heavy-tailed runtimes whose *estimate* is σ-noisy, so the
scheduler orders/packs under uncertainty. This is the live analogue of the sim σ
injection (eval-writeup §3.4–3.6): if the sim finding "σ↑ ⇒ RDSAC beats SAC"
transfers to live, RDSAC-cvar should cut the JCT tail (p99 / CVaR) here.

Design rationale: docs/eval-writeup.md §4.4. Engineering spec:
docs/live-ab-heavytail-spec.md. Only `philly` / `ali` (trace-derived, naturally
heavy-tailed). Reads sim.loader **read-only** — does not touch the simulator.

Invariants (paired comparison):
  * per-job common-random noise keyed on (seed, job_id) via zlib.crc32 → the same
    job gets the same (true, reported) multiplier under every arm.
  * sigma=0 ⇒ reported == true exactly (no RNG draw) → deterministic control arm.

The generator core (`gen_workload`, `peak_concurrent_mps`, `sbatch_cmd`) is pure
and unit-tested. `submit_stream` / `collect_sacct` shell out to Slurm and only run
on the live cluster.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np

from sim.loader import generate_by_family

LIVE_GPU_MPS = 100  # single RTX 4070 live GRES: mps:rtx4070:100
HEAVYTAIL_FAMILIES = ("philly", "ali")  # §4.4: trace-derived, naturally heavy-tailed


@dataclass(frozen=True)
class LiveJob:
    job_id: str
    arrival_offset_s: float   # seconds after stream start to submit
    true_runtime_s: float     # what the job actually sleeps (drives real JCT)
    reported_runtime_s: float # estimate fed to --time and /decide (σ-noisy)
    mps_req: int              # MPS slots in [1, 100]
    gpu_count: int = 1


def _job_noise(job_id: str, sigma: float, seed: int = 0) -> float:
    """Mean-preserving lognormal multiplier exp(σZ − σ²/2), E=1.

    sigma<=0 returns exactly 1.0 with NO draw → byte-identical deterministic arm.
    Common-random per (seed, job_id) so paired arms see the same multiplier.
    """
    if sigma <= 0:
        return 1.0
    key = zlib.crc32(f"{seed}:{job_id}".encode()) & 0xFFFFFFFF
    z = float(np.random.default_rng(key).standard_normal())
    return float(np.exp(sigma * z - 0.5 * sigma * sigma))


def peak_concurrent_mps(jobs: List[LiveJob]) -> int:
    """Demand-side peak: max simultaneous MPS if every job ran at submit time.

    Ignores queueing (we don't know real start times here) — it is a *demand*
    estimate proving the workload is oversubscribed by construction (> LIVE_GPU_MPS).
    """
    events: list[tuple[float, int]] = []
    for j in jobs:
        events.append((j.arrival_offset_s, +j.mps_req))
        events.append((j.arrival_offset_s + j.true_runtime_s, -j.mps_req))
    events.sort()
    cur = peak = 0
    for _t, d in events:
        cur += d
        peak = max(peak, cur)
    return peak


def gen_workload(
    family: str,
    n_jobs: int,
    *,
    seed: int = 42,
    sigma: float = 1.0,
    target_max_s: float = 180.0,
    min_runtime_s: float = 2.0,
    compress_pct: float = 95.0,
    mps_oversub: float = 4.0,
    arrival_window_frac: float = 0.1,
    arrival_mode: str = "burst",
) -> List[LiveJob]:
    """Produce a heavy-tail, oversubscribed, σ-noisy LiveJob stream.

    Steps (spec §2.3): generate_by_family → gpu_count≤1 filter → time-compress
    (preserve tail shape) → σ-noisy estimate → oversubscribe MPS to ~mps_oversub×100
    → arrivals (burst piles the queue up; poisson arrives faster than service).
    """
    if family not in HEAVYTAIL_FAMILIES:
        raise ValueError(f"heavy-tail A/B uses only {HEAVYTAIL_FAMILIES}; got {family!r}")
    jobs = generate_by_family(family, n_jobs=n_jobs, seed=seed)
    jobs = [j for j in jobs if j.gpu_count <= 1]
    if not jobs:
        return []

    runtimes = np.array([j.runtime for j in jobs], dtype=float)
    # Time-compress by a high PERCENTILE (not max): scaling by max lets a single
    # outlier squash the whole bulk to the floor (~43% of philly jobs), destroying
    # the heavy-tail shape we need for pillar (B). Anchoring on p95 keeps the bulk's
    # shape; the few jobs above p95 soft-cap at target_max_s, leaving a visible tail.
    anchor = float(np.percentile(runtimes, compress_pct))
    scale = target_max_s / anchor
    true = np.clip(runtimes * scale, min_runtime_s, target_max_s)

    ids = [str(j.job_id) for j in jobs]
    reported = np.array(
        [t * _job_noise(jid, sigma, seed) for t, jid in zip(true, ids)],
        dtype=float,
    )
    reported = np.maximum(min_runtime_s, reported)

    # Arrivals.
    arr_rng = np.random.default_rng(seed + 1)
    n = len(jobs)
    if arrival_mode == "burst":
        # Submit window ≪ durations → near-simultaneous → queue piles up.
        t_sub = arrival_window_frac * target_max_s
        arrivals = np.sort(arr_rng.uniform(0.0, t_sub, n))
    elif arrival_mode == "poisson":
        # Inter-arrival mean set so jobs arrive faster than they can be served by
        # mps_oversub×; service-ish time ≈ mean(true)·mean(mps_frac).
        mean_gap = float(true.mean()) / max(1.0, mps_oversub)
        gaps = arr_rng.exponential(mean_gap, n)
        arrivals = np.cumsum(gaps)
    else:
        raise ValueError(f"arrival_mode must be burst|poisson; got {arrival_mode!r}")

    base_mps = np.array([max(1, int(j.mps_req)) for j in jobs], dtype=float)

    # Scale MPS so demand-peak ≈ mps_oversub × capacity. Build a provisional stream
    # to measure the raw peak under these arrivals, then rescale.
    provisional = [
        LiveJob(ids[i], float(arrivals[i]), float(true[i]), float(reported[i]),
                int(np.clip(round(base_mps[i]), 1, LIVE_GPU_MPS)))
        for i in range(n)
    ]
    raw_peak = peak_concurrent_mps(provisional)
    target_peak = mps_oversub * LIVE_GPU_MPS
    mps_scale = target_peak / max(1, raw_peak)
    final_mps = np.clip(np.round(base_mps * mps_scale), 1, LIVE_GPU_MPS).astype(int)

    return [
        LiveJob(ids[i], float(arrivals[i]), float(true[i]), float(reported[i]),
                int(final_mps[i]))
        for i in range(n)
    ]


JOB_NAME_PREFIX = "htab"
HTAB_SACCT_FORMAT = "JobID,JobName%64,State,Submit,Start,End,ElapsedRaw"


def job_name(arm: str, round_idx: int, job_id: str) -> str:
    """Stable sacct join key shared by submission and collection."""
    return f"{JOB_NAME_PREFIX}_{arm}_{round_idx}_{job_id}"


def sbatch_cmd(job: LiveJob, arm: str, round_idx: int, *, partition: str = "gpu-rtx4070") -> list[str]:
    """Build the sbatch argv for one job. --time uses the (noisy) *reported*
    estimate the scheduler sees; the job actually sleeps the *true* runtime."""
    comment = json.dumps({
        "job_id": job.job_id, "true": round(job.true_runtime_s, 2),
        "reported": round(job.reported_runtime_s, 2), "mps": job.mps_req,
        "arm": arm, "round": round_idx,
    }, separators=(",", ":"))
    time_min = max(1, int(np.ceil(job.reported_runtime_s / 60.0)))
    return [
        "sbatch",
        f"--job-name={job_name(arm, round_idx, job.job_id)}",
        f"--partition={partition}",
        f"--gres=mps:{job.mps_req}",
        f"--time={time_min}",
        f"--comment={comment}",
        f"--wrap=sleep {int(round(job.true_runtime_s))}",
    ]


def parse_sacct_jct(raw_text: str) -> dict:
    """Parse `sacct -P --format=HTAB_SACCT_FORMAT` output → {JobName: record}.

    Pure (no cluster). record = {state, submit, start, end, jct, wait, elapsed}.
    JCT = End − Submit, wait = Start − Submit (None if a timestamp is missing or
    the job has not finished). Keyed by JobName so the runner joins back to its
    in-memory LiveJob stream via job_name(arm, round, job_id).
    """
    import csv
    from io import StringIO

    from sim.live_trace import parse_elapsed_seconds, parse_slurm_time

    clean = "\n".join(line for line in raw_text.splitlines() if line.strip())
    if not clean:
        return {}
    out: dict = {}
    for row in csv.DictReader(StringIO(clean), delimiter="|"):
        name = (row.get("JobName") or "").strip()
        if not name.startswith(JOB_NAME_PREFIX + "_"):
            continue
        submit = parse_slurm_time(row.get("Submit", ""))
        start = parse_slurm_time(row.get("Start", ""))
        end = parse_slurm_time(row.get("End", ""))
        jct = (end - submit) if (submit is not None and end is not None) else None
        wait = (start - submit) if (submit is not None and start is not None) else None
        out[name] = {
            "state": (row.get("State") or "").strip(),
            "submit": submit, "start": start, "end": end,
            "jct": jct, "wait": wait,
            "elapsed": parse_elapsed_seconds(row.get("ElapsedRaw", "")),
        }
    return out


def join_records(jobs: List[LiveJob], parsed: dict, arm: str, round_idx: int) -> list[dict]:
    """Join parsed sacct rows back to the workload stream for one (arm, round).

    Returns one record per job that COMPLETED with a usable JCT, carrying the
    ground-truth (true/reported/mps) plus measured jct/wait. Pure & testable.
    """
    records: list[dict] = []
    for j in jobs:
        row = parsed.get(job_name(arm, round_idx, j.job_id))
        if row is None or row["jct"] is None:
            continue
        if not row["state"].upper().startswith("COMPLETED"):
            continue
        records.append({
            "job_id": j.job_id, "arm": arm, "round": round_idx,
            "true_runtime_s": j.true_runtime_s,
            "reported_runtime_s": j.reported_runtime_s,
            "mps_req": j.mps_req,
            "jct": float(row["jct"]), "wait": float(row["wait"]) if row["wait"] is not None else None,
            "state": row["state"],
        })
    return records


def build_sacct_cmd(since: str, *, kubectl: str = "kubectl", namespace: str = "slurm",
                    controller_pod: str = "slurm-controller-0", until: Optional[str] = None) -> list[str]:
    import shlex
    inner = ["sacct", "-X", "-P", "-S", since, "--format", HTAB_SACCT_FORMAT]
    if until:
        inner.extend(["-E", until])
    return [*shlex.split(kubectl), "exec", "-n", namespace, controller_pod, "--", *inner]


def collect_sacct(since: str, *, kubectl: str = "kubectl", namespace: str = "slurm",
                  controller_pod: str = "slurm-controller-0", until: Optional[str] = None) -> dict:
    """Live wrapper: run sacct on the controller pod and parse. Cluster-only."""
    cmd = build_sacct_cmd(since, kubectl=kubectl, namespace=namespace,
                          controller_pod=controller_pod, until=until)
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"sacct failed ({proc.returncode})")
    return parse_sacct_jct(proc.stdout)


# ── live-only shell wrappers (not unit-tested against a cluster) ───────────────

def submit_stream(jobs: List[LiveJob], arm: str, round_idx: int, *,
                  dry_run: bool = True, partition: str = "gpu-rtx4070",
                  t0: Optional[float] = None,
                  exec_prefix: Optional[list[str]] = None) -> None:
    """Submit the stream honouring each job's arrival_offset. dry_run prints.

    `exec_prefix` (e.g. ["kubectl","exec","-n","slurm","pod/slurm-login-x","--"])
    is prepended so sbatch runs inside the cluster; each sbatch arg stays a single
    argv element (no shell), so `--wrap=sleep N` survives unquoted.
    """
    t0 = t0 if t0 is not None else time.time()
    for job in jobs:
        wait = job.arrival_offset_s - (time.time() - t0)
        if wait > 0 and not dry_run:
            time.sleep(wait)
        cmd = (exec_prefix or []) + sbatch_cmd(job, arm, round_idx, partition=partition)
        if dry_run:
            print(" ".join(cmd))
            continue
        for attempt in range(3):  # slurmctld can socket-timeout under burst load
            proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
            if proc.returncode == 0:
                break
            if attempt < 2:
                time.sleep(1.0 + attempt)
            else:
                sys.stderr.write(f"[htab] submit failed for {job.job_id}: {proc.stderr.strip()}\n")


def wait_drain(*, kubectl: str = "kubectl", namespace: str = "slurm",
               controller_pod: str = "slurm-controller-0", poll_s: float = 10.0,
               timeout_s: float = 3600.0, confirm: int = 2) -> bool:
    """Block until no htab_* jobs remain in squeue (cluster-only). Returns False on timeout.

    Hardened against a busy slurmctld: a FAILED squeue (non-zero rc, e.g. socket
    timeout) returns empty stdout — that must NOT be read as "drained". We only count
    an empty result when the command SUCCEEDED, and require `confirm` consecutive
    successful-empty polls before declaring drain.
    """
    import shlex
    cmd = [*shlex.split(kubectl), "exec", "-n", namespace, controller_pod, "--",
           "squeue", "-h", "-o", "%j"]
    deadline = time.time() + timeout_s
    empties = 0
    while time.time() < deadline:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            empties = 0  # query failed → unknown, do not trust as drained
            time.sleep(poll_s)
            continue
        remaining = [ln for ln in proc.stdout.splitlines()
                     if ln.strip().startswith(JOB_NAME_PREFIX + "_")]
        empties = empties + 1 if not remaining else 0
        if empties >= confirm:
            return True
        time.sleep(poll_s)
    return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Heavy-tail high-contention live A/B workload")
    p.add_argument("--family", choices=list(HEAVYTAIL_FAMILIES), default="philly")
    p.add_argument("--n-jobs", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sigma", type=float, default=1.0, help="0 = deterministic control arm")
    p.add_argument("--target-max-s", type=float, default=180.0)
    p.add_argument("--mps-oversub", type=float, default=4.0)
    p.add_argument("--arrival-mode", choices=["burst", "poisson"], default="burst")
    p.add_argument("--arm", default="score")
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--print-plan", action="store_true", help="print workload stats + dry-run sbatch")
    args = p.parse_args(argv)

    jobs = gen_workload(args.family, args.n_jobs, seed=args.seed, sigma=args.sigma,
                        target_max_s=args.target_max_s, mps_oversub=args.mps_oversub,
                        arrival_mode=args.arrival_mode)
    peak = peak_concurrent_mps(jobs)
    trues = np.array([j.true_runtime_s for j in jobs])
    print(f"[htab] family={args.family} n={len(jobs)} sigma={args.sigma} "
          f"peak_demand_mps={peak} ({peak / LIVE_GPU_MPS:.1f}× capacity)")
    print(f"[htab] true runtime  p50={np.percentile(trues,50):.1f}s "
          f"p99={np.percentile(trues,99):.1f}s max={trues.max():.1f}s")
    if args.print_plan:
        submit_stream(jobs, args.arm, args.round, dry_run=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
