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
import re
import subprocess
import sys
import time
import urllib.request as _urlreq
import zlib
from dataclasses import asdict, dataclass, field
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
    job_class: str = "batch"  # aiserve: inference / training
    slo_s: float = 0.0        # aiserve: latency-SLO deadline on JCT (>0 = inference)


@dataclass(frozen=True)
class WorkloadSpec:
    """Real-CUDA workload config (replaces `sleep N`).

    The heavy-tail job's ``true_runtime_s`` is its *idle-reference* duration:
    ``iters = round(true_runtime_s * iters_per_sec)`` fixes the WORK, calibrated
    so an unloaded reference card (the 4070, ~iters_per_sec) hits ~true_runtime_s.
    The same iters run LONGER on a slower card (the 3080) or under MPS contention
    — so heterogeneity + interference surface in JCT, unlike a `sleep` job whose
    runtime is placement-independent.

    CRN: the workload seed is derived from ``job_id`` so the same logical job is
    byte-identical across arms (paired comparison preserved).
    ``time_factor`` inflates Slurm ``--time`` (worst case ≈ slow-card × contention)
    so jobs aren't killed before finishing; applied uniformly across arms → fair.
    """
    bin_path: str = "/shared/bin/gpu_workload"
    dim: int = 4096
    vram_mb: int = 512
    iters_per_sec: float = 145.0   # idle 4070 @ dim=4096 (calibrated 2026-06-20)
    time_factor: float = 20.0      # --time = true_runtime_s × this (slow-card+contention margin)
    mps_pipe_dir: str = ""  # "" → inherit the prolog-provided device-plugin path

    def wrap(self, job: "LiveJob") -> str:
        iters = max(1, int(round(job.true_runtime_s * self.iters_per_sec)))
        seed = zlib.crc32(job.job_id.encode()) & 0xFFFFFFFF
        # The GPUs are Exclusive_Process, so concurrent CUDA jobs must route
        # through an MPS server. The cluster's NVIDIA device-plugin manages MPS
        # at /mps/nvidia.com/gpu/pipe and the 10-mps-env.sh prolog injects that
        # path into each job — so by default we DON'T override it (normal-user
        # path). mps_pipe_dir is an escape hatch for a manually-run daemon.
        env = f"CUDA_MPS_PIPE_DIRECTORY={self.mps_pipe_dir} " if self.mps_pipe_dir else ""
        return f"{env}{self.bin_path} {iters} {self.dim} {self.vram_mb} {seed}"

    def time_min(self, job: "LiveJob") -> int:
        return max(2, int(np.ceil(job.true_runtime_s * self.time_factor / 60.0)))


@dataclass(frozen=True)
class BertWorkloadSpec:
    """Real BERT GPU job (forward inference / fine-tune training) for live eval.

    Same wrap()/time_min() interface as WorkloadSpec, so the harness uses it
    interchangeably. ``job.job_class`` picks inference (BERT forward, no_grad)
    vs training (fine-tune: forward+backward+AdamW); n (batches/steps) =
    true_runtime_s × the calibrated per-mode rate; CRN seed from job_id. Uses the
    relocatable /shared/py python (torch cu124) by full path — robust in sbatch
    --wrap without needing the Lmod `module` shell function; bert_job.py loads
    BERT offline from /shared/hf_cache.
    """
    py: str = "/shared/py/bin/python3"
    script: str = "/shared/scripts/bert_job.py"
    batch_size: int = 16
    seq_len: int = 128
    infer_rate: float = 20.0   # inference batches/sec on idle 4070 (CALIBRATE)
    train_rate: float = 6.0    # fine-tune steps/sec on idle 4070 (CALIBRATE)
    time_factor: float = 25.0  # --time margin (slow card + contention)
    load_overhead_s: float = 45.0  # python+torch import + BERT load, added to --time

    def _mode_n(self, job: "LiveJob") -> tuple[str, int]:
        mode = "infer" if job.job_class == "inference" else "train"
        rate = self.infer_rate if mode == "infer" else self.train_rate
        return mode, max(1, int(round(job.true_runtime_s * rate)))

    def wrap(self, job: "LiveJob") -> str:
        mode, n = self._mode_n(job)
        seed = zlib.crc32(job.job_id.encode()) & 0xFFFFFFFF
        # gang (multi-GPU) → srun runs the payload on every allocated task/node.
        prefix = "srun " if job.gpu_count >= 2 else ""
        return (f"{prefix}{self.py} {self.script} --mode {mode} --n {n} "
                f"--batch-size {self.batch_size} --seq-len {self.seq_len} --seed {seed}")

    def time_min(self, job: "LiveJob") -> int:
        secs = job.true_runtime_s * self.time_factor + self.load_overhead_s
        return max(2, int(np.ceil(secs / 60.0)))


@dataclass(frozen=True)
class LlmWorkloadSpec:
    """Real small-LLM GPU job (batched autoregressive generation / fine-tune).

    Same wrap()/time_min() interface as WorkloadSpec, so the harness uses it
    interchangeably. ``job.job_class`` picks inference (batched generate, the
    "AI serving" shape) vs training (fine-tune: forward+backward+AdamW).
    Batched generation is *compute-bound*, so the job's runtime scales with both
    the MPS thread budget (Slurm --gres=mps:N → CUDA_MPS_ACTIVE_THREAD_PERCENTAGE)
    and card speed (4070 vs 3080) — the two levers the eval needs. n
    (rounds/steps) = true_runtime_s × the calibrated per-mode rate; CRN seed from
    job_id. Uses the relocatable /shared/py python by full path; llm_job.py loads
    Qwen2.5-0.5B offline from /shared/models/qwen05b.
    """
    py: str = "/shared/py/bin/python3"
    script: str = "/shared/scripts/llm_job.py"
    model: str = "/shared/models/qwen05b"
    # Prefill-heavy config: long prompt + short generation makes each round a
    # big compute-bound prefill matmul (long-context serving: RAG/summarisation),
    # so co-resident jobs actually contend (~2× slowdown at 2×, like the sgemm
    # bench) — decode-heavy generation is launch/bandwidth-bound and barely
    # contends (validated 2026-07-07). This is what gives the MPS-packing lever.
    # Sized to fit the 10GB 3080 under co-residency: batch=16/prompt=1024 runs
    # 2-way (mps:50) on the 3080 without OOM (validated 2026-07-07); the bigger
    # 2048/32 config OOMs at 2×. Still prefill-compute-bound → co-resident jobs
    # contend. For the 4-way mps:25 bucket the harness routes to cuBLAS instead
    # (HybridWorkloadSpec) — 4 concurrent LLM cold model-loads thrash NFS (~310s).
    batch_size: int = 16
    prompt_len: int = 1024
    gen_len: int = 8
    infer_rate: float = 0.9    # prefill rounds/sec on idle 4070 (calib 2026-07-07)
    train_rate: float = 5.0    # fine-tune steps/sec on idle 4070 (calib 2026-07-07)
    time_factor: float = 25.0  # --time margin (slow card + contention)
    load_overhead_s: float = 120.0  # python+torch import + model load; large enough
                                     # to cover the 3080's ~88s COLD NFS model read so
                                     # short jobs there don't TIMEOUT→FAIL→get dropped
                                     # (join_records keeps only COMPLETED) — the bias
                                     # that made score look worse in the first LLM run.

    def _mode_n(self, job: "LiveJob") -> tuple[str, int]:
        # Default to generation (serving): heavytail "batch" jobs become
        # variable-length generation requests. Only an explicit "training" class
        # maps to fine-tune (used by the aiserve inference+training mix).
        mode = "train" if job.job_class == "training" else "infer"
        rate = self.infer_rate if mode == "infer" else self.train_rate
        return mode, max(1, int(round(job.true_runtime_s * rate)))

    def wrap(self, job: "LiveJob") -> str:
        mode, n = self._mode_n(job)
        seed = zlib.crc32(job.job_id.encode()) & 0xFFFFFFFF
        prefix = "srun " if job.gpu_count >= 2 else ""
        return (f"{prefix}{self.py} {self.script} --mode {mode} --n {n} "
                f"--batch-size {self.batch_size} --prompt-len {self.prompt_len} "
                f"--gen-len {self.gen_len} --model {self.model} --seed {seed}")

    def time_min(self, job: "LiveJob") -> int:
        secs = job.true_runtime_s * self.time_factor + self.load_overhead_s
        return max(2, int(np.ceil(secs / 60.0)))


@dataclass(frozen=True)
class HybridWorkloadSpec:
    """Route each job to a real workload by its MPS request, so the full
    25/50/75/100 bucket structure runs on the VRAM-constrained 10GB 3080:
      mps < llm_min_mps (mps 25/50) → cuBLAS sgemm — tiny host+VRAM, no model load;
      mps >= llm_min_mps (mps 75/100) → real Qwen generation.
    Same wrap()/time_min() interface, so the harness treats it like any spec.

    Why llm_min_mps=75 (LLM never co-resides with another LLM): node-2 (the 3080
    box) has only ~7.5 GB HOST RAM, and each LLM job stages ~2-3 GB (torch cu124 +
    the 954 MB model + NFS cache) into host RAM before the GPU. Two concurrent LLM
    jobs there exhaust host RAM → OOM-killer → processes wedge in D-state
    (unkillable) → Slurm "Kill task failed" → node DRAINs. At the 75 threshold any
    two LLM jobs need mps≥75 each = ≥150 > 100 per card, so they can NEVER
    co-reside → node-2 stages at most one model at a time (~2.5 GB, fits); an
    mps:75 LLM job can still pack with an mps:25 cuBLAS job. cuBLAS (self-contained,
    negligible host RAM) carries the co-resident small buckets. (Validated
    2026-07-07: 2× LLM on the 3080 OOMs GPU too; 4× thrashes NFS ~310s.)
    """
    cublas: WorkloadSpec = field(default_factory=WorkloadSpec)
    llm: LlmWorkloadSpec = field(default_factory=LlmWorkloadSpec)
    llm_min_mps: int = 75

    def _pick(self, job: "LiveJob"):
        return self.llm if job.mps_req >= self.llm_min_mps else self.cublas

    def wrap(self, job: "LiveJob") -> str:
        return self._pick(job).wrap(job)

    def time_min(self, job: "LiveJob") -> int:
        return self._pick(job).time_min(job)


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


def gen_aiserve_workload(
    n_jobs: int = 80,
    *,
    seed: int = 42,
    n_gpus: int = 2,
    load: float = 0.7,
    inference_frac: float = 0.6,
    inf_runtime_s: float = 5.0,     # idle-ref median for inference (short)
    train_runtime_s: float = 40.0,  # idle-ref median for training (long)
    slo_factor: float = 4.0,
    inf_slo_s: float = 0.0,         # >0: FIXED inference SLO deadline (s) instead
                                    # of runtime×slo_factor. Live cuBLAS has a large
                                    # per-job overhead floor (~30s) that makes the
                                    # idle-ref×4 deadline unachievable; a fixed,
                                    # live-calibrated target (e.g. 75) is what an
                                    # actual latency SLA looks like.
    gang_frac: float = 0.0,        # >0: fraction of TRAINING jobs that are 2-GPU
                                    # gang (distributed) jobs spanning BOTH nodes. A
                                    # gang job at the head of the queue blocks BOTH
                                    # cards → head-of-line blocking that backfill
                                    # (multifactor/score) escapes but FCFS cannot;
                                    # this is where scheduling matters at 2×1.
) -> List["LiveJob"]:
    """AI-serving live workload: latency-SLO inference + best-effort training.

    Mirrors ``sim.loader.generate_ai_serving`` but emits LiveJobs with COMPRESSED
    idle-ref runtimes (inference ~secs, training ~tens of secs) so a full
    multi-round live run stays tractable while preserving the inference-behind-
    training contention the SLO metric measures. ``load`` sets offered load ρ via
    the arrival rate (moderate sweet spot, not saturated). slo_s = runtime ×
    slo_factor for inference; training is best-effort (slo_s=0).
    """
    rng = np.random.default_rng(seed)
    feats = []  # (class, gpu_count, mps, true_rt, slo)
    for _ in range(n_jobs):
        if rng.random() < inference_frac:
            rt = max(1.0, float(np.exp(np.log(inf_runtime_s) + 0.6 * rng.standard_normal())))
            mps = int(rng.choice([25, 50]))
            slo = inf_slo_s if inf_slo_s > 0 else rt * slo_factor
            feats.append(("inference", 1, mps, rt, slo))
        else:
            rt = max(5.0, float(np.exp(np.log(train_runtime_s) + 0.7 * rng.standard_normal())))
            # whole card so training fully occupies a GPU; a gang_frac of training
            # jobs are 2-GPU (both nodes) → head-of-line blocking.
            gpu = 2 if rng.random() < gang_frac else 1
            feats.append(("training", gpu, 100, rt, 0.0))
    mean_work = sum(rt * gc * (mps / 100.0) for _c, gc, mps, rt, _s in feats) / max(1, n_jobs)
    mean_gap = mean_work / max(1e-6, n_gpus * load)
    jobs: List[LiveJob] = []
    t = 0.0
    for i, (cls, gpu, mps, rt, slo) in enumerate(feats):
        t += float(rng.exponential(mean_gap)) if mean_gap > 0 else 0.0
        jobs.append(LiveJob(
            job_id=f"ai{i:04d}", arrival_offset_s=round(t, 3),
            true_runtime_s=round(rt, 3), reported_runtime_s=round(rt, 3),
            mps_req=mps, gpu_count=gpu, job_class=cls, slo_s=round(slo, 3)))
    return jobs


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
    mps_buckets: Optional[List[int]] = None,
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

    if mps_buckets:
        # Heterogeneous GPU-fraction demands snapped to clean MPS-slot buckets
        # (e.g. 25/50/75/100 = 1/2/3/4 of the 4-slot card; 100 = whole GPU, no
        # co-residency). Bucket chosen by the job's size RANK so bigger jobs ask
        # for more GPU (realistic) AND every bucket appears; CRN-stable because
        # the rank derives from the deterministic `true` runtimes (seed-fixed),
        # so the same job_id gets the same bucket across arms.
        buckets = sorted(int(b) for b in mps_buckets)
        order = np.argsort(np.argsort(true))          # 0..n-1 rank by runtime
        idx = (order * len(buckets) // max(1, n)).clip(0, len(buckets) - 1)
        final_mps = np.array([buckets[k] for k in idx], dtype=int)
        final_mps = np.clip(final_mps, 1, LIVE_GPU_MPS).astype(int)
    else:
        # Scale MPS so demand-peak ≈ mps_oversub × capacity. Build a provisional
        # stream to measure the raw peak under these arrivals, then rescale.
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
HTAB_SACCT_FORMAT = "JobID,JobName%64,State,Submit,Start,End,ElapsedRaw,NodeList"


def job_name(arm: str, round_idx: int, job_id: str) -> str:
    """Stable sacct join key shared by submission and collection."""
    return f"{JOB_NAME_PREFIX}_{arm}_{round_idx}_{job_id}"


def sbatch_cmd(job: LiveJob, arm: str, round_idx: int, *,
               partition: str = "gpu-rtx4070", hold: bool | None = None,
               nodelist: str | None = None,
               workload: "WorkloadSpec | None" = None,
               exclusive_gpu: bool = False) -> list[str]:
    """Build the sbatch argv for one job. --time uses the (noisy) *reported*
    estimate the scheduler sees; the job actually sleeps the *true* runtime.

    Placement (``-w <node>``): in the 2-node placement A/B the RL arms pick a
    node at submit time (Slurm 21.08 can't re-pin a job post-submit) and pass it
    as ``--nodelist``; the ``score`` arm omits ``-w`` → vanilla Slurm placement,
    a clean baseline.

    Hold (``-H``): legacy path for the slurmrestd placement controller (held →
    controller pins required_nodes → release). Unused by the submit-time ``-w``
    path; defaults to ``arm != "score"`` only when ``nodelist`` is not given.
    """
    if hold is None:
        hold = arm != "score" and nodelist is None
    comment = json.dumps({
        "job_id": job.job_id, "true": round(job.true_runtime_s, 2),
        "reported": round(job.reported_runtime_s, 2), "mps": job.mps_req,
        "arm": arm, "round": round_idx,
    }, separators=(",", ":"))
    # sleep mode: --time from the (noisy) reported estimate the scheduler sees.
    # CUDA mode: --time inflated for slow-card × contention so jobs aren't killed.
    if workload is not None:
        time_min = workload.time_min(job)
        wrap = workload.wrap(job)
    else:
        time_min = max(1, int(np.ceil(job.reported_runtime_s / 60.0)))
        wrap = f"sleep {int(round(job.true_runtime_s))}"
    cmd = ["sbatch"]
    if hold:
        cmd.append("-H")  # before the job spec → parsed as a submit flag
    if nodelist:
        cmd += ["-w", nodelist]  # submit-time RL node choice
    # Gang (multi-GPU) job: span gpu_count nodes (1 GPU/node at 2×1), one task per
    # node. Reserves ALL its cards at once → head-of-line blocking. The workload's
    # wrap() prefixes `srun` so the payload runs on every task.
    if job.gpu_count >= 2:
        cmd += [f"--nodes={job.gpu_count}", f"--ntasks={job.gpu_count}",
                "--ntasks-per-node=1"]
    # exclusive_gpu: each job takes the WHOLE GPU (mps:100) → one job per GPU,
    # no co-residency. Sidesteps MPS multiplexing (needed because node-2's MPS is
    # broken) while keeping the real-CUDA heterogeneity + queueing placement test.
    mps = 100 if (exclusive_gpu or job.gpu_count >= 2) else job.mps_req
    cmd += [
        f"--job-name={job_name(arm, round_idx, job.job_id)}",
        f"--partition={partition}",
        f"--gres=mps:{mps}",
        f"--time={time_min}",
        f"--comment={comment}",
        f"--wrap={wrap}",
    ]
    return cmd


# ── Submit-time RL placement (Slurm 21.08 can't re-pin post-submit) ───────────

def _mps_from_tres(tres_text: str) -> int:
    """Pull gres/mps=N out of a Cfg/AllocTRES string (0 if absent)."""
    m = re.search(r"gres/mps=(\d+)", tres_text or "")
    return int(m.group(1)) if m else 0


def parse_free_mps(scontrol_node_text: str) -> int:
    """free_mps = configured mps − allocated mps, from one `scontrol show node -o`
    line (CfgTRES=… AllocTRES=…)."""
    cfg = re.search(r"CfgTRES=(\S*)", scontrol_node_text)
    alloc = re.search(r"AllocTRES=(\S*)", scontrol_node_text)
    cfg_mps = _mps_from_tres(cfg.group(1) if cfg else "")
    alloc_mps = _mps_from_tres(alloc.group(1) if alloc else "")
    return max(0, cfg_mps - alloc_mps)


def _post_act(serve_url: str, payload: dict, timeout: float = 30.0) -> dict:
    """POST the rl-scheduler /act endpoint; returns its JSON (node_j, …)."""
    data = json.dumps(payload).encode()
    req = _urlreq.Request(serve_url.rstrip("/") + "/act", data=data,
                          headers={"Content-Type": "application/json"}, method="POST")
    with _urlreq.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


def _node_gpu_type(node_name: str) -> str:
    """Infer card type from the node name (…rtx3080… → rtx3080, else rtx4070).
    Lets the policy see card heterogeneity instead of every node looking like a
    4070 (item-1)."""
    return "rtx3080" if "3080" in node_name else "rtx4070"


def decide_node(serve_url: str, job: LiveJob, *, node_free_mps: list[int],
                node_names: list[str], mps_per_gpu: int = LIVE_GPU_MPS,
                now: float | None = None) -> str | None:
    """Ask the served RL model which node to place ``job`` on (submit-time).

    Builds a single-job /act request over the given nodes (index ↔ node_names),
    returns the chosen node name, or None if the model no-ops / picks an
    out-of-range node (caller then lets Slurm place it)."""
    payload = {
        "now": float(now if now is not None else 0.0),
        "n_nodes": len(node_names),
        "gpus_per_node": 1,
        "mps_per_gpu": mps_per_gpu,
        "pending_jobs": [{
            "job_id": job.job_id, "mps_req": job.mps_req, "gpu_count": 1,
            "gpu_type": "rtx4070", "runtime": job.reported_runtime_s,
            "submit_ts": 0.0, "can_fit": True,
        }],
        # Per-node real card type (item-1): the model now sees which node is the
        # slow 3080 vs the fast 4070, so it can route big/long jobs accordingly.
        "nodes": [{"gpus": [{"free_mps": int(fm),
                             "gpu_type": _node_gpu_type(nm)}]}
                  for fm, nm in zip(node_free_mps, node_names)],
    }
    resp = _post_act(serve_url, payload)
    nj = resp.get("node_j")
    if nj is None or not (0 <= int(nj) < len(node_names)):
        return None
    return node_names[int(nj)]


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
            "node": (row.get("NodeList") or "").strip(),
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
            "job_class": j.job_class, "slo_s": j.slo_s,
            "jct": float(row["jct"]), "wait": float(row["wait"]) if row["wait"] is not None else None,
            "submit_ts": float(row["submit"]) if row.get("submit") is not None else None,
            "end_ts": float(row["end"]) if row.get("end") is not None else None,
            "state": row["state"],
            "node": row.get("node", ""),
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
                  exec_prefix: Optional[list[str]] = None,
                  place_fn=None, placement: bool = False,
                  workload: "WorkloadSpec | None" = None,
                  exclusive_gpu: bool = False) -> None:
    """Submit the stream honouring each job's arrival_offset. dry_run prints.

    `exec_prefix` (e.g. ["kubectl","exec","-n","slurm","pod/slurm-login-x","--"])
    is prepended so sbatch runs inside the cluster; each sbatch arg stays a single
    argv element (no shell), so `--wrap=sleep N` survives unquoted.

    `place_fn(job) -> node_name | None` (RL arms in submit-time placement mode):
    when given, its result is passed as `--nodelist`, so the RL model's node
    choice is fixed at submit (Slurm 21.08 can't re-pin afterwards). None → no -w.
    """
    t0 = t0 if t0 is not None else time.time()
    for job in jobs:
        wait = job.arrival_offset_s - (time.time() - t0)
        if wait > 0 and not dry_run:
            time.sleep(wait)
        nodelist = None
        if place_fn is not None:
            try:
                nodelist = place_fn(job)
            except Exception as e:  # placement is best-effort; fall back to Slurm
                sys.stderr.write(f"[htab] place_fn failed for {job.job_id}: {e}\n")
        # Submit-time placement never holds (the RL choice is the -w node); the
        # -H path is only for the legacy slurmrestd controller.
        hold = False if placement else None
        cmd = (exec_prefix or []) + sbatch_cmd(job, arm, round_idx,
                                               partition=partition, nodelist=nodelist,
                                               hold=hold, workload=workload,
                                               exclusive_gpu=exclusive_gpu)
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
