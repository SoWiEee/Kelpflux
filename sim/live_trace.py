"""Convert Slurm sacct output into simulator normalized traces.

The live cluster restricts slurmdbd access to the controller pod, so the CLI
wrapper in ``scripts/collect-live-trace.py`` runs ``sacct`` from there and feeds
its parsable output into this module. The output remains compatible with
``sim.loader.load_auto``: extra live-only fields are preserved in JSON but ignored
by the simulator loader.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from typing import Iterable, Sequence

from .loader import MPS_PER_GPU

_GPU_TYPES = ("rtx4070", "rtx4080", "rtx4090", "a10", "h100", "v100", "p100")
_TERMINAL_OK = ("COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED")


@dataclass(frozen=True)
class LiveTraceStats:
    raw_rows: int
    emitted_jobs: int
    skipped_steps: int
    skipped_cpu: int
    skipped_state: int
    skipped_time: int


def _get(row: dict[str, str], *names: str) -> str:
    lowered = {k.lower(): v for k, v in row.items()}
    for name in names:
        if name in row and row[name]:
            return row[name]
        value = lowered.get(name.lower())
        if value:
            return value
    return ""


def parse_slurm_time(value: str) -> float | None:
    value = (value or "").strip()
    if not value or value in {"Unknown", "None", "N/A"}:
        return None
    # sacct normally emits local ISO timestamps: 2026-06-03T08:58:17.
    for fmt in (None, "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.fromisoformat(value) if fmt is None else datetime.strptime(value, fmt)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def parse_elapsed_seconds(value: str) -> float | None:
    value = (value or "").strip()
    if not value or value in {"Unknown", "None", "N/A"}:
        return None
    if value.isdigit():
        return float(value)
    days = 0
    if "-" in value:
        left, value = value.split("-", 1)
        if left.isdigit():
            days = int(left)
    parts = value.split(":")
    try:
        if len(parts) == 3:
            h, m, s = (int(float(x)) for x in parts)
            return float(days * 86400 + h * 3600 + m * 60 + s)
        if len(parts) == 2:
            m, s = (int(float(x)) for x in parts)
            return float(days * 86400 + m * 60 + s)
    except ValueError:
        return None
    return None


def parse_tres_int(text: str, names: Sequence[str]) -> int:
    if not text:
        return 0
    for name in names:
        patterns = (
            rf"(?:^|,){re.escape(name)}=(\d+)",
            rf"(?:^|,){re.escape(name)}:(?:[^,:]+:)?(\d+)",
            rf"(?:^|,)gres/{re.escape(name)}=(\d+)",
        )
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return int(m.group(1))
    return 0


def infer_gpu_type(*texts: str) -> str:
    joined = " ".join(t or "" for t in texts).lower()
    for gpu_type in _GPU_TYPES:
        if gpu_type in joined:
            return gpu_type
    return "rtx4070"


def _is_step(job_id: str) -> bool:
    return "." in job_id


def _state_ok(state: str) -> bool:
    upper = (state or "").upper()
    return any(upper.startswith(prefix) for prefix in _TERMINAL_OK)


def rows_from_sacct(text: str) -> list[dict[str, str]]:
    clean = "\n".join(line for line in text.splitlines() if line.strip())
    if not clean:
        return []
    reader = csv.DictReader(StringIO(clean), delimiter="|")
    return [dict(row) for row in reader]


def sacct_to_normalized(
    text: str,
    *,
    include_cpu: bool = False,
    relative_time: bool = True,
    min_runtime_seconds: float = 1.0,
) -> tuple[list[dict], LiveTraceStats]:
    rows = rows_from_sacct(text)
    jobs: list[dict] = []
    skipped_steps = skipped_cpu = skipped_state = skipped_time = 0

    for row in rows:
        job_id = _get(row, "JobIDRaw", "JobID", "JobId")
        if not job_id or _is_step(job_id):
            skipped_steps += 1
            continue
        state = _get(row, "State")
        if not _state_ok(state):
            skipped_state += 1
            continue
        submit = parse_slurm_time(_get(row, "Submit"))
        start = parse_slurm_time(_get(row, "Start"))
        end = parse_slurm_time(_get(row, "End"))
        elapsed = parse_elapsed_seconds(_get(row, "ElapsedRaw", "Elapsed"))
        if submit is None or start is None:
            skipped_time += 1
            continue
        runtime = (end - start) if end is not None and end >= start else (elapsed or 0.0)
        runtime = max(float(min_runtime_seconds), float(runtime))
        tres = ",".join(
            x for x in (
                _get(row, "AllocTRES", "AllocTres"),
                _get(row, "ReqTRES", "ReqTres"),
                _get(row, "TRESReq", "TresReq"),
                _get(row, "TRES_PER_NODE", "TresPerNode"),
            ) if x
        )
        mps_req = parse_tres_int(tres, ("mps",))
        gpu_count = parse_tres_int(tres, ("gpu",))
        if gpu_count <= 0 and mps_req > 0:
            gpu_count = 1
        if gpu_count <= 0 and not include_cpu:
            skipped_cpu += 1
            continue
        if gpu_count <= 0:
            gpu_count = 1
        if mps_req <= 0:
            mps_req = MPS_PER_GPU
        partition = _get(row, "Partition")
        node_list = _get(row, "NodeList", "Nodelist")
        jobs.append({
            "job_id": str(job_id),
            "user": _get(row, "User") or "live",
            "gpu_count": int(gpu_count),
            "gpu_type": infer_gpu_type(partition, node_list, tres, _get(row, "JobName")),
            "submit_ts": float(submit),
            "runtime": float(runtime),
            "mem_req": float(parse_tres_int(tres, ("mem",))),
            "mps_req": int(mps_req),
            "live_start_ts": float(start),
            "live_end_ts": float(end) if end is not None else None,
            "live_wait": float(max(0.0, start - submit)),
            "live_state": state,
            "partition": partition,
            "node_list": node_list,
            "alloc_tres": _get(row, "AllocTRES", "AllocTres"),
            "req_tres": _get(row, "ReqTRES", "ReqTres", "TRESReq", "TresReq"),
            "latency_class": classify_latency(partition=partition, node_list=node_list, state=state),
        })

    if relative_time and jobs:
        first_submit = min(j["submit_ts"] for j in jobs)
        for job in jobs:
            job["submit_ts"] = float(job["submit_ts"] - first_submit)
            job["live_start_ts"] = float(job["live_start_ts"] - first_submit)
            if job["live_end_ts"] is not None:
                job["live_end_ts"] = float(job["live_end_ts"] - first_submit)

    jobs.sort(key=lambda j: (j["submit_ts"], str(j["job_id"])))
    return jobs, LiveTraceStats(
        raw_rows=len(rows),
        emitted_jobs=len(jobs),
        skipped_steps=skipped_steps,
        skipped_cpu=skipped_cpu,
        skipped_state=skipped_state,
        skipped_time=skipped_time,
    )


def classify_latency(*, partition: str, node_list: str, state: str = "") -> str:
    part = (partition or "").lower()
    nodes = (node_list or "").lower()
    state_l = (state or "").lower()
    if "held" in state_l or "hold" in state_l:
        return "hard_placement"
    if "gpu" in part or "gpu" in nodes:
        return "gpu_warm"
    return "cpu_warm"


def write_trace(jobs: Iterable[dict], path: str) -> None:
    with open(path, "w") as fh:
        json.dump(list(jobs), fh, indent=2)
        fh.write("\n")
