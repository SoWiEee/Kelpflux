#!/usr/bin/env python3
"""Collect live Slurm accounting and emit a simulator normalized trace.

Default usage on the live k3s cluster::

    python3 scripts/collect-live-trace.py \
      --since now-7days \
      --output runs/live/live-trace.json \
      --latency-summary runs/live/live-latency.json

The command intentionally executes ``sacct`` from ``slurm-controller-0`` because
cluster NetworkPolicy allows controller -> slurmdbd, while login -> slurmdbd may
be denied.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim.live_trace import sacct_to_normalized, write_trace  # noqa: E402

SACCT_FORMAT = "JobID,User,JobName,Partition,State,Submit,Start,End,ElapsedRaw,NodeList,AllocTRES%120,ReqTRES%120"


def build_sacct_cmd(args: argparse.Namespace) -> list[str]:
    inner = [
        "sacct",
        "-X",
        "-P",
        "-S",
        args.since,
        "--format",
        args.sacct_format,
    ]
    if args.until:
        inner.extend(["-E", args.until])
    return [
        *shlex.split(args.kubectl),
        "exec",
        "-n",
        args.namespace,
        args.controller_pod,
        "--",
        *inner,
    ]


def run_sacct(args: argparse.Namespace) -> str:
    if args.input:
        return Path(args.input).read_text()
    cmd = build_sacct_cmd(args)
    if args.verbose:
        print("+ " + " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return proc.stdout


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    frac = k - lo
    return values[lo] + (values[hi] - values[lo]) * frac


def latency_summary(jobs: list[dict]) -> dict:
    grouped: dict[str, list[float]] = defaultdict(list)
    for job in jobs:
        grouped[str(job.get("latency_class") or "unknown")].append(float(job.get("live_wait") or 0.0))
    out = {}
    for key, values in sorted(grouped.items()):
        out[key] = {
            "count": len(values),
            "mean": statistics.fmean(values),
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
            "max": max(values),
        }
    all_values = [v for values in grouped.values() for v in values]
    if all_values:
        out["all"] = {
            "count": len(all_values),
            "mean": statistics.fmean(all_values),
            "p50": percentile(all_values, 50),
            "p90": percentile(all_values, 90),
            "p95": percentile(all_values, 95),
            "max": max(all_values),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect live sacct rows into sim normalized JSON.")
    parser.add_argument("--input", help="read existing sacct -P output instead of kubectl exec")
    parser.add_argument("--output", required=True, help="normalized JSON trace output path")
    parser.add_argument("--latency-summary", help="optional JSON summary of live_wait by latency_class")
    parser.add_argument("--include-cpu", action="store_true", help="include CPU-only jobs as generic single-slot jobs")
    parser.add_argument("--completed-only", action="store_true", help="keep only jobs whose sacct State starts with COMPLETED")
    parser.add_argument("--max-live-wait-seconds", type=float, default=0.0,
                        help="drop jobs with live_wait above this threshold; 0 disables")
    parser.add_argument("--absolute-time", action="store_true", help="keep epoch submit_ts instead of rebasing to first submit")
    parser.add_argument("--since", default="now-7days")
    parser.add_argument("--until", default="")
    parser.add_argument("--namespace", default="slurm")
    parser.add_argument("--controller-pod", default="slurm-controller-0")
    parser.add_argument("--kubectl", default=os.getenv("KUBECTL", "kubectl"))
    parser.add_argument("--sacct-format", default=SACCT_FORMAT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    raw = run_sacct(args)
    jobs, stats = sacct_to_normalized(
        raw,
        include_cpu=args.include_cpu,
        relative_time=not args.absolute_time,
    )
    before_filter = len(jobs)
    if args.completed_only:
        jobs = [j for j in jobs if str(j.get("live_state", "")).upper().startswith("COMPLETED")]
    if args.max_live_wait_seconds > 0:
        jobs = [j for j in jobs if float(j.get("live_wait") or 0.0) <= args.max_live_wait_seconds]
    filtered_jobs = before_filter - len(jobs)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_trace(jobs, args.output)
    if args.latency_summary:
        Path(args.latency_summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.latency_summary).write_text(json.dumps(latency_summary(jobs), indent=2) + "\n")
    print(json.dumps({
        **stats.__dict__,
        "filtered_jobs": filtered_jobs,
        "final_jobs": len(jobs),
        "output": args.output,
        "latency_summary": args.latency_summary,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
