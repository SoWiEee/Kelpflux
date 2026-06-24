"""Per-job and cluster-wide metric collection.

Tracks (submit, start, end) times for each job and emits:

- **JCT**       — completion time minus submit time, mean / p50 / p90 / p95
- **wait**      — start minus submit
- **slowdown**  — JCT / max(runtime, 60s) (Philly convention)
- **makespan**  — last end minus first submit
- **utilization** — mean MPS-slots-used over the makespan
- **bf_rate**   — fraction of jobs that started while an earlier-submitted
  job was still pending (a coarse backfill proxy used for §M6 later)
"""
from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class JobRecord:
    job_id: str
    user: str
    gpu_count: int
    mps_req: int
    submit_ts: float
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    runtime: float = 0.0
    slo_s: float = 0.0        # latency-SLO deadline (s); >0 = JCT target (inference)
    job_class: str = "batch"  # inference / training / batch

    @property
    def slo_violated(self) -> Optional[bool]:
        """True/False for SLO jobs (slo_s>0); None for best-effort jobs."""
        if self.slo_s <= 0 or self.jct is None:
            return None
        return self.jct > self.slo_s

    @property
    def jct(self) -> Optional[float]:
        if self.end_ts is None:
            return None
        return self.end_ts - self.submit_ts

    @property
    def wait(self) -> Optional[float]:
        if self.start_ts is None:
            return None
        return self.start_ts - self.submit_ts

    @property
    def slowdown(self) -> Optional[float]:
        if self.jct is None:
            return None
        return self.jct / max(self.runtime, 60.0)


@dataclass
class MetricCollector:
    records: Dict[str, JobRecord] = field(default_factory=dict)
    util_samples: List = field(default_factory=list)  # list of (t, frac)

    def record_submit(self, job_id: str, **kwargs) -> None:
        self.records[job_id] = JobRecord(job_id=job_id, **kwargs)

    def record_start(self, job_id: str, t: float) -> None:
        self.records[job_id].start_ts = t

    def record_end(self, job_id: str, t: float) -> None:
        self.records[job_id].end_ts = t

    def sample_util(self, t: float, frac: float) -> None:
        self.util_samples.append((t, frac))

    # -----
    def summary(self) -> dict:
        finished = [r for r in self.records.values() if r.end_ts is not None]
        if not finished:
            return {"n_jobs": 0}
        jcts = [r.jct for r in finished]
        waits = [r.wait for r in finished if r.wait is not None]
        slows = [r.slowdown for r in finished if r.slowdown is not None]
        first_submit = min(r.submit_ts for r in finished)
        last_end = max(r.end_ts for r in finished)
        # time-weighted utilization
        util = 0.0
        if len(self.util_samples) >= 2:
            total = 0.0
            wsum = 0.0
            for (t0, f0), (t1, _f1) in zip(self.util_samples, self.util_samples[1:]):
                dt = t1 - t0
                if dt > 0:
                    wsum += f0 * dt
                    total += dt
            util = wsum / total if total > 0 else 0.0
        # bf_rate: jobs that started while an earlier-submitted job was still pending
        ordered = sorted(finished, key=lambda r: r.submit_ts)
        bf = 0
        for i, r in enumerate(ordered):
            if r.start_ts is None:
                continue
            for prev in ordered[:i]:
                if prev.start_ts is None or prev.start_ts > r.start_ts:
                    bf += 1
                    break
        # SLO: violation rate over jobs carrying a deadline (slo_s>0, i.e. inference)
        slo_jobs = [r for r in finished if r.slo_s > 0]
        slo_violations = [r for r in slo_jobs if r.slo_violated]
        # per-class JCT breakdown (inference latency vs training throughput)
        by_class: dict = {}
        for cls in sorted({r.job_class for r in finished}):
            cj = [r.jct for r in finished if r.job_class == cls]
            by_class[cls] = {
                "n": len(cj),
                "jct_mean": statistics.fmean(cj),
                "jct_p99": _pct(cj, 99),
            }
        out = {
            "n_jobs": len(finished),
            "makespan": last_end - first_submit,
            "jct_mean": statistics.fmean(jcts),
            "jct_p50": _pct(jcts, 50),
            "jct_p90": _pct(jcts, 90),
            "jct_p95": _pct(jcts, 95),
            "jct_p99": _pct(jcts, 99),
            "wait_mean": statistics.fmean(waits) if waits else 0.0,
            "wait_p90": _pct(waits, 90) if waits else 0.0,
            "slowdown_mean": statistics.fmean(slows) if slows else 0.0,
            "slowdown_p90": _pct(slows, 90) if slows else 0.0,
            "slowdown_p95": _pct(slows, 95) if slows else 0.0,
            "slowdown_p99": _pct(slows, 99) if slows else 0.0,
            "utilization": util,
            "bf_rate": bf / len(finished),
            "slo_n": len(slo_jobs),
            "slo_violations": len(slo_violations),
            "slo_violation_rate": (len(slo_violations) / len(slo_jobs)) if slo_jobs else 0.0,
        }
        if by_class:
            out["by_class"] = by_class
        return out

    def write_per_job_csv(self, path: str) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["job_id", "user", "gpu_count", "mps_req", "job_class", "slo_s",
                 "submit_ts", "start_ts", "end_ts",
                 "runtime", "wait", "jct", "slowdown", "slo_violated"])
            for r in sorted(self.records.values(), key=lambda x: x.submit_ts):
                w.writerow([
                    r.job_id, r.user, r.gpu_count, r.mps_req, r.job_class,
                    f"{r.slo_s:.3f}",
                    f"{r.submit_ts:.3f}",
                    f"{r.start_ts:.3f}" if r.start_ts is not None else "",
                    f"{r.end_ts:.3f}" if r.end_ts is not None else "",
                    f"{r.runtime:.3f}",
                    f"{r.wait:.3f}" if r.wait is not None else "",
                    f"{r.jct:.3f}" if r.jct is not None else "",
                    f"{r.slowdown:.6f}" if r.slowdown is not None else "",
                    "" if r.slo_violated is None else int(r.slo_violated),
                ])


def _pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)
