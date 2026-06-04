"""Latency models for sim-to-live replay.

The resource simulator can run with zero dispatch latency for algorithmic
studies, or with a small live-like model calibrated from sacct traces. The
model intentionally stays simple: it estimates the delay between accepting an
allocation in the simulator and the job's observed Slurm start time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .loader import Job

Pool = Literal["cpu", "gpu"]


@dataclass
class LatencyModel:
    mode: str = "none"
    fixed_seconds: float = 0.0
    cpu_warm_seconds: float = 0.0
    cpu_cold_seconds: float = 0.0
    gpu_warm_seconds: float = 0.0
    gpu_cold_seconds: float = 0.0
    hard_placement_seconds: float = 0.0
    cold_after_seconds: float = 300.0
    active_by_pool: dict[Pool, int] = field(default_factory=lambda: {"cpu": 0, "gpu": 0})
    last_end_by_pool: dict[Pool, float] = field(default_factory=dict)

    @classmethod
    def none(cls, fixed_seconds: float = 0.0) -> "LatencyModel":
        return cls(mode="none", fixed_seconds=max(0.0, fixed_seconds))

    @classmethod
    def live_basic(
        cls,
        *,
        fixed_seconds: float = 0.0,
        cpu_warm_seconds: float = 0.0,
        cpu_cold_seconds: float = 3.0,
        gpu_warm_seconds: float = 3.0,
        gpu_cold_seconds: float = 15.0,
        hard_placement_seconds: float = 15.0,
        cold_after_seconds: float = 300.0,
    ) -> "LatencyModel":
        return cls(
            mode="live-basic",
            fixed_seconds=max(0.0, fixed_seconds),
            cpu_warm_seconds=max(0.0, cpu_warm_seconds),
            cpu_cold_seconds=max(0.0, cpu_cold_seconds),
            gpu_warm_seconds=max(0.0, gpu_warm_seconds),
            gpu_cold_seconds=max(0.0, gpu_cold_seconds),
            hard_placement_seconds=max(0.0, hard_placement_seconds),
            cold_after_seconds=max(0.0, cold_after_seconds),
        )

    def start_latency(self, job: Job, now: float) -> float:
        if self.mode == "none":
            return self.fixed_seconds
        cls = (getattr(job, "latency_class", "") or "").lower()
        if cls == "hard_placement":
            return self.fixed_seconds + self.hard_placement_seconds
        if cls == "gpu_warm":
            return self.fixed_seconds + self.gpu_warm_seconds
        if cls == "gpu_cold":
            return self.fixed_seconds + self.gpu_cold_seconds
        if cls == "cpu_warm":
            return self.fixed_seconds + self.cpu_warm_seconds
        if cls == "cpu_cold":
            return self.fixed_seconds + self.cpu_cold_seconds
        pool = pool_for_job(job)
        if self.active_by_pool.get(pool, 0) > 0:
            return self.fixed_seconds + self._warm_latency(pool)
        last_end = self.last_end_by_pool.get(pool)
        if last_end is None or now - last_end >= self.cold_after_seconds:
            return self.fixed_seconds + self._cold_latency(pool)
        return self.fixed_seconds + self._warm_latency(pool)

    def record_start(self, job: Job) -> None:
        pool = pool_for_job(job)
        self.active_by_pool[pool] = self.active_by_pool.get(pool, 0) + 1

    def record_end(self, job: Job, now: float) -> None:
        pool = pool_for_job(job)
        self.active_by_pool[pool] = max(0, self.active_by_pool.get(pool, 0) - 1)
        self.last_end_by_pool[pool] = now

    def _warm_latency(self, pool: Pool) -> float:
        return self.gpu_warm_seconds if pool == "gpu" else self.cpu_warm_seconds

    def _cold_latency(self, pool: Pool) -> float:
        return self.gpu_cold_seconds if pool == "gpu" else self.cpu_cold_seconds


def pool_for_job(job: Job) -> Pool:
    gpu_type = (getattr(job, "gpu_type", "") or "").lower()
    latency_class = (getattr(job, "latency_class", "") or "").lower()
    if job.gpu_count > 0 or "gpu" in gpu_type or "gpu" in latency_class:
        return "gpu"
    return "cpu"
