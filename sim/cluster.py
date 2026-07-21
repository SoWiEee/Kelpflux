"""Cluster model with MPS-aware GPU slots.

A ``Cluster`` is a list of nodes. Each node has ``gpus_per_node`` GPUs and
each GPU exposes ``mps_per_gpu`` slots. Jobs allocate either:

- whole GPUs (``mps_req == mps_per_gpu``), spread across nodes if needed; or
- a single GPU's MPS fraction (``gpu_count == 1`` + ``mps_req < mps_per_gpu``).

The simulator is intentionally simple — no preemption, no fragmentation
heuristics. Schedulers call :py:meth:`Cluster.try_allocate`; it returns a
list of ``Allocation`` records or ``None``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .loader import Job, MPS_PER_GPU


@dataclass
class Allocation:
    job_id: str
    node_id: int
    gpu_indices: List[int]  # which GPUs on that node
    mps_per_gpu: int        # slots reserved on each listed GPU


@dataclass
class _GPU:
    free_mps: int


@dataclass
class _Node:
    node_id: int
    gpus: List[_GPU]
    # Relative compute speed (1.0 = reference card). NOTE: on this cluster the two
    # cards are near-equal in compute (measured RTX 3080 ≈ 1.1× the RTX 4070, not
    # the slower card the old default 0.25 implied); the dominant, real asymmetry
    # is host RAM (below), not speed. ``speed`` is kept for legacy/homogeneous
    # paths, but per-(card, job_class) speed now lives in gym_env.SPEED_MATRIX.
    speed: float = 1.0
    # Card identity — drives the obs gpu-type one-hot directly (NOT via speed, so
    # near-equal speeds don't erase the signal) and selects the SPEED_MATRIX row.
    gpu_type: str = "rtx4070"
    # Host RAM budget usable by jobs (GB). Default huge = no constraint (legacy).
    # The real cluster is wildly asymmetric: node-1 (4070) ~62GB usable vs node-2
    # (3080) ~5GB after system/slurmd — the true placement lever on this platform.
    ram_gb: float = 1e9
    used_ram_gb: float = 0.0

    def free_ram_gb(self) -> float:
        return self.ram_gb - self.used_ram_gb

    def free_ram_ratio(self) -> float:
        return (self.free_ram_gb() / self.ram_gb) if self.ram_gb > 0 else 1.0

    def free_whole_gpus(self, mps_per_gpu: int) -> List[int]:
        return [i for i, g in enumerate(self.gpus) if g.free_mps == mps_per_gpu]

    def free_mps_total(self) -> int:
        return sum(g.free_mps for g in self.gpus)


@dataclass
class Cluster:
    n_nodes: int
    gpus_per_node: int
    mps_per_gpu: int = MPS_PER_GPU
    # Per-node relative speed; None → homogeneous (all 1.0, legacy behaviour).
    node_speeds: Optional[List[float]] = None
    # Per-node card identity and host-RAM budget (GB). None → legacy defaults
    # (all rtx4070, unbounded RAM) so existing callers/tests are unaffected.
    node_gpu_types: Optional[List[str]] = None
    node_ram_gb: Optional[List[float]] = None
    # Host-RAM overcommit factor (opt-in). 1.0 (default) = the hard OOM gate: a
    # placement whose nominal ram_req would exceed the node budget is refused
    # (structural guard, unchanged). >1.0 loosens the gate so nominally-over-budget
    # co-residency becomes *placeable* — the actual OOM then materialises
    # stochastically at run time (see gym_env's realized-peak model), turning a
    # hard constraint into a penalised tail event the policy must learn to avoid.
    ram_overcommit: float = 1.0
    nodes: List[_Node] = field(init=False)
    active: dict = field(init=False)      # job_id -> List[Allocation]
    active_ram: dict = field(init=False)  # job_id -> ram_req (GB), for release

    def __post_init__(self) -> None:
        speeds = self.node_speeds or [1.0] * self.n_nodes
        gtypes = self.node_gpu_types or ["rtx4070"] * self.n_nodes
        rams = self.node_ram_gb or [1e9] * self.n_nodes
        self.nodes = [
            _Node(i, [_GPU(self.mps_per_gpu) for _ in range(self.gpus_per_node)],
                  speed=float(speeds[i]), gpu_type=str(gtypes[i]),
                  ram_gb=float(rams[i]))
            for i in range(self.n_nodes)
        ]
        self.active = {}
        self.active_ram = {}

    def _ram_ok(self, node_i: int, ram_req: float) -> bool:
        """True if node_i can host ram_req more GB within its (overcommit-scaled) budget.

        With ``ram_overcommit == 1.0`` this is the exact hard gate. A larger factor
        admits nominally-over-budget placements so the stochastic-OOM model can
        decide their fate at run time instead of the allocator refusing outright.
        """
        n = self.nodes[node_i]
        return (n.used_ram_gb + ram_req) <= n.ram_gb * self.ram_overcommit + 1e-9

    # ----- introspection -----
    def total_gpus(self) -> int:
        return self.n_nodes * self.gpus_per_node

    def total_mps(self) -> int:
        return self.total_gpus() * self.mps_per_gpu

    def used_mps(self) -> int:
        return self.total_mps() - sum(n.free_mps_total() for n in self.nodes)

    def utilization(self) -> float:
        return self.used_mps() / self.total_mps() if self.total_mps() else 0.0

    def free_mps_per_node(self) -> List[int]:
        return [n.free_mps_total() for n in self.nodes]

    # ----- allocation -----
    def can_allocate(self, job: Job) -> bool:
        return self._plan(job) is not None

    def can_allocate_on(self, job: Job, node_i: int, gpu_i: int) -> bool:
        """Return True if job can be placed on the specific (node, gpu)."""
        return self._plan_on(job, node_i, gpu_i) is not None

    def try_allocate(self, job: Job) -> Optional[List[Allocation]]:
        plan = self._plan(job)
        if plan is None:
            return None
        for alloc in plan:
            node = self.nodes[alloc.node_id]
            for gi in alloc.gpu_indices:
                node.gpus[gi].free_mps -= alloc.mps_per_gpu
        self._reserve_ram(job.job_id, plan, getattr(job, "ram_req", 0.0))
        self.active[job.job_id] = plan
        return plan

    def _reserve_ram(self, job_id: str, plan: List[Allocation], ram_req: float) -> None:
        for ni in {a.node_id for a in plan}:   # charge each distinct host once
            self.nodes[ni].used_ram_gb += ram_req
        self.active_ram[job_id] = ram_req

    def try_allocate_on(
        self, job: Job, node_i: int, gpu_i: int
    ) -> Optional[List[Allocation]]:
        """Allocate job on the specified (node, gpu). Returns plan or None."""
        plan = self._plan_on(job, node_i, gpu_i)
        if plan is None:
            return None
        for alloc in plan:
            node = self.nodes[alloc.node_id]
            for gi in alloc.gpu_indices:
                node.gpus[gi].free_mps -= alloc.mps_per_gpu
        self._reserve_ram(job.job_id, plan, getattr(job, "ram_req", 0.0))
        self.active[job.job_id] = plan
        return plan

    def release(self, job_id: str) -> None:
        plan = self.active.pop(job_id, None)
        if plan is None:
            return
        for alloc in plan:
            node = self.nodes[alloc.node_id]
            for gi in alloc.gpu_indices:
                node.gpus[gi].free_mps += alloc.mps_per_gpu
        ram = self.active_ram.pop(job_id, 0.0)
        for ni in {a.node_id for a in plan}:
            self.nodes[ni].used_ram_gb = max(0.0, self.nodes[ni].used_ram_gb - ram)

    # ----- planners -----
    def _plan_on(self, job: Job, node_i: int, gpu_i: int) -> Optional[List[Allocation]]:
        """Plan placement on a specific (node, gpu).

        For MPS jobs (gpu_count==1, mps_req < mps_per_gpu): place on gpu_i of node_i.
        For whole-GPU jobs (gpu_count>1 or mps_req==mps_per_gpu): gpu_i is the
        *starting* GPU; we greedily fill remaining GPUs from node_i first, then
        other nodes.
        """
        if node_i >= self.n_nodes or gpu_i >= self.gpus_per_node:
            return None

        ram_req = getattr(job, "ram_req", 0.0)
        if job.gpu_count == 1 and job.mps_req < self.mps_per_gpu:
            # MPS fractional — must fit on the requested GPU AND the node's RAM.
            # The RAM gate is what makes a would-be-OOM placement structurally
            # illegal (it propagates to can_allocate_on → the action mask).
            if (self.nodes[node_i].gpus[gpu_i].free_mps >= job.mps_req
                    and self._ram_ok(node_i, ram_req)):
                return [Allocation(job.job_id, node_i, [gpu_i], job.mps_req)]
            return None

        # Whole-GPU job: start from (node_i, gpu_i), fill greedily
        needed = job.gpu_count
        plan: List[Allocation] = []
        node_order = [node_i] + [n for n in range(self.n_nodes) if n != node_i]
        for ni in node_order:
            if needed <= 0:
                break
            if not self._ram_ok(ni, ram_req):
                continue   # this node is out of host RAM for the job
            node = self.nodes[ni]
            free = node.free_whole_gpus(self.mps_per_gpu)
            if ni == node_i and gpu_i in free:
                # Prefer the requested GPU first
                ordered = [gpu_i] + [g for g in free if g != gpu_i]
            else:
                ordered = free
            take = ordered[: min(needed, len(ordered))]
            if take:
                plan.append(Allocation(job.job_id, ni, take, self.mps_per_gpu))
                needed -= len(take)
        return plan if needed <= 0 else None

    def _plan(self, job: Job) -> Optional[List[Allocation]]:
        ram_req = getattr(job, "ram_req", 0.0)
        # Single-GPU fractional MPS request
        if job.gpu_count == 1 and job.mps_req < self.mps_per_gpu:
            for node in self.nodes:
                if not self._ram_ok(node.node_id, ram_req):
                    continue   # node out of host RAM → skip (structural OOM guard)
                # First-fit: pick the GPU with the smallest matching residual
                best = None
                for gi, g in enumerate(node.gpus):
                    if g.free_mps >= job.mps_req:
                        residual = g.free_mps - job.mps_req
                        if best is None or residual < best[1]:
                            best = (gi, residual)
                if best is not None:
                    return [Allocation(job.job_id, node.node_id, [best[0]], job.mps_req)]
            return None

        # Whole-GPU job — may span nodes
        needed = job.gpu_count
        plan: List[Allocation] = []
        # prefer fewer nodes (best-fit by free whole-GPUs)
        ranked = sorted(
            range(self.n_nodes),
            key=lambda i: -len(self.nodes[i].free_whole_gpus(self.mps_per_gpu)),
        )
        for ni in ranked:
            if needed <= 0:
                break
            if not self._ram_ok(ni, ram_req):
                continue
            free_gpus = self.nodes[ni].free_whole_gpus(self.mps_per_gpu)
            if not free_gpus:
                continue
            take = free_gpus[: min(needed, len(free_gpus))]
            plan.append(Allocation(job.job_id, ni, take, self.mps_per_gpu))
            needed -= len(take)
        if needed > 0:
            return None
        return plan
