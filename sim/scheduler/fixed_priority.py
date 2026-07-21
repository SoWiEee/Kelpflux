"""Fixed global-priority scheduler — the search primitive for the headroom study.

Every job carries a caller-supplied integer priority (lower = dispatched first);
``order`` simply sorts the pending queue by it, with ``backfill=True`` so a job
that does not currently fit is skipped rather than head-of-line blocking. This
makes the whole schedule a single permutation of job priorities, so an outer
loop can enumerate (small instances) or search (full instances) over
permutations to bound the best achievable mean JCT.

Placement is unchanged — the cluster's ``try_allocate`` picks the GPU exactly as
it does for every other scheduler — so a fixed-priority run differs from ``score``
*only* in dispatch ordering. The gap between the best permutation found and
``score`` is therefore the headroom available from smarter ordering alone.
"""
from __future__ import annotations

from typing import Dict, List

from ..cluster import Cluster
from ..loader import Job


class FixedPriorityScheduler:
    name = "fixed-priority"
    backfill = True

    def __init__(self, priority: Dict[str, int]) -> None:
        self.priority = priority

    def order(self, pending: List[Job], cluster: Cluster, now: float) -> List[Job]:
        # Unknown jobs sort last (large default), ties broken by submit then id
        # for determinism.
        return sorted(
            pending,
            key=lambda j: (self.priority.get(j.job_id, 1 << 30), j.submit_ts, j.job_id),
        )
