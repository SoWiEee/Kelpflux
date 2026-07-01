"""Volcano-style binpack ordering (approximation).

Volcano's ``binpack`` plugin scores nodes so that pods consolidate onto the
most-utilized node first, reducing fragmentation and keeping other nodes free
for large gang jobs. Its observable effect as a *job ordering* is best-fit-
decreasing / longest-processing-time-first: place the largest-demand jobs first
so they claim contiguous capacity before small jobs fragment it.

We approximate that here by ordering pending jobs by descending resource demand
(GPU count, then MPS request), tie-broken by submit order. This is a stateless
ordering analog, not the full node-scoring plugin — consistent with the other
sim schedulers (see ``multifactor.py``).
"""
from __future__ import annotations

from typing import List

from ..cluster import Cluster
from ..loader import Job


class VolcanoBinpackScheduler:
    name = "volcano-binpack"
    backfill = True  # skip over a non-fitting large job to keep packing

    def order(self, pending: List[Job], cluster: Cluster, now: float) -> List[Job]:
        return sorted(
            pending,
            key=lambda j: (-j.gpu_count, -j.mps_req, j.submit_ts, j.job_id),
        )
