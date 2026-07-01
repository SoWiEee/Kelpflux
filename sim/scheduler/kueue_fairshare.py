"""Kueue-style fair-share ordering (approximation).

Kueue admits Workloads from ClusterQueues under quota, with fair-sharing so
that no single tenant monopolizes borrowable capacity. We do not model the full
ClusterQueue/Cohort borrowing tree; instead we capture the *spirit* — max-min
fairness across users — as a stateless queue ordering:

    each pending job is ranked by its position within its own user's
    submit-ordered pending jobs (0, 1, 2, ...); jobs are then served in
    rounds — every user's 1st job, then every user's 2nd, and so on.

This interleaves tenants so a single user's burst cannot head-of-line-block
others, which is the observable effect of Kueue fair-sharing on JCT. Like
``multifactor.py``, this is a deliberate approximation for the offline sim, not
a reproduction of the Kueue admission controller.
"""
from __future__ import annotations

from typing import List

from ..cluster import Cluster
from ..loader import Job


class KueueFairShareScheduler:
    name = "kueue-fairshare"
    backfill = True  # Kueue admits any workload whose quota is available

    def order(self, pending: List[Job], cluster: Cluster, now: float) -> List[Job]:
        by_user: dict = {}
        for j in sorted(pending, key=lambda j: (j.submit_ts, j.job_id)):
            by_user.setdefault(j.user, []).append(j)
        ranked = []
        for jobs in by_user.values():
            for pos, j in enumerate(jobs):
                ranked.append((pos, j.submit_ts, j.job_id, j))
        ranked.sort(key=lambda t: (t[0], t[1], t[2]))
        return [t[3] for t in ranked]
