from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collector import (  # noqa: E402
    ClusterStateCollector,
    _tres_per_node_fits,
)
from models import PartitionConfig  # noqa: E402


def _pool(**overrides):
    values = dict(
        partition="cpu",
        worker_statefulset="slurm-worker-cpu",
        min_replicas=1,
        max_replicas=4,
        scale_up_step=1,
        scale_down_step=1,
        scale_down_cooldown=60,
        cpus_per_node=4,
        memory_mb_per_node=3500,
        fallback=True,
    )
    values.update(overrides)
    return PartitionConfig(**values)


class FakeRest:
    def __init__(self, jobs):
        self.jobs = jobs

    def list_jobs(self, _partition):
        return self.jobs


def test_per_node_resource_requests_must_fit_pool_capacity():
    pool = _pool()

    assert _tres_per_node_fits("cpu=4,mem=3500M", pool)
    assert not _tres_per_node_fits("cpu=5,mem=1G", pool)
    assert not _tres_per_node_fits("cpu=2,mem=2Gc", pool)


def test_rest_queue_only_marks_schedulable_resource_waiters_for_scale_up():
    pool = _pool()
    rest = FakeRest([
        {"job_state": "PENDING", "state_reason": "Resources",
         "tres_per_node": "cpu=2,mem=2G"},
        {"job_state": "PENDING", "state_reason": "(Priority)",
         "tres_per_node": "cpu=2,mem=2G"},
        {"job_state": "PENDING", "state_reason": "Resources",
         "tres_per_node": "cpu=5,mem=1G"},
        {"job_state": "RUNNING", "state_reason": "None",
         "tres_per_node": "cpu=2,mem=2G"},
    ])
    collector = ClusterStateCollector(SimpleNamespace(), [pool], rest)

    jobs = collector._jobs_by_pool_and_state_rest("cpu")[pool.worker_statefulset]

    assert len(jobs["PENDING"]) == 3
    assert len(jobs["SCALE_PENDING"]) == 1
    assert len(jobs["RUNNING"]) == 1


def test_exec_queue_parses_parenthesized_slurm_reason():
    pool = _pool()

    class FakeClient:
        command = ""

        def exec_in_controller(self, command):
            self.command = command
            return "123|PENDING|(null)||cpu=2,mem=2G|(Resources)\n"

    client = FakeClient()
    collector = ClusterStateCollector(client, [pool])

    jobs = collector._jobs_by_pool_and_state_exec("cpu")[pool.worker_statefulset]

    assert "%R" in client.command
    assert len(jobs["SCALE_PENDING"]) == 1
