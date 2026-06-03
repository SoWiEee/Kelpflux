from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import PartitionConfig, PartitionState  # noqa: E402
from policy import CheckpointAwareQueuePolicy  # noqa: E402


def _cfg(**overrides):
    values = dict(
        partition="gpu-rtx4070",
        worker_statefulset="slurm-worker-gpu-rtx4070",
        min_replicas=0,
        max_replicas=2,
        scale_up_step=1,
        scale_down_step=1,
        scale_down_cooldown=0,
    )
    values.update(overrides)
    return PartitionConfig(**values)


def _state(**overrides):
    values = dict(
        partition="gpu-rtx4070",
        worker_statefulset="slurm-worker-gpu-rtx4070",
        current_replicas=2,
        pending_jobs=0,
        running_jobs=0,
        busy_nodes=0,
    )
    values.update(overrides)
    return PartitionState(**values)


def test_running_jobs_block_scale_down_even_without_checkpoint_path():
    decision = CheckpointAwareQueuePolicy(guard_enabled=False).evaluate(
        _cfg(checkpoint_path=""),
        _state(running_jobs=1, busy_nodes=0),
        checkpoint_age_seconds=None,
    )

    assert decision.action == "keep"
    assert decision.target_replicas == 2
    assert decision.reason == "running_jobs_block_scale_down"


def test_busy_nodes_hold_safe_floor_when_no_running_jobs_are_reported():
    decision = CheckpointAwareQueuePolicy(guard_enabled=True).evaluate(
        _cfg(),
        _state(current_replicas=2, running_jobs=0, busy_nodes=1),
        checkpoint_age_seconds=None,
    )

    assert decision.action == "scale_down"
    assert decision.target_replicas == 1
    assert decision.reason == "no_pending_jobs"


def test_idle_pool_can_scale_down_to_min_replicas():
    decision = CheckpointAwareQueuePolicy(guard_enabled=True).evaluate(
        _cfg(min_replicas=0),
        _state(current_replicas=1, running_jobs=0, busy_nodes=0),
        checkpoint_age_seconds=None,
    )

    assert decision.action == "scale_down"
    assert decision.target_replicas == 0
    assert decision.reason == "no_pending_jobs"
