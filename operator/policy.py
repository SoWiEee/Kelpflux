"""Scaling policy.

CheckpointAwareQueuePolicy evaluates cluster state and returns a ScalingDecision.
It is the only place that contains scaling business logic — no I/O, no side effects.
"""

from __future__ import annotations

from models import PartitionConfig, PartitionState, ScalingDecision


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


class CheckpointAwareQueuePolicy:
    def __init__(self, guard_enabled: bool):
        self.guard_enabled = guard_enabled

    def evaluate(
        self,
        partition_cfg: PartitionConfig,
        state: PartitionState,
        checkpoint_age_seconds: int | None,
        missing_since_seconds: float | None = None,
    ) -> ScalingDecision:
        if state.pending_jobs > 0:
            target = clamp(
                state.current_replicas + partition_cfg.scale_up_step,
                partition_cfg.min_replicas,
                partition_cfg.max_replicas,
            )
            return self._to_decision(state.current_replicas, target, "pending_jobs")

        safe_floor = max(partition_cfg.min_replicas, state.busy_nodes)
        candidate_target = clamp(
            state.current_replicas - partition_cfg.scale_down_step,
            safe_floor,
            partition_cfg.max_replicas,
        )

        if candidate_target < state.current_replicas and state.running_jobs > 0:
            return ScalingDecision(state.current_replicas, "keep", "running_jobs_block_scale_down")

        return self._to_decision(state.current_replicas, candidate_target, "no_pending_jobs")

    @staticmethod
    def _to_decision(current: int, target: int, reason: str) -> ScalingDecision:
        if target > current:
            return ScalingDecision(target, "scale_up", reason)
        if target < current:
            return ScalingDecision(target, "scale_down", reason)
        return ScalingDecision(target, "keep", reason)
