from __future__ import annotations

import os
import sys
from collections import defaultdict
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import PartitionConfig, PartitionState, ScalingDecision  # noqa: E402
from scale_actions import ScaleActionsMixin  # noqa: E402


class FakeClient:
    def __init__(self):
        self.resumed: list[str] = []
        self.drained: list[str] = []
        self.down: list[tuple[str, str]] = []
        self.annotations: list[tuple[str, str, str, str]] = []
        self.cpu_alloc: dict[str, int] = defaultdict(int)

    def resume_slurm_node(self, node_name: str) -> None:
        self.resumed.append(node_name)

    def drain_slurm_node(self, node_name: str) -> None:
        self.drained.append(node_name)

    def down_slurm_node(self, node_name: str, reason: str = "") -> None:
        self.down.append((node_name, reason))

    def get_node_cpu_alloc(self, node_name: str) -> int:
        return self.cpu_alloc[node_name]

    def set_annotation(self, resource: str, name: str, key: str, value: str) -> None:
        self.annotations.append((resource, name, key, value))


class FakeActuator:
    def __init__(self):
        self.patches: list[tuple[str, int]] = []

    def patch_replicas(self, statefulset: str, replicas: int) -> None:
        self.patches.append((statefulset, replicas))


class FakeLogger:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, event_type: str, **fields) -> None:
        self.events.append((event_type, fields))


class Harness(ScaleActionsMixin):
    def __init__(self):
        self.client = FakeClient()
        self.actuator = FakeActuator()
        self.cfg = SimpleNamespace(policy_name="checkpoint_aware_queue")
        self.last_scale_up_at = defaultdict(float)
        self._provisioning = {}
        self._draining_nodes = {}
        self._draining_started = {}
        self.logger = FakeLogger()


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


def test_scale_up_resumes_target_ordinals_before_patching_replicas():
    h = Harness()
    h._draining_nodes["slurm-worker-gpu-rtx4070"] = {"slurm-worker-gpu-rtx4070-1"}

    h._do_scale_up(
        _cfg(),
        _state(current_replicas=0, pending_jobs=1),
        ScalingDecision(target_replicas=2, action="scale_up", reason="pending_jobs"),
        "slurm-worker-gpu-rtx4070",
        now=123.0,
    )

    assert sorted(h.client.resumed) == [
        "slurm-worker-gpu-rtx4070-0",
        "slurm-worker-gpu-rtx4070-1",
    ]
    assert h.actuator.patches == [("slurm-worker-gpu-rtx4070", 2)]
    assert "slurm-worker-gpu-rtx4070" not in h._draining_nodes


def test_scale_down_marks_removed_idle_nodes_down_after_patch():
    h = Harness()

    h._do_scale_down(
        _cfg(),
        _state(current_replicas=2, running_jobs=0),
        ScalingDecision(target_replicas=0, action="scale_down", reason="no_pending_jobs"),
        "slurm-worker-gpu-rtx4070",
        cooldown_elapsed=999.0,
        cooldown_remaining=0,
    )

    assert sorted(h.client.drained) == [
        "slurm-worker-gpu-rtx4070-0",
        "slurm-worker-gpu-rtx4070-1",
    ]
    assert h.actuator.patches == [("slurm-worker-gpu-rtx4070", 0)]
    assert sorted(h.client.down) == [
        ("slurm-worker-gpu-rtx4070-0", "operator-scale-down"),
        ("slurm-worker-gpu-rtx4070-1", "operator-scale-down"),
    ]
    scale_events = [fields for event, fields in h.logger.events if event == "scale_action"]
    assert scale_events[-1]["slurm_nodes_down"] == [
        "slurm-worker-gpu-rtx4070-0",
        "slurm-worker-gpu-rtx4070-1",
    ]
