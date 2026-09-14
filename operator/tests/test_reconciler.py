from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import otel  # noqa: E402
import reconciler  # noqa: E402
from models import PartitionConfig, PartitionState  # noqa: E402
from policy import CheckpointAwareQueuePolicy  # noqa: E402
from reconciler import ReconcilerMixin  # noqa: E402
from scale_actions import ScaleActionsMixin  # noqa: E402


class FakeClient:
    def __init__(self, ready=0):
        self.ready = ready
        self.annotations = []
        self.future = []
        self.resumed = []

    def get_ready_replicas(self, _key):
        return self.ready

    def set_annotation(self, resource, key, name, value):
        self.annotations.append((resource, key, name, value))

    def future_slurm_node(self, node, reason=""):
        self.future.append((node, reason))

    def resume_slurm_node(self, node):
        self.resumed.append(node)


class FakeActuator:
    def __init__(self):
        self.patches = []

    def patch_replicas(self, key, replicas):
        self.patches.append((key, replicas))


class FakeLogger:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))


class Harness(ReconcilerMixin, ScaleActionsMixin):
    def __init__(self, ready=0):
        self.client = FakeClient(ready)
        self.actuator = FakeActuator()
        self.collector = SimpleNamespace(get_checkpoint_age_seconds=lambda _path: None)
        self.cfg = SimpleNamespace(
            policy_name="checkpoint_aware_queue",
            provisioning_timeout_seconds=10,
            provisioning_retry_seconds=20,
            checkpoint_guard_enabled=False,
        )
        self.policy = CheckpointAwareQueuePolicy(guard_enabled=False)
        self.rest = None
        self.last_scale_action_at = {"slurm-worker-cpu": 80.0}
        self._provisioning = {}
        self._provisioning_retry_at = {}
        self._checkpoint_missing_since = {}
        self._job_pending_trace = {}
        self._job_running_trace = {}
        self.logger = FakeLogger()


def _pool():
    return PartitionConfig(
        partition="cpu",
        worker_statefulset="slurm-worker-cpu",
        min_replicas=0,
        max_replicas=2,
        scale_up_step=1,
        scale_down_step=1,
        scale_down_cooldown=0,
    )


def _state(**overrides):
    values = dict(
        partition="cpu",
        worker_statefulset="slurm-worker-cpu",
        current_replicas=1,
        pending_jobs=1,
        running_jobs=0,
        busy_nodes=0,
        scale_pending_jobs=1,
    )
    values.update(overrides)
    return PartitionState(**values)


def test_provisioning_timeout_rolls_back_and_backs_off(monkeypatch):
    monkeypatch.setattr(reconciler.time, "time", lambda: 100.0)
    monkeypatch.setattr(otel, "enabled", lambda: False)
    h = Harness(ready=0)
    h._provisioning["slurm-worker-cpu"] = (80.0, 0, 1, None)

    h._process_pool(_pool(), {"slurm-worker-cpu": _state()})

    assert h.actuator.patches == [("slurm-worker-cpu", 0)]
    assert h.client.future == [("slurm-worker-cpu-0", "provisioning-timeout")]
    assert h._provisioning == {}
    assert h._provisioning_retry_at == {"slurm-worker-cpu": 120.0}
    assert h.last_scale_action_at["slurm-worker-cpu"] == 100.0
    assert any(event == "provisioning_timeout" for event, _ in h.logger.events)
    assert any(
        event == "scale_skipped" and fields["reason"] == "provisioning_retry_backoff"
        for event, fields in h.logger.events
    )


def test_ready_provisioning_resumes_only_new_ordinals_without_reconfigure(monkeypatch):
    monkeypatch.setattr(reconciler.time, "time", lambda: 100.0)
    monkeypatch.setattr(otel, "enabled", lambda: False)
    h = Harness(ready=2)
    h._provisioning["slurm-worker-cpu"] = (90.0, 1, 2, None)

    h._process_pool(
        _pool(),
        {"slurm-worker-cpu": _state(current_replicas=2, scale_pending_jobs=0)},
    )

    assert h.client.resumed == ["slurm-worker-cpu-1"]
    assert h.actuator.patches == []
    assert h._provisioning == {}
    assert any(
        name == "slurm.k8s/provisioning" and value == ""
        for _, _, name, value in h.client.annotations
    )
