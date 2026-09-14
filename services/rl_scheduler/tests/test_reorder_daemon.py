from services.rl_scheduler import reorder_daemon as rd
from services.rl_scheduler.placement_controller import SlurmJob, SlurmNode


def test_parse_eligible_jobs_limits_priority_scope_to_unheld_explicit_mps_gpu_jobs():
    doc = {"jobs": [
        {"job_id": 1, "name": "gpu", "job_state": ["PENDING"], "priority": 100,
         "partition": "gpu", "tres_req_str": "gres/mps=25", "submit_time": {"number": 1}},
        {"job_id": 2, "name": "cpu", "job_state": ["PENDING"], "priority": 100,
         "partition": "cpu", "tres_req_str": "cpu=2", "submit_time": {"number": 2}},
        {"job_id": 3, "name": "no-mps", "job_state": ["PENDING"], "priority": 100,
         "partition": "gpu", "tres_req_str": "gres/gpu=1", "submit_time": {"number": 3}},
        {"job_id": 4, "name": "held", "job_state": ["PENDING"], "priority": 0,
         "partition": "gpu", "state_reason": "JobHeldUser", "tres_req_str": "gres/mps=25"},
        {"job_id": 5, "name": "multi", "job_state": ["PENDING"], "priority": 100,
         "partition": "gpu", "tres_req_str": "gres/gpu=2,gres/mps=25"},
    ]}

    jobs = rd.parse_eligible_jobs(doc, gpu_partitions={"gpu", "gpu-rtx4070"})

    assert [job.job_id for job in jobs] == ["1"]
    assert jobs[0].mps_req == 25
    assert jobs[0].gpu_type == "rtx4070"


def test_rank_pending_respects_gpu_partition_and_uses_policy_selection():
    jobs = [
        SlurmJob("4070", "", "PENDING", "", 25, 1, "rtx4070", 10, 1,
                 partition="gpu-rtx4070", required_gpu_type="rtx4070"),
        SlurmJob("3080", "", "PENDING", "", 25, 1, "rtx3080", 10, 2,
                 partition="gpu-rtx3080", required_gpu_type="rtx3080"),
    ]
    nodes = [
        SlurmNode("gpu-4070", 50, 0, "rtx4070"),
        SlurmNode("gpu-3080", 50, 0, "rtx3080"),
    ]
    responses = iter([
        {"selected_job_id": "3080", "node_j": 1},
        {"selected_job_id": "4070", "node_j": 0},
    ])
    requests = []

    def post(payload, *, scheduler_url):
        requests.append(payload)
        return next(responses)

    order = rd.rank_pending(
        jobs, {node.name: node.free_mps for node in nodes}, mps_per_gpu=100,
        nodes=nodes, serve="http://rl", post=post, strict=True,
    )

    assert order == ["3080", "4070"]
    assert requests[0]["pending_jobs"][0]["eligible_node_indices"] == [0]
    assert requests[0]["pending_jobs"][1]["eligible_node_indices"] == [1]


def test_rank_fallback_uses_node_indices_and_respects_gpu_type():
    jobs = [
        SlurmJob("3080", "", "PENDING", "", 25, 1, "rtx3080", 10, 1,
                 required_gpu_type="rtx3080"),
    ]
    nodes = [
        SlurmNode("gpu-4070", 100, 0, "rtx4070"),
        SlurmNode("gpu-3080", 100, 0, "rtx3080"),
    ]

    order = rd.rank_pending(
        jobs, {node.name: node.free_mps for node in nodes}, mps_per_gpu=100,
        nodes=nodes, serve="http://rl", post=lambda *_args, **_kwargs: {},
    )

    assert order == ["3080"]


def test_rank_keeps_backlog_after_policy_noop_when_capacity_is_full():
    jobs = [
        SlurmJob(str(i), "", "PENDING", "", 100, 1, "rtx4070", 10, i)
        for i in range(3)
    ]
    nodes = [
        SlurmNode("gpu-4070", 100, 0, "rtx4070"),
        SlurmNode("gpu-3080", 100, 0, "rtx3080"),
    ]
    responses = iter([
        {"action": 0, "selected_job_id": "0", "node_j": 0},
        {"action": 1, "selected_job_id": "1", "node_j": 1},
        {"action": 32, "selected_job_id": None, "node_j": None},
    ])

    order = rd.rank_pending(
        jobs, {node.name: node.free_mps for node in nodes}, mps_per_gpu=100,
        nodes=nodes, serve="http://rl", post=lambda *_args, **_kwargs: next(responses),
        strict=True,
    )

    assert order == ["0", "1", "2"]


def test_reorder_cycle_writes_and_round_trips_reserved_priority(monkeypatch):
    jobs_doc = {"jobs": [{
        "job_id": 7, "name": "mps-canary", "job_state": ["PENDING"],
        "state_reason": "Resources", "priority": 100, "partition": "gpu",
        "tres_req_str": "gres/mps=25", "submit_time": {"number": 10},
    }]}
    jobs = rd.parse_eligible_jobs(jobs_doc, gpu_partitions={"gpu"})
    nodes = [
        SlurmNode("gpu-4070", 100, 0, "rtx4070"),
        SlurmNode("gpu-3080", 100, 0, "rtx3080"),
    ]
    writes = []

    monkeypatch.setattr(rd, "_read_state", lambda *a, **k: (jobs, nodes))
    monkeypatch.setattr(rd, "get_scheduler_healthz", lambda **k: {"obs_dim": 168, "n_actions": 33})
    monkeypatch.setattr(rd, "post_act", lambda *a, **k: {
        "selected_job_id": "7", "node_j": 0,
    })

    def set_priority(_base, _key, job_id, priority, _timeout):
        writes.append((job_id, priority))
        jobs_doc["jobs"][0]["priority"] = priority

    monkeypatch.setattr(rd, "_set_priority", set_priority)
    monkeypatch.setattr(rd, "http_json", lambda *a, **k: jobs_doc)

    order = rd.reorder_cycle(
        rest_base="http://rest/slurm/v0.0.39", jwt_key=b"key",
        scheduler_url="http://rl", node_names=["gpu-4070", "gpu-3080"],
        gpu_partitions={"gpu"},
    )

    assert order == ["7"]
    assert writes == [("7", rd.RELEASE_BASE)]


def test_reset_only_clears_priorities_in_reserved_band(monkeypatch):
    jobs_doc = {"jobs": [{
        "job_id": 7, "name": "mps", "job_state": ["PENDING"], "priority": rd.RELEASE_BASE,
        "partition": "gpu", "tres_req_str": "gres/mps=25",
    }]}
    writes = []

    monkeypatch.setattr(rd, "http_json", lambda *a, **k: jobs_doc)

    def set_priority(_base, _key, job_id, priority, _timeout):
        writes.append((job_id, priority))
        jobs_doc["jobs"][0]["priority"] = 100

    monkeypatch.setattr(rd, "_set_priority", set_priority)

    assert rd.reset_owned_priorities(
        "http://rest/slurm/v0.0.39", b"key",
    ) == ["7"]
    assert writes == [("7", rd.RELEASE_PRIORITY)]


def test_reset_cleans_owned_priority_after_job_leaves_gpu_scope(monkeypatch):
    jobs_doc = {"jobs": [{
        "job_id": 7, "name": "mps", "job_state": ["PENDING"],
        "priority": rd.RELEASE_BASE, "partition": "cpu", "tres_req_str": "cpu=2",
    }]}
    writes = []
    monkeypatch.setattr(rd, "http_json", lambda *a, **k: jobs_doc)

    def set_priority(_base, _key, job_id, priority, _timeout):
        writes.append((job_id, priority))
        jobs_doc["jobs"][0]["priority"] = 100

    monkeypatch.setattr(rd, "_set_priority", set_priority)

    assert rd.reset_owned_priorities(
        "http://rest/slurm/v0.0.39", b"key",
    ) == ["7"]
    assert writes == [("7", rd.RELEASE_PRIORITY)]


def test_reset_fails_if_slurm_leaves_infinite_priority_literal(monkeypatch):
    jobs_doc = {"jobs": [{
        "job_id": 7, "job_state": ["PENDING"], "priority": rd.RELEASE_BASE,
        "partition": "gpu", "tres_req_str": "gres/mps=25",
    }]}
    monkeypatch.setattr(rd, "http_json", lambda *a, **k: jobs_doc)
    monkeypatch.setattr(
        rd, "_set_priority",
        lambda _base, _key, _job_id, priority, _timeout: jobs_doc["jobs"][0].update(
            priority=priority,
        ),
    )

    try:
        rd.reset_owned_priorities(
            "http://rest/slurm/v0.0.39", b"key",
        )
    except RuntimeError as exc:
        assert "could not restore" in str(exc)
    else:
        raise AssertionError("an unchanged INFINITE priority must fail verification")


def test_reorder_cycle_detects_zero_priority_round_trip_and_rolls_back(monkeypatch):
    jobs_doc = {"jobs": [{
        "job_id": 7, "name": "mps-canary", "job_state": ["PENDING"],
        "state_reason": "Resources", "priority": 100, "partition": "gpu",
        "tres_req_str": "gres/mps=25", "submit_time": {"number": 10},
    }]}
    jobs = rd.parse_eligible_jobs(jobs_doc, gpu_partitions={"gpu"})
    nodes = [
        SlurmNode("gpu-4070", 100, 0, "rtx4070"),
        SlurmNode("gpu-3080", 100, 0, "rtx3080"),
    ]
    writes = []

    monkeypatch.setattr(rd, "_read_state", lambda *a, **k: (jobs, nodes))
    monkeypatch.setattr(rd, "get_scheduler_healthz", lambda **k: {"obs_dim": 168, "n_actions": 33})
    monkeypatch.setattr(rd, "post_act", lambda *a, **k: {
        "selected_job_id": "7", "node_j": 0,
    })

    def set_priority(_base, _key, job_id, priority, _timeout):
        writes.append((job_id, priority))
        jobs_doc["jobs"][0]["priority"] = 0 if priority == rd.RELEASE_BASE else priority

    monkeypatch.setattr(rd, "_set_priority", set_priority)
    monkeypatch.setattr(rd, "http_json", lambda *a, **k: jobs_doc)

    try:
        rd.reorder_cycle(
            rest_base="http://rest/slurm/v0.0.39", jwt_key=b"key",
            scheduler_url="http://rl", node_names=["gpu-4070", "gpu-3080"],
            gpu_partitions={"gpu"},
        )
    except RuntimeError as exc:
        assert "round-trip mismatch" in str(exc)
    else:
        raise AssertionError("priority=0 must not be filtered out during verification")

    assert jobs_doc["jobs"][0]["priority"] == 100
    assert writes == [("7", rd.RELEASE_BASE), ("7", 100)]


def test_reorder_cycle_rejects_queue_limit_above_checkpoint_capacity():
    try:
        rd.reorder_cycle(
            rest_base="http://rest/slurm/v0.0.39", jwt_key=b"key",
            scheduler_url="http://rl", node_names=["gpu-4070", "gpu-3080"],
            gpu_partitions={"gpu"}, max_jobs=rd.MAX_PENDING_JOBS + 1,
        )
    except ValueError as exc:
        assert "max_jobs" in str(exc)
    else:
        raise AssertionError("the priority band must bound max_jobs to checkpoint capacity")


def test_shadow_mode_never_resets_priorities(monkeypatch):
    class StopAfterOneCycle:
        waits = 0

        def is_set(self):
            return self.waits > 0

        def set(self):
            pass

        def wait(self, _seconds):
            self.waits += 1

    resets = []
    monkeypatch.setattr(rd.threading, "Event", StopAfterOneCycle)
    monkeypatch.setattr(rd.signal, "signal", lambda *_: None)
    monkeypatch.setattr(rd, "_read_jwt_key", lambda _path: b"key")
    monkeypatch.setattr(rd, "_slurm_base", lambda url, version: f"{url}/{version}")
    monkeypatch.setattr(rd, "reset_owned_priorities", lambda *a, **k: resets.append(True))
    monkeypatch.setattr(rd, "reorder_cycle", lambda **_kwargs: [])

    assert rd.main([
        "--rest-url", "http://rest", "--api-version", "v0.0.39",
        "--node-name", "gpu-4070", "--node-name", "gpu-3080",
        "--gpu-partition", "gpu", "--shadow", "--interval", "1",
    ]) == 0
    assert resets == []
