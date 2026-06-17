from services.rl_scheduler import placement_controller as pc


def test_parse_jobs_extracts_held_gpu_mps_job():
    doc = {
        "jobs": [
            {
                "job_id": 42,
                "name": "dsac-place-test",
                "job_state": ["PENDING"],
                "state_reason": "JobHeldUser",
                "tres_req_str": "cpu=1,mem=100M,gres/mps=25",
                "time_limit": {"number": 5},
                "submit_time": {"number": 1000},
            },
            {"job_id": 43, "job_state": ["RUNNING"]},
        ]
    }

    jobs = pc.parse_jobs(doc, default_runtime=600, default_mps=100)

    assert len(jobs) == 1
    assert jobs[0].job_id == "42"
    assert jobs[0].name == "dsac-place-test"
    assert jobs[0].mps_req == 25
    assert jobs[0].runtime == 300
    assert jobs[0].reason == "JobHeldUser"


def test_filter_held_jobs_keeps_only_held():
    held = pc.SlurmJob("1", "a", "PENDING", "JobHeldUser", 25, 1, "rtx4070", 60, 1)
    queued = pc.SlurmJob("2", "b", "PENDING", "Resources", 25, 1, "rtx4070", 60, 2)

    assert pc.filter_held_jobs([held, queued]) == [held]


def test_parse_nodes_computes_free_mps_and_skips_cpu():
    doc = {
        "nodes": [
            {
                "name": "slurm-worker-gpu-rtx4070-0",
                "state": ["MIXED"],
                "tres": "cpu=4,mem=3500M,gres/gpu=1,gres/mps=100",
                "alloc_tres": "cpu=1,mem=100M,gres/mps=25",
            },
            {"name": "slurm-worker-cpu-0", "state": ["IDLE"], "tres": "cpu=4,mem=3500M"},
        ]
    }

    nodes = pc.parse_nodes(doc, mps_per_gpu=100)

    assert len(nodes) == 1
    assert nodes[0].name == "slurm-worker-gpu-rtx4070-0"
    assert nodes[0].free_mps == 75
    assert nodes[0].running_jobs == 1
    assert nodes[0].gpu_type == "rtx4070"
    assert nodes[0].available is True


def test_parse_nodes_marks_drained_node_unavailable():
    doc = {
        "nodes": [
            {
                "name": "slurm-worker-gpu-rtx4070-0",
                "state": ["IDLE", "DRAIN", "NOT_RESPONDING"],
                "tres": "gres/gpu=1,gres/mps=100",
            }
        ]
    }

    nodes = pc.parse_nodes(doc, mps_per_gpu=100)

    assert nodes[0].available is False


def test_build_act_payload_maps_jobs_and_nodes_to_dsac_schema():
    jobs = [pc.SlurmJob("7", "job", "PENDING", "JobHeldUser", 25, 1, "rtx4070", 60, 10)]
    nodes = [
        pc.SlurmNode("gpu-0", 50, 0, "rtx4070"),
        pc.SlurmNode("gpu-1", 10, 1, "rtx4070"),
    ]

    payload = pc.build_act_payload(jobs, nodes, mps_per_gpu=100, now=20)

    assert payload["n_nodes"] == 2
    assert payload["gpus_per_node"] == 1
    assert payload["pending_jobs"][0]["job_id"] == "7"
    assert payload["pending_jobs"][0]["can_fit"] is True
    assert payload["nodes"][0]["gpus"][0]["free_mps"] == 50
    assert payload["nodes"][1]["gpus"][0]["free_mps"] == 10


def test_apply_hard_placement_posts_required_nodes_and_release(monkeypatch):
    calls = []

    def fake_http_json(method, url, *, jwt_key=None, body=None, timeout=10.0):
        calls.append((method, url, jwt_key, body))
        return {}

    monkeypatch.setattr(pc, "http_json", fake_http_json)

    pc.apply_hard_placement(
        rest_base="http://rest/slurm/v0.0.37",
        jwt_key=b"k",
        job_id="99",
        node_name="gpu-0",
        release=True,
    )

    assert calls == [
        (
            "POST",
            "http://rest/slurm/v0.0.37/job/99",
            b"k",
            {"required_nodes": "gpu-0", "priority": pc.RELEASE_PRIORITY},
        )
    ]


def test_apply_hard_placement_shadow_omits_priority_when_no_release(monkeypatch):
    calls = []
    monkeypatch.setattr(
        pc, "http_json",
        lambda method, url, *, jwt_key=None, body=None, timeout=10.0: calls.append(body) or {},
    )

    pc.apply_hard_placement(
        rest_base="http://rest/slurm/v0.0.37", jwt_key=None,
        job_id="5", node_name="gpu-0", release=False,
    )

    assert calls == [{"required_nodes": "gpu-0"}]


def _stub_http(monkeypatch, *, jobs, nodes, act):
    """Route GET /jobs, GET /nodes, POST /act to canned docs; capture job-update POSTs."""
    updates = []

    def fake_http_json(method, url, *, jwt_key=None, body=None, timeout=10.0):
        if url.endswith("/jobs"):
            return jobs
        if url.endswith("/nodes"):
            return nodes
        if url.endswith("/act"):
            return act
        if "/job/" in url:
            updates.append((url, body))
            return {}
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(pc, "http_json", fake_http_json)
    return updates


def test_choose_and_apply_actuates_when_not_shadow(monkeypatch):
    jobs = {"jobs": [{
        "job_id": 7, "name": "htab-1", "job_state": ["PENDING"],
        "state_reason": "JobHeldUser", "tres_req_str": "gres/mps=25",
        "submit_time": {"number": 1},
    }]}
    nodes = {"nodes": [{
        "name": "gpu-0", "state": ["IDLE"], "tres": "gres/gpu=1,gres/mps=100",
    }]}
    act = {"selected_job_id": "7", "node_j": 0, "gpu_k": 0, "action": 3, "value": 1.0, "entropy": 0.2}
    updates = _stub_http(monkeypatch, jobs=jobs, nodes=nodes, act=act)

    decision = pc.choose_and_apply(
        rest_url="http://rest", api_version="v0.0.37", scheduler_url="http://rl",
        jwt_key=b"k", shadow=False, auto_trim_model_topology=False,
    )

    assert decision.applied is True
    assert decision.node_name == "gpu-0"
    assert decision.reason == "applied"
    assert updates == [(
        "http://rest/slurm/v0.0.37/job/7",
        {"required_nodes": "gpu-0", "priority": pc.RELEASE_PRIORITY},
    )]


def test_choose_and_apply_shadow_does_not_post_update(monkeypatch):
    jobs = {"jobs": [{
        "job_id": 7, "name": "htab-1", "job_state": ["PENDING"],
        "state_reason": "JobHeldUser", "tres_req_str": "gres/mps=25", "submit_time": {"number": 1},
    }]}
    nodes = {"nodes": [{"name": "gpu-0", "state": ["IDLE"], "tres": "gres/gpu=1,gres/mps=100"}]}
    act = {"selected_job_id": "7", "node_j": 0, "gpu_k": 0}
    updates = _stub_http(monkeypatch, jobs=jobs, nodes=nodes, act=act)

    decision = pc.choose_and_apply(
        rest_url="http://rest", api_version="v0.0.37", scheduler_url="http://rl",
        jwt_key=b"k", shadow=True, auto_trim_model_topology=False,
    )

    assert decision.applied is False
    assert decision.reason == "shadow"
    assert updates == []


def test_choose_and_apply_abstains_on_dsac_no_op(monkeypatch):
    jobs = {"jobs": [{
        "job_id": 7, "name": "htab-1", "job_state": ["PENDING"],
        "state_reason": "JobHeldUser", "tres_req_str": "gres/mps=25", "submit_time": {"number": 1},
    }]}
    nodes = {"nodes": [{"name": "gpu-0", "state": ["IDLE"], "tres": "gres/gpu=1,gres/mps=100"}]}
    act = {"selected_job_id": None, "node_j": None, "gpu_k": None, "action": -1}
    updates = _stub_http(monkeypatch, jobs=jobs, nodes=nodes, act=act)

    decision = pc.choose_and_apply(
        rest_url="http://rest", api_version="v0.0.37", scheduler_url="http://rl",
        jwt_key=b"k", shadow=False, auto_trim_model_topology=False,
    )

    assert decision.reason == "dsac_no_op"
    assert decision.applied is False
    assert updates == []
