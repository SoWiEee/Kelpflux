from services.rl_scheduler import snapshot_agent as agent


def test_build_snapshot_extracts_pending_jobs_and_node_mps():
    jobs = {
        "jobs": [
            {
                "job_id": 42,
                "job_state": "PENDING",
                "submit_time": 100,
                "time_limit": 5,
                "tres_req_str": "cpu=1,gres/gpu=1,gres/mps=25",
            },
            {"job_id": 43, "job_state": "RUNNING", "tres_req_str": "cpu=1"},
        ]
    }
    nodes = {
        "nodes": [
            {
                "name": "slurm-worker-gpu-0",
                "state": "IDLE",
                "gres": "gpu:rtx4070:2,mps:200",
                "alloc_tres": "gres/mps=50",
            }
        ]
    }

    snap = agent.build_snapshot(jobs, nodes, now=200, mps_per_gpu=100, default_gpus_per_node=1)

    assert snap["n_nodes"] == 1
    assert snap["gpus_per_node"] == 1
    assert snap["pending_jobs"] == [
        {
            "job_id": "42",
            "mps_req": 25,
            "gpu_count": 1,
            "gpu_type": "rtx4070",
            "runtime": 300.0,
            "submit_ts": 100.0,
            "can_fit": True,
        }
    ]
    assert [g["free_mps"] for g in snap["nodes"][0]["gpus"]] == [100]


def test_build_snapshot_does_not_invent_gpu_when_no_nodes_are_available():
    snap = agent.build_snapshot(
        {"jobs": []},
        {"nodes": [{"name": "bad", "state": "DOWN", "gres": "gpu:rtx4070:1,mps:100"}]},
        now=10,
        mps_per_gpu=100,
    )

    assert snap["n_nodes"] == 0
    assert snap["nodes"] == []


def test_job_view_defaults_missing_gpu_request_to_full_mps():
    view = agent.job_view(
        {"job_id": "cpuish", "job_state": ["PENDING"], "submit_time": {"number": 5}, "tres_req_str": "cpu=1"},
        now=20,
        default_runtime=600,
        default_mps=100,
    )

    assert view["mps_req"] == 100
    assert view["gpu_count"] == 1
    assert view["runtime"] == 600.0
    assert view["submit_ts"] == 5.0


def test_build_snapshot_skips_cpu_nodes_when_gpu_nodes_exist():
    snap = agent.build_snapshot(
        {"jobs": []},
        {
            "nodes": [
                {"name": "slurm-worker-cpu-0", "state": "IDLE", "gres": ""},
                {"name": "slurm-worker-gpu-0", "state": "IDLE", "gres": "gpu:rtx4070:1,mps:100"},
            ]
        },
        now=10,
        mps_per_gpu=100,
        default_gpus_per_node=1,
    )

    assert snap["n_nodes"] == 1
    assert snap["snapshot" if False else "nodes"][0]["gpus"][0]["gpu_type"] == "rtx4070"


def test_run_once_sends_snapshot_bearer_token_only_to_scheduler(monkeypatch):
    calls = []

    def fake_http_json(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if url.endswith("/jobs"):
            return {"jobs": []}
        if url.endswith("/nodes"):
            return {"nodes": []}
        return {"ok": True}

    monkeypatch.setattr(agent, "http_json", fake_http_json)

    agent.run_once(
        rest_url="http://slurmrestd",
        api_version="v0.0.39",
        scheduler_url="http://rl-scheduler:8002",
        jwt_key=b"jwt-key",
        snapshot_token=b"snapshot-secret",
        mps_per_gpu=100,
        default_gpus_per_node=1,
        default_runtime=600,
    )

    assert calls[0][2].get("bearer_token") is None
    assert calls[1][2].get("bearer_token") is None
    assert calls[2][1] == "http://rl-scheduler:8002/snapshot"
    assert calls[2][2]["bearer_token"] == b"snapshot-secret"


def test_read_bearer_token_requires_a_nonempty_ascii_file(tmp_path):
    token_file = tmp_path / "api-token"
    token_file.write_bytes(b"api-secret\n")
    assert agent._read_bearer_token(str(token_file)) == b"api-secret"

    token_file.write_bytes(b"\n")
    try:
        agent._read_bearer_token(str(token_file))
    except RuntimeError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty API token must fail closed")
