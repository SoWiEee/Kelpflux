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


def test_build_snapshot_filters_down_nodes_and_falls_back():
    snap = agent.build_snapshot(
        {"jobs": []},
        {"nodes": [{"name": "bad", "state": "DOWN", "gres": "gpu:rtx4070:1,mps:100"}]},
        now=10,
        mps_per_gpu=100,
    )

    assert snap["n_nodes"] == 1
    assert snap["nodes"][0]["gpus"][0]["free_mps"] == 100
    assert snap["nodes"][0]["gpus"][0]["gpu_type"] == "rtx4070"


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
