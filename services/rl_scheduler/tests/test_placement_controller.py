import json

from services.rl_scheduler import placement_controller as pc


def test_parse_squeue_jobs_extracts_held_gpu_mps_job():
    raw = json.dumps(
        {
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
    )

    jobs = pc.parse_squeue_jobs(raw, default_runtime=600, default_mps=100)

    assert len(jobs) == 1
    assert jobs[0].job_id == "42"
    assert jobs[0].name == "dsac-place-test"
    assert jobs[0].mps_req == 25
    assert jobs[0].runtime == 300
    assert jobs[0].reason == "JobHeldUser"


def test_parse_scontrol_node_computes_free_mps():
    raw = """
    NodeName=slurm-worker-gpu-rtx4070-0 State=MIXED
       CfgTRES=cpu=4,mem=3500M,billing=4,gres/gpu=1,gres/mps=100
       AllocTRES=cpu=1,mem=100M,gres/mps=25
    """

    node = pc.parse_scontrol_node(raw, "slurm-worker-gpu-rtx4070-0", mps_per_gpu=100)

    assert node.free_mps == 75
    assert node.running_jobs == 1
    assert node.gpu_type == "rtx4070"
    assert node.available is True


def test_parse_scontrol_node_marks_drained_node_unavailable():
    raw = """
    NodeName=slurm-worker-gpu-rtx4070-0 State=IDLE+DRAIN+NOT_RESPONDING
       CfgTRES=cpu=4,mem=3500M,billing=4,gres/gpu=1,gres/mps=100
       AllocTRES=
    """

    node = pc.parse_scontrol_node(raw, "slurm-worker-gpu-rtx4070-0", mps_per_gpu=100)

    assert node.available is False


def test_build_act_payload_maps_jobs_and_nodes_to_dsac_schema():
    jobs = [
        pc.SlurmJob("7", "job", "PENDING", "JobHeldUser", 25, 1, "rtx4070", 60, 10),
    ]
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


class FakeRunner:
    def __init__(self):
        self.commands = []

    def run(self, args, **kwargs):
        self.commands.append(list(args))
        return ""


def test_apply_hard_placement_sets_req_node_list_and_releases():
    runner = FakeRunner()

    pc.apply_hard_placement(runner, job_id="99", node_name="gpu-0", release=True)

    assert runner.commands == [
        ["scontrol", "update", "JobId=99", "ReqNodeList=gpu-0"],
        ["scontrol", "release", "99"],
    ]
