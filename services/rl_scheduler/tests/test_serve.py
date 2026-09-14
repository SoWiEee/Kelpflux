from __future__ import annotations

import asyncio
import importlib
import sys
import time
import types

import httpx
import pytest
from fastapi.testclient import TestClient


class _FakeAgent:
    obs_dim = 161   # 1×1: 144 + 1*7 + 4 + 6 (GPU_FEAT_DIM=7 after host-RAM feature)
    n_actions = 17


class _FakeHolder:
    def __init__(self, *, action: int, value: float = 10.0, entropy: float = 0.0):
        self.agent = _FakeAgent()
        self.action = action
        self.value = value
        self.entropy = entropy
        self.seen_obs = None
        self.seen_mask = None

    def select(self, obs, mask):
        self.seen_obs = obs
        self.seen_mask = mask
        return self.action, self.value, self.entropy


@pytest.fixture()
def serve(monkeypatch):
    """Load serve.py without requiring a real torch DSAC checkpoint."""
    fake_dsac = types.ModuleType("services.rl_scheduler.dsac")

    class _UnusedDSACAgent:
        @classmethod
        def load(cls, _path):  # pragma: no cover - _AgentHolder is not used here
            raise AssertionError("tests should inject _holder directly")

    fake_dsac.DSACAgent = _UnusedDSACAgent
    fake_dsac._build_mlp = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "services.rl_scheduler.dsac", fake_dsac)
    sys.modules.pop("services.rl_scheduler.serve", None)
    module = importlib.import_module("services.rl_scheduler.serve")

    module._holder = None
    module._snapshot = None
    module.SHADOW_MODE = False
    module.SNAPSHOT_TTL_S = 30.0
    module.VALUE_ABSTAIN = -100000.0
    module.ENTROPY_ABSTAIN = 1.5
    module.PRIORITY_BOOST = 1000
    module._API_TOKEN = None
    module._SNAPSHOT_TOKEN = None
    module._CONTROL_TOKEN = None
    module.RL_SHADOW_MODE.set(0.0)
    module.RL_READY.set(0.0)
    return module


@pytest.fixture()
def client(serve):
    return TestClient(serve.app)


def _snapshot_payload(*, ts: float | None = None, free_mps: int = 100):
    payload = {
        "now": 10.0,
        "pending_jobs": [],
        "nodes": [
            {"gpus": [{"free_mps": free_mps, "running_jobs": 0, "gpu_type": "rtx4070"}]}
        ],
        "n_nodes": 1,
        "gpus_per_node": 1,
        "mps_per_gpu": 100,
    }
    if ts is not None:
        payload["ts"] = ts
    return payload


def _decide_payload(job_id: str = "job-1", *, mps_req: int = 25):
    return {
        "job_id": job_id,
        "mps_req": mps_req,
        "gpu_count": 1,
        "gpu_type": "rtx4070",
        "runtime": 60.0,
        "submit_ts": 10.0,
    }


def test_build_obs_and_mask_marks_only_feasible_actions(serve):
    req = serve.ActRequest(
        now=10.0,
        pending_jobs=[
            serve.JobView(job_id="fits", mps_req=25, gpu_count=1, runtime=60, submit_ts=1),
            serve.JobView(job_id="too-big", mps_req=125, gpu_count=1, runtime=60, submit_ts=2),
        ],
        nodes=[serve.NodeView(gpus=[serve.GpuView(free_mps=50)])],
        n_nodes=1,
        gpus_per_node=1,
        mps_per_gpu=100,
    )

    obs, mask, top_ids = serve.build_obs_and_mask(req)

    assert obs.shape == (161,)
    assert mask.shape == (17,)
    assert top_ids[:2] == ["fits", "too-big"]
    assert mask[0] is True or bool(mask[0]) is True
    assert mask[1] is False or bool(mask[1]) is False
    assert bool(mask[16]) is True  # no-op is always valid


def test_build_obs_and_mask_respects_slurm_node_eligibility(serve):
    req = serve.ActRequest(
        now=10.0,
        pending_jobs=[serve.JobView(
            job_id="3080-only", mps_req=25, gpu_count=1, runtime=60,
            submit_ts=1, eligible_node_indices=[1],
        )],
        nodes=[
            serve.NodeView(gpus=[serve.GpuView(free_mps=100, gpu_type="rtx4070")]),
            serve.NodeView(gpus=[serve.GpuView(free_mps=100, gpu_type="rtx3080")]),
        ],
        n_nodes=2,
        gpus_per_node=1,
        mps_per_gpu=100,
    )

    _, mask, _ = serve.build_obs_and_mask(req)

    assert not bool(mask[0])
    assert bool(mask[1])
    assert bool(mask[32])  # no-op


def test_act_rejects_checkpoint_topology_mismatch(serve, client):
    serve._holder = _FakeHolder(action=0)
    payload = _snapshot_payload()
    payload.update({"n_nodes": 2, "nodes": payload["nodes"] * 2})

    res = client.post("/act", json=payload)

    assert res.status_code == 409


def test_snapshot_endpoint_updates_prometheus_metrics(client):
    res = client.post("/snapshot", json=_snapshot_payload(free_mps=75))
    assert res.status_code == 200
    assert res.json()["pending"] == 0
    assert res.json()["nodes"] == 1

    metrics = client.get("/metrics").text
    assert "rl_scheduler_snapshot_free_mps 75.0" in metrics
    assert "rl_scheduler_snapshot_pending_jobs 0.0" in metrics


def test_configured_token_files_fail_closed(serve, monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_TOKEN_FILE", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="cannot read token file"):
        serve._load_configured_token("TEST_TOKEN_FILE")

    empty = tmp_path / "empty"
    empty.write_bytes(b"\n")
    monkeypatch.setenv("TEST_TOKEN_FILE", str(empty))
    with pytest.raises(RuntimeError, match="is empty"):
        serve._load_configured_token("TEST_TOKEN_FILE")


def test_mutation_endpoints_are_open_without_token_configuration(
    serve, monkeypatch, tmp_path,
):
    checkpoint = tmp_path / "agent.pt"
    checkpoint.write_bytes(b"test")
    monkeypatch.setattr(
        serve._AgentHolder,
        "from_checkpoint",
        classmethod(lambda cls, path: _FakeHolder(action=0)),
    )

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=serve.app),
            base_url="http://test",
        ) as client:
            assert (await client.post("/snapshot", json=_snapshot_payload())).status_code == 200
            serve._holder = _FakeHolder(action=0)
            assert (await client.post("/act", json=_snapshot_payload())).status_code == 200
            assert (await client.post("/decide", json=_decide_payload())).status_code == 200
            assert (await client.post(
                "/reload", json={"checkpoint": str(checkpoint)},
            )).status_code == 200
            assert (await client.post("/shadow", json={"shadow": True})).status_code == 200

    asyncio.run(exercise())


def test_mutation_endpoints_require_their_configured_bearer_tokens(
    serve, monkeypatch, tmp_path,
):
    checkpoint = tmp_path / "agent.pt"
    checkpoint.write_bytes(b"test")
    monkeypatch.setattr(
        serve._AgentHolder,
        "from_checkpoint",
        classmethod(lambda cls, path: _FakeHolder(action=0)),
    )
    serve._API_TOKEN = b"api-secret"
    serve._SNAPSHOT_TOKEN = b"snapshot-secret"
    serve._CONTROL_TOKEN = b"control-secret"
    requests = (
        ("/snapshot", _snapshot_payload(), "snapshot-secret", "api-secret", 200),
        ("/act", _snapshot_payload(), "api-secret", "snapshot-secret", 200),
        ("/decide", _decide_payload(), "api-secret", "control-secret", 200),
        ("/reload", {"checkpoint": str(checkpoint)}, "control-secret", "api-secret", 200),
        ("/shadow", {"shadow": True}, "control-secret", "snapshot-secret", 200),
    )

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=serve.app),
            base_url="http://test",
        ) as client:
            for path, payload, _, foreign_token, _ in requests:
                assert (await client.post(path, json=payload)).status_code == 401
                assert (await client.post(
                    path,
                    json=payload,
                    headers={"Authorization": "Bearer wrong"},
                )).status_code == 401
                assert (await client.post(
                    path,
                    json=payload,
                    headers={"Authorization": f"Bearer {foreign_token}"},
                )).status_code == 401

            assert serve._snapshot is None
            assert serve._holder is None
            assert serve.SHADOW_MODE is False
            assert (await client.get("/healthz")).status_code == 200
            assert (await client.get("/metrics")).status_code == 200
            serve._holder = _FakeHolder(action=0)
            for path, payload, token, _, expected_status in requests:
                assert (await client.post(
                    path,
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                )).status_code == expected_status

    asyncio.run(exercise())


def test_decide_abstains_when_snapshot_missing(client):
    res = client.post("/decide", json=_decide_payload())
    body = res.json()

    assert res.status_code == 200
    assert body["abstain"] is True
    assert body["priority_boost"] == 0
    assert body["abstain_reason"].startswith("snapshot_stale")


def test_decide_selected_job_returns_priority_boost(serve, client):
    serve._holder = _FakeHolder(action=0, value=5.0, entropy=0.0)
    client.post("/snapshot", json=_snapshot_payload())

    res = client.post("/decide", json=_decide_payload("job-1"))
    body = res.json()

    assert res.status_code == 200
    assert body["abstain"] is False
    assert body["rl_selected"] is True
    assert body["priority_boost"] == 1000
    assert body["rl_selected_job_id"] == "job-1"
    assert body["node_j"] == 0
    assert body["gpu_k"] == 0

    metrics = client.get("/metrics").text
    assert 'rl_scheduler_decisions_total{result="selected"} 1.0' in metrics
    assert "rl_scheduler_priority_boost_total 1.0" in metrics


def test_decide_no_op_does_not_boost(serve, client):
    serve._holder = _FakeHolder(action=16, value=1.0, entropy=0.0)
    client.post("/snapshot", json=_snapshot_payload())

    res = client.post("/decide", json=_decide_payload("job-1"))
    body = res.json()

    assert res.status_code == 200
    assert body["abstain"] is False
    assert body["rl_selected"] is False
    assert body["priority_boost"] == 0
    assert body["rl_selected_job_id"] is None
    assert body["node_j"] is None
    assert body["gpu_k"] is None


def test_decide_stale_snapshot_abstains_even_with_loaded_model(serve, client):
    serve._holder = _FakeHolder(action=0, value=5.0, entropy=0.0)
    serve.SNAPSHOT_TTL_S = 0.01
    client.post("/snapshot", json=_snapshot_payload(ts=time.time() - 10))

    res = client.post("/decide", json=_decide_payload("job-1"))
    body = res.json()

    assert res.status_code == 200
    assert body["abstain"] is True
    assert body["priority_boost"] == 0
    assert body["rl_selected"] is False


def test_decide_abstains_when_snapshot_has_no_available_gpu_nodes(serve, client):
    serve._holder = _FakeHolder(action=0, value=5.0, entropy=0.0)
    client.post(
        "/snapshot",
        json={"now": 10.0, "pending_jobs": [], "nodes": [], "n_nodes": 0,
              "gpus_per_node": 1, "mps_per_gpu": 100},
    )

    res = client.post("/decide", json=_decide_payload("job-1"))
    body = res.json()

    assert res.status_code == 200
    assert body["abstain"] is True
    assert body["abstain_reason"] == "no_available_gpu_nodes"
    assert body["priority_boost"] == 0
    assert serve._holder.seen_obs is None


def test_decide_low_value_abstains(serve, client):
    serve._holder = _FakeHolder(action=0, value=-5.0, entropy=0.0)
    serve.VALUE_ABSTAIN = -1.0
    client.post("/snapshot", json=_snapshot_payload())

    res = client.post("/decide", json=_decide_payload("job-1"))
    body = res.json()

    assert res.status_code == 200
    assert body["abstain"] is True
    assert body["priority_boost"] == 0
    assert body["abstain_reason"].startswith("low_value")


def test_decide_shape_mismatch_abstains_instead_of_500(serve, client):
    serve._holder = _FakeHolder(action=0, value=5.0, entropy=0.0)
    client.post(
        "/snapshot",
        json={
            "now": 10.0,
            "pending_jobs": [],
            "nodes": [
                {
                    "gpus": [
                        {"free_mps": 100, "running_jobs": 0, "gpu_type": "rtx4070"},
                        {"free_mps": 100, "running_jobs": 0, "gpu_type": "rtx4070"},
                    ]
                }
            ],
            "n_nodes": 1,
            "gpus_per_node": 2,
            "mps_per_gpu": 100,
        },
    )

    res = client.post("/decide", json=_decide_payload("job-1"))
    body = res.json()

    assert res.status_code == 200
    assert body["abstain"] is True
    assert body["priority_boost"] == 0
    assert body["abstain_reason"].startswith("shape_mismatch_obs")
