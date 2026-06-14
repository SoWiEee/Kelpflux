"""Tests for serve.py POST /reload — hot-swap of the served DSAC checkpoint.

Calls the endpoint function directly (httpx/TestClient is not installed) and
inspects the module global so the test needs no running HTTP server or cluster.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("prometheus_client")  # serve.py exposes /metrics; lib in the image
pytest.importorskip("fastapi")

from fastapi import HTTPException

import services.rl_scheduler.serve as serve
from services.rl_scheduler.dsac import DSACAgent


def _save_agent(path, *, n_actions=4, use_iqn=True, risk_mode="cvar"):
    agent = DSACAgent(obs_dim=8, n_actions=n_actions, device="cpu",
                      use_iqn=use_iqn, risk_mode=risk_mode)
    agent.save(str(path))
    return path


@pytest.fixture
def sac_ckpt(tmp_path):
    return _save_agent(tmp_path / "sac.pt", use_iqn=False)


@pytest.fixture
def rdsac_ckpt(tmp_path):
    return _save_agent(tmp_path / "rdsac_cvar.pt", use_iqn=True, risk_mode="cvar")


def _load_holder(ckpt):
    serve._holder = serve._AgentHolder.from_checkpoint(ckpt)


def test_reload_swaps_sac_to_rdsac(sac_ckpt, rdsac_ckpt):
    _load_holder(sac_ckpt)
    assert serve._holder.agent.use_iqn is False

    resp = serve.reload_checkpoint(serve.ReloadRequest(checkpoint=str(rdsac_ckpt)))

    assert resp["ok"] is True
    assert resp["variant"] == "RDSAC:cvar"
    assert resp["use_iqn"] is True
    assert serve._holder.agent.use_iqn is True       # global actually swapped


def test_reload_missing_file_keeps_current(sac_ckpt):
    _load_holder(sac_ckpt)
    before = serve._holder
    with pytest.raises(HTTPException) as ei:
        serve.reload_checkpoint(serve.ReloadRequest(checkpoint="/no/such/file.pt"))
    assert ei.value.status_code == 404
    assert serve._holder is before                   # unchanged on failure


def test_reload_dim_mismatch_keeps_current(sac_ckpt, tmp_path):
    _load_holder(sac_ckpt)                            # n_actions=4
    before = serve._holder
    bad = _save_agent(tmp_path / "bad.pt", n_actions=9, use_iqn=False)
    with pytest.raises(HTTPException) as ei:
        serve.reload_checkpoint(serve.ReloadRequest(checkpoint=str(bad)))
    assert ei.value.status_code == 400
    assert serve._holder is before                   # kept current model


def test_shadow_toggle_sets_global():
    serve.SHADOW_MODE = True
    r1 = serve.set_shadow(serve.ShadowRequest(shadow=False))
    assert r1["shadow_mode"] is False and serve.SHADOW_MODE is False
    r2 = serve.set_shadow(serve.ShadowRequest(shadow=True))
    assert r2["shadow_mode"] is True and serve.SHADOW_MODE is True


def test_variant_helper():
    assert serve._variant_of(DSACAgent(obs_dim=8, n_actions=4, device="cpu",
                                       use_iqn=False)) == "SAC"
    assert serve._variant_of(DSACAgent(obs_dim=8, n_actions=4, device="cpu",
                                       use_iqn=True, risk_mode="wang")) == "RDSAC:wang"
