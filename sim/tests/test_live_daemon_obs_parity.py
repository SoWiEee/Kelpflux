"""Parity guard: the live_daemon obs MUST equal the canonical gym_env obs.

The daemon logs (obs, act, rew, next_obs) for RLPD raw ``--online-log``. If its
obs construction drifts from ``gym_env`` (as it silently did — a 166-d obs fed
to a 168-d policy), the logged transitions are corrupt AND the buffer add
crashes. This builds one controlled scenario two ways:

  * canonical — a real ``sim.cluster.Cluster`` populated via the sim allocation
    API + ``sim.loader.Job`` pending, run through gym_env's ``_build_obs`` order;
  * live      — the equivalent ``LiveJob`` pending/running sets fed to
    ``live_daemon.build_obs_and_mask`` (which rehydrates sim objects internally).

and asserts the two obs vectors are identical, 168-d, and load into a real
2×1 checkpoint without the shape crash that motivated the fix.
"""
import glob
import math

import numpy as np
import pytest

from sim.cluster import Cluster
from sim.gym_env import (
    JOB_FEAT_DIM, TOP_K, env_dims,
    _job_feat, _gpu_feat, _topo_feat, _global_feat,
)
from sim.loader import Job
from services.rl_scheduler import live_daemon as ld

NODE_NAMES = ["kf-worker-gpu-rtx4070-0", "kf-worker-gpu-rtx3080-0"]
N_GPUS = 1
MPS = 100
NOW = 1_000_000.0


def _canonical_obs(pending, cluster):
    """Replicate gym_env.KubefluxSchedEnv._build_obs assembly exactly."""
    top = sorted(pending, key=lambda j: j.submit_ts)[:TOP_K]
    job_feats = []
    for i in range(TOP_K):
        if i < len(top):
            job_feats.append(_job_feat(top[i], NOW, MPS))
        else:
            job_feats.append(np.zeros(JOB_FEAT_DIM, dtype=np.float32))
    gpu_feats = [
        _gpu_feat(cluster, ni, gi)
        for ni in range(cluster.n_nodes)
        for gi in range(cluster.gpus_per_node)
    ]
    topo = _topo_feat(pending, cluster)
    glob = _global_feat(pending, cluster, NOW)
    return np.concatenate([*job_feats, *gpu_feats, topo, glob]).astype(np.float32)


def _scenario():
    """A mixed state: an llm running on the 3080, a whole-GPU batch on the 4070,
    plus three pending jobs of different classes/cards → exercises every feature
    (ram, slo, is_inference, free_ram, gpu one-hot, frag, cross-node)."""
    # canonical sim cluster — same knobs live_daemon uses internally.
    cl = Cluster(n_nodes=2, gpus_per_node=1, mps_per_gpu=MPS,
                 node_gpu_types=["rtx4070", "rtx3080"], node_ram_gb=[62.0, 5.0])
    run_llm = Job("r-llm", "u", 1, "rtx3080", 0.0, 140.0, 0.0, 75, "llm", 0.0, 2.5)
    run_bat = Job("r-bat", "u", 1, "rtx4070", 0.0, 90.0, 0.0, 100, "batch", 0.0, 1.0)
    assert cl.try_allocate_on(run_llm, 1, 0) is not None   # 3080
    assert cl.try_allocate_on(run_bat, 0, 0) is not None   # 4070

    pending = [
        Job("p-inf", "u", 1, "rtx4070", 5.0, 18.0, 0.0, 25, "inference", 36.0, 1.0),
        Job("p-trn", "u", 1, "rtx3080", 7.0, 300.0, 0.0, 50, "training", 0.0, 1.1),
        Job("p-llm", "u", 1, "rtx4070", 9.0, 180.0, 0.0, 100, "llm", 0.0, 2.5),
    ]

    # live mirror of the SAME state
    def _lj(j, state, nodelist=""):
        return ld.LiveJob(
            job_id=j.job_id, mps_req=j.mps_req, gpu_count=j.gpu_count,
            gpu_type=j.gpu_type, runtime=j.runtime, submit_ts=j.submit_ts,
            state=state, nodelist=nodelist, job_class=j.job_class,
            slo_s=j.slo_s, ram_req=j.ram_req)

    running_live = [_lj(run_llm, "RUNNING", NODE_NAMES[1]),
                    _lj(run_bat, "RUNNING", NODE_NAMES[0])]
    pending_live = [_lj(j, "PENDING") for j in pending]
    return cl, pending, pending_live, running_live


def test_live_obs_matches_canonical():
    cl, pending, pending_live, running_live = _scenario()
    obs_ref = _canonical_obs(pending, cl)
    obs_live, mask_live, top_ids = ld.build_obs_and_mask(
        pending_live, running_live, NODE_NAMES, N_GPUS, MPS, NOW)

    obs_dim, n_actions = env_dims(2, 1)
    assert obs_ref.shape == (obs_dim,) == (168,)
    assert obs_live.shape == (obs_dim,)
    np.testing.assert_allclose(obs_live, obs_ref, rtol=0, atol=1e-6)
    assert mask_live.shape == (n_actions,)
    assert mask_live[-1]  # no-op always legal
    # top_ids preserve submit-order for the action decode
    assert top_ids[:3] == ["p-inf", "p-trn", "p-llm"]


def test_reconstructed_cluster_matches_sim_allocation():
    cl, _pending, _pl, running_live = _scenario()
    cl_live = ld._reconstruct_sim_cluster(running_live, NODE_NAMES, N_GPUS, MPS)
    for ni in range(2):
        assert cl_live.nodes[ni].gpus[0].free_mps == cl.nodes[ni].gpus[0].free_mps
        assert math.isclose(cl_live.nodes[ni].used_ram_gb, cl.nodes[ni].used_ram_gb)
        assert math.isclose(cl_live.nodes[ni].free_ram_ratio(),
                            cl.nodes[ni].free_ram_ratio())
    assert set(cl_live.active) == set(cl.active)


def test_live_obs_adds_to_replay_buffer():
    """Reproduce the exact crash the fix targets: the daemon adds the live obs to
    a ``ReplayBuffer`` sized by ``env_dims`` (168). A 166-d obs raised
    ``could not broadcast (166,) into (168,)`` at rlpd_finetune.py:235."""
    from services.rl_scheduler.rlpd_finetune import ReplayBuffer, Transition
    obs_dim, n_actions = env_dims(2, 1)
    _cl, _p, pending_live, running_live = _scenario()
    obs, mask, _ = ld.build_obs_and_mask(
        pending_live, running_live, NODE_NAMES, N_GPUS, MPS, NOW)
    buf = ReplayBuffer(capacity=8, obs_dim=obs_dim, n_actions=n_actions)
    buf.add(Transition(obs=obs, act=0, rew=-0.1, next_obs=obs, done=False,
                       mask=mask, next_mask=mask))   # must not raise
    assert len(buf) == 1


def test_sacct_jct_uses_end_minus_submit(monkeypatch):
    """Reward JCT must come from sacct End−Submit, not wall-clock (which MinJobAge
    squeue retention inflates by ~300s)."""
    monkeypatch.setattr(ld, "_run",
                        lambda cmd, timeout=20: "2026-08-14T11:11:58|2026-08-14T11:13:13|COMPLETED")
    assert ld._sacct_jct("166314") == 75.0
    # non-terminal / unparseable → None so the caller falls back to now−ts
    monkeypatch.setattr(ld, "_run",
                        lambda cmd, timeout=20: "2026-08-14T11:11:58|Unknown|RUNNING")
    assert ld._sacct_jct("1") is None
    monkeypatch.setattr(ld, "_run", lambda cmd, timeout=20: "")
    assert ld._sacct_jct("1") is None


def test_logged_obs_loads_into_real_checkpoint():
    """The original bug: a live obs could not be added to the 168-d buffer /
    fed to the policy. Reproduce the exact path with a real checkpoint."""
    cks = sorted(glob.glob("runs/ckpts_aimix16/rdsac_cvar_s*.pt"))
    if not cks:
        pytest.skip("no 2×1 aimix checkpoint available")
    from services.rl_scheduler.dsac import DSACAgent
    agent = DSACAgent.load(cks[0], device="cpu")
    _cl, _p, pending_live, running_live = _scenario()
    obs, mask, _ = ld.build_obs_and_mask(
        pending_live, running_live, NODE_NAMES, N_GPUS, MPS, NOW)
    a = agent.select_action(obs, mask, greedy=True)   # must not raise on shape
    assert 0 <= int(a) < mask.shape[0]
