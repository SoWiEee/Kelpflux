"""Tests for the vectorized scheduling envs (sim/vec_env.py)."""
import numpy as np
import pytest

from sim.gym_env import KubefluxSchedEnv, env_dims
from sim.vec_env import (
    AsyncVectorSchedEnv,
    EnvSpec,
    FamilyJobFactory,
    SyncVectorSchedEnv,
    make_vector_env,
)


def _spec(num_jobs=8, base_seed=0):
    return EnvSpec(
        families=("philly",),
        n_jobs=num_jobs,
        total_gpus=1,
        n_nodes=1,
        gpus_per_node=1,
        max_steps=num_jobs * 50,
        base_seed=base_seed,
    )


def test_family_job_factory_is_picklable_and_preserves_stream():
    import pickle

    fac = FamilyJobFactory(("philly",), n_jobs=6, total_gpus=1, seed=123)
    first = [j.job_id for j in fac()]            # advance the stream once
    restored = pickle.loads(pickle.dumps(fac))   # snapshot mid-stream
    # Both continue from the same state → identical next traces.
    assert [j.job_id for j in fac()] == [j.job_id for j in restored()]
    assert first  # non-empty workload


def test_sync_vector_shapes_match_env_dims():
    spec = _spec()
    obs_dim, n_actions = env_dims(spec.n_nodes, spec.gpus_per_node)
    vec = SyncVectorSchedEnv(spec, num_envs=3)
    try:
        obs, masks = vec.reset(seed=7)
        assert obs.shape == (3, obs_dim)
        assert masks.shape == (3, n_actions)
        assert masks.dtype == bool
        acts = [int(np.flatnonzero(m)[0]) for m in masks]
        obs2, masks2, rews, dones, infos = vec.step(acts)
        assert obs2.shape == (3, obs_dim)
        assert rews.shape == (3,) and rews.dtype == np.float32
        assert dones.shape == (3,) and dones.dtype == bool
        assert len(infos) == 3
    finally:
        vec.close()


def test_sync_slot_matches_standalone_env_step_for_step():
    """A 1-wide sync vector must reproduce a standalone env exactly (same seed)."""
    spec = _spec(num_jobs=6, base_seed=0)
    vec = SyncVectorSchedEnv(spec, num_envs=1)
    standalone = spec.build(0)  # same index-0 seeding as the vector's slot 0

    try:
        v_obs, v_mask = vec.reset(seed=None)
        s_obs, _ = standalone.reset(seed=None)
        s_mask = standalone.action_mask()
        np.testing.assert_array_equal(v_obs[0], s_obs)
        np.testing.assert_array_equal(v_mask[0], s_mask)

        for _ in range(40):
            act = int(np.flatnonzero(v_mask[0])[0])
            v_obs, v_mask, v_rew, v_done, _ = vec.step([act])
            s_o, s_r, s_term, s_trunc, s_info = standalone.step(act)
            s_done = bool(s_term or s_trunc)
            assert bool(v_done[0]) == s_done
            assert float(v_rew[0]) == pytest.approx(s_r)
            if s_done:
                s_o, _ = standalone.reset()
            np.testing.assert_array_equal(v_obs[0], s_o)
            v_mask = v_mask  # next iter uses refreshed mask
    finally:
        vec.close()
        standalone.close()


def test_autoreset_exposes_terminal_obs_in_info():
    # Tiny episode budget → force a truncation so we exercise the autoreset path.
    spec = EnvSpec(families=("philly",), n_jobs=4, total_gpus=1, max_steps=3, base_seed=1)
    vec = SyncVectorSchedEnv(spec, num_envs=1)
    try:
        _, masks = vec.reset(seed=0)
        saw_done = False
        for _ in range(10):
            act = int(np.flatnonzero(masks[0])[0])
            obs, masks, _, dones, infos = vec.step([act])
            if dones[0]:
                saw_done = True
                assert "final_obs" in infos[0]
                assert "final_mask" in infos[0]
                assert infos[0]["final_obs"].shape == obs[0].shape
                break
        assert saw_done, "max_steps=3 should truncate within 10 steps"
    finally:
        vec.close()


def test_async_matches_sync_transitions():
    spec = _spec(num_jobs=6, base_seed=42)
    sync = SyncVectorSchedEnv(spec, num_envs=2)
    asyncv = AsyncVectorSchedEnv(spec, num_envs=2)
    try:
        s_obs, s_masks = sync.reset(seed=5)
        a_obs, a_masks = asyncv.reset(seed=5)
        np.testing.assert_array_equal(s_obs, a_obs)
        np.testing.assert_array_equal(s_masks, a_masks)

        for _ in range(25):
            acts = [int(np.flatnonzero(m)[0]) for m in s_masks]
            s_obs, s_masks, s_rews, s_dones, _ = sync.step(acts)
            a_obs, a_masks, a_rews, a_dones, _ = asyncv.step(acts)
            np.testing.assert_array_equal(s_obs, a_obs)
            np.testing.assert_array_equal(s_masks, a_masks)
            np.testing.assert_allclose(s_rews, a_rews)
            np.testing.assert_array_equal(s_dones, a_dones)
    finally:
        sync.close()
        asyncv.close()


def test_make_vector_env_falls_back_to_sync_for_single_env():
    assert isinstance(make_vector_env(_spec(), 1), SyncVectorSchedEnv)
    assert isinstance(make_vector_env(_spec(), 4, asynchronous=False), SyncVectorSchedEnv)
