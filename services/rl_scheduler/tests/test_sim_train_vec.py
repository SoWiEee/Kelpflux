"""Integration test: the vectorized sim_train path trains and saves a checkpoint."""
from pathlib import Path

from services.rl_scheduler.dsac import DSACAgent
from services.rl_scheduler.sim_train import sim_train
from sim.gym_env import env_dims


def test_sim_train_vectorized_sync_runs_and_saves(tmp_path: Path):
    obs_dim, n_actions = env_dims(1, 1)

    agent = sim_train(
        n_nodes=1, gpus_per_node=1, trace_family="philly", n_jobs=6,
        nstep_n=3, total_steps=600, warmup_steps=120,
        utd_ratio=1, batch_size=32, buf_capacity=5_000,
        seed=0, out_dir=tmp_path, device="cpu",
        score_warmup=False, potential_shaping=False, use_per=True,
        num_envs=4, async_envs=False,   # sync to stay deterministic + fast under pytest
    )

    assert isinstance(agent, DSACAgent)
    assert agent.obs_dim == obs_dim
    assert agent.n_actions == n_actions
    assert (tmp_path / "dsac.pt").exists()
    # training crossed warmup → episode log has content
    assert (tmp_path / "sim_train.jsonl").read_text().strip(), "expected episode logs"

    # checkpoint reloads to a matching-shape agent
    reloaded = DSACAgent.load(tmp_path / "dsac.pt")
    assert reloaded.obs_dim == obs_dim and reloaded.n_actions == n_actions


def test_sim_train_vectorized_async_smoke(tmp_path: Path):
    # Exercise the multiprocess backend end-to-end on a tiny budget.
    agent = sim_train(
        n_nodes=1, gpus_per_node=1, trace_family="philly", n_jobs=5,
        nstep_n=3, total_steps=160, warmup_steps=40,
        utd_ratio=1, batch_size=16, buf_capacity=2_000,
        seed=1, out_dir=tmp_path, device="cpu",
        score_warmup=False, potential_shaping=False, use_per=False,
        num_envs=2, async_envs=True,
    )
    assert (tmp_path / "dsac.pt").exists()
    assert agent.n_actions == env_dims(1, 1)[1]
