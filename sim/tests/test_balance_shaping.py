"""P1: load-balance shaping — φ penalizes uneven free-MPS across nodes."""
import pytest

pytest.importorskip("gymnasium")  # env construction needs gymnasium

from sim.gym_env import KubefluxSchedEnv
from sim.loader import generate_by_family


def _factory(seed=42):
    return [j for j in generate_by_family("philly", n_jobs=20, seed=seed)
            if j.gpu_count <= 2]


def test_imbalance_zero_when_balanced_and_positive_when_crowded():
    env = KubefluxSchedEnv(_factory, n_nodes=2, gpus_per_node=1,
                           balance_coef=1.0, max_steps=4000)
    env.reset(seed=0)
    cl = env._state.cluster
    # both nodes idle → balanced → imbalance 0
    assert env._imbalance() == 0.0
    # drain one node's GPU (simulate crowding) → imbalance > 0
    cl.nodes[0].gpus[0].free_mps = 0
    assert env._imbalance() > 0.0


def test_balance_shaping_off_by_default_and_potential_includes_term():
    env0 = KubefluxSchedEnv(_factory, n_nodes=2, gpus_per_node=1, max_steps=4000)
    env0.reset(seed=0)
    assert env0.balance_coef == 0.0
    # with balance on, crowding one node lowers the potential (more negative)
    env = KubefluxSchedEnv(_factory, n_nodes=2, gpus_per_node=1,
                           balance_coef=5.0, max_steps=4000)
    env.reset(seed=0)
    phi_balanced = env._potential()
    env._state.cluster.nodes[0].gpus[0].free_mps = 0
    phi_crowded = env._potential()
    assert phi_crowded < phi_balanced
