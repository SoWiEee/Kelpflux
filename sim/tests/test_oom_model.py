"""Tests for the opt-in stochastic host-RAM OOM model.

The model turns node-2's real failure mode (a job's peak host RSS is uncertain
at placement time, so a nominally-fitting co-residency can spike past the node
budget and OOM → drain) into a *penalised, learnable* tail event. These tests
pin the contract:

* off by default → bit-for-bit legacy behaviour (hard RAM gate, no penalty);
* ``ram_overcommit`` loosens the gate exactly as documented;
* the drain penalty fires iff summed realized peaks exceed the node budget;
* realized peaks are common-random (reproducible) and mean-preserving.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sim.cluster import Cluster
from sim.loader import Job
from sim.gym_env import KubefluxSchedEnv


def _llm(job_id: str, ram: float = 2.5) -> Job:
    return Job(job_id=job_id, user="u", gpu_count=1, gpu_type="any",
               submit_ts=0.0, runtime=100.0, mem_req=0.0, mps_req=1,
               job_class="llm", ram_req=ram)


def _env(**kw) -> KubefluxSchedEnv:
    env = KubefluxSchedEnv(
        jobs_factory=lambda: [_llm("j0")],
        n_nodes=1, gpus_per_node=1, node_ram_gb=[5.0], **kw)
    env.reset(seed=123)
    return env


# ── hard gate preserved (ram_overcommit default 1.0) ─────────────────────────

def test_hard_gate_unchanged_by_default():
    c = Cluster(n_nodes=1, gpus_per_node=1, node_ram_gb=[5.0])  # overcommit 1.0
    assert c.try_allocate(_llm("a")) is not None   # 2.5 ≤ 5
    assert c.try_allocate(_llm("b")) is not None   # 5.0 ≤ 5
    assert c.try_allocate(_llm("c")) is None       # 7.5 > 5 → refused (hard gate)


def test_overcommit_admits_over_budget_placement():
    c = Cluster(n_nodes=1, gpus_per_node=1, node_ram_gb=[5.0], ram_overcommit=2.0)
    assert c.try_allocate(_llm("a")) is not None   # 2.5 ≤ 10
    assert c.try_allocate(_llm("b")) is not None   # 5.0 ≤ 10
    assert c.try_allocate(_llm("c")) is not None   # 7.5 ≤ 10 → admitted (soft gate)


# ── penalty is off unless explicitly enabled ─────────────────────────────────

def test_penalty_off_by_default():
    env = _env()  # no oom params → oom_penalty_s == 0
    c = Cluster(n_nodes=1, gpus_per_node=1, node_ram_gb=[5.0], ram_overcommit=3.0)
    plan = c.try_allocate(_llm("a", ram=9.0))      # nominal 9 > 5 budget
    assert env._oom_penalty(plan, c) == 0.0        # disabled → no drain hit


# ── penalty fires when summed realized peaks exceed budget ───────────────────

def test_penalty_fires_on_overcommit_even_without_sigma():
    # oom_sigma=0 → peaks == nominal; overcommit lets 3 llm (7.5) onto the 5GB node.
    env = _env(oom_penalty_s=1000.0, oom_sigma=0.0, ram_overcommit=2.0)
    c = env._state.cluster
    for jid in ("a", "b", "c"):
        c.try_allocate(_llm(jid))                  # 2.5 each → 7.5 nominal
    plan = c.active["c"]
    assert env._oom_penalty(plan, c) == 1000.0     # 7.5 > 5 → OOM


def test_no_penalty_with_ram_headroom():
    # Two llm (5.0 nominal) exactly at budget, sigma small → typically no spike.
    env = _env(oom_penalty_s=1000.0, oom_sigma=0.0, ram_overcommit=2.0)
    c = env._state.cluster
    c.try_allocate(_llm("a"))                       # 2.5 ≤ 5, lots of headroom
    plan = c.active["a"]
    assert env._oom_penalty(plan, c) == 0.0         # 2.5 < 5 → safe


# ── realized-peak draw: reproducible + mean-preserving ───────────────────────

def test_ram_peak_reproducible_under_seed():
    env = _env(oom_penalty_s=1.0, oom_sigma=0.6)
    a = env._ram_peak("job-x", 2.5)
    b = env._ram_peak("job-x", 2.5)                 # same key → same draw
    assert a == b
    assert env._ram_peak("job-y", 2.5) != a         # different job → different luck


def test_ram_peak_zero_sigma_is_nominal():
    env = _env(oom_penalty_s=1.0, oom_sigma=0.0)
    assert env._ram_peak("job-x", 2.5) == 2.5


def test_ram_peak_mean_preserving():
    env = _env(oom_penalty_s=1.0, oom_sigma=0.5)
    env._episode_seed = None                        # sequential stream for sampling
    env._rng = np.random.default_rng(7)
    draws = [env._ram_peak(f"j{i}", 2.5) for i in range(20000)]
    assert math.isclose(float(np.mean(draws)), 2.5, rel_tol=0.03)


# ── realized runtime includes the drain hit end-to-end ───────────────────────

def test_realized_runtime_adds_oom_penalty():
    env = _env(oom_penalty_s=500.0, oom_sigma=0.0, ram_overcommit=2.0)
    c = env._state.cluster
    jobs = {jid: _llm(jid) for jid in ("a", "b", "c")}
    for jid in ("a", "b", "c"):
        c.try_allocate(jobs[jid])                   # 7.5 nominal on 5GB node
    rt = env._realized_runtime(jobs["c"], c.active["c"], c)
    assert rt >= 500.0                              # nominal 100 (÷speed) + 500 drain hit


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
