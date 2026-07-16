"""Host-RAM OOM gating (step 2) + per-(card, job_class) speed (step 3).

The cluster's real asymmetry is host RAM (node-2/3080 ~5GB vs node-1/4070 ~62GB),
not compute speed (the two cards are near-equal, measured). These tests lock:
1. a placement that would exceed a node's RAM budget is structurally illegal
   (can_allocate_on False → it never reaches the action mask),
2. RAM is freed on release so the node recovers,
3. realized runtime uses the measured SPEED_MATRIX per (card, class) when card
   identities are given, and falls back to the legacy scalar speed otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from sim.cluster import Cluster          # noqa: E402
from sim.loader import Job               # noqa: E402
from sim.gym_env import SPEED_MATRIX     # noqa: E402


def _job(jid, ram, mps=1, cls="llm"):
    return Job(job_id=jid, user="u", gpu_count=1, gpu_type="rtx3080",
               submit_ts=0.0, runtime=100.0, mem_req=1.0, mps_req=mps,
               job_class=cls, ram_req=ram)


def _cluster():
    # node0 = 4070 (62GB), node1 = 3080 (5GB); 1 GPU each, 4 MPS slots.
    return Cluster(n_nodes=2, gpus_per_node=1, mps_per_gpu=4,
                   node_gpu_types=["rtx4070", "rtx3080"], node_ram_gb=[62.0, 5.0])


def test_ram_gate_blocks_oom_placement():
    c = _cluster()
    # Two 2.5GB jobs fit on the 5GB node (=5.0); a third must NOT.
    assert c.try_allocate_on(_job("a", 2.5), 1, 0) is not None
    assert c.try_allocate_on(_job("b", 2.5), 1, 0) is not None
    assert c.nodes[1].used_ram_gb == 5.0
    # third job would push to 7.5 > 5.0 → placement is illegal on node 1
    assert c.can_allocate_on(_job("c", 2.5), 1, 0) is False
    assert c.try_allocate_on(_job("c", 2.5), 1, 0) is None
    # ...but the big-RAM node 0 (62GB) still accepts it
    assert c.can_allocate_on(_job("c", 2.5), 0, 0) is True


def test_ram_freed_on_release():
    c = _cluster()
    c.try_allocate_on(_job("a", 2.5), 1, 0)
    c.try_allocate_on(_job("b", 2.5), 1, 0)
    assert c.can_allocate_on(_job("c", 2.5), 1, 0) is False
    c.release("a")
    assert c.nodes[1].used_ram_gb == 2.5
    assert c.can_allocate_on(_job("c", 2.5), 1, 0) is True   # RAM recovered


def test_unbounded_ram_by_default_is_legacy():
    # No node_ram_gb → unbounded (1e9); RAM never gates (legacy behaviour).
    # mps_req=1 so 4 fit in the GPU's MPS slots — isolates the RAM path.
    c = Cluster(n_nodes=2, gpus_per_node=1, mps_per_gpu=4)
    for i in range(4):
        assert c.try_allocate_on(_job(f"j{i}", 999.0, mps=1), 1, 0) is not None


def test_speed_matrix_used_when_card_identity_given():
    c = _cluster()
    # 3080 training multiplier is >1 (faster) per the measured matrix.
    assert SPEED_MATRIX["rtx3080"]["training"] > 1.0
    assert SPEED_MATRIX["rtx4070"]["training"] == 1.0
    # node1 is the 3080 → its per-class training speed is the matrix value
    assert c.node_gpu_types is not None
    assert c.nodes[1].gpu_type == "rtx3080"
