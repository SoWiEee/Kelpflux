"""Gymnasium wrapper around the discrete-event runner for DRL training.

MDP spec (placement-aware):
- State : top-K=16 pending jobs × 9 feats
          + 2 nodes × 2 GPUs × 6 feats   (GPU slot state)
          + 4 topology feats
          + 6 global feats
          = 178 dims
- Action: Discrete(N_JOBS × N_NODES × N_GPUS + 1)
          = 16 × 2 × 2 + 1 = 65
          action a = job_i * (N_NODES*N_GPUS) + node_j * N_GPUS + gpu_k
          action 64 = no-op
- Reward: r_placement (per-action) + r_completion (per-job-end)
"""
from __future__ import annotations

import dataclasses
import heapq
import math
import zlib
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None  # type: ignore
    spaces = None  # type: ignore

from .cluster import Cluster
from .loader import Job, MPS_PER_GPU, RAM_REQ_GB

# ── Layout constants ───────────────────────────────────────────────────────
TOP_K     = 16
# Cluster is rtx4070 today; rtx3080 is the only planned addition (2nd node).
# This 2-entry tuple is the per-job GPU one-hot alphabet → feeds JOB_FEAT_DIM.
GPU_TYPES = ("rtx4070", "rtx3080")

JOB_FEAT_DIM    = 9
GPU_FEAT_DIM    = 7   # +1: per-node free-RAM ratio (host-RAM OOM lever)
TOPO_FEAT_DIM   = 4
GLOBAL_FEAT_DIM = 6

# ── Measured cluster physics (see docs; benchmarked 2026-07 on the live cluster) ──
# Compute speed multiplier per (card, job_class), relative to the RTX 4070 (=1.0);
# realized_runtime = nominal / speed. Warm compute-only best-of-N: the two cards are
# NEAR-EQUAL — the RTX 3080 is ~1.1× the 4070, NOT the 4× slower that the old
# node_speeds=[1.0,0.25] implied. So the real placement lever is host RAM, not speed.
SPEED_MATRIX = {
    "rtx4070": {"inference": 1.00, "training": 1.00, "llm": 1.00, "batch": 1.00},
    "rtx3080": {"inference": 1.09, "training": 1.18, "llm": 1.17, "batch": 1.10},
}
# RAM_REQ_GB (measured peak host RSS per class) lives in loader.py as the single
# source of truth; imported above. node-2 (3080) has only ~5GB usable, so concurrent
# llm jobs there OOM — the dominant, real asymmetry on this cluster.
RAM_REF_GB = 4.0   # obs normalizer for per-job ram_req

# ── Cluster size — current deployment vs. target ───────────────────────────
# LIVE (current): 1 host × 1 GPU (RTX 4070, MPS enabled)
#   obs_dim  = 16*9 + 1*1*6 + 4 + 6 = 160
#   n_actions = 16*1*1 + 1 = 17
#
# SIM training default (2×2): mirrors target 2-host cluster
#   obs_dim  = 16*9 + 2*2*6 + 4 + 6 = 178
#   n_actions = 16*2*2 + 1 = 65
#
# HOW TO ADD A SECOND GPU:
#   1. Set N_NODES=2, N_GPUS=2 below (or pass n_nodes=2, gpus_per_node=2 to env)
#   2. Retrain DSAC from scratch (obs_dim 160→178, n_actions 17→65 — checkpoint
#      is NOT compatible; different network input/output shape)
#   3. Update rlpd_finetune.py CLI defaults to match
#   4. In Slurm: verify two worker nodes are registered and GRES is correct
N_NODES = 1   # current: single host  ← change to 2 when second GPU is online
N_GPUS  = 1   # current: single GPU   ← change to 2 when second GPU is online

# Derived defaults (reflect N_NODES / N_GPUS above)
N_PLACEMENTS = N_NODES * N_GPUS
N_ACTIONS    = TOP_K * N_PLACEMENTS + 1
NO_OP        = N_ACTIONS - 1
OBS_DIM      = (TOP_K * JOB_FEAT_DIM
                + N_NODES * N_GPUS * GPU_FEAT_DIM
                + TOPO_FEAT_DIM
                + GLOBAL_FEAT_DIM)


# ── Co-location modes (B: MPS co-location as an action dimension) ───────────
# Each (job, placement) action carries a mode when colocation_actions=True:
#   PACK    — accept MPS sharing: place even on an occupied GPU (pays the A
#             interference slowdown if co-residents are present).
#   ISOLATE — require an empty GPU: legal only when the target GPU is idle, so
#             the job never shares and never pays interference, at the cost of
#             waiting for the GPU to drain.
MODE_PACK    = 0
MODE_ISOLATE = 1


def env_dims(n_nodes: int, gpus_per_node: int, top_k: int = TOP_K,
             colocation: bool = False) -> tuple[int, int]:
    """Return (obs_dim, n_actions) for a given cluster shape.

    With ``colocation=True`` each (job, placement) gains a PACK/ISOLATE mode, so
    the action count doubles. Use this to construct a matching DSACAgent::

        obs_dim, n_actions = env_dims(n_nodes=1, gpus_per_node=1)
        agent = DSACAgent(obs_dim=obs_dim, n_actions=n_actions)
        env = KubefluxSchedEnv(..., n_nodes=1, gpus_per_node=1)
    """
    obs = (top_k * JOB_FEAT_DIM
           + n_nodes * gpus_per_node * GPU_FEAT_DIM
           + TOPO_FEAT_DIM
           + GLOBAL_FEAT_DIM)
    n_modes = 2 if colocation else 1
    n_act = top_k * n_nodes * gpus_per_node * n_modes + 1
    return obs, n_act

# Floor on a realized (post-noise) runtime so stochastic draws can't produce
# zero/negative durations.
MIN_RUNTIME_S = 1.0


# ── Feature extractors ────────────────────────────────────────────────────

def _job_feat(job: Job, now: float, mps_per_gpu: int) -> np.ndarray:
    """9-dim per-job feature vector.

      [6] ram_req / RAM_REF — host-RAM footprint, so the policy can tell a heavy
          (llm) job from a light one and avoid the small-RAM node. (Was a dead
          duplicate of the wait/age slot before RAM modelling.)
      [7] slo_urgency — fraction of the latency deadline already consumed
          (wait / slo_s; >1 = already late). 0 for best-effort jobs (slo_s==0).
      [8] is_inference — 1.0 for the latency class, 0.0 for training/batch.
    """
    gpu_oh = [1.0 if job.gpu_type == t else 0.0 for t in GPU_TYPES]
    wait = max(0.0, now - job.submit_ts)
    slo_urgency = (wait / job.slo_s) if job.slo_s > 0 else 0.0
    is_inference = 1.0 if job.job_class == "inference" else 0.0
    return np.array([
        job.mps_req / mps_per_gpu,
        float(job.gpu_count),
        *gpu_oh,                        # 2 dims (rtx4070 / rtx3080)
        math.log1p(job.runtime),
        math.log1p(wait),
        getattr(job, "ram_req", 0.0) / RAM_REF_GB,  # host-RAM footprint (was a dup wait slot)
        slo_urgency,                    # SLO budget consumed (0 = best-effort)
        is_inference,                   # latency-class flag
    ], dtype=np.float32)


def _gpu_feat(cluster: Cluster, node_i: int, gpu_i: int) -> np.ndarray:
    """7-dim per-GPU feature vector."""
    node = cluster.nodes[node_i]
    gpu  = node.gpus[gpu_i]
    total_mps = cluster.mps_per_gpu
    free_ratio  = gpu.free_mps / total_mps
    vram_ratio  = gpu.free_mps / total_mps   # proxy: same scale as MPS in sim
    running_on_gpu = sum(
        1 for plan in cluster.active.values()
        for alloc in plan
        if alloc.node_id == node_i and gpu_i in alloc.gpu_indices
    )
    # gpu_type one-hot — from the node's CARD IDENTITY, not its speed. The two
    # cards are near-equal in compute, so a speed-derived flag would be blind;
    # keying on identity keeps the card signal even when speeds are ~equal.
    gpu_type_oh = [1.0 if node.gpu_type == t else 0.0 for t in GPU_TYPES] + [
        0.0 if node.gpu_type in GPU_TYPES else 1.0]   # [is_4070, is_3080, is_other]
    return np.array([
        free_ratio,
        vram_ratio,
        float(running_on_gpu),
        node.free_ram_ratio(),   # host-RAM headroom — the OOM lever the policy must see
        *gpu_type_oh,
    ], dtype=np.float32)


def _topo_feat(pending: List[Job], cluster: Cluster) -> np.ndarray:
    """4-dim topology/pressure feature vector.

    The former two dims were dead bandwidth placeholders (sim has no network
    model); they now carry global host-RAM pressure, the real cross-node lever:
      [0] min free-RAM ratio across nodes (how tight the tightest node is)
      [1] fraction of pending jobs that are RAM-heavy (ram_req ≥ 2GB, ~llm class)
    """
    min_free_ram = min((n.free_ram_ratio() for n in cluster.nodes), default=1.0)
    heavy = sum(1 for j in pending if getattr(j, "ram_req", 0.0) >= 2.0)
    ram_heavy_ratio = heavy / max(1, len(pending))
    # fraction of pending jobs needing >1 GPU (DDP pressure)
    ddp_ratio = (sum(1 for j in pending if j.gpu_count > 1) / max(1, len(pending)))
    # number of currently running cross-node allocations
    cross_node = sum(
        1 for plan in cluster.active.values()
        if len({alloc.node_id for alloc in plan}) > 1
    )
    return np.array([min_free_ram, ram_heavy_ratio, ddp_ratio, float(cross_node)],
                    dtype=np.float32)


def _global_feat(pending: List[Job], cluster: Cluster, now: float) -> np.ndarray:
    """6-dim global feature vector."""
    queue_len = len(pending)
    # predictor_spread: p90/p50 of pending runtimes (oracle in sim)
    if len(pending) >= 2:
        rts = sorted(j.runtime for j in pending)
        n   = len(rts)
        p50 = rts[int(n * 0.50)]
        p90 = rts[min(int(n * 0.90), n - 1)]
        spread = (p90 / p50) if p50 > 0 else 1.0
    else:
        spread = 1.0
    # fragmentation: coefficient of variation of free MPS across nodes
    if cluster.n_nodes > 1:
        free_per_node = cluster.free_mps_per_node()
        mean = max(1.0, sum(free_per_node) / len(free_per_node))
        var  = sum((x - mean) ** 2 for x in free_per_node) / len(free_per_node)
        frag = math.sqrt(var) / mean
    else:
        frag = 0.0
    tod = (now % 86400) / 86400.0
    return np.array([
        math.log1p(queue_len),
        spread,
        frag,
        math.sin(2 * math.pi * tod),
        math.cos(2 * math.pi * tod),
        0.0,   # reserved
    ], dtype=np.float32)


# ── Action codec ──────────────────────────────────────────────────────────

def encode_action(job_i: int, node_j: int, gpu_k: int) -> int:
    """Encode (job, node, gpu) triple into a flat action index."""
    return job_i * N_PLACEMENTS + node_j * N_GPUS + gpu_k


def decode_action(a: int) -> Tuple[int, int, int]:
    """Decode flat action index into (job_i, node_j, gpu_k). Raises if no-op."""
    if a == NO_OP:
        raise ValueError("no-op has no (job, node, gpu) decomposition")
    job_i  = a // N_PLACEMENTS
    rem    = a %  N_PLACEMENTS
    node_j = rem // N_GPUS
    gpu_k  = rem %  N_GPUS
    return job_i, node_j, gpu_k


def uxprl_task_reward(jct: float, job_class: str, slo_s: float,
                      c1: float = 1.0, c2: float = 2.0) -> float:
    """UXP-RL per-task reward (Lin et al. 2025, IEEE TNSM §IV-B.3).

        r = c1 / T_T                                if non-inference
        r = c2 / T_T                                if inference, T_T ≤ Δ^I
        r = c2 / T_T · 1/(T_T − Δ^I + 1)            if inference, T_T > Δ^I

    T_T is the task turnaround (= jct here); inference tasks (``job_class ==
    "inference"``, the paper's Z=3) earn c2 > c1, and turnaround past the Δ^I
    deadline (``slo_s``) is damped. An inference task with ``slo_s <= 0`` is
    treated as having no deadline (Δ^I = ∞ → no penalty).
    """
    t_t = max(1e-6, jct)
    if job_class == "inference":
        delta = slo_s if slo_s > 0.0 else float("inf")
        r = c2 / t_t
        if t_t > delta:
            r *= 1.0 / (t_t - delta + 1.0)
        return r
    return c1 / t_t


# ── Run state ─────────────────────────────────────────────────────────────

@dataclass
class _RunState:
    cluster:          Cluster
    pending:          List[Job]
    events:           list
    seq:              int
    now:              float
    by_id:            dict
    completed:        int
    n_jobs:           int
    jct_sum:          float
    completion_reward: float
    jcts:             list = dataclasses.field(default_factory=list)
    jct_records:      list = dataclasses.field(default_factory=list)  # (job_id, jct)


# ── Environment ───────────────────────────────────────────────────────────

class KubefluxSchedEnv:
    """Gymnasium-compatible placement-aware scheduling environment.

    Action a ∈ {0 … 63}: schedule top-K job job_i on node node_j / GPU gpu_k.
    Action 64 (NO_OP)  : do nothing; simulator advances to the next event.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        jobs_factory: Callable[[], List[Job]],
        *,
        n_nodes: int = N_NODES,
        gpus_per_node: int = N_GPUS,
        mps_per_gpu: int = MPS_PER_GPU,
        top_k: int = TOP_K,
        max_steps: int = 50_000,
        reward_mode: str = "jct_aligned",   # "jct_aligned" | "shaped" | "uxprl"
        reward_scale: float = 1000.0,   # NOTE: env default = 1000 (used by RLPD/live,
        #   whose online-log logs −JCT/1000). sim_train.py passes 20000 for the base
        #   policy training (§5.1). Keep the two consistent per-pipeline; don't assume
        #   one global scale — mismatching offline↔online reward scales breaks RLPD.
        placement_reward_scale: float = 0.01,
        uxprl_c1: float = 1.0,   # UXP-RL reward weight for non-inference tasks
        uxprl_c2: float = 2.0,   # UXP-RL reward weight for inference tasks (c2 > c1)
        mo_w_jct: float = 1.0,   # reward_mode="mo": weight on the −JCT (throughput) term
        mo_w_util: float = 0.05, # reward_mode="mo": weight on the per-step GPU-util term
        potential_shaping: bool = False,
        balance_coef: float = 0.0,
        fairness_coef: float = 0.0,   # convex (quadratic) per-job JCT penalty (obj-changing)
        node_speeds: Optional[list] = None,
        node_gpu_types: Optional[list] = None,  # per-node card id (rtx4070/rtx3080)
        node_ram_gb: Optional[list] = None,     # per-node usable host RAM (GB)
        runtime_sigma: float = 0.0,
        interference: float = 0.0,
        # ── Stochastic host-RAM OOM (opt-in; oom_penalty_s=0 → fully off) ──
        oom_sigma: float = 0.0,
        oom_penalty_s: float = 0.0,
        ram_overcommit: float = 1.0,
        colocation_actions: bool = False,
        slo_penalty: float = 0.0,
        noop_penalty: float = 0.0,
        # Joint-vs-decoupled ablation (None = joint, the default full action space):
        #   "placement_only" — job selection is frozen to the score scheduler's top
        #                       pick; the agent only chooses the placement.
        #   "job_only"       — placement is frozen to first-fit (smallest legal GPU
        #                       index); the agent only chooses which job to run.
        # Applied purely by restricting action_mask(), so obs_dim / n_actions are
        # identical across arms and one DSACAgent shape trains all three.
        ablation_mode: Optional[str] = None,
    ) -> None:
        if gym is None:
            raise ImportError("gymnasium is not installed")
        if ablation_mode not in (None, "placement_only", "job_only"):
            raise ValueError(f"ablation_mode={ablation_mode!r}")
        if reward_mode not in ("jct_aligned", "shaped", "uxprl", "mo"):
            raise ValueError(f"reward_mode={reward_mode!r}")

        self.jobs_factory           = jobs_factory
        self.n_nodes                = n_nodes
        self.gpus_per_node          = gpus_per_node
        self.mps_per_gpu            = mps_per_gpu
        self.top_k                  = top_k
        self.max_steps              = max_steps
        self.reward_mode            = reward_mode
        self.reward_scale           = float(reward_scale)
        self.placement_reward_scale = float(placement_reward_scale)
        # UXP-RL (Lin et al. 2025) reward weights: inference tasks earn c2 > c1.
        self.uxprl_c1               = float(uxprl_c1)
        self.uxprl_c2               = float(uxprl_c2)
        # Multi-objective (reward_mode="mo") weights: −JCT (finish fast) vs +GPU
        # utilization (keep cards busy). These genuinely trade off on this cluster —
        # tight MPS packing raises utilization but its interference raises JCT.
        self.mo_w_jct               = float(mo_w_jct)
        self.mo_w_util              = float(mo_w_util)
        self.reward_betas: tuple    = (1.0, 0.0)   # (β_jct, β_slowdown)
        self.potential_shaping      = potential_shaping
        # P1 (anti-over-concentration): potential-based node-balance shaping.
        # φ gains a −balance_coef·imbalance term so moving toward an even free-MPS
        # split across nodes earns shaping reward (Ng et al. 1999: optimal policy
        # unchanged, only exploration is guided away from crowding one card).
        self.balance_coef           = float(balance_coef)
        self.fairness_coef          = float(fairness_coef)
        # Item-1 (node heterogeneity): per-node relative speed. None → homogeneous.
        # A slow card (e.g. 3080 at 0.25) runs jobs ~4× longer AND shows a distinct
        # gpu-type one-hot in the obs, so the policy can learn to route big/long
        # jobs to the fast card instead of being blind to which node is which.
        self.node_speeds            = list(node_speeds) if node_speeds else None
        self.node_gpu_types         = list(node_gpu_types) if node_gpu_types else None
        self.node_ram_gb            = list(node_ram_gb) if node_ram_gb else None
        # ── Stochastic execution (opt-in; 0/0 = legacy deterministic env) ──
        # runtime_sigma : multiplicative mean-preserving lognormal noise on the
        #                 *realized* runtime. The obs still shows the nominal
        #                 runtime, so this is genuine outcome uncertainty — it
        #                 gives Z_R real spread for the distributional critic.
        # interference  : per-co-resident slowdown on the realized runtime when
        #                 a job shares its GPU(s) via MPS. Crowded placements
        #                 run slower → placement quality + tail risk to manage.
        self.runtime_sigma          = float(runtime_sigma)
        self.interference           = float(interference)
        # ── Stochastic host-RAM OOM (opt-in; oom_penalty_s == 0 → off) ──
        # Models node-2's real failure mode: a job's *peak* host RSS is uncertain
        # at placement time, so a nominally-fitting co-residency can still spike
        # past the node budget and OOM → drain. Each running job draws a realized
        # peak = ram_req · lognormal(oom_sigma) (mean-preserving, so nominal is the
        # expectation and only the tail grows). When a placement makes a node's
        # summed realized peaks exceed its RAM budget, the incoming job eats an
        # oom_penalty_s recovery hit — a rare, catastrophic tail cost that a
        # risk-aware (CVaR) policy can learn to avoid by leaving RAM headroom. The
        # obs already exposes per-job ram_req and per-node free-RAM ratio, so the
        # signal is *avoidable* → genuinely learnable, not a noise floor.
        # ram_overcommit > 1.0 additionally lets nominally-over-budget placements
        # through (harsher regime); at 1.0 the hard gate stands and OOM comes only
        # from realized > nominal on tight packings.
        self.oom_sigma              = float(oom_sigma)
        self.oom_penalty_s          = float(oom_penalty_s)
        self.ram_overcommit         = float(ram_overcommit)
        # ── Co-location action mode (opt-in; 1 mode = legacy action space) ──
        self.colocation_actions     = bool(colocation_actions)
        self.ablation_mode          = ablation_mode
        self._n_modes               = 2 if colocation_actions else 1
        # SLO-aware reward (opt-in; 0 = legacy). On an aiserve job's completion a
        # latency-class job (slo_s>0) that finished past its deadline incurs a
        # lateness penalty ∝ (jct − slo_s), teaching the policy to prioritise
        # latency-critical inference over best-effort training.
        self.slo_penalty            = float(slo_penalty)
        # Anti-idle (opt-in; 0 = legacy). At every decision point _advance_to_decision
        # guarantees ≥1 schedulable job, so choosing NO_OP there is always idling
        # when work is available. A small penalty discourages the cold-start
        # "learn to no-op" collapse (env gives placement −0.01 / no-op 0, so an
        # untrained policy hides in no-op → low utilization). Only charged when a
        # real placement is actually legal.
        self.noop_penalty           = float(noop_penalty)

        self._step_count = 0
        self._state: Optional[_RunState] = None
        self._rng = np.random.default_rng(None)
        self._episode_seed: Optional[int] = None
        self._oom_events = 0   # count of drain-penalised placements this episode

        n_act = top_k * n_nodes * gpus_per_node * self._n_modes + 1
        obs_d = (top_k * JOB_FEAT_DIM
                 + n_nodes * gpus_per_node * GPU_FEAT_DIM
                 + TOPO_FEAT_DIM
                 + GLOBAL_FEAT_DIM)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_d,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(n_act)
        self._n_placements = n_nodes * gpus_per_node
        self._no_op        = n_act - 1
        self._score_sched  = None   # lazily built for score_warmup_action()

    # ── Gym API ──────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        # Dedicated stream for runtime-noise draws — reproducible per seed and
        # independent of any global np.random state. When a seed is given the
        # idiosyncratic per-job noise becomes common-random (keyed on the job
        # id), so a paired eval sees the same multiplier for the same job under
        # every policy. Seedless (training) resets fall back to this stream.
        self._episode_seed = seed
        self._rng = np.random.default_rng(seed)
        self._oom_events = 0
        jobs    = self.jobs_factory()
        cluster = Cluster(
            n_nodes=self.n_nodes,
            gpus_per_node=self.gpus_per_node,
            mps_per_gpu=self.mps_per_gpu,
            node_speeds=self.node_speeds,
            node_gpu_types=self.node_gpu_types,
            node_ram_gb=self.node_ram_gb,
            ram_overcommit=self.ram_overcommit,
        )
        events: list = []
        seq = 0
        for j in sorted(jobs, key=lambda x: x.submit_ts):
            heapq.heappush(events, (j.submit_ts, seq, "submit", j.job_id))
            seq += 1
        self._state = _RunState(
            cluster=cluster, pending=[], events=events, seq=seq,
            now=0.0, by_id={j.job_id: j for j in jobs},
            completed=0, n_jobs=len(jobs),
            jct_sum=0.0, completion_reward=0.0,
        )
        self._step_count = 0
        self._advance_to_decision()
        return self._build_obs(), {}

    def step(self, action: int):
        if self._state is None:
            raise RuntimeError("call reset() first")
        st = self._state
        self._step_count += 1

        top        = self._top_k_jobs()
        r_place    = 0.0
        scheduled  = False
        phi_prev   = self._potential()

        if action != self._no_op:
            job_i, node_j, gpu_k, _mode = self._decode(action)
            if job_i < len(top):
                chosen = top[job_i]
                plan   = st.cluster.try_allocate_on(chosen, node_j, gpu_k)
                if plan is not None:
                    st.pending.remove(chosen)
                    realized = self._realized_runtime(chosen, plan, st.cluster)
                    end_ts   = st.now + realized
                    heapq.heappush(st.events, (end_ts, st.seq, "end", chosen.job_id))
                    st.seq += 1
                    scheduled = True
                    r_place   = self._placement_reward(chosen, node_j, gpu_k, st.cluster)
                else:
                    r_place = -0.01   # infeasible pick
            else:
                r_place = -0.01
        elif self.noop_penalty > 0.0 and any(st.cluster.can_allocate(j) for j in top):
            r_place -= self.noop_penalty   # idling while schedulable work exists

        st.completion_reward = 0.0
        if not scheduled and st.events:
            t, _s, kind, payload = heapq.heappop(st.events)
            st.now = t
            if kind == "submit":
                st.pending.append(st.by_id[payload])
            elif kind == "end":
                self._on_job_end(payload)

        self._advance_to_decision()

        terminated = (st.completed >= st.n_jobs) and not st.events
        truncated  = self._step_count >= self.max_steps

        end_charge = 0.0
        if terminated or truncated:
            for j in st.pending:
                end_charge -= (st.now - j.submit_ts) / self.reward_scale

        if self.reward_mode == "uxprl":
            # Faithful UXP-RL reward is *only* the completion-based term (positive,
            # per-task, inference-weighted). No placement shaping, no final-pending
            # charge, no potential shaping — those belong to other methods.
            reward = st.completion_reward
        elif self.reward_mode == "mo":
            # Multi-objective scalarization: −w_jct·JCT (completion, throughput) +
            # w_util·utilization (per step, keep GPUs busy). st.completion_reward
            # already holds −JCT/scale from _on_job_end (mo branch).
            reward = (self.mo_w_jct * st.completion_reward
                      + self.mo_w_util * st.cluster.utilization())
            # P1 balance shaping also applies under mo (potential-based → optimum-
            # preserving, Ng et al. 1999); guarded so balance_coef=0 leaves mo unchanged.
            if self.potential_shaping or self.balance_coef > 0.0:
                reward += 0.99 * self._potential() - phi_prev
        else:
            reward = r_place + st.completion_reward + end_charge
            if self.potential_shaping or self.balance_coef > 0.0:
                reward += 0.99 * self._potential() - phi_prev
        obs    = self._build_obs()
        info   = {
            "now": st.now, "queue_len": len(st.pending),
            "completed": st.completed, "n_jobs": st.n_jobs,
            "jct_sum": st.jct_sum,
            "avg_jct": st.jct_sum / max(1, st.completed) if st.completed else float("nan"),
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ── Action mask ──────────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        return self.action_mask()

    def action_mask(self) -> np.ndarray:
        """Bool mask over Discrete(N_ACTIONS). True = legal.

        With colocation modes on, ISOLATE is legal only when the target GPU is
        idle (free_mps == capacity); PACK is legal whenever the job fits.
        """
        st   = self._state
        assert st is not None
        top  = self._top_k_jobs()
        mask = np.zeros(self.action_space.n, dtype=bool)
        cap  = st.cluster.mps_per_gpu
        for i, job in enumerate(top):
            for nj in range(self.n_nodes):
                for gk in range(self.gpus_per_node):
                    if not st.cluster.can_allocate_on(job, nj, gk):
                        continue
                    placement = nj * self.gpus_per_node + gk
                    gpu_idle  = st.cluster.nodes[nj].gpus[gk].free_mps == cap
                    for mode in range(self._n_modes):
                        if mode == MODE_ISOLATE and not gpu_idle:
                            continue
                        mask[self._encode(i, placement, mode)] = True
        mask[self._no_op] = True
        if self.ablation_mode is not None:
            mask = self._restrict_mask_for_ablation(mask, top)
        return mask

    def _restrict_mask_for_ablation(
        self, mask: np.ndarray, top: List[Job]
    ) -> np.ndarray:
        """Collapse the legal mask onto a single decision axis for the ablation.

        ``placement_only`` keeps only the score scheduler's top job (agent varies
        placement); ``job_only`` keeps each job's first-fit placement only (agent
        varies which job). The no-op stays legal in both. Returns a new mask.
        """
        legal = [a for a in np.flatnonzero(mask) if a != self._no_op]
        if not legal:
            return mask  # only no-op available; nothing to restrict

        if self.ablation_mode == "job_only":
            # np.flatnonzero yields ascending action indices, and _encode is
            # monotonic in (job_i, placement, mode), so the first legal action for
            # each job is its first-fit (smallest placement, PACK) — keep just it.
            kept: dict[int, int] = {}
            for a in legal:
                job_i = self._decode(a)[0]
                if job_i not in kept:
                    kept[job_i] = a
            new = np.zeros_like(mask)
            for a in kept.values():
                new[a] = True
            new[self._no_op] = True
            return new

        # placement_only: freeze job to the score scheduler's top legal pick.
        head = self._ablation_head_index(top, legal)
        new = np.zeros_like(mask)
        for a in legal:
            if self._decode(a)[0] == head:
                new[a] = True
        new[self._no_op] = True
        return new

    def _ablation_head_index(self, top: List[Job], legal: List[int]) -> int:
        """Index (into ``top``) of the score scheduler's top job that still has a
        legal action; falls back to the first job with any legal action."""
        st = self._state
        assert st is not None
        if self._score_sched is None:
            from .scheduler.score import ScoreScheduler
            self._score_sched = ScoreScheduler()
        legal_jobs = {self._decode(a)[0] for a in legal}
        ordered = self._score_sched.order(st.pending, st.cluster, now=st.now)
        top_ids = [j.job_id for j in top]
        for job in ordered:
            idx = top_ids.index(job.job_id) if job.job_id in top_ids else None
            if idx is not None and idx in legal_jobs:
                return idx
        return self._decode(legal[0])[0]

    def score_warmup_action(self, mask: np.ndarray, rng: np.random.Generator) -> int:
        """Score-heuristic action for warmup seeding.

        Mirrors the single-env warmup block in ``sim_train``: order pending jobs
        by the score scheduler, then return the PACK action on the first
        placement of the highest-ranked job that is currently legal; fall back to
        a uniform random legal action. Lets the vectorized rollout reproduce the
        same high-quality warmup as the single-env path (in-process, so it can
        read ``self._state`` — Async workers call this inside their worker).
        """
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return self._no_op
        st = self._state
        if st is None or not st.pending or st.cluster is None:
            return int(rng.choice(legal))
        if self._score_sched is None:
            from .scheduler.score import ScoreScheduler
            self._score_sched = ScoreScheduler()
        ordered = self._score_sched.order(st.pending, st.cluster, now=st.now)
        act = int(rng.choice(legal))
        for job in ordered:
            job_idx = next((i for i, j in enumerate(st.pending)
                            if j.job_id == job.job_id), None)
            if job_idx is not None:
                cand = self._encode(job_idx, 0, 0)   # PACK on first placement
                if cand < mask.shape[0] and mask[cand]:
                    act = int(cand)
                    break
        return act

    # ── Internals ────────────────────────────────────────────────────────

    def _encode(self, job_i: int, placement: int, mode: int) -> int:
        """(job, placement_flat, mode) → flat action index. Mirrors _decode.

        With ``_n_modes == 1`` this is identical to the legacy
        ``job_i * n_placements + placement`` index.
        """
        return (job_i * self._n_placements + placement) * self._n_modes + mode

    def _decode(self, a: int) -> Tuple[int, int, int, int]:
        mode   = a %  self._n_modes
        pj     = a // self._n_modes
        job_i  = pj // self._n_placements
        rem    = pj %  self._n_placements
        node_j = rem // self.gpus_per_node
        gpu_k  = rem %  self.gpus_per_node
        return job_i, node_j, gpu_k, mode

    def _top_k_jobs(self) -> List[Job]:
        st = self._state
        assert st is not None
        return sorted(st.pending, key=lambda j: j.submit_ts)[: self.top_k]

    def _advance_to_decision(self) -> None:
        st = self._state
        assert st is not None
        while st.events:
            if st.pending and any(
                st.cluster.can_allocate(j) for j in st.pending
            ):
                return
            t, _s, kind, payload = heapq.heappop(st.events)
            st.now = t
            if kind == "submit":
                st.pending.append(st.by_id[payload])
            elif kind == "end":
                self._on_job_end(payload)

    def _realized_runtime(self, job: Job, plan, cluster: Cluster) -> float:
        """Realized (executed) runtime = nominal × interference × lognormal noise.

        With ``runtime_sigma == 0`` and ``interference == 0`` this returns the
        nominal runtime *exactly* (no RNG draw), keeping the legacy env
        bit-for-bit deterministic. ``plan`` is the just-applied allocation, so
        the chosen job already appears in ``cluster.active``.
        """
        base = job.runtime
        # Node heterogeneity. When card identities are provided (node_gpu_types set),
        # use the measured per-(card, job_class) SPEED_MATRIX; otherwise fall back to
        # the legacy scalar node.speed (keeps homogeneous/old-hetero paths bit-exact).
        node_id = plan[0].node_id if plan else 0
        node = cluster.nodes[node_id]
        if cluster.node_gpu_types is not None and node.gpu_type in SPEED_MATRIX:
            speed = SPEED_MATRIX[node.gpu_type].get(job.job_class, 1.0)
        else:
            speed = node.speed
        if speed != 1.0:
            base = max(MIN_RUNTIME_S, base / speed)
        if self.runtime_sigma > 0.0 or self.interference > 0.0:
            mult = 1.0
            if self.interference > 0.0:
                k = self._co_residents(cluster, plan, job.job_id)
                mult *= 1.0 + self.interference * k      # crowded GPU runs slower
            if self.runtime_sigma > 0.0:
                z = self._job_noise(job.job_id)
                # mean-preserving lognormal: E[exp(σZ − σ²/2)] = 1, so only the
                # variance grows with σ — the mean JCT stays comparable across σ.
                mult *= math.exp(self.runtime_sigma * z - 0.5 * self.runtime_sigma ** 2)
            base = max(MIN_RUNTIME_S, base * mult)
        # Stochastic host-RAM OOM (opt-in). Additive drain-recovery hit when this
        # placement pushes a node's summed realized peaks past its RAM budget.
        # Independent of the runtime-noise draw and off unless oom_penalty_s > 0.
        if self.oom_penalty_s > 0.0:
            base += self._oom_penalty(plan, cluster)
        return max(MIN_RUNTIME_S, base)

    def _oom_penalty(self, plan, cluster: Cluster) -> float:
        """Drain-recovery penalty if this placement OOMs any node it lands on.

        The chosen job is already reserved in ``cluster.active`` / ``active_ram``,
        so summing realized peaks over a node's residents includes it. A node OOMs
        when that sum exceeds its host-RAM budget; the incoming job then eats a
        single ``oom_penalty_s`` recovery hit. Realized peaks are common-random per
        (episode_seed, job_id), so a paired eval OOMs the same jobs under every
        policy — isolating the policy's RAM-headroom choices from draw luck.
        """
        nodes_in_plan = {a.node_id for a in plan}
        for ni in nodes_in_plan:
            budget = cluster.nodes[ni].ram_gb
            peak_sum = 0.0
            for jid, p in cluster.active.items():
                if any(a.node_id == ni for a in p):
                    peak_sum += self._ram_peak(jid, cluster.active_ram.get(jid, 0.0))
            if peak_sum > budget + 1e-9:
                self._oom_events += 1
                return self.oom_penalty_s
        return 0.0

    def _ram_peak(self, job_id: str, ram_req: float) -> float:
        """Realized peak host RSS = ram_req · mean-preserving lognormal(oom_sigma).

        Keyed independently of the runtime-noise stream (salt 0x52414d = 'RAM') so
        a job's memory luck and runtime luck are uncorrelated. With oom_sigma == 0
        this is exactly ram_req (peaks equal nominal → OOM only under overcommit).
        """
        if ram_req <= 0.0 or self.oom_sigma <= 0.0:
            return ram_req
        if self._episode_seed is None:
            z = float(self._rng.standard_normal())
        else:
            key = (int(self._episode_seed), int(zlib.crc32(job_id.encode())), 0x52414D)
            z = float(np.random.default_rng(key).standard_normal())
        return ram_req * math.exp(self.oom_sigma * z - 0.5 * self.oom_sigma ** 2)

    def _job_noise(self, job_id: str) -> float:
        """Standard-normal draw for a job's idiosyncratic runtime noise.

        Common-random per (episode_seed, job_id) so a paired eval gives the
        same multiplier to the same job under every policy — isolating the
        policy effect from noise luck. Seedless (training) resets draw from the
        sequential stream instead.
        """
        if self._episode_seed is None:
            return float(self._rng.standard_normal())
        key = (int(self._episode_seed), int(zlib.crc32(job_id.encode())))
        return float(np.random.default_rng(key).standard_normal())

    @staticmethod
    def _co_residents(cluster: Cluster, plan, job_id: str) -> int:
        """Count other active jobs sharing any GPU in ``plan`` (MPS neighbours)."""
        gpus = {(a.node_id, gi) for a in plan for gi in a.gpu_indices}
        others: set = set()
        for jid, p in cluster.active.items():
            if jid == job_id:
                continue
            if any((a.node_id, gi) in gpus for a in p for gi in a.gpu_indices):
                others.add(jid)
        return len(others)

    def _on_job_end(self, jid: str) -> None:
        st = self._state
        assert st is not None
        st.cluster.release(jid)
        st.completed += 1
        j   = st.by_id[jid]
        jct = st.now - j.submit_ts
        st.jct_sum += jct
        st.jcts.append(jct)
        st.jct_records.append((jid, jct))
        if self.reward_mode == "uxprl":
            st.completion_reward += uxprl_task_reward(
                jct, j.job_class, j.slo_s, self.uxprl_c1, self.uxprl_c2
            ) / self.reward_scale
        elif self.reward_mode == "shaped":
            b_jct, b_slow = self.reward_betas
            runtime  = max(1.0, j.runtime)
            slowdown = max(1.0, jct / runtime)
            st.completion_reward += (b_jct * (-jct / self.reward_scale)
                                     + b_slow * (-math.log(slowdown)))
        else:  # jct_aligned / mo
            st.completion_reward += -jct / self.reward_scale
            # Fairness/anti-starvation: a CONVEX per-job penalty on JCT. Squaring the
            # normalized JCT makes the objective min Σ(JCT + fairness·JCT²) = mean + a
            # tail/variance term. Unlike potential shaping (optimum-preserving), this
            # CHANGES the objective — trading a little mean for a bounded worst case,
            # which is exactly what the mean-JCT reward could not express.
            if self.fairness_coef > 0.0:
                st.completion_reward += -self.fairness_coef * (jct / self.reward_scale) ** 2
        # SLO lateness penalty (opt-in): latency-class jobs past deadline.
        if self.slo_penalty > 0.0 and j.slo_s > 0.0 and jct > j.slo_s:
            st.completion_reward += -self.slo_penalty * (jct - j.slo_s) / self.reward_scale

    def _placement_reward(
        self, job: Job, node_j: int, gpu_k: int, cluster: Cluster
    ) -> float:
        """Dense reward shaping from score function factors (scaled small)."""
        total = cluster.mps_per_gpu
        gpu   = cluster.nodes[node_j].gpus[gpu_k]
        # f_mps_fit: prefer GPU where MPS utilization is high after placement
        remaining = gpu.free_mps - job.mps_req
        mps_fit   = 1.0 - remaining / total  # higher = tighter fit
        # f_fragmentation: penalise uneven free MPS across all GPUs
        free_all = [
            cluster.nodes[ni].gpus[gi].free_mps
            for ni in range(cluster.n_nodes)
            for gi in range(cluster.gpus_per_node)
        ]
        mean = max(1.0, sum(free_all) / len(free_all))
        var  = sum((x - mean) ** 2 for x in free_all) / len(free_all)
        frag = math.sqrt(var) / mean
        r = (0.4 * mps_fit - 0.2 * frag)
        return r * self.placement_reward_scale

    def _build_obs(self) -> np.ndarray:
        st = self._state
        assert st is not None
        top = self._top_k_jobs()

        job_feats = []
        for i in range(self.top_k):
            if i < len(top):
                job_feats.append(_job_feat(top[i], st.now, self.mps_per_gpu))
            else:
                job_feats.append(np.zeros(JOB_FEAT_DIM, dtype=np.float32))

        gpu_feats = []
        for ni in range(self.n_nodes):
            for gi in range(self.gpus_per_node):
                if ni < st.cluster.n_nodes and gi < st.cluster.gpus_per_node:
                    gpu_feats.append(_gpu_feat(st.cluster, ni, gi))
                else:
                    gpu_feats.append(np.zeros(GPU_FEAT_DIM, dtype=np.float32))

        topo   = _topo_feat(st.pending, st.cluster)
        glob   = _global_feat(st.pending, st.cluster, st.now)

        return np.concatenate([*job_feats, *gpu_feats, topo, glob]).astype(np.float32)

    def _potential(self) -> float:
        """φ(s) = −Σ wait_time_i / (reward_scale × n_jobs) for potential-based shaping.

        Ng et al. 1999: F(s,s') = γφ(s') − φ(s) preserves the optimal policy.
        Scheduling a job reduces total wait time → positive shaping bonus each step.
        """
        if self._state is None:
            return 0.0
        st  = self._state
        now = st.now
        n   = max(1, st.n_jobs)
        phi = -sum(max(0.0, now - j.submit_ts)
                   for j in st.pending) / (self.reward_scale * n)
        if self.balance_coef > 0.0 and st.cluster.n_nodes > 1:
            phi -= self.balance_coef * self._imbalance()
        return phi

    def _imbalance(self) -> float:
        """Normalized free-MPS imbalance across nodes (0 = perfectly balanced).

        std(free_mps_per_node) / mps_per_gpu — penalizes concentrating load on a
        subset of nodes. Used by the P1 balance-shaping potential.
        """
        if self._state is None:
            return 0.0
        per_node = self._state.cluster.free_mps_per_node()
        if len(per_node) < 2:
            return 0.0
        mean = sum(per_node) / len(per_node)
        var  = sum((x - mean) ** 2 for x in per_node) / len(per_node)
        return math.sqrt(var) / max(1.0, float(self.mps_per_gpu))

    def episode_jcts(self) -> list[float]:
        """Per-job JCTs of completed jobs this episode (for tail metrics)."""
        return [] if self._state is None else list(self._state.jcts)

    def episode_records(self) -> list[dict]:
        """Per-completed-job records this episode for SLO/per-class analysis:
        {job_id, jct, slo_s, job_class, gpu_count, mps_req}."""
        if self._state is None:
            return []
        st = self._state
        out = []
        for jid, jct in st.jct_records:
            j = st.by_id[jid]
            out.append({"job_id": jid, "jct": jct, "slo_s": j.slo_s,
                        "job_class": j.job_class, "gpu_count": j.gpu_count,
                        "mps_req": j.mps_req})
        return out

    def cluster_utilization(self) -> float:
        return 0.0 if self._state is None else self._state.cluster.utilization()

    def render(self):
        return None

    def close(self):
        self._state = None
