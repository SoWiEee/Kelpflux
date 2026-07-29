"""RLPD-style Sim2Real fine-tune for the placement-aware DSAC agent.

RLPD (Ball et al., ICML 2023) — "Efficient Online RL with Offline Data".
Key idea: keep an offline replay buffer (from sim) and an online buffer
(from live cluster), each training batch drawn 50/50 from both.
LayerNorm + high UTD ratio closes the sim-to-real gap in ~10^3 live
transitions.

Pieces:
  ReplayBuffer         — FIFO numpy buffer (obs/act/rew/next_obs/done + masks)
  collect_sim_rollouts — fill offline buffer with uniform-random sim rollouts
  load_live_shadow_log — import live transitions from daemon JSONL logs
  rlpd_train           — 50/50 batch sampler + DSAC gradient loop

Run (after live_daemon has collected shadow-mode transitions):
    .venv-m11/bin/python -m services.rl_scheduler.rlpd_finetune \\
        --offline-steps 50000 --online-log shadow_logs/*.jsonl \\
        --out-dir runs/rlpd_$(date +%Y%m%d-%H%M%S)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from sim.gym_env import KubefluxSchedEnv
from sim.loader import load_auto
from sim.scheduler.score import ScoreScheduler
from services.rl_scheduler.dsac import _CategoricalActor, _build_mlp


class SumTree:
    """Binary SumTree for O(log n) priority sampling (used by PER).

    Leaves at indices [capacity, 2*capacity). tree[1] = total priority sum.
    tree[i] = tree[2i] + tree[2i+1].
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)

    def update(self, data_idx: int, priority: float) -> None:
        idx = data_idx + self.capacity
        self.tree[idx] = priority
        while idx > 1:
            idx >>= 1
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]

    def sample_one(self, val: float) -> int:
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if val <= self.tree[left]:
                idx = left
            else:
                val -= self.tree[left]
                idx = left + 1
        return idx - self.capacity

    @property
    def total(self) -> float:
        return float(self.tree[1])

    @property
    def max_priority(self) -> float:
        leaf_max = float(self.tree[self.capacity:].max())
        return leaf_max if leaf_max > 0 else 1.0


class PrioritizedReplayBuffer:
    """Replay buffer with Prioritized Experience Replay (PER, Schaul et al. 2016).

    Same add() / sample() interface as ReplayBuffer.  sample() additionally
    returns 'weights' (IS correction) and 'indices' (for priority updates).
    Call update_priorities(indices, td_errors) after each agent.update().
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        n_actions: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 100_000,
    ) -> None:
        self.capacity   = capacity
        self.obs_dim    = obs_dim
        self.n_actions  = n_actions
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_end   = beta_end
        self.beta_steps = beta_steps
        self._beta_step = 0
        self._tree      = SumTree(capacity)

        self.obs       = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.acts      = np.zeros((capacity,),            dtype=np.int64)
        self.rews      = np.zeros((capacity,),            dtype=np.float32)
        self.next_obs  = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.dones     = np.zeros((capacity,),            dtype=np.bool_)
        self.masks     = np.ones((capacity, n_actions),   dtype=np.bool_)
        self.next_masks= np.ones((capacity, n_actions),   dtype=np.bool_)
        self.gammas    = np.full((capacity,), 0.99,       dtype=np.float32)
        self._size     = 0
        self._idx      = 0

    def __len__(self) -> int:
        return self._size

    def add(self, t: "Transition", gamma: float = 0.99) -> None:
        i = self._idx
        self.obs[i]        = t.obs
        self.acts[i]       = t.act
        self.rews[i]       = t.rew
        self.next_obs[i]   = t.next_obs
        self.dones[i]      = t.done
        self.gammas[i]     = gamma
        if t.mask is not None:
            self.masks[i]      = t.mask
        if t.next_mask is not None:
            self.next_masks[i] = t.next_mask
        # New transitions get max priority → sampled at least once before update
        priority = self._tree.max_priority ** self.alpha
        self._tree.update(i, priority)
        self._idx  = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    @property
    def _beta(self) -> float:
        frac = min(1.0, self._beta_step / max(1, self.beta_steps))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def sample(self, n: int, rng: np.random.Generator) -> dict:
        total   = self._tree.total
        segment = total / n
        indices    = np.zeros(n, dtype=np.int64)
        priorities = np.zeros(n, dtype=np.float64)
        for i in range(n):
            val      = rng.uniform(segment * i, segment * (i + 1))
            data_idx = self._tree.sample_one(val)
            data_idx = min(data_idx, self._size - 1)
            indices[i]    = data_idx
            priorities[i] = self._tree.tree[data_idx + self._tree.capacity]

        probs   = priorities / total
        beta    = self._beta
        self._beta_step += 1
        weights = (self._size * probs) ** (-beta)
        weights /= weights.max()

        return {
            "obs":        self.obs[indices],
            "acts":       self.acts[indices],
            "rews":       self.rews[indices],
            "next_obs":   self.next_obs[indices],
            "dones":      self.dones[indices],
            "masks":      self.masks[indices],
            "next_masks": self.next_masks[indices],
            "gammas":     self.gammas[indices],
            "weights":    weights.astype(np.float32),
            "indices":    indices,
        }

    def update_priorities(
        self, indices: np.ndarray, td_errors: np.ndarray
    ) -> None:
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        for idx, p in zip(indices, priorities):
            self._tree.update(int(idx), float(p))


@dataclass
class Transition:
    obs: np.ndarray
    act: int
    rew: float
    next_obs: np.ndarray
    done: bool
    mask: Optional[np.ndarray] = None       # action_mask at obs
    next_mask: Optional[np.ndarray] = None  # action_mask at next_obs (needed by DSAC critic)


@dataclass
class ReplayBuffer:
    """FIFO numpy-backed replay. Stored as parallel arrays for cheap batch
    sampling. Pre-allocates to capacity to avoid repeated np.append.

    The `gammas` field stores the effective discount for each transition.
    For 1-step TD this is always γ.  For n-step returns (n>1) the caller
    pre-computes the discounted sum r + γr' + ... + γ^{n-1}r'' and stores
    γ^n here so the critic target becomes:
        y = n_step_return + gammas * (1 - done) * V(s_{t+n})
    """
    capacity: int
    obs_dim: int
    n_actions: int
    obs: np.ndarray = field(init=False)
    acts: np.ndarray = field(init=False)
    rews: np.ndarray = field(init=False)
    next_obs: np.ndarray = field(init=False)
    dones: np.ndarray = field(init=False)
    masks: np.ndarray = field(init=False)
    next_masks: np.ndarray = field(init=False)
    gammas: np.ndarray = field(init=False)   # γ^n per transition
    _size: int = 0
    _idx: int = 0

    def __post_init__(self):
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.acts = np.zeros((self.capacity,), dtype=np.int64)
        self.rews = np.zeros((self.capacity,), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)
        self.masks = np.ones((self.capacity, self.n_actions), dtype=np.bool_)
        self.next_masks = np.ones((self.capacity, self.n_actions), dtype=np.bool_)
        self.gammas = np.full((self.capacity,), 0.99, dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def add(self, t: Transition, gamma: float = 0.99) -> None:
        i = self._idx
        self.obs[i] = t.obs
        self.acts[i] = t.act
        self.rews[i] = t.rew
        self.next_obs[i] = t.next_obs
        self.dones[i] = t.done
        self.gammas[i] = gamma
        if t.mask is not None:
            self.masks[i] = t.mask
        if t.next_mask is not None:
            self.next_masks[i] = t.next_mask
        self._idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, n: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, self._size, size=n)
        return {
            "obs": self.obs[idx], "acts": self.acts[idx],
            "rews": self.rews[idx], "next_obs": self.next_obs[idx],
            "dones": self.dones[idx], "masks": self.masks[idx],
            "next_masks": self.next_masks[idx],
            "gammas": self.gammas[idx],
        }


def collect_sim_rollouts(*, n_transitions: int, trace_family: str,
                         n_jobs: int, n_nodes: int, gpus_per_node: int,
                         seed: int = 42) -> ReplayBuffer:
    """Fill offline buffer using a uniform-random masked policy in sim.

    We deliberately avoid the trained policy here — we want diverse coverage
    of the state space, not the policy's narrow on-policy trajectory.
    """
    from sim.loader import generate_by_family

    rng = np.random.default_rng(seed)

    total_gpus = n_nodes * gpus_per_node

    def _factory():
        jobs = generate_by_family(trace_family, n_jobs=n_jobs,
                                   seed=int(rng.integers(0, 2**31 - 1)))
        return [j for j in jobs if j.gpu_count <= total_gpus]

    env = KubefluxSchedEnv(
        _factory, n_nodes=n_nodes, gpus_per_node=gpus_per_node,
        max_steps=n_jobs * 100,
    )
    obs_dim   = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    buf = ReplayBuffer(capacity=n_transitions, obs_dim=obs_dim, n_actions=n_actions)

    obs, _ = env.reset()
    while len(buf) < n_transitions:
        mask  = env.action_mask()
        legal = np.flatnonzero(mask)
        act   = int(rng.choice(legal)) if len(legal) else 0
        next_obs, rew, term, trunc, _ = env.step(act)
        next_mask = env.action_mask()
        buf.add(Transition(
            obs=obs.astype(np.float32), act=act, rew=float(rew),
            next_obs=next_obs.astype(np.float32),
            done=bool(term or trunc), mask=mask, next_mask=next_mask,
        ))
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return buf


def _add_transition(buf: ReplayBuffer, t: Transition) -> None:
    if len(buf) >= buf.capacity:
        return
    buf.add(t)


def _score_demo_action(env: KubefluxSchedEnv, scheduler: ScoreScheduler) -> int:
    """Pick a legal action using the same score heuristic as the live fallback.

    This turns a normalized live sacct trace into RLPD demonstration data. The
    trace supplies real arrivals, runtimes, MPS requests, and observed latency
    class; the action is the score baseline's choice under the current simulated
    resource state. That makes the online buffer immediately usable even before
    we have full shadow-mode obs/next_obs logs for every live decision.
    """
    st = env._state  # noqa: SLF001 - replay adapter intentionally uses env internals
    assert st is not None
    top = env._top_k_jobs()  # noqa: SLF001
    mask = env.action_mask()
    best: tuple[float, int] | None = None
    for job_i, job in enumerate(top):
        base = scheduler.score(job, st.cluster)
        for node_j in range(env.n_nodes):
            for gpu_k in range(env.gpus_per_node):
                action = job_i * env._n_placements + node_j * env.gpus_per_node + gpu_k  # noqa: SLF001
                if action >= len(mask) or not mask[action]:
                    continue
                gpu = st.cluster.nodes[node_j].gpus[gpu_k]
                residual = gpu.free_mps - job.mps_req
                placement_fit = 1.0 - max(0.0, residual) / max(1, st.cluster.mps_per_gpu)
                value = base + 0.05 * placement_fit
                if best is None or value > best[0]:
                    best = (value, action)
    return best[1] if best is not None else env._no_op  # noqa: SLF001


def load_live_trace_rollouts(
    paths: list[str],
    *,
    obs_dim: int,
    n_actions: int,
    capacity: int,
    n_nodes: int,
    gpus_per_node: int,
    n_jobs: int,
) -> ReplayBuffer:
    """Replay normalized live traces into score-demonstration transitions.

    The input is the JSON emitted by scripts/collect-live-trace.py. Unlike
    load_live_shadow_log(), this does not require prior shadow-mode JSONL logs;
    it reconstructs obs/action/reward/next_obs by running the same trace through
    KubefluxSchedEnv and choosing score-policy actions.
    """
    buf = ReplayBuffer(capacity=capacity, obs_dim=obs_dim, n_actions=n_actions)
    files: list[str] = []
    for pattern in paths:
        files.extend(glob.glob(pattern))
    if not files:
        print(f"[rlpd] no normalized live trace files matched {paths}", file=sys.stderr)
        return buf

    scheduler = ScoreScheduler()
    for fp in files:
        jobs = load_auto(fp)
        if n_jobs > 0:
            jobs = jobs[:n_jobs]
        if not jobs:
            continue
        env = KubefluxSchedEnv(
            lambda _jobs=jobs: list(_jobs),
            n_nodes=n_nodes,
            gpus_per_node=gpus_per_node,
            max_steps=max(1000, len(jobs) * 200),
            reward_mode="jct_aligned",
        )
        if env.observation_space.shape[0] != obs_dim or env.action_space.n != n_actions:
            env.close()
            raise ValueError(
                f"live trace env dims mismatch for {fp}: "
                f"obs={env.observation_space.shape[0]} actions={env.action_space.n}; "
                f"expected obs={obs_dim} actions={n_actions}"
            )
        obs, _ = env.reset()
        done = False
        while not done and len(buf) < buf.capacity:
            mask = env.action_mask()
            act = _score_demo_action(env, scheduler)
            next_obs, rew, term, trunc, _info = env.step(act)
            next_mask = env.action_mask()
            _add_transition(buf, Transition(
                obs=obs.astype(np.float32),
                act=int(act),
                rew=float(rew),
                next_obs=next_obs.astype(np.float32),
                done=bool(term or trunc),
                mask=mask,
                next_mask=next_mask,
            ))
            obs = next_obs
            done = bool(term or trunc)
        env.close()
        if len(buf) >= buf.capacity:
            break
    return buf


def load_live_shadow_log(paths: list[str], *, obs_dim: int,
                          n_actions: int, capacity: int) -> ReplayBuffer:
    """Parse Phase D shadow-mode log lines (one JSON per /decide call) and
    materialise them as transitions.

    Expected line schema (emitted by Phase D log shipper — TBD):
        {"obs": [...], "act": int, "rew": float,
         "next_obs": [...], "done": bool, "mask": [bool ...]}

    Real-cluster reward is computed offline by joining each /decide row
    with the eventual JCT of the selected job (see Phase D pipeline)."""
    buf = ReplayBuffer(capacity=capacity, obs_dim=obs_dim, n_actions=n_actions)
    files = []
    for p in paths:
        files.extend(glob.glob(p))
    if not files:
        print(f"[rlpd] no shadow log files matched {paths}; "
              f"online buffer will be empty", file=sys.stderr)
        return buf
    for fp in files:
        with open(fp) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "obs" not in row or "next_obs" not in row:
                    continue
                buf.add(Transition(
                    obs=np.asarray(row["obs"], dtype=np.float32),
                    act=int(row.get("act", 0)),
                    rew=float(row.get("rew", 0.0)),
                    next_obs=np.asarray(row["next_obs"], dtype=np.float32),
                    done=bool(row.get("done", False)),
                    mask=np.asarray(row.get("mask",
                                            [True] * n_actions),
                                    dtype=bool),
                ))
    return buf


def mixed_batch(*, offline: ReplayBuffer, online: ReplayBuffer,
                batch_size: int, online_ratio: float,
                rng: np.random.Generator) -> dict:
    """RLPD core: each batch is online_ratio from live, rest from sim.
    If online is empty (e.g. cold-start before Phase D), fall back to
    100% offline."""
    if len(online) == 0:
        return offline.sample(batch_size, rng)
    n_online = max(1, int(batch_size * online_ratio))
    n_offline = batch_size - n_online
    a = online.sample(n_online, rng)
    b = offline.sample(n_offline, rng)
    out = {}
    for k in a:
        out[k] = np.concatenate([a[k], b[k]], axis=0)
    return out


class _EnsembleCritic(torch.nn.Module):
    """N independent discrete Q-nets, each a LayerNorm MLP obs→[n_actions].

    The LayerNorm (in ``_build_mlp``) and the ensemble are the two ingredients
    RLPD (Ball et al. 2023) identifies as essential for learning from offline
    data without value divergence / overestimation.
    """

    def __init__(self, obs_dim: int, n_actions: int, n_critics: int = 10,
                 hidden=(256, 256), layer_norm: bool = True) -> None:
        super().__init__()
        self.q = torch.nn.ModuleList(
            [_build_mlp(obs_dim, hidden, n_actions, layer_norm)
             for _ in range(n_critics)])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:   # → (N, B, A)
        return torch.stack([c(obs) for c in self.q], dim=0)


class RLPDAgent:
    """Faithful RLPD (Ball, Smith, Kostrikov, Levine 2023, arXiv:2302.02948),
    adapted to discrete masked actions (discrete SAC, Christodoulou 2019):

      • symmetric sampling — every gradient batch is 50 % offline (sim prior) +
        50 % online (real), regardless of buffer sizes (done by ``mixed_batch``);
      • LayerNorm critics in a large ENSEMBLE (default 10);
      • random subset of ``subset`` critics (default 2) min-reduced for the
        target → clipped-double-Q / REDQ-style pessimism that curbs the
        offline-data overestimation RLPD warns about;
      • SAC max-entropy backup with automatic temperature α;
      • high UTD (the caller loops ``utd_ratio`` updates per env step).

    Not a warm-started copy of a pretrained net: RLPD's premise is online RL
    *with* offline data, so the ensemble/critics train from scratch while the
    sim buffer supplies the prior via symmetric sampling. The actor may be
    warm-started from the sim policy (same _CategoricalActor architecture).
    """

    def __init__(self, obs_dim: int, n_actions: int, *, n_critics: int = 10,
                 subset: int = 2, hidden=(256, 256), gamma: float = 0.99,
                 tau: float = 0.005, lr: float = 3e-4,
                 target_entropy_ratio: float = 0.5, layer_norm: bool = True,
                 fixed_alpha: bool = False, init_alpha: float = 0.05,
                 device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = tuple(hidden)
        self.subset = min(subset, n_critics)
        self.gamma = gamma
        self.tau = tau
        # Auto-α (SAC/RLPD default) rails to huge values here because the action
        # mask leaves only ~2-3 legal actions, so max entropy ≪ ratio·log(A) and
        # the target is unreachable → α→∞ → near-random policy. fixed_alpha pins
        # it (matches the sim training recipe; robust for masked discrete SAC).
        self.fixed_alpha = fixed_alpha
        self.actor = _CategoricalActor(obs_dim, n_actions, hidden, layer_norm).to(self.device)
        self.q = _EnsembleCritic(obs_dim, n_actions, n_critics, hidden, layer_norm).to(self.device)
        self.q_targ = _EnsembleCritic(obs_dim, n_actions, n_critics, hidden, layer_norm).to(self.device)
        self.q_targ.load_state_dict(self.q.state_dict())
        for p in self.q_targ.parameters():
            p.requires_grad_(False)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.log_alpha = torch.full((1,), float(np.log(init_alpha)),
                                    device=self.device, requires_grad=not fixed_alpha)
        self.opt_alpha = (None if fixed_alpha
                          else torch.optim.Adam([self.log_alpha], lr=lr))
        self.target_entropy = target_entropy_ratio * float(np.log(n_actions))
        self.update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    def _t(self, x, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def update(self, batch: dict) -> dict:
        obs        = self._t(batch["obs"])
        next_obs   = self._t(batch["next_obs"])
        act        = self._t(batch["acts"], torch.long).view(-1)
        rew        = self._t(batch["rews"]).view(-1)
        done       = self._t(batch["dones"]).view(-1)
        mask       = self._t(batch["masks"], torch.bool)
        next_mask  = self._t(batch["next_masks"], torch.bool)
        gammas     = self._t(batch["gammas"]).view(-1) if "gammas" in batch \
            else torch.full_like(rew, self.gamma)
        B = obs.shape[0]
        alpha = self.alpha

        # ── critic: target = r + γ·V(s'), V from a RANDOM SUBSET min (REDQ) ──
        with torch.no_grad():
            probs_n, logp_n = self.actor.policy(next_obs, next_mask)          # (B, A)
            qt = self.q_targ(next_obs)                                        # (N, B, A)
            idx = torch.randperm(qt.shape[0], device=self.device)[:self.subset]
            qt_min = qt[idx].min(dim=0).values                               # (B, A)
            v_next = (probs_n * (qt_min - alpha * logp_n)).sum(dim=-1)        # (B,)
            y = rew + (1.0 - done) * gammas * v_next                         # (B,)

        q_all = self.q(obs)                                                  # (N, B, A)
        a_idx = act.view(1, B, 1).expand(q_all.shape[0], B, 1)
        q_taken = q_all.gather(-1, a_idx).squeeze(-1)                        # (N, B)
        loss_q = F.mse_loss(q_taken, y.unsqueeze(0).expand_as(q_taken))
        self.opt_q.zero_grad(set_to_none=True)
        loss_q.backward()
        self.opt_q.step()

        # ── actor: minimise E_a π(α·logπ − Q̄) (Q̄ = ensemble mean) ──
        probs, logp = self.actor.policy(obs, mask)                          # (B, A)
        with torch.no_grad():
            q_mean = self.q(obs).mean(dim=0)                                 # (B, A)
        loss_actor = (probs * (alpha * logp - q_mean)).sum(dim=-1).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        loss_actor.backward()
        self.opt_actor.step()

        # ── temperature: match target entropy (skipped when α is pinned) ──
        entropy = -(probs * logp).sum(dim=-1)                               # (B,)
        if not self.fixed_alpha:
            loss_alpha = (self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
            self.opt_alpha.zero_grad(set_to_none=True)
            loss_alpha.backward()
            self.opt_alpha.step()

        # ── soft target update ──
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_targ.parameters()):
                pt.mul_(1.0 - self.tau).add_(self.tau * p)
        self.update_count += 1
        return {"loss_critic": float(loss_q.item()),
                "loss_actor": float(loss_actor.item()),
                "alpha": float(alpha.item()),
                "entropy": float(entropy.mean().item())}

    def warm_start_actor(self, actor_state: dict) -> None:
        """Load a sim-trained _CategoricalActor's weights (same architecture)."""
        try:
            self.actor.load_state_dict(actor_state)
            print("[rlpd] warm-started actor from sim policy")
        except Exception as e:  # architecture mismatch → train actor from scratch
            print(f"[rlpd] actor warm-start skipped ({e})")

    def select_action(self, obs, mask, greedy: bool = True) -> int:
        with torch.no_grad():
            o = self._t(obs).unsqueeze(0)
            m = self._t(mask, torch.bool).unsqueeze(0)
            probs, _ = self.actor.policy(o, m)
            if greedy:
                return int(probs.argmax(dim=-1).item())
            return int(torch.multinomial(probs, 1).item())

    def save(self, path) -> None:
        """Export a **serve-compatible** checkpoint carrying the RLPD-trained actor.

        `serve.py` loads every non-UXP-RL checkpoint through `DSACAgent.load`,
        which requires the full DSAC key set (obs_dim, actor_target, q1/q2,
        targets, opt_*). The RLPD critic ensemble is never used at inference —
        `select_action` samples the categorical actor — so we copy the trained
        actor into a scalar-critic DSACAgent shell and save *that* as the primary
        (servable) checkpoint. The native RLPD ensemble is kept alongside as a
        `.rlpd` sidecar for provenance / resume. Without this, the faithful RLPD
        checkpoint is unservable and the eval's RLPD arm cannot run.
        """
        from .dsac import DSACAgent
        shell = DSACAgent(self.obs_dim, self.n_actions, hidden=self.hidden,
                          use_iqn=False, fixed_alpha=self.fixed_alpha,
                          device="cpu")
        shell.actor.load_state_dict(self.actor.state_dict())
        shell.actor_target.load_state_dict(self.actor.state_dict())
        with torch.no_grad():
            shell.log_alpha.fill_(float(self.log_alpha.detach().cpu()))
        shell.save(path)  # DSAC format → loadable by serve.py / DSACAgent.load
        # Provenance: the native RLPD ensemble critic (not used at inference).
        torch.save({"actor": self.actor.state_dict(), "q": self.q.state_dict(),
                    "log_alpha": self.log_alpha.detach().cpu(),
                    "obs_dim": self.obs_dim, "n_actions": self.n_actions,
                    "rlpd": True}, str(path) + ".rlpd")


def rlpd_train(*, base_policy_dir: Path, offline: ReplayBuffer,
               online: ReplayBuffer, n_updates: int,
               utd_ratio: int, batch_size: int, online_ratio: float,
               out_dir: Path,
               n_critics: int = 10, subset: int = 2,
               fixed_alpha: bool = False, init_alpha: float = 0.05,
               trace_family: str = "philly", n_jobs: int = 100,
               n_nodes: int = 1, gpus_per_node: int = 1) -> None:
    """Faithful RLPD fine-tune (see RLPDAgent): symmetric 50/50 offline+online
    batches, LayerNorm critic ensemble + random-subset target, high UTD.

    Each gradient step draws a mixed batch: online_ratio from live data, rest
    from the sim offline prior. High UTD closes the sim-to-real gap.
    """
    from .dsac import DSACAgent
    from sim.runner import run as sim_run
    from sim.loader import generate_by_family

    out_dir.mkdir(parents=True, exist_ok=True)
    obs_dim   = offline.obs_dim
    n_actions = offline.n_actions

    agent = RLPDAgent(obs_dim, n_actions, n_critics=n_critics, subset=subset,
                      fixed_alpha=fixed_alpha, init_alpha=init_alpha,
                      device="cuda" if torch.cuda.is_available() else "cpu")
    # Warm-start the actor from a sim-trained policy if one is given (the sim
    # buffer is still the offline prior via symmetric sampling; only the actor
    # weights are copied — the RLPD critic ensemble always trains from scratch).
    base_ckpt = Path(base_policy_dir) / "dsac.pt" if base_policy_dir else None
    if base_ckpt and base_ckpt.exists():
        print(f"[rlpd] warm-starting actor from {base_ckpt}")
        sd = torch.load(base_ckpt, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "actor" in sd:
            agent.warm_start_actor(sd["actor"])
    warm_start = out_dir / "dsac.pt"

    rng      = np.random.default_rng(0)
    log_path = out_dir / "rlpd_train.jsonl"

    print(f"[rlpd] {n_updates} updates × UTD={utd_ratio}  "
          f"offline={len(offline)}  online={len(online)}")

    with open(log_path, "w") as fh:
        for update in range(n_updates):
            loss_acc: dict = {}
            for _ in range(utd_ratio):
                batch  = mixed_batch(offline=offline, online=online,
                                     batch_size=batch_size,
                                     online_ratio=online_ratio, rng=rng)
                losses = agent.update(batch)
                for k, v in losses.items():
                    if k == "td_errors":
                        loss_acc["td_error_mean"] = (
                            loss_acc.get("td_error_mean", 0.0)
                            + float(np.mean(np.abs(v))) / utd_ratio
                        )
                        continue
                    loss_acc[k] = loss_acc.get(k, 0.0) + float(v) / utd_ratio

            row = {"update": update, "online_size": len(online),
                   "offline_size": len(offline), **loss_acc}
            fh.write(json.dumps(row) + "\n")

            if (update + 1) % 50 == 0:
                print(f"  update {update+1:4d}/{n_updates}  "
                      f"loss_critic={loss_acc.get('loss_critic', 0):.4f}  "
                      f"loss_actor={loss_acc.get('loss_actor', 0):.4f}  "
                      f"alpha={loss_acc.get('alpha', 0):.4f}  "
                      f"H={loss_acc.get('entropy', 0):.3f}")

    agent.save(warm_start)

    # Quick eval: 3 greedy episodes vs score baseline
    print("\n[rlpd] quick eval (3 seeds, greedy) ...")
    total_gpus = n_nodes * gpus_per_node
    dsac_jcts  = []
    score_jcts = []
    for ep_seed in [42, 43, 44]:
        env = KubefluxSchedEnv(
            lambda _s=ep_seed, _tg=total_gpus: [
                j for j in generate_by_family(trace_family, n_jobs=n_jobs, seed=_s)
                if j.gpu_count <= _tg
            ],
            n_nodes=n_nodes, gpus_per_node=gpus_per_node,
            max_steps=n_jobs * 200, reward_mode="jct_aligned",
        )
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            mask = env.action_mask()
            act  = agent.select_action(obs, mask, greedy=True)
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
        env.close()
        dsac_jcts.append(info.get("avg_jct", float("nan")))

        jobs = [j for j in generate_by_family(trace_family, n_jobs=n_jobs, seed=ep_seed)
                if j.gpu_count <= total_gpus]
        m, _ = sim_run(jobs, n_nodes=n_nodes, gpus_per_node=gpus_per_node,
                        scheduler_name="score")
        score_jcts.append(m.summary()["jct_mean"])

    dsac_mean  = float(np.nanmean(dsac_jcts))
    score_mean = float(np.mean(score_jcts))
    pct        = (score_mean - dsac_mean) / score_mean * 100
    print(f"  DSAC  mean JCT : {dsac_mean/3600:.3f}h")
    print(f"  Score mean JCT : {score_mean/3600:.3f}h")
    print(f"  Δ              : {pct:+.1f}%  "
          f"({'DSAC wins' if pct > 0 else 'score wins'})")
    print(f"[rlpd] policy → {warm_start}")
    print(f"[rlpd] log    → {log_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-policy", default=None,
                   help="dir with dsac.pt checkpoint to warm-start from (optional)")
    p.add_argument("--offline-steps", type=int, default=50_000)
    p.add_argument("--online-log", nargs="*", default=[],
                   help="shadow-mode JSONL transition logs")
    p.add_argument("--online-trace", nargs="*", default=[],
                   help="normalized live trace JSON from scripts/collect-live-trace.py; replayed as score demonstrations")
    p.add_argument("--n-updates", type=int, default=200)
    p.add_argument("--utd-ratio", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--online-ratio", type=float, default=0.5)
    p.add_argument("--n-critics", type=int, default=10,
                   help="RLPD LayerNorm critic ensemble size (Ball et al. 2023)")
    p.add_argument("--subset", type=int, default=2,
                   help="random critic subset min-reduced for the target (REDQ)")
    p.add_argument("--fixed-alpha", action="store_true",
                   help="pin the entropy temperature α (masked discrete SAC rails "
                        "auto-α to ∞ since target ratio·log(A) exceeds the max "
                        "entropy of the ~2-3 legal actions)")
    p.add_argument("--init-alpha", type=float, default=0.05,
                   help="α value when --fixed-alpha (else initial α)")
    p.add_argument("--trace-family", default="philly")
    p.add_argument("--n-jobs", type=int, default=300)
    # Cluster shape — must match the live deployment.
    # Current: 1 host × 1 GPU → obs_dim=160, n_actions=17.
    # When second GPU is online: change both to 2 and retrain from scratch.
    p.add_argument("--n-nodes", type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument("--out-dir",
                   default=f"runs/m11_rlpd_{time.strftime('%Y%m%d-%H%M%S')}")
    args = p.parse_args(argv)

    base = Path(args.base_policy) if args.base_policy else Path(args.out_dir)

    print(f"[rlpd] collecting offline buffer ({args.offline_steps} steps)...")
    offline = collect_sim_rollouts(
        n_transitions=args.offline_steps,
        trace_family=args.trace_family,
        n_jobs=args.n_jobs,
        n_nodes=args.n_nodes,
        gpus_per_node=args.gpus_per_node,
    )
    print(f"[rlpd] offline buffer size = {len(offline)}")

    online_capacity = max(10_000, args.offline_steps)
    online = load_live_shadow_log(
        args.online_log,
        obs_dim=offline.obs.shape[1],
        n_actions=offline.masks.shape[1],
        capacity=online_capacity,
    )
    if args.online_trace:
        trace_online = load_live_trace_rollouts(
            args.online_trace,
            obs_dim=offline.obs.shape[1],
            n_actions=offline.masks.shape[1],
            capacity=online_capacity,
            n_nodes=args.n_nodes,
            gpus_per_node=args.gpus_per_node,
            n_jobs=args.n_jobs,
        )
        for i in range(len(trace_online)):
            if len(online) >= online.capacity:
                break
            online.add(Transition(
                obs=trace_online.obs[i],
                act=int(trace_online.acts[i]),
                rew=float(trace_online.rews[i]),
                next_obs=trace_online.next_obs[i],
                done=bool(trace_online.dones[i]),
                mask=trace_online.masks[i],
                next_mask=trace_online.next_masks[i],
            ), gamma=float(trace_online.gammas[i]))
    print(f"[rlpd] online buffer size = {len(online)} "
          f"(0 = cold start, 100% offline)")

    rlpd_train(
        base_policy_dir=base,
        offline=offline,
        online=online,
        n_updates=args.n_updates,
        utd_ratio=args.utd_ratio,
        batch_size=args.batch_size,
        online_ratio=args.online_ratio if len(online) else 0.0,
        out_dir=Path(args.out_dir),
        n_critics=args.n_critics, subset=args.subset,
        fixed_alpha=args.fixed_alpha, init_alpha=args.init_alpha,
        trace_family=args.trace_family,
        n_jobs=args.n_jobs,
        n_nodes=args.n_nodes,
        gpus_per_node=args.gpus_per_node,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
