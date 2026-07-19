"""UXP-RL: User-eXperience-and-Performance-balanced RL scheduler.

Faithful discrete-action reproduction of the DQN-based scheduler of
Lin, Ling, Lai & Sudyana, "Reinforcement Learning for AI as a Service:
CPU-GPU Task Scheduling for Preprocessing, Training, and Inference Tasks,"
IEEE Transactions on Network and Service Management, vol. 22, no. 4, Aug 2025.

The method is a **value-based DQN** (paper Algorithm 1) with ε-greedy
exploration, a *hard*-updated target network, and a user-experience-balanced
reward that up-weights inference tasks (c2 > c1) and penalises inference
turnaround beyond the Δ^I latency target. Hyperparameters follow Table VI:

    γ = 0.9,  ε: 0.9 → 0.1 (decay ρ = 0.95),  lr = 5e-4,
    replay N = 500,  minibatch B = 32,  target-reset / ε-decay every I = 4000.

The reward function itself lives in ``sim/gym_env`` under ``reward_mode="uxprl"``
(it needs the env's per-task turnaround and job type); this module reproduces the
DQN core, the ε-greedy masked policy, the hard target update, and Table VI.

Substrate adaptation (documented, honest): the paper runs a two-agent GPU/CPU
split (Algorithm 1, line 5) because it decides CPU-vs-GPU placement. Kelpflux's
simulator only schedules GPU placement — there is no CPU-offload action — so the
CPU agent is degenerate and UXP-RL reduces to its single (GPU) agent form on this
substrate. The value-based algorithm is reproduced exactly; only the redundant
second agent is dropped.

The public surface (``select_action``, ``action_values``, ``save``/``load``,
``obs_dim``/``n_actions``) matches ``DSACAgent`` so the agent is a drop-in for
``serve.py`` (/act, /decide) and the eval harness. ``use_iqn`` is exposed as
``False`` so IQN-specific serve branches treat it as a scalar-value model.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from services.rl_scheduler.dsac import _build_mlp

# Paper Table VI — faithful defaults.
UXPRL_GAMMA = 0.9
UXPRL_LR = 5e-4
UXPRL_EPS_START = 0.9
UXPRL_EPS_MIN = 0.1
UXPRL_EPS_DECAY = 0.95
UXPRL_REPLAY_N = 500
UXPRL_BATCH = 32
UXPRL_UPDATE_FREQ = 4000   # I: target reset + ε decay interval (steps)


class _QNet(nn.Module):
    """Scalar action-value network Q(s,·) over a shared MLP trunk."""

    def __init__(self, obs_dim: int, n_actions: int,
                 hidden: Sequence[int] = (256, 256), layer_norm: bool = True) -> None:
        super().__init__()
        self.net = _build_mlp(obs_dim, hidden, n_actions, layer_norm)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class _UniformReplay:
    """Fixed-capacity uniform replay (paper N = 500). Stores masked transitions."""

    def __init__(self, capacity: int) -> None:
        self.buf: deque = deque(maxlen=capacity)

    def add(self, obs, act, rew, next_obs, done, next_mask) -> None:
        self.buf.append((obs, act, rew, next_obs, done, next_mask))

    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, len(self.buf), size=batch_size)
        items = [self.buf[i] for i in idx]
        obs, act, rew, next_obs, done, next_mask = zip(*items)
        return {
            "obs": np.asarray(obs, dtype=np.float32),
            "acts": np.asarray(act, dtype=np.int64),
            "rews": np.asarray(rew, dtype=np.float32),
            "next_obs": np.asarray(next_obs, dtype=np.float32),
            "dones": np.asarray(done, dtype=np.float32),
            "next_masks": np.asarray(next_mask, dtype=bool),
        }


class UXPRLAgent:
    """Faithful UXP-RL DQN (Lin et al. 2025) for masked discrete scheduling."""

    # Marker so serve.py / eval treat this as a scalar-value (non-IQN) model.
    use_iqn = False
    risk_mode = None

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: Sequence[int] = (256, 256),
        lr: float = UXPRL_LR,
        gamma: float = UXPRL_GAMMA,
        eps_start: float = UXPRL_EPS_START,
        eps_min: float = UXPRL_EPS_MIN,
        eps_decay: float = UXPRL_EPS_DECAY,
        target_sync: int = UXPRL_UPDATE_FREQ,
        replay_size: int = UXPRL_REPLAY_N,
        batch_size: int = UXPRL_BATCH,
        layer_norm: bool = True,
        seed: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = tuple(hidden)
        self.gamma = float(gamma)
        self.eps_start = float(eps_start)
        self.eps_min = float(eps_min)
        self.eps_decay = float(eps_decay)
        self.epsilon = float(eps_start)
        self.target_sync = int(target_sync)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)

        self.q = _QNet(obs_dim, n_actions, self.hidden, layer_norm).to(self.device)
        self.q_target = _QNet(obs_dim, n_actions, self.hidden, layer_norm).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.replay = _UniformReplay(replay_size)
        self._rng = np.random.default_rng(seed)
        self._update_count = 0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def action_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Per-action Q(s,·). (B, obs) → (B, A). Matches DSACAgent.action_values."""
        return self.q(obs)

    def select_action(self, obs: np.ndarray, mask: np.ndarray,
                      greedy: bool = False) -> int:
        """Masked ε-greedy. greedy=True (serve/eval) → pure argmax over legal Q."""
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return 0
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.choice(legal))
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            q = self.q(obs_t).squeeze(0).cpu().numpy()
        q_masked = np.where(mask.astype(bool), q, -np.inf)
        return int(np.argmax(q_masked))

    def add(self, obs, act, rew, next_obs, done, next_mask) -> None:
        self.replay.add(obs, int(act), float(rew), next_obs, bool(done), next_mask)

    # ------------------------------------------------------------------
    def update(self, batch: Optional[dict] = None) -> dict:
        """One DQN gradient step (paper Algorithm 1, lines 13-22).

        Samples uniformly from replay when ``batch`` is None. Bellman target
        ``y = r + γ·(1-done)·max_{a' legal} Q_target(s', a')`` (vanilla DQN, not
        DDQN — the paper prefers DQN's simplicity). MSE loss. Target net is *hard*-
        copied and ε decayed every ``target_sync`` steps.
        """
        if batch is None:
            if len(self.replay) < self.batch_size:
                return {}
            batch = self.replay.sample(self.batch_size, self._rng)

        def _t(k, dtype=torch.float32):
            return torch.as_tensor(batch[k], dtype=dtype, device=self.device)

        obs = _t("obs")
        acts = _t("acts", torch.long)
        rews = _t("rews")
        next_obs = _t("next_obs")
        dones = _t("dones")
        next_masks = _t("next_masks", torch.bool)

        q_sa = self.q(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.q_target(next_obs)
            next_q = next_q.masked_fill(~next_masks, -float("inf"))
            next_max = next_q.max(dim=1).values
            # A next-state with no legal action contributes 0 (fully masked row).
            next_max = torch.nan_to_num(next_max, neginf=0.0)
            target = rews + self.gamma * (1.0 - dones) * next_max

        loss = F.mse_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self._update_count += 1
        if self._update_count % self.target_sync == 0:
            self.q_target.load_state_dict(self.q.state_dict())
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

        return {"loss": float(loss.item()), "epsilon": self.epsilon}

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        torch.save({
            "algo": "uxprl",
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "opt": self.opt.state_dict(),
            "hidden": list(self.hidden),
            "obs_dim": self.obs_dim, "n_actions": self.n_actions,
            "gamma": self.gamma,
            "eps_start": self.eps_start, "eps_min": self.eps_min,
            "eps_decay": self.eps_decay, "epsilon": self.epsilon,
            "target_sync": self.target_sync, "batch_size": self.batch_size,
            "update_count": self._update_count,
        }, str(path))

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "UXPRLAgent":
        data = torch.load(str(path), map_location="cpu", weights_only=True)
        if data.get("algo") != "uxprl":
            raise ValueError(f"{path} is not a UXP-RL checkpoint (algo={data.get('algo')!r})")
        agent = cls(
            obs_dim=data["obs_dim"], n_actions=data["n_actions"],
            hidden=tuple(data.get("hidden", (256, 256))),
            gamma=data.get("gamma", UXPRL_GAMMA),
            eps_start=data.get("eps_start", UXPRL_EPS_START),
            eps_min=data.get("eps_min", UXPRL_EPS_MIN),
            eps_decay=data.get("eps_decay", UXPRL_EPS_DECAY),
            target_sync=data.get("target_sync", UXPRL_UPDATE_FREQ),
            batch_size=data.get("batch_size", UXPRL_BATCH),
            **kwargs)
        agent.q.load_state_dict(data["q"])
        agent.q_target.load_state_dict(data["q_target"])
        agent.opt.load_state_dict(data["opt"])
        agent.epsilon = float(data.get("epsilon", agent.eps_min))
        agent._update_count = data.get("update_count", 0)
        return agent

    @staticmethod
    def is_uxprl_checkpoint(path: str | Path) -> bool:
        try:
            data = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception:
            return False
        return data.get("algo") == "uxprl"


# ---------------------------------------------------------------------------
# Lean DQN training loop (Algorithm 1). Deliberately does NOT use score-warmup,
# n-step returns, or prioritized replay — none are part of UXP-RL. Faithfulness
# over shared machinery.
# ---------------------------------------------------------------------------
def train_uxprl(
    *,
    n_nodes: int = 1,
    gpus_per_node: int = 1,
    trace_family: str | list = "philly",
    n_jobs: int = 50,
    total_steps: int = 500_000,
    warmup_steps: int = 2_000,
    seed: int = 42,
    out_dir: Optional[Path] = None,
    reward_scale: float = 1.0,
    uxprl_c1: float = 1.0,
    uxprl_c2: float = 2.0,
    node_speeds: Optional[list] = None,
    curriculum: bool = False,
    curriculum_stages: Optional[list] = None,
    max_steps_mult: int = 200,
    log_every: int = 5_000,
    device: str = "cpu",
) -> UXPRLAgent:
    """Run online UXP-RL (DQN) training in sim. Returns the trained agent."""
    from sim.gym_env import KubefluxSchedEnv, env_dims
    from sim.loader import generate_by_family

    obs_dim, n_actions = env_dims(n_nodes, gpus_per_node)
    rng = np.random.default_rng(seed)
    total_gpus = n_nodes * gpus_per_node
    families = [trace_family] if isinstance(trace_family, str) else list(trace_family)

    if curriculum and curriculum_stages is None:
        curriculum_stages = [(10, 0.2), (30, 0.3), (50, 0.5)]
    active_n_jobs = n_jobs

    def _make_factory(nj: int):
        def _factory():
            family = families[int(rng.integers(0, len(families)))]
            jobs = generate_by_family(family, n_jobs=nj,
                                      seed=int(rng.integers(0, 2**31 - 1)))
            return [j for j in jobs if j.gpu_count <= total_gpus]
        return _factory

    env = KubefluxSchedEnv(
        _make_factory(active_n_jobs),
        n_nodes=n_nodes, gpus_per_node=gpus_per_node,
        max_steps=active_n_jobs * max_steps_mult,
        reward_mode="uxprl",
        reward_scale=reward_scale,
        node_speeds=node_speeds,
        uxprl_c1=uxprl_c1, uxprl_c2=uxprl_c2,
    )
    agent = UXPRLAgent(obs_dim=obs_dim, n_actions=n_actions,
                       seed=seed, device=device)

    log_fh = None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_fh = open(out_dir / "sim_train.jsonl", "w")

    obs, _ = env.reset(seed=seed)
    mask = env.action_mask()
    ep_steps = ep_reward = 0.0
    ep_count = 0
    last: dict = {}
    t0 = time.time()

    for step in range(total_steps):
        if curriculum_stages is not None:
            progress = step / total_steps
            cum = 0.0
            stage_n_jobs = curriculum_stages[0][0]
            for nj, frac in curriculum_stages:
                cum += frac
                if progress < cum:
                    stage_n_jobs = nj
                    break
            if stage_n_jobs != active_n_jobs:
                active_n_jobs = stage_n_jobs
                env.jobs_factory = _make_factory(active_n_jobs)
                env.max_steps = active_n_jobs * max_steps_mult
                print(f"  [curriculum] step={step}: n_jobs → {active_n_jobs}")

        # ε-greedy rollout (random-legal during warmup so replay fills).
        if step < warmup_steps:
            legal = np.flatnonzero(mask)
            act = int(rng.choice(legal)) if legal.size else 0
        else:
            act = agent.select_action(obs, mask)

        next_obs, rew, term, trunc, info = env.step(act)
        next_mask = env.action_mask()
        done = bool(term or trunc)
        agent.add(obs, act, rew, next_obs, done, next_mask)

        obs, mask = next_obs, next_mask
        ep_steps += 1
        ep_reward += float(rew)

        if done:
            ep_count += 1
            if log_fh:
                log_fh.write(json.dumps({
                    "step": step, "episode": ep_count,
                    "ep_steps": int(ep_steps), "ep_reward": ep_reward,
                    "avg_jct": info.get("avg_jct", float("nan")),
                    "completed": info.get("completed", 0),
                    "n_jobs": active_n_jobs,
                    "epsilon": agent.epsilon, "loss": last.get("loss"),
                }) + "\n")
            ep_steps = ep_reward = 0.0
            obs, _ = env.reset()
            mask = env.action_mask()

        if step >= warmup_steps:
            last = agent.update() or last

        if (step + 1) % log_every == 0:
            print(f"  step {step+1:6d}/{total_steps}  replay={len(agent.replay):5d}  "
                  f"eps={ep_count}  ε={agent.epsilon:.3f}  n_jobs={active_n_jobs}  "
                  f"elapsed={time.time() - t0:.0f}s")

    env.close()
    if log_fh:
        log_fh.close()
    if out_dir:
        ckpt = out_dir / "dsac.pt"   # same filename → serve.py auto-detects algo
        agent.save(ckpt)
        print(f"[uxprl] saved → {ckpt}")
    return agent
