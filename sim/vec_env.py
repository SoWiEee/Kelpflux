"""Vectorized scheduling environments for higher rollout throughput.

The simulator is a pure-Python discrete-event loop (~10 steps/s per env), which
is the compute wall for multi-seed / multi-arm studies (one σ block ≈ 4.6 h).
Throughput comes from running many envs in parallel:

- :class:`SyncVectorSchedEnv`  — ``N`` envs stepped in-process. Deterministic
  reference; batches obs/mask for the agent forward, but env steps stay
  sequential (no wall-clock win — used as the correctness oracle and for tiny N).
- :class:`AsyncVectorSchedEnv` — ``N`` envs in ``N`` worker processes. Real
  multi-core wall-clock speedup; identical interface and (per-seed) identical
  transitions to the sync version.

Both **auto-reset** a sub-env when its episode ends: the returned ``obs``/``mask``
for that slot are the *first* observation of the new episode, while the terminal
observation/mask are preserved under ``info["final_obs"]`` / ``info["final_mask"]``
(gymnasium's autoreset convention). The training collector must read those when
``done`` so the replay buffer stores the true terminal ``next_obs``.

The env factory is a picklable :class:`FamilyJobFactory` (not a closure) so the
spec can cross the process boundary to Async workers.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from sim.gym_env import MPS_PER_GPU, TOP_K, KubefluxSchedEnv
from sim.loader import Job, generate_by_family


class FamilyJobFactory:
    """Picklable no-arg job generator: pick a family, draw a seed, generate, filter.

    Each call advances an internal RNG, so successive ``reset()``s see fresh
    traces. RNG state is preserved across pickling, so an Async worker keeps
    producing the same stream it would in-process.
    """

    def __init__(self, families: Sequence[str], n_jobs: int, total_gpus: int, seed: int):
        self.families = tuple(families)
        self.n_jobs = int(n_jobs)
        self.total_gpus = int(total_gpus)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)

    def __call__(self) -> list[Job]:
        fam = self.families[int(self._rng.integers(0, len(self.families)))]
        jobs = generate_by_family(fam, n_jobs=self.n_jobs, seed=int(self._rng.integers(0, 2**31 - 1)))
        return [j for j in jobs if j.gpu_count <= self.total_gpus]

    # Preserve the RNG stream across the pickle to the worker process.
    def __getstate__(self) -> dict[str, Any]:
        return {
            "families": self.families,
            "n_jobs": self.n_jobs,
            "total_gpus": self.total_gpus,
            "seed": self.seed,
            "rng_state": self._rng.bit_generator.state,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["families"], state["n_jobs"], state["total_gpus"], state["seed"])
        self._rng.bit_generator.state = state["rng_state"]


@dataclass(frozen=True)
class EnvSpec:
    """Picklable description of one sub-env. ``build(index)`` makes a seeded env."""

    families: tuple[str, ...]
    n_jobs: int
    total_gpus: int
    n_nodes: int = 1
    gpus_per_node: int = 1
    mps_per_gpu: int = MPS_PER_GPU
    top_k: int = TOP_K
    max_steps: int = 10_000
    reward_mode: str = "jct_aligned"
    reward_scale: float = 20_000.0
    potential_shaping: bool = True
    runtime_sigma: float = 0.0
    interference: float = 0.0
    colocation_actions: bool = False
    base_seed: int = 0

    def build(self, index: int) -> KubefluxSchedEnv:
        # Distinct stream per env slot so parallel rollouts decorrelate.
        factory = FamilyJobFactory(
            self.families, self.n_jobs, self.total_gpus,
            seed=self.base_seed + 7919 * (index + 1),
        )
        return KubefluxSchedEnv(
            factory,
            n_nodes=self.n_nodes,
            gpus_per_node=self.gpus_per_node,
            mps_per_gpu=self.mps_per_gpu,
            top_k=self.top_k,
            max_steps=self.max_steps,
            reward_mode=self.reward_mode,
            reward_scale=self.reward_scale,
            potential_shaping=self.potential_shaping,
            runtime_sigma=self.runtime_sigma,
            interference=self.interference,
            colocation_actions=self.colocation_actions,
        )


# ── Shared single-env step/reset with autoreset (used by both backends) ─────

def _reset_one(env: KubefluxSchedEnv, seed: int | None) -> tuple[np.ndarray, np.ndarray]:
    obs, _ = env.reset(seed=seed)
    return obs, env.action_mask()


def _step_one(env: KubefluxSchedEnv, action: int) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
    obs, rew, term, trunc, info = env.step(int(action))
    done = bool(term or trunc)
    if done:
        info = dict(info)
        info["final_obs"] = obs
        info["final_mask"] = env.action_mask()
        obs, _ = env.reset()
    return obs, env.action_mask(), float(rew), done, info


class SyncVectorSchedEnv:
    """``N`` envs stepped in-process. Deterministic correctness oracle."""

    def __init__(self, spec: EnvSpec, num_envs: int):
        self.spec = spec
        self.num_envs = int(num_envs)
        self.envs = [spec.build(i) for i in range(self.num_envs)]
        self._rng = np.random.default_rng(spec.base_seed + 991)

    def score_actions(self) -> list[int]:
        """Per-env score-heuristic warmup action (mirrors single-env warmup)."""
        return [env.score_warmup_action(env.action_mask(), self._rng) for env in self.envs]

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        obs, masks = [], []
        for i, env in enumerate(self.envs):
            o, m = _reset_one(env, None if seed is None else seed + i)
            obs.append(o)
            masks.append(m)
        return np.stack(obs), np.stack(masks)

    def step(self, actions: Sequence[int]):
        obs, masks, rews, dones, infos = [], [], [], [], []
        for env, a in zip(self.envs, actions):
            o, m, r, d, info = _step_one(env, a)
            obs.append(o)
            masks.append(m)
            rews.append(r)
            dones.append(d)
            infos.append(info)
        return (
            np.stack(obs),
            np.stack(masks),
            np.asarray(rews, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()


# ── Async (multiprocess) backend ───────────────────────────────────────────

def _worker(remote, parent_remote, spec: EnvSpec, index: int) -> None:
    parent_remote.close()
    env = spec.build(index)
    wrng = np.random.default_rng(spec.base_seed + 991 + index)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send(_reset_one(env, data))
            elif cmd == "step":
                remote.send(_step_one(env, data))
            elif cmd == "score_action":
                remote.send(env.score_warmup_action(env.action_mask(), wrng))
            elif cmd == "close":
                break
            else:  # pragma: no cover - defensive
                raise RuntimeError(f"unknown command {cmd!r}")
    except (KeyboardInterrupt, EOFError):  # pragma: no cover
        pass
    finally:
        env.close()
        remote.close()


class AsyncVectorSchedEnv:
    """``N`` envs in ``N`` worker processes. Same interface as the sync version."""

    def __init__(self, spec: EnvSpec, num_envs: int, *, start_method: str | None = None):
        self.spec = spec
        self.num_envs = int(num_envs)
        ctx = mp.get_context(start_method or "spawn")
        self._closed = False
        self.parents, self.procs = [], []
        for i in range(self.num_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, parent, spec, i), daemon=True)
            proc.start()
            child.close()
            self.parents.append(parent)
            self.procs.append(proc)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        for i, parent in enumerate(self.parents):
            parent.send(("reset", None if seed is None else seed + i))
        results = [parent.recv() for parent in self.parents]
        obs = np.stack([o for o, _ in results])
        masks = np.stack([m for _, m in results])
        return obs, masks

    def score_actions(self) -> list[int]:
        """Per-env score-heuristic warmup action, computed inside each worker."""
        for parent in self.parents:
            parent.send(("score_action", None))
        return [int(parent.recv()) for parent in self.parents]

    def step(self, actions: Sequence[int]):
        for parent, a in zip(self.parents, actions):
            parent.send(("step", int(a)))
        results = [parent.recv() for parent in self.parents]
        obs = np.stack([r[0] for r in results])
        masks = np.stack([r[1] for r in results])
        rews = np.asarray([r[2] for r in results], dtype=np.float32)
        dones = np.asarray([r[3] for r in results], dtype=bool)
        infos = [r[4] for r in results]
        return obs, masks, rews, dones, infos

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for parent in self.parents:
            try:
                parent.send(("close", None))
            except (BrokenPipeError, OSError):  # pragma: no cover
                pass
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():  # pragma: no cover
                proc.terminate()

    def __del__(self):  # pragma: no cover - best-effort cleanup
        self.close()


def make_vector_env(spec: EnvSpec, num_envs: int, *, asynchronous: bool = True):
    """Build a vector env: Async (multiprocess) by default, Sync for num_envs==1."""
    if num_envs <= 1 or not asynchronous:
        return SyncVectorSchedEnv(spec, max(1, num_envs))
    return AsyncVectorSchedEnv(spec, num_envs)
