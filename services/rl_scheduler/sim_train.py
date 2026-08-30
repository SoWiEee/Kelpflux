"""Step 4: Online DSAC training loop inside KubefluxSchedEnv.

Runs the DSAC policy online in sim: agent collects its own transitions,
updates at UTD=4 after every env step (once the warmup buffer is full).
Saves checkpoint + JSONL episode log.

Improvements:
  n-step returns      : pre-compute discounted return over n steps (n=10 default)
  score warmup        : use score scheduler during warmup for high-quality seeds
  short episodes      : default n_jobs=50 → ~3× more distinct episodes
  potential shaping   : per-step reward φ(s) = −Σwait/scale, Ng et al. 1999
  PER                 : prioritized replay, sample by TD-error magnitude
  IQN                 : Implicit Quantile Network critic (opt-in)
  vectorized rollout  : N parallel envs (--num-envs) for multi-core throughput
  Curriculum          : n_jobs ramps from easy to hard over training

Usage::
    .venv-m11/bin/python -m services.rl_scheduler.sim_train \\
        --n-nodes 1 --gpus-per-node 1 --total-steps 500000 \\
        --trace philly --n-jobs 50 --out-dir runs/dsac_sim
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from sim.gym_env import KubefluxSchedEnv, env_dims
from sim.loader import generate_by_family
from services.rl_scheduler.dsac import DSACAgent
from services.rl_scheduler.distortion import RISK_MODES
from services.rl_scheduler.rlpd_finetune import (
    PrioritizedReplayBuffer, ReplayBuffer, Transition,
)


def _flush_nstep(nstep_buf: deque, buf, gamma: float) -> None:
    """Commit the oldest transition in nstep_buf with its n-step return."""
    if not nstep_buf:
        return
    horizon = len(nstep_buf)
    for h, t in enumerate(nstep_buf):
        if t.done:
            horizon = h + 1
            break

    t0 = nstep_buf[0]
    nstep_rew = 0.0
    g = 1.0
    for h in range(horizon):
        nstep_rew += g * nstep_buf[h].rew
        g *= gamma
    t_last = nstep_buf[horizon - 1]
    buf.add(
        Transition(
            obs=t0.obs, act=t0.act, rew=nstep_rew,
            next_obs=t_last.next_obs, done=t_last.done,
            mask=t0.mask, next_mask=t_last.next_mask,
        ),
        gamma=g,
    )


def _collect_and_train_vec(
    *, agent, buf, rng, log_fh, num_envs, async_envs,
    families, active_n_jobs, total_gpus, n_nodes, gpus_per_node,
    reward_mode, reward_scale, potential_shaping, runtime_sigma,
    interference, colocation, nstep_n, total_steps, warmup_steps,
    update_every, utd_ratio, batch_size, use_per, seed, log_every,
    curriculum_stages, score_warmup, slo_penalty=0.0,
) -> None:
    """Vectorized rollout: N parallel envs feed the shared agent/buffer.

    Throughput path for multi-seed/arm studies — the pure-Python sim is the wall.
    Async (multiprocess) envs step in parallel across cores; the learner stays in
    the main process. Score-warmup is honored via ``vec.score_actions()`` (each
    env computes the score-heuristic action in-process / inside its worker), so
    the vec path reproduces the single-env warmup instead of random-legal.
    """
    from sim.vec_env import EnvSpec, make_vector_env

    def _build_vec(nj: int):
        spec = EnvSpec(
            families=tuple(families), n_jobs=nj, total_gpus=total_gpus,
            n_nodes=n_nodes, gpus_per_node=gpus_per_node, max_steps=nj * 200,
            reward_mode=reward_mode, reward_scale=reward_scale,
            potential_shaping=potential_shaping, runtime_sigma=runtime_sigma,
            interference=interference, colocation_actions=colocation,
            slo_penalty=slo_penalty, base_seed=seed,
        )
        return make_vector_env(spec, num_envs, asynchronous=async_envs)

    if curriculum_stages is not None:
        active_n_jobs = curriculum_stages[0][0]
    print(f"  [sim_train] vectorized rollout: num_envs={num_envs} "
          f"async={async_envs and num_envs > 1} score_warmup={score_warmup}")

    vec = _build_vec(active_n_jobs)
    obs, masks = vec.reset(seed=seed)
    nstep_bufs = [deque(maxlen=nstep_n) for _ in range(num_envs)]
    ep_steps = np.zeros(num_envs)
    ep_reward = np.zeros(num_envs)
    ep_count = 0
    last_losses: dict = {}
    t0 = time.time()
    step = 0

    while step < total_steps:
        if curriculum_stages is not None:
            progress = step / total_steps
            cum = 0.0
            stage_n = curriculum_stages[0][0]
            for nj, frac in curriculum_stages:
                cum += frac
                if progress < cum:
                    stage_n = nj
                    break
            if stage_n != active_n_jobs:
                active_n_jobs = stage_n
                vec.close()
                vec = _build_vec(active_n_jobs)
                obs, masks = vec.reset()
                nstep_bufs = [deque(maxlen=nstep_n) for _ in range(num_envs)]
                ep_steps[:] = 0
                ep_reward[:] = 0
                print(f"  [curriculum] step={step}: n_jobs → {active_n_jobs}")

        if len(buf) < warmup_steps:
            if score_warmup:
                acts = vec.score_actions()
            else:
                acts = [int(rng.choice(np.flatnonzero(m))) for m in masks]
        else:
            acts = [agent.select_action(obs[i], masks[i]) for i in range(num_envs)]

        next_obs, next_masks, rews, dones, infos = vec.step(acts)
        for i in range(num_envs):
            nxt_o = infos[i]["final_obs"] if dones[i] else next_obs[i]
            nxt_m = infos[i]["final_mask"] if dones[i] else next_masks[i]
            nstep_bufs[i].append(Transition(
                obs=obs[i], act=acts[i], rew=float(rews[i]),
                next_obs=nxt_o, done=bool(dones[i]),
                mask=masks[i], next_mask=nxt_m,
            ))
            if len(nstep_bufs[i]) == nstep_n or dones[i]:
                _flush_nstep(nstep_bufs[i], buf, gamma=agent.gamma)
                if dones[i]:
                    nstep_bufs[i].clear()
            ep_steps[i] += 1
            ep_reward[i] += float(rews[i])
            if dones[i]:
                ep_count += 1
                if log_fh:
                    log_fh.write(json.dumps({
                        "step": step, "episode": ep_count, "env": i,
                        "ep_steps": int(ep_steps[i]), "ep_reward": float(ep_reward[i]),
                        "avg_jct": infos[i].get("avg_jct", float("nan")),
                        "completed": infos[i].get("completed", 0),
                        "n_jobs": active_n_jobs,
                        "alpha": last_losses.get("alpha"),
                        "entropy": last_losses.get("entropy"),
                        "loss_critic": last_losses.get("loss_critic"),
                        "loss_actor": last_losses.get("loss_actor"),
                    }) + "\n")
                ep_steps[i] = 0
                ep_reward[i] = 0

        obs, masks = next_obs, next_masks
        step += num_envs

        if len(buf) >= warmup_steps and (step // num_envs) % update_every == 0:
            for _ in range(utd_ratio):
                batch = buf.sample(min(batch_size, len(buf)), rng)
                losses = agent.update(batch)
                last_losses = losses
                if use_per and "indices" in batch and "td_errors" in losses:
                    buf.update_priorities(batch["indices"], losses["td_errors"])

        if step % log_every < num_envs:
            elapsed = time.time() - t0
            print(f"  step {step:6d}/{total_steps}  buf={len(buf):6d}  "
                  f"eps={ep_count}  n_jobs={active_n_jobs}  "
                  f"alpha={agent.alpha.item():.3f}  elapsed={elapsed:.0f}s")

    vec.close()


def sim_train(
    *,
    n_nodes: int = 1,
    gpus_per_node: int = 1,
    trace_family: str | list = "philly",
    n_jobs: int = 50,
    nstep_n: int = 10,
    total_steps: int = 500_000,
    warmup_steps: int = 2_000,
    score_warmup: bool = True,
    update_every: int = 1,
    utd_ratio: int = 4,
    batch_size: int = 256,
    buf_capacity: int = 100_000,
    seed: int = 42,
    out_dir: Optional[Path] = None,
    reward_mode: str = "jct_aligned",
    reward_scale: float = 20_000.0,   # base-policy training scale (§5.1). Differs from
    #   gym_env's env default (1000, used by RLPD/live to match the online-log −JCT/1000).
    mo_w_jct: float = 1.0,
    mo_w_util: float = 0.05,
    device: str = "cpu",
    log_every: int = 5_000,
    use_iqn: bool = True,
    # Temperature (entropy) controls
    fixed_alpha: bool = False,
    init_alpha: float = 0.1,
    target_entropy_ratio: float = 0.1,
    # New improvements
    potential_shaping: bool = True,
    balance_coef: float = 0.0,        # P1: potential-based node-balance shaping
    fairness_coef: float = 0.0,       # convex per-job JCT penalty (anti-starvation, obj-changing)
    node_speeds: Optional[list] = None,  # item-1: per-node relative speed (heterogeneity)
    node_gpu_types: Optional[list] = None,  # per-node card id → SPEED_MATRIX + obs one-hot
    node_ram_gb: Optional[list] = None,     # per-node usable host RAM (GB) → OOM gate
    normalize_reward: bool = False,   # P2: running-std reward normalization (PopArt-lite)
    use_per: bool = True,
    risk_mode: str = "mean",
    risk_beta: float = 0.25,
    value_clip: float = 0.0,          # Duan et al. 2021 target return-clip (0 = off)
    curriculum: bool = False,
    curriculum_stages: Optional[list] = None,
    # Stochastic execution (opt-in; gives Z_R spread for the distributional critic)
    runtime_sigma: float = 0.0,
    interference: float = 0.0,
    # Stochastic host-RAM OOM (opt-in; oom_penalty_s > 0 turns it on). Models
    # node-2's drain risk as a penalised tail event a risk-aware policy can avoid.
    oom_sigma: float = 0.0,
    oom_penalty_s: float = 0.0,
    ram_overcommit: float = 1.0,
    # Co-location action mode (B; opt-in, doubles the action space)
    colocation: bool = False,
    # Joint-vs-decoupled ablation: None | "placement_only" | "job_only"
    ablation_mode: Optional[str] = None,
    # SLO-aware reward (aiserve workload; 0 = off): lateness penalty on inference
    slo_penalty: float = 0.0,
    # Anti-idle (0 = off): penalty for NO_OP at a decision point where a job could
    # be scheduled — counters the cold-start "learn to no-op" low-utilization collapse
    noop_penalty: float = 0.0,
    # Episode length cap = n_jobs × this. 200 is generous; lower (e.g. 40) bounds
    # the cost of cold-start episodes where an untrained policy fails to schedule
    # and the episode would otherwise spin to the cap (good episodes are ~n_jobs).
    max_steps_mult: int = 200,
    # Vectorized rollout (Q2 throughput): >1 runs N parallel envs
    num_envs: int = 1,
    async_envs: bool = True,
) -> DSACAgent:
    """Run online DSAC training in sim. Returns the trained agent."""
    obs_dim, n_actions = env_dims(n_nodes, gpus_per_node, colocation=colocation)
    rng = np.random.default_rng(seed)

    total_gpus = n_nodes * gpus_per_node
    families = [trace_family] if isinstance(trace_family, str) else list(trace_family)

    # Curriculum stages: list of (n_jobs, fraction_of_total_steps)
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
        reward_mode=reward_mode,
        reward_scale=reward_scale,
        mo_w_jct=mo_w_jct, mo_w_util=mo_w_util,
        potential_shaping=potential_shaping,
        balance_coef=balance_coef,
        fairness_coef=fairness_coef,
        node_speeds=node_speeds,
        node_gpu_types=node_gpu_types,
        node_ram_gb=node_ram_gb,
        runtime_sigma=runtime_sigma,
        interference=interference,
        oom_sigma=oom_sigma,
        oom_penalty_s=oom_penalty_s,
        ram_overcommit=ram_overcommit,
        colocation_actions=colocation,
        slo_penalty=slo_penalty,
        noop_penalty=noop_penalty,
        ablation_mode=ablation_mode,
    )
    if num_envs > 1 and node_speeds:
        raise NotImplementedError("--node-speeds not wired into the vec path; use --num-envs 1")
    if num_envs > 1 and oom_penalty_s > 0.0:
        raise NotImplementedError("stochastic OOM not wired into the vec path; use --num-envs 1")
    agent = DSACAgent(
        obs_dim=obs_dim, n_actions=n_actions, device=device,
        use_iqn=use_iqn,
        risk_mode=risk_mode, risk_beta=risk_beta,
        value_clip=value_clip,
        fixed_alpha=fixed_alpha, init_alpha=init_alpha,
        target_entropy_ratio=target_entropy_ratio,
    )

    if use_per:
        buf = PrioritizedReplayBuffer(
            capacity=buf_capacity, obs_dim=obs_dim, n_actions=n_actions,
        )
    else:
        buf = ReplayBuffer(capacity=buf_capacity, obs_dim=obs_dim, n_actions=n_actions)

    score_sched = None
    if score_warmup:
        from sim.scheduler.score import ScoreScheduler
        score_sched = ScoreScheduler()

    nstep_buf: deque = deque(maxlen=nstep_n)

    log_fh = None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_fh = open(out_dir / "sim_train.jsonl", "w")

    # ── Vectorized rollout path (Q2): N parallel envs feed the shared learner ──
    if num_envs > 1:
        _collect_and_train_vec(
            agent=agent, buf=buf, rng=rng, log_fh=log_fh,
            num_envs=num_envs, async_envs=async_envs,
            families=families, active_n_jobs=active_n_jobs, total_gpus=total_gpus,
            n_nodes=n_nodes, gpus_per_node=gpus_per_node,
            reward_mode=reward_mode, reward_scale=reward_scale,
            potential_shaping=potential_shaping, runtime_sigma=runtime_sigma,
            interference=interference, colocation=colocation,
            nstep_n=nstep_n, total_steps=total_steps, warmup_steps=warmup_steps,
            update_every=update_every, utd_ratio=utd_ratio, batch_size=batch_size,
            use_per=use_per, seed=seed, log_every=log_every,
            curriculum_stages=curriculum_stages, score_warmup=score_warmup,
            slo_penalty=slo_penalty,
        )
        env.close()
        if log_fh:
            log_fh.close()
        if out_dir:
            ckpt = out_dir / "dsac.pt"
            agent.save(ckpt)
            print(f"[sim_train] saved → {ckpt}")
        return agent

    obs, _ = env.reset(seed=seed)
    ep_steps = ep_reward = 0.0
    ep_count = 0
    last_losses: dict = {}
    t0 = time.time()

    # P2: running-std reward normalization (PopArt-lite). Welford running second
    # moment of the reward; rew/(std+eps) keeps the critic target O(1) without a
    # hand-tuned reward_scale, reducing seed sensitivity. Mean is kept (only the
    # scale is normalized) so the sign/structure of the reward is preserved.
    _rn_count = 0
    _rn_mean = 0.0
    _rn_m2 = 0.0

    def _normalize(r: float) -> float:
        nonlocal _rn_count, _rn_mean, _rn_m2
        _rn_count += 1
        d = r - _rn_mean
        _rn_mean += d / _rn_count
        _rn_m2 += d * (r - _rn_mean)
        if not normalize_reward or _rn_count < 2:
            return r
        std = (_rn_m2 / _rn_count) ** 0.5
        return r / (std + 1e-6)

    for step in range(total_steps):
        # ── Curriculum: switch n_jobs when crossing a stage boundary ────
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
                env.max_steps    = active_n_jobs * 200
                print(f"  [curriculum] step={step}: n_jobs → {active_n_jobs}")

        mask = env.action_mask()

        if len(buf) < warmup_steps:
            state = env._state
            if (score_sched is not None and state is not None
                    and state.pending and state.cluster):
                ordered = score_sched.order(state.pending, state.cluster,
                                            now=state.now)
                legal = np.flatnonzero(mask)
                act = int(rng.choice(legal))
                for job in ordered:
                    job_idx = next(
                        (i for i, j in enumerate(state.pending)
                         if j.job_id == job.job_id),
                        None,
                    )
                    if job_idx is not None:
                        # PACK action on the first placement; with _n_modes==1
                        # this is the legacy job_idx * _n_placements index.
                        candidate = job_idx * env._n_placements * env._n_modes
                        if candidate < len(mask) and mask[candidate]:
                            act = int(candidate)
                            break
            else:
                act = int(rng.choice(np.flatnonzero(mask)))
        else:
            act = agent.select_action(obs, mask)

        next_obs, rew, term, trunc, info = env.step(act)
        next_mask = env.action_mask()

        done = bool(term or trunc)
        nstep_buf.append(Transition(
            obs=obs, act=act, rew=_normalize(float(rew)),
            next_obs=next_obs, done=done,
            mask=mask, next_mask=next_mask,
        ))

        if len(nstep_buf) == nstep_n or done:
            _flush_nstep(nstep_buf, buf, gamma=agent.gamma)
            if done:
                nstep_buf.clear()

        obs = next_obs
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
                    "alpha": last_losses.get("alpha"),
                    "entropy": last_losses.get("entropy"),
                    "loss_critic": last_losses.get("loss_critic"),
                    "loss_actor": last_losses.get("loss_actor"),
                }) + "\n")
            ep_steps = ep_reward = 0.0
            obs, _ = env.reset()

        # ── Gradient updates — only after warmup ────────────────────────
        if len(buf) >= warmup_steps and step % update_every == 0:
            for _ in range(utd_ratio):
                batch = buf.sample(min(batch_size, len(buf)), rng)
                losses = agent.update(batch)
                last_losses = losses
                # PER: update priorities with new TD errors
                if use_per and "indices" in batch and "td_errors" in losses:
                    buf.update_priorities(batch["indices"], losses["td_errors"])

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"  step {step+1:6d}/{total_steps}  buf={len(buf):6d}  "
                  f"eps={ep_count}  n_jobs={active_n_jobs}  "
                  f"alpha={agent.alpha.item():.3f}  elapsed={elapsed:.0f}s")

    env.close()
    if log_fh:
        log_fh.close()

    if out_dir:
        ckpt = out_dir / "dsac.pt"
        agent.save(ckpt)
        print(f"[sim_train] saved → {ckpt}")

    return agent


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-nodes",       type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument("--trace",         default=["philly", "ali"],
                   nargs="+", choices=["philly", "ali", "burst", "aiserve", "aimix"])
    p.add_argument("--n-jobs",        type=int, default=50)
    p.add_argument("--nstep-n",       type=int, default=10)
    p.add_argument("--no-score-warmup", action="store_true")
    p.add_argument("--total-steps",   type=int, default=500_000)
    p.add_argument("--warmup-steps",  type=int, default=2_000)
    p.add_argument("--utd-ratio",     type=int, default=4)
    p.add_argument("--batch-size",    type=int, default=256)
    p.add_argument("--seed",          type=int, default=42)
    # Reward is a single unified design (no single-/multi-objective toggle): the
    # −JCT completion term + convex fairness penalty (--fairness-coef) + potential
    # shaping (wait + node balance via --balance-coef), learned under co-residency
    # interference (--interference). The old --reward-mode / --mo-w-jct / --mo-w-util
    # knobs were removed; uxprl/shaped/jct_aligned remain internal (uxprl via --uxprl,
    # RLPD/eval construct the env directly).
    p.add_argument("--reward-scale",  type=float, default=20_000.0,
                   help="divisor on -JCT; larger → smaller returns "
                        "(keeps entropy term competitive with Q, default 20000)")
    p.add_argument("--device",        default="cpu")
    p.add_argument("--out-dir",
                   default=f"runs/dsac_sim_{time.strftime('%Y%m%d-%H%M%S')}")
    # Architecture flags
    p.add_argument("--no-iqn",               action="store_true",
                   help="vanilla scalar-critic SAC instead of the IQN "
                        "distributional critic (disables risk distortion)")
    p.add_argument("--risk-mode",            choices=list(RISK_MODES),
                   default="mean",
                   help="risk distortion in the RDSAC actor objective")
    p.add_argument("--value-clip",           type=float, default=0.0,
                   help="Duan et al. 2021 target return-clip boundary b (0 = off)")
    p.add_argument("--uxprl",                action="store_true",
                   help="UXP-RL (Lin et al. 2025): faithful value-based DQN with "
                        "ε-greedy + inference-weighted reward. Uses its own lean "
                        "training loop (no score-warmup/n-step/PER); most other "
                        "flags are ignored.")
    p.add_argument("--uxprl-c1",             type=float, default=1.0,
                   help="UXP-RL reward weight for non-inference tasks")
    p.add_argument("--uxprl-c2",             type=float, default=2.0,
                   help="UXP-RL reward weight for inference tasks (c2 > c1)")
    p.add_argument("--risk-beta",            type=float, default=0.25,
                   help="risk parameter (CVaR tail mass, Wang/CPW shape, MSD weight)")
    # Temperature (entropy) controls
    p.add_argument("--fixed-alpha",          action="store_true",
                   help="pin the entropy temperature α (disables auto-tuning)")
    p.add_argument("--init-alpha",           type=float, default=0.1,
                   help="initial α; with --fixed-alpha this is the constant value")
    p.add_argument("--target-entropy-ratio", type=float, default=0.1,
                   help="auto-α target entropy as a fraction of log(n_actions)")
    # Improvement flags
    p.add_argument("--no-potential-shaping", action="store_true",
                   help="disable potential-based reward shaping")
    p.add_argument("--no-per",               action="store_true",
                   help="disable Prioritized Experience Replay")
    p.add_argument("--curriculum",           action="store_true",
                   help="ramp n_jobs: 10→30→50 over training")
    # Stochastic execution (opt-in)
    p.add_argument("--runtime-sigma",        type=float, default=0.0,
                   help="lognormal (mean-preserving) noise on realized runtime; "
                        "0 = deterministic. Gives the distributional critic real "
                        "return spread to model.")
    p.add_argument("--interference",         type=float, default=0.0,
                   help="per-co-resident MPS slowdown on realized runtime; "
                        "0 = off")
    # Stochastic host-RAM OOM (opt-in): turns node-2's drain risk into a learnable
    # tail cost. Needs a bounded --node-ram-gb; single-env path (--num-envs 1) only.
    p.add_argument("--oom-penalty-s",        type=float, default=0.0,
                   help="drain-recovery seconds added when a placement's summed "
                        "realized host-RAM peaks exceed the node budget; 0 = off "
                        "(master switch for the stochastic-OOM model)")
    p.add_argument("--oom-sigma",            type=float, default=0.0,
                   help="lognormal (mean-preserving) spread of realized peak RSS "
                        "vs nominal ram_req; >0 makes tight-but-legal packings OOM "
                        "on unlucky draws")
    p.add_argument("--ram-overcommit",       type=float, default=1.0,
                   help="host-RAM overcommit factor; 1.0 = hard gate, >1.0 admits "
                        "nominally-over-budget co-residency (harsher OOM regime)")
    p.add_argument("--colocation",           action="store_true",
                   help="add PACK/ISOLATE co-location mode per placement "
                        "(doubles the action space; checkpoint-incompatible)")
    p.add_argument("--slo-penalty",          type=float, default=0.0,
                   help="SLO-aware reward (use with --trace aiserve): lateness "
                        "penalty on inference jobs past their deadline; 0 = off")
    p.add_argument("--max-steps-mult",       type=int, default=200,
                   help="episode length cap = n_jobs × this; lower (e.g. 40) "
                        "truncates cold-start spin episodes fast")
    p.add_argument("--noop-penalty",         type=float, default=0.0,
                   help="penalty for NO_OP when a job is schedulable; counters "
                        "the cold-start no-op/low-utilization collapse")
    # Vectorized rollout (Q2 throughput)
    p.add_argument("--num-envs",             type=int, default=1,
                   help="parallel rollout envs; >1 enables the vectorized path "
                        "(score-warmup honored via per-env score_actions())")
    p.add_argument("--sync-envs",            action="store_true",
                   help="with --num-envs>1, step envs in-process instead of in "
                        "worker processes (no wall-clock win; for debugging)")
    p.add_argument("--fairness-coef",        type=float, default=0.0,
                   help="convex (quadratic) per-job JCT penalty on completion: "
                        "min Σ(JCT+fairness·JCT²)=mean+tail (objective-changing, not φ)")
    p.add_argument("--balance-coef",         type=float, default=0.0,
                   help="P1: potential-based node-balance shaping coefficient (§3.6)")
    p.add_argument("--normalize-reward",     action="store_true",
                   help="P2: running-std reward normalization (§3.6)")
    p.add_argument("--node-speeds",          default="",
                   help="legacy scalar per-node speed, e.g. '1.0,0.25'. Superseded by "
                        "--node-gpu-types (per-(card,class) SPEED_MATRIX). Empty = homogeneous.")
    p.add_argument("--node-gpu-types",        default="",
                   help="comma-separated per-node card id, e.g. 'rtx4070,rtx3080'. Enables "
                        "the measured per-(card,job_class) speed matrix + obs card one-hot.")
    p.add_argument("--node-ram-gb",           default="",
                   help="comma-separated per-node usable host RAM in GB, e.g. '62,5'. "
                        "Enables the host-RAM OOM gate (would-be-OOM placements masked out).")
    p.add_argument("--hetero-cluster",        action="store_true",
                   help="shortcut for the real 2×1 cluster: --node-gpu-types rtx4070,rtx3080 "
                        "--node-ram-gb 62,5 (overrides those two if set).")
    args = p.parse_args(argv)
    if args.hetero_cluster:
        args.node_gpu_types = args.node_gpu_types or "rtx4070,rtx3080"
        args.node_ram_gb = args.node_ram_gb or "62,5"

    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[sim_train] CUDA not available, falling back to CPU")
        device = "cpu"

    traces = args.trace if len(args.trace) > 1 else args.trace[0]

    # ── UXP-RL (Lin et al. 2025): faithful DQN path with its own lean loop ──
    if args.uxprl:
        from services.rl_scheduler.uxprl import train_uxprl
        node_speeds = [float(s) for s in args.node_speeds.split(",") if s.strip()] or None
        print(f"[sim_train] arch=UXP-RL(DQN)  n={args.n_nodes}×{args.gpus_per_node}  "
              f"trace={traces}  steps={args.total_steps:,}  n_jobs={args.n_jobs}  "
              f"c1={args.uxprl_c1} c2={args.uxprl_c2}  "
              f"curriculum={args.curriculum}  device={device}")
        train_uxprl(
            n_nodes=args.n_nodes, gpus_per_node=args.gpus_per_node,
            trace_family=traces, n_jobs=args.n_jobs,
            total_steps=args.total_steps, warmup_steps=args.warmup_steps,
            seed=args.seed, out_dir=Path(args.out_dir), device=device,
            uxprl_c1=args.uxprl_c1, uxprl_c2=args.uxprl_c2,
            curriculum=args.curriculum, node_speeds=node_speeds,
            max_steps_mult=args.max_steps_mult,
        )
        return 0

    use_iqn = not args.no_iqn
    family = "RDSAC" if use_iqn else "SAC"
    arch = f"{family}+MLP"
    risk_str = f"risk={args.risk_mode}:{args.risk_beta}  " if use_iqn else ""
    print(f"[sim_train] arch={arch}  n={args.n_nodes}×{args.gpus_per_node}  "
          f"trace={traces}  steps={args.total_steps:,}  "
          f"n_jobs={args.n_jobs}  nstep={args.nstep_n}  "
          f"PER={not args.no_per}  shaping={not args.no_potential_shaping}  "
          f"{risk_str}"
          f"curriculum={args.curriculum}  device={device}")
    sim_train(
        n_nodes=args.n_nodes, gpus_per_node=args.gpus_per_node,
        trace_family=traces, n_jobs=args.n_jobs,
        nstep_n=args.nstep_n, score_warmup=not args.no_score_warmup,
        total_steps=args.total_steps, warmup_steps=args.warmup_steps,
        utd_ratio=args.utd_ratio, batch_size=args.batch_size,
        seed=args.seed, reward_mode="mo",   # single unified reward (no single/multi toggle)
        reward_scale=args.reward_scale,
        mo_w_jct=1.0, mo_w_util=0.0,
        out_dir=Path(args.out_dir), device=device,
        use_iqn=use_iqn,
        fixed_alpha=args.fixed_alpha, init_alpha=args.init_alpha,
        target_entropy_ratio=args.target_entropy_ratio,
        potential_shaping=not args.no_potential_shaping,
        use_per=not args.no_per,
        risk_mode=args.risk_mode,
        risk_beta=args.risk_beta,
        value_clip=args.value_clip,
        curriculum=args.curriculum,
        runtime_sigma=args.runtime_sigma,
        interference=args.interference,
        oom_sigma=args.oom_sigma,
        oom_penalty_s=args.oom_penalty_s,
        ram_overcommit=args.ram_overcommit,
        colocation=args.colocation,
        slo_penalty=args.slo_penalty,
        noop_penalty=args.noop_penalty,
        max_steps_mult=args.max_steps_mult,
        balance_coef=args.balance_coef,
        fairness_coef=args.fairness_coef,
        normalize_reward=args.normalize_reward,
        node_speeds=[float(s) for s in args.node_speeds.split(",") if s.strip()] or None,
        node_gpu_types=[s.strip() for s in args.node_gpu_types.split(",") if s.strip()] or None,
        node_ram_gb=[float(s) for s in args.node_ram_gb.split(",") if s.strip()] or None,
        num_envs=args.num_envs,
        async_envs=not args.sync_envs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
