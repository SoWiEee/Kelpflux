"""Risk-sensitive Distributional Soft Actor-Critic (RDSAC) with action masking.

Faithful discrete transpose of Ma et al. 2020/2025, "DSAC: Distributional Soft
Actor-Critic for Risk-Sensitive Reinforcement Learning" (arXiv:2004.14547),
official repo https://github.com/xtma/dsac.

The scheduler action space is finite and masked, so the continuous reparameterised
Gaussian actor of the paper is replaced by an explicit **categorical actor**
``π(a|s;φ)``; the distributional critic and the risk machinery follow the paper.

Critic (per Ma et al. §4.1, "RDSAC"): the soft return is split into a reward
distribution ``Z_R`` and an entropy distribution ``Z_H``, each parameterised by an
Implicit Quantile Network (Dabney et al. 2018) over the same shared trunk (heads
differ only in the final layer). Both are trained with quantile Huber regression
and double learning (twin critics, per-quantile min on the target).

Convention (α-external): ``Z_H`` regresses the pure entropy return in nats; the
combined value is ``Q = E[Z_R] + α·E[Z_H]``. Because α is auto-tuned, keeping it
outside means the entropy distribution does not relearn when α moves.

Actor (Ma et al. §4.1 objective, discrete categorical sum, masked):

    J_π(φ) = E_s Σ_a π(a|s) · [ α·log π(a|s) − ρ[Z_R(s,a)] − α·E[Z_H(s,a)] ]

where ρ is a risk distortion (mean / cvar / wang / cpw / msd) applied to the
reward distribution only — risk is injected into the policy objective, not just at
action selection. See ``distortion.py`` for the estimators.

References:
  Ma et al. 2020/2025 — DSAC (risk-sensitive distributional SAC)
  Dabney et al. 2018 ICML — Implicit Quantile Networks
  Christodoulou 2019 — Soft Actor-Critic for Discrete Action Spaces
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from services.rl_scheduler.distortion import RISK_MODES, distorted_values


def _build_mlp(in_dim: int, hidden: Sequence[int], out_dim: int,
               layer_norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    for m in layers:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)
    return nn.Sequential(*layers)


class _AttentionEncoder(nn.Module):
    """Permutation-invariant trunk: self-attention over job tokens → (B, d).

    Observation layout (must match gym_env.py _build_obs):
        obs = [job_0 (JOB_DIM) … job_{K-1} (JOB_DIM), cluster (rest)]
    Job slots whose features are all zero are treated as padding.
    """

    TOP_K: int = 16
    JOB_DIM: int = 11

    def __init__(self, obs_dim: int, d: int, n_heads: int = 4,
                 n_layers: int = 2) -> None:
        super().__init__()
        self.cluster_dim = obs_dim - self.TOP_K * self.JOB_DIM
        assert self.cluster_dim > 0, f"obs_dim={obs_dim} too small for attention trunk"
        self.job_embed = nn.Linear(self.JOB_DIM, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 2,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False)
        self.cluster_embed = nn.Linear(self.cluster_dim, d)
        self.proj = nn.Linear(d * 2, d)
        for m in [self.job_embed, self.cluster_embed, self.proj]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        k, jd = self.TOP_K, self.JOB_DIM
        jobs = obs[:, :k * jd].view(-1, k, jd)
        cluster = obs[:, k * jd:]
        pad = (jobs.abs().sum(dim=-1) == 0)
        tok = F.relu(self.job_embed(jobs))
        enc = self.transformer(tok, src_key_padding_mask=pad)
        non_pad = (~pad).float().unsqueeze(-1)
        n_valid = non_pad.sum(dim=1).clamp(min=1.0)
        queue_ctx = (enc * non_pad).sum(dim=1) / n_valid
        cluster_ctx = F.relu(self.cluster_embed(cluster))
        return F.relu(self.proj(torch.cat([queue_ctx, cluster_ctx], dim=-1)))


class _DualIQNCritic(nn.Module):
    """Dual-head Implicit Quantile Network: reward return Z_R and entropy return Z_H.

    Shared trunk (MLP or attention) → state embedding d; cosine-embedded quantile
    fraction τ multiplies it elementwise; two linear heads emit Z_R and Z_H
    quantiles, each (B, N_QUANT, n_actions).
    """

    N_QUANT: int = 32   # quantile samples per forward
    N_COS: int = 64     # cosine embedding dimension for τ

    def __init__(self, obs_dim: int, n_actions: int,
                 hidden: Sequence[int] = (256, 256), layer_norm: bool = True,
                 use_attention: bool = False) -> None:
        super().__init__()
        d = hidden[-1]
        if use_attention:
            self.encoder = _AttentionEncoder(obs_dim, d)
        else:
            enc_hidden = hidden[:-1] if len(hidden) > 1 else ()
            self.encoder = _build_mlp(obs_dim, enc_hidden, d, layer_norm)
        self.phi_embed = nn.Linear(self.N_COS, d)
        self.head_r = nn.Linear(d, n_actions)
        self.head_h = nn.Linear(d, n_actions)
        for m in (self.phi_embed, self.head_r, self.head_h):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)

    def quantile_q(self, obs: torch.Tensor, taus: torch.Tensor | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (Z_R, Z_H), each (B, N_QUANT, n_actions)."""
        b = obs.shape[0]
        if taus is None:
            taus = torch.rand(b, self.N_QUANT, device=obs.device)
        s = F.relu(self.encoder(obs))                                  # (B, d)
        i = torch.arange(1, self.N_COS + 1, device=obs.device, dtype=obs.dtype)
        cos = torch.cos(math.pi * taus.unsqueeze(-1) * i)              # (B, N, N_COS)
        phi = F.relu(self.phi_embed(cos))                              # (B, N, d)
        combined = s.unsqueeze(1) * phi                                # (B, N, d)
        return self.head_r(combined), self.head_h(combined)


class _CategoricalActor(nn.Module):
    """Explicit masked categorical policy π(a|s;φ)."""

    def __init__(self, obs_dim: int, n_actions: int,
                 hidden: Sequence[int] = (256, 256), layer_norm: bool = True) -> None:
        super().__init__()
        self.net = _build_mlp(obs_dim, hidden, n_actions, layer_norm)

    def policy(self, obs: torch.Tensor, mask: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked (probs, log_probs), each (B, n_actions). Masked log_probs = 0."""
        logits = self.net(obs).masked_fill(~mask, -1e9)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        log_probs = log_probs.masked_fill(~mask, 0.0)
        return probs, log_probs


def _quantile_huber(pred: torch.Tensor, target: torch.Tensor,
                    taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Per-sample quantile Huber loss. pred (B,Np), target (B,Nt), taus (B,Np) → (B,)."""
    u = target.unsqueeze(1) - pred.unsqueeze(2)                        # (B, Np, Nt)
    huber = torch.where(u.abs() < kappa, 0.5 * u ** 2,
                        kappa * (u.abs() - 0.5 * kappa))
    tau_w = (taus.unsqueeze(2) - (u.detach() < 0).float()).abs()       # (B, Np, Nt)
    return (tau_w * huber).mean(dim=(1, 2))                            # (B,)


class DSACAgent:
    """Risk-sensitive distributional SAC for masked scheduling (Ma et al. discrete).

    Usage::
        agent = DSACAgent(obs_dim=192, n_actions=17, risk_mode="cvar", risk_beta=0.25)
        a = agent.select_action(obs, mask)
        info = agent.update(batch)   # dict incl. td_errors for PER
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: Sequence[int] = (256, 256),
        lr_q: float = 3e-4,
        lr_pi: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_alpha: float = 0.1,
        target_entropy_ratio: float = 0.1,
        fixed_alpha: bool = False,
        layer_norm: bool = True,
        use_attention: bool = False,
        risk_mode: str = "mean",
        risk_beta: float = 0.25,
        risk_alpha: float | None = None,   # deprecated alias for risk_beta
        device: str = "cpu",
    ) -> None:
        if risk_mode not in RISK_MODES:
            raise ValueError(f"risk_mode must be one of {RISK_MODES}")
        if risk_alpha is not None:          # back-compat: old CVaR tail-mass arg
            risk_beta = risk_alpha

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = tuple(hidden)
        self.gamma = gamma
        self.tau = tau
        self.use_attention = use_attention
        self.risk_mode = risk_mode
        self.risk_beta = float(risk_beta)
        self.target_entropy_ratio = target_entropy_ratio
        self.fixed_alpha = fixed_alpha
        self.device = torch.device(device)

        def _critic():
            return _DualIQNCritic(obs_dim, n_actions, self.hidden, layer_norm,
                                  use_attention).to(self.device)

        self.q1, self.q2 = _critic(), _critic()
        self.q1_target, self.q2_target = _critic(), _critic()
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor = _CategoricalActor(obs_dim, n_actions, self.hidden,
                                       layer_norm).to(self.device)
        self.actor_target = _CategoricalActor(obs_dim, n_actions, self.hidden,
                                              layer_norm).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.log_alpha = torch.tensor(
            math.log(init_alpha), dtype=torch.float32,
            requires_grad=not fixed_alpha, device=self.device)

        self.opt_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr_q)
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=lr_pi)
        self.opt_alpha = (None if fixed_alpha
                          else torch.optim.Adam([self.log_alpha], lr=lr_alpha))
        self._update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    def _risk_value(self, z_r: torch.Tensor, z_h: torch.Tensor,
                    taus: torch.Tensor) -> torch.Tensor:
        """Combined action value ρ[Z_R] + α·E[Z_H]. z_* (B,N,A), taus (B,N) → (B,A)."""
        a = z_r.shape[-1]
        taus_a = taus.unsqueeze(1).expand(-1, a, -1)                   # (B, A, N)
        rho_r = distorted_values(z_r.permute(0, 2, 1), taus_a,
                                 self.risk_mode, self.risk_beta)       # (B, A)
        e_h = z_h.mean(dim=1)                                          # (B, A)
        return rho_r + self.alpha.detach() * e_h

    def _soft_update(self, src: nn.Module, tgt: nn.Module) -> None:
        for p, pt in zip(src.parameters(), tgt.parameters()):
            pt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

    def update(self, batch: Dict[str, np.ndarray]) -> dict:
        def _t(k, dtype=torch.float32):
            return torch.as_tensor(batch[k], dtype=dtype, device=self.device)

        obs = _t("obs")
        acts = _t("acts", torch.long)
        rews = _t("rews")
        next_obs = _t("next_obs")
        dones = _t("dones")
        masks = _t("masks", torch.bool)
        next_masks = _t("next_masks", torch.bool)
        gammas = _t("gammas") if "gammas" in batch else torch.full_like(rews, self.gamma)
        is_weights = _t("weights") if "weights" in batch else torch.ones_like(rews)
        b = obs.shape[0]
        n = self.q1.N_QUANT

        # ---- Critic: distributional soft targets (a' ~ target policy) ----
        with torch.no_grad():
            probs_n, logp_n = self.actor_target.policy(next_obs, next_masks)
            next_acts = torch.distributions.Categorical(probs=probs_n).sample()
            logp_next = logp_n.gather(1, next_acts.unsqueeze(1)).squeeze(1)
            taus_t = torch.rand(b, n, device=self.device)
            zr1, zh1 = self.q1_target.quantile_q(next_obs, taus_t)
            zr2, zh2 = self.q2_target.quantile_q(next_obs, taus_t)
            idx = next_acts.view(b, 1, 1).expand(-1, n, 1)
            zr_next = torch.minimum(zr1.gather(2, idx), zr2.gather(2, idx)).squeeze(2)
            zh_next = torch.minimum(zh1.gather(2, idx), zh2.gather(2, idx)).squeeze(2)
            g = (gammas * (1.0 - dones)).unsqueeze(1)                  # (B,1)
            target_r = rews.unsqueeze(1) + g * zr_next                 # (B,N)
            target_h = g * (zh_next - logp_next.unsqueeze(1))          # (B,N)

        loss_critic = obs.new_zeros(())
        td_accum = obs.new_zeros(b)
        aidx = acts.view(b, 1, 1).expand(-1, n, 1)
        for q in (self.q1, self.q2):
            taus = torch.rand(b, n, device=self.device)
            zr, zh = q.quantile_q(obs, taus)
            zr_a = zr.gather(2, aidx).squeeze(2)                       # (B,N)
            zh_a = zh.gather(2, aidx).squeeze(2)
            per_sample = (_quantile_huber(zr_a, target_r, taus)
                          + _quantile_huber(zh_a, target_h, taus))    # (B,)
            loss_critic = loss_critic + (is_weights * per_sample).mean()
            td_accum = td_accum + per_sample.detach()
        td_errors = (td_accum / 2.0).cpu().numpy()

        self.opt_q.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
        self.opt_q.step()

        self._update_count += 1
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        # ---- Actor: risk-sensitive policy objective (masked categorical sum) ----
        probs, log_probs = self.actor.policy(obs, masks)              # (B,A) grad
        with torch.no_grad():
            taus_a = torch.rand(b, n, device=self.device)
            zr1a, zh1a = self.q1.quantile_q(obs, taus_a)
            zr2a, zh2a = self.q2.quantile_q(obs, taus_a)
            zr_a = torch.minimum(zr1a, zr2a)
            zh_a = torch.minimum(zh1a, zh2a)
            q_action = self._risk_value(zr_a, zh_a, taus_a)           # (B,A)
        loss_actor = (probs * (self.alpha.detach() * log_probs - q_action)).sum(-1).mean()

        self.opt_pi.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.opt_pi.step()
        self._soft_update(self.actor, self.actor_target)

        # ---- Temperature α (auto-tune unless fixed) ----
        with torch.no_grad():
            entropy = -(probs * log_probs).sum(-1)                    # (B,)
        entropy_val = float(entropy.mean().item())
        loss_alpha_val = 0.0
        if not self.fixed_alpha:
            n_valid = masks.float().sum(-1).clamp(min=1.0)
            target_entropy = self.target_entropy_ratio * torch.log(n_valid)
            loss_alpha = (self.log_alpha * (entropy - target_entropy).detach()).mean()
            self.opt_alpha.zero_grad()
            loss_alpha.backward()
            self.opt_alpha.step()
            with torch.no_grad():
                # Upper bound generous (α≤~20): auto-tune needs head-room to
                # balance the entropy term against the return scale. A tight
                # ceiling (old 1.0 → α≤2.72) pins α and silently disables
                # temperature control when returns are O(10). Lower bound keeps
                # α from collapsing to zero (entropy term vanishes → no explore).
                self.log_alpha.clamp_(-5.0, 3.0)
            loss_alpha_val = float(loss_alpha.item())

        return {
            "loss_critic": float(loss_critic.item()),
            "loss_actor": float(loss_actor.item()),
            "loss_alpha": loss_alpha_val,
            "alpha": float(self.alpha.item()),
            "entropy": entropy_val,
            "td_errors": td_errors,
        }

    @torch.no_grad()
    def action_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Risk-adjusted action value ρ[Z_R] + α·E[Z_H] per action. (B,obs)→(B,A)."""
        b = obs.shape[0]
        taus = torch.rand(b, self.q1.N_QUANT, device=self.device)
        zr1, zh1 = self.q1.quantile_q(obs, taus)
        zr2, zh2 = self.q2.quantile_q(obs, taus)
        return self._risk_value(torch.minimum(zr1, zr2),
                                torch.minimum(zh1, zh2), taus)

    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray, mask: np.ndarray,
                      greedy: bool = False) -> int:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool,
                                     device=self.device).unsqueeze(0)
            probs, _ = self.actor.policy(obs_t, mask_t)
            p = probs.squeeze(0).cpu().numpy()
        p = p * mask.astype(np.float32)
        total = p.sum()
        if total < 1e-9:
            return int(np.flatnonzero(mask)[0])
        p /= total
        if greedy:
            return int(p.argmax())
        return int(np.random.choice(len(p), p=p))

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "opt_q": self.opt_q.state_dict(), "opt_pi": self.opt_pi.state_dict(),
            "opt_alpha": self.opt_alpha.state_dict() if self.opt_alpha else None,
            "log_alpha": self.log_alpha.item(),
            "fixed_alpha": self.fixed_alpha,
            "use_attention": self.use_attention,
            "risk_mode": self.risk_mode, "risk_beta": self.risk_beta,
            "target_entropy_ratio": self.target_entropy_ratio,
            "hidden": list(self.hidden),
            "obs_dim": self.obs_dim, "n_actions": self.n_actions,
            "update_count": self._update_count,
        }, str(path))

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "DSACAgent":
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        agent = cls(
            obs_dim=data["obs_dim"], n_actions=data["n_actions"],
            hidden=tuple(data.get("hidden", (256, 256))),
            fixed_alpha=data.get("fixed_alpha", False),
            use_attention=data.get("use_attention", False),
            risk_mode=kwargs.pop("risk_mode", data.get("risk_mode", "mean")),
            risk_beta=kwargs.pop("risk_beta", data.get("risk_beta", 0.25)),
            target_entropy_ratio=data.get("target_entropy_ratio", 0.1),
            **kwargs)
        agent.actor.load_state_dict(data["actor"])
        agent.actor_target.load_state_dict(data["actor_target"])
        agent.q1.load_state_dict(data["q1"])
        agent.q2.load_state_dict(data["q2"])
        agent.q1_target.load_state_dict(data["q1_target"])
        agent.q2_target.load_state_dict(data["q2_target"])
        agent.opt_q.load_state_dict(data["opt_q"])
        agent.opt_pi.load_state_dict(data["opt_pi"])
        with torch.no_grad():
            agent.log_alpha.fill_(float(data["log_alpha"]))
        if agent.opt_alpha and data.get("opt_alpha"):
            agent.opt_alpha.load_state_dict(data["opt_alpha"])
        agent._update_count = data.get("update_count", 0)
        return agent
