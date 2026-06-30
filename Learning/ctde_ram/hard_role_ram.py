"""Hard-role high-level RAM networks and learners.

The two supported methods share the exact token architecture:
3-channel allocentric map CNN + previous-role embedding + budget projection,
selectable preference FiLM/preference token, and Transformer-style fleet
self-attention.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from .popart import PopArtNorm
except ImportError:
    from popart import PopArtNorm


class AllocentricMapCNN(nn.Module):
    """One shared CNN over the complete three-channel map stack."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, d_model),
        )

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        return self.net(maps)


class PreferenceFiLM(nn.Module):
    def __init__(self, preference_dim: int = 2, d_model: int = 64):
        super().__init__()
        self.d_model = int(d_model)
        self.net = nn.Sequential(
            nn.Linear(preference_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, preference: torch.Tensor):
        preference = preference / preference.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        gamma_raw, beta = self.net(preference).chunk(2, dim=-1)
        return 1.0 + gamma_raw, beta


class AttentionEncoderLayer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, ff_dim=128, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.ReLU(),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        attn_out, attn_w = self.attn(
            x, x, x, need_weights=return_attn, average_attn_weights=False
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return (x, attn_w) if return_attn else x


class HardRoleAttentionTrunk(nn.Module):
    def __init__(
        self, d_model=64, n_heads=4, n_layers=1, ff_dim=128,
        preference_role_bias: bool = False,
        preference_conditioning: str = "film",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.preference_conditioning = str(preference_conditioning).lower()
        if self.preference_conditioning not in {"film", "pref_token"}:
            raise ValueError("preference_conditioning must be one of: film, pref_token")
        if self.preference_conditioning == "pref_token" and int(n_layers) < 2:
            raise ValueError("pref_token preference conditioning requires n_layers >= 2")
        self.map_cnn = AllocentricMapCNN(d_model)
        self.previous_role_embedding = nn.Embedding(2, d_model)
        self.budget_projection = nn.Linear(1, d_model)
        self.preference_film = (
            PreferenceFiLM(2, d_model)
            if self.preference_conditioning == "film" else None
        )
        self.preference_projection = (
            nn.Linear(2, d_model)
            if self.preference_conditioning == "pref_token" else None
        )
        self.layers = nn.ModuleList([
            AttentionEncoderLayer(d_model, n_heads, ff_dim) for _ in range(n_layers)
        ])
        self.preference_role_bias = bool(preference_role_bias)
        self.bias_mlp = nn.Linear(2, 2) if self.preference_role_bias else None

    def forward(
        self, maps: torch.Tensor, previous_roles: torch.Tensor,
        budget: torch.Tensor, preference: torch.Tensor,
        return_attn: bool = False,
    ):
        if maps.ndim != 5 or maps.shape[2] != 3:
            raise ValueError(f"Hard-role RAM expects [B,N,3,H,W] maps, got {tuple(maps.shape)}")
        batch, n_agents = maps.shape[:2]
        z = self.map_cnn(maps.reshape(batch * n_agents, *maps.shape[2:]))
        z = z.reshape(batch, n_agents, self.d_model)
        role_z = self.previous_role_embedding(previous_roles.long())
        # Preferred path: one remaining-budget fraction per agent [B,N,1].
        # Keep [B] / [B,1] support for old checkpoints and synthetic callers by
        # broadcasting the fleet scalar to every homogeneous agent.
        if budget.ndim == 1:
            budget = budget.unsqueeze(-1)
        if budget.ndim == 2:
            if budget.shape[1] == 1:
                budget_z = self.budget_projection(budget).unsqueeze(1).expand(-1, n_agents, -1)
            elif budget.shape[1] == n_agents:
                budget_z = self.budget_projection(budget.unsqueeze(-1))
            else:
                raise ValueError(f"Budget must be [B,1], [B,N], or [B,N,1], got {tuple(budget.shape)}")
        elif budget.ndim == 3 and budget.shape[1:] == (n_agents, 1):
            budget_z = self.budget_projection(budget)
        else:
            raise ValueError(f"Budget must be [B,1], [B,N], or [B,N,1], got {tuple(budget.shape)}")
        token = z + role_z + budget_z
        if self.preference_conditioning == "film":
            gamma, beta = self.preference_film(preference)
            token = gamma.unsqueeze(1) * token + beta.unsqueeze(1)
        else:
            normalized_preference = preference / preference.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            preference_token = self.preference_projection(normalized_preference).unsqueeze(1)
            token = torch.cat([token, preference_token], dim=1)

        attention = []
        for layer in self.layers:
            if return_attn:
                token, weights = layer(token, return_attn=True)
                attention.append(weights)
            else:
                token = layer(token)
        # The preference token participates in every attention layer but is not
        # exposed to the per-agent actor/Q heads or fleet-pooled critic.
        if self.preference_conditioning == "pref_token":
            token = token[:, :n_agents, :]
        return token, attention

    def role_bias(self, preference: torch.Tensor):
        return self.bias_mlp(preference).unsqueeze(1) if self.bias_mlp is not None else 0.0


class PPOActor(nn.Module):
    def __init__(self, **trunk_kwargs):
        super().__init__()
        self.trunk = HardRoleAttentionTrunk(**trunk_kwargs)
        self.role_head = nn.Linear(self.trunk.d_model, 2)
        nn.init.normal_(self.role_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.role_head.bias)

    def forward(self, maps, previous_roles, budget, preference, return_attn=False):
        h, attn = self.trunk(maps, previous_roles, budget, preference, return_attn)
        logits = self.role_head(h) + self.trunk.role_bias(preference)
        return (logits, attn) if return_attn else logits

    def act(self, maps, previous_roles, budget, preference, deterministic=False, return_attn=False):
        result = self.forward(maps, previous_roles, budget, preference, return_attn)
        logits, attn = result if return_attn else (result, None)
        dist = Categorical(logits=logits)
        roles = logits.argmax(dim=-1) if deterministic else dist.sample()
        logprob = dist.log_prob(roles).sum(dim=-1)
        return roles, logprob, dist.probs, attn


class PPOCritic(nn.Module):
    def __init__(self, n_values: int = 1, **trunk_kwargs):
        super().__init__()
        self.trunk = HardRoleAttentionTrunk(**trunk_kwargs)
        self.n_values = int(n_values)
        if self.n_values <= 0:
            raise ValueError("n_values must be positive")
        # Keep the legacy scalar key (`value_head.*`) checkpoint-compatible.
        # Vector critics use separate final layers so PopArt can preserve each
        # objective's unnormalized prediction independently.
        if self.n_values == 1:
            self.value_head = nn.Linear(self.trunk.d_model, 1)
        else:
            self.value_heads = nn.ModuleList([
                nn.Linear(self.trunk.d_model, 1) for _ in range(self.n_values)
            ])

    def forward(self, maps, previous_roles, budget, preference):
        h, _ = self.trunk(maps, previous_roles, budget, preference)
        pooled = h.mean(dim=1)
        if self.n_values == 1:
            return self.value_head(pooled).squeeze(-1)
        return torch.cat([head(pooled) for head in self.value_heads], dim=-1)

    def popart_heads(self):
        return [self.value_head] if self.n_values == 1 else list(self.value_heads)


class HardRoleQNetwork(nn.Module):
    def __init__(self, **trunk_kwargs):
        super().__init__()
        self.trunk = HardRoleAttentionTrunk(**trunk_kwargs)
        self.role_head = nn.Linear(self.trunk.d_model, 2)
        nn.init.normal_(self.role_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.role_head.bias)

    def forward(self, maps, previous_roles, budget, preference, return_attn=False):
        h, attn = self.trunk(maps, previous_roles, budget, preference, return_attn)
        q = self.role_head(h) + self.trunk.role_bias(preference)
        return (q, attn) if return_attn else q


class HardRoleQMixer(nn.Module):
    """Optional QMIX aggregation; the default baseline is a plain utility sum."""

    def __init__(self, n_agents: int, context_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        self.n_agents = int(n_agents)
        self.embed_dim = int(embed_dim)
        self.hyper_w1 = nn.Sequential(nn.Linear(context_dim, 64), nn.ReLU(), nn.Linear(64, n_agents * embed_dim))
        self.hyper_b1 = nn.Linear(context_dim, embed_dim)
        self.hyper_w2 = nn.Sequential(nn.Linear(context_dim, 64), nn.ReLU(), nn.Linear(64, embed_dim))
        self.v = nn.Sequential(nn.Linear(context_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, utilities: torch.Tensor, context: torch.Tensor):
        b = utilities.shape[0]
        w1 = self.hyper_w1(context).abs().view(b, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(context).unsqueeze(1)
        hidden = F.elu(torch.bmm(utilities.unsqueeze(1), w1) + b1)
        w2 = self.hyper_w2(context).abs().view(b, self.embed_dim, 1)
        return (torch.bmm(hidden, w2) + self.v(context).unsqueeze(1)).view(-1)


def q_context(network: HardRoleQNetwork, maps, previous_roles, budget, preference):
    h, _ = network.trunk(maps, previous_roles, budget, preference)
    return h.mean(dim=1)


@dataclass
class PPOMetrics:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    epochs_ran: int


class PPORAMLearner:
    def __init__(
        self, actor: PPOActor, critic: PPOCritic, actor_lr=3e-4, critic_lr=3e-4,
        epochs=4, minibatch_size=32, clip_eps=0.2, gae_lambda=0.95,
        entropy_coef=0.01, value_coef=0.5, gamma=0.99, target_kl=0.02,
        max_grad_norm=0.5, device="cpu", critic_mode="scalar",
        critic_popart=False, popart_alpha=1e-3,
        advantage_scalarization="ws", scalarization_power=3.0,
        ewc_p=1.0,
    ):
        self.actor, self.critic = actor, critic
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        self.epochs, self.minibatch_size = int(epochs), int(minibatch_size)
        self.clip_eps, self.gae_lambda = float(clip_eps), float(gae_lambda)
        self.entropy_coef, self.value_coef, self.gamma = float(entropy_coef), float(value_coef), float(gamma)
        self.target_kl = None if target_kl is None or target_kl <= 0 else float(target_kl)
        self.max_grad_norm = float(max_grad_norm)
        self.device = torch.device(device)
        self.critic_mode = critic_mode.lower()
        self.vector_critic = self.critic_mode == "vector"
        if self.critic_mode not in {"scalar", "vector"}:
            raise ValueError("critic_mode must be scalar or vector")
        self.advantage_scalarization = advantage_scalarization.lower()
        if self.advantage_scalarization not in {"ws", "wp", "wpop", "ewc"}:
            raise ValueError("advantage_scalarization must be one of: ws, wp, wpop, ewc")
        if not self.vector_critic and self.advantage_scalarization != "ws":
            raise ValueError("nonlinear advantage scalarization requires --ppo-critic-mode vector")
        self.scalarization_power, self.ewc_p = float(scalarization_power), float(ewc_p)
        self.scalarization_eps = 1e-8
        self.critic_popart = bool(critic_popart)
        n_values = 2 if self.vector_critic else 1
        self.popart = (
            [PopArtNorm(head, alpha=popart_alpha, rescale=True) for head in critic.popart_heads()]
            if self.critic_popart else []
        )

    def denormalize_values(self, values: torch.Tensor) -> torch.Tensor:
        if not self.popart:
            return values
        if self.vector_critic:
            return torch.stack([pa.denormalize(values[..., k]) for k, pa in enumerate(self.popart)], dim=-1)
        return self.popart[0].denormalize(values)

    def _normalize_value_targets(self, returns: torch.Tensor) -> torch.Tensor:
        if not self.popart:
            return returns
        for k, pa in enumerate(self.popart):
            targets = returns[..., k] if self.vector_critic else returns
            pa.update(targets)
        if self.vector_critic:
            return torch.stack([pa.normalize_target(returns[..., k]) for k, pa in enumerate(self.popart)], dim=-1)
        return self.popart[0].normalize_target(returns)

    def _scalarize_advantages(self, advantages: torch.Tensor, preference: torch.Tensor) -> torch.Tensor:
        # Per-objective standardization keeps a preference weight semantically
        # comparable even when the objective returns have different scales.
        mean = advantages.mean(dim=0, keepdim=True)
        std = advantages.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
        z = (advantages - mean) / std
        weights = preference / preference.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.advantage_scalarization == "ws":
            scalar = (weights * z).sum(dim=-1)
            return (scalar - scalar.mean()) / scalar.std(unbiased=False).clamp_min(1e-8)
        # Product/power scalarizers need non-negative inputs. A rollout-local,
        # per-objective min-max mapping retains ordering and avoids invalid
        # fractional powers of signed GAE values.
        lo, hi = z.amin(dim=0, keepdim=True), z.amax(dim=0, keepdim=True)
        positive = (z - lo) / (hi - lo).clamp_min(1e-8)
        positive = positive.clamp_min(self.scalarization_eps)
        if self.advantage_scalarization == "wp":
            scalar = (weights * positive.pow(self.scalarization_power)).sum(dim=-1)
        elif self.advantage_scalarization == "wpop":
            scalar = torch.prod(positive.pow(weights), dim=-1)
        else:
            weight_gain = torch.exp(self.ewc_p * weights) - 1.0
            scalar = (weight_gain * torch.exp(self.ewc_p * positive)).sum(dim=-1)
        # PPO needs a signed learning signal. Nonlinear scores are positive by
        # construction, so center/scale the final scalarized score.
        return (scalar - scalar.mean()) / scalar.std(unbiased=False).clamp_min(1e-8)

    def popart_state_dict(self):
        return [
            {"mu": pa.mu, "mu_sq": pa.mu_sq, "sigma": pa.sigma, "alpha": pa.alpha}
            for pa in self.popart
        ]

    def load_popart_state_dict(self, states):
        if not states:
            return
        for pa, state in zip(self.popart, states):
            pa.mu = float(state.get("mu", pa.mu))
            pa.mu_sq = float(state.get("mu_sq", pa.mu_sq))
            pa.sigma = float(state.get("sigma", pa.sigma))

    def update(self, rollout):
        if len(rollout) == 0:
            return None
        data = rollout.as_tensors(self.device)
        with torch.no_grad():
            next_values = self.denormalize_values(self.critic(data["next_maps"], data["next_previous_roles"], data["next_budget"], data["preference"]))
            discounts = self.gamma ** data["duration"]
            bootstrap_discount = discounts.unsqueeze(-1) if self.vector_critic else discounts
            bootstrap_mask = (1.0 - data["done"]).unsqueeze(-1) if self.vector_critic else (1.0 - data["done"])
            deltas = data["reward"] + bootstrap_discount * next_values * bootstrap_mask - data["value"]
            advantages = torch.zeros_like(deltas)
            gae = torch.zeros((), device=self.device)
            for t in range(len(deltas) - 1, -1, -1):
                gae = deltas[t] + discounts[t] * self.gae_lambda * (1.0 - data["done"][t]) * gae
                advantages[t] = gae
            returns = advantages + data["value"]
            if self.vector_critic:
                advantages = self._scalarize_advantages(advantages, data["preference"])
            else:
                advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
            value_targets = self._normalize_value_targets(returns)

        metrics = []
        count = len(advantages)
        epochs_ran = 0
        stop_early = False
        for _ in range(self.epochs):
            epoch_kls = []
            for idx in torch.randperm(count, device=self.device).split(self.minibatch_size):
                logits = self.actor(data["maps"][idx], data["previous_roles"][idx], data["budget"][idx], data["preference"][idx])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(data["roles"][idx]).sum(dim=-1)
                ratio = (new_logp - data["old_logprob"][idx]).exp()
                approx_kl = (data["old_logprob"][idx] - new_logp).mean()
                unclipped = ratio * advantages[idx]
                clipped = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                entropy = dist.entropy().mean()
                value = self.critic(data["maps"][idx], data["previous_roles"][idx], data["budget"][idx], data["preference"][idx])
                value_loss = F.mse_loss(value, value_targets[idx])

                self.actor_optim.zero_grad()
                (policy_loss - self.entropy_coef * entropy).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.zero_grad()
                (self.value_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()
                total = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                if not all(torch.isfinite(x) for x in (total, policy_loss, value_loss, entropy, approx_kl)):
                    raise FloatingPointError("Non-finite PPO loss/statistic")
                metrics.append((total.item(), policy_loss.item(), value_loss.item(), entropy.item(), approx_kl.item()))
                epoch_kls.append(max(float(approx_kl.item()), 0.0))
            epochs_ran += 1
            if self.target_kl is not None and epoch_kls and np.mean(epoch_kls) > self.target_kl:
                stop_early = True
            if stop_early:
                break
        rollout.clear()
        mean = np.asarray(metrics).mean(axis=0)
        return PPOMetrics(*map(float, mean), epochs_ran=epochs_ran)


class HardRoleQLearner:
    def __init__(
        self, online: HardRoleQNetwork, n_agents: int, lr=3e-4,
        target_update=50, mixer="none", gamma=0.99, per_eps=1e-6, device="cpu",
    ):
        self.online = online
        self.target = copy.deepcopy(online).eval()
        self.mixer_name = mixer
        self.mixer = HardRoleQMixer(n_agents, online.trunk.d_model).to(device) if mixer == "qmix" else None
        self.target_mixer = copy.deepcopy(self.mixer).eval() if self.mixer is not None else None
        params = list(online.parameters()) + ([] if self.mixer is None else list(self.mixer.parameters()))
        self.optim = torch.optim.Adam(params, lr=lr)
        self.target_update, self.gamma, self.per_eps = int(target_update), float(gamma), float(per_eps)
        self.device, self.updates = torch.device(device), 0

    def _total(self, network, mixer, q, roles, maps, previous_roles, budget, preference):
        selected = q.gather(-1, roles.unsqueeze(-1)).squeeze(-1)
        if mixer is None:
            return selected.sum(dim=1)
        context = q_context(network, maps, previous_roles, budget, preference)
        return mixer(selected, context)

    def update(self, replay, batch_size: int, beta: float):
        if replay.size < batch_size:
            return None
        data = replay.sample(batch_size, beta, self.device)
        q = self.online(data["maps"], data["previous_roles"], data["budget"], data["preference"])
        current = self._total(self.online, self.mixer, q, data["roles"], data["maps"], data["previous_roles"], data["budget"], data["preference"])
        with torch.no_grad():
            q_next_online = self.online(data["next_maps"], data["next_previous_roles"], data["next_budget"], data["preference"])
            next_roles = q_next_online.argmax(dim=-1)
            q_next_target = self.target(data["next_maps"], data["next_previous_roles"], data["next_budget"], data["preference"])
            next_total = self._total(self.target, self.target_mixer, q_next_target, next_roles, data["next_maps"], data["next_previous_roles"], data["next_budget"], data["preference"])
            target = data["reward"] + (self.gamma ** data["duration"]) * (1.0 - data["done"]) * next_total
        td = target - current
        loss = (data["weights"] * F.smooth_l1_loss(current, target, reduction="none")).mean()
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optim.step()
        replay.update_priorities(data["indices"], td.detach().abs().cpu().numpy() + self.per_eps)
        self.updates += 1
        if self.updates % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
            if self.mixer is not None:
                self.target_mixer.load_state_dict(self.mixer.state_dict())
        return float(loss.item()), float(td.abs().mean().item())
