"""Hard-role high-level RAM networks and learners.

The two supported methods share the exact token architecture:
3-channel allocentric map CNN + previous-role embedding + budget projection,
preference FiLM, and Transformer-style fleet self-attention.
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
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.map_cnn = AllocentricMapCNN(d_model)
        self.previous_role_embedding = nn.Embedding(2, d_model)
        self.budget_projection = nn.Linear(1, d_model)
        self.preference_film = PreferenceFiLM(2, d_model)
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
        if budget.ndim == 1:
            budget = budget.unsqueeze(-1)
        budget_z = self.budget_projection(budget).unsqueeze(1)
        token = z + role_z + budget_z
        gamma, beta = self.preference_film(preference)
        token = gamma.unsqueeze(1) * token + beta.unsqueeze(1)

        attention = []
        for layer in self.layers:
            if return_attn:
                token, weights = layer(token, return_attn=True)
                attention.append(weights)
            else:
                token = layer(token)
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
    def __init__(self, **trunk_kwargs):
        super().__init__()
        self.trunk = HardRoleAttentionTrunk(**trunk_kwargs)
        self.value_head = nn.Linear(self.trunk.d_model, 1)

    def forward(self, maps, previous_roles, budget, preference):
        h, _ = self.trunk(maps, previous_roles, budget, preference)
        return self.value_head(h.mean(dim=1)).squeeze(-1)


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
        max_grad_norm=0.5, device="cpu",
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

    def update(self, rollout):
        if len(rollout) == 0:
            return None
        data = rollout.as_tensors(self.device)
        with torch.no_grad():
            next_values = self.critic(data["next_maps"], data["next_previous_roles"], data["next_budget"], data["preference"])
            discounts = self.gamma ** data["duration"]
            deltas = data["reward"] + discounts * next_values * (1.0 - data["done"]) - data["value"]
            advantages = torch.zeros_like(deltas)
            gae = torch.zeros((), device=self.device)
            for t in range(len(deltas) - 1, -1, -1):
                gae = deltas[t] + discounts[t] * self.gae_lambda * (1.0 - data["done"][t]) * gae
                advantages[t] = gae
            returns = advantages + data["value"]
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

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
                value_loss = F.mse_loss(value, returns[idx])

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
