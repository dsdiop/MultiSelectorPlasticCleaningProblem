"""
RoleSelectorAttention: the scalable soft RAM used by CTDE-RAM V2/V3.

Hierarchical DRL meaning:
  - Input: per-agent low-level encodings z_i plus fleet context g.
  - Output before softmax: role Q/logit values for every (agent, role).
  - Output after softmax: W[i, k], the soft role/preference weight used to mix
    PopArt-normalized low-level Q-heads.

Why this exists:
  Elicit's final recommendation says the strong version should execute a soft
  role-weight matrix W, not only hard one-hot roles. It also says the scalable
  variant should use an attention-style role selector whose parameters do not
  grow with K**N. This class is that path.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMConditioner(nn.Module):
    def __init__(self, preference_dim: int, feature_dim: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.net = nn.Sequential(
            nn.Linear(preference_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 2 * feature_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[:feature_dim].fill_(1.0)

    def forward(self, preference):
        gamma, beta = self.net(preference).split(self.feature_dim, dim=-1)
        return gamma, beta


class DuelingRAMHead(nn.Module):
    def __init__(self, input_dim: int, N: int, K: int, discrete_actions: int = 0):
        super().__init__()
        self.N = int(N)
        self.K = int(K)
        self.discrete_actions = int(discrete_actions)
        if self.discrete_actions:
            self.value = nn.Linear(input_dim, 1)
            self.advantage = nn.Linear(input_dim, self.discrete_actions)
        else:
            self.value = nn.Linear(input_dim, self.N)
            self.advantage = nn.Linear(input_dim, self.N * self.K)

    def forward(self, hidden):
        advantage = self.advantage(hidden)
        if self.discrete_actions:
            return self.value(hidden) + advantage - advantage.mean(dim=-1, keepdim=True)
        advantage = advantage.view(-1, self.N, self.K)
        value = self.value(hidden).unsqueeze(-1)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q.flatten(start_dim=1)


class RoleSelectorAttention(nn.Module):
    def __init__(
        self, d_enc=128, d_ctx=256, K=2, d_role=64, d_extra=0,
        w_conditioning="concat",
    ):
        super().__init__()
        self.K = K
        self.scale = 1.0 / math.sqrt(d_role)
        self.role_keys = nn.Parameter(torch.randn(K, d_role) * self.scale)
        self.d_extra = int(d_extra)
        self.query_proj = nn.Linear(d_enc + d_ctx + self.d_extra, d_role)
        self.w_conditioning = w_conditioning
        self.film = FiLMConditioner(K, d_role) if w_conditioning == "film" else None

    def logits(self, z_all, g, extra=None):
        # z_all: (B, N, d_enc)
        # g:     (B, d_ctx)
        # extra: (B, d_extra), mission/preference features shared by all agents.
        # return: (B, N, K), one role value/logit per agent and role.
        B, N, _ = z_all.shape
        g_tiled = g.unsqueeze(1).expand(B, N, -1)
        parts = [z_all, g_tiled]
        if self.d_extra:
            if extra is None:
                extra = torch.zeros(B, self.d_extra, dtype=z_all.dtype, device=z_all.device)
            extra_tiled = extra.unsqueeze(1).expand(B, N, -1)
            parts.append(extra_tiled)
        x = torch.cat(parts, dim=-1)
        q = self.query_proj(x)
        if self.film is not None:
            gamma, beta = self.film(extra[:, -self.K:])
            q = gamma.unsqueeze(1) * q + beta.unsqueeze(1)
        return q @ self.role_keys.T * self.scale

    def forward(self, z_all, g, extra=None, temperature=1.0):
        # Soft RAM execution path:
        # W[i, k] is the probability/weight assigned to role k for agent i.
        scores = self.logits(z_all, g, extra=extra)
        return F.softmax(scores / max(float(temperature), 1e-6), dim=-1)
