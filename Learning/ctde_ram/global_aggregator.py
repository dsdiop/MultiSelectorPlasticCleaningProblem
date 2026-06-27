"""
GlobalAggregator: reduce the N agent encodings into one fleet context g.

Two modes are intentionally exposed for RQ3-style ablations:
  - attention: Elicit-style self-attention over agents, then mean-pool.
  - mean_pool: no relational attention, just mean-pool the agent encodings.

Both modes are permutation-invariant after the pool. The difference is whether each
agent encoding is first allowed to read the other agents through the transformer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAggregator(nn.Module):
    def __init__(self, d_enc=128, d_ctx=256, n_heads=4, n_layers=2, mode: str = "attention"):
        super().__init__()
        if mode not in {"attention", "mean_pool"}:
            raise ValueError("GlobalAggregator mode must be one of: attention, mean_pool")
        self.mode = mode

        # attention is the full Elicit relational aggregator. mean_pool is the
        # ablation: the projection/norm still learn a fleet context, but there is
        # no pairwise communication before pooling.
        if self.mode == "attention":
            layer = nn.TransformerEncoderLayer(
                d_model=d_enc,
                nhead=n_heads,
                dim_feedforward=d_enc * 2,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            self.transformer = None
        self.proj = nn.Linear(d_enc, d_ctx)
        self.norm = nn.LayerNorm(d_ctx)

    def forward(self, z_all):
        # z_all: (B, N, d_enc) -> g: (B, d_ctx)
        z = self.transformer(z_all) if self.transformer is not None else z_all
        z = z.mean(dim=1)
        g = self.norm(self.proj(z))
        return g


class MonotonicQMixer(nn.Module):
    """QMIX-style monotonic combination of selected per-agent role values."""

    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.n_agents = int(n_agents)
        self.embed_dim = int(embed_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, self.n_agents * self.embed_dim)
        )
        self.hyper_b1 = nn.Linear(state_dim, self.embed_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, self.embed_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def mixing_weights(self, state):
        batch = state.shape[0]
        w1 = self.hyper_w1(state).abs().view(batch, self.n_agents, self.embed_dim)
        w2 = self.hyper_w2(state).abs().view(batch, self.embed_dim, 1)
        return w1, w2

    def forward(self, agent_values, state):
        w1, w2 = self.mixing_weights(state)
        b1 = self.hyper_b1(state).unsqueeze(1)
        hidden = F.elu(torch.bmm(agent_values.unsqueeze(1), w1) + b1)
        return (torch.bmm(hidden, w2) + self.value(state).unsqueeze(1)).view(-1)
