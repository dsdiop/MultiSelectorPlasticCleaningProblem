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
