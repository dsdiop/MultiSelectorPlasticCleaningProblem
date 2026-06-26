"""
PopArt: per-head value normalization so K heads with different reward scales become
comparable before the RAM mixes them.

Reference: van Hasselt et al. 2016; Hessel et al. 2019.

LESSON BAKED IN (frozen heads):
  The original PopArt rescales the head's output layer every update. If the head is
  FROZEN (your pretrained contribution), we must NOT mutate its weights. So when
  `rescale=False` we only track running (mu, sigma) and use them to normalize -- the
  head stays untouched. This is the PopArt analogue of the greedy paper's fixed
  reward_min/reward_max normalization.
"""
import math
import torch
import torch.nn as nn


class PopArtNorm:
    def __init__(self, head_out: nn.Linear, alpha: float = 1e-3, rescale: bool = True):
        # head_out : the role head output adapter. For the toy MLP this is the final
        #            nn.Linear. For your DuelingDQN it is a small adapter that rescales
        #            the value and advantage outputs together.
        # alpha    : EMA decay for the running stats.
        # rescale  : if True, do the PopArt preserve-outputs transform on trainable heads.
        #            If False, only track stats (use this for frozen pretrained heads).
        self.head_out = head_out
        self.alpha = alpha
        self.rescale = rescale
        self.mu = 0.0
        self.mu_sq = 0.0
        self.sigma = 1.0

    def normalize(self, q_raw: torch.Tensor) -> torch.Tensor:
        return (q_raw - self.mu) / (self.sigma + 1e-8)

    def denormalize(self, q_norm: torch.Tensor) -> torch.Tensor:
        return q_norm * self.sigma + self.mu

    def normalize_target(self, td_target_raw: torch.Tensor) -> torch.Tensor:
        return (td_target_raw - self.mu) / (self.sigma + 1e-8)

    @torch.no_grad()
    def update(self, targets_raw: torch.Tensor) -> None:
        # Call once per batch with values in the ORIGINAL scale (TD targets, or Q
        # samples during a frozen-head warmup).
        batch_mean = float(targets_raw.mean().item())
        batch_sq = float((targets_raw ** 2).mean().item())
        mu_old, sigma_old = self.mu, self.sigma

        self.mu = (1 - self.alpha) * self.mu + self.alpha * batch_mean
        self.mu_sq = (1 - self.alpha) * self.mu_sq + self.alpha * batch_sq
        var = max(self.mu_sq - self.mu ** 2, 1e-4)
        self.sigma = math.sqrt(var)

        if self.rescale:
            # Preserve-outputs step from PopArt. If the output object knows how to
            # rescale itself (the dueling value/advantage case), delegate to it.
            if hasattr(self.head_out, "popart_rescale"):
                self.head_out.popart_rescale(mu_old, sigma_old, self.mu, self.sigma)
                return

            scale = sigma_old / self.sigma
            self.head_out.weight.data.mul_(scale)
            self.head_out.bias.data.copy_(
                (sigma_old * self.head_out.bias.data + (mu_old - self.mu)) / self.sigma
            )
