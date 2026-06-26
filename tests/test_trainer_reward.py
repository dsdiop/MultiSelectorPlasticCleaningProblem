import os
import sys
import unittest

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Learning", "ctde_ram_claude"))

from trainer import CTDERAMTrainer, RewardMinMaxNormalizer


class DeltaMetricRewardTest(unittest.TestCase):
    def test_minmax_normalizer_handles_zero_span(self):
        norm = RewardMinMaxNormalizer(2)
        norm.update(torch.tensor([[0.1, 0.2]], dtype=torch.float32))
        out = norm.normalize_tensor(torch.tensor([[0.1, 0.2]], dtype=torch.float32))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))

    def test_delta_metric_reward_uses_window_deltas(self):
        trainer = CTDERAMTrainer(
            obs_dim=4,
            N=2,
            K=2,
            A=4,
            device="cpu",
            seed=0,
            low_level_backend="mlp",
            obs_shape=(4,),
            action_space_n=4,
            movement_actions=4,
            freeze_low_level=True,
            ram_mode="soft_v2",
            role_state_mode="pooled",
            normalize_role_rewards=False,
            role_reward_norm="none",
            role_scalarization="ws",
            q_scalarization="ws",
            ram_reward_mode="delta_metrics",
        )

        prev_metrics = {"trash_cleaned": 0.2, "coverage": 0.1}
        reward_components = trainer._compute_step_reward_components(
            env=None,
            prev_metrics=prev_metrics,
            r_vecs=None,
            info={"trash_cleaned": 0.5, "coverage": 0.4},
        )

        expected = torch.tensor([0.3, 0.3], dtype=torch.float32)
        self.assertTrue(torch.allclose(reward_components, expected))


if __name__ == "__main__":
    unittest.main()
