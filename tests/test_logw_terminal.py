import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Learning.ctde_ram.hard_role_ram import PPOActor, PPOCritic, PPORAMLearner
from Learning.ctde_ram.replay_buffers import PPORoleRolloutBuffer


def test_logw_terminal_materializes_weighted_log_without_touching_reference():
    actor = PPOActor(d_model=4, n_heads=1, n_layers=1, ff_dim=8)
    critic = PPOCritic(d_model=4, n_heads=1, n_layers=1, ff_dim=8)
    learner = PPORAMLearner(
        actor,
        critic,
        epochs=0,
        ram_reward_mode="logw_terminal",
        ref_clean=0.1,
        ref_cov=0.2,
        ref_autocalibrate=True,
        wpop_eps=0.01,
    )
    rollout = PPORoleRolloutBuffer()
    maps = np.zeros((2, 3, 4, 4), dtype=np.float32)
    roles = np.zeros(2, dtype=np.int64)
    budget = np.ones(2, dtype=np.float32)
    preference = np.array([0.25, 0.75], dtype=np.float32)
    rollout.store(
        maps=maps,
        preference=preference,
        previous_roles=roles,
        roles=roles,
        old_logprob=0.0,
        value=0.0,
        reward=123.0,
        done=1.0,
        duration=1.0,
        budget=budget,
        next_maps=maps,
        next_previous_roles=roles,
        next_budget=budget,
        terminal_clean=0.4,
        terminal_cov=0.8,
    )

    learner.update(rollout)

    expected = 0.25 * np.log(0.4) + 0.75 * np.log(0.8)
    assert learner.last_terminal_utilities == pytest.approx([expected])
    assert learner.last_utility_ref == (0.1, 0.2)


def test_logw_terminal_clamps_zero_objectives():
    actor = PPOActor(d_model=4, n_heads=1, n_layers=1, ff_dim=8)
    critic = PPOCritic(d_model=4, n_heads=1, n_layers=1, ff_dim=8)
    learner = PPORAMLearner(
        actor, critic, epochs=0, ram_reward_mode="logw_terminal", wpop_eps=0.01
    )
    rollout = PPORoleRolloutBuffer()
    maps = np.zeros((1, 3, 4, 4), dtype=np.float32)
    roles = np.zeros(1, dtype=np.int64)
    rollout.store(
        maps=maps, preference=np.array([0.5, 0.5], dtype=np.float32),
        previous_roles=roles, roles=roles, old_logprob=0.0, value=0.0,
        reward=0.0, done=1.0, duration=1.0, budget=np.ones(1, dtype=np.float32),
        next_maps=maps, next_previous_roles=roles,
        next_budget=np.ones(1, dtype=np.float32), terminal_clean=0.0, terminal_cov=1.0,
    )

    learner.update(rollout)

    assert learner.last_terminal_utilities == pytest.approx([0.5 * np.log(0.01)])
