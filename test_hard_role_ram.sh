#!/usr/bin/env bash
set -euo pipefail

# Exhaustive CTDE hard-role RAM test runner.
#
# Fast/self-contained suite:
#   ./test_hard_role_ram.sh
#
# Include a real project-environment smoke run:
#   PATH_PLANNER_FOLDER=<folder> ./test_hard_role_ram.sh --project
#
# Optional output directory:
#   TEST_OUTPUT_DIR=/tmp/my_ctde_tests ./test_hard_role_ram.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

OUT="${TEST_OUTPUT_DIR:-/tmp/ctde_hard_role_tests}"
mkdir -p "${OUT}"

echo "======================================================================"
echo "CTDE-RAM hard-role exhaustive tests"
echo "repository : ${ROOT}"
echo "artifacts  : ${OUT}"
echo "======================================================================"

echo "[1/6] Static syntax and CLI checks"
python -m py_compile \
  Learning/ctde_ram/hard_role_ram.py \
  Learning/ctde_ram/replay_buffers.py \
  Learning/ctde_ram/project_env.py \
  Learning/ctde_ram/trainer.py \
  Learning/ctde_ram/run_experiment.py \
  Learning/ctde_ram/run_evaluation.py
bash -n run_hard_role_methods.sh
python Learning/ctde_ram/run_experiment.py --help > "${OUT}/run_experiment_help.txt"
for flag in \
  ppo_ram hard_role_q d-model n-attn-heads n-attn-layers attn-ff-dim \
  preference-role-bias ppo-epochs ppo-minibatch-size ppo-clip-eps \
  ppo-rollout-macro-steps ppo-target-kl ppo-max-grad-norm \
  ppo-preference-sampling weight-alpha-end weight-alpha-anneal-fraction \
  gae-lambda entropy-coef value-coef actor-lr critic-lr role-q-lr \
  role-q-target-update role-q-mixer role-q-gamma role-q-epsilon-start \
  role-q-epsilon-end role-q-epsilon-fraction per-alpha per-beta-start \
  per-beta-end per-eps; do
  grep -q -- "${flag}" "${OUT}/run_experiment_help.txt"
done

echo "[2/6] Network, attention, action, buffer, PPO, PER and QMIX tests"
OUT="${OUT}" python - <<'PY'
import os
import numpy as np
import torch

from Learning.ctde_ram.hard_role_ram import (
    PPOActor, PPOCritic, HardRoleQNetwork, PPORAMLearner, HardRoleQLearner,
)
from Learning.ctde_ram.replay_buffers import (
    PPORoleRolloutBuffer, PrioritizedHardRoleReplayBuffer,
)

torch.manual_seed(7)
np.random.seed(7)
B, N, H, W = 3, 4, 16, 20
kwargs = dict(d_model=64, n_heads=4, n_layers=2, ff_dim=128, preference_role_bias=False)
maps = torch.randn(B, N, 3, H, W)
previous = torch.randint(0, 2, (B, N))
budget = torch.rand(B, 1)
preference = torch.tensor([[1., 0.], [.5, .5], [0., 1.]])

actor = PPOActor(**kwargs)
critic = PPOCritic(**kwargs)
qnet = HardRoleQNetwork(**kwargs)

roles, joint_logp, probabilities, attention = actor.act(
    maps, previous, budget, preference, deterministic=False, return_attn=True
)
assert roles.shape == (B, N) and roles.dtype == torch.long
assert joint_logp.shape == (B,)
assert probabilities.shape == (B, N, 2)
assert torch.allclose(probabilities.sum(-1), torch.ones(B, N), atol=1e-6)
assert len(attention) == 2
assert all(x.shape == (B, 4, N, N) for x in attention)
assert critic(maps, previous, budget, preference).shape == (B,)
assert qnet(maps, previous, budget, preference).shape == (B, N, 2)
det_roles, _, _, _ = actor.act(maps, previous, budget, preference, deterministic=True)
assert torch.equal(det_roles, actor(maps, previous, budget, preference).argmax(-1))

# Preference enters only through FiLM by default: no logit/Q bias module.
assert actor.trunk.bias_mlp is None and qnet.trunk.bias_mlp is None
biased = PPOActor(**(kwargs | {"preference_role_bias": True}))
assert biased.trunk.bias_mlp is not None

# Reject anything other than [B,N,3,H,W].
try:
    actor(torch.randn(B, N, 2, H, W), previous, budget, preference)
    raise AssertionError("two-channel input was incorrectly accepted")
except ValueError:
    pass

# Permutation-equivariant shared network: identical agent inputs must produce
# identical logits. Individual budgets must be able to break that symmetry.
sym_actor = PPOActor(**kwargs).eval()
same_map = torch.randn(1, 1, 3, H, W).expand(1, N, -1, -1, -1).clone()
same_role = torch.zeros(1, N, dtype=torch.long)
same_budget = torch.full((1, N, 1), .5)
same_pref = torch.tensor([[.5, .5]])
same_logits = sym_actor(same_map, same_role, same_budget, same_pref)
assert torch.allclose(same_logits, same_logits[:, :1].expand_as(same_logits), atol=1e-6)
different_budget = torch.linspace(.1, .9, N).view(1, N, 1)
different_logits = sym_actor(same_map, same_role, different_budget, same_pref)
assert not torch.allclose(different_logits, different_logits[:, :1].expand_as(different_logits))

# FiLM sanity check after it has nonzero learned parameters: the same state must
# change under opposite preferences.
film_actor = PPOActor(**kwargs).eval()
with torch.no_grad():
    film_actor.trunk.preference_film.net[-1].weight.normal_(0.0, 0.05)
clean_logits = film_actor(same_map, same_role, same_budget, torch.tensor([[1., 0.]]))
explore_logits = film_actor(same_map, same_role, same_budget, torch.tensor([[0., 1.]]))
assert not torch.allclose(clean_logits, explore_logits)

# PPO rollout stores exact integer roles and performs a finite update.
rollout = PPORoleRolloutBuffer()
small_kwargs = dict(d_model=16, n_heads=4, n_layers=1, ff_dim=32, preference_role_bias=False)
small_actor, small_critic = PPOActor(**small_kwargs), PPOCritic(**small_kwargs)
ppo = PPORAMLearner(
    small_actor, small_critic, epochs=2, minibatch_size=2,
    gamma=.97, gae_lambda=.91,
)
for t, duration in enumerate((3, 2, 1, 4)):
    x = np.random.randn(N, 3, 8, 8).astype(np.float32)
    nx = np.random.randn(N, 3, 8, 8).astype(np.float32)
    prev = np.random.randint(0, 2, N, dtype=np.int64)
    pref = np.asarray([.4, .6], dtype=np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(x).unsqueeze(0)
        pt = torch.from_numpy(prev).unsqueeze(0)
        bt = torch.tensor([[.8]])
        wt = torch.from_numpy(pref).unsqueeze(0)
        action, logp, _, _ = small_actor.act(xt, pt, bt, wt)
        value = small_critic(xt, pt, bt, wt)
    rollout.store(
        maps=x, preference=pref, previous_roles=prev,
        roles=action[0].numpy(), old_logprob=logp.item(), value=value.item(),
        reward=float(t + 1), done=float(t == 3), duration=float(duration),
        budget=[.8], next_maps=nx, next_previous_roles=action[0].numpy(),
        next_budget=[.7],
    )
stored = rollout.as_tensors()
assert stored["roles"].dtype == torch.long and stored["roles"].shape == (4, N)
assert stored["duration"].tolist() == [3., 2., 1., 4.]
ppo_metrics = ppo.update(rollout)
assert ppo_metrics is not None and np.isfinite(ppo_metrics.loss)
assert len(rollout) == 0

# PER probabilities/importance weights, priority updates and exact hard actions.
replay = PrioritizedHardRoleReplayBuffer(32, N, (3, 8, 8), alpha=.6, eps=1e-5)
for t in range(12):
    hard_roles = np.asarray([(t + i) % 2 for i in range(N)], dtype=np.int64)
    replay.store(
        np.random.randn(N, 3, 8, 8), [.3, .7], np.zeros(N, dtype=np.int64),
        hard_roles, float(t), np.random.randn(N, 3, 8, 8), hard_roles,
        t == 11, (t % 4) + 1, [.9], [.8],
    )
sample = replay.sample(8, beta=.4)
assert sample["roles"].dtype == torch.long and sample["roles"].shape == (8, N)
assert sample["weights"].min() > 0 and sample["weights"].max() <= 1.0 + 1e-6
old = replay.priorities.copy()
replay.update_priorities([0, 1], [.25, 2.0])
assert replay.priorities[0] == .25 and replay.priorities[1] == 2.0
assert not np.array_equal(old[:2], replay.priorities[:2])

# Double-DQN updates for baseline sum and optional QMIX.
for mixer in ("none", "qmix"):
    network = HardRoleQNetwork(**small_kwargs)
    learner = HardRoleQLearner(
        network, N, lr=1e-3, target_update=1, mixer=mixer,
        gamma=.93, per_eps=1e-5,
    )
    result = learner.update(replay, batch_size=8, beta=.7)
    assert result is not None and all(np.isfinite(x) for x in result)
    for p, target_p in zip(learner.online.parameters(), learner.target.parameters()):
        assert torch.equal(p, target_p), "target_update=1 did not synchronize"
    if mixer == "qmix":
        assert learner.mixer is not None and learner.target_mixer is not None
    else:
        assert learner.mixer is None

print("network/buffer/learner tests: PASS")
PY

echo "[3/6] End-to-end trainer macro-step tests for PPO and HardRoleQ"
OUT="${OUT}" python - <<'PY'
import os
import numpy as np
from Learning.ctde_ram.trainer import CTDERAMTrainer

OUT = os.environ["OUT"]

class FakeExpertNuEnv:
    ctde_action_mode = "role_weights"
    N, K, A = 3, 2, 8
    obs_shape = (3, 20, 20)
    obs_dim = obs_shape
    action_space_n = 16

    def reset(self, seed=None):
        self.t = 0
        self.executed = []
        return self._obs()

    def _obs(self):
        return [np.full(self.obs_shape, (self.t + i) / 20, dtype=np.float32) for i in range(self.N)]

    def step(self, roles):
        roles = np.asarray(roles)
        assert roles.shape == (self.N,)
        assert roles.dtype.kind in "iu"
        assert np.all((roles == 0) | (roles == 1))
        self.executed.append(roles.copy())
        self.t += 1
        rewards = [np.asarray([1.0 - .2 * role, .2 + .8 * role], dtype=np.float32) for role in roles]
        return self._obs(), rewards, self.t >= 7, {"roles": roles.copy()}

    def budget_frac(self): return 1.0 - self.t / 7.0
    def budget_fracs(self):
        base = 1.0 - self.t / 7.0
        return np.clip(np.asarray([base, base - .1, base - .2], dtype=np.float32), 0, 1)
    def coverage_pct(self): return self.t / 7.0
    def trash_density(self): return .5
    def trash_cleaned_pct(self): return self.t / 14.0

def make(mode, mixer="none"):
    env = FakeExpertNuEnv()
    trainer = CTDERAMTrainer(
        obs_dim=env.obs_shape, obs_shape=env.obs_shape, N=env.N, K=2, A=8,
        low_level_backend="dueling_nu", ram_mode=mode, role_q_mixer=mixer,
        freeze_low_level=True, T_role=3, gamma=.9, batch_role=2,
        ppo_epochs=1, ppo_minibatch_size=2, ppo_rollout_macro_steps=1,
        tb_logdir=OUT, tb_runname=f"tb_{mode}_{mixer}", seed=11,
    )
    return env, trainer

for mode, mixer in (("ppo_ram", "none"), ("hard_role_q", "none"), ("hard_role_q", "qmix")):
    env, trainer = make(mode, mixer)
    metrics = trainer.run_episode(env, [.5, .5], epsilon_low=0.0, epsilon_ram=.3)
    assert metrics["n_steps"] == 7
    assert metrics["role_decisions"] == 3  # durations 3, 3, 1
    assert len(env.executed) == 7
    assert metrics["ram_losses"]
    if mode == "ppo_ram":
        assert trainer.ppo_rollout is not None and len(trainer.ppo_rollout) == 0
    else:
        assert trainer.hard_role_replay.size == 3
        np.testing.assert_array_equal(
            trainer.hard_role_replay.roles[:3],
            np.asarray([env.executed[0], env.executed[3], env.executed[6]]),
        )
        np.testing.assert_array_equal(trainer.hard_role_replay.durations[:3], [3., 3., 1.])
        assert trainer.hard_role_replay.budget.shape[1:] == (env.N, 1)
        assert not np.allclose(trainer.hard_role_replay.budget[0, 0], trainer.hard_role_replay.budget[0, 2])

    checkpoint = os.path.join(OUT, f"{mode}_{mixer}.pt")
    trainer.save_checkpoint(checkpoint)
    rows = trainer.probe_preference_sensitivity(
        env,
        scal_grid=[(1., 0.), (.8, .2), (.5, .5), (.2, .8), (0., 1.)],
        n_episodes_per_w=1,
        save_csv=os.path.join(OUT, f"probe_{mode}_{mixer}.csv"),
        plot_path=os.path.join(OUT, f"probe_{mode}_{mixer}.png"),
        pareto_plot_path=os.path.join(OUT, f"probe_{mode}_{mixer}.pareto.png"),
        attention_path=os.path.join(OUT, f"attention_{mode}_{mixer}.npz"),
    )
    assert len(rows) == 5
    required = {
        "role0_argmax_frac", "role1_argmax_frac", "mean_clean_logit_or_q",
        "mean_explore_logit_or_q", "coverage", "trash_cleaned",
        "switch_rate", "episode_return",
    }
    assert required <= set(rows[0])
    attention = np.load(os.path.join(OUT, f"attention_{mode}_{mixer}.npz"))
    assert attention["attention"].ndim == 5  # records,layers,heads,N,N
    assert attention["roles"].shape[-1] == env.N
    assert os.path.exists(os.path.join(OUT, f"probe_{mode}_{mixer}.png"))
    assert os.path.exists(os.path.join(OUT, f"probe_{mode}_{mixer}.pareto.png"))
    preference_check = trainer.diagnose_same_state_preferences(env, seed=0)
    assert np.isfinite(preference_check["mean_abs_score_difference"])
    forced = trainer.evaluate_forced_role_policies(env, n_episodes=1, seed_base=0)
    assert [row["forced_policy"] for row in forced] == [
        "all_clean", "all_explore", "split", "alternating", "random"
    ]

    _, restored = make(mode, mixer)
    restored.load_checkpoint(checkpoint)
    if mode == "hard_role_q":
        assert restored.hard_role_replay.size == trainer.hard_role_replay.size
    restored.close()
    trainer.close()

print("trainer/macro/checkpoint/probe tests: PASS")
PY

echo "[4/6] Expert_nu adapter hard-role determinism test"
python - <<'PY'
import numpy as np
from Learning.ctde_ram.project_env import ProjectPatrollingExpertNuCTDEEnv

# Test the adapter helpers without constructing the heavyweight project env.
adapter = object.__new__(ProjectPatrollingExpertNuCTDEEnv)
adapter.N = 4
adapter.rng = np.random.default_rng(0)
roles = np.asarray([0, 1, 1, 0], dtype=np.int64)
condition = adapter._hard_roles_to_condition(roles)
np.testing.assert_array_equal(condition, [False, True, True, False])
np.testing.assert_array_equal(adapter._role_weights_to_nu(roles), roles.astype(np.float32))
for invalid in (np.asarray([0, 2, 1, 0]), np.asarray([0, 1])):
    try:
        adapter._hard_roles_to_condition(invalid)
        raise AssertionError("invalid roles were accepted")
    except ValueError:
        pass
print("Expert_nu hard-role adapter tests: PASS")
PY

echo "[5/6] Regression guard: new methods reject non-map observations"
python - <<'PY'
import torch
from Learning.ctde_ram.trainer import CTDERAMTrainer

try:
    CTDERAMTrainer(obs_dim=10, N=2, K=2, A=5, ram_mode="ppo_ram")
    raise AssertionError("PPO-RAM accepted flat observations")
except ValueError as exc:
    assert "[3,H,W]" in str(exc)

# Default min-max normalization must happen before EWC. Even very large raw
# components must remain on the normalized EWC scale instead of exp(20).
trainer = object.__new__(CTDERAMTrainer)
trainer.K = 2
trainer.device = torch.device("cpu")
trainer.scalarization_eps = 1e-8
trainer.scalarization_power = 3.0
trainer.ewc_p = 1.0
trainer.role_scalarization = "ewc"
trainer.ram_reward_mode = "component_rewards"
trainer.role_reward_norm_name = "minmax"
from Learning.ctde_ram.trainer import RewardMinMaxNormalizer
trainer.role_reward_norm = RewardMinMaxNormalizer(2)
trainer.role_reward_norm.update(torch.tensor([0.0, 0.0]))
value = trainer._scalarize_role_reward(
    torch.tensor([1000.0, 5000.0]), torch.tensor([0.5, 0.5]), update_norm=True
)
assert torch.isfinite(value)
assert value.item() < 10.0, value.item()
print("input-contract regression test: PASS")
PY

python - <<'PY'
from Learning.ctde_ram.run_experiment import annealed_weight_alpha
assert annealed_weight_alpha(.4, 1.0, 0, 10_000) == .4
assert abs(annealed_weight_alpha(.4, 1.0, 5_000, 10_000) - .700030003) < 1e-8
assert annealed_weight_alpha(.4, 1.0, 9_999, 10_000) == 1.0
assert annealed_weight_alpha(.4, 1.0, 8_000, 10_000, .5) == 1.0
assert annealed_weight_alpha(.4, None, 8_000, 10_000) == .4
print("weight-alpha annealing tests: PASS")
PY

echo "[6/6] Optional real project smoke test"
if [[ "${1:-}" == "--project" ]]; then
  : "${PATH_PLANNER_FOLDER:?Set PATH_PLANNER_FOLDER for --project}"
  EPISODES="${PROJECT_TEST_EPISODES:-1}"
  for method in ppo hardq hardq_qmix; do
    PATH_PLANNER_FOLDER="${PATH_PLANNER_FOLDER}" \
    python -m Learning.ctde_ram.run_experiment \
      --env project \
      --project-control expert_nu \
      --path-planner-folder "${PATH_PLANNER_FOLDER}" \
      --episodes "${EPISODES}" \
      --eval-every 0 \
      --save-every 0 \
      --run-name "test_${method}" \
      --output-dir "${OUT}/project" \
      --freeze \
      $(case "${method}" in
          ppo) echo "--ram-mode ppo_ram --ppo-epochs 1" ;;
          hardq) echo "--ram-mode hard_role_q --role-q-mixer none" ;;
          hardq_qmix) echo "--ram-mode hard_role_q --role-q-mixer qmix" ;;
        esac)
  done
else
  echo "SKIP: pass --project and set PATH_PLANNER_FOLDER to run real environment tests"
fi

echo "======================================================================"
echo "ALL REQUESTED HARD-ROLE RAM TESTS PASSED"
echo "Artifacts: ${OUT}"
echo "======================================================================"
