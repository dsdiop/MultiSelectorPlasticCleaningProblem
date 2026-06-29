# CTDE-RAM for MultiSelector Plastic Cleaning

> **Current main methodology:** `PPO_RAM_FiLM_Attn`,
> `HardRoleQ_RAM_FiLM_Attn`, and optional-QMIX
> `HardRoleQ_RAM_FiLM_Attn_QMIX`. These methods emit and replay one hard
> clean/explore role per ASV. The older `soft_v2`/SAttn variants below are
> retained only as **legacy soft-learning / hard-execution RAM** and are not
> part of the main experiment table.

```bash
python -m Learning.ctde_ram.run_experiment --env project --project-control expert_nu \
  --path-planner-folder <folder> --ram-mode ppo_ram
python -m Learning.ctde_ram.run_experiment --env project --project-control expert_nu \
  --path-planner-folder <folder> --ram-mode hard_role_q --role-q-mixer none
python -m Learning.ctde_ram.run_experiment --env project --project-control expert_nu \
  --path-planner-folder <folder> --ram-mode hard_role_q --role-q-mixer qmix
```

The remaining text documents historical compatibility modes as well as current utilities.

This package implements two main two-timescale hard-role CTDE methods. Historical
soft-weight modes remain available only for reproducibility:

- **High level RAM**: chooses one hard role per ASV every `T_role` steps.
- **Low level controller**: receives that role as its branch condition, then chooses movement.
- **Role semantics in this project**:
  - `role 0 = nu=0 = cleaning / intensification`
  - `role 1 = nu=1 = exploration / coverage`

The toy environment is still available for smoke tests. The project path uses your real
`DQFDuelingVisualNetwork`: one shared visual encoder with two dueling heads.

## Files

- `trainer.py`: the orchestration class. This is where RAM version switches, reward normalization, replay, target updates, and training loops live.
- `role_selector.py`: the scalable attention soft RAM used by `--ram-mode soft_v2 --soft-ram-arch attention`.
- `nets.py`: toy MLP backend plus `DuelingNuQNetwork`, the wrapper for your real two-head DQN.
- `project_env.py`: adapter from `MultiAgentPatrolling` to the CTDE-RAM trainer contract.
- `replay_buffers.py`: low-level replay and role replay. The role buffer stores replayed `z_all`, mission `extra`, and executed `W(N,K)` so `GlobalAggregator(z_all)` is recomputed during RAM updates and still receives gradient.
- `popart.py`: PopArt normalization, including a dueling-head preserve-output adapter.
- `global_aggregator.py`: fleet context with `attention` or `mean_pool` mode.
- `pareto.py`: 2D Pareto front and hypervolume over coverage/cleaned metrics.
- `run_experiment.py`: CLI entry point.
- `run_evaluation.py`: offline evaluator for checkpoints saved by `run_experiment.py`.
- `experiment_io.py`: JSON/CSV/checkpoint/figure helpers used by train and eval.

## RAM Modes

Use `--ram-mode` to choose which high-level RAM is active.

### `--ram-mode random`

This is the random RAM control for RQ1.

- RAM action space: no learned action-value network.
- Execution: uniformly random hard one-hot `W`; each agent samples one role independently.
- Training: RAM update is skipped, so there is no high-level learner.
- Best for: proving the learned RAM beats a selector-free control on the exact same low-level substrate.

For a fair selector-only comparison, use it with `--freeze` and the same `--dqn-ckpt` as the learned RAM runs.

### `--ram-mode discrete`

This is the Elicit toy / Version 1 hard RAM.

- RAM action space: all `K^N` joint role assignments.
- Execution: one-hot `W`; each agent effectively uses one role/head.
- Training: Double-DQN over the `K^N` role action index.
- Best for: small fleets, reproducing the toy code faithfully, ablations.

### `--ram-mode factored`

This is a hard scalable ablation.

- RAM output: `N*K` per-agent role values.
- Execution: one-hot `W = argmax_k Q_role(i,k)`.
- Training: factorized Double-DQN-style backup, summed over agents.
- Best for: avoiding `K^N` blow-up while keeping hard roles.

### `--ram-mode soft_v2`

This is the Elicit V2 soft RAM.

- RAM output: role values converted to softmax weights `W(N,K)`.
- Execution: each agent acts with a weighted sum of PopArt-normalized Q-heads:

```text
Q_comp_i(a) = sum_k W[i,k] * PopArtNorm(Q_k(obs_i, a))
action_i = argmax_a Q_comp_i(a)
```

- Training: a soft Double-DQN-style backup. The online RAM chooses `W_next` with softmax; the target RAM evaluates that same soft `W_next`.
- Status: legacy soft-learning / hard-execution research path; not a current main method.

Choose the soft RAM network with `--soft-ram-arch`:

- `--soft-ram-arch mlp`: centralized fixed-size MLP emits `N*K` role values. Simple, but tied to a fixed `N`.
- `--soft-ram-arch attention`: scalable attention role selector. Parameters do not grow with `K^N`; this is the Elicit scalable version.

Useful soft RAM parameter:

- `--soft-ram-temperature 1.0`: lower values make soft weights closer to hard one-hot roles; higher values make weights smoother.

## Global Aggregator Modes

Use `--global-agg` to choose how the centralized fleet context `g` is built before the RAM head reads it.

### `--global-agg attention`

Elicit-style relational aggregation.

- Each agent encoding can attend to the other agents before pooling.
- This was the default relational aggregator for the legacy RAM modes.
- The new hard-role methods use their own residual attention trunk.

### `--global-agg mean_pool`

Mean-pool ablation.

- No transformer/self-attention before pooling.
- Still learns the final projection/norm into `g`.
- Use this to test whether relational aggregation helps, independent of `--soft-ram-arch`.

## Role State Modes

Use `--role-state-mode` to choose the mission/context vector appended to the learned fleet embedding `g`.

### `flat`

Stores the exact previous role matrix by flattening `prev_W`.

```text
extra = [coverage, trash_density, budget_frac, r_accum(K), prev_W(N*K), scalarization_weights(K)]
```

This matches the toy implementation closely, but the input dimension grows with `N*K`.

### `pooled`

Stores only the fleet role distribution `mean_i prev_W[i,k]`.

```text
extra = [coverage, trash_density, budget_frac, r_accum(K), role_distribution(K), scalarization_weights(K)]
```

This is the scalable Elicit-style state. Use it with:

```bash
--ram-mode soft_v2 --soft-ram-arch attention --role-state-mode pooled
```

### `auto`

Default behavior:

- `soft_v2 + attention` uses `pooled`.
- Other modes use `flat`.

## RAM Reward Normalization

Use `--role-reward-norm` to choose how the high-level RAM reward is normalized.

### `minmax`

Your greedy-agent style normalization.

- Normalize each objective component first.
- Then scalarize with the preference vector.
- Good when you want comparability with your greedy reward selector.
- Handles negative rewards because min/max bounds are learned online/warmup.

```text
norm_components = (r_components - min) / (max - min)
R_RAM_step = scalarize(w, norm_components)
```

### `running_mean_std`

Standard reward normalization for the RAM scalar reward.

- Scalarize first.
- Then normalize the scalar `R_RAM` with its own running mean/std.
- This is the normalization described in the final Elicit spec for the RAM level.
- It is separate from PopArt. PopArt remains per low-level Q-head.

```text
R_raw = scalarize(w, r_components)
R_RAM_step = (R_raw - running_mean) / running_std
```

### `none`

No RAM reward normalization. Useful for ablations.

Legacy shortcut:

```bash
--no-reward-normalization
```

is equivalent to:

```bash
--role-reward-norm none
```

## Scalarization Methods

There are two scalarization switches because they act at different levels.

### `--role-scalarization`

This chooses the scalar reward used to train the high-level RAM / RoleSelector.

```text
R_RAM_step = scalarize(w, reward_components)
```

Choices:

- `ws`: Weighted Sum. This is the original Elicit linear scalarization.
- `wp`: Weighted Power, matching your greedy `WP Reward`.
- `wpop`: Weighted Product Of Powers, matching your greedy `WPOP Reward`.
- `ewc`: Exponential Weighted Criterion, matching your greedy `EWC Reward`.

Recommended pairings:

- Main Elicit/convex-envelope run: `--role-scalarization ws`.
- Cheap non-convex-pressure ablation: `--role-scalarization ewc --q-scalarization ws`.
- Use `--role-reward-norm minmax` with `wp`, `wpop`, or `ewc`, because those methods were designed around normalized non-negative objective components in your greedy code.

Useful scalarization parameters:

- `--scalarization-power 3.0`: power `p` for `wp`; default matches your greedy code.
- `--ewc-p 1.0`: exponential parameter for `ewc`; default matches your greedy code.

### `--q-scalarization`

This chooses how the low-level Q heads are collapsed into one action score.

```text
score_i(a) = scalarize(W[i], [Q_clean(obs_i,a), Q_explore(obs_i,a)])
action_i = argmax_a score_i(a)
```

Choices are the same: `ws`, `wp`, `wpop`, `ewc`.

- `--q-scalarization ws` is the exact Elicit soft V2 execution and is a linear/convex Q mixture.
- `--q-scalarization wp|wpop|ewc` is the learned vector-Q style ablation: the PopArt-normalized Q heads are first normalized across actions to `[0,1]` for the current state, then the nonlinear scalarizer is applied before `argmax`.
- Nonlinear `q_scalarization` is the switch that breaks the strict weighted-sum action-selection geometry. Treat it as a separate method, not as the faithful Elicit V2 result.

## Quick Checks

From inside `Learning/ctde_ram`:

```bash
python run_experiment.py --smoke --episodes 1
python run_experiment.py --smoke --episodes 3 --ram-mode random --freeze
python run_experiment.py --smoke --episodes 3 --ram-mode discrete
python run_experiment.py --smoke --episodes 3 --ram-mode factored
python run_experiment.py --smoke --episodes 3 --ram-mode soft_v2 --soft-ram-arch mlp --role-reward-norm minmax
python run_experiment.py --smoke --episodes 3 --ram-mode soft_v2 --soft-ram-arch attention --role-state-mode pooled --role-reward-norm minmax
python run_experiment.py --smoke --episodes 3 --ram-mode soft_v2 --soft-ram-arch attention --global-agg mean_pool
python run_experiment.py --smoke --episodes 3 --ram-mode soft_v2 --role-scalarization ewc --q-scalarization ws
python run_experiment.py --smoke --episodes 3 --ram-mode soft_v2 --role-scalarization ewc --q-scalarization ewc
```

From the repository root:

```bash
python Learning/ctde_ram/run_experiment.py --smoke --episodes 1
```

## Monitoring, Saving, And Evaluation

Every training run now creates one run directory:

```text
Learning/ctde_ram/outputs/<run_name>/
```

Inside it:

```text
config.json                         exact CLI/config used
runtime.json                        resolved device, backend, TensorBoard path
done.json                           final checkpoint and best HV summary
metrics/train_episodes.csv          one row per episode
metrics/latest_train_episode.json   last episode metrics
metrics/eval_summary.csv            one row per eval sweep
metrics/latest_eval.json            last Pareto sweep
eval/eval_ep_XXXXX.{json,csv,png}   Pareto details per evaluation point
eval/eval_ep_<episodes>.{json,csv,png} guaranteed final-policy Pareto evaluation
checkpoints/latest.pt               latest checkpoint
checkpoints/final.pt                final checkpoint
checkpoints/best_hv.pt              best hypervolume checkpoint
checkpoints/episode_XXXXX.pt        periodic saves
tensorboard/                        TensorBoard event files when tensorboard is installed
```

Useful training flags:

```bash
--run-name my_run
--output-dir Learning/ctde_ram/outputs
--save-every 25
--eval-every 25
--eval-episodes 3
```

TensorBoard:

```bash
tensorboard --logdir Learning/ctde_ram/outputs/<run_name>/tensorboard
```

If training prints:

```text
[tensorboard] disabled: ... No module named 'tensorboard'
```

then no event file can be created in that Python environment. Install the optional
logging dependencies in the same environment you use for training:

```bash
pip install tensorboard matplotlib
```

`matplotlib` is only needed for saved PNG figures; CSV/JSON/checkpoints are saved
even without it.

Offline evaluation of a saved policy:

```bash
python Learning/ctde_ram/run_evaluation.py \
  --checkpoint Learning/ctde_ram/outputs/<run_name>/checkpoints/best_hv.pt \
  --episodes-per-w 5 \
  --points 11 \
  --probe
```

This rebuilds the env/trainer from the checkpoint config, loads the policy, saves
a new Pareto JSON/CSV/PNG, and can also run the preference-sensitivity probe.

## Real Project Run

### Low-level control substrate (`--project-control`)

`--env project` supports two ways for CTDE-RAM to drive `MultiAgentPatrolling`:

- `dqn_heads` (default): CTDE-RAM uses its own DQFDueling heads and sends
  movement actions. `--dqn-ckpt` (or `--path-planner-folder`) optionally
  initializes those heads; `--freeze` keeps them fixed for a selector-only study.
- `expert_nu`: CTDE-RAM only emits `W`/`nu`, and the loaded `Expert_nu` path
  planner converts `nu` into movement actions. This mirrors
  `Learning.utils.make_env` + `MultiAgentNuWrapper` exactly (`W[:,1]` is the
  exploration probability `nu`). The same checkpoint is reused as the frozen RAM
  encoder, low-level replay/PopArt are disabled, and the RAM is the only learner.

`expert_nu` requires the path planner checkpoint. Pass the paper folder name and
it resolves to `Learning/path_planner_algorithms/<folder>/Final_Policy.pth`:

```bash
python Learning/ctde_ram/run_experiment.py \
  --env project \
  --project-control expert_nu \
  --path-planner-folder Experimento_clean28_malaga_port_macro_plastic_random_nus_nsteps5_distbudget100_old_reward \
  --map-name malaga_port \
  --map-csv Environment/Maps/malaga_port.csv \
  --initial-positions "[[12,7],[14,5],[16,3],[18,1]]" \
  --N 4 \
  --episodes 500 \
  --T-role 20 \
  --ram-mode soft_v2 \
  --soft-ram-arch attention \
  --role-state-mode pooled \
  --role-reward-norm minmax \
  --warmup-episodes 10
```

Expert_nu masking matches `make_env` defaults (`masked_actions` and `consensus`
on); disable with `--no-expert-masked-actions` / `--no-expert-consensus`, and
change the expert head with `--expert-type`. `--freeze` is forced on in this mode
because the trainer's network is only the encoder. A direct `--dqn-ckpt` is still
accepted in place of `--path-planner-folder`.

The remaining `dqn_heads` examples below all accept `--project-control expert_nu`
plus `--path-planner-folder` to run the Expert_nu substrate instead.

Hard RAM baseline:

```bash
python Learning/ctde_ram/run_experiment.py \
  --env project \
  --map-name malaga_port \
  --map-csv Environment/Maps/malaga_port.csv \
  --initial-positions "[[12,7],[14,5],[16,3],[18,1]]" \
  --N 4 \
  --episodes 500 \
  --T-role 20 \
  --ram-mode discrete \
  --role-reward-norm minmax \
  --warmup-episodes 10 \
  --dqn-ckpt path/to/Final_Policy.pth
```

Soft V2 attention RAM:

```bash
python Learning/ctde_ram/run_experiment.py \
  --env project \
  --map-name malaga_port \
  --map-csv Environment/Maps/malaga_port.csv \
  --initial-positions "[[12,7],[14,5],[16,3],[18,1]]" \
  --N 4 \
  --episodes 500 \
  --T-role 20 \
  --ram-mode soft_v2 \
  --soft-ram-arch attention \
  --role-state-mode pooled \
  --role-reward-norm minmax \
  --role-scalarization ws \
  --q-scalarization ws \
  --warmup-episodes 10 \
  --dqn-ckpt path/to/Final_Policy.pth
```

Use `--freeze` to keep pretrained low-level DQN weights fixed. Without `--freeze`, the same DQN is trained alongside the RAM. If `--dqn-ckpt` is omitted, the DQN starts from scratch and is still trained with the RAM.

Keep `--role-reward-norm running_mean_std` as an ablation. For preference-conditioned comparisons, `minmax` is the safer main setting because it normalizes each objective component first and only then scalarizes with `w`.

Nonlinear scalarization run:

```bash
python Learning/ctde_ram/run_experiment.py \
  --env project \
  --map-name malaga_port \
  --map-csv Environment/Maps/malaga_port.csv \
  --initial-positions "[[12,7],[14,5],[16,3],[18,1]]" \
  --N 4 \
  --episodes 500 \
  --T-role 20 \
  --ram-mode soft_v2 \
  --soft-ram-arch attention \
  --role-state-mode pooled \
  --role-reward-norm minmax \
  --role-scalarization ewc \
  --q-scalarization ewc \
  --warmup-episodes 10 \
  --dqn-ckpt path/to/Final_Policy.pth
```

For the paper, keep `ws/ws` labelled as faithful Elicit soft V2. Label `ewc/ws` as nonlinear RAM reward training, and `ewc/ewc` as nonlinear Q-exit action scalarization.

## Diagnostics

Preference sensitivity probe:

```bash
python Learning/ctde_ram/run_experiment.py \
  --smoke \
  --episodes 3 \
  --ram-mode soft_v2 \
  --soft-ram-arch attention \
  --probe-preference-sensitivity \
  --probe-csv Learning/ctde_ram/preference_probe.csv \
  --probe-plot Learning/ctde_ram/preference_probe.png \
  --probe-pareto-plot Learning/ctde_ram/preference_probe.pareto.png
```

This sweeps `w=(1,0)` to `w=(0,1)`, prints both hard `argmax` role fractions and mean executed soft `W`, and plots the coverage-versus-cleaning Pareto front from the same probe rollouts. If the role values are flat, the RAM is not reacting to preferences.

Aggregator gradient check:

```bash
python Learning/ctde_ram/run_experiment.py \
  --smoke \
  --episodes 3 \
  --ram-mode soft_v2 \
  --soft-ram-arch attention \
  --check-aggregator-grad
```

Expected result after at least one RAM update:

```text
[check:aggregator_grad] PASS global_agg_grad_abs_sum=...
```

Frozen PopArt invariance check:

```bash
python Learning/ctde_ram/run_experiment.py \
  --smoke \
  --episodes 0 \
  --freeze \
  --check-frozen-popart
```

Expected result:

```text
[check:frozen_popart] PASS ... q_max_abs_diff=0.0 param_max_abs_diff=0.0 rescale_flags=[False, False]
```

## Fidelity To Elicit

Implemented now:

- Hard CTDE-RAM Version 1 (`discrete`).
- Random hard RAM control (`random`).
- Hard scalable/factored ablation (`factored`).
- Soft RAM Version 2 (`soft_v2`) that executes a real soft `W(N,K)`.
- Scalable attention soft RAM (`soft_v2 --soft-ram-arch attention`).
- Global aggregator switch: relational attention or mean-pool ablation.
- PopArt per low-level head.
- RAM reward normalization switch: `minmax`, `running_mean_std`, `none`.
- Role reward scalarization switch: `ws`, `wp`, `wpop`, `ewc`.
- Q-exit scalarization switch: `ws`, `wp`, `wpop`, `ewc`.
- Role state switch: exact flattened previous `W` or scalable pooled role distribution.
- Pareto evaluation over coverage and trash-cleaned metrics.
- Preference sensitivity probe over scalarization weights.
- Aggregator gradient and frozen PopArt invariance diagnostics.
- `make_env`/`Expert_nu` substrate via `--project-control expert_nu`: RAM emits
  `W`/`nu` and the fixed path planner navigates, matching `MultiAgentNuWrapper`.

Still intentionally not implemented here:

- Dedicated Stage 1 fixed-role pretraining loop. You can still load pretrained weights or train DQN+RAM jointly.

## Dependency Note

The toy smoke test only needs `torch` and `numpy`. The `--env project` path imports your existing project stack, including `gym`, `Environment`, and `Algorithm` modules.
