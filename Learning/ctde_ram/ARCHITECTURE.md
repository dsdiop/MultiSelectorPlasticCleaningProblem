# CTDE-RAM architecture (code-derived)

This document describes the code as inspected, not an intended design inferred from the README. Low-level DQN training, Hindsight Preference Replay (HPR), and PopArt internals are deliberately excluded. Low-level networks are covered only where their outputs drive execution.

Notation: `B` batch size, `N` agents, `K` roles/objectives, `A` movement actions, `O` observation shape, `D=prod(O)`, `E=d_enc`, `C=d_ctx`, `R=d_role`, and `X=d_extra`.

## 1. Discovered inventory

Files inspected:

- `README.md`
  - Classes/functions: none.
  - Important parameters: documents example invocations and design history.
  - Role: documentation only; not treated as authoritative over code.
- `aquatic_env.py`
  - Class: `AquaticMAEnv`; methods `reset`, `step`, metrics, observation/patch/drift/render helpers.
  - Constructor: `N=4,H=20,W=20,patch_radius=3,max_steps=200,trash_init_cells=30,trash_max=5,trash_pickup=1,drift_rate=0,seed=None`.
  - Role: toy environment adaptation/execution and metrics.
- `experiment_io.py`
  - Functions: `timestamp`, directory/run-name helpers, JSON conversion/read/write, metric flattening, CSV append/write, Pareto row/artifact/plot helpers.
  - Role: logging, checkpoint-adjacent artifact IO, plotting, utilities.
- `global_aggregator.py`
  - Classes: `GlobalAggregator`, `MonotonicQMixer`.
  - Important switches: aggregation `attention|mean_pool`; mixer state and embedding sizes.
  - Role: aggregation and factored RAM value mixing/network definition.
- `nets.py`
  - Classes: `SharedEncoder`, `TaskQHead`, `DuelingRolePopArtOutput`, `DuelingNuQNetwork`.
  - Functions: gradient toggle and MLP/DQN checkpoint loaders.
  - Role: low-level network definition and RAM input encoding. PopArt adapter internals are excluded.
- `pareto.py`
  - Functions: dominance, Pareto filtering, exact 2-D hypervolume, generic dispatch (rejects dimension >2), scalarization sweep.
  - Role: evaluation utilities.
- `popart.py`
  - Class: `PopArtNorm` (`normalize`, `denormalize`, target/stat updates).
  - Role: low-level value normalization. Excluded by request except that normalized head outputs enter action selection.
- `project_env.py`
  - Functions: checkpoint resolution, default positions, expert construction, project-env factory.
  - Classes: `ProjectPatrollingCTDEEnv`, `ProjectPatrollingExpertNuCTDEEnv`.
  - Role: environment adaptation, movement/role-weight execution, project configuration.
- `replay_buffers.py`
  - Classes: `LowLevelReplayBuffer`, `RoleReplayBuffer`; helper `_torch_from_array`.
  - Role: replay. Only `RoleReplayBuffer` is explained in depth; low-level replay is excluded.
- `requirements.txt`
  - Dependencies: `torch`, `numpy`, `tensorboard`, `matplotlib`, `tqdm`.
  - Role: packaging only.
- `role_selector.py`
  - Classes: `FiLMConditioner`, `DuelingRAMHead`, `RoleSelectorAttention`.
  - Role: preference conditioning and RAM heads/network definition/role selection.
- `run_evaluation.py`
  - Functions: evaluation parser, checkpoint namespace reconstruction, grid builder, `main`.
  - Role: offline evaluation, CLI/configuration, loading, output.
- `run_experiment.py`
  - Functions: preference sampling, progress/device/summary helpers, parser, env/trainer builders, `main`.
  - Role: training entry point, CLI/configuration, periodic evaluation, logging/checkpointing.
- `run_greedy_expert_nu.py`
  - Class: `GreedyExpertNuSelector`; functions parser, reward-stat collection, evaluation, main.
  - Role: separate one-step greedy evaluation baseline, scalarization, Pareto output. It does not train or execute RAM.
- `tb_logger.py`
  - Class: `TBLogger`; scalar/text/config/episode/Pareto/role logging and lifecycle methods.
  - Role: logging utilities.
- `trainer.py`
  - Classes: `RewardMinMaxNormalizer`, `RunningMeanStdNormalizer`, `CTDERAMTrainer`.
  - Functions: progress/tensor conversion helpers.
  - Role: complete execution and RAM-training orchestration, scalarization, replay use, evaluation, checkpointing.

Generated `__pycache__/` and `outputs/` directories contain no source architecture and are irrelevant.

## 2. Big picture and clocks

```text
CENTRALIZED HIGH-LEVEL CLOCK (initially, then each T_role steps or terminal)

 [obs_1 ... obs_N] --shared/frozen encoder--> z_all [N,E]
                                               |
                         +---------------------+--------------------+
                         | GlobalAggregator                         |
                         | attention/mean -> fleet g [C]            |
                         +---------------------+--------------------+
                                               | + mission extra [X]
                                               v
                random / joint MLP / factored MLP / soft selector
                                               |
                           selector W [N,K] (one-hot or soft)
                                               |
                              held constant for a role window
                                               v
PER-AGENT LOW-LEVEL CLOCK (every environment step)

 movement mode: obs_i -> K Q heads -> normalized Q_k(a) -> scalarize with W_i
                 -> argmax action_i -> env.step([action_i])

 expert_nu mode: W -> W[:,1]=nu -> Bernoulli hard condition -> Expert_nu
                 -> movement action dict -> underlying project env
```

At execution, decentralized low-level action scoring is per agent, but the RAM is centralized because it consumes every `z_i`, pooled fleet context, global mission metrics, previous fleet roles and preference. Training adds the fleet-wide role replay, online/target RAM networks, optimizer and loss.

## 3. Neural and architecture-relevant modules

| Module (file/class) | Active modes | Input -> output | Trainable / loss / target | dimensional dependence | Execution/training role |
|---|---|---|---|---|---|
| Shared encoder (`nets.SharedEncoder`) | toy, all RAM modes | `[B,D] -> [B,E]`; MLP `D->256->E` | Low-level optimizer unless frozen; copied to `encoder_tgt` | input depends on observation/N in toy; output E | creates `z_i` and low-level head input; replay stores detached z |
| Task Q head (`nets.TaskQHead`, K copies) | toy movement execution | `[B,E] -> [B,A]`, `E->128->A` | low-level optimizer/target; training details excluded | K copies, A outputs | its normalized Q outputs are combined using row `W_i` |
| Project dueling wrapper (`DuelingNuQNetwork`) | project | observation `[B,*O] -> z [B,256]`; `q_role -> [B,A]` | optionally frozen; target copy; low-level loss excluded | action split requires two equal A-sized heads; `role_to_head` maps K roles | supplies RAM encodings and per-role navigation Qs |
| Global aggregator (`GlobalAggregator`) | every trained RAM mode | `[B,N,E] -> [B,C]`; Transformer(E, 4 heads, 2 layers, FF=2E) or identity, mean over N, Linear E->C, LayerNorm | RAM optimizer; no target copy. Recomputed for current and next replay state; next under no-grad | parameters independent of N; attention compute roughly O(N²E), mean O(NE) | fleet context `g` |
| Central RAM MLP (`trainer.ram_q`) | discrete, factored, soft_v2+mlp | `[B,C+X] -> [B,K^N]` discrete or `[B,NK]` otherwise; two 256-ReLU layers | RAM optimizer, Huber TD loss; `ram_q_tgt` | discrete output exponential; factored/soft MLP output NK and flat X may grow NK | hard joint/per-agent values or soft logits |
| Dueling RAM head (`DuelingRAMHead`) | optional discrete/factored only | hidden `[B,256]`; joint V+adv -> `[B,K^N]`, or N values + NK advantages -> `[B,NK]` | replaces last RAM MLP linear; target inside copied `ram_q_tgt` | K^N or NK | value/advantage decomposition only; selection semantics unchanged |
| FiLM conditioner (`FiLMConditioner`) | `w_conditioning=film` | preference `[B,K] -> gamma,beta [B,F]` through `K->64->64->2F`; bounded version starts identity | RAM optimizer and target (`ram_film_tgt`) for MLP; embedded in selector and selector target for attention | depends K and feature F, not N | modulates MLP hidden F=256 or selector query F=R; concat still includes preference in extra |
| Attention role selector (`RoleSelectorAttention`) | soft_v2+attention | `z[B,N,E],g[B,C],extra[B,X] -> logits[B,N,K]`; query Linear(E+C+X,R), dot K learned role keys | RAM optimizer/Huber TD; `role_selector_tgt` | parameter count independent of N; role keys/query output depend K/R | scalable soft RAM logits; softmax produces W |
| Monotonic mixer (`MonotonicQMixer`) | factored + qmix | selected agent values `[B,N]`, state `[B,C+X] -> [B]`; state-conditioned nonnegative hypernet weights | RAM optimizer/Huber TD; mixer target | explicitly constructed for fixed N; hypernet emits N*32 | combines factored local values for TD prediction/target only; selection remains per-agent argmax |
| Random selector | random | RNG -> roles `[N]`, one-hot W `[N,K]` | not trainable/no target | O(NK) | control baseline |

The instantiated-but-inactive modules matter: `role_selector` is always constructed/checkpointed but trained only for soft_v2+attention; `ram_q` is always constructed/checkpointed but inactive in that path. Random mode constructs both but optimizes neither aggregator nor head.

## 4. RAM modes

| Mode | Networks trained | Meaning/output | execution | TD action evaluation | scaling/limits |
|---|---|---|---|---|---|
| `auto` | resolves to another mode | discrete if `K^N <= max_joint_role_actions`, else factored | inherited | inherited | threshold default 512 |
| `random` | none | uniform random role per agent; one-hot `[N,K]` | hard | no replay update | scalable baseline, preference-blind |
| `discrete` | aggregator + RAM MLP (+FiLM/dueling) | one Q per joint tuple; `[K^N]` | epsilon joint choice, tuple -> one-hot W | gather joint base-K index; online argmax, target gather | exponential output, combo list and compute/memory `K^N`; fixed N/K |
| `factored` | aggregator + RAM MLP (+FiLM/dueling, optional mixer) | per-agent role Qs `[N,K]` | epsilon roles or independent argmax -> one-hot W | selected per-agent values summed or QMIXed; online per-agent argmax, target gather | output/replay action NK; MLP and usually flat X retain fixed-N parameters |
| `soft_v2+mlp` | aggregator + RAM MLP (+FiLM) | per-agent logits `[N,K]`; softmax temperature -> W | selector W then `soft`, `hard_argmax`, or training-only ST Gumbel policy | dot executed/stored W with role values; online soft W evaluates target logits | NK output; fixed N; `ram_dueling` ignored |
| `soft_v2+attention` | aggregator + attention selector | per-agent query/key logits `[N,K]`; softmax -> W | same W execution choices | same soft weighted backup | selector parameters independent of N and avoids K^N; pooled state is N-independent, but replay/runtime still uses N rows |

```text
discrete: [g;extra] -> K^N joint values -> argmax tuple -> one-hot W
factored: [g;extra] -> N x K values -> N argmaxes -> one-hot W -> sum/QMIX in loss
soft MLP: [g;extra] -> N x K logits -> row softmax -> W
soft attention: each [z_i;g;extra] -> query_i dot role_keys -> logits_i -> W_i
random: RNG -> N role integers -> one-hot W
```

`w_execution`: `soft` preserves W; `hard_argmax` one-hots it; `st_gumbel` samples straight-through hard weights only while training. Evaluation always hardens any non-`soft` choice by argmax.

## 5. Configuration and architecture switches

### Trainer constructor/defaults

| Group | parameters (default) | effect |
|---|---|---|
| fleet/shape | `obs_dim`; `N`; `K`; `A`; `obs_shape=None` | required sizes; shape defaults from obs_dim |
| representation | `d_enc=128`, `d_ctx=256`, `d_role=64` | encoder, global context, selector query/key widths |
| clocks/discount | `T_role=20`, `gamma=.99`, `gamma_role=None` | macro cadence; default macro discount is `gamma**T_role` |
| RAM | `ram_mode=auto`, `max_joint_role_actions=512`, `global_agg_mode=attention`, `soft_ram_arch=attention`, `soft_ram_temperature=1`, `w_execution=soft`, `ram_dueling=False`, `hfactored_mixer=sum` | mode/head/aggregation/output behavior |
| state/conditioning | `role_state_mode=auto`, `w_conditioning=concat`, `film_parameterization=bounded` | pooled auto only for soft attention; otherwise flat; FiLM alternative |
| reward/scalarization | `normalize_role_rewards=True`, `role_reward_norm=minmax`, `role_scalarization=ws`, `q_scalarization=ws`, `scalarization_power=3`, `ewc_p=1`, `ram_reward_mode=component_rewards` | RAM target and low-level action scoring |
| switching | absolute and relative penalties `0` | subtracts switching cost from next stored RAM reward only; options mutually exclusive |
| RAM optimization | `lr_ram=3e-4`, `buf_role_cap=5000`, `batch_role=32`, `target_period_ram=50` | Adam, replay and hard target sync schedule |
| low-level/execution | backend `mlp`; `freeze=False`; DQN/encoder/head checkpoints; `action_space_n`, `movement_actions`; `number_of_features=1024`, `nettype=0`, `archtype=v1`, `role_to_head=(1,0)` | selects toy/project controller and role-to-head mapping; training algorithm excluded |
| logging | `tb_logdir=./runs`, `tb_runname=ctde_ram_v1` | TensorBoard location |
| excluded | HPR flags; low-level learning/replay/target/PopArt settings | exist but intentionally not explained here |

### Training CLI (additional/defaults)

`run_experiment.py` exposes all constructor switches above except dimensions and RAM optimizer sizes (those remain constructor defaults). It also defines:

| Group | CLI/default | effect |
|---|---|---|
| run | `--env toy`, `--episodes 2000`, `--N 4`, `--seed 0`, `--device -1`, `--smoke`, `--run-name`, `--output-dir Learning/ctde_ram/outputs` | environment/run/device/output; smoke forces toy N=2, <=6 episodes |
| schedule | `--warmup-episodes 0`; preference `--weight-sampling uniform|beta`, `--weight-alpha .5` | statistics warmup; training preferences (uniform simplex-like normalization of random values, or Dirichlet) |
| checkpoint/eval | `--save-every 50`, `--eval-every None` (3 smoke/50 normal), `--eval-episodes None` (2/3), `--eval-points 5` | periodic saves and paired-seed Pareto sweeps |
| diagnostics | preference probe flag; points 5, episodes 3, CSV/plots optional; frozen/aggregator checks | post-training sensitivity and invariance/gradient checks |
| project map | map CSV None, name `malaga_port`, positions None, distance budget 100, n-agents None, dynamic false, miopic true, detection 2, movement 1, collisions 15, reward `Distance Field`, uint8 false, ground truth `macro_plastic` | raw project environment construction |
| project controller | `dqn_heads|expert_nu` default dqn; path-planner folder/root; expert `ExpertByMapCoverage`; masked and consensus true | movement-action versus W/nu adapter |

Offline evaluation adds required checkpoint; output/eval name; device override; 5 episodes/weight; 11 points; optional JSON weights; optional probe with 3 episodes. The greedy baseline separately exposes the same project recipe plus `T_role=10`, 10 probe points/episodes, two warmup episodes, `greedy_type=wpop`, required run name/folder, and normalization toggle.

Config is saved in `config.json`; resolved backend/device/T_role in `runtime.json`; checkpoints also embed `run_config` and `trainer_runtime`. Offline loading starts with current parser defaults, overlays saved known keys, uses legacy FiLM for old checkpoints, then rebuilds environment/trainer before loading models.

## 6. Data structures and shapes

| name | created -> consumed | general shape | N=4,K=2 example | dependence/use |
|---|---|---|---|---|
| `obs_all` | env reset/step -> encoder/action selector | list N of `O`; batch `[N,*O]` | 4 observations | N/O; execution |
| `z_all` | `_encode_all` -> aggregator/replay | `[N,E]`; replay batch `[B,N,E]` | `[4,128]` toy, `[4,256]` project | N/E; detached before replay |
| `g` | aggregator -> role state/selector | `[C]`, batch `[B,C]` | `[256]` | C; centralized |
| metrics | adapters -> `_build_extra` | three scalars: coverage, trash density, budget fraction | `[3]` | neither N nor K |
| `r_accum` / window components | step rewards/metric deltas -> extra/target | `[K]` | `[2]` | K; reset per role window |
| previous role summary | previous W -> extra | flat `[NK]` or pooled `[K]` | 8 or 2 | flat grows N |
| preference `scal_weights` | sampler/grid -> extra/scalarizers | `[K]` | `[2]` | K |
| `extra` | `_build_extra` -> selector/replay | flat `3+K+NK+K`; pooled `3+K+K+K` | 15 flat, 9 pooled | mission context |
| `role_state` | concat(g,extra) -> central RAM/QMIX | `[C+X]`; batch `[B,C+X]` | 271 flat / 265 pooled | fixed-N only in flat |
| RAM values | heads -> role selection/TD | discrete `[B,K^N]`; otherwise `[B,N,K]` | `[B,16]` or `[B,4,2]` | mode-dependent |
| roles | argmax/combo -> one-hot/action index | `[N]`; replay-derived `[B,N]` | `[4]` | hard assignment |
| `W` selector/action | role selector -> execution/replay | `[N,K]`; batch `[B,N,K]` | `[4,2]` | soft or one-hot |
| low-level Q stack | K heads -> scalarizer | `[K,A]`; resulting composite `[A]` | `[2,A] -> [A]` | K/A; execution |
| per-agent rewards | env -> accumulation | list N of `[K]`; stacked `[N,K]` | `[4,2]` | N/K |
| role replay sample | buffer -> `update_ram` | z `[B,N,E]`, extra `[B,X]`, W `[B,N,K]`, scalar R `[B]`, components `[B,K]`, next z/extra, done `[B]` | default B=32 | training |
| evaluation point | evaluator -> Pareto | metrics dict; points `[P,2]`; front `[F,2]` | weight row `[2]` | exact 2-D only |

Role replay stores the actually selected selector W (`W`, before `_execution_weights`), not necessarily the hardened `executed_W`. Thus with `w_execution=hard_argmax` or `st_gumbel`, the soft_v2 TD prediction weights can differ from the environment-executed weights. This is a material off-policy/action-record mismatch in the current code.

## 7. Role state construction

```text
extra = concat(
  coverage_pct, trash_density, budget_frac,       # 3
  accumulated/current-window reward context,     # K
  flat(prev_W) [NK] OR mean_agents(prev_W) [K],
  requested Pareto preference w                  # K
)

role_state = concat(GlobalAggregator(z_all), extra)  # C + X
```

`auto` selects pooled only for soft_v2+attention and flat everywhere else. Pooled X=`3+3K` is independent of N; flat X=`3+3K+NK` grows with N. In training, `extra_prev` describes the start of a window. `extra_next` includes the just-finished window reward (min-max normalized only for component-reward/minmax), the selected W, current metrics, and same preference. After selection, accumulators reset, so this completed-window reward remains in the next state until the following boundary.

## 8. Aggregation and replay gradient path

```text
stored detached z [B,N,E]
          |
          +--> Transformer/identity --> mean_N --> Linear E->C --> LayerNorm --> g
                                                                                |
stored extra [B,X] -------------------------------------------------------------+--> RAM
```

Attention lets each agent token read all others before invariant mean pooling; mean_pool is the no-relational-attention ablation. Projection and LayerNorm train in both. The role buffer intentionally stores raw encodings instead of baked `g`, so `global_agg(z)` is recomputed inside `update_ram` and receives RAM-loss gradients. The low-level encoder does not: `z` was detached. There is no target aggregator: both online and target-head evaluations use the same current aggregator, with next computation inside `no_grad`.

## 9. Exact execution path

1. `run_experiment.main -> build_env`: toy returns `AquaticMAEnv`/MLP/T_role 20 (10 smoke); project returns an adapter/dueling backend/T_role 20 unless explicitly set.
2. `build_trainer -> CTDERAMTrainer.__init__` constructs low-level encoder, aggregator, all RAM heads, active optimizer/targets and replay.
3. `run_episode`: normalize requested preference; `env.reset`; initialize zero reward context and zero previous W.
4. `_encode_all`: stack N observations and produce `[N,E]`; detach.
5. `_build_extra`, then `select_roles` according to mode. `_execution_weights` produces the W held for the role window.
6. Every env step, `_step_env_with_current_W`:
   - Movement adapter: for each agent, `select_action` gets each role head's `[A]` raw Q, applies its already-maintained normalization, stacks `[K,A]`, scalarizes every action using `W_i`, and chooses epsilon-greedy argmax. It passes N integer actions to the env.
   - Expert-nu adapter: sends W itself. Adapter takes `nu=W[:,1]`, Bernoulli samples exploration/cleaning conditions, asks fixed `Expert_nu` for safe movement actions, then steps the raw env. No low-level replay/update occurs.
7. W remains unchanged until `ep_step % T_role == 0` or terminal. At the boundary, accumulate/scalarize reward, store RAM transition, optionally update RAM, construct next state and select new W.
8. Log losses, gradients, role/weight fractions, switches, metrics and buffer sizes; the driver appends training CSV and performs periodic evaluation/checkpointing.

## 10. RAM training path

```text
role window transition:
 (z, extra, W, scalar_R - switch_cost, raw_component_R, next_z, next_extra, done)
                                    |
                              RoleReplayBuffer
                                    |
 current: aggregator + online head --+--> q_pred(action W)
 next: aggregator + online chooses action; target head evaluates it --> q_next
                                    |
 target = R + gamma_role * (1-done) * q_next
 loss = SmoothL1(q_pred, target) -> clip norm 10 -> Adam -> periodic target copy
```

Discrete uses Double-DQN joint online argmax/target gather. Factored uses online independent argmax, target gathers N values, then sum or target QMIX. Soft uses online softmax logits as `W_next`, evaluates the dot sum `sum_{i,k} W_next*q_target`, and predicts the replayed `sum W_action*q_online`. Target sync occurs every 50 successful RAM updates by default. Aggregator and active head/mixer/FiLM receive gradients; low-level encoders do not. HPR code exists in `update_ram` but is excluded from this explanation.

The stored macro reward is one undiscounted sum over step components (or metric progress over the window), scalarized once; there is no within-window `gamma^t` accumulation. `gamma_role` defaults to `gamma**T_role` for bootstrapping.

## 11. Scalarization and reward processing

Weights are normalized to sum one. For objective vector `v`:

- `ws`: `sum_k w_k v_k` (main linear soft-head mixture).
- `wp`: `sum_k w_k v_k^p`, default p=3. Integer p permits negative values; fractional p clamps to epsilon.
- `wpop`: `product_k max(v_k,eps)^w_k`; requires meaningful nonnegative values.
- `ewc`: `sum_k (exp(p*w_k)-1) exp(p*clamp(v_k,-20,20))`, default p=1.

At RAM reward level, `component_rewards` sums all agents' per-step K-vectors over the window. `delta_metrics` uses nonnegative increases `[trash_cleaned, coverage]` from boundary metrics and forcibly bypasses reward normalization. Normalization options: component-wise historical min/max before scalarization; running mean/std after scalarization; or none. For nonlinear low-level Q scalarization, every head's actions are min-max mapped to `[0,1]` within the current state first; weighted sum uses normalized head Qs directly.

Example with `[cleaning, exploration]=[.2,.8]`, preference `[.75,.25]`: WS=.35; WP(p=3)=`.75*.008+.25*.512=.134`; WPOP uses `.2^.75*.8^.25`; EWC uses the exponential formula above.

Important ordering defect: the project adapter deliberately returns `[cleaning, exploration]`, while `AquaticMAEnv.step` returns `[exploration, cleaning]`. Metric-delta code also hardcodes `[cleaning, coverage]`. Therefore preference index semantics are not consistent between toy component-reward runs and project/metric-delta runs.

## 12. Environment adapters

Expected trainer interface: attributes `N,K,A,obs_dim/obs_shape`; `reset(seed?) -> list[N]`; `step(actions) -> (obs list, reward-vector list, done, info)`; metrics `coverage_pct`, `trash_density`, `budget_frac`, and preferably `trash_cleaned_pct`.

Toy observations concatenate two `(2r+1)^2` patches, normalized own row/column/time, and three relative features for each teammate. Hence `D=2(2r+1)^2+3+3(N-1)`: the toy observation and encoder parameters themselves change with N.

The project movement adapter converts dict observations to stable agent-ordered lists, retains last observation for inactive agents, scales uint8, flips underlying rewards from `[exploration,cleaning]` to `[cleaning,exploration]`, converts list actions to active-agent dict, and reports all-done. It derives coverage/trash from underlying percentages, density from ground-truth occupied visitable cells, and budget as mean remaining distance fraction.

The expert-nu adapter instead advertises `ctde_action_mode=role_weights`; maps W's role-1 column to exploration probability; Bernoulli samples a hard condition per agent; invokes masked/unmasked `Expert_nu`; and uses any-done to mirror the prior wrapper. Info includes nu and movement actions.

The factory loads the map CSV, parses JSON/default initial positions, truncates to N, and configures `MultiAgentPatrolling`. Defaults are mostly CLI values, but there is a code discrepancy: factory fallback `movement_length=2`, while both CLIs define default 1, so CLI-created runs use 1. Project action count exposed to RAM is A=8; raw action-space size configures the two dueling head slices.

## 13. Evaluation, Pareto and artifacts

Periodic training evaluation uses paired episode seeds for every preference, epsilon zero, and the same role cadence. It averages final coverage, trash cleaned and switches. It filters nondominated maximization points and computes exact 2-D hypervolume against `(0,0)`. Dimension >2 raises `NotImplementedError`. Periodic results save JSON/CSV/PNG, evaluation summary CSV/latest JSON, `best_hv.pt`, periodic/latest/final checkpoints (as driven by `main`), and TensorBoard metrics.

Offline evaluation reconstructs current defaults plus saved config, loads without optimizers, sweeps 11 linear K=2 preferences by default (custom JSON required for K!=2), writes progress JSON after each weight, Pareto artifacts, summary/config/checkpoint metadata, and optional sensitivity artifacts.

The preference-sensitivity probe records every decision's executed argmax fractions and mean executed W plus final task metrics; flat results expose a preference-insensitive selector. It can write CSV, role-fraction PNG, and Pareto PNG. Final post-training probing is optional; periodic evaluation measures outcomes but not mean W. The greedy script is a non-RAM baseline: it enumerates `2^N` hard nu assignments, simulates one-step rewards, scalarizes, and evaluates the chosen assignment every T_role.

Checkpoint payloads contain active and inactive RAM modules/targets, low-level model, normalizer states, role replay, optimizer states, counters, run config/runtime and metrics. Low-level replay is not checkpointed.

## 14. Scalability

| component | parameters/output | execution/training/memory implication |
|---|---|---|
| discrete | output `K^N`; final layer and combo list exponential | argmax/target vectors and compute exponential; replay W remains NK |
| factored MLP | output NK; flat default state X includes NK | linear output and state-dependent first layer grow with N; sum O(NK), QMIX hypernet fixed to N |
| soft MLP | output NK; flat default unless role mode overridden (`auto` is flat because arch=mlp) | same fixed-N issue, soft dot O(NK) |
| soft attention pooled | K role keys; query input E+C+3+3K; output runtime NK | parameters independent of N; selector O(NK R), aggregation attention O(N²E); replay O(capacity*N*E + capacity*N*K) |
| mean aggregator | E->C params, runtime O(NE) | no relational communication |
| observation | toy D grows `3(N-1)`; project O fixed by env | toy encoder parameters indirectly grow with N |
| low-level output | K heads each A actions | action scoring O(NKA) each env step |

Changing N at inference is architecturally plausible only for soft attention + pooled state and an observation encoder whose shape does not encode N. The trainer/buffer and environment are still instantiated with a fixed N, so this is parameter scalability, not plug-and-play dynamic fleet size.

## 15. Worked example: N=4, K=2

Use normal toy defaults: T_role=20, E=128, C=256, R=64. With patch radius 3, toy D=`98+3+9=110`. `auto` sees `2^4=16 <= 512` and therefore chooses discrete, not soft attention. For a soft-attention example explicitly set `ram_mode=soft_v2`; auto role state becomes pooled X=`3+6=9`.

1. Four `[110]` observations stack to `[4,110]`.
2. shared encoder produces z `[4,128]`.
3. two-layer four-head aggregator consumes `[1,4,128]` and emits g `[1,256]`.
4. extra is `[coverage,density,budget]` (3) + accumulated rewards (2) + mean previous W (2) + preference (2) = `[9]`.
5. each selector query consumes `128+256+9=393`, projects to 64, and dots two `[64]` role keys: logits `[4,2]`.
6. row softmax at temperature 1 produces W `[4,2]`; execution keeps it soft by default.
7. For each of 20 steps and each agent, two low-level `[A]` vectors become `[2,A]`; W row scalarizes to `[A]`; argmax movement is executed.
8. Fleet `[4,2]` rewards sum across agents/steps into `[2]`, scalarize to one R, and store `(z[4,128], extra[9], W[4,2], R, components[2], next_z[4,128], next_extra[9], done)`.
9. Once 32 role transitions exist, a RAM batch recomputes aggregator/selector, forms the soft Double-DQN target, applies Huber loss, and updates aggregator+selector.

## 16. Cheat sheets

### Execution

```text
reset -> obs[N,O] -> encoder -> z[N,E] -> aggregator + extra -> RAM -> W[N,K]
  -> repeat T_role: {Q heads[K,A] --W_i scalarize--> action_i} -> env
  -> accumulate K rewards/metrics -> next RAM decision
```

### RAM training

```text
window -> store raw z/extra/W/R/components/next/done
       -> sample B -> recompute trainable aggregator
       -> online chooses next joint/per-agent/soft action
       -> target evaluates -> macro TD target
       -> Huber -> aggregator + active RAM head (+FiLM/mixer) -> target sync/50
```

### Modes

| random | discrete | factored | soft_v2 MLP | soft_v2 attention |
|---|---|---|---|---|
| random hard NK | learned hard K^N | learned hard NK | learned soft NK, fixed-N | learned soft NK, N-independent parameters with pooled state |

### Important switches

| concern | switches |
|---|---|
| selector | `ram_mode`, `max_joint_role_actions`, `soft_ram_arch`, temperature, `w_execution` |
| state/aggregation | `global_agg`, `role_state_mode`, E/C/R constructor widths |
| conditioning/head | `w_conditioning`, FiLM parameterization, RAM dueling, factored mixer |
| reward | reward mode, reward normalization, role scalarizer, switch penalty |
| action execution | q scalarizer, project control, freeze/checkpoint, role-to-head constructor mapping |
| clocks/training | T_role, gamma/gamma_role, RAM LR/batch/capacity/target-period constructor settings |
| evaluation | eval frequency/points/episodes, probes, checkpoint/output settings |

## 17. Code-level cautions found by inspection

1. Toy and project reward component order differs, as described above.
2. Soft RAM replay stores selector W, not hardened executed W.
3. `GlobalAggregator` has no target copy; target heads see a shared current context transform.
4. `ROLE_COMBOS=list(product(...))` is built before mode resolution, so even explicitly scalable modes still allocate all `K^N` tuples at construction. Very large N/K can exhaust memory before factored/soft execution begins.
5. Metric-delta implementation writes indices 0 and 1 unconditionally; K<2 fails and K>2 leaves remaining objectives zero.
6. Evaluation accumulation always uses raw reward vectors for next-state extra, even when training uses metric-delta semantics; it does not mirror training's delta-metric state context.
7. Project movement adapter terminates on all agents done; expert-nu terminates on any agent done, producing different horizons by controller mode.
8. The greedy baseline's saved `eval_config.resolved.T_role` is hardcoded to 1 even though evaluation uses `args.T_role`; artifact metadata can therefore be wrong.
