"""
Run CTDE-RAM.

Use --env toy for the self-contained smoke test.
Use --env project to train on MultiAgentPatrolling with the real DQFDueling
low-level network.
"""
from __future__ import annotations

import argparse
import os
import random

# cuBLAS reads this when CUDA is initialized. Keep a reproducible workspace
# choice even when this module is launched directly instead of through the
# repository shell launcher.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

try:
    from tqdm.auto import trange, tqdm
except ImportError:
    trange = None
    tqdm = None

try:
    from .aquatic_env import AquaticMAEnv
    from .experiment_io import (
        append_csv_row,
        args_to_dict,
        build_probe_artifact_path,
        ensure_dir,
        make_run_name,
        resolve_run_dir,
        save_pareto_artifacts,
        write_csv_rows,
        write_json,
    )
    from .project_env import build_env_factory_from_existing_project, resolve_path_planner_ckpt
    from .trainer import CTDERAMTrainer
except ImportError:
    from aquatic_env import AquaticMAEnv
    from experiment_io import (
        append_csv_row,
        args_to_dict,
        build_probe_artifact_path,
        ensure_dir,
        make_run_name,
        resolve_run_dir,
        save_pareto_artifacts,
        write_csv_rows,
        write_json,
    )
    from project_env import build_env_factory_from_existing_project, resolve_path_planner_ckpt
    from trainer import CTDERAMTrainer


def sample_weights(rng, k: int = 2, mode: str = "uniform", alpha: float = 0.5):
    """Sample a preference weight on the K-simplex.

    uniform: original behavior (i.i.d. U(0,1) then L1-normalize). Mass concentrates
             toward the simplex center, so the front corners are under-sampled.
    beta:    Dirichlet(alpha) with alpha<1 is U-shaped (Beta(0.5,0.5) for K=2),
             over-representing the pure-preference corners. Use this when the
             selector has averaged over weights and ignores the front extremes.
    """
    if mode == "beta":
        return rng.dirichlet(np.full(k, float(alpha))).astype(np.float32)
    w = rng.uniform(0.0, 1.0, size=k).astype(np.float32)
    s = float(w.sum())
    return (w / s) if s > 0 else np.ones(k, dtype=np.float32) / k


def annealed_weight_alpha(
    start: float, end: float | None, episode: int, total_episodes: int,
    anneal_fraction: float = 1.0,
) -> float:
    """Linear preference-sampler concentration schedule."""
    if end is None:
        return float(start)
    if anneal_fraction <= 0.0:
        raise ValueError("weight_alpha_anneal_fraction must be positive")
    # Episodes are zero-indexed, so total_episodes-1 is the final sampled point.
    horizon = max(float(max(total_episodes - 1, 1)) * float(anneal_fraction), 1.0)
    progress = min(max(float(episode) / horizon, 0.0), 1.0)
    return float(start + progress * (float(end) - float(start)))


def progress_write(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def describe_device(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        return f"{device} ({name}, {total_gb:.1f} GB)"
    cuda_note = "cuda unavailable" if not torch.cuda.is_available() else "cpu requested"
    return f"cpu ({cuda_note})"


def print_experiment_summary(args, env, low_level_backend: str, t_role: int, device: str, run_dir: str, trainer=None) -> None:
    print("\n" + "=" * 80)
    print("CTDE-RAM experiment")
    print("=" * 80)
    print(f"run_name        : {args.run_name}")
    print(f"output_dir      : {run_dir}")
    print(f"env/backend     : {args.env} / {low_level_backend}")
    print(f"device          : {describe_device(device)}")
    print(f"episodes        : {args.episodes}")
    print(f"agents/objectives/actions : N={getattr(env, 'N', args.N)} K={env.K} A={env.A}")
    print(f"T_role          : {t_role}")
    print(f"seed            : {args.seed}")
    print(f"eval            : every={args.eval_every if args.eval_every is not None else ('3 smoke' if args.smoke else '50 default')} "
          f"episodes={args.eval_episodes if args.eval_episodes is not None else ('2 smoke' if args.smoke else '3 default')} "
          f"points={args.eval_points}")
    if trainer is not None:
        print(
            "trainer         : "
            f"ram_mode={trainer.ram_mode} global_agg={trainer.global_agg_mode} "
            f"soft_arch={trainer.soft_ram_arch} role_state={trainer.role_state_mode}"
        )
        print(
            "scalarization   : "
            f"role={trainer.role_scalarization} q={trainer.q_scalarization} "
            f"reward_norm={trainer.role_reward_norm_name} gamma_role={trainer.gamma_role:.4f}"
        )
    print("params:")
    for key, value in sorted(args_to_dict(args).items()):
        print(f"  {key}: {value}")
    print("=" * 80 + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["toy", "project"], default="toy")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--smoke", action="store_true", help="tiny fast end-to-end check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, default=-1, help="-1 for CPU, otherwise CUDA device index")
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use strict deterministic PyTorch/CUDA algorithms (default: enabled).",
    )
    p.add_argument("--run-name", type=str, default=None, help="Name for output/checkpoint/TensorBoard folder.")
    p.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("Learning", "ctde_ram", "outputs"),
        help="Root directory for configs, CSV metrics, figures, and checkpoints.",
    )
    p.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N episodes; 0 disables periodic saves.")
    p.add_argument("--eval-every", type=int, default=None, help="Evaluate every N episodes; default is 3 for smoke, 50 otherwise.")
    p.add_argument("--eval-episodes", type=int, default=None, help="Episodes per scalarization weight during eval.")
    p.add_argument("--eval-points", type=int, default=5, help="Number of weights in the evaluation Pareto sweep.")

    # Low-level DQFDueling options.
    p.add_argument("--freeze", action="store_true", help="freeze low-level DuelingDQN/heads")
    p.add_argument("--dqn-ckpt", type=str, default=None, help="checkpoint for DQFDuelingVisualNetwork")
    p.add_argument("--encoder-ckpt", type=str, default=None, help="toy MLP compatibility")
    p.add_argument("--heads-ckpt", type=str, default=None, help="toy MLP compatibility")
    p.add_argument("--number-of-features", type=int, default=1024)
    p.add_argument("--nettype", type=str, default="0")
    p.add_argument("--archtype", type=str, default="v1")

    # RAM and normalization.
    p.add_argument("--T-role", type=int, default=None)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gamma-role", type=float, default=None)
    p.add_argument(
        "--ram-mode",
        choices=["ppo_ram", "hard_role_q", "auto", "random", "discrete", "factored", "soft_v2"],
        default="hard_role_q",
        help="Main methods are ppo_ram and hard_role_q; remaining modes are legacy/baselines.",
    )
    p.add_argument("--max-joint-role-actions", type=int, default=512)
    p.add_argument(
        "--global-agg",
        choices=["attention", "mean_pool"],
        default="attention",
        help="Fleet context aggregator: Elicit-style attention or mean-pool ablation.",
    )
    p.add_argument(
        "--soft-ram-arch",
        choices=["mlp", "attention"],
        default="attention",
        help="Only for --ram-mode soft_v2: fixed-size MLP or scalable attention role selector.",
    )
    p.add_argument(
        "--soft-ram-temperature",
        type=float,
        default=1.0,
        help="Only for --ram-mode soft_v2: lower values make W closer to hard one-hot roles.",
    )
    p.add_argument(
        "--w-execution",
        choices=["soft", "hard_argmax", "st_gumbel"],
        default="soft",
        help="How selector weights are hardened before execution; T-role still controls cadence.",
    )
    p.add_argument(
        "--role-switch-penalty",
        type=float,
        default=0.0,
        help="Penalty applied only to RAM learning reward when the selected role changes.",
    )
    p.add_argument(
        "--role-switch-penalty-rel",
        type=float,
        default=0.0,
        help="Switch penalty as a fraction of the running mean absolute RAM reward.",
    )
    p.add_argument("--hpr", action="store_true", help="Enable hindsight preference replay.")
    p.add_argument("--hpr-fraction", type=float, default=0.5, help="Fraction of each RAM batch to relabel.")
    p.add_argument("--hpr-kappa", type=float, default=1.0, help="Dirichlet concentration scale for HPR.")
    p.add_argument(
        "--w-conditioning", choices=["concat", "film"], default="concat",
        help="Preference conditioning for the RAM role network.",
    )
    p.add_argument(
        "--film-parameterization", choices=["legacy", "bounded"], default="bounded",
        help="FiLM modulation: original direct outputs or bounded residual scaling.",
    )
    p.add_argument("--ram-dueling", action="store_true", help="Use a dueling RAM head for discrete/factored modes.")
    p.add_argument(
        "--hfactored-mixer", choices=["sum", "qmix"], default="sum",
        help="How factored per-agent role values are combined for RAM learning.",
    )
    p.add_argument(
        "--role-state-mode",
        choices=["auto", "flat", "pooled"],
        default="auto",
        help="flat stores prev_W exactly; pooled stores role distribution and is the scalable attention state.",
    )
    p.add_argument(
        "--role-reward-norm",
        choices=["minmax", "running_mean_std", "none"],
        default="minmax",
        help="RAM reward normalization: greedy-style minmax, standard running_mean_std, or none.",
    )
    p.add_argument(
        "--role-scalarization",
        choices=["ws", "wp", "wpop", "ewc"],
        default="ws",
        help="Scalarizer used to train the high-level RoleSelector/RAM reward.",
    )
    p.add_argument(
        "--ram-reward-mode",
        choices=["component_rewards", "delta_metrics"],
        default="component_rewards",
        help="RAM reward source: per-step component rewards or mission-metric deltas over each role window.",
    )
    p.add_argument(
        "--q-scalarization",
        choices=["ws", "wp", "wpop", "ewc"],
        default="ws",
        help="Scalarizer applied to low-level Q heads before argmax action selection.",
    )
    p.add_argument(
        "--scalarization-power",
        type=float,
        default=3.0,
        help="Power p for WP scalarization; default matches the greedy code.",
    )
    p.add_argument(
        "--ewc-p",
        type=float,
        default=1.0,
        help="Exponential parameter p for EWC scalarization; default matches the greedy code.",
    )
    p.add_argument("--warmup-episodes", type=int, default=0)
    # Shared main hard-role architecture.
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-attn-heads", type=int, default=4)
    p.add_argument("--n-attn-layers", type=int, default=1)
    p.add_argument("--attn-ff-dim", type=int, default=128)
    p.add_argument("--preference-role-bias", action="store_true", default=False)
    p.add_argument(
        "--hard-role-preference-conditioning",
        choices=["film", "pref_token"],
        default="film",
        help="Condition hard-role attention with FiLM or an additional preference token.",
    )
    # PPO-RAM.
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--ppo-minibatch-size", type=int, default=128)
    p.add_argument("--ppo-rollout-macro-steps", type=int, default=1024)
    p.add_argument("--ppo-clip-eps", type=float, default=0.1)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--entropy-coef", type=float, default=0.03)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument(
        "--ppo-critic-mode", choices=["scalar", "vector"], default="scalar",
        help="Scalar legacy critic or one value estimate per reward objective.",
    )
    p.add_argument(
        "--ppo-critic-popart", action="store_true",
        help="Apply independent PopArt normalization to each PPO critic output.",
    )
    p.add_argument(
        "--ppo-advantage-scalarization",
        choices=["ws", "wp", "wpop", "ewc"], default="ws",
        help="How vector GAE advantages are combined for the PPO actor update.",
    )
    p.add_argument("--ppo-target-kl", type=float, default=0.02)
    p.add_argument("--ppo-max-grad-norm", type=float, default=0.5)
    p.add_argument(
        "--ppo-preference-sampling", choices=["stratified", "random"], default="random",
        help="Stratified cycles through five fixed preferences so each PPO rollout mixes objectives.",
    )
    # HardRoleQ-RAM and PER.
    p.add_argument("--role-q-lr", type=float, default=3e-4)
    p.add_argument("--role-q-target-update", type=int, default=50)
    p.add_argument("--role-q-mixer", choices=["none", "qmix"], default="none")
    p.add_argument("--role-q-gamma", type=float, default=None)
    p.add_argument("--role-q-epsilon-start", type=float, default=1.0)
    p.add_argument("--role-q-epsilon-end", type=float, default=0.05)
    p.add_argument("--role-q-epsilon-fraction", type=float, default=0.75)
    p.add_argument("--per-alpha", type=float, default=0.6)
    p.add_argument("--per-beta-start", type=float, default=0.4)
    p.add_argument("--per-beta-end", type=float, default=1.0)
    p.add_argument("--per-eps", type=float, default=1e-6)
    p.add_argument(
        "--weight-sampling",
        choices=["uniform", "beta"],
        default="beta",
        help=(
            "Preference weight sampler for training. 'beta' uses Dirichlet(alpha) "
            "(alpha<1, Beta(0.5,0.5) for K=2) to over-represent the front corners "
            "when the selector averages over weights and ignores pure preferences."
        ),
    )
    p.add_argument(
        "--weight-alpha",
        type=float,
        default=0.4,
        help="Dirichlet/Beta concentration for --weight-sampling beta; <1 favors pure-preference extremes.",
    )
    p.add_argument(
        "--weight-alpha-end", type=float, default=None,
        help="Optional final Dirichlet/Beta concentration; enables linear annealing from --weight-alpha.",
    )
    p.add_argument(
        "--weight-alpha-anneal-fraction", type=float, default=1.0,
        help="Fraction of training over which weight alpha reaches --weight-alpha-end.",
    )
    p.add_argument(
        "--no-reward-normalization",
        action="store_true",
        help="Legacy shortcut; equivalent to --role-reward-norm none.",
    )
    p.add_argument(
        "--probe-preference-sensitivity",
        action="store_true",
        help="After training, sweep w=(1,0)->(0,1) and print role/nu sensitivity.",
    )
    p.add_argument("--probe-points", type=int, default=5, help="Number of weights in the preference probe.")
    p.add_argument("--probe-episodes", type=int, default=3, help="Evaluation episodes per probed weight.")
    p.add_argument("--probe-csv", type=str, default=None, help="Optional CSV path for the preference probe table.")
    p.add_argument("--probe-plot", type=str, default=None, help="Optional PNG path for the preference probe plot.")
    p.add_argument(
        "--probe-pareto-plot",
        type=str,
        default=None,
        help="Optional PNG path for the coverage-vs-cleaning Pareto front built from probe rollouts.",
    )
    p.add_argument(
        "--check-frozen-popart",
        action="store_true",
        help="Verify --freeze keeps raw low-level Q-values unchanged during PopArt stat updates.",
    )
    p.add_argument(
        "--check-aggregator-grad",
        action="store_true",
        help="After training, report whether GlobalAggregator received nonzero RAM gradient.",
    )
    p.add_argument("--forced-role-diagnostics", action="store_true", help="Evaluate fixed hard-role policies before training.")
    p.add_argument("--forced-role-episodes", type=int, default=10)
    p.add_argument("--same-state-preference-check", action="store_true", help="Compare extreme preferences on exactly the same state.")

    # Existing project environment recipe.
    p.add_argument("--map-csv", type=str, default=None)
    p.add_argument("--map-name", type=str, default="malaga_port")
    p.add_argument("--initial-positions", type=str, default=None)
    p.add_argument("--distance-budget", type=int, default=100)
    p.add_argument("--n-agents", type=int, default=None)
    p.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dynamic ground-truth evolution (default: enabled; use --no-dynamic to disable).",
    )
    p.add_argument("--no-miopic", dest="miopic", action="store_false")
    p.set_defaults(miopic=True)
    p.add_argument("--detection-length", type=int, default=2)
    p.add_argument("--movement-length", type=int, default=1)
    p.add_argument("--max-collisions", type=int, default=15)
    p.add_argument("--reward-type", type=str, default="Distance Field")
    p.add_argument("--convert-to-uint8", action="store_true")
    p.add_argument("--ground-truth-type", type=str, default="macro_plastic")

    # Low-level control substrate for --env project.
    p.add_argument(
        "--project-control",
        choices=["dqn_heads", "expert_nu"],
        default="dqn_heads",
        help=(
            "dqn_heads: CTDE-RAM drives the DQFDueling heads and emits movement actions. "
            "expert_nu: CTDE-RAM emits W/nu and the loaded Expert_nu path planner converts "
            "nu into movement actions, mirroring Learning.utils.make_env."
        ),
    )
    p.add_argument(
        "--path-planner-folder",
        type=str,
        default=None,
        help=(
            "Folder under Learning/path_planner_algorithms/ holding Final_Policy.pth, e.g. "
            "Experimento_clean28_malaga_port_macro_plastic_random_nus_nsteps5_distbudget100_old_reward. "
            "Resolved into the Expert_nu path planner (and reused as the frozen RAM encoder)."
        ),
    )
    p.add_argument(
        "--path-planner-root",
        type=str,
        default=None,
        help="Override the path_planner_algorithms root used to resolve --path-planner-folder.",
    )
    p.add_argument(
        "--expert-type",
        type=str,
        default="ExpertByMapCoverage",
        help="Expert_nu expert head (matches make_env default).",
    )
    p.add_argument(
        "--no-expert-masked-actions",
        dest="expert_masked_actions",
        action="store_false",
        help="Disable Expert_nu safe action masking (default on, matches make_env).",
    )
    p.set_defaults(expert_masked_actions=True)
    p.add_argument(
        "--no-expert-consensus",
        dest="expert_consensus",
        action="store_false",
        help="Disable Expert_nu consensus safe action masking (default on, matches make_env).",
    )
    p.set_defaults(expert_consensus=True)
    return p.parse_args(argv)


def build_env(args):
    if args.smoke:
        args.env = "toy"
        args.N, args.episodes = 2, min(args.episodes, 6)

    if args.env == "toy":
        if args.smoke:
            env = AquaticMAEnv(N=2, H=8, W=8, patch_radius=2, max_steps=40, trash_init_cells=8, seed=args.seed)
            t_role = args.T_role or 10
        else:
            env = AquaticMAEnv(N=args.N, H=20, W=20, max_steps=200, seed=args.seed)
            t_role = args.T_role or 20
        return env, "mlp", t_role

    if args.n_agents is not None:
        args.N = args.n_agents

    # Resolve the path-planner checkpoint once. In expert_nu mode it is shared:
    # the env's Expert_nu converts nu->movement, and the trainer reuses the same
    # weights as the RAM encoder. Threading it into args.dqn_ckpt is what feeds
    # the trainer, so --path-planner-folder and --dqn-ckpt converge to one path.
    resolved_ckpt = resolve_path_planner_ckpt(
        path_planner_folder=getattr(args, "path_planner_folder", None),
        dqn_ckpt=args.dqn_ckpt,
        path_planner_root=getattr(args, "path_planner_root", None),
    )
    if resolved_ckpt is not None:
        args.dqn_ckpt = resolved_ckpt

    if args.project_control == "expert_nu":
        if resolved_ckpt is None:
            raise ValueError(
                "--project-control expert_nu requires --path-planner-folder (or --dqn-ckpt)."
            )
        # The trainer's low-level network is only a frozen encoder here; Expert_nu
        # owns navigation and the encoder is detached everywhere, so freezing just
        # avoids building a no-op optimizer over parameters that never get grad.
        args.freeze = True

    env_factory = build_env_factory_from_existing_project(args)
    env = env_factory(0)
    t_role = args.T_role or 20
    return env, "dueling_nu", t_role


def build_trainer(args, env, low_level_backend, t_role, device, tb_logdir=None, tb_runname=None):
    return CTDERAMTrainer(
        obs_dim=getattr(env, "obs_shape", getattr(env, "obs_dim")),
        obs_shape=getattr(env, "obs_shape", None),
        N=getattr(env, "N", args.N),
        K=env.K,
        A=env.A,
        T_role=t_role,
        gamma=args.gamma,
        gamma_role=args.gamma_role,
        device=device,
        seed=args.seed,
        low_level_backend=low_level_backend,
        action_space_n=getattr(env, "action_space_n", None),
        movement_actions=getattr(env, "A", None),
        number_of_features=args.number_of_features,
        nettype=args.nettype,
        archtype=args.archtype,
        dqn_ckpt=args.dqn_ckpt,
        freeze_low_level=args.freeze,
        encoder_ckpt=args.encoder_ckpt,
        heads_ckpt=args.heads_ckpt,
        ram_mode=args.ram_mode,
        max_joint_role_actions=args.max_joint_role_actions,
        global_agg_mode=args.global_agg,
        soft_ram_arch=args.soft_ram_arch,
        soft_ram_temperature=args.soft_ram_temperature,
        w_execution=args.w_execution,
        role_switch_penalty=args.role_switch_penalty,
        role_switch_penalty_rel=args.role_switch_penalty_rel,
        hpr=args.hpr,
        hpr_fraction=args.hpr_fraction,
        hpr_kappa=args.hpr_kappa,
        w_conditioning=args.w_conditioning,
        film_parameterization=args.film_parameterization,
        ram_dueling=args.ram_dueling,
        hfactored_mixer=args.hfactored_mixer,
        role_state_mode=args.role_state_mode,
        normalize_role_rewards=not args.no_reward_normalization,
        role_reward_norm=args.role_reward_norm,
        role_scalarization=args.role_scalarization,
        q_scalarization=args.q_scalarization,
        scalarization_power=args.scalarization_power,
        ewc_p=args.ewc_p,
        ram_reward_mode=args.ram_reward_mode,
        d_model=args.d_model,
        n_attn_heads=args.n_attn_heads,
        n_attn_layers=args.n_attn_layers,
        attn_ff_dim=args.attn_ff_dim,
        preference_role_bias=args.preference_role_bias,
        hard_role_preference_conditioning=args.hard_role_preference_conditioning,
        ppo_epochs=args.ppo_epochs,
        ppo_minibatch_size=args.ppo_minibatch_size,
        ppo_rollout_macro_steps=args.ppo_rollout_macro_steps,
        ppo_clip_eps=args.ppo_clip_eps,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        ppo_critic_mode=args.ppo_critic_mode,
        ppo_critic_popart=args.ppo_critic_popart,
        ppo_advantage_scalarization=args.ppo_advantage_scalarization,
        ppo_target_kl=args.ppo_target_kl,
        ppo_max_grad_norm=args.ppo_max_grad_norm,
        role_q_lr=args.role_q_lr,
        role_q_target_update=args.role_q_target_update,
        role_q_mixer=args.role_q_mixer,
        role_q_gamma=args.role_q_gamma,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_end=args.per_beta_end,
        per_eps=args.per_eps,
        tb_logdir=tb_logdir or "./runs",
        tb_runname=tb_runname or ("ctde_ram_smoke" if args.smoke else f"ctde_ram_{args.env}"),
        batch_low=16 if args.smoke else 64,
        batch_role=8 if args.smoke else 32,
    )


def main():
    args = parse_args()
    if args.weight_alpha <= 0.0:
        raise ValueError("--weight-alpha must be positive")
    if args.weight_alpha_end is not None and args.weight_alpha_end <= 0.0:
        raise ValueError("--weight-alpha-end must be positive")
    if args.weight_alpha_anneal_fraction <= 0.0:
        raise ValueError("--weight-alpha-anneal-fraction must be positive")
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(args.deterministic)
    torch.use_deterministic_algorithms(bool(args.deterministic))
    args.cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    args.pythonhashseed = os.environ.get("PYTHONHASHSEED")

    env, low_level_backend, t_role = build_env(args)
    eval_env, eval_backend, eval_t_role = build_env(args)
    if eval_backend != low_level_backend or eval_t_role != t_role:
        raise RuntimeError("training and evaluation environments resolved inconsistently")
    device = "cpu" if args.device < 0 or not torch.cuda.is_available() else f"cuda:{args.device}"
    args.run_name = make_run_name(args)
    run_dir = resolve_run_dir(args.output_dir, args.run_name)
    if not args.probe_csv:
        args.probe_csv = build_probe_artifact_path(run_dir, args.run_name, args.episodes, "csv")
    if not args.probe_plot:
        args.probe_plot = build_probe_artifact_path(run_dir, args.run_name, args.episodes, "png")
    if not args.probe_pareto_plot:
        args.probe_pareto_plot = build_probe_artifact_path(run_dir, args.run_name, args.episodes, "pareto.png")
    metrics_dir = ensure_dir(os.path.join(run_dir, "metrics"))
    eval_dir = ensure_dir(os.path.join(run_dir, "eval"))
    ckpt_dir = ensure_dir(os.path.join(run_dir, "checkpoints"))
    fig_dir = ensure_dir(os.path.join(run_dir, "figures"))
    write_json(os.path.join(run_dir, "config.json"), args_to_dict(args))
    print_experiment_summary(args, env, low_level_backend, t_role, device, run_dir)

    trainer = build_trainer(
        args,
        env,
        low_level_backend,
        t_role,
        device,
        tb_logdir=run_dir,
        tb_runname="tensorboard",
    )
    trainer.tb.log_hparams_text(args_to_dict(args))

    if trainer.main_hard_role_mode and args.forced_role_diagnostics:
        forced_rows = trainer.evaluate_forced_role_policies(
            eval_env, n_episodes=args.forced_role_episodes, seed_base=args.seed
        )
        write_csv_rows(os.path.join(metrics_dir, "forced_role_diagnostics.csv"), forced_rows)
        write_json(os.path.join(metrics_dir, "forced_role_diagnostics.json"), {"rows": forced_rows})
        for row in forced_rows:
            progress_write(
                f"[forced] {row['forced_policy']} coverage={row['coverage']:.4f} "
                f"cleaned={row['trash_cleaned']:.4f} "
                f"r_clean={row['clean_reward_return']:.2f} r_explore={row['explore_reward_return']:.2f}"
            )

    if trainer.main_hard_role_mode and args.same_state_preference_check:
        preference_report = trainer.diagnose_same_state_preferences(eval_env, seed=args.seed)
        write_json(os.path.join(metrics_dir, "same_state_preference_initial.json"), preference_report)
        progress_write(
            f"[preference-check] initial mean_abs_score_difference="
            f"{preference_report['mean_abs_score_difference']:.8f} "
            f"roles_differ={preference_report['roles_differ']}"
        )

    if args.warmup_episodes > 0:
        print(f"[warmup] episodes={args.warmup_episodes}")
        trainer.warmup_normalizers(env, n_episodes=args.warmup_episodes)

    print(
        f"[config] env={args.env} backend={low_level_backend} ram_mode={trainer.ram_mode} "
        f"global_agg={trainer.global_agg_mode} soft_arch={trainer.soft_ram_arch} role_state={trainer.role_state_mode} "
        f"reward_norm={trainer.role_reward_norm_name} role_scalarization={trainer.role_scalarization} "
        f"q_scalarization={trainer.q_scalarization} gamma_role={trainer.gamma_role:.4f} device={device}"
    )
    write_json(os.path.join(run_dir, "runtime.json"), {
        "device": device,
        "low_level_backend": low_level_backend,
        "T_role": t_role,
        "paired_evaluation_seed_base": int(args.seed),
        "tensorboard_dir": trainer.tb.log_dir,
    })

    if args.check_frozen_popart:
        report = trainer.check_frozen_popart_invariance(eval_env)
        status = "PASS" if report["ok"] else "FAIL"
        print(
            f"[check:frozen_popart] {status} reason={report['reason']} "
            f"q_max_abs_diff={report['q_max_abs_diff']} "
            f"param_max_abs_diff={report['param_max_abs_diff']} "
            f"rescale_flags={report['popart_rescale_flags']}"
        )

    best_hv = -float("inf")
    latest_eval = None
    eval_every = args.eval_every if args.eval_every is not None else (3 if args.smoke else 50)
    eval_episodes = args.eval_episodes if args.eval_episodes is not None else (2 if args.smoke else 3)
    run_config = args_to_dict(args)
    ppo_preference_grid = [
        np.asarray(w, dtype=np.float32)
        for w in ((1.0, 0.0), (0.8, 0.2), (0.5, 0.5), (0.2, 0.8), (0.0, 1.0))
    ]

    def run_eval(eval_episode: int):
        nonlocal best_hv, latest_eval
        if args.eval_points <= 1:
            eval_points = [(1.0, 0.0)]
        else:
            eval_points = [
                (float(1.0 - t), float(t))
                for t in np.linspace(0.0, 1.0, args.eval_points)
            ]
        latest_eval = trainer.evaluate_pareto(
            eval_env,
            scal_grid=[(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)] if args.smoke
            else eval_points,
            n_episodes_per_w=eval_episodes,
        )
        prefix = f"eval_ep_{eval_episode:05d}"
        paths = save_pareto_artifacts(latest_eval, eval_dir, prefix)
        append_csv_row(os.path.join(metrics_dir, "eval_summary.csv"), {
            "episode": eval_episode,
            "hypervolume": float(latest_eval["hypervolume"]),
            "front_size": int(latest_eval["pareto_front"].shape[0]),
            "n_points": int(latest_eval["all_points"].shape[0]),
            "json": paths["json"],
            "csv": paths["csv"],
            "png": paths["png"],
        })
        write_json(os.path.join(metrics_dir, "latest_eval.json"), latest_eval)
        if trainer.main_hard_role_mode:
            sensitivity = trainer.diagnose_same_state_preferences(eval_env, seed=args.seed)
            write_json(
                os.path.join(eval_dir, f"same_state_preference_ep_{eval_episode:05d}.json"),
                sensitivity,
            )
            trainer.tb.log_scalar(
                "eval/same_state_preference_score_delta",
                sensitivity["mean_abs_score_difference"],
                step=eval_episode,
            )
        if float(latest_eval["hypervolume"]) > best_hv:
            best_hv = float(latest_eval["hypervolume"])
            best_path = os.path.join(ckpt_dir, "best_hv.pt")
            trainer.save_checkpoint(
                best_path,
                run_config=run_config,
                episode=eval_episode,
                metrics={"best_hv": best_hv},
            )
            progress_write(f"[checkpoint] best_hv={best_hv:.4f} saved={best_path}")

    if trange is not None:
        episode_iter = trange(args.episodes, desc="train", unit="ep", dynamic_ncols=True)
    else:
        episode_iter = range(args.episodes)

    for ep in episode_iter:
        eps_low = max(0.05, 1.0 - ep / max(1, args.episodes * 0.5))
        if args.ram_mode == "hard_role_q":
            progress = min(ep / max(1, args.episodes * args.role_q_epsilon_fraction), 1.0)
            eps_ram = args.role_q_epsilon_start + progress * (args.role_q_epsilon_end - args.role_q_epsilon_start)
        else:
            eps_ram = max(0.05, 1.0 - ep / max(1, args.episodes * 0.75))
        current_weight_alpha = annealed_weight_alpha(
            args.weight_alpha, args.weight_alpha_end, ep, args.episodes,
            args.weight_alpha_anneal_fraction,
        )
        if args.ram_mode == "ppo_ram" and args.ppo_preference_sampling == "stratified":
            w = ppo_preference_grid[ep % len(ppo_preference_grid)].copy()
        else:
            w = sample_weights(rng, env.K, mode=args.weight_sampling, alpha=current_weight_alpha)
        m = trainer.run_episode(env, w, eps_low, eps_ram)
        head_loss = np.mean(m["head_losses"]) if m["head_losses"] else float("nan")
        ram_loss = np.mean(m["ram_losses"]) if m["ram_losses"] else float("nan")
        train_row = {
            "episode": ep,
            "w0": float(w[0]),
            "w1": float(w[1]) if env.K > 1 else 0.0,
            "weight_alpha": current_weight_alpha,
        }
        train_row.update(m)
        append_csv_row(os.path.join(metrics_dir, "train_episodes.csv"), train_row)
        write_json(os.path.join(metrics_dir, "latest_train_episode.json"), train_row)
        trainer.tb.log_scalar("train/weight_0", float(w[0]), step=ep)
        trainer.tb.log_scalar("train/weight_alpha", current_weight_alpha, step=ep)
        if env.K > 1:
            trainer.tb.log_scalar("train/weight_1", float(w[1]), step=ep)
        trainer.tb.flush()
        if tqdm is not None:
            postfix = dict(
                w=f"({w[0]:.2f},{w[1]:.2f})",
                R=f"{m['total_reward']:.2f}",
                switches=m["n_switches"],
                ram=f"{ram_loss:.4f}",
            )
            if trainer.main_hard_role_mode:
                postfix["role1"] = f"{m['role1_frac']:.3f}"
                if trainer.ram_mode == "ppo_ram":
                    postfix["rollout"] = m["ppo_rollout_size"]
            else:
                postfix["head"] = f"{head_loss:.4f}"
            episode_iter.set_postfix(refresh=False, **postfix)
        else:
            progress_write(
                f"[ep {ep:4d}] w=({w[0]:.2f},{w[1]:.2f}) R={m['total_reward']:.2f} "
                f"switches={m['n_switches']} "
                f"head_loss={head_loss:.4f} "
                f"ram_loss={ram_loss:.4f}"
            )

        if eval_every > 0 and (ep + 1) % eval_every == 0:
            run_eval(ep + 1)

        if args.save_every > 0 and (ep + 1) % args.save_every == 0:
            periodic_path = os.path.join(ckpt_dir, f"episode_{ep + 1:05d}.pt")
            latest_path = os.path.join(ckpt_dir, "latest.pt")
            trainer.save_checkpoint(periodic_path, run_config=run_config, episode=ep, metrics=m)
            trainer.save_checkpoint(latest_path, run_config=run_config, episode=ep, metrics=m)
            progress_write(f"[checkpoint] saved={periodic_path}")

    # Do not discard a final partial mixed-preference PPO rollout.
    final_ppo_update = trainer.flush_main_method_updates()
    if final_ppo_update is not None:
        progress_write(
            f"[ppo] flushed partial rollout loss={final_ppo_update.loss:.6f} "
            f"kl={final_ppo_update.approx_kl:.6f}"
        )

    # Run one additional sweep after all updates so the fully trained policy always
    # has an eval_ep_<episodes> artifact (for example eval_ep_05000).
    if eval_every > 0:
        run_eval(args.episodes)

    if args.check_aggregator_grad:
        grad = trainer.last_global_agg_grad_abs_sum
        if grad is None:
            print("[check:aggregator_grad] FAIL no RAM update occurred; increase episodes or lower batch_role.")
        else:
            status = "PASS" if grad > 0.0 else "FAIL"
            print(f"[check:aggregator_grad] {status} global_agg_grad_abs_sum={grad:.6f}")

    if args.probe_preference_sensitivity:
        if args.probe_points <= 1:
            scal_grid = [(1.0, 0.0)]
        else:
            scal_grid = [(float(1.0 - t), float(t)) for t in np.linspace(0.0, 1.0, args.probe_points)]
        trainer.probe_preference_sensitivity(
            eval_env,
            scal_grid=scal_grid,
            n_episodes_per_w=args.probe_episodes,
            save_csv=args.probe_csv,
            plot_path=args.probe_plot,
            pareto_plot_path=args.probe_pareto_plot,
            attention_path=os.path.join(fig_dir, "hard_role_attention.npz") if trainer.main_hard_role_mode else None,
        )

    final_path = os.path.join(ckpt_dir, "final.pt")
    trainer.save_checkpoint(final_path, run_config=run_config, episode=args.episodes - 1, metrics={"best_hv": best_hv})
    trainer.save_checkpoint(os.path.join(ckpt_dir, "latest.pt"), run_config=run_config, episode=args.episodes - 1, metrics={"best_hv": best_hv})
    write_json(os.path.join(run_dir, "done.json"), {
        "final_checkpoint": final_path,
        "best_hv": best_hv,
        "latest_eval_hv": None if latest_eval is None else float(latest_eval["hypervolume"]),
        "tensorboard_dir": trainer.tb.log_dir,
    })
    print(f"[checkpoint] final={final_path}")
    print(f"[tensorboard] view with: tensorboard --logdir \"{os.path.join(run_dir, 'tensorboard')}\"")
    trainer.close()
    if hasattr(env, "close"):
        env.close()
    if hasattr(eval_env, "close"):
        eval_env.close()
    print("done.")


if __name__ == "__main__":
    main()
