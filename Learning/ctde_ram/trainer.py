"""
CTDERAMTrainer -- CTDE-RAM with a real low-level DuelingDQN option controller.

Hierarchical DRL view:
  - The high level is the Role Assignment Module (RAM). It chooses one role/option
    per agent every T_role steps.
  - In your ASV project the role is exactly nu:
        role 0 = nu=0 = cleaning / intensification
        role 1 = nu=1 = exploration / coverage
  - The low level is a Q-network conditioned by that role. With
    low_level_backend="dueling_nu" this is your DQFDuelingVisualNetwork: one
    shared visual encoder and two dueling heads.

The toy MLP backend is kept for smoke tests, but the project path no longer
pretends that there are separate SharedEncoder and TaskQHead modules.
"""
from __future__ import annotations

import copy
import csv
import itertools
import os
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from .nets import (
        DuelingNuQNetwork,
        SharedEncoder,
        TaskQHead,
        load_dqn_state_if_exists,
        load_state_if_exists,
        set_requires_grad,
    )
    from .popart import PopArtNorm
    from .replay_buffers import LowLevelReplayBuffer, RoleReplayBuffer
    from .global_aggregator import GlobalAggregator
    from .role_selector import FiLMConditioner, RoleSelectorAttention
    from .pareto import sweep_scalarizations
    from .tb_logger import TBLogger
except ImportError:
    from nets import (
        DuelingNuQNetwork,
        SharedEncoder,
        TaskQHead,
        load_dqn_state_if_exists,
        load_state_if_exists,
        set_requires_grad,
    )
    from popart import PopArtNorm
    from replay_buffers import LowLevelReplayBuffer, RoleReplayBuffer
    from global_aggregator import GlobalAggregator
    from role_selector import FiLMConditioner, RoleSelectorAttention
    from pareto import sweep_scalarizations
    from tb_logger import TBLogger


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, dynamic_ncols=True, **kwargs)


def _progress_write(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def _tensor_to_numpy(value, dtype=np.float32):
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    return np.asarray(value, dtype=dtype)


class RewardMinMaxNormalizer:
    """Greedy-agent style component normalizer for role-level scalarization.

    This is not PopArt. PopArt normalizes value targets inside the Q-learning
    heads. This min/max normalizer mirrors the greedy role selector idea: before
    mixing cleaning and exploration rewards with a preference vector, put both
    components on a comparable [0, 1] scale. It handles negative rewards because
    the lower bound is learned during warmup and then updated online.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        self.dim = int(dim)
        self.eps = float(eps)
        self.min_value = np.full(self.dim, np.inf, dtype=np.float32)
        self.max_value = np.full(self.dim, -np.inf, dtype=np.float32)

    @property
    def ready(self) -> bool:
        return bool(np.all(np.isfinite(self.min_value)) and np.all(np.isfinite(self.max_value)))

    def update(self, values) -> None:
        arr = _tensor_to_numpy(values, dtype=np.float32)
        arr = arr.reshape(-1, self.dim)
        self.min_value = np.minimum(self.min_value, arr.min(axis=0))
        self.max_value = np.maximum(self.max_value, arr.max(axis=0))

    def normalize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        if not self.ready:
            return values
        lo = torch.as_tensor(self.min_value, dtype=values.dtype, device=values.device)
        hi = torch.as_tensor(self.max_value, dtype=values.dtype, device=values.device)
        return (values - lo) / torch.clamp(hi - lo, min=self.eps)


class RunningMeanStdNormalizer:
    """Standard reward normalization for the role-level RAM reward.

    This is the normalization name used in the implementation literature: a
    running mean and running standard deviation. It is separate from PopArt.

    Important distinction:
      - PopArt is per low-level Q-head and rescales value targets/outputs.
      - running_mean_std is for the high-level RAM scalar reward only.

    Elicit's final spec asks for this at the RAM level: normalize the scalarized
    fleet reward R_RAM with its own running mean/std, and do not share these
    statistics with the low-level PopArt heads.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    @property
    def ready(self) -> bool:
        return self.count > 1

    @property
    def std(self) -> float:
        if self.count <= 1:
            return 1.0
        return float(np.sqrt(max(self.m2 / (self.count - 1), self.eps)))

    def update(self, values) -> None:
        arr = _tensor_to_numpy(values, dtype=np.float32)
        for x in arr.reshape(-1):
            self.count += 1
            delta = float(x) - self.mean
            self.mean += delta / self.count
            delta2 = float(x) - self.mean
            self.m2 += delta * delta2

    def normalize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        if not self.ready:
            return values
        return (values - self.mean) / (self.std + self.eps)


class CTDERAMTrainer:
    def __init__(
        self,
        obs_dim,
        N: int,
        K: int,
        A: int,
        d_enc: int = 128,
        d_ctx: int = 256,
        d_role: int = 64,
        T_role: int = 20,
        gamma: float = 0.99,
        gamma_role: Optional[float] = None,
        lr_low: float = 3e-4,
        lr_ram: float = 3e-4,
        buf_low_cap: int = 50_000,
        buf_role_cap: int = 5_000,
        batch_low: int = 64,
        batch_role: int = 32,
        target_period_low: int = 500,
        target_period_ram: int = 50,
        popart_alpha: float = 1e-3,
        device: str = "cpu",
        seed: int = 0,
        # Low-level controller options.
        low_level_backend: str = "mlp",
        obs_shape: Optional[Sequence[int]] = None,
        action_space_n: Optional[int] = None,
        movement_actions: Optional[int] = None,
        number_of_features: int = 1024,
        nettype: str = "0",
        archtype: str = "v1",
        role_to_head: Sequence[int] = (1, 0),
        freeze_low_level: bool = False,
        dqn_ckpt: Optional[str] = None,
        encoder_ckpt: Optional[str] = None,
        heads_ckpt: Optional[str] = None,
        # RAM/version switches.
        #
        # ram_mode:
        #   auto      -> hard discrete RAM for small K^N, factored hard RAM for large K^N.
        #   random    -> untrained random hard roles; control baseline for RQ1.
        #   discrete  -> Elicit toy/V1 hard RAM: DQN over all K^N joint role assignments.
        #   factored  -> hard per-agent role values: avoids K^N, still executes one-hot roles.
        #   soft_v2   -> Elicit V2 soft RAM: executes W(N,K) softmax role weights and mixes
        #                PopArt-normalized low-level Q-heads with those weights.
        ram_mode: str = "auto",
        max_joint_role_actions: int = 512,
        # global_agg_mode is separate from soft_ram_arch on purpose:
        #   soft_ram_arch chooses the selector head used by soft_v2.
        #   global_agg_mode asks whether the shared fleet embedding g uses
        #   relational attention or the mean-pool ablation.
        global_agg_mode: str = "attention",
        # Only used when ram_mode="soft_v2".
        #
        # soft_ram_arch:
        #   mlp       -> centralized fixed-size MLP emits N*K role values.
        #   attention -> scalable attention role selector emits per-agent role values
        #                with parameters independent of K^N.
        soft_ram_arch: str = "attention",
        soft_ram_temperature: float = 1.0,
        w_execution: str = "soft",
        role_switch_penalty: float = 0.0,
        hpr: bool = False,
        hpr_fraction: float = 0.5,
        hpr_kappa: float = 1.0,
        w_conditioning: str = "concat",
        # role_state_mode controls the mission/context vector appended to g.
        #
        #   flat   -> previous W is flattened; matches the original toy implementation,
        #             but d_extra grows as N*K.
        #   pooled -> previous W is summarized as role distribution; this is the
        #             scalable Elicit-style state because d_extra is independent of N.
        #   auto   -> pooled for soft_v2+attention, flat otherwise.
        role_state_mode: str = "auto",
        # Reward normalization options.
        normalize_role_rewards: bool = True,
        # role_reward_norm:
        #   minmax           -> your greedy-agent style component min/max normalization.
        #   running_mean_std -> standard RAM scalar reward normalization from the Elicit spec.
        #   none             -> no RAM reward normalization.
        role_reward_norm: str = "minmax",
        # Scalarization switches.
        #
        # role_scalarization chooses how the macro reward vector is collapsed
        # into the scalar target that trains the high-level RAM/RoleSelector.
        #
        # q_scalarization chooses how the K low-level Q-head outputs are
        # collapsed into one action score at execution time.
        #
        # ws   = Weighted Sum, the exact Elicit soft_v2 linear mixture.
        # wp   = Weighted Power from your greedy code.
        # wpop = Weighted Product Of Powers from your greedy code.
        # ewc  = Exponential Weighted Criterion from your greedy code.
        role_scalarization: str = "ws",
        q_scalarization: str = "ws",
        scalarization_power: float = 3.0,
        ewc_p: float = 1.0,
        ram_reward_mode: str = "component_rewards",
        tb_logdir: str = "./runs",
        tb_runname: str = "ctde_ram_v1",
    ):
        self.N, self.K, self.A = int(N), int(K), int(A)
        self.T_role = int(T_role)
        self.gamma = float(gamma)
        # gamma_role is the SMDP/option discount over a T_role-length macro step.
        # Leaving it as gamma**T_role is principled; exposing it makes sensitivity
        # checks explicit when T_role changes.
        self.gamma_role = float(gamma ** T_role if gamma_role is None else gamma_role)
        self.batch_low, self.batch_role = int(batch_low), int(batch_role)
        self.target_period_low = int(target_period_low)
        self.target_period_ram = int(target_period_ram)
        self.device = torch.device(device)
        self.d_ctx = int(d_ctx)
        self.popart_alpha = float(popart_alpha)
        self.freeze_low_level = bool(freeze_low_level)
        self.low_level_backend = low_level_backend
        self.global_agg_mode = global_agg_mode
        self.soft_ram_arch = soft_ram_arch
        self.soft_ram_temperature = float(soft_ram_temperature)
        self.w_execution = w_execution
        self.role_switch_penalty = float(role_switch_penalty)
        if self.role_switch_penalty < 0.0:
            raise ValueError("role_switch_penalty must be non-negative")
        self.hpr = bool(hpr)
        self.hpr_fraction = float(hpr_fraction)
        self.hpr_kappa = float(hpr_kappa)
        if not 0.0 <= self.hpr_fraction <= 1.0:
            raise ValueError("hpr_fraction must be in [0, 1]")
        if self.hpr_kappa <= 0.0:
            raise ValueError("hpr_kappa must be positive")
        self.w_conditioning = w_conditioning
        if self.w_conditioning not in {"concat", "film"}:
            raise ValueError("w_conditioning must be one of: concat, film")
        self.role_scalarization = role_scalarization.lower()
        self.q_scalarization = q_scalarization.lower()
        self.scalarization_power = float(scalarization_power)
        self.ewc_p = float(ewc_p)
        self.scalarization_eps = 1e-8
        self.ram_reward_mode = ram_reward_mode.lower()
        if self.ram_reward_mode not in {"component_rewards", "delta_metrics"}:
            raise ValueError("ram_reward_mode must be one of: component_rewards, delta_metrics")
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        if obs_shape is None:
            obs_shape = tuple(obs_dim) if isinstance(obs_dim, (tuple, list)) else (int(obs_dim),)
        self.obs_shape = tuple(obs_shape)
        self.obs_dim_flat = int(np.prod(self.obs_shape))

        self._build_low_level(
            d_enc=d_enc,
            action_space_n=action_space_n,
            movement_actions=movement_actions,
            number_of_features=number_of_features,
            nettype=nettype,
            archtype=archtype,
            role_to_head=role_to_head,
            dqn_ckpt=dqn_ckpt,
            encoder_ckpt=encoder_ckpt,
            heads_ckpt=heads_ckpt,
        )

        # Joint role combinations for the hard discrete RAM. For large fleets the
        # factored hard RAM avoids K**N outputs by producing N*K role values.
        # The soft_v2 mode is opt-in because it changes the executed policy from
        # one-hot roles to soft role/preference weights.
        self.ROLE_COMBOS = list(itertools.product(range(self.K), repeat=self.N))
        self.n_ram_actions = len(self.ROLE_COMBOS)
        if ram_mode == "auto":
            self.ram_mode = "discrete" if self.n_ram_actions <= max_joint_role_actions else "factored"
        else:
            self.ram_mode = ram_mode
        if self.ram_mode not in {"random", "discrete", "factored", "soft_v2"}:
            raise ValueError("ram_mode must be one of: auto, random, discrete, factored, soft_v2")
        if self.global_agg_mode not in {"attention", "mean_pool"}:
            raise ValueError("global_agg_mode must be one of: attention, mean_pool")
        if self.soft_ram_arch not in {"mlp", "attention"}:
            raise ValueError("soft_ram_arch must be one of: mlp, attention")
        if self.w_execution not in {"soft", "hard_argmax", "st_gumbel"}:
            raise ValueError("w_execution must be one of: soft, hard_argmax, st_gumbel")
        valid_scalarizations = {"ws", "wp", "wpop", "ewc"}
        if self.role_scalarization not in valid_scalarizations:
            raise ValueError("role_scalarization must be one of: ws, wp, wpop, ewc")
        if self.q_scalarization not in valid_scalarizations:
            raise ValueError("q_scalarization must be one of: ws, wp, wpop, ewc")

        if role_state_mode == "auto":
            self.role_state_mode = "pooled" if (self.ram_mode == "soft_v2" and self.soft_ram_arch == "attention") else "flat"
        else:
            self.role_state_mode = role_state_mode
        if self.role_state_mode not in {"flat", "pooled"}:
            raise ValueError("role_state_mode must be one of: auto, flat, pooled")

        # High-level CTDE state: fleet context g plus mission variables.
        # The preference vector is included so one RAM learns the Pareto front.
        if self.role_state_mode == "flat":
            # Toy/Elicit V1 state: exact previous role matrix is available.
            # Dimension grows with fleet size because prev_W is flattened.
            self.d_extra = 3 + self.K + self.N * self.K + self.K
        else:
            # Scalable Elicit-style state: RAM sees the role distribution instead
            # of every agent's previous W row. This keeps the extra state
            # independent of N and is the right companion for attention soft RAM.
            self.d_extra = 3 + self.K + self.K + self.K
        self.d_role_state = self.d_ctx + self.d_extra

        self.global_agg = GlobalAggregator(self.d_enc, d_ctx, mode=self.global_agg_mode).to(self.device)
        self.role_selector = RoleSelectorAttention(
            self.d_enc, d_ctx, self.K, d_role, d_extra=self.d_extra,
            w_conditioning=self.w_conditioning,
        ).to(self.device)
        self.role_selector_tgt = copy.deepcopy(self.role_selector).eval()
        ram_out = self.n_ram_actions if self.ram_mode == "discrete" else self.N * self.K
        self.ram_q = nn.Sequential(
            nn.Linear(self.d_role_state, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, ram_out),
        ).to(self.device)
        self.ram_q_tgt = copy.deepcopy(self.ram_q).eval()
        self.ram_film = (
            FiLMConditioner(self.K, 256).to(self.device)
            if self.w_conditioning == "film" else None
        )
        self.ram_film_tgt = copy.deepcopy(self.ram_film).eval() if self.ram_film is not None else None

        self.bufs_low = [LowLevelReplayBuffer(buf_low_cap, self.obs_shape) for _ in range(self.K)]
        self.buf_role = RoleReplayBuffer(buf_role_cap, self.N, self.d_enc, self.d_extra, self.K)

        if not self.freeze_low_level:
            low_params = [p for p in self._low_level_parameters() if p.requires_grad]
            self.optim_low = torch.optim.Adam(low_params, lr=lr_low)
        else:
            self.optim_low = None
        ram_params = self._ram_trainable_parameters()
        self.optim_ram = torch.optim.Adam(ram_params, lr=lr_ram) if ram_params else None

        # Backwards compatibility: older code passed normalize_role_rewards=False.
        # That now maps to role_reward_norm="none". Prefer using the explicit
        # role_reward_norm parameter in new experiments so the method is named
        # in logs/configs.
        if not normalize_role_rewards:
            role_reward_norm = "none"
        self.role_reward_norm_name = role_reward_norm
        if self.role_reward_norm_name not in {"minmax", "running_mean_std", "none"}:
            raise ValueError("role_reward_norm must be one of: minmax, running_mean_std, none")
        if self.role_reward_norm_name == "minmax":
            self.role_reward_norm = RewardMinMaxNormalizer(self.K)
        elif self.role_reward_norm_name == "running_mean_std":
            self.role_reward_norm = RunningMeanStdNormalizer()
        else:
            self.role_reward_norm = None

        self.step_count_low = 0
        self.step_count_ram = 0
        # Diagnostics filled after each RAM update. The first one is the exact
        # dead-path check: if this stays None/0 after updates, the aggregator is
        # not receiving gradient from replayed z_all.
        self.last_global_agg_grad_abs_sum = None
        self.last_ram_grad_abs_sum = None
        self.tb = TBLogger(log_dir=tb_logdir, run_name=tb_runname)

    # ---------- network construction ----------
    def _build_low_level(
        self,
        d_enc: int,
        action_space_n: Optional[int],
        movement_actions: Optional[int],
        number_of_features: int,
        nettype: str,
        archtype: str,
        role_to_head: Sequence[int],
        dqn_ckpt: Optional[str],
        encoder_ckpt: Optional[str],
        heads_ckpt: Optional[str],
    ) -> None:
        if self.low_level_backend == "mlp":
            if len(self.obs_shape) != 1:
                raise ValueError("The toy MLP backend expects flat observations. Use dueling_nu for image observations.")
            self.d_enc = int(d_enc)
            self.encoder = SharedEncoder(self.obs_dim_flat, self.d_enc).to(self.device)
            self.heads = nn.ModuleList([TaskQHead(self.d_enc, self.A) for _ in range(self.K)]).to(self.device)
            load_state_if_exists(self.encoder, encoder_ckpt, self.device)
            if heads_ckpt:
                load_state_if_exists(self.heads, heads_ckpt, self.device)
            if self.freeze_low_level:
                set_requires_grad(self.encoder, False)
                set_requires_grad(self.heads, False)
                self.encoder.eval()
                self.heads.eval()
            self.encoder_tgt = copy.deepcopy(self.encoder).eval()
            self.heads_tgt = copy.deepcopy(self.heads).eval()
            self.popart = [
                PopArtNorm(self.heads[k].out, alpha=self.popart_alpha, rescale=not self.freeze_low_level)
                for k in range(self.K)
            ]
            return

        if self.low_level_backend == "dueling_nu":
            movement_actions = int(movement_actions or self.A)
            action_space_n = int(action_space_n or movement_actions * self.K)
            self.low_level = DuelingNuQNetwork(
                obs_shape=self.obs_shape,
                action_space_n=action_space_n,
                movement_actions=movement_actions,
                number_of_features=number_of_features,
                archtype=archtype,
                nettype=nettype,
                role_to_head=role_to_head,
            ).to(self.device)
            self.A = self.low_level.A
            self.d_enc = self.low_level.d_enc
            if self.K != self.low_level.K:
                raise ValueError(f"K={self.K} does not match role_to_head length {self.low_level.K}")
            load_dqn_state_if_exists(self.low_level, dqn_ckpt, self.device)
            if self.freeze_low_level:
                set_requires_grad(self.low_level, False)
                self.low_level.eval()
            self.low_level_tgt = copy.deepcopy(self.low_level).eval()
            self.popart = [
                PopArtNorm(self.low_level.role_output_adapter(k), alpha=self.popart_alpha, rescale=not self.freeze_low_level)
                for k in range(self.K)
            ]
            return

        raise ValueError("low_level_backend must be 'mlp' or 'dueling_nu'")

    def _low_level_parameters(self):
        if self.low_level_backend == "mlp":
            return list(self.encoder.parameters()) + list(self.heads.parameters())
        return list(self.low_level.parameters())

    def _ram_trainable_parameters(self):
        """Parameters trained by the high-level RAM optimizer.

        This switch is intentionally explicit because it is the main place where
        experiments differ:
          - random: train no RAM parameters; only samples hard roles uniformly.
          - discrete/factored: train the MLP role-value network `ram_q`.
          - soft_v2 + mlp: train the same MLP, but execute softmax W instead of
            one-hot argmax roles.
          - soft_v2 + attention: train RoleSelectorAttention, the scalable
            soft RAM from the final Elicit recommendation.
        """
        if self.ram_mode == "random":
            return []
        params = list(self.global_agg.parameters())
        if self.ram_mode == "soft_v2" and self.soft_ram_arch == "attention":
            params += list(self.role_selector.parameters())
        else:
            params += list(self.ram_q.parameters())
            if self.ram_film is not None:
                params += list(self.ram_film.parameters())
        return params

    def _sync_ram_target(self) -> None:
        if self.ram_mode == "random":
            return
        if self.ram_mode == "soft_v2" and self.soft_ram_arch == "attention":
            self.role_selector_tgt.load_state_dict(self.role_selector.state_dict())
        else:
            self.ram_q_tgt.load_state_dict(self.ram_q.state_dict())
            if self.ram_film is not None:
                self.ram_film_tgt.load_state_dict(self.ram_film.state_dict())

    def _sync_low_target(self) -> None:
        if self.low_level_backend == "mlp":
            self.encoder_tgt.load_state_dict(self.encoder.state_dict())
            self.heads_tgt.load_state_dict(self.heads.state_dict())
        else:
            self.low_level_tgt.load_state_dict(self.low_level.state_dict())

    def _obs_batch_tensor(self, obs_all) -> torch.Tensor:
        return torch.as_tensor(np.stack(obs_all), dtype=torch.float32, device=self.device)

    def _encode_batch(self, obs_t: torch.Tensor, target: bool = False) -> torch.Tensor:
        if self.low_level_backend == "mlp":
            return (self.encoder_tgt if target else self.encoder)(obs_t)
        return (self.low_level_tgt if target else self.low_level).encode(obs_t)

    def _q_role(self, obs_t: torch.Tensor, role: int, target: bool = False) -> torch.Tensor:
        if self.low_level_backend == "mlp":
            enc = self._encode_batch(obs_t, target=target)
            return (self.heads_tgt if target else self.heads)[role](enc)
        return (self.low_level_tgt if target else self.low_level).q_role(obs_t, role)

    # ---------- helpers ----------
    def _encode_all(self, obs_all):
        return self._encode_batch(self._obs_batch_tensor(obs_all))

    def _env_value(self, env, name: str, default: float = 0.0) -> float:
        value = getattr(env, name, default)
        if callable(value):
            value = value()
        return float(value)

    def _build_extra(self, env, r_accum, prev_W, scal_weights):
        # The extra vector is the non-neural mission context appended to the
        # learned fleet embedding g.
        #
        # flat:
        #   stores every previous W[i,k] exactly. This is closest to the toy
        #   Elicit code and is useful for N fixed at training/evaluation time.
        #
        # pooled:
        #   stores only mean_i W[i,k], i.e. the current role distribution. This
        #   is the scalable Elicit-style state because its dimension does not
        #   grow with the number of ASVs.
        prev_role_summary = (
            prev_W.flatten()
            if self.role_state_mode == "flat"
            else prev_W.mean(dim=0)
        )
        return torch.cat([
            torch.tensor([
                self._env_value(env, "coverage_pct", 0.0),
                self._env_value(env, "trash_density", 0.0),
                self._env_value(env, "budget_frac", 1.0),
            ], dtype=torch.float32, device=self.device),
            r_accum.to(self.device),
            prev_role_summary.to(self.device),
            torch.as_tensor(scal_weights, dtype=torch.float32, device=self.device),
        ])

    def _stack_rewards(self, r_vecs) -> torch.Tensor:
        return torch.stack([
            torch.as_tensor(r, dtype=torch.float32, device=self.device) for r in r_vecs
        ])

    def _metric_reward_components(self, prev_metrics, curr_metrics):
        cleaned_prev = float(prev_metrics.get("trash_cleaned", 0.0))
        coverage_prev = float(prev_metrics.get("coverage", 0.0))
        cleaned_now = float(curr_metrics.get("trash_cleaned", 0.0))
        coverage_now = float(curr_metrics.get("coverage", 0.0))
        comps = torch.zeros(self.K, dtype=torch.float32, device=self.device)
        comps[0] = max(cleaned_now - cleaned_prev, 0.0)
        comps[1] = max(coverage_now - coverage_prev, 0.0)
        return comps

    def _compute_step_reward_components(self, env, prev_metrics, r_vecs=None, info=None, curr_metrics=None):
        if self.ram_reward_mode == "delta_metrics":
            if curr_metrics is None:
                coverage_now = self._env_value(env, "coverage_pct", float(info.get("coverage", 0.0)) if info else 0.0)
                cleaned_now = self._env_value(env, "trash_cleaned_pct", float(info.get("trash_cleaned", 0.0)) if info else 0.0)
                curr_metrics = {"trash_cleaned": cleaned_now, "coverage": coverage_now}
            return self._metric_reward_components(prev_metrics, curr_metrics)

        reward_mat = self._stack_rewards(r_vecs)
        return reward_mat.sum(dim=0)

    def _roles_to_W(self, roles) -> torch.Tensor:
        roles = np.asarray(roles, dtype=np.int64)
        W = torch.zeros(self.N, self.K, device=self.device)
        W[torch.arange(self.N, device=self.device), torch.as_tensor(roles, dtype=torch.long, device=self.device)] = 1.0
        return W

    def _roles_to_combo_idx(self, roles_t: torch.Tensor) -> torch.Tensor:
        idx = torch.zeros(roles_t.shape[0], dtype=torch.long, device=roles_t.device)
        for agent_i in range(self.N):
            idx = idx * self.K + roles_t[:, agent_i].long()
        return idx

    def _grad_abs_sum(self, parameters) -> float:
        """Diagnostic gradient sum, deliberately simple to compare across runs."""
        total = 0.0
        for p in parameters:
            if p.grad is not None:
                total += float(p.grad.detach().abs().sum().item())
        return total

    def _normalize_weights_tensor(self, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.to(self.device)
        return weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=self.scalarization_eps)

    def _pow_for_weighted_power(self, values: torch.Tensor) -> torch.Tensor:
        # Your greedy WP uses p_wp=3.0. Integer powers are safe for negative raw
        # rewards; fractional powers are only valid on non-negative inputs.
        nearest_int = round(self.scalarization_power)
        if abs(self.scalarization_power - nearest_int) < 1e-8:
            return values.pow(int(nearest_int))
        return values.clamp_min(self.scalarization_eps).pow(self.scalarization_power)

    def _scalarize_components(
        self,
        values: torch.Tensor,
        weights: torch.Tensor,
        method: str,
    ) -> torch.Tensor:
        """Scalarize a vector of objective values with one named method.

        `values` has objective dimension last: (..., K).
        `weights` can be (K,) or broadcastable to (..., K).

        These formulas mirror the greedy selector in Learning/utils.py:
          ws   -> sum_i w_i r_i
          wp   -> sum_i w_i r_i^p
          wpop -> product_i r_i^w_i
          ewc  -> sum_i (exp(p*w_i)-1) * exp(p*r_i)

        For WPOP and fractional WP the inputs must be non-negative. The main
        paper setting should therefore pair nonlinear scalarizations with
        --role-reward-norm minmax. For Q-values, the caller normalizes per
        action set before using nonlinear methods.
        """
        method = method.lower()
        values = values.to(self.device)
        weights = self._normalize_weights_tensor(weights.to(dtype=values.dtype, device=values.device))
        while weights.dim() < values.dim():
            weights = weights.unsqueeze(0)

        if method == "ws":
            return (weights * values).sum(dim=-1)

        if method == "wp":
            return (weights * self._pow_for_weighted_power(values)).sum(dim=-1)

        if method == "wpop":
            safe_values = values.clamp_min(self.scalarization_eps)
            return torch.prod(safe_values.pow(weights), dim=-1)

        if method == "ewc":
            # Clamp only as a numerical guard. With minmax-normalized inputs the
            # values are already in [0, 1], which is the intended greedy/EWC use.
            safe_values = torch.clamp(values, min=-20.0, max=20.0)
            weight_gain = torch.exp(self.ewc_p * weights) - 1.0
            value_gain = torch.exp(self.ewc_p * safe_values)
            return (weight_gain * value_gain).sum(dim=-1)

        raise ValueError(f"Unknown scalarization method: {method}")

    def _normalize_q_for_nonlinear_scalarization(self, q_by_role: torch.Tensor) -> torch.Tensor:
        """State-local Q normalization for nonlinear action scalarization.

        Q heads after PopArt are comparable, but they can still be negative.
        WP/WPOP/EWC in your greedy selector were designed for normalized reward
        components, so before applying those methods at the Q exit we map each
        head's action values to [0, 1] within the current state.
        """
        q_min = q_by_role.min(dim=-1, keepdim=True).values
        q_max = q_by_role.max(dim=-1, keepdim=True).values
        return (q_by_role - q_min) / torch.clamp(q_max - q_min, min=self.scalarization_eps)

    def _scalarize_q_values(self, q_by_role: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Collapse K role/objective Q heads into one score per action.

        q_by_role: (K, A)
        weights:   (K,) from the current RAM output row W[i].

        q_scalarization=ws is the original Elicit line:
            Q_comp(a) = sum_k W[k] * PopArtNorm(Q_k(a))

        q_scalarization in {wp,wpop,ewc} is the learned/vector-Q style ablation:
        normalize the per-head action scores to [0,1], then apply the same
        nonlinear scalarizer used by the greedy baselines.
        """
        if self.q_scalarization == "ws":
            return self._scalarize_components(
                q_by_role.transpose(0, 1),
                weights,
                method="ws",
            )
        q_norm01 = self._normalize_q_for_nonlinear_scalarization(q_by_role)
        return self._scalarize_components(
            q_norm01.transpose(0, 1),
            weights,
            method=self.q_scalarization,
        )

    def _scalarize_role_reward(self, fleet_r_raw: torch.Tensor, scal_weights: torch.Tensor, update_norm: bool) -> torch.Tensor:
        # This function is deliberately method-switched because the paper
        # interpretation changes depending on which normalizer you choose.
        #
        # minmax:
        #   Your greedy-agent style normalization. It normalizes each objective
        #   component first, then scalarizes with the preference vector.
        #
        # running_mean_std:
        #   Standard reward normalization for RAM. It scalarizes first, then
        #   normalizes the scalar R_RAM with a running mean/std. This is the
        #   high-level reward normalization described in the final Elicit spec.
        #
        # none:
        #   Raw scalarized fleet reward, useful for ablations.
        raw_scalar = self._scalarize_components(
            fleet_r_raw,
            scal_weights,
            method=self.role_scalarization,
        )
        if self.role_reward_norm_name == "none":
            return raw_scalar
        if self.role_reward_norm_name == "minmax":
            if update_norm:
                self.role_reward_norm.update(fleet_r_raw)
            fleet_for_ram = self.role_reward_norm.normalize_tensor(fleet_r_raw)
            return self._scalarize_components(
                fleet_for_ram,
                scal_weights,
                method=self.role_scalarization,
            )
        if update_norm:
            self.role_reward_norm.update(raw_scalar)
        return self.role_reward_norm.normalize_tensor(raw_scalar)

    # ---------- warmup ----------
    @torch.no_grad()
    def warmup_normalizers(self, env, n_episodes: int = 5, max_steps_per_episode: Optional[int] = None) -> None:
        """Warm up PopArt and reward min/max stats before learning.

        This is the CTDE-RAM equivalent of the greedy agent's reward-statistics
        warmup. For frozen pretrained heads, PopArt updates only mu/sigma and
        does not mutate the DQN weights.
        """
        if n_episodes <= 0:
            return
        # In Expert_nu mode the low-level PopArt heads are never read (Expert_nu
        # owns navigation), so warmup only seeds the RAM reward statistics and
        # drives the env with random soft roles instead of movement actions.
        expert_nu_mode = self._env_uses_expert_nu(env)
        for _ in _progress(range(int(n_episodes)), desc="warmup", unit="ep", leave=False):
            obs_all = env.reset()
            prev_metrics = {
                "trash_cleaned": self._env_value(env, "trash_cleaned_pct", 0.0),
                "coverage": self._env_value(env, "coverage_pct", 0.0),
            }
            done = False
            steps = 0
            while not done:
                if expert_nu_mode:
                    W_np = self.rng.dirichlet(
                        np.ones(self.K, dtype=np.float32), size=self.N
                    ).astype(np.float32)
                    obs_next, r_vecs, done, info = env.step(W_np)
                else:
                    obs_t = self._obs_batch_tensor(obs_all)
                    for k in range(self.K):
                        self.popart[k].update(self._q_role(obs_t, k).reshape(-1))

                    actions = self.rng.integers(0, self.A, size=self.N).tolist()
                    obs_next, r_vecs, done, info = env.step(actions)
                curr_metrics = {
                    "trash_cleaned": self._env_value(env, "trash_cleaned_pct", float(info.get("trash_cleaned", 0.0)) if info else 0.0),
                    "coverage": self._env_value(env, "coverage_pct", float(info.get("coverage", 0.0)) if info else 0.0),
                }
                fleet_r = self._compute_step_reward_components(env, prev_metrics, r_vecs=r_vecs, info=info, curr_metrics=curr_metrics)
                prev_metrics = curr_metrics
                if self.role_reward_norm_name == "minmax":
                    self.role_reward_norm.update(fleet_r)
                elif self.role_reward_norm_name == "running_mean_std":
                    # During warmup no Pareto preference is active yet, so use
                    # equal weights just to seed the scalar R_RAM statistics.
                    equal_w = torch.ones(self.K, dtype=torch.float32, device=self.device) / self.K
                    self.role_reward_norm.update(
                        self._scalarize_components(
                            fleet_r,
                            equal_w,
                            method=self.role_scalarization,
                        )
                    )

                obs_all = obs_next
                steps += 1
                if max_steps_per_episode is not None and steps >= max_steps_per_episode:
                    break

    # ---------- action selection (low level) ----------
    @torch.no_grad()
    def select_action(self, obs_i, w_i, epsilon: float):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.A))
        obs_t = torch.as_tensor(obs_i, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_stack = []
        for k in range(self.K):
            q_raw = self._q_role(obs_t, k).squeeze(0)
            q_stack.append(self.popart[k].normalize(q_raw))
        Q_norm = torch.stack(q_stack, dim=0)
        Q_comp = self._scalarize_q_values(Q_norm, w_i.to(self.device))
        return int(Q_comp.argmax().item())

    def _soft_role_q_values(self, z_batch, extra_batch, target: bool = False):
        """Return per-agent role values for soft/factored RAM modes.

        Shape:
          z_batch:     (B, N, d_enc)
          extra_batch: (B, d_extra)
          output:      (B, N, K)

        For soft_v2+attention this is the scalable Elicit path:
        RoleSelectorAttention reads each agent encoding plus shared context and
        emits role values with parameters independent of K**N.
        """
        if self.ram_mode == "soft_v2" and self.soft_ram_arch == "attention":
            g = self.global_agg(z_batch)
            selector = self.role_selector_tgt if target else self.role_selector
            return selector.logits(z_batch, g, extra=extra_batch)

        g = self.global_agg(z_batch)
        role_state = torch.cat([g, extra_batch], dim=-1)
        return self._ram_q_values(role_state, target=target).view(-1, self.N, self.K)

    def _ram_q_values(self, role_state: torch.Tensor, target: bool = False) -> torch.Tensor:
        q_net = self.ram_q_tgt if target else self.ram_q
        film = self.ram_film_tgt if target else self.ram_film
        if film is None:
            return q_net(role_state)
        h = q_net[1](q_net[0](role_state))
        gamma, beta = film(role_state[:, -self.K:])
        h = gamma * h + beta
        for layer in q_net[2:]:
            h = layer(h)
        return h

    def _soft_weights_from_q(self, q_values: torch.Tensor) -> torch.Tensor:
        # Elicit V2 execution: W is a softmax role/preference distribution per
        # agent. Lower temperature -> closer to hard roles. Higher -> smoother.
        return F.softmax(q_values / max(self.soft_ram_temperature, 1e-6), dim=-1)

    def _execution_weights(self, W: torch.Tensor, training: bool) -> torch.Tensor:
        """Map selector weights to the weights actually executed by the env."""
        if self.w_execution == "soft":
            return W
        if self.w_execution == "hard_argmax" or not training:
            return F.one_hot(W.argmax(dim=-1), num_classes=self.K).to(W.dtype)
        # In this off-policy DQN backup, the straight-through gradient benefit is partial.
        logits = torch.log(W.clamp_min(1e-8))
        return F.gumbel_softmax(
            logits,
            tau=max(self.soft_ram_temperature, 1e-6),
            hard=True,
            dim=-1,
        )

    def _role_switch_cost(
        self,
        new_W: torch.Tensor,
        previous_W: torch.Tensor,
        new_executed_W: torch.Tensor,
        previous_executed_W: torch.Tensor,
    ) -> float:
        if self.role_switch_penalty == 0.0:
            return 0.0
        if self.ram_mode == "soft_v2" and self.w_execution == "soft":
            distance = torch.norm(new_W - previous_W, p=1)
        else:
            distance = (new_executed_W.argmax(dim=-1) != previous_executed_W.argmax(dim=-1)).sum()
        return self.role_switch_penalty * float(distance.item())

    def _env_uses_expert_nu(self, env) -> bool:
        """True when env.step expects W/nu instead of movement actions."""
        return getattr(env, "ctde_action_mode", "movement_actions") == "role_weights"

    def _step_env_with_current_W(self, env, obs_all, W: torch.Tensor, epsilon_low: float):
        """Execute one low-level step under the currently selected high-level W.

        Normal CTDE-RAM path:
          W chooses how to mix/choose DQN heads; trainer sends movement actions.

        Expert_nu path:
          W itself is the action. The env adapter reads W[:,1] as exploration
          probability nu and Expert_nu converts that nu to movement actions.
          In this mode low-level replay is intentionally disabled because the
          path planner is the fixed Expert_nu checkpoint, not a learner here.
        """
        if self._env_uses_expert_nu(env):
            obs_next, r_vecs, done, info = env.step(_tensor_to_numpy(W, dtype=np.float32))
            return obs_next, r_vecs, done, info, None

        actions = [self.select_action(obs_all[i], W[i], epsilon_low) for i in range(self.N)]
        obs_next, r_vecs, done, info = env.step(actions)
        return obs_next, r_vecs, done, info, actions

    # ---------- role selection (high level RAM) ----------
    @torch.no_grad()
    def select_roles(self, z_all, extra, epsilon_ram: float):
        if self.ram_mode == "random":
            # RQ1 control: choose a hard role per agent uniformly at random.
            # This intentionally ignores epsilon_ram and ignores all RAM networks.
            # Use --freeze with this mode when you want a pure selector baseline
            # over the exact same fixed low-level DQN heads.
            roles = self.rng.integers(0, self.K, size=self.N, dtype=np.int64)
            return roles, self._roles_to_W(roles)

        g = self.global_agg(z_all.unsqueeze(0)).squeeze(0)
        role_state = torch.cat([g, extra], dim=0)

        if self.ram_mode == "discrete":
            if self.rng.random() < epsilon_ram:
                combo_idx = int(self.rng.integers(0, self.n_ram_actions))
            else:
                combo_idx = int(self._ram_q_values(role_state.unsqueeze(0)).squeeze(0).argmax().item())
            roles = np.asarray(self.ROLE_COMBOS[combo_idx], dtype=np.int64)
            return roles, self._roles_to_W(roles)

        if self.ram_mode == "soft_v2":
            if self.rng.random() < epsilon_ram:
                # Exploration for a soft RAM should also be soft, otherwise V2
                # collapses into the hard ablation during epsilon-heavy early
                # training. Dirichlet(1) gives a uniform random simplex row.
                W_np = self.rng.dirichlet(np.ones(self.K, dtype=np.float32), size=self.N).astype(np.float32)
                W = torch.as_tensor(W_np, dtype=torch.float32, device=self.device)
            else:
                q_roles = self._soft_role_q_values(
                    z_all.unsqueeze(0),
                    extra.unsqueeze(0),
                    target=False,
                ).squeeze(0)
                W = self._soft_weights_from_q(q_roles)
            roles = _tensor_to_numpy(W.argmax(dim=-1), dtype=np.int64)
            return roles, W

        if self.rng.random() < epsilon_ram:
            roles = self.rng.integers(0, self.K, size=self.N, dtype=np.int64)
        else:
            q = self._ram_q_values(role_state.unsqueeze(0)).view(self.N, self.K)
            roles = _tensor_to_numpy(q.argmax(dim=-1), dtype=np.int64)
        return roles, self._roles_to_W(roles)

    # ---------- low-level update ----------
    def update_head(self, k: int):
        if self.freeze_low_level or self.bufs_low[k].size < self.batch_low:
            return None
        obs, acts, rews, next_obs, dones = self.bufs_low[k].sample(self.batch_low)
        obs, next_obs = obs.to(self.device), next_obs.to(self.device)
        acts, rews, dones = acts.to(self.device), rews.to(self.device), dones.to(self.device)

        with torch.no_grad():
            q_next_online = self._q_role(next_obs, k, target=False)
            best = q_next_online.argmax(dim=1)
            q_next_tgt = self._q_role(next_obs, k, target=True)
            q_next_sel = q_next_tgt.gather(1, best.unsqueeze(1)).squeeze(1)
            q_next_raw = self.popart[k].denormalize(q_next_sel)
            td_target_raw = rews + self.gamma * q_next_raw * (1.0 - dones)
            td_target_norm = self.popart[k].normalize_target(td_target_raw)

        q_pred = self._q_role(obs, k, target=False).gather(1, acts.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_pred, td_target_norm)
        self.optim_low.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._low_level_parameters(), 10.0)
        self.optim_low.step()
        self.popart[k].update(td_target_raw)

        self.step_count_low += 1
        if self.step_count_low % self.target_period_low == 0:
            self._sync_low_target()

        lv = float(loss.item())
        self.tb.log_step(f"train/head_{k}_loss", lv)
        self.tb.log_step(f"train/head_{k}_sigma", self.popart[k].sigma)
        self.tb.step()
        return lv

    # ---------- RAM update ----------
    def update_ram(self):
        if self.ram_mode == "random":
            self.last_global_agg_grad_abs_sum = None
            self.last_ram_grad_abs_sum = None
            return None
        if self.buf_role.size < self.batch_role:
            return None
        z, extra, W_action, R, r_components, zn, en, d = self.buf_role.sample(self.batch_role)
        z, extra = z.to(self.device), extra.to(self.device)
        zn, en = zn.to(self.device), en.to(self.device)
        W_action = W_action.to(self.device)
        roles = W_action.argmax(dim=-1).long()
        R, d = R.to(self.device), d.to(self.device)
        r_components = r_components.to(self.device)

        if self.hpr and self.hpr_fraction > 0.0:
            relabel_count = min(self.batch_role, int(round(self.batch_role * self.hpr_fraction)))
            if relabel_count > 0:
                relabel_idx = torch.randperm(self.batch_role, device=self.device)[:relabel_count]
                w_orig = extra[relabel_idx, -self.K:]
                concentration = self.hpr_kappa * w_orig.clamp_min(0.0) + 1e-6
                w_relabel = torch.distributions.Dirichlet(concentration).sample()
                extra = extra.clone()
                en = en.clone()
                R = R.clone()
                extra[relabel_idx, -self.K:] = w_relabel
                en[relabel_idx, -self.K:] = w_relabel
                R[relabel_idx] = self._scalarize_role_reward(
                    r_components[relabel_idx], w_relabel, update_norm=False
                )

        g = self.global_agg(z)
        role_state = torch.cat([g, extra], dim=-1)

        with torch.no_grad():
            g_next = self.global_agg(zn)
            rs_next = torch.cat([g_next, en], dim=-1)
            if self.ram_mode == "discrete":
                q_next_online = self._ram_q_values(rs_next)
                best = q_next_online.argmax(dim=1)
                q_next_tgt = self._ram_q_values(rs_next, target=True)
                q_next_sel = q_next_tgt.gather(1, best.unsqueeze(1)).squeeze(1)
            elif self.ram_mode == "soft_v2":
                # Soft Double-DQN-style backup:
                #   online network chooses the soft W_next,
                #   target network evaluates that W_next.
                # This is the continuous/soft analogue of argmax+gather used by
                # the hard RAM, and is what lets V2 actually execute W(N,K).
                q_next_online = self._soft_role_q_values(zn, en, target=False)
                W_next = self._soft_weights_from_q(q_next_online)
                q_next_tgt = self._soft_role_q_values(zn, en, target=True)
                q_next_sel = (W_next * q_next_tgt).sum(dim=(1, 2))
            else:
                q_next_online = self._ram_q_values(rs_next).view(-1, self.N, self.K)
                best_roles = q_next_online.argmax(dim=-1, keepdim=True)
                q_next_tgt = self._ram_q_values(rs_next, target=True).view(-1, self.N, self.K)
                q_next_sel = q_next_tgt.gather(2, best_roles).squeeze(-1).sum(dim=1)
            target = R + self.gamma_role * q_next_sel * (1.0 - d)

        if self.ram_mode == "discrete":
            action_idx = self._roles_to_combo_idx(roles)
            q_pred = self._ram_q_values(role_state).gather(1, action_idx.unsqueeze(1)).squeeze(1)
        elif self.ram_mode == "soft_v2":
            q_roles = self._soft_role_q_values(z, extra, target=False)
            q_pred = (W_action * q_roles).sum(dim=(1, 2))
        else:
            q_roles = self._ram_q_values(role_state).view(-1, self.N, self.K)
            q_pred = q_roles.gather(2, roles.unsqueeze(-1)).squeeze(-1).sum(dim=1)

        loss = F.smooth_l1_loss(q_pred, target)
        self.optim_ram.zero_grad()
        loss.backward()
        self.last_global_agg_grad_abs_sum = self._grad_abs_sum(self.global_agg.parameters())
        self.last_ram_grad_abs_sum = self._grad_abs_sum(self._ram_trainable_parameters())
        nn.utils.clip_grad_norm_(self._ram_trainable_parameters(), 10.0)
        self.optim_ram.step()

        self.step_count_ram += 1
        if self.step_count_ram % self.target_period_ram == 0:
            self._sync_ram_target()

        lv = float(loss.item())
        self.tb.log_step("train/ram_loss", lv)
        self.tb.log_step("train/global_agg_grad_abs_sum", self.last_global_agg_grad_abs_sum)
        return lv

    # ---------- one training episode ----------
    def run_episode(self, env, scal_weights, epsilon_low: float, epsilon_ram: float):
        scal_weights = np.asarray(scal_weights, dtype=np.float32)
        scal_weights = scal_weights / max(float(scal_weights.sum()), 1e-8)
        sw_t = torch.as_tensor(scal_weights, dtype=torch.float32, device=self.device)

        obs_all = env.reset()
        done = False
        ep_step = 0
        R_role = 0.0
        window_start_metrics = {
            "trash_cleaned": self._env_value(env, "trash_cleaned_pct", 0.0),
            "coverage": self._env_value(env, "coverage_pct", 0.0),
        }
        r_accum = torch.zeros(self.K, device=self.device)
        window_r_components = torch.zeros(self.K, device=self.device)
        prev_W = torch.zeros(self.N, self.K, device=self.device)

        z_all = self._encode_all(obs_all).detach()
        extra = self._build_extra(env, r_accum, prev_W, scal_weights)
        roles, W = self.select_roles(z_all, extra, epsilon_ram)
        executed_W = self._execution_weights(W, training=True)
        window_switch_penalty = 0.0
        role_counts = np.zeros(self.K, dtype=np.float64)
        W_sum = np.zeros(self.K, dtype=np.float64)
        executed_role_counts = np.zeros(self.K, dtype=np.float64)
        executed_W_sum = np.zeros(self.K, dtype=np.float64)
        n_role_decisions = 0
        role_counts += np.bincount(_tensor_to_numpy(W.argmax(dim=-1), dtype=np.int64), minlength=self.K)
        W_sum += _tensor_to_numpy(W, dtype=np.float32).mean(axis=0)
        executed_role_counts += np.bincount(
            _tensor_to_numpy(executed_W.argmax(dim=-1), dtype=np.int64), minlength=self.K
        )
        executed_W_sum += _tensor_to_numpy(executed_W, dtype=np.float32).mean(axis=0)
        n_role_decisions += 1
        z_all_prev, extra_prev = z_all, extra

        m = dict(
            total_reward=0.0,
            head_losses=[],
            ram_losses=[],
            ram_grad_abs_sums=[],
            global_agg_grad_abs_sums=[],
            n_switches=0,
            role_switch_penalties=[],
        )

        while not done:
            obs_next, r_vecs, done, info, actions = self._step_env_with_current_W(
                env, obs_all, executed_W, epsilon_low
            )
            reward_mat = self._stack_rewards(r_vecs)
            curr_metrics = {
                "trash_cleaned": self._env_value(env, "trash_cleaned_pct", float(info.get("trash_cleaned", 0.0)) if info else 0.0),
                "coverage": self._env_value(env, "coverage_pct", float(info.get("coverage", 0.0)) if info else 0.0),
            }
            if self.ram_reward_mode == "delta_metrics":
                fleet_r = self._compute_step_reward_components(env, window_start_metrics, r_vecs=r_vecs, info=info, curr_metrics=curr_metrics)
            else:
                fleet_r = self._compute_step_reward_components(env, window_start_metrics, r_vecs=r_vecs, info=info)

            # Expert_nu mode returns actions=None: the path planner is a fixed
            # checkpoint, so there is no low-level head to train and no low-level
            # replay to fill. The RAM still learns from the role-level buffer.
            if actions is not None:
                for i in range(self.N):
                    for k in range(self.K):
                        self.bufs_low[k].store(
                            obs_all[i],
                            actions[i],
                            float(reward_mat[i, k].item()),
                            obs_next[i],
                            done,
                        )

            if self.ram_reward_mode == "delta_metrics":
                r_accum += fleet_r
            else:
                window_r_components += fleet_r
                R_step = float(self._scalarize_role_reward(fleet_r, sw_t, update_norm=True).item())
                R_role += R_step
                m["total_reward"] += R_step

            for k in range(self.K):
                lv = self.update_head(k)
                if lv is not None:
                    m["head_losses"].append(lv)

            obs_all = obs_next
            ep_step += 1
            window_start_metrics = curr_metrics

            if (ep_step % self.T_role == 0) or done:
                if self.ram_reward_mode == "delta_metrics":
                    fleet_r = r_accum
                    window_r_components = r_accum.clone()
                    R_step = float(self._scalarize_role_reward(fleet_r, sw_t, update_norm=True).item())
                    R_role = R_step
                    m["total_reward"] += R_step
                learning_R_role = R_role - window_switch_penalty
                z_next = self._encode_all(obs_all).detach()
                extra_next = self._build_extra(env, r_accum, W, scal_weights)
                self.buf_role.store(
                    _tensor_to_numpy(z_all_prev, dtype=np.float32),
                    _tensor_to_numpy(extra_prev, dtype=np.float32),
                    _tensor_to_numpy(W, dtype=np.float32),
                    learning_R_role,
                    _tensor_to_numpy(z_next, dtype=np.float32),
                    _tensor_to_numpy(extra_next, dtype=np.float32),
                    done,
                    r_components=_tensor_to_numpy(window_r_components, dtype=np.float32),
                )

                lr = self.update_ram()
                if lr is not None:
                    m["ram_losses"].append(lr)
                    m["ram_grad_abs_sums"].append(float(self.last_ram_grad_abs_sum or 0.0))
                    m["global_agg_grad_abs_sums"].append(float(self.last_global_agg_grad_abs_sum or 0.0))

                new_roles, new_W = self.select_roles(z_next, extra_next, epsilon_ram)
                new_executed_W = self._execution_weights(new_W, training=True)
                next_window_switch_penalty = self._role_switch_cost(
                    new_W, W, new_executed_W, executed_W
                )
                m["role_switch_penalties"].append(next_window_switch_penalty)
                if not torch.equal(new_executed_W, executed_W):
                    m["n_switches"] += 1
                z_all_prev, extra_prev = z_next, extra_next
                roles, W, prev_W = new_roles, new_W, new_W
                executed_W = new_executed_W
                window_switch_penalty = next_window_switch_penalty
                role_counts += np.bincount(_tensor_to_numpy(W.argmax(dim=-1), dtype=np.int64), minlength=self.K)
                W_sum += _tensor_to_numpy(W, dtype=np.float32).mean(axis=0)
                executed_role_counts += np.bincount(
                    _tensor_to_numpy(executed_W.argmax(dim=-1), dtype=np.int64), minlength=self.K
                )
                executed_W_sum += _tensor_to_numpy(executed_W, dtype=np.float32).mean(axis=0)
                n_role_decisions += 1
                window_start_metrics = curr_metrics
                R_role = 0.0
                r_accum = torch.zeros(self.K, device=self.device)
                window_r_components = torch.zeros(self.K, device=self.device)

        role_frac = role_counts / max(float(role_counts.sum()), 1.0)
        mean_W = W_sum / max(float(n_role_decisions), 1.0)
        executed_role_frac = executed_role_counts / max(float(executed_role_counts.sum()), 1.0)
        mean_executed_W = executed_W_sum / max(float(n_role_decisions), 1.0)
        m.update({
            "n_steps": ep_step,
            "epsilon_low": float(epsilon_low),
            "epsilon_ram": float(epsilon_ram),
            "final_coverage": self._env_value(env, "coverage_pct", 0.0),
            "final_trash_cleaned": self._env_value(env, "trash_cleaned_pct", 0.0),
            "role_decisions": int(n_role_decisions),
            "buf_role_size": int(self.buf_role.size),
            "buf_low_min_size": int(min(buf.size for buf in self.bufs_low)) if self.bufs_low else 0,
            "last_global_agg_grad_abs_sum": float(self.last_global_agg_grad_abs_sum or 0.0),
            "last_ram_grad_abs_sum": float(self.last_ram_grad_abs_sum or 0.0),
            "mean_role_switch_penalty": float(np.mean(m["role_switch_penalties"]))
            if m["role_switch_penalties"] else 0.0,
        })
        for k in range(self.K):
            m[f"role{k}_frac"] = float(role_frac[k])
            m[f"role{k}_mean_W"] = float(mean_W[k])
            m[f"executed_role{k}_frac"] = float(executed_role_frac[k])
            m[f"executed_role{k}_mean_W"] = float(mean_executed_W[k])
        if self.K == 2:
            m["mean_nu_role1"] = float(role_frac[1])
        self.tb.log_episode_metrics(m, prefix="train")
        return m

    # ---------- greedy single-weight evaluation ----------
    @torch.no_grad()
    def _evaluate_single(self, env, scal_weights, n_episodes: int = 5, show_progress: bool = False, progress_desc: Optional[str] = None):
        scal_weights = np.asarray(scal_weights, dtype=np.float32)
        scal_weights = scal_weights / max(float(scal_weights.sum()), 1e-8)
        results = []
        role_counts = np.zeros(self.K, dtype=np.int64)
        episode_iter = range(n_episodes)
        if show_progress:
            episode_iter = _progress(
                episode_iter,
                desc=progress_desc or "eval episodes",
                unit="ep",
                leave=False,
            )
        for _ in episode_iter:
            obs_all = env.reset()
            done = False
            ep_step = 0
            r_accum = torch.zeros(self.K, device=self.device)
            prev_W = torch.zeros(self.N, self.K, device=self.device)
            z_all = self._encode_all(obs_all).detach()
            extra = self._build_extra(env, r_accum, prev_W, scal_weights)
            _, W = self.select_roles(z_all, extra, epsilon_ram=0.0)
            executed_W = self._execution_weights(W, training=False)
            metrics = dict(coverage=0.0, trash_cleaned=0.0, n_switches=0)
            info = {}
            while not done:
                obs_all, r_vecs, done, info, _ = self._step_env_with_current_W(
                    env, obs_all, executed_W, epsilon_low=0.0
                )
                r_accum += self._stack_rewards(r_vecs).sum(dim=0)
                ep_step += 1
                if (ep_step % self.T_role == 0) or done:
                    z_next = self._encode_all(obs_all).detach()
                    extra = self._build_extra(env, r_accum, W, scal_weights)
                    _, new_W = self.select_roles(z_next, extra, epsilon_ram=0.0)
                    new_executed_W = self._execution_weights(new_W, training=False)
                    if not torch.equal(new_executed_W, executed_W):
                        metrics["n_switches"] += 1
                    W = new_W
                    executed_W = new_executed_W
                    for k in _tensor_to_numpy(executed_W.argmax(dim=-1), dtype=np.int64):
                        role_counts[k] += 1
                    r_accum = torch.zeros(self.K, device=self.device)
            metrics["coverage"] = self._env_value(env, "coverage_pct", 0.0)
            metrics["trash_cleaned"] = self._env_value(env, "trash_cleaned_pct", float(info.get("trash_cleaned", 0.0)))
            results.append(metrics)
        avg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
        avg["_role_counts"] = role_counts
        return avg

    # ---------- diagnostics ----------
    @torch.no_grad()
    def check_frozen_popart_invariance(self, env):
        """Verify that frozen low-level Q-values do not move during PopArt updates.

        Why this exists:
          PopArt normally rescales the output layer when its statistics change.
          That is correct when the Q head is trainable. With --freeze, however,
          the pretrained DQN is the fixed substrate for selector comparisons, so
          PopArt must update only mu/sigma and leave the raw Q network untouched.

        This check updates PopArt statistics on one fixed batch, compares raw
        Q-values and parameters before/after, then restores the PopArt stats so
        the diagnostic does not perturb the following experiment.
        """
        if not self.freeze_low_level:
            return {
                "ok": False,
                "reason": "freeze_low_level is False; run this check with --freeze.",
                "q_max_abs_diff": None,
                "param_max_abs_diff": None,
                "popart_rescale_flags": [bool(pa.rescale) for pa in self.popart],
            }

        obs_all = env.reset()
        obs_t = self._obs_batch_tensor(obs_all)
        q_before = [self._q_role(obs_t, k).detach().clone() for k in range(self.K)]
        params_before = [p.detach().clone() for p in self._low_level_parameters()]
        popart_state = [(pa.mu, pa.mu_sq, pa.sigma) for pa in self.popart]

        for k in range(self.K):
            self.popart[k].update(q_before[k].reshape(-1))

        q_after = [self._q_role(obs_t, k).detach().clone() for k in range(self.K)]
        params_after = [p.detach().clone() for p in self._low_level_parameters()]

        # Keep the check non-invasive: it proves the update behavior, then puts
        # the normalizer statistics back exactly where they were.
        for pa, (mu, mu_sq, sigma) in zip(self.popart, popart_state):
            pa.mu = mu
            pa.mu_sq = mu_sq
            pa.sigma = sigma

        q_max_abs_diff = max(float((a - b).abs().max().item()) for a, b in zip(q_before, q_after))
        param_max_abs_diff = max(float((a - b).abs().max().item()) for a, b in zip(params_before, params_after))
        flags = [bool(pa.rescale) for pa in self.popart]
        ok = (not any(flags)) and q_max_abs_diff <= 1e-8 and param_max_abs_diff <= 1e-8
        return {
            "ok": ok,
            "reason": "raw Q-values and frozen parameters stayed fixed" if ok else "frozen PopArt invariance failed",
            "q_max_abs_diff": q_max_abs_diff,
            "param_max_abs_diff": param_max_abs_diff,
            "popart_rescale_flags": flags,
        }

    @torch.no_grad()
    def probe_preference_sensitivity(
        self,
        env,
        scal_grid=None,
        n_episodes_per_w: int = 3,
        save_csv: Optional[str] = None,
        plot_path: Optional[str] = None,
    ):
        """Sweep preference weights and report whether the selected roles move.

        This is the CTDE-RAM version of the old track_nu sanity gate. For each
        weight vector w, it records every high-level role decision:
          - argmax role fractions: hard nu/role counts, comparable to track_nu.
          - mean W: the actual executed role weights, important for soft_v2.

        If these columns are flat from w=(1,0) to w=(0,1), the selector is not
        preference-sensitive even if a Pareto/hypervolume number looks decent.
        """
        if scal_grid is None:
            scal_grid = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]

        rows = []
        for scal_weights in _progress(list(scal_grid), desc="probe weights", unit="w", leave=False):
            scal_weights = np.asarray(scal_weights, dtype=np.float32)
            scal_weights = scal_weights / max(float(scal_weights.sum()), 1e-8)
            W_records = []
            role_records = []
            coverages = []
            cleaned = []

            weight_label = ",".join(f"{float(x):.2f}" for x in scal_weights)
            for _ in _progress(
                range(int(n_episodes_per_w)),
                desc=f"probe episodes w=({weight_label})",
                unit="ep",
                leave=False,
            ):
                obs_all = env.reset()
                done = False
                ep_step = 0
                r_accum = torch.zeros(self.K, device=self.device)
                prev_W = torch.zeros(self.N, self.K, device=self.device)
                info = {}

                z_all = self._encode_all(obs_all).detach()
                extra = self._build_extra(env, r_accum, prev_W, scal_weights)
                _, W = self.select_roles(z_all, extra, epsilon_ram=0.0)
                executed_W = self._execution_weights(W, training=False)
                W_records.append(_tensor_to_numpy(executed_W, dtype=np.float32))
                role_records.append(_tensor_to_numpy(executed_W.argmax(dim=-1), dtype=np.int64))

                while not done:
                    obs_all, r_vecs, done, info, _ = self._step_env_with_current_W(
                        env, obs_all, executed_W, epsilon_low=0.0
                    )
                    r_accum += self._stack_rewards(r_vecs).sum(dim=0)
                    ep_step += 1

                    if (ep_step % self.T_role == 0) or done:
                        z_next = self._encode_all(obs_all).detach()
                        extra = self._build_extra(env, r_accum, W, scal_weights)
                        _, W = self.select_roles(z_next, extra, epsilon_ram=0.0)
                        executed_W = self._execution_weights(W, training=False)
                        W_records.append(_tensor_to_numpy(executed_W, dtype=np.float32))
                        role_records.append(_tensor_to_numpy(executed_W.argmax(dim=-1), dtype=np.int64))
                        r_accum = torch.zeros(self.K, device=self.device)

                coverages.append(self._env_value(env, "coverage_pct", 0.0))
                cleaned.append(self._env_value(env, "trash_cleaned_pct", float(info.get("trash_cleaned", 0.0))))

            W_arr = np.stack(W_records, axis=0)
            role_arr = np.stack(role_records, axis=0)
            role_counts = np.bincount(role_arr.reshape(-1), minlength=self.K).astype(np.float32)
            role_frac = role_counts / max(float(role_counts.sum()), 1.0)
            mean_W = W_arr.mean(axis=(0, 1))

            row = {
                "w0": float(scal_weights[0]),
                "w1": float(scal_weights[1]) if self.K > 1 else 0.0,
                "coverage": float(np.mean(coverages)),
                "trash_cleaned": float(np.mean(cleaned)),
                "n_role_decisions": int(role_arr.shape[0]),
            }
            for k in range(self.K):
                row[f"role{k}_argmax_frac"] = float(role_frac[k])
                row[f"role{k}_mean_weight"] = float(mean_W[k])
            if self.K == 2:
                # In this project role 1 is the exploration/coverage nu. This
                # single column makes flat-nu failures obvious at a glance.
                row["mean_nu_role1"] = float(role_frac[1])
            rows.append(row)

        _progress_write("[probe] preference sensitivity")
        for row in rows:
            argmax_bits = " ".join(
                f"role{k}_frac={row[f'role{k}_argmax_frac']:.3f}"
                for k in range(self.K)
            )
            mean_bits = " ".join(
                f"meanW{k}={row[f'role{k}_mean_weight']:.3f}"
                for k in range(self.K)
            )
            _progress_write(
                f"   w=({row['w0']:.2f},{row['w1']:.2f}) "
                f"{argmax_bits} {mean_bits} "
                f"coverage={row['coverage']:.3f} cleaned={row['trash_cleaned']:.3f}"
            )

        if save_csv:
            dir_name = os.path.dirname(save_csv)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(save_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            _progress_write(f"[probe] wrote csv: {save_csv}")

        if plot_path:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                dir_name = os.path.dirname(plot_path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                x = [row["w0"] for row in rows]
                fig, ax = plt.subplots(figsize=(6, 4))
                for k in range(self.K):
                    y = [row[f"role{k}_argmax_frac"] for row in rows]
                    ax.plot(x, y, marker="o", label=f"role {k} argmax frac")
                ax.set_xlabel("w0")
                ax.set_ylabel("selected role fraction")
                ax.set_ylim(-0.05, 1.05)
                ax.invert_xaxis()
                ax.legend(loc="best")
                ax.set_title("Preference sensitivity")
                fig.tight_layout()
                fig.savefig(plot_path, dpi=160)
                plt.close(fig)
                _progress_write(f"[probe] wrote plot: {plot_path}")
            except Exception as ex:
                _progress_write(f"[probe] plot skipped: {ex!r}")

        return rows

    # ---------- Pareto sweep ----------
    @torch.no_grad()
    def evaluate_pareto(
        self,
        env,
        scal_grid=((1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)),
        n_episodes_per_w: int = 3,
        objective_keys=("coverage", "trash_cleaned"),
        ref_point=(0.0, 0.0),
    ):
        scal_grid = list(scal_grid)
        eval_bar = tqdm(
            total=len(scal_grid),
            desc="eval weights",
            unit="w",
            dynamic_ncols=True,
            leave=False,
        ) if tqdm is not None else None

        def _evaluator(w):
            weight_label = ",".join(f"{float(x):.2f}" for x in w)
            mm = self._evaluate_single(
                env,
                w,
                n_episodes=n_episodes_per_w,
                show_progress=tqdm is not None,
                progress_desc=f"eval episodes w=({weight_label})",
            )
            self.tb.log_role_histogram(mm.pop("_role_counts", np.zeros(self.K, dtype=np.int64)))
            if eval_bar is not None:
                eval_bar.set_postfix(
                    coverage=f"{mm['coverage']:.3f}",
                    cleaned=f"{mm['trash_cleaned']:.3f}",
                    refresh=False,
                )
                eval_bar.update(1)
            return mm

        try:
            result = sweep_scalarizations(_evaluator, scal_grid, objective_keys, ref_point)
        finally:
            if eval_bar is not None:
                eval_bar.close()
        self.tb.log_pareto(result)
        _progress_write(
            f"[eval] hypervolume={result['hypervolume']:.4f} "
            f"front_size={result['pareto_front'].shape[0]} n_points={result['all_points'].shape[0]}"
        )
        for w, mm in result["per_weight"]:
            _progress_write(
                f"   w={tuple(round(float(x), 2) for x in w)} "
                f"coverage={mm['coverage']:.3f} cleaned={mm['trash_cleaned']:.3f}"
            )
        return result

    # ---------- checkpointing ----------
    def _popart_state(self):
        return [
            {
                "mu": float(pa.mu),
                "mu_sq": float(pa.mu_sq),
                "sigma": float(pa.sigma),
                "alpha": float(pa.alpha),
                "rescale": bool(pa.rescale),
            }
            for pa in self.popart
        ]

    def _load_popart_state(self, states) -> None:
        if not states:
            return
        for pa, state in zip(self.popart, states):
            pa.mu = float(state.get("mu", pa.mu))
            pa.mu_sq = float(state.get("mu_sq", pa.mu_sq))
            pa.sigma = float(state.get("sigma", pa.sigma))

    def _role_reward_norm_state(self):
        if self.role_reward_norm is None:
            return None
        if self.role_reward_norm_name == "minmax":
            return {
                "name": "minmax",
                "min_value": self.role_reward_norm.min_value.tolist(),
                "max_value": self.role_reward_norm.max_value.tolist(),
            }
        if self.role_reward_norm_name == "running_mean_std":
            return {
                "name": "running_mean_std",
                "count": int(self.role_reward_norm.count),
                "mean": float(self.role_reward_norm.mean),
                "m2": float(self.role_reward_norm.m2),
            }
        return None

    def _load_role_reward_norm_state(self, state) -> None:
        if not state or self.role_reward_norm is None:
            return
        if self.role_reward_norm_name == "minmax":
            self.role_reward_norm.min_value = np.asarray(state["min_value"], dtype=np.float32)
            self.role_reward_norm.max_value = np.asarray(state["max_value"], dtype=np.float32)
        elif self.role_reward_norm_name == "running_mean_std":
            self.role_reward_norm.count = int(state.get("count", self.role_reward_norm.count))
            self.role_reward_norm.mean = float(state.get("mean", self.role_reward_norm.mean))
            self.role_reward_norm.m2 = float(state.get("m2", self.role_reward_norm.m2))

    def _low_level_state(self):
        if self.low_level_backend == "mlp":
            return {
                "encoder": self.encoder.state_dict(),
                "heads": self.heads.state_dict(),
                "encoder_tgt": self.encoder_tgt.state_dict(),
                "heads_tgt": self.heads_tgt.state_dict(),
            }
        return {
            "low_level": self.low_level.state_dict(),
            "low_level_tgt": self.low_level_tgt.state_dict(),
        }

    def _load_low_level_state(self, state) -> None:
        if not state:
            return
        if self.low_level_backend == "mlp":
            self.encoder.load_state_dict(state["encoder"])
            self.heads.load_state_dict(state["heads"])
            self.encoder_tgt.load_state_dict(state.get("encoder_tgt", state["encoder"]))
            self.heads_tgt.load_state_dict(state.get("heads_tgt", state["heads"]))
        else:
            self.low_level.load_state_dict(state["low_level"])
            self.low_level_tgt.load_state_dict(state.get("low_level_tgt", state["low_level"]))

    def checkpoint_state(self, run_config: Optional[dict] = None, episode: Optional[int] = None, metrics: Optional[dict] = None):
        """Return a full training/evaluation checkpoint payload."""
        optim_state = {}
        if self.optim_low is not None:
            optim_state["low"] = self.optim_low.state_dict()
        if self.optim_ram is not None:
            optim_state["ram"] = self.optim_ram.state_dict()
        return {
            "format": "ctde_ram_checkpoint_v1",
            "episode": episode,
            "metrics": metrics or {},
            "run_config": run_config or {},
            "trainer_runtime": {
                "N": self.N,
                "K": self.K,
                "A": self.A,
                "T_role": self.T_role,
                "gamma": self.gamma,
                "gamma_role": self.gamma_role,
                "ram_mode": self.ram_mode,
                "soft_ram_arch": self.soft_ram_arch,
                "soft_ram_temperature": self.soft_ram_temperature,
                "w_execution": self.w_execution,
                "role_switch_penalty": self.role_switch_penalty,
                "hpr": self.hpr,
                "hpr_fraction": self.hpr_fraction,
                "hpr_kappa": self.hpr_kappa,
                "w_conditioning": self.w_conditioning,
                "role_state_mode": self.role_state_mode,
                "role_reward_norm": self.role_reward_norm_name,
                "role_scalarization": self.role_scalarization,
                "q_scalarization": self.q_scalarization,
                "ram_reward_mode": self.ram_reward_mode,
                "global_agg_mode": self.global_agg_mode,
                "low_level_backend": self.low_level_backend,
                "freeze_low_level": self.freeze_low_level,
                "obs_shape": self.obs_shape,
                "d_enc": self.d_enc,
                "d_ctx": self.d_ctx,
                "d_extra": self.d_extra,
            },
            "steps": {
                "low": int(self.step_count_low),
                "ram": int(self.step_count_ram),
                "tb_episode": int(self.tb.episode),
                "tb_global_step": int(self.tb.global_step),
            },
            "models": {
                "global_agg": self.global_agg.state_dict(),
                "role_selector": self.role_selector.state_dict(),
                "role_selector_tgt": self.role_selector_tgt.state_dict(),
                "ram_q": self.ram_q.state_dict(),
                "ram_q_tgt": self.ram_q_tgt.state_dict(),
                "ram_film": None if self.ram_film is None else self.ram_film.state_dict(),
                "ram_film_tgt": None if self.ram_film_tgt is None else self.ram_film_tgt.state_dict(),
                "low_level": self._low_level_state(),
            },
            "normalizers": {
                "popart": self._popart_state(),
                "role_reward": self._role_reward_norm_state(),
            },
            "replay_buffers": {
                "role": self.buf_role.state_dict(),
            },
            "optimizers": optim_state,
        }

    def save_checkpoint(self, path: str, run_config: Optional[dict] = None, episode: Optional[int] = None, metrics: Optional[dict] = None) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.checkpoint_state(run_config=run_config, episode=episode, metrics=metrics), path)
        return path

    def load_checkpoint(self, path: str, load_optimizers: bool = False, map_location: Optional[str] = None):
        ckpt = torch.load(path, map_location=map_location or self.device, weights_only=False)
        models = ckpt["models"]
        self.global_agg.load_state_dict(models["global_agg"])
        self.role_selector.load_state_dict(models["role_selector"])
        self.role_selector_tgt.load_state_dict(models.get("role_selector_tgt", models["role_selector"]))
        self.ram_q.load_state_dict(models["ram_q"])
        self.ram_q_tgt.load_state_dict(models.get("ram_q_tgt", models["ram_q"]))
        if self.ram_film is not None and models.get("ram_film") is not None:
            self.ram_film.load_state_dict(models["ram_film"])
            self.ram_film_tgt.load_state_dict(models.get("ram_film_tgt", models["ram_film"]))
        self._load_low_level_state(models.get("low_level"))
        self._load_popart_state(ckpt.get("normalizers", {}).get("popart"))
        self._load_role_reward_norm_state(ckpt.get("normalizers", {}).get("role_reward"))
        self.buf_role.load_state_dict(ckpt.get("replay_buffers", {}).get("role"))

        steps = ckpt.get("steps", {})
        self.step_count_low = int(steps.get("low", self.step_count_low))
        self.step_count_ram = int(steps.get("ram", self.step_count_ram))
        if load_optimizers:
            optimizers = ckpt.get("optimizers", {})
            if self.optim_low is not None and "low" in optimizers:
                self.optim_low.load_state_dict(optimizers["low"])
            if self.optim_ram is not None and "ram" in optimizers:
                self.optim_ram.load_state_dict(optimizers["ram"])
        return ckpt

    def close(self):
        self.tb.close()
