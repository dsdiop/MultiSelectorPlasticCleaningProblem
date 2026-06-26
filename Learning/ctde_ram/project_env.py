"""
Adapter from the existing MultiAgentPatrolling project to CTDE-RAM.

The trainer uses the hierarchical view:
  RAM role 0 = nu=0 = cleaning
  RAM role 1 = nu=1 = exploration

Two execution substrates are supported:
  - dqn_heads: CTDE-RAM uses the DQFDueling heads directly and sends movement
    actions to MultiAgentPatrolling.
  - expert_nu: CTDE-RAM sends W/nu to Expert_nu, and Expert_nu converts nu into
    movement actions with the loaded path planner. This mirrors Learning.utils.make_env.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Sequence

import numpy as np
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

PATH_PLANNER_ROOT = os.path.join(PROJECT_ROOT, "Learning", "path_planner_algorithms")


def resolve_path_planner_ckpt(path_planner_folder=None, dqn_ckpt=None, path_planner_root=None) -> str | None:
    """Resolve the DQFDueling checkpoint used by Expert_nu and/or the trainer.

    Preferred project CLI:
      --path-planner-folder Experimento_clean28_malaga_port_...

    That resolves to:
      Learning/path_planner_algorithms/<folder>/Final_Policy.pth

    --dqn-ckpt is still accepted as a direct checkpoint path for compatibility.
    """
    if dqn_ckpt:
        path = os.path.abspath(dqn_ckpt)
        if not os.path.exists(path):
            raise FileNotFoundError(f"--dqn-ckpt does not exist: {path}")
        return path

    if not path_planner_folder:
        return None

    root = os.path.abspath(path_planner_root or PATH_PLANNER_ROOT)
    folder = str(path_planner_folder)
    if folder.endswith(".pth"):
        # Be permissive: if someone passes a direct .pth here, use it. The normal
        # paper path should still be the folder name.
        path = folder if os.path.isabs(folder) else os.path.abspath(folder)
    else:
        path = os.path.join(root, folder, "Final_Policy.pth")

    if not os.path.exists(path):
        available = []
        if os.path.isdir(root):
            available = sorted(
                name for name in os.listdir(root)
                if os.path.isdir(os.path.join(root, name))
            )
        msg = f"Path planner checkpoint does not exist: {path}"
        if available:
            msg += "\nAvailable --path-planner-folder values:\n  " + "\n  ".join(available[:30])
        raise FileNotFoundError(msg)
    return path


def default_initial_positions(map_name: str, n_agents: int) -> np.ndarray:
    defaults = {
        "malaga_port": np.array([[12, 7], [14, 5], [16, 3], [18, 1]], dtype=int),
        "alamillo_lake": np.array([[68, 26], [64, 26], [60, 26], [56, 26]], dtype=int),
        "ypacarai_map": np.array([[24, 21], [28, 24], [27, 19], [24, 24]], dtype=int),
    }
    if map_name not in defaults:
        raise ValueError(
            f"No default initial positions for map '{map_name}'. Pass --initial-positions."
        )
    return defaults[map_name][:n_agents].copy()


class ProjectPatrollingCTDEEnv:
    """Thin CTDE contract around MultiAgentPatrolling.

    Contract exposed to CTDERAMTrainer:
      reset() -> [obs_i]
      step([movement_action_i]) -> ([obs_i], [reward_vec_i], done, info)

    reward_vec_i is returned in role order:
      index 0: cleaning
      index 1: exploration

    The underlying project environment returns reward vectors as
    [exploration, cleaning], so the default reward_component_order=(1, 0)
    intentionally flips them.
    """

    def __init__(self, env, reward_component_order: Sequence[int] = (1, 0)):
        self.env = env
        self.N = int(env.number_of_agents)
        self.K = len(tuple(reward_component_order))
        self.A = 8
        self.reward_component_order = tuple(int(i) for i in reward_component_order)
        self.obs_shape = tuple(env.observation_space.shape)
        self.obs_dim = self.obs_shape
        self.action_space_n = int(env.action_space.n)
        self._last_obs = [np.zeros(self.obs_shape, dtype=np.float32) for _ in range(self.N)]

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _active_ids(self):
        active = getattr(self.env, "active_agents", None)
        if active is None:
            return list(range(self.N))
        if isinstance(active, dict):
            return [int(i) for i, flag in active.items() if flag]
        return [int(i) for i, flag in enumerate(active) if flag]

    def _format_obs(self, obs_dict) -> list[np.ndarray]:
        out = []
        for agent_id in range(self.N):
            if isinstance(obs_dict, dict) and agent_id in obs_dict:
                obs = obs_dict[agent_id]
            else:
                obs = self._last_obs[agent_id]
            obs = np.asarray(obs)
            if getattr(self.env, "convert_to_uint8", False):
                obs = obs.astype(np.float32) / 255.0
            else:
                obs = obs.astype(np.float32)
            self._last_obs[agent_id] = obs
            out.append(obs)
        return out

    def _format_rewards(self, reward_dict) -> list[np.ndarray]:
        rewards = []
        for agent_id in range(self.N):
            raw = np.asarray(
                reward_dict.get(agent_id, np.zeros(max(self.reward_component_order) + 1, dtype=np.float32)),
                dtype=np.float32,
            )
            if raw.size <= max(self.reward_component_order):
                padded = np.zeros(max(self.reward_component_order) + 1, dtype=np.float32)
                padded[: raw.size] = raw
                raw = padded
            rewards.append(raw[list(self.reward_component_order)].astype(np.float32))
        return rewards

    def reset(self):
        obs, _ = self.env.reset()
        return self._format_obs(obs)

    def step(self, actions):
        action_dict = {agent_id: int(actions[agent_id]) for agent_id in self._active_ids()}
        obs, reward, done_dict, info = self.env.step(action_dict)
        done = bool(all(done_dict.values())) if isinstance(done_dict, dict) else bool(done_dict)
        info = dict(info or {})
        info["trash_cleaned"] = self.trash_cleaned_pct()
        info["coverage"] = self.coverage_pct()
        info["reward_components_role_order"] = self._format_rewards(reward)
        return self._format_obs(obs), self._format_rewards(reward), done, info

    def coverage_pct(self) -> float:
        return float(getattr(self.env, "percentage_of_map_visited", 0.0))

    def trash_cleaned_pct(self) -> float:
        return float(getattr(self.env, "percentage_of_trash_cleaned", 0.0))

    def trash_density(self) -> float:
        gt = getattr(self.env, "gt", None)
        gt_map = getattr(gt, "map", None)
        if gt_map is None:
            return 0.0
        visitable = np.asarray(self.env.scenario_map).astype(bool)
        denom = max(int(np.count_nonzero(visitable)), 1)
        return float(np.count_nonzero(np.asarray(gt_map)[visitable] > 0) / denom)

    def budget_frac(self) -> float:
        distances = np.asarray(self.env.fleet.get_distances(), dtype=np.float32)
        remaining = 1.0 - distances / max(float(self.env.distance_budget), 1.0)
        return float(np.clip(remaining, 0.0, 1.0).mean())

    def close(self):
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


class ProjectPatrollingExpertNuCTDEEnv(ProjectPatrollingCTDEEnv):
    """CTDE contract that executes roles through Expert_nu.

    Contract exposed to CTDERAMTrainer:
      reset() -> [raw per-agent obs_i]
      step(W_or_nu) -> ([obs_i], [reward_vec_i], done, info)

    W_or_nu can be:
      - W with shape (N, 2): W[:, 1] is the exploration probability nu.
      - roles/probabilities with shape (N,): 0=cleaning, 1=exploration.

    This mirrors MultiAgentNuWrapper.nu_to_actions but keeps observations as the
    per-agent image tensors needed by the RAM encoder.
    """

    ctde_action_mode = "role_weights"

    def __init__(self, env, expert_nu, reward_component_order: Sequence[int] = (1, 0), seed: int = 0):
        super().__init__(env, reward_component_order=reward_component_order)
        self.expert_nu = expert_nu
        self.rng = np.random.default_rng(seed)

    def _role_weights_to_nu(self, role_weights) -> np.ndarray:
        arr = np.asarray(role_weights, dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[0] != self.N:
                raise ValueError(f"Expected W with N={self.N} rows, got shape {arr.shape}")
            if arr.shape[1] < 2:
                raise ValueError("Expert_nu execution expects K>=2 so W[:,1] is exploration probability.")
            nu = arr[:, 1]
        else:
            if arr.shape[0] != self.N:
                raise ValueError(f"Expected nu vector length N={self.N}, got shape {arr.shape}")
            nu = arr
        return np.clip(nu.astype(np.float32), 0.0, 1.0)

    def _state_float32(self, active_ids):
        state = getattr(self.env, "state", {})
        out = {}
        for agent_id in active_ids:
            obs = state[agent_id]
            if getattr(self.env, "convert_to_uint8", False):
                obs = (obs / 255.0).astype(np.float32)
            else:
                obs = obs.astype(np.float32)
            out[agent_id] = obs
        return out

    def _nu_to_actions(self, nu: np.ndarray) -> dict[int, int]:
        active_ids = self._active_ids()
        condition = np.zeros(self.N, dtype=bool)
        # Same semantics as MultiAgentNuWrapper:
        #   nu=0 -> condition False -> cleaning head
        #   nu=1 -> condition True  -> exploration head
        #   soft nu in [0,1] samples exploration with probability nu.
        condition[: len(nu)] = nu > self.rng.random(len(nu))
        state_float32 = self._state_float32(active_ids)
        if not self.expert_nu.masked_actions:
            actions = self.expert_nu.select_action(state_float32, condition=condition)
        else:
            actions = self.expert_nu.select_masked_action(
                states=state_float32,
                positions=self.env.fleet.get_positions(),
                condition=condition,
            )
        return {agent_id: int(action) for agent_id, action in actions.items() if agent_id in active_ids}

    def step(self, role_weights):
        nu = self._role_weights_to_nu(role_weights)
        action_dict = self._nu_to_actions(nu)
        obs, reward, done_dict, info = self.env.step(action_dict)
        # MultiAgentNuWrapper used any(done.values()). Keep that behavior here
        # because this mode is meant to reproduce the old make_env substrate.
        done = bool(any(done_dict.values())) if isinstance(done_dict, dict) else bool(done_dict)
        info = dict(info or {})
        info["nu"] = nu
        info["movement_actions"] = action_dict
        info["trash_cleaned"] = self.trash_cleaned_pct()
        info["coverage"] = self.coverage_pct()
        info["reward_components_role_order"] = self._format_rewards(reward)
        return self._format_obs(obs), self._format_rewards(reward), done, info


def build_expert_nu_for_env(env, args, device: str, ckpt_path: str):
    """Build Expert_nu exactly with the path-planner recipe from make_env."""
    from Algorithm.RainbowDQL.Agent.Expert_nu import Expert_nu
    from Algorithm.RainbowDQL.Networks.network import DQFDuelingVisualNetwork

    path_planner = DQFDuelingVisualNetwork(
        env.observation_space.shape,
        [8, env.action_space.n - 8],
        getattr(args, "number_of_features", 1024),
        getattr(args, "archtype", "v1"),
        getattr(args, "nettype", "0"),
    ).to(device)
    path_planner.load_state_dict(torch.load(ckpt_path, map_location=device))
    path_planner.eval()

    return Expert_nu(
        env=env,
        device=device,
        path_planner=path_planner,
        expert=getattr(args, "expert_type", "ExpertByMapCoverage"),
        masked_actions=getattr(args, "expert_masked_actions", True),
        consensus=getattr(args, "expert_consensus", True),
    )


def build_env_factory_from_existing_project(args):
    """Create a MultiAgentPatrolling factory using the existing project recipe.

    project_control="dqn_heads":
      Instantiate MultiAgentPatrolling directly and let CTDE-RAM choose movement
      actions with its own DuelingDQN heads.

    project_control="expert_nu":
      Mirror Learning.utils.make_env: load the path planner from
      Learning/path_planner_algorithms/<folder>/Final_Policy.pth, build Expert_nu,
      and let CTDE-RAM choose only W/nu.
    """
    from Environment.PatrollingEnvironments import MultiAgentPatrolling

    map_name = getattr(args, "map_name", "malaga_port")
    map_csv = getattr(args, "map_csv", None)
    if map_csv is None:
        map_csv = os.path.join(PROJECT_ROOT, "Environment", "Maps", f"{map_name}.csv")
    sc_map = np.genfromtxt(map_csv, delimiter=",")

    initial_positions_arg = getattr(args, "initial_positions", None)
    if initial_positions_arg:
        initial_positions = np.asarray(json.loads(initial_positions_arg), dtype=int)
    else:
        initial_positions = default_initial_positions(map_name, int(getattr(args, "N", getattr(args, "n_agents", 4))))

    n_agents = int(getattr(args, "N", getattr(args, "n_agents", initial_positions.shape[0])))
    initial_positions = initial_positions[:n_agents]

    env_kwargs = dict(
        scenario_map=sc_map,
        fleet_initial_positions=initial_positions,
        distance_budget=getattr(args, "distance_budget", 100),
        number_of_vehicles=n_agents,
        seed=getattr(args, "seed", 0),
        miopic=getattr(args, "miopic", True),
        dynamic=getattr(args, "dynamic", False),
        detection_length=getattr(args, "detection_length", 2),
        movement_length=getattr(args, "movement_length", 2),
        max_collisions=getattr(args, "max_collisions", 15),
        reward_type=getattr(args, "reward_type", "Distance Field"),
        convert_to_uint8=getattr(args, "convert_to_uint8", False),
        ground_truth_type=getattr(args, "ground_truth_type", "macro_plastic"),
        obstacles=False,
        frame_stacking=1,
    )
    project_control = getattr(args, "project_control", "dqn_heads")
    if project_control not in {"dqn_heads", "expert_nu"}:
        raise ValueError("project_control must be one of: dqn_heads, expert_nu")

    device_int = int(getattr(args, "device", -1))
    device = "cpu" if device_int < 0 or not torch.cuda.is_available() else f"cuda:{device_int}"
    ckpt_path = resolve_path_planner_ckpt(
        path_planner_folder=getattr(args, "path_planner_folder", None),
        dqn_ckpt=getattr(args, "dqn_ckpt", None),
        path_planner_root=getattr(args, "path_planner_root", None),
    )

    def factory(idx: int = 0):
        kwargs = dict(env_kwargs)
        kwargs["seed"] = int(env_kwargs["seed"]) + int(idx)
        raw_env = MultiAgentPatrolling(**kwargs)
        if project_control == "expert_nu":
            if ckpt_path is None:
                raise ValueError(
                    "--project-control expert_nu requires either --path-planner-folder "
                    "or --dqn-ckpt."
                )
            expert_nu = build_expert_nu_for_env(raw_env, args, device=device, ckpt_path=ckpt_path)
            return ProjectPatrollingExpertNuCTDEEnv(raw_env, expert_nu, seed=kwargs["seed"])
        return ProjectPatrollingCTDEEnv(raw_env)

    return factory
