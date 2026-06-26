import gym
import numpy as np
import os
import torch
import torch.nn as nn
from gym import spaces

import sys
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)
from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.Expert_nu import Expert_nu
from typing import Dict
from torch.distributions import Bernoulli
import argparse
import itertools
import copy

class MultiAgentNuWrapper(gym.Wrapper):
    # This wrapper will:

#     Accept nu values (continuous in [0, 1]) from the PPO agent.

#     Use Expert_nu to convert nu into discrete actions for each agent.

#     Manage the multi-agent-to-single-agent conversion for PPO.
    def __init__(self, env, expert_nu, preference_weight=(1.0, 1.0)):
        super().__init__(env)
        self.expert_nu = expert_nu
        self.num_agents = env.number_of_agents
        
        # Initialize last coverage metrics
        self.last_trash_coverage = 0.0
        self.last_map_coverage = 0.0
        self.simulate_steps = False  # Flag to control simulation mode

        self.initial_budgets = self.num_agents * self.env.distance_budget  # Store initial budgets
        self.weights_rng = np.random.default_rng(self.env.seed)
        self.preference_weight = preference_weight
        
        def make_observation_space(n_agents: int, map_shape: tuple, shared_scalar_dim: int = 5):
            H, W = map_shape

            return spaces.Dict({
                "trash_map": spaces.Box(low=0, high=1, shape=(1, H, W), dtype=np.float32),
                "shared_scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(shared_scalar_dim,), dtype=np.float32),
                "agent_data": spaces.Tuple([
                    spaces.Dict({
                        "ego_pos": spaces.Box(low=0, high=1, shape=(1, H, W), dtype=np.float32),
                        "other_pos": spaces.Box(low=0, high=1, shape=(1, H, W), dtype=np.float32),
                        "distance_budget": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                    }) for _ in range(n_agents)
                ])
            })
        self.single_observation_space = make_observation_space(
            n_agents=self.num_agents,
            map_shape=env.scenario_map.shape,  # Assuming env.scenario_map is a numpy array with shape (H, W)
            shared_scalar_dim=2  # Adjust based on your environment's shared scalars
        )
        self.observation_space = self.single_observation_space
        
        # PPO interacts with nu values for all agents
        # self.single_action_space = spaces.Box(
        #     low=0.0, 
        #     high=1.0, 
        #     shape=(self.num_agents,),  # One nu per agent
        #     dtype=np.float32
        # )
        # For multi-agent discrete actions mu=0 or nu=1,  we use MultiBinary
        self.single_action_space = spaces.MultiBinary(self.num_agents)

        # ✅ Flattened observation space: assumes get_agent_state returns np.ndarray
        sample_obs = self._flatten_obs(env.reset()[0])
        # self.single_observation_space = spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=sample_obs.shape,
        #     dtype=np.float32
        # )
        # self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
    def reset(self, preference_weight=None):
        obs , _= self.env.reset()
        
        # Initialize last coverage metrics
        self.last_trash_coverage = 0.0
        self.last_map_coverage = 0.0
        if preference_weight is not None:
            self.preference_weight = preference_weight
        else:
            weights = self.weights_rng.uniform(0, 1, size=(2,))
            # normalize the weights
            weights /= np.sum(weights)
            self.preference_weight = tuple(weights)
        
        # self.preference_weight = preference_weight
        self.initial_budgets = self.num_agents * self.env.distance_budget  # Store initial budgets
        return self._flatten_obs(obs) , _
    
    def nu_to_actions(self, nu, sim_state=None, sim_fleet=None):
        # Convert nu to actions using Expert_nu
        condition = np.array([nu[i] > np.random.rand() for i in range(self.env.number_of_agents)])
        if self.simulate_steps:
            state_float32 = {
                i: (sim_state[i] / 255.0).astype(np.float32) if self.env.convert_to_uint8 else sim_state[i]
                for i in sim_state.keys()
            }
        else:
            state_float32 = {
                i: (self.env.state[i] / 255.0).astype(np.float32) if self.env.convert_to_uint8 else self.env.state[i]
                for i in self.env.state.keys()
            }

        if not self.expert_nu.masked_actions:
            actions = self.expert_nu.select_action(state_float32, condition=condition)
        else:
            if self.simulate_steps:
                actions = self.expert_nu.select_masked_action(states=state_float32, positions=sim_fleet.get_positions(), condition=condition)
            else:
                actions = self.expert_nu.select_masked_action(states=state_float32, positions=self.env.fleet.get_positions(), condition=condition)

        return {agent_id: action for agent_id, action in actions.items()}
    
    def step(self, action_nu, last_actions=None):
        """
        Handles real or simulated step depending on self.simulate_steps.

        If simulate_steps is True:
        - action_nu can be:
            - (num_agents,) array → simulate 1 step
            - list of (num_agents,) arrays → simulate multiple steps
            - flat list of length k * num_agents → reshaped to simulate k steps
        """
        
        if self.simulate_steps:
            # Normalize input into a list of action arrays
            # if isinstance(action_nu, np.ndarray):
            #     action_nu_seq = [action_nu]  # single step
            # elif isinstance(action_nu, list):
            #     if all(isinstance(el, (np.ndarray, list)) and len(el) == self.env.number_of_agents for el in action_nu):
            #         action_nu_seq = [np.array(el) for el in action_nu]  # list of arrays
            #     elif len(action_nu) % self.env.number_of_agents == 0:
            #         action_nu_seq = np.array(action_nu).reshape(-1, self.env.number_of_agents)
            #     else:
            #         raise ValueError("Invalid shape for action_nu list.")
            # else:
            #     raise TypeError("Unsupported action_nu format for simulation.")
            if isinstance(action_nu, np.ndarray):
                if action_nu.ndim == 1:
                    action_nu = action_nu[np.newaxis]  # single step
                    
            sim_state = self.env.state
            sim_fleet = self.env.fleet
            sim_dict = {}
            
            total_reward = 0.0
            total_length = 0
            for nu in action_nu:
                action = self.nu_to_actions(nu, sim_state=sim_state, sim_fleet=sim_fleet)
            # action_sequence = [nu_to_actions(nu) for nu in [action_nu]]
                # Check if action has been taken before in last_actions
                if last_actions is not None and str(action) in last_actions:
                    reward_dict = last_actions[str(action)]
                else:
                    reward_dict = self.env.simulate_step(action)
                    if last_actions is not None:
                        last_actions[str(action)] = reward_dict
                # reward_dict = self.env.simulate_step(action)
                # for every agent, multiply the reward by the preference weight
                # Multiply each agent's reward by its preference weight
                weighted_rewards = [
                    reward_dict[agent_id] * self.preference_weight
                    for agent_id in reward_dict.keys()
                ]
                total_reward += np.sum(weighted_rewards)
                # total_reward += np.sum([np.sum(r) for r in reward_dict.values()])
                total_length += 1
                # if any(done_dict.values()):
                #     break
                # sim_state = next_obs  # Update simulated state
                # sim_fleet = info_["fleet"]  # Update simulated fleet
                # sim_dict = info_  # Update simulation info
                # current_trash = info["trash_coverage"]
                # current_map = info["map_coverage"]
                # self.last_trash_coverage = current_trash
                # self.last_map_coverage = current_map
            info = {
                "cumulative_steps": total_length,
                'last_actions': last_actions
            }
            # total reward is an np array of shape (num_agents,num reward components) of reward_dict
            total_reward = np.asarray(list(reward_dict.values()))
            return None, total_reward, None, None, info

        # 🔁 Standard env step using the same action conversion
        actions = self.nu_to_actions(action_nu)

        next_obs, reward, done, info = self.env.step(actions)

        current_trash = info["trash_coverage"]
        current_map = info["map_coverage"]
        self.last_trash_coverage = current_trash
        self.last_map_coverage = current_map
        weighted_rewards = [
            reward[agent_id] * self.preference_weight
            for agent_id in reward.keys()
        ]
        total_reward = np.sum(weighted_rewards)
        info['reward_components'] = np.asarray(list(reward.values()))
        # total_reward = np.sum(list(reward.values()))
        return self._flatten_obs(next_obs), total_reward, any(done.values()), any(done.values()), info

    def get_q_values(self, nu):
        sim_state = self.env.state
        sim_fleet = self.env.fleet
        sim_dict = {}
        action = self.nu_to_actions(nu, sim_state=sim_state, sim_fleet=sim_fleet)
        # array such as action that is self.preference_weight[0] if nu[agent_id] == 0 else self.preference_weight[1]
        # action_weights = [self.preference_weight[0] if nu[agent_id] == 0 else self.preference_weight[1] for agent_id in sim_state.keys()]
        return [self.expert_nu.values4consensus[agent_id][action[agent_id]] for agent_id in sim_state.keys()]

    def _get_agent_state(self, agent_id: int) -> np.ndarray:
        """Implement this method in your env to return agent-specific observations"""
        return self.env.get_agent_state(agent_id)

    def _flatten_obs_(self, obs: Dict[int, np.ndarray]) -> np.ndarray:
        # return np.concatenate([obs[agent_id] for agent_id in sorted(obs.keys())])
        """Stack all agent observations into a single vector"""
        # We want: [trash_map, egopos_0, otherpos_0, egopos_1, otherpos_1, ..., egopos_N, otherpos_N]
        if len(obs) == 0:
            return np.array([])
        # Assume trash_map is the same for all agents, so take from agent 0
        trash_map = obs[0][0:1]  # shape (1, H, W) or (1, ...)
        fleet_obs = [trash_map]
        for agent_id in sorted(obs.keys()):
            fleet_obs.append(obs[agent_id][1:])  # egopos_i, otherpos_i
        # fleet_obs = np.concatenate(fleet_obs, axis=0)
        # obs = {0: fleet_obs}
        # return np.concatenate([obs[agent_id] for agent_id in sorted(obs.keys())])
        # Flatten the observations
        return np.concatenate(fleet_obs, axis=0)  # shape (num_agents * obs_dim,)
    def _flatten_obs(self, obs: Dict[int, np.ndarray]) -> Dict:
        # Shared inputs (from any agent)
        trash_map = obs[0][0:1]            # (1, H, W)
        distances = np.asarray(self.env.fleet.get_distances())  # List of distances for each agent
        shared_scalars = np.array([
            # self.initial_budgets - np.sum(distances),   # optional: avg budget
            self.preference_weight[0],
            self.preference_weight[1]
            # self.last_trash_coverage,
            # self.last_map_coverage
        ], dtype=np.float32)

        # Per-agent inputs
        agent_data = []
        for agent_id in sorted(obs.keys()):
            full_obs = obs[agent_id]  # shape: (3, H, W)
            ego_pos = full_obs[1:2]   # (1, H, W)
            other_pos = full_obs[2:3] # (1, H, W)
            budget = (self.env.distance_budget - distances[agent_id]) / self.env.distance_budget  # Calculate budget for this agent
            agent_data.append({
                "ego_pos": ego_pos,
                "other_pos": other_pos,
                "distance_budget": np.array([budget], dtype=np.float32)
            })

        return {
            "trash_map": trash_map,
            "shared_scalars": shared_scalars,
            "agent_data": agent_data  # length = N agents
        }

        # return obs

def make_env(
    env_fn: callable,  # Function that returns a MultiAgentPatrolling instance
    expert_nu_fn: callable,  # Function that returns an Expert_nu instance
    env_kwargs: dict,  # Arguments for MultiAgentPatrolling
    expert_kwargs: dict,  # Arguments for Expert_nu
    seed: int,
    device_int: int,  # Device for PyTorch
    args: argparse.Namespace,  # Arguments from the command line
    idx: int,
    capture_video: bool = False,
    run_name: str = None,
):
    def thunk():
        # Initialize environment with custom args
        env_kwargs['seed'] = seed
        env = env_fn(**env_kwargs)
        # path planner is a DQFDuelingVisualNetwork
        from Algorithm.RainbowDQL.Networks.network import DQFDuelingVisualNetwork
        nettype = '0'
        arch = 'v1'
        device = 'cpu' if device_int == -1 else f'cuda:{device_int}'
        path_planner = DQFDuelingVisualNetwork(env.observation_space.shape, [8, env.action_space.n - 8], 1024,arch,nettype).to(device)
        # Experimento_clean26_alamillo_lake_macro_plastic_random_nus_nsteps5 Experimento_clean26_malaga_port_macro_plastic_random_nus_nsteps5
        
        path_to_file = f"{data_path}/Learning/path_planner_algorithms/Experimento_clean26_{args.map}_{args.benchmark}_random_nus_nsteps5/Final_Policy.pth"
        path_to_file = f"{data_path}/Learning/path_planner_algorithms/Experimento_clean28_{args.map}_{args.benchmark}_random_nus_nsteps5_distbudget100_old_reward/Final_Policy.pth"
        if not os.path.exists(path_to_file):
            raise FileNotFoundError(f"Path to file {path_to_file} does not exist. Please check the path.")
        
        path_planner.load_state_dict(torch.load(path_to_file, map_location=device))
        ########################
        
        # Initialize Expert_nu with env instance + custom args
        expert_nu = expert_nu_fn(env=env, path_planner=path_planner, **expert_kwargs)
        
        # Wrap the environment
        env = MultiAgentNuWrapper(env, expert_nu)
        
        # Standard Gym wrappers
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # if capture_video and idx == 0:
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        
        # Seeding
        # env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(envs.single_observation_space.shape[0], 32, 8, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        n_flatten = np.prod(self.network(torch.zeros(size=(1,
                                                               envs.single_observation_space.shape[0],
                                                               envs.single_observation_space.shape[1],
                                                               envs.single_observation_space.shape[2]))).shape)
        self.linear = nn.Sequential(
            layer_init(nn.Linear(n_flatten, 512)),
            nn.ReLU(),
        )
        n_actions = envs.single_action_space.n if isinstance(envs.single_action_space, spaces.MultiBinary) else envs.single_action_space.shape[0]
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)

        # self.actor = nn.Sequential(
        #                             layer_init(nn.Linear(512, envs.single_action_space.shape[0]), std=0.01),
        #                             nn.Sigmoid()  # To ensure outputs ∈ [0,1])
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        x = self.network(x)
        x = self.linear(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.network(x)
        x = self.linear(x)
        
        logits = self.actor(x)  # shape: (batch, n_agents)
        dist = Bernoulli(logits=logits)  # or probs=torch.sigmoid(logits)

        if action is None:
            action = dist.sample()
        
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return action, logprob, entropy, self.critic(x)
class PreferencePPOAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.observation_space = envs.single_observation_space
        self.n_agents = envs.single_action_space.n
        self.action_dim = envs.single_action_space.n

        # === Extract input shapes from the observation space ===
        trash_map_shape = self.observation_space["trash_map"].shape
        scalar_dim = self.observation_space["shared_scalars"].shape[0]
        ego_shape = self.observation_space["agent_data"][0]["ego_pos"].shape
        budget_shape = self.observation_space["agent_data"][0]["distance_budget"].shape

        # === Trash map encoder: CNN ===
        self.trash_map_encoder = nn.Sequential(
            layer_init(nn.Conv2d(trash_map_shape[0], 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 32, kernel_size=3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 16, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten()
        )
        dummy_trash = torch.zeros((1,) + trash_map_shape)
        trash_latent_dim = self.trash_map_encoder(dummy_trash).shape[-1]
        
        trash_out_dim = 512  # Output dimension after trash map encoding
        self.trash_bottleneck = nn.Sequential(
            layer_init(nn.Linear(trash_latent_dim, trash_out_dim)),     # Compress
            nn.ReLU()
        )
        # === Shared scalars encoder: MLP ===
        scalar_out_dim = 64
        self.scalar_encoder = nn.Sequential(
            layer_init(nn.Linear(scalar_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, scalar_out_dim)),
            nn.ReLU()
        )

        # === Ego+Other Position encoder: CNN ===
        self.agent_pos_encoder = nn.Sequential(
            layer_init(nn.Conv2d(2, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 32, kernel_size=3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 16, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten()
        )
        dummy_pos = torch.zeros((1, 2) + ego_shape[1:])
        pos_latent_dim = self.agent_pos_encoder(dummy_pos).shape[-1]
        
        agent_out_dim = 256  # Output dimension after agent position encoding
        self.agent_bottleneck = nn.Sequential(
            layer_init(nn.Linear(pos_latent_dim, agent_out_dim)),
            nn.ReLU()
        )
        # === Agent feature size ===
        self.per_agent_feature_dim = agent_out_dim + budget_shape[0]
        total_fused_dim = trash_out_dim + scalar_out_dim + self.per_agent_feature_dim * self.n_agents

        fusion_out_dim = 512  # Output dimension after fusion layer
        # === Fusion layer ===
        self.fusion = nn.Sequential(
            layer_init(nn.Linear(total_fused_dim, fusion_out_dim)),
            nn.ReLU()
        )

        # === Actor and Critic ===
        self.actor = layer_init(nn.Linear(fusion_out_dim, self.action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(fusion_out_dim, 1), std=1.0)

    def forward(self, obs):
        """
        obs: a dict containing:
            - trash_map: (B, 1, H, W)
            - shared_scalars: (B, D)
            - agent_data: list of N dicts with keys:
                - ego_pos: (B, 1, H, W)
                - other_pos: (B, 1, H, W)
                - distance_budget: (B, 1)
        """
        B = obs["trash_map"].shape[0]

        # Encode shared components
        trash_feat = self.trash_bottleneck(self.trash_map_encoder(obs["trash_map"]))           # (B, D1)
        scalar_feat = self.scalar_encoder(obs["shared_scalars"])        # (B, 128)

        # Encode each agent's data
        agent_features = []
        for i in range(self.n_agents):
            ego = obs["agent_data"][i]["ego_pos"]       # (B, 1, H, W)
            other = obs["agent_data"][i]["other_pos"]   # (B, 1, H, W)
            budget = obs["agent_data"][i]["distance_budget"]  # (B, 1)

            pos_stack = torch.cat([ego, other], dim=1)   # (B, 2, H, W)
            pos_feat = self.agent_bottleneck(self.agent_pos_encoder(pos_stack)) # (B, D2)
            agent_feat = torch.cat([pos_feat, budget], dim=1)  # (B, D2+1)
            agent_features.append(agent_feat)

        # Concatenate all per-agent features
        agents_fused = torch.cat(agent_features, dim=1)  # (B, N*(D2+1))

        # Fuse all together
        fused_input = torch.cat([trash_feat, scalar_feat, agents_fused], dim=1)  # (B, total_fused_dim)
        fused = self.fusion(fused_input)  # (B, 512)

        logits = self.actor(fused)  # (B, action_dim)
        value = self.critic(fused)  # (B, 1)

        return logits, value

    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        dist = Bernoulli(logits=logits)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, value

    def get_value(self, obs):
        _, value = self.forward(obs)
        return value

class PreferencePPOAgent_v0(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.observation_space = envs.single_observation_space
        self.n_agents = envs.single_action_space.n
        self.action_dim = envs.single_action_space.n
        # === Extract input shapes from the observation space ===
        trash_map_shape = self.observation_space["trash_map"].shape        # (1, H, W)
        scalar_dim = self.observation_space["shared_scalars"].shape[0]     # D
        ego_shape = self.observation_space["agent_data"][0]["ego_pos"].shape  # (1, H, W)
        budget_shape = self.observation_space["agent_data"][0]["distance_budget"].shape  # (1,)

        # === Trash map encoder: CNN ===
        self.trash_map_encoder = nn.Sequential(
            nn.Conv2d(trash_map_shape[0], 16, kernel_size=3, padding=1),  # (1, H, W) → (16, H, W)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),                  # → (32, H, W)
            nn.ReLU(),
            nn.Flatten()
        )
        dummy_trash = torch.zeros((1,) + trash_map_shape)  # (1, 1, H, W)
        trash_latent_dim = self.trash_map_encoder(dummy_trash).shape[-1]

        # === Shared scalars encoder: MLP ===
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # === Ego+Other Position encoder: CNN ===
        self.agent_pos_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),  # (2, H, W)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        dummy_pos = torch.zeros((1, 2) + ego_shape[1:])  # (1, 2, H, W)
        pos_latent_dim = self.agent_pos_encoder(dummy_pos).shape[-1]

        # === Agent feature size ===
        self.per_agent_feature_dim = pos_latent_dim + budget_shape[0]
        total_fused_dim = trash_latent_dim + 64 + self.per_agent_feature_dim * self.n_agents

        # === Fusion layer ===
        self.fusion = nn.Sequential(
            nn.Linear(total_fused_dim, 512),
            nn.ReLU()
        )

        # === Actor and Critic ===
        self.actor = nn.Sequential(
            nn.Linear(512, self.action_dim),
            nn.Sigmoid()  # useful for MultiBinary or [0,1] actions
        )
        self.critic = nn.Linear(512, 1)

    def forward(self, obs):
        """
        obs: a dict containing:
            - trash_map: (B, 1, H, W)
            - shared_scalars: (B, D)
            - agent_data: list of N dicts with keys:
                - ego_pos: (B, 1, H, W)
                - other_pos: (B, 1, H, W)
                - distance_budget: (B, 1)
        """
        B = obs["trash_map"].shape[0]

        # Encode shared components
        trash_feat = self.trash_map_encoder(obs["trash_map"])           # (B, D1)
        scalar_feat = self.scalar_encoder(obs["shared_scalars"])        # (B, 64)

        # Encode each agent's data
        agent_features = []
        for i in range(self.n_agents):
            ego = obs["agent_data"][i]["ego_pos"]       # (B, 1, H, W)
            other = obs["agent_data"][i]["other_pos"]   # (B, 1, H, W)
            budget = obs["agent_data"][i]["distance_budget"]  # (B, 1)

            pos_stack = torch.cat([ego, other], dim=1)   # (B, 2, H, W)
            pos_feat = self.agent_pos_encoder(pos_stack) # (B, D2)
            agent_feat = torch.cat([pos_feat, budget], dim=1)  # (B, D2+1)
            agent_features.append(agent_feat)

        # Concatenate all per-agent features
        agents_fused = torch.cat(agent_features, dim=1)  # (B, N*(D2+1))

        # Fuse all together
        fused_input = torch.cat([trash_feat, scalar_feat, agents_fused], dim=1)  # (B, total_fused_dim)
        fused = self.fusion(fused_input)  # (B, 512)

        logits = self.actor(fused)  # (B, action_dim)
        value = self.critic(fused)  # (B, 1)

        return logits, value

    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        dist = Bernoulli(probs=logits)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, value

    def get_value(self, obs):
        _, value = self.forward(obs)
        return value

class PreferenceMAPPOAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.observation_space = envs.single_observation_space
        self.n_agents = len(envs.single_action_space) if isinstance(envs.single_action_space, list) else envs.single_action_space.n
        self.action_dim = 1  # Bernoulli per agent

        # === Extract input shapes ===
        trash_map_shape = self.observation_space["trash_map"].shape
        ego_shape = self.observation_space["agent_data"][0]["ego_pos"].shape
        budget_shape = self.observation_space["agent_data"][0]["distance_budget"].shape
        scalar_dim = self.observation_space["shared_scalars"].shape[0]  # two scalars

        # === Trash map encoder ===
        self.trash_map_encoder = nn.Sequential(
            layer_init(nn.Conv2d(trash_map_shape[0], 32, 3, 1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 16, 3, 1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 8, 3, 1)),
            nn.ReLU(),
            nn.Flatten()
        )
        dummy_trash = torch.zeros((1,) + trash_map_shape)
        trash_latent_dim = self.trash_map_encoder(dummy_trash).shape[-1]
        trash_out_dim = 16
        self.trash_bottleneck = nn.Sequential(
            layer_init(nn.Linear(trash_latent_dim, trash_out_dim)),
            nn.ReLU()
        )
        self.trash_norm = nn.LayerNorm(trash_out_dim)  # LayerNorm for trash
        
        # === Agent pos encoder ===
        self.agent_pos_encoder = nn.Sequential(
            layer_init(nn.Conv2d(2, 32, 3, 1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 16, 3, 1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 8, 3, 1)),
            nn.ReLU(),
            nn.Flatten()
        )
        dummy_pos = torch.zeros((1, 2) + ego_shape[1:])
        pos_latent_dim = self.agent_pos_encoder(dummy_pos).shape[-1]
        
        agent_bottleneck_out_dim = 4
        self.agent_bottleneck = nn.Sequential(
            layer_init(nn.Linear(pos_latent_dim, agent_bottleneck_out_dim)),
            nn.ReLU()
        )
        self.agent_norm = nn.LayerNorm(agent_bottleneck_out_dim)  # LayerNorm for agent features
        self.per_agent_feature_dim = agent_bottleneck_out_dim + budget_shape[0]

        # === Per-agent actor heads ===
        actor_in_dim = trash_out_dim + scalar_dim + self.per_agent_feature_dim
        # self.actor_heads = nn.ModuleList([
        #     layer_init(nn.Linear(actor_in_dim, self.action_dim), std=0.1)
        #     for _ in range(self.n_agents)
        # ])
        # Actor heads with hidden layers
        actor_hidden_dim1 = 128
        # actor_hidden_dim2 = actor_hidden_dim1 
        actor_hidden_dim2 = actor_hidden_dim1 // 2
        self.actor_heads = nn.ModuleList([
            nn.Sequential(
                layer_init(nn.Linear(actor_in_dim, actor_hidden_dim1)),
                nn.ReLU(),
                layer_init(nn.Linear(actor_hidden_dim1, actor_hidden_dim2)),
                nn.ReLU(),
                layer_init(nn.Linear(actor_hidden_dim2, self.action_dim), std=np.sqrt(2))
            )
            for _ in range(self.n_agents)
        ])
        # === Centralized critic ===
        critic_in_dim = trash_out_dim + scalar_dim + self.per_agent_feature_dim * self.n_agents
        # Critic with hidden layers
        critic_hidden_dim1 = 128
        critic_hidden_dim2 = critic_hidden_dim1 // 2

        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_in_dim, critic_hidden_dim1)),
            nn.ReLU(),
            layer_init(nn.Linear(critic_hidden_dim1, critic_hidden_dim2)),
            nn.ReLU(),
            layer_init(nn.Linear(critic_hidden_dim2, 1), std=1.0)
        )

        # self.critic = layer_init(nn.Linear(critic_in_dim, 1), std=1.0)

    def forward(self, obs, action=None):
        B = obs["trash_map"].shape[0]

        # === Encode trash map ===
        trash_feat = self.trash_bottleneck(self.trash_map_encoder(obs["trash_map"]))
        trash_feat = self.trash_norm(trash_feat)  # LayerNorm applied
        
        # === Encode agent features ===
        agent_features = []
        for i in range(self.n_agents):
            ego = obs["agent_data"][i]["ego_pos"]
            other = obs["agent_data"][i]["other_pos"]
            budget = obs["agent_data"][i]["distance_budget"]

            pos_stack = torch.cat([ego, other], dim=1)
            pos_feat = self.agent_bottleneck(self.agent_pos_encoder(pos_stack))
            pos_feat = self.agent_norm(pos_feat)  # LayerNorm applied
            agent_feat = torch.cat([pos_feat, budget], dim=1)
            agent_features.append(agent_feat)

        weights = obs["shared_scalars"]  # (B, 2)

        # === Actions per agent ===
        actions, logprobs, entropies = [], [], []
        for i, head in enumerate(self.actor_heads):
            input_feat = torch.cat([trash_feat, weights, agent_features[i]], dim=1)
            dist = Bernoulli(logits=head(input_feat))
            act = dist.sample() if action is None else action[:, i].unsqueeze(-1)
            # print(dist.probs.detach().cpu().numpy())
            actions.append(act.squeeze(-1))
            logprobs.append(dist.log_prob(act).sum(-1))
            entropies.append(dist.entropy().sum(-1))

        # === Centralized critic ===
        critic_input = torch.cat([trash_feat, weights] + agent_features, dim=1)
        value = self.critic(critic_input)
        
        # === Stack outputs ===
        actions = torch.stack(actions, dim=1)
        logprobs = torch.stack(logprobs, dim=1)
        entropies = torch.stack(entropies, dim=1)

        return actions, logprobs, entropies, value
    
    def get_action_and_value(self, obs, action=None):
        return self.forward(obs, action)

    def get_value(self, obs):
        _, _, _, value = self.forward(obs)
        return value


# Suppose you want to evaluate C candidates each step
# candidate_count = C
# vec_env = SyncVectorEnv([make_env for _ in range(candidate_count)])
# obs_batch = vec_env.reset()
class GreedyAgent:
    def __init__(self, env, greedy_type="WPOP Reward", reward_stats=None, normalize_rewards=True):
        self.num_agents = env.num_agents        
        self.env = env
        self.greedy_type = greedy_type
        self.normalize_rewards = normalize_rewards
        # greedy_type = "QValue"
        if greedy_type == "WS Reward":
            self.scalarization = self._weighted_sum
        elif greedy_type == "WPOP Reward":
            self.scalarization = self._weighted_product_of_power
        elif greedy_type == "WP Reward":
            self.scalarization = self._weighted_power
            self.p_wp = 3.0 # impar para que no invierta reward negativos
        elif greedy_type == "EWC Reward":
            self.scalarization = self._ewc_reward
            self.p_ewc = 1.0
        elif greedy_type == "QValue":
            self.act = self.act_q_value_greedy
            
        if normalize_rewards:
            if reward_stats is not None:
                self.reward_min = reward_stats['min_reward']
                self.reward_max = reward_stats['max_reward']
            else:
                self.reward_min = np.zeros(2)
                self.reward_max = np.ones(2)
        # import time
        # print the time it takes to copy the environment
        # start_time = time.time()
        # self.sim_env = copy.deepcopy(env)
        # end_time = time.time()
        # print(f"Time taken to copy the environment: {end_time - start_time:.4f} seconds")
        # start_time = time.time()
        # _ = copy.deepcopy(env.fleet)
        # end_time = time.time()
        # print(f"Time taken to copy the environment: {end_time - start_time:.4f} seconds")
        # start_time = time.time()
        # self.sim_env = copy.deepcopy(env.gt)
        # end_time = time.time()
        # print(f"Time taken to copy the environment: {end_time - start_time:.4f} seconds")
        # self.sim_env.reset()
        # Ensure sim_env is a SyncVectorEnv for batch processing
        # if not isinstance(self.sim_env, SyncVectorEnv):
        #     self.sim_env = SyncVectorEnv([lambda: self.sim_env for _ in range(1)])
    # def normalize_reward(self, reward):
    def _weighted_sum(self, norm_rewards):
        return  (
            norm_rewards[:, 0] * self.env.preference_weight[0]
            + norm_rewards[:, 1] * self.env.preference_weight[1]
        )
    def _weighted_power(self, norm_rewards):
        return  (
            self.env.preference_weight[0] * (norm_rewards[:, 0] ** self.p_wp)
            + self.env.preference_weight[1] * (norm_rewards[:, 1] ** self.p_wp)
        )
    def _weighted_product_of_power(self, norm_rewards):
        return  (
            (norm_rewards[:, 0] ** self.env.preference_weight[0]) * (norm_rewards[:, 1] ** self.env.preference_weight[1])
        )
    def _ewc_reward(self, norm_rewards):
        """
        Exponential Weighted Criterion (EWC) scalarization.
        Args:
            norm_rewards: np.array of shape (num_candidates, 2), normalized rewards
        Returns:
            weighted_rewards: np.array of shape (num_candidates,)
        """
        # Tunable exponential parameter
        # p = 3.0  # you can make this self.p if you want it configurable

        # Compute EWC: sum_i (exp(p * w_i) - 1) * exp(p * r_i)
        # Vectorized over all candidates
        w_exp = np.exp(self.p_ewc * np.array(self.env.preference_weight)) - 1  # shape (2,)
        ewc_matrix = np.exp(self.p_ewc * norm_rewards)  # shape (num_candidates, 2)
        weighted_rewards = np.sum(w_exp * ewc_matrix, axis=1)  # shape (num_candidates,)

        return weighted_rewards

    def act_q_value_greedy(self, obs):
        best_nu = None
        best_sum_q_value = -float("inf")
        self.env.simulate_steps = True  
        for nu in itertools.product([0, 1], repeat=self.num_agents):
            nu_vec = np.array(nu, dtype=np.float32)
            q_values = self.env.get_q_values(nu_vec)
            sum_q_values = sum(q_values)
            if sum_q_values > best_sum_q_value:
                best_sum_q_value = sum_q_values
                best_nu = nu_vec
        self.env.simulate_steps = False
        return best_nu

    def act(self, obs):
        best_nu = None
        self.env.simulate_steps = True
        last_actions = {}

        # Generate all nu combinations as a 2D array (shape: num_combinations x num_agents)
        num_combinations = 2 ** self.num_agents
        nu_combinations = np.array(list(itertools.product([0, 1], repeat=self.num_agents)), dtype=np.float32)

        # Preallocate array to store sum rewards (shape: num_combinations x 2)
        sum_rewards = np.zeros((num_combinations, 2), dtype=np.float32)

        # Evaluate all candidate nu in a single loop (could be parallelized if env supports batch)
        for i, nu_vec in enumerate(nu_combinations):
            _, reward, _, _, info = self.env.step(nu_vec, last_actions)
            last_actions = info['last_actions']
            sum_rewards[i] = np.sum(reward, axis=0)
        if self.normalize_rewards:
            # 🟢 Update global min/max reward bounds BEFORE normalization
            self.reward_min = np.minimum(self.reward_min, np.min(sum_rewards, axis=0))
            self.reward_max = np.maximum(self.reward_max, np.max(sum_rewards, axis=0))

            # Normalize all sum_rewards
            denom = np.maximum(self.reward_max - self.reward_min, 1e-8)
            norm_rewards = (sum_rewards - self.reward_min) / denom
        else:
            norm_rewards = sum_rewards

        # Compute weighted scalarization for all candidates
        weighted_rewards = self.scalarization(norm_rewards)

        # Pick best nu
        best_idx = np.argmax(weighted_rewards)
        best_nu = nu_combinations[best_idx]

        self.env.simulate_steps = False
        return best_nu
