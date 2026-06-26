import sys
import os
data_path = os.path.join(os.path.dirname(__file__), '..','..','..')
sys.path.append(data_path)
from typing import Dict, List, Tuple
import gym
import numpy as np
from tqdm import trange
from copy import copy
import json
import os
import torch
import joblib

from Algorithm.RainbowDQL.ActionMasking.ActionMaskingUtils import SafeActionMasking, ConsensusSafeActionMasking

class Expert_nu:
    def __init__(self,
			env: gym.Env,
            device: str = 'cpu',
            path_planner=None,
            expert: str = 'ExpertByMapCoverage',
			# Masked actions
			masked_actions= True,
			consensus= True,):
        self.device = device
        # Initialize the environment and its properties
        self.env = env
        self.save_state_in_uint8 = self.env.convert_to_uint8
        action_dim = env.action_space.n
        self.action_dim = [8, action_dim - 8]
  
        # self.nu_intervals = nu_intervals
        # self.nu = self.nu_intervals[0][1]
        
        self.path_planner = path_planner
        self.expert = expert
        
        self.masked_actions = masked_actions
        self.consensus = consensus
                # Masking utilities #
        if self.masked_actions:
            self.safe_masking_module = SafeActionMasking(action_space_dim = self.action_dim[0], movement_length = self.env.movement_length)
        
        if self.consensus:
            self.consensus_safe_action_masking = ConsensusSafeActionMasking(self.env.scenario_map, action_space_dim = self.action_dim[0], movement_length = self.env.movement_length)
            self.values4consensus = {agent_id: np.zeros((1, self.env.action_space.n)) for agent_id in range(self.env.number_of_agents)}
            
    def predict_masked_action(self, state: np.ndarray, agent_id: int, position: np.ndarray, condition: bool = True) -> int:
        """Predict the action for the given state and agent_id."""
        
		# Update the state of the safety module #
        self.safe_masking_module.update_state(position = position, new_navigation_map = self.env.scenario_map)
        # q_values = self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy()
        action_values = self.get_action_values(state, agent_id=agent_id, condition=condition)
        action_values, selected_action = self.safe_masking_module.mask_action(q_values = action_values.flatten())
        if self.consensus:
            self.values4consensus[agent_id] = action_values

        return selected_action

    def get_action_values(self, state: np.ndarray, agent_id: int, condition: bool = True) -> np.ndarray:
        """Get the action values for the given state and agent_id."""
        
        # Here we assume that the action values are computed by some model, e.g., a neural network.
        # For simplicity, we will return random values as a placeholder.
        # action_values = np.random.rand(self.action_dim[0])
        q_values = self.path_planner(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy()
        if condition:
            action_values = q_values.squeeze(0)[:self.action_dim[0]]
        else:
            action_values = q_values.squeeze(0)[self.action_dim[0]:]
        return action_values
    
    def get_condition(self) -> np.ndarray:
        """Determine the condition based on the selected expert strategy."""
        if self.expert == 'ExpertByMapCoverage':
            return self._ExpertByMapCoverage()
        elif self.expert == 'default':
            return self._condition_default()
        # You can add more expert types here
        else:
            raise ValueError(f"Unknown expert type: {self.expert}")
    
    def select_action(self, states: dict, condition=None) -> dict:
        if condition is None:
            condition = self.get_condition()
        actions = {agent_id: self.predict_action(state, condition=condition[agent_id]) for agent_id, state in states.items()}

        return actions

    def select_masked_action(self, states: dict, positions: np.ndarray, condition=None):

        if self.consensus:
            self.values4consensus = {agent_id: np.zeros((1, self.env.action_space.n)) for agent_id, state in states.items()}
            
        if condition is None:
            # See which phase we are by choosing self.nu > np.random.rand()
            condition = self.get_condition()
        # Predict the action for each agent
        actions = {agent_id: self.predict_masked_action(state, agent_id=agent_id, position=positions[agent_id], condition=condition[agent_id]) for agent_id, state in states.items()}

        if self.consensus:
            actions = self.consensus_safe_action_masking.query_actions(self.values4consensus,positions)
            actions = {agent_id: actions[agent_id] for agent_id, state in states.items()}

        return actions
        
   
    # def anneal_nu(self):

    #     if p <= p2[0] and p1[0]!=p2[0]:
    #         first_p = p1
    #         second_p = p2
    #     elif p <= p3[0] and p2[0]!=p3[0]:
    #         first_p = p2
    #         second_p = p3
    #     elif p <= p4[0]:
    #         first_p = p3
    #         second_p = p4

    #     return (second_p[1] - first_p[1]) / (second_p[0] - first_p[0]) * (p - first_p[0]) + first_p[1]

    def run_episodes(self, episodes, render = False):
        
        # Reset metrics #
        episodic_reward_vector = []
        record = np.array([-np.inf, -np.inf])
        mean_clean_record = -np.inf
        percentage_of_map_visited_record = -np.inf
        max_movements = self.env.distance_budget
        
        for episode in trange(1, int(episodes) + 1):
            # Reset the environment #
            state = self.env.reset()
            if render:
                self.env.render()
            done = {i:False for i in range(self.env.number_of_agents)}
            state = self.env.reset()
            score = 0
            length = 0
            
            while not all(done.values()):
                
                # Select the action using the current policy
                state_float32 = {i:None for i in state.keys()}
                if self.save_state_in_uint8:
                    for agent_id in state.keys():
                        state_float32[agent_id] = (state[agent_id] / 255.0).astype(np.float32)
                else:
                    state_float32 = state

                if not self.masked_actions:
                    actions = self.select_action(state_float32)
                else:
                    actions = self.select_masked_action(states=state_float32, positions=self.env.fleet.get_positions())


                actions = {agent_id: action for agent_id, action in actions.items() if not done[agent_id]}
                # Process the agent step #
                next_state, reward, done = self.step(actions)

                # Update the state
                state = next_state
                # Accumulate indicators
                score += np.mean(list(reward.values()),axis=0)  # The mean reward among the agents
                length += 1
                if render:
                    self.env.render()
                # print(f"Episode: {episode}, Percentage of map visited: {self.env.percentage_of_map_visited}, nu: {self.nu}")
                # if episode ends
                if all(done.values()):


                    # Compute average metrics #
                    self.episodic_reward = score
                    self.episodic_length = length
                    episodic_reward_vector.append(self.episodic_reward)
           
    def step(self, action: dict) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""

        next_state, reward, done, _ = self.env.step(action)

        return next_state, reward, done
    
    def _ExpertByMapCoverage(self) -> np.ndarray:
        """Original default logic for condition."""
        condition = np.ndarray(shape=(self.env.number_of_agents,), dtype=bool)
        for agent_id in range(self.env.number_of_agents):
            if self.env.percentage_of_map_visited > (agent_id / self.env.number_of_agents):
                nu = 0
            else:
                nu = 1
            condition[agent_id] = nu > np.random.rand()
            # print(f"Agent: {agent_id}, Percentage visited: {self.env.percentage_of_map_visited}, nu: {nu}")
        return condition