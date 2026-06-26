import os
import sys
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)

import json
import argparse
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt

from Learning.utils import make_env, GreedyAgent
from Utils.metrics_wrapper import MetricsDataCreator

data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)
# Configuration
RUNS_DIR  =  [#"malaga_port__macro_plastic__0__1749796562", # 10*(delta_trash + delta_map)/2 - 1
            #  "malaga_port__macro_plastic__0__1749862821" #100*(delta_trash + delta_map)/2 - 1
            #  "malaga_port__macro_plastic__0__1749930844" #  np.mean(list(reward.values())) + 10*(delta_trash + delta_map)/2 - 1
            #  'malaga_port__macro_plastic__0__1749987331'  #   np.sum(list(reward.values())) + 100*(delta_trash + delta_map)/2 - 1
            #   "malaga_port__macro_plastic__0__1749715625",  # np.sum
            #   "malaga_port__macro_plastic__0__1749751301"  # np.mean
              #"malaga_port__macro_plastic__0__1749774774"  # (delta_trash + delta_map)/2 - 1
            #   'malaga_port__macro_plastic__0__1750173760', # np.sum - 8 channel state
            #   'malaga_port__macro_plastic__0__1750185322', # np.sum steps 256 - numenvs 16
              'malaga_port__macro_plastic__0__1750189305' # np.sum 10000000 total steps
]
DEFAULT_RUNS_DIR = [f"{data_path}/runs/{run}" for run in RUNS_DIR]
DEFAULT_MODEL_NAME = "ppo_agent.pth"
DEFAULT_ARGS_NAME = "args.json"
METRICS_DIRECTORY = f"{data_path}/Evaluation/Results/"
# metrics_directory= f'{data_path}Results_seed_{seed}_nu_steps_dist_field_{args.map}_30keps/{policy_type}_{nu_step}',

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one or multiple PPO agents.")
    parser.add_argument("--num-episodes", type=int, default=200, help="Number of episodes to evaluate.")
    parser.add_argument("--seed", type=int, default=30, help="Random seed for reproducibility.")
    parser.add_argument("--runs", nargs='+', type=str, default=DEFAULT_RUNS_DIR,
                        help="List of run directories under /runs to evaluate.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help="Name of the model .pth file inside each run directory.")
    parser.add_argument("--args_path", type=str, default=DEFAULT_ARGS_NAME,
                        help="Name of the args .json file inside each run directory.")
    parser.add_argument("--metrics_directory", type=str, default=METRICS_DIRECTORY,
                        help="Directory to save evaluation metrics.")
    parser.add_argument("--device", type=int, default=0, choices=[-1, 0, 1],
                        help="Device to run evaluation on: -1 for CPU, 0 for GPU.")
    parser.add_argument("--render", type=bool, default=False, 
                        help="Render the environment during evaluation.")
    args = parser.parse_args()
    # check if runs directory is a list of directories
    if isinstance(args.runs, str):
        args.runs = [args_eval.runs]
    return args




if __name__ == "__main__":
    args_eval = parse_args()

    with open(args_eval.args_path, "r") as f:
        args_env_dict = json.load(f)
    args_env = argparse.Namespace(**args_env_dict)

    torch.manual_seed(args_eval.seed)
    np.random.seed(args_eval.seed)

    # Env setup
    N = args_env.n_agents
    map_name = args_env.map
    benchmark_name = args_env.benchmark
    sc_map = np.genfromtxt(f"{data_path}/Environment/Maps/{map_name}.csv", delimiter=',')

    if map_name == 'malaga_port':
        initial_positions = np.array([[12, 7], [14, 5], [16, 3], [18, 1]])[:N, :]
    elif map_name == 'alamillo_lake':
        initial_positions = np.array([[68, 26], [64, 26], [60, 26], [56, 26]])[:N, :]
    elif map_name == 'ypacarai_map':
        initial_positions = np.array([[24, 21],[28,24],[27,19],[24,24]])

    env_kwargs = {
        "scenario_map": sc_map,
        "fleet_initial_positions": initial_positions,
        "distance_budget": args_env.distance_budget,
        "number_of_vehicles": N,
        "seed": args_eval.seed,
        "miopic": args_env.miopic,
        "dynamic": args_env.dynamic,
        "detection_length": args_env.detection_length,
        "movement_length": args_env.movement_length,
        "max_collisions": 15,
        "reward_type": args_env.reward_type,
        "convert_to_uint8": args_env.convert_to_uint8,
        "ground_truth_type": benchmark_name,
        "obstacles": False,
        "frame_stacking": 1
    }

    expert_kwargs = {
        "device": args_eval.device,
        "masked_actions": True,
        "consensus": True,
    }

    from Environment.PatrollingEnvironments import MultiAgentPatrolling
    from Algorithm.RainbowDQL.Agent.Expert_nu import Expert_nu

    env_fn = make_env(
        env_fn=MultiAgentPatrolling,
        expert_nu_fn=Expert_nu,
        env_kwargs=env_kwargs,
        expert_kwargs=expert_kwargs,
        seed=args_eval.seed,
        device_int=args_eval.device,
        args=args_env,
        idx=0,
        capture_video=False,
        run_name="greedy_agent"
    )
    env = env_fn()
    agent = GreedyAgent(env)

    # Set up metrics logging
    metrics_directory = f"{args_eval.metrics_directory}Results_seed_{args_eval.seed}_{map_name}_{benchmark_name}/greedy_"
    os.makedirs(metrics_directory, exist_ok=True)

    metrics = MetricsDataCreator(
        metrics_names=['Policy Name', 'Accumulated Reward', 'Total Length', 'nu', 'Percentage of Trash Cleaned', 'Percentage Visited'],
        algorithm_name='Greedy',
        experiment_name='GreedyResults',
        directory=metrics_directory
    )

    paths = MetricsDataCreator(
        metrics_names=['vehicle', 'x', 'y'],
        algorithm_name='Greedy',
        experiment_name='Greedy_paths',
        directory=metrics_directory
    )

    if os.path.exists(metrics_directory + 'GreedyResults.csv'):
        metrics.load_df(metrics_directory + 'GreedyResults.csv')
    if os.path.exists(metrics_directory + 'Greedy_paths.csv'):
        paths.load_df(metrics_directory + 'Greedy_paths.csv')

    for episode in trange(args_eval.num_episodes, desc="Evaluating Greedy"):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        total_length = 0

        nu_ = np.asarray([0. for _ in env.fleet.vehicles])
        percentage_visited = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)
        percentage_cleaned = 0.0

        metrics_list = ["Greedy", total_reward, total_length, nu_, percentage_cleaned, percentage_visited]
        metrics.register_step(run_num=episode, step=total_length, metrics=metrics_list)

        for veh_id, veh in enumerate(env.fleet.vehicles):
            paths.register_step(run_num=episode, step=total_length, metrics=[veh_id, veh.position[0], veh.position[1]])

        while not done:
            nu_vector = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(nu_vector)
            done = terminated or truncated

            total_reward += reward
            obs = next_obs
            total_length += 1
            percentage_visited = env.percentage_of_map_visited
            percentage_cleaned = env.percentage_of_trash_cleaned
            nu_ = nu_vector

            metrics_list = ["Greedy", total_reward, total_length, nu_, percentage_cleaned, percentage_visited]
            metrics.register_step(run_num=episode, step=total_length, metrics=metrics_list)

            for veh_id, veh in enumerate(env.fleet.vehicles):
                paths.register_step(run_num=episode, step=total_length, metrics=[veh_id, veh.position[0], veh.position[1]])

            if args_eval.render:
                env.render()

    metrics.register_experiment()
    paths.register_experiment()
    print("✅ Greedy agent evaluation completed.")
    env.close()
