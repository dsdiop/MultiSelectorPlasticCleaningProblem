import sys
import os
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)

from Learning.utils import make_env, Agent, GreedyAgent
from Utils.metrics_wrapper import MetricsDataCreator

import torch
import gym
import json
import numpy as np
import argparse
import random
from tqdm import tqdm
from tqdm import trange
import matplotlib.pyplot as plt

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
            #   'malaga_port__macro_plastic__0__1750189305' # np.sum 10000000 total steps
              'malaga_port__macro_plastic__0__1751293154' # np.sum 10000000 total steps 100 budget
]
DEFAULT_RUNS_DIR = [f"{data_path}/runs/{run}" for run in RUNS_DIR]
DEFAULT_MODEL_NAME = "ppo_agent.pth"
DEFAULT_ARGS_NAME = "args.json"
METRICS_DIRECTORY = f"{data_path}/Evaluation/Results/"
# metrics_directory= f'{data_path}Results_seed_{seed}_nu_steps_dist_field_{args.map}_30keps/{policy_type}_{nu_step}',

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO or Greedy agent.")
    parser.add_argument("--num-episodes", type=int, default=200, help="Number of episodes to evaluate.")
    parser.add_argument("--seed", type=int, default=30, help="Random seed for reproducibility.")
    parser.add_argument("--runs", nargs='+', type=str, default=DEFAULT_RUNS_DIR,
                        help="List of run directories under /runs to evaluate.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help="Name of the model .pth file inside each run directory.")
    parser.add_argument("--args_name", type=str, default=DEFAULT_ARGS_NAME,
                        help="Name of the args .json file inside each run directory.")
    parser.add_argument("--metrics_directory", type=str, default=METRICS_DIRECTORY,
                        help="Directory to save evaluation metrics.")
    parser.add_argument("--device", type=int, default=0, choices=[-1, 0, 1])
    parser.add_argument("--render", type=bool, default=False, help="Render the environment during evaluation.")
    parser.add_argument("--agent", type=str, choices=["ppo", "greedy"], default="greedy", help="Agent type to evaluate")
    args = parser.parse_args()
    if isinstance(args.runs, str):
        args.runs = [args.runs]
    return args


if __name__ == "__main__":
    args_eval = parse_args()
    for run_dir in tqdm(args_eval.runs, desc="Evaluating runs"):
        model_path = os.path.join(run_dir, args_eval.model_name)
        args_path = os.path.join(run_dir, args_eval.args_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
        if not os.path.exists(args_path):
            raise FileNotFoundError(f"Args file {args_path} does not exist.")

        with open(args_path, "r") as f:
            args_env_dict = json.load(f)
        args_env = argparse.Namespace(**args_env_dict)
    # TRY NOT TO MODIFY: seeding
        random.seed(args_eval.seed)
        np.random.seed(args_eval.seed)
        torch.manual_seed(args_eval.seed)
        # env setup
        # envs = gym.vector.SyncVectorEnv(
        #     [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
        # )
        
        # Environment setup
        N = args_env.n_agents
        sc_map = np.genfromtxt(f"{data_path}/Environment/Maps/{args_env.map}.csv", delimiter=',')

        if args_env.map == 'malaga_port':
            initial_positions = np.array([[12, 7], [14, 5], [16, 3], [18, 1]])[:N, :]
        elif args_env.map == 'alamillo_lake':
            initial_positions = np.array([[68, 26], [64, 26], [60, 26], [56, 26]])[:N, :]
        elif args_env.map == 'ypacarai_map':
            initial_positions = np.asarray([[24, 21],[28,24],[27,19],[24,24]])




        device = torch.device("cuda" if torch.cuda.is_available() and args_eval.device >= 0 else "cpu")

        env_kwargs = {
            "scenario_map": sc_map,
            "fleet_initial_positions": initial_positions,
            "distance_budget": 100,
            "number_of_vehicles": N,  # Or use args.num_agents if defined
            "seed": args_eval.seed,
            "miopic": args_env.miopic,
            "dynamic": args_env.dynamic,
            "detection_length": args_env.detection_length,
            "movement_length": args_env.movement_length,
            "max_collisions": 15,
            "reward_type": args_env.reward_type,
            "convert_to_uint8": args_env.convert_to_uint8,
            "ground_truth_type": args_env.benchmark,
            "obstacles": False,
            "frame_stacking": 1
        }

        expert_kwargs = {
        # "env": env, # environment will be passed in the thunk
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
            capture_video=True,
            run_name=os.path.basename(run_dir),
        )
        env = env_fn()
        
        if args_eval.agent == "ppo":
            agent = Agent(env).to(device)
            agent.load_state_dict(torch.load(model_path, map_location=device))
            agent.eval()
        else:
            agent = GreedyAgent(env)

        
        metrics_directory = f"{args_eval.metrics_directory}Results_seed_{args_eval.seed}_{args_env.map}_{args_env.benchmark}"
        if not os.path.exists(metrics_directory):
            os.makedirs(metrics_directory)
        # if greey agent, use greedy as policy name and use HeuristicResults as metrics name
        # if ppo agent, use the run directory name as policy name and DRLResults as metrics name
        if args_eval.agent == "ppo":
            algorithm_name = 'PPO'
            policy_name = algorithm_name + '_' + run_dir.split('/')[-1]
            
        if args_eval.agent == "greedy":
            algorithm_name = 'Greedy'
            policy_name = "Greedy5_"
        else:
            algorithm_name = 'DRL'
            policy_name = run_dir.split('/')[-1]
        metrics_directory= f"{metrics_directory}/{policy_name}_"
        metrics = MetricsDataCreator(metrics_names=['Policy Name',
                                                        'Accumulated Reward',
                                                        'Total Length',
                                                        'nu',
                                                        'Percentage of Trash Cleaned',
                                                        'Percentage Visited'],
                                        algorithm_name=algorithm_name,
                                        experiment_name=f'{algorithm_name}Results',
                                        directory=metrics_directory)
        if os.path.exists(metrics_directory + f'{algorithm_name}Results' + '.csv'):
            metrics.load_df(metrics_directory + f'{algorithm_name}Results' + '.csv')

        paths = MetricsDataCreator(metrics_names=['vehicle', 'x', 'y'],
                                algorithm_name=algorithm_name,
                                experiment_name=f'{algorithm_name}_paths',
                                directory=metrics_directory)

        if os.path.exists(metrics_directory + f'{algorithm_name}_paths' + '.csv'):
            paths.load_df(metrics_directory + f'{algorithm_name}_paths' + '.csv')

        for episode in trange(args_eval.num_episodes, desc=f"Evaluating {policy_name}"):
            obs, _ = env.reset()
            
            if args_eval.render:# and run==0:
                fig = env.render()
            done = False
            
            total_reward = 0
            total_length = 0
            instantaneous_percentage_of_trash_cleaned = 0
            percentage_of_trash_cleaned = 0
            percentage_visited = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)
            # nu_ is a list of nu of all the agents
            nu_ = np.asarray([0. for veh in env.fleet.vehicles])
            metrics_list = [policy_name, total_reward,
                            total_length, nu_,
                            percentage_of_trash_cleaned,
                            percentage_visited]
            total_reward = 0
            
            # Initial register #
            metrics.register_step(run_num=episode, step=total_length, metrics=metrics_list)
            for veh_id, veh in enumerate(env.fleet.vehicles):
                paths.register_step(run_num=episode, step=total_length, metrics=[veh_id, veh.position[0], veh.position[1]])

            while not done:
                if args_eval.agent == "ppo":
                    obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                    with torch.no_grad():
                        action, _, _, _ = agent.get_action_and_value(obs_tensor)
                    nu_ = action.cpu().numpy()[0]
                else:
                    nu_ = agent.act(obs)

                next_obs, reward, terminated, truncated, info = env.step(nu_)
                done = terminated or truncated
                total_reward += reward
                total_length += 1
                if args_eval.render:
                    print(f"Episode/Step: {episode}/{total_length}, Percentage of map visited/cleaned: {env.percentage_of_map_visited:.2f}/{env.percentage_of_trash_cleaned:.2f},  nu: {nu_}")
                    env.render()
                    
                obs = next_obs
                
                percentage_visited = env.percentage_of_map_visited
                total_reward += reward
                percentage_of_trash_cleaned = env.percentage_of_trash_cleaned
                # nu_ = action.cpu().numpy()[0]
                metrics_list = [policy_name, total_reward,
                            total_length, nu_,
                            percentage_of_trash_cleaned,
                            percentage_visited]
                metrics.register_step(run_num=episode, step=total_length, metrics=metrics_list)
                for veh_id, veh in enumerate(env.fleet.vehicles):
                    paths.register_step(run_num=episode, step=total_length, metrics=[veh_id, veh.position[0], veh.position[1]])

        if not args_eval.render:
            metrics.register_experiment()
            paths.register_experiment()
        else:
            plt.close()
        print(f"✅ Evaluation for run '{policy_name}' complete.")
        env.close()