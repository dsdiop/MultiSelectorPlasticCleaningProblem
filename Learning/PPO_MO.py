import sys
import os
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)

from Learning.utils import make_env, PreferencePPOAgent, GreedyAgent, PreferenceMAPPOAgent
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator

import torch
import gym
import json
import numpy as np
import argparse
import random
from tqdm import tqdm
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV


from mpl_toolkits.mplot3d import Axes3D

ARCHIVE_PATH = "ppo_evaluation_archive_MAPPO_steps_256_numenvs_16.csv"

def load_archive(path=ARCHIVE_PATH):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame(columns=["weight_clean", "weight_explore", "cleaning", "exploration", "time"])

def save_archive(df, path=ARCHIVE_PATH):
    df.to_csv(path, index=False)

def construct_3d_pareto_front(df):
    objectives = df[["cleaning", "exploration", "time"]].to_numpy()
    # Since pymoo does minimization, and we want to maximize cleaning/exploration and minimize time,
    # we invert the cleaning and exploration scores.
    objectives_to_minimize = np.column_stack([-objectives[:, 0], -objectives[:, 1], objectives[:, 2]])

    nds = NonDominatedSorting().do(objectives_to_minimize, only_non_dominated_front=True)
    pareto_df = df.iloc[nds]
    return pareto_df
def construct_2d_pareto_front(df, objective_1="cleaning", objective_2="exploration"):
    objectives = df[[objective_1, objective_2]].to_numpy()
    # Since pymoo does minimization, and we want to maximize cleaning/exploration,
    # we invert the cleaning and exploration scores.
    if "time" in objective_1:
        sign_1 = 1
    else:
        sign_1 = -1
    if "time" in objective_2:
        sign_2 = 1
    else:
        sign_2 = -1

    objectives_to_minimize = np.column_stack([sign_1 * objectives[:, 0], sign_2 * objectives[:, 1]])

    nds = NonDominatedSorting().do(objectives_to_minimize, only_non_dominated_front=True)
    pareto_df = df.iloc[nds]
    return pareto_df
def plot_2d_pareto(df, objective_1="cleaning", objective_2="exploration"):
    if "time" in objective_1:
        units_1 = "steps"
        mult_1 = 1
    else:
        units_1 = "%"
        mult_1 = 100
    if "time" in objective_2:
        units_2 = "steps"
        mult_2 = 1
    else:
        units_2 = "%"
        mult_2 = 100
    # Create a scatter plot for the 2D Pareto front
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[objective_1]*mult_1, df[objective_2]*mult_2, c='red', marker='o')
    ax.set_xlabel(f"{objective_1.capitalize()} ({units_1})")
    ax.set_ylabel(f"{objective_2.capitalize()} ({units_2})")
    ax.set_title("2D Pareto Front (Greedy Agent)")
    ax.grid(True)

    # Annotate points with their weights
    for i, row in df.iterrows():
        ax.annotate(f"{row['weight_explore']:.2f}, {row['weight_clean']:.2f}",
                    (row[objective_1]*mult_1, row[objective_2]*mult_2),
                    textcoords="offset points", xytext=(0, 5), ha='center')
    plt.tight_layout()
    plt.show()
    
def plot_3d_pareto(df):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(df["cleaning"]*100, df["exploration"]*100, df["time"], c='red', marker='o')
    ax.set_xlabel("Cleaning (%)")
    ax.set_ylabel("Exploration (%)")
    ax.set_zlabel("Time (steps)")
    ax.set_title("3D Pareto Front (Greedy Agent)")
    # Annotate points with their weights
    for i, row in df.iterrows():
        ax.text(row["cleaning"]*100, row["exploration"]*100, row["time"],
                f"{row['weight_explore']:.2f}, {row['weight_clean']:.2f}",
                size=8, zorder=1, color='k')
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    
def compute_hypervolume(pareto_df, ref_point=np.array([-0.0, -0.0, 250.0])):
    # Negate the objectives we want to maximize (cleaning and exploration)
    objectives = pareto_df[["cleaning", "exploration", "time"]].to_numpy()
    objectives_to_minimize = np.column_stack([-100*objectives[:, 0], -100*objectives[:, 1], objectives[:, 2]])

    hv = HV(ref_point=ref_point)
    volume = hv(objectives_to_minimize)
    return volume
def compute_hypervolume_2d(pareto_df, objective_1="cleaning", objective_2="exploration", ref_point=np.array([-0.0, -0.0])):
    # Negate the objectives we want to maximize (cleaning and exploration)
    objectives = pareto_df[[objective_1, objective_2]].to_numpy()
    if "time" in objective_1:
        mult_1 = 1
    else:
        mult_1 = -100
    if "time" in objective_2:
        mult_2 = 1
    else:
        mult_2 = -100
    objectives_to_minimize = np.column_stack([mult_1*objectives[:, 0], mult_2*objectives[:, 1]])

    hv = HV(ref_point=ref_point)
    volume = hv(objectives_to_minimize)
    return volume
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
            #   'malaga_port__macro_plastic__0__1751293154' # np.sum 10000000 total steps 100 budget
            # 'malaga_port__macro_plastic__0__1753111554' # np.sum 1000000 total steps preference agent
            # 'malaga_port__macro_plastic__0__1753134709' # np.sum 1000000 total steps preference agent bottleneck
            # 'malaga_port__macro_plastic__0__1756814856' # np.sum 1000000 total steps preference MAPPO agent 
            # 'malaga_port__macro_plastic__0__MAPPO_1_1757035844', # Preference MAPPO agent actor/critic with same hidden layers
            # 'malaga_port__macro_plastic__0__MAPPO_1_1756994574' # Preference MAPPO agent actor/critic with different hidden layers
            'malaga_port__macro_plastic__0__MAPPO_1_1757036545' # Preference MAPPO agent same actor/critic steps 256 - numenvs 16
]

DEFAULT_RUNS_DIR = [f"{data_path}/runs/{run}" for run in RUNS_DIR]
DEFAULT_MODEL_NAME = "ppo_agent.pth"
DEFAULT_ARGS_NAME = "args.json"
METRICS_DIRECTORY = f"{data_path}/Evaluation/Results/"
# metrics_directory= f'{data_path}Results_seed_{seed}_nu_steps_dist_field_{args.map}_30keps/{policy_type}_{nu_step}',

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO or Greedy agent.")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes to evaluate.")
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
    parser.add_argument("--agent", type=str, choices=["ppo", "greedy"], default="ppo", help="Agent type to evaluate")
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




        device = torch.device(f"cuda:{args_eval.device}" if torch.cuda.is_available() and args_eval.device >= 0 else "cpu")

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
        # create a rng to create an array of two float values
        weights_rng = np.random.default_rng(args_eval.seed)
        num_weights = 100
        for run in tqdm(range(num_weights)):
            weights = weights_rng.uniform(0, 1, size=(2,))
            # normalize the weights
            weights /= np.sum(weights)
            archive_df = load_archive()
            w_explore, w_clean = round(weights[0], 5), round(weights[1], 5)

            # Check if weights already evaluated
            existing = archive_df[
                (archive_df["weight_clean"] == w_clean) & (archive_df["weight_explore"] == w_explore)
            ]
            if not existing.empty:
                print(f"✅ Skipping already evaluated weights: {weights}")
                continue
            
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
            
            agent = PreferenceMAPPOAgent(env).to(device)
            agent.load_state_dict(torch.load(model_path, map_location=device))
            agent.eval()

            
            total_length_sum = 0
            percentage_of_trash_cleaned = 0
            percentage_visited = 0

            for episode in trange(args_eval.num_episodes, desc=f"Evaluating "):
                obs, _ = env.reset(preference_weight=tuple(weights))
                
                if args_eval.render:# and run==0:
                    fig = env.render()
                done = False
                
                total_reward = 0
                total_length = 0
                # nu_ is a list of nu of all the agents
                nu_ = np.asarray([0. for veh in env.fleet.vehicles])
                
                while not done:
                    if args_eval.agent == "ppo":
                        # obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                        obs_tensor = {
                            "trash_map": torch.tensor(obs["trash_map"], dtype=torch.float32, device=device).unsqueeze(0),
                            "shared_scalars": torch.tensor(obs["shared_scalars"], dtype=torch.float32, device=device).unsqueeze(0),
                            "agent_data": []
                        }
                        N = len(obs["agent_data"])
                        for i in range(N):
                            agent_obs = obs["agent_data"][i]
                            obs_tensor["agent_data"].append({
                                "ego_pos": torch.tensor(agent_obs["ego_pos"], dtype=torch.float32, device=device).unsqueeze(0),
                                "other_pos": torch.tensor(agent_obs["other_pos"], dtype=torch.float32, device=device).unsqueeze(0),
                                "distance_budget": torch.tensor(agent_obs["distance_budget"], dtype=torch.float32, device=device).unsqueeze(0),
                            })
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
                    
                percentage_visited += env.percentage_of_map_visited
                percentage_of_trash_cleaned += env.percentage_of_trash_cleaned
                total_length_sum += total_length
                    # nu_ = action.cpu().numpy()[0]
            avg_cleaning = percentage_of_trash_cleaned / args_eval.num_episodes
            avg_exploration = percentage_visited / args_eval.num_episodes
            avg_time = total_length_sum / args_eval.num_episodes

            archive_df.loc[len(archive_df)] = [w_clean, w_explore, avg_cleaning, avg_exploration, avg_time]
            print(f"✅ Evaluated weights {weights} → Clean: {avg_cleaning:.2f}, Explore: {avg_exploration:.2f}, Time: {avg_time}")
            save_archive(archive_df)
            
            if args_eval.render:
                plt.close()
            # print(f"✅ Evaluation for run '{policy_name}' complete.")
            env.close()
        archive = load_archive()
        pareto_df = construct_3d_pareto_front(archive)

        # Compute hypervolume
        hv = compute_hypervolume(pareto_df)
        print(f"🌌 Hypervolume of Pareto front: {hv:.4f}")

        # List weight pairs on the Pareto front
        weights_on_pf = pareto_df[["weight_clean", "weight_explore", "cleaning", "exploration", "time"]].to_numpy()

        print("🔍 Weight pairs on 3D Pareto front:")
        for i, (wc, we, clean, expl, time_) in enumerate(weights_on_pf):
            print(f"  {i+1:2d}: weight_clean = {wc:.4f}, weight_explore = {we:.4f}, cleaning = {clean:.4f}, exploration = {expl:.4f}, time = {time_:.2f}")

        # How many points on the Pareto Front repeats
        num_repeats = pareto_df.duplicated(subset=["cleaning", "exploration", "time"]).sum()
        print(f"🔄 Number of repeated points on 3D Pareto front: {num_repeats}")

        # How many points on the Pareto Front are unique
        num_unique = pareto_df.drop_duplicates(subset=["cleaning", "exploration", "time"]).shape[0]
        print(f"🔄 Number of unique points on 3D Pareto front: {num_unique}")

        # Plot the Pareto front
        plot_3d_pareto(pareto_df)

        objective_1="exploration"
        objective_2="cleaning"
        pareto_df = construct_2d_pareto_front(archive, objective_1=objective_1, objective_2=objective_2)
        # Compute Hypervolume
        # if "time" in objective_i then ref_point[i] of that objective is 250 else is -0.0
        ref_point = np.array([-0.0, -0.0])
        if "time" in objective_1:
            ref_point[0] = 250.0
        elif "time" in objective_2:
            ref_point[1] = 250.0
            
        hv_2d = compute_hypervolume_2d(pareto_df, objective_1=objective_1, objective_2=objective_2, ref_point=ref_point)
        print(f"\n🌌 Hypervolume of 2D Pareto front: {hv_2d:.4f}")

        # List weight pairs on the 2D Pareto front
        weights_on_pf = pareto_df[["weight_clean", "weight_explore", "cleaning", "exploration"]].to_numpy()

        print("🔍 Weight pairs on 2D Pareto front:")
        for i, (wc, we, clean, expl) in enumerate(weights_on_pf):
            print(f"  {i+1:2d}: weight_clean = {wc:.4f}, weight_explore = {we:.4f}, cleaning = {clean:.4f}, exploration = {expl:.4f}")

        # How many points on the Pareto Front repeats
        num_repeats = pareto_df.duplicated(subset=["cleaning", "exploration"]).sum()
        print(f"🔄 Number of repeated points on 2D Pareto front: {num_repeats}")

        # How many points on the Pareto Front are unique
        num_unique = pareto_df.drop_duplicates(subset=["cleaning", "exploration"]).shape[0]
        print(f"🔄 Number of unique points on 2D Pareto front: {num_unique}")
        # Plot the 2D Pareto front
        plot_2d_pareto(pareto_df, objective_1=objective_1, objective_2=objective_2)
