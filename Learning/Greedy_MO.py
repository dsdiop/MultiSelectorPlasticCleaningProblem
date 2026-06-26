import sys
import os
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)

from Learning.utils import make_env, Agent, GreedyAgent
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

from scipy.stats import wilcoxon

from mpl_toolkits.mplot3d import Axes3D

ARCHIVE_PATH = ["greedy_evaluation_WP_reward.csv",
                "greedy_evaluation_WS_reward.csv",
                "greedy_evaluation_WPOP_reward_normalized.csv",
                "greedy_evaluation_EWC_reward_normalized.csv",
                # "ppo_evaluation_archive_MAPPO_steps_256_numenvs_16.csv",
                # "greedy_evaluation_EWC_reward.csv",
                # "greedy_evaluation_WS_reward_normalized.csv",
                # "greedy_evaluation_WP_reward_normalized.csv"
                ]
ARCHIVE_PATH = ["greedy_evaluation_EWC_alamillo.csv",
                "greedy_evaluation_WPOP_alamillo.csv",
                "greedy_evaluation_WS_alamillo.csv",
                "greedy_evaluation_WP_alamillo.csv"
                ]
ARCHIVE_PATH0 = ["greedy_evaluation_EWC_1.csv",
                "greedy_evaluation_EWC_3.csv",
                "greedy_evaluation_EWC_5.csv",
                "greedy_evaluation_EWC_7.csv",]
ARCHIVE_PATH = ["greedy_evaluation_WP_1.csv",
                "greedy_evaluation_WP_3.csv",
                "greedy_evaluation_WP_5.csv",
                "greedy_evaluation_WP_7.csv",
                "greedy_evaluation_WP_9.csv",]
def load_archive(path=ARCHIVE_PATH):
    full_path = f"{data_path}/{path}"
    if os.path.exists(full_path):
        return pd.read_csv(full_path)
    else:
        return pd.DataFrame(columns=["weight_clean", "weight_explore", "cleaning", "exploration", "time"])


def save_archive(df, path=ARCHIVE_PATH):
    df.to_csv(path, index=False)
    
def compare_pareto_metrics(df1, df2, objective_1="cleaning", objective_2="exploration"):
    """Compare two Pareto fronts using Wilcoxon signed-rank test for both objectives."""
    common_cols = [objective_1, objective_2]
    results = {}
    for obj in common_cols:
        min_len = min(len(df1[obj]), len(df2[obj]))
        if min_len < 5:
            results[obj] = {"stat": np.nan, "pvalue": np.nan, "note": "too few samples"}
            continue
        try:
            stat, pvalue = wilcoxon(df1[obj].iloc[:min_len], df2[obj].iloc[:min_len])
            results[obj] = {"stat": stat, "pvalue": pvalue}
        except ValueError as e:
            results[obj] = {"stat": np.nan, "pvalue": np.nan, "note": str(e)}
    return results

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
    objectives_to_minimize = np.column_stack([-objectives[:, 0], -objectives[:, 1], objectives[:, 2]])

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

def compute_spread(pareto_points): ## mal calculated
    """
    Spread (Δ metric): how evenly spaced the solutions are along the Pareto front.
    pareto_points: (N,2) array of Pareto-optimal solutions.
    """
    # Sort by first objective
    pareto_points = pareto_points[np.argsort(pareto_points[:,0])]
    distances = np.linalg.norm(np.diff(pareto_points, axis=0), axis=1)
    d_bar = np.mean(distances)
    # Distances between extreme points and extremes of front
    d_f = np.linalg.norm(pareto_points[0] - pareto_points[-1])
    # Δ = (d_f + sum(|d_i - d_bar|)) / (d_f + (N-1)*d_bar)
    delta = (d_f + np.sum(np.abs(distances - d_bar))) / (d_f + (len(distances) * d_bar))
    return delta

def compute_spacing(pareto_points):
    """
    Spacing metric: measures variance of distances between consecutive solutions.
                    This metric better captures gaps along the front than nearest-neighbor spacing in 2D (Original Spacing metric).
    """
    pareto_points = pareto_points[np.argsort(pareto_points[:,0])]
    distances = np.linalg.norm(np.diff(pareto_points, axis=0), axis=1)
    return np.std(distances, ddof=1)

def M3_metric(pareto_points):
    """
    Compute Zitzler's M3 metric extent of the front 
    Args:
        front: np.ndarray of shape (N, m)
    Returns:
        M3: float
    """
    max_dists = np.ptp(pareto_points, axis=0)   # peak to peak (max - min) for each column)
    M3 = np.sqrt(np.sum(max_dists))
    return M3

def collect_reward_stats(env_fn, num_episodes=10):
    """Run warmup episodes to estimate reward normalization values."""
    rewards = []
    print(f"\n🔥 Running {num_episodes} warmup episodes to normalize rewards...")
    env = env_fn()
    for ep in trange(num_episodes, desc="Warmup"):
        obs, _ = env.reset()
        done, total_reward = False, 0
        while not done:
            # Random nu of 0 or 1 for warmup
            nu_ = np.asarray([np.random.choice([0, 1]) for _ in env.fleet.vehicles])
            next_obs, reward, terminated, truncated, info = env.step(nu_)
            done = terminated or truncated
            sum_reward = np.sum(info['reward_components'], axis=0)
            rewards.append(sum_reward)
    env.close()
    return {"min_reward": np.min(np.asarray(rewards), axis=0), "max_reward": np.max(np.asarray(rewards), axis=0)}

def compute_igd(pareto_points, reference_front, k=100):
    """
    Inverted Generational Distance (IGD):
    Average distance from points in a reference front to the closest Pareto point.
    - pareto_points: (N,2) array of your solutions
    - reference_front: (M,2) array of true/approximate front, or None to auto-generate linearly
    """
    if reference_front is None:
        # auto-generate a linear reference front between extremes
        x = np.linspace(pareto_points[:,0].min(), pareto_points[:,0].max(), k)
        y = np.linspace(pareto_points[:,1].max(), pareto_points[:,1].min(), k)
        reference_front = np.column_stack([x, y])

    dists = []
    for r in reference_front:
        d = np.min(np.linalg.norm(pareto_points - r, axis=1))
        dists.append(d)
    return np.mean(dists)

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
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes to evaluate.")
    parser.add_argument("--seed", type=int, default=3, help="Random seed for reproducibility.")
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
    parser.add_argument("--track_nu",type=bool, default=True,  help="Track nu values over time.")
    args = parser.parse_args()
    if isinstance(args.runs, str):
        args.runs = [args.runs]
    return args


if __name__ == "__main__":
    # ARCHIVE_PATH can now be a string or list
    ARCHIVE_PATHS = ARCHIVE_PATH if isinstance(ARCHIVE_PATH, list) else [ARCHIVE_PATH]
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
        # create a rng to create an array of two float values
        weights_rng = np.random.default_rng(args_eval.seed)
        num_weights = 0
        #####################
        # # weights list are of length num_weights and each element is an array of two float values ranging from (0,1) to (1,0) evenly spaced
        weights_list = [1 - 0.324 for i in range(num_weights)] # 0.0090 - EWC ---- 0.8377 - WS --- 0.32400 - WPOP 
        #########################
        for run in tqdm(range(num_weights)):
            # weights = weights_rng.uniform(0, 1)
            ##########################
            weights = weights_list[run]
            #############################
            # normalize the weights
            # weights /= np.sum(weights)
            weights = np.array([weights, 1 - weights])
            print(f"\n🎯 Evaluating weights: {weights}")
            archive_df = load_archive()
            w_explore, w_clean = round(weights[0], 3), round(weights[1], 3)

            # Check if weights already evaluated
            existing = archive_df[
                (archive_df["weight_clean"] == w_clean) & (archive_df["weight_explore"] == w_explore)
            ]
            if not existing.empty and False:
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
            if run == 0:
                # Collect reward stats only once per run
                reward_stats = collect_reward_stats(env_fn, num_episodes=2)
                print(f"Reward stats: {reward_stats}")
            env = env_fn()
            agent = GreedyAgent(env, reward_stats=reward_stats)
            print(f"Using {agent.greedy_type} Greedy Agent for evaluation.")
            
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
                        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                        with torch.no_grad():
                            action, _, _, _ = agent.get_action_and_value(obs_tensor)
                        nu_ = action.cpu().numpy()[0]
                    else:
                        nu_ = agent.act(obs)
                    if args_eval.track_nu:
                        # Track nu_ evolution for each agent
                        if total_length == 0:
                            nu_history = [[] for _ in range(len(nu_))]
                        for i, nu_val in enumerate(nu_):
                            nu_history[i].append(nu_val)
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
                if args_eval.track_nu:
                    # Plot nu_ evolution for each agent in a subplot, color for each agent
                    
                    for i in range(len(nu_history)):
                        mean_nu = np.cumsum(nu_history[i]) / np.arange(1, len(nu_history[i]) + 1)
                        if i == 0:
                            mean_nu_history = [mean_nu]
                        else:
                            mean_nu_history.append(mean_nu)
                    fig, axs = plt.subplots(len(nu_history), 1, figsize=(8, 2 * len(nu_history)))
                    mean_nu = []
                    for i, nu_vals in enumerate(nu_history):
                        mean_nu_ = np.cumsum(nu_history[i]) / np.arange(1, len(nu_history[i]) + 1)
                        mean_nu.append(round(mean_nu_[-1], 2))
                        print(f"Agent {i} - Mean nu: {mean_nu[i]:.2f}")
                        axs[i].plot(nu_vals, drawstyle='steps-post', color=plt.cm.tab10(i / len(nu_history)), linewidth=3)
                        # axs[i].plot(mean_nu_, color=plt.cm.tab10(i / len(nu_history)), linestyle='--')
                        axs[i].set_title(f"Agent {i}", fontsize=20)    
                        # Apply custom binary ticks
                        axs[i].set_yticks([0, 1])
                        axs[i].set_yticklabels(["Cleaning", "Exploration"], fontsize=20)
                        # axs[i].set_xlabel("Time step")
                        # axs[i].set_ylabel("nu value")
                        axs[i].tick_params(axis='both', which='major', labelsize=20)
                        axs[i].set_ylim(-0.1, 1.1)
                        # axs[i].legend(["nu value", "Mean nu value"])
                    plt.xlabel("Time step", fontsize=20)
                    #set one y label one big centered label
                    # Add a single centered y-label for the entire figure (supports older matplotlib versions)
                    try:
                        fig.supylabel(r"$\nu$ value", fontsize=20)
                    except AttributeError:
                        fig.text(0.04, 0.5, "nu value", va="center", rotation="vertical", fontsize=18)
                    # plt.yticks([0, 1], ["Cleaning", "Exploration"])
                    # plt.ylabel("nu value")
                    # plt.suptitle(f"Nu values over time for weights {weights}")
                    # plot for each step, the mean nu value of each agent at that step,
                    # i.e, for each agent at each step, plot the mean nu value 
                    # (averaged over the step until that step) of that agent at that step,
                    plt.tight_layout()
                    # plt.savefig(f"{data_path}/Evaluation/Results/nu_evolution_weights_{w_explore}_{w_clean}_greedy_type_{agent.greedy_type}.png")
                    plt.show()
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
        # archive = load_archive()
        # pareto_df = construct_3d_pareto_front(archive)

        # # Compute hypervolume
        # hv = compute_hypervolume(pareto_df)
        # print(f"🌌 Hypervolume of Pareto front: {hv:.4f}")

        # # List weight pairs on the Pareto front
        # weights_on_pf = pareto_df[["weight_clean", "weight_explore", "cleaning", "exploration", "time"]].to_numpy()

        # print("🔍 Weight pairs on Pareto front:")
        # for i, (wc, we, clean, expl, time_) in enumerate(weights_on_pf):
        #     print(f"  {i+1:2d}: weight_clean = {wc:.4f}, weight_explore = {we:.4f}, cleaning = {clean:.4f}, exploration = {expl:.4f}, time = {time_:.2f}")

        # # Plot the Pareto front
        # plot_3d_pareto(pareto_df)
        objective_1 = "cleaning"
        objective_2 = "exploration"

        ref_point = np.array([-0.0, -0.0])
        if "time" in objective_1:
            ref_point[0] = 250.0
        elif "time" in objective_2:
            ref_point[1] = 250.0

        # --- Prepare unified plot ---
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(ARCHIVE_PATHS)))
        # different markers for each archive
        markers = ['o', 's', 'H', '^', 'v', '<', '>', 'p', '*', 'D']
        pareto_dfs = []  # Keep all DF for pairwise tests
        all_points = []  # For combined Pareto front
        for i, path in enumerate(ARCHIVE_PATHS):
            archive = load_archive(path)
            pareto_df = construct_2d_pareto_front(archive, objective_1=objective_1, objective_2=objective_2)
            pareto_dfs.append((path, pareto_df))
            # add the Scalarization column if not present
            if 'Scalarization' not in pareto_df.columns:
                is_normalized = "normalized" in path
                reward_type_str = os.path.basename(path).split("_")[-2] if not is_normalized else f'{os.path.basename(path).split("_")[-3]}'
                pareto_df['Scalarization'] = f"{reward_type_str}" # + ("_normalized" if is_normalized else "")
            all_points.append(pareto_df)
            # Compute metrics
            hv_2d = compute_hypervolume_2d(pareto_df, objective_1=objective_1, objective_2=objective_2, ref_point=ref_point)
            pareto_points = pareto_df[[objective_1, objective_2]].to_numpy()
            spacing_val = compute_spacing(100 * pareto_points)
            m3_val = M3_metric(100 * pareto_points)

            print(f"\n📁 {path}")
            print(f"🌌 Hypervolume: {hv_2d:.4f}")
            print(f"M3 metric: {m3_val}")
            print(f"Spacing: {spacing_val}")

            num_repeats = pareto_df.duplicated(subset=[objective_1, objective_2]).sum()
            num_unique = pareto_df.drop_duplicates(subset=[objective_1, objective_2]).shape[0]
            print(f"🔄 Repeated: {num_repeats}, Unique: {num_unique}")

            # --- Overlay custom scatter ---
            if "time" in objective_1:
                units_1, mult_1 = "steps", 1
            else:
                units_1, mult_1 = "%", 100
            if "time" in objective_2:
                units_2, mult_2 = "steps", 1
            else:
                units_2, mult_2 = "%", 100
            reward_type_str = os.path.basename(path).split("_")[-2] if "normalized" not in path else f'{os.path.basename(path).split("_")[-3]}'
            reward_type_str += f" (a={(os.path.basename(path).split('_')[-1]).replace('.csv','')})"
            # plot points united by lines
            #first sort pareto_df by objective_1
            pareto_df = pareto_df.sort_values(by=objective_1, ascending=False)
            if "W" not in reward_type_str and False:
                reward_type_str = f"PPO Conditioned Network"
            ax.plot(
                pareto_df[objective_1] * mult_1,
                pareto_df[objective_2] * mult_2,
                label=f"{reward_type_str} Pareto (HV={hv_2d:.3f})",
                color=colors[i],
                marker=markers[i % len(markers)],
                markersize=5,
                alpha=0.9,
                linestyle='--',
            )
        # --- Build and plot combined Pareto front ---
        combined_df = pd.concat(all_points, ignore_index=True)
        combined_pareto = construct_2d_pareto_front(combined_df, objective_1=objective_1, objective_2=objective_2)
        combined_points = combined_pareto[[objective_1, objective_2]].to_numpy()
        hv_combined = compute_hypervolume_2d(combined_pareto, objective_1=objective_1, objective_2=objective_2, ref_point=ref_point)
        spacing_combined = compute_spacing(100 * combined_points)
        m3_combined = M3_metric(100 * combined_points)

        print("\n📊 Combined Pareto Front (All CSVs):")
        print(f"🌌 Hypervolume: {hv_combined:.4f}")
        print(f"M3 metric: {m3_combined}")
        print(f"Spacing: {spacing_combined}")
        print(f"Points: {len(combined_pareto)}")
        # print combined pareto front points and its weights
        # sort combined_pareto by objective_1
        combined_pareto = combined_pareto.sort_values(by=objective_1, ascending=False)
        print("🔍 Combined Pareto Front Points:")
        for i, row in combined_pareto.iterrows():
            print(f"  {row['weight_clean']:.4f}, {row['weight_explore']:.4f} | {objective_1} = {row[objective_1]:.4f}, {objective_2} = {row[objective_2]:.4f}")
        
        # ax.plot(
        #     combined_pareto[objective_1] * mult_1,
        #     combined_pareto[objective_2] * mult_2,
        #     label=f"Combined Pareto (HV={hv_combined:.3f})",
        #     color='black',
        #     marker='X',
        #     markersize=5,
        #     alpha=0.6,
        #     linestyle='-',
        # )
        # PLOT THE WEIGHTS WHERE THE POINTS ARE
        # for i, row in combined_pareto.iterrows():
        #     ax.annotate(f"{row['weight_explore']:.4f}, {row['weight_clean']:.4f}",
        #                 (row[objective_1]*mult_1, row[objective_2]*mult_2),
        #                 textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='black')
        pareto_dfs.append(("Combined Pareto", combined_pareto))
        # --- Wilcoxon pairwise tests ---
        if len(pareto_dfs) > 1:
            print("\n📊 Pairwise Wilcoxon Tests Between Pareto Fronts:")
            for i in range(len(pareto_dfs)):
                for j in range(i + 1, len(pareto_dfs)):
                    name_i, df_i = pareto_dfs[i]
                    name_j, df_j = pareto_dfs[j]
                    results = compare_pareto_metrics(df_i, df_j, objective_1, objective_2)
                    print(f"\n🧪 {os.path.basename(name_i)} vs {os.path.basename(name_j)}:")
                    for obj, res in results.items():
                        if "note" in res:
                            print(f"  {obj.capitalize()}: {res['note']}")
                        else:
                            signif = "✅ Significant" if res["pvalue"] < 0.05 else "❌ Not significant"
                            print(f"  {obj.capitalize()}: stat={res['stat']:.4f}, p={res['pvalue']:.4f} → {signif}")

        
        # --- Final formatting ---
        ax.set_xlabel(f"{objective_1.capitalize()} ({units_1})", fontsize=16)
        ax.set_ylabel(f"{objective_2.capitalize()} ({units_2})", fontsize=16)
        ax.set_title("2D Pareto Fronts (Comparison)", fontsize=17)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=16)
        # ticks size
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        # plt.savefig("combined_pareto_front1.png", dpi=300)
        plt.show()
        
        
        # PLOT THE COMBINED PARETO FRONT WITH THE LINES REPRESENTING THE WEIGHTS IN A SUBPLOT
        # Subplot: top = combined Pareto, bottom = weight tendencies
        cp_sorted = combined_pareto.sort_values(by=objective_1).reset_index(drop=True)

        fig2, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1]})

        # Top: Combined Pareto Front (bigger)
        ax_top.plot(
            cp_sorted[objective_1] * mult_1,
            cp_sorted[objective_2] * mult_2,
            marker='X',
            color='black',
            markersize=6,
            linestyle='-',
            alpha=0.9,
            label=f"Combined Pareto (HV={hv_combined:.3f})"
        )
        # # annotate scalarization on each point
        # for i, row in cp_sorted.iterrows():
        #     ax_top.annotate(f"{row['Scalarization']}",
        #                 (row[objective_1]*mult_1, row[objective_2]*mult_2),
        #                 textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='black')

        ax_top.set_xlabel(f"{objective_1.capitalize()} ({units_1})")
        ax_top.set_ylabel(f"{objective_2.capitalize()} ({units_2})")
        ax_top.set_title("Combined Pareto Front")
        ax_top.grid(True, linestyle="--", alpha=0.6)
        ax_top.legend()

        # Bottom: Weights Trend
        # Bottom: Weights Trend
        # x positions correspond to the x-axis of the top plot (objective_1 values scaled)
        x = cp_sorted[objective_1].to_numpy() * mult_1

        # Ensure Scalarization column exists
        if 'Scalarization' not in cp_sorted.columns:
            cp_sorted['Scalarization'] = 'Unknown'

        unique_scalars = cp_sorted['Scalarization'].unique()
        for i, scalar in enumerate(unique_scalars):
            mask = cp_sorted['Scalarization'] == scalar
            y = np.full_like(x, np.nan, dtype=float)
            y[mask.to_numpy()] = cp_sorted.loc[mask, 'weight_clean'].to_numpy()
            # Delete points where y is nan to avoid breaks in the line
            mask_nan = ~np.isnan(y)
            ax_bottom.plot(
                x[mask_nan],
                y[mask_nan],
                label=str(scalar),
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linewidth=2,
                linestyle='--',
                alpha=0.9,
            )
            # Annotate first and last valid points for quick visual cue
            # valid_idx = np.where(~np.isnan(y))[0]
            # if valid_idx.size:
            #     ax_bottom.annotate(f"{y[valid_idx[0]]:.2f}", (x[valid_idx[0]], y[valid_idx[0]]),
            #                        textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8)
            #     ax_bottom.annotate(f"{y[valid_idx[-1]]:.2f}", (x[valid_idx[-1]], y[valid_idx[-1]]),
            #                        textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8)

        # # annotate scalarization on each point
        
        ax_bottom.set_title("Weights Trend Along Combined Pareto Front")
        # ax_bottom.set_xlabel(f"{objective_1.capitalize()} ({units_1})")
        # NO TICK LABELS ON X AXIS OF TOP PLOT
        ax_bottom.set_xticklabels([])
        ax_bottom.set_ylabel("Weight value")
        ax_bottom.set_ylim(-0.05, 1.05)
        ax_bottom.grid(True, linestyle="--", alpha=0.6)
        ax_bottom.legend()
        # show markers and a light vertical line for each point to link top/bottom visually
        for xv in x:
            ax_bottom.axvline(x=xv, color='gray', alpha=0.18)

        plt.tight_layout()
        # plt.savefig(f"{data_path}/Evaluation/Results/combined_pareto_with_weights_trend.png", dpi=300)
        plt.show()
        plt.close(fig2)
        
        
        
        # --- Count contributions from each CSV to combined Pareto front ---
        contrib_counts = {os.path.basename(p): 0 for p in ARCHIVE_PATHS}

        for src in ARCHIVE_PATHS:
            src_base = os.path.basename(src)
            src_df = pd.concat([df for name, df in pareto_dfs if os.path.basename(name) == src_base], ignore_index=True)
            # Count matches between combined Pareto and this CSV’s points
            for _, row in combined_pareto.iterrows():
                # if row (all it columns ) exists in src_df
                matches = (src_df[objective_1] == row[objective_1]) & (src_df[objective_2] == row[objective_2])
                if np.any(matches):
                    contrib_counts[src_base] += 1
        # change contrib_count keys to the reward type only and the values to percentage of points contributed
        contrib_counts = {k.split("_")[-2] if "normalized" not in k else f"{k.split('_')[-3]}": v/len(combined_pareto) * 100.0  for k, v in contrib_counts.items()}
        print("\n📈 Contribution to Combined Pareto Front:")
        for name, count in contrib_counts.items():
            print(f"  {name}: {count:.2f}%")

        # --- Plot histogram of contributions ---
        plt.figure(figsize=(8, 6))
        plt.bar(contrib_counts.keys(), contrib_counts.values(), color=colors[:len(contrib_counts)], alpha=0.8)
        plt.title("Contribution to Combined Pareto Front",fontsize=17)
        plt.xlabel("Scalarization methods", fontsize=16)
        plt.ylabel("Percentage of Points Contributing (%)", fontsize=16)
        # tick size
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        # plt.savefig("pareto_contributions.png", dpi=300)
        plt.show()
        plt.close()
        # pareto_front_malaga.csv has another df to compare with, with the columns PTC, PMV, Policy. 
        # PTC is cleaning, PMV is exploration, plot its pareto front on the same 2D plot as
        # "combined pareto fronts" for malaga_port map. This is to compare our results with existing literature.
        # if args_env.map == "malaga_port":
        reference_path = "pareto_front_malaga_port.csv"
        reference_path = f"{data_path}/{reference_path}"
        if os.path.exists(reference_path):
            ref_df = pd.read_csv(reference_path)
            ref_pareto = construct_2d_pareto_front(ref_df, objective_1="PTC", objective_2="PMV")
            points_ref = ref_pareto[["PTC", "PMV"]].to_numpy()
            hv_ref = compute_hypervolume_2d(ref_pareto, objective_1="PTC", objective_2="PMV", ref_point=np.array([-0.0, -0.0]))
            spacing_ref = compute_spacing(100 * points_ref)
            m3_ref = M3_metric(100 * points_ref)
            print("\n📊 Reference Pareto Front (Literature):")
            print(f"🌌 Hypervolume: {hv_ref:.4f}")
            print(f"📏 Spacing: {spacing_ref:.4f}")
            print(f"📐 M3: {m3_ref:.4f}")
            #plot the reference pareto front and the combined pareto front together
            plt.figure(figsize=(8, 6))
            plt.plot(
                ref_pareto["PTC"] * 100,
                ref_pareto["PMV"] * 100,
                label="FP2S Pareto (HV={:.3f})".format(hv_ref),
                color='green',
                marker='^',
                markersize=8,
                alpha=0.9,
                linestyle='-',
            )
            print("\n✅ Plotted reference Pareto front from literature.")
            # plot combined pareto front again
            plt.plot(
                combined_pareto[objective_1] * mult_1,
                combined_pareto[objective_2] * mult_2,
                label=f"Combined Pareto (HV={hv_combined:.3f})",
                color='black',
                marker='X',
                markersize=8,
                alpha=0.9,
                linestyle='-',
            )
            # Final formatting
            plt.xlabel(f"{objective_1.capitalize()} ({units_1})", fontsize=16)
            plt.ylabel(f"{objective_2.capitalize()} ({units_2})", fontsize=16)
            plt.title("FP2S vs Combined Pareto Fronts", fontsize=17)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(fontsize=16)
            # tick size
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.tight_layout()
            # plt.savefig("combined_pareto_front_with_reference_v2.png", dpi=300)
            plt.show()