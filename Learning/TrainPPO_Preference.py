
import argparse
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from gym import spaces
from typing import Dict, Tuple
import sys
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)

from Learning.utils import make_env, PreferencePPOAgent
from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.Expert_nu import Expert_nu
from gym.spaces import MultiBinary
from torch.distributions import Bernoulli
from tqdm import trange

def convert_obs(obs_dict, device):
    obs_out = {
        "trash_map": torch.tensor(obs_dict["trash_map"], dtype=torch.float32, device=device),
        "shared_scalars": torch.tensor(obs_dict["shared_scalars"], dtype=torch.float32, device=device),
        "agent_data": []
    }
    N = len(obs_dict["agent_data"])
    for i in range(N):
        agent_obs = obs_dict["agent_data"][i]
        obs_out["agent_data"].append({
            "ego_pos": torch.tensor(agent_obs["ego_pos"], dtype=torch.float32, device=device),
            "other_pos": torch.tensor(agent_obs["other_pos"], dtype=torch.float32, device=device),
            "distance_budget": torch.tensor(agent_obs["distance_budget"], dtype=torch.float32, device=device),
        })
    return obs_out

import torch

def convert_obs_batched(obs_list, device):
    """
    Convert an orderedDict of (from vectorized envs) into
    batched torch tensors on the given device.

    Args:
        obs_list (List[Dict]): observations from envs.reset() or envs.step()
        device: torch.device

    Returns:
        Dict[str, Tensor or List[Dict]]: structured batched tensor observation
    """
    num_envs = len(obs_list)
    n_agents = len(obs_list[0]["agent_data"])

    # Stack shared inputs
    trash_map = torch.stack([torch.tensor(obs["trash_map"], dtype=torch.float32) for obs in obs_list], dim=0)
    shared_scalars = torch.stack([torch.tensor(obs["shared_scalars"], dtype=torch.float32) for obs in obs_list], dim=0)

    # Initialize per-agent lists
    ego_pos_list = [[] for _ in range(n_agents)]
    other_pos_list = [[] for _ in range(n_agents)]
    distance_budget_list = [[] for _ in range(n_agents)]

    # Collect per-agent data
    for obs in obs_list:
        for i in range(n_agents):
            agent_obs = obs["agent_data"][i]
            ego_pos_list[i].append(torch.tensor(agent_obs["ego_pos"], dtype=torch.float64))
            other_pos_list[i].append(torch.tensor(agent_obs["other_pos"], dtype=torch.float64))
            distance_budget_list[i].append(torch.tensor(agent_obs["distance_budget"], dtype=torch.float64))

    # Stack per-agent tensors across environments
    agent_data = []
    for i in range(n_agents):
        agent_data.append({
            "ego_pos": torch.stack(ego_pos_list[i], dim=0).to(device),          # (num_envs, 1, H, W)
            "other_pos": torch.stack(other_pos_list[i], dim=0).to(device),      # (num_envs, 1, H, W)
            "distance_budget": torch.stack(distance_budget_list[i], dim=0).to(device),  # (num_envs, 1)
        })

    return {
        "trash_map": trash_map.to(device),             # (num_envs, 1, H, W)
        "shared_scalars": shared_scalars.to(device),   # (num_envs, D)
        "agent_data": agent_data                       # List[N] of dicts
    }

## Code adapted from the 'ppo_atari.py' example in the following repository:
## https://github.com/vwxyzjn/ppo-implementation-details

def parse_args():
    parser = argparse.ArgumentParser('Train a PPO agent to solve the multiobjective cleaning problem.')
    # Environment specific arguments
    parser.add_argument('--map', type=str, default='malaga_port', choices=['malaga_port','alamillo_lake','ypacarai_map'], 
        help='The map to use.')
    parser.add_argument('--distance_budget', type=int, default=100, 
        help='The maximum distance of the agents.')
    parser.add_argument('--n_agents', type=int, default=4, 
        help='The number of agents to use.')
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    parser.add_argument('--miopic', type=bool, default=True, 
        help='If True the scenario is miopic.')
    parser.add_argument('--detection_length', type=int, default=2, 
        help='The influence radius of the agents.')
    parser.add_argument('--movement_length', type=int, default=1, 
        help='The movement length of the agents.')
    parser.add_argument('--reward_type', type=str, default='Distance Field', 
        help='The reward type to train the agent.')
    parser.add_argument('--convert_to_uint8', type=bool, default=False, 
        help='If convert the state to unit8 to store it (to save memory).')
    parser.add_argument('--benchmark', type=str, default='macro_plastic', choices=['shekel', 'algae_bloom','macro_plastic'], 
        help='The benchmark to use.')
    parser.add_argument('--dynamic', type=bool, default=True, 
        help='Simulate dynamic')
    parser.add_argument('--device', type=int, default=0, help='The device to use.', choices=[-1, 0, 1])
    
    # Path planner specific arguments
    parser.add_argument('--path-planner-model', type=str, default='vaeUnet', choices=['miopic', 'vaeUnet'], 
        help='The model to use.')

    # fmt: off
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.map}__{args.benchmark}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    import json

    # Save all command-line arguments to a JSON file
    with open(f"runs/{run_name}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    print("Run name:", run_name)
    print("✅ Arguments saved to args.json")
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device_str = 'cpu' if args.device == -1 else f'cuda:{args.device}'
    device = torch.device(device_str if (torch.cuda.is_available() and args.device != -1) else "cpu")
    print(f"\nUsing device: {device}\n")
    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    # )
    
    # Environment setup
    N = args.n_agents
    sc_map = np.genfromtxt(f"{data_path}/Environment/Maps/{args.map}.csv", delimiter=',')

    if args.map == 'malaga_port':
        initial_positions = np.array([[12, 7], [14, 5], [16, 3], [18, 1]])[:N, :]
    elif args.map == 'alamillo_lake':
        initial_positions = np.array([[68, 26], [64, 26], [60, 26], [56, 26]])[:N, :]
    elif args.map == 'ypacarai_map':
        initial_positions = np.asarray([[24, 21],[28,24],[27,19],[24,24]])


    env_kwargs = {
        "scenario_map": sc_map,
        "fleet_initial_positions": initial_positions,
        "distance_budget": args.distance_budget,
        "number_of_vehicles": N,  # Or use args.num_agents if defined
        "seed": args.seed,
        "miopic": args.miopic,
        "dynamic": args.dynamic,
        "detection_length": args.detection_length,
        "movement_length": args.movement_length,
        "max_collisions": 15,
        "reward_type": args.reward_type,
        "convert_to_uint8": args.convert_to_uint8,
        "ground_truth_type": args.benchmark,
        "obstacles": False,
        "frame_stacking": 1
    }

    expert_kwargs = {
    # "env": env, # environment will be passed in the thunk
    "device": device_str,
    "masked_actions": True,
    "consensus": True,
    }


    env_fns = [
        make_env(
            env_fn=MultiAgentPatrolling,
            expert_nu_fn=Expert_nu,
            env_kwargs=env_kwargs,
            expert_kwargs=expert_kwargs,
            seed=args.seed + i,
            device_int=args.device,
            args=args,
            idx=i,
            capture_video=(i == 0),
            run_name=run_name
        ) for i in range(args.num_envs)
    ]

    envs = gym.vector.SyncVectorEnv(env_fns)
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only box action discrete is supported"
    # For shape extraction only; agent will work with envs later
    env_for_shapes = env_fns[0]()
    agent = PreferencePPOAgent(env_for_shapes).to(device)
    print(f"Agent structure:\n{agent}\n")
    # agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    ### MODIFIED PPO TRAINING LOOP FOR STRUCTURED OBSERVATIONS ###
    H, W = sc_map.shape
    # === Storage for structured observation ===

    obs_buf = {
    "trash_map": torch.zeros((args.num_steps, args.num_envs) + env_for_shapes.single_observation_space["trash_map"].shape, dtype=torch.float32, device=device),
    "shared_scalars": torch.zeros((args.num_steps, args.num_envs) + env_for_shapes.single_observation_space["shared_scalars"].shape, dtype=torch.float32, device=device),
    "agent_data": {
        "ego_pos": torch.zeros((args.num_steps, args.num_envs, N) + env_for_shapes.single_observation_space["agent_data"][0]["ego_pos"].shape, dtype=torch.float32, device=device),
        "other_pos": torch.zeros((args.num_steps, args.num_envs, N) + env_for_shapes.single_observation_space["agent_data"][0]["other_pos"].shape, dtype=torch.float32, device=device),
        "distance_budget": torch.zeros((args.num_steps, args.num_envs, N) + env_for_shapes.single_observation_space["agent_data"][0]["distance_budget"].shape, dtype=torch.float32, device=device),
    }
}
    actions = torch.zeros((args.num_steps, args.num_envs, N), dtype=torch.float32, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    # === Init ===
    global_step = 0
    start_time = time.time()
    next_obs = convert_obs(envs.reset()[0], device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    for update in trange(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        total_reward = 0.0
        for step in range(args.num_steps):
            global_step += args.num_envs

            # Store observations
            #########################
            obs_buf["trash_map"][step] = next_obs["trash_map"]
            obs_buf["shared_scalars"][step] = next_obs["shared_scalars"]

            for i in range(N):
                obs_buf["agent_data"]["ego_pos"][step, :, i] = next_obs["agent_data"][i]["ego_pos"]
                obs_buf["agent_data"]["other_pos"][step, :, i] = next_obs["agent_data"][i]["other_pos"]
                obs_buf["agent_data"]["distance_budget"][step, :, i] = next_obs["agent_data"][i]["distance_budget"]
            #########################
            
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device)
            #############################
            next_obs = convert_obs(next_obs, device)  # You must define this
            next_done = torch.tensor(done, dtype=torch.float32, device=device)
            #############################
            # next_done = torch.tensor(done, dtype=torch.float32).to(device)

            # next_obs = collate_obs(next_obs_raw, N)
            total_reward += np.mean(reward)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # Reshape for batch processing
        ###############################
        # B = num_steps * num_envs
        B = args.num_steps * args.num_envs
        D = obs_buf["shared_scalars"].shape[-1]  # Dimension of shared scalars
        b_obs = {
            "trash_map": obs_buf["trash_map"].reshape(B, 1, H, W),
            "shared_scalars": obs_buf["shared_scalars"].reshape(B, D),
            "agent_data": []
        }
        for i in range(N):
            b_obs["agent_data"].append({
                "ego_pos": obs_buf["agent_data"]["ego_pos"][:, :, i].reshape(B, 1, H, W),
                "other_pos": obs_buf["agent_data"]["other_pos"][:, :, i].reshape(B, 1, H, W),
                "distance_budget": obs_buf["agent_data"]["distance_budget"][:, :, i].reshape(B, 1),
            })
        ###############################
        
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO updates
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # mb_obs = {k: v[mb_inds] for k, v in b_obs.items()}
                ##################################
                # Convert observations to batched tensors
                mb_obs = {
                    "trash_map": b_obs["trash_map"][mb_inds],
                    "shared_scalars": b_obs["shared_scalars"][mb_inds],
                    "agent_data": []
                }
                for i in range(N):
                    mb_obs["agent_data"].append({
                        "ego_pos": b_obs["agent_data"][i]["ego_pos"][mb_inds],
                        "other_pos": b_obs["agent_data"][i]["other_pos"][mb_inds],
                        "distance_budget": b_obs["agent_data"][i]["distance_budget"][mb_inds],
                    })
                ##################################

                # Extract minibatch data
                mb_actions = b_actions[mb_inds]
                mb_logprobs = b_logprobs[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_values = b_values[mb_inds]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("\nSteps", int(global_step), "SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        # record rewards for plotting purposes
        writer.add_scalar("charts/mean_reward", total_reward, global_step) # Mean reward across all environments

    envs.close()
    writer.close()
    # Save the trained PPO agent
    torch.save(agent.state_dict(), f"runs/{run_name}/ppo_agent.pth")
    print("✅ Agent saved to", f"ppo_agent.pth")