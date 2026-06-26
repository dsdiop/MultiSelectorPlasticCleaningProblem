import sys
import os
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)
from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.DuelingDQNAgent_alternative import MultiAgentDuelingDQNAgent
import numpy as np
import torch
import optuna
from optuna.trial import TrialState
import joblib

import argparse


parser = argparse.ArgumentParser(description='Train a multiagent DQN agent to solve the multiobjective cleaning problem.')
parser.add_argument('--map', type=str, default='malaga_port', choices=['malaga_port','alamillo_lake','ypacarai_map'], help='The map to use.')
parser.add_argument('--distance_budget', type=int, default=200, help='The maximum distance of the agents.')
parser.add_argument('--n_agents', type=int, default=4, help='The number of agents to use.')
parser.add_argument('--seed', type=int, default=0, help='The seed to use.')
parser.add_argument('--miopic', type=bool, default=True, help='If True the scenario is miopic.')
parser.add_argument('--detection_length', type=int, default=2, help='The influence radius of the agents.')
parser.add_argument('--movement_length', type=int, default=1, help='The movement length of the agents.')
parser.add_argument('--reward_type', type=str, default='Distance Field', help='The reward type to train the agent.')
parser.add_argument('--convert_to_uint8', type=bool, default=False, help='If convert the state to unit8 to store it (to save memory).')
parser.add_argument('--benchmark', type=str, default='macro_plastic', choices=['shekel', 'algae_bloom','macro_plastic'], help='The benchmark to use.')

parser.add_argument('--model', type=str, default='vaeUnet', choices=['miopic', 'vaeUnet'], help='The model to use.')
parser.add_argument('--device', type=int, default=0, help='The device to use.', choices=[-1, 0, 1])
parser.add_argument('--dynamic', type=bool, default=True, help='Simulate dynamic')

# Compose a name for the experiment
args = parser.parse_args()

N = args.n_agents

sc_map = np.genfromtxt(f"{data_path}/Environment/Maps/{args.map}.csv", delimiter=',')

if args.map == 'malaga_port':
    initial_positions = np.array([[12, 7], [14, 5], [16, 3], [18, 1]])[:N, :]
elif args.map == 'alamillo_lake':
    initial_positions = np.array([[68, 26], [64, 26], [60, 26], [56, 26]])[:N, :]
elif args.map == 'ypacarai_map':
    initial_positions = np.asarray([[24, 21],[28,24],[27,19],[24,24]])


device = 'cpu' if args.device == -1 else f'cuda:{args.device}'

nettype = '0'

arch = 'v1'
policy_name = f'Experimento_clean27_{args.map}_{args.benchmark}_random_nus_nsteps5'
# logdir=f'./Learning/runs/Vehicles_{N}/SecondPaper/Nu_w_Optuna/{args.map}_{args.benchmark}'
logdir=f'./Learning/runs/Vehicles_{N}/SecondPaper'
if 'malaga' in args.map:
    logdir_optimized_policy = f'./Learning/runs/Vehicles_{N}/SecondPaper/Optuna/malaga_study/DRL_OPTUNA_TRIAL_0'
elif 'alamillo' in args.map:
    logdir_optimized_policy = f'./Learning/runs/Vehicles_{N}/SecondPaper/Optuna/alamillo_study/DRL_OPTUNA_TRIAL_2'
# split args.map to get the name of the map


if __name__ == "__main__":

    # Create a directory for the study
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    # Load the pkl of the study
    study = joblib.load(f"{logdir_optimized_policy}/DRL_hyperparam_study.pkl")
    
    # Load the best parameters
    if 'malaga' in args.map:
        best_params = study
    elif 'alamillo' in args.map:
        best_params = study.best_params
        
    # nu_steps = np.linspace(1,0,num=11)
    nu_steps = [1]
    eps = 60000
    for nu_step in nu_steps:
        #policy_name = "RandomNu_w_Optuna_" + str(nu_step)
        # Train the agent with the best parameters
        env = MultiAgentPatrolling(scenario_map=sc_map, 
                                    fleet_initial_positions=initial_positions,
                                    distance_budget=args.distance_budget,
                                    number_of_vehicles=N,
                                    seed=args.seed,
                                    miopic=args.miopic,
                                    dynamic=args.dynamic,
                                    detection_length=args.detection_length,
                                    movement_length=args.movement_length,
                                    max_collisions=15,
                                    reward_type=args.reward_type,
                                    convert_to_uint8=args.convert_to_uint8,
                                    ground_truth_type=args.benchmark,
                                    obstacles=False,

                                    frame_stacking=1,
                                    state_index_stacking=(1, 2, 3)
                                    )
        multiagent = MultiAgentDuelingDQNAgent(env=env,
                                            memory_size=int(1E6),
                                            batch_size=128,#64
                                            target_update=eps//5,
                                            soft_update=True,
                                            tau=0.001,
                                            epsilon_values=[1.0, 0.05],
                                            epsilon_interval=[0.0, 0.5],
                                            learning_starts=100, # 100
                                            gamma=0.99,
                                            alpha= 0.2,
                                            beta = 0.4,
                                            n_steps=5,
                                            lr=1e-4,
                                            number_of_features=1024,
                                            noisy=False,
                                            nettype=nettype,
                                            archtype=arch,
                                            device=device,
                                            weighted=False,
                                            train_every=7,
                                            save_every=1000,
                                            distributional=False,
                                            logdir=logdir+'/'+policy_name,
                                            prewarmed_memory=None,
                                            use_nu=True,
                                            nu_intervals=[[0., 1], [nu_step, 1], [nu_step, 0.], [1., 0.]],
                                            concatenatedDQN = False,
                                            eval_episodes=50,
                                            masked_actions= True,
                                            consensus = True,
                                            eval_every=500,
                                            weighting_method=None,
                                            weight_methods_parameters=dict()
                                            )
        if "1.0" not in policy_name and False:
            multiagent.load_model(initial_network + '/Final_Policy.pth')
            eps = 5000
        multiagent.train(episodes=eps)
        initial_network = multiagent.logdir
        torch.cuda.empty_cache()
            
