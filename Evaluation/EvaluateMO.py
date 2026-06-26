
import sys
import os
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)
from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from Environment.GroundTruthsModels.AlgaeBloomGroundTruth import algae_colormap,background_colormap
import numpy as np
import matplotlib.pyplot as plt
import torch
from Utils.metrics_wrapper import MetricsDataCreator
import json
from tqdm import trange
import pandas as pd 
import seaborn as sns 
#from Evaluation.Utils.path_plotter import plot_trajectory
#from Algorithm.RainbowDQL.Agent.pruebas import plot_visits, plot_state
from Evaluation.Utils.EvaluationUtils import run_path_planners_evaluation

import argparse
parser = argparse.ArgumentParser(description='Train a multiagent DQN agent to solve the multiobjective cleaning problem.')
parser.add_argument('--map', type=str, default='malaga_port', choices=['malaga_port','alamillo_lake','ypacarai_map'], help='The map to use.')
parser.add_argument('--distance_budget', type=int, default=100, help='The maximum distance of the agents.')
parser.add_argument('--n_agents', type=int, default=4, help='The number of agents to use.')
parser.add_argument('--seed', type=int, default=30, help='The seed to use.')
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
args.map = 'alamillo_lake'
args.map = 'malaga_port'
N = args.n_agents
sc_map = np.genfromtxt(f"{data_path}/Environment/Maps/{args.map}.csv", delimiter=',')
if args.map == 'malaga_port':
    initial_positions = np.array([[12, 7], [14, 5], [16, 3], [18, 1]])[:N, :]
elif args.map == 'alamillo_lake':
    initial_positions = np.array([[68, 26], [64, 26], [60, 26], [56, 26]])[:N, :]
elif args.map == 'ypacarai_map':
    initial_positions = np.asarray([[24, 21],[28,24],[27,19],[24,24]])


device = 'cpu' if args.device == -1 else f'cuda:{args.device}'

imm = []
def EvaluateMultiagent(number_of_agents: int,
                       sc_map,
                       visitable_locations,
                       initial_positions,
                       num_of_eval_episodes: int,
                       policy_path: str,
                       policy_type: str,
                       seed: int,
                       policy_name: str='DDQN',
                       metrics_directory: str= './',
                       nu_interval = None,
                       agent_config=None,
                       environment_config=None,
                       render=False
                       ):

    N = number_of_agents
    random_index = np.random.choice(np.arange(0,len(visitable_locations)), N, replace=False)
    #imm = []
    
    if agent_config is None:
        agent_config = json.load(open(f'{policy_path}experiment_config.json', 'rb'))
    
    if environment_config is None:
        environment_config = json.load(open(f'{policy_path}environment_config.json', 'rb'))

    ## Some Sanity checks
    try:
        environment_config['movement_length']=environment_config['movement_length']
    except:
        environment_config['movement_length'] = 2
        
    try:
        environment_config['frame_stacking']=environment_config['frame_stacking']
        environment_config['state_index_stacking']=environment_config['state_index_stacking']
    except:
        if 'fstack' in policy_path:
            environment_config['frame_stacking'] = 2
        else:
            environment_config['frame_stacking'] = 1
        
        environment_config['state_index_stacking']=(2, 3, 4)
    try:
        environment_config['trail_length']=environment_config['trail_length']
    except:
        environment_config['trail_length'] = 1    
    env = MultiAgentPatrolling(scenario_map=sc_map,
                                fleet_initial_positions=environment_config['fleet_initial_positions'],
                                distance_budget=environment_config['distance_budget'],
                                number_of_vehicles=environment_config['number_of_agents'],
                                seed=seed,
                                miopic=environment_config['miopic'],
                                detection_length=environment_config['detection_length'],
                                movement_length=environment_config['movement_length'],
                                max_collisions=environment_config['max_number_of_colissions'],
                                #networked_agents=False,
                                ground_truth_type=environment_config['ground_truth'],
                                obstacles=False,
                                frame_stacking=environment_config['frame_stacking'],
                                state_index_stacking=environment_config['state_index_stacking'],
				                reward_type=environment_config['reward_type'],
     			                trail_length = environment_config['trail_length']
                                #reward_weights=environment_config['reward_weights']
                                )
    env.convert_to_uint8 = False
    ## Some Sanity checks
    try:
        agent_config['nettype']=agent_config['nettype']
    except:
        agent_config['nettype'] = '0'
        
    try:
        agent_config['archtype']=agent_config['archtype']
    except:
        if 'v2' in policy_path:
            agent_config['archtype'] = 'v2'
        else:
            agent_config['archtype'] = 'v1'
    try:
        agent_config['concatenatedDQN'] = agent_config['concatenatedDQN']  
    except:
        if '_2DQN' in policy_name:
            agent_config['concatenatedDQN'] = True
        elif 'one_phase' in policy_name:
            agent_config['concatenatedDQN'] = True
        else:
            agent_config['concatenatedDQN'] = False

    multiagent = MultiAgentDuelingDQNAgent(env=env,
                                        memory_size=int(1),
                                        batch_size=64,
                                        target_update=1000,
                                        soft_update=True,
                                        tau=0.001,
                                        epsilon_values=[0, 0],
                                        epsilon_interval=[0.0, 0.5],
                                        learning_starts=100, # 100
                                        gamma=0.99,
                                        lr=1e-4,
                                        number_of_features=1024 if 'n_of_features_512' not in policy_path else 512,
                                        noisy=False,
                                        nettype=agent_config['nettype'],
                                        archtype=agent_config['archtype'],
                                        device='cuda',
                                        train_every=15,
                                        save_every=1000,
                                        distributional=False,
                                        logdir=f'Learning/runs/Vehicles_{N}/{policy_path}',
                                        use_nu=agent_config['use_nu'],
                                        nu_intervals= agent_config['nu_intervals'] if nu_interval is None else nu_interval,
                                        concatenatedDQN = agent_config['concatenatedDQN'],
                                        eval_episodes=num_of_eval_episodes,
                                        eval_every=1000)

    multiagent.load_model(policy_path+policy_type)
    metrics = MetricsDataCreator(metrics_names=['Policy Name',
                                                'Accumulated Reward Cleaning',
                                                'Accumulated Reward Exploration',
                                                'Total Length',
                                                'nu',
                                                'Percentage of Trash Cleaned',
                                                'Percentage Visited'],
                                algorithm_name='DRL',
                                experiment_name='DRLResults',
                                directory=metrics_directory)
    
    paths = MetricsDataCreator(metrics_names=['vehicle', 'x', 'y','nu','trash_collected','done'],
                            algorithm_name='DRL',
                            experiment_name='DRL_paths',
                            directory=metrics_directory)
    
    path_directory = os.path.dirname(metrics_directory)
    if os.path.exists(path_directory):
        if os.path.exists(metrics_directory + 'DRLResults' + '.csv'):
            metrics.load_df(metrics_directory + 'DRLResults' + '.csv')
        if os.path.exists(metrics_directory + 'DRL_paths' + '.csv'):
            paths.load_df(metrics_directory + 'DRL_paths' + '.csv')
    else:
        # create the path
        os.makedirs(path_directory)
        
    

        
    """ Evaluate the agent on the environment for a given number of episodes with a deterministic policy """

    multiagent.dqn.eval()
    max_movements = env.distance_budget
    multiagent.epsilon = 0
    
    save_txt= None
    
    for run in trange(num_of_eval_episodes):

        # Reset the environment #
        state, _ = env.reset()

        if render:# and run==0:
            fig = env.render()
        done = {agent_id: False for agent_id in range(env.number_of_agents)}

        total_reward = 0
        total_reward_cleaning = 0
        list_total_reward_cleaning = []
        total_reward_exploration = 0
        list_total_reward_exploration = []
        total_length = 0
        instantaneous_percentage_of_trash_cleaned = 0
        percentage_of_trash_cleaned = 0
        percentage_visited = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)

        # Exploration reward components
        total_visit_reward_exploration = 0  
        list_total_visit_reward_exploration = []
        total_innactivity_penalty_exploration = 0
        list_total_innactivity_penalty_exploration = []
        total_redundancy_penalty_exploration = 0
        list_total_redundancy_penalty_exploration = []
        # Cleaning reward components
        total_trash_collecting_reward_cleaning = 0
        list_total_trash_collecting_reward_cleaning = []
        total_distance_reward_cleaning = 0
        list_total_distance_reward_cleaning = []
        total_model_update_reward_cleaning = 0
        list_total_model_update_reward_cleaning = []
        total_time_penalty_cleaning = 0
        list_total_time_penalty_cleaning = []
        
        
        if multiagent.use_nu:
            multiagent.nu = multiagent.anneal_nu(p= 0,
                                    p1=multiagent.nu_intervals[0],
                                    p2=multiagent.nu_intervals[1],
                                    p3=multiagent.nu_intervals[2],
                                    p4=multiagent.nu_intervals[3])
            nu_ = multiagent.nu
        else:
            nu_ = None
        metrics_list = [policy_name, total_reward_cleaning,
                        total_reward_exploration,
                        total_length, nu_,
                        percentage_of_trash_cleaned,
                        percentage_visited]
        # Initial register #
        metrics.register_step(run_num=run, step=total_length, metrics=metrics_list)
        for veh_id, veh in enumerate(env.fleet.vehicles):
            paths.register_step(run_num=run, step=total_length, metrics=[veh_id, veh.position[0], veh.position[1], nu_, env.n_trash_cleaned[veh_id], done[veh_id]])

        while not all(done.values()):

            total_length += 1
            if multiagent.use_nu:
                distance = np.min([np.max(env.fleet.get_distances()), max_movements])
                multiagent.nu = multiagent.anneal_nu(p= distance / max_movements,
                                        p1=multiagent.nu_intervals[0],
                                        p2=multiagent.nu_intervals[1],
                                        p3=multiagent.nu_intervals[2],
                                        p4=multiagent.nu_intervals[3])
                if render:
                    print('nu: ', multiagent.nu)
                nu_ = multiagent.nu
            else:
                nu_ = None
            # Select the action using the current policy
            # Select the action using the current policy
            state_float32 = {i:None for i in state.keys()}
            if env.convert_to_uint8:
                for agent_id in state.keys():
                    state_float32[agent_id] = (state[agent_id] / 255.0).astype(np.float32)
            else:
                state_float32 = state
     
            if not  multiagent.masked_actions:
                actions = multiagent.select_action(state_float32)
            else:
                actions = multiagent.select_masked_action(states=state_float32, positions=env.fleet.get_positions())
         

            actions = {agent_id: action for agent_id, action in actions.items() if not done[agent_id]}
            #print(env.fleet.get_positions())
            # Process the agent step #
            next_state, reward, done = multiagent.step(actions)
            
            #print("Number of steps where the agent has cleaned")
            # print("Number of trash collected: ", env.total_n_trash_cleaned)
            # print('Number of percentage of trash collected: ', env.percentage_of_trash_cleaned)
            # print('Number of collisions: ', env.fleet.fleet_collisions)
 
            if render:# and run==0:
                fig.texts.clear()
                if multiagent.use_nu:
                    nu = multiagent.nu  # 1 is Exploration, 0 is Cleaning, otherwise is Transition
                    if nu == 1:
                        phase_title = "Exploration"
                        title_color = "blue"
                    elif nu == 0:
                        phase_title = "Cleaning"
                        title_color = "red"
                    else:
                        phase_title = "Transition"
                        title_color = "purple"

                    bbox_props_exp = dict(boxstyle="round,pad=0.3", facecolor="blue", edgecolor="black", alpha=.1)
                    bbox_props_cln = dict(boxstyle="round,pad=0.3", facecolor="red", edgecolor="black", alpha=.1)
                    bbox_props_title = dict(boxstyle="round,pad=0.3", facecolor=title_color, edgecolor="white", alpha=.03)
                    
                    fig.text(0.7, 0.025, f'Percentage of map visited: {percentage_visited*100:.2f}%', ha='center', fontsize=16, bbox=bbox_props_exp)
                    #fig.suptitle(f"{phase_title} Phase", fontsize=40, color=title_color,y=0.99)
                    fig.text(0.5, 0.91, f"{phase_title} Phase", ha='center', fontsize=32, color =title_color, bbox=bbox_props_title)
                    fig.text(0.3, 0.025, f'Percentage of trash cleaned: {percentage_of_trash_cleaned*100:.2f}%', ha='center', fontsize=16, bbox=bbox_props_cln)
                
                
                fig = env.render()
            state = next_state
            percentage_visited = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)
            rewards = np.asarray(list(reward.values()))

            total_reward_exploration += np.sum(rewards[:,0])
            list_total_reward_exploration.append(total_reward_exploration)
            total_reward_cleaning += np.sum(rewards[:,1])
            list_total_reward_cleaning.append(total_reward_cleaning)
            total_reward = total_reward_exploration + total_reward_cleaning
            # Exploration reward components
            total_visit_reward_exploration += np.nansum(env.visit_reward_exploration)
            list_total_visit_reward_exploration.append(total_visit_reward_exploration)
            total_innactivity_penalty_exploration -= np.nansum(env.innactivity_penalty_exploration)
            list_total_innactivity_penalty_exploration.append(total_innactivity_penalty_exploration)
            total_redundancy_penalty_exploration -= np.nansum(env.redundancy_penalty_exploration)
            list_total_redundancy_penalty_exploration.append(total_redundancy_penalty_exploration)
            # Cleaning reward components
            total_trash_collecting_reward_cleaning += np.nansum(env.trash_collecting_reward_cleaning)
            list_total_trash_collecting_reward_cleaning.append(total_trash_collecting_reward_cleaning)
            total_distance_reward_cleaning += np.nansum(env.distance_reward_cleaning)
            list_total_distance_reward_cleaning.append(total_distance_reward_cleaning)
            total_model_update_reward_cleaning += np.nansum(env.model_update_reward_cleaning)
            list_total_model_update_reward_cleaning.append(total_model_update_reward_cleaning)
            total_time_penalty_cleaning -= np.nansum(env.time_penalty_cleaning)
            list_total_time_penalty_cleaning.append(total_time_penalty_cleaning)
            
            percentage_of_trash_cleaned = env.percentage_of_trash_cleaned
            #imm.append(instantaneous_global_idleness)
            metrics_list = [policy_name, total_reward_cleaning,
                        total_reward_exploration,
                        total_length, nu_,
                        percentage_of_trash_cleaned,
                        percentage_visited]
            metrics.register_step(run_num=run, step=total_length, metrics=metrics_list)
            for veh_id, veh in enumerate(env.fleet.vehicles):
                paths.register_step(run_num=run, step=total_length, metrics=[veh_id, veh.position[0], veh.position[1], nu_, env.n_trash_cleaned[veh_id], done[veh_id]])
            # save an image of env.gt.read() and env.known_information for the start and end of the cleaning phase
            if ((nu_ == 0 and save_txt is None) or all(done.values())) and False:
                if save_txt is None:
                    tt = 'initial'
                else:
                    tt = 'final'
                save_txt = True
                path_to_save = os.path.join(path_directory,'gt_model')
                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)
                np.savetxt(f'{path_to_save}/gt_cleaning_phase_{policy_name}_{tt}.csv', env.gt.read(), delimiter=',')
                np.savetxt(f'{path_to_save}/model_cleaning_phase_{policy_name}_{tt}.csv', env.known_information, delimiter=',')
    if not render:
        metrics.register_experiment()
        paths.register_experiment()
    else:
        plt.close()
    mean_reward_inf = total_reward_cleaning / num_of_eval_episodes
    mean_reward_exp = total_reward_exploration / num_of_eval_episodes
    mean_reward = total_reward / num_of_eval_episodes
    mean_length = total_length / num_of_eval_episodes
    
    return mean_reward_inf, mean_reward_exp, mean_reward, mean_length




if __name__ == '__main__':
    if False:
        num_of_eval_episodes = 200
        #sc_map = np.genfromtxt('Environment/Maps/malaga_port.csv', delimiter=',')

        # N = 4
        # initial_positions = np.array([[12, 7], [14, 5], [16, 3], [18, 1]])[:N, :]
        visitable_locations = np.vstack(np.where(sc_map != 0)).T
        policy_types = ['Final_Policy','BestCleaningPolicy','BestPolicy_perc_map_visited']#,'BestPolicy_reward_cleaning','BestPolicy_reward_exploration']
        # policy_types = ['BestPolicy_perc_map_visited']
        nu_intervals ={'1':[[0., 1], [0.10, 1], [0.90, 1.], [1., 1.]],
                       '2':[[0., 1], [0.80, 1], [0.90, 0.], [1., 0.]],
                       '3':[[0., 1], [0.70, 1], [0.80, 0.], [1., 0.]],
                       '4':[[0., 1], [0.60, 1], [0.70, 0.], [1., 0.]],
                       '5':[[0., 1], [0.50, 1], [0.60, 0.], [1., 0.]],
                       '6':[[0., 1], [0.40, 1], [0.50, 0.], [1., 0.]],
                       '7':[[0., 1], [0.30, 1], [0.40, 0.], [1., 0.]],
                       '8':[[0., 1], [0.20, 1], [0.30, 0.], [1., 0.]],
                       '9':[[0., 1], [0.10, 1], [0.20, 0.], [1., 0.]],
                       '10':[[0., 0], [0.10, 0], [0.20, 0.], [1., 0.]],
                       'original':[[0., 1], [0.30, 1], [0.60, 0.], [1., 0.]]}
        """for n in nu_intervals.keys():
            n='6'"""
        seeds = [17,43,45,3,31]
        # data_path1 = f"{data_path}/../Learning/runs/Vehicles_4/SecondPaper/Nu_w_Optuna/{args.map}_{args.benchmark}"
        # policy_names = [folder for folder in os.listdir(data_path1) if os.path.isdir(os.path.join(data_path1, folder))]
        data_path1 = f"{data_path}/../Learning/runs/Vehicles_4/SecondPaper/Experimento_clean27_{args.map}_{args.benchmark}_random_nus_nsteps5"
        data_path1 = f'C:\\Users\\dames\\Downloads\\Experimento_clean26_{args.map}_{args.benchmark}_random_nus_nsteps5'
        # num_veh = 20
        # data_path1 = f"C:\\Users\\dames\\Downloads\\Vehicles_{num_veh}/SecondPaper/Experimento_clean28_{args.map}_{args.benchmark}__20veh_random_nus_nsteps5"
        data_path1 = f"C:\\Users\\dames\\Downloads\\Vehicles_{N}/SecondPaper/Experimento_clean28_{args.map}_{args.benchmark}__{N}veh_random_nus_nsteps5"
        
        # policy_names = [folder for folder in os.listdir(data_path1) if os.path.isdir(os.path.join(data_path1, folder))]
        policy_names = ['a']
        pf_policies_alamillo =['BestPolicy_perc_map_visited_0.9', 'Final_Policy_0.9', 
                                'Final_Policy_0.8', 'Final_Policy_0.7',
                                'BestCleaningPolicy_0.6', 'BestCleaningPolicy_0.5',
                                'BestPolicy_perc_map_visited_0.4', 'BestCleaningPolicy_0.3','BestCleaningPolicy_0.1']
        pf_policies_malaga = ['Final_Policy_1.0', 'BestCleaningPolicy_0.9',
                              'Final_Policy_0.8', 'BestPolicy_perc_map_visited_0.8',
                              'Final_Policy_0.7', 'BestCleaningPolicy_0.6',
                              'Final_Policy_0.5', 'BestCleaningPolicy_0.5',
                              'Final_Policy_0.3', 'BestCleaningPolicy_0.3',
                              'Final_Policy_0.2']
        
        if args.map == 'malaga_port':
            pf_policies = pf_policies_malaga
        elif args.map == 'alamillo_lake':
            pf_policies = pf_policies_alamillo
        types = ['old_reward', 'new_reward']
        nu_steps=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        for type_ in types:
            data_path1 = f"C:\\Users\\dames\\OneDrive\\Escritorio\\ADS\\Tesis\\thirdpaper\\ThirdPaper\\Experimento_clean28_malaga_port_macro_plastic_random_nus_nsteps5_distbudget100_{type_}"
            for nu_step in nu_steps:
                for i,policy_name in enumerate(policy_names):
                    # nu_step = round(float(policy_name.split('_')[-1]),1)
                    # if nu_step != 0.6:
                    #     pass
                    # policy_path = f'{data_path1}/{policy_name}/'
                    policy_path = f'{data_path1}/'
                    seed = 30#30#43#45#3#31# 22 #46 en el video del alamillo
                    pol_type = []
                    for policy_type in policy_types:
                        if 'Final_Policy' not in policy_type:
                            policy_type_1 = f'{policy_type}_{nu_step}'
                        else:
                            policy_type_1 = policy_type
                        # if f'{policy_type}_{nu_step}' not in pf_policies:
                        #     continue
                        #policy_name = f"same_map_{policy_name}_{policy_type}"
                        print(policy_path,policy_type,nu_step)
                        EvaluateMultiagent(number_of_agents=N,
                                        sc_map=sc_map,
                                        visitable_locations=visitable_locations,
                                        initial_positions=initial_positions,
                                        num_of_eval_episodes=num_of_eval_episodes,
                                        policy_path=policy_path,
                                        policy_type=policy_type_1+'.pth',
                                        seed=seed,
                                        policy_name=f'{policy_type}_{nu_step}',
                                        metrics_directory= f'{data_path}/Evaluation/Results_paper/Results/Results_budget100/Results_seed_{seed}_random_nus_{args.map}_{type_}/{policy_type}_{nu_step}',
                                        nu_interval =[[0., 1], [nu_step, 1], [nu_step, 0.], [1., 0.]],#None,#nu_intervals[nu_interval],
                                        render = False
                                        )


    if True:
        import glob
        map_ = 'alamillo_lake'
        map_ = 'malaga_port'
        print(map_)
        # csv_list = glob.glob(f"./Evaluation/Results_chapter/Results_seed_30_nu_steps_dist_field_{map_}/*DRLResults.csv")
        csv_list = glob.glob(f"./Evaluation/Results_paper/Results/Results_seed_30_nu_steps_dist_field_{map_}_30keps/*DRLResults.csv")
        csv_list = glob.glob(f"./Evaluation/Results_paper/Results/Results_seed_30_random_nus_{map_}/*DRLResults.csv")
        
        # csv_list = glob.glob(f"./Evaluation/Results/Results/Results_seed_30_nu_steps_dist_field_{map_}_30keps/*DRLResults.csv")
        # csv_list = glob.glob(f"./Evaluation/Results/Results/Results_seed_30_random_nus_{map_}/*DRLResults.csv")
        
        # csv_list = glob.glob(f'./Evaluation/Results_paper/Results/Results_2_veh/Results_seed_30_random_nus_{map_}_2_veh/*DRLResults.csv')
        # csv_list = glob.glob(f'./Evaluation/Results_paper/Results/Results_10_veh/Results_seed_30_random_nus_{map_}_10_veh/*DRLResults.csv')
        # csv_list = glob.glob(f'./Evaluation/Results_paper/Results/Results_20_veh/Results_seed_30_random_nus_{map_}_20_veh/*DRLResults.csv')
        csv_list = glob.glob(f'./Evaluation/Results_paper/Results/Results_budget100/Results_seed_30_random_nus_{map_}_old_reward/*DRLResults.csv')
        # csv_list = glob.glob(f'./Evaluation/Results_paper/Results/Results_budget100/Results_seed_30_random_nus_{map_}_new_reward/*DRLResults.csv')
        Finalpolicies = {csv_list[i].split('\\')[-1].split('DRLResults.csv')[0]:pd.read_csv(csv_list[i]) for i in range(len(csv_list))}
        
        dictrename = {'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_1':'Only Exploration',
                    'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_2':'80-90',
                    'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_3':'70-80',
                    'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_4':'60-70',
                    'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_5':'50-60',
                    'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_6':'40-50',
                    'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_7':'30-40',
                    'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_8':'20-30',
                    'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_9':'10-20',
                    'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_10':'Only Intensification',}

        #values_to_evaluate =['Accumulated Reward Cleaning', 'Accumulated Reward Exploration','Total Length', 'Percentage of Trash Cleaned','Percentage Visited']
        values_to_evaluate =['Percentage of Trash Cleaned',
                            'Percentage Visited']
        objective_evaluations = {key:[[Finalpolicies[key].groupby(['Run'])[values_to_evaluate[i]].tail(1).mean(),
                                       Finalpolicies[key].groupby(['Run'])[values_to_evaluate[i]].tail(1).std()] 
                    for i in range(len(values_to_evaluate))]
                    for key in Finalpolicies.keys()} # policy:[[mean objective 1, std objective 1], [mean objective 2, std objective 2]]
        # Remove BestPolicy_reward_exploration and BestPolicy_reward_cleaning from the dictionary
        objective_evaluations = {key: value for key, value in objective_evaluations.items() if 'BestPolicy_reward_exploration' not in key and 'BestPolicy_reward_cleaning' not in key}
        
        # Remove duplicated items and keep only one
        unique_objective_evaluations = {}
        for key, value in objective_evaluations.items():
            if value not in unique_objective_evaluations.values():
                unique_objective_evaluations[key] = value

        # Update the objective evaluations dictionary
        objective_evaluations = unique_objective_evaluations
        #grouped_data = Finalpolicy.groupby(['Policy Name', 'Run', 'Step'])[values_to_evaluate].mean().reset_index()
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        from pymoo.indicators.hv import HV
        from pymoo.indicators.igd import IGD
        from pymoo.indicators.spacing import SpacingIndicator
        from pymoo.indicators.distance_indicator import DistanceIndicator
        # Extract the two objective values into a numpy array
        objs = np.array([[-objective_evaluations[key][0][0], -objective_evaluations[key][1][0]] for key in objective_evaluations.keys()])
        stds = np.array([[objective_evaluations[key][0][1], objective_evaluations[key][1][1]] for key in objective_evaluations.keys()])
        policies = list(objective_evaluations.keys())

        # Perform non-dominated sorting
        nds = NonDominatedSorting()
        fronts = nds.do(objs)

        # Get the solutions in the first front (i.e., the Pareto front)
        pf = -objs[fronts[0]]
        pf_stds = stds[fronts[0]]
        pf_policies = [policies[i] for i in fronts[0]]
        
        sort_index = np.argsort(pf[:,0])
        pf = pf[sort_index]
        pf_stds = pf_stds[sort_index]
        pf_policies = [pf_policies[i] for i in sort_index]
        ci99 = 2.58*pf_stds/np.sqrt(200)
        coef_disper = pf_stds/pf
        # pLOt the Pareto front, the percentage of trash cleaned vs the percentage visited, 
        # with confidence intervals of 95%, the means are united by a line, 
        # and the confidence intervals are represented by uniting the upper limits of the confidence intervals with a line
        # and the lower limits of the confidence intervals with another line,
        # the space between the two lines is filled with a color that represents the policy
        # plt.figure(figsize=(10, 6))
        # # sort the pf by the percentage of trash cleaned
        # pf = np.array(sorted(pf, key=lambda x: x[0]))
        # plt.plot(pf[:, 0], pf[:, 1], 'k--', label='95% Confidence Interval')
        # plt.scatter(pf[:, 0], pf[:, 1], s=100)
        # # Plot the confidence intervals  by uniting the upper limits of the confidence intervals with a line
        # plt.plot(pf[:, 0], pf[:, 1] + 1.96*pf_stds[:, 1]/np.sqrt(200), 'r-', label='95% Confidence Interval')
        # plt.plot(pf[:, 0], pf[:, 1] - 1.96*pf_stds[:, 1]/np.sqrt(200), 'r-', label='95% Confidence Interval')
        # plt.plot(pf[:, 0] + 1.96*pf_stds[:, 0]/np.sqrt(200), pf[:, 1], 'b-', label='95% Confidence Interval')
        # plt.plot(pf[:, 0] - 1.96*pf_stds[:, 0]/np.sqrt(200), pf[:, 1], 'b-', label='95% Confidence Interval')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        
        # Calculate the hypervolume
        hv = HV(ref_point=np.array([0., 0.]))
        hypervolume = hv.do(100*objs[fronts[0]])
        print(f"Hypervolume: {hypervolume}")
        # Calculate the IGD
        igd = IGD(ref_point=np.array([100., 100.]), pf=100*pf)
        igd_value = igd.do(100*objs)
        print(f"IGD: {igd_value}")
        # Calculate the Spacing
        spacing = SpacingIndicator()
        spacing_value = spacing.do(100*objs[fronts[0]])
        print(f"Spacing: {spacing_value}")
        # Calculate the Average Distance
        # distance = DistanceIndicator(pf=100*pf, dist_func="euclidean",axis=0)
        # distance_value = distance.do(100*objs[fronts[0]])
        # print(f"Average Distance: {distance_value}")
        # Plot the Pareto front
        plt.figure(figsize=(15, 10))
        plt.rcParams['font.family'] = 'STIXGeneral'
        plt.rcParams['font.family'] = 'Times New Roman'
        nu_val_ant=None
        for i, (obj, std, policy) in enumerate(zip(pf, ci99, pf_policies)):
            obj = obj*100
            std = std*100
            #plt.errorbar(obj[0], obj[1], xerr=std[0], yerr=std[1], fmt='o', label=policy, capsize=5)
            # change policy name: FinalPolicy_reward_exploration_0.6 -> FP_{0.6} is a mathematical notation
            if 'Final_Policy' in policy:
                nu_val = policy.split('Final_Policy_')[-1]
                policy ='FP$\mathregular{_{'+nu_val+'}}$'
            elif 'BestPolicy_perc_map_visited_' in policy:
                nu_val = policy.split('BestPolicy_perc_map_visited_')[-1]
                policy = 'BestPMV$\mathregular{_{'+nu_val+'}}$'
            elif 'BestCleaningPolicy' in policy:
                nu_val = policy.split('BestCleaningPolicy_')[-1]
                policy = 'BestPTC$\mathregular{_{'+nu_val+'}}$'
            if nu_val_ant == nu_val or '0.6' in nu_val:
                va = 'bottom'
                si = -1
            else:
                va = 'top'
                si = 1
            nu_val_ant = nu_val
            # if nu_val == '0.7':
            #     va = 'bottom'
            #     si = -1
            # plot a shaded rectangle withing the confidence interval each rectangle with different color and type of shading
            #plt.fill_betweenx([obj[1]-std[1], obj[1]+std[1]], obj[0]-std[0], obj[0]+std[0], alpha=0.6, label=policy)
            
            plt.fill_between([obj[0]-std[0], obj[0]+std[0]], obj[1]-std[1], obj[1]+std[1], alpha=0.6, label=policy)
            plt.scatter(obj[0], obj[1], s=50, c='red', marker='x')
            plt.annotate(f'{nu_val}', (obj[0], obj[1]), textcoords="offset points", fontsize=30, xytext=(0,si*30), ha='center',va=va, arrowprops=dict(facecolor='blue', shrink=0.05), fontweight='bold')
            print(f"{policy}: cleaning {obj[0]} +- {std[0]}, exploration {obj[1]} +- {std[1]}")
        font_size = 40
        plt.xlabel('PTC (%)',fontsize=font_size, fontweight='bold')
        plt.ylabel('PMV (%)',fontsize=font_size, fontweight='bold')
        # save objective values to a csv file
        # create a DataFrame from the Pareto front
        # pf_df = pd.DataFrame(pf, columns=['PTC', 'PMV'])
        # pf_df['Policy'] = pf_policies
        # # pf_df['CI99'] = ci99
        # pf_df.to_csvMacro(f'pareto_front_{map_}.csv', index=False)
        # fix the axis limits
        if map_ == 'malaga_port':
            ax_limit_x = 30 #99.5 #45 #55 #80 #30 #40
            ax_limit_y = 75 #99.5 #70 #75#90 #40 #85
            title_ = 'Malaga Port'
        elif map_ == 'alamillo_lake':
            ax_limit_x = 75 #50 #75
            ax_limit_y = 75 #90 #75
            title_ = 'Alamillo Lake'
        
        plt.title(title_, fontsize=font_size, fontweight='bold')
        # import matplotlib.font_manager as font_manager 
        # font = font_manager.FontProperties(weight='black',
        #                            style='normal', size=font_size)
        plt.legend(loc='lower left', bbox_to_anchor=(1.0, 0.00), fontsize=35)
        
            
        plt.xlim(ax_limit_x, 100)
        plt.ylim(ax_limit_y, 100)
        # make the aspect ratio equal
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        # make the grid lines every 0.1
        fontsize_ticks =40
        plt.xticks(np.arange(ax_limit_x, 105, 5), fontsize=fontsize_ticks, fontweight='bold')
        plt.yticks(np.arange(ax_limit_y, 105, 5), fontsize=fontsize_ticks, fontweight='bold')
        # plt.xticks(fontsize=fontsize_ticks, fontweight='bold')
        # plt.yticks(fontsize=fontsize_ticks, fontweight='bold')
        # make the space between grid lines more widely distributed
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.tight_layout()
        
        # plt.savefig(f'./new_images_/pareto_front_{map_}_rand.png')
        plt.show()
        ####################################################
        import itertools
        import pandas as pd
        import numpy as np
        from scipy.stats import wilcoxon

        def wilcoxon_matrix(Finalpolicies, policies, metric="Percentage of Trash Cleaned", alpha=0.05):
            """
            Build a pairwise Wilcoxon signed-rank test table across policies.
            Also prints which comparisons are significant.
            
            Finalpolicies : dict of {policy_name: DataFrame}
            policies      : list of policy names to compare
            metric        : evaluation metric (string)
            alpha         : significance threshold (default 0.05)
            """
            # collect per-run values
            policy_run_values = {}
            for key, df in Finalpolicies.items():
                if key in policies:
                    values = df.groupby("Run")[metric].tail(1).values
                    policy_run_values[key] = values
            
            # prepare empty matrix
            n = len(policies)
            pval_matrix = np.ones((n, n))   # diagonal = 1.0
            
            # fill upper triangle with Wilcoxon p-values
            significant = []
            for i, j in itertools.combinations(range(n), 2):
                p1, p2 = policies[i], policies[j]
                data1, data2 = policy_run_values[p1], policy_run_values[p2]
                
                # ensure equal length (important!)
                min_len = min(len(data1), len(data2))
                data1, data2 = data1[:min_len], data2[:min_len]
                
                stat, p = wilcoxon(data1, data2)
                pval_matrix[i, j] = p
                pval_matrix[j, i] = p  # symmetric
                
                if p >= alpha:
                    significant.append((p1, p2, p))
            
            # convert to DataFrame
            df_pvals = pd.DataFrame(pval_matrix, index=policies, columns=policies)
            
            # Print significance summary
            print("\nSignificant Comparisons (p >= {:.2f}):".format(alpha))
            if significant:
                for (p1, p2, p) in significant:
                    print(f"  {p1} vs {p2}: p = {p:.4f}")
            else:
                print("  None")
            
            return df_pvals

        # Example usage:
        wilcoxon_metric1="Percentage Visited"
        wilcoxon_metric2="Percentage of Trash Cleaned"
        for wilcoxon_metric in [wilcoxon_metric1,wilcoxon_metric2]:
            pval_table = wilcoxon_matrix(Finalpolicies, pf_policies, metric=wilcoxon_metric)
            # check if _nu_steps_dist_field or random_nus in csv_list[0]
            if "_nu_steps_dist_field" in csv_list[0]:
                algo_nu_steps = "DWS"
            elif "random_nus" in csv_list[0]:
                algo_nu_steps = "nu-RS"

            #print wilcoxon in a txt file, appending the results if the file already exists
            with open("./wilcoxon_results1.txt", "a") as f:
                f.write(f"\n{title_} {algo_nu_steps}\n")
                f.write(f"Wilcoxon Signed-Rank Test Results for Metric: {wilcoxon_metric}\n")
                f.write("\nWilcoxon p-value Matrix:\n")
                f.write(pval_table.round(4).to_string())
                f.write("\n")

###########################################
        # Calculate the Pareto front using the median instead of the mean
        objective_evaluations_median = {key:[[Finalpolicies[key].groupby(['Run'])[values_to_evaluate[i]].tail(1).median(),Finalpolicies[key].groupby(['Run'])[values_to_evaluate[i]].tail(1).std()] 
                            for i in range(len(values_to_evaluate))]
                            for key in Finalpolicies.keys()} # policy:[[median objective 1, std objective 1], [median objective 2, std objective 2]]
        # Regroup all tail values of all the policies in one dataframe to represent the boxplot
        all_tail_values = pd.concat([Finalpolicies[key].groupby(['Run'])[values_to_evaluate].tail(1).assign(Policy=key) for key in Finalpolicies.keys()])
        all_tail_values = all_tail_values.reset_index(drop=True)
        # # Boxplot of the distribution of the percentage_visited values (vertical)
        # plt.figure(figsize=(12, 8))
        # sns.boxplot(data=all_tail_values, x='Policy', y='Percentage Visited')
        # plt.xticks(rotation=90)
        # plt.title('Distribution of Percentage Visited (All Policies)')
        # plt.show()

        # # Boxplot of the percentage of trash cleaned (horizontal)
        # plt.figure(figsize=(12, 8))
        # sns.boxplot(data=all_tail_values, y='Policy', x='Percentage of Trash Cleaned')
        # plt.yticks(rotation=0)
        # plt.title('Distribution of Percentage of Trash Cleaned (All Policies)')
        # plt.show()
        objs_median = np.array([[-objective_evaluations_median[key][0][0], -objective_evaluations_median[key][1][0]] for key in objective_evaluations_median.keys()])
        stds_median = np.array([[objective_evaluations_median[key][0][1], objective_evaluations_median[key][1][1]] for key in objective_evaluations_median.keys()])
        policies_median = list(objective_evaluations_median.keys())

        nds_median = NonDominatedSorting()
        fronts_median = nds_median.do(objs_median)

        pf_median = -objs_median[fronts_median[0]]
        pf_stds_median = stds_median[fronts_median[0]]
        pf_policies_median = [policies_median[i] for i in fronts_median[0]]
        
        # Plot the Pareto front with median values
        plt.figure(figsize=(10, 6))
        for i, (obj, std, policy) in enumerate(zip(pf_median, pf_stds_median, pf_policies_median)):
            plt.errorbar(obj[0], obj[1], xerr=std[0], yerr=std[1], fmt='o', label=policy, capsize=5)
            plt.scatter(obj[0], obj[1], s=100)
            plt.annotate(policy.split('_')[-1].split('DRLResults')[0], (obj[0], obj[1]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel('Percentage of Trash Cleaned')
        plt.ylabel('Percentage of the Map Visited')
        plt.title('Pareto Front (Median)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=all_tail_values[all_tail_values['Policy'].isin(pf_policies_median)][['Percentage Visited','Policy']], 
                    x='Policy', 
                    y='Percentage Visited', hue='Policy')
        plt.xticks(rotation=0)
        plt.title('Distribution of Percentage Visited (Pareto Front)')
        plt.legend(title='Policy')
        plt.show()
        # Boxplot of the distribution of the percentage_visited values (vertical)
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=all_tail_values, x='Percentage of Trash Cleaned', y='Percentage Visited', order=pf_median[:,0])
        plt.xticks(rotation=90)
        plt.title('Distribution of Percentage Visited (Pareto Front)')
        plt.show()

        # Boxplot of the percentage of trash cleaned (horizontal)
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=all_tail_values, y='Percentage Visited', x='Percentage of Trash Cleaned', order=pf_median[:,1])
        plt.yticks(rotation=0)
        plt.title('Distribution of Percentage of Trash Cleaned (Pareto Front)')
        plt.show()
        #################
        data_path7 = 'C:\\Users\\dames\\OneDrive\\Documentos\\GitHub\\MultiAgentPatrollingProblem\\Evaluation\\Results_seed30_nu_intervals'

        pol_resul = 'Final_PolicyDRLResults.csv'
        
        Finalpolicy = pd.read_csv(f'{data_path7}/{pol_resul}')
        values_to_evaluate =['Accumulated Reward Intensification',
                            'Accumulated Reward Exploration',
                            'Total Accumulated Reward',
                            'Total Length',
                            'nu',
                            'Instantaneous Global Idleness Intensification',
                            'Instantaneous Global Idleness Exploration',
                            'Average Global Idleness Intensification',
                            'Average Global Idleness Exploration',
                            'Percentage Visited']
        grouped_data = Finalpolicy.groupby(['Policy Name', 'Run', 'Step'])[values_to_evaluate].mean().reset_index()

        ###########################
        
        # Plotting
        plt.figure(figsize=(12, 8))
        steps = -1
        Distint = 'Nu-intervals'
        line_styles = ['-', '--', '-.', ':']
        file_output = 'output_CAEPIA.txt'
        # Loop through each metric
        for metric in values_to_evaluate:
            # Loop through each Policy Name
            i=0
            for policy_name, group in grouped_data.groupby('Policy Name'):
                # Calculate mean values for each step
                mean_values = group.groupby('Step')[metric].mean()
                std_values = group.groupby('Step')[metric].std()
                mean_values = mean_values[:steps]
                std_values = std_values[:steps]
                
                if 'Global Idleness Exploration' in metric:
                    mean_values = mean_values[:31]
                    std_values = std_values[:31]
                    with open(file_output, "a") as f:
                        print(policy_name,metric, 'Mean', mean_values[30], file=f)
                        print(policy_name,metric, 'std', std_values[30], file=f)
                
                if 'Global Idleness Intensification' in metric:
                    with open(file_output, "a") as f:
                        print(policy_name,metric, 'Mean', mean_values.min(), file=f)                   
                # Plot a line for each Policy Name
                if 'Percentage Visited' in metric:
                    with open(file_output, "a") as f:
                        print(policy_name,metric, 'Mean', mean_values[30], file=f)
                plt.plot(mean_values.index, mean_values, label=f'{policy_name}', linestyle=line_styles[i%4])
                plt.fill_between(mean_values.index, mean_values - std_values, mean_values + std_values, alpha=0.1)
                i = i+1
            plt.xlabel('Step')
            mt = metric
            if 'nu' in metric:
                metric = r'$\nu$' 
            plt.ylabel(metric)
            plt.title(f'{metric}')
            """plt.axvspan(0, 30, alpha=0.1, color='blue', hatch='/', label='Exploration Phase')
            if 'Global Idleness Exploration' not in metric:
                plt.axvspan(30, 60, alpha=0.1, color='gray', label='Transition Phase')
                plt.axvspan(60, 100, alpha=0.1, color='green', hatch='\\\\', label='Intensification Phase')"""
            plt.legend(handlelength=4, handletextpad=1)
            plt.grid()
            #plt.axvline(x=30, color='red')

            if 'nu' in mt:
                metric = mt 
            #plt.savefig(f'{data_path1}/imagenes/{Distint}_{metric}.png',bbox_inches='tight')
            plt.show(block=True)

        Accum_per_episode = Finalpolicy.groupby(['Policy Name','Run'])[values_to_evaluate].tail(1) 
        #print(Accum_per_episode.to_markdown(),'\n \n \n')
        # merge the result dataframe with the original dataframe on the 'group' and 'value' columns
        Finalpolicy_accum = Finalpolicy.loc[Accum_per_episode.index]

        #print(Finalpolicy_accum.to_markdown(),'\n \n \n')
        Mean_per_episode = Finalpolicy_accum.groupby('Policy Name')[values_to_evaluate].mean()
        std_per_episode = Finalpolicy_accum.groupby('Policy Name')[values_to_evaluate].std()
        # Filter out rows based on their index
        #Mean_per_episode = Mean_per_episode[~Mean_per_episode.index.isin(indexes_to_skip)]
        with open(file_output, "a") as f:
            print(Mean_per_episode.sort_values('Average Global Idleness Intensification',ascending=True).to_markdown(),'\n \n \n', file=f)
            print(std_per_episode.sort_values('Average Global Idleness Intensification',ascending=True).to_markdown(),'\n \n \n', file=f)
            # para pasar a latex
            #print(Mean_per_episode.sort_values('Mean Weighted Idleness Exploration',ascending=True).style.to_latex(),'\n \n \n', file=f)
        from pymoo.factory import get_performance_indicator
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

        # extract the two objective values from the dataframe into a numpy array
        objs = - Mean_per_episode[['Mean Weighted Idleness Intensification', 'Mean Weighted Idleness Exploration',"Percentage Visited Exploration","Percentage Visited"]].to_numpy()
        objs =  Mean_per_episode[["Average Global Idleness Intensification","Average Global Idleness Exploration"]].to_numpy()


        nds = NonDominatedSorting()
        fronts = nds.do(objs)

        # get the solutions in the first front (i.e., the Pareto front)
        pf = objs[fronts[0]]

        # print the Pareto front
        with open(file_output, "a") as f:
            print("Pareto front:", file=f)
            print(pf,file=f)
        
        vals = ["Percentage Visited Exploration","Percentage Visited", "Accumulated Reward Intensification", "Accumulated Reward Exploration",
                "Average global idleness Intensification","Average global idleness Exploration"]
        vals = ["Percentage Visited Exploration","Percentage Visited",'Mean Weighted Idleness Intensification','Mean Weighted Idleness Exploration',
                'Mean Weighted Idleness', "Average Global Idleness Intensification","Average Global Idleness Exploration"]
        for val in vals:
            my_order =Mean_per_episode.sort_values(val,ascending=False).index
            plt.figure(figsize=(20,10))
            sns.set_style("whitegrid")
            sns.set(font_scale=1.8)
            ax=sns.boxplot(
            data=Finalpolicy_accum,
            x='Policy Name', y=val, hue='Policy Name',order=my_order,dodge=False
        )
            """if 'Mean Weighted Idleness Intensification' in val:
                current_ylim = ax.get_ylim()
                ax.set_ylim((current_ylim[0], current_ylim[1]*2))"""
            plt.title(val)
            plt.legend(fontsize = "25")
            plt.title(val, fontsize = "40")
            plt.ylabel(val, fontsize = "30")
            plt.show()
            #plt.savefig(f'{data_path5}/imagenes/{val}.png',bbox_inches='tight')
            plt.close()

    # to print with colorbar 
    """fig,ax=plt.subplots()
    im = ax.imshow(env.im1.get_array(),cmap='rainbow_r',vmin=0,vmax=1.0)
    plt.colorbar(im,ax=ax)"""