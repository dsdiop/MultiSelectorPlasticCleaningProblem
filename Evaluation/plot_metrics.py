import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import glob

map_ = 'alamillo_lake'
map_ = 'malaga_port'
print(map_)

csv_list = glob.glob("./Evaluation/Results/Results_seed_30_malaga_port_macro_plastic/*malaga_port__macro_plastic__0__1751293154_DRLResults.csv")

Finalpolicies = {csv_list[i].split('\\')[-1].split('.csv')[0]:pd.read_csv(csv_list[i]) for i in range(len(csv_list))}

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
# Plot the Pareto front
plt.figure(figsize=(15, 10))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.family'] = 'Times New Roman'
nu_val_ant=None
for i, (obj, std, policy) in enumerate(zip(pf, ci99, pf_policies)):
    obj = obj*100
    std = std*100
   
    # if nu_val == '0.7':
    #     va = 'bottom'
    #     si = -1
    # plot a shaded rectangle withing the confidence interval each rectangle with different color and type of shading
    #plt.fill_betweenx([obj[1]-std[1], obj[1]+std[1]], obj[0]-std[0], obj[0]+std[0], alpha=0.6, label=policy)
    
    plt.fill_between([obj[0]-std[0], obj[0]+std[0]], obj[1]-std[1], obj[1]+std[1], alpha=0.6, label=policy.split('__')[-1])
    plt.scatter(obj[0], obj[1], s=50, c='red', marker='x')
    # plt.annotate(f'{nu_val}', (obj[0], obj[1]), textcoords="offset points", fontsize=25, xytext=(0,si*30), ha='center',va=va, arrowprops=dict(facecolor='blue', shrink=0.05), fontweight='bold')
    print(f"{policy}: cleaning {obj[0]} +- {std[0]}, exploration {obj[1]} +- {std[1]}")
font_size = 40
font_size = 40
plt.xlabel('PTC (%)',fontsize=font_size, fontweight='bold')
plt.ylabel('PMV (%)',fontsize=font_size, fontweight='bold')
# fix the axis limits
if map_ == 'malaga_port':
    ax_limit_x = 70
    ax_limit_y = 85
    title_ = 'Malaga Port'
elif map_ == 'alamillo_lake':
    ax_limit_x = 75
    ax_limit_y = 60
    title_ = 'Alamillo Lake'

plt.title(title_, fontsize=font_size, fontweight='bold')
# import matplotlib.font_manager as font_manager 
# font = font_manager.FontProperties(weight='black',
#                            style='normal', size=font_size)
plt.legend(loc='lower left', bbox_to_anchor=(1.0, 0.00), fontsize=30)

    
plt.xlim(ax_limit_x, 100)
plt.ylim(ax_limit_y, 100)
# make the aspect ratio equal
#plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
# make the grid lines every 0.1
plt.xticks(np.arange(ax_limit_x, 105, 5), fontsize=30, fontweight='bold')
plt.yticks(np.arange(ax_limit_y, 105, 5), fontsize=30, fontweight='bold')
# make the space between grid lines more widely distributed
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.tight_layout()
# plt.savefig(f'pareto_front_{map_}_chapter.png')
plt.show()
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