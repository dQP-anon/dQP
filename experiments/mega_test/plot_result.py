import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('data_name', metavar='data_name', type=str)
parser.add_argument('--time_accuracy',action='store_true',default=False, help="Plot dim vs time and accuracy")
parser.add_argument('--times',action='store_true',default=False, help="Plot forward/backward times")
parser.add_argument('--scatter',action='store_true',default=False, help="Plot scatter plot of time vs accuracy")

args = parser.parse_args()
data_name = args.data_name
file_path = 'results/' + data_name + '.csv'
save_path = 'results/fig/'
results= pd.read_csv(file_path)
results['Primal residual'] = results[['Eq residual', 'Ineq residual']].max(axis=1)
df_plt = results[results['status']=='Success']
solvers = list(set(df_plt.solver))
solvers.sort()
colors = plt.cm.tab10.colors[:len(solvers)][::-1]

x_var = 'dim'
label_font = 15
legend_font = 12


#%% time vs dim

def plot_total_time(axes):
    """Plot total time on the given axes."""
    y_var = 'Total time'
    for i, (k, v) in enumerate(df_plt.groupby("solver")):

        color = colors[i % len(colors)]
        tmp = np.sort(v[x_var].unique())
        med = (v.groupby(x_var)[y_var]).median().reindex(tmp)
        label = 'Ours' if k[:3] == "dQP" else k
        label = k 
        axes.plot(tmp, med, label=label, color=color)

        # Calculate quantiles for filling (if needed)
        q1 = (v.groupby(x_var)[y_var]).quantile(0.25).reindex(tmp)
        q3 = (v.groupby(x_var)[y_var]).quantile(0.75).reindex(tmp)
        axes.fill_between(tmp, q1, q3, alpha=0.2, color=color)

    axes.set_xscale('log')
    axes.set_yscale('log')
    


def plot_time_accuracy():
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))  
    
    plot_total_time(axes[0])
    
    benchmarks = ['Duality gap']
    for j, column in enumerate(benchmarks):
        for i, (k, v) in enumerate(df_plt.groupby("solver")):
            color = colors[i % len(colors)]
            v_sorted = v.sort_values(by='dim')
            
            grouped = v_sorted.groupby('dim')[column].agg(['median'])
            grouped['Q1'] = v_sorted.groupby('dim')[column].quantile(0.25)
            grouped['Q3'] = v_sorted.groupby('dim')[column].quantile(0.75)
            axes[1].plot(grouped.index, grouped['median'], label=k, color=color)
            axes[1].fill_between(grouped.index, grouped['Q1'], grouped['Q3'], color=color, alpha=0.2)
    axes[1].set_xlabel('Dim',fontsize=label_font)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    
    legend = axes[0].legend(fontsize=legend_font,loc="upper left")
    fig.suptitle(data_name,fontsize=label_font)
    plt.tight_layout()
    plt.savefig(save_path  + data_name+ '_time_accuracy'+'.pdf', bbox_inches='tight')
    

#%%
def plot_times():
    labels = []
    legends = []
    x_var = 'dim'
    y_vars = ['Total time','Forward time', 'Backward time']
    
    dot_size = 5
    fig, axes = plt.subplots(2, 1, figsize=(10,12))
    
    for (j,y_var) in enumerate(y_vars):
        
        for i, (k, v) in enumerate(df_plt.groupby("solver")):
            if j==0:
                color = colors[i % len(colors)]
                tmp = np.sort(v[x_var].unique())
                med = (v.groupby(x_var)[y_var]).median().reindex(tmp)
                axes[j].plot(tmp, med, color=color, label=k )
                
            else:
    
                color = colors[i % len(colors)]
                tmp = np.sort(v[x_var].unique())
                med = (v.groupby(x_var)[y_var]).median().reindex(tmp)
                if y_var=="Forward time":
                    axes[1].plot(tmp, med, color=color, label=k + ' Forward')
                else: 
                    axes[1].plot(tmp, med, color=color, label=k + ' Backward', linestyle='--')
                    
        if j==0:
            axes[j].set_xscale('log')
            axes[j].set_yscale('log')
            axes[j].set_xlabel(x_var,fontsize=label_font)
            axes[j].set_title(y_var)
            axes[j].legend(loc="upper left",fontsize=legend_font)
        
            
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel(x_var,fontsize=label_font)
    axes[1].set_title("Forward/ Backward time") 
    axes[1].legend(loc="upper left",fontsize=legend_font)
    fig.suptitle(data_name,fontsize=label_font)    
    plt.savefig(save_path  + data_name+ '_times'+'.pdf', bbox_inches='tight')





#%% Scatter plot with success rate in the legend 
# Each dot is a problem with dotsize reflects its dim 
def plot_scatter():
    
    df_plt = results[(results['status']=='Success')]
    bm = ['Total time', 'Duality gap']
    n_prob = len(set(results.Problem))
    plt.figure(figsize=(10,9))
    
    for (j,solver) in enumerate(solvers):
        suc_rate = int(np.ceil(100 * df_plt[df_plt['solver']==solvers[j]]['Problem'].count() / n_prob ))
        color = colors[j % len(colors)]
        if solver[:3]=='dQP':
            df_act = df_plt[df_plt['solver']==solver]
            df_noact = df_plt[df_plt['solver']!=solver]
            prob_act = df_act.Problem 
            prob_noact = list(set(df_noact.Problem))
            prob_all_solved = [element for element in prob_act if element in prob_noact ]
            prob_act_only = [element for element in prob_act if element not in prob_noact ]
            data_all_solved = df_plt[(df_plt['solver'] == solver) & (df_plt['Problem'].isin(prob_all_solved))]
            data_act_only = df_plt[(df_plt['solver'] == solver) & (df_plt['Problem'].isin(prob_act_only))]
            act_solver = df_plt.groupby("dQP solver")["Problem"].count().sort_values(ascending=False)
            legend_label = "\n".join([f"  {solver:10} {value}" for solver, value in act_solver.items()])
            plt.scatter(data_all_solved[bm[0]],
                        data_all_solved[bm[1]],
                        label=r'$\bf{Ours (100\%)}$' + "\n" + legend_label,
                        s=15 * np.log10(data_all_solved['dim']),
                        color=color,
                        linewidth=0,
                        edgecolors='black'
                        ) 
            plt.scatter(data_act_only[bm[0]],
                        data_act_only[bm[1]],
                        label=None,
                        s=15 * np.log10(data_act_only['dim']),
                        edgecolors='black',  
                        color=color,
                        linewidth=2) 
        else:
            plt.scatter(df_plt[df_plt['solver']==solvers[j]][bm[0]],
                        df_plt[df_plt['solver']==solvers[j]][bm[1]],
                        label=solvers[j] + ' ('+ str(suc_rate)  +  str('%') + ')',
                        s=15 * np.log10(df_plt[df_plt['solver']==solvers[j]]['dim']),
                        color=color,
                        linewidth=0,
                        edgecolors='black')
    legend = plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0,fontsize=legend_font)
    
    
    
    plt.axhline(y=1e-6, color='gray', linestyle='--', linewidth=0.7)
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Running Time',fontsize=label_font)
    plt.ylabel('Duality Gap',fontsize=label_font)
    
    y_ticks = plt.yticks()[0]  
    y_ticks = np.unique(np.append(y_ticks, [1e-6])) 
    plt.yticks(y_ticks)  
    plt.title(data_name,fontsize=label_font)
    plt.savefig(save_path  + data_name + '_scatter'+'.pdf', bbox_inches='tight')

if args.time_accuracy:
    plot_time_accuracy()
if args.times:  
    plot_times()
if args.scatter:
    plot_scatter()


