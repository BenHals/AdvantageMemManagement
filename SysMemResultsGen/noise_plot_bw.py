import pickle
import random
import math
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats

sns.set()
sns.set_context("paper")
sns.set_style("ticks")
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

cmap = plt.get_cmap('viridis')
cmap = plt.get_cmap('cubehelix')
cmap = plt.get_cmap('magma')
end_shift_prop = 0.1
end_shift_amount = cmap.N * end_shift_prop
indices = np.linspace(end_shift_amount, cmap.N-end_shift_amount, 7)
my_colors = [cmap(int(i)) for i in indices]
print(my_colors)
# random.shuffle(my_colors)
for i in range(1, math.floor(len(my_colors)/2), 2):
    my_colors[i], my_colors[-(i + 1)] = my_colors[-(i+1)], my_colors[i]


print(my_colors)
sns.set_palette(my_colors)
# sns.set_palette(['#000000', '#ABABAB'], n_colors=100)

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file",
    help="filename", default="len6vall.pickle")
ap.add_argument("-m", "--metric",
    help="metric", default="accuracy", choices=["time", "memory", "f1", "f1 by System", "drift accuract", "detect accuracy", "accuracy", "Overall accuracy", "average_purity", 'Max Recall',  'Max Precision',  'Precision for Max Recall',  'Recall for Max Precision',  'MR by System',  'MP by System',  'PMR by System',  'RMP by System',  'Num Good System Concepts',  'Max Recall_alt',  'Max Precision_alt',  'Precision for Max Recall_alt',  'Recall for Max Precision_alt',  'MR by System_alt',  'MP by System_alt',  'PMR by System_alt',  'RMP by System_alt',  'Num Good System Concepts_alt', ])
ap.add_argument("-iv", "--indvar",
    help="iv", default="run_noise")
ap.add_argument("-ttv", "--ttestvar",
    help="iv", default= None)


args = vars(ap.parse_args())

metric_replace = {'driftdetect_accuracy_50': "Drift point accuracy", 'accuracy':"Accuracy", 'f1':"$F1_c$", 'f1 by System': "$F1_s$", 'Num Good System Concepts' : '$s \\approx c$',
    "time": "Time (s)", "memory": "Memory (KB)", "run_noise": "Noise Proportion", "merge_similarity" : "Kappa Agreement Threshold"}
metric = args['metric']
iv = args['indvar']
ttestvar = args['ttestvar']
save_name = f"{args['file'].split('.')[0]}-{metric.replace(' ', '')}"
with open(f"Mem_Manage_Results/{args['file']}", 'rb') as f:
    # df = pickle.load(f)
    df = pd.read_pickle(f)

def custom_mean(series):
    m = np.mean(series)
    std = np.std(series)

    return f"{m:.4} ({std:.2})"

def custom_ttest_index(index):
    def custom_ttest(series):
        mm_levels = np.unique(series.index.get_level_values(index))
        print(mm_levels)

        a = series.loc[series.index.get_level_values(index) == mm_levels[0]].dropna().values
        b = series.loc[series.index.get_level_values(index) == mm_levels[1]].dropna().values
        print(a)
        print(b)
        return scipy.stats.ttest_ind(a, b, equal_var= False)
    
    return custom_ttest

def series_ttest(series):
    return scipy.stats.ttest_1samp(series, 1)



print(df)

make_table = True

df = df.loc[df.index.get_level_values('sys_learner') == "HN"]
df = df.groupby(["cl", "st", "mem_manage", "rep", "drift_window", "poisson", "sens", "sys_learner", "run_noise", "ml"]).aggregate([(metric, np.max)])
print(df)
df.columns = df.columns.droplevel(1)
print(df)
if make_table:
    df_table = df.unstack(level = iv)
    print(df_table.index)
    print(df_table)
    df_table = df_table.rename(index = {'def': '0'}, level = 'poisson')
    df_table = df_table.rename(index = {'def': 'arf'}, level = 'mem_manage')
    df_table = df_table.rename(index = {'arf': 'ARF'}, level = 'mem_manage')
    df_table = df_table.rename(index = {'rA': '# Evolutions'}, level = 'mem_manage')
    df_table = df_table.rename(index = {'RCStreamType-RBF': 'RBF'}, level = 'st')
    df_table = df_table.rename(index = {'RCStreamType-TREE': 'TREE'}, level = 'st')
    df_table = df_table.rename(index = {'RCStreamType-WINDSIM': 'WINDSIM'}, level = 'st')
    
    print(df_table.index)
    df_table = df_table[metric]

    print(df_table.columns)
    print(df_table)
    try:
        noise_levels = sorted(list(df_table.columns), key = lambda x: float(x))
    except:
        noise_levels = df_table.columns
        print("cant sort")
    print(noise_levels)
    
    
    for c_i, c in enumerate(noise_levels[1:], start = 1):
        n0 = noise_levels[c_i - 1]
        n1 = noise_levels[c_i]
        colname = f"{n0} -\> {n1}"
        col = df_table.loc[:, n1] / df_table.loc[:, n0]
        print(col)
        df_table[colname] = col
    
    df_table = df_table.dropna()
    print(df_table)
    
        # df_table_val[colname] = df_table_val.loc[:, n0] - df_table_val.loc[:, n1]
    # df_table = df_table.groupby(level=['poisson', 'st', 'ml', 'sys_learner',  'mem_manage',]).aggregate([('Mean (STD)', custom_mean)])
    

    df_table_ttest = df_table.groupby(level=['st']).aggregate([ custom_ttest_index(ttestvar) if not ttestvar is None else series_ttest])
    df_table = df_table.groupby(level=['st', 'mem_manage',]).aggregate([('Mean (STD)', custom_mean)])

    print(df_table)
    df_table = df_table.loc[df_table.index.get_level_values(1) != 'arf']
    df_table = df_table.loc[df_table.index.get_level_values(1) != 'rAAuc']



    print(df_table)

    

    print(df_table)
    df_table.columns.droplevel(1)
    print(df_table)
    df_table.index.names = [x if x != 'st' else 'Stream Source' for x in df_table.index.names]
    df_table.index.names = [x if x != 'mem_manage' else 'Memory Management Strategy' for x in df_table.index.names]
    print(df_table)
    


    
    print(df_table)
    print(df_table_ttest)

    # print(df_table.xs('mean', axis = 1, level = 1, drop_level = False))
    # print(df_table.xs('std', axis = 1, level = 1, drop_level = False))
    # print(df_table.xs('mean', axis = 1, level = 1, drop_level = False).map(str) + df_table.xs('std', axis = 1, level = 1, drop_level = False).map(str))
    # df_table.loc['mean'] = df_table.xs('mean', axis = 1, level = 1, drop_level = False) + df_table.xs('std', axis = 1, level = 1, drop_level = False)
    with open(f'Mem_Manage_Results/{save_name}_{iv}_plot_bw.txt', 'w') as f:
        print(df_table)
        df_table.to_latex(f)
    with open(f'Mem_Manage_Results/{save_name}_{iv}_ttest_plot_bw.txt', 'w') as f:
        print(df_table_ttest)
        df_table_ttest.to_latex(f)
    # exit()


df = df.unstack(level=iv)
df = df.unstack(level='mem_manage')

df_grouped = df.groupby(level=['ml', 'sys_learner', 'sens', 'poisson', 'st']).aggregate([np.mean, np.std])
df_grouped = df_grouped.unstack(level='st')

print(df_grouped.columns.levels)
cl_vals = [float(x) for x in list(df_grouped.columns.get_level_values(1))]
mm_vals = list(df_grouped.columns.get_level_values(2))
print(cl_vals)
print(np.any(np.asarray(cl_vals) == 2))
cl_vals = np.asarray(cl_vals) < 30
mm_vals = np.asarray(mm_vals) == 'def'
# cl_vals = np.bitwise_or(np.asarray(cl_vals) % 5 == 0, np.asarray(cl_vals) == 1, np.asarray(cl_vals) < 30)
print(cl_vals)
# exit()
print(mm_vals)
# df_grouped = df_grouped.iloc[:, np.bitwise_and(cl_vals,mm_vals)]
print(df_grouped.shape)



st_levels = df_grouped.columns.get_level_values('st').unique()
print(st_levels)
overall = None
poisson_labels = {}

for st in st_levels:
    st_df = df_grouped.iloc[:, df_grouped.columns.get_level_values('st') == st]
    print(st_df)
    ml_levels = st_df.index.get_level_values('ml').unique()
    for ml in ml_levels:
        ml_df = st_df.loc[st_df.index.get_level_values('ml') == ml]
        
        if ml == 'rcd':
            continue
            
        print(ml_df)
        poisson_levels = ml_df.index.get_level_values('poisson').unique()

        for p in poisson_levels:
            print(f"st: {st}, ml: {ml}, p: {p}")
            ml_p_df = ml_df.loc[ml_df.index.get_level_values('poisson') == p]

            melt = pd.melt(ml_p_df)
            
            
            # melt.columns = ['res', 'cl', 'mem_manage', 'res_agg', 'value']
            melt.columns = ['res', iv, 'mem_manage', 'res_agg', 'st', 'value']
            if ml == 'arf':
                melt['mem_manage'] = 'arf'
            if ml == 'rcd':
                melt['mem_manage'] = 'rcd'
            
            
            
            melt = melt.loc[melt['res'] == metric]
            melt[iv] = pd.to_numeric(melt[iv])
            melt = melt.sort_values(by = [iv])
            melt = melt.dropna()
            print(len(melt))
            if len(melt) < 1:
                continue
            melt['name'] = f"{ml}:{p}:" + melt['mem_manage']
            melt['creator'] = melt['mem_manage'].map(lambda x: "theirs" if x in ['acc', 'age', 'LRU', 'div'] else 'base' if x in ['arf', 'rcd'] else 'mine')
            melt['poisson'] = melt['name'].map(lambda x: p)

            
            melt_std = melt.loc[melt['res_agg'] == 'std']
            # print(melt_std)
            melt = melt.loc[melt['res_agg'] == 'mean']


            print(melt)
            melt = pd.merge(melt, melt_std[[iv, 'name', 'value']], on = ['name', iv], how = 'left')
            melt = melt.rename({'value_x': 'mean', 'value_y': 'std'}, axis='columns')
            print(melt)
            if ml == 'arf':
                melt[iv] = melt[iv] * 2
                melt = melt.loc[melt[iv] <= 30]
            melt = melt.loc[melt['mem_manage'] != 'rAAuc']
            
            
            final_x = melt[iv].unique()[-1]
            avg_final_y = melt[melt[iv] == final_x]['mean'].mean()
            
            if overall is None:
                overall = melt
            else:
                overall = overall.append(melt)
            
            if p.isdigit():
                if st not in poisson_labels:
                    poisson_labels[st] = []
                poisson_labels[st].append((final_x + 0.25, avg_final_y, f"poisson = {p}"))

print(poisson_labels)
print(overall)

# Facet

# g = sns.FacetGrid(overall, col = 'st', hue = 'mem_manage')
# g.map(plt.plot, iv, 'mean')
# plt.show()

# Lineplot

def plot_graph(data, poisson_labels, ax, show_legend = True, st = "Default"):
    # cmap = plt.get_cmap('viridis')
    # indices = np.linspace(25, cmap.N-25, 2)
    # my_colors = random.shuffle([cmap(int(i)) for i in indices])

    g = sns.lineplot(x=iv, y='mean', err_style = 'bars', ax=ax,
    # linewidth=1, data=data, hue='mem_manage', hue_order=['score', 'rA', 'auc', 'rAAuc', 'arf', 'rcd', 'age', 'LRU', 'acc', 'div'], sort=True, style= 'creator', style_order= ['mine', 'theirs', 'base'], units = "poisson", estimator=None)
    linewidth=2, data=data, hue='mem_manage', hue_order=['rA', 'LRU'], dashes=["", (4, 2), (4, 2)], sort=True, style= 'creator', style_order= ['mine', 'theirs', 'base'], units = "poisson", estimator=None)

    if show_legend:
        h, l = g.get_legend_handles_labels()
        rename_strategies = {'rA': "#E", "auc": "AAC", "score": "EP", "acc": "Acc", "age":"FIFO", "LRU":"LRU", "div":"DP", 'arf': 'ARF'}
        h_ours = [x[0] for x in zip(h, l) if x[1] in ['rA', 'auc', 'score']]
        h_theirs = [x[0] for x in zip(h, l) if x[1] in ['acc', 'age', 'LRU', 'arf', 'div', 'rcd']]


        l_ours = [rename_strategies[x[1]] for x in zip(h, l) if x[1] in ['rA', 'auc', 'score']]
        l_theirs = [rename_strategies[x[1]] for x in zip(h, l) if x[1] in ['acc', 'age', 'LRU', 'arf', 'div', 'rcd']]

        for line, label in zip(h_theirs, l_theirs):
            line.set_linestyle("--")
            if label == 'arf':
                line.set_linestyle(":")

        # box = g.get_position()
        # g.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position
        
        # ours = ax.legend(h_ours, l_ours, 
        #                     fancybox=True, ncol=1, title = "proposed", bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.)
        # theirs = ax.legend(h_theirs, l_theirs, 
        #                     fancybox=True, ncol=1, title = "baselines", bbox_to_anchor=(1.05, 0.25), loc=2, borderaxespad=0.1)
        # ax.add_artist(ours)
        l = ax.legend()
        l.remove()
    else:
        l = ax.legend()
        l.remove()

    for p_label in poisson_labels:
        continue
        # ax.text(p_label[0], p_label[1], p_label[2], verticalalignment = 'center')

    g.set_title(f"{st.split('-')[-1]}")
    ax.set_xlabel(metric_replace[iv] if iv in metric_replace else iv)
    ax.set_ylabel(metric_replace[metric] if metric in metric_replace else metric)
def plot_graph_minimal(data, poisson_labels, ax, show_legend = True, st = "Default", m = 'accuracy'):
    # cmap = plt.get_cmap('viridis')
    # indices = np.linspace(25, cmap.N-25, 2)
    # my_colors = random.shuffle([cmap(int(i)) for i in indices])
    g = sns.lineplot(x=iv, y='mean', err_style = 'bars', ax=ax,
    # linewidth=1, data=data, hue='mem_manage', hue_order=['score', 'rA', 'auc', 'rAAuc', 'arf', 'rcd', 'age', 'LRU', 'acc', 'div'], sort=True, style= 'creator', style_order= ['mine', 'theirs', 'base'], units = "poisson", estimator=None)
    linewidth=3,  data=data, hue='mem_manage', hue_order=['rA', 'LRU'], sort=True, dashes=["", (4, 2), (4, 2)], style= 'creator', style_order= ['mine', 'theirs', 'base'], units = "poisson", estimator=None)

    if show_legend or True:
        h, l = g.get_legend_handles_labels()
        rename_strategies = {'rA': "#E", "auc": "AAC", "score": "EP", "acc": "Acc", "age":"FIFO", "LRU":"LRU", "div":"DP", 'arf': 'ARF'}
        h_ours = [x[0] for x in zip(h, l) if x[1] in ['rA', 'auc', 'score']]
        h_theirs = [x[0] for x in zip(h, l) if x[1] in ['acc', 'age', 'LRU', 'arf', 'div', 'rcd']]


        l_ours = [rename_strategies[x[1]] for x in zip(h, l) if x[1] in ['rA', 'auc', 'score']]
        l_theirs = [rename_strategies[x[1]] for x in zip(h, l) if x[1] in ['acc', 'age', 'LRU', 'arf', 'div', 'rcd']]

        for line, label in zip(h_theirs, l_theirs):
            line.set_linestyle("--")
            if label == 'arf':
                line.set_linestyle(":")

        # box = g.get_position()
        # g.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position
        
        # ours = ax.legend(h_ours, l_ours, 
        #                     fancybox=True, ncol=1, title = "proposed", bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.)
        # theirs = ax.legend(h_theirs, l_theirs, 
        #                     fancybox=True, ncol=1, title = "baselines", bbox_to_anchor=(1.05, 0.25), loc=2, borderaxespad=0.1)
        # ax.add_artist(ours)
        l = ax.legend()
        l.remove()
    else:
        l = ax.legend()
        l.remove()

    for p_label in poisson_labels:
        continue
        # ax.text(p_label[0], p_label[1], p_label[2], verticalalignment = 'center')

    g.set_title(f"{st.split('-')[-1]}")
    ax.set_xlabel(metric_replace[iv] if iv in metric_replace else iv)
    ax.set_ylabel(metric_replace[metric] if metric in metric_replace else metric)
    
    return h_ours, h_theirs, l_ours, l_theirs

unique_stream_types = overall['st'].unique()
# unique_stream_types = unique_stream_types[:2]
# fig, axs = plt.subplots(1, len(unique_stream_types), sharey=True, figsize = (20,5))
fig, axs = plt.subplots(1, len(unique_stream_types), figsize = (20,5))
# fig, axs = plt.subplots(1, len(unique_stream_types), figsize = (13.33,5))
if len(unique_stream_types) < 2:
    axs = [axs]

for st_i, st in enumerate(unique_stream_types):
    print(st)
    df_st = overall.loc[overall['st'] == st]
    # print(df_st.sample(5))
    plot_graph(df_st, poisson_labels[st], axs[st_i], st_i == len(unique_stream_types) - 1, st)

for st_i, st in enumerate(unique_stream_types):
    m_savename = f"{args['file'].split('.')[0]}-{[x.replace(' ', '') for x in ['accuracy']]}"
    min_fig = plt.figure()
    ax = min_fig.gca()
    try:
        st_name = unique_stream_types[st_i].split('-')[1]
    except:
        st_name = unique_stream_types[st_i]

    # Save just the portion _inside_ the second axis's boundaries
    print(st)
    df_st = overall.loc[overall['st'] == st]

    h_ours, h_theirs, l_ours, l_theirs = plot_graph_minimal(df_st, poisson_labels[st], ax, False, st, 'accuracy')
    min_fig.savefig(f'Mem_Manage_Results/{m_savename}{st_name}_figure_bw.png', bbox_inches='tight')


pic_suffix = ""
# pic_suffix = "ss"
fig.savefig(f"Mem_Manage_Results/{save_name}{pic_suffix}_{iv}_bw.png", dpi=200, bbox_inches='tight')
fig.savefig(f"Mem_Manage_Results/{save_name}{pic_suffix}H_{iv}_bw.png", dpi=500, bbox_inches='tight')
fig.savefig(f"Mem_Manage_Results/{save_name}{pic_suffix}_{iv}_bw.pdf", bbox_inches='tight')

figsize = (3, 0.1)
fig_leg = plt.figure(figsize=figsize)
ax_leg = fig_leg.add_subplot(111)


print(h_ours, h_theirs, l_ours, l_theirs)
ours = ax_leg.legend(h_ours, l_ours, 
                    fancybox=True, ncol=len(h_ours), title = "Proposed", bbox_to_anchor=(0, 0, 0.4, 0.1), loc="upper left", borderaxespad=0, mode='expand', frameon=False)
ours.get_title().set_fontsize('14')
theirs = ax_leg.legend(h_theirs, l_theirs, 
                    fancybox=True, ncol=len(h_theirs), title = "Baseline", bbox_to_anchor=(0.4, 0, 0.6, 0.1), loc="upper left", borderaxespad=0, mode='expand', frameon=False )
theirs.get_title().set_fontsize('14')
# ours = m_fig.legend(h_ours, l_ours, 
#                     fancybox=True, ncol=len(h_ours), title = "proposed", bbox_to_anchor=(0, 1.02), loc="lower left", borderaxespad=0.1)
# theirs = m_fig.legend(h_theirs, l_theirs, 
#                     fancybox=True, ncol=len(h_theirs), title = "baselines", bbox_to_anchor=(1, 1.02), loc="lower right", borderaxespad=0.1)
ax_leg.add_artist(ours)
# add the legend from the previous axes
# hide the axes frame and the x/y labels
ax_leg.axis('off')
fig_leg.tight_layout()
fig_leg.savefig(f'Mem_Manage_Results/{save_name}_legend_bw.pdf', bbox_inches='tight')



plt.show()