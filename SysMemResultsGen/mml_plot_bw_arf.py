import pickle
import argparse
import math 
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
sns.set()
sns.set_context("paper")
sns.set_style("ticks")
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

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
sns.set_palette(my_colors)

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file",
    help="filename", default="len6vall.pickle")
ap.add_argument("-m", "--metric", type=str, nargs="+",
    help="metric", default="accuracy")

args = vars(ap.parse_args())

metric = args['metric']
metric_replace = {'driftdetect_accuracy_50': "Drift point accuracy", 'accuracy':"Accuracy", 'f1':"$F1_c$", 'f1 by System': "$F1_s$", 'Num Good System Concepts' : '$s \\approx c$',
    "time": "Time (s)", "memory": "Memory (KB)"}
save_name = f"{args['file'].split('.')[0]}-{[x.replace(' ', '') for x in metric]}"
with open(f"Mem_Manage_Results/{args['file']}", 'rb') as f:
    # df = pickle.load(f)
    df = pd.read_pickle(f)

def custom_mean(series):
    multiplier = 100 if '_accuracy' in series.name[0] else 1
    series = series * multiplier
    m = np.mean(series)
    std = np.std(series)

    return f"{m:.2f}({std:.2f})"


# df = df.loc[df.index.get_level_values('st') == "RCStreamType-TREE"]

print(df)
# exit()

make_table = False

if make_table:
    df_table = df.unstack(level = 'cl')

    # print(df_table.iloc[:, df_table.columns.get_level_values(1) == '25'])

    # print(df_table.index)
    df_table = df_table.rename(index = {'def': '0'}, level = 'poisson')
    df_table = df_table.rename(index = {'def': 'arf'}, level = 'mem_manage')
    df_table = df_table.rename(index = {'RCStreamType-RBF': 'RBF'}, level = 'st')
    df_table = df_table.rename(index = {'RCStreamType-TREE': 'TREE'}, level = 'st')
    df_table = df_table.rename(index = {'RCStreamType-WINDSIM': 'WINDSIM'}, level = 'st')
    # print(df_table.index)


    # df_table = df_table.groupby(level=['poisson', 'st', 'ml', 'sys_learner',  'mem_manage',]).aggregate([('Mean (STD)', custom_mean)])
    df_table_val = df_table.groupby(level=['st', 'mem_manage',]).aggregate([np.mean])
    df_table = df_table.groupby(level=['st', 'mem_manage',]).aggregate([('', custom_mean)])

    df_table_val = df_table_val.loc[df_table.index.get_level_values(1) != 'arf']
    df_table = df_table.loc[df_table.index.get_level_values(1) != 'arf']
    df_table_val = df_table_val.loc[df_table.index.get_level_values(1) != 'rAAuc']
    df_table = df_table.loc[df_table.index.get_level_values(1) != 'rAAuc']
    # skip weird arf values (wrong first run)
    cl_vals = [int (x) for x in list(df_table.columns.get_level_values(1))]
    # cl_vals = (np.asarray(cl_vals) % 5 == 0) | (np.asarray(cl_vals) == 1)
    cl_vals = (np.isin(np.asarray(cl_vals), [5, 15, 25, 35]))
    # cl_vals = (np.isin(np.asarray(cl_vals), [5, 20, 35]))
    # cl_vals = (np.isin(np.asarray(cl_vals), [5, 20]))
    # print(cl_vals)
    df_table_val.iloc[:, cl_vals]
    df_table = df_table.iloc[:, cl_vals]



    # print(df_table)

    df_table = df_table[metric]
    df_table_val = df_table_val[metric]
    print(df_table.columns)
    # df_table.columns = df_table.columns.swaplevel(0, 1)
    print(df_table.columns)
    s = sorted(df_table.columns.levels[1], key = lambda x: float(x) if x.isdigit() else x)


    
    # print(s)
    df_table = df_table.reindex(s, axis=1, level = 1)
    df_table_val = df_table_val.reindex(s, axis=1, level = 1)
    print(df_table.columns.levels[1])
    df_table = df_table.rename(columns = {'driftdetect_accuracy_50': "Drift point accuracy", 'accuracy':"Accuracy", 'f1':"$F1_c$", 'f1 by System': "$F1_s$", 'Num Good System Concepts' : '$s \\approx c$'}, level = 0)
    df_table_val = df_table_val.rename(columns = {'accuracy':"Accuracy", 'f1':"$f1_c$", 'f1 by System': "$f1_s$"}, level = 0)
    print(df_table)
    # exit()
    
    # print(df_table)
    df_table = df_table.reindex(['LRU','acc', 'age', 'div', 'auc', 'rA', 'score'], axis=0, level = 'mem_manage')
    df_table_val = df_table_val.reindex(['LRU','acc', 'age', 'div', 'auc', 'rA', 'score'], axis=0, level = 'mem_manage')
    # print(df_table)
    # df_table.columns.droplevel(2)
    # df_table_val.droplevel(2)
    # print(df_table)
    # print(df_table_val.loc[:, '25'] - df_table_val.loc[:, '15'])
    # df_table['$\delta$ Acc 25 $->$ 15 '] = df_table_val.loc[:, '25'] - df_table_val.loc[:, '15']
    # df_table['\multicolumn{1}{c}{Accuracy by Memory Plot}'] = df_table.index.get_level_values(0)
    # # df_table['\multicolumn{1}{c}{Accuracy by Memory Plot}'] = df_table['\multicolumn{1}{c}{Accuracy by Memory Plot}'].apply(lambda x: x.split('-')[1])
    # df_table['\multicolumn{1}{c}{Accuracy by Memory Plot}'] = df_table['\multicolumn{1}{c}{Accuracy by Memory Plot}'].apply(lambda x: f"\multirow{{7}}{{*}}{{\includegraphics[width=4cm]{{images/{save_name}{x}_figure.png}}}}")
    # changed = df_table['\multicolumn{1}{c}{Accuracy by Memory Plot}'].ne(df_table['\multicolumn{1}{c}{Accuracy by Memory Plot}'].shift(fill_value = "na"))
    # print(df_table)
    # print(df_table['\multicolumn{1}{c}{Accuracy by Memory Plot}'].loc[~changed])
    # df_table.loc[~changed, "\multicolumn{1}{c}{Accuracy by Memory Plot}"] = ""
    df_table = df_table.rename(index = {'RBF': '\multirow{7}*{\\rotatebox{90}{RBF}}'}, level = 'st')
    df_table = df_table.rename(index = {'TREE': '\multirow{7}*{\\rotatebox{90}{TREE}}'}, level = 'st')
    df_table = df_table.rename(index = {'WINDSIM': '\multirow{7}*{\\rotatebox{90}{WINDSIM}}'}, level = 'st')
    # print(df_table)
    

    # print(df_table.xs('mean', axis = 1, level = 1, drop_level = False))
    # print(df_table.xs('std', axis = 1, level = 1, drop_level = False))
    # print(df_table.xs('mean', axis = 1, level = 1, drop_level = False).map(str) + df_table.xs('std', axis = 1, level = 1, drop_level = False).map(str))
    # df_table.loc['mean'] = df_table.xs('mean', axis = 1, level = 1, drop_level = False) + df_table.xs('std', axis = 1, level = 1, drop_level = False)
    df_table.index.names = [x if x != 'st' else '' for x in df_table.index.names]
    df_table.index.names = [x if x != 'mem_manage' else 'Strategy' for x in df_table.index.names]
    with pd.option_context("max_colwidth", 1000):
        table_latex = df_table.to_latex(escape = False)
    table_latex_lines = table_latex.splitlines()
    new_latex_lines = []
    
    max_cols = {}
    st = ""
    for l_i,l in enumerate(table_latex_lines):

        if 'RBF' in l:
            st = 'RBF'
        if 'TREE' in l:
            st = 'TREE'
        if 'WINDSIM' in l:
            st = 'WINDSIM'

        cells = l.split('&')
        # if 'RBF' in cells[0]:
        #     cells[0] = '\multirow{5}*{\\rotatebox{90}{RBF}}'
        # if 'TREE' in l:
        #     cells[0] = '\multirow{5}*{\\rotatebox{90}{TREE}}'
        # if 'WINDSIM' in cells[0]:
        #     cells[0] = '\multirow{5}*{\\rotatebox{90}{WINDSIM}}'
        for c_i,c in enumerate(cells):
            c = c.strip()
            if len(c) == 0:
                continue
            
            if c.strip()[0].isdigit():
                print(f"{c.strip()}, {c.strip() == '0.00 (0.00)'}")
                if c.strip() == '0.00 (0.00)':
                    cells[c_i] = '0(0)' 
                val = c.split('(')[0].split('\\')[0]
                val = float(val)
                if (c_i, st) not in max_cols:
                    max_cols[(c_i, st)] = [(l_i, val, st)]
                else:
                    if val > max_cols[(c_i, st)][0][1]:
                        max_cols[(c_i, st)] = [(l_i, val, st)]
                    elif val == max_cols[(c_i, st)][0][1]:
                        max_cols[(c_i, st)].append((l_i, val, st))
        table_latex_lines[l_i] = '&'.join(cells)
    for c_i, st in max_cols:
        for l_i, val, st in max_cols[(c_i, st)]:
            cells = table_latex_lines[l_i].split('&')
             
            new_content = cells[c_i] if not '\\\\' in cells[c_i] else cells[c_i].split('\\\\')[0]
            cells[c_i] = f"\\textbf{{{cells[c_i].strip()}}}" if not '\\\\' in cells[c_i] else f"\\textbf{{{new_content.strip()}}}" + "\\\\"
            table_latex_lines[l_i] = '&'.join(cells)
    
    # print(table_latex_lines)
    colors = sns.color_palette().as_hex()
    # print(colors)
    for l_i,l in enumerate(table_latex_lines):
        add_break = False
        cells = l.split('&')
        if l_i == 3:
            cells[0] = "$N$"
        if 'LRU' in l:
            cells[1] = 'LRU'
            # for c in range(1, len(cells)):
            #     cells[c] = f"\cellcolor{{snso}}{cells[c]}"
        if '&acc&' in l:
            cells[1] = 'Accuracy'
        if ' age ' in l:
            cells[1] = 'FIFO'
        if 'div' in l:
            cells[1] = 'DP'
            # add_break = True
        if 'rA' in l:
            cells[1] = '\\textbf{\\texttt{\#}E}'
            # for c in range(1, len(cells)):
            #     cells[c] = f"\cellcolor{{snsb}}{cells[c]}"
        if 'auc' in l:
            cells[1] = '\\textbf{AAC}'
        if 'score' in l:
            cells[1] = '\\textbf{EP}'
            add_break = True
        
        # if l_i > 1:
        #     cells.insert(11, " ")
        #     cells.insert(8, " ")
        #     cells.insert(5, " ")
        new_line = '&'.join(cells)
        # print(new_line)
        new_latex_lines.append(new_line)
        if add_break:
            new_latex_lines.append("\\\\")
    table_latex = '\n'.join(new_latex_lines)
    
    with open(f'Mem_Manage_Results/{save_name}_plot_bw.txt', 'w') as f:
        print(df_table)
        f.write(table_latex)
            
    # exit()



# print(df.index.get_level_values('sens'))

# df = df.iloc[df.index.get_level_values('sens') != '0.02']
# df = df.unstack(level='cl')
# df = df.unstack(level='mem_manage')

# df = df.unstack(level='poisson')
# df_grouped = df.groupby(level=['ml', 'sys_learner', 'sens', 'poisson']).aggregate([np.mean, np.std])
df_grouped = df.groupby(level=['ml', 'sys_learner', 'sens', 'poisson', 'st', 'cl', 'mem_manage']).aggregate([np.mean, np.std])
# df_grouped = df.groupby(level=['ml', 'sys_learner', 'sens', 'poisson', 'st']).aggregate([np.mean, np.std])
# df_grouped = df_grouped.unstack(level='st')
df_grouped.reset_index(level=['st', 'cl', 'mem_manage'], inplace=True)
print(df_grouped)
# print(df_grouped.columns.levels)
cl_vals = [int (x) for x in df_grouped['cl'].unique()]
mm_vals = list(df_grouped['mem_manage'].unique())
# print(cl_vals)
# print(np.any(np.asarray(cl_vals) == 2))
cl_vals = np.asarray(cl_vals) < 30
cl_vals = np.asarray(cl_vals) < 40
mm_vals = np.asarray(mm_vals) == 'def'
# cl_vals = np.bitwise_or(np.asarray(cl_vals) % 5 == 0, np.asarray(cl_vals) == 1, np.asarray(cl_vals) < 30)
# print(cl_vals)
# exit()
# print(mm_vals)
# df_grouped = df_grouped.iloc[:, np.bitwise_and(cl_vals,mm_vals)]
# print(df_grouped.shape)



st_levels = df_grouped['st'].unique()
print(st_levels)
st_levels = ["RCStreamType-RBF", "RCStreamType-TREE"]

def plot_graph(data, poisson_labels, ax, fig, show_legend = True, st = "Default", m = "accuracy", show_title = True, show_ytitle = True, show_xtitle = True):
    print(data)
    # cmap = plt.get_cmap('viridis')
    # indices = np.linspace(25, cmap.N-25, 7)
    # my_colors = [cmap(int(i)) for i in indices]
    g = sns.lineplot(x='cl', y='mean', err_style = 'bars', ax=ax,
    # linewidth=1, data=data, hue='mem_manage',markers=True, hue_order=['score', 'rA', 'auc', 'age', 'LRU', 'acc', 'div'], dashes=["", (5, 5), (5, 5)], sort=True, style= 'creator', style_order= ['mine', 'theirs', 'base'], units = "poisson", estimator=None)
    # linewidth=1, data=data, markersize=10, hue='mem_manage',markers=True, dashes=False, hue_order=['score', 'rA', 'auc', 'age', 'LRU', 'acc', 'div'], sort=True, style= 'mem_manage', units = "poisson", estimator=None)
    linewidth=3, data=data, hue='mem_manage', hue_order=['rA', 'arf'], sort=True, style= 'creator', style_order= ['mine', 'base', 'theirs'], units = "poisson", estimator=None)
    # ax.set_yticklabels(["{:.2f}".format(t) for t in ax.get_yticks()])

    lines = g.get_lines()
    # for l in lines[3:7]:
    # for l in lines[2:4]:
    #     l.set_dashes((5, 5))

    if show_legend:
        h, l = g.get_legend_handles_labels()
        # h_ours = [x[0] for x in zip(h, l) if x[1] in ['rA', 'auc', 'score']]
        h_ours = [x[0] for x in zip(h, l) if x[1] in ['rA']]
        # h_theirs = [x[0] for x in zip(h, l) if x[1] in ['acc', 'age', 'LRU', 'arf', 'div', 'rcd']]
        # h_theirs = [x[0] for x in zip(h, l) if x[1] in ['acc', 'age', 'LRU', 'div']]
        h_theirs = [x[0] for x in zip(h, l) if x[1] in ['arf']]

        rename_strategies = {'rA': "#E", "auc": "AAC", "score": "EP", "acc": "Acc", "age":"FIFO", "LRU":"LRU", "div":"DP", "arf": "ARF"}
        # l_ours = [rename_strategies[x[1]] for x in zip(h, l) if x[1] in ['rA', 'auc', 'score']]
        l_ours = [rename_strategies[x[1]] for x in zip(h, l) if x[1] in ['rA']]
        # l_theirs = [x[1] for x in zip(h, l) if x[1] in ['acc', 'age', 'LRU', 'arf', 'div', 'rcd']]
        # l_theirs = [rename_strategies[x[1]] for x in zip(h, l) if x[1] in ['acc', 'age', 'LRU', 'div']]
        l_theirs = [rename_strategies[x[1]] for x in zip(h, l) if x[1] in ['arf']]

        for line, label in zip(h_theirs, l_theirs):
            line.set_linestyle("--")
            if label == 'arf':
                line.set_linestyle(":")

        # box = g.get_position()
        # g.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position
        
        # ours = fig.legend(h_ours, l_ours, 
        #                     fancybox=True, ncol=len(h_ours), title = "proposed", bbox_to_anchor=(0, 1.02), loc="lower left", borderaxespad=0.1)
        # theirs = fig.legend(h_theirs, l_theirs, 
        #                     fancybox=True, ncol=len(h_theirs), title = "baselines", bbox_to_anchor=(2, 1.02), loc="lower right", borderaxespad=0.1)
        # fig.add_artist(ours)
    else:
        h_ours, h_theirs, l_ours, l_theirs = None, None, None, None

    l = ax.legend()
    l.remove()

    for p_label in poisson_labels:
        continue
        # ax.text(p_label[0], p_label[1], p_label[2], verticalalignment = 'center')

    if show_title:
        g.set_title(f"{st.split('-')[-1]}")
    if show_xtitle:
        ax.set_xlabel("# Stored Models")

    if show_ytitle:
        ax.set_ylabel(m if m not in metric_replace else metric_replace[m])
    else:
        ax.set_ylabel("")

    # ax.set_yticks(ax.get_yticks()[::2])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.set_yticklabels(["{:.2e}".format(t) for t in ax.get_yticks()])
    
    return h_ours, h_theirs, l_ours, l_theirs 



def plot_graph_minimal(data, poisson_labels, ax, show_legend = True, st = "Default", m = "accuracy"):
    
    print(data)
    g = sns.lineplot(x='cl', y='mean', err_style = 'bars', ax=ax,
    linewidth=3, data=data, hue='mem_manage', hue_order=['rA', 'LRU'], dashes=["", (5, 5), (5, 5)], sort=True, style='creator', style_order= ['mine', 'theirs', 'base'], units = "poisson", estimator=None)
    ax.set_yticklabels(["{:.2e}".format(t) for t in ax.get_yticks()])
    if show_legend:
        h, l = g.get_legend_handles_labels()
        h_ours = [x[0] for x in zip(h, l) if x[1] in ['rA', 'auc', 'score']]
        # h_theirs = [x[0] for x in zip(h, l) if x[1] in ['acc', 'age', 'LRU', 'arf', 'div', 'rcd']]
        h_theirs = [x[0] for x in zip(h, l) if x[1] in ['acc', 'age', 'LRU', 'div']]


        l_ours = [x[1] for x in zip(h, l) if x[1] in ['rA', 'auc', 'score']]
        # l_theirs = [x[1] for x in zip(h, l) if x[1] in ['acc', 'age', 'LRU', 'arf', 'div', 'rcd']]
        l_theirs = [x[1] for x in zip(h, l) if x[1] in ['acc', 'age', 'LRU', 'div']]

        for line, label in zip(h_theirs, l_theirs):
            line.set_linestyle("--")
            if label == 'arf':
                line.set_linestyle(":")

        box = g.get_position()
        g.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position
        
        ours = ax.legend(h_ours, l_ours, 
                            fancybox=True, ncol=1, title = "proposed", bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.)
        theirs = ax.legend(h_theirs, l_theirs, 
                            fancybox=True, ncol=1, title = "baselines", bbox_to_anchor=(1.05, 0.25), loc=2, borderaxespad=0.1)
        ax.add_artist(ours)
    else:
        l = ax.legend()
        l.remove()

    for p_label in poisson_labels:
        continue
        # ax.text(p_label[0], p_label[1], p_label[2], verticalalignment = 'center')

    g.set_title(f"{st.split('-')[-1]}")
    ax.set_xlabel("# Stored Models")
    ax.set_ylabel(m if m not in metric_replace else metric_replace[m])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)

print(len(st_levels))
# num_sts = 3
num_sts = 2
# num_sts = len(st_levels)
m_fig, m_axs = plt.subplots(num_sts, len(metric), figsize = (20,10), sharex='all')
print(df_grouped)
for m_i, m in enumerate(metric):
    overall = None
    poisson_labels = {}
    m_savename = f"{args['file'].split('.')[0]}-{[x.replace(' ', '') for x in [m]]}"

    # for st in st_levels:
    # for st in ['RCStreamType-RBF', 'RCStreamType-TREE', 'RCStreamType-WINDSIM']:
    for st in ['RCStreamType-RBF', 'RCStreamType-TREE']:
        # if st == 'RCStreamType-WINDSIM':
        #     continue
        # st_df = df_grouped.iloc[:, df_grouped.columns.get_level_values('st') == st]
        st_df = df_grouped[df_grouped['st'] == st]
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
                ml_p_df = ml_df[ml_df.index.get_level_values('poisson') == p]
                print(ml_p_df)
                melt = ml_p_df
                # melt = pd.melt(ml_p_df)
                # melt = melt.dropna()
                # print(melt)
                
                
                
                
                # melt.columns = ['res', 'cl', 'mem_manage', 'res_agg', 'value']
                # melt.columns = ['res', 'cl', 'mem_manage', 'res_agg', 'st', 'value']
                if ml == 'arf':
                    melt['mem_manage'] = 'arf'
                if ml == 'rcd':
                    melt.loc[:, 'mem_manage'] = 'rcd'
                
                # melt = melt.loc[melt['res'] == m]
                melt = melt[['st', 'cl', 'mem_manage', m]]
                print(melt)


                if m == 'memory':
                    melt.loc[:, (m, 'mean')] = melt.loc[:, (m, 'mean')] / 1000
                    melt.loc[:, (m, 'std')] = melt.loc[:, (m, 'std')] / 1000
                melt["cl"] = pd.to_numeric(melt["cl"])
                melt = melt.sort_values(by = ['cl'])
                # melt = melt.dropna()
                print(melt)
                
                print(len(melt))
                if len(melt) < 1:
                    continue
                melt['name'] = f"{ml}:{p}:" + melt['mem_manage']
                melt['creator'] = melt['mem_manage'].map(lambda x: "theirs" if x in ['acc', 'age', 'LRU', 'div'] else 'base' if x in ['arf', 'rcd'] else 'mine')
                melt['poisson'] = melt['name'].map(lambda x: p)
                print(melt)
                melt["mean"] = melt.loc[:, (m, "mean")]
                melt["std"] = melt.loc[:, (m, "std")]
                print(melt)
                # melt_std = melt.loc[melt['res_agg'] == 'std']
                melt_std = melt[["st", "cl", "mem_manage", "std"]]
                print(melt_std)
                print(melt)
                # print(melt_std)
                # melt = melt.loc[melt['res_agg'] == 'mean']


                print(melt)
                print(melt_std)
                # melt = pd.merge(melt, melt_std[['cl', 'name', 'value']], on = ['name', 'cl'], how = 'left')
                # melt = melt.rename({'value_x': 'mean', 'value_y': 'std'}, axis='columns')
                print(melt)
                
                # if ml == 'arf':
                #     melt['cl'] = melt['cl'] * 2
                #     melt = melt.loc[melt['cl'] <= 30]
                melt = melt.loc[melt['mem_manage'] != 'rAAuc']
                # melt = melt.loc[melt['mem_manage'] != 'arf']
                
                if len(melt['cl'].unique()) < 1:
                    continue
                final_x = melt['cl'].unique()[-1]
                avg_final_y = melt[melt['cl'] == final_x]['mean'].mean()
                
                if overall is None:
                    overall = melt
                else:
                    overall = overall.append(melt)
                try:
                    if p.isdigit():
                        if st not in poisson_labels:
                            poisson_labels[st] = []
                        poisson_labels[st].append((final_x + 0.25, avg_final_y, f"poisson = {p}"))
                except:
                    pass
    
    print(poisson_labels)
    print(overall)

# Facet

# g = sns.FacetGrid(overall, col = 'st', hue = 'mem_manage')
# g.map(plt.plot, 'cl', 'mean')
# plt.show()

# Lineplot



    unique_stream_types = overall['st'].unique()
    unique_stream_types = ["RCStreamType-RBF", "RCStreamType-TREE"]
    # fig, axs = plt.subplots(1, len(unique_stream_types), sharey=True, figsize = (20,5))
    fig, axs = plt.subplots(1, len(unique_stream_types), figsize = (20,5))
    if len(unique_stream_types) < 2:
        axs = [axs]
    for st_i, st in enumerate(unique_stream_types):
        print(st)
        df_st = overall.loc[overall['st'] == st]
        print(df_st)
        # print(df_st.sample(5))
        try:
            h_ours, h_theirs, l_ours, l_theirs = plot_graph(df_st, poisson_labels[st], axs[st_i], fig, st_i == len(unique_stream_types) - 1, st, m)
        except Exception as e:
            print(e)

    pic_suffix = ""
    # pic_suffix = "ss"
    
    plt.savefig(f"Mem_Manage_Results/{m_savename}{pic_suffix}_bw.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"Mem_Manage_Results/{m_savename}{pic_suffix}H_bw.png", dpi=500, bbox_inches='tight')
    plt.savefig(f"Mem_Manage_Results/{m_savename}{pic_suffix}_bw.pdf", bbox_inches='tight')
    img = plt.imread(f"Mem_Manage_Results/{m_savename}{pic_suffix}H_bw.png")
    plt.imshow(img, cmap="gray")
    plt.imsave( f"Mem_Manage_Results/{m_savename}{pic_suffix}_grey_bw.png", img, cmap="gray")
    figsize = (20, 2)
    fig_leg = plt.figure(figsize=figsize)
    ax_leg = fig_leg.add_subplot(111)


    print(h_ours, h_theirs, l_ours, l_theirs)
    ours = ax_leg.legend(h_ours, l_ours, 
                        fancybox=True, ncol=len(h_ours), title = "Proposed", bbox_to_anchor=(0.3, 0, 0.1, 1), loc="upper left", borderaxespad=0, mode='expand', frameon=False)
                        # fancybox=True, ncol=len(h_ours), markerscale=2, title = "Proposed", bbox_to_anchor=(0, 0, 0.3, 1), loc="upper left", borderaxespad=0, mode='expand', frameon=False)
    ours.get_title().set_fontsize('14')
    theirs = ax_leg.legend(h_theirs, l_theirs, 
                        fancybox=True, ncol=len(h_theirs), title = "Baseline", bbox_to_anchor=(0.6, 0, 0.1, 1), loc="upper left", borderaxespad=0, mode='expand', frameon=False )
                        # fancybox=True, ncol=len(h_theirs), markerscale=2, title = "Baseline", bbox_to_anchor=(0.6, 0, 0.4, 1), loc="upper left", borderaxespad=0, mode='expand', frameon=False )
    theirs.get_title().set_fontsize('14')
    # ours = m_fig.legend(h_ours, l_ours, 
    #                     fancybox=True, ncol=len(h_ours), title = "proposed", bbox_to_anchor=(0, 1.02), loc="lower left", borderaxespad=0.1)
    # theirs = m_fig.legend(h_theirs, l_theirs, 
    #                     fancybox=True, ncol=len(h_theirs), title = "baselines", bbox_to_anchor=(1, 1.02), loc="lower right", borderaxespad=0.1)
    ax_leg.add_artist(ours)
    # add the legend from the previous axes
    # hide the axes frame and the x/y labels
    ax_leg.axis('off')
    # fig_leg.tight_layout()
    fig_leg.savefig(f'Mem_Manage_Results/{save_name}_legend_bw.pdf', bbox_inches='tight')

    for st_i, st in enumerate(unique_stream_types):
        print(st)
        df_st = overall.loc[overall['st'] == st]
        # print(df_st.sample(5))
        if len(metric) == 1:
            legend_vals = plot_graph(df_st, poisson_labels[st], m_axs[st_i], m_fig, st_i == 0 and m_i == 0, st, m, show_title = st_i >= 0, show_ytitle = True, show_xtitle = st_i == len(unique_stream_types))
        else:
            legend_vals = plot_graph(df_st, poisson_labels[st], m_axs[st_i, m_i], m_fig, st_i == 0 and m_i == 0, st, m, show_title = st_i >= 0, show_ytitle = True, show_xtitle = st_i == len(unique_stream_types))
        if not legend_vals[0] is None:
            h_ours, h_theirs, l_ours, l_theirs = legend_vals

    pic_suffix = "all"
    # pic_suffix = "ss"
    




    # for st_i, st in enumerate(unique_stream_types):
    #     fig = plt.figure()
    #     ax = fig.gca()
    #     try:
    #         st_name = unique_stream_types[st_i].split('-')[1]
    #     except:
    #         st_name = unique_stream_types[st_i]

    #     # Save just the portion _inside_ the second axis's boundaries
    #     print(st)
    #     df_st = overall.loc[overall['st'] == st]

    #     try:
    #         plot_graph_minimal(df_st, poisson_labels[st], ax, False, st, m)
    #         fig.savefig(f'Mem_Manage_Results/{m_savename}{st_name}_figure_bw.pdf', bbox_inches='tight')
    #     except Exception as e:
    #         print(e)


# then create a new image
# adjust the figure size as necessary
figsize = (3, 3)
fig_leg = plt.figure(figsize=figsize)
ax_leg = fig_leg.add_subplot(111)


print(h_ours, h_theirs, l_ours, l_theirs)
ours = ax_leg.legend(h_ours, l_ours, 
                    fancybox=True, ncol=len(h_ours), title = "Ours", bbox_to_anchor=(0, 0, 0.4, 1), loc="upper left", borderaxespad=0, mode='expand')
theirs = ax_leg.legend(h_theirs, l_theirs, 
                    fancybox=True, ncol=len(h_theirs), title = "Baselines", bbox_to_anchor=(0.4, 0, 0.6, 1), loc="upper left", borderaxespad=0, mode='expand')
# ours = m_fig.legend(h_ours, l_ours, 
#                     fancybox=True, ncol=len(h_ours), title = "proposed", bbox_to_anchor=(0, 1.02), loc="lower left", borderaxespad=0.1)
# theirs = m_fig.legend(h_theirs, l_theirs, 
#                     fancybox=True, ncol=len(h_theirs), title = "baselines", bbox_to_anchor=(1, 1.02), loc="lower right", borderaxespad=0.1)
ax_leg.add_artist(ours)
# add the legend from the previous axes
# hide the axes frame and the x/y labels
ax_leg.axis('off')
# fig_leg.tight_layout()
fig_leg.savefig(f'Mem_Manage_Results/legend_bw.png')

m_fig.subplots_adjust(wspace = 0.19, hspace = 0.11)
m_fig.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
            mode="expand", borderaxespad=0, ncol=4)
m_fig.tight_layout()
m_fig.savefig(f"Mem_Manage_Results/{save_name}{pic_suffix}_bw.png", dpi=200)
m_fig.savefig(f"Mem_Manage_Results/{save_name}{pic_suffix}H_bw.png", dpi=500)
m_fig.savefig(f"Mem_Manage_Results/{save_name}{pic_suffix}_bw.pdf")
# m_fig.savefig(f"Mem_Manage_Results/{save_name}{pic_suffix}.png", dpi=200, bbox_inches='tight')
# m_fig.savefig(f"Mem_Manage_Results/{save_name}{pic_suffix}H.png", dpi=500, bbox_inches='tight')
# m_fig.savefig(f"Mem_Manage_Results/{save_name}{pic_suffix}.pdf", bbox_inches='tight')
plt.show()





# df = df_g.iloc[(df_g.index.get_level_values('ml') == 'system') & (df_g.index.get_level_values('sys_learner') == 'HN') & (df_g.index.get_level_values('poisson') == '0')]
# # df = df_g.iloc[(df_g.index.get_level_values('ml') == 'system') & (df_g.index.get_level_values('sys_learner') == 'HN')]
# sys_arf = df_g.iloc[(df_g.index.get_level_values('ml') == 'system') & (df_g.index.get_level_values('poisson') == '3')]
# sys_hat = df_g.iloc[(df_g.index.get_level_values('ml') == 'system') & (df_g.index.get_level_values('sys_learner') == 'HAT')]
# print(df)
# print(sys_arf)
# arf = df_g.iloc[(df_g.index.get_level_values('ml') == 'arf') & (df_g.index.get_level_values('sys_learner') == 'pyn')]
# # arf = df_g.iloc[(df_g.index.get_level_values('ml') == 'arf')]
# print(arf)
# rcd = df_orig.iloc[(df_orig.index.get_level_values('ml') == 'rcd')]
# print(rcd)
# melt = pd.melt(df)
# melt_sysarf = pd.melt(sys_arf)
# melt_syshat = pd.melt(sys_hat)
# melt_arf = pd.melt(arf)
# melt_rcd = pd.melt(rcd)
# melt_arf.columns = ['res', 'cl', 'mem_manage', 'res_agg', 'value']
# melt_rcd.columns = ['res', 'cl', 'mem_manage', 'res_agg', 'value']
# melt.columns = ['res', 'cl', 'mem_manage', 'res_agg', 'value']
# melt_sysarf.columns = ['res', 'cl', 'mem_manage', 'res_agg', 'value']
# melt_syshat.columns = ['res', 'cl', 'mem_manage', 'res_agg', 'value']

# melt = melt.loc[melt['res'] == 'accuracy']
# melt = melt.loc[melt['res_agg'] == 'mean']
# melt = melt.sort_values(by = ['cl'])
# melt["cl"] = pd.to_numeric(melt["cl"])

# melt_sysarf = melt_sysarf.loc[melt_sysarf['res'] == 'accuracy']
# melt_sysarf = melt_sysarf.loc[melt_sysarf['res_agg'] == 'mean']
# melt_sysarf = melt_sysarf.sort_values(by = ['cl'])
# melt_sysarf['mem_manage'] += 'sys_arf'
# melt_sysarf["cl"] = pd.to_numeric(melt_sysarf["cl"])

# melt_syshat = melt_syshat.loc[melt_syshat['res'] == 'accuracy']
# melt_syshat = melt_syshat.loc[melt_syshat['res_agg'] == 'mean']
# melt_syshat = melt_syshat.sort_values(by = ['cl'])
# melt_syshat['mem_manage'] = 'sys_hat'
# melt_syshat["cl"] = pd.to_numeric(melt_syshat["cl"])

# melt_arf = melt_arf.loc[melt_arf['res'] == 'accuracy']
# melt_arf = melt_arf.loc[melt_arf['res_agg'] == 'mean']
# melt_arf = melt_arf.loc[melt_arf['mem_manage'] == 'def']
# arf_len = melt_arf.shape[0]
# melt_arf['mem_manage'] = 'arf'
# melt_arf["cl"] = pd.to_numeric(melt_arf["cl"])
# melt_arf = melt_arf.sort_values(by = ['cl'])

# melt_rcd = melt_rcd.loc[melt_rcd['res'] == 'accuracy']
# melt_rcd = melt_rcd.loc[melt_rcd['res_agg'] == 'mean']
# melt_rcd = melt_rcd.loc[melt_rcd['mem_manage'] == 'def']
# melt_rcd['mem_manage'] = 'rcd'
# melt_rcd["cl"] = pd.to_numeric(melt_rcd["cl"])
# melt_rcd = melt_rcd.sort_values(by = ['cl'])
# print(melt)
# print(melt_arf)
# print(melt_rcd)

# # melt = melt.append(melt_arf).append(melt_rcd)
# melt = melt.append(melt_arf)
# melt = melt.append(melt_sysarf)
# melt = melt.append(melt_syshat)

# print(melt)
# sns.lineplot(x='cl', y='value',
#     linewidth=1, data=melt, hue='mem_manage', sort=True)
# plt.show()