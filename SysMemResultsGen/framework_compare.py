import pickle
import argparse
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
    df = pickle.load(f)

print(df.groupby('ml').aggregate(['mean']))
# exit()

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
    # cl_vals = (np.isin(np.asarray(cl_vals), [5, 15, 25, 35]))
    cl_vals = (np.isin(np.asarray(cl_vals), [5, 20, 35]))
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
    
    with open(f'Mem_Manage_Results/{save_name}_plot.txt', 'w') as f:
        print(df_table)
        f.write(table_latex)
            
    # exit()



df = df.reset_index()
print(df.head())

cl_vals = [int (x) for x in list(df['cl'].unique())]
mm_vals = list(df['mem_manage'].unique())
df['cl'] = df['cl'].astype(int)
# df = df[df['ml'] == 'system']
df['sys-policy'] = df['ml'] + df['mem_manage'] + df['sys_learner']
df = df[(df['ml'] != "system") | (df['mem_manage'] == 'rA')]
print(cl_vals)
print(mm_vals)

st_levels = df['st'].unique()
print(st_levels)
sns.lineplot(x='cl', y='accuracy', data=df, hue='sys-policy')
plt.show()

print(df[df['sys-policy'] == 'systemEDDMrAHN'])
print(df[df['sys-policy'] == 'systemrANBN'])
df_cl = df[df['cl'] == 15]
print(df_cl.groupby(['sys-policy']).aggregate(('mean', 'std', 'count'))['accuracy'])
exit()

def plot_graph(data, poisson_labels, ax, fig, show_legend = True, st = "Default", m = "accuracy", show_title = True, show_ytitle = True, show_xtitle = True):
    print(data)
    g = sns.lineplot(x='cl', y='mean', err_style = 'bars', ax=ax,
    # linewidth=1, data=data, hue='mem_manage', hue_order=['score', 'rA', 'auc', 'age', 'LRU', 'acc', 'div'], sort=True, style= 'creator', style_order= ['mine', 'theirs', 'base'], units = "poisson", estimator=None)
    linewidth=1, data=data, hue='mem_manage', hue_order=['rA', 'arf'], sort=True, style= 'creator', style_order= ['mine', 'theirs', 'base'], units = "poisson", estimator=None)
    ax.set_yticklabels(["{:.2e}".format(t) for t in ax.get_yticks()])

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
    
    g = sns.lineplot(x='cl', y='mean', err_style = 'bars', ax=ax,
    linewidth=3, data=data, hue='mem_manage', hue_order=['rA', 'LRU'], sort=True, style= 'creator', style_order= ['mine', 'theirs', 'base'], units = "poisson", estimator=None)
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
num_sts = 2
# num_sts = len(st_levels)
m_fig, m_axs = plt.subplots(num_sts, len(metric), figsize = (20,10), sharex='all')
print(df_grouped)
for m_i, m in enumerate(metric):
    overall = None
    poisson_labels = {}
    m_savename = f"{args['file'].split('.')[0]}-{[x.replace(' ', '') for x in [m]]}"

    for st in st_levels:
        if st == 'RCStreamType-WINDSIM':
            continue
        st_df = df_grouped.iloc[:, df_grouped.columns.get_level_values('st') == st]
        print(st_df)
        
        ml_levels = st_df.index.get_level_values('ml').unique()
        for ml in ml_levels:
            ml_df = st_df.loc[st_df.index.get_level_values('ml') == ml]
            
                
            print(ml_df)
            poisson_levels = ml_df.index.get_level_values('poisson').unique()

            for p in poisson_levels:
                print(f"st: {st}, ml: {ml}, p: {p}")
                ml_p_df = ml_df.loc[ml_df.index.get_level_values('poisson') == p]

                melt = pd.melt(ml_p_df)
                melt = melt.dropna()
                print(melt)
                input()
                
                
                
                # melt.columns = ['res', 'cl', 'mem_manage', 'res_agg', 'value']
                melt.columns = ['res', 'cl', 'mem_manage', 'res_agg', 'st', 'value']
                if ml == 'arf':
                    melt['mem_manage'] = 'arf'
                if ml == 'rcd':
                    melt['mem_manage'] = 'rcd'
                
                
                
                melt = melt.loc[melt['res'] == m]
                if m == 'memory':
                    melt['value'] = melt['value'] / 1000
                melt["cl"] = pd.to_numeric(melt["cl"])
                melt = melt.sort_values(by = ['cl'])
                melt = melt.dropna()
                print(melt)
                
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
                print(melt_std)
                melt = pd.merge(melt, melt_std[['cl', 'name', 'value']], on = ['name', 'cl'], how = 'left')
                melt = melt.rename({'value_x': 'mean', 'value_y': 'std'}, axis='columns')
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
                
                if p.isdigit():
                    if st not in poisson_labels:
                        poisson_labels[st] = []
                    poisson_labels[st].append((final_x + 0.25, avg_final_y, f"poisson = {p}"))
    # exit()
    print(poisson_labels)
    print(overall)

# Facet

# g = sns.FacetGrid(overall, col = 'st', hue = 'mem_manage')
# g.map(plt.plot, 'cl', 'mean')
# plt.show()

# Lineplot



    unique_stream_types = overall['st'].unique()
    # fig, axs = plt.subplots(1, len(unique_stream_types), sharey=True, figsize = (20,5))
    fig, axs = plt.subplots(1, len(unique_stream_types), figsize = (20,5))
    if len(unique_stream_types) < 2:
        axs = [axs]
    for st_i, st in enumerate(unique_stream_types):
        print(st)
        df_st = overall.loc[overall['st'] == st]
        # print(df_st.sample(5))
        try:
            h_ours, h_theirs, l_ours, l_theirs = plot_graph(df_st, poisson_labels[st], axs[st_i], fig, st_i == len(unique_stream_types) - 1, st, m)
        except:
            print("no")

    pic_suffix = ""
    # pic_suffix = "ss"
    
    plt.savefig(f"Mem_Manage_Results/{m_savename}{pic_suffix}.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"Mem_Manage_Results/{m_savename}{pic_suffix}H.png", dpi=500, bbox_inches='tight')
    plt.savefig(f"Mem_Manage_Results/{m_savename}{pic_suffix}.pdf", bbox_inches='tight')

    figsize = (20, 2)
    fig_leg = plt.figure(figsize=figsize)
    ax_leg = fig_leg.add_subplot(111)


    print(h_ours, h_theirs, l_ours, l_theirs)
    ours = ax_leg.legend(h_ours, l_ours, 
                        fancybox=True, ncol=len(h_ours), title = "Proposed", bbox_to_anchor=(0.3, 0, 0.1, 1), loc="upper left", borderaxespad=0, mode='expand', frameon=False)
    theirs = ax_leg.legend(h_theirs, l_theirs, 
                        fancybox=True, ncol=len(h_theirs), title = "Baseline", bbox_to_anchor=(0.6, 0, 0.1, 1), loc="upper left", borderaxespad=0, mode='expand', frameon=False )
    # ours = m_fig.legend(h_ours, l_ours, 
    #                     fancybox=True, ncol=len(h_ours), title = "proposed", bbox_to_anchor=(0, 1.02), loc="lower left", borderaxespad=0.1)
    # theirs = m_fig.legend(h_theirs, l_theirs, 
    #                     fancybox=True, ncol=len(h_theirs), title = "baselines", bbox_to_anchor=(1, 1.02), loc="lower right", borderaxespad=0.1)
    ax_leg.add_artist(ours)
    # add the legend from the previous axes
    # hide the axes frame and the x/y labels
    ax_leg.axis('off')
    # fig_leg.tight_layout()
    fig_leg.savefig(f'Mem_Manage_Results/{save_name}_legend.pdf', bbox_inches='tight')

    for st_i, st in enumerate(unique_stream_types):
        print(st)
        df_st = overall.loc[overall['st'] == st]
        # print(df_st.sample(5))
        legend_vals = plot_graph(df_st, poisson_labels[st], m_axs[st_i, m_i], m_fig, st_i == 0 and m_i == 0, st, m, show_title = st_i >= 0, show_ytitle = True, show_xtitle = st_i == len(unique_stream_types) - 1)
        if not legend_vals[0] is None:
            h_ours, h_theirs, l_ours, l_theirs = legend_vals

    pic_suffix = "all"
    # pic_suffix = "ss"
    




    for st_i, st in enumerate(unique_stream_types):
        fig = plt.figure()
        ax = fig.gca()
        try:
            st_name = unique_stream_types[st_i].split('-')[1]
        except:
            st_name = unique_stream_types[st_i]

        # Save just the portion _inside_ the second axis's boundaries
        print(st)
        df_st = overall.loc[overall['st'] == st]

        try:
            plot_graph_minimal(df_st, poisson_labels[st], ax, False, st, m)
            fig.savefig(f'Mem_Manage_Results/{m_savename}{st_name}_figure.pdf', bbox_inches='tight')
        except:
            print("plot failed")


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
fig_leg.savefig(f'Mem_Manage_Results/legend.png')

m_fig.subplots_adjust(wspace = 0.19, hspace = 0.11)
m_fig.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
            mode="expand", borderaxespad=0, ncol=4)
m_fig.tight_layout()
m_fig.savefig(f"Mem_Manage_Results/{save_name}{pic_suffix}.png", dpi=200)
m_fig.savefig(f"Mem_Manage_Results/{save_name}{pic_suffix}H.png", dpi=500)
m_fig.savefig(f"Mem_Manage_Results/{save_name}{pic_suffix}.pdf")
# m_fig.savefig(f"Mem_Manage_Results/{save_name}{pic_suffix}.png", dpi=200, bbox_inches='tight')
# m_fig.savefig(f"Mem_Manage_Results/{save_name}{pic_suffix}H.png", dpi=500, bbox_inches='tight')
# m_fig.savefig(f"Mem_Manage_Results/{save_name}{pic_suffix}.pdf", bbox_inches='tight')
plt.show()
