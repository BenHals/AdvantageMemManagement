import sys, os
import glob
import argparse
import random
import pickle
import results_df

import numpy as np
import pandas as pd
import tailer as tl
import io

import pathlib


def start_run(base_directory, subdir, csv_filename, drift_data, save_overall_graph, readall):
    if not os.path.exists(subdir):
        print('No Directory')
        exit()
    name = csv_filename[:-4]

    full_filename = os.sep.join([subdir, csv_filename])
    print(full_filename)

    version = 10
    
    store_filename_all = f"{subdir}{os.sep}{name}_result_{version}_True.pickle"
    store_filename_cropped = f"{subdir}{os.sep}{name}_result_{version}_False.pickle"
    print(f"attempting to read {store_filename_all}")
    try_files = [store_filename_all, store_filename_cropped]
    if readall:
        try_files = [store_filename_all]

    if not save_overall_graph:
        if any([os.path.exists(x) for x in try_files]):
            for x in try_files:
                if os.path.exists(x):
                    try:
                        with open(x, "rb") as f:
                            result = pickle.load(f)
                        print(result)
                        return result
                    except Exception as e:
                        print(e)
                        print("Cant read storage file")
        else:
            print("storage file does not exist")
            
        txt_filename = f"{subdir}{os.sep}{name}_acc_{version}.txt"
        txt_files = [f"{subdir}{os.sep}{name}_acc_True_{version}.txt", f"{subdir}{os.sep}{name}_acc_{version}.txt", f"{subdir}{os.sep}{name}_acc_False_{version}.txt"]

        if readall:
            # txt_files = [f"{subdir}{os.sep}{name}_acc_True.txt", f"{subdir}{os.sep}{name}_acc.txt"]
            txt_files = [f"{subdir}{os.sep}{name}_acc_True_{version}.txt"]

        print(f"attempting to read {txt_filename}")
        if any([os.path.exists(x) for x in txt_files]):
                for x in txt_files:
                    if os.path.exists(x):
                        result = {}
                        try:
                            with open(x, "r") as f:
                                for line in f:
                                    key = line.split(': ')[0]
                                    value = float(line.split(': ')[1].strip())
                                    result[key] = value
                            print(result)
                            if len(result.keys()) > 10:
                                return result
                        except Exception as e:
                            print(e)
                            print("Cant read txt file")
        else:
            print("txt file does not exist")

    try:
        if not readall:
            with open(full_filename, 'r') as f_ob:
                lastLines = tl.tail(f_ob,5)[1:]
                print(lastLines)

            # data = pd.read_csv(full_filename)
            dnames = pd.read_csv(full_filename, nrows=1)
            print('\n'.join(lastLines))
            data = pd.read_csv(io.StringIO('\n'.join(lastLines)), header=None)

            data.columns = dnames.columns
            print(data.head())
        else:
            data = pd.read_csv(full_filename)
            print(data.head())
    except:
        print("no data")
        return None
    if 'ground_truth_concept' not in data.columns and not drift_data is None:
        if 'ex' in data.columns:
            data.rename(columns={'ex': 'example'}, inplace=True)
        data = data.merge(drift_data, on = 'example', how = 'left')
    
    merge_fn = f"{full_filename[:-4]}-merges.pickle"
    print(merge_fn)
    merges_ref = None

    if os.path.exists(merge_fn):
        print('merges found')
        with open(merge_fn, 'rb') as mf:
            merges_ref = pickle.load(mf)

    if 'system_concept' in data.columns and not (merges_ref is None):
        print("following merges")
        print(merges_ref)
        print(data['system_concept'])
        for i,sysc in enumerate(data['system_concept'].values):
            update_value = sysc
            while update_value in merges_ref:
                update_value = merges_ref[update_value]
            if update_value != sysc:
                data.at[i, 'system_concept'] = update_value
        print(data['system_concept'])
    print("calling results_df")
    result = results_df.log_accuracy_from_df(data, name, subdir, merges_ref, readall)
    if save_overall_graph:
        results_df.plot_outerFSM([data], [name], subdir)
        # exit()
        results_df.plot_system_acc_from_df([data], [name], subdir)
        results_df.plot_concepts_from_df(data, name, subdir)
    return result


def subdir_run(base_directory, type, save_overall_graph, produce_table = False, save_name = 'res', group_by = ['ml', 'cl', 'mm_manage'], readall = False, noalt = False):
    list_of_directories = {}
    print(pathlib.Path(base_directory))
    print(pathlib.Path(base_directory).exists())
    num_files_found = 0
    files_processed = 0
    for (dirpath, dirnames, filenames) in os.walk(base_directory):
        csv_files = []
        for filename in filenames:
            if filename.endswith('.csv'):
                csv_files.append(filename)
                num_files_found += 1
        if len(csv_files) > 0:
            list_of_directories[dirpath] = csv_files
    print(list_of_directories)
    # for fn in pathlib.Path(base_directory).glob('**/*.csv'):
    #     if str(fn.parent) not in list_of_directories:
    #         list_of_directories[str(fn.parent)] = []
    #     list_of_directories[str(fn.parent)].append(str(fn.stem))
    list_of_directories = {}
    csv_files = list(pathlib.Path(base_directory).glob('**/*.csv'))
    for res_file in csv_files:
        parent_dir_name = str(res_file.parent)
        if parent_dir_name not in list_of_directories:
            list_of_directories[parent_dir_name] = []
        list_of_directories[parent_dir_name].append(res_file)
        num_files_found += 1


    info_names = ['noise', 'nc', 'hd', 'ed', 'ha', 'ea', 'hp', 'epa', 'st', 'gradual', 'drift_window']
    indexes = []
    values = []
    results = {}
    seen_dir = {}



    print(list_of_directories)
    # exit()
    for subdir in list_of_directories:
        print(subdir)
        parent_dirs = subdir.split(os.sep)
        print(parent_dirs)
        print(list_of_directories[subdir])
        if len(list_of_directories[subdir]) <= 1:
            continue
        # input()
        can_make_table = False
        if produce_table:
            experiment_info = None
            if len(parent_dirs) > 2:
                info_dir = parent_dirs[-2]
                info = tuple(info_dir.split('_'))
                info = info[:len(info_names)]
                print(info)
                drift_window = 0
                if len(parent_dirs) > 3:
                    drift_dir = parent_dirs[-3]
                    print(drift_dir)
                    if drift_dir[-1] == 'w':
                        try:
                            drift_window = int(drift_dir[:-1])
                        except:
                            print("cant_convert to drift window")
                    else:
                        print("no w")
                info = tuple([x for x in info] + [drift_window])
                if len(info) == len(info_names):
                    can_make_table = True

                    experiment_info = dict(zip(info_names, info))
            if experiment_info is None:
                info = (-1, -1, -1, -1, -1, -1, -1, -1, parent_dirs[-1], -1, -1)
                experiment_info = dict(zip(info_names, info))
                can_make_table = True
            print(experiment_info)
            if info not in seen_dir:
                seen_dir[info] = 0
            else:
                seen_dir[info] += 1
        drift_fn = f'{subdir}{os.sep}drift_info.csv'
        if os.path.exists(drift_fn):

            drift_data = pd.read_csv(drift_fn)
            drift_data.columns = ['example', 'ground_truth_concept', 'drift_occured']
            print(drift_data.tail())
            last_col = drift_data.iloc[-1]
            print(last_col)
            
            
            print(last_col)
            drift_data = drift_data.append({"example": last_col['example'] + 1, "ground_truth_concept": last_col['ground_truth_concept'], "drift_occured": last_col['drift_occured']}, ignore_index= True)
            print(drift_data.tail())
        else:
            drift_data = None
        for csv_filename in list_of_directories[subdir]:
            print(f"Files processed {files_processed}/{num_files_found}")
            files_processed += 1
            # print(csv_filename)
            # input()
            if 'drift_info' in str(csv_filename.name):
                continue
            try:
                result = start_run(base_directory, subdir, str(csv_filename.name), drift_data, save_overall_graph, readall)
                print("GOT RESULT")
                if noalt:
                    compat_result = {}
                    for k in result.keys():
                        if not '_alt' in k:
                            compat_result[k] = result[k]
                    result = compat_result
                    # print(result)
                
                if len(result.keys()) <= 1:
                    continue
                # exit()
            except Exception as e:
                print(e)
                # input()
            if result is None:
                continue
            if can_make_table:
                dash_split = str(csv_filename.name).replace('--', '-').split('-')
                print(dash_split)

                run_name = dash_split[0]
                run_noise = 0
                cl = 'def'
                mm = 'def'
                sensitivity = 'def'
                window = 'def'
                sys_learner = 'def'
                poisson = "def"
                optimal_drift = False
                similarity = 'def'
                merge = 'def'
                time = -1
                memory = -1
                merge_similarity = 0.9

                memory_filename = f"{subdir}{os.sep}{str(csv_filename.name)[:-4]}_memory.txt"
                time_filename = f"{subdir}{os.sep}{str(csv_filename.name)[:-4]}_timer.txt"

                print(f"Checking for memory file: {memory_filename}")
                if os.path.exists(memory_filename):
                    with open(memory_filename, 'r') as f:
                        lines = f.readlines()
                        memory = float(lines[1].split(': ')[1])
                else:
                    print("No memory file")
                print(f"Checking for time file: {time_filename}")
                if os.path.exists(time_filename):
                    with open(time_filename, 'r') as f:
                        lines = f.readlines()
                        time = float(lines[0].split(': ')[1])
                else:
                    print("No time file")

                if run_name == 'system' or run_name == 'systemEDDM':
                    run_noise = dash_split[1]
                    cl = dash_split[2].split('.')[0]
                    if 'ARF' in str(csv_filename.name):
                        sys_learner = 'ARF'
                    if 'HAT' in str(csv_filename.name):
                        sys_learner = 'HAT'
                    if 'HATN' in str(csv_filename.name):
                        sys_learner = 'HATN'
                    if 'HN' in str(csv_filename.name):
                        sys_learner = 'HN'
                    if 'NBN' in str(csv_filename.name):
                        sys_learner = 'NBN'



                    if str(csv_filename.name)[-5].isnumeric() and str(csv_filename.name)[-6] == '-':
                        print(str(csv_filename.name))
                        print("not a final csv")
                        continue
                    if len(dash_split) > 3:
                        mm = dash_split[3].split('.')[0]
                    else:
                        mm = 'def'
                    if len(dash_split) > 4:
                        sensitivity = dash_split[4]
                        if 'e' in sensitivity:
                            sensitivity = dash_split[4] + dash_split[5]
                            if len(dash_split) > 6:
                                window = dash_split[6]
                            else:
                                window = 'def'
                        else:
                            if len(dash_split) > 5:
                                window = dash_split[5]
                            else:
                                window = 'def'
                    else:
                        sensitivity = 'def'
                    if len(dash_split) > 8:
                        if len(str(dash_split[8].split('.')[0])) < 3:
                            poisson = str(dash_split[8].split('.')[0])
                    if len(dash_split) > 10:
                        optimal_drift = dash_split[10] == 'True'
                    if len(dash_split) > 11:
                        similarity = dash_split[11]
                    if len(dash_split) > 12:
                        merge = dash_split[12]
                    if len(dash_split) > 13:
                        merge_similarity = '.'.join(dash_split[13].split('.')[:-1])
                    # if len(dash_split) > 13:
                    #     merge = dash_split[13]

                # elif run_name == 'rcd':
                else:
                    if str(csv_filename.name)[-5].isnumeric() and str(csv_filename.name)[-6] == '-':
                        print(str(csv_filename.name))
                        print("not a final csv")
                        continue
                    cl = dash_split[1].split('.')[0]
                    if 'py' in str(csv_filename.name):
                        sys_learner = 'py'
                    if 'pyn' in str(csv_filename.name):
                        sys_learner = 'pyn'
                    if len(dash_split) > 4:
                        run_noise = dash_split[4]
                # else:
                #     cl = 0



                rep = seen_dir[info]
                seed = pathlib.Path(csv_filename).resolve().parent.stem
                print(pathlib.Path(csv_filename))
                print(pathlib.Path(csv_filename).parent)
                print(pathlib.Path(csv_filename).parent.stem)
                extended_names = info_names + ['ml', 'cl', 'mem_manage', 'rep', 'seed', 'sens', 'window', 'sys_learner', 'poisson', 'od', 'sm', 'merge', 'run_noise', 'merge_similarity']
                extended_info = tuple(list(info) + [run_name, cl, mm, rep, seed, sensitivity, window, sys_learner, poisson, optimal_drift, similarity, merge, run_noise, merge_similarity])
                print(list(zip(extended_names, extended_info)))
                # if run_name == 'systemEDDM' or sys_learner == 'NBN':
                #     input()
                for ii, index_val in enumerate(extended_info):
                    if len(indexes) <= ii:
                        indexes.append([])
                    indexes[ii].append(index_val)
                r = []
                if 'Max Recall' in result and 'Precision for Max Recall' in result:
                    if result['Precision for Max Recall'] + result['Max Recall'] == 0:
                        result['F1 GT'] = 0
                    else:
                        result['F1 GT'] = 2 * ((result['Precision for Max Recall'] * result['Max Recall']) / (result['Precision for Max Recall'] + result['Max Recall']))
                if 'MR by System' in result and 'PMR by System' in result:
                    if result['PMR by System'] + result['MR by System'] == 0:
                        result['F1 Sys'] = 0
                    else:
                        result['F1 Sys'] = 2 * ((result['PMR by System'] * result['MR by System']) / (result['PMR by System'] + result['MR by System']))
                # print(result)
                result['time'] = time
                result['memory'] = memory
                for v in result.values():
                    r.append(v)
                if len(values) > 0:
                    num_cols = len(values[0])
                    new_row_cols = len(r)
                    print(num_cols)
                    print(new_row_cols)
                    if new_row_cols != num_cols:
                        print(result)
                        exit()
                values.append(r)
                while extended_info in results:
                    rep += 1
                    extended_info = tuple(list(info) + [run_name, cl, mm, rep, sensitivity, window, sys_learner, poisson, optimal_drift, similarity, merge, run_noise, merge_similarity])

                results[extended_info] = result
                print(np.array(values))
                print(np.array(values).shape)
                #print(np.array(indexes))
                #print(np.array(indexes).shape)
                # print(list(results.keys()))
                # print(extended_names)
                index = pd.MultiIndex.from_tuples(list(results.keys()), names=extended_names)
                # print(np.array(values))
                # print(index)
                # print(list(result.keys()))
                try:
                    df = pd.DataFrame(np.array(values), index=index, columns = list(result.keys()))
                except:
                    del results[extended_info]
                    continue
                df.to_pickle(f'Mem_Manage_Results/{save_name}.pickle')
                #print(df.head())
                # grouped = df.groupby(level=group_by).aggregate([np.mean, np.std])
                # print(grouped.tail()['accuracy'])
                # with open(f'Mem_Manage_Results/{save_name}.txt', 'w') as f:
                #     grouped.to_latex(f)
                
                

    df = pd.DataFrame(np.array(values), index=index, columns = list(result.keys()))
    df.to_pickle(f'Mem_Manage_Results/{save_name}.pickle')
    print(df.head())
if __name__ == "__main__":
    # Set config params, get commandline params
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory",
        help="tdata generator for stream", default="datastreams")
    ap.add_argument("-t", "--type",
        help="learner", default="sys", choices=["sys", "moa"])
    ap.add_argument("-og", "--overallgraph", action="store_true",
        help="save overall graph")
    ap.add_argument("-tb", "--table", action="store_true",
        help="save overall graph")
    ap.add_argument("-ra", "--readall", action="store_true",
        help="save overall graph")
    ap.add_argument("-na", "--noalt", action="store_true",
        help="no alt for compat")
    ap.add_argument("-sn", "--savename", default='res',
        help="save overall graph")
    ap.add_argument("-gb", "--groupby", type=str, nargs="+",
        help="list of concepts")
    args = vars(ap.parse_args())

    subdir_run(args['directory'], args['type'], args['overallgraph'], args['table'], args['savename'], args['groupby'], args['readall'], args['noalt'])


    # run_fsm(datastream, options, suppress = False, display = True, save_stream = False,
    #     fsm = None, system_stats=None, detector = None, stream_examples = None):