import sys, os
import argparse
import random
import pickle
import json
from DriftStream.streamGen import RCStreamType
from DriftStream.genConceptChain import generateExperimentConceptChain
from DriftStream.genConceptChain import datastreamOptions
from DriftStream.streamGen import RecurringConceptStream
from DriftStream.streamGen import RecurringConceptGradualStream

import numpy as np
import pandas as pd

class ExperimentOptions:
    def __init__(self, seed, stream_type, directory):
        self.seed = seed
        self.stream_type = stream_type
        self.experiment_directory = directory
        self.batch_size = 1

def makeReuseFolder(experiment_directory):
    if not os.path.exists(experiment_directory):
        print('making directory')
        print(experiment_directory)
        os.makedirs(experiment_directory)
        os.makedirs(f'{experiment_directory}{os.sep}archive')

def get_concepts(gt_concepts, ex_index, num_samples):
    """ Given [(gt_concept, start_i, end_i)...]
        Return the ground truth occuring at a given index."""
    
    gt_concept = None
    for gt_c, s_i, e_i in gt_concepts:
        if s_i <= ex_index < e_i:
            gt_concept = gt_c
            break 
    return (gt_concept)
    
def get_model_drifts(num_samples, datastream):
    detections = np.zeros(num_samples)
    for d_i, d in enumerate(datastream.get_drift_info().keys()):
        if d >= len(detections):
            continue
        detections[d] = 1
    return detections

def get_concept_by_example(num_samples, ground_truth_concepts):
    gt_by_ex = []
    for ex in range(num_samples):
        sample_gt_concept = get_concepts(ground_truth_concepts, ex, num_samples)
        gt_by_ex.append(sample_gt_concept)
    return gt_by_ex

def get_concepts_from_model(concept_chain, num_samples):  
    # Have a dict of {ts: concept}
    # Transform to [(concept, start_ts, end_ts)]
    switch_indexes = list(concept_chain.keys())
    gt_concept = concept_chain[switch_indexes[0]]
    start = switch_indexes[0]
    seen_unique = []
    ground_truth_concepts = []
    for ts_i,ts in enumerate(switch_indexes[1:]):
        end, new_gt_concept = ts, concept_chain[ts]
        if gt_concept not in seen_unique:
            seen_unique.append(gt_concept)
        gt_concept_index = seen_unique.index(gt_concept)
        ground_truth_concepts.append((gt_concept_index, start, end))
        gt_concept, start = new_gt_concept, end
    end = num_samples
    if gt_concept not in seen_unique:
        seen_unique.append(gt_concept)
    gt_concept_index = seen_unique.index(gt_concept)
    ground_truth_concepts.append((gt_concept_index, start, end))

    return get_concept_by_example(num_samples, ground_truth_concepts)


def saveStreamToArff(filename, stream_examples):
    with open(f"{filename}", 'w') as f:
        f.write(f"@RELATION stream\n")
        if len(stream_examples) > 0:
            first_example = stream_examples[0]
            for i, x in enumerate(first_example[0].tolist()[0]):
                values = []
                for row in stream_examples:
                    try:
                        values.append(row[0].tolist()[0][i])
                    except:
                        print(row)
                
                    # values = np.unique(np.array([x[0].tolist()[0][i] for x in stream_examples]))
                values = np.unique(np.array(values))
                if len(values) < 10:
                    # f.write(f"@ATTRIBUTE x{i}  {{{','.join([str(x) for x in values.tolist()])}}}\n")
                    f.write(f"@ATTRIBUTE x{i}  NUMERIC\n")
                else:
                    f.write(f"@ATTRIBUTE x{i}  NUMERIC\n")

            for i, y in enumerate(first_example[1].tolist()):
                values = np.unique(np.array([y[1].tolist()[i] for y in stream_examples]))
                if len(values) < 10:
                    # f.write(f"@ATTRIBUTE y{i}  {{{','.join([str(x) for x in values.tolist()])}}}\n")
                    f.write(f"@ATTRIBUTE y{i}  NUMERIC\n")
                else:
                    f.write(f"@ATTRIBUTE y{i}  NUMERIC\n")
            f.write(f"@DATA\n")
            for l in stream_examples:
                for x in l[0].tolist()[0]:
                    f.write(f"{x},")
                for y in l[1].tolist():
                    f.write(f"{y},")
                f.write(f"\n")


def save_stream(options, ds_options):
    cc, ns, desc = generateExperimentConceptChain(ds_options, options.sequential)
    options.ds_length = ns
    options.concept_chain = cc
    print(desc)

    if ds_options.gradual:
        datastream = RecurringConceptGradualStream(options.stream_type, ns, ds_options.noise, options.concept_chain, window_size = options.window_size, seed = options.seed, desc = desc)
    else:
        datastream = RecurringConceptStream(options.stream_type, ns, ds_options.noise, options.concept_chain, seed = options.seed, desc = desc)
    with open(f"{options.experiment_directory}{os.sep}{ds_options.seed}_concept_chain.pickle", "wb") as f:
        pickle.dump(datastream.concept_chain, f)
    with open(f"{options.experiment_directory}{os.sep}{ds_options.seed}_dsinfo.txt", "w+") as f:
        f.write(json.dumps(options.__dict__, default=lambda o: '<not serializable>'))
        f.write('\n')
        f.write(json.dumps(ds_options.__dict__, default=lambda o: '<not serializable>'))
    ns = datastream.num_samples
    print(datastream.concept_chain)
    stream_examples = []
    update_percent = ns // 1000
    ex = 0
    while datastream.has_more_samples():
        X,y = datastream.next_sample(options.batch_size)
        stream_examples.append((X, y))
        ex += 1
        # print(f"{ex}\r", end = "")
        if ex % update_percent == 0:
            print(f"{ex / update_percent}%\r", end = "")

    gts = get_concepts_from_model(datastream.concept_chain, ns)

    ground_truth = np.array(gts)
    sys_results = {}
    sys_results['ground_truth_concept'] = np.array(gts)
    sys_results['drift_occured'] = get_model_drifts(ns, datastream)
    res_data = pd.DataFrame(data = sys_results)
    res_data.to_csv(f'{options.experiment_directory}{os.sep}drift_info.csv')

    arff_full_filename = f"{options.experiment_directory}{os.sep}{ds_options.seed}.ARFF"
    arff_filename = f"{ds_options.seed}.ARFF"
    saveStreamToArff(arff_full_filename, stream_examples)

if __name__ == '__main__':
        # Set config params, get commandline params
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--seed", type=int,
        help="Random seed", default=None)
    ap.add_argument("-w", "--window", type=int,
        help="window", default=1000)
    ap.add_argument("-g", "--gradual", action="store_true",
        help="set if gradual shift is desired")
    ap.add_argument("-m", "--many", action="store_true",
        help="Generate many, not from options")
    ap.add_argument("-u", "--uniform", action="store_true",
        help="layout concepts sequentially")
    ap.add_argument("-st", "--streamtype",
        help="tdata generator for stream", default="STAGGER")
    ap.add_argument("-d", "--directory",
        help="tdata generator for stream", default="datastreams")
    ap.add_argument("-n", "--noise", type=float,
            help="noise", default=0.0)
    ap.add_argument("-nc", "--numconcepts", type=int,
        help="Number of Concepts", default=25)

    ap.add_argument("-hd", "--harddifficulty", type=int,
        help="Difficulty for a hard concept", default=3)
    ap.add_argument("-ed", "--easydifficulty", type=int,
        help="Difficulty for an easy concept", default=1)
    ap.add_argument("-ha", "--hardappear", type=int,
        help="How many times a hard concept appears", default=15)
    ap.add_argument("-ea", "--easyappear", type=int,
        help="How many times an easy concept appears", default=15)
    ap.add_argument("-hp", "--hardprop", type=float,
        help="Proportion of hard to easy concepts", default=0.5)
    ap.add_argument("-epa", "--examplesperapp", type=int,
        help="How many examples each concept appearence lasts for", default=5000)
    ap.add_argument("-r", "--repeat", type=int,
        help="Number of Concepts", default=1)
    args = vars(ap.parse_args())

    seed = args['seed']
    if seed == None:
        seed = random.randint(0, 10000)
        args['seed'] = seed

    noise = args['noise']
    num_concepts = args['numconcepts']
    st = args['streamtype']
    print(args['many'])
    if args['many']:
        # for st in ['RBF', 'TREE', 'WINDSIM']:
        
            # for noise in [0, 0.05, 0.1, 0.25]:
        for noise in [0, 0.05, 0.1]:
            for st in ['TREE', 'RBF']:
                # for num_concepts in [5, 25, 50, 100]:
                # for num_concepts in [50]:
                for nc in [50]:
                    for d in [1, 2, 3]:
                        
                        for hp in [0, 0.05, 0.1, 0.15]:
                            if st == 'TREE' and d == 1 and hp == 0:
                                continue
                            for r in range(0, 3):
                                seed = random.randint(0, 10000)
                                args['seed'] = seed
                                ds_options = datastreamOptions(noise, num_concepts, args['harddifficulty'] + d, args['easydifficulty'] + d, args['hardappear'],
                                        args['easyappear'], hp, args['examplesperapp'], RCStreamType[st], seed, args['gradual'])
                                experiment_info = ds_options.__dict__.copy()
                                experiment_info.pop('seed')
                                experiment_info = list(experiment_info.values())
                                experiment_name = '_'.join((str(x) for x in experiment_info)).replace('.', '-')
                                experiment_directory = f"{os.getcwd()}{os.sep}{args['directory']}{os.sep}{noise}{os.sep}{experiment_name}{os.sep}{ds_options.seed}"
                                options = ExperimentOptions(seed, ds_options.stream_type, experiment_directory)
                                makeReuseFolder(options.experiment_directory)
                                save_stream(options, ds_options)
    else:
        for r in range(args['repeat']):
            ds_options = datastreamOptions(noise, num_concepts, args['harddifficulty'], args['easydifficulty'], args['hardappear'],
                    args['easyappear'], args['hardprop'], args['examplesperapp'], RCStreamType[st], seed, args['gradual'])
            
            experiment_info = ds_options.__dict__.copy()
            experiment_info.pop('seed')
            experiment_info = list(experiment_info.values())
            experiment_name = '_'.join((str(x) for x in experiment_info)).replace('.', '-')
            experiment_directory = f"{os.getcwd()}{os.sep}{args['directory']}{os.sep}{noise}{os.sep}{experiment_name}{os.sep}{ds_options.seed}"
            options = ExperimentOptions(seed, ds_options.stream_type, experiment_directory)
            options.sequential = args['uniform']
            options.window_size = args['window']
            makeReuseFolder(options.experiment_directory)
            save_stream(options, ds_options)
            seed = random.randint(0, 10000)
            args['seed'] = seed
        
