import sys, os
import shlex
import glob
import argparse
import random
import pickle

import moaLink
import config
import subprocess


import numpy as np
import pandas as pd
from time import process_time

from time import process_time
from skmultiflow.evaluation import EvaluatePrequential
from scipy.io import arff
from skmultiflow.data import DataStream
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from skmultiflow.meta.adaptive_random_forests import ARFBaseLearner

from evaluate_prequential import evaluate_prequential

def get_votes_for_instance(self, X):
    return self.classifier.get_votes_for_instance(X).copy()

ARFBaseLearner.get_votes_for_instance = get_votes_for_instance





def start_run(options):
    if not os.path.exists(options.experiment_directory):
        print('No Directory')
        return
    name = '-'.join([options.moa_learner, str(options.concept_limit), 'pyn', str(options.seed)])
    print(name)
    datastream_filename = None
    datastream_pickle_filename = None
    fns = glob.glob(os.sep.join([options.experiment_directory, "*.ARFF"]))
    print(fns)
    for fn in fns:
        if fn.split('.')[-1] == 'ARFF':
            actual_fn = fn.split(os.sep)[-1]
            fn_path = os.sep.join(fn.split(os.sep)[:-1])
            print(actual_fn)
            print(fn_path)
            csv_fn = f"{name}.csv"
            csv_full_fn = os.sep.join([fn_path, csv_fn])
            print(csv_full_fn)
            skip_file = False
            if os.path.exists(csv_full_fn):
                if os.path.getsize(csv_full_fn) > 2000:
                    skip_file = True
            if not skip_file:
                datastream_filename = fn
                break
            else:
                print('csv exists')
    if datastream_filename == None:
        print('Not datastream file')
        return
    print(datastream_filename)

    datastream_filename = f"{os.sep.join(datastream_filename.split(os.sep)[:-1])}{os.sep}{datastream_filename.split(os.sep)[-1]}"
    data = arff.loadarff(datastream_filename)
    df = pd.DataFrame(data[0])
    for c in df.columns:
        print(f"Factoizing {c}")
        if pd.api.types.is_string_dtype(df[c]):
            print(pd.factorize(df[c])[0].shape)
            df[c] = pd.factorize(df[c])[0]


    bat_filename = f"{options.experiment_directory}{os.sep}{name}.{'bat' if not options.using_linux else 'sh'}"
    if not os.path.exists(bat_filename) or True:
        num_examples = df.shape[0]
        stream_string = moaLink.get_moa_stream_from_filename(os.sep.join(datastream_filename.split(os.sep)[:-1]), datastream_filename.split(os.sep)[-1])
        moa_string = moaLink.make_moa_command(
            stream_string,
            options.moa_learner,
            options.concept_limit,
            'int',
            num_examples,
            config.report_window_length,
            options.experiment_directory,
            is_bat= not options.using_linux,
            name = name
        )
        moaLink.save_moa_bat(moa_string, bat_filename, not options.using_linux)
        # datastream = None
    t_start = process_time()
    command = f"{bat_filename} {options.moa_location}"
    print(command)
    print(options.moa_learner)
    if options.moa_learner != 'arf' or options.use_moa:
        if options.using_linux:
            
            subprocess.run(['chmod' ,'+x', bat_filename])
            subprocess.run([bat_filename, options.moa_location])
        else:
            subprocess.run(command)
    else:

        # df['y0'] = df['y0'].astype('int64')
        # df["y0"] = df["y0"].astype('category')
        print(df.info())
        datastream = DataStream(df)
        datastream.prepare_for_use()

        print(datastream.target_values)
        learner = AdaptiveRandomForest(n_estimators= int(options.concept_limit))
        avg_memory, max_memory = evaluate_prequential(datastream= datastream, classifier= learner, directory= options.experiment_directory, name = name)
        # right = 0
        # wrong = 0
        # overall_log = []
        # while datastream.has_more_samples():
        #     X,y = datastream.next_sample()
        #     prediction = learner.predict(X)
        #     is_correct = prediction[0] == y[0]
        #     if is_correct:
        #         right += 1
        #     else:
        #         wrong += 1
        #     learner.partial_fit(X, y)
        #     if (right + wrong) > 0 and (right + wrong) % 200 == 0:
        #         overall_log.append((right+ wrong, right / (right + wrong)))
        #         print(f'ex: {right + wrong}, Acc: {right / (right + wrong)}\r', end = "")
        # overall = pd.DataFrame(overall_log, columns = ['ex', 'overall_accuracy'])
        # overall.to_csv(f"{options.experiment_directory}{os.sep}{name}.csv")
        # print("")
        # print(f'Accuracy: {right / (right + wrong)}')
    #fsm, system_stats, concept_chain, ds, stream_examples =  fsmsys.run_fsm(datastream, options, suppress = True, name = name, save_checkpoint=True)
    t_stop = process_time()
    print("")
    print("Elapsed time during the whole program in seconds:", 
                                         t_stop-t_start)
    with open(f"{options.experiment_directory}{os.sep}{name}_timer.txt", "w") as f:
        f.write(f"Elapsed time during the whole program in seconds: {t_stop-t_start}")
    with open(f"{options.experiment_directory}{os.sep}{name}_memory.txt", "w") as f:
        f.write(f"Average: {avg_memory}\n")
        f.write(f"Max: {max_memory}")
    # display.results.stitch_csv(options.experiment_directory, name)

def subdir_run(options):
    base_directory = options.experiment_directory
    list_of_directories = []
    for (dirpath, dirnames, filenames) in os.walk(base_directory):
        for filename in filenames:
            if filename.endswith('.ARFF'): 
                list_of_directories.append(dirpath)
    if not options.reverse:
        list_of_directories.sort(reverse = True)
    concept_limit_step = options.concept_limit
    for subdir in list_of_directories:
        options.experiment_directory = subdir
        if options.concept_limit_range > 0:
            for cl in range(options.concept_limit_start, int(options.concept_limit_range), max(concept_limit_step, 1)):
                options.concept_limit = cl
                start_run(options)
        else:
            start_run(options)

class MoaOptions:
    def __init__(self, concept_limit, moa_location, using_linux, directory, moa_learner):
        self.concept_limit = concept_limit
        self.moa_location = moa_location
        self.using_linux = using_linux
        self.experiment_directory = directory
        self.seed = None
        self.moa_learner = moa_learner

if __name__ == "__main__":
    # Set config params, get commandline params
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--seed", type=int,
        help="Random seed", default=None)
    ap.add_argument("-n", "--noise", type=float,
        help="Noise", default=0)
    ap.add_argument("-cl", "--conceptlimit", type=int,
        help="Concept limit", default=-1)
    ap.add_argument("-clr", "--conceptlimitrange", type=int,
        help="Concept limit", default=-1)
    ap.add_argument("-cls", "--conceptlimitstart", type=int,
        help="Concept limit start", default=1)
    ap.add_argument("-d", "--directory",
        help="tdata generator for stream", default="datastreams")
    ap.add_argument("-m", "--moa",
        help="Moa location", default=f"moa{os.sep}lib{os.sep}")
    ap.add_argument("-ml", "--moalearner",
        help="Moa location", default=f"rcd", choices = ['ht', 'rcd', 'arf', 'obag', 'ecpf'])
    ap.add_argument("-l", "--linux", action="store_true",
        help="running on linux")
    ap.add_argument("-r", "--reverse", action="store_true",
        help="reverse order")
    ap.add_argument("-um", "--usemoa", action="store_true",
        help="running on linux")
    args = vars(ap.parse_args())
    options = MoaOptions(args['conceptlimit'], args['moa'], args['linux'], args['directory'], args['moalearner'])
    options.concept_limit_range = args['conceptlimitrange']
    options.concept_limit_start = args['conceptlimitstart']
    options.reverse = args['reverse']
    options.use_moa = args['usemoa']
    seed = args['seed']
    if seed == None:
        seed = random.randint(0, 10000)
        args['seed'] = seed
    options.seed = seed


    subdir_run(options)


