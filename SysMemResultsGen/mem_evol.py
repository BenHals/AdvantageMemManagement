import sys, os
import glob
from pathlib import Path
import argparse
import random
import pickle
import results_df

import numpy as np
import pandas as pd
import io
import statistics


def subdir_run(directory):
    # fns = glob.glob(f"{directory}{os.sep}*", recursive=True)
    fns = Path(directory).glob('**/*memorytree*')
    evols = []
    totals = []
    for fn in fns:
        print(fn.parts)
        filename = fn.parts[-1]
        if 'rA' not in filename:
            continue
        datastream = fn.parts[-3]
        num_models = int(filename.split('-')[2])
        stream = datastream.split('_')[8]
        print(num_models)
        print(stream)
        with open(fn, 'r') as f:
            lines = f.readlines()
            total = int(lines[-1])
            evol_total = 0
            model_non_evol = 0
            print(total)
            for l in lines:
                if 'evolution' in l:
                    value = int(l.split(': ')[1])
                    evol_total += value

            print(f"Total evol: {evol_total}, is {evol_total / total} of total")
            if evol_total > 0:
                evols.append(evol_total)
            totals.append(total)
    print(f"Mean per model no evol: {(statistics.mean([(t - e) / num_models for t,e in zip(totals, evols)]))}, ({(statistics.stdev([(t - e) / num_models for t,e in zip(totals, evols)]))})")
    print([(t - e) / num_models for t,e in zip(totals, evols)])
    print(evols)
    print(totals)
    print(f"N required to equal extra model: {(statistics.mean([(t) / num_models for t,e in zip(totals, evols)])) / (statistics.mean([(e) / num_models for t,e in zip(totals, evols)]))}")
    print(f"Mean Evol: {statistics.mean(evols)}, mean Evol per model: {statistics.mean(evols) / num_models}({np.std(np.array(evols) / num_models)}), mean prop total: {statistics.mean(evols) / statistics.mean(totals)}, mean pt per model: {statistics.mean(evols) / num_models / statistics.mean(totals)}")


if __name__ == "__main__":
    # Set config params, get commandline params
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory",
        help="tdata generator for stream", default="datastreams")
    args = vars(ap.parse_args())

    subdir_run(args['directory'])