#!/usr/bin/env python3
from setup_paths import *

import numpy as np

import csv
import importlib
import datasets

OUTPUT_DIR = ROOT_PATH.joinpath('modules/AP-examples/data-cv').absolute()

def dataset_from_name(dataset):
    return getattr(importlib.import_module('datasets'), dataset)

def convert_to_ap_perf_format():
    for dataset_name in datasets.REAL:
        # output files
        data_fname = OUTPUT_DIR.joinpath("{}.csv".format(dataset_name))
        trainidx_fname = OUTPUT_DIR.joinpath("{}.train".format(dataset_name))
        testidx_fname = OUTPUT_DIR.joinpath("{}.test".format(dataset_name))

        # get data
        print("Dataset: {}".format(dataset_name))
        data = dataset_from_name(dataset_name)()

        train_idxs = []
        test_idxs = []
        with open(data_fname, 'w') as csvfile:
            writer = csv.writer(csvfile)
            # 1-indexed array in julia
            i = 1
            for split in ['train', 'val', 'test']:
                print("Split: {}".format(split))
                for j in range(len(data[split]['X'])):
                    # label is the last column
                    row = np.concatenate((data[split]['X'][j], [data[split]['y'][j]]))
                    writer.writerow(row)
                    if split in ['train', 'val']:
                        train_idxs.append(i)
                    else:
                        test_idxs.append(i)
                    i += 1
        with open(trainidx_fname, 'w') as f:
            f.write(','.join([str(idx) for idx in train_idxs]))
        with open(testidx_fname, 'w') as f:
            f.write(','.join([str(idx) for idx in test_idxs]))

def main():
    convert_to_ap_perf_format()

if __name__ == '__main__':
    main()
