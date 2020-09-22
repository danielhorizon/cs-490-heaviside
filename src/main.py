#!/usr/bin/env python3
from setup_paths import *
import concurrent

import argparse
import copy
import datetime
import logging
import re
import shutil
import traceback

import numpy as np

import losses
import datasets

from experiment import Experiment
from models import Model

nowstring = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def train_and_eval(args):
    # don't load until we're in a forked thread
    import tensorflow as tf
    gpu_devices = tf.config.list_physical_devices('GPU')
    if not len(gpu_devices):
        raise RuntimeError("no GPU in: {}".format(gpu_devices))
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

    experiment = Experiment(args)
    # in a forked process, we set the seed to the current experiment's seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    experiment.log.info("===START===")
    model = Model(experiment)
    model.train_and_evaluate()
    experiment.log.info("===END===")
    return True

def add_seed_and_execute(args):
    _args = copy.deepcopy(args)
    _args.prefix = '{}_{:05d}'.format(nowstring,0)
    _args.seed = np.random.randint(np.iinfo(np.uint32).max)
    try:
        data = train_and_eval(_args)
    except Exception as exc:
        logging.error("{}: {}".format(args.prefix, traceback.format_exc()))
    else:
        logging.info('{} is complete'.format(args))


def parallel(args):
    all_args = []
    for t in range(args.trials):
        _args = copy.deepcopy(args)
        # use the trial number with timestamp to avoid name collision
        _args.prefix = '{}_{:05d}'.format(nowstring,t)
        # set a random seed before forking to avoid time-based seed collision
        _args.seed = np.random.randint(np.iinfo(np.uint32).max)
        all_args.append(_args)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.nprocs) as executor:
        # Start the load operations and mark each future with its URL
        future_call = {executor.submit(train_and_eval, args): args for args in all_args}
        for future in concurrent.futures.as_completed(future_call):
            args = future_call[future]
            try:
                data = future.result()
            except Exception as exc:
                logging.error("{}: {}".format(args.prefix, traceback.format_exc()))
            else:
                logging.info('{} is complete'.format(args))

def main():
    parser = argparse.ArgumentParser(description='run an experiment')
    parser.add_argument('--loss', type=str,
            required=True,
            default='all',
            choices=['all', 'heaviside', 'lm', 'lmp', 'mvs', 'fb', 'reported', 'hs_a_r'] + losses.ALL)
    parser.add_argument('--dataset', type=str,
            required=True,
            default='all',
            choices=['all', 'synthetic', 'real', 'reported'] + datasets.ALL)
    parser.add_argument("--invert_datasets", default=False, action="store_true" , help="add inverted datasets")
    parser.add_argument('--experiment', type=str,
            required=True,
            help="defines the folder for a set of experiments")

    # watch out this deletes any previous experiment with this name in the running folder
    parser.add_argument('--overwrite', default=False, action='store_true', help="use with care: deletes a running experiment with the same name before running this experiment")

    # parallelism
    parser.add_argument('--trials', type=int, default=10, help="number of experiment instances to run")
    parser.add_argument('--nprocs', type=int, default=6, help="number of threads to run")

    # training
    parser.add_argument('--train_verbose', default=False, action='store_true', help="verbose training output")
    parser.add_argument('--early_stopping_patience', type=int, default=100, help="early stopping patience")
    parser.add_argument('--epochs', type=int, default=5000, help="epochs")
    parser.add_argument('--batch_size', type=int, default=2048, help="batch size")
    parser.add_argument('--prefix', type=str,
            default=nowstring, help="prefix to each folder in an experiment, defaults with the time")
    parser.add_argument('--output_folder', default='experiments/running',
            help='path name, relative to the project root, where the experiment folder is placed')
    parser.add_argument("--eval_batch_f1", default=False, action="store_true" , help="evaluate batch f1 scores")
    args = parser.parse_args()

    experiment_path = ROOT_PATH.joinpath(args.output_folder).joinpath(args.experiment)
    if experiment_path.exists():
        if args.overwrite:
            shutil.rmtree(experiment_path)
        else:
            raise RuntimeError("Experiment path '{}', already exists, pass --overwrite to remove old experiment and run again".format(experiment_path))
    experiment_path.mkdir(parents=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s][%(threadName)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(experiment_path.joinpath("experiments.log")),
            logging.StreamHandler()
        ]
    )

    if args.trials <= 1:
        add_seed_and_execute(args)
        # train_and_eval(args)
    else:
        parallel(args)

if __name__ == "__main__":
    main()
