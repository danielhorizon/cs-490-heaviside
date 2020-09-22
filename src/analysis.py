#!/usr/bin/env python3
import argparse

from setup_paths import *

import dataloader
import plots


def analysis(args):
    experiments, paths = dataloader.load_experiments(args)
    chart_folder = pathlib.Path(paths[0]).parent.parent.joinpath('charts')
    chart_folder.mkdir(exist_ok=True)

    plots.max_score(experiments, chart_folder, args)
    plots.mean_score(experiments, chart_folder, args)
    plots.balance_over_thresholds(experiments, chart_folder, args)
    plots.evaluation(experiments, chart_folder, args)
    plots.training(experiments, chart_folder, args)
    #plots.output_distribution(experiments, chart_folder, args)

def main():
    parser = argparse.ArgumentParser(description='builds charts and stuffs')
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2048, help="batch size")
    parser.add_argument('--nprocs', type=int, default=10, help="number of threads to run")
    args = parser.parse_args()
    analysis(args)

if __name__ == "__main__":
    main()
