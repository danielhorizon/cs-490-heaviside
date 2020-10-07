import pathlib
import concurrent
import glob
import traceback

from inference import test_eval

import tensorflow as tf
from tensorflow.keras import backend as K

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import losses

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sns.set_palette("pastel")

mpl.rcParams['figure.figsize'] = (24, 20)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#MEASURES = ['accuracy', 'precision', 'recall', 'roc_auc', 'pr_auc', 'f1_mean', 'f1_max']
MEASURES = ['accuracy', 'precision', 'recall',
            'roc_auc', 'pr_auc', 'f1_mean', 'f1']
# computed in inference.py via sklearn over all thresholds
INFERENCE_MEASURES = ['f1', 'ap', 'accuracy', 'pr_auc', 'tpr', 'fpr']
#LOSSES_AND_METRICS = ['accuracy', 'f1', 'ap', 'pr_auc', 'tpr', 'fpr']


def current_path(path):
    if not pathlib.Path(path).exists():
        path = 'complete'.join(str(path).split('running'))
    return path


def training_metrics_df(experiments):
    ds_name = list(experiments[0]['histories'].keys())[0]
    print(ds_name)
    loss_name = list(experiments[0]['histories'][ds_name].keys())[0]
    print(loss_name)
    _metrics = list(
        set(experiments[0]['histories'][ds_name][loss_name].keys()) - set('loss'))
    print("Training Metrics DF metrics: {}".format(_metrics))
    metrics = {'train': [], 'val': []}
    metrics['train'] = [m for m in _metrics if not m.startswith('val_')]
    metrics['val'] = [m for m in _metrics if m.startswith('val_')]
    df_dict = {'epoch': [], 'loss': [], 'dataset': [], 'split': []}
    metric_names = set()
    for experiment in experiments:
        for dataset in experiment['histories'].keys():
            for loss in experiment['histories'][dataset].keys():
                for split in ['train', 'val']:
                    metric_len = 0
                    for i, metric in enumerate(metrics[split]):
                        metric_name = "{}_metric".format(
                            metric.split('val_')[-1])
                        metric_names.add(metric_name)
                        if metric_name not in df_dict:
                            df_dict[metric_name] = []
                        df_dict[metric_name].extend(
                            experiment['histories'][dataset][loss][metric])
                        metric_len = len(
                            experiment['histories'][dataset][loss][metric])
                    df_dict['epoch'].extend(np.arange(0, metric_len))
                    df_dict['loss'].extend(np.repeat(loss, metric_len))
                    df_dict['dataset'].extend(np.repeat(dataset, metric_len))
                    df_dict['split'].extend(np.repeat(split, metric_len))
    for metric_name in list(metric_names):
        df_dict[metric_name] = np.array(df_dict[metric_name]).reshape(-1)
    return pd.DataFrame.from_dict(df_dict)


def evaluation_metrics_df(experiments):
    cols = ['dataset', 'loss', 'split', 'metric', 'value']
    df_dict = {}
    for col in cols:
        df_dict[col] = []
    for experiment in experiments:
        if 'results' not in experiment:
            print('MISSING RESULTS:')
            print(experiment.keys())
        for dataset_name in experiment['results'].keys():
            for loss in experiment['results'][dataset_name].keys():
                for split in experiment['results'][dataset_name][loss].keys():
                    # array of metric values across trials
                    for metric in MEASURES:
                        df_dict['loss'].append(loss)
                        df_dict['dataset'].append(dataset_name)
                        df_dict['split'].append(split)
                        df_dict['metric'].append(metric)
                        df_dict['value'].append(
                            experiment['results'][dataset_name][loss][split][metric])
    return pd.DataFrame.from_dict(df_dict)


def output_distribution_df(experiments):
    df_dict = {'dataset': [], 'loss': [], 'split': [], 'prediction': []}
    for experiment in experiments:
        for dataset in experiment['predictions'].keys():
            for loss in experiment['predictions'][dataset].keys():
                for split, v in experiment['predictions'][dataset][loss].items():
                    prediction_len = v.shape[0]
                    df_dict['dataset'].extend(
                        np.repeat(dataset, prediction_len))
                    df_dict['loss'].extend(np.repeat(loss, prediction_len))
                    df_dict['split'].extend(np.repeat(split, prediction_len))
                    df_dict['prediction'].extend(v)

    df_dict['prediction'] = np.array(df_dict['prediction']).reshape(-1)
    return pd.DataFrame.from_dict(df_dict)


def test_score_df(experiments, chart_folder, args):
    pickle_paths = []
    to_run = []
    datasets = {}

    # compute final results across all thresholds
    for experiment in experiments:
        if 'models' not in experiment:
            print('MISSING MODELS:')
            print(experiment.keys())
            continue
        for dataset_name in experiment['models'].keys():
            for loss in experiment['models'][dataset_name].keys():
                model_path = current_path(
                    experiment['models'][dataset_name][loss])
                pickle_path = model_path.replace('.h5', '_test_final.pkl')
                # delete existing pkls if you want to re-run results
                if pathlib.Path(pickle_path).exists():
                    pickle_paths.append(pickle_path)
                    continue
                to_run.append((dataset_name, model_path,
                               loss, pickle_path, args))

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.nprocs) as executor:
        # Start the load operations and mark each future with its URL
        future_call = {executor.submit(
            test_eval, *args): args for args in to_run}
        for future in concurrent.futures.as_completed(future_call):
            args = future_call[future]
            try:
                pickle_paths.append(future.result())
            except Exception as exc:
                print("EXCEPTION: {}".format(traceback.format_exc()))
            else:
                print('{} is complete'.format(args))

    # find the max score per experiment
    maxes_paths = []
    for experiment in experiments:
        experiment_path = pathlib.Path(current_path(experiment['path']))
        max_path = experiment_path.joinpath('max_final.pkl')
        maxes_paths.append(max_path)
        if max_path.exists():
            continue
        maxes = {'dataset': [], 'loss': [],
                 'threshold': [], 'measure': [], 'val': []}
        glb = str(experiment_path.joinpath('*_test_final.pkl'))
        for fp in glob.iglob(glb):
            df = pd.read_pickle(fp)
            # print(df[measure])
            for measure in INFERENCE_MEASURES:
                row = df.loc[df[measure].idxmax()]
                maxes['dataset'].append(row.dataset)
                maxes['loss'].append(row.loss)
                maxes['threshold'].append(row.threshold)
                maxes['measure'].append(measure)
                maxes['val'].append(row[measure])
        print("SAVING {}".format(max_path))
        max_df = pd.DataFrame.from_dict(maxes)
        print(max_df.describe())
        max_df.to_pickle(max_path)
    return {
        'all': pd.concat([pd.read_pickle(fp) for fp in pickle_paths], ignore_index=True),
        'maxes': pd.concat([pd.read_pickle(fp) for fp in maxes_paths], ignore_index=True)
    }


def training(experiments, chart_folder, args):
    df = training_metrics_df(experiments)
    show_metrics = ["{}_metric".format(m) for m in ['loss'] + MEASURES]
    for dataset_name in experiments[0]['histories'].keys():
        for metric in show_metrics:
            plt.figure()
            plt.ylabel(metric)
            output_file = chart_folder.joinpath(
                'training_{}_{}.jpg'.format(dataset_name, metric))
            print("writing: {}".format(output_file))
            sns.lineplot(x="epoch", y=metric, hue='loss', style='split',
                         data=df[(df['dataset'] == dataset_name)])
            if metric == 'loss_metric':
                plt.yscale('log')
            plt.suptitle(' '.join([dataset_name, metric]))
            plt.savefig(output_file)
            plt.close()


def evaluation(experiments, chart_folder, args):
    df = evaluation_metrics_df(experiments)

    for dataset_name in experiments[0]['histories'].keys():
        for split in df['split'].unique():
            output_file = chart_folder.joinpath(
                'evaluation_{}_{}.jpg'.format(dataset_name, split))
            print("writing metrics to: {}".format(output_file))
            plt.figure()
            sns.barplot(x="metric",
                        hue="loss",
                        y="value",
                        data=df[(df['dataset'] == dataset_name) &
                                (df['split'] == split)], capsize=.1
                        ).set(title="{}: {} split".format(dataset_name, split))
            plt.savefig(output_file)
            plt.close()


def output_distribution(experiments, chart_folder, args):
    df = output_distribution_df(experiments)
    split = 'test'
    datasets = df.dataset.unique()
    for dataset in datasets:
        losses = df.loss.unique()
        for i, loss in enumerate(df.loss.unique()):
            #import pdb; pdb.set_trace(); 1
            plt.figure()
            output_file = chart_folder.joinpath(
                'output_distribution_{}_{}_{}.jpg'.format(dataset, split, loss))
            print("writing output_distribution to: {}".format(output_file))
            plt.suptitle("{}, split: {}, loss: {}".format(
                dataset, split, loss))
            plt.hist(df[(df['dataset'] == dataset) & (df['loss'] == loss) & (
                df['split'] == split)].prediction.to_numpy(), log=True)
            plt.savefig(output_file)
            plt.close()


def max_score(experiments, chart_folder, args):
    # max over all thresholds, mean over all experiments
    df = test_score_df(experiments, chart_folder, args)['maxes']
    for measure in INFERENCE_MEASURES:
        plt.figure()
        plt.suptitle('dataset by {} score'.format(measure))
        output_file = chart_folder.joinpath('max_score_{}.jpg'.format(measure))
        print("writing max_scores to: {}".format(output_file))
        sns.barplot(x="dataset", y="val",
                    data=df[df['measure'] == measure], hue="loss", capsize=.1, errwidth=1.0)
        plt.savefig(output_file)
        plt.close()


def mean_score(experiments, chart_folder, args):
    # max over all thresholds, mean over all experiments
    df = test_score_df(experiments, chart_folder, args)['all']
    for measure in INFERENCE_MEASURES:
        plt.figure()
        plt.suptitle('dataset by {} score'.format(measure))
        output_file = chart_folder.joinpath(
            'mean_over_thresholds_score_{}.jpg'.format(measure))
        print("writing mean_scores to: {}".format(output_file))
        sns.barplot(x="dataset", y=measure, data=df,
                    hue="loss", capsize=.1, errwidth=1.0)
        plt.savefig(output_file)
        plt.close()


def balance_over_thresholds(experiments, chart_folder, args):
    ''' based on test eval (sklearn) '''
    # mean over all thresholds over all experiments
    df = test_score_df(experiments, chart_folder, args)['all']
    dfs = []
    for metric in ['f1', 'accuracy', 'precision', 'recall']:
        dfs.append(df[['dataset', 'loss', 'threshold', metric]])
        dfs[-1]['metric'] = metric
        dfs[-1].columns = ['dataset', 'loss', 'threshold', 'val', 'metric']
    plotdf = pd.concat(dfs, ignore_index=True).dropna()
    #plotdf = plotdf[plotdf.loss.isin(plot_losses)]
    for dataset_name in plotdf.dataset.unique():
        print(dataset_name)
        for loss in plotdf.loss.unique():
            plt.figure()
            plt.suptitle("{}: {}".format(dataset_name, loss))
            sns.lineplot(x="threshold", y="val",
                         data=plotdf[(plotdf['dataset'] == dataset_name) & (plotdf['loss'] == loss)], hue="metric")
            output_file = chart_folder.joinpath(
                'balance_over_thresholds_{}_{}.jpg'.format(dataset_name, loss))
            print("writing balance_over_thresholds to: {}".format(output_file))
            plt.savefig(output_file)
            plt.close()


def roc(experiments, chart_folder, args):
    ''' TODO '''
    pass


def pr(experiments, chart_folder, args):
    ''' TODO '''
    pass
