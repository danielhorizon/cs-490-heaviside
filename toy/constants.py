import importlib
import sklearn.metrics as skmetrics

REAL_DATASETS = [
    'iris', 
    'cifar10', 
    'mnist'
]

DS_WEIGHTS = {
    'cocktailparty': [0.72, 1.65],
    'uci_adult': [0.66, 2.07],
    'mammography': [0.51, 21.55],
    'kaggle_cc_fraud': [0.50, 290.25],
}
SYN_DATASETS = [
    'synthetic_05',
    'synthetic_33',
]
DATASETS = REAL_DATASETS + SYN_DATASETS
LOSSES = [
    'bce',
    'dice', 'sig-dice',
    'approx-f1',
    'approx-f2',
    'approx-f3',
    'approx-accuracy',
    'approx-f1-sig',
    'approx-f2-sig',
    'approx-f3-sig',
    'approx-accuracy-sig',
    'approx-auroc',
    'approx-auroc-sig',
    'ap-perf-f1',
    'wmw'
]
LOSS_COLLECTIONS = {
    'all': LOSSES,
    'test': ['bce', 'approx-f1'],
    'auroc': ['bce', 'approx-auroc', 'sig-auroc'],
    'batchsizes': ['bce', 'approx-f1', 'approx-f1-sig', 'approx-accuracy', 'approx-accuracy-sig'],
    'image': ['bce', 'approx-f1', 'approx-f1-sig', 'approx-accuracy', 'approx-accuracy-sig']
}
PLUGIN_METRIC = [
    skmetrics.f1_score,
    skmetrics.accuracy_score
]
EPS = 1e-7


def model_path(args):
    return '/'.join([args.output, 'running', args.experiment, args.dataset])


def dataset_from_name(dataset):
    return getattr(importlib.import_module('datasets'), dataset)
