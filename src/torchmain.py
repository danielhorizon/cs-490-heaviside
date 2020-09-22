#!/usr/bin/env python3
from setup_paths import *

import argparse

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix, Precision, Recall, MetricsLambda
from ignite.handlers import ModelCheckpoint, EarlyStopping

from ap_perf import PerformanceMetric, MetricLayer

import importlib
import shutil

from datasets import ALL as ALL_DATASETS

def dataset_from_name(dataset):
    return getattr(importlib.import_module('datasets'), dataset)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, ds_split):
        self.X = torch.from_numpy(ds_split['X']).float()
        self.y = torch.from_numpy(ds_split['y']).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index, :], self.y[index]

# Clear cuda cache between training/testing
def empty_cuda_cache(engine):
    torch.cuda.empty_cache()
    import gc
    gc.collect()

class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        #output = F.sigmoid(x)
        return x.squeeze()

# metric definition
class Fbeta(PerformanceMetric):
    def __init__(self, beta):
        self.beta = beta

    def define(self, C):
        return ((1 + self.beta ** 2) * C.tp) / ((self.beta ** 2) * C.ap + C.pp)

def threshold(y_pred, t=torch.tensor([0.0])):
    return (y_pred > t).float()

# creating trainer,evaluator
def thresholded_output_transform(output):
    y_pred, y = output
    return threshold(y_pred), y

def getmetrics():
    precision = Precision(thresholded_output_transform, average=False)
    recall = Recall(thresholded_output_transform, average=False)
    def Fbetaf(r, p, beta):
        return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)).item()
    return {
        'f1': MetricsLambda(Fbetaf, recall, precision, 1),
        'accuracy': Accuracy(thresholded_output_transform),
    #    'cm': ConfusionMatrix(num_classes=1)
    }

def train(args):
    if torch.cuda.is_available():
      device = "cuda:0"
    else:
      device = "cpu"

    dataset_name = 'mammography'
    ds = mammography()
    dataparams = {'batch_size': args.batch_size,
                  'shuffle': True,
                  'num_workers': 1}

    trainset = Dataset(ds['train'])
    train_loader = DataLoader(trainset, **dataparams)
    validationset = Dataset(ds['val'])
    val_loader = DataLoader(validationset, **dataparams)
    testset = Dataset(ds['test'])
    test_loader = DataLoader(testset, **dataparams)

    input_dim = ds['train']['X'][0].shape[0]

    # initialize metric
    f1_score = Fbeta(1)
    f1_score.initialize()
    f1_score.enforce_special_case_positive()

    # create a model and criterion layer
    model = Net(input_dim).to(device)
    if args.loss == 'bce':
        # negative log likelihood
        loss_name = 'bce'
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'ap-perf-f1':
        # F1 loss
        loss_name = 'f1'
        criterion = MetricLayer(f1_score).to(device)
    else:
        raise RuntimeError("Unknown loss {}".format(args.loss))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    metrics = getmetrics()
    metrics['loss'] = Loss(criterion)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    training_history = {'f1':[],'accuracy':[],'loss':[]}
    validation_history = {'f1':[],'accuracy':[],'loss':[]}
    last_epoch = []

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    def val_loss_score_function(engine):
        return -engine.state.metrics['loss']

    model_path = '/'.join([args.output, dataset_name, args.loss])
    best_model_handler = ModelCheckpoint(dirname=model_path,
                                         filename_prefix="best",
                                         n_saved=3,
                                         global_step_transform=global_step_from_engine(trainer),
                                         score_name=loss_name,
                                         score_function=val_loss_score_function)
    val_evaluator.add_event_handler(Events.COMPLETED, best_model_handler, {'model': model, })

    handler = EarlyStopping(patience=100, score_function=val_loss_score_function, trainer=trainer)
    val_evaluator.add_event_handler(Events.COMPLETED, handler)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)

    #### Logging ####
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        accuracy = metrics['accuracy']*100
        last_epoch.append(0)
        training_history['f1'].append(metrics['f1'])
        training_history['accuracy'].append(accuracy)
        training_history['loss'].append(metrics['loss'])
        print("Train - Epoch ({}): f1: {:.2f} | Accuracy: {:.2f} | Loss: {:.2f}"
              .format(trainer.state.epoch, metrics['f1'], accuracy, metrics['loss']))

    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        accuracy = metrics['accuracy']*100
        validation_history['f1'].append(metrics['f1'])
        validation_history['accuracy'].append(accuracy)
        validation_history['loss'].append(metrics['loss'])
        print("Val - Epoch ({}): f1: {:.2f} | Accuracy: {:.2f} | Loss: {:.2f}"
              .format(trainer.state.epoch, metrics['f1'], accuracy, metrics['loss']))

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

    # clear cuda cache beteween train/test
    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
    train_evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)
    val_evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)

    # train
    trainer.run(train_loader, max_epochs=args.epochs)

def test(args):
    if torch.cuda.is_available():
      device = "cuda:0"
    else:
      device = "cpu"

    dataset_name = 'mammography'
    ds = mammography()
    dataparams = {'batch_size': args.batch_size,
                  'shuffle': True,
                  'num_workers': 1}

    trainset = Dataset(ds['train'])
    train_loader = DataLoader(trainset, **dataparams)
    validationset = Dataset(ds['val'])
    val_loader = DataLoader(validationset, **dataparams)
    testset = Dataset(ds['test'])
    test_loader = DataLoader(testset, **dataparams)

    input_dim = ds['train']['X'][0].shape[0]

    best_model = Net(input_dim).to(device)
    best_model_path = None
    minval = None
    globpath = '/'.join([args.output, dataset_name, args.loss, "best_model_*_{}*.pth".format(args.loss)])
    for path in glob.iglob(globpath):
        val = float(path.split('=')[-1].split('.')[0])
        if minval is None or val < minval:
            minval = val
            best_model_path = path
    print("loading: {}".format(best_model_path))
    best_model.load_state_dict(torch.load(best_model_path))

    def inference_update_with_tta(engine, batch):
        best_model.eval()
        with torch.no_grad():
            x, y = batch        
            y_pred = threshold(best_model(x))
            return y_pred, y
    inferencer = Engine(inference_update_with_tta)

    for name, metric in getmetrics().items():
        metric.attach(inferencer, name)

    #ProgressBar(desc="Inference").attach(inferencer)

    result_state = inferencer.run(test_loader, max_epochs=1)
    print(result_state.metrics)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--experiment', type=str,
            required=True,
            help="defines the folder for a set of experiments")
    parser.add_argument('--overwrite', default=False, action='store_true',
            help="use with care: deletes a running experiment with the same name before running this experiment")

    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 5000)')
    parser.add_argument('--output_folder', default='torch_models/running',
            help='path name, relative to the project root, where the experiment folder is placed')
    parser.add_argument('--loss', type=str,
            required=True,
            default='all',
            choices=['all', 'f1'])
    parser.add_argument('--dataset', type=str,
            required=True,
            default='all',
            choices=['all', 'synthetic', 'real'] + ALL_DATASETS)

    args = parser.parse_args()

    experiment_path = ROOT_PATH.joinpath(args.output_folder).joinpath(args.experiment)
    if experiment_path.exists():
        if args.overwrite:
            shutil.rmtree(experiment_path)
        else:
            raise RuntimeError("Experiment path '{}', already exists, pass --overwrite to remove old experiment and run again".format(experiment_path))
    experiment_path.mkdir(parents=True)
    args.experiment_path = experiment_path

    train(args)

if __name__ == '__main__':
    main()
