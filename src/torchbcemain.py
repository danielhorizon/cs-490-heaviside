#!/usr/bin/env python3
#
# Nathan Tsoi © 2020
#
# Installation:
#   pip3 install ap_perf pytorch-ignite imbalanced-learn
#
# Running:
# ./src/torchbcemain.py --loss bce --mode train --batch_size 2048
#
# To compare models, first run bce loss. This will save an initial weights that can be loaded when running the other losses
#
#    ./src/torchbcemain.py --loss bce --mode train --batch_size 2048 --experiment TORCH8
#
#  Then run the other losses, passing the --initial_weights [timestamp] flag. The timestamp will be shown in the output folder or tensorboard:
#
#    ./src/torchbcemain.py --loss approx-f1 --mode train --batch_size 2048 --initial_weights 1595761371 --experiment TORCH8
#
#  Note the batch size limitation for the ap-perf loss, learning rate was provided by the author
#
#    ./src/torchbcemain.py --loss ap-perf-f1 --mode train --batch_size 20 --lr 0.0003 --initial_weights 1595761371 --experiment TORCH8
#

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix, Precision, Recall, MetricsLambda
from ignite.contrib.metrics import AveragePrecision, ROC_AUC
from ignite.handlers import ModelCheckpoint, global_step_from_engine, EarlyStopping

from ap_perf import PerformanceMetric, MetricLayer
from ap_perf.metric import CM_Value

from torchconfusion import confusion

import copy
import importlib
import logging
import shutil

DATASETS = [
    'cocktailparty',
    'uci_adult',
    'mammography',
    'kaggle_cc_fraud',
]
LOSSES = [
    'bce',
    'approx-auroc',
    'ap-perf-f1',
    'approx-f1',
    'approx-accuracy',
    #'approx-ap',
    #'auc-roc'
]
EPS = 1e-7

import time

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

def mean_f1_approx_loss_on(device, thresholds=torch.arange(0.1, 1, 0.1)):
    thresholds = thresholds.to(device)
    def loss(pt, gt):
        """Approximate F1:
            - Linear interpolated Heaviside function
            - Harmonic mean of precision and recall
            - Mean over a range of thresholds
        """
        classes = pt.shape[1] if len(pt.shape) == 2 else 1
        mean_f1s = torch.zeros(classes, dtype=torch.float32).to(device)
        # mean over all classes
        for i in range(classes):
            thresholds = torch.arange(0.1, 1, 0.1).to(device)
            tp, fn, fp, _ = confusion(gt, pt[:,i] if classes > 1 else pt, thresholds)
            precision = tp/(tp+fp+EPS)
            recall = tp/(tp+fn+EPS)
            mean_f1s[i] = torch.mean(2 * (precision * recall) / (precision + recall + EPS))
        loss = 1 - mean_f1s.mean()
        return loss
    return loss

def mean_accuracy_approx_loss_on(device, thresholds=torch.arange(0.1, 1, 0.1)):
    thresholds = thresholds.to(device)
    def loss(pt, gt):
        """Approximate Accuracy:
            - Linear interpolated Heaviside function
            - (TP + TN) / (TP + TN + FP + FN)
            - Mean over a range of thresholds
        """
        classes = pt.shape[1] if len(pt.shape) == 2 else 1
        mean_accs = torch.zeros(classes, dtype=torch.float32).to(device)
        # mean over all classes
        for i in range(classes):
            tp, fn, fp, tn = confusion(gt, pt[:,i] if classes > 1 else pt, thresholds)
            mean_accs[i] = torch.mean((tp + tn) / (tp + tn + fp + fn))
        loss = 1 - mean_accs.mean()
        return loss
    return loss

def roc_auc_score(device):
    ''' TODO: Not working correctly, poor results (loss not converging) '''
    def direct_auc_loss(pt, gt):
        """
        '_y_true' and 'y_pred' are tensors, 'gamma' and 'power' are constants
        """
        gamma = 0.2
        power = 3

        gt_bool = gt >= 0.5
        pos = pt[gt_bool]
        neg = pt[~gt_bool]

        _pos= pos.view(-1,1).expand(-1,neg.shape[0]).reshape(-1)
        _neg = neg.repeat(pos.shape[0])
        diff = _pos - _neg - gamma
        masked = diff[diff<0.0]
        return torch.sum(torch.pow(-masked, power))

    return direct_auc_loss

def mean_ap_approx_loss_on(device, linspacing=11):
    ''' TODO: Not working correctly, poor results (loss not converging) '''
    def loss(pt, gt):
        """Approximate AP:
            - Linear interpolated Heaviside function
            - interpolated ap (11 point by default)
            - Mean over a range of thresholds
        """
        classes = pt.shape[1] if len(pt.shape) == 2 else 1
        mean_rhos = torch.zeros(classes, dtype=torch.float32).to(device)
        thresholds = torch.linspace(0, 1, linspacing).to(device)
        # mean over all classes
        for i in range(classes):
            tp, fn, fp, _ = confusion(gt, pt[:,i] if classes > 1 else pt, thresholds)
            #print('tp', tp)
            #print('fn', fn)
            #print('fp', fp)
            pres = tp/(tp+fp+EPS)
            recs = tp/(tp+fn+EPS)
            #print('thresholds', thresholds)
            #print('pres', pres)
            #print('recs', recs)

            ## non-interpolated
            #rhos = []
            #for j in range(len(thresholds)):
            #    #print(f"threshold: {thresholds[j]}")
            #    #print(f"tp: {tp}")
            #    #print(f"fn: {fn}")
            #    #print(f"fp: {fp}")
            #    if j > 0:
            #        rhos.append((recs[j] - recs[j-1])*pres[j])
            #    else:
            #        rhos.append(recs[j]*pres[j])
            #stacked = torch.stack(rhos)
            #print(stacked)
            #mean_rhos[i] = torch.sum(stacked)

            # interpolated
            rhos = []
            for t in thresholds:
                cond = recs >= t
                #print("pres[cond]", pres[cond])
                if cond.any():
                    rhos.append(torch.max(pres[cond]))
            #print("rhos", rhos)
            if len(rhos):
                mean_rhos[i] = torch.sum(torch.stack(rhos))/linspacing
            #print("mean_rhos", mean_rhos)
        loss = 1 - mean_rhos.mean()
        #print('loss', loss)
        return loss
    return loss

def area(x,y):
    ''' area under curve via trapezoidal rule '''
    direction = 1
    # the following is equivalent to: dx = np.diff(x)
    dx = x[1:] - x[:-1]
    if torch.any(dx < 0):
        if torch.all(dx <= 0):
            direction = -1
        else:
            logging.warn("x is neither increasing nor decreasing\nx: {}\ndx: {}.".format(x, dx))
            return 0
    return direction * torch.trapz(y, x)

def mean_auroc_approx_loss_on(device, linspacing=11):
    def loss(pt, gt):
        """Approximate auroc:
            - Linear interpolated Heaviside function
            - roc (11-point approximation)
            - integrate via trapezoidal rule under curve
        """
        classes = pt.shape[1] if len(pt.shape) == 2 else 1
        thresholds = torch.linspace(0, 1, linspacing).to(device)
        areas = []
        # mean over all classes
        for i in range(classes):
            tp, fn, fp, tn = confusion(gt, pt[:,i] if classes > 1 else pt, thresholds)
            fpr = fp/(fp+tn+EPS)
            tpr = tp/(tp+fn+EPS)
            a = area(fpr, tpr)
            if a > 0:
                areas.append(a)
        loss = 1 - torch.stack(areas).mean()
        return loss
    return loss

class Net(nn.Module):
    def __init__(self, input_dim, sigmoid_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid_out = sigmoid_out

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        if self.sigmoid_out:
            x = self.sigmoid(x)
        return x.squeeze()

# metric definition
class Fbeta(PerformanceMetric):
    def __init__(self, beta):
        self.beta = beta

    def define(self, C):
        return ((1 + self.beta ** 2) * C.tp) / ((self.beta ** 2) * C.ap + C.pp)

# metric definition
class AccuracyMetric(PerformanceMetric):
    def define(self, C):
        return (C.tp + C.tn) / C.all

# F1
class F1Score(PerformanceMetric):
    def define(self, C):
        return (2 * C.tp) / (C.ap + C.pp)


def threshold_pred(y_pred, t):
    return (y_pred > t).float()

# creating trainer,evaluator
def thresholded_output_transform(threshold, device):
    def transform(output):
        y_pred, y = output
        return threshold_pred(y_pred, t=torch.tensor([threshold]).to(device)), y
    return transform

def inference_engine(model, device, threshold=0.5):
    def inference_update_with_tta(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            y_pred = threshold_pred(model(x.to(device)).to(device), t=torch.tensor([threshold]).to(device))
            return y_pred, y
    engine = Engine(inference_update_with_tta)
    for name, metric in getmetrics(threshold, device).items():
        metric.attach(engine, name)
    return engine

def inference_over_range(model, device, data_loader, thresholds=np.arange(0.1,1,0.1)):
    res = []
    for threshold in thresholds:
        inferencer = inference_engine(model, device, threshold=threshold)
        res.append(inferencer.run(data_loader))
    return res

def getmetrics(threshold, device):
    precision = Precision(thresholded_output_transform(threshold, device), average=False)
    recall = Recall(thresholded_output_transform(threshold, device), average=False)
    def Fbetaf(r, p, beta):
        return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)).item()
    return {
        'f1': MetricsLambda(Fbetaf, recall, precision, 1),
        'ap': AveragePrecision(thresholded_output_transform(threshold, device)),
        'auroc': ROC_AUC(thresholded_output_transform(threshold, device)),
        'accuracy': Accuracy(thresholded_output_transform(threshold, device)),
    #    'cm': ConfusionMatrix(num_classes=1)
    }

## combine 2 confusion matrices
def add_cm_val(cmv1, cmv2):
    res = CM_Value(np.array([]),np.array([]))
    res.all = cmv1.all + cmv2.all
    res.tp = cmv1.tp + cmv2.tp
    res.ap = cmv1.ap + cmv2.ap
    res.pp = cmv1.pp + cmv2.pp

    res.an = cmv1.an + cmv2.an
    res.pn = cmv1.pn + cmv2.pn

    res.fp = cmv1.fp + cmv2.fp
    res.fn = cmv1.fn + cmv2.fn
    res.tn = cmv1.tn + cmv2.tn

    return res

# compute metric value from cunfusion matrix
def compute_metric_from_cm(metric, C_val):
    # check for special cases
    if metric.special_case_positive:
        if C_val.ap == 0 and C_val.pp == 0:
            return 1.0
        elif C_val.ap == 0:
            return 0.0
        elif C_val.pp == 0:
            return 0.0

    if metric.special_case_negative:
        if C_val.an == 0 and C_val.pn == 0:
            return 1.0
        elif C_val.an == 0:
            return 0.0
        elif C_val.pn == 0:
            return 0.0

    val = metric.metric_expr.compute_value(C_val)
    return val

def model_path(args):
    return '/'.join([args.output, 'running', args.experiment, args.dataset])

def train(args):
    if torch.cuda.is_available():
        print("device = cuda")
        device = f"cuda:{args.gpu}"
    else:
        print("device = cpu")
        device = "cpu"

    # ap-perf only works on cpu
    if args.loss == 'ap-perf-f1':
        print("device = cpu")
        device = 'cpu'

    ds = dataset_from_name(args.dataset)()
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

    # accuracy metric
    accm = AccuracyMetric()
    accm.initialize()

    threshold = 0.5

    # create a model and criterion layer
    sigmoid_out = False
    if args.loss in ['approx-f1', 'approx-accuracy', 'approx-auroc', 'approx-ap', 'auc-roc']:
        sigmoid_out = True
    model = Net(input_dim, sigmoid_out).to(device)

    # set run timestamp or load from args
    now = int(time.time())
    if args.initial_weights:
        now = args.initial_weights

    Path(model_path(args)).mkdir(parents=True, exist_ok=True)
    run_name = f"{args.dataset}-{args.loss}-batch_{args.batch_size}-lr_{args.lr}_{now}"
    log_path = '/'.join([model_path(args), f"{run_name}.log"])
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fm = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fm)
    ch.setFormatter(fm)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logging.info(f"Configured logging to output to: {log_path} and terminal")

    initial_model_file_path = '/'.join([model_path(args), '{}_initial.pth'.format(now)])

    # load or save initial weights
    if args.initial_weights:
        logging.info(f"[{now}] loading {initial_model_file_path}")
        model.load_state_dict(torch.load(initial_model_file_path))
    else:
        # persist the initial weights for future use
        torch.save(model.state_dict(), initial_model_file_path)

    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'ap-perf-f1':
        threshold = 0.0
        criterion = MetricLayer(f1_score).to(device)
    elif args.loss == 'approx-f1':
        criterion = mean_f1_approx_loss_on(device, thresholds=torch.tensor([0.5]))
    elif args.loss == 'approx-accuracy':
        criterion = mean_accuracy_approx_loss_on(device, thresholds=torch.tensor([0.5]))
    elif args.loss == 'approx-auroc':
        criterion = mean_auroc_approx_loss_on(device)
    elif args.loss == 'approx-ap':
        criterion = mean_ap_approx_loss_on(device)
    elif args.loss == 'auc-roc':
        criterion = roc_auc_score(device)
    else:
        raise RuntimeError("Unknown loss {}".format(fargs.loss))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    patience = args.early_stopping_patience

    tensorboard_path = '/'.join([args.output, 'running', args.experiment, 'tensorboard', run_name])
    writer = SummaryWriter(tensorboard_path)

    # early stopping
    early_stopping = False
    best_f1 = None
    best_f1_apperf = 0

    best_test = {
        'now': now,
        'loss': args.loss,
        'accuracy_05_score':0,
        'f1_05_score':0,
        'ap_05_score':0,
        'auroc_05_score':0,
        'accuracy_mean_score': 0,
        'f1_mean_score': 0,
        'ap_mean_score':0,
        'auroc_mean_score':0,
    }

    for epoch in range(args.epochs):
        if early_stopping:
            logging.info("[{}] Early Stopping at Epoch {}/{}".format(now, epoch, args.epochs))
            logging.info("  [val best f1] apperf: {:.4f} | ignite: {:.4f}".format(best_f1_apperf, best_f1))
            break

        data_cm = CM_Value(np.array([]),np.array([]))

        losses = []
        accuracies = []
        f1s = []
        aps = []
        rocs = []

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            model.train()
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            losses.append(loss)
            loss.backward()
            optimizer.step()

            ## check prediction
            model.eval()    # switch to evaluation
            y_pred = model(inputs)
            y_pred_thresh = (y_pred >= threshold).float()
            np_pred = y_pred_thresh.cpu().numpy()
            np_labels = labels.cpu().numpy()
            
            batch_cm = CM_Value(np_pred, np_labels)
            data_cm = add_cm_val(data_cm, batch_cm)
            # sklearn.metrics to tensorboard
            accuracies.append(metrics.accuracy_score(np_labels, np_pred))
            f1s.append(metrics.f1_score(np_labels,np_pred))
            aps.append(metrics.average_precision_score(np_labels,np_pred))
            # undefined if predicting only 1 value
            # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_ranking.py#L224
            if len(np.unique(np_labels)) == 2:
                rocs.append(metrics.roc_auc_score(np_labels,np_pred))

        acc_val = compute_metric_from_cm(accm, data_cm)
        f1_val = compute_metric_from_cm(f1_score, data_cm)

        mloss = np.array(loss.cpu().detach()).mean()
        writer.add_scalar('loss', mloss, epoch)
        writer.add_scalar('train/accuracy', np.array(acc_val).mean(), epoch)
        writer.add_scalar('train/f1', np.array(f1_val).mean(), epoch)
        writer.add_scalar('train/ap', np.array(aps).mean(), epoch)
        writer.add_scalar('train/auroc', np.array(rocs).mean(), epoch)

        logging.info("Train - Epoch ({}): Loss: {:.4f} Accuracy: {:.4f} | F1: {:.4f}".format(epoch, mloss, acc_val, f1_val))

        ### Validation
        model.eval()
        with torch.no_grad():
            data_cm = CM_Value(np.array([]),np.array([]))
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)

                ## prediction
                pred = (output >= 0).float()
                batch_cm = CM_Value(pred.cpu().numpy(), labels.cpu().numpy())
                data_cm = add_cm_val(data_cm, batch_cm)

            acc_val = compute_metric_from_cm(accm, data_cm)
            f1_val_apperf = compute_metric_from_cm(f1_score, data_cm)
            writer.add_scalar('val_ap_perf/accuracy', acc_val, epoch)
            writer.add_scalar('val_ap_perf/f1', f1_val_apperf, epoch)
            if best_f1_apperf < f1_val_apperf:
                best_f1_apperf = f1_val_apperf

            logging.info("Val - Epoch ({}): Accuracy: {:.4f} | F1: {:.4f}".format(epoch, acc_val, f1_val))

            # check early stopping per epoch
            patience -= 1
            if best_f1 is None or best_f1 < f1_val_ignite:
                # save the best model
                model_file_path = '/'.join([model_path(args), '{}_best_model_{}_{}_{}={}.pth'.format(now, epoch, args.dataset, args.loss, f1_val_ignite)])
                torch.save(model, model_file_path)
                logging.info("Saving best model to {}".format(model_file_path))
                best_f1 = f1_val_ignite
                patience = args.early_stopping_patience
                # check test set results
                results = inference_over_range(model, device, test_loader)
                # values correspond to the thresholds: np.arange(0.1,1,0.1), so index 4 has t=0.5
                accuracies = [r.metrics['accuracy'] for r in results]
                test_f1s = [r.metrics['f1'] for r in results]
                aps = [r.metrics['ap'] for r in results]
                aurocs = [r.metrics['auroc'] for r in results]
                # record the best to print at the end
                if best_test['accuracy_05_score'] < accuracies[4]:
                    best_test['accuracy_05_score'] = accuracies[4]
                    best_test['accuracy_05_model_file'] = model_file_path
                if best_test['f1_05_score'] < test_f1s[4]:
                    best_test['f1_05_score'] = test_f1s[4]
                    best_test['f1_05_model_file'] = model_file_path
                if best_test['ap_05_score'] < aps[4]:
                    best_test['ap_05_score'] = aps[4]
                    best_test['ap_05_model_file'] = model_file_path
                if best_test['auroc_05_score'] < aurocs[4]:
                    best_test['auroc_05_score'] = aurocs[4]
                    best_test['auroc_05_model_file'] = model_file_path
                mean_accuracy = np.mean(accuracies)
                mean_f1 = np.mean(test_f1s)
                mean_ap = np.mean(aps)
                mean_auroc = np.mean(aurocs)
                if best_test['accuracy_mean_score'] < mean_accuracy:
                    best_test['accuracy_mean_score'] = mean_accuracy
                    best_test['accuracy_mean_model_file'] = model_file_path
                if best_test['f1_mean_score'] < mean_f1:
                    best_test['f1_mean_score'] = mean_f1
                    best_test['f1_mean_model_file'] = model_file_path
                if best_test['ap_mean_score'] < mean_ap:
                    best_test['ap_mean_score'] = mean_ap
                    best_test['ap_mean_model_file'] = model_file_path
                if best_test['auroc_mean_score'] < mean_auroc:
                    best_test['auroc_mean_score'] = mean_auroc
                    best_test['auroc_mean_model_file'] = model_file_path
                # write to tensorboard
                writer.add_scalar('test/accuracy_05', accuracies[4], epoch)
                writer.add_scalar('test/f1_05', test_f1s[4], epoch)
                writer.add_scalar('test/ap_05', aurocs[4], epoch)
                writer.add_scalar('test/auroc_05', aurocs[4], epoch)
                writer.add_scalar('test/accuracy_mean', mean_accuracy, epoch)
                writer.add_scalar('test/f1_mean', mean_f1, epoch)
                writer.add_scalar('test/ap_mean', mean_auroc, epoch)
                writer.add_scalar('test/auroc_mean', mean_auroc, epoch)
            logging.info(f"[{now}] {args.loss}, patience: {patience}")
            if patience <= 0:
                early_stopping = True

    logging.info(f"{args.experiment} {now}")
    logging.info(best_test)
    pd.DataFrame({k: [v] for k, v in best_test.items()}).to_csv('/'.join([model_path(args), f"{run_name}.csv"]))
    return now


def test(args):
    now = int(time.time())
    if args.initial_weights:
        now = args.initial_weights

    if torch.cuda.is_available():
      device = f"cuda:{args.gpu}"
    else:
      device = "cpu"

    threshold = 0.5
    if args.loss == 'ap-perf-f1':
        threshold = 0.0
    ds = dataset_from_name(args.dataset)()
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

    best_model_path = None
    maxval = None
    globpath = '/'.join([args.output, 'running', args.experiment, dataset_name, args.loss, "{}_best_model_*_{}*.pth".format(now, args.loss)])
    for path in glob.iglob(globpath):
        val = float(path.split('=')[-1].split('.pth')[0])
        print(val)
        if maxval is None or val > maxval:
            maxval = val
            best_model_path = path
    print("loading: {}".format(best_model_path))
    #best_model.load_state_dict(torch.load(best_model_path))
    best_model = torch.load(best_model_path)

    inferencer = inference_engine(model, device, threshold=threshold)

    #ProgressBar(desc="Inference").attach(inferencer)

    result_state = inferencer.run(test_loader)
    print(result_state.metrics)

    return now


def train_for_datasets(args):
    if args.dataset == 'all':
        for dataset in DATASETS:
            _args = copy.deepcopy(args)
            _args.dataset = dataset
            train_for_losses(_args)
        return ts
    else:
        return train_for_losses(args)

def train_for_losses(args, ts=None):
    if args.loss == 'all':
        for loss in LOSSES:
            _args = copy.deepcopy(args)
            _args.loss = loss
            if ts is not None:
                _args.initial_weights = ts
            # params from author
            if loss == 'ap-perf-f1':
                _args.batch_size = 20
                _args.lr = 3e-4
            ts = train(_args)
        return ts
    else:
        return train(args)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--experiment', type=str, help='experiment name')
    parser.add_argument('--initial_weights', type=int, help='to load a prior model initial weights')
    parser.add_argument('--mode', type=str,
            required=True,
            default='train',
            choices=['train', 'test'])

    parser.add_argument('--early_stopping_patience', type=int, default=100, help="early stopping patience")
    parser.add_argument('--batch_size', type=int, default=2048, metavar='N',
                        help='input batch size for training (default: 2048)')
    parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                        help='number of epochs to train (default: 5000)')
    parser.add_argument('--loss', type=str,
            required=True,
            default='bce',
            choices=LOSSES + ['all'])
    parser.add_argument('--dataset', type=str,
            default='mammography',
            choices=DATASETS + ['all'])
    parser.add_argument('--output', type=str, default="experiments",
                        help='output path for experiment data')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                        help='input batch size for training (default: 2048)')
    parser.add_argument('--gpu', type=int, default=0, metavar='N',
                        help='which gpu to use? [0, n]')

    args = parser.parse_args()

    if args.mode == 'train':
        # loop over all datasets and losses, if necessary
        train_for_datasets(args)
    elif args.mode == 'test':
        if args.loss == 'all':
            for loss in LOSSES:
                args.loss = loss
                test(args)
        else:
            test(args)
    else:
        raise RuntimeError("Unknown mode")

if __name__ == '__main__':
    main()

