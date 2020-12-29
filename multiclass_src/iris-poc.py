import os
import json
import time
import torch
import click
import logging
import random
import math

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import metrics

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable

# torchconfuson files
from thresh_torchconfusion import *
from gradient_flow import *


EPS = 1e-7
_IRIS_DATA_PATH = "../data/iris.csv"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, ds_split):
        self.X = torch.from_numpy(ds_split['X']).float()
        self.y = torch.from_numpy(ds_split['y']).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index, :], self.y[index]


class Model(nn.Module):
    # http://airccse.org/journal/ijsc/papers/2112ijsc07.pdf
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8620866&tag=1
    def __init__(self, input_features=4, hidden_layer1=50, hidden_layer2=20,
                 output_features=3):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc3 = nn.Linear(hidden_layer2, output_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def record_results(results_path, best_test, output_file):
    # reading in the data from the existing file.
    file_path = "/".join([results_path, output_file])
    if os.path.isfile(file_path):
        with open(file_path, "r+") as f:
            data = json.load(f)
            data.append(best_test)
            f.close()
        with open(file_path, "w") as outfile:
            json.dump(data, outfile)
    ## if the file doesn't eixst:
    else:
        best_test = [best_test]
        with open(file_path, "w") as outfile:
            json.dump(best_test, outfile)


def load_iris(shuffle=True, seed=None):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8620866&tag=1
    raw_df = pd.read_csv(_IRIS_DATA_PATH)
    mappings = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }
    raw_df["species"] = raw_df["species"].apply(lambda x: mappings[x])

    # split and shuffle; shuffle=true will shuffle the elements before the split.
    set_seed(seed)
    train_df, test_df = train_test_split(
        raw_df, test_size=0.20, shuffle=shuffle)
    train_df, val_df = train_test_split(
        train_df, test_size=0.25, shuffle=shuffle)  # 0.25 * 0.8 = 0.2

    train_labels = np.array(train_df.pop("species"))
    val_labels = np.array(val_df.pop("species"))
    test_labels = np.array(test_df.pop("species"))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    # scaling data.
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    # scaling validation and test based on training data.
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    print('iris training labels shape: {}'.format(train_labels.shape))
    print("iris training features.shape: {}".format(train_features.shape))

    print('iris validation labels shape: {}'.format(val_labels.shape))
    print("iris validation features.shape: {}".format(val_features.shape))

    print('iris test labels shape: {}'.format(test_labels.shape))
    print("iris test features.shape: {}".format(test_features.shape))

    return {
        'train': {
            'X': train_features,
            'y': train_labels
        },
        'val': {
            'X': val_features,
            'y': val_labels
        },
        'test': {
            'X': test_features,
            'y': test_labels
        },
    }


def create_loaders(data_splits, batch_size, seed):
    dataparams = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    trainset = Dataset(data_splits['train'])
    validationset = Dataset(data_splits['val'])
    testset = Dataset(data_splits['test'])
    set_seed(seed)
    train_loader = DataLoader(trainset, **dataparams)
    set_seed(seed)
    val_loader = DataLoader(validationset, **dataparams)
    set_seed(seed)
    test_loader = DataLoader(testset, **dataparams)
    return train_loader, val_loader, test_loader


def evaluation_f1_across_thresholds(device, y_labels=None, y_preds=None, thresholds=None):
    classes = len(y_labels[0])
    mean_f1s = torch.zeros(classes, dtype=torch.float32)
    precisions = torch.zeros(classes, dtype=torch.float32)
    recalls = torch.zeros(classes, dtype=torch.float32)
    class_losses = torch.zeros(classes, dtype=torch.float32)

    # y_labels = tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])
    # y_preds = tensor([[0.0981, 0.0968, 0.0977, 0.0869, 0.1180, 0.1081, 0.0972, 0.0919, 0.1003, 0.1050]]

    ## for each class
    for i in range(classes):
        gt_list = torch.Tensor([x[i] for x in y_labels]).to(device)
        pt_list = y_preds[:, i]

        num_thresh = len(thresholds)
        thresh_pr, thresh_re, thresh_f1, thresh_loss = [
            None] * num_thresh, [None] * num_thresh, [None] * num_thresh, [None] * num_thresh
        ## loop across all of the thresholds.
        for j in range(num_thresh):
            # activation, using sklearn to compute metrics.
            pt_list = torch.Tensor(
                [1 if x >= thresholds[j] else 0 for x in pt_list])
            tn, fp, fn, tp = confusion_matrix(y_true=gt_list.cpu().numpy(),
                                              y_pred=pt_list.cpu().numpy(), labels=[0, 1]).ravel()
            # converting to tensors
            tp, fn, fp, tn = torch.tensor([tp]).to(device), torch.tensor([fn]).to(
                device), torch.tensor([fp]).to(device), torch.tensor([tn]).to(device)

            precision = tp/(tp+fp+EPS)
            recall = tp/(tp+fn+EPS)
            temp_f1 = torch.mean(2 * (precision * recall) /
                                 (precision + recall + EPS))

            thresh_pr[j] = precision.detach().item()
            thresh_re[j] = recall.detach().item()
            thresh_f1[j] = temp_f1.detach().item()
            thresh_loss[j] = 1 - thresh_f1[j]

        mean_f1s[i] = np.array(thresh_f1).mean()
        precisions[i] = np.array(thresh_pr).mean()
        recalls[i] = np.array(thresh_re).mean()
        class_losses[i] = np.array(thresh_loss).mean()

    # return class-wise metrics.
    return mean_f1s, precisions, recalls, class_losses


def evaluation_f1(device, y_labels=None, y_preds=None, threshold=None):
    classes = len(y_labels[0])
    mean_f1s = torch.zeros(classes, dtype=torch.float32)
    precisions = torch.zeros(classes, dtype=torch.float32)
    recalls = torch.zeros(classes, dtype=torch.float32)

    '''
    y_labels = tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])
    y_preds = tensor([[0.0981, 0.0968, 0.0977, 0.0869, 0.1180, 0.1081, 0.0972, 0.0919, 0.1003, 0.1050]])
    '''
    for i in range(classes):
        gt_list = torch.Tensor([x[i] for x in y_labels]).to(device)
        pt_list = y_preds[:, i]

        pt_list = torch.Tensor([1 if x >= threshold else 0 for x in pt_list])

        tn, fp, fn, tp = confusion_matrix(y_true=gt_list.cpu().numpy(),
                                          y_pred=pt_list.cpu().numpy(), labels=[0, 1]).ravel()

        # converting to tensors
        tp, fn, fp, tn = torch.tensor([tp]).to(device), torch.tensor([fn]).to(
            device), torch.tensor([fp]).to(device), torch.tensor([tn]).to(device)
        precision = tp/(tp+fp+EPS)
        recall = tp/(tp+fn+EPS)
        temp_f1 = torch.mean(2 * (precision * recall) /
                             (precision + recall + EPS))
        mean_f1s[i] = temp_f1
        precisions[i] = precision
        recalls[i] = recall

    # return class wise f1, and the mean of the f1s.
    return mean_f1s, mean_f1s.mean(), precisions, recalls


def train_iris(data_splits, loss_metric, epochs, seed, run_name, cuda, train_tau, batch_size, patience, output_file):
    using_gpu = False
    if torch.cuda.is_available():
        print("device = cuda :{}".format(type(cuda)))
        using_gpu = True
        if cuda == "0":
            device = "cuda:0"
        elif cuda == "1":
            device = "cuda:1"
        elif cuda == "2":
            device = "cuda:2"
        elif cuda == "3":
            device = "cuda:3"
        else:
            device = "cuda:0"
    else:
        print("device = cpu")
        device = "cpu"

    # setting seeds
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # using DataSet and DataLoader
    train_loader, val_loader, test_loader = create_loaders(
        data_splits, batch_size, seed)

    # setting seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # storing metrics
    train_dxn, test_dxn, valid_dxn = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    best_test = {
        "best-epoch": 0,
        "loss": float('inf'),
        "test_wt_f1_score": 0,
        "best-class-epoch": [0, 0, 0],
        "test_accuracy": 0,
        "learning_rate": 0,
        "imbalanced": False,
        "loss_metric": loss_metric,
        "run_name": run_name,
        "train_dxn": None,
        "test_dxn": None,
        "valid_dxn": None,
        "seed": seed,
        "batch_size": batch_size,
        "model_file_path": None,
        "patience": None
    }

    # initialization
    learning_rate = 0.001
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_test['learning_rate'] = learning_rate
    best_test['patience'] = patience

    if run_name:
        experiment_name = run_name
        tensorboard_path = "/".join(["tensorboard", "iris-poc", experiment_name])
        writer = SummaryWriter(tensorboard_path)

    # criterion
    approx = False
    if loss_metric == "ce":
        criterion = nn.CrossEntropyLoss()
    # evaluating this across thresholds, not for a single threshold.
    elif loss_metric == "approx-f1":
        approx = True
        train_threshold = float(train_tau)
        threshold_tensor = torch.Tensor([train_threshold]).to(device)
        criterion = mean_f1_approx_loss_on(
            device=device, threshold=threshold_tensor)
    else:
        raise RuntimeError("Unknown loss {}".format(loss_metric))

    ## setting patience to be an array
    early_stopping_per_class = [False for i in range(3)]
    patience_classes = [int(patience) for i in range(3)]
    print("patience: {}".format(patience))
    reset_patience = [int(patience) for i in range(3)]

    losses = []
    best_valid_class_losses = [None for i in range(3)]
    for epoch in range(epochs):
        # checking if all classes have been trained thoroughly
        print("patience: {}".format(patience_classes))
        print("reset_patience: {}".format(reset_patience))
        if all(early_stopping_per_class):
            break

        accs, microf1s, macrof1s, wf1s = [], [], [], []
        micro_prs, macro_prs, weighted_prs = [], [], []
        micro_recalls, macro_recalls, weighted_recalls = [], [], []
        class_f1_scores = {0: [], 1: [], 2: []}
        class_precision = {0: [], 1: [], 2: []}
        class_recall = {0: [], 1: [], 2: []}

        if epoch == 0:
            if run_name:
                writer.add_scalar("loss", 0, epoch)
                writer.add_scalar("train/accuracy", 0, epoch)
                writer.add_scalar("train/w-f1", 0, epoch)
                writer.add_scalar("train/micro-f1", 0, epoch)
                writer.add_scalar("train/macro-f1", 0, epoch)
                writer.add_scalar("train/w-recall", 0, epoch)
                writer.add_scalar("train/micro-recall", 0, epoch)
                writer.add_scalar("train/macro-recall", 0, epoch)
                writer.add_scalar("train/w-precision", 0, epoch)
                writer.add_scalar("train/micro-precision", 0, epoch)
                writer.add_scalar("train/macro-precision", 0, epoch)

                # adding per-class f1, precision, and recall
                for i in range(3):
                    title = "train/class-" + str(i) + "-f1"
                    writer.add_scalar(title, 0, epoch)
                    title = "train/class-" + str(i) + "-precision"
                    writer.add_scalar(title, 0, epoch)
                    title = "train/class-" + str(i) + "-recall"
                    writer.add_scalar(title, 0, epoch)

        else:
            tr_class_losses = []
            for batch, (inputs, labels) in enumerate(train_loader):
                # setting into train mode.
                model.train()

                # for class distribution
                labels_list = labels.numpy()
                for label in labels_list:
                    train_dxn[int(label)] += 1

                inputs = inputs.to(device)
                # labels = labels.type(torch.LongTensor).to(device)
                labels = labels.to(device)

                # zero grad
                optimizer.zero_grad()
                output = model(inputs)

                ## if we're doing regular ce
                labels = labels.type(torch.int64)
                if not approx:
                    loss = criterion(output, labels)

                ## if we're doing approx
                else:
                    train_labels = torch.zeros(len(labels), len(output[0])).to(device).scatter_(
                        1, labels.unsqueeze(1), 1.).to(device)
                    output=output.to(device)
                    loss, batch_class_losses = criterion(
                        y_labels=train_labels, y_preds=output)

                losses.append(loss)
                loss.backward()
                optimizer.step()

                ## appending in class based losses
                tr_class_losses.append(batch_class_losses)

                # checking prediction via evaluation (for every batch)
                model.eval()
                y_pred = model(inputs)
                _, train_preds = torch.max(y_pred, 1)

                # storing metrics for each batch
                # accs = array of each batch's accuracy -> averaged at each epoch
                accs.append(accuracy_score(
                    y_true=labels.cpu(), y_pred=train_preds.cpu()))
                microf1s.append(f1_score(y_true=labels.cpu(),
                                         y_pred=train_preds.cpu(), average="micro"))
                macrof1s.append(f1_score(y_true=labels.cpu(),
                                         y_pred=train_preds.cpu(), average="macro"))
                wf1s.append(f1_score(y_true=labels.cpu(),
                                     y_pred=train_preds.cpu(), average="weighted"))

                # precision
                micro_prs.append(precision_score(
                    y_true=labels.cpu(), y_pred=train_preds.cpu(), average="micro"))
                macro_prs.append(precision_score(
                    y_true=labels.cpu(), y_pred=train_preds.cpu(), average="macro"))
                weighted_prs.append(precision_score(
                    y_true=labels.cpu(), y_pred=train_preds.cpu(), average="weighted"))

                # recall
                micro_recalls.append(recall_score(
                    y_true=labels.cpu(), y_pred=train_preds.cpu(), average="micro"))
                macro_recalls.append(recall_score(
                    y_true=labels.cpu(), y_pred=train_preds.cpu(), average="macro"))
                weighted_recalls.append(recall_score(
                    y_true=labels.cpu(), y_pred=train_preds.cpu(), average="weighted"))

                class_f1s = f1_score(y_true=labels.cpu(),
                                     y_pred=train_preds.cpu(), average=None)
                class_re = recall_score(
                    y_true=labels.cpu(), y_pred=train_preds.cpu(), average=None)
                class_pr = precision_score(
                    y_true=labels.cpu(), y_pred=train_preds.cpu(), average=None)

                for i in range(len(class_f1s)):
                    class_f1_scores[i].append(class_f1s[i])
                    class_precision[i].append(class_pr[i])
                    class_recall[i].append(class_re[i])

            # https://github.com/rizalzaf/ap_perf/blob/master/examples/tabular.py
            m_loss = torch.mean(torch.stack(losses)) if using_gpu else np.array(
                [x.item() for x in losses]).mean()
            m_accs = np.array(accs).mean()
            m_weightedf1s = np.array(microf1s).mean()
            m_microf1s = np.array(microf1s).mean()
            m_macrof1s = np.array(macrof1s).mean()
            print("Train - Epoch ({}): | Acc: {:.3f} | W F1: {:.3f} | Micro F1: {:.3f}| Macro F1: {:.3f}".format(
                epoch, m_accs, m_weightedf1s, m_microf1s, m_macrof1s)
            )
            if run_name:
                writer.add_scalar("train/train-loss", m_loss, epoch)
                writer.add_scalar("train/accuracy", m_accs, epoch)
                writer.add_scalar("train/w-f1", m_weightedf1s, epoch)
                writer.add_scalar("train/micro-f1", m_microf1s, epoch)
                writer.add_scalar("train/macro-f1", m_macrof1s, epoch)
                writer.add_scalar("train/w-recall",
                                  np.array(weighted_recalls).mean(), epoch)
                writer.add_scalar("train/micro-recall",
                                  np.array(micro_recalls).mean(), epoch)
                writer.add_scalar("train/macro-recall",
                                  np.array(macro_recalls).mean(), epoch)
                writer.add_scalar("train/w-precision",
                                  np.array(weighted_prs).mean(), epoch)
                writer.add_scalar("train/micro-precision",
                                  np.array(micro_prs).mean(), epoch)
                writer.add_scalar("train/macro-precision",
                                  np.array(macro_prs).mean(), epoch)

                # adding per-class f1, precision, and recall
                for i in range(3):
                    title = "train/class-" + str(i) + "-f1"
                    writer.add_scalar(title, np.array(
                        class_f1_scores[i]).mean(), epoch)
                    title = "train/class-" + str(i) + "-precision"
                    writer.add_scalar(title, np.array(
                        class_precision[i]).mean(), epoch)
                    title = "train/class-" + str(i) + "-recall"
                    writer.add_scalar(title, np.array(
                        class_recall[i]).mean(), epoch)

        ## purely for tensorboard logging
        ## this is looking at class-based train losses 
        model.eval()
        with torch.no_grad():
            tau_thresholds = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
            for tau in tau_thresholds:
                # go through all the thresholds, and test them out again.
                tr_preds, tr_labels = [], []
                for i, (inputs, labels) in enumerate(train_loader):
                    # stacking onto tensors.
                    inputs, labels = inputs.to(device), labels.to(device)

                    # passing it through our finalized model.
                    output = model(inputs)
                    labels = labels.type(torch.int64)
                    labels = torch.zeros(len(labels), len(output[0])).to(device).scatter_(
                        1, labels.unsqueeze(1), 1.).to(device)

                    pred_arr = output.detach().cpu().numpy()
                    label_arr = labels.detach().cpu().numpy()

                    # appending results.
                    tr_preds.append(pred_arr)
                    tr_labels.append(label_arr)

                tr_preds = torch.tensor(tr_preds[0])
                tr_labels = torch.tensor(tr_labels[0])

                tr_class_f1s, tr_mean_f1, _, _ = evaluation_f1(device=device, y_labels=tr_labels, y_preds=tr_preds, threshold=tau)

                ## writing out results to tensorboard
                title = "train/mean-f1-{}".format(tau)
                writer.add_scalar(title, tr_mean_f1, epoch)

                for i in range(3):
                    title = "train/class-" + str(i) + "-loss-{}".format(tau)
                    writer.add_scalar(title, 1-tr_class_f1s[i], epoch)

        # ---------- test ----------
        model.eval()
        test_losses, test_class_losses = [], []
        test_preds, test_labels = np.array([]), np.array([])
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                labels_list = labels.cpu().numpy()
                for label in labels_list:
                    test_dxn[int(label)] += 1

                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)
                _, predicted = torch.max(output, 1)

                pred_arr = predicted.cpu().numpy()
                label_arr = labels.cpu().numpy()

                test_labels = np.concatenate([test_labels, label_arr])
                test_preds = np.concatenate([test_preds, pred_arr])

                labels = labels.type(torch.int64)
                if approx:
                    trans_labels = torch.zeros(len(labels), len(output[0])).to(device).scatter_(
                        1, labels.unsqueeze(1), 1.).to(device)
                    output = output.to(device)

                    batch_test_loss, batch_test_class_losses = criterion(
                        y_labels=trans_labels, y_preds=output)
                else:
                    batch_test_loss = criterion(output, labels)

            # adding in test loss
            test_losses.append(batch_test_loss.detach().cpu().numpy())
            test_class_losses.append(batch_test_class_losses)
            test_acc = accuracy_score(y_true=test_labels, y_pred=test_preds)
            test_f1_micro = f1_score(
                y_true=test_labels, y_pred=test_preds, average='micro')
            test_f1_macro = f1_score(
                y_true=test_labels, y_pred=test_preds, average='macro')
            test_f1_weighted = f1_score(
                y_true=test_labels, y_pred=test_preds, average='weighted')
            test_class_f1s = f1_score(
                y_true=test_labels, y_pred=test_preds, average=None)
            test_class_prs = precision_score(
                y_true=test_labels, y_pred=test_preds, average=None)
            test_class_rec = recall_score(
                y_true=test_labels, y_pred=test_preds, average=None)

            test_loss = np.mean(test_losses)
            if run_name:
                writer.add_scalar("test/test-loss", test_loss, epoch)
                writer.add_scalar("test/accuracy", test_acc, epoch)
                writer.add_scalar("test/micro-f1", test_f1_micro, epoch)
                writer.add_scalar("test/macro-f1", test_f1_macro, epoch)
                writer.add_scalar("test/w-f1", test_f1_weighted, epoch)
                # adding per-class f1, precision, and recall
                for i in range(3):
                    title = "test/class-" + str(i) + "-f1"
                    writer.add_scalar(title, np.array(
                        test_class_f1s[i]).mean(), epoch)
                    title = "test/class-" + str(i) + "-precision"
                    writer.add_scalar(title, np.array(
                        test_class_prs[i]).mean(), epoch)
                    title = "test/class-" + str(i) + "-recall"
                    writer.add_scalar(title, np.array(
                        test_class_rec[i]).mean(), epoch)

            if epoch != 0:
                if best_test['loss'] > m_loss:
                    best_test['loss'] = m_loss
                    best_test['best-epoch'] = epoch
                if best_test['test_wt_f1_score'] < test_f1_weighted:
                    best_test['test_wt_f1_score'] = test_f1_weighted
                if best_test['test_accuracy'] < test_acc:
                    best_test['test_accuracy'] = test_acc

            print("Test - Epoch ({}): | Loss: {:.4f} | Acc: {:.3f} | W F1: {:.3f} | Micro F1: {:.3f} | Macro F1: {:.3f}".format(
                epoch, test_loss, test_acc, test_f1_weighted, test_f1_micro, test_f1_macro)
            )

        # ---------- validation ----------
        # Calculate metrics after going through all the batches
        model.eval()
        with torch.no_grad():
            ## looping through every single batch.
            eval_preds, eval_labels = [], []
            for i, (inputs, labels) in enumerate(val_loader):
                # logging validation distribution
                labels_list = labels.numpy()
                for label in labels_list:
                    valid_dxn[int(label)] += 1

                # stacking onto tensors.
                inputs, labels = inputs.to(device), labels.to(device)

                # passing it through our finalized model.
                output = model(inputs)
                labels = labels.type(torch.int64)
                labels = torch.zeros(len(labels), len(output[0])).to(device).scatter_(
                    1, labels.unsqueeze(1), 1.).to(device)

                # appending results.
                pred_arr = output.detach().cpu().numpy()
                label_arr = labels.detach().cpu().numpy()
                eval_preds.append(pred_arr)
                eval_labels.append(label_arr)

            eval_preds = torch.tensor(eval_preds[0])
            eval_labels = torch.tensor(eval_labels[0])

            ## evaluating across a range of thresholds...
            eval_class_f1s, eval_class_prs, eval_class_recs, eval_class_losses = evaluation_f1_across_thresholds(
                device=device, y_labels=eval_labels, y_preds=eval_preds, thresholds=torch.arange(0.1, 1, 0.1))

            valid_loss = np.array(eval_class_losses).mean()
            valid_mean_f1 = np.array(eval_class_f1s).mean()

            # add in per-class metrics
            if run_name:
                writer.add_scalar("val/valid-loss", valid_loss, epoch)
                writer.add_scalar("val/f1", valid_mean_f1, epoch)

                # adding per-class f1, precision, and recall
                for i in range(3):
                    title = "val/class-" + str(i) + "-f1"  # logging f1
                    writer.add_scalar(title, eval_class_f1s[i], epoch)
                    title = "val/class-" + \
                        str(i) + "-precision"  # logging precision
                    writer.add_scalar(title, eval_class_prs[i], epoch)
                    title = "val/class-" + str(i) + "-recall"  # logging recall
                    writer.add_scalar(title, eval_class_recs[i], epoch)
                    title = "val/class-" + str(i) + "-loss"  # logging losses
                    writer.add_scalar(title, eval_class_losses[i], epoch)

            print("Val - Epoch ({}): | Loss: {:.4f} | Mean F1: {:.4f} \n".format(
                epoch, valid_loss, valid_mean_f1)
            )

            ## adding in class_losses, and checking for early stopping here
            for i in range(3):
                # if the loss is less (per class), reset that class's patience
                if (best_valid_class_losses[i] == None) or (eval_class_losses[i] < best_valid_class_losses[i]):
                    if best_valid_class_losses[i] != None:
                        print("Class {}: Valid loss decreased {:.5f} -> {:.5f}! Resetting patience to: {}".format(
                            i + 1, best_valid_class_losses[i], eval_class_losses[i], patience))

                    ## saving model for that class, only if it hasn't hit negative patience
                    if early_stopping_per_class[i] == False:
                        today_date = time.strftime('%Y%m%d')
                        model_file_path = "/".join(["/app/timeseries/multiclass_src/models/iris-poc",
                                                    '{}-class-{}-best-model-{}.pth'.format(
                                                        today_date, i+1, run_name
                                                    )])
                        torch.save(model, model_file_path)

                    ## setting the new best loss
                    best_valid_class_losses[i] = eval_class_losses[i]
                    patience_classes[i] = reset_patience[i]
                else:
                    patience_classes[i] -= 1
                    ## storing values into best run json
                    if patience_classes[i] <= 0:
                        early_stopping_per_class[i] = True
                        best_test['best-class-epoch'][i] = epoch


    # ----- FINAL EVALUATION STEP, USING FULLY TRAINED MODEL -----
    print("--- Finished Training - Entering Final Evaluation Step\n")
    # saving the model.
    model_file_path = "/".join(["/app/timeseries/multiclass_src/models/iris-poc",
                                '{}-overfit-model-{}.pth'.format(
                                    time.strftime('%Y%m%d'), run_name
                                )])
    torch.save(model, model_file_path)

    # ----- recording results in a json.
    if torch.is_tensor(best_test['loss']):
        best_test['loss'] = best_test['loss'].item()
    if torch.is_tensor(best_test['test_wt_f1_score']):
        best_test['test_wt_f1_score'] = best_test['test_wt_f1_score'].item()
    best_test['loss'] = round(best_test['loss'], 5)
    best_test['test_wt_f1_score'] = round(best_test['test_wt_f1_score'], 5)
    best_test['train_dxn'] = train_dxn
    best_test['test_dxn'] = test_dxn
    best_test['valid_dxn'] = valid_dxn

    if output_file == None:
        output_file = "testing.json"

    record_results(best_test=best_test, results_path="/app/timeseries/multiclass_src/results/iris",
                   output_file=output_file)
    return


@click.command()
@click.option("--loss", required=True)
@click.option("--epochs", required=True)
@click.option("--batch_size", required=True)
@click.option("--run_name", required=False)
@click.option("--cuda", required=False)
@click.option("--patience", required=True)
@click.option("--train_tau", required=True)
@click.option("--output_file", required=True)
def run(loss, epochs, batch_size, run_name, cuda, train_tau, patience, output_file):
    print(run_name)
    
    batch_size = int(batch_size)
    epochs = int(epochs)

    seeds = [1,2,3,4,5]

    for i in range(len(seeds)): 
        data_splits = load_iris(seed=seeds[i])
        temp_name = str(run_name) + "-" + str(i)
        train_iris(data_splits, loss_metric=loss, epochs=epochs, train_tau=train_tau, seed=seeds[i], run_name=temp_name,
                cuda=cuda, batch_size=batch_size, patience=patience, output_file=output_file)


def main():
    os.environ['LC_ALL'] = 'C.UTF-8'
    os.environ['LANG'] = "C.UTF-8"
    run()


if __name__ == '__main__':
    main()

'''
python3 iris-poc.py --epochs=20000 --loss="approx-f1" --run_name="poc-af1-0.1" --cuda=0 --train_tau=0.1 --batch_size=1024 --patience=100 --output_file="poc_results.json"
python3 iris-poc.py --epochs=20000 --loss="approx-f1" --run_name="poc-af1-0.125" --cuda=0 --train_tau=0.125 --batch_size=1024 --patience=100 --output_file="poc_results.json"
python3 iris-poc.py --epochs=20000 --loss="approx-f1" --run_name="poc-af1-0.2" --cuda=0 --train_tau=0.2 --batch_size=1024 --patience=100 --output_file="poc_results.json"
python3 iris-poc.py --epochs=20000 --loss="approx-f1" --run_name="poc-af1-0.3" --cuda=0 --train_tau=0.3 --batch_size=1024 --patience=100 --output_file="poc_results.json"
python3 iris-poc.py --epochs=20000 --loss="approx-f1" --run_name="poc-af1-0.4" --cuda=0 --train_tau=0.4 --batch_size=1024 --patience=100 --output_file="poc_results.json"

python3 iris-poc.py --epochs=20000 --loss="approx-f1" --run_name="poc-af1-0.5" --cuda=1 --train_tau=0.5 --batch_size=1024 --patience=100 --output_file="poc_results.json"
python3 iris-poc.py --epochs=20000 --loss="approx-f1" --run_name="poc-af1-0.6" --cuda=1 --train_tau=0.6 --batch_size=1024 --patience=100 --output_file="poc_results.json"
python3 iris-poc.py --epochs=20000 --loss="approx-f1" --run_name="poc-af1-0.7" --cuda=2 --train_tau=0.7 --batch_size=1024 --patience=100 --output_file="poc_results.json"
python3 iris-poc.py --epochs=20000 --loss="approx-f1" --run_name="poc-af1-0.8" --cuda=2 --train_tau=0.8 --batch_size=1024 --patience=100 --output_file="poc_results.json"
python3 iris-poc.py --epochs=20000 --loss="approx-f1" --run_name="poc-af1-0.9" --cuda=2 --train_tau=0.9 --batch_size=1024 --patience=100 --output_file="poc_results.json"



'''
