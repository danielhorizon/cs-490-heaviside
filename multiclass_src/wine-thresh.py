
import warnings
import os
import time
import torch
import click
import logging
import math
import random 
import json 

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from mc_metrics import get_confusion

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import metrics

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable

from mc_torchconfusion import *
from gradient_flow import *

EPS = 1e-7
_WHITE_WINE = "../data/winequality-white.csv"
_RED_WINE = "../data/winequality-red.csv"

warnings.filterwarnings('ignore')


'''
https://www.koreascience.or.kr/article/JAKO201832073079660.pdf
    - Actual baseline, waiting on email from professor 

https://github.com/simonneutert/wine_quality_data                   - Data exploration purposes 
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7475021        - Regression on quality 
http://rstudio-pubs-static.s3.amazonaws.com/565136_b4395e2500ec4c129ab776b9e8dd24de.html


https://link.springer.com/article/10.1007/s11063-019-10125-6

Actual wine-quality BASELINE
https://link.springer.com/chapter/10.1007/978-3-030-52249-0_27
https://link.springer.com/chapter/10.1007/978-3-030-52249-0_27#enumeration
'''


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_white_wine(shuffle=True, seed=None):
    set_seed(seed)
    # https://link.springer.com/chapter/10.1007/978-3-030-52249-0_27
    # solely doing this on white wine
    raw_df = pd.read_csv(_WHITE_WINE, sep=';')
    raw_df.columns = raw_df.columns.str.replace(' ', '-')
    raw_df = raw_df[raw_df.quality != 3]
    raw_df = raw_df[raw_df.quality != 9]
    print("shape before: {}".format(raw_df.shape))

    # Re-labeling quality data.
    mappings = {
        4: 0,  # low
        5: 1,  # below average
        6: 2,  # average
        7: 3,  # above average/high
        8: 3
    }
    raw_df["quality"] = raw_df["quality"].apply(lambda x: mappings[x])

    # Removing outliers that are +/- 3 SD's, as done in the paper
    # only for the x columns, not for quality.
    x_columns = ['fixed-acidity', 'volatile-acidity', 'citric-acid', 'residual-sugar',
                 'chlorides', 'free-sulfur-dioxide', 'total-sulfur-dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']
    raw_df[x_columns] = raw_df[x_columns][raw_df[x_columns].apply(
        lambda x:(x-x.mean()).abs() <= (3*x.std())).all(1)]

    raw_df = raw_df.dropna()
    print("shape after: {}".format(raw_df.shape))

    # Split and shuffle
    train_df, test_df = train_test_split(raw_df, test_size=0.2, shuffle=shuffle)
    train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=shuffle)

    train_labels = np.array(train_df.pop("quality"))
    val_labels = np.array(val_df.pop("quality"))
    test_labels = np.array(test_df.pop("quality"))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    # scaling according to the train data. 
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)
    print("whitewine mean: {}".format(train_features.mean()))
    print("whitewine variance: {}".format(train_features.var()))
    print("whitewine min: {}, max: {}".format(
        train_features.min(), train_features.max()))

    print('whitewine Training labels shape: {}'.format(train_labels.shape))
    print('whitewine Validation labels shape: {}'.format(val_labels.shape))
    print('whitewine Test labels shape: {}'.format(test_labels.shape))
    print('whitewine Training features shape: {}'.format(train_features.shape))
    print('whitewine Validation features shape: {}'.format(val_features.shape))
    print('whitewine Test features shape: {}'.format(test_features.shape))

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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, ds_split):
        self.X = torch.from_numpy(ds_split['X']).float()
        self.y = torch.from_numpy(ds_split['y']).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index, :], self.y[index]


class Model(nn.Module):
    '''
    - Input Layer: 11, output = 1000 (using ReLU) activation function
    - Hidden layer 1: output = 500 (using ReLU) 
    - Hidden layer 2: output = 250 (using ReLU)
    - Output layer: output = 4 with "Softmax" activation and L2 regularization of 0.0001 
    75 epochs and 60% classification accuracy 
    '''

    def __init__(self, input_features=11, hidden_layer1=1000, hidden_layer2=500, output_inputs=250, output_features=4):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_layer1)  # 11 -> 1000
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)  # 1000 -> 500
        self.fc3 = nn.Linear(hidden_layer2, output_inputs)  # 500 -> 250
        self.output = nn.Linear(output_inputs, output_features)  # 250 -> 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


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


def train_wine(data_splits, loss_metric, epochs, seed, run_name, cuda, train_tau, batch_size, patience, output_file):
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
    train_dxn, test_dxn, valid_dxn = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    best_test = {
        "best-epoch": 0,
        "loss": float('inf'),
        "test_wt_f1_score": 0,
        'val_wt_f1_score':float("inf"),
        "test_accuracy": 0,
        "val_accuracy": 0,
        "learning_rate": 0,
        "imbalanced": False,
        "loss_metric": loss_metric,
        "run_name": run_name,
        "train_dxn": None,
        "test_dxn": None,
        "valid_dxn": None,
        "seed": seed,
        "batch_size": batch_size,
        "evaluation": None,
        "patience": None
    }

    # initialization
    learning_rate = 0.0001
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_test['learning_rate'] = learning_rate
    best_test['patience'] = patience

    if run_name:
        experiment_name = run_name
        tensorboard_path = "/".join(["tensorboard", "wine", experiment_name])
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
    elif loss_metric == "approx-ap": 
        approx = True
        train_threshold = float(train_tau)
        threshold_tensor = torch.Tensor([train_threshold]).to(device)
        criterion = mean_ap_approx_loss_on(
            device=device, thresholds=threshold_tensor)
    elif loss_metric == "approx-ap2":
        approx = True
        train_threshold = float(train_tau)
        threshold_tensor = torch.Tensor([train_threshold]).to(device)
        criterion = old_mean_ap_approx_loss_on(
            device=device, thresholds=threshold_tensor)
    else:
        raise RuntimeError("Unknown loss {}".format(loss_metric))

    ## --- training ---
    early_stopping = False
    lowest_f1_loss = None
    print("patience: {}".format(patience))
    patience = int(patience)
    reset_patience = patience

    losses = []
    for epoch in range(epochs):
        ## early stopping
        if early_stopping:
            print("Early stopping at Epoch {}/{}".format(epoch, epochs))
            break

        accs, microf1s, macrof1s, wf1s = [], [], [], []
        micro_prs, macro_prs, weighted_prs = [], [], []
        micro_recalls, macro_recalls, weighted_recalls = [], [], []
        class_f1_scores = {0: [], 1: [], 2: [], 3:[]}
        class_precision = {0: [], 1: [], 2: [], 3:[]}
        class_recall = {0: [], 1: [], 2: [], 3:[]}

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
                for i in range(4):
                    title = "train/class-" + str(i) + "-f1"
                    writer.add_scalar(title, 0, epoch)
                    title = "train/class-" + str(i) + "-precision"
                    writer.add_scalar(title, 0, epoch)
                    title = "train/class-" + str(i) + "-recall"
                    writer.add_scalar(title, 0, epoch)

        else:
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
                    loss = criterion(y_labels=train_labels, y_preds=output)

                losses.append(loss)
                loss.backward()
                optimizer.step()

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
            print("Train Loss: {}".format(m_loss))
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
                for i in range(4):
                    title = "train/class-" + str(i) + "-f1"
                    writer.add_scalar(title, np.array(
                        class_f1_scores[i]).mean(), epoch)
                    title = "train/class-" + str(i) + "-precision"
                    writer.add_scalar(title, np.array(
                        class_precision[i]).mean(), epoch)
                    title = "train/class-" + str(i) + "-recall"
                    writer.add_scalar(title, np.array(
                        class_recall[i]).mean(), epoch)

        # ---------- test ----------
        model.eval()
        test_losses = []
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
                    batch_test_loss  = criterion(y_labels=trans_labels, y_preds=output)
                else:
                    batch_test_loss = criterion(output, labels)

            # adding in test loss
            test_losses.append(batch_test_loss.detach().cpu().numpy())
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
            macro_prs = precision_score(y_true=test_labels, y_pred=test_preds, average='macro')
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
                for i in range(4):
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
            
            print("Test - Epoch ({}): | Macro PR: {:.4f}".format(
                epoch, macro_prs
            ))
            print("Test - Epoch ({}): | Loss: {:.4f} | Acc: {:.3f} | W F1: {:.3f} | Micro F1: {:.3f} | Macro F1: {:.3f}".format(
                epoch, test_loss, test_acc, test_f1_weighted, test_f1_micro, test_f1_macro)
            )

        # ---------- validation ----------
        # Calculate metrics after going through all the batches
        model.eval()
        with torch.no_grad():
            val_preds, val_labels = np.array([]), np.array([])
            valid_losses = []

            for batch, (inputs, labels) in enumerate(val_loader):
                labels_list = labels.cpu().numpy()
                for label in labels_list:
                    valid_dxn[int(label)] += 1

                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)
                _, predicted = torch.max(output, 1)

                # calculate metrics
                pred_arr = predicted.cpu().numpy()
                label_arr = labels.cpu().numpy()

                val_labels = np.concatenate([val_labels, label_arr])
                val_preds = np.concatenate([val_preds, pred_arr])

                # valid loss if APPROX
                labels = labels.type(torch.int64)
                if approx:
                    valid_labels = torch.zeros(len(labels), len(output[0])).to(
                        device).scatter_(1, labels.unsqueeze(1), 1.).to(device)

                    output = output.to(device)
                    batch_val_loss = criterion(
                        y_labels=valid_labels, y_preds=output)
                # using regular CE
                else:
                    batch_val_loss = criterion(output, labels)

                valid_losses.append(batch_val_loss.detach().cpu().numpy())

            val_acc = accuracy_score(y_true=val_labels, y_pred=val_preds)
            val_f1_micro = f1_score(
                y_true=val_labels, y_pred=val_preds, average='micro')
            val_f1_macro = f1_score(
                y_true=val_labels, y_pred=val_preds, average='macro')
            val_f1_weighted = f1_score(
                y_true=val_labels, y_pred=val_preds, average='weighted')

            class_val_f1 = f1_score(
                y_true=val_labels, y_pred=val_preds, average=None)
            class_val_pr = precision_score(
                y_true=val_labels, y_pred=val_preds, average=None)
            class_val_re = recall_score(
                y_true=val_labels, y_pred=val_preds, average=None)
            valid_loss = np.mean(valid_losses)

            val_macro_pr = precision_score(
                y_true=val_labels, y_pred=val_preds, average='macro')

            if run_name:
                writer.add_scalar("valid/train-loss", valid_loss, epoch)
                writer.add_scalar("valid/accuracy", val_acc, epoch)
                writer.add_scalar("valid/w-f1", val_f1_weighted, epoch)
                writer.add_scalar("valid/micro-f1", val_f1_micro, epoch)
                writer.add_scalar("valid/macro-f1", val_f1_macro, epoch)

                # adding per-class f1, precision, and recall
                for i in range(4):
                    title = "valid/class-" + str(i) + "-f1"
                    writer.add_scalar(title, class_val_f1[i], epoch)
                    title = "valid/class-" + str(i) + "-precision"
                    writer.add_scalar(title, class_val_pr[i], epoch)
                    title = "valid/class-" + str(i) + "-recall"
                    writer.add_scalar(title, class_val_re[i], epoch)

            if epoch != 0:
                if best_test['val_wt_f1_score'] < val_f1_weighted:
                    best_test['val_wt_f1_score'] = val_f1_weighted
                if best_test['val_accuracy'] < val_acc:
                    best_test['val_accuracy'] = val_acc

            print("Val - Epoch ({}): | Loss: {:.3f} | Macro PR: {:.3f}\n".format(
                epoch, valid_loss, val_macro_pr)
            )
            print("Val - Epoch ({}): | Acc: {:.3f} | W F1: {:.3f} | Micro F1: {:.3f} | Macro F1: {:.3f}\n".format(
                epoch, val_acc, val_f1_weighted, val_f1_micro, val_f1_macro)
            )

            ## checking early stopping per epoch
            patience -= 1
            adjust = False
            if lowest_f1_loss is None or valid_loss < lowest_f1_loss:
                adjust = True
                if lowest_f1_loss != None:
                    print("Valid loss decreased {:.5f} -> {:.5f}! Resetting patience to: {}".format(
                        lowest_f1_loss, valid_loss, reset_patience))

                today_date = time.strftime('%Y%m%d')

                # TODO(dlee): add in support for balanced dataset.
                model_file_path = "/".join(["/app/timeseries/multiclass_src/models/wine-ap",
                                            '{}-best_model-{}.pth'.format(
                                                today_date, run_name
                                            )])
                torch.save(model, model_file_path)
                patience = reset_patience
                lowest_f1_loss = valid_loss

                best_test['model_file_path'] = model_file_path

            ## if early stopping has begun, print it like this.
            if not adjust:
                print("Early stopping {}/{}...".format(reset_patience -
                                                       patience, reset_patience))
            if patience <= 0:
                early_stopping = True

    # ----- FINAL EVALUATION STEP, USING FULLY TRAINED MODEL -----
    print("--- Finished Training - Entering Final Evaluation Step\n")
    # saving the model.
    model_file_path = "/".join(["/app/timeseries/multiclass_src/models/wine-ap",
                                '{}-overfit-model-{}.pth'.format(
                                    time.strftime('%Y%m%d'), run_name
                                )])
    torch.save(model, model_file_path)

    # ----- recording results in a json.
    if torch.is_tensor(best_test['loss']):
        best_test['loss'] = best_test['loss'].item()
    if torch.is_tensor(best_test['test_wt_f1_score']):
        best_test['test_wt_f1_score'] = best_test['test_wt_f1_score'].item()
    if torch.is_tensor(best_test['val_wt_f1_score']):
        best_test['val_wt_f1_score'] = best_test['val_wt_f1_score'].item()

    best_test['loss'] = round(best_test['loss'], 5)
    best_test['test_wt_f1_score'] = round(best_test['test_wt_f1_score'], 5)
    best_test['val_wt_f1_score'] = round(best_test['val_wt_f1_score'], 5)
    best_test['train_dxn'] = train_dxn
    best_test['test_dxn'] = test_dxn
    best_test['valid_dxn'] = valid_dxn

    if output_file == None:
        output_file = "testing.json"

    record_results(best_test=best_test, results_path="/app/timeseries/multiclass_src/results/wine",
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

    # seeds = [1, 3, 5]
    seeds = [1]
    for i in range(len(seeds)): 
        data_splits = load_white_wine(seed=seeds[i])
        temp_name = str(run_name) + "-" + str(i)
        train_wine(data_splits, loss_metric=loss, epochs=epochs, train_tau=train_tau, seed=seeds[i], run_name=temp_name,
                cuda=cuda, batch_size=batch_size, patience=patience, output_file=output_file)


def main():
    os.environ['LC_ALL'] = 'C.UTF-8'
    os.environ['LANG'] = "C.UTF-8"
    run()


if __name__ == '__main__':
    main()

'''

python3 wine-thresh.py --epochs=5000 --loss="approx-f1" --run_name="traintau-af1-0.1" --cuda=0 --train_tau=0.1 --batch_size=1024 --patience=100 --output_file="thresh_results.json"
python3 wine-thresh.py --epochs=5000 --loss="approx-f1" --run_name="traintau-af1-0.125" --cuda=0 --train_tau=0.125 --batch_size=1024 --patience=100 --output_file="thresh_results.json"
python3 wine-thresh.py --epochs=5000 --loss="approx-f1" --run_name="traintau-af1-0.2" --cuda=0 --train_tau=0.2 --batch_size=1024 --patience=100 --output_file="thresh_results.json"
python3 wine-thresh.py --epochs=5000 --loss="approx-f1" --run_name="traintau-af1-0.3" --cuda=0 --train_tau=0.3 --batch_size=1024 --patience=100 --output_file="thresh_results.json"
python3 wine-thresh.py --epochs=5000 --loss="approx-f1" --run_name="traintau-af1-0.4" --cuda=0 --train_tau=0.4 --batch_size=1024 --patience=100 --output_file="thresh_results.json"

python3 wine-thresh.py --epochs=5000 --loss="approx-f1" --run_name="traintau-af1-0.5" --cuda=1 --train_tau=0.5 --batch_size=1024 --patience=100 --output_file="thresh_results.json"
python3 wine-thresh.py --epochs=5000 --loss="approx-f1" --run_name="traintau-af1-0.6" --cuda=1 --train_tau=0.6 --batch_size=1024 --patience=100 --output_file="thresh_results.json"
python3 wine-thresh.py --epochs=5000 --loss="approx-f1" --run_name="traintau-af1-0.7" --cuda=2 --train_tau=0.7 --batch_size=1024 --patience=100 --output_file="thresh_results.json"
python3 wine-thresh.py --epochs=5000 --loss="approx-f1" --run_name="traintau-af1-0.8" --cuda=2 --train_tau=0.8 --batch_size=1024 --patience=100 --output_file="thresh_results.json"
python3 wine-thresh.py --epochs=5000 --loss="approx-f1" --run_name="traintau-af1-0.9" --cuda=2 --train_tau=0.9 --batch_size=1024 --patience=100 --output_file="thresh_results.json"


'''
