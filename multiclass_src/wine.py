
import os
import time
import torch
import click
import logging
import math
import random 

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
from mc_torchconfusion_weighted import wt_mean_f1_approx_loss_on
from gradient_flow import *

# for early stopping.
from pytorchtools import EarlyStopping

EPS = 1e-7
_WHITE_WINE = "../data/winequality-white.csv"
_RED_WINE = "../data/winequality-red.csv"


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


def load_white_wine(shuffle=True):
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


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def train_wine(data_splits, loss_metric, epochs, run_name):
    using_gpu = False
    if torch.cuda.is_available():
        print("device = cuda")
        device = "cuda"
        using_gpu = True
    else:
        print("device = cpu")
        device = "cpu"

    # setting train, validation, and test sets
    X_train, y_train = data_splits['train']['X'], data_splits['train']['y']
    X_valid, y_valid = data_splits['val']['X'], data_splits['val']['y']
    X_test, y_test = data_splits['test']['X'], data_splits['test']['y']

    X_train = Variable(torch.Tensor(X_train).float(), requires_grad=True)
    X_test = Variable(torch.Tensor(X_test).float())
    X_valid = Variable(torch.Tensor(X_valid).float())
    y_train = Variable(torch.Tensor(y_train).long())
    y_test = Variable(torch.Tensor(y_test).long())
    y_valid = Variable(torch.Tensor(y_valid).long())

    # using DataSet and DataLoader
    dataparams = {'batch_size': 128, 'shuffle': True, 'num_workers': 1}
    trainset = Dataset(data_splits['train'])
    validationset = Dataset(data_splits['val'])
    testset = Dataset(data_splits['test'])
    set_seed(0)
    train_loader = DataLoader(trainset, **dataparams)
    set_seed(0)
    val_loader = DataLoader(validationset, **dataparams)
    set_seed(0)
    test_loader = DataLoader(testset, **dataparams)

    # initialization
    early_stopping = False
    approx = False
    model = Model().to(device)

    # initialize the early_stopping object
    patience = 50
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # setting up tensorboard
    tensorboard_path = "/".join(["tensorboard", "wine", run_name])
    writer = SummaryWriter(tensorboard_path)

    avg_val_losses = []
    best_test = {
        "best-epoch": 0,
        "loss": float('inf'),
        "f1_score": 0,
        "accuracy": 0
    }

    # setting optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    first_run = True

    # criterion
    if loss_metric == "ce":
        criterion = nn.CrossEntropyLoss()
    elif loss_metric == "approx-f1":
        approx = True
        criterion = mean_f1_approx_loss_on(device=device)
    elif loss_metric == "approx-acc":
        approx = True
        criterion = mean_accuracy_approx_loss_on(device=device)
    elif loss_metric == "approx-auroc":
        approx = True
        criterion = mean_auroc_approx_loss_on(device=device)
    elif loss_metric == "approx-f1-wt":
        approx = True
        criterion = wt_mean_f1_approx_loss_on(device=device)
    else:
        raise RuntimeError("Unknown loss {}".format(loss_metric))

    # ----- TRAINING -----
    losses = []
    for epoch in range(epochs):
        if first_run:
            X_train = X_train.to(device)
            y_train = y_train.numpy()
            X_test = X_test.to(device)
            y_test = y_test.numpy()
            X_valid = X_valid.to(device)
            y_valid = y_valid.numpy()
            first_run = False

        if epochs != 0:
            for i, (inputs, labels) in enumerate(train_loader):
                labels = labels.type(torch.LongTensor).to(device)
                inputs = inputs.to(device)
                # setting into train mode.
                model.train()

                # zeroing out gradients
                optimizer.zero_grad()

                # making predictions
                y_pred = model(inputs)

                if not approx:
                    loss = criterion(y_pred, labels)
                else:
                    train_labels = torch.zeros(len(labels), 4).to(device).scatter_(
                        1, labels.unsqueeze(1), 1.).to(device)
                    loss = criterion(y_labels=train_labels, y_preds=y_pred)

                losses.append(loss)
                loss.backward()
                optimizer.step()

            # TRAIN Metrics: Accuracy, F1, Loss
            model.eval()
            if using_gpu:
                mloss = torch.mean(torch.stack(losses))
            else:
                mloss = np.array([x.item for x in losses]).mean()
            tr_output = model(X_train)
            tr_pred = torch.Tensor([torch.argmax(x)
                                    for x in tr_output]).to(device)
            tr_pred_np = [int(x) for x in tr_pred.cpu().numpy()]
            y_preds = tr_pred_np

            tr_acc = accuracy_score(y_true=y_train, y_pred=y_preds)
            tr_f1_micro = f1_score(
                y_true=y_train, y_pred=y_preds, average='micro')
            tr_f1_macro = f1_score(
                y_true=y_train, y_pred=y_preds, average='macro')
            tr_f1_weighted = f1_score(
                y_true=y_train, y_pred=y_preds, average='weighted')
            class_f1s = f1_score(y_true=y_train, y_pred=y_preds, average=None)
            class_re = recall_score(
                y_true=y_train, y_pred=y_preds, average=None)
            class_pr = precision_score(
                y_true=y_train, y_pred=y_preds, average=None)

            if run_name:
                writer.add_scalar("loss", mloss, epoch)
                writer.add_scalar("train/accuracy", tr_acc, epoch)
                writer.add_scalar("train/w-f1", tr_f1_weighted, epoch)
                writer.add_scalar("train/micro-f1", tr_f1_micro, epoch)
                writer.add_scalar("train/macro-f1", tr_f1_macro, epoch)
                # adding per-class f1, precision, and recall
                for i in range(4):
                    title = "train/class-" + str(i) + "-f1"
                    writer.add_scalar(title, np.array(
                        class_f1s[i]).mean(), epoch)
                    title = "train/class-" + str(i) + "-precision"
                    writer.add_scalar(title, np.array(
                        class_pr[i]).mean(), epoch)
                    title = "train/class-" + str(i) + "-recall"
                    writer.add_scalar(title, np.array(
                        class_re[i]).mean(), epoch)
        else:
            # adding in 0 for now.
            writer.add_scalar("loss", 0, epoch)
            writer.add_scalar("train/accuracy", 0, epoch)
            writer.add_scalar("train/w-f1", 0, epoch)
            writer.add_scalar("train/micro-f1", 0, epoch)
            writer.add_scalar("train/macro-f1", 0, epoch)
            for i in range(4):
                title = "train/class-" + str(i) + "-f1"
                writer.add_scalar(title, 0, epoch)
                title = "train/class-" + str(i) + "-precision"
                writer.add_scalar(title, 0, epoch)
                title = "train/class-" + str(i) + "-recall"
                writer.add_scalar(title, 0, epoch)

        # Printing out train metrics:
        print("Train - Epoch ({}): | Acc: {:.3f} | W F1: {:.3f} | Micro F1: {:.3f}| Macro F1: {:.3f}".format(
            epoch, tr_acc, tr_f1_weighted, tr_f1_micro, tr_f1_macro)
        )

        # ----- TEST SET -----
        # Calculate metrics after going through all the batches
        model.eval() 
        test_preds, test_labels = np.array([]), np.array([])
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device) 
            output = model(inputs)
            _, predicted = torch.max(output, 1)

            pred_arr = predicted.cpu().numpy()
            label_arr = labels.cpu().numpy() 

            test_labels = np.concatenate([test_labels, label_arr])
            test_preds = np.concatenate([test_preds, pred_arr])
        
        test_acc = accuracy_score(y_true=test_labels, y_pred=test_preds)
        test_f1_micro = f1_score(y_true=test_labels, y_pred=test_preds, average='micro')
        test_f1_macro = f1_score(y_true=test_labels, y_pred=test_preds, average='macro')
        test_f1_weighted = f1_score(y_true=test_labels, y_pred=test_preds, average='weighted')
        test_class_f1s = f1_score(
            y_true=test_labels, y_pred=test_preds, average=None)
        test_class_prs = precision_score(
            y_true=test_labels, y_pred=test_preds, average=None)
        test_class_rec = recall_score(
            y_true=test_labels, y_pred=test_preds, average=None)

        # adding in tensorboard metrics 
        if run_name: 
            writer.add_scalar("test/accuracy", test_acc, epoch)
            writer.add_scalar("test/micro-f1", test_f1_micro, epoch)
            writer.add_scalar("test/macro-f1", test_f1_macro, epoch)
            writer.add_scalar("test/w-f1", test_f1_weighted, epoch)
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

                tp, fp, fn, tn = get_confusion(
                    gt=test_labels, pt=test_preds, class_value=i)
                tp_title = 'test/class-' + str(i) + "-TP"
                fp_title = 'test/class-' + str(i) + "-FP"
                fn_title = 'test/class-' + str(i) + "-FN"
                tn_title = 'test/class-' + str(i) + "-TN"
                writer.add_scalar(tp_title, tp, epoch)
                writer.add_scalar(fp_title, fp, epoch)
                writer.add_scalar(fn_title, fn, epoch)
                writer.add_scalar(tn_title, tn, epoch)


        # storing test metrics in dict based on loss
        if epoch != 0: 
            if best_test['loss'] > mloss:
                best_test['loss'] = mloss
                best_test['best-epoch'] = epoch
            if best_test['f1_score'] < test_f1_weighted:
                best_test['f1_score'] = test_f1_weighted
            if best_test['accuracy'] < test_acc:
                best_test['accuracy'] = test_acc

        print("Test - Epoch ({}): | Acc: {:.3f} | W F1: {:.3f} | Micro F1: {:.4f} | Macro F1: {:.3f}".format(
            epoch, test_acc, test_f1_weighted, test_f1_micro, test_f1_macro)
        )
        print(classification_report(y_true=test_labels, y_pred=test_preds,
                                    target_names=['0', '1', '2', '3']))

        # ----- VALIDATION -----
        model.eval()
        valid_losses = []
        with torch.no_grad(): 
            val_preds, val_labels = np.array([]), np.array([])
            for i, (inputs, labels) in enumerate(val_loader): 
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = model(inputs)
                _, predicted = torch.max(output, 1)

                # calculate metrics 
                model.eval()
                pred_arr = predicted.cpu().numpy()
                label_arr = labels.cpu().numpy() 
                val_labels = np.concatenate([val_labels, label_arr])
                val_preds = np.concatenate([val_preds, pred_arr])

                if approx:
                    labels = labels.type(torch.int64)
                    # there are 3 wine classes
                    valid_labels = torch.zeros(len(labels), 4).to(device).scatter_(
                        1, labels.unsqueeze(1), 1.)
                    curr_val_loss = criterion(
                        y_labels=valid_labels, y_preds=output)
                else:
                    labels = labels.type(torch.int64)
                    curr_val_loss = criterion(output, labels)

                # appending loss to master list. 
                valid_losses.append(curr_val_loss.detach().cpu().numpy())

            # computing validation metrics via val_data
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
            # adding in per-class metrics: 
            if run_name: 
                writer.add_scalar("val/accuracy", val_acc, epoch)
                writer.add_scalar("val/micro-f1", val_f1_micro, epoch)
                writer.add_scalar("val/macro-f1", val_f1_macro, epoch)
                writer.add_scalar("val/w-f1", val_f1_weighted, epoch)

                # adding per-class f1, precision, and recall
                for i in range(4):
                    title = "val/class-" + str(i) + "-f1"
                    writer.add_scalar(title, np.array(class_val_f1[i]).mean(), epoch)
                    title = "val/class-" + str(i) + "-precision"
                    writer.add_scalar(title, np.array(class_val_pr[i]).mean(), epoch)
                    title = "val/class-" + str(i) + "-recall"
                    writer.add_scalar(title, np.array(class_val_re[i]).mean(), epoch)

                    # adding in per class training
                    # get_confusion(gt, pt, class_value=None):
                    tp, fp, fn, tn = get_confusion(
                        gt=val_labels, pt=val_preds, class_value=i)
                    tp_title = 'val/class-' + str(i) + "-TP"
                    fp_title = 'val/class-' + str(i) + "-FP"
                    fn_title = 'val/class-' + str(i) + "-FN"
                    tn_title = 'val/class-' + str(i) + "-TN"
                    writer.add_scalar(tp_title, tp, epoch)
                    writer.add_scalar(fp_title, fp, epoch)
                    writer.add_scalar(fn_title, fn, epoch)
                    writer.add_scalar(tn_title, tn, epoch)

            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early Stopping")
                break

            print("Val - Epoch ({}): | Acc: {:.3f} | W F1: {:.3f} | Micro F1: {:.3f} | Macro F1: {:.3f}\n".format(
                epoch, val_acc, val_f1_weighted, val_f1_micro, val_f1_macro)
            )

    print(best_test)
    return


@click.command()
@click.option("--loss", required=True)
@click.option("--epochs", required=True)
@click.option("--run_name", required=False)
def run(loss, epochs, run_name):
    data_splits = load_white_wine()
    train_wine(data_splits, loss_metric=loss,
               epochs=int(epochs), run_name=run_name)


def main():
    os.environ['LC_ALL'] = 'C.UTF-8'
    os.environ['LANG'] = "C.UTF-8"
    run()


if __name__ == '__main__':
    main()


'''
- Write loss for the weighted classes, and for the different F1's 


'''
