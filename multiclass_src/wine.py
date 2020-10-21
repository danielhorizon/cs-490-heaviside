

import time
import torch
import click
import logging
import math

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchconfusion import confusion
from keras.utils import to_categorical


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

    # Re-labeling quality data.
    mappings = {
        4: 0,
        5: 1,
        6: 2,
        7: 3,
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

    # Split and shuffle
    train_df, test_df = train_test_split(raw_df, test_size=0.2, shuffle=shuffle)
    train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=shuffle)

    train_labels = np.array(train_df.pop("quality"))
    val_labels = np.array(val_df.pop("quality"))
    test_labels = np.array(test_df.pop("quality"))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    # Scale data
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
    
    def __init__(self, 
        input_features = 11, 
        hidden_layer1 = 1000, 
        hidden_layer2 = 500, 
        output_inputs = 250, 
        output_features = 4):

        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_layer1) # 11 -> 1000 
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2) # 1000 -> 500 
        self.fc3 = nn.Linear(hidden_layer2, output_inputs)  # 500 -> 250
        self.output = nn.Linear(output_inputs, output_features) # 250 -> 4 
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


def train_wine(data_splits, loss_metric, epochs):
    # setting train, validation, and test sets
    X_train, y_train = data_splits['train']['X'], data_splits['train']['y']
    X_valid, y_valid = data_splits['val']['X'], data_splits['val']['y']
    X_test, y_test = data_splits['test']['X'], data_splits['test']['y']

    X_train = Variable(torch.Tensor(X_train).float())
    X_test = Variable(torch.Tensor(X_test).float())
    X_valid = Variable(torch.Tensor(X_valid).float())
    y_train = Variable(torch.Tensor(y_train).long())
    y_test = Variable(torch.Tensor(y_test).long())
    y_valid = Variable(torch.Tensor(y_valid).long())

    # initialization
    now = int(time.time())
    early_stopping = False
    model = Model()
    print(model)

    val_losses = []
    best_test = {
        "now": now,
        "best-epoch": 0,
        "loss": float('inf'),
        "f1_score": 0,
        "accuracy": 0
    }

    # criterion
    if loss_metric == "ce":
        criterion = nn.CrossEntropyLoss()
    # elif loss_metric == "approx-f1":
    #     criterion = mean_f1_approx_loss_on(thresholds=torch.tensor([0.5]))
    # elif loss_metric == 'approx-accuracy':
    #     criterion = mean_accuracy_approx_loss_on(
    #         thresholds=torch.tensor([0.5]))
    else:
        raise RuntimeError("Unknown loss {}".format(loss_metric))

    # setting optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # ----- TRAINING -----
    losses = []
    for epoch in range(epochs):
        if early_stopping:
            logging.info(
                "[{}] Early Stopping at Epoch {}/{}".format(now, epoch, epochs))
            break

        model.train()
        optimizer.zero_grad()
        y_pred = model.forward(X_train)
        print("y pred: {}".format(y_pred))
        print("label: {}".format(y_train))
        loss = criterion(y_pred, y_train)
        losses.append(loss)
        loss.backward()
        optimizer.step()

        # ----- EVALUATE -----
        model.eval()
        mloss = np.array([x.detach().numpy() for x in losses]).mean()
        # https://github.com/rizalzaf/ap_perf/blob/master/examples/tabular.py

        # TRAIN Metrics: Accuracy, F1, Loss
        tr_output = model(X_train)
        # for each array, get the array of max index
        tr_pred = torch.Tensor([torch.argmax(x) for x in tr_output])
        tr_pred_np = [int(x) for x in tr_pred.cpu().numpy()]

        # TODO(dlee): you should do these manually, not using sci-kit
        tr_acc = accuracy_score(y_true=y_train, y_pred=tr_pred_np)
        tr_f1_micro = f1_score(
            y_true=y_train, y_pred=tr_pred_np, average='micro')
        tr_f1_macro = f1_score(
            y_true=y_train, y_pred=tr_pred_np, average='macro')
        tr_f1_weighted = f1_score(
            y_true=y_train, y_pred=tr_pred_np, average='weighted')

        # TEST Metrics: Accuracy, F1, Loss
        test_data = torch.Tensor(X_test)
        ts_output = model(test_data)
        ts_pred = torch.Tensor([torch.argmax(x) for x in ts_output])
        ts_pred_np = [int(x) for x in ts_pred.cpu().numpy()]
        ts_acc = accuracy_score(y_true=y_test, y_pred=ts_pred_np)
        ts_f1_micro = f1_score(
            y_true=y_test, y_pred=ts_pred_np, average='micro')
        ts_f1_macro = f1_score(
            y_true=y_test, y_pred=ts_pred_np, average='macro')
        ts_f1_weighted = f1_score(
            y_true=y_test, y_pred=ts_pred_np, average='weighted')

        # storing test metrics in dict based on loss
        if best_test['loss'] > mloss:
            best_test['loss'] = mloss
            best_test['now'] = int(time.time()) - now
            best_test['best-epoch'] = epoch
        if best_test['f1_score'] < ts_f1_weighted:
            best_test['f1_score'] = ts_f1_weighted
        if best_test['accuracy'] < ts_acc:
            best_test['accuracy'] = ts_acc

        print("-- Mean Loss: {:.3f}".format(mloss))
        print("Train - Epoch ({}): | Acc: {:.3f} | W F1: {:.3f} | Macro F1: {:.3f}".format(
            epoch, tr_acc, tr_f1_weighted, tr_f1_macro)
        )
        print("Test - Epoch ({}): | Acc: {:.3f} | W F1: {:.3f} | Macro F1: {:.3f}".format(
            epoch, ts_acc, ts_f1_weighted, ts_f1_macro)
        )

        # ----- VALIDATION -----
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            output = model.forward(X_valid)
            curr_val_loss = criterion(output, y_valid)

            # computing validation metrics via val_data
            val_output = model(X_valid)
            val_pred = torch.Tensor([torch.argmax(x) for x in val_output])
            val_pred_np = [int(x) for x in val_pred.cpu().numpy()]
            val_acc = accuracy_score(y_true=y_valid, y_pred=val_pred_np)
            val_f1_micro = f1_score(
                y_true=y_valid, y_pred=val_pred_np, average='micro')
            val_f1_macro = f1_score(
                y_true=y_valid, y_pred=val_pred_np, average='macro')
            val_f1_weighted = f1_score(
                y_true=y_valid, y_pred=val_pred_np, average='weighted')

            # breaking out of training if validation moves in other direction after last 2
            if epoch > 5:
                if (curr_val_loss > val_losses[-1]):
                    early_stopping = True
                if (curr_val_loss == val_losses[-1]) & (curr_val_loss == val_losses[-2]):
                    early_stopping = True

            val_losses.append(curr_val_loss)
            print("Valid - Epoch ({}): | Acc: {:.3f} | W F1: {:.3f} | Macro F1: {:.3f}".format(
                epoch, val_acc, val_f1_weighted, val_f1_macro)
            )

    print(best_test)
    return

@click.command()
@click.option("--loss", required=True)
@click.option("--epochs", required=True)
def run(loss, epochs):
    data_splits = load_white_wine() 
    train_wine(data_splits, loss_metric=loss, epochs=int(epochs))


def main():
    run()


if __name__ == '__main__':
    main()
