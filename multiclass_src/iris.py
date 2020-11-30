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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable

from mc_torchconfusion import *
from gradient_flow import *

# for early stopping.
from pytorchtools import EarlyStopping

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


def record_results(best_test, output_file):
    # reading in the data from the existing file.
    results_path = "/app/timeseries/multiclass_src/results"
    file_path = "/".join([results_path, output_file])
    with open(file_path, "r+") as f:
        data = json.load(f)
        data.append(best_test)
        f.close()

    with open(file_path, "w") as outfile:
        json.dump(data, outfile)


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
    train_df, test_df = train_test_split(raw_df, test_size=0.20, shuffle=shuffle)
    train_df, val_df = train_test_split(train_df, test_size=0.20, shuffle=shuffle)

    train_labels = np.array(train_df.pop("species"))
    val_labels = np.array(val_df.pop("species"))
    test_labels = np.array(test_df.pop("species"))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    # scaling data.
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
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
    dataparams = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 1}
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

        # GT LIST:tensor([0., 0., 1.,  ..., 0., 1., 0.], device='cuda:0')
        # PT LIST: tensor([0.1047, 0.1021, 0.1016,  ..., 0.1004, 0.1035, 0.1009], device='cuda:0', grad_fn= < SelectBackward > )

        # print("GT LIST:{}".format(gt_list))
        # print("PT LIST:{}".format(pt_list))
        # tensor([1., 1., 1.,  ..., 1., 1., 1.])
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


def train_iris(data_splits, loss_metric, epochs, seed, run_name, batch_size):
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
    y_train = Variable(torch.Tensor(y_train).long())
    X_test = Variable(torch.Tensor(X_test).float())
    X_valid = Variable(torch.Tensor(X_valid).float())
    y_test = Variable(torch.Tensor(y_test).long())
    y_valid = Variable(torch.Tensor(y_valid).long())

    # using DataSet and DataLoader
    train_loader, val_loader, test_loader = create_loaders(data_splits, batch_size, seed)

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
        "val_wt_f1_score": 0,
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
        "evaluation": None
    }

    # initialization
    approx = False
    model = Model().to(device)
    patience = 10
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    learning_rate = 0.003
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_test['learning_rate'] = learning_rate

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
    else:
        raise RuntimeError("Unknown loss {}".format(loss_metric))


    # ----- TRAINING -----
    losses = []
    for epoch in range(epochs):
        accs, microf1s, macrof1s, wf1s = [], [], [], []
        micro_prs, macro_prs, weighted_prs = [], [], []
        micro_recalls, macro_recalls, weighted_recalls = [], [], []
        class_f1_scores = {0: [], 1: [], 2: []}
        class_precision = {0: [], 1: [], 2: []}
        class_recall = {0: [], 1: [], 2: []} 

        if epoch == 0:
            print("--- MODEL PARAMS ---")
            for param in model.parameters():
                print(param.data[1])
                break

        for batch, (inputs, labels) in enumerate(train_loader):
            # setting into train mode.
            model.train()

            # for class distribution 
            labels_list = labels.numpy() 
            for label in labels_list: 
                train_dxn[int(label)] += 1 
            
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor).to(device)

            # zero grad
            optimizer.zero_grad()
            output = model(inputs)

            if not approx:
                loss = criterion(output, labels)
            else:
                # TODO(dlee): this is hard coded in (the 3 part)
                train_labels = torch.zeros(len(labels), 3).scatter_(1, labels.unsqueeze(1), 1.)
                loss, _, _, _, _, _, _, _, _ = criterion(
                    y_labels=train_labels, y_preds=output)

            losses.append(loss)
            loss.backward()
            optimizer.step()

            # checking prediction via evaluation (for every batch)
            model.eval() 
            y_pred = model(inputs)
            _, train_preds = torch.max(y_pred, 1)
            # storing metrics for each batch
            # accs = array of each batch's accuracy -> averaged at each epoch
            accs.append(accuracy_score(y_true=labels.cpu(), y_pred=train_preds.cpu()))
            microf1s.append(f1_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="micro"))
            macrof1s.append(f1_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="macro"))
            wf1s.append(f1_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="weighted"))

            # precision
            micro_prs.append(precision_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="micro"))
            macro_prs.append(precision_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="macro"))
            weighted_prs.append(precision_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="weighted"))

            # recall
            micro_recalls.append(recall_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="micro"))
            macro_recalls.append(recall_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="macro"))
            weighted_recalls.append(recall_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="weighted"))

            class_f1s = f1_score(y_true=labels.cpu(),y_pred=train_preds.cpu(), average=None)
            class_re = recall_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average=None)
            class_pr = precision_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average=None)

            for i in range(len(class_f1s)):
                class_f1_scores[i].append(class_f1s[i])
                class_precision[i].append(class_pr[i])
                class_recall[i].append(class_re[i])

        # https://github.com/rizalzaf/ap_perf/blob/master/examples/tabular.py
        m_loss = torch.mean(torch.stack(losses)) if using_gpu else np.array([x.item() for x in losses]).mean()
        m_accs = np.array(accs).mean()
        m_weightedf1s = np.array(microf1s).mean()
        m_microf1s = np.array(microf1s).mean()
        m_macrof1s = np.array(macrof1s).mean()
        print("Train - Epoch ({}): | Acc: {:.3f} | W F1: {:.3f} | Micro F1: {:.3f}| Macro F1: {:.3f}".format(
            epoch, m_accs, m_weightedf1s, m_microf1s, m_macrof1s)
        )

        # ----- TEST SET -----
        # Calculate metrics after going through all the batches
        model.eval() 
        test_preds, test_labels = np.array([]), np.array([])
        for i, (inputs, labels) in enumerate(test_loader):
            labels_list = labels.numpy()
            for label in labels_list:
                test_dxn[int(label)] += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            # print("output: {}".format(output))
            _, predicted = torch.max(output, 1)

            pred_arr = predicted.cpu().numpy()
            label_arr = labels.cpu().numpy()

            test_labels = np.concatenate([test_labels, label_arr])
            test_preds = np.concatenate([test_preds, pred_arr])

        test_acc = accuracy_score(y_true=test_labels, y_pred=test_preds)
        test_f1_micro = f1_score(y_true=test_labels, y_pred=test_preds, average='micro')
        test_f1_macro = f1_score(y_true=test_labels, y_pred=test_preds, average='macro')
        test_f1_weighted = f1_score(y_true=test_labels, y_pred=test_preds, average='weighted')

        test_class_f1s = f1_score(y_true=test_labels, y_pred=test_preds, average=None)
        test_class_prs = precision_score(y_true=test_labels, y_pred=test_preds, average=None)
        test_class_rec = recall_score(y_true=test_labels, y_pred=test_preds, average=None)

        if epoch != 0:
            if best_test['loss'] > m_loss:
                best_test['loss'] = m_loss
                best_test['best-epoch'] = epoch
            if best_test['test_wt_f1_score'] < test_f1_weighted:
                best_test['test_wt_f1_score'] = test_f1_weighted
            if best_test['test_accuracy'] < test_acc:
                best_test['test_accuracy'] = test_acc

        print("Test - Epoch ({}): | Acc: {:.3f} | W F1: {:.3f} | Micro F1: {:.3f} | Macro F1: {:.3f}".format(
            epoch, test_acc, test_f1_weighted, test_f1_micro, test_f1_macro)
        )

        # ---------- VALIDATION ----------
        # Calculate metrics after going through all the batches
        model.eval()
        valid_losses = []
        with torch.no_grad():
            val_preds, val_labels = np.array([]), np.array([])

            for batch, (inputs, labels)  in enumerate(val_loader):
                labels_list = labels.numpy()
                for label in labels_list:
                    valid_dxn[int(label)] += 1

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


                # valid loss if APPROX
                if approx:
                    # print(target)
                    labels = labels.type(torch.int64)
                    valid_labels = torch.zeros(len(labels), 3).scatter_(1, labels.unsqueeze(1), 1.)
                    curr_val_loss, hclass_tp, hclass_fn, hclass_fp, hclass_tn, hclass_pr, hclass_re, hclass_f1, hclass_acc = criterion(
                        y_labels=valid_labels, y_preds=output)
                # using regular CE
                else:
                    labels = labels.type(torch.int64)
                    curr_val_loss = criterion(output, labels)

                valid_losses.append(curr_val_loss.detach().cpu().numpy())

            val_acc = accuracy_score(y_true=val_labels, y_pred=val_preds)
            val_f1_micro = f1_score(y_true=val_labels, y_pred=val_preds, average='micro')
            val_f1_macro = f1_score(y_true=val_labels, y_pred=val_preds, average='macro')
            val_f1_weighted = f1_score(y_true=val_labels, y_pred=val_preds, average='weighted')

            class_val_f1 = f1_score(y_true=val_labels, y_pred=val_preds, average=None)
            class_val_pr = precision_score(y_true=val_labels, y_pred=val_preds, average=None)
            class_val_re = recall_score(y_true=val_labels, y_pred=val_preds, average=None)
            valid_loss = np.mean(valid_losses)

            # computing the losses
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early Stopping")
                break

            print("Val - Epoch ({}): | Acc: {:.3f} | W F1: {:.3f} | Micro F1: {:.3f} | Macro F1: {:.3f}\n".format(
                    epoch, val_acc, val_f1_weighted, val_f1_micro, val_f1_macro)
            )
            if epoch != 0:
                if best_test['val_wt_f1_score'] < val_f1_weighted:
                    best_test['val_wt_f1_score']=val_f1_weighted
                if best_test['val_accuracy'] < val_acc:
                    best_test['val_accuracy']=val_acc

        # inits.
    model.eval()
    test_thresholds=[0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]

    eval_json={
        "run_name": None,
        "seed": seed,
        "0.1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None, "mean_f1": None, "eval_dxn": None},
        "0.2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None, "mean_f1": None, "eval_dxn": None},
        "0.3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None, "mean_f1": None, "eval_dxn": None},
        "0.4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None, "mean_f1": None, "eval_dxn": None},
        "0.45": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None, "mean_f1": None, "eval_dxn": None},
        "0.5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None, "mean_f1": None, "eval_dxn": None},
        "0.55": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None, "mean_f1": None, "eval_dxn": None},
        "0.6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None, "mean_f1": None, "eval_dxn": None},
        "0.7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None, "mean_f1": None, "eval_dxn": None},
        "0.8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None, "mean_f1": None, "eval_dxn": None},
        "0.9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None, "mean_f1": None, "eval_dxn": None}
    }

    with torch.no_grad():
        for tau in test_thresholds:
            # go through all the thresholds, and test them out again.
            final_test_dxn = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            test_preds, test_labels = [], []
            for i, (inputs, labels) in enumerate(test_loader):
                # updating distribution of labels.
                labels_list = labels.numpy()
                for label in labels_list:
                    final_test_dxn[label] += 1

                # stacking onto tensors.
                inputs = inputs.to(device)
                labels = labels.to(device)

                # passing it through our finalized model.
                output = model(inputs)
                labels = torch.zeros(len(labels), 10).to(device).scatter_(
                    1, labels.unsqueeze(1), 1.).to(device)

                pred_arr=output.detach().cpu().numpy()
                label_arr=labels.detach().cpu().numpy()

                # appending results.
                test_preds.append(pred_arr)
                test_labels.append(label_arr)

            test_preds=torch.tensor(test_preds[0])
            test_labels=torch.tensor(test_labels[0])

            class_f1s, mean_f1, precisions, recalls= evaluation_f1(
                device=device, y_labels=test_labels, y_preds=test_preds, threshold=tau)

            tau = str(tau)
            eval_json[tau]['class_f1s'] = class_f1s.numpy().tolist()
            eval_json[tau]['mean_f1'] = mean_f1.item()
            eval_json[tau]['eval_dxn'] = final_test_dxn
            eval_json[tau]['class_precisions'] = precisions.numpy().tolist()
            eval_json[tau]['class_recalls'] = recalls.numpy().tolist()

    eval_json['run'] = run_name
    eval_json['seed'] = seed
    best_test['evaluation'] = eval_json

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

    record_results(best_test, "20201129_iris.json")
    return


@click.command()
@click.option("--loss", required=True)
@click.option("--epochs", required=True)
@click.option("--batch_size", required=True)
@click.option("--run_name", required=False)
def run(loss, epochs, batch_size, run_name):
    seed = 1
    data_splits = load_iris(seed=seed)
    batch_size = int(batch_size)
    epochs = int(epochs)
    train_iris(data_splits, loss_metric=loss, epochs=epochs, seed=seed, run_name=run_name, batch_size=batch_size)


def main():
    os.environ['LC_ALL'] = 'C.UTF-8'
    os.environ['LANG'] = "C.UTF-8"
    run()


if __name__ == '__main__':
    main()

'''
python3 iris.py --loss="approx-f1" --epochs=100 --batch_size=256 --run_name="iris-256-approx-f1" 
'''
