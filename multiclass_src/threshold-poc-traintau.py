'''
This file is for loading in the models themselves, and then running through the test loader and 
picking the appropriate model for each class. 

Models were trained on taus [0.1, 0.125, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Models are evaluated on [0.1, 0.2, ... , 0.9]

'''
import os
import torch
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# CUSTOM IMPORT
from download_cifar import *


EPS = 1e-7


def record_results(best_test, output_file):
    # reading in the data from the existing file.
    results_path = "/app/timeseries/multiclass_src/results/max_tau"
    file_path = "/".join([results_path, output_file])
    with open(file_path, "r+") as f:
        data = json.load(f)
        data.append(best_test)
        f.close()

    with open(file_path, "w") as outfile:
        json.dump(data, outfile)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


class Dataset(torch.utils.data.Dataset):
    def __init__(self, ds_split):
        self.X = torch.from_numpy(np.array(ds_split['X'])).float()
        self.y = torch.from_numpy(np.array(ds_split['y']))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index, :], self.y[index]


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_model(models_path, model_name):
    model_path = "/".join([models_path, model_name])
    
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # model.load_state_dict(torch.load(model_path))
    model = torch.load(model_path).to("cuda:3")
    
    return model



def get_test_loader(batch_size, seed):
    data_splits = load_imb_data(seed)
    train_set = Dataset(data_splits['train'])
    validation_set = Dataset(data_splits['val'])
    test_set = Dataset(data_splits['test'])

    data_params = {'batch_size': batch_size, 'shuffle': True,
                   'num_workers': 1, 'worker_init_fn': np.random.seed(seed)}
    set_seed(seed)
    train_loader = DataLoader(train_set, **data_params)
    set_seed(seed)
    val_loader = DataLoader(validation_set, **data_params)
    set_seed(seed)
    test_loader = DataLoader(test_set, **data_params)
    return train_loader, val_loader, test_loader


def evaluation_f1(device, y_labels=None, y_preds=None, threshold=None):
    classes = len(y_labels[0])
    mean_f1s = torch.zeros(classes, dtype=torch.float32)
    precisions = torch.zeros(classes, dtype=torch.float32)
    recalls = torch.zeros(classes, dtype=torch.float32)

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

    return mean_f1s, mean_f1s.mean(), precisions, recalls


def get_metrics(device, batch_size, seed, results_path, models_path, models_list):
    # LOAD IN TEST LOADER 
    test_loader, _, _ = get_test_loader(batch_size=batch_size, seed=seed)

    # EVALUATION
    results_json = {   
        "0.1": {
            "0.1": {"mean_f1": None, "eval_dxn": None},
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
        },
        "0.125":  {
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
        },
        "0.2": {
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
        },
        "0.3": {
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
        },
        "0.4":  {
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
        },
        "0.5": {
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
        },
        "0.6":  {
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
        },
        "0.7": {
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
        }, 
        "0.8":  {
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
        },
        "0.9":  {
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
        },
    }

    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
    trained_taus = ["0.1", "0.125", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]

    with torch.no_grad():    
        # for each trained model, go through all the thresholds and evaluate the performance again. 
        for x in range(len(models_list)):
            # loading in model 
            print("loading in model: {}".format(models_list[x]))
            model = load_model(models_path, models_list[x])
            model.eval()

            for tau in test_thresholds:
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
                    labels = torch.zeros(len(labels), 10).to(device).scatter_(1, labels.unsqueeze(1), 1.).to(device)

                    pred_arr = output.detach().cpu().numpy()
                    label_arr = labels.detach().cpu().numpy()

                    # appending results.
                    test_preds.append(pred_arr)
                    test_labels.append(label_arr)

                test_preds = torch.tensor(test_preds[0])
                test_labels = torch.tensor(test_labels[0])

                class_f1s, mean_f1, precisions, recalls = evaluation_f1(device=device, y_labels=test_labels, y_preds=test_preds, threshold=tau)

                tau = str(tau)
                print("For Model {}, Eval Tau: {}, Mean F1: {}".format(trained_taus[x], tau, mean_f1.item()))
                print("For Model {}, Eval Tau: {}, Class F1: {}".format(
                    trained_taus[x], tau, class_f1s.numpy().tolist() ))

                results_json[trained_taus[x]][tau]['class_f1s'] = class_f1s.numpy().tolist() 
                results_json[trained_taus[x]][tau]['mean_f1'] = mean_f1.item()
                results_json[trained_taus[x]][tau]['eval_dxn'] = final_test_dxn
                results_json[trained_taus[x]][tau]['class_precisions'] = precisions.numpy().tolist()
                results_json[trained_taus[x]][tau]['class_recalls'] = recalls.numpy().tolist()

    record_results(results_json, results_path)
    return results_json


if __name__ == '__main__':
    train_tau_models_100p = [
        "20201204_best_model_1024_approx-f1_run3-train_tau-approx-f1-imb-0.1-2.pth",
        "20201204_best_model_1024_approx-f1_run3-train_tau-approx-f1-imb-0.125-2.pth",
        "20201204_best_model_1024_approx-f1_run3-train_tau-approx-f1-imb-0.2-2.pth",
        "20201204_best_model_1024_approx-f1_run3-train_tau-approx-f1-imb-0.3-2.pth",
        "20201204_best_model_1024_approx-f1_run3-train_tau-approx-f1-imb-0.4-2.pth",
        "20201204_best_model_1024_approx-f1_run3-train_tau-approx-f1-imb-0.5-2.pth",
        "20201204_best_model_1024_approx-f1_run3-train_tau-approx-f1-imb-0.6-2.pth",
        "20201204_best_model_1024_approx-f1_run3-train_tau-approx-f1-imb-0.7-2.pth",
        "20201204_best_model_1024_approx-f1_run3-train_tau-approx-f1-imb-0.8-2.pth",
        "20201204_best_model_1024_approx-f1_run3-train_tau-approx-f1-imb-0.9-2.pth"
    ]
    overfit_train_tau_models_100p = [
        "20201204_overfit_model_1024_100_0.1_run3-train_tau-approx-f1-imb-0.1-0.pth",
        "20201204_overfit_model_1024_100_0.125_run3-train_tau-approx-f1-imb-0.125-0.pth",
        "20201204_overfit_model_1024_100_0.2_run3-train_tau-approx-f1-imb-0.2-0.pth",
        "20201204_overfit_model_1024_100_0.3_run3-train_tau-approx-f1-imb-0.3-0.pth",
        "20201204_overfit_model_1024_100_0.4_run3-train_tau-approx-f1-imb-0.4-0.pth",
        "20201204_overfit_model_1024_100_0.5_run3-train_tau-approx-f1-imb-0.5-0.pth",
        "20201204_overfit_model_1024_100_0.6_run3-train_tau-approx-f1-imb-0.6-0.pth",
        "20201204_overfit_model_1024_100_0.7_run3-train_tau-approx-f1-imb-0.7-0.pth",
        "20201204_overfit_model_1024_100_0.8_run3-train_tau-approx-f1-imb-0.8-0.pth",
        "20201204_overfit_model_1024_100_0.9_run3-train_tau-approx-f1-imb-0.9-0.pth",
    ]

    train_tau_models_150p = [
        '20201206_best_model_1024_approx-f1_run4-150p-train_tau-approx-f1-imb-0.1-2.pth',
        '20201206_best_model_1024_approx-f1_run4-150p-train_tau-approx-f1-imb-0.125-2.pth',
        '20201206_best_model_1024_approx-f1_run4-150p-train_tau-approx-f1-imb-0.2-2.pth',
        "20201206_best_model_1024_approx-f1_run4-150p-train_tau-approx-f1-imb-0.3-2.pth",
        "20201206_best_model_1024_approx-f1_run4-150p-train_tau-approx-f1-imb-0.4-2.pth",
        "20201206_best_model_1024_approx-f1_run4-150p-train_tau-approx-f1-imb-0.5-2.pth",
        "20201206_best_model_1024_approx-f1_run4-150p-train_tau-approx-f1-imb-0.6-2.pth",
        "20201206_best_model_1024_approx-f1_run4-150p-train_tau-approx-f1-imb-0.7-2.pth",
        "20201206_best_model_1024_approx-f1_run4-150p-train_tau-approx-f1-imb-0.8-2.pth",
        "20201206_best_model_1024_approx-f1_run4-150p-train_tau-approx-f1-imb-0.9-2.pth"
    ]
    overfit_train_tau_models_150p = [
        "20201206_overfit_model_1024_patience-150_0.1_run4-150p-train_tau-approx-f1-imb-0.1-2.pth",
        "20201206_overfit_model_1024_patience-150_0.125_run4-150p-train_tau-approx-f1-imb-0.125-2.pth",
        "20201206_overfit_model_1024_patience-150_0.2_run4-150p-train_tau-approx-f1-imb-0.2-2.pth",
        "20201206_overfit_model_1024_patience-150_0.3_run4-150p-train_tau-approx-f1-imb-0.3-2.pth",
        "20201206_overfit_model_1024_patience-150_0.4_run4-150p-train_tau-approx-f1-imb-0.4-2.pth",
        "20201206_overfit_model_1024_patience-150_0.5_run4-150p-train_tau-approx-f1-imb-0.5-2.pth",
        "20201206_overfit_model_1024_patience-150_0.6_run4-150p-train_tau-approx-f1-imb-0.6-2.pth",
        "20201206_overfit_model_1024_patience-150_0.7_run4-150p-train_tau-approx-f1-imb-0.7-2.pth",
        "20201206_overfit_model_1024_patience-150_0.8_run4-150p-train_tau-approx-f1-imb-0.8-2.pth",
        "20201206_overfit_model_1024_patience-150_0.9_run4-150p-train_tau-approx-f1-imb-0.9-2.pth"
    ]
    max_tau_models = [
        "20201206_best_model_1024_approx-f1_maxtau-approx-f1-imb-0.pth"
    ]

    # get_metrics(device="cuda:3", batch_size=1024, seed=11,  results_path="overfit_20201207_traintau_eval_run2.json",
    #     models_path="/app/timeseries/multiclass_src/models/tau_trained/20201206/")
    # Need to pull in results across all runs and average them.

    # 20201206_best_model_1024_approx-f1_maxtau-approx-f1-imb-0.pth
    models_list = [
        "20201206_best_model_1024_approx-f1_searchtau-v2-e10-1024-0.pth", 
        "20201206_best_model_1024_approx-f1_searchtau-v2-e10-1024-1.pth", 
        # "20201206_best_model_1024_approx-f1_searchtau-v2-e5-1024-0.pth", 
        # "20201206_best_model_1024_approx-f1_searchtau-v2-e5-1024-1.pth"
    ]
    get_metrics(device="cuda:3", batch_size=1024, seed=11,  results_path="20201207_searchtau_eval_run0.json",
                models_path="/app/timeseries/multiclass_src/models/searchtau/", models_list=models_list)
